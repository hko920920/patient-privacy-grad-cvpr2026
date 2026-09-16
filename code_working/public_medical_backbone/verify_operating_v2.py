"""Saved-tensor verification and separately recorded single-reviewer development summary."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .verify_pilot import Check,check_packet,arrays,png_pixels,image_from_packet,sha,read,save,cfg,self_test

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/public_operating_diagnostic_20260916_v1'
SALT='public-operating-diagnostic-20260916-v1'
def verify(out):
    started=time.perf_counter();torch.set_num_threads(1);check=Check();c=read(out/'contract.json');run=read(out/'execution.json')
    for p,h in c['source_sha256'].items():check.require(sha(p)==h,'source binding '+str(p))
    for name,key in [('contract.json','contract_sha256'),('inputs.pt','inputs_sha256'),('manifest.json','manifest_sha256')]:
        check.require(sha(out/name)==run[key],'execution input '+name)
    check.require(run['complete'] and run['images']==64 and run['unet_calls']==1920 and run['unet_examples']==2880 and run['backward']==0,'fixed budget')
    previous=Path(c['previous_directory']);prior=read(previous/'contract.json')
    check.require(read(previous/'confirmation_decision.json')['passes_gate'] is False and not read(previous/'adoption_status.json')['approved_for_private_head'],'old failure retained')
    with (Path(prior['plan_directory'])/'generation_tasks.csv').open(encoding='utf-8-sig',newline='') as f:old_tasks=list(csv.DictReader(f))
    expected_prompts={k:next(x['prompt'] for x in old_tasks if x['prompt_id']==k) for k in ('generic','normal','effusion','cardiomegaly')}
    check.require(c['prompts']==expected_prompts,'unchanged prompt strings')
    seeds=[]
    for phase,key,prefix in [('development','development_seeds','S'),('confirmation','reserved_confirmation_seeds','C')]:
        expected={f'{prefix}{i}':int(hashlib.sha256(f'{SALT}|{phase}|{i}'.encode()).hexdigest()[:15],16) for i in range(4)}
        check.require(c[key]==expected,'fixed seeds '+phase);seeds+=list(expected.values())
    check.require(len(set(seeds))==8 and not set(seeds)&{int(t['seed']) for t in old_tasks},'all new distinct seeds')
    inputs=torch.load(out/'inputs.pt',map_location='cpu',weights_only=True);cache=torch.load(previous/'cache.pt',map_location='cpu',weights_only=True)
    check.require(set(inputs['initials'])==set(c['development_seeds']) and set(inputs['conditional'])==set(expected_prompts),'input keys')
    for sid,seed in c['development_seeds'].items():
        z=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
        check.equal(inputs['initials'][sid].numpy(),z.numpy(),'actual CPU initial '+sid)
    for pid,prompt in expected_prompts.items():check.equal(inputs['conditional'][pid].numpy(),cache['hidden'][prompt].float().numpy(),'cached conditional '+pid)
    check.require(inputs['null'].shape==(1,77,1024) and inputs['null'].dtype==torch.float32 and bool(torch.isfinite(inputs['null']).all()),'finite empty conditioning')
    check.require(inputs['empty_token_ids'].shape==(1,77) and inputs['empty_token_ids'].dtype==torch.int64,'empty token shape/dtype')
    # Reconstruct BOS/EOS/padding from the bound vocabulary, including SD2's pad token.
    tc=read(Path(c['snapshot'])/'tokenizer/tokenizer_config.json');vocab=read(Path(c['snapshot'])/'tokenizer/vocab.json')
    # The separately bound special_tokens_map overrides the legacy tokenizer config.
    tc.update(read(Path(c['snapshot'])/'tokenizer/special_tokens_map.json'))
    def token_id(name):
        value=tc[name];return vocab[value['content'] if isinstance(value,dict) else value]
    expected_tokens=np.full((1,77),token_id('pad_token'),dtype=np.int64)
    expected_tokens[0,0]=token_id('bos_token');expected_tokens[0,1]=token_id('eos_token')
    check.equal(inputs['empty_token_ids'].numpy(),expected_tokens,'empty CLIP token identities')
    config=read(Path(c['snapshot'])/'scheduler/scheduler_config.json')
    for key in ['prediction_type','beta_start','beta_end','beta_schedule','num_train_timesteps','clip_sample','steps_offset','set_alpha_to_one']:
        check.require(run['scheduler_config'][key]==config[key],'scheduler '+key)
    training=read(previous/'training_report.json');basehashes=set()
    for cp in ('E4','E8'):
        guard=run['model_guards'][cp];check.require(guard['base_before']==guard['base_after'] and guard['adapter_exact'] and guard['adapter_dtype']=='float32','unchanged actual model '+cp)
        check.require(guard['checkpoint_sha256']==training['checkpoints'][cp]['sha256'],'actual adapter checkpoint '+cp);basehashes.add(guard['base_before'])
    check.require(len(basehashes)==1,'same base across checkpoints')
    manifest=read(out/'manifest.json');check.require(len(manifest)==64,'all64 records')
    expected={(cp,g,sid,pid) for cp in ('E4','E8') for g in (1.,7.5) for sid in c['development_seeds'] for pid in expected_prompts}
    check.require({(r['checkpoint'],r['guidance'],r['seed_id'],r['prompt_id']) for r in manifest}==expected,'complete factorial cells')
    check.require(len({r['image_id'] for r in manifest})==64,'unique blind IDs')
    first={};details={};input_hashes={};gridids=[]
    for r in manifest:
        cp,g,sid,pid=r['checkpoint'],r['guidance'],r['seed_id'],r['prompt_id']
        wanted=hashlib.sha256(f'{SALT}|{cp}|{g}|{sid}|{pid}'.encode()).hexdigest()[:12]
        check.require(r['image_id']==wanted,'blind label')
        check.require(r['checkpoint_sha256']==training['checkpoints'][cp]['sha256'],'record checkpoint')
        for key,hkey in [('path','sha256'),('image_path','image_sha256')]:
            check.require(sha(out/r[key])==r[hkey],'raw file binding');input_hashes[r[key]]=r[hkey]
        packet=arrays(out/r['path']);details[wanted]=check_packet(check,packet,'operating_fp32',g,config)
        expected_h=inputs['conditional'][pid]
        if g>1:expected_h=torch.cat([inputs['null'],expected_h],dim=0)
        check.equal(packet['conditioning'],expected_h.numpy(),'actual branch order and conditioning '+wanted)
        check.equal(packet['initial_from_prepare'],inputs['initials'][sid].numpy(),'common actual initial '+wanted)
        check.equal(png_pixels(out/r['image_path']),image_from_packet(packet),'PNG '+wanted)
        first[(cp,g,sid,pid)]=packet['raw_eps'][0][-1:].copy()
    first_differences={}
    for cp in ('E4','E8'):
        for sid in c['development_seeds']:
            for pid in expected_prompts:
                a,b=first[(cp,1.,sid,pid)],first[(cp,7.5,sid,pid)]
                check.close(a,b,'conditional first-step batch rounding '+cp+sid+pid,atol=1e-4,rtol=1e-4)
                first_differences[cp+'/'+sid+'/'+pid]=float(np.max(np.abs(a-b)))
    # Read all bound sources again to detect mutation during verification.
    for path,h in c['source_sha256'].items():check.require(sha(path)==h,'source unchanged after verification')
    result=dict(status='PASS_PAIRED_OPERATING_SAVED_TENSORS',complete=True,checks=check.count,seconds=time.perf_counter()-started,
        images=64,ddim_transitions=1920,UNet_examples=2880,first_step_conditional_batch_difference=first_differences,
        code_sha256=sha(__file__),contract_sha256=sha(out/'contract.json'),manifest_sha256=sha(out/'manifest.json'),
        correction=dict(kind='special_tokens_map precedence for empty padding',original_code_sha256=sha(Path(__file__).with_name('verify_operating.py')),first_failure_sha256=sha(out/'verification_attempt1_failed.json'),tolerances_changed=False,generated_data_changed=False),
        raw_files_sha256=input_hashes,trajectory_checks=details,errors=check.errors,
        source_of_null_embedding='source/weights/token identities checked; CLIP not independently rerun',
        model_forward_rerun=False,visual_quality_assessed=False,automatic_adoption=False)
    save(out/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('status','checks','seconds','images','ddim_transitions')}))
def summarize(out):
    c=read(out/'contract.json');v=read(out/'verification.json');check=Check()
    check.require(v['complete'] and v['status'].startswith('PASS'),'numerical verification first')
    check.require(v['manifest_sha256']==sha(out/'manifest.json'),'verified manifest unchanged')
    review=read(out/'review_root.json');manifest=read(out/'manifest.json')
    votes={r['image_id']:r for r in review['images']}
    check.require(review['labels_hidden'] is True and len(review['images'])==len(votes)==64 and set(votes)=={r['image_id'] for r in manifest},'complete blinded root votes')
    check.require(all(type(r['pass']) is bool and r['note'].strip() for r in votes.values()),'explicit per-image decisions')
    by_cell={(r['checkpoint'],r['guidance'],r['seed_id'],r['prompt_id']):bool(votes[r['image_id']]['pass']) for r in manifest}
    results={}
    for cp in ('E4','E8'):
        for g in (1.,7.5):
            counts={pid:sum(by_cell[cp,g,sid,pid] for sid in c['development_seeds']) for pid in c['prompts']}
            blocks={sid:sum(by_cell[cp,g,sid,pid] for pid in c['prompts']) for sid in c['development_seeds']}
            total=sum(counts.values());results[f'{cp}_CFG{g}']=dict(total_pass=total,per_prompt_pass=counts,per_seed_pass=blocks,development_screen_pass=total>=12 and min(counts.values())>=2)
    paired={}
    for cp in ('E4','E8'):
        pairs=[(by_cell[cp,1.,sid,pid],by_cell[cp,7.5,sid,pid]) for sid in c['development_seeds'] for pid in c['prompts']]
        paired[cp]=dict(CFG1_fail_to_CFG75_pass=sum(not a and b for a,b in pairs),CFG1_pass_to_CFG75_fail=sum(a and not b for a,b in pairs),same=sum(a==b for a,b in pairs))
    nominee=next(([cp,7.5] for cp in ('E4','E8') if results[f'{cp}_CFG7.5']['development_screen_pass']),None)
    result=dict(complete=True,results=results,paired_CFG_effect=paired,nominated_for_separate_confirmation=nominee,
        approved_for_private_head=False,confirmation_executed=False,reviewers=1,independent_seed_blocks=4,
        original_pilot_failure_unchanged=True,manifest_sha256=sha(out/'manifest.json'),review_sha256=sha(out/'review_root.json'),
        verification_sha256=sha(out/'verification.json'),interpretation='descriptive single-reviewer development diagnosis, not clinical or population evidence')
    save(out/'development_summary.json',result)
    for cp in ('E4','E8'):
        for g in (1.,7.5):
            rows=[r for r in manifest if r['checkpoint']==cp and r['guidance']==g]
            rows.sort(key=lambda r:(list(c['prompts']).index(r['prompt_id']),r['seed_id']))
            grid=Image.new('RGB',(1024,1152),'white');draw=ImageDraw.Draw(grid)
            draw.text((6,8),f'{cp} CFG{g} | {results[f"{cp}_CFG{g}"]["total_pass"]}/16 | one-reviewer development',fill='black')
            for i,r in enumerate(rows):
                x=i%4*256;y=32+i//4*280;good=votes[r['image_id']]['pass']
                draw.text((x+3,y+4),r['prompt_id']+' '+r['seed_id']+' '+('PASS' if good else 'FAIL'),fill='darkgreen' if good else 'darkred')
                with Image.open(out/r['image_path']) as im:grid.paste(im,(x,y+24))
            target=out/f'{cp}_CFG{g}_reviewed.png';check.require(not target.exists(),'existing presentation');grid.save(target)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('verify','summarize','self-test'));p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    if a.phase=='verify':verify(a.out)
    elif a.phase=='summarize':summarize(a.out)
    else:
        tests=self_test();u=np.zeros((1,4,32,32),np.float32);conditional=np.ones_like(u)
        if not np.array_equal(cfg(np.concatenate([u,conditional]),7.5),conditional*7.5):raise AssertionError('CFG7.5 branch fixture')
        if not np.array_equal(cfg(conditional,1.),conditional):raise AssertionError('CFG1 fixture')
        print(json.dumps(dict(previous_numeric_self_test=tests,extra_CFG_fixtures='PASS')))
