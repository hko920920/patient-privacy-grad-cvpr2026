"""Independent saved-tensor checks and fixed two-ballot confirmation decision."""
import argparse,csv,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .verify_pilot import Check,check_packet,arrays,png_pixels,image_from_packet,sha,read,save

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/public_operating_confirmation_20260916_v1'
SALT='public-operating-diagnostic-20260916-v1'
ID_SALT='public-operating-confirmation-20260916-v1'

def verify(out):
    started=time.perf_counter();torch.set_num_threads(1);ch=Check();c=read(out/'contract.json');run=read(out/'execution.json')
    for p,h in c['source_sha256'].items():ch.require(sha(p)==h,'frozen source '+p)
    for name,key in [('contract.json','contract_sha256'),('inputs.pt','inputs_sha256'),('manifest.json','manifest_sha256')]:
        ch.require(sha(out/name)==run[key],'execution binding '+name)
    ch.require(run['complete'] and (run['images'],run['unet_calls'],run['unet_examples'],run['backward'],run['vae_decodes'],run['new_text_encoding_calls'])==(16,480,960,0,16,0),'fixed executed budget')
    dev=Path(c['development_directory']);dc=read(dev/'contract.json');ds=read(dev/'development_summary.json')
    previous=Path(c['previous_directory']);prior=read(previous/'contract.json')
    ch.require(c['checkpoint']=='E4' and c['guidance']==7.5 and ds['nominated_for_separate_confirmation']==['E4',7.5],'previous nominee exact')
    ch.require(not read(previous/'confirmation_decision.json')['passes_gate'] and not read(previous/'adoption_status.json')['approved_for_private_head'],'historical failure preserved')
    seeds={f'C{i}':int(hashlib.sha256(f'{SALT}|confirmation|{i}'.encode()).hexdigest()[:15],16) for i in range(4)}
    ch.require(c['confirmation_seeds']==seeds==dc['reserved_confirmation_seeds'],'pre-result reserved seeds')
    with (Path(prior['plan_directory'])/'generation_tasks.csv').open(encoding='utf-8-sig',newline='') as f:old_tasks=list(csv.DictReader(f))
    ch.require(not set(seeds.values())&({int(t['seed']) for t in old_tasks}|set(dc['development_seeds'].values())),'unseen seeds')
    prompts={k:next(t['prompt'] for t in old_tasks if t['prompt_id']==k) for k in ['generic','normal','effusion','cardiomegaly']}
    ch.require(c['prompts']==dc['prompts']==prompts,'unchanged exact prompt strings')
    inputs=torch.load(out/'inputs.pt',map_location='cpu',weights_only=True);old=torch.load(dev/'inputs.pt',map_location='cpu',weights_only=True)
    ch.require(set(inputs['initials'])==set(seeds) and set(inputs['conditional'])==set(prompts),'input keys')
    for sid,seed in seeds.items():
        z=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
        ch.equal(inputs['initials'][sid].numpy(),z.numpy(),'actual reserved CPU latent '+sid)
        ch.require(all(not torch.equal(z,v) for v in old['initials'].values()),'new actual input '+sid)
    for pid in prompts:ch.equal(inputs['conditional'][pid].numpy(),old['conditional'][pid].numpy(),'conditional cache '+pid)
    for key in ['null','empty_token_ids']:ch.equal(inputs[key].numpy(),old[key].numpy(),'exact verified cache '+key)
    config=read(Path(c['snapshot'])/'scheduler/scheduler_config.json')
    dev_run=read(dev/'execution.json');ch.require(run['scheduler_config']==dev_run['scheduler_config'],'complete scheduler unchanged')
    training=read(previous/'training_report.json');guard=run['model_guard']
    ch.require(guard['base_before']==guard['base_after']==dev_run['model_guards']['E4']['base_before'],'base unchanged across runs')
    ch.require(guard['adapter_exact'] and guard['adapter_dtype']=='float32' and guard['checkpoint_sha256']==training['checkpoints']['E4']['sha256'],'exact checkpoint')
    manifest=read(out/'manifest.json');ch.require(len(manifest)==16 and len({x['image_id'] for x in manifest})==16,'16 unique images')
    expected={(sid,pid) for sid in seeds for pid in prompts}
    ch.require({(x['seed_id'],x['prompt_id']) for x in manifest}==expected,'all reserved cells')
    actual_tasks=[{k:x[k] for k in ['image_id','checkpoint','guidance','seed_id','prompt_id','phase']} for x in manifest]
    ch.require(actual_tasks==c['tasks'],'fixed execution task order')
    raw={};details={}
    for row in manifest:
        sid,pid=row['seed_id'],row['prompt_id'];iid=hashlib.sha256(f'{ID_SALT}|{sid}|{pid}'.encode()).hexdigest()[:12]
        ch.require(row['image_id']==iid and row['phase']=='confirmation' and row['checkpoint']=='E4' and row['guidance']==7.5,'task identity')
        ch.require(row['checkpoint_sha256']==guard['checkpoint_sha256'],'per-image checkpoint')
        for key,hkey in [('path','sha256'),('image_path','image_sha256')]:
            ch.require(sha(out/row[key])==row[hkey],'raw file');raw[row[key]]=row[hkey]
        packet=arrays(out/row['path']);details[iid]=check_packet(ch,packet,'confirmation_'+iid,7.5,config)
        h=torch.cat([inputs['null'],inputs['conditional'][pid]],dim=0)
        ch.equal(packet['conditioning'],h.numpy(),'actual branch order '+iid)
        ch.equal(packet['initial_from_prepare'],inputs['initials'][sid].numpy(),'common actual initial '+iid)
        ch.equal(png_pixels(out/row['image_path']),image_from_packet(packet),'actual PNG '+iid)
    ch.require(read(out/'blind_ids.json')==sorted(x['image_id'] for x in manifest),'all blind labels')
    for p,h in c['source_sha256'].items():ch.require(sha(p)==h,'source unchanged after verification')
    result=dict(status='PASS_RESERVED_CONFIRMATION_SAVED_TENSORS',complete=True,checks=ch.count,seconds=time.perf_counter()-started,
        images=16,ddim_transitions=480,UNet_examples=960,code_sha256=sha(__file__),contract_sha256=sha(out/'contract.json'),
        manifest_sha256=sha(out/'manifest.json'),raw_files_sha256=raw,trajectory_checks=details,errors=ch.errors,
        source_of_null_embedding='exact reuse of prior verified and hash-bound tensor',model_forward_rerun=False,
        visual_quality_assessed=False,automatic_adoption=False)
    save(out/'verification.json',result)
    print({k:result[k] for k in ['status','checks','seconds','images']},flush=True)

def decide(out):
    ch=Check();c=read(out/'contract.json');v=read(out/'verification.json');manifest=read(out/'manifest.json')
    ch.require(v['complete'] and v['status'].startswith('PASS') and v['manifest_sha256']==sha(out/'manifest.json'),'verified unchanged manifest')
    ch.require(v['code_sha256']==sha(__file__) and v['contract_sha256']==sha(out/'contract.json'),'fixed verifier and contract')
    for p,h in c['source_sha256'].items():ch.require(sha(p)==h,'frozen source')
    ids={x['image_id'] for x in manifest};ballots={};review_sha={};reviewer_counts={}
    for reviewer in c['review']['reviewers']:
        path=out/('review_'+reviewer+'.json');review=read(path)
        ch.require(review['reviewer']==reviewer and review['labels_hidden'] is True and review['saw_other_review'] is False,'separate hidden-label ballots')
        if reviewer=='independent':ch.require(review['checkpoint_blinded'] is True,'independent candidate-blind reviewer')
        values=review['images'];votes={x['image_id']:x for x in values}
        ch.require(len(values)==len(votes)==16 and set(votes)==ids,'complete review')
        ch.require(all(type(x['pass']) is bool and x['note'].strip() for x in values),'explicit per-image decision')
        ballots[reviewer]=votes;review_sha[path.name]=sha(path);reviewer_counts[reviewer]=sum(x['pass'] for x in values)
    joint={iid:all(votes[iid]['pass'] for votes in ballots.values()) for iid in ids}
    per_prompt={pid:sum(joint[x['image_id']] for x in manifest if x['prompt_id']==pid) for pid in c['prompts']}
    per_seed={sid:sum(joint[x['image_id']] for x in manifest if x['seed_id']==sid) for sid in c['confirmation_seeds']}
    total=sum(joint.values());passed=total>=c['review']['minimum_total'] and min(per_prompt.values())>=c['review']['minimum_per_prompt']
    disagreed=sorted(iid for iid in ids if len({b[iid]['pass'] for b in ballots.values()})>1)
    decision=dict(complete=True,passes_gate=passed,total_pass=total,per_prompt_pass=per_prompt,per_seed_pass=per_seed,
        reviewer_pass_counts=reviewer_counts,disagreements=disagreed,joint_pass_by_image=joint,manifest_sha256=sha(out/'manifest.json'),
        review_sha256=review_sha,verification_sha256=sha(out/'verification.json'),contract_sha256=sha(out/'contract.json'),
        code_sha256=sha(__file__),checks=ch.count,independent_latent_blocks=4,clinical_quality_proven=False,
        higher_CFG_superiority_proven=False,private_utility_proven=False,DP_efficacy_proven=False,original_failed_pilot_unchanged=True)
    save(out/'confirmation_decision.json',decision)
    save(out/'adoption_status.json',dict(approved_for_nonprivate_head_feasibility=passed,approved_for_private_head=passed,
        scope='conditional research operating backbone; only fresh non-DP head comparison next',
        DP_experiment_approved=False,clinical_use_approved=False,checkpoint='E4',guidance=7.5,
        checkpoint_sha256=read(out/'execution.json')['model_guard']['checkpoint_sha256'],
        confirmation_decision_sha256=sha(out/'confirmation_decision.json'),contract_sha256=sha(out/'contract.json'),
        next_action='fresh backbone features and public versus pooled non-DP generation' if passed else 'close current recipe; one stronger public alternative only',
        previous_adoption_file_not_modified=True))
    grid=Image.new('RGB',(1024,1152),'white');draw=ImageDraw.Draw(grid)
    draw.text((6,8),f'Fixed confirmation | joint {total}/16 | four shared latent blocks',fill='black')
    order=sorted(manifest,key=lambda x:(list(c['prompts']).index(x['prompt_id']),x['seed_id']))
    for i,row in enumerate(order):
        x=i%4*256;y=32+i//4*280;iid=row['image_id']
        text=row['prompt_id']+' '+row['seed_id']+' '+('PASS' if joint[iid] else 'FAIL')
        draw.text((x+3,y+4),text,fill='darkgreen' if joint[iid] else 'darkred')
        with Image.open(out/row['image_path']) as im:grid.paste(im,(x,y+24))
    grid.save(out/'confirmation_reviewed.png')
    print({k:decision[k] for k in ['passes_gate','total_pass','per_prompt_pass','reviewer_pass_counts','disagreements']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('verify','decide'));p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    verify(a.out) if a.phase=='verify' else decide(a.out)
