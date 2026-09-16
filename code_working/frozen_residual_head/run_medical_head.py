"""Fixed CFG-aware non-DP head extraction and fitting on the adopted public LoRA."""
import argparse,csv,hashlib,shutil,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from . import residual_math as rm
from . import run_capacity as cap
from public_medical_backbone.run_pilot import CODE,RESEARCH,models,require,sha,read,save,save_torch,base_sha,now
from u_patient_audit.common import RUN,IMAGES

OUT=CODE/'_reports/medical_head_20260916_v1'
CONF=CODE/'_reports/public_operating_confirmation_20260916_v1'
CAP=CODE/'_reports/frozen_residual_capacity_20260916_v1'
PUB=CODE/'_reports/frozen_residual_public32_20260916_v1'
PROTOCOL=RESEARCH/'TRACK1_MEDICAL_HEAD_PROTOCOL_20260916.md'
S=7.5

def npz(path,**values):
    with Path(path).open('xb') as f:np.savez(f,**values)

def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def prepare(out):
    require(not out.exists(),'Existing medical head experiment')
    adoption=read(CONF/'adoption_status.json');confirm=read(CONF/'contract.json')
    require(adoption['approved_for_nonprivate_head_feasibility'] and adoption['checkpoint']=='E4' and adoption['guidance']==S,'Adopted configuration required')
    require(sha(CONF/'confirmation_decision.json')==adoption['confirmation_decision_sha256'],'Adoption binding')
    sources=dict(confirm['source_sha256']);old=read(CAP/'contract.json');public=read(PUB/'contract.json')
    sources.update(old['source_sha256']);sources.update(public['source_sha256'])
    for p,h in sources.items():require(sha(p)==h,'Frozen input changed '+p)
    images=read(PUB/'images.json')+read(CAP/'images.json')
    counts={sp:len({x['patient_id'] for x in images if x['split']==sp}) for sp in ('public','train','eval')}
    require(counts==dict(public=32,train=80,eval=40),'Fixed patient counts')
    sets={sp:{x['patient_id'] for x in images if x['split']==sp} for sp in counts}
    require(sum(map(len,sets.values()))==len(set.union(*sets.values())),'Patient overlap')
    backbone=Path(confirm['previous_directory']);bc=read(backbone/'contract.json')
    backbone_rows=csvrows(Path(bc['plan_directory'])/'images.csv')
    bp={x['patient_id'] for x in backbone_rows if x['backbone_role']=='train'}
    require(not bp&set.union(*sets.values()),'Head patients overlap backbone training')
    all_eval=csvrows(RUN/'cohort/evaluation_images.csv');all_aux=csvrows(RUN/'cohort/auxiliary_images.csv')
    excluded={x['patient_id'] for x in all_eval+all_aux}|bp
    tasks=CODE/'_reports/nih_cxr14_k5_evaluator_preflight_protocol_v1_001/real_evaluator_tasks_local.csv'
    reference=[]
    for condition in ('no_finding','pleural_effusion'):
        eligible=[x for x in csvrows(tasks) if x['matched_condition']==condition and x['patient_id'] not in excluded]
        eligible.sort(key=lambda x:hashlib.sha256(('medical-head-reference-v1|'+x['patient_id']).encode()).hexdigest())
        require(len(eligible)>=20,'Insufficient untouched development references')
        for row in eligible[:20]:reference.append(dict(**row,path=str(IMAGES/row['image_filename']),sha256=sha(IMAGES/row['image_filename'])))
    require(len({x['patient_id'] for x in reference})==40,'Unique reference patients')
    for row in images:require(sha(row['path'])==row['sha256'],'Source image changed')
    out.mkdir(parents=True);shutil.copyfile(CAP/'projection.npz',out/'projection.npz')
    require(sha(out/'projection.npz')==old['projection_sha256'],'Projection exact')
    save(out/'images.json',images);save(out/'reference.json',reference)
    plan=cap.sample_plan(old,images);require(len(plan)==4352,'Fixed record count');save(out/'extraction_tasks.json',plan)
    generation_seeds={f'S{i:02d}':int(hashlib.sha256(f'medical-head-generation-20260916-v1|seed|{i}'.encode()).hexdigest()[:15],16) for i in range(16)}
    dev=read(Path(confirm['development_directory'])/'contract.json')
    old_seeds=set(dev['development_seeds'].values())|set(dev['reserved_confirmation_seeds'].values())
    old_seeds|={int(x['seed']) for x in csvrows(Path(bc['plan_directory'])/'generation_tasks.csv')}
    require(len(set(generation_seeds.values()))==16 and not old_seeds&set(generation_seeds.values()),'Fresh fixed generation seeds')
    files=[Path(__file__),Path(__file__).with_name('verify_medical_head.py'),Path(rm.__file__),Path(cap.__file__),PROTOCOL,
        CONF/'contract.json',CONF/'adoption_status.json',CONF/'confirmation_decision.json',CONF/'verification.json',CONF/'inputs.pt',
        CAP/'contract.json',CAP/'images.json',CAP/'projection.npz',PUB/'contract.json',PUB/'images.json',tasks,
        RUN/'cohort/evaluation_images.csv',RUN/'cohort/auxiliary_images.csv']
    for p in files:sources[str(p)]=sha(p)
    witness={sp:next(x['image_id'] for x in images if x['split']==sp) for sp in counts}
    save(out/'contract.json',dict(schema='medical-guided-residual-head/v1',created_utc=now(),source_sha256=sources,
        snapshot=confirm['snapshot'],backbone_directory=str(backbone),confirmation_directory=str(CONF),checkpoint='E4',
        checkpoint_sha256=adoption['checkpoint_sha256'],guidance=S,prompts=confirm['prompts'],generation_seeds=generation_seeds,
        expected_records=4352,expected_patients=counts,expected_images=dict(public=64,train=320,eval=160),draws_per_image=8,
        ridge_lambda=.001,projection_sha256=sha(out/'projection.npz'),images_sha256=sha(out/'images.json'),
        tasks_sha256=sha(out/'extraction_tasks.json'),reference_sha256=sha(out/'reference.json'),witness_images=witness,
        time_basis_logsnr_min=old['time_basis_logsnr_min'],time_basis_logsnr_max=old['time_basis_logsnr_max'],
        objective='r=epsilon-guided_base; X=7.5*phi_conditional; mean patients/draws/spatial SUM_channels + .001||W||^2',
        old_conditional_MSE_not_comparable=True,new_training=False,new_backbone_backward=0,non_DP=True,
        generation_methods=['backbone','public','pooled'],generation_images_per_method=64,reference_patients=40,
        stage='stage2_track1_development',no_outcome_tuning=True,statistics_tolerance=dict(atol=2e-10,rtol=2e-10)))
    print('MEDICAL_HEAD_PREPARED '+sha(out/'contract.json'),flush=True)

def check(out):
    c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Bound input changed '+p)
    for fn,key in [('images.json','images_sha256'),('projection.npz','projection_sha256'),('extraction_tasks.json','tasks_sha256'),('reference.json','reference_sha256')]:require(sha(out/fn)==c[key],'Bound output changed '+fn)
    return c

def extract(out):
    c=check(out);require(not(out/'raw').exists(),'No implicit extraction resume');(out/'raw').mkdir()
    models.setup();start=time.perf_counter();cache=models.load_cache();sched=models.scheduler()
    proj=np.load(out/'projection.npz');require(np.array_equal(sched.alphas_cumprod.numpy(),proj['alphas']),'Same public schedule')
    cp=Path(c['backbone_directory'])/'checkpoints/E4.pt';require(sha(cp)==c['checkpoint_sha256'],'Checkpoint exact')
    unet=models.load_unet(cp,training=False).float();expected=torch.load(cp,map_location='cpu',weights_only=True)['adapter']
    initial_adapter=models.adapter_state(unet)
    require(set(initial_adapter)==set(expected) and all(torch.equal(v,initial_adapter[k]) for k,v in expected.items()),'Actual LoRA exact')
    before=base_sha(unet);p=torch.from_numpy(proj['P'].astype(np.float32)).cuda()
    null=torch.load(Path(c['confirmation_directory'])/'inputs.pt',map_location='cpu',weights_only=True)['null']
    captured={}
    def tap(module,inputs):captured['h']=inputs[0].detach()
    hook=unet.conv_out.register_forward_pre_hook(tap);manifest=[];timings=[];plan=read(out/'extraction_tasks.json')
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for i,row in enumerate(plan):
                z=cache['latents'][row['image_id']].float();eps=torch.randn(z.shape,generator=torch.Generator(device='cpu').manual_seed(row['noise_seed']),dtype=torch.float32)
                xt,target=models.diffusion_input(z,torch.tensor([row['timestep']]),eps,sched)
                hcond=cache['hidden'][row['prompt']].float();hh=torch.cat([null,hcond],0).cuda()
                xx=torch.cat([xt,xt],0).cuda();tt=torch.tensor([row['timestep']],device='cuda')
                if i==0:unet(xx,tt,hh).sample;torch.cuda.synchronize()
                tick=time.perf_counter();raw=unet(xx,tt,hh).sample
                guided=raw[:1]+S*(raw[1:]-raw[:1]);hidden=captured.pop('h')[-1].permute(1,2,0).reshape(-1,320)
                f=torch.cat([torch.ones((1024,1),device='cuda'),hidden@p],1)
                basis=rm.legendre_time_basis(np.int64(row['timestep']),proj['alphas'],logsnr_min=c['time_basis_logsnr_min'],logsnr_max=c['time_basis_logsnr_max'])
                values=dict(features=f.cpu().numpy().copy(),raw_eps=raw.cpu().numpy().copy(),base=guided[0].permute(1,2,0).reshape(-1,4).cpu().numpy().copy(),
                    target=target[0].permute(1,2,0).reshape(-1,4).numpy().copy(),basis=basis,timestep=np.array(row['timestep'],np.int64))
                if row['image_id']==c['witness_images'][row['split']] and row['draw_id'] in (0,7):
                    values.update(hidden=hidden.cpu().numpy().copy(),noisy=xt.numpy().copy(),conditioning=hh.cpu().numpy().copy())
                require(all(np.isfinite(a).all() for a in values.values()),'Nonfinite extraction')
                path=out/'raw'/f'{i:06d}.npz';npz(path,**values);torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
                manifest.append(dict(**row,raw_path=path.relative_to(out).as_posix(),raw_sha256=sha(path)))
                if i==7:
                    save(out/'profile.json',dict(records=8,mean_record_seconds=float(np.mean(timings)),estimated_record_loop_seconds=float(np.mean(timings))*4352,peak_bytes=torch.cuda.max_memory_allocated(),performance_not_used_for_tuning=True))
                if i%128==0 or i+1==len(plan):print({'phase':'extract','done':i+1,'total':4352,'seconds':round(time.perf_counter()-start,2)},flush=True)
    finally:hook.remove()
    after=base_sha(unet);actual=models.adapter_state(unet)
    require(before==after and all(torch.equal(actual[k],expected[k]) for k in expected),'Model changed')
    require(all(not p.requires_grad and p.grad is None for p in unet.parameters()),'Unexpected parameter gradients')
    save(out/'manifest.json',manifest);save(out/'extraction.json',dict(complete=True,records=len(manifest),seconds=time.perf_counter()-start,mean_record_seconds=float(np.mean(timings)),
        unet_calls=4353,unet_examples=8706,backward=0,warmup_calls=1,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        base_before=before,base_after=after,adapter_exact=True,contract_sha256=sha(out/'contract.json'),manifest_sha256=sha(out/'manifest.json')))
    print('EXTRACTION_COMPLETE',flush=True)

def fit(out):
    c=check(out);start=time.perf_counter();execution=read(out/'extraction.json');require(execution['complete'] and execution['manifest_sha256']==sha(out/'manifest.json'),'Complete bound extraction')
    groups=defaultdict(list)
    for row in read(out/'manifest.json'):
        path=out/row['raw_path'];require(sha(path)==row['raw_sha256'],'Raw input changed')
        with np.load(path) as d:
            stat=rm.sufficient_statistics_separable(S*d['features'].astype(np.float64),d['target'].astype(np.float64)-d['base'].astype(np.float64),d['basis'])
        groups[(row['split'],row['patient_id'])].append(stat)
    patients=[];aa=[];bb=[];qq=[]
    for key,stats in groups.items():
        stat=rm.mean_statistics(stats);patients.append(dict(split=key[0],patient_id=key[1],records=len(stats)))
        aa.append(stat['A']);bb.append(stat['B']);qq.append(stat['Q'])
    aa=np.stack(aa);bb=np.stack(bb);qq=np.array(qq);npz(out/'patient_statistics.npz',A=aa,B=bb,Q=qq);save(out/'patients.json',patients)
    indices={sp:[i for i,x in enumerate(patients) if x['split']==sp] for sp in ('public','train','eval')}
    weights={};fit_objectives={}
    for name,ii in [('public',indices['public']),('pooled',indices['public']+indices['train'])]:
        a,b,q=aa[ii].mean(0),bb[ii].mean(0),float(qq[ii].mean());w=rm.solve_ridge(a,b,.001);weights[name]=w
        fit_objectives[name]=rm.loss_from_statistics(a,b,q,w,ridge=.001)
    weights['backbone']=np.zeros((64,4));npz(out/'weights.npz',**weights)
    losses={name:[rm.loss_from_statistics(aa[i],bb[i],qq[i],w)/4 for i in indices['eval']] for name,w in weights.items()}
    result=dict(complete=True,seconds=time.perf_counter()-start,patient_counts={k:len(v) for k,v in indices.items()},fit_objectives=fit_objectives,
        development_MSE={k:float(np.mean(v)) for k,v in losses.items()},development_losses=losses,development_patient_ids=[patients[i]['patient_id'] for i in indices['eval']],
        contract_sha256=sha(out/'contract.json'),patient_statistics_sha256=sha(out/'patient_statistics.npz'),patients_sha256=sha(out/'patients.json'),weights_sha256=sha(out/'weights.npz'),
        primary_generation_utility_not_yet_evaluated=True,nonDP=True)
    save(out/'fit.json',result);print({k:result[k] for k in ['seconds','patient_counts','development_MSE']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','extract','fit'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a.out)
