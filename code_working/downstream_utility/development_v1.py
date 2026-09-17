"""Fixed-compute non-DP study; byte-identical corrected kernels with observation hooks."""
from .common import *
import sys,time,gc
from contextlib import contextmanager
from collections import Counter
import numpy as np
import torch
from PIL import Image
from . import data_v2,train_v2

PRIOR=CODE/'_reports/downstream_profile_20260917_v3'
REPLAY=CODE/'_reports/downstream_all_arm_replay_20260917_v1'
OUT=CODE/'_reports/downstream_development_20260917_v1'
NAMES=['real/public','real/private']+['synthetic/'+m for m in METHODS]
AUDIT=CODE/'_reports/patient_usage_audit_20260917_v1'
ACCESS_GUARD=False
def install_pixel_guard():
    global ACCESS_GUARD
    if ACCESS_GUARD:return
    import os
    allowed={os.path.normcase(os.path.abspath(r['path'])) for r in allowed_real()}
    raw_root=os.path.normcase(os.path.abspath(RAW))+os.sep
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if p.startswith(raw_root) and p not in allowed:raise RuntimeError('Blocked non-development raw pixel access')
    sys.addaudithook(audit);ACCESS_GUARD=True
def log(**x):print(json.dumps(dict(utc=now(),**x)),flush=True)
def npz(path,**x):
    with Path(path).open('xb') as f:np.savez_compressed(f,**x)
def tsave(obj,path):
    with Path(path).open('xb') as f:torch.save(obj,f)
def runtime():
    require(not {'downstream_utility.data','downstream_utility.train'}&set(sys.modules),'Legacy imported')
    require(any(type(x).__name__=='RejectLegacy' for x in sys.meta_path),'Import guard absent')
    require(Path(sys.modules['__main__'].__file__).name=='run_development_v1.py','Full-study entrypoint required')
    return {n:str(m.__file__) for n,m in sys.modules.items() if n.startswith('downstream_utility') and getattr(m,'__file__',None)}
def contract():
    runtime();c=read(OUT/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Frozen source changed: '+p)
    require(not c['expert_final_allowed'] and not c['reserved_allowed'] and not c['DP_allowed'],'Forbidden phase')
    install_pixel_guard();return c
def completed(folder):
    if not (folder/'complete.json').exists():return False
    for p,h in read(folder/'complete.json')['files'].items():require(sha(folder/p)==h,'Completed artifact changed')
    return True
def seal(folder,files,**extra):
    import os
    save(folder/'complete.json',dict(created_utc=now(),files={os.path.relpath(p,folder):sha(p) for p in files},**extra))
def fresh(folder):
    if completed(folder):return False
    require(not folder.exists(),'Partial output preserved; explicit technical-resume amendment required: '+str(folder))
    folder.mkdir(parents=True);return True
def allowed_real():
    rr=rows(PRIOR/'real_manifest_private.csv')
    denied={int(r['patient_id']) for r in rows(AUDIT/'patient_usage_private.csv') if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    denied|={int(r['patient_id']) for r in rows(AUDIT/'provisional_patient_split_private.csv') if r['provisional_role']=='reserved_confirmation'}
    require(len(rr)==11277 and len({r['image_id'] for r in rr})==11277,'Manifest size/unique')
    require(not ({int(r['patient_id']) for r in rr}&denied),'Final/reserved patient access blocked')
    require(Counter(r['role'] for r in rr)==dict(public=813,private=320,classifier_selection=5097,method_development=5047),'Role counts')
    for r in rr:
        p=Path(r['path']).resolve();require(p.parent==RAW.resolve() and p.name==r['image_id'],'Raw path/image binding')
    return rr
def prepare_contract():
    runtime();require(not OUT.exists(),'No contract overwrite')
    require(read(REPLAY/'verification.json')['status']=='PASS_CORRECTED_ALL_ARM_ONE_STEP_INTEGRATION_REPLAY','Corrected gate')
    previous=read(PRIOR/'contract.json');plan=read(RESEARCH/'spec_sources/downstream_master_plan_20260917.json')
    require(sha(MODEL_FILE)==MODEL_SHA,'ImageNet digest');allowed_real()
    old=torch.load(PRIOR/'generation_inputs.pt',map_location='cpu',weights_only=True)
    nums=set(previous['previous_numeric_seeds'])|set(previous['generation_seeds'].values())
    seeds={f'L{i:03d}':seed('full-development-bank-v1',i) for i in range(64)}
    require(len(set(seeds.values()))==64 and not(set(seeds.values())&nums),'Fresh numeric seeds')
    initials={k:torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(v),dtype=torch.float32) for k,v in seeds.items()}
    oldz=list(old['initials'].values())
    for p in [MEDICAL/'generation_inputs.pt',Path(previous['medical_config']['confirmation_directory'])/'inputs.pt']:
        oldz+=list(torch.load(p,map_location='cpu',weights_only=True)['initials'].values())
    require(not({tensor_sha(z) for z in initials.values()}&{tensor_sha(z) for z in oldz}),'Fresh actual latents')
    negative=['normal']*22+['effusion']*21+['cardiomegaly']*21;cells=[]
    for i,latent in enumerate(seeds):
        for label,prompt in [(1,'pneumothorax'),(0,negative[i])]:
            cells.append(dict(cell=f'C{i:03d}_{label}',latent=latent,seed=seeds[latent],label=label,prompt=prompt))
    here=Path(__file__).parent
    sources=[here/n for n in ['run_development_v1.py','development_v1.py','verify_development_v1.py','run_v2.py','run.py','common.py','data_v2.py','train_v2.py']]
    sources += [RESEARCH/'TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md',RESEARCH/'spec_sources/downstream_master_plan_20260917.json',
        PRIOR/'contract.json',PRIOR/'real_manifest_private.csv',PRIOR/'real224.npy',PRIOR/'image_audit.json',PRIOR/'data.json',
        PRIOR/'generation_inputs.pt',PRIOR/'weights.npz',REPLAY/'contract.json',REPLAY/'verification.json',MODEL_FILE,
        AUDIT/'patient_usage_private.csv',AUDIT/'provisional_patient_split_private.csv',MEDICAL/'projection.npz',Path(previous['checkpoint'])]
    sources += [CODE/p for p in ['frozen_residual_head/medical_head_sampling.py','frozen_residual_head/run_lora_positive_control.py',
        'public_medical_backbone/run_pilot.py','u_patient_audit/models.py','u_patient_audit/common.py']]
    sources += [Path(p) for p in previous['source_sha256'] if str(p).startswith(previous['snapshot'])]
    OUT.mkdir();tsave(dict(initials=initials,conditional=old['conditional'],null=old['null']),OUT/'generation_inputs.pt');sources.append(OUT/'generation_inputs.pt')
    save(OUT/'contract.json',dict(schema='nonDP-downstream-development/v1',created_utc=now(),source_sha256={str(p):sha(p) for p in sources},
        master_plan=plan,unchanged_corrected_kernel_sha256={n:sha(here/n) for n in ['data_v2.py','train_v2.py']},
        snapshot=previous['snapshot'],checkpoint=previous['checkpoint'],medical_config=previous['medical_config'],
        cells=cells,generation_seeds=seeds,methods=list(METHODS),arms=list(ARMS),namespaces=NAMES,
        generator=dict(CFG=7.5,steps=30,eta=0.,precision='FP32',resolution=256,correction='conditional only; guided delta=7.5*delta'),
        calibration=dict(seed=11,lrs=[1e-4,3e-4],trajectory_steps=800,checkpoints=[400,800],tie_AUROC=.002,optimizer_updates=1600),
        classifier_seeds=[11,23,37],development_runs=21,batch_size=32,weight_decay=1e-4,
        instrumentation='Unchanged train_v2.train_steps; temporary resnet factory forward-pre-hook saves completed step400 before forward401; no math/RNG/optimizer change.',
        instrumentation_parity_updates=4,development_cluster_bootstrap=2000,bootstrap_seed=seed('development-cluster-bootstrap'),
        resume='Only hash-complete cells/runs skipped. Partial outputs require explicit technical amendment; no overwrite or outcome selection.',
        expert_final_allowed=False,reserved_allowed=False,DP_allowed=False,final_ready=False,runtime_modules=runtime()))
    log(phase='contract_frozen',images=512,calibration_updates=1600,development_runs=21)
def load_inputs(profile=False):
    rr=allowed_real();audits={r['image_id']:r for r in read(PRIOR/'image_audit.json')};cache=np.load(PRIOR/'real224.npy',mmap_mode='r')
    for r in rr:r.update(array_key='real',namespace='real/'+r['role'],source_file_sha256=r['sha256'],array_sha256=audits[r['image_id']]['tensor_sha256'])
    directory=PRIOR if profile else OUT;gm=read(directory/'generation_manifest.json');images=[]
    for i,r in enumerate(gm):
        p=directory/r['image_path'];require(sha(p)==r['image_sha256'],'Synthetic file binding')
        with Image.open(p) as im:a=data_v2.letterbox(im)
        images.append(a);r.update(index=i,array_key='synthetic',namespace='synthetic/'+r['method'],
            patient_id=r['cell'],role='synthetic_'+r['method'],source_file_sha256=r['image_sha256'],array_sha256=tensor_sha(a))
    source=data_v2.source_pools(rr,gm);require(set(source)==set(NAMES),'Six namespaces')
    return rr,gm,{'real':cache,'synthetic':np.stack(images)},source
@contextmanager
def observer(folder,run_id,save_steps,total):
    original=train_v2.resnet18;handles=[];start=time.perf_counter()
    def factory(*args,**kwargs):
        model=original(*args,**kwargs);count=[0]
        def before_forward(m,inputs):
            done=count[0]
            if done in save_steps:
                state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                tsave(state,folder/f'state_{done}.pt')
                save(folder/f'checkpoint_{done}.json',dict(step=done,state_sha256=train_v2.state_hash(state),created_utc=now()))
            if done in (10,20) or (done and done%100==0):
                seconds=time.perf_counter()-start
                log(phase='train_progress',run=run_id,steps=done,total=total,elapsed_seconds=seconds,estimated_remaining_seconds=seconds/done*(total-done))
            count[0]+=1
        handles.append(model.register_forward_pre_hook(before_forward));return model
    train_v2.resnet18=factory
    try:yield
    finally:
        train_v2.resnet18=original
        for h in handles:h.remove()
def preflight():
    setup();contract()
    if (OUT/'preflight.json').exists():return
    rr,gm,arrays,source=load_inputs(profile=True);begin=time.perf_counter()
    for r in rr:
        require(sha(r['path'])==r['sha256'],'Raw SHA changed')
        with Image.open(r['path']) as im:a=data_v2.letterbox(im)
        require(np.array_equal(a,arrays['real'][int(r['index'])]) and tensor_sha(a)==r['array_sha256'],'Raw/cache exact')
    state=train_v2.initial_state(11);plain,pt=train_v2.train_steps(state,'R1',source,arrays,2)
    folder=OUT/'instrumentation';folder.mkdir()
    with observer(folder,'instrumentation-parity',[1],2):observed,ot=train_v2.train_steps(state,'R1',source,arrays,2)
    require(train_v2.state_hash(plain)==train_v2.state_hash(observed),'Observer exact final state')
    keys=['image_ids','roles','array_indices','source_namespaces','input_sha256','logits','labels','loss','source_pixel_sha256']
    require(all(a[k]==b[k] for a,b in zip(pt['trace'],ot['trace']) for k in keys),'Observer exact trace')
    first=torch.load(folder/'state_1.pt',map_location='cpu',weights_only=True);prior=read(REPLAY/'runs/R1_0/trace.json')
    require(train_v2.state_hash(first)==prior['final_state_sha256'],'Observed checkpoint equals validated one-step state')
    save(folder/'plain_trace.json',pt);save(folder/'observed_trace.json',ot)
    save(OUT/'preflight.json',dict(status='PASS',raw_rebound=11277,observer_exact=True,checkpoint_exact=True,
        extra_optimizer_updates=4,seconds=time.perf_counter()-begin,expert_pixels=0,reserved_pixels=0,runtime_modules=runtime()))
    log(phase='preflight_pass',raw_images=11277,observer_updates=4)
def generate_512():
    setup();c=contract();require(read(OUT/'preflight.json')['status']=='PASS','Preflight required')
    if completed(OUT/'generation_done'):return
    from types import SimpleNamespace
    from diffusers import AutoencoderKL,DDIMScheduler
    from diffusers.image_processor import VaeImageProcessor
    from frozen_residual_head.medical_head_sampling import GuidedHead,sample
    from public_medical_backbone.run_pilot import models,manual,base_sha
    t0=time.perf_counter();inputs=torch.load(OUT/'generation_inputs.pt',map_location='cpu',weights_only=True)
    unet=models.load_unet(Path(c['checkpoint']),training=False).float();before=base_sha(unet)
    expected=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)['adapter']
    require(all(torch.equal(v,models.adapter_state(unet)[k]) for k,v in expected.items()),'E4 exact')
    snap=Path(c['snapshot']);vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).cuda()
    sched=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True);require(sched.config.prediction_type=='epsilon','Scheduler epsilon')
    pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=sched,image_processor=VaeImageProcessor(vae_scale_factor=8))
    p=dict(np.load(MEDICAL/'projection.npz'));weights=dict(np.load(PRIOR/'weights.npz'));med=c['medical_config']
    make=lambda method:GuidedHead(unet,p['P'],weights[method],p['alphas'],med['time_basis_logsnr_min'],med['time_basis_logsnr_max'])
    hooks=len(unet.conv_out._forward_pre_hooks);manifest=[];new=0
    for cell in c['cells']:
        for method in METHODS:
            stem=cell['cell']+'_'+method;folder=OUT/'generation'/stem
            if completed(folder):manifest.append(read(folder/'record.json'));continue
            require(fresh(folder),'Fresh cell');z=inputs['initials'][cell['latent']].cuda();h=torch.cat([inputs['null'],inputs['conditional'][cell['prompt']]],0).cuda()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            packet=manual(pipe,z,h,7.5,False) if method=='backbone' else sample(pipe,z,h,make(method))
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            require(all(np.isfinite(a).all() for a in packet.values()),'Finite generation')
            require(len(unet.conv_out._forward_pre_hooks)==hooks,'No leaked hook')
            io=time.perf_counter();npz(folder/'trace.npz',**packet)
            Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(folder/'image.png')
            r=dict(**cell,method=method,image_id=stem,path=(folder/'trace.npz').relative_to(OUT).as_posix(),
                image_path=(folder/'image.png').relative_to(OUT).as_posix(),sha256=sha(folder/'trace.npz'),image_sha256=sha(folder/'image.png'),
                initial_sha256=tensor_sha(z),conditioning_sha256=tensor_sha(h),seconds=elapsed,output_IO_seconds=time.perf_counter()-io,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            save(folder/'record.json',r);seal(folder,[folder/'trace.npz',folder/'image.png',folder/'record.json']);manifest.append(r);new+=1
        if len(manifest)%16==0:log(phase='generation',done=len(manifest),total=512,seconds=time.perf_counter()-t0)
    require(len(manifest)==512 and Counter(r['method'] for r in manifest)==dict.fromkeys(METHODS,128),'512 complete')
    require(before==base_sha(unet) and all(torch.equal(v,models.adapter_state(unet)[k]) for k,v in expected.items()),'Generator unchanged')
    require(all(p.grad is None and not p.requires_grad for p in unet.parameters()),'No generator training')
    save(OUT/'generation_manifest.json',manifest);save(OUT/'generation.json',dict(complete=True,images=512,new_decodes=new,backward=0,
        seconds=time.perf_counter()-t0,base_before=before,base_after=before,checkpoint_sha256=sha(c['checkpoint']),
        scheduler_config=dict(sched.config),inputs_sha256=sha(OUT/'generation_inputs.pt'),expert_pixels=0,reserved_pixels=0))
    folder=OUT/'generation_done';folder.mkdir();seal(folder,[OUT/'generation_manifest.json',OUT/'generation.json'])
    del pipe,unet,vae;gc.collect();torch.cuda.empty_cache();log(phase='generation_complete',images=512)
def exposure(trace,records):
    lookup={r['image_id']:r for r in records};pi=Counter(x for t in trace for x in t['image_ids']);pp=Counter()
    for iid,n in pi.items():pp[(lookup[iid]['namespace'],lookup[iid]['patient_id'])]+=n
    detail=[dict(image_id=i,patient_id=lookup[i]['patient_id'],namespace=lookup[i]['namespace'],label=int(lookup[i]['label']),exposures=n) for i,n in sorted(pi.items())]
    groups={}
    for ns in NAMES:
        for label in (0,1):
            eligible=[r for r in records if r['namespace']==ns and int(r['label'])==label];values=[pi[r['image_id']] for r in eligible]
            pv=[pp[(ns,p)] for p in {r['patient_id'] for r in eligible}]
            groups[ns+'/'+str(label)]=dict(population_images=len(values),population_patients=len(pv),draws=sum(values),
                unique_images=sum(v>0 for v in values),unique_patients=sum(v>0 for v in pv),
                image_exposure_max=max(values,default=0),image_exposure_median=float(np.median(values)) if values else 0,
                patient_exposure_max=max(pv,default=0),patient_exposure_median=float(np.median(pv)) if pv else 0)
    return dict(groups=groups,per_image=detail,per_patient=[dict(namespace=k[0],patient_id=k[1],exposures=v) for k,v in sorted(pp.items())])
def fit(folder,arm,run_seed,lr,steps,arrays,source,records,save_steps=()):
    if completed(folder):return read(folder/'trace.json')
    require(fresh(folder),'Fresh classifier');state=train_v2.initial_state(run_seed)
    save(folder/'started.json',dict(utc=now(),arm=arm,seed=run_seed,lr=lr,steps=steps,initial_state_sha256=train_v2.state_hash(state),runtime_modules=runtime()))
    with observer(folder,folder.name,save_steps,steps):final,result=train_v2.train_steps(state,arm,source,arrays,steps,run_seed=run_seed,lr=lr)
    tsave(final,folder/f'state_{steps}.pt');result.update(arm=arm,run_seed=run_seed,runtime_modules=runtime())
    save(folder/'trace.json',result);save(folder/'exposure.json',exposure(result['trace'],records))
    for n in save_steps:save(folder/f'exposure_{n}.json',exposure(result['trace'][:n],records))
    seal(folder,list(folder.glob('*')),optimizer_updates=steps)
    log(phase='train_complete',run=folder.name,steps=steps,seconds=result['seconds'],warm_step_seconds=float(np.mean([t['step_seconds'] for t in result['trace'][10:20]])))
    return result
def predict(state_path,rr,arrays,role):
    require(role in ('classifier_selection','method_development'),'Forbidden evaluation role')
    selected=[r for r in rr if r['role']==role];require(len(selected)==(5097 if role=='classifier_selection' else 5047),'Evaluation count')
    model=train_v2.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,1)
    model.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True),strict=True);model.cuda().eval();logits=[];tick=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(selected),32):
            records=selected[offset:offset+32];a=[arrays['real'][int(r['index'])] for r in records]
            require(all(tensor_sha(x)==r['array_sha256'] for r,x in zip(records,a)),'Evaluation pixels')
            x=data_v2.preprocess_batch(a,'R0',0,0,'cuda');y=model(x).flatten();require(bool(torch.isfinite(y).all()),'Finite logits');logits+=y.cpu().tolist()
    del model;torch.cuda.empty_cache()
    return dict(logits=np.asarray(logits,dtype=np.float64),labels=np.array([int(r['label']) for r in selected]),
        patients=np.array([int(r['patient_id']) for r in selected]),image_ids=np.array([r['image_id'] for r in selected])),time.perf_counter()-tick
def metrics(packet):
    from sklearn.metrics import roc_auc_score,average_precision_score
    return dict(AUROC=float(roc_auc_score(packet['labels'],packet['logits'])),AP=float(average_precision_score(packet['labels'],packet['logits'])))
def calibrate_R1():
    setup();c=contract();require(completed(OUT/'generation_done'),'Generation complete first')
    if (OUT/'calibration_choice.json').exists():require(completed(OUT/'calibration_done'),'Calibration sealed');return
    rr,gm,arrays,source=load_inputs();records=[r for r in rr if r['role'] in ('public','private')]+gm
    for lr in c['calibration']['lrs']:fit(OUT/'calibration'/f'R1_lr{lr:g}','R1',11,lr,800,arrays,source,records,[400])
    candidates=[]
    for lr in c['calibration']['lrs']:
        folder=OUT/'calibration'/f'R1_lr{lr:g}'
        for step in (400,800):
            path=folder/f'selection_{step}.npz'
            if path.exists():packet=dict(np.load(path));seconds=None
            else:
                packet,seconds=predict(folder/f'state_{step}.pt',rr,arrays,'classifier_selection');npz(path,**packet)
            candidates.append(dict(lr=lr,steps=step,**metrics(packet),prediction_sha256=sha(path),inference_seconds=seconds))
    best=max(x['AUROC'] for x in candidates);chosen=sorted([x for x in candidates if best-x['AUROC']<=.002],key=lambda x:(x['steps'],x['lr']))[0]
    save(OUT/'calibration_choice.json',dict(chosen=chosen,candidates=candidates,rule='within .002 of highest AUROC: fewer steps then lower lr',
        model_selection_role='classifier_selection',method_development_accessed=False,created_utc=now(),fixed_for_all_21_runs=True))
    folder=OUT/'calibration_done';folder.mkdir();seal(folder,[OUT/'calibration_choice.json',*list((OUT/'calibration').glob('*/selection_*.npz'))])
    log(phase='calibration_frozen',lr=chosen['lr'],steps=chosen['steps'],selection_AUROC=chosen['AUROC'])
def train_21_runs():
    setup();c=contract();require(completed(OUT/'calibration_done'),'Calibration freeze required')
    chosen=read(OUT/'calibration_choice.json')['chosen'];rr,gm,arrays,source=load_inputs();records=[r for r in rr if r['role'] in ('public','private')]+gm
    for run_seed in c['classifier_seeds']:
        for arm in ARMS:fit(OUT/'runs'/f'{arm}_{run_seed}',arm,run_seed,chosen['lr'],chosen['steps'],arrays,source,records)
    log(phase='all_21_training_complete',optimizer_updates=21*chosen['steps'])
def evaluate_method_development():
    setup();c=contract();chosen=read(OUT/'calibration_choice.json')['chosen']
    require(all(completed(OUT/'runs'/f'{a}_{s}') for s in c['classifier_seeds'] for a in ARMS),'All21 complete before development metrics')
    if (OUT/'result.json').exists():return
    rr,gm,arrays,source=load_inputs();evaluation=OUT/'evaluation';evaluation.mkdir(exist_ok=True);scores={}
    for arm in ARMS:
        scores[arm]=[]
        for s in c['classifier_seeds']:
            path=evaluation/f'{arm}_{s}.npz'
            if path.exists():packet=dict(np.load(path));seconds=None
            else:
                packet,seconds=predict(OUT/'runs'/f'{arm}_{s}'/f"state_{chosen['steps']}.pt",rr,arrays,'method_development');npz(path,**packet)
            scores[arm].append(dict(seed=s,**metrics(packet),prediction_sha256=sha(path),inference_seconds=seconds))
        log(phase='development_inference',arm=arm,complete=len(scores)*3,total=21)
    means={a:{m:float(np.mean([r[m] for r in runs])) for m in ('AUROC','AP')} for a,runs in scores.items()};gate={}
    for a in ('S2','S3'):
        comparisons={}
        for b in ('S1','R1'):
            da=[x['AUROC']-y['AUROC'] for x,y in zip(scores[a],scores[b])];dp=[x['AP']-y['AP'] for x,y in zip(scores[a],scores[b])]
            comparisons[b]=dict(AUROC_delta=means[a]['AUROC']-means[b]['AUROC'],AP_delta=means[a]['AP']-means[b]['AP'],
                seed_AUROC_deltas=da,seed_AP_deltas=dp,positive_seeds=sum(x>0 for x in da))
        passed=all(v['AUROC_delta']>=.01 and v['positive_seeds']>=2 and v['AP_delta']>=0 for v in comparisons.values())
        gate[a]=dict(passed=passed,comparisons=comparisons)
    save(OUT/'result.json',dict(status='DEVELOPMENT_EFFICACY_COMPLETE_UNVERIFIED',created_utc=now(),scores=scores,means=means,gate=gate,
        engineering_gate_passed=any(x['passed'] for x in gate.values()),classification_selection=chosen,generation_images=512,
        calibration_optimizer_updates=1600,development_optimizer_updates=21*chosen['steps'],instrumentation_updates=4,
        expert_pixels=0,reserved_pixels=0,new_DP=0,final_ready=False,weak_label_development_only=True,
        interpretation='Fixed-compute half-batch replacement; synthetic labels are requested conditions, not expert adjudications.',
        contract_sha256=sha(OUT/'contract.json'),calibration_choice_sha256=sha(OUT/'calibration_choice.json')))
    log(phase='development_metrics_complete',means=means,gate=gate)
def verify_all():
    contract()
    from .verify_development_v1 import verify
    verify()
