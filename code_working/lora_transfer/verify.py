"""Independent saved-artifact verification; no producer sampler, training, or metric import."""
from .common import *
import numpy as np
from PIL import Image
import torch

def weighted_metrics(y,s,w):
    order=np.argsort(-s,kind='mergesort');ys=y[order];ws=w[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(ws*ys)[ends];fp=np.cumsum(ws*(1-ys))[ends]
    pos=tp[-1];neg=fp[-1]
    require(pos>0 and neg>0,'Bootstrap both classes')
    auc=float(np.trapz(np.r_[0.,tp/pos],np.r_[0.,fp/neg]))
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
    ap=float(np.sum(np.diff(np.r_[0.,tp])/pos*precision))
    return np.array([auc,ap])

def verify():
    c=validate_sources();result=read(OUT/'result.json');count=0;errors={}
    def check(ok,msg):
        nonlocal count
        require(ok,msg);count+=1
    old=read(OLD/'result.json');plan=c['plan'];schedule=rows(PLAN_DIR/'training_schedule_private.csv')
    images=read(PLAN_DIR/'training_images_private.json');byid={r['image_id']:r for r in images}
    for role in ('public','private'):
        cf=OUT/('cache_'+role);check(completed(cf),'Cache sealed');cache=torch.load(cf/'cache.pt',map_location='cpu',weights_only=True)
        expected={r['image_id'] for r in images if r['split']==('public' if role=='public' else 'train')};check(set(cache['latents'])==expected,'Cache role-only ids')
        for k,v in cache['latents'].items():check(tuple(v.shape)==(1,4,32,32) and bool(torch.isfinite(v).all()),'Finite latent')
    check(completed(OUT/'e4_replay'),'E4 replay complete')
    e4=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)['adapter'];initial_hash=model_state_hash(e4)
    for arm in ARMS:
        folder=OUT/'training'/arm;check(completed(folder),'Training seal');report=read(folder/'report.json');init=read(folder/'initial.json')
        trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];ss=[r for r in schedule if r['arm']==arm];step=0;exposure=Counter()
        check(init['adapter_state_sha256']==initial_hash,'Exact E4 continued independently');check(init['fresh_optimizer_state']==0,'Fresh AdamW')
        for attempt,t in enumerate(trace,1):
            b=ss[step*4:(step+1)*4]
            check(t['attempt']==attempt and t['step']==step+1,'Attempt sequence')
            for k,expected in [('image_ids',[r['image_id'] for r in b]),('patient_ids',[r['patient_id'] for r in b]),('roles',[r['role'] for r in b]),('timesteps',[int(r['timestep']) for r in b]),('noise_seed',int(b[0]['noise_seed']))]:check(t[k]==expected,'Actual training schedule '+k)
            check(np.isfinite(t['loss']),'Finite loss')
            if t['committed']:
                step+=1;exposure.update(t['image_ids']);check(t['scale_after']>=t['scale_before'] and np.isfinite(t['gradient_norm']),'Successful AMP')
            else:check(t['scale_after']<t['scale_before'],'Overflow skipped')
            check(t['optimizer_step_calls']==step and set(t['optimizer_state_steps'])==({step} if step else set()),'Actual AdamW steps')
        check(step==448 and len(trace)-step<=16,'Fixed successful updates')
        check(dict(exposure)==c['expected_exposure'][arm]['images'],'Actual exposure invariant')
        check(report['base_before']==report['base_after'],'Frozen base unchanged')
        if arm=='L_public':check(not any('cache_private' in p for p in report['access']['cache_files']) and not report['access']['raw_files'],'Public training private-cache boundary')
        for step in (224,448):
            z=torch.load(folder/f'step_{step}.pt',map_location='cpu',weights_only=True)
            check(z['step']==step and set(z['adapter'])==set(e4),'Fixed checkpoint keys')
            check(all(int(v['step'].item())==step for v in z['optimizer']['state'].values()),'Checkpoint successful steps')
            check(all(bool(torch.isfinite(v).all()) for v in z['adapter'].values()),'Finite checkpoint')
    gen=read(OUT/'generation_manifest.json');check(len(gen)==256,'Exactly256 new images')
    inp=torch.load(OLD/'generation_inputs.pt',map_location='cpu',weights_only=True)
    config=read(Path(c['snapshot'])/'scheduler/scheduler_config.json')
    check(config['beta_schedule']=='scaled_linear' and not config['clip_sample'],'DDIM formula assumptions')
    betas=torch.linspace(config['beta_start']**.5,config['beta_end']**.5,1000,dtype=torch.float32)**2;alphas=torch.cumprod(1-betas,dim=0).numpy()
    ddim_max=0.;cfg_max=0.;generated_arrays={}
    for r in gen:
        folder=(OUT/r['image_path']).parent;check(completed(folder),'Cell sealed');p=np.load(OUT/r['path'])
        check(np.array_equal(p['latents'][0],inp['initials'][r['latent']].numpy()),'Shared actual initial')
        cond=torch.cat([inp['null'],inp['conditional'][r['prompt']]]).float().numpy()
        check(np.array_equal(p['conditioning'],cond),'Null/conditional order exact')
        check(all(np.isfinite(p[k]).all() for k in p.files),'Finite packet')
        for j,t in enumerate(p['timesteps']):
            raw=p['raw_eps'][j];epsilon=raw[:1]+np.float32(7.5)*(raw[1:]-raw[:1]);d=float(np.max(np.abs(epsilon-p['eps'][j])));cfg_max=max(cfg_max,d);check(d==0,'CFG exact both adapted branches')
            a=alphas[int(t)];prev=int(t)-1000//30;ap=alphas[prev] if prev>=0 else (np.float32(1) if config.get('set_alpha_to_one',True) else alphas[0])
            x0=(p['latents'][j]-np.sqrt(1-a)*epsilon)/np.sqrt(a)
            expected=np.sqrt(ap)*x0+np.sqrt(1-ap)*epsilon;d=float(np.max(np.abs(expected-p['latents'][j+1])));ddim_max=max(ddim_max,d);check(d<=1e-5,'Independent DDIM transition')
        with Image.open(OUT/r['image_path']) as im:
            check(np.array_equal(np.asarray(im),np.clip(p['image_float']*255,0,255).astype(np.uint8)),'Saved PNG pixels')
            im=im.convert('L');w,h=im.size;scale=224/max(w,h);nw=max(1,round(w*scale));nh=max(1,round(h*scale));can=Image.new('L',(224,224));can.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),((224-nw)//2,(224-nh)//2));generated_arrays[r['image_id']]=np.asarray(can).copy()
    errors.update(ddim_max_abs=ddim_max,cfg_max_abs=cfg_max)
    check(completed(OUT/'classifier_replay'),'Six-update preflight sealed')
    rr=rows(PROFILE/'real_manifest_private.csv');audit={r['image_id']:r for r in read(PROFILE/'image_audit.json')}
    public=[r for r in rr if r['role']=='public'];gm={r['image_id']:r for r in gen}
    def draw(items,run_seed,step,stream):
        # Independently reconstruct patient-then-image sampling.
        ordered=sorted(items,key=lambda r:r['image_id']);rng=np.random.default_rng(seed(run_seed,step,stream));out=[]
        for label in (0,1):
            pool={}
            for r in ordered:
                if int(r['label'])==label:pool.setdefault(r.get('patient_id',r.get('cell')),[]).append(r)
            ids=sorted(pool)
            for _ in range(8):
                pid=ids[int(rng.integers(len(ids)))];pr=pool[pid];out.append(pr[int(rng.integers(len(pr)))])
        return out
    bce_max=0.
    for arm in ARMS:
        synth=[r for r in gen if r['arm']==arm]
        for s in SEEDS:
            folder=OUT/'runs'/f'{arm}_{s}';check(completed(folder),'Classifier seal');tr=read(folder/'trace.json')
            check(tr['steps']==400 and tr['initial_state_sha256']==read(OLD/'runs'/f'S1_{s}'/'trace.json')['initial_state_sha256'],'Same init/steps')
            oldtr=read(OLD/'runs'/f'S1_{s}'/'trace.json')['trace']
            for i,t in enumerate(tr['trace']):
                expected=draw(public,s,i,'public-shared')+draw(synth,s,i,'synthetic-extra')
                check(t['image_ids']==[r['image_id'] for r in expected],'Independent source selection')
                check(t['image_ids'][:16]==oldtr[i]['image_ids'][:16],'Shared historical real stream')
                check(t['patient_ids'][16:]==oldtr[i]['patient_ids'][16:],'Shared historical cells')
                check(t['source_namespaces']==['real/public']*16+['synthetic/'+METHODS[arm]]*16,'Disjoint namespaces')
                check(t['labels']==[int(r['label']) for r in expected] and sum(t['labels'])==16,'Class balance')
                check(t['source_file_sha256']==[r.get('image_sha256',r.get('sha256')) for r in expected],'Actual file hashes')
                pixels=[audit[r['image_id']]['tensor_sha256'] if j<16 else tensor_sha(generated_arrays[r['image_id']]) for j,r in enumerate(expected)]
                check(t['source_pixel_sha256']==pixels,'Actual array pixel hashes')
                check(t['finite_loss'] and t['finite_gradients'] and set(t['optimizer_steps'])=={i+1},'Finite successful classifier update')
                logits=np.array(t['logits']);y=np.array(t['labels']);loss=float(np.mean(np.logaddexp(0,logits)-y*logits));delta=abs(loss-t['loss']);bce_max=max(bce_max,delta);check(delta<=2e-6,'Independent BCE')
            state=torch.load(folder/'state_400.pt',map_location='cpu',weights_only=True)
            import hashlib
            digest=hashlib.sha256()
            for k,v in sorted(state.items()):digest.update(k.encode());digest.update(v.contiguous().numpy().tobytes())
            check(digest.hexdigest()==tr['final_state_sha256'],'Actual final classifier state')
    errors['BCE_max_abs']=bce_max
    names=list(ARMS)+list(old['scores']);packets={};metric_max=0.
    reference=None
    for a in names:
        packets[a]=[]
        for s in SEEDS:
            p=dict(np.load((OUT if a in ARMS else OLD)/'evaluation'/f'{a}_{s}.npz'));packets[a].append(p)
            if reference is None:reference=p
            for k in ('labels','patients','image_ids'):check(np.array_equal(p[k],reference[k]),'Same development evaluation order')
            m=weighted_metrics(p['labels'],p['logits'],np.ones(len(p['labels'])));stored=next(x for x in result['scores'][a] if x['seed']==s)
            delta=max(abs(m[0]-stored['AUROC']),abs(m[1]-stored['AP']));metric_max=max(metric_max,delta);check(delta<1e-12,'Independent AUROC/AP')
    check(len(reference['labels'])==5047 and len(np.unique(reference['patients']))==2026 and sum(reference['labels'])==223,'Method-development only')
    errors['metric_max_abs']=metric_max
    # Shared patient-resampling multiplicities, preserve original image-level labels.
    bootstrap=OUT/'cluster_bootstrap.npz'
    if not bootstrap.exists():
        ids,inverse=np.unique(reference['patients'],return_inverse=True);rng=np.random.default_rng(plan['evaluation']['bootstrap_seed']);values=np.empty((2000,len(names),3,2));draw_sha=hashlib.sha256()
        for b in range(2000):
            draw=rng.integers(len(ids),size=len(ids));counts=np.bincount(draw,minlength=len(ids));draw_sha.update(counts.astype(np.int32).tobytes());w=counts[inverse].astype(float)
            for ai,a in enumerate(names):
                for si,p in enumerate(packets[a]):values[b,ai,si]=weighted_metrics(p['labels'],p['logits'],w)
            if b and b%400==0:log(phase='independent_patient_bootstrap',done=b,total=2000)
        npz(bootstrap,values=values,arms=np.array(names),seeds=np.array(SEEDS),draw_sha256=np.array(draw_sha.hexdigest()))
    bp=np.load(bootstrap);values=bp['values'].mean(axis=2);intervals={}
    for a,b in [('L_pooled','L_public'),('L_pooled','R1'),('L_pooled','R0'),('L_pooled','S0'),('L_public','S1'),('L_pooled','S3')]:
        d=values[:,names.index(a)]-values[:,names.index(b)];intervals[a+'-'+b]={m:np.quantile(d[:,i],[.025,.975]).tolist() for i,m in enumerate(['AUROC','AP'])}
    d=values[:,names.index('L_pooled')]-values[:,names.index('L_public')]-values[:,names.index('S3')]+values[:,names.index('S1')]
    intervals['interaction_descriptive']={m:np.quantile(d[:,i],[.025,.975]).tolist() for i,m in enumerate(['AUROC','AP'])}
    means={a:{m:float(np.mean([r[m] for r in result['scores'][a]])) for m in ('AUROC','AP')} for a in names}
    gate=True
    for b in ('L_public','R1','R0'):
        ds=[x['AUROC']-y['AUROC'] for x,y in zip(result['scores']['L_pooled'],result['scores'][b])]
        gate &= means['L_pooled']['AUROC']-means[b]['AUROC']>=.01 and sum(x>0 for x in ds)>=2 and means['L_pooled']['AP']>=means[b]['AP']
    check(gate==result['engineering_gate_passed'],'Independent engineering decision')
    # Recheck frozen dependencies after all execution, including original failed controls.
    validate_sources()
    if not (OUT/'verification.json').exists():
        save(OUT/'verification.json',dict(status='PASS_INTEGRITY_NOT_AUTOMATIC_EFFICACY',created_utc=now(),checks=count,errors=errors,engineering_gate_passed=gate,
            bootstrap_repeats=2000,patient_cluster_intervals_95=intervals,bootstrap_sha256=sha(bootstrap),result_sha256=sha(OUT/'result.json'),contract_sha256=sha(OUT/'contract.json'),
            original_frozen_sources_unchanged=True,expert_pixels=0,reserved_pixels=0,new_DP=0,final_ready=False,unit='Single LoRA trajectory and bank per arm; three classifier seeds, development patients reused'))
    log(phase='independent_verification_pass',checks=count,errors=errors,engineering_gate=gate,intervals=intervals)
