"""Independent stored-output arithmetic, manifests and limited raw-input replay.

Does not import production preprocessing, sampler, model forward or trainer.
"""
import csv,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'_reports/downstream_profile_20260917_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(ok,msg):
    if not bool(ok):raise RuntimeError(msg)
def state_sha(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def verify(out=OUT):
    tick=time.perf_counter();c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():check(sha(p)==h,'Source mutation '+p)
    mf=rows(out/'real_manifest_private.csv');byid={r['image_id']:r for r in mf}
    check(len(mf)==len(byid)==11277,'Manifest uniqueness')
    check(Counter(r['role'] for r in mf)==dict(public=813,private=320,classifier_selection=5097,method_development=5047),'Manifest role counts')
    groups={role:{r['patient_id'] for r in mf if r['role']==role} for role in {r['role'] for r in mf}}
    for a in groups:
        for b in groups:
            if a<b:check(not groups[a]&groups[b],'Patient overlap')
    ledger=rows(BASE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv')
    locked={r['patient_id'] for r in ledger if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    reserved={r['patient_id'] for r in rows(BASE/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv') if r['provisional_role']=='reserved_confirmation'}
    check(not {r['patient_id'] for r in mf}&(locked|reserved),'Locked/reserved access')
    d=read(out/'data.json');cache=np.load(out/'real224.npy',mmap_mode='r');audit=read(out/'image_audit.json')
    check(sha(out/'real224.npy')==d['cache_sha256'] and len(audit)==len(mf),'Cache binding')
    for i,(r,a) in enumerate(zip(mf,audit)):
        check(r['image_id']==a['image_id'] and r['sha256']==a['raw_sha256'],'Audit row')
        check(hashlib.sha256(cache[i].tobytes()).hexdigest()==a['tensor_sha256'],'Cache tensor row')
    replay=[]
    for role in sorted(groups):
        candidates=sorted([r for r in mf if r['role']==role],key=lambda x:hashlib.sha256(('independent224|'+x['image_id']).encode()).hexdigest())[:16]
        for r in candidates:
            check(sha(r['path'])==r['sha256'],'Raw replay hash')
            with Image.open(r['path']) as im:
                im=im.convert('L');w,h=im.size;ratio=224/max(w,h)
                rw=int(round(w*ratio));rh=int(round(h*ratio));small=np.asarray(im.resize((rw,rh),Image.Resampling.BILINEAR))
            top=(224-rh)//2;left=(224-rw)//2
            independent=np.pad(small,((top,224-rh-top),(left,224-rw-left)),mode='constant')
            check(np.array_equal(independent,cache[int(r['index'])]),'Independent full-frame preprocessing')
            replay.append(r['image_id'])
    g=read(out/'generation.json');gm=read(out/'generation_manifest.json')
    check(len(gm)==28 and g['total_decodes']==31 and g['unet_calls']==931,'Generation cap/count')
    check(g['base_before']==g['base_after'] and g['adapter_exact'],'Model immutability')
    check(sha(out/'generation_manifest.json')==g['manifest_sha256'] and sha(out/'generation_inputs.pt')==g['inputs_sha256'],'Generation bindings')
    inp=torch.load(out/'generation_inputs.pt',map_location='cpu',weights_only=True)
    weights=dict(np.load(out/'weights.npz'));projection=dict(np.load(BASE/'_reports/medical_head_20260916_v1/projection.npz'))
    alpha=projection['alphas'];med=c['medical_config'];eps=np.finfo(np.float32).eps
    cfg=g['scheduler_config'];ddim_checks=0;correction_checks=0
    for r in gm:
        check(sha(out/r['path'])==r['sha256'] and sha(out/r['image_path'])==r['image_sha256'],'Generated binding')
        z=dict(np.load(out/r['path']));check(all(np.isfinite(v).all() for v in z.values()),'Finite trace')
        check(np.array_equal(z['initial_from_prepare'],inp['initials'][r['latent']].numpy()),'Shared latent')
        h=torch.cat([inp['null'],inp['conditional'][r['prompt']]],0).numpy()
        check(np.array_equal(z['conditioning'],h),'Shared conditioning')
        check(hashlib.sha256(h.tobytes()).hexdigest()==r['conditioning_sha256'],'Condition hash')
        check(np.array_equal(z['model_inputs'][:,0],z['latents'][:-1,0]) and np.array_equal(z['model_inputs'][:,1],z['latents'][:-1,0]),'DDIM model-input scale equals one')
        with Image.open(out/r['image_path']) as im:check(np.array_equal(np.asarray(im),np.clip(z['image_float']*255,0,255).astype(np.uint8)),'PNG postprocess')
        for j,tv in enumerate(z['timesteps']):
            t=int(tv);at=np.float32(alpha[t]);prev=t-cfg['num_train_timesteps']//30
            ap=np.float32(alpha[prev] if prev>=0 else 1. if cfg['set_alpha_to_one'] else alpha[0])
            residual=z['eps'][j];xt=z['latents'][j]
            x0=(xt-np.sqrt(np.float32(1)-at)*residual)/np.sqrt(at)
            check(not cfg['clip_sample'] and not cfg['thresholding'],'Unclipped fixed DDIM')
            term1=np.sqrt(ap)*x0;term2=np.sqrt(np.float32(1)-ap)*residual;next_z=term1+term2
            bound=64*eps*(1+np.abs(term1)+np.abs(term2)+np.abs(z['latents'][j+1]))
            check(np.all(np.abs(next_z-z['latents'][j+1])<=bound),'Independent DDIM recurrence')
            raw=z['raw_eps'][j];guided=raw[:1]+np.float32(7.5)*(raw[1:]-raw[:1])
            check(np.allclose(guided,residual,atol=2e-6,rtol=2e-6),'CFG branch arithmetic');ddim_checks+=1
            if r['method']!='backbone':
                a=float(alpha[t]);u=2*((np.log(a)-np.log1p(-a))-med['time_basis_logsnr_min'])/(med['time_basis_logsnr_max']-med['time_basis_logsnr_min'])-1
                basis=np.array([1,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2],np.float64)
                phi=np.concatenate([z['head_features'][j].astype(np.float64)*b for b in basis],axis=1)
                delta=(phi@weights[r['method']]).astype(np.float32).reshape(32,32,4).transpose(2,0,1)[None]
                check(np.allclose(delta,z['conditional_delta'][j],atol=2e-6,rtol=2e-6),'Independent residual')
                base=z['base_raw_eps'][j];check(np.array_equal(raw[:1],base[:1]),'Null branch unchanged')
                check(np.array_equal(raw[1:],base[1:]+delta),'Conditional add')
                gb=base[:1]+np.float32(7.5)*(base[1:]-base[:1])
                bound=64*eps*(1+np.abs(base[:1])+15*np.abs(base[1:])+15*np.abs(delta))
                check(np.all(np.abs(residual-gb-np.float32(7.5)*delta)<=bound),'7.5 correction amplification');correction_checks+=1
    refs={r['method']:dict(np.load(out/r['path'])) for r in gm if r['cell']==c['cells'][0]['cell']}
    for rr in g['replays']:
        check(sha(out/rr['path'])==rr['sha256'],'Replay binding');z=dict(np.load(out/rr['path']))
        ref=refs['private_only' if rr['kind']=='private_replay' else 'backbone']
        for k in ref:check(np.array_equal(z[k],ref[k]),'Full replay '+rr['kind']+' '+k)
    cw=dict(np.load(out/'private_witness.npz'))
    phi=np.concatenate([cw['original_features'].astype(np.float64)*b for b in cw['original_basis']],axis=1)
    check(np.allclose(phi@weights['private_only'],cw['residual'],atol=2e-6,rtol=2e-6),'Independent saved private witness')
    cl=read(out/'classifier.json');check(cl['optimizer_updates']==98,'Update cap')
    initial=torch.load(out/'classifier_initial.pt',map_location='cpu',weights_only=True);check(state_sha(initial)==cl['initial_state_sha256'],'Initial weights')
    gen_byid={r['image_id']:r for r in gm};traces={};exposure=Counter();max_loss_error=0.
    for arm in c['classifier_arms']:
        aa=read(out/'classifier_runs'/f'{arm}_0.json');bb=read(out/'classifier_runs'/f'{arm}_1.json')
        check(aa['steps']==bb['steps']==7 and aa['initial_state_sha256']==cl['initial_state_sha256'],'Same step/initial state')
        wa=torch.load(out/'classifier_runs'/f'{arm}_0.pt',map_location='cpu',weights_only=True)
        wb=torch.load(out/'classifier_runs'/f'{arm}_1.pt',map_location='cpu',weights_only=True)
        check(state_sha(wa)==aa['final_state_sha256'] and all(torch.equal(wa[k],wb[k]) for k in wa),'Actual replay state')
        traces[arm]=aa['trace']
        for a,b in zip(aa['trace'],bb['trace']):
            for k in ('step','loss','logits','labels','input_sha256','image_ids','patient_ids','roles'):check(a[k]==b[k],'Exact training replay')
            labels=np.array(a['labels']);logits=np.array(a['logits']);loss=float(np.mean(np.logaddexp(0,logits)-labels*logits))
            err=abs(loss-a['loss']);max_loss_error=max(max_loss_error,err);check(err<=2e-6,'Independent BCE loss')
            check(labels.sum()==16 and labels[:16].sum()==labels[16:].sum()==8,'Class balance')
            check(a['roles'][:16]==['public']*16,'Shared public half')
            for j,iid in enumerate(a['image_ids']):
                if iid in byid:
                    rr=byid[iid];check(a['patient_ids'][j]==rr['patient_id'] and labels[j]==int(rr['label']),'Real batch provenance')
                    exposure[iid]+=1
                else:
                    rr=gen_byid[iid];check(labels[j]==rr['label'] and a['patient_ids'][j]==rr['cell'],'Synthetic batch provenance')
            second='public' if arm in ('R0','R1') else 'private' if arm=='Dreal' else 'synthetic_'+dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]
            check(a['roles'][16:]==[second]*16,'Source composition')
    for j in range(7):
        ref=traces['R1'][j]['image_ids'][:16]
        check(all(traces[a][j]['image_ids'][:16]==ref for a in c['classifier_arms']),'Paired public draws')
        check(traces['R0'][j]['image_ids']==traces['R1'][j]['image_ids'],'Plain/augment same images')
        cells=[gen_byid[i]['cell'] for i in traces['S0'][j]['image_ids'][16:]]
        check(all([gen_byid[i]['cell'] for i in traces[a][j]['image_ids'][16:]]==cells for a in ('S1','S2','S3')),'Paired synthetic cells')
    timing={}
    for arm in c['classifier_arms']:
        rr=[read(out/'classifier_runs'/f'{arm}_{rep}.json') for rep in (0,1)]
        steps=[z['step_seconds'] for run in rr for z in run['trace'][2:]]
        timing[arm]=dict(warmed_step_median=float(np.median(steps)),warmed_step_max=max(steps),
            peak_allocated=max(z['peak_allocated'] for z in rr),peak_reserved=max(z['peak_reserved'] for z in rr))
    generation_timing={m:dict(mean_seconds=float(np.mean([r['seconds'] for r in gm if r['method']==m])),
        peak_allocated=max(r['peak_allocated'] for r in gm if r['method']==m),peak_reserved=max(r['peak_reserved'] for r in gm if r['method']==m)) for m in c['methods']}
    cost=dict(generation512_compute_seconds=128*sum(x['mean_seconds'] for x in generation_timing.values()),
        classifier21_runs400_steps_seconds=3*400*sum(x['warmed_step_median'] for x in timing.values()),
        classifier21_runs800_steps_seconds=3*800*sum(x['warmed_step_median'] for x in timing.values()),
        public_calibration1600_steps_seconds=1600*timing['R1']['warmed_step_median'],
        full_validation_forward_seconds_per_image=cl['inference_seconds']/cl['inference_images'],
        note='Extrapolation from7-step runs/128-image forward; excludes model IO, full validation, statistical analysis and verification')
    result=dict(status='PASS_BOUNDED_PROFILE_SAVED_ARITHMETIC_AND_PROVENANCE',complete=True,seconds=time.perf_counter()-tick,
        real_images_bound=len(mf),independent_raw_preprocessing_replay_images=len(replay),cache_rows_hashed=len(mf),
        independent_DDIM_transitions=ddim_checks,independent_guided_corrections=correction_checks,
        generation_decodes=31,classifier_optimizer_updates=98,classifier_exact_replays=7,max_BCE_error=max_loss_error,
        generator_timing=generation_timing,classifier_timing=timing,cost_extrapolation=cost,
        expert_pixels=0,reserved_pixels=0,AUROC_AP_computed=False,DP_executed=False,private_utility_established=False,
        limitation='Saved trace arithmetic and64 independently decoded real inputs; not a second UNet/VAE run for every cell or efficacy test',
        verifier_sha256=sha(Path(__file__)),contract_sha256=sha(out/'contract.json'))
    with (out/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)
