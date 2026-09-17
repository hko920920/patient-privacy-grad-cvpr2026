"""Independent packet/source/metric audit. Does not import production sampling/training functions."""
import csv,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
CODE=Path(__file__).resolve().parents[1]
OUT=CODE/'_reports/downstream_development_20260917_v1'
PRIOR=CODE/'_reports/downstream_profile_20260917_v3'
COUNT=0
def need(ok,msg):
    global COUNT
    COUNT+=1
    if not bool(ok):raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def letterbox(p):
    with Image.open(p) as im:
        im=im.convert('L');w,h=im.size;ratio=224/max(w,h);nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
        im=im.resize((nw,nh),Image.Resampling.BILINEAR);result=Image.new('L',(224,224),0);result.paste(im,((224-nw)//2,(224-nh)//2))
        return np.asarray(result,dtype=np.uint8).copy()
def ranking(y,z,w=None):
    y=np.asarray(y,dtype=np.float64);w=np.ones(len(y)) if w is None else w
    order=np.argsort(-z,kind='stable');s=z[order];yy=y[order];ww=w[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(ww*yy)[ends];fp=np.cumsum(ww*(1-yy))[ends]
    need(tp[-1]>0 and fp[-1]>0,'Both weighted classes')
    ap=np.sum(np.diff(np.r_[0,tp])/tp[-1]*tp/np.maximum(tp+fp,1e-300))
    auc=np.sum(np.diff(np.r_[0,fp])/fp[-1]*(np.r_[0,tp[:-1]]+tp)/(2*tp[-1]))
    return np.array([auc,ap])
def draw(records,n,rng_seed):
    rng=np.random.default_rng(rng_seed);result=[]
    for lab in (0,1):
        pool={}
        for r in sorted(records,key=lambda z:z['image_id']):
            if int(r['label'])==lab:pool.setdefault(r['patient_id'],[]).append(r)
        keys=sorted(pool)
        for _ in range(n//2):
            patient=keys[int(rng.integers(len(keys)))];options=pool[patient];result.append(options[int(rng.integers(len(options)))])
    return result
def seed(*x):return int(hashlib.sha256(('downstream-profile-20260917-v1|'+'|'.join(map(str,x))).encode()).hexdigest()[:15],16)
def verify():
    global COUNT
    tick=time.perf_counter();c=read(OUT/'contract.json');result=read(OUT/'result.json');gm=read(OUT/'generation_manifest.json')
    need(len(gm)==512,'512 images')
    for p,h in c['source_sha256'].items():need(sha(p)==h,'Frozen source')
    cache=np.load(PRIOR/'real224.npy',mmap_mode='r');rr=rows(PRIOR/'real_manifest_private.csv')
    audits={r['image_id']:r for r in read(PRIOR/'image_audit.json')}
    lookup={}
    for r in rr:
        r=dict(r,namespace='real/'+r['role'],source_file_sha256=r['sha256'],array_sha256=audits[r['image_id']]['tensor_sha256'])
        need(ah(cache[int(r['index'])])==r['array_sha256'],'Real cache index');lookup[r['image_id']]=r
    pub=[r for r in lookup.values() if r['role']=='public'];priv=[r for r in lookup.values() if r['role']=='private']
    synthetic={m:[] for m in c['methods']};same={};cfg=read(OUT/'generation.json')['scheduler_config']
    alphas=np.load(CODE/'_reports/medical_head_20260916_v1/projection.npz')['alphas']
    max_ddim=0.;max_residual=0.
    weights=np.load(PRIOR/'weights.npz')
    for i,r in enumerate(gm):
        p=OUT/r['image_path'];need(sha(p)==r['image_sha256'] and sha(OUT/r['path'])==r['sha256'],'Generated hashes')
        a=letterbox(p);row=dict(r,namespace='synthetic/'+r['method'],role='synthetic_'+r['method'],patient_id=r['cell'],
            source_file_sha256=r['image_sha256'],array_sha256=ah(a),index=i)
        lookup[r['image_id']]=row;synthetic[r['method']].append(row)
        packet=dict(np.load(OUT/r['path']));need(all(np.isfinite(z).all() for z in packet.values()),'Finite packet')
        need(ah(packet['initial_from_prepare'])==r['initial_sha256'],'Initial binding')
        need(ah(packet['conditioning'])==r['conditioning_sha256'],'Condition binding')
        signature=(r['initial_sha256'],r['conditioning_sha256'],r['label'],r['prompt'])
        need(r['cell'] not in same or same[r['cell']]==signature,'Shared cells');same[r['cell']]=signature
        with Image.open(p) as im:need(np.array_equal(np.asarray(im),np.clip(packet['image_float']*255,0,255).astype(np.uint8)),'PNG decode')
        ts=packet['timesteps'];need(len(ts)==30 and len(packet['latents'])==31,'30 transitions')
        for j,t in enumerate(ts):
            prev=int(t)-cfg['num_train_timesteps']//30
            ap=np.float32(alphas[int(t)]);app=np.float32(alphas[prev]) if prev>=0 else np.float32(1. if cfg['set_alpha_to_one'] else alphas[0])
            x=packet['latents'][j];eps=packet['eps'][j]
            x0=(x-np.sqrt(np.float32(1)-ap)*eps)/np.sqrt(ap)
            if cfg['clip_sample']:x0=np.clip(x0,-cfg['clip_sample_range'],cfg['clip_sample_range'])
            predicted=np.sqrt(app)*x0+np.sqrt(np.float32(1)-app)*eps
            err=float(np.max(np.abs(predicted-packet['latents'][j+1])));max_ddim=max(max_ddim,err)
            need(np.allclose(predicted,packet['latents'][j+1],atol=1e-4,rtol=2e-5),'Independent DDIM')
        if r['method']!='backbone':
            f=packet['head_features'].astype(np.float64);b=packet['head_basis']
            residual=(b[:,None,:,None]*f[:,:,None,:]).reshape(30,1024,64)@weights[r['method']]
            err=float(np.max(np.abs(residual-packet['head_residual'])));max_residual=max(max_residual,err)
            need(np.allclose(residual,packet['head_residual'],atol=2e-6,rtol=2e-6),'Residual')
            raw=packet['raw_eps'];need(np.array_equal(raw[:,:1],packet['base_raw_eps'][:,:1]),'Null unchanged')
            guided=raw[:,:1]+np.float32(7.5)*(raw[:,1:]-raw[:,:1])
            need(np.array_equal(guided,packet['eps']),'CFG arithmetic')
    need(len(same)==128,'128 paired cells')
    for m,records in synthetic.items():need(Counter(int(r['label']) for r in records)=={0:64,1:64},'Balanced method bank')
    # Independently reconstruct every source draw, exposure, and optimization step.
    files=sorted((OUT/'calibration').glob('*/trace.json'))+sorted((OUT/'runs').glob('*/trace.json'))
    need(len(files)==23,'Two calibration plus21 development')
    run_traces={};updates=0
    for path in files:
        tr=read(path);arm=tr['arm'];s=tr['run_seed'];updates+=tr['steps'];run_traces[(path.parent.parent.name,arm,s,tr['optimizer']['lr'])]=tr
        need(tr['fresh_optimizer_state_count']==0 and len(tr['changed_trainable_parameters'])==62,'Fresh optimizer and actual updates')
        counts=Counter()
        for t in tr['trace']:
            step=t['step']-1;first=draw(pub,16,seed(s,step,'public-shared'))
            if arm in ('R0','R1'):second=draw(pub,16,seed(s,step,'public-extra'))
            elif arm=='Dreal':second=draw(priv,16,seed(s,step,'private-extra'))
            else:second=draw(synthetic[dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]],16,seed(s,step,'synthetic-extra'))
            plan=first+second;ids=[r['image_id'] for r in plan]
            need(ids==t['image_ids'],'Independent source draws')
            need([int(r['label']) for r in plan]==t['labels'] and sum(t['labels'])==16,'Labels/balance')
            need([r['namespace'] for r in plan]==t['source_namespaces'],'Namespaces')
            need([r['source_file_sha256'] for r in plan]==t['source_file_sha256'],'Raw/PNG sources')
            need([r['array_sha256'] for r in plan]==t['source_pixel_sha256'],'Actual pixel bindings')
            need([int(r['index']) for r in plan]==t['array_indices'],'Array indices')
            need(set(t['optimizer_steps'])=={step+1} and t['finite_gradients'] and t['finite_loss'],'AdamW/finite')
            z=np.asarray(t['logits'],np.float64);y=np.asarray(t['labels']);loss=np.mean(np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z))))
            need(abs(loss-t['loss'])<=2e-6,'Independent BCE');counts.update(ids)
        exp=read(path.parent/'exposure.json');need(dict(counts)=={r['image_id']:r['exposures'] for r in exp['per_image']},'Exposure recount')
        marker=read(path.parent/'complete.json')
        for p,h in marker['files'].items():need(sha(path.parent/p)==h,'Completed run immutable')
    selected=result['classification_selection'];need(updates==1600+21*selected['steps'],'Optimizer budget')
    selection_rows=[r for r in rr if r['role']=='classifier_selection'];cal=read(OUT/'calibration_choice.json');candidates=[]
    for lr in c['calibration']['lrs']:
        for step in (400,800):
            packet=dict(np.load(OUT/'calibration'/f'R1_lr{lr:g}'/f'selection_{step}.npz'))
            need(packet['image_ids'].tolist()==[r['image_id'] for r in selection_rows],'Selection row order')
            need(packet['labels'].tolist()==[int(r['label']) for r in selection_rows],'Selection image labels')
            values=ranking(packet['labels'],packet['logits']);reported=next(x for x in cal['candidates'] if x['lr']==lr and x['steps']==step)
            need(max(abs(values[0]-reported['AUROC']),abs(values[1]-reported['AP']))<1e-12,'Calibration metrics')
            candidates.append(dict(lr=lr,steps=step,AUROC=float(values[0])))
    best=max(x['AUROC'] for x in candidates);choice=sorted([x for x in candidates if best-x['AUROC']<=.002],key=lambda x:(x['steps'],x['lr']))[0]
    need(choice['lr']==selected['lr'] and choice['steps']==selected['steps'],'Independent calibration choice')
    import torch
    initial_by_seed={}
    for path in files:
        tr=read(path);step=tr['steps'];state=torch.load(path.parent/f'state_{step}.pt',map_location='cpu',weights_only=True);digest=hashlib.sha256()
        for k,v in sorted(state.items()):
            digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes());need(bool(torch.isfinite(v).all()),'Finite final state')
        need(digest.hexdigest()==tr['final_state_sha256'],'Actual final state digest')
        sid=tr['run_seed'];need(sid not in initial_by_seed or initial_by_seed[sid]==tr['initial_state_sha256'],'Shared initial state')
        initial_by_seed[sid]=tr['initial_state_sha256']
        if path.parent.parent.name=='runs':need(step==selected['steps'] and tr['optimizer']['lr']==selected['lr'],'Common frozen setting')
    # Score calculations are independent of sklearn, including tied scores.
    max_metric=0.;packets={};evaluators=[r for r in rr if r['role']=='method_development']
    for arm in c['arms']:
        for k,s in enumerate(c['classifier_seeds']):
            packet=dict(np.load(OUT/'evaluation'/f'{arm}_{s}.npz'));packets[(arm,s)]=packet
            need(packet['image_ids'].tolist()==[r['image_id'] for r in evaluators],'Development row order')
            need(packet['labels'].tolist()==[int(r['label']) for r in evaluators],'Development image labels')
            values=ranking(packet['labels'],packet['logits']);reported=result['scores'][arm][k]
            err=max(abs(values[0]-reported['AUROC']),abs(values[1]-reported['AP']));max_metric=max(max_metric,err);need(err<1e-12,'AUROC/AP exact')
    # Paired patient-cluster bootstrap, seed-wise metrics then mean (not ensemble).
    base=packets[('R1',11)];patients,inv=np.unique(base['patients'],return_inverse=True);rng=np.random.default_rng(c['bootstrap_seed'])
    wanted=['R1','S1','S2','S3','Dreal'];B=c['development_cluster_bootstrap'];boot=np.empty((B,len(wanted),3,2))
    # Presort logits once; vectorize all cluster draws within modest memory.
    counts=np.zeros((B,len(patients)),dtype=np.int16)
    for b in range(B):counts[b]=np.bincount(rng.integers(len(patients),size=len(patients)),minlength=len(patients))
    sample_weights=counts[:,inv].astype(np.float64)
    need(np.all(sample_weights@base['labels']>0) and np.all(sample_weights@(1-base['labels'])>0),'Bootstrap both classes')
    for ai,a in enumerate(wanted):
        for si,s in enumerate(c['classifier_seeds']):
            p=packets[(a,s)];order=np.argsort(-p['logits'],kind='stable');z=p['logits'][order];y=p['labels'][order]
            ends=np.r_[np.flatnonzero(z[1:]!=z[:-1]),len(z)-1];w=sample_weights[:,order]
            tp=np.cumsum(w*y,axis=1)[:,ends];fp=np.cumsum(w*(1-y),axis=1)[:,ends]
            dt=np.diff(np.column_stack([np.zeros(B),tp]),axis=1);df=np.diff(np.column_stack([np.zeros(B),fp]),axis=1)
            boot[:,ai,si,0]=np.sum(df*(np.column_stack([np.zeros(B),tp[:,:-1]])+tp),axis=1)/(2*tp[:,-1]*fp[:,-1])
            boot[:,ai,si,1]=np.sum(dt*tp/np.maximum(tp+fp,1e-300),axis=1)/tp[:,-1]
        print(json.dumps(dict(phase='cluster_bootstrap',arm=a,draws=B)),flush=True)
    summaries={}
    for a,b in [('S2','S1'),('S2','R1'),('S3','S1'),('S3','R1'),('Dreal','R1')]:
        delta=boot[:,wanted.index(a)].mean(axis=1)-boot[:,wanted.index(b)].mean(axis=1)
        summaries[a+'-'+b]={m:dict(mean_delta=result['means'][a][m]-result['means'][b][m],
            percentile95=np.quantile(delta[:,j],[.025,.975]).tolist()) for j,m in enumerate(['AUROC','AP'])}
    gate={}
    for a in ['S2','S3']:
        gate[a]=all(result['means'][a]['AUROC']-result['means'][b]['AUROC']>=.01 and
            result['means'][a]['AP']>=result['means'][b]['AP'] and
            sum(result['scores'][a][i]['AUROC']>result['scores'][b][i]['AUROC'] for i in range(3))>=2 for b in ['S1','R1'])
        need(gate[a]==result['gate'][a]['passed'],'Independent gate')
    need(any(gate.values())==result['engineering_gate_passed'],'Joint gate')
    with (OUT/'cluster_bootstrap.npz').open('xb') as f:np.savez_compressed(f,patient_ids=patients,patient_counts=counts,metrics=boot,arms=np.array(wanted))
    report=dict(status='PASS_INTEGRITY_NOT_EFFICACY',checks=COUNT,seconds=time.perf_counter()-tick,max_metric_difference=max_metric,
        max_DDIM_absolute_difference=max_ddim,max_residual_difference=max_residual,optimizer_updates=updates,
        generation_images=512,generation_transitions=15360,independent_gate=gate,cluster_bootstrap=summaries,
        bootstrap_scope='Development exploratory; paired patient clusters, same draws across arms/seeds; conditional on one synthetic bank/three trained classifiers.',
        expert_pixels=0,reserved_pixels=0,DP=0,result_sha256=sha(OUT/'result.json'),contract_sha256=sha(OUT/'contract.json'))
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)
