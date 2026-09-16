"""Independent CPU checks and fixed descriptive summaries for a single mask intervention.

No endpoint producer import, neural inference, fitting, AUC, CI or selection.
Saved FP32 inputs are replayed in their original family, never homogenized.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch
from safetensors import safe_open
from .verify_fp32_noising import verify_fp32_noising

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'_reports/cvpr_u_pilot_v1_001'
BASE=RUN/'baseline_screen_20260915'
OUT=BASE/'masked_slot_endpoints_v2'
PID='14393'
IMAGES=['00014393_002.png','00014393_006.png','00014393_001.png','00014393_004.png']
BRANCHES=['treatment','control']
PROMPT='a frontal chest radiograph'
PRODUCER_SHA='c0d5af1eb1cda141a33f9a5663b0e8ec8b2004654f9bfa2d91855ea365146262'
NOISING_SHA='ba710cb9d80fb92e6e70b4b850333df4d5087c12b855e425bfda7b271c94c195'
STATUS='PASS_MASKED_ENDPOINT_SAVED_ARITHMETIC_SOURCE_REUSE_AND_TRAINING_GATES'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def exact(a,b,name='value'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,name
        assert torch.equal(a,b),name
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),name
        for k in a:exact(a[k],b[k],name+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),name
        for i in range(len(a)):exact(a[i],b[i],name+'/'+str(i))
    else:assert type(a)==type(b) and a==b,name


def csv_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def noise(iid,j):
    text=f'promptloss-v1|260915|{iid}|140' if j==0 else f'promptloss-repeat-v1|260915|{iid}|140|{j}'
    seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%(2**63-1)
    return seed,torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)


def policy():
    return dict(schema='masked-endpoint-independent-analysis/v1',patient_id=PID,current_step=2,
        stored_cells=4464,paired_cells=2232,new_forward=2260,reused_cells=2204,backward=0,
        orientation='negative image loss before patient mean/max; MoFit h=conditioned-minus-null MSE',
        change='treatment minus control; contribution slots8 versus0, all other scheduled slots fixed',
        generic_t140=dict(seeds=list(range(8)),prefixes=[1,2,4,8],
            primary='negative saved GPU FP32 MSE',additional=['negative residual FP64 MSE','negative residual FP64 L2']),
        cdi_dl_t100=dict(seeds=list(range(5)),prefixes=[1,2,4,5],
            primary='negative saved GPU FP32 L2',additional=['negative CPU FP32 MSE','negative residual FP64 MSE','negative residual FP64 L2']),
        fixed_mofit=dict(queries=IMAGES,source_embeddings=IMAGES,source_model='original model_1',
            seeds=list(range(5)),fresh_mean=[1,2,3,4],conditions=['null']+IMAGES,
            h='conditioned-minus-same-query-same-noise-null MSE; FP32 primary and FP64 sensitivity',
            pooling='average fixed noise set within each query/condition cell, then mean/max the two E or U photos; differences last'),
        patient_summaries='all41 separately; p14393 highlighted only by prespecified identity',
        selection_context='40 paired patient changes, all/A/B descriptive count, mean, median, SD, quartiles, min, max and sign counts',
        AUC=False,CI=False,fitting=False,reference_selection=False,
        limitations=['One selected patient and one fixed training realization; no population effect or attack performance.',
            'Control masks eight losses with original denominator and forward slots, not a resized-dataset retrain.',
            'Runtime full-weight fingerprints are checked against artifact/checkpoint values; neural predictions are not independently rerun.',
            'FP64 residual arithmetic checks saved predictions, not an independently executed FP64 neural model.',
            'MoFit embeddings and posterior latents remain treatment-derived; control optimization is not rerun.',
            'Selection A/B are context groups; their membership does not change across this intervention.',
            'No stage2 completion or new-method superiority follows automatically.'])


def scope():
    rows=[r for r in csv_rows(RUN/'cohort/evaluation_images.csv') if r['eval_role']=='selection'
          or (r['eval_role']=='fit' and r['patient_id']==PID)]
    order=[PID]+sorted({r['patient_id'] for r in rows if r['eval_role']=='selection'},key=int)
    result=[]
    for p in order:
        for s,role in [('E','train_candidate'),('U','U_observed')]:
            rr=sorted([r for r in rows if r['patient_id']==p and r['record_role']==role],key=lambda r:r['image_id'])
            assert len(rr)==2
            result.extend(dict(r,scenario=s) for r in rr)
    assert len(order)==41 and len(result)==164 and [r['image_id'] for r in result[:4]]==IMAGES
    assert Counter(r['assignment_group'] for r in result if r['eval_role']=='selection')=={'A':80,'B':80}
    return result,order


def training(c):
    d=Path(c['training_directory']);v=read(c['training_verification']);e=read(d/'execution.json');p=read(d/'protocol.json')
    assert v['status']=='PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC' and v['complete'] is True
    assert e['complete'] is True and e['status']=='PASS_REPLAY_AND_MASKED_CONTRIBUTION_EXECUTION_PENDING_INDEPENDENT_VERIFICATION'
    for file,key in [('execution.json','execution_sha256'),('protocol.json','protocol_sha256')]:assert sha(d/file)==v[key]
    assert sha(p['contract_path'])==p['contract_sha256']==v['contract_sha256']==e['contract_sha256']
    exact(p['contract'],read(p['contract_path']))
    con=read(d/'conformance.json')
    assert con['status']=='PASS_EXACT_TREATMENT_REPLAY_AND_SHARED_PREFIX' and len(con['gates'])==3 and all(x['exact'] is True for x in con['gates'])
    assert sha(d/'conformance.json')==e['conformance_sha256']
    states={}
    for b in BRANCHES:
        states[b]={}
        for step in [43,250,1000]:
            path=d/b/f'step_{step:04d}.pt'
            assert sha(path)==v['checkpoint_sha256'][b][str(step)]
            states[b][step]=torch.load(path,map_location='cpu',weights_only=True)
            exact(states[b][step]['contract'],p['contract']['original_training_contract'])
    for step in [250,1000]:
        original=torch.load(RUN/f'training_coverage_v2/model_1/step_{step:04d}.pt',map_location='cpu',weights_only=True)
        exact(states['treatment'][step],original,f'original full checkpoint{step}')
    exact(states['treatment'][43],states['control'][43],'shared full checkpoint43')
    a,b=states['treatment'][1000],states['control'][1000]
    expected=dict(a['exposures']);assert [expected.pop(i) for i in IMAGES[:2]]==[4,4]
    exact(b['exposures'],expected);assert sum(expected.values())==3992 and sum(a['exposures'].values())==4000
    for branch,x in [('treatment',a),('control',b)]:
        assert x['step']==x['attempt']==1000 and len(x['losses'])==1000
        assert sha(c['checkpoint_paths'][branch])==c['checkpoint_sha256'][branch]
        exact(x,torch.load(c['checkpoint_paths'][branch],map_location='cpu',weights_only=True))
    return {'treatment':a,'control':b},dict(training_verification_sha256=sha(c['training_verification']),
        full_checkpoints_250_1000_original_exact=True,shared_full_checkpoint43_exact=True,
        numerical_contract_included=True,treatment_effective_exposures=4000,control_effective_exposures=3992)


def sources(c,ids):
    ds={k:Path(v) for k,v in c['source_directories'].items()}
    names={'prompt':('verification.json','PASS_PROMPT_LOSS_RAW_ARITHMETIC_AND_COHORT'),
        'repeat':('verification_v2.json','PASS_REPEAT_NOISE_SAVED_TENSORS_AND_REUSE'),
        'cross':('verification.json','PASS_CROSS_RESPONSE_SAVED_ARITHMETIC_SOURCE_AND_REPLAYS'),
        'cdi_E':('verification.json','PASS_SAVED_E_COHORT_ARITHMETIC_AND_INTEGRITY'),
        'cdi_U':('verification.json','PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY')}
    tables={};protocols={}
    for k,d in ds.items():
        n,status=names[k];v=read(d/n);e=read(d/'execution.json');p=read(d/'protocol.json')
        assert v['status']==status
        for f,field in [('results.json','results_sha256'),('protocol.json','protocol_sha256')]:
            assert sha(d/f)==e[field]
            if field in v:assert sha(d/f)==v[field]
        assert sha(p['contract_path'])==p['contract_sha256'];exact(p['contract'],read(p['contract_path']))
        protocols[k]=p;tables[k]=read(d/'results.json')
    mapping={}
    for key in ['prompt','repeat','cdi_E','cdi_U']:
        mapping[key]={r['image_id']:r for r in tables[key] if r['model']=='model_1' and r['image_id'] in ids}
    assert len(mapping['prompt'])==164 and len(mapping['repeat'])==160
    assert len(mapping['cdi_E'])==len(mapping['cdi_U'])==82
    cross=torch.load(ds['cross']/'raw.pt',map_location='cpu',weights_only=True)
    assert sha(ds['cross']/'raw.pt')==read(ds['cross']/'execution.json')['raw_sha256']
    index={(r['query_image_id'],r['source_image_id'] or 'null',r['noise_index']):r for r in tables['cross']
           if r['recipient_model']=='model_1' and r['source_model'] in [None,'model_1']}
    assert len(index)==100
    return ds,mapping,cross,index,protocols


def expected_fingerprints(c,states):
    paths=[p for p in c['frozen_sha256'] if Path(p).name=='diffusion_pytorch_model.fp16.safetensors' and Path(p).parent.name=='unet']
    assert len(paths)==1
    hashes={b:hashlib.sha256() for b in BRANCHES}
    with safe_open(paths[0],framework='pt',device='cpu') as reader:
        names={re.sub(r'\.(to_q|to_k|to_v|to_out\.0)\.(weight|bias)$',r'.\1.base_layer.\2',n):n for n in reader.keys()}
        assert len(names)==len(list(reader.keys()))
        assert set(states['treatment']['adapter'])==set(states['control']['adapter'])
        allnames=sorted(set(names)|set(states['treatment']['adapter']))
        assert not set(names)&set(states['treatment']['adapter'])
        for name in allnames:
            base=reader.get_tensor(names[name]).float() if name in names else None
            for b in BRANCHES:
                t=(base if base is not None else states[b]['adapter'][name].float()).contiguous()
                assert torch.isfinite(t).all()
                hashes[b].update(name.encode()+b'\0');hashes[b].update(str(t.dtype).encode())
                hashes[b].update(json.dumps(list(t.shape)).encode());hashes[b].update(t.numpy().tobytes())
    return {b:dict(sha256=h.hexdigest(),state_tensors=len(allnames)) for b,h in hashes.items()}


def arithmetic(q):
    p,n=q['prediction'],q['epsilon']
    for t in [p,n,q['noised_latent']]:assert t.dtype==torch.float32 and t.shape==(1,4,32,32) and torch.isfinite(t).all()
    pp=p.double().numpy().ravel();nn=n.double().numpy().ravel();r=pp-nn
    ss=math.fsum(float(x)*float(x) for x in r);mse=ss/len(r);l2=math.sqrt(ss)
    for key,v in [('loss_mse_fp64',mse),('loss_l2_fp64',l2)]:
        assert math.isfinite(q[key]) and math.isclose(q[key],v,rel_tol=3e-14,abs_tol=1e-30),key
    # Error bound includes FP32 residual subtraction/square, arbitrary positive
    # accumulation of4096 terms and final division/sqrt. Not an efficacy gate.
    u=2**-24;gamma=(len(r)+6)*u/(1-(len(r)+6)*u)
    bound=gamma*math.fsum(float(x)**2 for x in np.abs(pp)+np.abs(nn))+1e-35
    assert abs(q['loss_mse_fp32']-mse)<=bound/len(r)
    assert abs(q['loss_l2_fp32']-l2)<=math.sqrt(max(ss+bound,0))-math.sqrt(ss)+2*u*l2+1e-30
    return mse,l2,dict(mse_absolute_error=abs(q['loss_mse_fp32']-mse),
        l2_absolute_error=abs(q['loss_l2_fp32']-l2),mse_conservative_bound=bound/len(r))


def distribution(v):
    a=np.asarray(v,dtype=np.float64);assert a.ndim==1 and len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,min=float(a.min()),max=float(a.max()),
        q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),
        positive=int((a>0).sum()),zero=int((a==0).sum()),negative=int((a<0).sum()))


def settings(family):
    n=8 if family=='generic_t140' else 5
    return [(f'noise_{j}',[j]) for j in range(n)]+[(f'mean_{k}',list(range(k))) for k in ([1,2,4,8] if n==8 else [1,2,4,5])]


def summaries(values,rows,order):
    image_output=[];patient_output=[];context=[];by_patient=defaultdict(list)
    for r in rows:by_patient[r['patient_id'],r['scenario']].append(r['image_id'])
    groups={r['patient_id']:r['assignment_group'] for r in rows}
    role={r['patient_id']:r['eval_role'] for r in rows}
    for family in ['generic_t140','cdi_dl_t100']:
        metrics=['negative_MSE_FP32','negative_MSE_FP64','negative_L2_FP64']
        if family=='cdi_dl_t100':metrics.append('negative_L2_FP32')
        for metric in metrics:
            field={'negative_MSE_FP32':'loss_mse_fp32','negative_MSE_FP64':'loss_mse_fp64',
                   'negative_L2_FP32':'loss_l2_fp32','negative_L2_FP64':'loss_l2_fp64'}[metric]
            for setting,js in settings(family):
                scores={}
                for row in rows:
                    iid=row['image_id'];pair={b:-float(np.mean([values[b,family,iid,'generic',j][field] for j in js])) for b in BRANCHES}
                    scores[iid]=pair
                    image_output.append(dict(family=family,metric=metric,setting=setting,**{k:row[k] for k in ['patient_id','image_id','scenario','eval_role','assignment_group']},
                        **pair,delta=pair['treatment']-pair['control']))
                for scenario in ['E','U']:
                    for pool in ['mean','max']:
                        records=[]
                        for p in order:
                            ids=by_patient[p,scenario];assert len(ids)==2
                            pair={b:float(np.mean([scores[i][b] for i in ids]) if pool=='mean' else max(scores[i][b] for i in ids)) for b in BRANCHES}
                            record=dict(family=family,metric=metric,setting=setting,scenario=scenario,pool=pool,patient_id=p,
                                eval_role=role[p],assignment_group=groups[p],**pair,delta=pair['treatment']-pair['control'])
                            records.append(record);patient_output.append(record)
                        for group in ['all','A','B']:
                            rr=[r for r in records if r['eval_role']=='selection' and (group=='all' or r['assignment_group']==group)]
                            assert len(rr)==(40 if group=='all' else 20)
                            context.append(dict(family=family,metric=metric,setting=setting,scenario=scenario,pool=pool,group=group,
                                delta=distribution([r['delta'] for r in rr])))
    mofit_cells=[];mofit_pooled=[]
    for precision in ['FP32','FP64']:
        field='loss_mse_'+precision.lower()
        for setting,js in [(f'noise_{j}',[j]) for j in range(5)]+[('fresh_mean_4',[1,2,3,4])]:
            for condition in ['null']+IMAGES:
                for metric in (['negative_MSE'] if condition=='null' else ['negative_MSE','h']):
                    image_scores={}
                    for iid in IMAGES:
                        pair={}
                        for b in BRANCHES:
                            v=np.array([values[b,'fixed_mofit',iid,condition,j][field] for j in js])
                            if metric=='h':v-=np.array([values[b,'fixed_mofit',iid,'null',j][field] for j in js])
                            else:v=-v
                            pair[b]=float(v.mean())
                        image_scores[iid]=pair
                        mofit_cells.append(dict(precision=precision,setting=setting,condition=condition,metric=metric,
                            query_image_id=iid,**pair,delta=pair['treatment']-pair['control']))
                    for scenario,ids in [('E',IMAGES[:2]),('U',IMAGES[2:])]:
                        for pool in ['mean','max']:
                            pair={b:float(np.mean([image_scores[i][b] for i in ids]) if pool=='mean' else max(image_scores[i][b] for i in ids)) for b in BRANCHES}
                            mofit_pooled.append(dict(precision=precision,setting=setting,condition=condition,metric=metric,
                                scenario=scenario,pool=pool,**pair,delta=pair['treatment']-pair['control']))
            # Patient-matched diagonal: each query uses its own original M1 embedding.
            for scenario,ids in [('E',IMAGES[:2]),('U',IMAGES[2:])]:
                for pool in ['mean','max']:
                    pair={}
                    for b in BRANCHES:
                        z=[float(np.mean([values[b,'fixed_mofit',i,i,j][field]-values[b,'fixed_mofit',i,'null',j][field] for j in js])) for i in ids]
                        pair[b]=float(np.mean(z) if pool=='mean' else max(z))
                    mofit_pooled.append(dict(precision=precision,setting=setting,condition='same_image_diagonal',metric='h',
                        scenario=scenario,pool=pool,**pair,delta=pair['treatment']-pair['control']))
    return dict(image_scores=image_output,patient_scores=patient_output,selection_context=context,
        patient14393=[r for r in patient_output if r['patient_id']==PID],
        fixed_mofit_cells=mofit_cells,fixed_mofit_patient_pools=mofit_pooled)


def run(args):
    started=time.perf_counter();out=args.run_dir
    assert args.expected_code_sha256 and sha(__file__)==args.expected_code_sha256
    assert sha(ROOT/'u_patient_audit/verify_fp32_noising.py')==NOISING_SHA
    p=read(out/'protocol.json');c=p['contract'];e=read(out/'execution.json')
    assert p['code_sha256']==PRODUCER_SHA and sha(ROOT/'u_patient_audit/run_masked_slot_endpoints_v2.py')==PRODUCER_SHA
    assert e['status']=='PASS_MASKED_ENDPOINT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION' and e['complete'] is True
    exact(c,read(p['contract_path']));assert sha(p['contract_path'])==p['contract_sha256']==e['contract_sha256']
    assert sha(out/'protocol.json')==e['protocol_sha256'] and sha(out/'results.json')==e['results_sha256']
    assert c['schema']=='masked-slot-endpoint-contract/v2'
    assert c['verification_binding']['analysis_policy_frozen_before_gpu'] is True
    assert c['verification_binding']['independent_verifier_code_frozen_before_gpu'] is False
    assert str(Path(__file__).resolve()) not in c['frozen_sha256']
    assert c['policy']['expected_all_cells']==4464 and c['policy']['expected_new_forward']==2260
    for f,h in c['frozen_sha256'].items():assert sha(f)==h,f
    dest=out/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    new(dest/'protocol.json',dict(schema='masked-endpoint-independent-verification/v1',analysis_policy=policy(),
        source_code_sha256=args.expected_code_sha256,extraction_contract_sha256=p['contract_sha256'],
        extraction_protocol_sha256=sha(out/'protocol.json'),extraction_results_sha256=sha(out/'results.json'),
        helper_sha256=NOISING_SHA,source_inputs_sha256=c['frozen_sha256'],
        code_timing='Analysis policy fixed before GPU; this independent verifier implementation finalized separately, not claimed pre-GPU frozen'))
    try:
        torch.set_num_threads(4)
        rows,order=scope();exact(order,c['patient_order']);exact([r['image_id'] for r in rows],c['image_order'])
        metadata={r['image_id']:r for r in rows};ids=set(metadata)
        states,training_report=training(c)
        ds,mapping,cross,crossindex,source_protocols=sources(c,ids)
        cache=torch.load(RUN/'cache/cache.pt',map_location='cpu',weights_only=True)
        assert cache['audit_prompt']==PROMPT;hidden=cache['hidden'][PROMPT].float()
        scheduler_paths=[f for f in c['frozen_sha256'] if Path(f).name=='scheduler_config.json']
        assert len(scheduler_paths)==1;sc=read(scheduler_paths[0])
        assert sc['beta_schedule']=='scaled_linear' and sc['trained_betas'] is None and sc['num_train_timesteps']==1000 and sc['prediction_type']=='epsilon'
        betas=torch.linspace(sc['beta_start']**.5,sc['beta_end']**.5,1000,dtype=torch.float32)**2
        alphas=torch.cumprod(1-betas,dim=0)
        source_packets={}
        def packet(kind,iid):
            key=kind,iid
            if key not in source_packets:
                row=mapping[kind][iid];path=ds[kind]/row['raw_path'];assert sha(path)==row['raw_sha256']
                source_packets[key]=torch.load(path,map_location='cpu',weights_only=True)
            return source_packets[key]
        exposure_patient={b:Counter() for b in BRANCHES}
        trainrows=csv_rows(RUN/'cohort/model_1_train.csv')
        for b in BRANCHES:
            for r in trainrows:exposure_patient[b][r['patient_id']]+=states[b]['exposures'].get(r['image_id'],0)
        results=read(out/'results.json');assert len(results)==4464
        grouped=defaultdict(list)
        for r in results:grouped[r['raw_path']].append(r)
        assert len(grouped)==696
        values={};check_counts=Counter();bounds=[];noising=[]
        for raw_path,rr in grouped.items():
            path=out/raw_path;assert path.resolve().is_relative_to(out.resolve()) and sha(path)==rr[0]['raw_sha256']
            raw=torch.load(path,map_location='cpu',weights_only=True);family=raw['family'];m=raw['metadata'];b=m['branch'];iid=m['image_id'];source=metadata[iid]
            assert b in BRANCHES and family in ['generic_t140','cdi_dl_t100','fixed_mofit']
            assert len(rr)==len(raw['cells'])==(8 if family=='generic_t140' else 5)
            for k in ['patient_id','image_id','scenario','eval_role','assignment_group','record_role']:assert m[k]==source[k]
            pex=exposure_patient[b][source['patient_id']];iex=states[b]['exposures'].get(iid,0)
            assert m['effective_patient_exposures']==pex and m['effective_image_exposures']==iex
            assert m['effective_patient_member']==int(pex>0) and m['effective_image_member']==int(iex>0)
            assert m['checkpoint_sha256']==c['checkpoint_sha256'][b] and m['base_pretraining_membership']=='unknown'
            condition=rr[0]['condition']
            if family!='fixed_mofit':
                assert condition=='generic';exact(raw['latent'],cache['latents'][iid].float());exact(raw['hidden'],hidden)
            else:
                assert iid in IMAGES and condition in ['null']+IMAGES
                exact(raw['latent'],cross['shared']['latents'][iid])
                exact(raw['hidden'],cross['shared']['null_hidden'] if condition=='null' else cross['shared']['embeddings']['model_1'][condition])
            for j,(q,r) in enumerate(zip(raw['cells'],rr)):
                assert q['repeat']==j==r['cell_index'] and r['raw_sha256']==rr[0]['raw_sha256'] and r['condition']==condition
                for k,v in m.items():exact(v,r[k])
                for k,v in q.items():
                    if not torch.is_tensor(v):exact(v,r[k])
                key=b,family,iid,condition,j;assert key not in values
                original_prediction=None;source_scalar=None
                if family=='generic_t140':
                    kind='prompt' if source['patient_id']==PID else 'repeat';old=packet(kind,iid);oldrow=mapping[kind][iid]
                    expected_seed,eps=noise(iid,j);assert q['seed']==expected_seed;exact(q['epsilon'],eps)
                    exact(raw['latent'],old['latent'])
                    assert th(hidden)==(old['prompt_hidden_sha256']['generic'] if kind=='prompt' else old['hidden_sha256'])
                    noising.append(verify_fp32_noising(q['noised_latent'],raw['latent'],eps,alphas))
                    if kind=='repeat':
                        exact(old['alphas'],alphas);oo=old['draws'][j];exact(q['noised_latent'],oo['noised_latent']);exact(eps,oo['epsilon'])
                        original_prediction=oo['prediction'];source_scalar=('loss_mse_fp32',oo['loss_mse_fp32'])
                    elif j==0:
                        exact(q['noised_latent'],old['noised_latent']);exact(eps,old['epsilon'])
                        original_prediction=old['predictions']['generic'];source_scalar=('loss_mse_fp32',oldrow['losses']['generic'])
                    sp=ds[kind]/oldrow['raw_path']
                elif family=='cdi_dl_t100':
                    kind='cdi_'+source['scenario'];old=packet(kind,iid);oldrow=mapping[kind][iid]
                    assert th(hidden)==old['hidden_sha256'];assert source_protocols[kind]['contract']['prompt']==PROMPT
                    exact(old['alphas'],alphas);exact(raw['latent'],old['latent'])
                    oo=old['predictions']['denoising_loss'][j];n=old['noise_draws']['denoising_loss'][j]
                    assert oo['timestep']==100 and q['seed']==n['seed'];exact(q['epsilon'],n['noise']);exact(q['noised_latent'],oo['input'])
                    pc=source_protocols[kind]['contract'];payload=f"cdi-v1|{pc['master_seed']}|{iid}|{pc['stream']}|denoising_loss|{j}"
                    seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')%(2**63-1)
                    assert seed==q['seed'];exact(q['epsilon'],torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(seed),dtype=torch.float32))
                    a=float(alphas[100]);exact(q['noised_latent'],math.sqrt(a)*raw['latent']+math.sqrt(1-a)*q['epsilon'])
                    original_prediction=oo['epsilon'];source_scalar=('loss_l2_fp32',float(old['module_outputs']['denoising_loss'][0,j,0]))
                    assert float(old['module_outputs']['denoising_loss'].mean())==oldrow['features'][0]
                    sp=ds[kind]/oldrow['raw_path']
                else:
                    oldrow=crossindex[iid,condition,j];oo=cross['cells'][oldrow['cell_id']]
                    assert q['seed']==oldrow['noise_seed'];exact(q['epsilon'],oo['target']);exact(q['noised_latent'],oo['noised_latent'])
                    noising.append(verify_fp32_noising(q['noised_latent'],raw['latent'],q['epsilon'],alphas))
                    original_prediction=oo['prediction'];source_scalar=('loss_mse_fp32',oldrow['loss'])
                    sp=ds['cross']/'raw.pt'
                assert Path(q['source_path']).resolve()==sp.resolve() and q['source_raw_sha256']==c['frozen_sha256'][str(sp.resolve())]
                reuse=b=='treatment' and original_prediction is not None
                assert q['reused'] is reuse and q['forward']==int(not reuse) and q['backward']==0
                if reuse:
                    exact(q['prediction'],original_prediction,'exact reused prediction');assert q[source_scalar[0]]==source_scalar[1]
                assert q['mse_fp32_reduction']==('CPU secondary' if family=='cdi_dl_t100' else 'source-matched GPU primary')
                assert q['l2_fp32_reduction']==('source-matched GPU primary' if family=='cdi_dl_t100' else 'CPU secondary')
                mse,l2,bb=arithmetic(q);bounds.append(bb)
                residual=q['prediction']-q['epsilon']
                if family=='cdi_dl_t100':assert q['loss_mse_fp32']==float(residual.square().mean())
                else:assert q['loss_l2_fp32']==float(torch.norm(residual.reshape(1,-1),p=2,dim=-1)[0])
                values[key]={k:q[k] for k in ['loss_mse_fp32','loss_l2_fp32','loss_mse_fp64','loss_l2_fp64']}
                check_counts['cells']+=1;check_counts['reused']+=int(reuse);check_counts['forward']+=q['forward'];check_counts[family]+=1
        expected={(b,f,i,'generic',j) for b in BRANCHES for f,n in [('generic_t140',8),('cdi_dl_t100',5)] for i in ids for j in range(n)}
        expected|={(b,'fixed_mofit',i,s,j) for b in BRANCHES for i in IMAGES for s in ['null']+IMAGES for j in range(5)}
        assert set(values)==expected and len(expected)==4464
        assert check_counts['reused']==2204 and check_counts['forward']==2260
        for f,v in [('records',4464),('new_forward',2260),('backward',0),('vae_forward',0),('reused_cells',2204),('patients',41),('unique_images',164)]:assert e[f]==v
        assert all(e[k] is False for k in ['new_target_training','new_optimization','new_fitting','AUC_computed','stage2_completion_claim'])
        assert e['frozen_inputs_unchanged'] is True
        fingerprints=expected_fingerprints(c,states);assert len(e['model_reports'])==2
        for report,b in zip(e['model_reports'],BRANCHES):
            assert report['branch']==b and report['counts']==dict(forward=28 if b=='treatment' else 2232,backward=0)
            for k,v in fingerprints[b].items():assert report['initial_state'][k]==report['final_state'][k]==v
            assert all(report[k] is True for k in ['all_parameters_frozen','no_parameter_gradients','parameter_versions_unchanged','actual_checkpoint_adapter_exact'])
            assert report['unet_training'] is False
        analysis=summaries(values,rows,order)
        analysis.update(status='PASS_FIXED_SINGLE_PATIENT_MASK_ENDPOINT_DESCRIPTIVE_SUMMARIES',policy=policy())
        new(dest/'analysis.json',analysis)
        for f,h in c['frozen_sha256'].items():assert sha(f)==h
        verification=dict(status=STATUS,complete=True,seconds=time.perf_counter()-started,
            source_code_sha256=args.expected_code_sha256,results_sha256=sha(out/'results.json'),
            protocol_sha256=sha(out/'protocol.json'),execution_sha256=sha(out/'execution.json'),
            contract_sha256=p['contract_sha256'],verification_protocol_sha256=sha(dest/'protocol.json'),
            analysis_path=str((dest/'analysis.json').resolve()),analysis_sha256=sha(dest/'analysis.json'),
            counts=dict(check_counts),paired_cells=2232,training_gates=training_report,expected_full_FP32_weight_fingerprints=fingerprints,
            noising=dict(scheduler_FP32_exact_cells=len(noising),CDI_literal_and_source_exact_cells=1640,
                max_error_to_operand_scaled_bound=max(x['max_error_to_rounding_bound_ratio'] for x in noising)),
            loss_reconstruction=dict(max_MSE_FP32_vs_FP64_error=max(x['mse_absolute_error'] for x in bounds),
                max_L2_FP32_vs_FP64_error=max(x['l2_absolute_error'] for x in bounds),
                FP32_check='explicit conservative rounding bound plus exact source replay/CPU secondary reducer',FP64_check='independent math.fsum residual squares'),
            limitations=policy()['limitations'],CUDA_initialized=torch.cuda.is_initialized())
        assert verification['CUDA_initialized'] is False
        new(dest/'verification.json',verification);new(out/'verification.json',verification)
        print(json.dumps(dict(status=STATUS,seconds=verification['seconds'],cells=4464)))
    except BaseException:
        new(dest/'failure.json',dict(status='FAILED_PRESERVED_MASKED_ENDPOINT_CPU_VERIFICATION',traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
        raise


def self_test():
    a=distribution([-2.,0.,1.,3.]);assert a['mean']==.5 and a['positive']==2 and a['negative']==1
    p=torch.linspace(-1,1,4096).reshape(1,4,32,32);n=torch.flip(p,[-1]);r=p-n;d=p.double()-n.double()
    q=dict(prediction=p,epsilon=n,noised_latent=n,loss_mse_fp32=float(r.square().mean()),loss_l2_fp32=float(torch.norm(r)),
        loss_mse_fp64=float(d.square().mean()),loss_l2_fp64=float(torch.linalg.vector_norm(d)))
    arithmetic(q)
    q['loss_mse_fp32']+=.1
    try:arithmetic(q)
    except AssertionError:pass
    else:raise AssertionError('corrupt residual scalar accepted')
    # Exact source arithmetic is distinct from attack direction; max(-loss)=-min(loss).
    assert max([-1.,-3.])==-1. and max([-1.,-3.])-max([-2.,-2.])==1.
    assert .5*((5.-1.)+(3.-2.))==2.5
    # Full policy path on synthetic scalars: sign precedes max, and every fixed
    # family/seed/prefix/precision/condition is emitted, with no data-driven choice.
    rows=[];order=[PID]+[str(90000+i) for i in range(40)];values={}
    for pi,pid in enumerate(order):
        for ii in range(4):
            iid=IMAGES[ii] if pi==0 else f'{pid}_{ii}.png'
            row=dict(patient_id=pid,image_id=iid,scenario='E' if ii<2 else 'U',
                eval_role='fit' if pi==0 else 'selection',assignment_group='A' if pi<=20 else 'B')
            rows.append(row)
            for family,n in [('generic_t140',8),('cdi_dl_t100',5)]:
                for b in BRANCHES:
                    for j in range(n):
                        value=float(1+pi+ii+j+(b=='control'))
                        values[b,family,iid,'generic',j]={f'loss_{loss}_{prec}':value for loss in ['mse','l2'] for prec in ['fp32','fp64']}
    for b in BRANCHES:
        for ii,iid in enumerate(IMAGES):
            for si,s in enumerate(['null']+IMAGES):
                for j in range(5):
                    value=float(2+ii+si+j+(b=='control')*(si+1))
                    values[b,'fixed_mofit',iid,s,j]={f'loss_{loss}_{prec}':value for loss in ['mse','l2'] for prec in ['fp32','fp64']}
    result=summaries(values,rows,order)
    assert len(result['image_scores'])==11808 and len(result['patient_scores'])==11808
    assert len(result['patient14393'])==288 and len(result['selection_context'])==864
    assert len(result['fixed_mofit_cells'])==432 and len(result['fixed_mofit_patient_pools'])==480
    assert all(x['delta']==1. for x in result['patient_scores'])
    assert all(x['delta']['mean']==1. for x in result['selection_context'])
    first=next(x for x in result['patient14393'] if x['family']=='generic_t140' and x['metric']=='negative_MSE_FP32'
               and x['setting']=='noise_0' and x['scenario']=='E' and x['pool']=='max')
    assert first['treatment']==-1. and first['control']==-2.
    return dict(status='PASS_MASKED_ENDPOINT_CPU_ARITHMETIC_AND_POLICY_TESTS',CUDA_initialized=torch.cuda.is_initialized())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=OUT)
    p.add_argument('--output-tag',default='analysis_v1');p.add_argument('--expected-code-sha256')
    p.add_argument('--self-test',action='store_true');p.add_argument('--write-policy',type=Path)
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    if a.write_policy:
        new(a.write_policy,dict(policy=policy(),analyzer_sha256=sha(__file__),producer_sha256=PRODUCER_SHA,noising_helper_sha256=NOISING_SHA));return
    run(a)


if __name__=='__main__':main()
