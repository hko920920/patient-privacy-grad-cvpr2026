"""Independent CPU saved-tensor checks and fixed MoFit crossing arithmetic."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,RUN,digest,read_csv
from .models import scheduler

BASE=RUN/'baseline_screen_20260915'
OUT=BASE/'mofit_cross_response_v1'
CONTRACT=BASE/'mofit_cross_response_contract_v1.json'
RESEARCH=next(ROOT.parent.glob('*/research_2026-09-10'))
FREEZE=RESEARCH/'spec_sources/mofit_cross_response_analysis_freeze_20260915.json'
MODELS=('model_1','model_2')
IMAGES=('00014393_002.png','00014393_006.png','00014393_001.png','00014393_004.png')
SEEDS=(260915,26091501,26091502,26091503,26091504)
PRIOR=(BASE/'mofit_medical_full_v1',BASE/'mofit_medical_target2_full_v1',BASE/'mofit_medical_remaining_pair_v1')
STATUSES=('PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE',
          'PASS_TARGET2_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_NONMEMBERSHIP',
          'PASS_REMAINING_PAIR_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_MEMBERSHIP')
EXPECTED_CONTRACT_SHA='29638fa556e28056167b2481574124b76740a5eb0bc1bceedbaa3cc9abefa3df'
EXPECTED_PRODUCER_SHA='99b9dd67366a7f372d2f860198f617546ee0224256cd7400cca9dc6c48995dea'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def need(v,s):
    if not v:raise AssertionError(s)
def thash(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def close(a,b,s,rtol=2e-12,atol=2e-12):
    a,b=np.asarray(a),np.asarray(b)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a,b,rtol=rtol,atol=atol),s)
def tensor(v,shape):
    need(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.device.type=='cpu'
         and tuple(v.shape)==shape and torch.isfinite(v).all(),'finite CPU tensor shape/dtype')
def cid(m,i,sm,si,r):return f'{m}|{i}|{sm or "null"}|{si or "null"}|{r}'
def policy():
    return dict(schema='mofit-cross-response-analysis-policy/v1',current_step=2,
        patient_id='14393',models=list(MODELS),images=list(IMAGES),noise_seeds=list(SEEDS),
        matrix='All4x4 query/source images for each recipient/source model, noises0..4 and fixed fresh_mean4',
        noise_mean_order='For fresh_mean4, average each image/source/recipient h cell over noises1..4 first; then image pooling',
        diagonal_units='Every individual image plus E_mean/E_max/U_mean/U_max',
        max_order='Pool the two same-image 2x2 cell matrices elementwise; then decompose, never max a difference',
        decomposition=dict(A='h(M1,e1)',B='h(M1,e2)',C='h(M2,e1)',D='h(M2,e2)',
            recipient='.5*((A-C)+(B-D))',embedding='.5*((A-B)+(C-D))',sum='A-D'),
        precision='Main h from verified saved FP32 MSE; parallel saved-prediction FP64 MSE decomposition retained',
        replays='All8 original optimized and8 null, exact/close/error separately; original producer tolerances1e-5/1e-6',
        original_source='after Adam update299, complete77x1024 embedding and saved original posterior sample',
        limits=['Only one patient, same-patient cross images, two whole-cohort targets',
            'No population AUC/CI, no individual-patient causal inference',
            'New-noise test holds embeddings fixed, not noise-specific MoFit reoptimization',
            'Symmetric recipient/embedding decomposition is an algebraic allocation, not identified causal effects',
            'Cross-image persistence does not establish patient specificity',
            'No favorable noise, matrix cell, score direction, gamma or condition selection',
            'No stage2 completion or novel method claim'],
        AUC=False,CI=False,new_fitting=False,new_optimization=False,new_GPU_calls=0)

def prepare():
    need(not FREEZE.exists(),'immutable analysis freeze')
    need(digest(CONTRACT)==EXPECTED_CONTRACT_SHA,'fixed producer contract')
    producer=ROOT/'u_patient_audit/run_mofit_cross_response.py'
    need(digest(producer)==EXPECTED_PRODUCER_SHA,'fixed producer source')
    c=read(CONTRACT)
    f=dict(schema='mofit-cross-response-analysis-freeze/v1',analysis_source_sha256=digest(__file__),
        producer_source_sha256=EXPECTED_PRODUCER_SHA,extraction_contract_sha256=EXPECTED_CONTRACT_SHA,
        extraction_contract_path=str(CONTRACT),policy=policy(),
        original_source_bindings=c['original_source_bindings'],
        before_new_GPU_results=True)
    write(FREEZE,f)
    print(json.dumps(dict(status='ANALYSIS_FROZEN_BEFORE_GPU',source_sha256=digest(__file__),freeze_sha256=digest(FREEZE))))

def load_sources(c):
    packets={};rows={}
    for k,d in enumerate(PRIOR):
        v,e=read(d/'verification.json'),read(d/'execution.json')
        need(v['status']==STATUSES[k] and e['complete'] is True,'verified prior source')
        for n in ('results','raw','protocol'):
            p=d/(n+('.pt' if n=='raw' else '.json'))
            need(v[n+'_sha256']==e[n+'_sha256']==digest(p),'prior complete source checksum')
        raw=torch.load(d/'raw.pt',map_location='cpu',weights_only=True)
        for row in read(d/'results.json'):
            m,i=row['model'],row['image_id'];key=(m,i)
            need(key not in packets and m in MODELS and i in IMAGES,'unique source8 identity')
            q=raw[m][i] if k==2 else raw[i]
            need(row['stage1_steps']==1000 and row['stage2_steps']==300,'full source optimization')
            tensor(q['final']['embedding'],(1,77,1024))
            need(q['embedding_updates'][-1]['iteration']==299
                and torch.equal(q['final']['embedding'],q['embedding_updates'][-1]['after'])
                and thash(q['final']['embedding'])==q['traces']['embedding'][-1]['after_sha256'],'post-update embedding source')
            for vname in ('u','h','v'):
                need(row[vname]==q['final']['original_scores'][vname]==q['traces']['embedding'][-1][vname],'source scalar phase')
            tensor(q['posterior']['original']['scaled_sample'],(1,4,32,32))
            for name in ('null_cell','optimized_cell'):
                cell=q['final']['original_scores'][name]
                need(torch.equal(q['posterior']['original']['scaled_sample'],cell['scaled_latent']),'original source latent')
                need(torch.equal(q['draws']['diffusion_noise'],cell['target']),'original source diffusion noise')
            packets[key]=q;rows[key]=row
    need(len(packets)==8,'all8 embeddings')
    for p,h in c['original_source_bindings'].items():need(digest(p)==h,'unchanged source binding')
    return packets,rows

def agreement(a,b):
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'replay tensor shape')
    diff=np.abs(a.astype(float)-b.astype(float));den=max(float(np.linalg.norm(b.astype(float).ravel())),1e-30)
    return dict(exact=bool(np.array_equal(a,b)),close=bool(np.all(diff<=1e-6+1e-5*np.abs(b))),
        max_absolute_error=float(diff.max()),relative_l2_error=float(np.linalg.norm(diff.ravel())/den))

def verify():
    start=time.perf_counter();torch.set_num_threads(2)
    f=read(FREEZE);c=read(CONTRACT)
    need(f['analysis_source_sha256']==digest(__file__) and f['policy']==policy(),'analysis freeze binding')
    need(f['extraction_contract_sha256']==digest(CONTRACT)==EXPECTED_CONTRACT_SHA,'fixed extraction contract')
    need(digest(ROOT/'u_patient_audit/run_mofit_cross_response.py')==f['producer_source_sha256']==EXPECTED_PRODUCER_SHA,'frozen producer')
    p,e=read(OUT/'protocol.json'),read(OUT/'execution.json')
    need(p['contract']==c and p['contract_sha256']==e['contract_sha256']==EXPECTED_CONTRACT_SHA,'executed contract')
    for n,suffix in (('raw','.pt'),('results','.json'),('protocol','.json'),('original_replay','.json')):
        need(e[n+'_sha256']==digest(OUT/(n+suffix)),'execution file hash '+n)
    need(e['complete'] is True and e['records']==360 and e['optimized_records']==320 and e['null_records']==40
        and e['forward']==360 and e['backward']==0 and e['vae_forward']==0 and e['original_replay_records']==16,'complete fixed counts')
    need(c['policy']['noise_seeds']==list(SEEDS) and c['policy']['image_order']==list(IMAGES)
        and c['policy']['model_order']==list(MODELS) and c['policy']['timestep']==140,'fixed scope')
    for filename,h in c['frozen_sha256'].items():need(digest(filename)==h,'frozen file '+filename)
    source,source_rows=load_sources(c)
    raw=torch.load(OUT/'raw.pt',map_location='cpu',weights_only=True)
    shared=raw['shared'];rows=read(OUT/'results.json');ix={r['cell_id']:r for r in rows}
    expected={cid(m,i,sm,si,r) for m in MODELS for i in IMAGES for r in range(5)
              for sm,si in [(None,None)]+[(sm,si) for sm in MODELS for si in IMAGES]}
    need(len(rows)==len(ix)==360 and set(ix)==set(raw['cells'])==expected,'exact360 cell identities')
    alpha=float(scheduler().alphas_cumprod[140]);need(alpha==c['alpha_cumprod_t140'],'scheduler alpha')
    embedding_checks=[];source_latents={}
    for i in IMAGES:
        tensor(shared['latents'][i],(1,4,32,32))
        for m in MODELS:
            q=source[m,i]
            need(torch.equal(shared['latents'][i],q['posterior']['original']['scaled_sample']),'same original posterior across targets')
            need(torch.equal(shared['embeddings'][m][i],q['final']['embedding']),'actual saved final embedding')
            need(torch.equal(shared['null_hidden'],q['null_hidden']),'same null hidden')
            embedding_checks.append(dict(model=m,image_id=i,after_update299=True,
                embedding_sha256=thash(shared['embeddings'][m][i]),latent_sha256=thash(shared['latents'][i])))
    for r,seed in enumerate(SEEDS):
        noise=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
        need(torch.equal(shared['noises'][str(r)],noise),'independent CPU noise replay')
        if r==0:
            for q in source.values():need(torch.equal(q['draws']['diffusion_noise'],noise),'original noise')
    metadata={}
    for m in MODELS:
        cp=RUN/'training_coverage_v2'/m/'step_1000.pt';st=torch.load(cp,map_location='cpu',weights_only=True)
        train=read_csv(RUN/'cohort'/(m+'_train.csv'))
        need(digest(cp)==c['checkpoint_sha256'][m] and st['step']==1000 and sum(st['exposures'].values())==4000,'actual target checkpoint')
        need(set(st['exposures'])=={r['image_id'] for r in train},'actual target train manifest')
        own=[r for r in train if r['patient_id']=='14393'];member=int(bool(own))
        patient_ex=sum(st['exposures'][r['image_id']] for r in own)
        need(member==int(m=='model_1') and patient_ex==8*member,'actual patient exposure')
        for i in IMAGES:
            n=st['exposures'].get(i,0);s='E' if i in IMAGES[:2] else 'U'
            need(n==(4 if member and s=='E' else 0),'actual image exposure')
            md=dict(patient_id='14393',eval_role='fit',assignment_group='A',scenario=s,
                member=member,image_member=int(n>0),actual_training_exposures=n,patient_training_exposures=patient_ex)
            need(c['metadata'][m][i]==md,'contract actual metadata');metadata[m,i]=md
    mse64={};max_mse_error=0.
    for key,row in ix.items():
        m,i,sm,si,r=row['recipient_model'],row['query_image_id'],row['source_model'],row['source_image_id'],row['noise_index']
        need(key==cid(m,i,sm,si,r) and row['noise_seed']==SEEDS[r],'cell identity fields')
        need(row['condition']==('null' if sm is None else 'optimized'),'conditioning label')
        need(row['query_metadata']==metadata[m,i] and row['source_metadata']==(metadata[sm,si] if sm else None),'recipient/source metadata')
        need(row['query_scenario']==metadata[m,i]['scenario'] and row['source_scenario']==(metadata[sm,si]['scenario'] if sm else None),'scenario semantics')
        need(row['forward']==1 and row['backward']==0,'cell counts')
        cell=raw['cells'][key];noise=shared['noises'][str(r)];latent=shared['latents'][i]
        for v in ('prediction','noised_latent','target'):tensor(cell[v],(1,4,32,32))
        need(torch.equal(cell['target'],noise),'cell target fixed noise')
        expected_noisy=math.sqrt(alpha)*latent.double().numpy()+math.sqrt(1-alpha)*noise.double().numpy()
        close(cell['noised_latent'].numpy(),expected_noisy,'FP64 independent noising',rtol=2e-6,atol=1e-7)
        residual=cell['prediction'].double().numpy()-noise.double().numpy()
        loss=float(np.mean(residual*residual));mse64[key]=loss
        need(row['loss']==cell['loss'],'stored scalar/cell agreement')
        close(cell['loss'],loss,'FP64 residual MSE',rtol=2e-6,atol=1e-7)
        max_mse_error=max(max_mse_error,abs(cell['loss']-loss))
        null=ix[cid(m,i,None,None,r)]['loss']
        need(row['null_loss']==null and row['h']==row['loss']-null,'exact same-recipient null subtraction')
    replay=read(OUT/'original_replay.json');rx={r['cell_id']:r for r in replay}
    expected_replay={cid(m,i,sm,si,0) for m in MODELS for i in IMAGES for sm,si in ((None,None),(m,i))}
    need(len(replay)==len(rx)==16 and set(rx)==expected_replay,'all16 original replays')
    rep=[]
    for key in sorted(expected_replay):
        row=ix[key];m,i=row['recipient_model'],row['query_image_id'];null=row['condition']=='null'
        old_key='null_cell' if null else 'optimized_cell';old=source[m,i]['final']['original_scores'][old_key]
        need(rx[key]['original_cell']==old_key,'replay source phase')
        actual={}
        for name in ('prediction','noised_latent','target'):
            actual[name]=agreement(raw['cells'][key][name].numpy(),old[name].numpy())
        actual['loss']=agreement(row['loss'],old['loss'])
        actual['h']=agreement(row['h'],0. if null else source_rows[m,i]['h'])
        for name,ag in actual.items():
            saved=rx[key]['comparisons'][name]
            need(saved['exact']==ag['exact'] and saved['close']==ag['close'],'independent replay flags')
            close(saved['max_absolute_error'],ag['max_absolute_error'],'replay max error')
            close(saved['relative_l2_error'],ag['relative_l2_error'],'replay norm error')
            need(ag['close'],'fixed FP32 replay tolerance')
        rep.append(dict(cell_id=key,condition=row['condition'],model=m,image_id=i,
            original_h=0. if null else source_rows[m,i]['h'],replayed_h=row['h'],
            raw_h_exact=row['h']==(0. if null else source_rows[m,i]['h']),comparisons=actual))
    need(len(e['model_reports'])==2 and {r['model'] for r in e['model_reports']}==set(MODELS),'two model reports')
    for r in e['model_reports']:
        m=r['model']
        need(r['before_fingerprint']==r['after_fingerprint']==c['prior_unet_fingerprints'][m],'unchanged target state')
        need(r['forward_calls']==r['forward_examples']==180 and r['backward']==0 and not r['model_training'],'counts/model mode')
        for k in ('checkpoint_adapter_exact','adapter_tensors_unchanged','parameter_versions_unchanged',
                  'all_parameters_frozen','no_parameter_gradients'):need(r[k] is True,'model guard '+k)
    for filename,h in c['frozen_sha256'].items():need(digest(filename)==h,'inputs unchanged after verification')
    v=dict(status='PASS_CROSS_RESPONSE_SAVED_ARITHMETIC_SOURCE_AND_REPLAYS',predictions_verified=360,
        optimized_records=320,null_records=40,source_embeddings=embedding_checks,original_replays=rep,
        original_optimized_h_bitwise_exact_count=sum(r['raw_h_exact'] for r in rep if r['condition']=='optimized'),
        original_null_h_exact_count=sum(r['raw_h_exact'] for r in rep if r['condition']=='null'),
        max_FP32_FP64_MSE_difference=max_mse_error,new_GPU_calls=0,
        independent_neural_forward_replay=False,FP64_scope='CPU residual reduction of saved FP32 prediction',
        scientific_efficacy_pass=False,results_sha256=digest(OUT/'results.json'),raw_sha256=digest(OUT/'raw.pt'),
        protocol_sha256=digest(OUT/'protocol.json'),execution_sha256=digest(OUT/'execution.json'),
        source_sha256=digest(__file__),freeze_sha256=digest(FREEZE),seconds_cpu=time.perf_counter()-start)
    write(OUT/'verification.json',v)
    return ix,mse64,v

def decompose(matrix):
    A,B,C,D=np.asarray(matrix,float).reshape(-1).tolist()
    recipient=.5*((A-C)+(B-D));embedding=.5*((A-B)+(C-D));total=A-D
    close(recipient+embedding,total,'exact symmetric sum decomposition')
    return dict(A=A,B=B,C=C,D=D,recipient_component=recipient,embedding_component=embedding,
        A_minus_D=total,component_sum=recipient+embedding,arithmetic_residual=recipient+embedding-total)

def summarize(ix,mse64):
    H=np.empty((5,2,2,4,4));H64=np.empty_like(H)
    for r in range(5):
        for mi,m in enumerate(MODELS):
            for smi,sm in enumerate(MODELS):
                for ii,i in enumerate(IMAGES):
                    for sii,si in enumerate(IMAGES):
                        key=cid(m,i,sm,si,r);null=cid(m,i,None,None,r)
                        H[r,mi,smi,ii,sii]=ix[key]['h']
                        H64[r,mi,smi,ii,sii]=mse64[key]-mse64[null]
    variants=[('noise_'+str(r),[r],H[r],H64[r]) for r in range(5)]
    variants += [('fresh_mean4',[1,2,3,4],H[1:].mean(0),H64[1:].mean(0))]
    matrices=[];decompositions=[]
    for setting,draws,h,h64 in variants:
        for mi,m in enumerate(MODELS):
            for smi,sm in enumerate(MODELS):
                matrices.append(dict(setting=setting,noise_indices=draws,recipient_model=m,source_model=sm,
                    query_image_order=list(IMAGES),source_image_order=list(IMAGES),
                    h_matrix=h[mi,smi].tolist(),saved_prediction_FP64_h_matrix=h64[mi,smi].tolist()))
        units=[(i,[j],'image') for j,i in enumerate(IMAGES)]
        units += [(s+'_'+pool,inds,pool) for s,inds in (('E',[0,1]),('U',[2,3])) for pool in ('mean','max')]
        for unit,inds,pool in units:
            diag=np.stack([h[:,:,j,j] for j in inds],axis=0)
            diag64=np.stack([h64[:,:,j,j] for j in inds],axis=0)
            a=diag.max(0) if pool=='max' else diag.mean(0)
            b=diag64.max(0) if pool=='max' else diag64.mean(0)
            row=dict(setting=setting,noise_indices=draws,unit=unit,query_images=[IMAGES[j] for j in inds],
                pool=pool,cell_matrix=a.tolist(),**decompose(a),FP64_residual_decomposition=decompose(b))
            if pool=='max':
                row['note']='max each A/B/C/D cell over the two images, then decompose'
            decompositions.append(row)
    need(len(matrices)==24 and len(decompositions)==48,'all fixed matrices/decompositions')
    return matrices,decompositions

def analyze(ix,mse64,v):
    out=OUT/'analysis_v1';out.mkdir(exist_ok=False)
    matrices,decomp=summarize(ix,mse64)
    result=dict(status='PASS_FIXED_CROSS_RESPONSE_DECOMPOSITION',policy=policy(),matrices=matrices,
        same_image_and_pooled_decompositions=decomp,original_replays=v['original_replays'],
        all_noise_conditions_retained=True,best_condition_selected=False,AUC_computed=False,
        CI_computed=False,individual_causal_claim=False,stage2_completion_claim=False)
    write(out/'protocol.json',dict(policy=policy(),source_sha256=digest(__file__),freeze_sha256=digest(FREEZE),
        extraction_contract_sha256=EXPECTED_CONTRACT_SHA,raw_verification_sha256=digest(OUT/'verification.json')))
    write(out/'analysis.json',result)
    write(out/'provenance.json',dict(source_sha256=digest(__file__),freeze_sha256=digest(FREEZE),
        input_sha256={n:digest(OUT/n) for n in ('protocol.json','results.json','raw.pt','execution.json','original_replay.json','verification.json')},
        output_sha256={n:digest(out/n) for n in ('protocol.json','analysis.json')}))
    print(json.dumps(dict(status=result['status'],matrices=24,decompositions=48,
        original_h_exact=v['original_optimized_h_bitwise_exact_count'],CPU_verify_seconds=v['seconds_cpu'])))

def self_test():
    # Noncommuting max control: max(A-D) must not replace max(A)-max(D).
    two=np.array([[[1.,5.],[2.,3.]],[[4.,2.],[6.,1.]]])
    d=decompose(two.max(0))
    need(d['A_minus_D']==1 and np.max(two[:,0,0]-two[:,1,1])==3,'max ordering')
    need(decompose([[1.,3.],[2.,-1.]])['A_minus_D']==2,'basic diagonal difference')
    need(agreement([1.],[1.])['exact'] and not agreement([1.],[2.])['close'],'independent replay check')
    # Entire reporting path with identifiable synthetic 360-cell matrix.
    ix={};mse={}
    for r in range(5):
        for mi,m in enumerate(MODELS):
            for ii,i in enumerate(IMAGES):
                n=cid(m,i,None,None,r);mse[n]=1.
                for smi,sm in enumerate(MODELS):
                    for sii,si in enumerate(IMAGES):
                        k=cid(m,i,sm,si,r);h=.1*mi+.01*smi+.001*ii+.0001*sii+.00001*r
                        ix[k]={'h':h};mse[k]=1+h
    matrices,decomp=summarize(ix,mse)
    need(len(matrices)==24 and len(decomp)==48,'complete synthetic reporting')
    import builtins,symtable
    missing=[]
    def walk(scope):
        if scope.get_type()=='function':
            for s in scope.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in globals() and not hasattr(builtins,s.get_name()):
                    missing.append((scope.get_name(),s.get_name()))
        for child in scope.get_children():walk(child)
    walk(symtable.symtable(Path(__file__).read_text(encoding='utf8'),__file__,'exec'))
    need(not missing,'global dependencies '+repr(missing))
    print('PASS_SYNTHETIC_360_CELL_REPORTING_MAX_ORDER_DECOMPOSITION_DEPENDENCIES')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-freeze',action='store_true');g.add_argument('--self-test',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--expected-code-sha256');p.add_argument('--expected-freeze-sha256');a=p.parse_args()
    if a.self_test:self_test()
    elif a.prepare_freeze:prepare()
    else:
        need(a.expected_code_sha256==digest(__file__) and a.expected_freeze_sha256==digest(FREEZE),'externally bound analysis freeze')
        ix,mse,v=verify();analyze(ix,mse,v)

