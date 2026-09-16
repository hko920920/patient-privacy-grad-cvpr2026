"""Saved tensor verification and predeclared repeated-noise analysis, CPU only."""
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .common import ROOT,RUN,digest,read_csv
from .models import scheduler
from .verify_fp32_noising import verify_fp32_noising, self_test as noising_self_test
ORIGINAL_SHA='9988ca024ed3203a578c3c1a85427931c93950196078e808937d031f6bf5d99c'
NOISING_HELPER_SHA='ba710cb9d80fb92e6e70b4b850333df4d5087c12b855e425bfda7b271c94c195' 

OLD=RUN/'baseline_screen_20260914/prompt_loss_t140_v1'
OUT=RUN/'baseline_screen_20260915/repeat_noise_t140_v1'
MODELS=('model_1','model_2')
SCENARIOS=('E','U')
METRICS=('MSE_FP32','L2_FP64_saved_prediction')
POOLS=('mean','max')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def need(ok,label):
    if not ok:raise AssertionError(label)
def close(a,b,label,rtol=2e-12,atol=2e-12):
    need(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=rtol,atol=atol),label)
def thash(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rho(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).statistic)
def policy_seed(i,r):
    text=f'promptloss-v1|260915|{i}|140' if r==0 else f'promptloss-repeat-v1|260915|{i}|140|{r}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%(2**63-1)
def analysis_policy():
    return dict(current_step=2,scope='Fixed selection40 t140 generic loss noise diagnostic',
        repeats=list(range(8)),prefix_means=[1,2,4,8],metrics=list(METRICS),patient_pools=list(POOLS),
        primary='negative MSE, two-image patient mean; report all declared alternatives',
        ordering='For each image average losses over the declared repeats, negate, then patient mean/max',
        bootstrap=dict(reuse=str(OLD/'analysis_v1/bootstrap.json'),resamples=2000,seed=260914,
            units='patient stratified20A20B; same indices for models/scenarios/seeds/contrasts',
            uncertainty='conditional on fixed8 noise draws and development patients; not a noise-population CI'),
        AUC_policy='All8 individual draws plus prefix1/2/4/8; each prefix contrast against prefix1',
        reliability='All28 seed-pair Spearman; disjoint1/2/4 draw blocks; descriptive one-way ICC(1,1)',
        paired_change='Same image/repeat noise across targets; score_when_member-score_when_nonmember per patient',
        no_best_seed=True,no_best_prefix=True,new_fitting=False,new_target_training=False,
        guardrails=['selection40 already used for development, not an untouched test',
            'The coupled targets change whole A/B training groups; no individual causal effect',
            'MC averaging is an existing baseline, not new patient-structure novelty',
            'A positive result does not diagnose all CDI features or full MoFit optimization',
            'A negative result does not establish absent leakage',
            'SecMI/PIA have no external random noise draw in this implementation',
            'DL t100 already averages5 independent noise norms; this test is t140 generic MSE/L2',
            'L2 is double reduction of saved FP32 predictions, not FP64 neural inference',
            'No 1%FPR claim with20 negatives; no overall stage2 gate'],
        overall_stage2_gate=None)

def verify(directory):
    start=time.perf_counter();torch.set_num_threads(2)
    protocol=read(directory/'protocol.json');c=protocol['contract'];e=read(directory/'execution.json')
    need(c['analysis_code_sha256']==ORIGINAL_SHA and digest(ROOT/'u_patient_audit/analyze_verify_repeat_noise_t140.py')==ORIGINAL_SHA,'original frozen verifier source preserved')
    need(digest(ROOT/'u_patient_audit/verify_fp32_noising.py')==NOISING_HELPER_SHA,'bound FP32 noising helper')
    need(c['analysis_policy']==analysis_policy(),'predeclared complete analysis policy')
    need(digest(protocol['contract_path'])==protocol['contract_sha256']==e['contract_sha256'],'contract binding')
    need(read(protocol['contract_path'])==c,'embedded contract')
    for p,h in c['frozen_sha256'].items():need(digest(p)==h,'frozen input '+p)
    need(e['complete'] is True and e['records']==320 and e['forward']==2240 and e['backward']==0
        and e['stored_cells']==2560 and e['reused_cells']==320,'exact extraction counts')
    need(e['protocol_sha256']==digest(directory/'protocol.json') and e['results_sha256']==digest(directory/'results.json'),'outputs binding')
    oldv=read(OLD/'verification.json');oldrows=read(OLD/'results.json')
    need(oldv['status']=='PASS_PROMPT_LOSS_RAW_ARITHMETIC_AND_COHORT','verified original draw')
    need(oldv['results_sha256']==digest(OLD/'results.json'),'original results provenance')
    original={(r['model'],r['image_id']):r for r in oldrows if r['model'] in MODELS and r['eval_role']=='selection'}
    rows=read(directory/'results.json');index={(r['model'],r['image_id']):r for r in rows}
    need(len(rows)==len(index)==len(original)==320 and set(index)==set(original),'exact320 identities')
    cache=torch.load(RUN/'cache/cache.pt',map_location='cpu',weights_only=True)
    alphas=scheduler().alphas_cumprod.cpu().float()
    ledgers={}
    for m in MODELS:
        cp=Path(c['checkpoint_paths'][m]);st=torch.load(cp,map_location='cpu',weights_only=True)
        need(digest(cp)==c['checkpoint_sha256'][m] and sum(st['exposures'].values())==4000,'bound checkpoints')
        train=read_csv(RUN/'cohort'/(m+'_train.csv'))
        need(set(st['exposures'])=={r['image_id'] for r in train},'actual manifest')
        patient={}
        for r in train:patient[r['patient_id']]=patient.get(r['patient_id'],0)+st['exposures'][r['image_id']]
        ledgers[m]=(st['exposures'],patient)
    raw_hashes={};values={};shared={};max_mse_error=0.;reused=0;noising_checks=[]
    for (m,i),row in index.items():
        source=original[m,i]
        for k in ('patient_id','image_id','eval_role','scenario','record_role','assignment_group','member',
            'image_member','actual_training_exposures','patient_training_exposures','checkpoint_sha256','cache_sha256','cohort_lock_sha256'):
            need(row[k]==source[k],'original metadata '+k)
        ex,patient=ledgers[m]
        need(row['member']==int(patient.get(row['patient_id'],0)>0),'actual patient member')
        need(row['actual_training_exposures']==ex.get(i,0),'actual image exposure')
        need(row['image_member']==int(ex.get(i,0)>0),'actual image member')
        need(row['forward']==7 and row['backward']==0 and row['repeats']==8,'row counts')
        need(row['contract_sha256']==protocol['contract_sha256'],'row contract')
        rp=directory/row['raw_path'];need(digest(rp)==row['raw_sha256'],'raw checksum')
        need(row['original_raw_sha256']==source['raw_sha256']==digest(OLD/source['raw_path']),'original raw checksum')
        raw=torch.load(rp,map_location='cpu',weights_only=True)
        old=torch.load(OLD/source['raw_path'],map_location='cpu',weights_only=True)
        z=cache['latents'][i].float()
        need(raw['model']==m and raw['image_id']==i and raw['timestep']==140,'raw identity')
        need(torch.equal(raw['latent'],z) and torch.equal(raw['alphas'],alphas),'cache latent/alphas')
        need(raw['hidden_sha256']==row['hidden_sha256']==thash(cache['hidden'][cache['audit_prompt']].float()),'fixed actual hidden')
        need([d['repeat'] for d in raw['draws']]==list(range(8)) and len(row['draws'])==8,'all repeats in order')
        vv=[]
        for d,summary in zip(raw['draws'],row['draws']):
            r=d['repeat'];seed=policy_seed(i,r)
            eps=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
            need(d['seed']==summary['seed']==seed and torch.equal(eps,d['epsilon']),'independent noise replay')
            need(d['noise_sha256']==summary['noise_sha256']==thash(eps),'noise hash')
            pred=d['prediction'];noisy=d['noised_latent']
            for t in (pred,noisy,eps):
                need(t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==(1,4,32,32)
                    and torch.isfinite(t).all(),'finite FP32 input/output')
            nc=verify_fp32_noising(noisy,z,eps,alphas,140)
            noising_checks.append(dict(model=m,image_id=i,repeat=r,**nc))
            resid=pred.double().numpy()-eps.double().numpy();mse=float(np.mean(resid*resid));l2=math.sqrt(float(np.sum(resid*resid)))
            close(d['loss_mse_fp32'],mse,'FP64 residual MSE',rtol=2e-6,atol=1e-7)
            close(d['loss_l2_fp64'],l2,'independent FP64 L2')
            close(summary['loss_mse_fp32'],d['loss_mse_fp32'],'saved MSE summary')
            close(summary['loss_l2_fp64'],l2,'saved L2 summary')
            need(d['reused_original']==summary['reused_original']==(r==0),'reuse label')
            max_mse_error=max(max_mse_error,abs(d['loss_mse_fp32']-mse))
            if r==0:
                need(torch.equal(pred,old['predictions']['generic']) and torch.equal(noisy,old['noised_latent'])
                    and torch.equal(eps,old['epsilon']),'exact original tensors')
                need(d['loss_mse_fp32']==source['losses']['generic'],'exact original MSE0')
                reused+=1
            key=(i,r);fingerprint=(thash(z),thash(eps),thash(noisy),raw['hidden_sha256'])
            if key in shared:need(shared[key]==fingerprint,'same inputs across targets')
            else:shared[key]=fingerprint
            vv.append([d['loss_mse_fp32'],l2,mse])
        values[m,i]=np.asarray(vv);raw_hashes[row['raw_path']]=row['raw_sha256']
    need(reused==320 and len(shared)==1280,'reuse320/distinctnoise1280')
    need(len(e['model_reports'])==2 and {r['model'] for r in e['model_reports']}==set(MODELS),'two model guards')
    for r in e['model_reports']:
        prior=next(v for v in read(OLD/'execution.json')['model_reports'] if v['model']==r['model'])
        need(r['initial_state']['sha256']==r['final_state']['sha256']==prior['initial_state']['sha256'],'exact model state')
        need(r['forward']==1120 and r['backward']==0 and r['state_values_exactly_unchanged']
             and r['adapter_matches_checkpoint'] and r['parameter_versions_and_no_gradients']
             and r['all_parameters_frozen'] and not r['unet_training'],'guard evidence')
    for p,h in c['frozen_sha256'].items():need(digest(p)==h,'unchanged after verification')
    v=dict(status='PASS_REPEAT_NOISE_SAVED_TENSORS_AND_REUSE',records=320,losses_verified=2560,
        original_draw0_exact_reuses=reused,distinct_noises=1280,max_MSE_FP32_FP64_difference=max_mse_error,
        raw_sha256=raw_hashes,results_sha256=digest(directory/'results.json'),protocol_sha256=digest(directory/'protocol.json'),
        execution_sha256=digest(directory/'execution.json'),code_sha256=digest(__file__),
        independent_neural_forward_replay=False,new_GPU_calls=0,seconds_cpu=time.perf_counter()-start)
    v['original_verifier_source_sha256']=ORIGINAL_SHA
    v['noising_helper_sha256']=NOISING_HELPER_SHA
    v['version']='v2; original arithmetic/statistics preserved except FP32 noising validation'
    v['noising_validation']=dict(CPU_FP32_exact_cells=sum(x['CPU_FP32_exact'] for x in noising_checks),
        old_tolerance_failed_cells=sum(x['old_tolerance_failed_elements']>0 for x in noising_checks),
        old_tolerance_failed_elements=sum(x['old_tolerance_failed_elements'] for x in noising_checks),
        max_error_to_rounding_bound_ratio=max(x['max_error_to_rounding_bound_ratio'] for x in noising_checks),
        first_reproduced_failure=next((x for x in noising_checks if x['old_tolerance_failed_elements']),None),
        method='Exact CPU FP32 scheduler output AND coefficient-error + gamma2 operand-scaled roundoff bound',
        neural_GPU_rerun=False)
    write(directory/'verification_v2.json',v)
    write(directory/'verification_v1_failure_reproduction.json',dict(
        original_code_sha256=ORIGINAL_SHA,original_check='independent noising rtol2e-6 atol1e-7 to ideal FP64',
        reproduction=v['noising_validation'],GPU_raw_preserved=True,verification_only_fix=True))
    return c,index,values

def auc_boot(score,label,ix):
    pos=np.flatnonzero(label);neg=np.flatnonzero(1-label)
    credit=(score[pos,None]>score[None,neg]).astype(float)+.5*(score[pos,None]==score[None,neg])
    point=float(credit.mean());close(point,roc_auc_score(label,score),'independent AUC')
    cnt=np.asarray([np.bincount(j,minlength=len(score)) for j in ix],float)
    bs=np.einsum('bi,ij,bj->b',cnt[:,pos],credit,cnt[:,neg])/400
    # Independent explicit resampled comparisons on every bootstrap draw.
    v=score[ix];y=label[ix];a=v[y==1].reshape(len(ix),20);b=v[y==0].reshape(len(ix),20)
    direct=((a[:,:,None]>b[:,None,:]).sum((1,2))+.5*(a[:,:,None]==b[:,None,:]).sum((1,2)))/400
    close(bs,direct,'independent bootstrap')
    return point,bs

def analyze(directory,c,index,values):
    out=directory/'analysis_v2';out.mkdir(exist_ok=False);start=time.perf_counter()
    patient_order=c['patient_order'];groups={p:next(r['assignment_group'] for r in index.values() if r['patient_id']==p) for p in patient_order}
    need(list(groups.values()).count('A')==list(groups.values()).count('B')==20,'20A20B')
    bs=read(OLD/'analysis_v1/bootstrap.json')
    remap={i:patient_order.index(p) for i,p in enumerate(bs['patient_order'])}
    ix=np.array([[remap[i] for i in row] for row in bs['indices']])
    need(ix.shape==(2000,40),'same2000 patient bootstrap')
    cube=np.empty((2,2,40,2,8,3),float)
    for mi,m in enumerate(MODELS):
        for si,s in enumerate(SCENARIOS):
            for pi,p in enumerate(patient_order):
                images=sorted(r['image_id'] for r in index.values() if r['model']==m and r['patient_id']==p and r['scenario']==s)
                need(len(images)==2,'two images per patient/scenario')
                for ii,i in enumerate(images):cube[mi,si,pi,ii]=values[m,i]
    labels=np.array([[int(groups[p]==('A' if m=='model_1' else 'B')) for p in patient_order] for m in MODELS])
    settings=[('seed_'+str(r),[r]) for r in range(8)]+[('mean_'+str(k),list(range(k))) for k in (1,2,4,8)]
    stats=[];contrasts=[];patient_rows=[];reliability=[];paired=[];score_cache={};boot_cache={}
    for metric,metric_name in enumerate(METRICS):
        for si,s in enumerate(SCENARIOS):
            for mi,m in enumerate(MODELS):
                for pool in POOLS:
                    for setting,draws in settings:
                        # np.take avoids advanced-index axis reordering.
                        image_score=-np.take(cube[mi,si,:,:,:,metric],draws,axis=-1).mean(-1)
                        score=image_score.mean(1) if pool=='mean' else image_score.max(1)
                        key=(metric,si,mi,pool,setting);point,b=auc_boot(score,labels[mi],ix)
                        score_cache[key]=score;boot_cache[key]=b
                        score64=-np.take(cube[mi,si,:,:,:,2],draws,axis=-1).mean(-1)
                        score64=score64.mean(1) if pool=='mean' else score64.max(1)
                        stats.append(dict(metric=metric_name,scenario=s,model=m,pool=pool,setting=setting,draws=draws,
                            AUC=point,CI95=np.quantile(b,[.025,.975]).tolist(),
                            residual_FP64_AUC=float(roc_auc_score(labels[mi],score64)) if metric==0 else None))
                        for pi,p in enumerate(patient_order):
                            patient_rows.append(dict(metric=metric_name,scenario=s,model=m,pool=pool,setting=setting,
                                patient_id=p,group=groups[p],member=int(labels[mi,pi]),score=float(score[pi])))
                    for k in (2,4,8):
                        ka=(metric,si,mi,pool,'mean_'+str(k));kb=(metric,si,mi,pool,'mean_1')
                        a=float(roc_auc_score(labels[mi],score_cache[ka]));b=float(roc_auc_score(labels[mi],score_cache[kb]))
                        contrasts.append(dict(metric=metric_name,scenario=s,model=m,pool=pool,mean_count=k,
                            delta_AUC_vs_mean1=a-b,paired_CI95=np.quantile(boot_cache[ka]-boot_cache[kb],[.025,.975]).tolist()))
                    single=np.stack([score_cache[metric,si,mi,pool,'seed_'+str(r)] for r in range(8)],axis=1)
                    within=float(np.var(single,axis=1,ddof=1).mean());between=float(np.var(single.mean(1),ddof=1))
                    denominator=8*between+7*within
                    disjoint=[]
                    for k in (1,2,4):
                        for off in range(0,8,2*k):
                            a=-cube[mi,si,:,:,off:off+k,metric].mean(-1)
                            b=-cube[mi,si,:,:,off+k:off+2*k,metric].mean(-1)
                            a=a.mean(1) if pool=='mean' else a.max(1);b=b.mean(1) if pool=='mean' else b.max(1)
                            disjoint.append(dict(draws_a=list(range(off,off+k)),draws_b=list(range(off+k,off+2*k)),count=k,
                                Spearman=rho(a,b),score_difference_RMS=float(np.sqrt(np.mean((a-b)**2)))))
                    reliability.append(dict(metric=metric_name,scenario=s,model=m,pool=pool,
                        mean_within_patient_seed_variance=within,between_patient_mean_of_single_draw_scores_variance=between,
                        one_way_ICC_1_1=(8*between-within)/denominator if denominator else None,
                        ICC_scope='Repeated single-draw patient scores; max then draw-average differs from primary draw-average then max. Can be negative; not participation specificity',
                        seed_pairs=[dict(a=a,b=b,Spearman=rho(single[:,a],single[:,b])) for a,b in itertools.combinations(range(8),2)],
                        disjoint_block_comparisons=disjoint))
            for pool in POOLS:
                for setting,draws in settings:
                    a=score_cache[metric,si,0,pool,setting];b=score_cache[metric,si,1,pool,setting]
                    delta=np.where(labels[0]==1,a-b,b-a)
                    paired.append(dict(metric=metric_name,scenario=s,pool=pool,setting=setting,draws=draws,
                        participant_minus_nonparticipant_mean=float(delta.mean()),
                        CI95_patient_mean=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),
                        positive_patients=int((delta>0).sum()),zero_patients=int((delta==0).sum()),
                        cross_target_Spearman=rho(a,b),patient_deltas=delta.tolist(),
                        individual_causal_claim=False))
                d=np.stack([np.where(labels[0]==1,
                    score_cache[metric,si,0,pool,'seed_'+str(r)]-score_cache[metric,si,1,pool,'seed_'+str(r)],
                    score_cache[metric,si,1,pool,'seed_'+str(r)]-score_cache[metric,si,0,pool,'seed_'+str(r)]) for r in range(8)],1)
                paired.append(dict(metric=metric_name,scenario=s,pool=pool,setting='seed_variability_summary',
                    mean_within_patient_delta_seed_variance=float(np.var(d,axis=1,ddof=1).mean()),
                    patient_delta_seed_sd=np.std(d,axis=1,ddof=1).tolist(),
                    patient_positive_draw_counts=(d>0).sum(1).tolist(),mean_of_single_draw_patient_deltas=d.mean(1).tolist()))
    original=read(OLD/'analysis_v1/analysis.json')['metrics'];restoration=[]
    for s in SCENARIOS:
        for m in MODELS:
            old=next(r for r in original if r['scenario']==s and r['model']==m and r['prompt']=='generic'
                and r['score_type']=='negative_target_loss')
            new=next(r for r in stats if r['metric']=='MSE_FP32' and r['scenario']==s and r['model']==m
                and r['pool']=='mean' and r['setting']=='mean_1')
            close(new['AUC'],old['selection_AUC'],'exact original mean1 AUC')
            close(new['CI95'],old['CI95'],'exact original mean1 bootstrap')
            restoration.append(dict(scenario=s,model=m,original_AUC=old['selection_AUC'],
                repeated_mean1_AUC=new['AUC'],CI95=new['CI95'],exact_restoration=True))
    need(len(stats)==192 and len(contrasts)==48 and len(reliability)==16,'predeclared table counts')
    result=dict(status='PASS_FIXED_REPEAT_NOISE_ANALYSIS',policy=analysis_policy(),patient_order=patient_order,
        metrics=stats,prefix_contrasts=contrasts,reliability=reliability,paired_target_change=paired,
        original_restoration=restoration,seconds_cpu=time.perf_counter()-start,overall_stage2_gate=None)
    write(out/'protocol.json',dict(contract_sha256=read(directory/'protocol.json')['contract_sha256'],
        extraction_protocol_sha256=digest(directory/'protocol.json'),
        code_sha256=digest(__file__),policy=analysis_policy(),raw_verification_sha256=digest(directory/'verification_v2.json')))
    write(out/'analysis.json',result);write(out/'patient_scores.json',patient_rows)
    write(out/'bootstrap.json',dict(patient_order=patient_order,indices=ix.tolist(),original_sha256=digest(OLD/'analysis_v1/bootstrap.json')))
    write(out/'provenance.json',dict(code_sha256=digest(__file__),results_sha256=digest(directory/'results.json'),
        verification_sha256=digest(directory/'verification_v2.json'),
        output_sha256={n:digest(out/n) for n in ('protocol.json','analysis.json','patient_scores.json','bootstrap.json')}))
    print(json.dumps(dict(status=result['status'],metrics=192,contrasts=48,reliability=16,seconds=result['seconds_cpu'])))

def self_test():
    noising_self_test()
    y=np.r_[np.ones(20,dtype=int),np.zeros(20,dtype=int)];s=np.arange(40,dtype=float)
    ix=np.tile(np.arange(40),(2000,1));p,b=auc_boot(s,y,ix)
    need(p==0 and np.all(b==0),'AUC orientation')
    x=np.zeros((2,2,40,2,8,3))
    need(np.take(x[0,0,:,:,:,0],[0,1],axis=-1).mean(-1).shape==(40,2),'axis contract')
    need(analysis_policy()['overall_stage2_gate'] is None,'no efficacy gate')
    import builtins,symtable
    missing=[]
    def walk(scope):
        if scope.get_type()=='function':
            for sym in scope.get_symbols():
                if sym.is_referenced() and sym.is_global() and sym.get_name() not in globals() and not hasattr(builtins,sym.get_name()):
                    missing.append((scope.get_name(),sym.get_name()))
        for ch in scope.get_children():walk(ch)
    walk(symtable.symtable(Path(__file__).read_text(encoding='utf8'),__file__,'exec'))
    need(not missing,'global dependencies '+repr(missing))
    print('PASS_CPU_SYNTHETIC_AXES_AUC_BOOTSTRAP_DEPENDENCIES; no actual results read')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--expected-code-sha256');a=p.parse_args()
    if a.self_test:self_test()
    else:
        need(digest(__file__)==a.expected_code_sha256,'externally frozen verifier/analyzer')
        c,index,values=verify(OUT);analyze(OUT,c,index,values)
