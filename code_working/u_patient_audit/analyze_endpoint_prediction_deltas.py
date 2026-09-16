"""Post-intervention prediction/loss geometry; no fitting or attack evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'_reports/cvpr_u_pilot_v1_001'
BASE=RUN/'baseline_screen_20260915'
ENDPOINT=BASE/'masked_slot_endpoints_v1'
POLICY=BASE/'endpoint_prediction_delta_policy_v1.json'
PREFIXES={'generic_t140':[1,2,4,8],'cdi_dl_t100':[1,2,4,5]}
REPEATS={'generic_t140':8,'cdi_dl_t100':5,'fixed_mofit':5}
METRICS=['delta_rms','signed_projection','squared_term','delta_mse','abs_delta_mse']
VERIFICATION_STATUS='PASS_MASKED_ENDPOINT_SAVED_ARITHMETIC_SOURCE_REUSE_AND_TRAINING_GATES'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def policy():
    return dict(schema='endpoint-prediction-delta-policy/v1',current_step=2,
        required_verification_status=VERIFICATION_STATUS,
        question='Does the fixed U response move in prediction space while its MSE change is small relative to other patients?',
        pairs=2232,branches=['treatment','control'],delta='prediction_treatment - prediction_control',
        residual='prediction_control - shared_epsilon',
        identity='delta_MSE = 2*sum(residual_control*delta)/D + sum(delta**2)/D',
        precision='All decomposition terms recomputed from saved FP32 tensors using float64 and math.fsum',
        identity_roundoff='Record error; 32*float64 epsilon*sum of absolute component energies, not efficacy threshold',
        metrics=METRICS,source_runtime_fp32_delta_also_saved=True,
        families=list(REPEATS),repeats=REPEATS,prefixes=PREFIXES,
        aggregation='All per-image repeat cells; patient mean over the two E or U images for every repeat and each fixed prefix. fixed_mofit also all-five mean.',
        delta_rms_aggregation='sqrt(mean squared_term), pooling coordinates/images/noises; not norm of a noise-averaged prediction',
        max='Not decomposed: endpoint max report is preserved separately because branch selectors can differ',
        comparisons='For generic_t140 and cdi_dl_t100 only: p14393 versus fixed selection40, within A20/B20/all40, same scenario and noise scope',
        empirical_rank='(number of reference values below case + 0.5*number equal)/reference count; descriptive midrank, not p-value',
        distribution=['n','min','q25','median','q75','max','mean'],
        mofit_scope='One patient only; all four fixed source embeddings and null. No fabricated selection40 rank.',
        no_fitting=True,no_new_scoring_rule=True,no_seed_selection=True,no_threshold_selection=True,
        no_GPU=True,no_cross_image_cosine=True,no_MIA_AUC=True,no_CI=True,
        limits=['Same known training intervention for all query patients; observations are not independent training replicates',
            'Paired counterfactual predictions are analyst-only inputs, not the limited-access auditor setting',
            'Large delta norm or cancellation does not establish patient-specific membership signal',
            'Small observed difference does not prove absence of patient information',
            'CDI original scalar is L2; this MSE identity is a diagnostic, not a replacement of its original score',
            'MoFit rows decompose each conditioned or null raw MSE, not h as a vector norm; the original conditioned-minus-null report is preserved',
            'MoFit inputs use fixed treatment-derived embeddings and a distinct posterior-sample latent family'])


def prepare():
    assert not POLICY.exists()
    producer=ROOT/'u_patient_audit/run_masked_slot_endpoints.py'
    p=dict(policy=policy(),created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        analysis_code_sha256=sha(Path(__file__)),source_producer=str(producer),source_producer_sha256=sha(producer))
    write(POLICY,p)
    return dict(status='PASS_POLICY_FROZEN_WITHOUT_ENDPOINT_RESULTS',policy_sha256=sha(POLICY),
                analysis_code_sha256=p['analysis_code_sha256'])


def components(pt,pc,epsilon):
    assert pt.dtype==pc.dtype==epsilon.dtype==torch.float32 and pt.shape==pc.shape==epsilon.shape
    assert tuple(pt.shape)==(1,4,32,32)
    t=pt.double().reshape(-1).numpy();c=pc.double().reshape(-1).numpy();e=epsilon.double().reshape(-1).numpy()
    assert np.isfinite(t).all() and np.isfinite(c).all() and np.isfinite(e).all()
    d=t-c;rt=t-e;rc=c-e;n=d.size
    mt=math.fsum((rt*rt).tolist())/n;mc=math.fsum((rc*rc).tolist())/n
    linear=2*math.fsum((rc*d).tolist())/n
    quad=math.fsum((d*d).tolist())/n
    delta=mt-mc
    error=abs(delta-(linear+quad))
    energy=mt+mc+2*math.fsum(np.abs(rc*d).tolist())/n+quad
    bound=32*np.finfo(np.float64).eps*energy+np.nextafter(0.,1.)
    assert error<=bound,'Float64 loss difference identity failure'
    return dict(mse_treatment=mt,mse_control=mc,delta_mse=delta,abs_delta_mse=abs(delta),
        signed_projection=linear,squared_term=quad,delta_rms=math.sqrt(quad),
        identity_absolute_error=error,identity_roundoff_bound=float(bound),coordinates=n)


def aggregate(rows):
    n=len(rows)
    q={key:math.fsum(r[key] for r in rows)/n for key in
       ['mse_treatment','mse_control','delta_mse','signed_projection','squared_term','runtime_delta_mse']}
    q['delta_rms']=math.sqrt(q['squared_term']);q['abs_delta_mse']=abs(q['delta_mse'])
    q['identity_absolute_error']=abs(q['delta_mse']-q['signed_projection']-q['squared_term'])
    scale=q['mse_treatment']+q['mse_control']+abs(q['signed_projection'])+q['squared_term']
    q['identity_roundoff_bound']=math.fsum(r['identity_roundoff_bound'] for r in rows)/n+32*np.finfo(float).eps*scale+1e-320
    assert q['identity_absolute_error']<=q['identity_roundoff_bound']
    q['runtime_minus_fp64_delta']=q['runtime_delta_mse']-q['delta_mse']
    q['paired_cells']=n
    return q


def describe(values,case):
    a=np.asarray(values,dtype=float);assert len(a) in [20,40]
    below=int(np.sum(a<case));equal=int(np.sum(a==case))
    return dict(n=len(a),min=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),
        q75=float(np.quantile(a,.75)),max=float(a.max()),mean=math.fsum(values)/len(a),
        case_value=case,reference_below=below,reference_equal=equal,
        empirical_midrank_fraction=(below+.5*equal)/len(a))


def self_test():
    # Nonzero vector change with exact cancellation of the two identity terms.
    e=torch.zeros((1,4,32,32));c=torch.ones_like(e);t=-torch.ones_like(e)
    q=components(t,c,e)
    assert q['delta_mse']==0 and q['delta_rms']==2 and q['signed_projection']==-4 and q['squared_term']==4
    q['runtime_delta_mse']=0.;a=aggregate([q,q])
    assert a['delta_mse']==0 and a['delta_rms']==2
    z=components(c,c,e);assert z['delta_rms']==z['delta_mse']==0
    r=describe([1.]*20,1.);assert r['empirical_midrank_fraction']==.5
    return dict(status='PASS_CPU_IDENTITY_CANCELLATION_AGGREGATION_AND_TIES',CUDA_initialized=torch.cuda.is_initialized())


def run(args):
    started=time.perf_counter();frozen=read(POLICY)
    assert frozen['policy']==policy() and frozen['analysis_code_sha256']==sha(Path(__file__))==args.expected_code_sha256
    assert sha(frozen['source_producer'])==frozen['source_producer_sha256']
    out=args.run_dir;v=read(args.verification);execution=read(out/'execution.json');ep=read(out/'protocol.json')
    assert v['status']==args.required_verification_status==VERIFICATION_STATUS
    assert execution['complete'] is True and execution['status']=='PASS_MASKED_ENDPOINT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION'
    assert execution['records']==4464 and execution['new_forward']==2260 and execution['reused_cells']==2204
    assert execution['results_sha256']==sha(out/'results.json') and execution['protocol_sha256']==sha(out/'protocol.json')
    assert v['execution_sha256']==sha(out/'execution.json')
    for field,name in [('results_sha256','results.json'),('protocol_sha256','protocol.json')]:
        assert v[field]==sha(out/name)
    assert sha(ep['contract_path'])==ep['contract_sha256']==execution['contract_sha256']
    assert v['contract_sha256']==ep['contract_sha256']
    records=read(out/'results.json');paths={r['raw_path']:r['raw_sha256'] for r in records}
    inputs={str((out/p).resolve()):h for p,h in paths.items()}
    for p in [Path(__file__),POLICY,args.verification,out/'execution.json',out/'protocol.json',out/'results.json',Path(ep['contract_path'])]:
        inputs[str(p.resolve())]=sha(p)
    for p,h in inputs.items():assert sha(p)==h,p
    dest=out/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    write(dest/'protocol.json',dict(schema='endpoint-prediction-delta-execution/v1',policy=frozen,
        policy_sha256=sha(POLICY),input_sha256=inputs,required_verification_status=args.required_verification_status))
    paired=defaultdict(dict)
    packets={}
    for r in records:
        k=(r['family'],r['image_id'],r['condition'],r['repeat'])
        assert r['branch'] not in paired[k]
        paired[k][r['branch']]=r
    assert len(paired)==2232 and all(set(v)=={'treatment','control'} for v in paired.values())
    cells=[]
    for key,pair in paired.items():
        rs=[pair[b] for b in ['treatment','control']];cs=[];ps=[]
        for r in rs:
            if r['raw_path'] not in packets:
                packets[r['raw_path']]=torch.load(out/r['raw_path'],map_location='cpu',weights_only=True)
            packet=packets[r['raw_path']];ps.append(packet);cs.append(packet['cells'][r['cell_index']])
            assert cs[-1]['repeat']==r['repeat'] and cs[-1]['seed']==r['seed']
        a,b=rs;ct,cc=cs
        for f in ['patient_id','scenario','eval_role','assignment_group','condition','repeat','seed']:
            assert a[f]==b[f]
        for f in ['latent','hidden']:assert torch.equal(ps[0][f],ps[1][f])
        for f in ['noised_latent','epsilon']:assert torch.equal(ct[f],cc[f])
        q=components(ct['prediction'],cc['prediction'],ct['epsilon'])
        runtime=ct['loss_mse_fp32']-cc['loss_mse_fp32']
        cells.append(dict(family=key[0],image_id=key[1],condition=key[2],repeat=key[3],seed=a['seed'],
            patient_id=a['patient_id'],scenario=a['scenario'],eval_role=a['eval_role'],assignment_group=a['assignment_group'],
            **q,runtime_delta_mse=runtime,runtime_minus_fp64_delta=runtime-q['delta_mse']))
    grouped=defaultdict(list)
    for r in cells:grouped[r['family'],r['patient_id'],r['scenario'],r['condition']].append(r)
    patient=[]
    for (family,pid,scenario,condition),rs in grouped.items():
        image_ids=sorted({r['image_id'] for r in rs});assert len(image_ids)==2
        assert len(rs)==2*REPEATS[family]
        scopes=[('repeat',[i]) for i in range(REPEATS[family])]
        scopes += [('prefix',list(range(n))) for n in PREFIXES[family]] if family in PREFIXES else [('all',list(range(5)))]
        for scope,indices in scopes:
            selected=[r for r in rs if r['repeat'] in indices]
            assert len(selected)==2*len(indices)
            patient.append(dict(family=family,patient_id=pid,scenario=scenario,condition=condition,
                eval_role=rs[0]['eval_role'],assignment_group=rs[0]['assignment_group'],
                noise_scope=scope,noise_indices=indices,image_ids=image_ids,
                aggregation='patient_mean',**aggregate(selected)))
    contexts=[]
    for case in patient:
        if case['patient_id']!='14393' or case['family']=='fixed_mofit':continue
        refs=[q for q in patient if q['eval_role']=='selection' and all(q[k]==case[k]
            for k in ['family','scenario','condition','noise_scope','noise_indices'])]
        assert len(refs)==40 and len({q['patient_id'] for q in refs})==40
        for group in ['A','B','all']:
            selected=[q for q in refs if group=='all' or q['assignment_group']==group]
            assert len(selected)==(40 if group=='all' else 20)
            for metric in METRICS:
                contexts.append(dict(family=case['family'],scenario=case['scenario'],condition=case['condition'],
                    noise_scope=case['noise_scope'],noise_indices=case['noise_indices'],reference_group=group,
                    metric=metric,case_patient_id='14393',**describe([q[metric] for q in selected],case[metric])))
    for p,h in inputs.items():assert sha(p)==h,p
    assert len(patient)==1782 and len(contexts)==630
    write(dest/'cell_components.json',cells);write(dest/'patient_means.json',patient);write(dest/'reference_context.json',contexts)
    assert torch.cuda.is_initialized() is False
    write(dest/'analysis.json',dict(status='PASS_SAVED_PREDICTION_MSE_IDENTITY_AND_DESCRIPTIVE_CONTEXT',
        current_step=2,paired_cells=len(cells),patient_mean_rows=len(patient),context_rows=len(contexts),
        maximum_identity_absolute_error=max(q['identity_absolute_error'] for q in cells),
        outputs_sha256={n:sha(dest/n) for n in ['cell_components.json','patient_means.json','reference_context.json']},
        protocol_sha256=sha(dest/'protocol.json'),policy_sha256=sha(POLICY),
        elapsed_seconds=time.perf_counter()-started,limits=policy()['limits'],
        MIA_AUC=False,new_score=False,causal_population_claim=False,CUDA_initialized=torch.cuda.is_initialized()))
    print(json.dumps(dict(status='PASS_SAVED_PREDICTION_MSE_IDENTITY_AND_DESCRIPTIVE_CONTEXT',paired_cells=len(cells),seconds=time.perf_counter()-started)))


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-policy',action='store_true');g.add_argument('--self-test',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--run-dir',type=Path,default=ENDPOINT);p.add_argument('--output-tag',default='prediction_delta_v1')
    p.add_argument('--verification',type=Path,default=ENDPOINT/'verification.json')
    p.add_argument('--required-verification-status',default=VERIFICATION_STATUS)
    p.add_argument('--expected-code-sha256')
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    if a.prepare_policy:print(json.dumps(prepare()));return
    assert a.expected_code_sha256
    run(a)


if __name__=='__main__':main()
