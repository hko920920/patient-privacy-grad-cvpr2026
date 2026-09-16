"""Saved, fixed-scorer responses to the two predeclared patient interventions.

No fitting, feature selection, GPU inference, AUC, uncertainty interval or test.
The reused numerical scorer is source-hash bound; raw extraction has a separate
mandatory independent verification gate. E uses the existing U-fitted scorers.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
RESEARCH=next(ROOT.parent.glob('CVPR *'))/'research_2026-09-10'
POLICY=RESEARCH/'spec_sources/patient_cross_intervention_analysis_contract_20260915.json'
POLICY_SHA='b0855a8bdec2e1a5f738d35ee12930bf0dfd7a21c603cf55f343ba99ccf5bc70'
HELPER=Path(__file__).with_name('analyze_masked_cdi_fixed_scorers.py')
HELPER_SHA='06eb9dab58ec19191ad111506e9b9ee5e57e6245183d7224250b3c0411b9c6f8'
RAW_STATUS='PASS_CROSS_PATIENT_CDI_SAVED_ARITHMETIC_PROVENANCE_AND_P_DL_REPLAY'
STATUS='PASS_CROSS_PATIENT_FIXED_SCORER_RESTORATION_AND_DESCRIPTIVE_ANALYSIS'
BRANCHES=['treatment','p_control','q_control']
EPS=np.finfo(float).eps


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def helper():
    assert sha(HELPER)==HELPER_SHA
    spec=importlib.util.spec_from_file_location('frozen_masked_cdi_scorer',HELPER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def arithmetic_envelope(values):
    return float(64*EPS*max(1.,np.abs(np.asarray(values,dtype=float)).sum()))


def sign_report(value,envelope):
    return dict(value=float(value),raw_sign=int(np.sign(value)),FP64_arithmetic_envelope=float(envelope),
        sign_outside_arithmetic_envelope='positive' if value>envelope else 'negative' if value<-envelope else 'unresolved')


def cross_contrast(scores,bounds,indices):
    """Rows interventions p/q; columns queried patients p/q. Same scorer."""
    st,sp,sq=(np.asarray(scores[b],dtype=float) for b in BRANCHES)
    bt,bp,bq=(np.asarray(bounds[b],dtype=float) for b in BRANCHES)
    ip,iq=indices;dp=st-sp;dq=st-sq;h=dp-dq
    d=np.asarray([[dp[ip],dp[iq]],[dq[ip],dq[iq]]])
    rp=float(h[ip]);rq=float(-h[iq]);k=rp+rq
    base_cancelled=float((sq[ip]-sp[ip])-(sq[iq]-sp[iq]))
    centered=d-d.mean(axis=1,keepdims=True)-d.mean(axis=0,keepdims=True)+d.mean()
    ck=float(centered[0,0]+centered[1,1]-centered[0,1]-centered[1,0])
    identity_bound=arithmetic_envelope([st[ip],st[iq],sp[ip],sp[iq],sq[ip],sq[iq],*d.ravel()])
    assert abs(k-base_cancelled)<=identity_bound and abs(k-ck)<=identity_bound
    # Bounds characterize repeated saved FP64 scorer arithmetic, not statistical
    # uncertainty, neural-network error, or an efficacy threshold.
    rpb=float(bp[ip]+bq[ip]+identity_bound)
    rqb=float(bp[iq]+bq[iq]+identity_bound)
    kb=rpb+rqb
    result=dict(matrix_rows=['p_intervention','q_intervention'],matrix_columns=['p_query','q_query'],
        D=d.tolist(),R_p=sign_report(rp,rpb),R_q=sign_report(rq,rqb),K=sign_report(k,kb),
        diagonal_mean=float(np.trace(d)/2),off_diagonal_mean=float((d[0,1]+d[1,0])/2),
        diagonal_minus_off_diagonal=float(k/2),base_cancelled_K=base_cancelled,
        row_column_centered_D=centered.tolist(),centered_K=ck,
        arithmetic_identity_envelope=identity_bound,
        both_own_minus_cross_positive_raw=bool(rp>0 and rq>0),
        both_own_minus_cross_positive_outside_arithmetic_envelope=bool(rp>rpb and rq>rqb),
        K_positive_does_not_imply_both=True)
    return dp,dq,h,result


def context_indices(patients,p,q):
    masks={
        'selection_A':[i for i,r in enumerate(patients) if r['role']=='selection' and r['group']=='A'],
        'selection_B':[i for i,r in enumerate(patients) if r['role']=='selection' and r['group']=='B'],
        'selection_all':[i for i,r in enumerate(patients) if r['role']=='selection'],
        'all41':list(range(len(patients))),
        'background39_A':[i for i,r in enumerate(patients) if r['patient_id'] not in [p,q] and r['group']=='A'],
        'background39_B':[i for i,r in enumerate(patients) if r['patient_id'] not in [p,q] and r['group']=='B'],
        'background39_all':[i for i,r in enumerate(patients) if r['patient_id'] not in [p,q]],
        'p_others40':[i for i,r in enumerate(patients) if r['patient_id']!=p],
        'q_others40':[i for i,r in enumerate(patients) if r['patient_id']!=q]}
    assert [len(x) for x in masks.values()]==[20,20,40,41,19,20,39,40,40]
    return masks


def policy_and_sources():
    assert sha(POLICY)==POLICY_SHA
    c=read(POLICY);assert c['schema']=='patient-cross-intervention-analysis/v1'
    assert c['no_new_fitting'] and c['no_new_C_search'] and c['no_new_AUC'] and c['no_CI'] and c['no_p_value']
    assert c['branches']==BRANCHES and c['focal_patients']=={'p':'14393','q':'4092'}
    for f,h in c['frozen_sha256'].items():assert sha(f)==h,f
    m=helper();oc,oa=m.validate_policy();assert c['methods']==oc['methods']
    assert c['clean_primary_methods']==oa['clean_primary_methods'] and len(c['methods'])==18
    prior=Path(c['prior_scorer_directory']);audit=read(prior/'independent_saved_scorer_audit.json')
    assert audit['status']=='PASS_INDEPENDENT_MASKED_CDI_FIT79_INPUT_SCALER_PREDICTION_AUDIT'
    for f,h in audit['input_sha256'].items():assert sha(f)==h
    provenance=read(prior/'provenance.json')
    for f,h in provenance['outputs_sha256'].items():assert sha(prior/f)==h
    assert provenance['source_code_sha256']==HELPER_SHA
    params={c['arms'][0]:{r['method']:r for r in read(prior/'original_fitted_parameters.json')},
        c['arms'][1]:{r['method']:r for r in read(prior/'leave_patient_out79_parameters.json')}}
    p,q=c['focal_patients'].values()
    for arm,pp in params.items():
        assert len(pp)==10
        for method,r in pp.items():
            ids=r['fitting_patient_ids'];assert q not in ids and len(ids)==len(set(ids))
            if arm==c['primary_arm']:assert len(ids)==79 and p not in ids
            else:assert len(ids)==80 and p in ids
            spec=next(s for s in c['methods'] if s['id']==method)
            assert r['warnings']==[]
            if spec['C_policy']=='fixed':assert r['chosen_C']==1.
    saved=read(prior/'patient_scores.json');assert len(saved)==37*41
    oldscore={(r['arm'],r['method'],r['patient_id']):r for r in saved};assert len(oldscore)==len(saved)
    return c,m,params,oldscore


def load_matrices(c,m,run_dir):
    v=read(run_dir/'verification.json');e=read(run_dir/'execution.json');proto=read(run_dir/'protocol.json')
    assert v['status']==RAW_STATUS and v['complete'] is True and e['complete'] is True
    for name,key in [('results.json','results_sha256'),('protocol.json','protocol_sha256'),('execution.json','execution_sha256')]:
        assert sha(run_dir/name)==v[key],name
    assert proto['contract_sha256']==v['contract_sha256']==e['contract_sha256']
    contract=proto['contract'];assert read(proto['contract_path'])==contract and sha(proto['contract_path'])==proto['contract_sha256']
    assert contract['frozen_sha256'][str(POLICY.resolve())]==POLICY_SHA
    for f,h in contract['frozen_sha256'].items():assert sha(f)==h,f
    fresh=read(run_dir/'results.json');assert len(fresh)==90
    assert Counter((r['branch'],r['scenario']) for r in fresh)=={('q_control','U'):82,('q_control','E'):4,('p_control','E'):4}
    old={s:[r for r in read(Path(c['original_directory_'+s])/'results.json') if r['model']=='model_1'] for s in ['U','E']}
    pold=read(Path(c['p_control_U_directory'])/'results.json');assert len(pold)==82
    matrices={};sources=[]
    for scenario in ['U','E']:
        patients=[r for r in c['patient_rows'] if r['scenario']==scenario]
        assert [p['patient_id'] for p in patients]==c['patient_order_'+scenario]
        wanted={iid for p in patients for iid in p['images']}
        for branch in BRANCHES:
            rr=old[scenario] if branch=='treatment' else pold if branch=='p_control' and scenario=='U' else [r for r in fresh if r['branch']==branch and r['scenario']==scenario]
            idx={r['image_id']:r for r in rr if r['image_id'] in wanted};assert set(idx)==wanted
            assert len(idx)==sum(r['image_id'] in wanted for r in rr)
            for p in patients:
                for iid in p['images']:
                    r=idx[iid];assert r['scenario']==scenario and r['patient_id']==p['patient_id']
                    assert r['eval_role']==p['role'] and r['assignment_group']==p['group']
                    if scenario=='U':assert r['image_member']==0 and r['actual_training_exposures']==0
                    sources.append(dict(branch=branch,scenario=scenario,patient_id=p['patient_id'],image_id=iid,
                        feature_names=r['feature_names'],features=r['features'],NO_optimizer_return_objective=float(r['modules']['noise_optim']['optimizer']['fun'])))
            matrices[scenario,branch]=m.matrix(idx,patients)
    return matrices,sources,v,proto


def run(args):
    started=time.perf_counter();assert args.expected_code_sha256 and sha(__file__)==args.expected_code_sha256
    c,m,params,saved=policy_and_sources();run_dir=Path(c['new_output_directory']) if args.run_dir is None else args.run_dir
    assert run_dir.resolve()==Path(c['new_output_directory']).resolve()
    matrices,sources,v,proto=load_matrices(c,m,run_dir)
    dest=run_dir/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    bindings=dict(c['frozen_sha256']);bindings[str(POLICY.resolve())]=POLICY_SHA;bindings[str(HELPER.resolve())]=HELPER_SHA
    bindings.update({str((run_dir/n).resolve()):sha(run_dir/n) for n in ['results.json','protocol.json','execution.json','verification.json']})
    new(dest/'protocol.json',dict(schema='cross-patient-existing-scorer-analysis/v1',policy_sha256=POLICY_SHA,
        source_code_sha256=args.expected_code_sha256,scorer_helper_sha256=HELPER_SHA,
        extraction_contract_sha256=proto['contract_sha256'],inputs_sha256=bindings,
        numerical_policy_fixed_before_new_interventions=True,analyzer_source_bound_at_execution=True,
        new_fitting=False,new_GPU=False,independent_raw_verification_status=v['status']))
    try:
        outputs=[];contrasts=[];contexts=[];restoration=[];image_probs=[]
        for scenario in ['U','E']:
            patients=[r for r in c['patient_rows'] if r['scenario']==scenario];ids=[r['patient_id'] for r in patients]
            p,q=c['focal_patients'].values();focal=[ids.index(p),ids.index(q)]
            for arm in c['arms']:
                specs=c['methods']+([dict(id='dino_existing',kind='unchanged')] if arm==c['arms'][0] and scenario=='U' else [])
                for spec in specs:
                    method=spec['id'];scores={};bounds={};parameter=params[arm].get(method)
                    clean=arm==c['primary_arm'] and method in c['clean_primary_methods']
                    metadata=dict(arm=arm,method=method,scenario=scenario,clean_fixed_C_primary=clean,
                        historical_tuned_C_selection_included_p=spec.get('C_policy')=='tuned',
                        source_quirk_diagnostic=spec.get('source_quirk_diagnostic',False),
                        no_fitting_in_this_analysis=True,chosen_C=parameter['chosen_C'] if parameter else None,
                        p_in_prior_coefficient_fit=bool(parameter and p in parameter['fitting_patient_ids']),
                        q_in_prior_coefficient_fit=False,unchanged_by_construction=spec['kind']=='unchanged')
                    for branch in BRANCHES:
                        if spec['kind']=='unchanged':
                            scores[branch]=np.asarray([saved[arm,method,i]['treatment_score'] for i in ids]);bounds[branch]=np.zeros(len(ids));im=None
                        else:scores[branch],bounds[branch],im=m.predict(matrices[scenario,branch],spec,parameter)
                        if im is not None:
                            image_probs.extend([dict(**metadata,branch=branch,patient_id=pp['patient_id'],images=pp['images'],probabilities=im[j].tolist(),pool=spec['patient_probability_pool']) for j,pp in enumerate(patients)])
                    dp,dq,h,k=cross_contrast(scores,bounds,focal)
                    contrasts.append(dict(**metadata,**k,patient_order=[p,q],E_uses_U_fitted_scorer=scenario=='E'))
                    for j,pp in enumerate(patients):
                        outputs.append(dict(**metadata,patient_id=pp['patient_id'],role=pp['role'],group=pp['group'],images=pp['images'],
                            scores={b:float(scores[b][j]) for b in BRANCHES},score_arithmetic_envelopes={b:float(bounds[b][j]) for b in BRANCHES},
                            D_p_i=float(dp[j]),D_q_i=float(dq[j]),h_i=float(h[j])))
                        if scenario=='U':
                            old=saved[arm,method,pp['patient_id']];checks={}
                            for field,branch in [('treatment_score','treatment'),('control_score','p_control')]:
                                error=abs(float(scores[branch][j])-old[field]);bound=float(bounds[branch][j]);assert error<=bound
                                checks[field]=dict(saved=old[field],reproduced=float(scores[branch][j]),absolute_error=error,envelope=bound,exact=error==0)
                            de=abs(float(dp[j])-old['delta']);db=float(bounds['treatment'][j]+bounds['p_control'][j]+arithmetic_envelope([old['delta'],dp[j]]));assert de<=db
                            checks['delta']=dict(saved=old['delta'],reproduced=float(dp[j]),absolute_error=de,envelope=db,exact=de==0)
                            restoration.append(dict(arm=arm,method=method,patient_id=pp['patient_id'],checks=checks))
                    if scenario=='U':
                        groups=context_indices(patients,p,q);assert list(groups)==c['contexts']
                        for name,jj in groups.items():
                            contexts.append(dict(**metadata,context=name,patient_ids=[ids[j] for j in jj],includes_p=p in [ids[j] for j in jj],includes_q=q in [ids[j] for j in jj],
                                D_p_i=m.distribution(dp[jj]),D_q_i=m.distribution(dq[jj]),h_i=m.distribution(h[jj]),
                                descriptive_only=True,independent_intervention_repetitions=False))
        assert len(restoration)==1517 and len(outputs)==1589 and len(contrasts)==73 and len(contexts)==333
        checks=[c for row in restoration for c in row['checks'].values()]
        new(dest/'patient_scores.json',outputs);new(dest/'cross_contrasts.json',contrasts);new(dest/'patient_contexts.json',contexts)
        new(dest/'image_probabilities.json',image_probs);new(dest/'prior_score_restoration.json',restoration)
        new(dest/'source_features.json',sources);new(dest/'saved_parameter_identity.json',{arm:{k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in pp.items()} for arm,pp in params.items()})
        result=dict(status=STATUS,complete=True,current_step=2,patient_ids=c['focal_patients'],patient_score_rows=len(outputs),
            clean_primary=[r for r in contrasts if r['clean_fixed_C_primary']],all_method_scenario_contrasts=contrasts,
            prior_restoration=dict(patient_rows=len(restoration),scalar_checks=len(checks),exact=sum(x['exact'] for x in checks),
                max_absolute_error=max(x['absolute_error'] for x in checks),all_within_FP64_arithmetic_envelope=True),
            saved_scorer_sets=2,saved_learned_scorers=20,new_fitting=False,new_C_search=False,new_GPU=False,
            AUC=False,CI=False,p_value=False,efficacy_gate=False,stage2_completion_claim=False,
            context_warning=c['context_warning'],decision_policy=c['decision_policy'],limits=c['limits'],
            source_code_sha256=args.expected_code_sha256,analysis_policy_sha256=POLICY_SHA,
            extraction_verification_sha256=sha(run_dir/'verification.json'),seconds_cpu=time.perf_counter()-started)
        for f,h in bindings.items():assert sha(f)==h
        assert sha(__file__)==args.expected_code_sha256
        new(dest/'analysis.json',result)
        new(dest/'provenance.json',dict(source_code_sha256=args.expected_code_sha256,helper_sha256=HELPER_SHA,
            policy_sha256=POLICY_SHA,inputs_sha256=bindings,outputs_sha256={f.name:sha(f) for f in dest.iterdir() if f.is_file()}))
        print(json.dumps(dict(status=STATUS,seconds_cpu=result['seconds_cpu'],rows=len(outputs),restored=len(restoration),contrasts=len(contrasts))))
    except BaseException:
        new(dest/'failure.json',dict(status='FAILED_PRESERVED_CROSS_PATIENT_FIXED_SCORER_ANALYSIS',traceback=traceback.format_exc()))
        raise


def self_test():
    # Additive row/query offsets disappear from K, while a positive total can
    # coexist with a negative q-own contrast. This is not a performance gate.
    scores={'treatment':np.array([8.,7.]),'p_control':np.array([4.,5.]),'q_control':np.array([7.,6.])}
    bounds={b:np.zeros(2) for b in BRANCHES};_,_,_,r=cross_contrast(scores,bounds,[0,1])
    assert r['R_p']['value']==3 and r['R_q']['value']==-1 and r['K']['value']==2
    assert not r['both_own_minus_cross_positive_raw'] and r['K']['raw_sign']==1
    shifted={b:v+np.array([100.,-50.]) for b,v in scores.items()}
    assert cross_contrast(shifted,bounds,[0,1])[3]['K']['value']==2
    shifted['p_control']+=17;shifted['q_control']-=12
    assert cross_contrast(shifted,bounds,[0,1])[3]['K']['value']==2
    c,m,params,_=policy_and_sources();patients=[p for p in c['patient_rows'] if p['scenario']=='U']
    assert context_indices(patients,'14393','4092')['background39_all']
    assert m.self_test()['new_fitting'] is False
    return dict(status='PASS_CROSS_CONTRAST_SIGNS_ADDITIVE_CANCELLATION_CONTEXTS_AND_FROZEN_SCORER_INPUTS',new_fitting=False,new_GPU=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--output-tag',default='existing_scorer_analysis_v1')
    p.add_argument('--expected-code-sha256');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    run(a)


if __name__=='__main__':main()
