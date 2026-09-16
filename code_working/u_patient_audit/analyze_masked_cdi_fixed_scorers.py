"""CPU CDI scorer responses to one fixed masked-patient training intervention.

Original fitted scorers are preserved. A separately predeclared fit79 arm removes
the target patient from auxiliary fitting. Neither arm tunes on control outputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'_reports/cvpr_u_pilot_v1_001'
OUT=RUN/'baseline_screen_20260915/cdi_masked_u82_v1'
RESEARCH=next(ROOT.parent.glob('CVPR *'))/'research_2026-09-10'
POLICY=RESEARCH/'spec_sources/masked_cdi_fixed_scorer_analysis_contract_20260915.json'
ADDENDUM=RESEARCH/'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json'
POLICY_SHA='e224582bf32a045e7d2593e7b19385cb6c1574578c6a8648f89a66f5b6465518'
ADDENDUM_SHA='bb5147a5c40cbdb249d2e64ee5f3a942fea91962ff2d0ed975b84d6981bc16d1'
RAW_STATUS='PASS_MASKED_CDI_U82_SAVED_ARITHMETIC_PROVENANCE_AND_DL_REPLAY'
PID='14393'
ORIGINAL_ARM='original_fit80_frozen_patient_seen'
REFIT_ARM='leave_patient_out79_refit'
FEATURE_NAMES=['Denoising Loss','SecMI$_{stat}$','PIA','PIAN']+[f'Gradient Masking_{i}' for i in range(10)]+[f'Multiple Loss_{i}' for i in range(10)]+['Noise Optimization_0','Noise Optimization_1']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def validate_policy():
    assert sha(POLICY)==POLICY_SHA and sha(ADDENDUM)==ADDENDUM_SHA
    c,a=read(POLICY),read(ADDENDUM)
    assert a['original_policy_sha256']==POLICY_SHA and a['methods']==c['methods']
    assert len(c['methods'])==18 and len({s['id'] for s in c['methods']})==18
    assert len(a['training_patient_ids'])==len(set(a['training_patient_ids']))==79 and PID not in a['training_patient_ids']
    for f,h in c['frozen_sha256'].items():assert sha(f)==h,f
    return c,a


def features(row):
    assert row['feature_names']==FEATURE_NAMES
    x=np.asarray(row['features'],dtype=np.float64)
    assert x.shape==(26,) and np.isfinite(x).all()
    # The pre-existing 27D diagnostic uses the optimizer-return objective fun,
    # not the released-code exported second-noising residual in feature24.
    objective=float(row['modules']['noise_optim']['optimizer']['fun']);assert math.isfinite(objective)
    return np.r_[x,objective]


def representation(x,name):
    assert x.ndim==3 and x.shape[1:]==(2,27)
    if name=='image26':return x[:,:,:26].reshape(-1,26)
    if name=='mean26':return x[:,:,:26].mean(axis=1)
    assert name in ['meanmax52','meanmax54']
    z=x[:,:,:26] if name=='meanmax52' else x
    return np.concatenate([z.mean(axis=1),z.max(axis=1)],axis=1)


def predict(x,spec,parameter=None):
    if spec['kind']=='scalar':
        assert spec['direction']==-1 and spec['feature_index'] in range(4)
        z=-x[:,:,spec['feature_index']]
        result=z.mean(axis=1) if spec['pool_after_direction']=='mean' else z.max(axis=1)
        # Only final two-image summation/division can differ from a saved scalar.
        bound=8*np.finfo(float).eps*np.maximum(1,np.abs(z).sum(axis=1))
        return result,bound,None
    p=parameter['parameters'];assert p['classes']==[0,1]
    xx=representation(x,spec['representation']);d=xx.shape[1]
    mean=np.asarray(p['scaler_mean']);scale=np.asarray(p['scaler_scale'])
    coef=np.asarray(p['coef']);intercept=np.asarray(p['intercept'])
    assert mean.shape==scale.shape==(d,) and coef.shape==(1,d) and intercept.shape==(1,)
    assert np.isfinite(xx).all() and np.isfinite(coef).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale>0)
    z=(xx-mean)/scale;logits=(z@coef.T+intercept).reshape(-1);prob=expit(logits)
    # Conservative dimension-scaled FP64 arithmetic envelope to compare the same
    # stored scaler/logistic computation across batch shapes. No efficacy cutoff.
    unit=np.finfo(float).eps;g=(d+8)*unit/(1-(d+8)*unit)
    error=8*g*(np.sum(np.abs(z*coef),axis=1)+abs(intercept[0])+1)
    image_probability=None
    if spec['representation']=='image26':
        image_probability=prob.reshape(len(x),2)
        result=image_probability.mean(axis=1) if spec['patient_probability_pool']=='mean' else image_probability.max(axis=1)
        error=error.reshape(len(x),2).max(axis=1)+8*unit
    else:result=prob
    assert result.shape==(len(x),) and np.isfinite(result).all()
    return result,error,image_probability


def fit79(x,y,ids,spec,chosen_C,config):
    assert len(x)==len(y)==len(ids)==79 and PID not in ids and len(set(ids))==79
    assert Counter(y)=={0:40,1:39}
    xx=representation(x,spec['representation']);yy=np.repeat(y,2) if spec['representation']=='image26' else y
    assert len(xx)==len(yy)==(158 if spec['representation']=='image26' else 79)
    assert config['solver']=='liblinear' and config['penalty']=='l2' and config['class_weight'] is None
    scaler=StandardScaler(with_mean=True,with_std=True)
    transformed=scaler.fit_transform(xx)
    lr=LogisticRegression(C=chosen_C,solver=config['solver'],penalty=config['penalty'],max_iter=config['max_iter'],
        random_state=config['random_state'],class_weight=None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');lr.fit(transformed,yy)
    convergence=[str(w.message) for w in captured if issubclass(w.category,ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Invalid unconverged fixed-policy fit79 '+spec['id']+': '+'; '.join(convergence))
    assert lr.classes_.tolist()==[0,1] and np.isfinite(lr.coef_).all()
    return dict(arm=REFIT_ARM,model='original_model_1_auxiliary_features',method=spec['id'],
        C_policy=spec['C_policy'],chosen_C=float(chosen_C),representation=spec['representation'],
        fitting_patient_ids=ids,fitting_rows=len(yy),excluded_patient_id=PID,
        source_feature_matrix_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
        fitting_labels=y.tolist(),solver=config['solver'],penalty=config['penalty'],
        max_iter=config['max_iter'],random_state=config['random_state'],class_weight=None,
        selection_or_control_used_in_fitting=False,new_C_search=False,
        historical_tuned_C_selection_included_patient=spec['C_policy']=='tuned',
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],
        parameters=dict(scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),scaler_var=scaler.var_.tolist(),
            scaler_n_samples_seen=int(scaler.n_samples_seen_),coef=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),
            classes=lr.classes_.tolist(),n_iter=lr.n_iter_.tolist()))


def distribution(x):
    x=np.asarray(x,dtype=np.float64);assert len(x) and np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),sample_sd=float(x.std(ddof=1)) if len(x)>1 else None,
        min=float(x.min()),max=float(x.max()),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),
        positive=int((x>0).sum()),zero=int((x==0).sum()),negative=int((x<0).sum()))


def original_data(c):
    old=Path(c['original_directory']);ad=old/'analysis_v1'
    v=read(old/'verification.json');p=read(old/'protocol.json');e=read(old/'execution.json')
    assert v['status']=='PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY'
    assert v['analysis_verification']['status']=='PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS'
    for name,key in [('results.json','results_sha256'),('protocol.json','protocol_sha256'),('execution.json','execution_sha256')]:
        assert sha(old/name)==v[key]
    assert e['results_sha256']==v['results_sha256'] and e['protocol_sha256']==v['protocol_sha256']
    provenance=read(ad/'provenance.json');assert sha(ad/'provenance.json')==v['analysis_verification']['provenance_sha256']
    for f,h in provenance['output_sha256'].items():assert sha(ad/f)==h
    contract=read(ad/'contract_copy.json');assert c['methods']==contract['methods']
    rows=[r for r in read(old/'results.json') if r['model']=='model_1']
    assert len(rows)==240
    index={r['image_id']:r for r in rows};assert len(index)==240
    for r in rows:
        assert r['scenario']=='U' and r['image_member']==0 and r['actual_training_exposures']==0
        assert int(r['member'])==int(r['assignment_group']=='A')
    params={r['method']:r for r in read(ad/'fit_parameters.json') if r['model']=='model_1'}
    assert len(params)==10 and all(PID in r['fitting_patient_ids'] for r in params.values())
    predictions={(r['method'],r['patient_id']):r for r in read(ad/'predictions.json') if r['model']=='model_1'}
    assert len(predictions)==19*120
    return old,contract,index,params,predictions


def matrix(index,patients):
    values=[]
    for p in patients:
        rr=[index[i] for i in p['images']];assert len(rr)==2
        for r in rr:
            assert r['patient_id']==p['patient_id'] and r['eval_role']==p['role'] and r['assignment_group']==p['group']
        values.append([features(r) for r in rr])
    result=np.asarray(values);assert result.shape==(len(patients),2,27)
    return result


def run(args):
    started=time.perf_counter();assert args.expected_code_sha256 and sha(__file__)==args.expected_code_sha256
    c,a=validate_policy();out=args.run_dir
    assert out.resolve()==Path(c['extraction_directory']).resolve()
    verification=read(out/'verification.json');execution=read(out/'execution.json');protocol=read(out/'protocol.json')
    assert verification['status']==RAW_STATUS and verification['complete'] is True
    assert execution['complete'] is True and execution['status']=='PASS_MASKED_CDI_U82_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION'
    for name,key in [('results.json','results_sha256'),('protocol.json','protocol_sha256'),('execution.json','execution_sha256')]:
        assert sha(out/name)==verification[key],name
    assert protocol['contract_sha256']==verification['contract_sha256']==execution['contract_sha256']
    extraction=protocol['contract'];assert sha(protocol['contract_path'])==protocol['contract_sha256']
    assert extraction==read(protocol['contract_path'])
    assert extraction['schema']=='cdi-masked-u82-contract/v1' and extraction['expected_records']==82
    frozen=extraction['frozen_sha256']
    assert frozen[str(POLICY.resolve())]==POLICY_SHA and frozen[str(ADDENDUM.resolve())]==ADDENDUM_SHA
    for f,h in frozen.items():assert sha(f)==h
    assert Path(extraction['treatment_directory']).resolve()==Path(c['original_directory']).resolve()
    dest=out/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    bindings={str((out/name).resolve()):sha(out/name) for name in ['results.json','protocol.json','execution.json','verification.json']}
    bindings.update(c['frozen_sha256']);bindings[str(POLICY.resolve())]=POLICY_SHA;bindings[str(ADDENDUM.resolve())]=ADDENDUM_SHA
    new(dest/'protocol.json',dict(schema='masked-cdi-fixed-and-excluded-patient-analysis/v1',
        source_code_sha256=args.expected_code_sha256,original_policy_sha256=POLICY_SHA,addendum_sha256=ADDENDUM_SHA,
        extraction_contract_sha256=protocol['contract_sha256'],inputs_sha256=bindings,
        policy_fixed_before_GPU=True,analyzer_code_timing='Bound at analysis execution; no claim this implementation was frozen before GPU.',
        original_arm_new_fitting=False,additional_arm_fitting='only original U79 auxiliary features, fixed historical C; no control/selection fitting'))
    try:
        old,original_contract,index,params,old_predictions=original_data(c)
        patients=c['patient_rows'];assert patients[0]['patient_id']==PID and len(patients)==41
        ids=[p['patient_id'] for p in patients];assert len(set(ids))==41
        assert Counter(p['group'] for p in patients if p['role']=='selection')=={'A':20,'B':20}
        treatment=matrix(index,patients)
        control_rows=read(out/'results.json');assert len(control_rows)==82
        control_index={r['image_id']:r for r in control_rows};assert set(control_index)=={i for p in patients for i in p['images']}
        for p in patients:
            for iid in p['images']:
                r=control_index[iid]
                assert r['model']=='masked_control' and r['branch']=='control' and r['scenario']=='U'
                assert int(r['member'])==int(p['group']=='A' and p['patient_id']!=PID)
                assert r['actual_training_exposures']==0 and r['image_member']==0
                assert (r['patient_training_exposures']>0)==bool(r['member'])
                assert r['checkpoint_sha256']==extraction['checkpoint_sha256']['control']
                assert r['contract_sha256']==protocol['contract_sha256']
        control=matrix(control_index,patients)
        # Fit only historical original-U data. The control matrix is never passed
        # to fit79 or used for any hyperparameter/feature/reference choice.
        fitpatients=[p for p in original_contract['patient_rows'] if p['patient_id'] in a['training_patient_ids']]
        assert [p['patient_id'] for p in fitpatients]==a['training_patient_ids']
        assert all(p['role']=='fit' and p['patient_id']!=PID for p in fitpatients)
        fitx=matrix(index,fitpatients);fity=np.asarray([int(p['group']=='A') for p in fitpatients])
        assert Counter(fity)=={0:40,1:39}
        refits={};restoration=[];output=[];image_probabilities=[];contexts=[]
        for spec in c['methods']:
            method=spec['id'];parameter=params.get(method)
            if parameter:
                assert parameter['chosen_C']==a['historical_chosen_C'][method]
                if spec['C_policy']=='fixed':assert parameter['chosen_C']==a['logistic_regression']['fixed_C']==1.
                refits[method]=fit79(fitx,fity,a['training_patient_ids'],spec,parameter['chosen_C'],a['logistic_regression'])
            for arm,chosen_parameter in [(ORIGINAL_ARM,parameter),(REFIT_ARM,refits.get(method))]:
                st,bt,it=predict(treatment,spec,chosen_parameter);sc,bc,ic=predict(control,spec,chosen_parameter)
                if arm==ORIGINAL_ARM:
                    for j,p in enumerate(patients):
                        original=old_predictions[method,p['patient_id']]
                        assert original['role']==p['role'] and original['group']==p['group']
                        error=abs(st[j]-original['score']);assert error<=bt[j]
                        restoration.append(dict(method=method,patient_id=p['patient_id'],original=original['score'],reproduced=float(st[j]),
                            absolute_error=float(error),FP64_arithmetic_envelope=float(bt[j]),bitwise_equal=bool(st[j]==original['score'])))
                for j,p in enumerate(patients):
                    clean=arm==REFIT_ARM and method in a['clean_primary_methods']
                    row=dict(arm=arm,method=method,patient_id=p['patient_id'],role=p['role'],group=p['group'],
                        treatment_member=int(p['group']=='A'),control_member=int(p['group']=='A' and p['patient_id']!=PID),
                        treatment_score=float(st[j]),control_score=float(sc[j]),delta=float(st[j]-sc[j]),
                        clean_fixed_C_primary=clean,source_quirk_diagnostic=spec.get('source_quirk_diagnostic',False),
                        historical_tuned_C_selection_included_patient=spec.get('C_policy')=='tuned',
                        target_patient_seen_in_coefficient_fit=arm==ORIGINAL_ARM and spec['kind']=='logistic_regression',
                        scalar_no_fit=spec['kind']=='scalar',chosen_C=chosen_parameter['chosen_C'] if chosen_parameter else None)
                    output.append(row)
                    if it is not None:
                        image_probabilities.append(dict(arm=arm,method=method,patient_id=p['patient_id'],images=p['images'],
                            treatment=it[j].tolist(),control=ic[j].tolist(),pool=spec['patient_probability_pool']))
                for group in ['A','B','all']:
                    jj=[j for j,p in enumerate(patients) if p['role']=='selection' and (group=='all' or p['group']==group)]
                    assert len(jj)==(40 if group=='all' else 20)
                    contexts.append(dict(arm=arm,method=method,group=group,delta=distribution((st-sc)[jj])))
        # Target-independent public encoder control is copied, not retrained.
        for p in patients:
            oldscore=old_predictions['dino_existing',p['patient_id']]['score']
            output.append(dict(arm=ORIGINAL_ARM,method='dino_existing',patient_id=p['patient_id'],role=p['role'],group=p['group'],
                treatment_member=int(p['group']=='A'),control_member=int(p['group']=='A' and p['patient_id']!=PID),
                treatment_score=oldscore,control_score=oldscore,delta=0.,unchanged_by_construction=True,
                clean_fixed_C_primary=False,target_patient_fit_exclusion_not_performed=True))
        for group in ['A','B','all']:
            contexts.append(dict(arm=ORIGINAL_ARM,method='dino_existing',group=group,
                delta=distribution([0.]*(40 if group=='all' else 20)),unchanged_by_construction=True))
        assert len(restoration)==18*41 and len(output)==(19+18)*41 and len(refits)==10
        warning_count=sum(len(p['warnings']) for p in refits.values())
        new(dest/'patient_scores.json',output);new(dest/'selection_context.json',contexts)
        new(dest/'image_probabilities.json',image_probabilities)
        new(dest/'original_score_restoration.json',restoration)
        new(dest/'original_fitted_parameters.json',list(params.values()))
        new(dest/'leave_patient_out79_parameters.json',list(refits.values()))
        new(dest/'leave_patient_out79_fit_inputs.json',dict(patient_rows=fitpatients,image_feature_order=FEATURE_NAMES+['NO_optimizer_return_objective'],
            original_model1_feature_matrix=fitx.tolist(),labels=fity.tolist(),source='original verified U cohort; no control or selection fit',
            feature_matrix_sha256=hashlib.sha256(fitx.astype('<f8').tobytes()).hexdigest()))
        analysis=dict(status='PASS_MASKED_CDI_FROZEN_AND_LEAVE_PATIENT_OUT79_SCORER_ANALYSIS_PENDING_INDEPENDENT_AUDIT',
            current_step=2,patient_id=PID,scenario='U',patients=41,original_methods=18,original_DINO_reuse=1,
            leave_patient_out_methods=18,refitted_learned_scorers=10,fit_patients=79,fit_labels={'nonmember':40,'member':39},
            clean_primary_methods=a['clean_primary_methods'],
            patient14393=[r for r in output if r['patient_id']==PID],
            selection_context=contexts,
            original_restoration=dict(rows=len(restoration),max_absolute_error=max(r['absolute_error'] for r in restoration),
                bitwise_exact=sum(r['bitwise_equal'] for r in restoration),all_within_FP64_arithmetic_envelope=True),
            warnings=warning_count,limits=c['limits']+a['limits']+[a['tuned_C_caveat']],
            original_policy_sha256=POLICY_SHA,leave_out_addendum_sha256=ADDENDUM_SHA,
            new_GPU=False,new_target_training=False,new_C_search=False,control_or_selection_used_for_fitting=False,
            AUC=False,CI=False,method_selection=False,stage2_completion_claim=False,
            seconds=time.perf_counter()-started)
        for f,h in bindings.items():assert sha(f)==h
        assert sha(__file__)==args.expected_code_sha256
        new(dest/'analysis.json',analysis)
        new(dest/'provenance.json',dict(source_code_sha256=args.expected_code_sha256,inputs_sha256=bindings,
            extraction_verification_sha256=sha(out/'verification.json'),protocol_sha256=sha(dest/'protocol.json'),
            outputs_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
            neural_raw_verification='Separate frozen raw verifier; this module performs only saved-feature score analysis and authorized auxiliary fit79'))
        print(json.dumps(dict(status=analysis['status'],seconds=analysis['seconds'],patient_score_rows=len(output),refits=len(refits),warnings=warning_count)))
    except BaseException:
        new(dest/'failure.json',dict(status='FAILED_PRESERVED_MASKED_CDI_SCORER_ANALYSIS',traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
        raise


def self_test():
    x=np.zeros((2,2,27));x[:,0,0]=[1,4];x[:,1,0]=[3,8]
    assert np.array_equal(representation(x,'meanmax52')[:,[0,26]],[[2,3],[6,8]])
    scalar=dict(kind='scalar',direction=-1,feature_index=0,pool_after_direction='max')
    assert np.array_equal(predict(x,scalar)[0],[-1,-4])
    spec=dict(kind='logistic_regression',representation='image26',patient_probability_pool='mean')
    p=dict(parameters=dict(classes=[0,1],scaler_mean=[0]*26,scaler_scale=[1]*26,
        coef=[[1]+[0]*25],intercept=[0]))
    pred,_,_=predict(x,spec,p)
    assert np.allclose(pred,[(expit(1)+expit(3))/2,(expit(4)+expit(8))/2])
    assert not np.allclose(pred,expit([2,6]))
    assert distribution([-1,0,2])['positive']==1
    return dict(status='PASS_FIXED_SCORER_REPRESENTATION_DIRECTION_AND_PROBABILITY_POOLING',new_fitting=False,new_GPU=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=OUT)
    p.add_argument('--output-tag',default='fixed_scorer_analysis_v1');p.add_argument('--expected-code-sha256')
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    run(a)


if __name__=='__main__':main()
