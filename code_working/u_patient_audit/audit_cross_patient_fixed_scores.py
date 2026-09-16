"""Independent saved-score arithmetic for the fixed p/q interventions.

Does not import a score producer, fit a classifier, or run a neural model.
Independent sums/sigmoid and scalar distribution formulae use Python float64.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / '_reports/cvpr_u_pilot_v1_001'
RESEARCH = next(ROOT.parent.glob('*/research_2026-09-10'))
POLICY = RESEARCH / 'spec_sources/patient_cross_intervention_analysis_contract_20260915.json'
POLICY_SHA = 'b0855a8bdec2e1a5f738d35ee12930bf0dfd7a21c603cf55f343ba99ccf5bc70'
CORRECTION = RESEARCH / 'spec_sources/cross_intervention_dino_metadata_correction_20260915.json'
CORRECTION_SHA = 'ecdd2c3af6dbe659371b13c9f878255aa1a049199df8a623d9b7cb97d9457926'
PRODUCER = Path(__file__).with_name('analyze_cross_patient_intervention_v2.py')
PRODUCER_SHA = 'e3c191ba1e1430cf5bf6b1f6349cff3d71b2437db9bae43e417af5d10f30601b'
HELPER_SHA = '06eb9dab58ec19191ad111506e9b9ee5e57e6245183d7224250b3c0411b9c6f8'
PRIOR_AUDITOR_SHA = 'c90246e17337a427250cf85fecd74ebdb546f699a1d07126c80db88c20aff1e4'
OUTPUT = RUN / 'baseline_screen_20260915/cdi_cross_patient_intervention_v1/existing_scorer_analysis_v2'
BRANCHES = ['treatment', 'p_control', 'q_control']
ORIGINAL, REFIT = 'original_fit80_frozen_patient_seen', 'leave_patient_out79_refit'
STATUS = 'PASS_INDEPENDENT_CROSS_PATIENT_FIXED_SCORER_ARITHMETIC_AND_PROVENANCE'
EPS = sys.float_info.epsilon


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def design(pair, name):
    assert len(pair) == 2 and all(len(row) == 27 for row in pair)
    if name == 'image26':
        return [row[:26] for row in pair]
    d = 27 if name == 'meanmax54' else 26
    mean = [math.fsum([pair[0][j], pair[1][j]]) / 2 for j in range(d)]
    if name == 'mean26':
        return [mean]
    assert name in ['meanmax52', 'meanmax54']
    return [mean + [max(pair[0][j], pair[1][j]) for j in range(d)]]


def predict(pair, spec, par):
    if spec['kind'] == 'scalar':
        assert spec['direction'] == -1 and spec['feature_index'] in range(4)
        z = [-row[spec['feature_index']] for row in pair]
        score = math.fsum(z)/2 if spec['pool_after_direction'] == 'mean' else max(z)
        return score, 8*EPS*max(1., math.fsum(map(abs, z))), None
    p = par['parameters']
    assert p['classes'] == [0, 1] and len(p['coef']) == len(p['intercept']) == 1
    prob, error = [], []
    for row in design(pair, spec['representation']):
        d = len(row)
        assert len(p['coef'][0]) == len(p['scaler_mean']) == len(p['scaler_scale']) == d
        assert all(v > 0 for v in p['scaler_scale'])
        terms = [(x-m)/s*w for x,m,s,w in zip(row,p['scaler_mean'],p['scaler_scale'],p['coef'][0])]
        assert all(math.isfinite(v) for v in terms)
        prob.append(sigmoid(math.fsum(terms) + p['intercept'][0]))
        g = (d+8)*EPS / (1-(d+8)*EPS)
        error.append(8*g*(math.fsum(map(abs,terms))+abs(p['intercept'][0])+1))
    if spec['representation'] == 'image26':
        assert spec['patient_probability_pool'] in ['mean', 'max']
        return (math.fsum(prob)/2 if spec['patient_probability_pool'] == 'mean' else max(prob),
                max(error)+8*EPS, prob)
    assert len(prob) == 1
    return prob[0], error[0], None


def quantile(z, fraction):
    z = sorted(z)
    pos = fraction*(len(z)-1)
    left = math.floor(pos)
    if left == len(z)-1:
        return z[left]
    return z[left] + (z[left+1]-z[left])*(pos-left)


def distribution(z):
    assert len(z) > 1 and all(math.isfinite(v) for v in z)
    mean = math.fsum(z)/len(z)
    return dict(n=len(z), mean=mean, median=quantile(z,.5),
        sample_sd=math.sqrt(math.fsum((v-mean)**2 for v in z)/(len(z)-1)),
        min=min(z), max=max(z), q25=quantile(z,.25), q75=quantile(z,.75),
        positive=sum(v>0 for v in z), zero=sum(v==0 for v in z), negative=sum(v<0 for v in z))


def groups(patients):
    result = {}
    rules = [('selection_A',lambda p:p['role']=='selection' and p['group']=='A'),
        ('selection_B',lambda p:p['role']=='selection' and p['group']=='B'),
        ('selection_all',lambda p:p['role']=='selection'),('all41',lambda p:True),
        ('background39_A',lambda p:p['patient_id'] not in ['14393','4092'] and p['group']=='A'),
        ('background39_B',lambda p:p['patient_id'] not in ['14393','4092'] and p['group']=='B'),
        ('background39_all',lambda p:p['patient_id'] not in ['14393','4092']),
        ('p_others40',lambda p:p['patient_id']!='14393'),('q_others40',lambda p:p['patient_id']!='4092')]
    for name, rule in rules:
        result[name] = [p['patient_id'] for p in patients if rule(p)]
    assert [len(v) for v in result.values()] == [20,20,40,41,19,20,39,40,40]
    return result


def envelope(values):
    return 64*EPS*max(1.,math.fsum(map(abs, values)))


def contrast(scores, bounds):
    # Inputs contain two queries, p then q, under one unchanged scorer.
    t,p,q = (scores[b] for b in BRANCHES)
    dp = [t[i]-p[i] for i in range(2)]
    dq = [t[i]-q[i] for i in range(2)]
    d = [dp,dq]
    rp, rq = dp[0]-dq[0], -(dp[1]-dq[1])
    k = rp+rq
    ib = envelope([*t,*p,*q,*dp,*dq])
    rb = [bounds['p_control'][i]+bounds['q_control'][i]+ib for i in range(2)]
    rowmean = [math.fsum(row)/2 for row in d]
    colmean = [math.fsum([d[0][j],d[1][j]])/2 for j in range(2)]
    grand = math.fsum([*dp,*dq])/4
    center = [[d[i][j]-rowmean[i]-colmean[j]+grand for j in range(2)] for i in range(2)]
    return dict(D=d,R_p=rp,R_q=rq,K=k,bounds=rb+[math.fsum(rb)],
        diagonal_mean=math.fsum([d[0][0],d[1][1]])/2,
        off_diagonal_mean=math.fsum([d[0][1],d[1][0]])/2,
        diagonal_minus_off_diagonal=k/2,
        base_cancelled_K=(q[0]-p[0])-(q[1]-p[1]),
        row_column_centered_D=center,
        centered_K=center[0][0]+center[1][1]-center[0][1]-center[1][0],
        arithmetic_identity_envelope=ib)


class Audit:
    def __init__(self):
        self.inputs, self.errors, self.counts = {}, {}, Counter()

    def bind(self,path,expected=None):
        path = Path(path).resolve(); key = str(path)
        if key not in self.inputs:
            self.inputs[key] = sha(path)
        if expected is not None:
            assert self.inputs[key] == expected, key

    def bindings(self,mapping):
        for path,h in mapping.items(): self.bind(path,h)

    def close(self,x,y,label,rtol=2e-11,atol=2e-12):
        x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(),label
        err = float(np.max(np.abs(x-y))) if x.size else 0.
        self.errors[label] = max(self.errors.get(label,0.),err); self.counts[label] += 1
        assert np.allclose(x,y,rtol=rtol,atol=atol),(label,err)


def parameter_sources(a,c):
    prior = Path(c['prior_scorer_directory'])
    priorv = read(prior/'independent_saved_scorer_audit.json')
    assert priorv['status'] == 'PASS_INDEPENDENT_MASKED_CDI_FIT79_INPUT_SCALER_PREDICTION_AUDIT'
    assert priorv['source_code_sha256'] == PRIOR_AUDITOR_SHA
    a.bindings(priorv['input_sha256']);a.bind(prior/'independent_saved_scorer_audit.json')
    original = read(prior/'original_fitted_parameters.json')
    refit = read(prior/'leave_patient_out79_parameters.json')
    params = {ORIGINAL:{r['method']:r for r in original},REFIT:{r['method']:r for r in refit}}
    old = Path(c['original_directory_U'])/'analysis_v1'
    oldparams = {r['method']:r for r in read(old/'fit_parameters.json') if r['model']=='model_1'}
    assert params[ORIGINAL] == oldparams
    add = read(RESEARCH/'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json')
    a.bind(old/'fit_parameters.json');a.bind(RESEARCH/'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json')
    fitinput = read(prior/'leave_patient_out79_fit_inputs.json')
    assert len(fitinput['patient_rows']) == 79 and Counter(fitinput['labels']) == {0:40,1:39}
    ids = [p['patient_id'] for p in fitinput['patient_rows']]
    assert ids == add['training_patient_ids'] and not {'14393','4092'} & set(ids)
    methods = {s['id']:s for s in c['methods']}
    for arm,ps in params.items():
        assert len(ps) == 10 and len(ps) == len(set(ps))
        for method,par in ps.items():
            spec = methods[method]
            assert spec['kind'] == 'logistic_regression' and not par['warnings']
            pp = par['fitting_patient_ids']
            assert len(pp) == len(set(pp)) == (79 if arm==REFIT else 80)
            assert '4092' not in pp and ('14393' in pp) == (arm==ORIGINAL)
            assert par['chosen_C'] == params[ORIGINAL][method]['chosen_C'] == add['historical_chosen_C'][method]
            if spec['C_policy']=='fixed': assert par['chosen_C'] == 1.
            if arm==REFIT:
                assert pp == ids and par['fitting_labels'] == fitinput['labels']
                assert par['excluded_patient_id'] == '14393'
                assert par['selection_or_control_used_in_fitting'] is False and par['new_C_search'] is False
                assert par['historical_tuned_C_selection_included_patient'] == (spec['C_policy']=='tuned')
    saved = read(prior/'patient_scores.json')
    idx = {(r['arm'],r['method'],r['patient_id']):r for r in saved}
    assert len(idx) == len(saved) == 1517
    # The DINO diagnostic itself is a saved fit80 scaler/LR (including p).
    # It is target-independent, not fitting-free. No new DINO fit is performed.
    dino = read(RUN/'encoder_control/scores.json')
    dino = {r['patient_id']:r for r in dino if r['model']=='model_1'}
    assert len(dino)==120 and dino['14393']['role']=='fit' and dino['4092']['role']=='selection'
    for pid,r in dino.items():
        key = ORIGINAL,'dino_existing',pid
        if key in idx: assert idx[key]['treatment_score']==idx[key]['control_score']==r['score']
    a.bind(RUN/'encoder_control/scores.json');a.bind(RUN/'encoder_control/report.json')
    a.bind(ROOT/'u_patient_audit/encoder_control.py')
    return params,idx


def feature_sources(a,c,run):
    v,e,proto = [read(run/n) for n in ['verification.json','execution.json','protocol.json']]
    assert v['status']=='PASS_CROSS_PATIENT_CDI_SAVED_ARITHMETIC_PROVENANCE_AND_P_DL_REPLAY'
    assert v['complete'] is True and e['complete'] is True
    for name,key in [('results.json','results_sha256'),('execution.json','execution_sha256'),('protocol.json','protocol_sha256')]:
        a.bind(run/name,v[key])
    a.bind(run/'verification.json')
    assert proto['contract_sha256']==v['contract_sha256']==e['contract_sha256']
    contract = proto['contract']
    assert read(proto['contract_path'])==contract
    a.bind(proto['contract_path'],proto['contract_sha256']);a.bindings(contract['frozen_sha256'])
    fresh = read(run/'results.json')
    assert Counter((r['branch'],r['scenario']) for r in fresh)=={('q_control','U'):82,('q_control','E'):4,('p_control','E'):4}
    old = {}
    for scenario in ['E','U']:
        path = Path(c['original_directory_'+scenario])/'results.json'
        a.bind(path)
        old[scenario] = [r for r in read(path) if r['model']=='model_1']
    path = Path(c['p_control_U_directory'])/'results.json';a.bind(path)
    pold = read(path)
    matrices,source_rows = {},{}
    for scenario in ['U','E']:
        patients = [r for r in c['patient_rows'] if r['scenario']==scenario]
        assert [p['patient_id'] for p in patients]==c['patient_order_'+scenario]
        wanted = {iid for p in patients for iid in p['images']}
        assert len(wanted)==2*len(patients)
        for branch in BRANCHES:
            rr = old[scenario] if branch=='treatment' else pold if branch=='p_control' and scenario=='U' else [r for r in fresh if r['branch']==branch and r['scenario']==scenario]
            rr = [r for r in rr if r['image_id'] in wanted]
            idx = {r['image_id']:r for r in rr}
            assert len(idx)==len(rr)==len(wanted) and set(idx)==wanted
            for p in patients:
                pair = []
                for iid in p['images']:
                    row = idx[iid]
                    assert (row['patient_id'],row['scenario'],row['eval_role'],row['assignment_group']) == (p['patient_id'],scenario,p['role'],p['group'])
                    if scenario=='U': assert row['image_member']==0 and row['actual_training_exposures']==0
                    xx = row['features']+[row['modules']['noise_optim']['optimizer']['fun']]
                    assert len(row['feature_names'])==26 and len(xx)==27 and all(math.isfinite(x) for x in xx)
                    pair.append(xx)
                    source_rows[branch,scenario,iid] = dict(branch=branch,scenario=scenario,patient_id=p['patient_id'],image_id=iid,
                        feature_names=row['feature_names'],features=row['features'],NO_optimizer_return_objective=float(xx[-1]))
                matrices[branch,scenario,p['patient_id']] = pair
    assert len(source_rows)==258 and len(matrices)==129
    return matrices,source_rows


def metadata(row,arm,spec,scenario,par,c):
    dino = spec['kind']=='unchanged'
    assert row['arm']==arm and row['method']==spec['id'] and row['scenario']==scenario
    assert row['clean_fixed_C_primary']==(arm==REFIT and spec['id'] in c['clean_primary_methods'])
    assert row['historical_tuned_C_selection_included_p']==(spec.get('C_policy')=='tuned')
    assert row['source_quirk_diagnostic']==spec.get('source_quirk_diagnostic',False)
    assert row['no_fitting_in_this_analysis'] is True and row['q_in_prior_coefficient_fit'] is False
    assert row['chosen_C']==(par['chosen_C'] if par else None)
    assert row['p_in_prior_coefficient_fit']==(dino or bool(par and '14393' in par['fitting_patient_ids']))
    assert row['unchanged_by_construction']==dino


def run(args):
    started=time.perf_counter();a=Audit();code=sha(__file__)
    assert args.expected_code_sha256==code
    a.bind(__file__,code);a.bind(PRODUCER,PRODUCER_SHA);a.bind(POLICY,POLICY_SHA);a.bind(CORRECTION,CORRECTION_SHA)
    c=read(POLICY)
    assert c['schema']=='patient-cross-intervention-analysis/v1' and c['branches']==BRANCHES
    assert c['focal_patients']=={'p':'14393','q':'4092'} and c['arms']==[ORIGINAL,REFIT]
    assert len(c['methods'])==18 and c['no_new_fitting'] and c['no_new_C_search'] and c['no_new_AUC'] and c['no_CI'] and c['no_p_value']
    a.bindings(c['frozen_sha256'])
    ad=args.analysis_dir.resolve();dest=ad/'independent_fixed_scorer_audit_v1'
    assert ad.parent==Path(c['new_output_directory']).resolve()
    analysis,proto,prov=[read(ad/n) for n in ['analysis.json','protocol.json','provenance.json']]
    assert analysis['status']=='PASS_CROSS_PATIENT_FIXED_SCORER_RESTORATION_AND_DESCRIPTIVE_ANALYSIS' and analysis['complete'] is True
    assert proto['schema']=='cross-patient-existing-scorer-analysis/v2'
    assert proto['source_code_sha256']==prov['source_code_sha256']==analysis['source_code_sha256']==PRODUCER_SHA
    assert proto['policy_sha256']==prov['policy_sha256']==analysis['analysis_policy_sha256']==POLICY_SHA
    assert proto['metadata_correction_sha256']==CORRECTION_SHA
    assert proto['scorer_helper_sha256']==prov['helper_sha256']==HELPER_SHA
    assert proto['inputs_sha256']==prov['inputs_sha256']
    a.bindings(prov['inputs_sha256'])
    for name,h in prov['outputs_sha256'].items(): a.bind(ad/name,h)
    a.bind(ad/'provenance.json')
    for flag in ['new_fitting','new_C_search','new_GPU','AUC','CI','p_value','efficacy_gate','stage2_completion_claim']:
        assert analysis[flag] is False
    assert proto['new_fitting'] is False and proto['new_GPU'] is False
    assert analysis['extraction_verification_sha256']==sha(ad.parent/'verification.json')
    assert analysis['context_warning']==c['context_warning'] and analysis['decision_policy']==c['decision_policy'] and analysis['limits']==c['limits']
    dest.mkdir(exist_ok=False)
    new(dest/'protocol.json',dict(schema='independent-cross-patient-scorer-audit/v1',
        source_code_sha256=code,producer_sha256=PRODUCER_SHA,policy_sha256=POLICY_SHA,metadata_correction_sha256=CORRECTION_SHA,
        producer_import=False,new_fitting=False,new_GPU=False,
        scope='Saved inputs, coefficients, independent float64 sigmoid and pooling, cross identities, descriptive distributions',
        verification_code_bound_now=True,preexecution_policy_source_bound_by_producer=True))
    try:
        params,saved=parameter_sources(a,c)
        matrices,sources=feature_sources(a,c,ad.parent)
        actual_sources=read(ad/'source_features.json')
        assert len(actual_sources)==len(sources)
        assert {(r['branch'],r['scenario'],r['image_id']):r for r in actual_sources}==sources
        identities=read(ad/'saved_parameter_identity.json')
        expected={arm:{m:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() for m,p in pp.items()} for arm,pp in params.items()}
        assert identities==expected
        rows=read(ad/'patient_scores.json');contrasts=read(ad/'cross_contrasts.json')
        contexts=read(ad/'patient_contexts.json');images=read(ad/'image_probabilities.json')
        restoration=read(ad/'prior_score_restoration.json')
        index={(r['arm'],r['method'],r['scenario'],r['patient_id']):r for r in rows}
        crossidx={(r['arm'],r['method'],r['scenario']):r for r in contrasts}
        contextidx={(r['arm'],r['method'],r['context']):r for r in contexts}
        imageidx={(r['arm'],r['method'],r['scenario'],r['branch'],r['patient_id']):r for r in images}
        restoreidx={(r['arm'],r['method'],r['patient_id']):r for r in restoration}
        assert len(index)==len(rows)==1589 and len(crossidx)==len(contrasts)==73
        assert len(contextidx)==len(contexts)==333 and len(restoreidx)==len(restoration)==1517
        assert len(imageidx)==len(images)==1032
        independent_sign_differences=Counter();all_errors=[]
        for scenario in ['U','E']:
            patients=[p for p in c['patient_rows'] if p['scenario']==scenario]
            masks=groups(patients) if scenario=='U' else {}
            if masks: assert list(masks)==c['contexts']
            for arm in c['arms']:
                specs=c['methods']+([{'id':'dino_existing','kind':'unchanged'}] if scenario=='U' and arm==ORIGINAL else [])
                for spec in specs:
                    method=spec['id'];par=params[arm].get(method)
                    for patient in patients:
                        pid=patient['patient_id'];row=index[arm,method,scenario,pid]
                        metadata(row,arm,spec,scenario,par,c)
                        assert (row['role'],row['group'],row['images'])==(patient['role'],patient['group'],patient['images'])
                        computed={}
                        for branch in BRANCHES:
                            if spec['kind']=='unchanged':
                                score,bound,im=saved[arm,method,pid]['treatment_score'],0.,None
                            else: score,bound,im=predict(matrices[branch,scenario,pid],spec,par)
                            computed[branch]=score
                            a.close(score,row['scores'][branch],'independent_branch_score')
                            a.close(bound,row['score_arithmetic_envelopes'][branch],'score_arithmetic_envelope',rtol=2e-11,atol=1e-25)
                            assert abs(score-row['scores'][branch])<=bound+2*EPS
                            if im is not None:
                                ir=imageidx[arm,method,scenario,branch,pid]
                                metadata(ir,arm,spec,scenario,par,c)
                                assert ir['images']==patient['images'] and ir['pool']==spec['patient_probability_pool']
                                a.close(im,ir['probabilities'],'individual_image_sigmoid')
                                expectedpool=math.fsum(ir['probabilities'])/2 if ir['pool']=='mean' else max(ir['probabilities'])
                                assert expectedpool==row['scores'][branch]
                        direct={'D_p_i':row['scores']['treatment']-row['scores']['p_control'],
                                'D_q_i':row['scores']['treatment']-row['scores']['q_control']}
                        direct['h_i']=direct['D_p_i']-direct['D_q_i']
                        assert all(row[k]==v for k,v in direct.items())
                        dd={'D_p_i':computed['treatment']-computed['p_control'],'D_q_i':computed['treatment']-computed['q_control']}
                        dd['h_i']=dd['D_p_i']-dd['D_q_i']
                        for key,value in dd.items():
                            a.close(value,row[key],'independent_'+key)
                            independent_sign_differences[key]+=int((value>0)-(value<0)!=(row[key]>0)-(row[key]<0))
                        if scenario=='U':
                            old=saved[arm,method,pid];rr=restoreidx[arm,method,pid]
                            for field,value in [('treatment_score',row['scores']['treatment']),('control_score',row['scores']['p_control']),('delta',row['D_p_i'])]:
                                ck=rr['checks'][field]
                                assert ck['saved']==old[field] and ck['reproduced']==value
                                assert ck['absolute_error']==abs(value-old[field]) and ck['exact']==(value==old[field])
                                assert ck['absolute_error']<=ck['envelope'];all_errors.append(ck)
                                a.close(value,old[field],'prior_restoration')
                    cr=crossidx[arm,method,scenario];metadata(cr,arm,spec,scenario,par,c)
                    assert cr['patient_order']==['14393','4092'] and cr['E_uses_U_fitted_scorer']==(scenario=='E')
                    focal=[index[arm,method,scenario,p] for p in ['14393','4092']]
                    sc={b:[r['scores'][b] for r in focal] for b in BRANCHES}
                    bd={b:[r['score_arithmetic_envelopes'][b] for r in focal] for b in BRANCHES}
                    calc=contrast(sc,bd)
                    for key in ['D','diagonal_mean','off_diagonal_mean','diagonal_minus_off_diagonal','base_cancelled_K','row_column_centered_D','centered_K']:
                        a.close(calc[key],cr[key],'cross_'+key)
                    a.close(calc['arithmetic_identity_envelope'],cr['arithmetic_identity_envelope'],'identity_envelope',rtol=2e-11,atol=1e-25)
                    for i,key in enumerate(['R_p','R_q','K']):
                        v,b=calc[key],calc['bounds'][i];r=cr[key]
                        a.close(v,r['value'],'cross_'+key)
                        a.close(b,r['FP64_arithmetic_envelope'],'cross_'+key+'_envelope',rtol=2e-11,atol=1e-25)
                        assert r['raw_sign']==int((v>0)-(v<0))
                        sign='positive' if v>b else 'negative' if v<-b else 'unresolved'
                        assert r['sign_outside_arithmetic_envelope']==sign
                    assert cr['matrix_rows']==['p_intervention','q_intervention'] and cr['matrix_columns']==['p_query','q_query']
                    assert cr['both_own_minus_cross_positive_raw']==(calc['R_p']>0 and calc['R_q']>0)
                    assert cr['both_own_minus_cross_positive_outside_arithmetic_envelope']==(calc['R_p']>calc['bounds'][0] and calc['R_q']>calc['bounds'][1])
                    assert cr['K_positive_does_not_imply_both'] is True
                    assert abs(calc['K']-calc['base_cancelled_K'])<=calc['arithmetic_identity_envelope']
                    assert abs(calc['K']-calc['centered_K'])<=calc['arithmetic_identity_envelope']
                    for context,ids in masks.items():
                        cx=contextidx[arm,method,context];metadata(cx,arm,spec,scenario,par,c)
                        assert cx['patient_ids']==ids and cx['includes_p']==('14393' in ids) and cx['includes_q']==('4092' in ids)
                        assert cx['descriptive_only'] is True and cx['independent_intervention_repetitions'] is False
                        for field in ['D_p_i','D_q_i','h_i']:
                            values=[index[arm,method,scenario,p][field] for p in ids]
                            for stat,value in distribution(values).items():
                                if stat in ['n','positive','zero','negative']: assert value==cx[field][stat]
                                else: a.close(value,cx[field][stat],'context_'+stat)
        assert analysis['all_method_scenario_contrasts']==contrasts
        assert analysis['clean_primary']==[r for r in contrasts if r['clean_fixed_C_primary']]
        assert len(analysis['clean_primary'])==8 and analysis['patient_score_rows']==1589
        assert analysis['saved_scorer_sets']==2 and analysis['saved_learned_scorers']==20
        restoration_summary=analysis['prior_restoration']
        assert restoration_summary['patient_rows']==1517 and restoration_summary['scalar_checks']==4551
        assert restoration_summary['exact']==sum(c['exact'] for c in all_errors)
        assert restoration_summary['max_absolute_error']==max(c['absolute_error'] for c in all_errors)
        assert restoration_summary['all_within_FP64_arithmetic_envelope'] is True
        assert sha(__file__)==code
        report=dict(status=STATUS,complete=True,seconds_cpu=time.perf_counter()-started,
            source_code_sha256=code,producer_sha256=PRODUCER_SHA,policy_sha256=POLICY_SHA,metadata_correction_sha256=CORRECTION_SHA,
            protocol_sha256=sha(dest/'protocol.json'),analysis_sha256=sha(ad/'analysis.json'),
            input_sha256=a.inputs,counts=dict(patient_scores=1589,branch_scores=4767,source_images=258,
                learned_scorers_reused=20,image_probability_rows=1032,cross_contrasts=73,
                prior_patient_restorations=1517,prior_scalar_restorations=4551,contexts=333,
                clean_primary_method_scenarios=8,original_fit=80,excluded_fit=79),
            max_absolute_errors=a.errors,checks=dict(a.counts),independent_rounding_raw_delta_sign_differences=dict(independent_sign_differences),
            DINO_fit80_includes_p=True,DINO_q_not_in_fit=True,DINO_target_response_exact_zero=True,
            new_fits=0,new_neural_queries=0,new_analysis_choices=False,
            limits=['Saved original and U79 parameters are bound unchanged; this audit does not rerun the logistic optimizer.',
                'All three branch predictions use independently summed logits and sigmoid; preserved score arithmetic is also checked exactly.',
                'DINO uses an earlier fit80 auxiliary LR including p, while its fixed target-independent scores have zero intervention response.',
                'Only four fixed-C U79 methods are clean primary; tuned C historically used p, and source target features were learned with p.',
                'Numerical envelopes are arithmetic diagnostics, not confidence intervals, neural error bounds, or evidence of attack efficacy.',
                'Two fixed interventions in one training realization; background patients do not supply independent intervention replications.'])
        new(dest/'verification.json',report)
        print(json.dumps(dict(status=STATUS,seconds_cpu=report['seconds_cpu'],verification_sha256=sha(dest/'verification.json'),
            counts=report['counts'],max_absolute_errors=a.errors)))
    except BaseException:
        new(dest/'failure.json',dict(status='FAILED_PRESERVED_INDEPENDENT_CROSS_PATIENT_SCORER_AUDIT',traceback=traceback.format_exc()))
        raise


def self_test():
    assert sigmoid(0)==.5 and sigmoid(1000)==1 and sigmoid(-1000)==0
    spec={'kind':'scalar','direction':-1,'feature_index':0,'pool_after_direction':'max'}
    pair=[[2.]*27,[3.]*27]
    assert predict(pair,spec,None)[0]==-2
    assert design(pair,'meanmax52')==[([2.5]*26)+([3.]*26)]
    p={'parameters':{'classes':[0,1],'coef':[[1.]*26],'intercept':[-65.],
                    'scaler_mean':[0.]*26,'scaler_scale':[1.]*26}}
    spec={'kind':'logistic_regression','representation':'image26','patient_probability_pool':'mean'}
    score,_,im=predict(pair,spec,p)
    assert abs(score-.5)<EPS and len(im)==2 and im[0]<im[1]
    scores={'treatment':[8.,7.],'p_control':[4.,5.],'q_control':[7.,6.]}
    bounds={b:[0.,0.] for b in BRANCHES};cc=contrast(scores,bounds)
    assert cc['R_p']==3 and cc['R_q']==-1 and cc['K']==2
    assert cc['base_cancelled_K']==cc['centered_K']==2
    shifted={b:[x+v for x,v in zip(z,[100.,-50.])] for b,z in scores.items()}
    shifted['p_control']=[v+17 for v in shifted['p_control']]
    shifted['q_control']=[v-12 for v in shifted['q_control']]
    assert contrast(shifted,bounds)['K']==2
    assert distribution([1.,2.,3.,4.])==dict(n=4,mean=2.5,median=2.5,sample_sd=math.sqrt(5/3),
        min=1.,max=4.,q25=1.75,q75=3.25,positive=4,zero=0,negative=0)
    a=Audit();a.bind(POLICY,POLICY_SHA);a.bind(CORRECTION,CORRECTION_SHA);a.bind(PRODUCER,PRODUCER_SHA)
    c=read(POLICY);groups([p for p in c['patient_rows'] if p['scenario']=='U'])
    params,saved=parameter_sources(a,c)
    assert len(params[ORIGINAL])==len(params[REFIT])==10 and len(saved)==1517
    return dict(status='PASS_CPU_INDEPENDENT_SIGMOID_POOLING_CROSS_IDENTITIES_CONTEXTS_AND_PRIOR_PARAMETER_BINDINGS',
                producer_import=False,new_fitting=False,new_neural_queries=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis-dir',type=Path,default=OUTPUT)
    p.add_argument('--expected-code-sha256');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:
        if args.expected_code_sha256: assert sha(__file__)==args.expected_code_sha256
        print(json.dumps(self_test()));return
    assert args.expected_code_sha256,'A frozen audit source hash is required'
    run(args)


if __name__=='__main__':
    main()
