"""Independent CPU arithmetic audit of saved original/leave-patient-out scorers.

Does not import the score producer, refit a classifier, or execute a neural model.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / '_reports/cvpr_u_pilot_v1_001'
OUT = RUN / 'baseline_screen_20260915/cdi_masked_u82_v1/fixed_scorer_analysis_v1'
OLD = RUN / 'baseline_screen_20260914/cdi_u_cohort_v1'
RESEARCH = next(ROOT.parent.glob('*/research_2026-09-10'))
POLICY = RESEARCH / 'spec_sources/masked_cdi_fixed_scorer_analysis_contract_20260915.json'
ADDENDUM = RESEARCH / 'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json'
PRODUCER_SHA = '06eb9dab58ec19191ad111506e9b9ee5e57e6245183d7224250b3c0411b9c6f8'
ORIGINAL, REFIT = 'original_fit80_frozen_patient_seen', 'leave_patient_out79_refit'
PID = '14393'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def features(row):
    x = row['features'] + [row['modules']['noise_optim']['optimizer']['fun']]
    assert len(x) == 27 and np.isfinite(x).all()
    return x


def design(x, representation):
    if representation == 'image26':
        return x[:, :, :26].reshape(-1, 26)
    d = 27 if representation == 'meanmax54' else 26
    means = (x[:, 0, :d] + x[:, 1, :d]) / 2
    if representation == 'mean26':
        return means
    assert representation in ('meanmax52', 'meanmax54')
    return np.column_stack((means, np.maximum(x[:, 0, :d], x[:, 1, :d])))


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def prediction(x, spec, par):
    if spec['kind'] == 'scalar':
        assert spec['direction'] == -1
        z = -x[:, :, spec['feature_index']]
        return (z[:, 0] + z[:, 1]) / 2 if spec['pool_after_direction'] == 'mean' else z.max(1), None
    p = par['parameters']
    assert p['classes'] == [0, 1]
    xx = design(x, spec['representation'])
    mean, scale, w = (np.asarray(p[k], dtype=float) for k in ('scaler_mean', 'scaler_scale', 'coef'))
    assert mean.shape == scale.shape == (xx.shape[1],) and w.shape == (1, xx.shape[1])
    assert (scale > 0).all() and np.isfinite(w).all()
    probs = np.array([sigmoid(math.fsum(((r - mean) / scale) * w[0]) + p['intercept'][0]) for r in xx])
    if spec['representation'] == 'image26':
        per_image = probs.reshape(len(x), 2)
        result = (per_image[:, 0] + per_image[:, 1]) / 2 if spec['patient_probability_pool'] == 'mean' else per_image.max(1)
        return result, per_image
    return probs, None


def distribution(z):
    z = list(z)
    mean = math.fsum(z) / len(z)
    return dict(n=len(z), mean=mean, median=float(np.median(z)),
        sample_sd=math.sqrt(math.fsum((v - mean) ** 2 for v in z) / (len(z) - 1)),
        min=min(z), max=max(z), q25=float(np.quantile(z, .25)), q75=float(np.quantile(z, .75)),
        positive=sum(v > 0 for v in z), zero=sum(v == 0 for v in z), negative=sum(v < 0 for v in z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, default=OUT)
    parser.add_argument('--expected-code-sha256', required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    code_hash = sha(__file__)
    assert code_hash == args.expected_code_sha256
    ad = args.analysis_dir.resolve()
    output = ad / 'independent_saved_scorer_audit.json'
    assert not output.exists()
    assert sha(ROOT / 'u_patient_audit/analyze_masked_cdi_fixed_scorers.py') == PRODUCER_SHA
    c, add = read(POLICY), read(ADDENDUM)
    protocol, provenance, analysis = (read(ad / n) for n in ('protocol.json', 'provenance.json', 'analysis.json'))
    assert protocol['source_code_sha256'] == provenance['source_code_sha256'] == PRODUCER_SHA
    assert protocol['original_policy_sha256'] == sha(POLICY) == add['original_policy_sha256']
    assert protocol['addendum_sha256'] == sha(ADDENDUM)
    assert analysis['status'] == 'PASS_MASKED_CDI_FROZEN_AND_LEAVE_PATIENT_OUT79_SCORER_ANALYSIS_PENDING_INDEPENDENT_AUDIT'
    assert analysis['warnings'] == 0 and analysis['fit_patients'] == 79
    for key in ('new_GPU', 'new_target_training', 'new_C_search', 'control_or_selection_used_for_fitting', 'AUC', 'CI', 'method_selection', 'stage2_completion_claim'):
        assert analysis[key] is False
    for name, h in provenance['outputs_sha256'].items():
        assert sha(ad / name) == h
    assert provenance['extraction_verification_sha256'] == sha(ad.parent / 'verification.json')
    rawv = read(ad.parent / 'verification.json')
    assert rawv['status'] == 'PASS_MASKED_CDI_U82_SAVED_ARITHMETIC_PROVENANCE_AND_DL_REPLAY'
    assert rawv['results_sha256'] == sha(ad.parent / 'results.json')
    for source, h in provenance['inputs_sha256'].items():
        assert sha(source) == h
    oldrows = {r['image_id']: r for r in read(OLD / 'results.json') if r['model'] == 'model_1'}
    newrows = {r['image_id']: r for r in read(ad.parent / 'results.json')}
    oldcontract = read(OLD / 'analysis_v1/contract_copy.json')
    patients = c['patient_rows']
    assert len(patients) == 41 and patients[0]['patient_id'] == PID
    assert set(newrows) == {i for p in patients for i in p['images']}
    fitpatients = [p for p in oldcontract['patient_rows'] if p['patient_id'] in add['training_patient_ids']]
    assert [p['patient_id'] for p in fitpatients] == add['training_patient_ids'] and len(fitpatients) == 79
    assert all(p['patient_id'] != PID and p['role'] == 'fit' for p in fitpatients)
    assert not {p['patient_id'] for p in fitpatients} & {p['patient_id'] for p in patients if p['role'] == 'selection'}
    def matrix(index, pp):
        for p in pp:
            assert len(p['images']) == 2
            for i in p['images']:
                r = index[i]
                assert (r['patient_id'], r['eval_role'], r['assignment_group']) == (p['patient_id'], p['role'], p['group'])
        return np.asarray([[features(index[i]) for i in p['images']] for p in pp], dtype=np.float64)
    treatment, control, fitx = matrix(oldrows, patients), matrix(newrows, patients), matrix(oldrows, fitpatients)
    labels = [int(p['group'] == 'A') for p in fitpatients]
    assert Counter(labels) == {0: 40, 1: 39}
    savedfit = read(ad / 'leave_patient_out79_fit_inputs.json')
    assert savedfit['patient_rows'] == fitpatients and savedfit['labels'] == labels
    assert np.array_equal(savedfit['original_model1_feature_matrix'], fitx)
    matrix_hash = hashlib.sha256(fitx.astype('<f8').tobytes()).hexdigest()
    assert savedfit['feature_matrix_sha256'] == matrix_hash
    original = {r['method']: r for r in read(ad / 'original_fitted_parameters.json')}
    oldparams = {r['method']: r for r in read(OLD / 'analysis_v1/fit_parameters.json') if r['model'] == 'model_1'}
    assert original == oldparams and len(original) == 10
    refit = {r['method']: r for r in read(ad / 'leave_patient_out79_parameters.json')}
    assert set(refit) == set(original)
    methods = {s['id']: s for s in c['methods']}
    assert c['methods'] == add['methods'] and len(methods) == 18
    errors, counts = {}, Counter()
    def close(x, y, label, rtol=2e-11, atol=2e-12):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        error = float(np.max(np.abs(x - y))) if x.size else 0.
        errors[label] = max(errors.get(label, 0.), error)
        counts[label] += 1
        assert np.allclose(x, y, rtol=rtol, atol=atol), (label, error)
    for method, par in refit.items():
        spec = methods[method]
        assert par['fitting_patient_ids'] == add['training_patient_ids'] and par['excluded_patient_id'] == PID
        assert par['fitting_labels'] == labels and par['source_feature_matrix_sha256'] == matrix_hash
        assert par['chosen_C'] == original[method]['chosen_C'] == add['historical_chosen_C'][method]
        assert not par['warnings'] and par['selection_or_control_used_in_fitting'] is False and par['new_C_search'] is False
        assert par['historical_tuned_C_selection_included_patient'] == (spec['C_policy'] == 'tuned')
        if spec['C_policy'] == 'fixed':
            assert par['chosen_C'] == 1.
        xx = design(fitx, spec['representation'])
        assert par['fitting_rows'] == len(xx) == par['parameters']['scaler_n_samples_seen']
        assert len(xx) == (158 if spec['representation'] == 'image26' else 79)
        means = np.array([math.fsum(xx[:, j]) / len(xx) for j in range(xx.shape[1])])
        variances = np.array([math.fsum((xx[:, j] - means[j]) ** 2) / len(xx) for j in range(xx.shape[1])])
        eps = np.finfo(float).eps
        constant = variances <= len(xx) * eps * variances + (len(xx) * means * eps) ** 2
        scales = np.where(constant, 1., np.sqrt(variances))
        close(means, par['parameters']['scaler_mean'], 'fit79_scaler_mean')
        close(variances, par['parameters']['scaler_var'], 'fit79_scaler_variance')
        close(scales, par['parameters']['scaler_scale'], 'fit79_scaler_scale')
        assert par['parameters']['classes'] == [0, 1]
    scores = read(ad / 'patient_scores.json')
    index = {(r['arm'], r['method'], r['patient_id']): r for r in scores}
    assert len(index) == len(scores) == 1517
    oldpred = {(r['method'], r['patient_id']): r for r in read(OLD / 'analysis_v1/predictions.json') if r['model'] == 'model_1'}
    imageprobs = {(r['arm'], r['method'], r['patient_id']): r for r in read(ad / 'image_probabilities.json')}
    for method, spec in methods.items():
        for arm, params in ((ORIGINAL, original), (REFIT, refit)):
            st, it = prediction(treatment, spec, params.get(method))
            sc, ic = prediction(control, spec, params.get(method))
            for j, p in enumerate(patients):
                r = index[arm, method, p['patient_id']]
                assert (r['role'], r['group']) == (p['role'], p['group'])
                assert r['treatment_member'] == int(p['group'] == 'A') and r['control_member'] == int(p['group'] == 'A' and p['patient_id'] != PID)
                assert r['clean_fixed_C_primary'] == (arm == REFIT and method in add['clean_primary_methods'])
                assert r['historical_tuned_C_selection_included_patient'] == (spec.get('C_policy') == 'tuned')
                assert r['target_patient_seen_in_coefficient_fit'] == (arm == ORIGINAL and spec['kind'] == 'logistic_regression')
                close(st[j], r['treatment_score'], 'saved_treatment_prediction')
                close(sc[j], r['control_score'], 'saved_control_prediction')
                assert r['delta'] == r['treatment_score'] - r['control_score']
                if arm == ORIGINAL:
                    close(st[j], oldpred[method, p['patient_id']]['score'], 'original18_score_restoration')
                if it is not None:
                    q = imageprobs[arm, method, p['patient_id']]
                    assert q['images'] == p['images'] and q['pool'] == spec['patient_probability_pool']
                    close(it[j], q['treatment'], 'image_probability_treatment')
                    close(ic[j], q['control'], 'image_probability_control')
    for p in patients:
        r = index[ORIGINAL, 'dino_existing', p['patient_id']]
        assert r['treatment_score'] == r['control_score'] == oldpred['dino_existing', p['patient_id']]['score']
        assert r['delta'] == 0 and r['unchanged_by_construction'] is True
    restoration = read(ad / 'original_score_restoration.json')
    assert len(restoration) == 738
    for r in restoration:
        assert r['original'] == oldpred[r['method'], r['patient_id']]['score']
        assert r['reproduced'] == index[ORIGINAL, r['method'], r['patient_id']]['treatment_score']
        assert r['absolute_error'] == abs(r['original'] - r['reproduced'])
        assert r['bitwise_equal'] == (r['original'] == r['reproduced'])
    contexts = read(ad / 'selection_context.json')
    assert len(contexts) == 111
    for r in contexts:
        rr = [s['delta'] for s in scores if s['arm'] == r['arm'] and s['method'] == r['method'] and s['role'] == 'selection'
              and (r['group'] == 'all' or s['group'] == r['group'])]
        assert len(rr) == (40 if r['group'] == 'all' else 20)
        for k, value in distribution(rr).items():
            close(value, r['delta'][k], 'context_' + k)
    assert analysis['clean_primary_methods'] == add['clean_primary_methods']
    assert len(add['clean_primary_methods']) == 4 and all(methods[m]['C_policy'] == 'fixed' and not methods[m]['source_quirk_diagnostic'] for m in add['clean_primary_methods'])
    assert analysis['patient14393'] == [r for r in scores if r['patient_id'] == PID]
    assert analysis['selection_context'] == contexts
    paths = [ad / n for n in ('protocol.json', 'provenance.json', 'analysis.json', 'patient_scores.json',
        'selection_context.json', 'original_fitted_parameters.json', 'leave_patient_out79_parameters.json',
        'leave_patient_out79_fit_inputs.json', 'original_score_restoration.json', 'image_probabilities.json')]
    paths += [POLICY, ADDENDUM, ad.parent / 'verification.json', ad.parent / 'results.json', OLD / 'results.json', Path(__file__)]
    report = dict(status='PASS_INDEPENDENT_MASKED_CDI_FIT79_INPUT_SCALER_PREDICTION_AUDIT', seconds_cpu=time.perf_counter() - start,
        source_code_sha256=code_hash, score_producer_sha256=PRODUCER_SHA,
        input_sha256={str(p.resolve()): sha(p) for p in paths},
        counts=dict(fit_patients=79, nonmembers=40, members=39, excluded_patient_id=PID, learned_scorers=10,
            saved_patient_scores=1517, branch_predictions=2952, original18_restorations=738, unchanged_DINO41=41, contexts=111),
        max_absolute_errors=errors, checks=dict(counts), clean_primary_methods=add['clean_primary_methods'],
        all_historical_C_preserved=True, no_new_C_search=True, selected_patient_absent_from_auxiliary_fit=True,
        no_selection_or_control_features_in_fit=True, new_classifier_fits_in_this_audit=0, new_neural_queries=0,
        limits=['Checks saved original-U79 inputs, scaler statistics and coefficients-to-prediction arithmetic; does not independently rerun the logistic solver.',
            'Fixed-C four methods exclude the selected patient from auxiliary coefficient/scaler fitting; historical tuned C still involved the patient.',
            'Feature-producing original target was trained with the selected patient; auxiliary sample exclusion does not make its target weights patient-independent.',
            'One masked-patient intervention; no population AUC, CI, causal generality or method-selection claim.'])
    assert sha(__file__) == code_hash
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(status=report['status'], seconds_cpu=report['seconds_cpu'], audit_sha256=sha(output),
        counts=report['counts'], max_absolute_errors=errors)))


if __name__ == '__main__':
    main()
