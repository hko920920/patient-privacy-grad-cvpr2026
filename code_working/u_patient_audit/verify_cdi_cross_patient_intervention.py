"""Independent CPU arithmetic and actual-branch provenance for 90 CDI packets.

Frozen U/E arithmetic helpers are reused only after their exact source hashes
are checked. No producer imports, target inference, fitting or efficacy gates.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / '_reports/cvpr_u_pilot_v1_001'
BASE = RUN / 'baseline_screen_20260915'
OUT = BASE / 'cdi_cross_patient_intervention_v1'
OLDU = RUN / 'baseline_screen_20260914/cdi_u_cohort_v1'
OLDE = RUN / 'baseline_screen_20260914/cdi_e_cohort_v1'
OLDP = BASE / 'cdi_masked_u82_v1'
ENDPOINT = BASE / 'masked_slot_endpoints_v2'
P_TRAIN = BASE / 'masked_patient_training_v1'
Q_TRAIN = BASE / 'masked_patient_4092_training_v1'
P, Q = '14393', '4092'
STATUS = 'PASS_CROSS_PATIENT_CDI_SAVED_ARITHMETIC_PROVENANCE_AND_P_DL_REPLAY'
PRODUCER_SHA = '7fbffff8806ab452186f75a52869df1cb216c65e8a384dc9bdbcfc224299cf2a'
POLICY_SHA = 'b0855a8bdec2e1a5f738d35ee12930bf0dfd7a21c603cf55f343ba99ccf5bc70'
HELPERS = {
    'verify_cdi_u_cohort.py': 'da899f1efbc46f8571baebd2113d204038921b5b5374caba7ad22c7d58b64869',
    'verify_cdi_e_cohort.py': '66c8f4621d1a4f55357a9dace42d6a4ea33403ffcfD94864a9fc53d053ee9f52'.lower(),
    'verify_cdi_kernel.py': '57b7719c54fee4c32c2d92e29535ce44aceb5b784669b4be2fff43aa6dd7d342',
    'analyze_verify_masked_slot_endpoints.py': 'e24aac2f16802fd563f594ab0cd28ded3be8d53c43e84dba7ee7e77bdb5ac913',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(value, label):
    if not value:
        raise AssertionError(label)


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def same(a, b, label):
    if isinstance(a, torch.Tensor):
        need(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
             and torch.equal(a, b), label)
    elif isinstance(a, dict):
        need(isinstance(b, dict) and a.keys() == b.keys(), label + ': keys')
        for k in a:
            same(a[k], b[k], label + '.' + str(k))
    elif isinstance(a, (list, tuple)):
        need(type(a) is type(b) and len(a) == len(b), label + ': type/length')
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, label + '.' + str(i))
    else:
        need(type(a) is type(b) and a == b, label)


def csvrows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def inside(directory, relative):
    p = (directory / relative).resolve()
    need(p.is_relative_to(directory.resolve()), 'source path containment')
    return p


def verified(directory, status, names):
    v = read(directory / 'verification.json')
    need(v['status'] == status, 'independent source verification ' + str(directory))
    if 'complete' in v:
        need(v['complete'] is True, 'source complete')
    for name in names:
        need(v[name + '_sha256'] == sha(directory / (name + '.json')), 'source binding ' + name)
    return v


def scope():
    cohort = csvrows(RUN / 'cohort/evaluation_images.csv')
    selected = sorted({r['patient_id'] for r in cohort if r['eval_role'] == 'selection'}, key=int)
    need(len(selected) == 40 and P not in selected and Q in selected, 'original fit p and selection q')
    order = [P] + selected
    u, e = [], []
    for pid in order:
        rr = sorted((r for r in cohort if r['patient_id'] == pid and r['record_role'] == 'U_observed'), key=lambda r: r['image_id'])
        need(len(rr) == 2, 'fixed U2')
        u.extend(rr)
    for pid in [P, Q]:
        rr = sorted((r for r in cohort if r['patient_id'] == pid and r['record_role'] == 'train_candidate'), key=lambda r: r['image_id'])
        need(len(rr) == 2 and all(r['assignment_group'] == 'A' for r in rr), 'fixed focal E2')
        e.extend(rr)
    need(Counter(r['assignment_group'] for r in u if r['eval_role'] == 'selection') == {'A': 40, 'B': 40}, 'selection A20/B20')
    need(all(r['eval_role'] == ('fit' if r['patient_id'] == P else 'selection') for r in u + e), 'cohort roles')
    for name, value in read(RUN / 'cohort/lock.json')['files'].items():
        need(sha(RUN / 'cohort' / name) == value, 'locked cohort file')
    return u, e, order


def ledgers(c, e_rows):
    pv = verified(P_TRAIN, 'PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC', ['execution', 'protocol'])
    qv = verified(Q_TRAIN, 'PASS_MASKED_PATIENT_4092_SAVED_TRAINING_AND_SOURCE_GATES', ['execution', 'protocol'])
    for directory, v in [(P_TRAIN, pv), (Q_TRAIN, qv)]:
        proto = read(directory / 'protocol.json')
        need(v['contract_sha256'] == proto['contract_sha256'] == sha(proto['contract_path']), 'verified training contract')
    paths = {'treatment': RUN / 'training_coverage_v2/model_1/step_1000.pt',
             'p_control': P_TRAIN / 'control/step_1000.pt', 'q_control': Q_TRAIN / 'control/step_1000.pt'}
    states = {b: torch.load(path, map_location='cpu', weights_only=True) for b, path in paths.items()}
    same(states['treatment'], torch.load(P_TRAIN / 'treatment/step_1000.pt', map_location='cpu', weights_only=True), 'verified exact original treatment')
    need(sha(P_TRAIN / 'treatment/step_1000.pt') == pv['checkpoint_sha256']['treatment']['1000'], 'treatment verification binding')
    need(sha(paths['p_control']) == pv['checkpoint_sha256']['control']['1000'], 'p actual checkpoint')
    need(sha(paths['q_control']) == qv['checkpoint_sha256']['1000'], 'q actual checkpoint')
    manifest = csvrows(RUN / 'cohort/model_1_train.csv')
    image_patient = {r['image_id']: r['patient_id'] for r in manifest}
    need(len(manifest) == len(image_patient) == 912, 'unchanged train manifest')
    need(set(states['treatment']['exposures']) == set(image_patient) and sum(states['treatment']['exposures'].values()) == 4000, 'treatment 912/4000')
    totals = {}
    for branch, state in states.items():
        need(Path(c['checkpoint_paths'][branch]).resolve() == paths[branch].resolve(), 'checkpoint path ' + branch)
        need(c['checkpoint_sha256'][branch] == sha(paths[branch]), 'checkpoint hash ' + branch)
        need(state['step'] == state['attempt'] == 1000, '1000 committed updates/no overflow')
        same(state['contract'], states['treatment']['contract'], 'same numerical contract')
        if branch != 'treatment':
            removed = P if branch == 'p_control' else Q
            expected = dict(states['treatment']['exposures'])
            ids = {r['image_id'] for r in manifest if r['patient_id'] == removed}
            need(ids == {r['image_id'] for r in e_rows if r['patient_id'] == removed} and len(ids) == 2, 'only focal E2 removal')
            for iid in ids:
                need(expected.pop(iid) == 4, 'four removed loss contributions per E image')
            same(state['exposures'], expected, 'only eight loss contributions removed')
            need(sum(expected.values()) == 3992 and len(expected) == 910, 'control 3992/910')
        totals[branch] = Counter()
        for iid, n in state['exposures'].items():
            totals[branch][image_patient[iid]] += n
    need(totals['p_control'][P] == totals['q_control'][Q] == 0 and totals['p_control'][Q] == totals['q_control'][P] == 8, 'opposite focal exposures')
    return states, totals


def references(c, u_rows, e_rows):
    verified(OLDU, 'PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY', ['results', 'protocol'])
    verified(OLDE, 'PASS_SAVED_E_COHORT_ARITHMETIC_AND_INTEGRITY', ['results', 'protocol'])
    verified(OLDP, 'PASS_MASKED_CDI_U82_SAVED_ARITHMETIC_PROVENANCE_AND_DL_REPLAY', ['results', 'protocol', 'execution'])
    verified(ENDPOINT, 'PASS_MASKED_ENDPOINT_SAVED_ARITHMETIC_SOURCE_REUSE_AND_TRAINING_GATES', ['results', 'protocol', 'execution'])
    expected = {(b, scenario, r['image_id']): (d, r) for b, scenario, d, desired in
        [('treatment', 'U', OLDU, u_rows), ('treatment', 'E', OLDE, e_rows), ('p_control', 'U', OLDP, u_rows)]
        for r in read(d / 'results.json') if r['image_id'] in {x['image_id'] for x in desired}
        and (b != 'treatment' or r['model'] == 'model_1')}
    need(len(expected) == len(c['reused_sources']) == 168, '168 exact reused source rows')
    need({(s['branch'], s['scenario'], s['image_id']) for s in c['reused_sources']} == set(expected), 'reuse branch/scenario/image set')
    original = {}
    for s in c['reused_sources']:
        d, r = expected[s['branch'], s['scenario'], s['image_id']]
        rawpath = inside(d, r['raw_path']); rowpath = rawpath.with_suffix('.json')
        need(Path(s['directory']).resolve() == d.resolve() and Path(s['raw_path']).resolve() == rawpath, 'actual source directory/path')
        need(Path(s['row_path']).resolve() == rowpath and s['patient_id'] == r['patient_id'] and r['scenario'] == s['scenario'], 'actual source row metadata')
        need(sha(rawpath) == r['raw_sha256'] == s['raw_sha256'] and sha(rowpath) == s['row_sha256'], 'source raw/row hashes')
        same(read(rowpath), r, 'original aggregate/per-row identity')
        if s['branch'] == 'treatment':
            need(r['checkpoint_sha256'] == c['checkpoint_sha256']['treatment'], 'source treatment checkpoint')
            original[r['image_id']] = (s, r)
        else:
            need(r['checkpoint_sha256'] == c['checkpoint_sha256']['p_control'], 'source p control checkpoint')
    ids = {r['image_id'] for r in e_rows}
    eps = {(r['image_id'], r['repeat']): r for r in read(ENDPOINT / 'results.json')
           if r['branch'] == 'control' and r['family'] == 'cdi_dl_t100' and r['image_id'] in ids}
    need(set(eps) == {(iid, j) for iid in ids for j in range(5)}, '20 prior p-control E DL endpoints')
    return original, eps


def p_dl_replay(raw, row, eps):
    iid = row['image_id']; er0 = eps[iid, 0]
    path = inside(ENDPOINT, er0['raw_path'])
    need(sha(path) == er0['raw_sha256'], 'prior p E raw hash')
    packet = torch.load(path, map_location='cpu', weights_only=True)
    same(packet['latent'], raw['latent'], 'p E replay latent')
    for j in range(5):
        er = eps[iid, j]; cell = packet['cells'][j]
        pred = raw['predictions']['denoising_loss'][j]; noise = raw['noise_draws']['denoising_loss'][j]
        need(er['raw_path'] == er0['raw_path'] and er['raw_sha256'] == er0['raw_sha256'] and er['cell_index'] == j, 'p E replay packet/order')
        same(pred['input'], cell['noised_latent'], 'p E exact DL input')
        same(pred['epsilon'], cell['prediction'], 'p E exact DL prediction')
        same(noise['noise'], cell['epsilon'], 'p E exact DL noise')
        need(noise['seed'] == cell['seed'] == er['seed'] and cell['repeat'] == j, 'p E exact DL seed/repeat')
        need(float(raw['module_outputs']['denoising_loss'][0, j, 0]) == cell['loss_l2_fp32'] == er['loss_l2_fp32'], 'p E exact saved DL score')
    return 5


def verify(directory):
    started = time.perf_counter()
    for name, value in HELPERS.items():
        need(sha(ROOT / 'u_patient_audit' / name) == value, 'frozen helper ' + name)
    from . import verify_cdi_u_cohort as uh
    from . import verify_cdi_e_cohort as eh
    from .analyze_verify_masked_slot_endpoints import expected_fingerprints
    p, e, rows = (read(directory / n) for n in ['protocol.json', 'execution.json', 'results.json'])
    c = p['contract']
    need(p['schema'] == 'cdi-cross-patient-intervention-execution/v1' and c['schema'] == 'cdi-cross-patient-intervention-contract/v1', 'protocol schemas')
    need(e['schema'] == 'cdi-cross-patient-intervention-result/v1' and e['complete'] is True
         and e['status'] == 'PASS_CDI_CROSS_PATIENT_INTERVENTION_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION', 'complete extraction')
    need(read(p['contract_path']) == c and sha(p['contract_path']) == p['contract_sha256'] == e['contract_sha256'], 'immutable extraction contract')
    for n in ['results', 'protocol']:
        need(sha(directory / (n + '.json')) == e[n + '_sha256'], 'extraction output binding')
    fixed = dict(current_step=2, focal_patient=P, second_patient=Q, branch_order=['q_control', 'p_control'],
                 expected_records=90, expected_unique_images=86, expected_patients=41, expected_reused_records=168,
                 expected_branch_records={'q_control':86, 'p_control':4}, expected_scenario_records={'U':82, 'E':8},
                 master_seed=260914, stream='primary', batch_size=1, prediction_type='epsilon', dtype='float32',
                 prompt='a frontal chest radiograph', code_policy='released_code_literal', perform_fitting=False,
                 perform_performance_evaluation=False, new_target_training=False)
    for k, v in fixed.items():
        need(c[k] == v, 'predeclared extraction ' + k)
    need(Path(c['output_directory']).resolve() == directory.resolve(), 'output path')
    need(sha(c['analysis_policy_path']) == c['analysis_policy_sha256'] == POLICY_SHA, 'fixed outcome-independent policy')
    need(sha(ROOT / 'u_patient_audit/run_cdi_cross_patient_intervention.py') == PRODUCER_SHA, 'frozen producer')
    need(sha(ROOT / 'u_patient_audit/cdi_adapter.py') == '70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce', 'literal adapter')
    for path, value in c['frozen_sha256'].items():
        need(sha(path) == value, 'frozen source/input ' + path)
    same(p['source_bindings'], c['source_bindings'], 'contract/execution source bindings')
    same(p['source_bindings'], read(OLDU / 'protocol.json')['source_bindings'], 'unchanged released source')
    need(c['feature_names'] == uh.NAMES == eh.NAMES == c['source_bindings']['feature_names'], 'literal26 order')
    for name, value in c['source_bindings']['files_sha256'].items():
        path = (Path(c['source_bindings']['source_root']) / name).resolve()
        need(sha(path) == value == c['frozen_sha256'][str(path)], 'upstream code hash')
    u_rows, e_rows, order = scope()
    need(c['patient_order'] == order, 'fixed U patient order')
    policy = read(c['analysis_policy_path'])
    need(policy['patient_order_U'] == order and policy['patient_order_E'] == [P, Q], 'fixed analysis order')
    states, totals = ledgers(c, e_rows)
    original, eps = references(c, u_rows, e_rows)
    need(p['reused_sources'] == c['reused_sources'] and p['measurement_plan'] == c['measurement_plan'], 'protocol reuse/plan exact')
    plans = [('q_control', r) for r in u_rows + e_rows] + [('p_control', r) for r in e_rows]
    need(len(rows) == len(c['measurement_plan']) == len(plans) == 90, 'exact new90')
    cachepath = RUN / 'cache/cache.pt'; cachehash = sha(cachepath); lockhash = sha(RUN / 'cohort/lock.json')
    need(c['cache_sha256'] == cachehash and c['cohort_lock_sha256'] == lockhash, 'cache/cohort source hashes')
    cache = torch.load(cachepath, map_location='cpu', weights_only=True)
    hiddenhash = hashlib.sha256(cache['hidden'][c['prompt']].float().contiguous().numpy().tobytes()).hexdigest()
    checks, rawhashes, replay = [], {}, 0
    for row, planrow, (branch, cr) in zip(rows, c['measurement_plan'], plans):
        iid, pid = cr['image_id'], cr['patient_id']; removed = Q if branch == 'q_control' else P
        scenario = 'U' if cr['record_role'] == 'U_observed' else 'E'
        need(row['image_id'] == iid and row['branch'] == row['model'] == branch and row['removed_patient_id'] == removed, 'actual branch/row identity')
        for k, v in planrow.items():
            need(row[k] == v, 'declared row ' + k)
        for k in ['patient_id', 'eval_role', 'record_role', 'assignment_group']:
            need(row[k] == cr[k], 'locked cohort row ' + k)
        imcount, pcount = states[branch]['exposures'].get(iid, 0), totals[branch][pid]
        need(row['scenario'] == scenario and row['member'] == row['effective_patient_member'] == int(pcount > 0), 'actual patient membership')
        need(row['patient_training_exposures'] == row['effective_patient_exposures'] == pcount, 'actual patient exposures')
        need(row['image_member'] == row['effective_image_member'] == int(imcount > 0)
             and row['actual_training_exposures'] == row['effective_image_exposures'] == imcount, 'actual image membership/exposures')
        need(imcount == (0 if scenario == 'U' or pid == removed else 4), 'role/removed E exposure')
        need(row['treatment_member'] == int(totals['treatment'][pid] > 0)
             and row['treatment_patient_training_exposures'] == totals['treatment'][pid]
             and row['treatment_image_member'] == int(states['treatment']['exposures'].get(iid, 0) > 0), 'original treatment membership')
        need(row['membership_label_scope'] == 'NIH effective loss contributions only' and row['pretrained_membership'] == 'unknown', 'limited membership scope')
        need(row['checkpoint_sha256'] == c['checkpoint_sha256'][branch] and row['cache_sha256'] == cachehash and row['cohort_lock_sha256'] == lockhash, 'actual row artifact binding')
        need(row['contract_sha256'] == p['contract_sha256'] and row['source_noise_and_input_exact'] is True
             and row['reused_record'] is row['reused_from_kernel'] is False, 'new measured row')
        path = inside(directory, row['raw_path'])
        need(row['raw_path'] == branch + '/' + Path(iid).stem + '.pt' and sha(path) == row['raw_sha256'], 'new actual packet path/hash')
        same(read(path.with_suffix('.json')), row, 'new aggregate/per-row exact')
        source, priorrow = original[iid]
        same(row['treatment_source'], source, 'original reference identity')
        raw = torch.load(path, map_location='cpu', weights_only=True)
        prior = torch.load(source['raw_path'], map_location='cpu', weights_only=True)
        for k in ['latent', 'alphas', 'noise_draws']:
            same(raw[k], prior[k], 'all original module inputs/randomness ' + k)
        need(raw['hidden_sha256'] == prior['hidden_sha256'] == hiddenhash and raw['schema'] == prior['schema'] == 'cdi-literal-26-raw/v1', 'generic conditioning/schema exact')
        # Only legacy enum guard/return changes, never a numeric or label input.
        helper = uh if scenario == 'U' else eh
        checked = helper.verify_image(dict(row, model='model_1'), raw, cache['latents'][iid], {}, OLDU if scenario == 'U' else OLDE)
        checked.update(model=branch, branch=branch, scenario=scenario, actual_member=row['member'], actual_image_member=row['image_member'])
        need(row['features'] == [float(x) for x in raw['features'].reshape(-1)] if 'features' in raw else True, 'raw feature-vector copy when present')
        need(row['features'][0] == float(raw['module_outputs']['denoising_loss'].mean()), 'literal FP32 five-draw mean')
        if branch == 'p_control':
            replay += p_dl_replay(raw, row, eps)
        checks.append(checked); rawhashes[str(path)] = row['raw_sha256']
    need(replay == 20, 'p control E20 exact replay')
    q = sum(r['NO_objective_evaluations'] for r in checks)
    forward, backward = sum(r['forward'] for r in rows), sum(r['backward'] for r in rows)
    need(e['forward'] == forward == 90 * 51 + q and e['backward'] == backward == 90 * 10 + q
         and e['NO_objective_evaluations'] == q and e['vae_forward'] == 0, 'actual F/B/NO cost')
    need(e['records'] == 90 and e['unique_images'] == 86 and e['unique_patients'] == 41 and e['reused_records'] == 168, 'executed scope')
    need([r['branch'] for r in e['branch_reports']] == ['q_control', 'p_control'], 'execution branch order')
    expected_states = {}
    for branch in ['q_control', 'p_control']:
        fingerprints = expected_fingerprints(c, {'treatment': states['treatment'], 'control': states[branch]})
        expected_states[branch] = fingerprints['control']
        report = next(r for r in e['branch_reports'] if r['branch'] == branch)
        same(read(directory / branch / 'execution.json'), report, 'aggregate branch execution')
        same(report['initial_state'], report['final_state'], 'unchanged full FP32 model')
        for k, value in expected_states[branch].items():
            need(report['initial_state'][k] == value, 'FP32 base-plus-actual-checkpoint full reconstruction ' + k)
        need(report['initial_state']['artifact_tensors_exactly_checked'] == 0, 'producer fingerprint field scope')
        for k in ['all_parameters_frozen', 'no_parameter_gradients', 'parameter_versions_unchanged', 'actual_checkpoint_adapter_exact', 'adapter_tensors_unchanged']:
            need(report[k] is True, 'model runtime guard ' + k)
        rr = [r for r in rows if r['branch'] == branch]
        need(report['records'] == len(rr) and report['forward'] == sum(r['forward'] for r in rr)
             and report['backward'] == sum(r['backward'] for r in rr) and report['checkpoint_sha256'] == c['checkpoint_sha256'][branch], 'branch cost/checkpoint')
    previous_state = read(OLDP / 'execution.json')['initial_state']
    for k, value in expected_states['p_control'].items():
        need(previous_state[k] == value, 'previous verified p full model identity ' + k)
    for k in ['source_noise_and_input_exact_all90', 'frozen_inputs_unchanged', 'reused_sources_unchanged']:
        need(e[k] is True, 'runtime source guard ' + k)
    need(e['new_target_training'] is e['perform_fitting'] is e['perform_performance_evaluation'] is False, 'no target update/fitting/evaluation')
    for path, value in c['frozen_sha256'].items():
        need(sha(path) == value, 'source unchanged after verification')
    need(not torch.cuda.is_initialized(), 'CPU only')
    return dict(status=STATUS, complete=True, current_step=2, seconds_cpu=time.perf_counter()-started,
        results_sha256=sha(directory/'results.json'), protocol_sha256=sha(directory/'protocol.json'),
        execution_sha256=sha(directory/'execution.json'), contract_sha256=p['contract_sha256'],
        records=90, unique_images=86, patients=41, reused_records=168, literal_features_reconstructed=2340,
        regenerated_noise_draws=720, p_control_E_DL_exact_cells=20, forward=forward, backward=backward,
        NO_objective_evaluations=q, expected_full_FP32_model_states=expected_states, raw_sha256=rawhashes,
        checked_images=checks, helpers_sha256=HELPERS, CUDA_initialized=False,
        limitations=['Saved tensor arithmetic and provenance; no independent neural inference or training rerun.',
            'Full FP32 base-plus-adapter states reconstructed independently; runtime guards remain producer observations.',
            'GM/NO recorded gradients checked where algebra permits; neural Jacobians not independently recomputed.',
            'Legacy helper model_1 is metadata-only; wrapper checks actual p/q checkpoint and effective labels.',
            'p E DL20 predictions exactly replay previous saved measurement; q predictions have no independent replay.',
            'Fixed intervention paths; not a population causal estimate, deployable counterfactual attack or privacy guarantee.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--verification-tag', default='verification_v1')
    parser.add_argument('--expected-code-sha256', required=True)
    args = parser.parse_args(); codehash = sha(__file__)
    need(codehash == args.expected_code_sha256, 'frozen verifier source')
    directory = args.run_dir.resolve(); dest = directory / args.verification_tag
    need(dest.parent == directory and not (directory / 'verification.json').exists(), 'preserve verification outputs')
    bindings = {str(directory / n): sha(directory / n) for n in ['protocol.json', 'execution.json', 'results.json']}
    bindings.update({str(ROOT/'u_patient_audit'/n): sha(ROOT/'u_patient_audit'/n) for n in HELPERS})
    bindings[str(Path(__file__).resolve())] = codehash
    dest.mkdir(exist_ok=False)
    new_json(dest / 'protocol.json', dict(schema='cross-patient-CDI-independent-verification/v1',
        source_code_sha256=codehash, input_sha256=bindings, CPU_only=True,
        verifier_frozen_before_own_execution=True, verifier_claimed_frozen_before_extraction=False))
    try:
        torch.set_num_threads(2)
        result = verify(directory)
        for path, value in bindings.items():
            need(sha(path) == value, 'own input unchanged')
        result.update(source_code_sha256=codehash, verification_protocol_sha256=sha(dest/'protocol.json'))
        new_json(dest/'verification.json', result); new_json(directory/'verification.json', result)
        print(json.dumps({k:result[k] for k in ['status','seconds_cpu','records','forward','backward','p_control_E_DL_exact_cells']}))
    except BaseException:
        new_json(dest/'failure.json', dict(status='FAILED_PRESERVED_CROSS_PATIENT_CDI_VERIFICATION', traceback=traceback.format_exc(), source_code_sha256=codehash))
        raise


if __name__ == '__main__':
    main()
