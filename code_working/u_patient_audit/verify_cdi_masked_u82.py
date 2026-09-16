"""Independent saved-packet checks for literal CDI on a masked control model.

No model inference or fitting. The frozen cohort verifier supplies numerical
arithmetic only; actual masked-model identity and exposure are checked here.
"""
import argparse
from collections import Counter, defaultdict
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
OUT = BASE / 'cdi_masked_u82_v1'
OLD = RUN / 'baseline_screen_20260914/cdi_u_cohort_v1'
ENDPOINT = BASE / 'masked_slot_endpoints_v2'
TRAIN = BASE / 'masked_patient_training_v1'
PID = '14393'
MASKED = {'00014393_002.png', '00014393_006.png'}
HELPER_SHA = 'da899f1efbc46f8571baebd2113d204038921b5b5374caba7ad22c7d58b64869'
KERNEL_SHA = '57b7719c54fee4c32c2d92e29535ce44aceb5b784669b4be2fff43aa6dd7d342'
PRODUCER_SHA = '5cc78b30a64c8fe296692ade7c25f792f042f481d4252c116aec8f52592a5515'
ADAPTER_SHA = '70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce'
STATUS = 'PASS_MASKED_CDI_U82_SAVED_ARITHMETIC_PROVENANCE_AND_DL_REPLAY'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(value, label):
    if not value:
        raise AssertionError(label)


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def csvrows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def same(a, b, label):
    if isinstance(a, torch.Tensor):
        need(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
             and torch.equal(a, b), label)
    elif isinstance(a, dict):
        need(isinstance(b, dict) and a.keys() == b.keys(), label + ': keys')
        for k in a:
            same(a[k], b[k], label + '.' + str(k))
    elif isinstance(a, (list, tuple)):
        need(type(a) is type(b) and len(a) == len(b), label + ': length/type')
        for j, (x, y) in enumerate(zip(a, b)):
            same(x, y, label + '.' + str(j))
    else:
        need(type(a) is type(b) and a == b, label)


def inside(directory, relative):
    p = (directory / relative).resolve()
    need(p.is_relative_to(directory.resolve()), 'packet remains inside source directory')
    return p


def cohort_and_ledgers():
    cohort = csvrows(RUN / 'cohort/evaluation_images.csv')
    selection = sorted({r['patient_id'] for r in cohort if r['eval_role'] == 'selection'}, key=int)
    need(len(selection) == 40 and PID not in selection, 'fixed selection40 and fit patient')
    order = [PID] + selection
    selected = []
    for pid in order:
        rr = [r for r in cohort if r['patient_id'] == pid and r['record_role'] == 'U_observed']
        need(len(rr) == 2 and all(r['eval_role'] == ('fit' if pid == PID else 'selection') for r in rr),
             'fixed U2/role for ' + pid)
        selected.extend(sorted(rr, key=lambda r: r['image_id']))
    need(Counter(r['assignment_group'] for r in selected if r['eval_role'] == 'selection')
         == {'A': 40, 'B': 40}, 'selection20A20B U2')
    lock = read(RUN / 'cohort/lock.json')
    for name, value in lock['files'].items():
        need(sha(RUN / 'cohort' / name) == value, 'locked cohort ' + name)
    tv = read(TRAIN / 'verification.json')
    need(tv['complete'] is True and tv['status'] == 'PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC',
         'completed independent masked training verification')
    for name in ('execution', 'protocol'):
        need(tv[name + '_sha256'] == sha(TRAIN / (name + '.json')), 'verified training ' + name)
    oldcp = RUN / 'training_coverage_v2/model_1/step_1000.pt'
    newcp = TRAIN / 'control/step_1000.pt'
    control = torch.load(newcp, map_location='cpu', weights_only=True)
    original = torch.load(oldcp, map_location='cpu', weights_only=True)
    need(sha(newcp) == tv['checkpoint_sha256']['control']['1000'], 'actual verified control checkpoint')
    need(sha(TRAIN / 'treatment/step_1000.pt') == tv['checkpoint_sha256']['treatment']['1000'],
         'verified treatment checkpoint')
    same(original, torch.load(TRAIN / 'treatment/step_1000.pt', map_location='cpu', weights_only=True),
         'complete original/treatment checkpoint exact including numerical contract')
    need(control['step'] == original['step'] == 1000, 'checkpoint step1000')
    same(control['contract'], original['contract'], 'original numerical checkpoint contract')
    manifest = csvrows(RUN / 'cohort/model_1_train.csv')
    need(len(manifest) == 912 and set(original['exposures']) == {r['image_id'] for r in manifest}
         and sum(original['exposures'].values()) == 4000, 'original912/4000 exposures')
    expected = dict(original['exposures'])
    need({r['image_id'] for r in manifest if r['patient_id'] == PID} == MASKED, 'only selected E2 belongs to patient')
    for iid in MASKED:
        need(expected.pop(iid) == 4, 'four realized slots per removed E image')
    same(control['exposures'], expected, 'control is exactly original ledger minus eight selected contributions')
    need(sum(expected.values()) == 3992 and len(expected) == 910, 'control910/3992')
    pexp = defaultdict(int)
    for r in manifest:
        pexp[r['patient_id']] += expected.get(r['image_id'], 0)
    return selected, order, control, original, dict(pexp), sha(newcp), sha(oldcp)


def reference_sources(ids):
    oldv = read(OLD / 'verification.json')
    need(oldv['status'] == 'PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY', 'verified original full U cohort')
    need(oldv['results_sha256'] == sha(OLD / 'results.json'), 'original results binding')
    old = {r['image_id']: r for r in read(OLD / 'results.json') if r['model'] == 'model_1' and r['image_id'] in ids}
    need(set(old) == set(ids), 'exact original M1 U82 source rows')
    ev = read(ENDPOINT / 'verification.json')
    need(ev['complete'] is True and ev['status'] == 'PASS_MASKED_ENDPOINT_SAVED_ARITHMETIC_SOURCE_REUSE_AND_TRAINING_GATES',
         'verified endpoint arithmetic and training gates')
    for name in ('results', 'protocol', 'execution'):
        need(ev[name + '_sha256'] == sha(ENDPOINT / (name + '.json')), 'verified endpoint ' + name)
    endpoints = {(r['image_id'], r['repeat']): r for r in read(ENDPOINT / 'results.json')
                 if r['branch'] == 'control' and r['family'] == 'cdi_dl_t100' and r['image_id'] in ids}
    need(set(endpoints) == {(i, j) for i in ids for j in range(5)}, 'exact410 control DL endpoint cells')
    return old, endpoints, oldv, ev


def check_numeric_and_replay(row, raw, original_row, cache, endpoints, helper):
    iid = row['image_id']
    # The helper's model label is only a legacy enum/return field, never a seed,
    # tensor, score or checkpoint input. Actual control identity was checked by
    # this wrapper. Preserve the true label in the final report.
    numeric_row = dict(row, model='model_1')
    checked = helper.verify_image(numeric_row, raw, cache['latents'][iid], {}, OLD)
    checked['model'] = row['model']
    checked['arithmetic_helper_legacy_enum'] = 'model_1 (metadata-only compatibility; actual model masked_control)'
    original_path = inside(OLD, original_row['raw_path'])
    need(sha(original_path) == original_row['raw_sha256'], 'original raw source hash ' + iid)
    prior = torch.load(original_path, map_location='cpu', weights_only=True)
    for key in ('latent', 'alphas', 'noise_draws'):
        same(raw[key], prior[key], 'unchanged source ' + iid + '.' + key)
    hidden = cache['hidden']['a frontal chest radiograph'].float().contiguous().numpy()
    need(raw['hidden_sha256'] == prior['hidden_sha256'] == hashlib.sha256(hidden.tobytes()).hexdigest(),
         'same generic cached conditioning')
    need(raw['schema'] == prior['schema'] == 'cdi-literal-26-raw/v1', 'literal raw packet schema')
    ep_path = inside(ENDPOINT, endpoints[iid, 0]['raw_path'])
    need(sha(ep_path) == endpoints[iid, 0]['raw_sha256'], 'verified endpoint raw hash ' + iid)
    ep = torch.load(ep_path, map_location='cpu', weights_only=True)
    same(raw['latent'], ep['latent'], 'endpoint latent ' + iid)
    for j in range(5):
        er = endpoints[iid, j]
        need(er['raw_path'] == endpoints[iid, 0]['raw_path'] and er['cell_index'] == j,
             'same endpoint packet and order')
        q = ep['cells'][j]
        pred = raw['predictions']['denoising_loss'][j]
        noise = raw['noise_draws']['denoising_loss'][j]
        same(pred['input'], q['noised_latent'], 'exact DL input ' + iid)
        same(pred['epsilon'], q['prediction'], 'exact DL prediction ' + iid)
        same(noise['noise'], q['epsilon'], 'exact DL noise ' + iid)
        need(noise['seed'] == q['seed'] == er['seed'] and q['repeat'] == j,
             'identical original module seed')
        value = float(raw['module_outputs']['denoising_loss'][0, j, 0])
        need(value == q['loss_l2_fp32'] == er['loss_l2_fp32'], 'exact original GPU L2 replay ' + iid)
    need(row['features'][0] == float(raw['module_outputs']['denoising_loss'].mean()),
         'literal source FP32 five-draw DL mean')
    checked['saved_control_DL_endpoint_cells_exact'] = 5
    return checked


def inspect_records(directory, results, selected, control, pexp, checkpoint_hash, original_rows, endpoints):
    need(sha(ROOT / 'u_patient_audit/verify_cdi_u_cohort.py') == HELPER_SHA,
         'frozen numerical helper source')
    need(sha(ROOT / 'u_patient_audit/verify_cdi_kernel.py') == KERNEL_SHA,
         'frozen original arithmetic-copy source')
    from . import verify_cdi_u_cohort as helper
    need(len(results) == 82 and [r['image_id'] for r in results] == [r['image_id'] for r in selected],
         'exact U82 record order and count')
    cache_path = RUN / 'cache/cache.pt'
    cache_hash, lock_hash = sha(cache_path), sha(RUN / 'cohort/lock.json')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    checked, raw_hashes = [], {}
    for row, cr in zip(results, selected):
        iid, pid = row['image_id'], cr['patient_id']
        need(row['model'] == 'masked_control' and row['branch'] == 'control', 'actual masked control identity')
        for key in ('patient_id', 'eval_role', 'record_role', 'assignment_group'):
            need(row[key] == cr[key], 'cohort metadata ' + iid + '.' + key)
        need(row['scenario'] == 'U' and row['record_role'] == 'U_observed', 'only unseen patient images')
        need(row['member'] == int(pexp.get(pid, 0) > 0)
             and row['patient_training_exposures'] == pexp.get(pid, 0), 'actual patient contribution')
        need(row['effective_patient_member'] == row['member']
             and row['effective_patient_exposures'] == row['patient_training_exposures']
             and row['effective_image_member'] == row['effective_image_exposures'] == 0,
             'explicit effective exposure fields')
        need(row['treatment_member'] == original_rows[iid]['member']
             and row['treatment_patient_training_exposures'] == original_rows[iid]['patient_training_exposures'],
             'original treatment participation')
        need(row['pretrained_membership'] == 'unknown'
             and row['membership_label_scope'] == 'NIH effective loss contributions only', 'no inferred base membership')
        need(row['image_member'] == row['actual_training_exposures'] == 0
             and iid not in control['exposures'], 'all U image exposures zero')
        need(row['checkpoint_sha256'] == checkpoint_hash and row['cache_sha256'] == cache_hash
             and row['cohort_lock_sha256'] == lock_hash, 'actual checkpoint/cache/cohort binding')
        need(row['feature_names'] == original_rows[iid]['feature_names'], 'unchanged source feature names/order')
        path = inside(directory, row['raw_path'])
        need(row['raw_path'] == 'control/' + Path(iid).stem + '.pt', 'exact immutable control packet path')
        need(read(path.with_suffix('.json')) == row, 'per-image JSON/aggregate exact')
        need(sha(path) == row['raw_sha256'], 'saved control raw hash ' + iid)
        raw_hashes[str(path.resolve())] = row['raw_sha256']
        raw = torch.load(path, map_location='cpu', weights_only=True)
        checked.append(check_numeric_and_replay(row, raw, original_rows[iid], cache, endpoints, helper))
    need(Counter(r['member'] for r in results) == {0: 42, 1: 40}, '20 contributing and21 noncontributing patients U2')
    return checked, raw_hashes


def verify(directory):
    started = time.perf_counter()
    p, e, rows = (read(directory / name) for name in ('protocol.json', 'execution.json', 'results.json'))
    c = p['contract']
    need(p['schema'] == 'cdi-masked-u82-execution/v1' and c['schema'] == 'cdi-masked-u82-contract/v1',
         'declared new masked extraction schemas')
    need(e['schema'] == 'cdi-masked-u82-result/v1' and e['complete'] is True
         and e['status'] == 'PASS_MASKED_CDI_U82_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION', 'completed extraction')
    need(read(Path(p['contract_path'])) == c
         and sha(p['contract_path']) == p['contract_sha256'] == e['contract_sha256'], 'immutable external contract')
    for name in ('results', 'protocol'):
        need(sha(directory / (name + '.json')) == e[name + '_sha256'], 'completed output hash ' + name)
    expected = dict(current_step=2, model='masked_control', branch='control', scenario='U',
        expected_patients=41, expected_records=82, expected_unique_images=82,
        master_seed=260914, stream='primary', batch_size=1, prediction_type='epsilon', dtype='float32',
        prompt='a frontal chest radiograph', code_policy='released_code_literal',
        no_control_module_reuse=True, original_treatment_raw_reused_without_changes=True,
        perform_fitting=False, perform_performance_evaluation=False, new_target_training=False,
        no_new_patients=True, no_E_expansion=True)
    for k, value in expected.items():
        need(c[k] == value, 'fixed extraction policy ' + k)
    need(Path(c['output_directory']).resolve() == directory.resolve()
         and Path(c['treatment_directory']).resolve() == OLD.resolve(), 'fixed source/output directories')
    for path, value in c['frozen_sha256'].items():
        need(sha(path) == value, 'frozen extraction source/input ' + path)
    for name, value in (('run_cdi_masked_u82.py', PRODUCER_SHA), ('cdi_adapter.py', ADAPTER_SHA)):
        path = ROOT / 'u_patient_audit' / name
        need(sha(path) == value == c['frozen_sha256'][str(path.resolve())], 'pinned literal producer/adapter')
    selected, order, control, original, pexp, ch, oh = cohort_and_ledgers()
    need(c['patient_order'] == order and c['image_order'] == [r['image_id'] for r in selected], 'fixed U82 scope/order')
    need(c['checkpoint_sha256'] == {'control': ch, 'treatment': oh}, 'actual checkpoint hashes')
    need(Path(c['checkpoint_paths']['control']).resolve() == (TRAIN / 'control/step_1000.pt').resolve()
         and Path(c['control_checkpoint']).resolve() == (TRAIN / 'control/step_1000.pt').resolve()
         and Path(c['checkpoint_paths']['treatment']).resolve() == (RUN / 'training_coverage_v2/model_1/step_1000.pt').resolve(),
         'actual checkpoint paths')
    old, endpoints, oldv, ev = reference_sources(c['image_order'])
    oldp = read(OLD / 'protocol.json')
    need(p['source_bindings'] == c['source_bindings'] == oldp['source_bindings'], 'same released-source implementation')
    need(c['feature_names'] == c['source_bindings']['feature_names'] == oldp['source_bindings']['feature_names'],
         'same26 feature order')
    for name, value in c['source_bindings']['files_sha256'].items():
        path = (Path(c['source_bindings']['source_root']) / name).resolve()
        need(sha(path) == value == c['frozen_sha256'][str(path)], 'pinned upstream source ' + name)
    need(p['measurement_plan'] == c['measurement_plan'] and len(c['measurement_plan']) == 82,
         'predeclared row metadata')
    sources = {s['image_id']: s for s in c['treatment_sources']}
    need(set(sources) == set(c['image_order']) and len(c['treatment_sources']) == 82, 'exact treatment source list')
    for r, plan in zip(rows, c['measurement_plan']):
        for k, value in plan.items():
            need(r[k] == value, 'predeclared row metadata ' + k)
        need(r['contract_sha256'] == p['contract_sha256'] and r['source_noise_and_input_exact'] is True
             and r['reused_record'] is r['reused_from_kernel'] is False, 'new control kernel and source binding')
        src = sources[r['image_id']]
        need(r['treatment_source'] == src, 'individual treatment provenance')
        prior = old[r['image_id']]
        need(Path(src['raw_path']).resolve() == inside(OLD, prior['raw_path'])
             and src['raw_sha256'] == prior['raw_sha256'] == sha(src['raw_path']), 'actual original raw source')
        need(Path(src['row_path']).resolve() == Path(src['raw_path']).with_suffix('.json').resolve()
             and sha(src['row_path']) == src['row_sha256'] and read(src['row_path']) == prior,
             'actual original per-image row')
    need(c['cache_sha256'] == sha(RUN / 'cache/cache.pt') and c['cohort_lock_sha256'] == sha(RUN / 'cohort/lock.json'),
         'fixed actual cache/cohort')
    checked, raw_hashes = inspect_records(directory, rows, selected, control, pexp, ch, old, endpoints)
    q = sum(r['NO_objective_evaluations'] for r in checked)
    forward, backward = sum(r['forward'] for r in rows), sum(r['backward'] for r in rows)
    need(e['forward'] == forward == 82 * 51 + q and e['backward'] == backward == 82 * 10 + q
         and e['NO_objective_evaluations'] == q and e['vae_forward'] == 0, 'reconciled literal F/B counters')
    need(e['records'] == e['unique_images'] == 82 and e['unique_patients'] == 41
         and e['model'] == 'masked_control' and e['branch'] == 'control', 'completed scope')
    for key in ('all_parameters_frozen', 'no_parameter_gradients', 'parameter_versions_unchanged',
                'actual_checkpoint_adapter_exact', 'adapter_tensors_unchanged', 'source_noise_and_input_exact_all82',
                'frozen_inputs_unchanged'):
        need(e[key] is True, 'runtime model/source integrity ' + key)
    need(e['new_target_training'] is e['perform_fitting'] is e['perform_performance_evaluation'] is False,
         'no target training or performance evaluation in extraction')
    endpoint_control = next(r for r in read(ENDPOINT / 'execution.json')['model_reports'] if r['branch'] == 'control')
    same(e['initial_state'], e['final_state'], 'unchanged full masked model state')
    same(e['initial_state'], endpoint_control['initial_state'], 'same previously verified masked full model state')
    need(e['checkpoint_sha256'] == ch, 'execution actual checkpoint')
    for path, value in c['frozen_sha256'].items():
        need(sha(path) == value, 'unchanged bound extraction input ' + path)
    summary = dict(status=STATUS, complete=True, seconds_cpu=time.perf_counter() - started,
        results_sha256=sha(directory / 'results.json'), protocol_sha256=sha(directory / 'protocol.json'),
        execution_sha256=sha(directory / 'execution.json'), contract_sha256=p['contract_sha256'],
        records=82, patients=41, scenario='U', model='masked_control', actual_checkpoint_sha256=ch,
        effective_exposures=3992, selected_patient_exposures=0, all_U_image_exposures=0,
        literal_features_independently_reconstructed=82 * 26, regenerated_noise_draws=82 * 8,
        saved_control_DL_endpoint_exact_images=82, saved_control_DL_endpoint_exact_cells=410,
        actual_control_full_state_sha256=e['initial_state']['sha256'], forward=forward, backward=backward,
        NO_objective_evaluations=q, arithmetic_helper_sha256=HELPER_SHA, kernel_copy_source_sha256=KERNEL_SHA,
        original_treatment_results_sha256=oldv['results_sha256'], endpoint_results_sha256=ev['results_sha256'],
        raw_sha256=raw_hashes, checked_images=checked, CUDA_initialized=torch.cuda.is_initialized(),
        limitations=['Saved arithmetic, source, exposure and recorded model-state verification; no independent neural CPU/GPU rerun.',
            'GM/NO recorded input gradients and output-loss gradients are checked; full neural Jacobians are not independently recomputed.',
            'Treatment/control latent, alphas, generic hidden and original random draws match; model-dependent GM masks/NO trajectories need not match.',
            'Legacy numeric helper receives model_1 only for its enum guard/return field; actual masked checkpoint and membership are independently checked here.',
            'One patient-contribution intervention; no new-method efficacy, population MIA, AUC, CI or privacy guarantee claim.',
            'This verifier code is bound in its own execution protocol after extraction policy freeze; not claimed frozen before GPU extraction.'])
    need(summary['CUDA_initialized'] is False, 'CPU-only verifier')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--verification-tag', default='verification_v1')
    parser.add_argument('--expected-code-sha256', required=True)
    args = parser.parse_args()
    code_hash = sha(__file__)
    need(code_hash == args.expected_code_sha256, 'frozen independent verifier code')
    directory = args.run_dir.resolve()
    need((directory / 'execution.json').exists(), 'completed extraction required')
    need(not (directory / 'verification.json').exists(), 'preserve existing root verification')
    dest = directory / args.verification_tag
    need(dest.parent == directory, 'simple verification subdirectory')
    dest.mkdir(exist_ok=False)
    bindings = {str((directory / n).resolve()): sha(directory / n)
                for n in ('protocol.json', 'execution.json', 'results.json')}
    bindings.update({str(p.resolve()): sha(p) for p in
        [Path(__file__), ROOT / 'u_patient_audit/verify_cdi_u_cohort.py',
         ROOT / 'u_patient_audit/verify_cdi_kernel.py', ENDPOINT / 'verification.json',
         ENDPOINT / 'results.json', TRAIN / 'verification.json', OLD / 'verification.json']})
    write(dest / 'protocol.json', dict(schema='masked-cdi-u82-independent-verification/v1',
        source_code_sha256=code_hash, input_sha256=bindings, numerical_helper_sha256=HELPER_SHA,
        code_bound_before_verification=True, verifier_code_claimed_frozen_before_extraction=False))
    try:
        torch.set_num_threads(2)
        summary = verify(directory)
        for path, value in bindings.items():
            need(sha(path) == value, 'verification source/input unchanged ' + path)
        summary.update(source_code_sha256=code_hash, verification_protocol_sha256=sha(dest / 'protocol.json'))
        write(dest / 'verification.json', summary)
        write(directory / 'verification.json', summary)
        print(json.dumps({k: summary[k] for k in ('status', 'seconds_cpu', 'records', 'forward', 'backward',
            'saved_control_DL_endpoint_exact_cells', 'CUDA_initialized')}))
    except BaseException:
        write(dest / 'failure.json', dict(status='FAILED_PRESERVED_MASKED_CDI_U82_VERIFICATION',
            traceback=traceback.format_exc(), source_code_sha256=code_hash))
        raise


if __name__ == '__main__':
    main()
