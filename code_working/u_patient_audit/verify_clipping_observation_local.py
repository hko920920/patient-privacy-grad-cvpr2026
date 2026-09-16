"""CPU-only saved-tensor audit; does not import either measurement producer."""
import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / '_reports/cvpr_u_pilot_v1_001/protection_observation_20260915/clipping_observation_local_v1'
PROMPT = 'a frontal chest radiograph'
SALT = 'cvpr-clipping-observation-local-v1-20260915'
EPS32 = np.finfo(np.float32).eps


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(ok, why):
    if not bool(ok):
        raise AssertionError(why)


def close(actual, expected, why, atol=2e-12, rtol=2e-9):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), why + ': shape/finite')
    require(np.all(np.abs(a-b) <= atol + rtol*np.abs(b)), why + ': mismatch')


def arr(tensor):
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'CPU tensor')
    require(torch.isfinite(tensor).all(), 'finite tensor')
    return tensor.detach().numpy()


def length(x):
    a = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.dot(a, a)))


def inner(x, y):
    return float(np.dot(np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()))


def cos(x, y):
    return inner(x, y) / max(length(x)*length(y), 1e-300)


def clip32(x, c):
    n = length(x)
    a = min(1.0, c/n) if n else 1.0
    return np.multiply(x, np.float32(a), dtype=np.float32), a


def mse(pred, target):
    r = np.asarray(pred, dtype=np.float64)-np.asarray(target, dtype=np.float64)
    return float(np.mean(r*r, dtype=np.float64))


def score_difference(pa, pb, target):
    # Independent difference-of-squares and direct MSE paths.
    a, b, e = [np.asarray(x, dtype=np.float64) for x in (pa, pb, target)]
    stable = float(np.mean((b-a)*(b+a-2*e), dtype=np.float64))
    direct = mse(b, e)-mse(a, e)
    close(stable, direct, 'MSE difference identity', atol=4e-14, rtol=1e-9)
    return stable


def equal_tree(a, b, label):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), label)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), label)
        for k in a:
            equal_tree(a[k], b[k], label + '/' + str(k))
    elif isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), label)
        for i, (x, y) in enumerate(zip(a, b)):
            equal_tree(x, y, label + '/' + str(i))
    else:
        require(a == b, label)


def cpu_adam(state, gradient):
    theta = state['theta']
    params, pos = [], 0
    for shape in state['parameter_shapes']:
        n = math.prod(shape)
        p = torch.nn.Parameter(theta[pos:pos+n].reshape(shape).clone())
        p.grad = torch.from_numpy(np.asarray(gradient[pos:pos+n], dtype=np.float32).copy()).reshape(shape)
        params.append(p)
        pos += n
    require(pos == theta.numel(), 'parameter slicing')
    opt = torch.optim.AdamW(params, lr=1e-4, foreach=False, fused=False)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    original_steps = [float(v['step']) for v in opt.state.values()]
    opt.step()
    require([float(v['step']) for v in opt.state.values()] == [s+1 for s in original_steps], 'CPU Adam step advance')
    return torch.cat([p.detach().reshape(-1) for p in params])


def compare_adam(expected, observed, initial):
    # Cross-device float32 implementation agreement, not an empirical efficacy gate.
    e, o, t = [arr(x).astype(np.float64) for x in (expected, observed, initial)]
    err = np.abs(e-o)
    envelope = 32*EPS32*(np.abs(t)+np.abs(e-t)+np.abs(o-t)+np.finfo(np.float32).tiny)
    require(np.all(err <= envelope), 'CPU/CUDA Adam outside fixed float32 arithmetic envelope')
    return dict(exact=bool(np.array_equal(e, o)), max_absolute=float(err.max()),
                l2_absolute=length(e-o), relative_update_l2=length(e-o)/max(length(e-t), 1e-300),
                max_envelope_fraction=float(np.max(err/envelope)),
                envelope='32*eps32*(abs(theta)+abs(cpu_update)+abs(saved_update)+tiny32)')


def state_check(path, source, step):
    state = torch.load(path, map_location='cpu', weights_only=True)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    names = state['parameter_names']
    require(names == list(saved['adapter']), 'named parameter order matches source adapter order')
    require(state['parameter_shapes'] == [list(saved['adapter'][n].shape) for n in names], 'parameter shapes')
    expected_theta = torch.cat([saved['adapter'][n].reshape(-1) for n in names])
    require(expected_theta.dtype == torch.float32 and torch.equal(state['theta'], expected_theta), 'source theta exact')
    require(expected_theta.numel() == 1659904 and len(names) == 256, 'LoRA dimensions')
    equal_tree(state['optimizer'], saved['optimizer'], 'source optimizer exact')
    require(all(float(v['step']) == step for v in state['optimizer']['state'].values()), 'saved step')
    group = state['optimizer']['param_groups']
    require(len(group) == 1 and group[0]['foreach'] is False and group[0]['fused'] is False, 'Adam mode')
    require(group[0]['amsgrad'] is False and group[0]['maximize'] is False, 'Adam flags')
    for v in state['optimizer']['state'].values():
        require(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all(), 'finite moments')
    return state


def protocol_check(run, p, report):
    activation = p['schema'] == 'clipping-activation-control/v1'
    require(p['schema'] in ('clipping-observation-local/v1', 'clipping-activation-control/v1'), 'schema')
    require(p['clip_norm'] == (.01 if activation else .28448700606156724), 'prescribed C')
    require(p['timestep'] == 500 and p['checkpoint_steps'] == [250, 1000], 'timestep/states')
    require(p['audit_and_update_prompt'] == PROMPT and p['seed_salt'] == SALT, 'conditioning/seed')
    require(p['gradient_noise_sigma'] == 0 and p['outer_denominator'] == 1 and p['other_patients_in_local_batch'] == 0, 'local bridge scope')
    require(p['historical_AMP_replay'] is False, 'not AMP replay')
    counts = (114, 0, 0, 32) if activation else (166, 52, 24, 36)
    require((p['expected_forward'], p['expected_backward'], p['extra_VAE_images'], p['expected_parameter_optimizer_steps']) == counts, 'protocol counts')
    require((report['forward'], report['backward'], report['vae_images']) == counts[:3], 'report counts')
    require(report['patients'] == 8 and report['state_conditions'] == 16 and report['source_checkpoints_unchanged'], 'completion scope')
    require(report['status'] == ('COMPLETED_ACTIVATION_CONTROL_PENDING_INDEPENDENT_VERIFICATION' if activation else 'COMPLETED_LOCAL_MEASUREMENTS_PENDING_INDEPENDENT_VERIFICATION'), 'producer status')
    for key, filename in [('results_sha256', 'results.json'), ('protocol_sha256', 'protocol.json'), ('inputs_sha256', 'inputs.pt')]:
        require(report[key] == sha(run/filename), 'report hash/' + filename)
    for rel, expected in p['code_sha256'].items():
        require(sha(ROOT/rel) == expected, 'source code hash/' + rel)
    for path, expected in p['input_sha256'].items():
        require(sha(path) == expected, 'input binding/' + path)
    for binding in p['checkpoint_bindings']:
        require(sha(binding['path']) == binding['sha256'], 'checkpoint binding')
    return activation


def check_inputs(run, p, fixture):
    inputs = torch.load(run/'inputs.pt', map_location='cpu', weights_only=True)
    ids = [r['image_id'] for r in fixture['records']]
    require(len(set(ids)) == 24 and set(inputs['latents']) == set(ids) == set(inputs['queries']), 'image input IDs')
    cache_dir = ROOT/'_reports/cvpr_u_pilot_v1_001/cache'
    cache_report = read(cache_dir/'summary.json')
    require(sha(cache_dir/'cache.pt') == cache_report['cache_sha256'], 'cache self-binding')
    cache = torch.load(cache_dir/'cache.pt', map_location='cpu', weights_only=True)
    require(torch.equal(inputs['hidden'], cache['hidden'][PROMPT].float()), 'generic hidden exact cache')
    del cache
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(repo_id=p['model_id'], revision=p['model_revision'], local_files_only=True))
    config = read(snap/'scheduler/scheduler_config.json')
    require(config['prediction_type'] == 'epsilon' and config['num_train_timesteps'] == 1000, 'actual epsilon scheduler')
    require(config['beta_schedule'] == 'scaled_linear' and config.get('trained_betas') is None and not config.get('rescale_betas_zero_snr', False), 'pinned scheduler arithmetic')
    betas = torch.linspace(math.sqrt(config['beta_start']), math.sqrt(config['beta_end']), 1000, dtype=torch.float32).square()
    alpha = torch.cumprod(1-betas, dim=0)[500]
    for image in ids:
        latent = inputs['latents'][image]
        require(latent.dtype == torch.float16 and tuple(latent.shape) == (1, 4, 32, 32), 'latent contract')
        arr(latent)
        seed = int(hashlib.sha256((SALT+'|'+image).encode()).hexdigest()[:15], 16)
        noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
        noisy = torch.sqrt(alpha)*latent.float()+torch.sqrt(1-alpha)*noise
        query = inputs['queries'][image]
        require(torch.equal(query['target'], noise), 'noise seed exact/' + image)
        require(torch.equal(query['noisy'], noisy), 'independent noising exact/' + image)
    return inputs, {str(cache_dir/'cache.pt'): sha(cache_dir/'cache.pt'), str(snap/'scheduler/scheduler_config.json'): sha(snap/'scheduler/scheduler_config.json')}


def verify_packet(row, raw, state, inputs, c, first, activation):
    require(raw['patient_id'] == row['patient_id'] and raw['step'] == row['step'], 'packet identity')
    records = raw['records']
    require([r['role'] for r in records] == ['E_local_1', 'E_local_2', 'U_local'], 'roles')
    require(all(r['patient_id'] == row['patient_id'] for r in records), 'patient image identity')
    gs = arr(raw['gradients'])
    require(gs.dtype == np.float32 and gs.shape == (3, 1659904), 'gradient shape/dtype')
    g1, g2, gu = gs
    gm = np.multiply(np.add(g1, g2, dtype=np.float32), np.float32(.5), dtype=np.float32)
    x1, a1 = clip32(g1, c); x2, a2 = clip32(g2, c)
    va = np.multiply(np.add(x1, x2, dtype=np.float32), np.float32(.5), dtype=np.float32)
    vb, b = clip32(gm, c)
    abar = (a1+a2)/2
    ct = np.float32(.5)*(np.float32(a1-abar)*(g1-gm)+np.float32(a2-abar)*(g2-gm))
    cp = ct-np.float32(inner(ct, gm)/max(inner(gm, gm), 1e-300))*gm
    algebra = va-(np.float32(abar)*gm+ct)
    theta = state['theta']
    da, db = raw['delta_A'], raw['delta_B']
    require(da.dtype == db.dtype == torch.float32 and da.shape == db.shape == theta.shape, 'delta dtype/shape')
    ta, tb = theta+da, theta+db
    # Check reconstructability because producer saved displacements instead of endpoints.
    require(torch.equal(ta-theta, da) and torch.equal(tb-theta, db), 'saved endpoint reconstruction')
    dab = arr(ta-tb)
    cpu_a, cpu_b = cpu_adam(state, va), cpu_adam(state, vb)
    adam = {'A': compare_adam(cpu_a, ta, theta), 'B': compare_adam(cpu_b, tb, theta)}
    computed = dict(gradient_norms=[length(g) for g in gs], mean_gradient_norm=length(gm),
        clip_factors=[a1, a2], mean_clip_factor=b, gradient_cosine_E1_E2=cos(g1, g2),
        gradient_cosine_Emean_U=cos(gm, gu), clipped_A_B_cosine=cos(va, vb),
        clipped_A_B_norm_difference=length(va-vb), c_perpendicular_norm=length(cp),
        c_algebra_l2_error=length(algebra), delta_A_norm=length(arr(da)), delta_B_norm=length(arr(db)),
        delta_A_B_norm=length(dab), delta_A_B_cosine=cos(arr(da), arr(db)),
        raw_SGD_gradient_contrast=[inner(g, va-vb) for g in gs])
    for key, value in computed.items():
        close(row[key], value, key, atol=2e-10, rtol=3e-6)
    require(row['both_gradients_unclipped'] == (a1 == a2 == 1.), 'unclipped metadata')
    require(length(va) <= c*(1+8*EPS32) and length(vb) <= c*(1+8*EPS32), 'C bounded vectors')
    losses = {key: [] for key in ['base', 'A', 'B']}
    contrasts = []
    for i, record in enumerate(records):
        target = arr(inputs['queries'][record['image_id']]['target'])
        predictions = {key: arr(raw['prediction_'+key][i]) for key in ['base', 'A', 'B']}
        require(all(v.shape == target.shape and v.dtype == np.float32 for v in predictions.values()), 'prediction contract')
        for key, pred in predictions.items():
            losses[key].append(mse(pred, target))
        contrasts.append(score_difference(predictions['A'], predictions['B'], target))
    linear = [-inner(g, dab) for g in gs]
    for key, value in [('base_losses', losses['base']), ('loss_A', losses['A']), ('loss_B', losses['B']),
                       ('score_A_minus_B', contrasts), ('linear_prediction_A_minus_B', linear),
                       ('Emean_score_A_minus_B', sum(contrasts[:2])/2), ('U_score_A_minus_B', contrasts[2]),
                       ('Emean_linear_prediction', sum(linear[:2])/2), ('U_linear_prediction', linear[2])]:
        close(row[key], value, key)
    require(row['E_U_raw_sign_opposition'] == ((sum(contrasts[:2])/2)*contrasts[2] < 0), 'raw sign metadata')
    require(row['E_U_linear_sign_opposition'] == ((sum(linear[:2])/2)*linear[2] < 0), 'linear sign metadata')
    controls = raw['controls']
    control_audit = {}
    if first:
        require(controls['baseline_replay_exact'] == [True]*3, 'producer replay assertion')
        if 'baseline_replay_predictions' in controls:
            require(all(torch.equal(a, b) for a, b in zip(controls['baseline_replay_predictions'], raw['prediction_base'])), 'saved baseline replay exact')
            control_audit['replay_scope'] = 'independent exact saved prediction comparison'
        else:
            control_audit['replay_scope'] = 'producer assertion only; replay predictions were not saved in original primary run'
        if not activation:
            measured_gm = arr(controls['mean_gradient'])
            close(controls['mean_gradient_max_abs'], float(np.max(np.abs(measured_gm-gm))), 'mean gradient max error')
            rel = length(measured_gm-gm)/max(length(gm), 1e-300)
            close(controls['mean_gradient_relative_l2'], rel, 'mean gradient relative error')
            require(rel < 2e-5, 'predeclared producer mean-gradient accumulation tolerance')
            na = (clip32(g1, float('inf'))[0]+clip32(g2, float('inf'))[0])*np.float32(.5)
            nb = clip32(gm, float('inf'))[0]
            require(np.array_equal(na, nb), 'independent no-clip vector identity')
            require(torch.equal(cpu_adam(state, na), cpu_adam(state, nb)), 'independent no-clip CPU Adam identity')
            require(controls['inactive_gradient_exact'] and controls['inactive_optimizer_exact'], 'producer no-clip flags')
        halves = controls['half_predictions']
        half = [score_difference(arr(halves['A'][i]), arr(halves['B'][i]), arr(inputs['queries'][r['image_id']]['target'])) for i, r in enumerate(records)]
        close(controls['half_displacement_score_A_minus_B'], half, 'half score raw reduction')
        half_da = (theta+da*.5)-theta
        half_db = (theta+db*.5)-theta
        half_diff = arr((theta+half_da)-(theta+half_db))
        half_linear = [-inner(g, half_diff) for g in gs]
        control_audit.update(half_score=half, half_linear_using_actual_fp32_interpolation=half_linear,
            half_absolute_prediction_error=[abs(a-b) for a, b in zip(half, half_linear)],
            note='Measured remainder, not a mandatory small-error or successful mechanism gate')
    else:
        require(controls == {}, 'controls only first patient per state')
    public_control = {k: v for k, v in controls.items() if k not in ('half_predictions', 'mean_gradient', 'baseline_replay_predictions')}
    equal_tree(public_control, row['controls'], 'controls JSON mirror')
    return dict(patient_id=row['patient_id'], step=row['step'], adam_cpu_comparison=adam,
        score_A_minus_B=contrasts, linear_prediction=linear,
        absolute_prediction_error=[abs(a-b) for a, b in zip(contrasts, linear)],
        Emean_score=sum(contrasts[:2])/2, U_score=contrasts[2], controls=control_audit,
        note='Saved arithmetic audit; no sign/magnitude efficacy pass threshold')


def audit(run, output, expected_code):
    require(sha(__file__) == expected_code.lower(), 'expected verifier code hash')
    require(not torch.cuda.is_initialized(), 'CPU entry')
    started = time.perf_counter()
    p, report = read(run/'protocol.json'), read(run/'report.json')
    bindings = {str(run/f): sha(run/f) for f in ['protocol.json', 'report.json', 'results.json', 'inputs.pt', 'state_250.pt', 'state_1000.pt']}
    output.mkdir(parents=False, exist_ok=False)
    write_new(output/'protocol.json', dict(schema='independent-clipping-audit/v1', source_sha256=expected_code,
        input_sha256=bindings, numpy_version=np.__version__, torch_version=torch.__version__,
        no_GPU=True, policy='Independent saved arithmetic and CPU Adam comparison; not performance certification',
        arithmetic_tolerances=dict(scalar_atol=2e-12, scalar_rtol=2e-9,
            float32_derived_atol=2e-10, float32_derived_rtol=3e-6, adam_envelope_eps_multiplier=32)))
    activation = protocol_check(run, p, report)
    fixture_paths = [Path(k) for k in p['input_sha256'] if Path(k).name == 'public_triplet_fixture.json']
    require(len(fixture_paths) == 1, 'fixture binding')
    fixture = read(fixture_paths[0])
    require(sha(fixture_paths[0]) == p['fixture_sha256'], 'fixture digest')
    require(sha(fixture['source']) == fixture['source_sha256'], 'source metadata hash')
    with Path(fixture['source']).open(encoding='utf-8-sig', newline='') as f:
        source_rows = {r['image_id']: r for r in csv.DictReader(f)}
    patients = list(dict.fromkeys(r['patient_id'] for r in fixture['records']))
    require(patients == p['patient_order'] and len(patients) == 8, 'patient order')
    for r in fixture['records']:
        require(sha(r['path']) == r['sha256'], 'raw image hash')
        require(source_rows[r['image_id']]['partition'] == 'public_development', 'public role')
        require(source_rows[r['image_id']]['patient_id'] == r['patient_id'], 'source patient identity')
    for binding in fixture['excluded_manifest_bindings']:
        require(sha(binding['path']) == binding['sha256'], 'exclusion manifest hash')
        with Path(binding['path']).open(encoding='utf-8-sig', newline='') as f:
            require(not set(patients) & {r['patient_id'] for r in csv.DictReader(f)}, 'no prior pilot patients')
    inputs, inspected_input_hashes = check_inputs(run, p, fixture)
    results = read(run/'results.json')
    require([(r['step'], r['patient_id']) for r in results] == [(s, x) for s in [250, 1000] for x in patients], '16 fixed ordered cases')
    if activation:
        original = Path(p['source_run'])
        require(sha(original/'results.json') == p['source_results_sha256'] and sha(original/'protocol.json') == p['source_protocol_sha256'], 'activation source binding')
        require(sha(p['calibration_path']) == p['calibration_sha256'], 'pre-existing calibration grid hash')
        require(min(read(p['calibration_path'])['selection_rule']['grid']) == p['clip_norm'], 'minimum public grid C')
        for name in ['inputs.pt', 'state_250.pt', 'state_1000.pt']:
            require(sha(run/name) == sha(original/name), 'activation input/state exact reuse')
        primary_rows = read(original/'results.json')
        require(all(r['both_gradients_unclipped'] and r['delta_A_B_norm'] == 0 for r in primary_rows), 'explicit posthoc trigger')
    audits = []
    for b in p['checkpoint_bindings']:
        state = state_check(run/f"state_{b['step']}.pt", Path(b['path']), b['step'])
        for index, row in enumerate(r for r in results if r['step'] == b['step']):
            raw_path = run/row['raw_file']
            require(raw_path.resolve().parent == run.resolve() and sha(raw_path) == row['raw_sha256'], 'packet path/hash')
            raw = torch.load(raw_path, map_location='cpu', weights_only=True)
            expected_records = sorted([r for r in fixture['records'] if r['patient_id'] == row['patient_id']], key=lambda r: r['role'])
            require(raw['records'] == expected_records, 'packet exact fixture records')
            if activation:
                source_path = Path(raw['source_raw_file'])
                require(source_path.parent.resolve() == original.resolve() and sha(source_path) == raw['source_raw_sha256'], 'activation packet binding')
                old = torch.load(source_path, map_location='cpu', weights_only=True)
                for key in ['gradients', 'prediction_base', 'records', 'patient_id', 'step']:
                    equal_tree(raw[key], old[key], 'unchanged reused/' + key)
            audits.append(verify_packet(row, raw, state, inputs, p['clip_norm'], index == 0, activation))
    for path, expected in bindings.items():
        require(sha(path) == expected, 'input immutable after audit')
    require(sha(__file__) == expected_code.lower() and not torch.cuda.is_initialized(), 'self unchanged/CPU exit')
    arithmetic = dict(cases=audits, interpretation='Posthoc C=.01 activation control' if activation else 'Prespecified public-p80 local primary',
        hypothesis_decision='not assigned by integrity verifier')
    write_new(output/'arithmetic.json', arithmetic)
    final = dict(status='PASS_SAVED_LOCAL_CLIPPING_ADAM_ARITHMETIC_AND_PROVENANCE', complete=True,
        activation_control=activation, cases=16, patients=8, forward=report['forward'], backward=report['backward'],
        elapsed_seconds=time.perf_counter()-started, source_code_sha256=expected_code,
        results_sha256=sha(run/'results.json'), protocol_sha256=sha(run/'protocol.json'), report_sha256=sha(run/'report.json'),
        arithmetic_sha256=sha(output/'arithmetic.json'), inspected_reference_sha256=inspected_input_hashes,
        limitations=['No independent UNet/VAE inference or full backward recomputation.',
            'Primary baseline replay retains only producer assertions; activation raw replay is independently compared.',
            'Cross-device Adam differences are measured under a fixed float32 arithmetic envelope.',
            'Reference cache/scheduler inspected at verification time; producer original binding scope is preserved.',
            'No DP guarantee, population inference, MIA metric, or mechanism-success gate.'])
    write_new(output/'verification.json', final)
    print(json.dumps({k: final[k] for k in ['status', 'cases', 'elapsed_seconds', 'activation_control']}))


def self_test():
    # Exact score sign/tie identity and an independently derived learned-moment Adam step.
    p = np.array([1., 2., -3.], dtype=np.float32)
    q = np.array([1.1, 1.9, -3.1], dtype=np.float32)
    e = np.array([.1, .2, .3], dtype=np.float32)
    close(score_difference(p, q, e), mse(q, e)-mse(p, e), 'synthetic score')
    require(score_difference(p, p, e) == 0, 'exact tied prediction')
    theta = torch.tensor([.2, -.3, .4], dtype=torch.float32)
    parameter = torch.nn.Parameter(theta.clone())
    opt = torch.optim.AdamW([parameter], lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    opt.state[parameter] = dict(step=torch.tensor(250.), exp_avg=torch.tensor([.002,-.003,.001]), exp_avg_sq=torch.tensor([.0003,.0002,.0001]))
    state = dict(theta=theta, parameter_shapes=[[3]], optimizer=copy.deepcopy(opt.state_dict()))
    g = np.array([.01,-.02,.03], dtype=np.float32)
    actual = cpu_adam(state, g)
    m = .9*np.array([.002,-.003,.001])+.1*g.astype(np.float64)
    v = .999*np.array([.0003,.0002,.0001])+.001*g.astype(np.float64)**2
    expected = theta.double().numpy()*(1-1e-6)-1e-4*(m/(1-.9**251))/(np.sqrt(v/(1-.999**251))+1e-8)
    close(actual.numpy(), expected, 'independent Adam equation', atol=8e-8, rtol=0)
    a, b = np.array([4.,0.], dtype=np.float32), np.array([0.,1.], dtype=np.float32)
    va = (clip32(a,1)[0]+clip32(b,1)[0])/np.float32(2)
    vb = clip32((a+b)/np.float32(2),1)[0]
    close(va, [.5,.5], 'analytic clipping A')
    close(vb, np.array([4,1])/np.sqrt(17), 'analytic clipping B', atol=1e-7)
    require(not np.array_equal(va, vb), 'active control discriminates order')
    require(not torch.cuda.is_initialized(), 'CPU synthetic tests')
    print('PASS_SYNTHETIC_SCORE_CLIPPING_AND_LEARNED_ADAM_CPU')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--run-dir', type=Path, default=DEFAULT)
    ap.add_argument('--output-tag', default='verification_v1')
    ap.add_argument('--expected-code-sha256')
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test()
    else:
        require(bool(args.expected_code_sha256), 'expected self hash required')
        require(Path(args.output_tag).name == args.output_tag, 'simple immutable output tag')
        audit(args.run_dir, args.run_dir/args.output_tag, args.expected_code_sha256)
