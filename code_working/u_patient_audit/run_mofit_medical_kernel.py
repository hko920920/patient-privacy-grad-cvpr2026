"""Immutable MoFit input-gradient conformance/runtime attempts; no MIA statistics.

Create and inspect the CPU contract first. A separate --run invocation is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import re
import time
import traceback

import torch

from .common import ROOT, RUN, IMAGES, digest, read_csv, write_json, verify_inputs
from .models import setup, load_unet, snapshot, scheduler, adapter_state
from .mofit_medical_adapter import MoFitMedicalAdapter, kernel_policy, tensor_hash, make_draws
from .prepare_mofit_medical_inputs import OUTPUT as INPUT_DIR

RESEARCH = ROOT.parent / 'CVPR 주제 탐색/research_2026-09-10'
OUTPUT_ROOT = RUN / 'baseline_screen_20260915'
CHECKPOINT = RUN / 'training_coverage_v2/model_1/step_1000.pt'
IMAGE_ORDER = ['00014393_002.png', '00014393_001.png']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def assert_hashes(bindings):
    for filename, expected in bindings.items():
        if digest(filename) != expected:
            raise AssertionError('Frozen file changed: ' + filename)


def environment_files():
    import diffusers
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    base = Path(diffusers.__file__).parent
    return [Path(inspect.getfile(torch.optim.Adam)), Path(inspect.getfile(AutoencoderKL)),
            Path(inspect.getfile(UNet2DConditionModel)), Path(inspect.getfile(DDPMScheduler)),
            base / 'models/autoencoders/vae.py', base / 'models/unets/unet_2d_blocks.py']


def create_contract(path):
    if Path(path).exists():
        raise RuntimeError('Contract exists; preserved')
    verify_inputs()
    summary = load(INPUT_DIR / 'summary.json')
    assert digest(INPUT_DIR / 'inputs.pt') == summary['inputs_sha256']
    assert digest(INPUT_DIR / 'preparation_protocol.json') == summary['protocol_sha256']
    frozen = dict(summary['source_sha256'])
    assert_hashes(frozen)
    upstream = RESEARCH / 'code_sources/X12'
    snap = snapshot()
    paths = [Path(__file__), ROOT / 'u_patient_audit/mofit_medical_adapter.py',
             ROOT / 'u_patient_audit/verify_mofit_medical_kernel.py',
             ROOT / 'u_patient_audit/common.py', ROOT / 'u_patient_audit/models.py',
             ROOT / 'u_patient_audit/prepare_mofit_medical_inputs.py',
             ROOT / 'dp_protocol/calibrate_xray_public_clip_norms.py',
             CHECKPOINT, RUN / 'cohort/model_1_train.csv', RUN / 'training_coverage_v2/protocol.json',
             INPUT_DIR / 'inputs.pt', INPUT_DIR / 'summary.json', INPUT_DIR / 'preparation_protocol.json',
             upstream / 'COCO/MoFit_COCO.py', upstream / 'COCO/MoFit_COCO.sh',
             upstream / 'COCO/diffusers_0_18_2/pipelines/stable_diffusion/pipeline_stable_diffusion.py',
             upstream / 'COCO/diffusers_0_18_2/schedulers/scheduling_ddim.py']
    for component in ['unet', 'vae', 'scheduler']:
        paths.extend(p for p in (snap / component).iterdir() if p.is_file())
    paths.extend(environment_files())
    rows = {r['image_id']: r for r in read_csv(RUN / 'cohort/evaluation_images.csv')}
    paths.extend(IMAGES / rows[i]['image_filename'] for i in summary['images'])
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
        frozen[str(p.resolve())] = digest(p)
    sched = scheduler()  # CPU config only, no GPU context.
    assert sched.config.prediction_type == 'epsilon'
    contract = {
        'schema': 'mofit-medical-execution-contract/v1', 'current_step': 2,
        'created_utc': now(), 'patient_id': '14393', 'eval_role': 'fit',
        'model': 'model_1', 'checkpoint_step': 1000, 'dtype': 'float32',
        'checkpoint_sha256': digest(CHECKPOINT), 'input_directory': str(INPUT_DIR),
        'inputs_sha256': summary['inputs_sha256'], 'model_snapshot': str(snap),
        'policy': kernel_policy(), 'frozen_sha256': frozen,
        'benchmark': {'image_order': [IMAGE_ORDER[0]], 'stage1_steps': 2, 'stage2_steps': 2,
                      'fd_steps': [.01, .005], 'endpoint_diagnostics': False,
                      'CFG_control': 'reuse first literal CFG2 update; one same-point CFG1 gradient call',
                      'expected_counts': {'unet_forward_calls': 19, 'unet_forward_examples': 25,
                                          'unet_backward_calls': 5, 'vae_forward': 9, 'vae_backward': 3}},
        'full': {'image_order': IMAGE_ORDER, 'stage1_steps': 1000, 'stage2_steps': 300,
                 'fd_steps': [], 'endpoint_diagnostics': True,
                 'expected_counts_per_image': {'unet_forward_calls': 2202,
                       'unet_forward_examples': 3203, 'unet_backward_calls': 1300,
                       'vae_forward': 1003, 'vae_backward': 1000}},
        'alpha_cumprod_t140': float(sched.alphas_cumprod[140]),
        'vae_scaling_factor': .18215, 'UNet_training': False, 'VAE_training': False,
        'checkpointing': 'eval mode: checkpoint flags may exist but no training-gated block recomputation',
        'optimizer': kernel_policy()['stage2']['Adam'],
        'E_U_initialization': 'same frozen BLIP captions/hidden and four CPU draws from inputs.pt; no resampling',
        'FD_diagnostic': {
            'raw_scale': 'eps32*(abs(Lplus)+abs(Lminus))/(2h)',
            'independent_diagnostic_floor': '8*eps32*max(abs(Lplus),abs(Lminus),abs(L0))/(2h)',
            'meaning': 'heuristic scalar cancellation scales; neither bounds full neural-network evaluation error',
            'no_universal_model_tolerance': True},
        'full_trace_scope': 'all iteration scalar/hash traces; only declared five pixel and five embedding full packets',
        'weight_guards': 'all parameters frozen; version/grad checks; full UNet/VAE tensor fingerprints before and after',
        'scope': 'one patient, target1 E/U one image each, literal schedule numerical/runtime kernel',
        'perform_membership_performance_evaluation': False, 'new_target_training': False,
        'caption_selection_or_editing': False, 'automatic_low_precision_or_CFG_fallback': False,
        'stage2_completion_claim': False,
    }
    new_json(path, contract)
    return contract


def validate(contract):
    for k, v in {'schema': 'mofit-medical-execution-contract/v1', 'current_step': 2,
                 'patient_id': '14393', 'eval_role': 'fit', 'model': 'model_1',
                 'checkpoint_step': 1000, 'dtype': 'float32',
                 'perform_membership_performance_evaluation': False,
                 'new_target_training': False, 'caption_selection_or_editing': False,
                 'automatic_low_precision_or_CFG_fallback': False}.items():
        assert contract[k] == v, k
    assert contract['policy'] == kernel_policy()
    assert contract['benchmark']['image_order'] == IMAGE_ORDER[:1]
    assert contract['full']['image_order'] == IMAGE_ORDER
    assert contract['benchmark']['stage1_steps'] == contract['benchmark']['stage2_steps'] == 2
    assert contract['full']['stage1_steps'] == 1000 and contract['full']['stage2_steps'] == 300
    assert contract['benchmark']['fd_steps'] == [.01, .005] and contract['full']['fd_steps'] == []
    assert contract['benchmark']['endpoint_diagnostics'] is False
    assert contract['full']['endpoint_diagnostics'] is True
    assert_hashes(contract['frozen_sha256'])
    assert digest(CHECKPOINT) == contract['checkpoint_sha256']
    assert digest(INPUT_DIR / 'inputs.pt') == contract['inputs_sha256']
    payload = torch.load(INPUT_DIR / 'inputs.pt', map_location='cpu', weights_only=True)
    assert payload['policy'] == load(INPUT_DIR / 'preparation_protocol.json')
    assert payload['policy']['kernel_policy'] == kernel_policy()
    expected_draws = make_draws()
    for k, v in expected_draws.items():
        assert torch.equal(payload['draws'][k], v) if isinstance(v, torch.Tensor) else payload['draws'][k] == v
    assert torch.equal(payload['hidden'][''], payload['null_hidden'])
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert state['step'] == 1000 and sum(state['exposures'].values()) == 4000
    manifest = {r['image_id'] for r in read_csv(RUN / 'cohort/model_1_train.csv')}
    assert set(state['exposures']) == manifest
    rows = {}
    for image_id in IMAGE_ORDER:
        item = payload['images'][image_id]
        row = dict(item['metadata'])
        assert row['patient_id'] == '14393' and row['eval_role'] == 'fit' and row['assignment_group'] == 'A'
        e = row['record_role'] == 'train_candidate'
        exposure = int(state['exposures'].get(image_id, 0))
        assert (exposure > 0) == e
        assert digest(IMAGES / row['image_filename']) == item['original_file_sha256'] == row['sha256']
        assert tensor_hash(item['pixels']) == item['pixels_sha256']
        assert tuple(item['pixels'].shape) == (1, 3, 256, 256)
        assert tuple(payload['hidden'][item['caption']].shape) == (1, 77, 1024)
        rows[image_id] = {'image_id': image_id, 'patient_id': '14393', 'eval_role': 'fit',
                         'assignment_group': 'A', 'scenario': 'E' if e else 'U',
                         'record_role': row['record_role'], 'model': 'model_1', 'member': 1,
                         'image_member': int(e), 'actual_training_exposures': exposure,
                         'caption': item['caption'], 'pixels_sha256': item['pixels_sha256'],
                         'initial_hidden_sha256': tensor_hash(payload['hidden'][item['caption']])}
    assert [rows[i]['scenario'] for i in IMAGE_ORDER] == ['E', 'U']
    return payload, rows


def fingerprint(model):
    h = hashlib.sha256()
    tensors = 0
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode('utf-8'))
            h.update(str(tuple(value.shape)).encode('ascii'))
            h.update(str(value.dtype).encode('ascii'))
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
            tensors += 1
    return {'sha256': h.hexdigest(), 'tensors': tensors}


def numerical_drift(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    return {'max_absolute_difference': float((a - b).abs().max()),
            'l2_difference': float((a - b).norm()), 'literal_l2': na, 'control_l2': nb,
            'cosine': float(torch.dot(a, b) / (a.norm() * b.norm())) if na and nb else None,
            'exact_equal': bool(torch.equal(a, b))}


def execute(args, contract, payload, rows):
    from diffusers import AutoencoderKL
    mode = args.mode
    tag = args.attempt_tag or (mode + '_v1')
    if not re.fullmatch('[A-Za-z0-9_-]+', tag):
        raise ValueError('Unsafe attempt tag')
    out = OUTPUT_ROOT / ('mofit_medical_' + tag)
    out.mkdir(parents=True, exist_ok=False)
    contract_sha = digest(args.contract)
    started = time.perf_counter()
    protocol = {'schema': 'mofit-medical-kernel-execution/v1', 'mode': mode,
                'started_utc': now(), 'contract': contract, 'contract_path': str(args.contract.resolve()),
                'contract_sha256': contract_sha, 'alpha_cumprod_t140': contract['alpha_cumprod_t140'],
                'environment': {'python': platform.python_version(),
                    **{n: version(n) for n in ['torch', 'diffusers', 'transformers', 'peft', 'numpy']}}}
    new_json(out / 'protocol.json', protocol)
    unet = vae = adapter = None
    results, raw = [], {}
    cfg_raw = cfg_report = None
    totals = dict(unet_forward_calls=0, unet_forward_examples=0, unet_backward_calls=0,
                  vae_forward=0, vae_backward=0)
    try:
        setup()
        unet = load_unet(CHECKPOINT, training=False).float().requires_grad_(False).eval()
        vae = AutoencoderKL.from_pretrained(snapshot() / 'vae', torch_dtype=torch.float16,
                    variant='fp16', use_safetensors=True, local_files_only=True).float().to('cuda').eval()
        vae.requires_grad_(False)
        assert float(vae.config.scaling_factor) == contract['vae_scaling_factor'] == .18215
        sched = scheduler()
        assert float(sched.alphas_cumprod[140]) == contract['alpha_cumprod_t140']
        adapter = MoFitMedicalAdapter(unet, vae, sched, payload['null_hidden'])
        before = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        before_adapter = adapter_state(unet)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measurement_start = time.perf_counter()
        mode_contract = contract[mode]
        print(json.dumps({'event': 'MoFit_target_and_VAE_loaded', 'mode': mode,
                          'device': torch.cuda.get_device_name(0), 'seconds': time.perf_counter() - started}), flush=True)
        for image_id in mode_contract['image_order']:
            item = payload['images'][image_id]
            def progress(stage, done, total, counts, loss):
                if done <= 2 or done % 50 == 0 or done == total:
                    row = {'event': 'MoFit_progress', 'image_id': image_id,
                           'stage': stage, 'done': done, 'total': total, 'counts': counts,
                           'loss': loss, 'seconds': time.perf_counter() - measurement_start}
                    write_json(out / 'progress.json', row)
                    print(json.dumps(row, allow_nan=False), flush=True)
            measured = adapter.run(item['pixels'], image_id, payload['hidden'][item['caption']],
                  payload['draws'], stage1_steps=mode_contract['stage1_steps'],
                  stage2_steps=mode_contract['stage2_steps'], fd_steps=mode_contract['fd_steps'],
                  endpoint_diagnostics=mode_contract['endpoint_diagnostics'], progress=progress)
            result = {**rows[image_id], **measured['report']}
            raw[image_id] = measured['raw']
            expected_primary = (dict(unet_forward_calls=18, unet_forward_examples=24,
                                unet_backward_calls=4, vae_forward=8, vae_backward=2)
                                if mode == 'benchmark' else mode_contract['expected_counts_per_image'])
            assert result['counts'] == expected_primary, (result['counts'], expected_primary)
            for k, v in result['counts'].items():
                totals[k] += v
            results.append(result)
            torch.save(raw[image_id], out / (Path(image_id).stem + '.pt'))
            new_json(out / (Path(image_id).stem + '.json'), result)
            if mode == 'benchmark':
                literal = measured['raw']['pixel_updates'][0]
                control_start = time.perf_counter()
                loss, gradient, cell = adapter.pixel_cell(literal['before'],
                        payload['hidden'][item['caption']].to('cuda', torch.float32),
                        payload['draws']['diffusion_noise'].to('cuda', torch.float32),
                        gradient=True, mode='unconditional_only_diagnostic')
                control_counts = {k: adapter.counts[k] - result['counts'][k] for k in totals}
                assert control_counts == dict(unet_forward_calls=1, unet_forward_examples=1,
                                               unet_backward_calls=1, vae_forward=1, vae_backward=1)
                cfg_raw = {'image_id': image_id, 'point': literal['before'], 'gradient': gradient,
                           'cell': cell, 'literal_reference': 'raw[image_id].pixel_updates[0]'}
                cfg_report = {'image_id': image_id, 'counts': control_counts,
                              'loss_literal': literal['loss'], 'loss_unconditional_only': loss,
                              'loss_difference': loss - literal['loss'],
                              'gradient': numerical_drift(literal['gradient'], gradient),
                              'prediction': numerical_drift(literal['cell']['prediction'], cell['prediction']),
                              'seconds': time.perf_counter() - control_start,
                              'scope': 'same first-point gradient only; not equivalence of 1000 sign updates'}
                for k, v in control_counts.items():
                    totals[k] += v
                torch.save(cfg_raw, out / 'cfg_control.pt')
                new_json(out / 'cfg_control.json', cfg_report)
            adapter.assert_weights_unchanged()
        torch.cuda.synchronize()
        measurement_seconds = time.perf_counter() - measurement_start
        after = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        assert before == after
        after_adapter = adapter_state(unet)
        assert before_adapter.keys() == after_adapter.keys()
        assert all(torch.equal(before_adapter[k], after_adapter[k]) for k in before_adapter)
        adapter.assert_weights_unchanged()
        assert_hashes(contract['frozen_sha256'])
        assert digest(args.contract) == contract_sha
        if mode == 'benchmark':
            assert totals == mode_contract['expected_counts']
        new_json(out / 'results.json', results)
        torch.save(raw, out / 'raw.pt')
        execution = {'schema': 'mofit-medical-kernel-result/v1',
                     'status': 'PASS_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION',
                     'complete': True, 'current_step': 2, 'mode': mode, 'ended_utc': now(),
                     'images': len(results), 'patients': 1, 'model': 'model_1',
                     'counts': totals, 'measurement_seconds': measurement_seconds,
                     'total_seconds': time.perf_counter() - started,
                     'before_fingerprints': before, 'after_fingerprints': after,
                     'target_and_VAE_fingerprints_equal': True,
                     'adapter_tensors_exact_equal': True,
                     'parameter_versions_and_no_weight_gradients': True,
                     'frozen_files_unchanged': True,
                     'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                     'peak_memory_scope': 'counters reset after models/fingerprints loaded; kernel measurement window',
                     'cuda_device': torch.cuda.get_device_name(0),
                     'results_sha256': digest(out / 'results.json'), 'raw_sha256': digest(out / 'raw.pt'),
                     'protocol_sha256': digest(out / 'protocol.json'),
                     'cfg_control_sha256': digest(out / 'cfg_control.pt') if cfg_raw else None,
                     'cfg_control_report_sha256': digest(out / 'cfg_control.json') if cfg_raw else None,
                     'perform_membership_performance_evaluation': False,
                     'new_target_training': False, 'stage2_completion_claim': False}
        new_json(out / 'execution.json', execution)
        print(json.dumps(execution, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        failure = {'status': 'FAILED_INCOMPLETE_MOFIT_KERNEL', 'mode': mode,
                   'ended_utc': now(), 'seconds': time.perf_counter() - started,
                   'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                   'completed_images': [r['image_id'] for r in results],
                   'completed_counts': totals, 'current_adapter_counts': dict(adapter.counts) if adapter else None,
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                   'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,
                   'automatic_fallback_performed': False,
                   'limitation': 'an interrupted adapter call may have no complete raw packet; progress/counters are retained'}
        new_json(out / 'failure.json', failure)
        raise
    finally:
        del unet, vae, adapter
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-contract', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--mode', choices=['benchmark', 'full'], default='benchmark')
    parser.add_argument('--attempt-tag')
    args = parser.parse_args()
    if args.create_contract:
        contract = create_contract(args.contract)
        print(json.dumps({'status': 'CPU_CONTRACT_CREATED_NO_GPU', 'path': str(args.contract),
                          'sha256': digest(args.contract), 'frozen_files': len(contract['frozen_sha256'])}), flush=True)
        return
    contract = load(args.contract)
    payload, rows = validate(contract)
    if args.dry_run:
        print(json.dumps({'status': 'PASS_CPU_MOFIT_CONTRACT_PREFLIGHT_NO_GPU',
                          'images': list(rows), 'inputs_sha256': contract['inputs_sha256'],
                          'contract_sha256': digest(args.contract)}), flush=True)
        return
    execute(args, contract, payload, rows)


if __name__ == '__main__':
    main()
