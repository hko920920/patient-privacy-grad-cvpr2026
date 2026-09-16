"""Fixed smaller-step pixel directional-derivative check; no new backward/optimization.

Consumes the frozen benchmark's first pixel point, gradient, direction and inputs.
"""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
import traceback

import torch

from .common import ROOT, RUN, digest, write_json
from .models import setup, snapshot, scheduler, load_unet
from .mofit_medical_adapter import MoFitMedicalAdapter, tensor_hash
from .run_mofit_medical_kernel import load, new_json, now, fingerprint, assert_hashes, CHECKPOINT

BENCHMARK = RUN / 'baseline_screen_20260915/mofit_medical_benchmark_v1'
IMAGE = '00014393_002.png'
HS = [.002, .001, .0005, .00025, .0001]
COUNTS = dict(unet_forward_calls=10, unet_forward_examples=20,
              unet_backward_calls=0, vae_forward=10, vae_backward=0)


def create_contract(path):
    if path.exists():
        raise RuntimeError('Immutable contract already exists')
    protocol = load(BENCHMARK / 'protocol.json')
    verification = load(BENCHMARK / 'verification.json')
    require_status = 'PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE'
    assert verification['status'] == require_status
    frozen = dict(protocol['contract']['frozen_sha256'])
    assert_hashes(frozen)
    paths = [Path(__file__), ROOT / 'u_patient_audit/verify_mofit_pixel_fd_refinement.py',
             Path(protocol['contract_path'])]
    paths.extend(BENCHMARK / name for name in ['protocol.json', 'execution.json', 'results.json',
                 'raw.pt', 'verification.json', 'verification_protocol.json', 'cfg_control.json', 'cfg_control.pt'])
    for p in paths:
        frozen[str(p.resolve())] = digest(p)
    raw = torch.load(BENCHMARK / 'raw.pt', map_location='cpu', weights_only=True)[IMAGE]
    point, gradient, fd = raw['pixel_updates'][0]['before'], raw['pixel_updates'][0]['gradient'], raw['fd']['pixel']
    assert torch.equal(point, fd['point'])
    contract = {'schema': 'mofit-pixel-fd-refinement-contract/v1', 'current_step': 2,
                'created_utc': now(), 'benchmark_directory': str(BENCHMARK),
                'benchmark_image_id': IMAGE, 'checkpoint': str(CHECKPOINT),
                'checkpoint_sha256': digest(CHECKPOINT), 'h_values': HS,
                'direction_seed': 1731, 'dtype': 'float32', 'mode': 'literal_cfg2',
                'alpha_cumprod_t140': protocol['alpha_cumprod_t140'],
                'model_snapshot': protocol['contract']['model_snapshot'],
                'vae_scaling_factor': .18215, 'point_sha256': tensor_hash(point),
                'gradient_sha256': tensor_hash(gradient), 'direction_sha256': tensor_hash(fd['direction']),
                'cached_AD': fd['ADdot'], 'base_loss': raw['pixel_updates'][0]['loss'],
                'expected_counts': COUNTS, 'frozen_sha256': frozen,
                'purpose': 'directional derivative convergence/cancellation check before full optimization',
                'diagnostic_floor': '8*eps32*max(abs(Lplus),abs(Lminus),abs(L0))/(2*h)',
                'floor_is_neural_network_error_bound': False,
                'selection_rule': 'report all original and new h; no best-h pass selection',
                'same_point_cached_gradient_only': True, 'new_backward': False,
                'new_target_training': False, 'attack_parameter_tuning': False,
                'membership_performance_claim': False, 'stage2_completion_claim': False}
    new_json(path, contract)
    return contract


def validate(contract):
    assert contract['schema'] == 'mofit-pixel-fd-refinement-contract/v1'
    assert contract['benchmark_directory'] == str(BENCHMARK)
    assert contract['benchmark_image_id'] == IMAGE and contract['h_values'] == HS
    assert contract['direction_seed'] == 1731 and contract['dtype'] == 'float32'
    assert contract['mode'] == 'literal_cfg2' and contract['expected_counts'] == COUNTS
    for flag in ['new_backward', 'new_target_training', 'attack_parameter_tuning',
                 'membership_performance_claim', 'stage2_completion_claim']:
        assert contract[flag] is False
    assert_hashes(contract['frozen_sha256'])
    verification = load(BENCHMARK / 'verification.json')
    execution = load(BENCHMARK / 'execution.json')
    assert verification['status'] == 'PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE'
    for key, name in [('protocol_sha256', 'protocol.json'), ('results_sha256', 'results.json'), ('raw_sha256', 'raw.pt')]:
        assert verification[key] == execution[key] == digest(BENCHMARK / name)
    assert digest(CHECKPOINT) == contract['checkpoint_sha256']
    raw = torch.load(BENCHMARK / 'raw.pt', map_location='cpu', weights_only=True)[IMAGE]
    first, fd = raw['pixel_updates'][0], raw['fd']['pixel']
    assert first['iteration'] == 0 and torch.equal(first['before'], fd['point'])
    assert tensor_hash(first['before']) == contract['point_sha256']
    assert tensor_hash(first['gradient']) == contract['gradient_sha256']
    assert tensor_hash(fd['direction']) == contract['direction_sha256']
    regenerated = (torch.randint(0, 2, first['before'].shape,
                    generator=torch.Generator().manual_seed(1731)) * 2 - 1).float()
    assert torch.equal(regenerated, fd['direction'])
    ad = float((first['gradient'].double() * fd['direction'].double()).sum())
    assert ad == contract['cached_AD'] == fd['ADdot']
    assert first['loss'] == contract['base_loss']
    return raw, execution


def summarize(packet, origin, base_loss):
    ad = float(packet['ADdot'])
    eps = torch.finfo(torch.float32).eps
    return [{'h': r['h'], 'origin': origin, 'FD': r['finite_difference'], 'AD': ad,
             'absolute_difference': abs(r['finite_difference'] - ad),
             'relative_difference': abs(r['finite_difference'] - ad) / max(abs(ad), abs(r['finite_difference']), 1e-300),
             'diagnostic_floor': 8 * eps * max(abs(r['plus_loss']), abs(r['minus_loss']), abs(base_loss)) / (2 * r['h']),
             'sign_agreement': bool(r['finite_difference'] * ad > 0)} for r in packet['cells']]


def execute(args, contract, source, benchmark_execution):
    from diffusers import AutoencoderKL
    tag = args.attempt_tag or 'v1'
    assert re.fullmatch('[A-Za-z0-9_-]+', tag)
    out = RUN / 'baseline_screen_20260915' / ('mofit_pixel_fd_refinement_' + tag)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract_sha = digest(args.contract)
    protocol = {'schema': 'mofit-pixel-fd-refinement-execution/v1', 'started_utc': now(),
                'contract': contract, 'contract_path': str(args.contract.resolve()),
                'contract_sha256': contract_sha, 'alpha_cumprod_t140': contract['alpha_cumprod_t140'],
                'environment': {n: version(n) for n in ['torch', 'diffusers', 'peft', 'numpy']}}
    new_json(out / 'protocol.json', protocol)
    unet = vae = adapter = None
    try:
        setup()
        unet = load_unet(CHECKPOINT, training=False).float().requires_grad_(False).eval()
        vae = AutoencoderKL.from_pretrained(snapshot() / 'vae', torch_dtype=torch.float16,
                  variant='fp16', use_safetensors=True, local_files_only=True).float().to('cuda').eval()
        vae.requires_grad_(False)
        assert float(vae.config.scaling_factor) == contract['vae_scaling_factor']
        sched = scheduler()
        assert float(sched.alphas_cumprod[140]) == contract['alpha_cumprod_t140']
        adapter = MoFitMedicalAdapter(unet, vae, sched, source['null_hidden'])
        before = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        assert before == benchmark_execution['before_fingerprints'] == benchmark_execution['after_fingerprints']
        hidden = source['initial_hidden'].to('cuda', torch.float32)
        noise = source['draws']['diffusion_noise'].to('cuda', torch.float32)
        first = source['pixel_updates'][0]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        packet = adapter._fd(first['before'], first['gradient'],
              lambda point: adapter.pixel_cell(point, hidden, noise, mode='literal_cfg2'), HS, 1731)
        torch.cuda.synchronize()
        measurement_seconds = time.perf_counter() - measured_start
        assert packet['ADdot'] == contract['cached_AD']
        assert torch.equal(packet['point'], first['before'])
        assert torch.equal(packet['direction'], source['fd']['pixel']['direction'])
        assert adapter.counts == COUNTS
        adapter.assert_weights_unchanged()
        after = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        assert before == after
        assert_hashes(contract['frozen_sha256'])
        assert digest(args.contract) == contract_sha
        results = {'image_id': IMAGE, 'cached_AD': packet['ADdot'], 'base_loss': first['loss'],
                   'rows': summarize(source['fd']['pixel'], 'benchmark', first['loss']) +
                           summarize(packet, 'refinement', first['loss']),
                   'counts': dict(adapter.counts), 'no_best_h_selection': True,
                   'claim': 'all step sizes reported; derivative evidence requires interpretation'}
        new_json(out / 'results.json', results)
        torch.save(packet, out / 'raw.pt')
        execution = {'schema': 'mofit-pixel-fd-refinement-result/v1',
                     'status': 'PASS_FD_EXECUTION_PENDING_INDEPENDENT_VERIFICATION',
                     'complete': True, 'current_step': 2, 'ended_utc': now(),
                     'image_id': IMAGE, 'counts': dict(adapter.counts),
                     'measurement_seconds': measurement_seconds, 'total_seconds': time.perf_counter() - started,
                     'before_fingerprints': before, 'after_fingerprints': after,
                     'weight_versions_and_no_gradients': True, 'frozen_files_unchanged': True,
                     'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                     'results_sha256': digest(out / 'results.json'), 'raw_sha256': digest(out / 'raw.pt'),
                     'protocol_sha256': digest(out / 'protocol.json'),
                     'new_backward': False, 'new_target_training': False,
                     'attack_parameter_tuning': False, 'membership_performance_claim': False,
                     'stage2_completion_claim': False}
        new_json(out / 'execution.json', execution)
        print(json.dumps({'execution': execution, 'rows': results['rows']}, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        new_json(out / 'failure.json', {'status': 'FAILED_FD_REFINEMENT', 'ended_utc': now(),
                 'seconds': time.perf_counter() - started, 'error_type': type(exc).__name__,
                 'error': str(exc), 'traceback': traceback.format_exc(),
                 'counts': dict(adapter.counts) if adapter else None, 'automatic_fallback': False})
        raise
    finally:
        del unet, vae, adapter
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    # Match the benchmark's CPU double reduction order for the cached AD check.
    torch.set_num_threads(4)
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--create-contract', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--run', action='store_true')
    p.add_argument('--attempt-tag')
    args = p.parse_args()
    if args.create_contract:
        c = create_contract(args.contract)
        print(json.dumps({'status': 'CPU_CONTRACT_CREATED_NO_GPU', 'sha256': digest(args.contract),
                          'frozen_files': len(c['frozen_sha256'])}), flush=True)
        return
    c = load(args.contract)
    source, execution = validate(c)
    if args.dry_run:
        print(json.dumps({'status': 'PASS_CPU_FD_REFINEMENT_PREFLIGHT_NO_GPU',
                          'cached_AD': c['cached_AD'], 'h_values': HS,
                          'contract_sha256': digest(args.contract)}), flush=True)
        return
    execute(args, c, source, execution)


if __name__ == '__main__':
    main()
