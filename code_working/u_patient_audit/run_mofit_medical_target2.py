"""Same fixed MoFit full kernel on target2, where patient14393 is absent.

Only target/checkpoint/exposure metadata changes. Frozen numerical producer,
captions, pixels, all four draws, and 1000+300 schedule are reused unchanged.
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

from .common import ROOT, RUN, IMAGES, digest, read_csv, write_json
from .models import setup, snapshot, scheduler, load_unet, adapter_state
from .mofit_medical_adapter import MoFitMedicalAdapter, kernel_policy, tensor_hash, make_draws
from .run_mofit_medical_kernel import INPUT_DIR, load, new_json, now, fingerprint, assert_hashes

CHECKPOINT = RUN / 'training_coverage_v2/model_2/step_1000.pt'
ORIGINAL_CONTRACT = RUN / 'baseline_screen_20260915/mofit_medical_contract_v1.json'
FD_DIR = RUN / 'baseline_screen_20260915/mofit_pixel_fd_refinement_v1'
IMAGES_ORDER = ['00014393_002.png', '00014393_001.png']
COUNTS = dict(unet_forward_calls=2202, unet_forward_examples=3203,
              unet_backward_calls=1300, vae_forward=1003, vae_backward=1000)
KERNEL_ARGUMENTS = dict(stage1_steps=1000, stage2_steps=300, nominal_stage1_steps=1000,
                        mode='literal_cfg2', monitor='literal_every_step', fd_steps=[],
                        endpoint_diagnostics=True,
                        capture_pixel=[0, 1, 99, 499, 999], capture_embedding=[0, 1, 29, 149, 299])


def actual_exposure():
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    manifest = read_csv(RUN / 'cohort/model_2_train.csv')
    assert state['step'] == 1000 and sum(state['exposures'].values()) == 4000
    assert set(state['exposures']) == {r['image_id'] for r in manifest}
    patient_rows = [r for r in manifest if r['patient_id'] == '14393']
    assert not patient_rows, 'Patient14393 must be absent from target2 manifest'
    assert all(state['exposures'].get(i, 0) == 0 for i in IMAGES_ORDER)
    return state, {'patient_training_images': 0, 'patient_training_exposures': 0,
                   'manifest_images': len(manifest), 'training_exposures_total': 4000}


def create_contract(path):
    if path.exists():
        raise RuntimeError('Immutable target2 contract already exists')
    old = load(ORIGINAL_CONTRACT)
    frozen = dict(old['frozen_sha256'])
    assert_hashes(frozen)
    paths = [Path(__file__), ROOT / 'u_patient_audit/verify_mofit_medical_target2.py',
             ORIGINAL_CONTRACT, CHECKPOINT, RUN / 'cohort/model_2_train.csv']
    paths.extend(FD_DIR / name for name in ['protocol.json', 'results.json', 'raw.pt',
                 'execution.json', 'verification.json', 'verification_protocol.json'])
    fd_protocol = load(FD_DIR / 'protocol.json')
    paths.append(Path(fd_protocol['contract_path']))
    for p in paths:
        frozen[str(p.resolve())] = digest(p)
    _, exposure = actual_exposure()
    contract = {'schema': 'mofit-medical-target2-contract/v1', 'current_step': 2,
                'created_utc': now(), 'patient_id': '14393', 'eval_role': 'fit',
                'assignment_group': 'A', 'model': 'model_2', 'checkpoint_step': 1000,
                'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': digest(CHECKPOINT),
                'input_directory': str(INPUT_DIR), 'inputs_sha256': old['inputs_sha256'],
                'image_order': IMAGES_ORDER, 'member': 0, 'image_member_by_image': {i: 0 for i in IMAGES_ORDER},
                'actual_training_exposures_by_image': {i: 0 for i in IMAGES_ORDER},
                'actual_exposure': exposure, 'model_snapshot': old['model_snapshot'],
                'alpha_cumprod_t140': old['alpha_cumprod_t140'], 'vae_scaling_factor': .18215,
                'dtype': 'float32', 'kernel_policy': kernel_policy(), 'kernel_arguments': KERNEL_ARGUMENTS,
                'checkpoint_adapter_loading': 'LoRA parameters stay FP32 in PEFT load_unet; compare saved FP32 values exactly. Base FP16 artifact is promoted separately.',
                'expected_counts_per_image': COUNTS, 'expected_total_counts': {k: 2*v for k,v in COUNTS.items()},
                'numerical_contract_reference': str(ORIGINAL_CONTRACT),
                'pixel_FD_diagnostic_directory': str(FD_DIR),
                'E_meaning': 'train_candidate photo role in the fixed cohort; neither image is in target2 training',
                'U_meaning': 'U_observed photo role; image nonmember in both fixed targets',
                'paired_comparison_scope': 'same one patient, E002/U001, target1 participant versus target2 nonparticipant',
                'causal_limit': 'whole A/B training cohorts differ; not an isolated individual-patient causal intervention',
                'initialization': 'reuse exact frozen pixels/captions/hidden and all four saved CPU draws across targets',
                'scope_change_from_target1': ['model2 checkpoint', 'model2 manifest/exposures',
                        'member=0', 'both image_member=0', 'both image exposure=0', 'patient exposure=0'],
                'no_new_backward_check': 'target1 local FD supports the common implementation; target2 full gradients are not independently re-evaluated',
                'frozen_sha256': frozen, 'new_target_training': False,
                'attack_parameter_tuning': False, 'cohort_expansion': False,
                'perform_membership_performance_evaluation': False,
                'stage2_completion_claim': False}
    new_json(path, contract)
    return contract


def validate(c):
    for key, value in {'schema': 'mofit-medical-target2-contract/v1', 'current_step': 2,
            'patient_id': '14393', 'eval_role': 'fit', 'assignment_group': 'A', 'model': 'model_2',
            'checkpoint_step': 1000, 'member': 0, 'dtype': 'float32', 'new_target_training': False,
            'attack_parameter_tuning': False, 'cohort_expansion': False,
            'perform_membership_performance_evaluation': False, 'stage2_completion_claim': False}.items():
        assert c[key] == value, key
    assert c['image_order'] == IMAGES_ORDER and c['kernel_arguments'] == KERNEL_ARGUMENTS
    assert c['kernel_policy'] == kernel_policy() and c['expected_counts_per_image'] == COUNTS
    assert c['expected_total_counts'] == {k: 2*v for k,v in COUNTS.items()}
    assert_hashes(c['frozen_sha256'])
    assert digest(CHECKPOINT) == c['checkpoint_sha256']
    state, exposure = actual_exposure()
    assert exposure == c['actual_exposure']
    assert digest(INPUT_DIR / 'inputs.pt') == c['inputs_sha256']
    payload = torch.load(INPUT_DIR / 'inputs.pt', map_location='cpu', weights_only=True)
    assert payload['policy'] == load(INPUT_DIR / 'preparation_protocol.json')
    assert payload['policy']['kernel_policy'] == kernel_policy()
    for key, expected in make_draws().items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(payload['draws'][key], expected)
        else:
            assert payload['draws'][key] == expected
    assert torch.equal(payload['hidden'][''], payload['null_hidden'])
    rows = {}
    for image in IMAGES_ORDER:
        item = payload['images'][image]
        source = item['metadata']
        assert source['patient_id'] == '14393' and source['eval_role'] == 'fit' and source['assignment_group'] == 'A'
        e = source['record_role'] == 'train_candidate'
        assert state['exposures'].get(image, 0) == c['actual_training_exposures_by_image'][image] == 0
        assert c['image_member_by_image'][image] == 0
        assert digest(IMAGES / source['image_filename']) == item['original_file_sha256'] == source['sha256']
        assert tensor_hash(item['pixels']) == item['pixels_sha256']
        rows[image] = dict(image_id=image, patient_id='14393', eval_role='fit', assignment_group='A',
            scenario='E' if e else 'U', record_role=source['record_role'], model='model_2',
            member=0, image_member=0, actual_training_exposures=0, patient_training_exposures=0,
            caption=item['caption'], pixels_sha256=item['pixels_sha256'],
            initial_hidden_sha256=tensor_hash(payload['hidden'][item['caption']]))
    assert [rows[i]['scenario'] for i in IMAGES_ORDER] == ['E', 'U']
    return payload, rows, state


def execute(args, c, payload, metadata, checkpoint_state):
    from diffusers import AutoencoderKL
    tag = args.attempt_tag or 'v1'
    assert re.fullmatch('[A-Za-z0-9_-]+', tag)
    out = RUN / 'baseline_screen_20260915' / ('mofit_medical_target2_full_' + tag)
    out.mkdir(parents=True, exist_ok=False)
    contract_sha = digest(args.contract)
    started = time.perf_counter()
    protocol = {'schema': 'mofit-medical-target2-execution/v1', 'current_step': 2, 'mode': 'full',
                'started_utc': now(), 'contract': c, 'contract_path': str(args.contract.resolve()),
                'contract_sha256': contract_sha, 'alpha_cumprod_t140': c['alpha_cumprod_t140'],
                'environment': {n: version(n) for n in ['torch', 'diffusers', 'transformers', 'peft', 'numpy']}}
    new_json(out / 'protocol.json', protocol)
    unet = vae = adapter = None
    results, raw = [], {}
    totals = {k: 0 for k in COUNTS}
    try:
        setup()
        unet = load_unet(CHECKPOINT, training=False).float().requires_grad_(False).eval()
        loaded_adapter = adapter_state(unet)
        expected_adapter = checkpoint_state['adapter']
        assert loaded_adapter.keys() == expected_adapter.keys()
        assert all(torch.equal(loaded_adapter[k], expected_adapter[k].float()) for k in loaded_adapter)
        vae = AutoencoderKL.from_pretrained(snapshot() / 'vae', torch_dtype=torch.float16,
                  variant='fp16', use_safetensors=True, local_files_only=True).float().to('cuda').eval()
        vae.requires_grad_(False)
        assert float(vae.config.scaling_factor) == c['vae_scaling_factor']
        sched = scheduler()
        assert float(sched.alphas_cumprod[140]) == c['alpha_cumprod_t140']
        adapter = MoFitMedicalAdapter(unet, vae, sched, payload['null_hidden'])
        before = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        model1_reference = load(FD_DIR / 'execution.json')['before_fingerprints']
        assert before['vae'] == model1_reference['vae']
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        for image in IMAGES_ORDER:
            item = payload['images'][image]
            def progress(stage, done, total, counts, loss):
                if done <= 2 or done % 50 == 0 or done == total:
                    row = dict(event='MoFit_target2_progress', image_id=image, model='model_2',
                               stage=stage, done=done, total=total, counts=counts, loss=loss,
                               seconds=time.perf_counter() - measured_start)
                    write_json(out / 'progress.json', row)
                    print(json.dumps(row, allow_nan=False), flush=True)
            measured = adapter.run(item['pixels'], image, payload['hidden'][item['caption']],
                                   payload['draws'], **KERNEL_ARGUMENTS, progress=progress)
            report = {**metadata[image], **measured['report']}
            assert report['counts'] == COUNTS
            results.append(report)
            raw[image] = measured['raw']
            for key in COUNTS:
                totals[key] += report['counts'][key]
            torch.save(raw[image], out / (Path(image).stem + '.pt'))
            new_json(out / (Path(image).stem + '.json'), report)
            adapter.assert_weights_unchanged()
        torch.cuda.synchronize()
        measurement_seconds = time.perf_counter() - measured_start
        after = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
        assert before == after
        after_adapter = adapter_state(unet)
        assert all(torch.equal(loaded_adapter[k], after_adapter[k]) for k in loaded_adapter)
        adapter.assert_weights_unchanged()
        assert_hashes(c['frozen_sha256'])
        assert digest(args.contract) == contract_sha
        assert totals == c['expected_total_counts']
        new_json(out / 'results.json', results)
        torch.save(raw, out / 'raw.pt')
        execution = {'schema': 'mofit-medical-target2-result/v1',
            'status': 'PASS_TARGET2_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION',
            'complete': True, 'current_step': 2, 'mode': 'full', 'model': 'model_2', 'ended_utc': now(),
            'patients': 1, 'images': 2, 'counts': totals,
            'measurement_seconds': measurement_seconds, 'total_seconds': time.perf_counter() - started,
            'before_fingerprints': before, 'after_fingerprints': after,
            'target_and_VAE_fingerprints_equal': True, 'checkpoint_adapter_values_exact': True,
            'adapter_tensors_exact_equal': True, 'parameter_versions_and_no_weight_gradients': True,
            'frozen_files_unchanged': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            'peak_memory_scope': 'reset after model loading/fingerprints; full kernel window',
            'cuda_device': torch.cuda.get_device_name(0),
            'results_sha256': digest(out / 'results.json'), 'raw_sha256': digest(out / 'raw.pt'),
            'protocol_sha256': digest(out / 'protocol.json'), 'actual_exposure': c['actual_exposure'],
            'cfg_control_sha256': None, 'cfg_control_report_sha256': None,
            'perform_membership_performance_evaluation': False, 'new_target_training': False,
            'attack_parameter_tuning': False, 'cohort_expansion': False, 'stage2_completion_claim': False}
        new_json(out / 'execution.json', execution)
        print(json.dumps(execution, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        new_json(out / 'failure.json', {'status': 'FAILED_INCOMPLETE_TARGET2_KERNEL', 'ended_utc': now(),
            'seconds': time.perf_counter() - started, 'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'completed_images': [r['image_id'] for r in results],
            'completed_counts': totals, 'current_adapter_counts': dict(adapter.counts) if adapter else None,
            'automatic_fallback': False})
        raise
    finally:
        del unet, vae, adapter
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-contract', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    p.add_argument('--attempt-tag')
    args = p.parse_args()
    if args.create_contract:
        c = create_contract(args.contract)
        print(json.dumps({'status': 'CPU_TARGET2_CONTRACT_CREATED_NO_GPU',
                          'sha256': digest(args.contract), 'frozen_files': len(c['frozen_sha256'])}), flush=True)
        return
    c = load(args.contract)
    payload, metadata, state = validate(c)
    if args.dry_run:
        print(json.dumps({'status': 'PASS_CPU_TARGET2_PREFLIGHT_NO_GPU', 'actual_exposure': c['actual_exposure'],
                          'image_member': {k: v['image_member'] for k,v in metadata.items()},
                          'contract_sha256': digest(args.contract)}), flush=True)
        return
    execute(args, c, payload, metadata, state)


if __name__ == '__main__':
    main()
