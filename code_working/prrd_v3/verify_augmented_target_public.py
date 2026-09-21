"""One bounded four-image public check of the new B augmentation target.

No optimization updates, full128 profile, Q/V pixels, receiver or DP calls.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from .contracts import CODE, OUT, setup, sha, seed, save, source_hashes, require, now
from .datasets import manifests, PixelAccess, TRAIN, DEV
from .encoders import Encoder, BIO_PATH, state_hash, pil_tensor, preprocess
from .augmented_targets import RULE, extract_augmented_points, patient_augmented_moments
from .render import Renderer, augment, augment_parameters
from .guarded_objectives import (LearnerPolicy, bind_objective, one_pass, two_pass,
                                statistical_loss, bank_moments)
from .bank_runtime import objective_target_state, tree_digest, digest
from .verify_guarded_image_objective import PixelTrace
from .w1_boundary_debug import grads, grad_error

PATHS = CODE/'_reports/prrd_step2_paths_20260918_v1/public_paths'


def choose_public_fixture(group):
    chosen = []
    used = set()
    for label in (0, 1):
        candidates = sorted((r for r in group if int(r['label']) == label),
                            key=lambda r: seed('prrd-B-aug-public-fixture', 101, label, r['image_id']))
        count = 0
        for row in candidates:
            if row['patient_id'] in used:
                continue
            chosen.append(row); used.add(row['patient_id']); count += 1
            if count == 2:
                break
        require(count == 2, 'Insufficient public fixture patients')
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing public attempt')
    args.output.mkdir(parents=True)
    groups = manifests()
    fixture = choose_public_fixture(groups['P'])
    inputs = [BIO_PATH, OUT/'source_P_projection.npz', PATHS/'source_P_relation_projection.npz',
              PATHS/'source_P_relation_targets.npz', TRAIN, DEV]
    config = {
        'schema': 'prrd.B-augmented-target-public-check/v1', 'created': now(),
        'scope': 'P_FOUR_IMAGES_TECHNICAL_ONLY',
        'fixture': [{k: r[k] for k in ('image_id', 'patient_id', 'label', 'sha256')} for r in fixture],
        'rule': RULE, 'rule_sha256': digest(RULE),
        'cases': [{'arm': 'B', 'microbatch': m, 'renderer_step': 500, 'seed': 101} for m in (1, 4)],
        'extraction_microbatch_views': 16, 'expected_source_forward_images': 76,
        'expected_source_backward_images': 36, 'optimizer_updates': 0,
        'criteria': {'loss_abs': 2e-6, 'gradient_atol': 2e-7, 'gradient_rtol_peak': 5e-4,
                     'augmentation_reference_atol': 2e-6, 'prepared_CPU_GPU_atol': 2e-6,
                     'identity_feature_reference_atol': 2e-6},
        'numerical_contract_amendment': 'v2: compare augmentation to existing FP32 affine_grid/grid_sample, not ideal exact identity; prior gradient criteria unchanged',
        'inputs': {str(p): sha(p) for p in inputs}, 'source_sha256': source_hashes(),
        'target_scope': 'clean P cache; augmented four-image technical fixture ONLY',
        'main_bank_reuse': False, 'Q_V_pixels': False, 'recipient': False,
        'DP_expert_reserved': False, 'new_patient_efficacy': False,
    }
    # Freeze counts, inputs and tolerances before any model or image operation.
    save(args.output/'public_contract.json', config)
    setup(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    access = PixelAccess(groups, ('P',)).install()
    encoder = Encoder('source').use_projection(OUT/'source_P_projection.npz')
    encoder.use_relation_projection(PATHS/'source_P_relation_projection.npz')
    before = state_hash(encoder)
    counts = {'forward_calls': 0, 'forward_images': 0, 'backward_calls': 0, 'backward_images': 0}

    def hook(module, inputs, output):
        n = len(inputs[0])
        counts['forward_calls'] += 1; counts['forward_images'] += n
        if output.projected_global_embedding.requires_grad:
            def backward(g):
                counts['backward_calls'] += 1; counts['backward_images'] += n
                return g
            output.projected_global_embedding.register_hook(backward)
    handle = encoder.model.register_forward_hook(hook)
    features = extract_augmented_points(encoder, fixture, access, 'P', microbatch=16)
    np.savez(args.output/'public_augmented_features.npz', **features)
    assembled = patient_augmented_moments(features['z'], features['labels'],
                                         features['patient_ids'], features['roles'])
    augmented = assembled['target']
    np.savez(args.output/'public_augmented_target.npz',
             aug_m=augmented['m'], aug_A=augmented['A'], counts=augmented['counts'],
             rule_sha256=np.array(digest(RULE)))
    # Independent loops on saved features, keeping patient-class weighting explicit.
    direct = {'m': np.zeros((2, 16)), 'A': np.zeros((2, 16, 16))}
    for c in (0, 1):
        pids = sorted(set(features['patient_ids'][features['labels'] == c]))
        for pid in pids:
            indexes = np.flatnonzero((features['patient_ids'] == pid) & (features['labels'] == c))
            for i in indexes:
                for z in features['z'][i].astype(np.float64):
                    direct['m'][c] += z/(len(pids)*len(indexes)*4)
                    direct['A'][c] += np.outer(z, z)/(len(pids)*len(indexes)*4)
    aggregation_error = max(float(np.max(abs(direct[k]-augmented[k]))) for k in direct)
    require(aggregation_error <= 1e-12, 'Public augmented patient aggregation mismatch')
    # Check native augmentation placement, with identity transform before resize.
    native_cpu = torch.stack([pil_tensor(access.image(r)) for r in fixture])
    native = native_cpu.cuda()
    theta = torch.eye(2, 3, device='cuda').repeat(4, 1, 1)
    with torch.no_grad():
        identity = augment(native, (theta, torch.ones(4, device='cuda')), list(range(4)))
        pixel_error = float((identity-native).abs().max())
        reference_identity = F.grid_sample(native, F.affine_grid(theta, native.shape, align_corners=False),
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        reference_error = float((identity-reference_identity).abs().max())
        actual_theta = torch.as_tensor(features['theta'][:, 0], device='cuda')
        actual_brightness = torch.as_tensor(features['brightness'][:, 0], device='cuda')
        actual = augment(native, (actual_theta, actual_brightness), list(range(4)))
        actual_ref = (F.grid_sample(native, F.affine_grid(actual_theta, native.shape, align_corners=False),
                                   mode='bilinear', padding_mode='zeros', align_corners=False)
                      * actual_brightness[:, None, None, None]).clamp(0, 1)
        actual_reference_error = float((actual-actual_ref).abs().max())
        cpu_prepared = preprocess(native_cpu, 'source')
        gpu_prepared = preprocess(native, 'source')
        prepared_error = float((cpu_prepared-gpu_prepared.cpu()).abs().max())
        zreference, zidentity = encoder(reference_identity), encoder(identity)
        identity_feature_error = float((zreference-zidentity).abs().max())
    native_check = {'identity_vs_ideal_max_abs': pixel_error,
                    'identity_vs_reference_max_abs': reference_error,
                    'sampled_augmentation_vs_reference_max_abs': actual_reference_error,
                    'prepared_CPU_GPU_max_abs': prepared_error,
                    'identity_feature_vs_reference_max_abs': identity_feature_error}
    save(args.output/'native_path_check.json', native_check)
    require(max(reference_error, actual_reference_error, prepared_error,
                identity_feature_error) <= 2e-6, 'Native/preprocess reference parity failed')
    templates = torch.stack([TF.resize(x, [224, 224], antialias=True) for x in native_cpu]).cuda()
    del native, identity, reference_identity, actual, actual_ref, cpu_prepared, gpu_prepared, zreference, zidentity
    with np.load(PATHS/'source_P_relation_targets.npz', allow_pickle=False) as archive:
        target = {k: archive['C_'+k].copy() for k in ('m', 'A')}
    objective = bind_objective(target, 'B', {}, LearnerPolicy(0., .1, kappa=.1),
                               eta=1., device='cuda', aug_target=augmented)
    target_digest = tree_digest(objective_target_state(objective))
    rows = []
    for i, case in enumerate(config['cases']):
        model = Renderer(templates, 'B', 101); model.set_step(500)
        initial = state_hash(model)
        reference_trace = PixelTrace(encoder)
        value = one_pass(model, reference_trace, objective, 500, 101, case['microbatch'])
        value['loss'].backward()
        reference = grads(model); reference_loss = float(value['loss'].detach())
        del value
        replay_trace = PixelTrace(encoder)
        replay = two_pass(model, replay_trace, objective, 500, 101, case['microbatch'])
        parameter_error = grad_error(grads(model), reference)
        require(len(reference_trace.pixel_gradients) == len(replay_trace.pixel_gradients),
                'Pixel replay length differs')
        require(all(v is not None and bool(torch.isfinite(v).all())
                    for v in reference_trace.pixel_gradients+replay_trace.pixel_gradients),
                'Missing or nonfinite image gradient')
        pixel_gradient_error = grad_error(replay_trace.pixel_gradients, reference_trace.pixel_gradients)
        loss_error = abs(reference_loss-replay['loss'])
        passed = (loss_error <= 2e-6 and
                  all(err['max_abs'] <= 2e-7+5e-4*err['reference_peak']
                      for err in (parameter_error, pixel_gradient_error)))
        fixed = state_hash(model) == initial and tree_digest(objective_target_state(objective)) == target_digest
        row = dict(case, loss_abs=loss_error, parameter_gradient=parameter_error,
                   pixel_gradient=pixel_gradient_error, passed=bool(passed and fixed),
                   renderer_and_targets_unchanged=fixed, trace=replay)
        rows.append(row); save(args.output/f'gradient_case_{i}.json', row)
        print(json.dumps(row), flush=True)
        require(passed and fixed, 'Changed-target one/two-pass check failed')
        del model, reference, reference_trace, replay_trace
        torch.cuda.empty_cache()
    # Demonstrate gradient from the augmented statistics, excluding clean/prior/function terms.
    x = templates.detach().clone().requires_grad_(True)
    params = augment_parameters(4, 101, 500, False, 'cuda')
    paths = encoder.forward_paths(augment(x, params, list(range(4))))
    aug_loss = statistical_loss(bank_moments(paths, 'B', {}), objective.aug_moments, 'B')
    image_gradient, = torch.autograd.grad(aug_loss, x)
    aug_probe = {'loss': float(aug_loss.detach()), 'gradient_norm': float(image_gradient.norm()),
                 'finite': bool(torch.isfinite(image_gradient).all()),
                 'clean_function_prior_included': False}
    require(aug_probe['finite'] and aug_probe['gradient_norm'] > 0, 'Augmented target detached from image')
    del x, paths, aug_loss, image_gradient
    handle.remove(); torch.cuda.synchronize()
    require(counts['forward_images'] == 76 and counts['backward_images'] == 36,
            'Predeclared model operation counts differ')
    require(state_hash(encoder) == before, 'Frozen source changed')
    require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()),
            'Source gradient leakage')
    require(source_hashes() == config['source_sha256'], 'Source code changed during check')
    summary = {
        'status': 'PASS_B_AUGMENTED_TARGET_PUBLIC_IMAGE_CONNECTION_ONLY',
        'conditions': len(rows), 'rows': rows, 'aggregation_max_abs': aggregation_error,
        'native_identity_pixel_max_abs': pixel_error, 'CPU_GPU_prepared_max_abs': prepared_error,
        'native_path_check': native_check,
        'identity_point_reference_feature_max_abs': identity_feature_error, 'augmentation_only_probe': aug_probe,
        'source_state_unchanged': True, 'source_state_sha256': before,
        'counts': counts, 'access': access.report(), 'seconds': time.perf_counter()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'GPU': torch.cuda.get_device_name(), 'contract_sha256': sha(args.output/'public_contract.json'),
        'features_sha256': sha(args.output/'public_augmented_features.npz'),
        'target_sha256': sha(args.output/'public_augmented_target.npz'),
        'new_training_updates': 0, 'full128_profile': False, 'main_bank_reuse': False,
        'Q_V_pixels': 0, 'recipient_runs': 0, 'DP_releases': 0,
        'expert_reserved': False, 'patient_utility_evaluated': False,
    }
    save(args.output/'verification.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'conditions', 'counts', 'access', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
