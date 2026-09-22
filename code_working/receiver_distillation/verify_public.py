"""Bounded public BioViL check; no bank optimization or receiver evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from prrd_v3.contracts import CODE, OUT, RESEARCH, now, require, save, seed, setup, sha, source_hashes
from prrd_v3.datasets import manifests, PixelAccess, TRAIN, DEV
from prrd_v3.encoders import Encoder, BIO_PATH, pil_tensor, state_hash
from prrd_v3.render import Renderer
from prrd_v3.w1_boundary_debug import grads, grad_error
from .signals import (conditions, heads, per_image_gradient, patient_class_sums,
                      bind_objective, objective_digest)
from .objectives import one_pass, two_pass


class PixelTrace:
    def __init__(self, encoder):
        self.encoder = encoder
        self.gradients = []
    def __call__(self, x):
        view = x.view_as(x)
        if view.requires_grad:
            index = len(self.gradients)
            self.gradients.append(None)
            def capture(g):
                self.gradients[index] = g.detach().cpu().clone()
                return g
            view.register_hook(capture)
        return self.encoder(view)


def fixture(group):
    chosen, used = [], set()
    for label in (0, 1):
        options = sorted((r for r in group if int(r['label']) == label),
                         key=lambda r: seed('receiver-public-fixture-v1', 101, label, r['image_id']))
        count = 0
        for row in options:
            if row['patient_id'] in used:
                continue
            chosen.append(row)
            used.add(row['patient_id'])
            count += 1
            if count == 2:
                break
        require(count == 2, 'Insufficient public fixture')
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cpu-result', type=Path, required=True)
    args = ap.parse_args()
    require(json.loads(args.cpu_result.read_text())['status'] == 'PASS', 'CPU prerequisite failed')
    require(not args.output.exists(), 'Preserve previous public attempt')
    args.output.mkdir(parents=True)
    groups = manifests()
    selected = fixture(groups['P'])
    cs = conditions()
    hs = heads('biovil', 16)
    bindings = [BIO_PATH, OUT/'source_P_projection.npz', TRAIN, DEV, args.cpu_result,
                RESEARCH/'RECEIVER_DISTILLATION_DEVELOPMENT_PLAN_20260922.md']
    own_hashes = {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    contract = {
        'created': now(), 'scope': 'PUBLIC_FOUR_IMAGE_CONDITION_SIGNAL_CONNECTION_ONLY',
        'seed': 101, 'conditions': 4, 'encoder': 'BioViL-T', 'point_dimension': 16,
        'point_path': 'existing public PCA/q95; no whitening; no raw relation path',
        'patient_rule': 'mean visits within patient/class; mean class-present patients; 0.5 each class',
        'targets': 'four explicit shared augmentation and fixed linear-head conditions; NOT augmean',
        'renderer_arm': 'B', 'renderer_step': 500, 'optimizer_updates': 0,
        'cases': [{'microbatch': m} for m in (1, 4)],
        'expected_source_forward_images': 116, 'expected_source_backward_images': 68,
        'criteria': {'loss_abs': 2e-6, 'gradient_atol': 2e-7, 'gradient_rtol_peak': 5e-4,
                     'independent_target_atol': 1e-12},
        'inputs': {str(p): sha(p) for p in bindings},
        'public_fixture': [{'image_id': r['image_id'], 'sha256': r['sha256'],
                            'label': int(r['label'])} for r in selected],
        'conditions_and_heads': [
            {'index': cond.index, 'theta': cond.theta.tolist(), 'brightness': cond.brightness.tolist(),
             'weight': head.weight.tolist(), 'bias': head.bias.tolist()} for cond, head in zip(cs, hs)],
        'legacy_source_hashes': source_hashes(), 'new_source_hashes': own_hashes,
        'Q_V_pixels': False, 'recipient': False, 'DP': False, 'expert_reserved': False,
        'reusable_main_bank': False,
    }
    # Contract is sealed before loading the model or opening any patient image.
    save(args.output/'public_contract.json', contract)
    setup()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    access = PixelAccess(groups, ('P',)).install()
    encoder = Encoder('source').use_projection(OUT/'source_P_projection.npz')
    before = state_hash(encoder)
    counts = {'forward_calls': 0, 'forward_images': 0, 'backward_calls': 0, 'backward_images': 0}
    def hook(module, inputs, output):
        n = len(inputs[0])
        counts['forward_calls'] += 1
        counts['forward_images'] += n
        if output.projected_global_embedding.requires_grad:
            def backward(g):
                counts['backward_calls'] += 1
                counts['backward_images'] += n
                return g
            output.projected_global_embedding.register_hook(backward)
    handle = encoder.model.register_forward_hook(hook)
    native_cpu = torch.stack([pil_tensor(access.image(r)) for r in selected])
    labels = [int(r['label']) for r in selected]
    patients = [r['patient_id'] for r in selected]
    with torch.no_grad():
        native = native_cpu.cuda()
        z = torch.stack([encoder(cond.apply(native)).cpu() for cond in cs])
    np.savez(args.output/'public_features.npz', z=z.numpy(), labels=np.array(labels),
             patients=np.array(patients))
    with np.load(args.output/'public_features.npz', allow_pickle=False) as archive:
        saved = torch.from_numpy(archive['z'].copy()).double()
    gs = torch.stack([per_image_gradient(saved[k], labels, h) for k, h in enumerate(hs)])
    stats = patient_class_sums(gs, labels, patients)
    target = stats.balanced()
    # Independent CE loss and autograd, recomputed from persisted features.
    independent = []
    for k, h in enumerate(hs):
        w, b = h.weight.clone().requires_grad_(), h.bias.clone().requires_grad_()
        loss = 0
        for c in (0, 1):
            pids = sorted({p for p, y in zip(patients, labels) if y == c})
            for p in pids:
                ids = [i for i in range(len(labels)) if patients[i] == p and labels[i] == c]
                loss = loss + .5/len(pids)*F.cross_entropy(
                    saved[k, ids] @ w.T+b, torch.tensor([labels[i] for i in ids]))
        gw, gb = torch.autograd.grad(loss, (w, b))
        independent.append(torch.cat((gw.flatten(), gb)))
    target_error = float((target-torch.stack(independent)).abs().max())
    save(args.output/'target_check.json', {'maximum_abs': target_error, 'counts': stats.counts.tolist(),
                                         'gradient_shape': list(target.shape)})
    require(target_error <= 1e-12, 'Independent real signal mismatch')
    np.savez(args.output/'public_target.npz', gradients=target.numpy(), counts=stats.counts.numpy())
    hs_gpu = heads('biovil', 16, device='cuda')
    objective = bind_objective(cs, {'biovil': hs_gpu}, {'biovil': target.cuda()}, labels)
    digest = objective_digest(objective)
    templates = torch.stack([TF.resize(x, [224, 224], antialias=True) for x in native_cpu]).cuda()
    del native, z, saved, gs
    rows = []
    for index, case in enumerate(contract['cases']):
        model = Renderer(templates, 'B', 101)
        model.set_step(500)
        initial = state_hash(model)
        trace_one = PixelTrace(encoder)
        values = one_pass(model, {'biovil': trace_one}, objective, case['microbatch'])
        values['loss'].backward()
        reference = grads(model)
        loss = float(values['loss'].detach())
        del values
        trace_two = PixelTrace(encoder)
        replay = two_pass(model, {'biovil': trace_two}, objective, case['microbatch'])
        err = grad_error(grads(model), reference)
        require(len(trace_one.gradients) == len(trace_two.gradients), 'Pixel replay length')
        require(all(g is not None and bool(torch.isfinite(g).all())
                    for g in trace_one.gradients+trace_two.gradients), 'Invalid image gradient')
        pixel_error = grad_error(trace_two.gradients, trace_one.gradients)
        loss_error = abs(loss-replay['loss'])
        passed = loss_error <= 2e-6 and all(
            e['max_abs'] <= 2e-7+5e-4*e['reference_peak'] for e in (err, pixel_error))
        fixed = state_hash(model) == initial and objective_digest(objective) == digest
        row = dict(case, parameter_gradient=err, pixel_gradient=pixel_error,
                   loss_abs=loss_error, passed=bool(passed and fixed),
                   renderer_target_unchanged=fixed, loss_components=replay)
        save(args.output/f'case_{index}.json', row)
        print(json.dumps(row), flush=True)
        rows.append(row)
        require(passed and fixed, 'Condition signal one/two-pass mismatch')
        del model, reference, trace_one, trace_two
        torch.cuda.empty_cache()
    # Verify signal-only loss reaches image pixels, independent of anchor/TV.
    x = templates.detach().clone().requires_grad_(True)
    probe = bind_objective(cs[:1], {'biovil': hs_gpu[:1]},
                           {'biovil': target[:1].cuda()}, labels)
    feature = encoder(cs[0].apply(x))[None]
    loss, _ = probe.matching({'biovil': feature})
    gradient = torch.autograd.grad(loss, x)[0]
    signal_only = {'loss': float(loss.detach()), 'pixel_gradient_norm': float(gradient.norm()),
                   'finite': bool(torch.isfinite(gradient).all()), 'prior_included': False}
    require(signal_only['finite'] and signal_only['pixel_gradient_norm'] > 0, 'No task-signal pixel gradient')
    handle.remove()
    torch.cuda.synchronize()
    require(counts['forward_images'] == 116 and counts['backward_images'] == 68,
            'Predeclared operation counts differ')
    require(state_hash(encoder) == before, 'Frozen encoder state changed')
    require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()), 'Encoder gradient leak')
    require(source_hashes() == contract['legacy_source_hashes'], 'Legacy code changed')
    require({str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))} == own_hashes,
            'New source changed during check')
    require(all(sha(p) == h for p, h in contract['inputs'].items()), 'Bound input changed')
    summary = {'status': 'PASS_PUBLIC_CONDITION_SIGNAL_IMAGE_CONNECTION_ONLY',
               'rows': rows, 'independent_target_max_abs': target_error,
               'signal_only_probe': signal_only, 'encoder_state_unchanged': True,
               'legacy_code_unchanged': True, 'target_unchanged': objective_digest(objective) == digest,
               'counts': counts, 'access': access.report(), 'optimizer_updates': 0,
               'elapsed_seconds': time.perf_counter()-started,
               'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
               'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
               'artifacts': {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    save(args.output/'verification.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

