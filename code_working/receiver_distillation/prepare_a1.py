"""P/Q extraction only; keep four conditions separate and verify saved signals."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from prrd_v3.contracts import CODE, OUT, RESEARCH, now, read, require, save, sha, setup, source_hashes
from prrd_v3.datasets import manifests, PixelAccess, TRAIN, DEV
from prrd_v3.encoders import Encoder, BIO_PATH, pil_tensor, state_hash
from prrd_v3.bank_runtime import digest
from .signals import conditions, heads, per_image_gradient, patient_class_sums, pool_patient_sums
from .target_io import descriptor, load_target


def independent(saved_paths, target_path, rule):
    """NumPy CE algebra and explicit patient weights, not production aggregation."""
    hs = [(np.asarray(t['weight']), np.asarray(t['bias'])) for t in rule['tuples']]
    populations = {}
    max_error = 0.0
    metadata_ok = True
    expected_groups = manifests()
    for role, path in saved_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            z, ids, labels, images = (archive[k].copy() for k in ('z','patient_ids','labels','image_ids'))
            require(np.array_equal(archive['condition_ids'], np.arange(4)), 'Lost condition axes')
        group = expected_groups[role]
        require(np.array_equal(images, [r['image_id'] for r in group]), 'Image order changed')
        require(np.array_equal(ids, [r['patient_id'] for r in group]), 'Patient order changed')
        require(np.array_equal(labels, [int(r['label']) for r in group]), 'Label order changed')
        sums = np.zeros((4, 2, 34), dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        for c in (0, 1):
            people = sorted(set(ids[labels == c]))
            counts[c] = len(people)
            terms = [[] for _ in range(4)]
            for pid in people:
                idx = np.flatnonzero((ids == pid) & (labels == c))
                for k, (w, b) in enumerate(hs):
                    x = z[k, idx].astype(np.float64)
                    logits = x @ w.T+b
                    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
                    delta = exp/exp.sum(axis=1, keepdims=True)
                    delta[:, c] -= 1
                    weight = np.einsum('ni,nj->nij', delta, x).reshape(len(idx), -1)
                    terms[k].append(np.concatenate([weight, delta], axis=1).mean(0))
            for k in range(4):
                sums[k, c] = [math.fsum(float(v[j]) for v in terms[k]) for j in range(34)]
        populations[role] = {'sums': sums, 'counts': counts}
    populations['pooled'] = {
        'sums': populations['P']['sums']+populations['Q']['sums'],
        'counts': populations['P']['counts']+populations['Q']['counts']}
    with np.load(target_path, allow_pickle=False) as actual:
        for role, p in populations.items():
            g = .5*(p['sums']/np.maximum(p['counts'], 1)[None,:,None]).sum(1)
            require(np.array_equal(actual[role+'_counts'], p['counts']), 'Class patient count changed')
            for key, expected in (('sums', p['sums']), ('gradient', g)):
                max_error = max(max_error, float(np.max(abs(actual[role+'_'+key]-expected))))
    require(max_error <= 1e-12, 'Independent target error exceeds fixed tolerance')
    return {'status': 'PASS_STORED_CONDITION_SIGNAL_AND_PATIENT_WEIGHTS',
            'maximum_abs': max_error, 'criteria_atol': 1e-12, 'metadata_order_exact': metadata_ok,
            'counts': {r: p['counts'].tolist() for r,p in populations.items()},
            'conditions_separate': 4, 'independent_encoder_rerun': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), 'Do not overwrite a target preparation attempt')
    args.output.mkdir(parents=True)
    private = args.output/'private'
    private.mkdir()
    started = time.perf_counter()
    groups = manifests()
    rule = descriptor()
    legacy_before = source_hashes()
    own_files = [Path(__file__), Path(__file__).with_name('signals.py'), Path(__file__).with_name('target_io.py')]
    bindings = {str(p):sha(p) for p in own_files+[BIO_PATH, OUT/'source_P_projection.npz', TRAIN, DEV,
                RESEARCH/'RECEIVER_A1_TARGET_RUNTIME_CONTRACT_20260922.md']}
    old = read(CODE/'_reports/prrd_b_aug_targets_20260921_v1/contract.json')
    for name, h in old['bindings'].items():
        if name.endswith('.npz') or name.endswith('prrd_source_readout_20260921_v1/result.json'):
            require(sha(name) == h, 'Existing clean reference changed')
            bindings[name] = h
    contract = {
        'schema': 'receiver.A1-target-preparation/v1', 'created': now(),
        'scope': 'P_Q_CONDITION_TARGETS_ONLY_NONDP', 'rule': rule, 'rule_sha256': digest(rule),
        'bindings': bindings, 'legacy_sources': legacy_before,
        'microbatch_images': 16, 'K': 4, 'expected_forward_images': 23640,
        'expected_backward_images': 0, 'pixel_roles': ['P','Q'],
        'independent_target_atol': 1e-12, 'private_artifacts': True,
        'V_recipient_DP_final': False, 'optimizer_updates': 0,
    }
    save(args.output/'contract.json', contract)
    save(private/'condition_rule.json', rule)
    snapshot = args.output/'implementation_snapshot'
    snapshot.mkdir()
    for p in own_files:
        shutil.copy2(p, snapshot/p.name)
    setup()
    access = PixelAccess(groups, ('P','Q')).install()
    encoder = Encoder('source').use_projection(OUT/'source_P_projection.npz')
    encoder_before = state_hash(encoder)
    torch.cuda.reset_peak_memory_stats()
    counts = {'forward_images': 0, 'backward_images': 0}
    def count_forward(module, inputs, output):
        counts['forward_images'] += len(inputs[0])
        require(not output.projected_global_embedding.requires_grad, 'Target extraction has image graph')
    hook = encoder.model.register_forward_hook(count_forward)
    cs, hs = conditions(), heads('biovil', 16)
    saved_paths, stats, times = {}, {}, {}
    for role in ('P', 'Q'):
        tick = time.perf_counter()
        group = groups[role]
        require(all(int(r['width']) == int(r['height']) == 1024 for r in group), 'Native geometry changed')
        feature = np.empty((4, len(group), 16), dtype=np.float32)
        last_print = -1
        for start in range(0, len(group), 16):
            rows = group[start:start+16]
            native = torch.stack([pil_tensor(access.image(r)) for r in rows]).cuda()
            with torch.no_grad():
                for k, cond in enumerate(cs):
                    z = encoder(cond.apply(native))
                    require(bool(torch.isfinite(z).all()) and float(z.norm(dim=-1).max()) <= 1+1e-6,
                            'Invalid bounded point features')
                    feature[k, start:start+len(rows)] = z.cpu().numpy()
            del native, z
            bucket = (start+len(rows))//256
            if bucket != last_print or start+len(rows) == len(group):
                print(json.dumps({'phase':'condition_extract','role':role,
                                  'images_done':start+len(rows),'images_total':len(group),
                                  'seconds':time.perf_counter()-tick}), flush=True)
                last_print = bucket
        path = private/(role+'_condition_features.npz')
        np.savez(path, z=feature, condition_ids=np.arange(4),
                 image_ids=np.asarray([r['image_id'] for r in group]),
                 patient_ids=np.asarray([r['patient_id'] for r in group]),
                 labels=np.asarray([int(r['label']) for r in group]), roles=np.full(len(group), role),
                 rule_sha256=np.array(digest(rule)))
        saved_paths[role] = path
        grad = torch.stack([per_image_gradient(torch.from_numpy(feature[k]), [int(r['label']) for r in group], h)
                            for k, h in enumerate(hs)])
        stats[role] = patient_class_sums(grad, [int(r['label']) for r in group],
                                        [r['patient_id'] for r in group])
        times[role] = time.perf_counter()-tick
        del feature, grad
    stats['pooled'] = pool_patient_sums(stats['P'], stats['Q'])
    arrays = {'schema':np.array('receiver.separate-condition-target/v1'),
              'rule_sha256':np.array(digest(rule)), 'condition_ids':np.arange(4)}
    for name, s in stats.items():
        arrays.update({name+'_sums':s.sums.numpy(), name+'_counts':s.counts.numpy(),
                       name+'_gradient':s.balanced().numpy()})
    target_file = private/'A1_condition_targets.npz'
    np.savez(target_file, **arrays)
    hook.remove()
    require(counts == {'forward_images':23640, 'backward_images':0}, 'Extraction count mismatch')
    require(state_hash(encoder) == encoder_before, 'Frozen source changed')
    del encoder
    torch.cuda.empty_cache()
    verification = independent(saved_paths, target_file, rule)
    save(args.output/'independent_verification.json', verification)
    handoff = {
        'schema':'receiver.target-handoff/v1',
        'target_file':str(target_file), 'target_sha256':sha(target_file),
        'rule_file':str(private/'condition_rule.json'), 'rule_file_sha256':sha(private/'condition_rule.json'),
        'rule_sha256':digest(rule), 'contract_file':str(args.output/'contract.json'),
        'contract_sha256':sha(args.output/'contract.json'),
        'population':'pooled', 'privacy':'NONDP_INTERNAL_Q_DERIVED',
        'not_an_execution_authorization':True,
    }
    save(private/'target_handoff.json', handoff)
    obj, _ = load_target(handoff, (0,)*64+(1,)*64)
    require(torch.equal(obj.targets['biovil'], stats['pooled'].balanced()), 'Runtime loader differs')
    require(source_hashes() == legacy_before, 'Legacy production code changed')
    require(all(sha(p) == h for p,h in bindings.items()), 'Bound input/code changed during preparation')
    result = {
        'status':'PASS_A1_P_Q_CONDITION_TARGETS_READY',
        'images':5910, 'patients':2699, 'conditions':4, 'gradient_shape':[4,34],
        'per_role_seconds':times, 'elapsed_seconds':time.perf_counter()-started,
        'counts':counts, 'access':access.report(), 'verification':verification,
        'loader_exact':True, 'source_fixed':True, 'legacy_and_clean_inputs_unchanged':True,
        'target_sha256':sha(target_file), 'handoff_sha256':sha(private/'target_handoff.json'),
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'stored_bytes':sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
        'free_bytes':shutil.disk_usage(CODE).free, 'optimizer_updates':0, 'V_recipient_DP_final':False,
    }
    save(args.output/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
