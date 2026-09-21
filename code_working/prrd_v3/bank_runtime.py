"""Bank-level execution, immutable checkpoints and verified final PNG artifacts.

No data/model loading and no authorization is granted by this module.
The CLI binds an explicit main authorization or a bounded public technical test.
"""
from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
from PIL import Image
import torch

from .contracts import require, sha
from .datasets import png_roundtrip
from .encoders import state_hash
from .guarded_objectives import two_pass
from .render import LEVELS, ACTIVATE


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def frozen_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: frozen_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [frozen_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(frozen_tree(v) for v in value)
    return value


def tree_digest(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update(canonical(['tensor', str(a.dtype), list(a.shape)]))
            h.update(a.numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in sorted(v, key=repr):
                add(key); add(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v:
                add(item)
        else:
            h.update(canonical([type(v).__name__, v]))
    add(value)
    return h.hexdigest()


def rng_state():
    n = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])),
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
    }


def restore_rng(value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value['torch_cpu'])
    require(len(value['torch_cuda']) == (torch.cuda.device_count()
            if torch.cuda.is_available() else 0), 'CUDA RNG device count changed')
    if value['torch_cuda']:
        torch.cuda.set_rng_state_all(value['torch_cuda'])


@contextmanager
def file_lock(path):
    """OS lock is released on process exit, including an unclean exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b'\0'); stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        stream.close()
        raise RuntimeError('Another process owns this bank or authorization') from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def atomic_bytes(path, data, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if replace:
        os.replace(tmp, path)
    else:
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(tmp, path)
        tmp.unlink()


def atomic_json(path, value, *, replace=False):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8') + b'\n'
    atomic_bytes(path, data, replace=replace)


def append_event(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def capture(renderer, optimizer, completed, signature):
    return {
        'schema': 'prrd.bank-checkpoint/v1', 'signature': signature,
        'completed_updates': completed, 'next_step': completed + 1,
        'active_levels': renderer.active,
        'requires_grad': [p.requires_grad for p in renderer.parameters()],
        'renderer': frozen_tree(renderer.state_dict()),
        'optimizer': frozen_tree(optimizer.state_dict()),
        'rng': rng_state(),
    }


def save_checkpoint(directory, renderer, optimizer, completed, signature):
    directory = Path(directory)
    payload = capture(renderer, optimizer, completed, signature)
    cpdir = directory / 'checkpoints'
    cpdir.mkdir(exist_ok=True)
    name = 'step_%06d_%s.pt' % (completed, uuid.uuid4().hex)
    tmp = cpdir / ('.' + name + '.tmp')
    with tmp.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    target = cpdir / name
    os.link(tmp, target); tmp.unlink()
    record = {'path': 'checkpoints/' + name, 'sha256': sha(target),
              'completed_updates': completed, 'next_step': completed + 1,
              'payload_digest': tree_digest(payload), 'signature': signature}
    atomic_json(cpdir / (name + '.json'), record)
    # The latest pointer is the commit point; orphaned snapshots remain evidence.
    atomic_json(directory / 'latest_checkpoint.json', record, replace=True)
    return record


def checkpoint_record(directory):
    d = Path(directory)
    record = json.loads((d / 'latest_checkpoint.json').read_text(encoding='utf-8'))
    rel = Path(record['path'])
    require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == 'checkpoints',
            'Invalid checkpoint location')
    p = d / rel
    require(sha(p) == record['sha256'], 'Checkpoint bytes changed')
    side = json.loads(p.with_name(p.name + '.json').read_text(encoding='utf-8'))
    require(side == record, 'Checkpoint pointer and commit receipt differ')
    return record, p


def load_checkpoint(directory, renderer, optimizer, signature):
    record, path = checkpoint_record(directory)
    require(record['signature'] == signature, 'Checkpoint is bound to another run')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['schema'] == 'prrd.bank-checkpoint/v1' and
            payload['signature'] == signature, 'Checkpoint contract mismatch')
    require(tree_digest(payload) == record['payload_digest'], 'Checkpoint payload changed')
    done = payload['completed_updates']
    require(isinstance(done, int) and done >= 0 and
            payload['next_step'] == done + 1 == record['next_step'] and
            done == record['completed_updates'], 'Invalid checkpoint step')
    renderer.load_state_dict(payload['renderer'], strict=True)
    renderer.set_step(max(1, done))
    require(renderer.active == payload['active_levels'] and
            [p.requires_grad for p in renderer.parameters()] == payload['requires_grad'],
            'Pyramid state is inconsistent with successful updates')
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    restored = capture(renderer, optimizer, done, signature)
    require(tree_digest(restored) == tree_digest(payload),
            'Restored parameter/optimizer/RNG/step/activation state is not exact')
    return done, {'checkpoint_sha256': record['sha256'],
                  'restored_state_digest': tree_digest(restored),
                  'restored_exact': True, 'completed_updates': done,
                  'next_step': done + 1, 'active_levels': renderer.active}


def image_layout(spec):
    n = spec['images']; q = n // 4
    relation = spec['arm'] not in ('A', 'B')
    rows, pairs = [], []
    for i in range(n):
        pair = relation and (q <= i < 2*q or 3*q <= i < 4*q)
        rows.append({'synthetic_id': 's%03d' % i, 'label': int(i >= n//2),
                     'bank_type': 'relation' if pair else 'marginal',
                     'file': 'images/s%03d.png' % i})
    if relation:
        for i in range(q):
            pairs.append({'virtual_pair_id': 'v%03d' % i,
                          'positive_id': 's%03d' % (3*q+i),
                          'negative_id': 's%03d' % (q+i)})
    return rows, pairs


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    w = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    w.writeheader(); w.writerows(rows)
    return stream.getvalue().encode('utf-8')


def verify_artifact(directory, spec, signature, expected_pixels=None):
    d = Path(directory)
    seal = json.loads((d / 'artifact_seal.json').read_text(encoding='utf-8'))
    require(seal['signature'] == signature and seal['bank_id'] == spec['id'],
            'Artifact belongs to another bank')
    require(seal['successful_updates'] == spec['updates'] and
            seal['artifact_kind'] == spec['artifact_kind'], 'Intermediate output is not final')
    require(set(seal['files_sha256']) ==
            {p.relative_to(d).as_posix() for p in d.rglob('*')
             if p.is_file() and p.name != 'artifact_seal.json'}, 'Artifact file set changed')
    for rel, h in seal['files_sha256'].items():
        p = Path(rel)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe artifact path')
        require(sha(d / p) == h, 'Exported file hash mismatch')
    with (d / 'images.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    with (d / 'pairs.csv').open(encoding='utf-8', newline='') as f:
        pairs = list(csv.DictReader(f))
    expected_rows, expected_pairs = image_layout(spec)
    require(len(rows) == spec['images'] and pairs == expected_pairs, 'Wrong image/pair count or binding')
    exact = 0
    for i, (row, want) in enumerate(zip(rows, expected_rows)):
        require({k: str(row[k]) for k in want} == {k: str(v) for k, v in want.items()},
                'Image label, order, type or pair binding changed')
        require(row['png_sha256'] == sha(d / want['file']), 'PNG hash differs from manifest')
        with Image.open(d / want['file']) as im:
            require(im.mode == 'L' and im.size == (224, 224), 'Wrong PNG mode or geometry')
            actual = np.asarray(im).copy()
        require(actual.dtype == np.uint8, 'PNG is not uint8')
        if expected_pixels is not None:
            require(np.array_equal(actual, expected_pixels[i]), 'Float-to-PNG pixels differ')
            exact += 1
    return {'status': 'PASS_FINAL_PNG_LABEL_PAIR_HASH', 'images': len(rows),
            'pairs': len(pairs), 'expected_pixel_arrays_exact': exact,
            'artifact_kind': spec['artifact_kind'], 'files_verified': len(seal['files_sha256'])}


def export_artifact(directory, renderer, spec, signature, checkpoint):
    require(checkpoint['completed_updates'] == spec['updates'], 'Cannot export intermediate checkpoint')
    final = Path(directory) / 'artifact'
    with torch.no_grad():
        image = torch.cat([renderer(list(range(i, min(i+spec['microbatch'], spec['images'])))).cpu()
                           for i in range(0, spec['images'], spec['microbatch'])])
    require(tuple(image.shape) == (spec['images'], 1, 224, 224) and
            bool(torch.isfinite(image).all()), 'Invalid final image tensor')
    # Independent expected rounding, not the png_roundtrip helper's returned array.
    expected = np.rint(np.clip(image[:, 0].numpy(), 0, 1) * 255).astype(np.uint8)
    if final.exists():
        # Recover a crash after publishing the complete artifact but before COMPLETED.
        report = verify_artifact(final, spec, signature, expected)
        seal = json.loads((final / 'artifact_seal.json').read_text(encoding='utf-8'))
        require(seal['checkpoint_sha256'] == checkpoint['sha256'], 'Artifact/checkpoint mismatch')
        return report
    staging = Path(directory) / ('.export_' + uuid.uuid4().hex)
    staging.mkdir(); (staging / 'images').mkdir()
    rows, pairs = image_layout(spec)
    for i, row in enumerate(rows):
        _, data = png_roundtrip(image[i, 0].numpy())
        atomic_bytes(staging / row['file'], data)
        row['png_sha256'] = sha(staging / row['file'])
    atomic_bytes(staging / 'images.csv', csv_bytes(rows, list(rows[0])))
    atomic_bytes(staging / 'pairs.csv', csv_bytes(pairs,
                 ['virtual_pair_id', 'positive_id', 'negative_id']))
    atomic_json(staging / 'learning_contract.json', {
        'schema': 'prrd.synthetic-learning/v1', 'arm': spec['arm'],
        'learner_policy': spec['learner_policy'], 'eta_during_synthesis': spec['eta'],
        'point_bank_only_for_point_risk': True, 'pair_bank_only_for_relation_risk': True,
        'recipient_uses_own_public_transform_and_relation_scale': True,
        'output': 'uint8 grayscale224 PNG; image labels are synthetic targets',
        'artifact_kind': spec['artifact_kind'], 'not_for_main_study': spec['artifact_kind'] != 'MAIN_BANK',
        'source_representation_contract_sha256': spec['contract_sha256'],
    })
    atomic_json(staging / 'provenance_public.json', {
        'schema': 'prrd.synthetic-provenance/v1', 'bank_id': spec['id'],
        'contract_sha256': spec['contract_sha256'], 'implementation_sha256': spec['implementation_sha256'],
        'initialization': 'PUBLIC_P_ONLY', 'seed': spec['seed'],
        'successful_updates': spec['updates'], 'privacy_receipt': None,
        'DP_applied': False, 'artifact_kind': spec['artifact_kind'],
        'main_reuse_allowed': spec['artifact_kind'] == 'MAIN_BANK',
    })
    files = {p.relative_to(staging).as_posix(): sha(p)
             for p in sorted(staging.rglob('*')) if p.is_file()}
    atomic_json(staging / 'artifact_seal.json', {
        'signature': signature, 'bank_id': spec['id'], 'successful_updates': spec['updates'],
        'artifact_kind': spec['artifact_kind'], 'checkpoint_sha256': checkpoint['sha256'],
        'files_sha256': files,
    })
    report = verify_artifact(staging, spec, signature, expected)
    require(not final.exists(), 'Final artifact overwrite prohibited')
    os.rename(staging, final)
    return report


def objective_target_state(objective):
    state = {'moments': dict(objective.moments),
             'M': objective.functional_target.matrix,
             'w': objective.functional_target.weight}
    if objective.aug_moments is not None:
        state['aug_moments'] = dict(objective.aug_moments)
    return state


def check_augmented_target_binding(objective, spec):
    """Bind content as well as provenance before checkpoint creation/replay."""
    augmented = getattr(objective, 'aug_moments', None)
    keys = ('aug_target_sha256', 'aug_target_rule_sha256', 'aug_target_digest')
    if augmented is None:
        require(not any(k in spec for k in keys), 'Augmented target binding without target')
        return
    require(objective.arm in ('A', 'B'), 'Augmented target requires a pointwise arm')
    require(all(isinstance(spec.get(k), str) and len(spec[k]) == 64
                and all(c in '0123456789abcdef' for c in spec[k]) for k in keys),
            'Missing or invalid augmented target binding')
    require(tree_digest(dict(augmented)) == spec['aug_target_digest'],
            'Augmented target content differs from run binding')


def run_bank(directory, renderer, encoder, objective, optimizer, spec, *,
             resume=False, stop_after=None, deadline=None, on_step=None):
    """Execute one immutable bank. Loader and authorization remain outside."""
    directory = Path(directory)
    require(spec['images'] == renderer.n and spec['updates'] > 0 and
            spec['checkpoint_every'] == 25, 'Invalid bank dimensions or checkpoint interval')
    require(spec['arm'] == renderer.arm == objective.arm, 'Arm mismatch')
    require(type(spec['microbatch']) is int and spec['microbatch'] in (4, 8, 16)
            and spec['seed'] in (101, 202, 303), 'Bound runtime differs')
    require(spec['artifact_kind'] in ('MAIN_BANK', 'TECHNICAL_TEST_ONLY'), 'Unknown artifact role')
    if spec['artifact_kind'] == 'MAIN_BANK':
        require(spec['images'] == 128 and spec['updates'] == 500, 'Main recipe changed')
    else:
        require(spec['images'] <= 8 and spec['updates'] <= 84, 'Technical check expanded')
    stop = spec['updates'] if stop_after is None else stop_after
    require(0 <= stop <= spec['updates'], 'Invalid stop boundary')
    check_augmented_target_binding(objective, spec)
    signature = digest(spec)
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    else:
        require(directory.is_dir(), 'No bank to resume')
    with file_lock(directory / '.process.lock'):
        require(not (directory / 'COMPLETED.json').exists(), 'Completed bank is sealed; overwrite/resume prohibited')
        if resume:
            saved = json.loads((directory / 'run_spec.json').read_text(encoding='utf-8'))
            require(saved == spec, 'Resume code/input/recipe binding changed')
            completed, restoration = load_checkpoint(directory, renderer, optimizer, signature)
        else:
            renderer.set_step(1)
            atomic_json(directory / 'run_spec.json', spec)
            completed = 0; restoration = None
            save_checkpoint(directory, renderer, optimizer, 0, signature)
        require(completed <= stop, 'Requested stop precedes saved state')
        attempt = uuid.uuid4().hex
        start = time.monotonic(); last_duration = 0.; stopped_for_budget = False
        source_before = state_hash(encoder)
        target_before = tree_digest(objective_target_state(objective))
        prior_completed = completed
        while completed < stop:
            if deadline is not None and time.monotonic() + max(15., last_duration*1.25) >= deadline:
                stopped_for_budget = True; break
            step = completed + 1
            renderer.set_step(step)
            tick = time.monotonic()
            trace = two_pass(renderer, encoder, objective, step, spec['seed'], spec['microbatch'])
            require(all(math.isfinite(v) for v in trace.values()), 'Nonfinite loss/readout')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in renderer.parameters()),
                    'Nonfinite parameter after optimizer update')
            for state in optimizer.state.values():
                require(all(not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all())
                            for v in state.values()), 'Nonfinite optimizer state')
            completed = step
            last_duration = time.monotonic() - tick
            append_event(directory / 'updates.jsonl', {
                'attempt': attempt, 'pid': os.getpid(), 'successful_step': step,
                'active_levels': renderer.active, 'trace': trace,
            })
            if on_step is not None:
                on_step(completed, trace)
            if completed % spec['checkpoint_every'] == 0:
                save_checkpoint(directory, renderer, optimizer, completed, signature)
        latest, _ = checkpoint_record(directory)
        if latest['completed_updates'] != completed:
            latest = save_checkpoint(directory, renderer, optimizer, completed, signature)
        require(state_hash(encoder) == source_before, 'Fixed encoder weights/buffers/projections changed')
        require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()),
                'Gradient leaked into fixed encoder')
        require(tree_digest(objective_target_state(objective)) == target_before,
                'Fixed target changed')
        final = completed == spec['updates']
        png = None
        if final:
            png = export_artifact(directory, renderer, spec, signature, latest)
            atomic_json(directory / 'COMPLETED.json', {
                'status': 'COMPLETE_MAIN_BANK' if spec['artifact_kind'] == 'MAIN_BANK'
                          else 'COMPLETE_TECHNICAL_TEST_ONLY',
                'bank_id': spec['id'], 'successful_updates': completed,
                'signature': signature, 'checkpoint_sha256': latest['sha256'],
                'artifact_seal_sha256': sha(directory / 'artifact/artifact_seal.json'),
                'not_for_main_study': spec['artifact_kind'] != 'MAIN_BANK',
            })
        return {'status': 'COMPLETED' if final else 'PAUSED',
                'artifact_kind': spec['artifact_kind'], 'bank_id': spec['id'],
                'pid': os.getpid(), 'updates_this_process': completed-prior_completed,
                'completed_updates': completed, 'next_step': completed+1,
                'active_levels': renderer.active,
                'requires_grad': [p.requires_grad for p in renderer.parameters()],
                'restoration': restoration, 'checkpoint_sha256': latest['sha256'],
                'source_state_sha256': source_before, 'source_unchanged': True,
                'target_unchanged': True, 'PNG': png,
                'stopped_for_budget': stopped_for_budget,
                'process_bank_seconds': time.monotonic()-start}
