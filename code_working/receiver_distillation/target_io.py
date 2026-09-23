"""Condition-specific target handoff; all Q-derived files are NON-DP internal."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from prrd_v3.contracts import OUT, require, sha
from prrd_v3.encoders import BIO_PATH
from prrd_v3.bank_runtime import digest
from .signals import Condition, Head, conditions, heads, bind_objective


def descriptor():
    cs, hs = conditions(), heads('biovil', 16)
    return {
        'schema': 'receiver.fixed-condition-linear-ce/v1',
        'encoder_id': 'biovil', 'conditions': 4, 'dimension': 16, 'public_seed': 101,
        'checkpoint': str(BIO_PATH), 'checkpoint_sha256': sha(BIO_PATH),
        'projection': str(OUT/'source_P_projection.npz'),
        'projection_sha256': sha(OUT/'source_P_projection.npz'),
        'feature_path': 'FP32 bounded source point; public PCA16/q95; no whitening',
        'placement': 'native grayscale affine then source resize512 crop448; synthetic224 same relative affine',
        'gradient_layout': 'vec(W[2,16]) then bias[2]; float64 CE derivative',
        'patient_weight': 'visit mean within patient/class then present-patient mean then 0.5 each class',
        'empty_class': 'zero contribution; do not renormalize remaining class',
        'loss': 'mean over K of cosine distance; equal encoder mean; anchor0.01 TV0.0001 once',
        'tuples': [{'index': c.index, 'theta': c.theta.tolist(), 'brightness': c.brightness.tolist(),
                    'weight': h.weight.tolist(), 'bias': h.bias.tolist()} for c, h in zip(cs, hs)],
    }


def reconstruct(rule, device):
    require(rule['schema'] == 'receiver.fixed-condition-linear-ce/v1', 'Unsupported signal rule')
    cs, hs = [], []
    for i, t in enumerate(rule['tuples']):
        require(t['index'] == i, 'Condition order changed')
        cs.append(Condition(i, torch.tensor(t['theta'], dtype=torch.float32),
                            torch.tensor(t['brightness'], dtype=torch.float32)))
        hs.append(Head(torch.tensor(t['weight'], dtype=torch.float64, device=device),
                       torch.tensor(t['bias'], dtype=torch.float64, device=device)))
    return tuple(cs), tuple(hs)


def load_target(handoff, labels, *, population='pooled', device='cpu'):
    require(population in ('P', 'Q', 'pooled'), 'Invalid target population')
    if isinstance(handoff, (str, Path)):
        handoff = json.loads(Path(handoff).read_text(encoding='utf-8'))
    require(handoff['schema'] == 'receiver.target-handoff/v1', 'Wrong handoff type')
    target_file, rule_file = Path(handoff['target_file']), Path(handoff['rule_file'])
    require(sha(target_file) == handoff['target_sha256'], 'Condition target bytes changed')
    require(sha(rule_file) == handoff['rule_file_sha256'], 'Condition rule bytes changed')
    rule = json.loads(rule_file.read_text(encoding='utf-8'))
    require(digest(rule) == handoff['rule_sha256'], 'Rule content changed')
    require(digest(rule) == digest(descriptor()), 'Current head/transform/preprocess binding differs')
    with np.load(target_file, allow_pickle=False) as d:
        require(str(d['schema']) == 'receiver.separate-condition-target/v1', 'Not a condition target')
        require(str(d['rule_sha256']) == digest(rule), 'Target/condition binding mismatch')
        value = d[population+'_gradient'].copy()
        require(value.shape == (4, 34) and np.isfinite(value).all(), 'Condition target shape/values')
    cs, hs = reconstruct(rule, device)
    objective = bind_objective(cs, {'biovil': hs},
                               {'biovil': torch.as_tensor(value, device=device)}, labels)
    return objective, rule

