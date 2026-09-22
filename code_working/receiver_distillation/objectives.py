"""Global condition-specific matching with exact microbatch replay.

Uses the existing renderer/anchor/TV. No synthetic optimizer step is performed.
"""
from __future__ import annotations
import torch
from prrd_v3.contracts import require
from prrd_v3.objectives import regularization


def batches(n, microbatch):
    require(microbatch > 0, 'Invalid microbatch')
    return [list(range(i, min(i + microbatch, n))) for i in range(0, n, microbatch)]


def collect(renderer, encoders, objective, microbatch):
    ids_list = batches(renderer.n, microbatch)
    result = {}
    for name in sorted(encoders):
        result[name] = torch.stack([
            torch.cat([encoders[name](condition.apply(renderer(ids))) for ids in ids_list])
            for condition in objective.conditions])
    return result


def prior(renderer, microbatch):
    return sum(regularization(renderer(ids), renderer.templates[ids]) * len(ids) / renderer.n
               for ids in batches(renderer.n, microbatch))


def one_pass(renderer, encoders, objective, microbatch):
    renderer.zero_grad(set_to_none=True)
    features = collect(renderer, encoders, objective, microbatch)
    match, blocks = objective.matching(features)
    reg = prior(renderer, microbatch)
    return {'loss': match + reg, 'matching': match, 'prior': reg, 'blocks': blocks}


def two_pass(renderer, encoders, objective, microbatch):
    renderer.zero_grad(set_to_none=True)
    with torch.no_grad():
        features = collect(renderer, encoders, objective, microbatch)
        reg = prior(renderer, microbatch)
    leaves = {name: z.detach().requires_grad_(True) for name, z in features.items()}
    match, blocks = objective.matching(leaves)
    names = sorted(leaves)
    derivatives = torch.autograd.grad(match, tuple(leaves[n] for n in names))
    # Replay exactly the same renderer partition and condition, with global
    # feature derivatives. Never recompute a per-microbatch class denominator.
    for name, upstream in zip(names, derivatives):
        for k, condition in enumerate(objective.conditions):
            for ids in batches(renderer.n, microbatch):
                z = encoders[name](condition.apply(renderer(ids)))
                (z * upstream[k, ids]).sum().backward()
    # Anchor/TV are counted once, not once per encoder/condition.
    for ids in batches(renderer.n, microbatch):
        (regularization(renderer(ids), renderer.templates[ids]) * len(ids) / renderer.n).backward()
    return {'loss': float((match.detach() + reg).cpu()),
            'matching': float(match.detach().cpu()), 'prior': float(reg.cpu()),
            'blocks': {k: float(v.detach().cpu()) for k, v in blocks.items()}}

