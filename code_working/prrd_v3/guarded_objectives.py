"""Stage3 dual-path statistics PLUS fixed-target functional matching.

The historical objectives.py remains available for the old W1 reproduction.
This module has no data access, optimizer step, recipient selection or DP code.
All moments use full-bank denominators; augmentation receives statistics only.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
import torch
from .contracts import require
from .guarded_readout import readout_from_moments, freeze_functional_target, functional_loss
from .objectives import regularization
from .relation_paths import RELATION_ARMS, paired_relation
from .render import augment, augment_parameters

ARMS = ('A', 'B') + RELATION_ARMS


@dataclass(frozen=True)
class LearnerPolicy:
    beta: float
    ridge: float
    rho: float | None = None
    kappa: float | None = None
    guard: bool = True

    def solve(self, moments):
        return readout_from_moments(moments, beta=self.beta, ridge=self.ridge,
                                    rho=self.rho, kappa=self.kappa, guard=self.guard)


def bank_moments(paths, arm, scales):
    """Point slots and virtual pair slots follow the existing two-bank layout.

    A/B use all images for point matching. Relation arms use only marginal
    slots for point moments, and only paired slots for the auxiliary risk.
    D's shuffled PRIVATE target is formed upstream; synthetic pairs are fixed.
    """
    require(arm in ARMS, 'Unknown guarded objective arm')
    z, a = paths['z'].double(), paths['a'].double()
    require(z.ndim == 2 and z.shape == a.shape and len(z) > 0 and len(z) % 4 == 0,
            'Expected balanced point/pair feature blocks in common coordinates')
    q = len(z) // 4
    neg, pos = (z[:q], z[2*q:3*q]) if arm in RELATION_ARMS else (z[:2*q], z[2*q:])
    result = {'m': torch.stack([neg.mean(0), pos.mean(0)]),
              'A': torch.stack([neg.T @ neg / len(neg), pos.T @ pos / len(pos)])}
    if arm in RELATION_ARMS:
        rows = paired_relation(z[3*q:], z[q:2*q], a[3*q:], a[q:2*q], arm, scales)
        r = rows['r']
        result.update(delta=r.mean(0), C=r.T @ r / len(r))
        if arm == 'R_joint':
            u = rows['u']
            result.update(u=u.mean(0), U=u.T @ u / len(u))
    return result


def statistical_loss(moments, target, arm):
    loss = ((moments['m']-target['m']).square().sum()
            + (moments['A']-target['A']).square().sum()) / 4
    if arm in RELATION_ARMS:
        mean, second = ('u', 'U') if arm == 'R_joint' else ('delta', 'C')
        loss = loss + ((moments[mean]-target[mean]).square().sum()
                       + (moments[second]-target[second]).square().sum()) / 2
    return loss


@dataclass(frozen=True)
class BoundObjective:
    arm: str
    scales: object
    policy: LearnerPolicy
    eta: float
    moments: object
    target_readout: object
    functional_target: object


def bind_objective(target, arm, scales, policy, *, eta, device='cpu'):
    """Make detached COPIES once. Callers must not mutate this bound target.

    Target and synthetic learners use the SAME policy but their OWN moments.
    M_T is fixed for the functional metric; it is never replaced with M_S.
    """
    require(arm in ARMS and math.isfinite(float(eta)) and eta >= 0,
            'Invalid arm or functional coefficient')
    scales = {k: float(v) for k, v in scales.items()}
    require(all(math.isfinite(v) and v > 0 for v in scales.values()), 'Invalid public scales')
    keys = ['m', 'A']
    if arm in RELATION_ARMS: keys += ['delta', 'C']
    if arm == 'R_joint': keys += ['u', 'U']
    moments = {k: torch.as_tensor(target[k], dtype=torch.float64, device=device).detach().clone()
               for k in keys}
    require(all(bool(torch.isfinite(v).all()) for v in moments.values()), 'Nonfinite target')
    if arm == 'R_joint':
        d = moments['m'].shape[1]
        eye = torch.eye(d, dtype=torch.float64, device=device)
        linear = torch.cat([eye, -eye], dim=1) / math.sqrt(2)
        require(torch.allclose(moments['delta'], linear @ moments['u'], atol=1e-12, rtol=1e-10)
                and torch.allclose(moments['C'], linear @ moments['U'] @ linear.T,
                                   atol=1e-12, rtol=1e-10),
                'Joint target readout must derive from ALREADY-bounded endpoint moments')
    with torch.no_grad():
        readout = policy.solve(moments)
        fixed = freeze_functional_target(readout)
    return BoundObjective(arm, MappingProxyType(scales), policy, float(eta),
                          MappingProxyType(moments), readout, fixed)


def feature_objective(clean, augmented, objective):
    """Functional matching occurs ONCE, on the unaugmented synthetic learner."""
    moments = bank_moments(clean, objective.arm, objective.scales)
    aug_moments = bank_moments(augmented, objective.arm, objective.scales)
    readout = objective.policy.solve(moments)
    stats = statistical_loss(moments, objective.moments, objective.arm)
    aug_stats = statistical_loss(aug_moments, objective.moments, objective.arm)
    func = functional_loss(readout.weight, objective.functional_target)
    return {'loss': stats + .1 * aug_stats + objective.eta * func,
            'statistics': stats, 'augmentation_matching': aug_stats,
            'functional': func, 'readout': readout}


def _runtime(renderer, objective, microbatch):
    require(renderer.arm == objective.arm, 'Renderer/target arm mismatch')
    require(renderer.relation == (objective.arm in RELATION_ARMS), 'Pair renderer mismatch')
    require(isinstance(microbatch, int) and microbatch > 0, 'Invalid microbatch')


def _cat(parts):
    return {k: torch.cat([p[k] for p in parts]) for k in ('z', 'a')}


def one_pass(renderer, encoder, objective, step, bank_seed, microbatch):
    """Retain graph, partitioning renderer/augmentation/encoder like replay."""
    _runtime(renderer, objective, microbatch)
    n = renderer.n
    params = augment_parameters(n, bank_seed, step, renderer.relation, renderer.base.device)
    clean, augmented, reg = [], [], 0.
    for start in range(0, n, microbatch):
        ids = list(range(start, min(n, start+microbatch)))
        x = renderer(ids)
        clean.append(encoder.forward_paths(x))
        augmented.append(encoder.forward_paths(augment(x, params, ids)))
        reg = reg + regularization(x, renderer.templates[ids]) * len(ids) / n
    result = feature_objective(_cat(clean), _cat(augmented), objective)
    result['regularization'] = reg
    result['loss'] = result['loss'] + reg
    return result


def two_pass(renderer, encoder, objective, step, bank_seed, microbatch):
    """Accumulate exact global response gradients, with no optimizer update.

    Replay both point z and raw-affine a through the SAME encoder forward.
    Only small feature/moment/readout graphs are retained between passes.
    """
    _runtime(renderer, objective, microbatch)
    n = renderer.n
    params = augment_parameters(n, bank_seed, step, renderer.relation, renderer.base.device)
    clean, augmented, reg = [], [], 0.
    with torch.no_grad():
        for start in range(0, n, microbatch):
            ids = list(range(start, min(n, start+microbatch)))
            x = renderer(ids)
            clean.append(encoder.forward_paths(x))
            augmented.append(encoder.forward_paths(augment(x, params, ids)))
            reg += float(regularization(x, renderer.templates[ids])) * len(ids) / n
    clean = {k: v.detach().requires_grad_(True) for k, v in _cat(clean).items()}
    augmented = {k: v.detach().requires_grad_(True) for k, v in _cat(augmented).items()}
    result = feature_objective(clean, augmented, objective)
    leaves = [clean['z'], clean['a'], augmented['z'], augmented['a']]
    gradients = torch.autograd.grad(result['loss'], leaves, allow_unused=True)
    gz, gr, gaz, gar = [torch.zeros_like(v) if g is None else g.detach()
                        for v, g in zip(leaves, gradients)]
    renderer.zero_grad(set_to_none=True)
    for start in range(0, n, microbatch):
        ids = list(range(start, min(n, start+microbatch)))
        x = renderer(ids)
        paths = encoder.forward_paths(x)
        response = ((paths['z'] * gz[ids]).sum() + (paths['a'] * gr[ids]).sum()
                    + regularization(x, renderer.templates[ids]) * len(ids) / n)
        response.backward()
        # Free clean encoder activations before replaying augmentation.
        del response, paths, x
        x = renderer(ids)
        paths = encoder.forward_paths(augment(x, params, ids))
        ((paths['z'] * gaz[ids]).sum() + (paths['a'] * gar[ids]).sum()).backward()
        del paths, x
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in renderer.parameters()),
            'Nonfinite guarded synthesis gradient')
    readout = result['readout']
    values = {k: float(result[k].detach()) for k in ('loss', 'statistics', 'augmentation_matching', 'functional')}
    values.update(loss=values['loss'] + reg, regularization=reg,
                  alpha=float(readout.alpha.detach()),
                  radius_squared=float(readout.radius_squared.detach()),
                  proposed_shift_squared=float(readout.proposed_shift_squared.detach()),
                  source_image_forwards=4*n, source_image_backwards=2*n)
    return values
