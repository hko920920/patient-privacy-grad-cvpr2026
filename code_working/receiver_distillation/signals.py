"""Fixed public conditions and class-balanced, patient-equal CE signals.

No private normalization, clipping, noise, or accountant is hidden here.
These are NON-DP development signals; class counts are private for Q.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch
from torch.nn import functional as F
from prrd_v3.contracts import require, seed
from prrd_v3.render import augment_parameters, augment


@dataclass(frozen=True)
class Condition:
    index: int
    theta: torch.Tensor
    brightness: torch.Tensor

    def apply(self, images):
        n = len(images)
        params = (self.theta.to(images.device).expand(n, -1, -1),
                  self.brightness.to(images.device).expand(n))
        return augment(images, params, list(range(n)))


@dataclass(frozen=True)
class Head:
    weight: torch.Tensor
    bias: torch.Tensor


def conditions(count=4, public_seed=101):
    require(count > 0, 'No signal conditions')
    result = []
    for k in range(count):
        theta, brightness = augment_parameters(
            1, seed('receiver-condition-v1', public_seed, k), 1, False, 'cpu')
        result.append(Condition(k, theta, brightness))
    return tuple(result)


def heads(encoder_id, dimension, count=4, public_seed=101, device='cpu'):
    require(dimension > 0 and count > 0, 'Invalid head shape')
    result = []
    for k in range(count):
        g = torch.Generator().manual_seed(seed('receiver-head-v1', public_seed, encoder_id, k))
        bound = 1 / math.sqrt(dimension)
        w = torch.empty(2, dimension, dtype=torch.float64).uniform_(-bound, bound, generator=g)
        b = torch.empty(2, dtype=torch.float64).uniform_(-bound, bound, generator=g)
        result.append(Head(w.to(device), b.to(device)))
    return tuple(result)


def per_image_gradient(features, labels, head):
    """Exact CE dL/d[vec(W),b], differentiable with respect to features."""
    z = features.double()
    y = torch.as_tensor(labels, dtype=torch.long, device=z.device)
    require(z.ndim == 2 and y.shape == (len(z),), 'Signal shape mismatch')
    require(head.weight.shape == (2, z.shape[1]) and head.bias.shape == (2,), 'Head mismatch')
    require(not head.weight.requires_grad and not head.bias.requires_grad, 'Head must be fixed')
    residual = (z @ head.weight.T + head.bias).softmax(-1) - F.one_hot(y, 2)
    return torch.cat(((residual[:, :, None] * z[:, None, :]).flatten(1), residual), dim=-1)


@dataclass(frozen=True)
class PatientSums:
    # sums[K, class, gradient_coordinate]; counts are class-present patients.
    sums: torch.Tensor
    counts: torch.Tensor
    patients: frozenset

    def balanced(self):
        denominator = self.counts.clamp_min(1).to(self.sums.device)
        return (self.sums / denominator[None, :, None]).sum(1) * 0.5


def patient_class_sums(signals, labels, patient_ids):
    require(signals.ndim == 3, 'Expected K x images x coordinates')
    ys = [int(y) for y in labels]
    require(len(ys) == len(patient_ids) == signals.shape[1], 'Patient metadata shape')
    require(set(ys) <= {0, 1}, 'Expected binary labels')
    patients = frozenset(patient_ids)
    class_sums, counts = [], []
    for c in (0, 1):
        means = []
        for p in sorted(patients):
            ids = [i for i, (pid, y) in enumerate(zip(patient_ids, ys)) if pid == p and y == c]
            if ids:
                means.append(signals[:, ids].mean(1))
        # Preserve differentiability even for an absent class.
        class_sums.append(torch.stack(means).sum(0) if means else signals.sum(1) * 0)
        counts.append(len(means))
    return PatientSums(torch.stack(class_sums, dim=1),
                       torch.tensor(counts, dtype=torch.float64), patients)


def pool_patient_sums(*parts):
    require(len(parts) > 0, 'No populations to combine')
    patients = frozenset()
    for part in parts:
        require(not (patients & part.patients), 'Populations share a patient')
        patients = patients | part.patients
    return PatientSums(sum((p.sums for p in parts)), sum((p.counts for p in parts)), patients)


def synthetic_signal(features, labels, head):
    g = per_image_gradient(features, labels, head)
    ys = torch.as_tensor(labels, dtype=torch.long, device=g.device)
    pieces = [g[ys == c].mean(0) if bool((ys == c).any()) else g.sum(0) * 0 for c in (0, 1)]
    return 0.5 * (pieces[0] + pieces[1])


@dataclass(frozen=True)
class Objective:
    conditions: tuple
    heads: dict
    targets: dict
    labels: tuple

    def matching(self, features):
        require(set(features) == set(self.targets) == set(self.heads), 'Encoder set mismatch')
        blocks = {}
        for name in sorted(features):
            z = features[name]
            require(z.shape[0] == len(self.conditions) == len(self.heads[name]), 'Condition mismatch')
            signal = torch.stack([synthetic_signal(z[k], self.labels, head)
                                  for k, head in enumerate(self.heads[name])])
            target = self.targets[name]
            require(not target.requires_grad and signal.shape == target.shape, 'Invalid frozen target')
            blocks[name] = (1 - F.cosine_similarity(signal, target, dim=-1, eps=1e-12)).mean()
        # Mean over conditions within each encoder, then equal mean over encoders.
        # This is not smooth-max, CVaR or gradient balancing.
        return torch.stack(list(blocks.values())).mean(), blocks


def bind_objective(condition_tuple, head_map, target_map, labels):
    targets = {}
    for name, target in target_map.items():
        target = target.detach().clone().double()
        require(bool(torch.isfinite(target).all()), 'Nonfinite target')
        require(bool((target.norm(dim=-1) > 1e-12).all()), 'Zero target cosine undefined')
        targets[name] = target
    return Objective(condition_tuple, head_map, targets, tuple(int(y) for y in labels))


def tensor_digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        x = t.detach().cpu().contiguous()
        h.update(str((x.dtype, tuple(x.shape))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def objective_digest(objective):
    tensors = []
    for condition in objective.conditions:
        tensors.extend((condition.theta, condition.brightness))
    for name in sorted(objective.heads):
        for head in objective.heads[name]:
            tensors.extend((head.weight, head.bias))
        tensors.append(objective.targets[name])
    return tensor_digest(tensors)

