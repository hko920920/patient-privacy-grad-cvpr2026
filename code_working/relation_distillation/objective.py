"""Differentiable *whole-bank* matching; input is features, never patient files."""
import torch


def bank_moments(z, relation):
    if z.ndim != 2 or z.shape[0] != 128:
        raise ValueError('Expected exactly 128 feature rows')
    # Shared layout: 0:64 negative, 64:128 positive.
    # C/D point subsets: 0:32 and 64:96; pairs: 32:64 <-> 96:128.
    negative = z[:32] if relation else z[:64]
    positive = z[64:96] if relation else z[64:]
    m = torch.stack([negative.mean(0), positive.mean(0)])
    a = torch.stack([negative.T @ negative / len(negative),
                     positive.T @ positive / len(positive)])
    difference = (z[96:128] - z[32:64]) / 2
    return {'m': m, 'A': a, 'delta': difference.mean(0),
            'C': difference.T @ difference / len(difference)}


def matching_loss(z, target, relation, beta=1.0):
    current = bank_moments(z, relation)
    t = {k: torch.as_tensor(target[k], dtype=z.dtype, device=z.device)
         for k in ('m', 'A', 'delta', 'C')}
    point = ((current['m']-t['m']).square().sum()
             + (current['A']-t['A']).square().sum()) / 2
    paired = ((current['delta']-t['delta']).square().sum()
              + (current['C']-t['C']).square().sum()) if relation else z.new_zeros(())
    return point + beta * paired


def feature_gradient(z, target, relation, beta=1.0):
    """First pass: global dL/dz. Replay encoder microbatches with this gradient.

    Averaging separately computed microbatch moment losses is NOT equivalent.
    Frozen eval-mode encoder, fixed images and precision are required on replay.
    """
    leaf = z.detach().clone().requires_grad_(True)
    loss = matching_loss(leaf, target, relation, beta)
    gradient, = torch.autograd.grad(loss, leaf)
    return loss.detach(), gradient.detach()
