"""Finite real-augmentation point targets; no automatic data access or execution.

P/Q access must be granted by the caller's PixelAccess. This module does not
load an encoder, select projections, evaluate V, or implement a DP release.
"""
import math
import numpy as np
import torch
from .contracts import require, seed
from .encoders import pil_tensor
from .render import augment

RULE = {
    'schema': 'prrd.real-augmentation-mean/v1',
    'views_per_image': 4, 'bank_seed': 101,
    'seed_domain': 'prrd-B-realaug-v1',
    'seed_fields': ['domain', 'bank_seed', 'role', 'image_id', 'view_index'],
    'distribution': {'rotation_degrees': [-3., 3.],
                     'theta_translation': [-.02, .02],
                     'scale': [.98, 1.02], 'brightness': [.98, 1.02]},
    'placement': 'native grayscale -> augment -> existing source preprocess -> bounded point z',
    'native_geometry': [1024, 1024],
    'sampling': 'bilinear zero-padding; align_corners=False; clamp[0,1]',
    'aggregation': 'mean views/images within patient-class, then mean present patients',
    'second_moment': 'mean outer(z,z), not outer(mean(z),mean(z))',
    'statistic_dtype': 'float64', 'feature_dtype': 'float32',
}


def view_parameters(image_id, role, *, device='cpu'):
    require(role in ('P', 'Q') and isinstance(image_id, str) and image_id,
            'Real augmentation requires a P/Q image identity')
    seeds = [seed(RULE['seed_domain'], RULE['bank_seed'], role, image_id, k)
             for k in range(RULE['views_per_image'])]
    u = torch.stack([torch.rand(5, generator=torch.Generator(device='cpu').manual_seed(s))
                     for s in seeds])
    angle = (u[:, 0]*2-1)*3*math.pi/180
    scale = .98+.04*u[:, 3]
    theta = torch.zeros(len(seeds), 2, 3)
    theta[:, 0, 0] = angle.cos()/scale
    theta[:, 0, 1] = -angle.sin()/scale
    theta[:, 1, 0] = angle.sin()/scale
    theta[:, 1, 1] = angle.cos()/scale
    theta[:, 0, 2] = (u[:, 1]*2-1)*.02
    theta[:, 1, 2] = (u[:, 2]*2-1)*.02
    brightness = .98+.04*u[:, 4]
    return {'seeds': np.asarray(seeds, dtype=np.int64),
            'theta': theta.to(device), 'brightness': brightness.to(device)}


def patient_augmented_moments(z, labels, patient_ids, roles):
    """Patient weighting is unchanged by the number of visits or views."""
    z = np.asarray(z, dtype=np.float64)
    labels, ids, roles = np.asarray(labels), np.asarray(patient_ids), np.asarray(roles)
    require(z.ndim == 3 and all(n > 0 for n in z.shape), 'Expected nonempty image/view/feature array')
    n, _, d = z.shape
    require(labels.shape == ids.shape == roles.shape == (n,), 'Feature/metadata shape mismatch')
    require(np.isin(labels, (0, 1)).all() and np.isin(roles, ('P', 'Q')).all(),
            'Invalid label or protected role')
    require(ids.dtype.kind in 'US' and all(str(v) for v in ids), 'Patient identities must be strings')
    require(np.isfinite(z).all() and np.max(np.linalg.norm(z, axis=-1)) <= 1+1e-6,
            'Nonfinite or unbounded point features')
    people = np.unique(ids)
    present = np.zeros((len(people), 2), dtype=np.int64)
    means = np.zeros((len(people), 2, d), dtype=np.float64)
    seconds = np.zeros((len(people), 2, d, d), dtype=np.float64)
    patient_roles = []
    for i, pid in enumerate(people):
        selected = ids == pid
        assigned = np.unique(roles[selected])
        require(len(assigned) == 1, 'Patient appears in multiple roles')
        patient_roles.append(assigned[0])
        for c in (0, 1):
            visits = z[selected & (labels == c)]
            if not len(visits):
                continue
            flat = visits.reshape(-1, d)
            present[i, c] = 1
            means[i, c] = flat.mean(axis=0)
            seconds[i, c] = flat.T @ flat / len(flat)
    patient_roles = np.asarray(patient_roles)
    populations = {}
    for role in ('P', 'Q'):
        take = patient_roles == role
        populations[role] = {
            'counts': present[take].sum(axis=0),
            'm_sum': means[take].sum(axis=0),
            'A_sum': seconds[take].sum(axis=0)}
    counts = sum(p['counts'] for p in populations.values())
    denom = np.maximum(counts, 1)
    target = {'counts': counts,
              'm': sum(p['m_sum'] for p in populations.values()) / denom[:, None],
              'A': sum(p['A_sum'] for p in populations.values()) / denom[:, None, None]}
    return {'target': target, 'populations': populations,
            'patients': {'patient_ids': people, 'roles': patient_roles,
                         'present': present, 'm': means, 'A': seconds}}


def extract_augmented_points(encoder, group, access, role, *, microbatch=16, on_chunk=None):
    """Stream native image -> four augmented views -> existing frozen point path.

    Returns internal features/transform bindings only. No augmented pixels are
    saved. The caller supplies and seals the data/model/phase contract.
    """
    require(role in ('P', 'Q') and role in access.allowed_roles, 'Unapproved extraction role')
    require(encoder.kind == 'source' and not encoder.training
            and not any(p.requires_grad for p in encoder.parameters()), 'Expected frozen source')
    require(microbatch in (4, 8, 16) and len(group) > 0, 'Invalid view microbatch/group')
    require(len({r['image_id'] for r in group}) == len(group), 'Duplicate physical image')
    for row in group:
        actual_role, trusted = access.check(row['path'])
        require(actual_role == role and row == trusted, 'Extraction role/manifest mismatch')
        require([int(row['height']), int(row['width'])] == RULE['native_geometry'],
                'Unexpected native geometry')
    pieces = []
    images_per_chunk = microbatch // RULE['views_per_image']
    for start in range(0, len(group), images_per_chunk):
        rows = group[start:start+images_per_chunk]
        native = torch.stack([pil_tensor(access.image(r)) for r in rows]).to(encoder.device_name)
        params = [view_parameters(r['image_id'], role, device=encoder.device_name) for r in rows]
        theta = torch.cat([p['theta'] for p in params])
        brightness = torch.cat([p['brightness'] for p in params])
        with torch.no_grad():
            images = native.repeat_interleave(RULE['views_per_image'], dim=0)
            images = augment(images, (theta, brightness), list(range(len(images))))
            features = encoder(images).reshape(len(rows), RULE['views_per_image'], -1)
        require(bool(torch.isfinite(features).all()), 'Nonfinite augmented feature')
        pieces.append({'z': features.cpu().numpy(),
                       'view_seeds': np.stack([p['seeds'] for p in params]),
                       'theta': theta.cpu().numpy().reshape(len(rows), 4, 2, 3),
                       'brightness': brightness.cpu().numpy().reshape(len(rows), 4)})
        if on_chunk:
            on_chunk(start, len(rows), pieces[-1])
        del native, images, features, params, theta, brightness
    result = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    result.update(image_ids=np.asarray([r['image_id'] for r in group]),
                  patient_ids=np.asarray([r['patient_id'] for r in group]),
                  labels=np.asarray([int(r['label']) for r in group], dtype=np.int64),
                  roles=np.full(len(group), role))
    return result

