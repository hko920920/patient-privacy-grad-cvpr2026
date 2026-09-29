"""2-4A design algebra. Reads public P/carrier only; creates no labels or scores."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
THESIS = ROOT.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    paths = {
        'selection': ROOT / 'contribution_public128_control_20260928_v1/public128_selection.npz',
        'DINO_P': THESIS / 'code_working/_reports/receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz',
        'RN_P': THESIS / 'code_working/_reports/receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
    }
    expected = {
        'selection': '7e49a9ba1ab1ea10ecb7ae6f8cd91792ccd458e14cb1da1534b31154370de3c4',
        'DINO_P': '4093700089740598561abe4fef5246c3f1626d9efc9165183872f157f7bfba6f',
        'RN_P': 'bd07854ecc436ccea6845698f0087fd465c71ec163dceb7d3d720e9903626274',
    }
    before = {k: digest(p) for k,p in paths.items()}
    assert before == expected
    with np.load(paths['selection'], allow_pickle=False) as f:
        sel = {k:f[k] for k in f.files}
    with np.load(paths['DINO_P'], allow_pickle=False) as f:
        dp = {k:f[k] for k in f.files}
    with np.load(paths['RN_P'], allow_pickle=False) as f:
        rp = {k:f[k] for k in f.files}
    assert set(dp['roles']) == set(rp['roles']) == {'P'}
    for k in ['image_ids','patient_ids','labels']:
        assert np.array_equal(dp[k],rp[k])
    xd = dp['z'].transpose(1,0,2).reshape(813,64).astype(float)
    xr = rp['z'].astype(float)
    xs = sel['DINO'].astype(float)
    rs = sel['ResNet18'].astype(float)
    assert np.array_equal(xd[sel['indices']],xs)
    assert np.array_equal(xr[sel['indices']],rs)
    ids = dp['patient_ids']; labels = dp['labels']
    # Same public task weighting: half per class, equal patient within class.
    weights = np.zeros(813)
    patient_class_rows = []
    patient_weights = []
    for patient in np.unique(ids):
        classes = np.unique(labels[ids == patient])
        for c in classes:
            mask = (ids == patient) & (labels == c)
            weights[mask] = .5 / len(np.unique(ids[labels == c])) / mask.sum()
            patient_class_rows.append(xr[mask].mean(axis=0))
            patient_weights.append(1 / len(np.unique(ids)) / len(classes))
    assert abs(weights.sum()-1) < 1e-12
    rows = np.stack(patient_class_rows)
    patient_weights = np.asarray(patient_weights)
    hr = xr.T @ (weights[:,None]*xr) + .1*np.eye(128)
    ev, evec = np.linalg.eigh(hr)
    sqrt_h = (evec*np.sqrt(ev)) @ evec.T
    invsqrt_h = (evec/np.sqrt(ev)) @ evec.T
    _, ds, dv = np.linalg.svd(xs.T, full_matrices=True)
    dtol = max(xs.T.shape)*np.finfo(float).eps*ds[0]
    drank = int((ds > dtol).sum())
    nspace = dv[drank:].T
    br = np.linalg.solve(rs.T@rs/128+.1*np.eye(128),rs.T/128)
    m = sqrt_h @ br @ nspace
    u, sv, _ = np.linalg.svd(m, full_matrices=False)
    mtol = max(m.shape)*np.finfo(float).eps*sv[0]
    mrank = int((sv > mtol).sum())
    u = u[:,:mrank]
    # In the H metric the compatible target is U' H^-1/2 * class mean.
    full = rows @ invsqrt_h
    reduced = full @ u
    fullnorm = np.linalg.norm(full,axis=1)
    reducednorm = np.linalg.norm(reduced,axis=1)
    retained = np.sum(patient_weights*reducednorm**2)/np.sum(patient_weights*fullnorm**2)
    # Descriptive public tails, not a selected privacy calibration.
    same_c = float(np.quantile(fullnorm,.95))
    before_clip = full * np.minimum(1,same_c/np.maximum(fullnorm,1e-300))[:,None]
    before_projected = before_clip @ u
    after_projected = reduced * np.minimum(1,same_c/np.maximum(reducednorm,1e-300))[:,None]
    e_before = np.linalg.norm(before_projected-reduced,axis=1)
    e_after = np.linalg.norm(after_projected-reduced,axis=1)
    assert np.all(e_after <= e_before + 1e-12)
    # A projection after isotropic Gaussian noise has identical retained covariance.
    cov_error = float(np.max(np.abs(u.T@u-np.eye(mrank))))
    source_error = float(np.max(np.abs(xs.T@nspace)))
    projection_error = float(np.max(np.abs(m-u@(u.T@m))))
    assert source_error < 1e-10 and projection_error < 1e-10 and cov_error < 1e-10
    result = {
        'scope':'PUBLIC_GEOMETRY_DESIGN_ONLY_NOT_EFFICACY',
        'P_patients':int(len(np.unique(ids))), 'P_images':813,
        'P_patient_class_rows':len(rows), 'carrier_images':128,
        'DINO_feature_rank':drank, 'source_preserving_label_dimensions':nspace.shape[1],
        'receiver_compatible_dimensions':mrank,
        'M_singular_max':float(sv[0]),'M_singular_min':float(sv[mrank-1]),
        'M_condition_number':float(sv[0]/sv[mrank-1]),
        'patient_equal_retained_norm_energy_fraction':float(retained),
        'public_norm_q95_full':same_c,
        'public_norm_q95_reduced':float(np.quantile(reducednorm,.95)),
        'public_q95_definition':'Unweighted empirical quantile over 675 patient/class rows, descriptive only.',
        'same_C_preprojection_clipped_rows':int((fullnorm>same_c).sum()),
        'same_C_postprojection_clipped_rows':int((reducednorm>same_c).sum()),
        'patient_weighted_per_row_projection_bias_before':float(patient_weights@e_before),
        'patient_weighted_per_row_projection_bias_after':float(patient_weights@e_after),
        'source_nullspace_max_abs':source_error,
        'receiver_range_projection_max_abs':projection_error,
        'retained_noise_covariance_identity_max_abs':cov_error,
        'per_patient_bias_inequality_checked':True,
        'raw_Q_reads':0,'V_reads':0,'DP_release_reads':0,'new_DP_releases':0,
        'label_solves':0,'model_forwards':0,'receiver_evaluations':0,
        'Expert_Reserved_reads':0,
        'inputs':{k:{'path':str(p),'sha256':before[k]} for k,p in paths.items()},
        'warning':'No clinical utility, general transfer, Q-tail behavior, or novelty is established.',
    }
    assert {k:digest(p) for k,p in paths.items()} == before
    result['seconds'] = time.monotonic()-start
    destination = OUT/'public_geometry.json'
    assert not destination.exists(), 'Do not silently overwrite the design record'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
