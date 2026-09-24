"""Public/previously released synthetic feature geometry; no Q/V/Reserved reads."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json
import numpy as np

def packet(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def moments(data, patient=False):
    z=np.asarray(data['z'],np.float64)
    y=np.asarray(data['labels'],int)
    weights=np.empty(len(z))
    for c in (0,1):
        ids=np.flatnonzero(y==c)
        if patient:
            counts=Counter(str(data['patient_ids'][i]) for i in ids)
            for i in ids: weights[i]=.5/len(counts)/counts[str(data['patient_ids'][i])]
        else: weights[ids]=.5/len(ids)
    assert abs(weights.sum()-1)<1e-12
    return z.T@(weights[:,None]*z), z.T@(weights*(2*y-1))

def run(root, output):
    r=root/'_reports'
    mapping={
        'DenseNet':{
            'public': 'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz',
            'FEATURE_PUBLIC':'receiver_feature_followup_20260924_v1/evaluation/DenseNet_FEATURE_PUBLIC_PNG_features.npz',
            'FEATURE_DP1':'receiver_feature_dp8_20260924_v1/evaluation/DenseNet_PNG_features.npz',
            'FEATURE_DP2':'receiver_feature_followup_20260924_v1/evaluation/DenseNet_FEATURE_DP2_PNG_features.npz',
            'GRAD_DP1':'receiver_dino_dp8_20260924_v1/evaluation/DenseNet_DINO_DP_PNG_features.npz',
            'GRAD_DP2':'receiver_dino_dp8_noise_repeat_20260924_v1/evaluation/DenseNet_DINO_DP2_PNG_features.npz',
            'A2_DP1':'receiver_dp8_first_20260923_v1/evaluation/DenseNet_DP8_PNG_features.npz',
            'A2_DP2':'receiver_dp8_noise_repeat_20260923_v1/evaluation/DenseNet_DP2_PNG_features.npz'},
        'ResNet18':{
            'public':'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
            'FEATURE_PUBLIC':'receiver_feature_followup_20260924_v1/evaluation/ResNet18_FEATURE_PUBLIC_PNG_features.npz',
            'FEATURE_DP1':'receiver_feature_dp8_20260924_v1/evaluation/ResNet18_PNG_features.npz',
            'FEATURE_DP2':'receiver_feature_followup_20260924_v1/evaluation/ResNet18_FEATURE_DP2_PNG_features.npz',
            'GRAD_DP1':'receiver_resnet18_reuse_20260924_v1/DINO_DP1_features.npz',
            'GRAD_DP2':'receiver_resnet18_reuse_20260924_v1/DINO_DP2_features.npz',
            'A2_DP1':'receiver_resnet18_reuse_20260924_v1/A2_DP1_features.npz',
            'A2_DP2':'receiver_resnet18_reuse_20260924_v1/A2_DP2_features.npz'}
    }
    result={'scope':'geometry diagnostic; public P and fixed synthetic outputs only; no V efficacy',
            'new_private_queries':0,'new_synthesis':0,'ridge':.1,'receivers':{},'input_hashes':{}}
    for rec,files in mapping.items():
        stats={}
        for name,rel in files.items():
            path=r/rel
            d=packet(path)
            result['input_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if name=='public': assert set(d['roles'].tolist())=={'P'}
            stats[name]=moments(d,patient=name=='public')
        covp,_=stats['public']; eye=np.eye(len(covp)); hp=covp+.1*eye
        rows={}
        for name,(cov,m) in stats.items():
            h=cov+.1*eye
            w=np.linalg.solve(h,m); wp=np.linalg.solve(hp,m)
            eig=np.linalg.eigvalsh(h)
            cos=float(w@wp/(np.linalg.norm(w)*np.linalg.norm(wp)))
            # Exact equal-mean different-covariance identity.
            rhs=np.linalg.solve(h,(covp-cov)@wp)
            err=float(np.max(np.abs(w-wp-rhs)))
            assert err<1e-12
            rows[name]={
                'condition_number':float(eig[-1]/eig[0]),
                'min_eigenvalue':float(eig[0]),
                'max_eigenvalue':float(eig[-1]),
                'public_covariance_weight_cosine':cos,
                'public_covariance_relative_weight_change':float(np.linalg.norm(w-wp)/np.linalg.norm(w)),
                'class_signal_norm':float(np.linalg.norm(m)),
                'covariance_frobenius':float(np.linalg.norm(cov)),
                'public_covariance_distance':float(np.linalg.norm(cov-covp)),
                'identity_error':err
            }
        result['receivers'][rec]=rows
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['receivers'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)

