"""Exact local inverse-risk calculation and independently drawn Monte Carlo checks."""
from pathlib import Path
import argparse,json
import numpy as np

def risk(A,B,K,tau,sigma):
    E=K@A-np.eye(A.shape[1])
    bias=tau**2*np.linalg.norm(B@E,'fro')**2
    variance=sigma**2*np.linalg.norm(B@K,'fro')**2
    return {'bias':float(bias),'variance':float(variance),'total':float(bias+variance)}

def run(root,out):
    with np.load(out/'public_probe/public_operators.npz') as p:
        A=p['A']; Bs={r:p[r+'_B'] for r in ('DenseNet','ResNet18')}
    U,s,Vt=np.linalg.svd(A,full_matrices=False)
    V=Vt.T;tau=.02;sigma=.25*np.median(s)*tau;k=8
    Bn={r:b/np.linalg.norm(b,'fro') for r,b in Bs.items()}
    mean_B=np.concatenate(list(Bn.values()),axis=0)/np.sqrt(len(Bn))
    mass=np.sum((mean_B@V)**2,axis=0)
    gain=mass*(tau**2-sigma**2/s**2)
    selected=np.argsort(-gain)[:k]
    def inverse_modes(ix):return (V[:,ix]/s[ix])@U[:,ix].T
    Ks={
        'source_singular_top8':inverse_modes(np.arange(k)),
        'receiver_risk_top8':inverse_modes(selected),
        'all16':inverse_modes(np.arange(len(s))),
        'anchor_no_update':np.zeros((A.shape[1],A.shape[0]))
    }
    # Honest P-only signal-covariance projection; not a full Dosser reproduction.
    fpath=root/'_reports/receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz'
    with np.load(fpath,allow_pickle=False) as p:
        assert set(p['roles'].tolist())=={'P'}
        z=p['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)
        ids=p['patient_ids']
    unique,inverse=np.unique(ids,return_inverse=True)
    counts=np.bincount(inverse)
    pw=1/(len(unique)*counts[inverse])
    blocks=[]
    for bound in (1.9061329951133719,1.5917066731966771):
        features=.5*z*np.minimum(1.,bound/np.maximum(np.linalg.norm(z,axis=1),1e-300))[:,None]/bound
        centered=features-np.sum(features*pw[:,None],axis=0)
        blocks.append(centered.T@(pw[:,None]*centered))
    C=np.zeros((128,128));C[:64,:64]=blocks[0];C[64:,64:]=blocks[1]
    eig,P=np.linalg.eigh(C);P=P[:,-k:]
    Ks['public_signal_PCA8']=np.linalg.pinv(P.T@A)@P.T
    rng=np.random.default_rng(73219)
    aa=rng.normal(size=(20000,16))*tau
    noise=rng.normal(size=(20000,128))*sigma
    yy=aa@A.T+noise
    result={'scope':'local linearized public operator experiment; NOT patient-DP/X-ray utility',
      'coefficient_prior_sd':tau,'illustrative_query_noise_sd':float(sigma),
      'rank_budget':k,'selected_source_indices':selected.tolist(),
      'receiver_risk_selects_same_as_source':set(selected)==set(range(k)),
      'weights':mass.tolist(),'benefit':gain.tolist(),'methods':{}}
    for name,K in Ks.items():
        got={'per_receiver':{},'public_mean_normalized_risk':risk(A,mean_B,K,tau,sigma)}
        for r,B in Bs.items():
            theory=risk(A,B,K,tau,sigma)
            err=(yy@K.T-aa)@B.T
            empirical=float(np.mean(np.sum(err**2,axis=1)))
            theory.update(monte_carlo=empirical,relative_MC_error=abs(empirical/theory['total']-1))
            assert theory['relative_MC_error']<.05
            got['per_receiver'][r]=theory
        result['methods'][name]=got
    # A sensitivity-calibrated exact linear example: more coordinates need not help.
    A1=np.diag([1.,.1]); additions={'duplicate':A1,'redundant':np.diag([1.,0.]),'complementary':np.diag([.1,1.])}
    examples={}
    for name,A2 in additions.items():
        AM=np.concatenate([A1,A2],axis=0)
        # Public per-encoder bound C=1; conservative joint sum-query sensitivity sqrt(M).
        examples[name]={'single_noise_risk':float(np.trace(np.linalg.inv(A1.T@A1))),
                        'joint_noise_risk':float(2*np.trace(np.linalg.inv(AM.T@AM))),
                        'coordinate_noise_variance_ratio':2.}
    result['linear_gaussian_source_addition_examples']=examples
    np.savez_compressed(out/'public_probe/candidate_maps.npz',**Ks)
    (out/'public_probe/risk_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

