"""Public affine receiver transport of existing DP feature targets; exploratory."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet,moments

def fit(x,y,w):
    w=w/w.sum();mx=w@x;my=w@y;xc=x-mx;yc=y-my
    cov=xc.T@(w[:,None]*xc)
    lam=1e-3*np.trace(cov)/x.shape[1]
    T=np.linalg.solve(cov+lam*np.eye(x.shape[1]),xc.T@(w[:,None]*yc))
    return T,my-mx@T,float(lam)
def run(root,out):
    r=root/'_reports'
    source=packet(r/'receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz')
    assert set(source['roles'].tolist())=={'P'}
    z=source['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)
    pid=source['patient_ids'].astype(str);labels=source['labels'];image_ids=source['image_ids']
    public={
      'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz'),
      'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz')}
    vfiles={'DenseNet':'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz','ResNet18':'receiver_resnet18_reuse_20260924_v1/V_features.npz'}
    targetpaths={
      'PUBLIC':'receiver_feature_followup_20260924_v1/public_target/target.npz',
      'DP1':'receiver_feature_dp8_20260924_v1/one_release/protected_target/target.npz',
      'DP2':'receiver_feature_followup_20260924_v1/one_release/protected_target/target.npz'}
    targets={key:packet(r/p)['target'].reshape(2,64) for key,p in targetpaths.items()}
    people=sorted(set(pid));xs=[];indices=[];pids=[]
    for p in people:
        for c in (0,1):
            ix=np.flatnonzero((pid==p)&(labels==c))
            if len(ix):xs.append(z[ix].mean(0));indices.append(ix);pids.append(p)
    xs=np.array(xs);pcounts=Counter(pids);w=np.array([1/len(people)/pcounts[p] for p in pids])
    folds=np.array([int(hashlib.sha256(('public-transport-v1'+p).encode()).hexdigest()[:8],16)%5 for p in pids])
    bounds=[1.9061329951133719,1.5917066731966771]
    result={'scope':'EXPLORATORY_SUMMARY_TRANSPORT_NOT_SYNTHETIC_IMAGE_RESULT',
       'new_private_queries':0,'new_synthesis':0,'new_encoders':0,'Reserved_Expert':False,
       'P_patients':len(people),'P_patient_class_rows':len(pids),'receivers':{}}
    for receiver,dp in public.items():
        assert np.array_equal(dp['image_ids'],image_ids) and set(dp['roles'].tolist())=={'P'}
        ys=np.array([dp['z'][ix].astype(np.float64).mean(0) for ix in indices])
        maps=[];checks=[]
        for cl,bound in enumerate(bounds):
            x=xs*np.minimum(1,bound/np.maximum(np.linalg.norm(xs,axis=1),1e-300))[:,None]
            T,b,lam=fit(x,ys,w);maps.append((T,b))
            pred=np.zeros_like(ys);constant=np.zeros_like(ys)
            for fold in range(5):
                tr=folds!=fold;te=~tr
                Tf,bf,_=fit(x[tr],ys[tr],w[tr]);pred[te]=x[te]@Tf+bf
                constant[te]=np.average(ys[tr],axis=0,weights=w[tr])
            err=float(np.sum(w[:,None]*(pred-ys)**2));base=float(np.sum(w[:,None]*(constant-ys)**2))
            checks.append({'class_clip':cl,'relative_ridge':lam,'patient_cv_R2':1-err/base,
                           'weighted_cv_squared_error':err,'constant_error':base})
        # Source maps are now fixed before V is read.
        vp=packet(r/vfiles[receiver]);zv=vp['z'].astype(np.float64);yv=vp['labels']
        cov,bp=moments(dp,patient=True);H=cov+.1*np.eye(len(bp))
        scores={};metrics={}
        for name,target in targets.items():
            mu=np.stack([target[c]@maps[c][0]+maps[c][1] for c in (0,1)])
            b=.5*(mu[1]-mu[0]);weight=np.linalg.solve(H,b);score=zv@weight
            scores[name]=score
            metrics[name]={'AUROC':float(roc_auc_score(yv,score)),'AP':float(average_precision_score(yv,score))}
        rp=zv@np.linalg.solve(H,bp)
        metrics['DIRECT_REAL_P']={'AUROC':float(roc_auc_score(yv,rp)),'AP':float(average_precision_score(yv,rp))}
        result['receivers'][receiver]={'public_cross_validation':checks,'metrics':metrics,
          'DP_minus_matched_public':{name:{m:metrics[name][m]-metrics['PUBLIC'][m] for m in ('AUROC','AP')} for name in ('DP1','DP2')}}
    (out/'public_transport_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

