"""Fixed-image soft-label compilation, fixed before outcomes; no image optimization."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet

def pw(d):
    y=d['labels'];p=d['patient_ids'].astype(str);w=np.empty(len(y))
    for c in (0,1):
        ix=np.flatnonzero(y==c);ct=Counter(p[ix])
        for i in ix:w[i]=.5/len(ct)/ct[p[i]]
    return w
def run(root,out):
    tick=time.monotonic();r=root/'_reports'
    pub={
      'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz'),
      'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz')}
    val={'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz'),
         'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/V_features.npz')}
    trans=packet(out/'transport_predictions_private.npz')
    syn={};ops={}
    for draw in ('DP1','DP2'):
        for rec in pub:
            path=(r/'receiver_feature_dp8_20260924_v1/evaluation'/(rec+'_PNG_features.npz')) if draw=='DP1' else (r/'receiver_feature_followup_20260924_v1/evaluation'/(rec+'_FEATURE_DP2_PNG_features.npz'))
            d=packet(path);z=d['z'].astype(np.float64);n=len(z)
            assert n==128 and np.array_equal(d['labels'],[0]*64+[1]*64)
            syn[draw,rec]=d
            ops[draw,rec]=np.linalg.solve(z.T@z/n+.1*np.eye(z.shape[1]),z.T/n)
    results={};soft={};saved={}
    for draw in ('DP1','DP2'):
        for calibrate in pub:
            d=pub[calibrate];assert set(d['roles'].tolist())=={'P'}
            zp=d['z'].astype(np.float64);w=pw(d)
            A=zp@ops[draw,calibrate];target=zp@trans[calibrate+'_TRANSPORT_'+draw+'_weight']
            lam=.001*np.sum(w[:,None]*A*A)/A.shape[1]
            original=2*syn[draw,calibrate]['labels']-1
            design=np.vstack([np.sqrt(w[:,None])*A,np.sqrt(lam)*np.eye(128)])
            response=np.concatenate([np.sqrt(w)*target,np.sqrt(lam)*original])
            fitted=lsq_linear(design,response,bounds=(-1.,1.),tol=1e-12,max_iter=500)
            assert fitted.success,fitted.message
            labels=fitted.x;name=draw+'_fit_'+calibrate;soft[name]=labels
            result={'calibration_receiver':calibrate,'draw':draw,'regularizer':lam,
              'public_target_relative_RMSE':float(np.sqrt(np.sum(w*(A@labels-target)**2)/np.sum(w*target**2))),
              'hard_label_relative_RMSE':float(np.sqrt(np.sum(w*(A@original-target)**2)/np.sum(w*target**2))),
              'soft_label_min':float(labels.min()),'soft_label_max':float(labels.max()),
              'labels_at_bound':int(np.sum(abs(labels)>1-1e-6)),'solver_optimality':float(fitted.optimality),
              'receivers':{}}
            for evaluate in pub:
                weight=ops[draw,evaluate]@labels
                v=val[evaluate];s=v['z'].astype(np.float64)@weight
                assert np.array_equal(v['image_ids'],trans['image_ids'])
                result['receivers'][evaluate]={'AUROC':float(roc_auc_score(v['labels'],s)),
                    'AP':float(average_precision_score(v['labels'],s)),'used_for_label_fit':evaluate==calibrate}
                saved[name+'_eval_'+evaluate]=s
            results[name]=result
    np.savez_compressed(out/'fixed_image_soft_labels.npz',**soft)
    np.savez_compressed(out/'fixed_image_compilation_predictions_private.npz',
      names=np.array(list(saved)),scores=np.column_stack(list(saved.values())),
      labels=trans['labels'],patient_ids=trans['patient_ids'],image_ids=trans['image_ids'])
    res={'scope':'Exploratory soft-label correction on existing 128 PNGs. No new pixels or Q queries.',
      'results':results,'seconds':time.monotonic()-tick,
      'plan_sha256':hashlib.sha256((out/'FIXED_IMAGE_COMPILATION_PLAN.md').read_bytes()).hexdigest()}
    (out/'fixed_image_compilation_results.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
