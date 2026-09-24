"""Matched public label controls for the fixed-image transport probe."""
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet
from fixed_image_compilation import pw
def run(root,out):
    start=time.monotonic();r=root/'_reports';sys.path.insert(0,str(root))
    from prrd_pilot_20260921.common import bootstrap_metrics
    pub={'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz'),
         'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz')}
    val={'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz'),
         'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/V_features.npz')}
    trans=packet(out/'transport_predictions_private.npz')
    original_results=packet(out/'fixed_image_compilation_predictions_private.npz')
    orig_labels=packet(out/'fixed_image_soft_labels.npz')
    oldboot=packet(out/'transport_paired_bootstrap.npz')
    boot=packet(r/'downstream_development_20260917_v1/cluster_bootstrap.npz')
    syn={};ops={}
    for draw in ('DP1','DP2'):
        for rec in pub:
            path=r/'receiver_feature_dp8_20260924_v1/evaluation'/(rec+'_PNG_features.npz') if draw=='DP1' else r/'receiver_feature_followup_20260924_v1/evaluation'/(rec+'_FEATURE_DP2_PNG_features.npz')
            d=packet(path);z=d['z'].astype(np.float64);syn[draw,rec]=d
            ops[draw,rec]=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T/128)
    scores={};fits={};labelpackets={}
    for draw in ('DP1','DP2'):
        for cal in pub:
            zp=pub[cal]['z'].astype(np.float64);pwgt=pw(pub[cal]);A=zp@ops[draw,cal]
            lam=.001*np.sum(pwgt[:,None]*A*A)/128
            original=2*syn[draw,cal]['labels']-1
            design=np.vstack([np.sqrt(pwgt[:,None])*A,np.sqrt(lam)*np.eye(128)])
            for kind,weight_name in [('DP',cal+'_TRANSPORT_'+draw+'_weight'),('PUBLIC',cal+'_TRANSPORT_PUBLIC_weight'),('REAL_P',cal+'_REAL_P_weight')]:
                key=draw+'_fit_'+cal+'_'+kind
                t=zp@trans[weight_name]
                response=np.concatenate([np.sqrt(pwgt)*t,np.sqrt(lam)*original])
                solved=lsq_linear(design,response,bounds=(-1.,1.),tol=1e-12,max_iter=500)
                assert solved.success
                l=solved.x;labelpackets[key]=l
                if kind=='DP':assert np.max(abs(l-orig_labels[draw+'_fit_'+cal]))<1e-12
                fits[key]={'relative_public_RMSE':float(np.sqrt(np.sum(pwgt*(A@l-t)**2)/np.sum(pwgt*t**2))),
                           'optimality':float(solved.optimality)}
                for ev in pub:
                    score=val[ev]['z'].astype(np.float64)@(ops[draw,ev]@l)
                    n=key+'_eval_'+ev;scores[n]=score
                    if kind=='DP':
                        oi=list(original_results['names']).index(draw+'_fit_'+cal+'_eval_'+ev)
                        assert np.max(abs(score-original_results['scores'][:,oi]))<1e-12
    y=trans['labels'];pid=trans['patient_ids'].astype(np.int64);_,inv=np.unique(pid,return_inverse=True)
    metrics={};boots={};skerr=0.
    for n,s in scores.items():
        metrics[n]={'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))}
        boots[n]=bootstrap_metrics(y,s,pid,boot['patient_counts'],boot['patient_ids'])
        for j in (0,17,1999):
            weights=boot['patient_counts'][j,inv]
            ref=np.array([roc_auc_score(y,s,sample_weight=weights),average_precision_score(y,s,sample_weight=weights)])
            skerr=max(skerr,float(np.max(abs(ref-boots[n][j]))))
    assert skerr<1e-12
    comparisons={}
    for draw in ('DP1','DP2'):
        for cal in pub:
            for ev in pub:
                key=draw+'_fit_'+cal
                a=key+'_DP_eval_'+ev
                bn=ev+'_FEATURE_'+draw
                old_scores=trans['scores'][:,list(trans['names']).index(bn)]
                references={'ORIGINAL_LABELS':(oldboot['metrics'][:,list(oldboot['names']).index(bn),:],
                    {'AUROC':float(roc_auc_score(y,old_scores)),'AP':float(average_precision_score(y,old_scores))})}
                for kind in ('PUBLIC','REAL_P'):
                    n=key+'_'+kind+'_eval_'+ev;references[kind+'_LABEL_CONTROL']=(boots[n],metrics[n])
                for label,(bboot,bmetric) in references.items():
                    name=a+'_minus_'+label
                    comparisons[name]={'receiver_used_for_label_fit':cal==ev,**{m:{'delta':metrics[a][m]-bmetric[m],
                        'patient_cluster_95':np.quantile((boots[a]-bboot)[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}}
    names=list(scores)
    np.savez_compressed(out/'compiler_control_predictions_private.npz',names=np.array(names),scores=np.column_stack([scores[n] for n in names]),
        labels=y,patient_ids=pid,image_ids=trans['image_ids'])
    np.savez_compressed(out/'compiler_control_bootstrap.npz',names=np.array(names),metrics=np.stack([boots[n] for n in names],1))
    np.savez_compressed(out/'compiler_control_soft_labels.npz',**labelpackets)
    result={'scope':'Exploratory; original DP pixels reused; public controls remove DP signal from label calculation, not from pixels.',
      'new_Q_access':0,'new_DP_release':0,'new_pixels':0,'Expert_Reserved':False,'metrics':metrics,'fits':fits,
      'comparisons':comparisons,'bootstrap_sklearn_error':skerr,'seconds':time.monotonic()-start,
      'plan_sha256':hashlib.sha256((out/'COMPILER_CONTROLS_PLAN.md').read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'compiler_control_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'metrics':metrics,'other_receiver_contrasts':{n:v for n,v in comparisons.items() if not v['receiver_used_for_label_fit']},
                     'seconds':result['seconds']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

