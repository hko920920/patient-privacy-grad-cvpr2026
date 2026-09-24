"""DINO+publicly transported ResNet18 functional label compiler; DenseNet excluded."""
import argparse,json,time,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet,moments
from fixed_image_compilation import pw
def run(root,out):
    tick=time.monotonic();r=root/'_reports';sys.path.insert(0,str(root))
    from prrd_pilot_20260921.common import bootstrap_metrics
    src=packet(r/'receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz')
    source={**src,'z':src['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)}
    public={'DINO':source,'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz')}
    sf=packet(out/'existing_png_dino_K4_features.npz')
    trans=packet(out/'transport_predictions_private.npz')
    source_predictions=packet(out/'source_label_baseline_predictions_private.npz')
    compiler_predictions=packet(out/'compiler_control_predictions_private.npz')
    boot=packet(r/'downstream_development_20260917_v1/cluster_bootstrap.npz')
    val={'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz'),
         'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/V_features.npz')}
    old=np.array([-1.]*64+[1.]*64)
    labels={};scores={};fit_report={}
    for draw,tpath in [('DP1','receiver_feature_dp8_20260924_v1/one_release/protected_target/target.npz'),('DP2','receiver_feature_followup_20260924_v1/one_release/protected_target/target.npz')]:
        target=packet(r/tpath)['target'].reshape(2,64)
        designparts=[];targetparts=[];synthetic={}
        for m in ('DINO','ResNet18'):
            assert set(public[m]['roles'].tolist())=={'P'}
            xp=public[m]['z'].astype(np.float64)
            if m=='DINO':
                xs=sf[draw]
                H,_=moments(source,True)
                wt=np.linalg.solve(H+.1*np.eye(64),.5*(target[1]-target[0]))
            else:
                f=r/'receiver_feature_dp8_20260924_v1/evaluation/ResNet18_PNG_features.npz' if draw=='DP1' else r/'receiver_feature_followup_20260924_v1/evaluation/ResNet18_FEATURE_DP2_PNG_features.npz'
                xs=packet(f)['z'].astype(np.float64);wt=trans[m+'_TRANSPORT_'+draw+'_weight']
            synthetic[m]=xs
            op=np.linalg.solve(xs.T@xs/128+.1*np.eye(xs.shape[1]),xs.T/128)
            A=xp@op;t=xp@wt;w=pw(public[m]);scale=np.sqrt(np.sum(w*t*t))
            assert scale>1e-12
            designparts.append(np.sqrt(.5*w[:,None])*A/scale)
            targetparts.append(np.sqrt(.5*w)*t/scale)
        A=np.vstack(designparts);t=np.concatenate(targetparts)
        lam=.001*np.sum(A*A)/128
        fitted=lsq_linear(np.vstack([A,np.sqrt(lam)*np.eye(128)]),
                         np.concatenate([t,np.sqrt(lam)*old]),bounds=(-1,1),tol=1e-12,max_iter=500)
        assert fitted.success
        l=fitted.x;labels[draw]=l
        fit_report[draw]={'relative_public_RMSE_DINO':float(np.linalg.norm(designparts[0]@l-targetparts[0])*np.sqrt(2)),
            'relative_public_RMSE_ResNet18':float(np.linalg.norm(designparts[1]@l-targetparts[1])*np.sqrt(2)),
            'original_joint_residual':float(np.linalg.norm(A@old-t)),
            'final_joint_residual':float(np.linalg.norm(A@l-t)),
            'regularizer':lam,'optimality':float(fitted.optimality),'at_bound':int(np.sum(abs(l)>1-1e-6))}
        for rec in val:
            if rec=='ResNet18':xs=synthetic[rec]
            else:
                f=r/'receiver_feature_dp8_20260924_v1/evaluation/DenseNet_PNG_features.npz' if draw=='DP1' else r/'receiver_feature_followup_20260924_v1/evaluation/DenseNet_FEATURE_DP2_PNG_features.npz'
                xs=packet(f)['z'].astype(np.float64)
            w=np.linalg.solve(xs.T@xs/128+.1*np.eye(xs.shape[1]),xs.T@l/128)
            scores[draw+'_JOINT_eval_'+rec]=val[rec]['z'].astype(np.float64)@w
            for source_label,n in [('ORIGINAL',rec+'_FEATURE_'+draw)]:
                scores[draw+'_'+source_label+'_eval_'+rec]=trans['scores'][:,list(trans['names']).index(n)]
            for name in ('DIRECT_SOURCE_PSEUDOLABEL','SOURCE_LABEL_SOLVE'):
                n=draw+'_'+name+'_eval_'+rec;scores[n]=source_predictions['scores'][:,list(source_predictions['names']).index(n)]
            n=draw+'_fit_ResNet18_DP_eval_'+rec
            scores[draw+'_RESNET_ONLY_eval_'+rec]=compiler_predictions['scores'][:,list(compiler_predictions['names']).index(n)]
    y=trans['labels'];pids=trans['patient_ids'].astype(np.int64)
    boots={n:bootstrap_metrics(y,s,pids,boot['patient_counts'],boot['patient_ids']) for n,s in scores.items()}
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    contrasts={}
    for draw in ('DP1','DP2'):
        for rec in val:
            a=draw+'_JOINT_eval_'+rec
            for kind in ('ORIGINAL','DIRECT_SOURCE_PSEUDOLABEL','SOURCE_LABEL_SOLVE','RESNET_ONLY'):
                b=draw+'_'+kind+'_eval_'+rec
                contrasts[a+'_minus_'+kind]={m:{'delta':metrics[a][m]-metrics[b][m],
                    'patient_cluster_95':np.quantile((boots[a]-boots[b])[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
    names=list(scores)
    np.savez_compressed(out/'joint_compiler_soft_labels.npz',**labels)
    np.savez_compressed(out/'joint_compiler_predictions_private.npz',names=np.array(names),scores=np.column_stack([scores[n] for n in names]),
        labels=y,patient_ids=pids,image_ids=trans['image_ids'])
    np.savez_compressed(out/'joint_compiler_bootstrap.npz',names=np.array(names),metrics=np.stack([boots[n] for n in names],1))
    result={'scope':'Exploratory joint DINO/ResNet18 functional soft-label compiler; DenseNet excluded from construction loss.',
      'new_Q_access':0,'new_releases':0,'new_pixels':0,'Expert_Reserved':False,
      'fits':fit_report,'metrics':metrics,'comparisons':contrasts,'seconds':time.monotonic()-tick,
      'plan_sha256':hashlib.sha256((out/'JOINT_COMPILER_PLAN.md').read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'joint_compiler_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'fits':fit_report,'joint_metrics':{n:v for n,v in metrics.items() if 'JOINT' in n},
        'DenseNet_comparisons':{n:v for n,v in contrasts.items() if 'DenseNet' in n},'seconds':result['seconds']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

