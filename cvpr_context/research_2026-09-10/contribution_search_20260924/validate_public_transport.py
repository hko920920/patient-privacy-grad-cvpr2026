"""Verify fixed public transport; same existing 2,000 patient bootstrap draws."""
import argparse, hashlib, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from geometry_diagnostic import packet, moments
from public_transport_probe import fit

def run(root,out):
    started=time.monotonic(); r=root/'_reports'
    sys.path.insert(0,str(root))
    from prrd_pilot_20260921.common import bootstrap_metrics
    hashes={}
    def read(rel):
        path=r/rel
        hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        return packet(path)
    source=read('receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz')
    assert set(source['roles'].tolist())=={'P'}
    ximg=source['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)
    ids=source['patient_ids'].astype(str); labels=source['labels']; people=sorted(set(ids))
    rows=[]; groups=[]; row_ids=[]; row_classes=[]
    for pid in people:
        for cl in (0,1):
            ix=np.flatnonzero((ids==pid)&(labels==cl))
            if len(ix):
                rows.append(ximg[ix].mean(0));groups.append(ix);row_ids.append(pid);row_classes.append(cl)
    xraw=np.asarray(rows);row_classes=np.asarray(row_classes)
    mult=Counter(row_ids);pw=np.array([1/len(people)/mult[p] for p in row_ids])
    bounds=[1.9061329951133719,1.5917066731966771]
    xs=[xraw*np.minimum(1,c/np.maximum(np.linalg.norm(xraw,axis=1),1e-300))[:,None] for c in bounds]
    targets={key:read(rel)['target'].reshape(2,64).astype(np.float64) for key,rel in {
        'PUBLIC':'receiver_feature_followup_20260924_v1/public_target/target.npz',
        'DP1':'receiver_feature_dp8_20260924_v1/one_release/protected_target/target.npz',
        'DP2':'receiver_feature_followup_20260924_v1/one_release/protected_target/target.npz'}.items()}
    reconstructed_public=np.stack([xs[c][row_classes==c].mean(0) for c in (0,1)])
    public_error=float(np.max(abs(reconstructed_public-targets['PUBLIC'])))
    assert public_error<1e-12,public_error
    frozen=json.loads((out/'public_transport_results.json').read_text(encoding='utf-8'))
    old=read('receiver_feature_followup_20260924_v1/evaluation/predictions_private.npz')
    boot=read('downstream_development_20260917_v1/cluster_bootstrap.npz')
    pids=old['patient_ids'].astype(np.int64);y=old['labels']
    unique,inverse=np.unique(pids,return_inverse=True)
    assert np.array_equal(unique,boot['patient_ids'])
    scores={};weights={};maps={};bootstrap={};checks={}
    for rec,public_file,vfile in [
        ('DenseNet','prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz','prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz'),
        ('ResNet18','receiver_resnet18_reuse_20260924_v1/P_projected_features.npz','receiver_resnet18_reuse_20260924_v1/V_features.npz')]:
        pp=read(public_file);vv=read(vfile)
        assert set(pp['roles'].tolist())=={'P'}
        for key in ('image_ids','patient_ids','labels'):
            assert np.array_equal(pp[key].astype(str),source[key].astype(str)),key
            assert np.array_equal(vv[key].astype(str),old[key].astype(str)),key
        ys=np.array([pp['z'][ix].astype(np.float64).mean(0) for ix in groups])
        cov,bpub=moments(pp,True);H=cov+.1*np.eye(len(bpub))
        ms=[];map_error=0.;dual_error=0.
        for c,x in enumerate(xs):
            T,b,lam=fit(x,ys,pw)
            # Independent augmented-design solve via least-squares with an unpenalized intercept.
            aug=np.column_stack([x,np.ones(len(x))])
            reg=np.zeros((64,65));reg[:,:64]=np.sqrt(lam)*np.eye(64)
            design=np.vstack([np.sqrt(pw[:,None])*aug,reg])
            response=np.vstack([np.sqrt(pw[:,None])*ys,np.zeros((64,ys.shape[1]))])
            coef=np.linalg.lstsq(design,response,rcond=None)[0]
            map_error=max(map_error,float(np.max(abs(coef[:-1]-T))),float(np.max(abs(coef[-1]-b))))
            ms.append((T,b))
            maps[rec+'_T'+str(c)]=T;maps[rec+'_b'+str(c)]=b
            # Exact dual equivalence: public signed weights -> target model class mean.
            mx=pw@x;xc=x-mx;cx=xc.T@(pw[:,None]*xc)
            for name,target in targets.items():
                a=pw+pw*(xc@np.linalg.solve(cx+lam*np.eye(64),target[c]-mx))
                dual_error=max(dual_error,float(np.max(abs(a@ys-(target[c]@T+b)))))
                maps[rec+'_'+name+'_public_weights_c'+str(c)]=a
        assert map_error<1e-8 and dual_error<1e-10,(map_error,dual_error)
        checks[rec]={'augmented_lstsq_max_error':map_error,'public_signed_weight_dual_error':dual_error}
        for name,target in targets.items():
            mu=np.stack([target[c]@ms[c][0]+ms[c][1] for c in (0,1)])
            w=np.linalg.solve(H,.5*(mu[1]-mu[0]));n=rec+'_TRANSPORT_'+name
            weights[n]=w;scores[n]=vv['z'].astype(np.float64)@w
            value=[roc_auc_score(y,scores[n]),average_precision_score(y,scores[n])]
            expected=frozen['receivers'][rec]['metrics'][name]
            assert abs(value[0]-expected['AUROC'])<1e-13 and abs(value[1]-expected['AP'])<1e-13
        n=rec+'_REAL_P';weights[n]=np.linalg.solve(H,bpub);scores[n]=vv['z'].astype(np.float64)@weights[n]
        for name in ('PUBLIC','DP1','DP2'):
            oldname=rec+'_FEATURE_'+name
            scores[oldname]=old['scores'][:,list(old['names']).index(oldname)]
        # Existing strongest internal DP controls, all draws retained.
        for name in ('DINO_DP1','DINO_DP2'):
            oldname=rec+'_'+name
            scores[oldname]=old['scores'][:,list(old['names']).index(oldname)]
    for n,s in scores.items():
        bootstrap[n]=bootstrap_metrics(y,s,pids,boot['patient_counts'],boot['patient_ids'])
    metrics={};independent_error=0.
    for n,s in scores.items():
        point=[roc_auc_score(y,s),average_precision_score(y,s)]
        metrics[n]={metric:{'point':float(point[j]),'patient_cluster_95':np.quantile(bootstrap[n][:,j],[.025,.975]).tolist()} for j,metric in enumerate(('AUROC','AP'))}
        for k in (0,17,1999):
            sw=boot['patient_counts'][k,inverse]
            ref=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
            independent_error=max(independent_error,float(abs(ref-bootstrap[n][k]).max()))
    assert independent_error<1e-12,independent_error
    contrasts={}
    for rec in ('DenseNet','ResNet18'):
        for draw in ('DP1','DP2'):
            a=rec+'_TRANSPORT_'+draw
            for short in ('TRANSPORT_PUBLIC','REAL_P','FEATURE_'+draw,'DINO_DP1','DINO_DP2'):
                b=rec+'_'+short
                contrasts[a+'_minus_'+short]={m:{'delta':metrics[a][m]['point']-metrics[b][m]['point'],'patient_cluster_95':np.quantile((bootstrap[a]-bootstrap[b])[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
    names=list(scores)
    np.savez_compressed(out/'transport_predictions_private.npz',names=np.array(names),scores=np.column_stack([scores[n] for n in names]),
        labels=y,image_ids=old['image_ids'],patient_ids=pids,**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(out/'transport_paired_bootstrap.npz',names=np.array(names),metrics=np.stack([bootstrap[n] for n in names],1),
        source_draws_sha256=np.array(hashes['downstream_development_20260917_v1/cluster_bootstrap.npz']))
    np.savez_compressed(out/'transport_public_maps_and_dp_weights.npz',**maps)
    for local in ('PUBLIC_TRANSPORT_PLAN.md','public_transport_probe.py','validate_public_transport.py'):
        hashes[local]=hashlib.sha256((out/local).read_bytes()).hexdigest()
    result={'scope':'EXPLORATORY_DIRECT_SUMMARY_TRANSPORT_NOT_NEW_SYNTHETIC_IMAGES',
        'new_Q_access':0,'new_private_releases':0,'new_synthetic_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
        'evaluation':'Image-level AUROC/AP; paired patient-cluster bootstrap, 2000 saved draws; conditional on fixed DP releases.',
        'public_target_reconstruction_max_abs_error':public_error,'independent_bootstrap_max_error':independent_error,
        'map_checks':checks,'input_hashes':hashes,'metrics':metrics,'contrasts':contrasts,'seconds':time.monotonic()-started}
    (out/'transport_validation_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'checks':checks,'public_target_error':public_error,'bootstrap_error':independent_error,'contrasts':contrasts,'seconds':result['seconds']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

