"""Public-only follow-up to the fixed LOFO analysis: weight-swap design algebra."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, time
import numpy as np
from analyze_public_maps import CAPS, affine, clipped, weighted_rms

def main(root, rr, out):
    t0=time.perf_counter();log=[]
    files={
      'P_DINO':root/'_reports/receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz',
      'P_RN':root/'_reports/receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
      'P_TARGET':root/'_reports/receiver_feature_followup_20260924_v1/public_target/target.npz',
      'HEADS':rr/'contribution_kd_control_20260928_v1/target_weights.npz',
      'OPS':rr/'contribution_kd_corrections_20260928_v1/public_operators.npz',
      'REF':rr/'contribution_kd_corrections_20260928_v1/corrected_target_weights.npz',
    }
    keys={
      'P_DINO':['z','roles','image_ids','patient_ids','labels'],
      'P_RN':['z','roles','image_ids','patient_ids','labels'],
      'P_TARGET':['target'],'HEADS':['DP1_DINO','DP2_DINO'],
      'OPS':['AK','AC','wDP','wRP','lambda_C'],
      'REF':['DP1_ACKD_RN_ONLY','DP2_ACKD_RN_ONLY'],
    }
    data={}
    for role,p in files.items():
        with np.load(p,allow_pickle=False) as f:data[role]={k:f[k] for k in keys[role]}
        log.append({'path':str(p),'role':role,'keys':keys[role],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dp,rp=data['P_DINO'],data['P_RN']
    assert set(dp['roles'].tolist())==set(rp['roles'].tolist())=={'P'}
    assert all(np.array_equal(dp[k],rp[k]) for k in ['image_ids','patient_ids','labels'])
    d=dp['z'].transpose(1,0,2).reshape(-1,64).astype(float);r=rp['z'].astype(float)
    pid=dp['patient_ids'].astype(str);cl=dp['labels'].astype(int);people=sorted(set(pid))
    ix=[];gp=[];gc=[]
    for p in people:
        for c in (0,1):
            k=np.flatnonzero((pid==p)&(cl==c))
            if len(k):ix.append(k);gp.append(p);gc.append(c)
    gp=np.array(gp);gc=np.array(gc);pc=Counter(gp)
    a=np.array([1/len(people)/pc[p] for p in gp]);b=np.array([.5/np.sum(gc==c) for c in gc])
    ua=np.zeros(len(d));wb=np.zeros(len(d))
    for j,k in enumerate(ix):ua[k]=a[j]/len(k);wb[k]=b[j]/len(k)
    x=np.stack([d[k].mean(0) for k in ix]);y=np.stack([r[k].mean(0) for k in ix])
    gd=d.T@(wb[:,None]*d);hd=gd+.1*np.eye(64)
    hr=r.T@(wb[:,None]*r)+.1*np.eye(128)
    public=data['P_TARGET']['target'].reshape(2,64);mid=public.mean(0)
    dpub=.5*(public[1]-public[0]);wdpub=np.linalg.solve(hd,dpub)
    wrpub=np.linalg.solve(hr,r.T@(wb*(2*cl-1)))
    lam=float(data['OPS']['lambda_C']);ops={};targets={};checks={}
    for name,weights in [('B',wb),('U',ua)]:
        g=d.T@(weights[:,None]*d);c=r.T@(weights[:,None]*d)
        op=np.linalg.solve(hr,c@np.linalg.solve(g+lam*np.eye(64),hd))
        # Independent solve order for the same operator.
        ind=np.linalg.solve(hr,c)@np.linalg.solve(g+lam*np.eye(64),hd)
        checks['operator_association_'+name]=float(np.max(np.abs(op-ind)))
        ops[name]=op
        for release in ('DP1','DP2'):
            wd=data['HEADS'][release+'_DINO']
            targets['ACKD_'+name+'_'+release]=wrpub+op@(wd-wdpub)
    checks['AC_operator_reproduction']=float(np.max(np.abs(ops['B']-data['OPS']['AC'])))
    checks['source_public_anchor']=float(np.max(np.abs(wdpub-data['OPS']['wDP'])))
    checks['receiver_public_anchor']=float(np.max(np.abs(wrpub-data['OPS']['wRP'])))
    for release in ('DP1','DP2'):
        checks['ACKD_reproduction_'+release]=float(np.max(np.abs(targets['ACKD_B_'+release]-data['REF'][release+'_ACKD_RN_ONLY'])))
    assert max(checks.values())<1e-10,checks
    # Keep original per-cap HR lambdas for both weights; no retuning.
    lambdas=[]
    for cap in CAPS:lambdas.append(affine(clipped(x,cap),y,a)[2])
    maps={}
    for name,w in [('U',a),('B',b)]:
        maps[name]=[affine(clipped(x,cap),y,w,lambdas[c])[:2] for c,cap in enumerate(CAPS)]
        for release in ('DP1','DP2'):
            delta=hd@data['HEADS'][release+'_DINO'];mu=[mid-delta,mid+delta]
            mm=[mu[c]@maps[name][c][0]+maps[name][c][1] for c in (0,1)]
            targets['HR_'+name+'_'+release]=np.linalg.solve(hr,.5*(mm[1]-mm[0]))
    # Exact weighted-ridge perturbation identity, including fitted intercept.
    # This is standard algebra, not a new theorem or a utility guarantee.
    for c,cap in enumerate(CAPS):
        z=np.column_stack([clipped(x,cap),np.ones(len(x))])
        th_u=np.vstack(maps['U'][c]);th_b=np.vstack(maps['B'][c])
        penalty=np.diag(np.r_[np.full(64,lambdas[c]),0.])
        rhs=np.linalg.solve(z.T@(b[:,None]*z)+penalty,z.T@((b-a)[:,None]*(y-z@th_u)))
        err=float(np.max(np.abs(th_b-th_u-rhs)))
        checks['weighted_ridge_perturbation_cap'+str(c)]=err
        assert err<1e-10,err
    public_diff={}
    for method in ['HR','ACKD']:
        for release in ['DP1','DP2']:
            w0=targets[method+'_U_'+release];w1=targets[method+'_B_'+release]
            public_diff[method+'_'+release]={
              'U_minus_B_score_RMS':weighted_rms(r@(w0-w1),wb),
              'relative_to_U_score_RMS':weighted_rms(r@(w0-w1),wb)/weighted_rms(r@w0,wb)}
    # Six deterministic public-positive patient deletion diagnostics, map-only.
    # Full-P task heads, source summaries, task weights and lambda stay fixed.
    deletion={name:{release:[] for release in ['DP1','DP2']} for name in ['U','B']}
    pospatients=sorted(set(gp[gc==1]))
    for p in pospatients:
        keep=gp!=p
        unique=sorted(set(gp[keep]));counts=Counter(gp[keep])
        au=np.array([1/len(unique)/counts[q] for q in gp[keep]])
        cb=np.array([.5/np.sum(gc[keep]==c) for c in gc[keep]])
        for name,w in [('U',au),('B',cb)]:
            mapsdel=[affine(clipped(x[keep],cap),y[keep],w,lambdas[c])[:2] for c,cap in enumerate(CAPS)]
            for release in ['DP1','DP2']:
                delta=hd@data['HEADS'][release+'_DINO'];mu=[mid-delta,mid+delta]
                mm=[mu[c]@mapsdel[c][0]+mapsdel[c][1] for c in (0,1)]
                new=np.linalg.solve(hr,.5*(mm[1]-mm[0]));old=targets['HR_'+name+'_'+release]
                deletion[name][release].append(weighted_rms(r@(new-old),wb))
    sensitivity={name:{release:{'mean_public_target_score_RMS':float(np.mean(values)),'max_public_target_score_RMS':float(np.max(values)),'all_six_RMS':values} for release,values in per.items()} for name,per in deletion.items()}
    report={'item':'2-1R2','scope':'Public calibration-weight bridge; no labels/readouts/V/Q.',
      'read_log':log,'checks':checks,'fixed_CKD_lambda':lam,'fixed_HR_lambdas':lambdas,
      'public_target_weight_swap':public_diff,'leave_one_positive_public_patient_out_map_sensitivity':sensitivity,
      'notes':['Deletion diagnostics use all six public positives, never private patients.',
        'Kish effective sample size and mapping-CV errors do not estimate downstream AUROC or prove the cause of existing HR gains.',
        'Next efficacy design is selected using these public diagnostics, so it is exploratory development, not an untouched confirmatory test.',
        'Proposed2x2 swaps only correspondence-map weights. Task/source/receiver head weights, protected head, public source midpoint, actual ACKD receiver anchor, lambda numbers and label solver remain fixed. HR fitted means/intercepts and its implied mapped public anchor change as a consequence of map weighting; this is not a slope-only ablation.'],
      'calculation_seconds':time.perf_counter()-t0}
    (out/'weight_bridge_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'weight_bridge_targets_NOT_LABELS.npz',**targets)
    print(json.dumps({k:report[k] for k in ['checks','public_target_weight_swap','leave_one_positive_public_patient_out_map_sensitivity','calculation_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rr',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.rr,a.out)
