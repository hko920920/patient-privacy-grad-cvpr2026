"""2-1R2: public map algebra only; no labels, V, Q, images or model forwards."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, json, time
import numpy as np

CAPS = (1.9061329951133719, 1.5917066731966771)

def clipped(x, cap):
    return x * np.minimum(1., cap / np.maximum(np.linalg.norm(x, axis=1), 1e-300))[:, None]

def affine(x, y, weight, lam=None, center=True):
    weight = weight / weight.sum()
    mx = weight @ x if center else np.zeros(x.shape[1])
    my = weight @ y if center else np.zeros(y.shape[1])
    xc, yc = x-mx, y-my
    cov = xc.T @ (weight[:, None]*xc)
    cross = xc.T @ (weight[:, None]*yc)
    if lam is None:
        lam = .001 * np.trace(cov) / x.shape[1]
    t = np.linalg.solve(cov + lam*np.eye(x.shape[1]), cross)
    b = my-mx@t
    # Independent augmented weighted least-squares with unpenalized intercept.
    xa = np.column_stack([x, np.ones(len(x))]) if center else x
    penalty = np.diag(np.r_[np.full(x.shape[1], np.sqrt(lam)), 0.]) if center else np.sqrt(lam)*np.eye(x.shape[1])
    aug = np.vstack([np.sqrt(weight)[:, None]*xa, penalty])
    rhs = np.vstack([np.sqrt(weight)[:, None]*y, np.zeros((penalty.shape[0], y.shape[1]))])
    independent = np.linalg.lstsq(aug, rhs, rcond=None)[0]
    err = np.max(np.abs(independent[:x.shape[1]]-t))
    if center:
        err = max(err, np.max(np.abs(independent[-1]-b)))
    assert err < 1e-9, err
    return t, b, float(lam), float(err)

def weighted_rms(x, weights):
    return float(np.sqrt(np.sum(weights*x*x)))

def main(root, rr, out):
    started = time.perf_counter()
    paths = {
      'P_DINO': root/'_reports/receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz',
      'P_RN': root/'_reports/receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
      'P_TARGET': root/'_reports/receiver_feature_followup_20260924_v1/public_target/target.npz',
      'OLD_SOURCE_HEADS': rr/'contribution_kd_control_20260928_v1/target_weights.npz',
      'PUBLIC_MAPS': rr/'contribution_search_20260924/transport_public_maps_and_dp_weights.npz',
      'OLD_HR_TARGETS': rr/'contribution_head_reconstruction_20260928_v1/HR_target_weights.npz',
    }
    selected_keys = {
      'P_DINO':['z','patient_ids','image_ids','labels','roles'],
      'P_RN':['z','patient_ids','image_ids','labels','roles'],
      'P_TARGET':['target'],
      'OLD_SOURCE_HEADS':['DP1_DINO','DP2_DINO'],
      'PUBLIC_MAPS':['ResNet18_T0','ResNet18_T1','ResNet18_b0','ResNet18_b1'],
      'OLD_HR_TARGETS':['DP1_HR_RN_ONLY','DP2_HR_RN_ONLY'],
    }
    data, log = {}, []
    for role, path in paths.items():
        with np.load(path, allow_pickle=False) as f:
            data[role] = {key:f[key] for key in selected_keys[role]}
        log.append({'role':role, 'path':str(path), 'keys':selected_keys[role],
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    sd, rd = data['P_DINO'], data['P_RN']
    assert set(sd['roles'].tolist()) == set(rd['roles'].tolist()) == {'P'}
    for key in ('image_ids','patient_ids','labels'):
        assert np.array_equal(sd[key], rd[key])
    d = sd['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)
    r = rd['z'].astype(np.float64)
    pids, labels = sd['patient_ids'].astype(str), sd['labels'].astype(int)
    people = sorted(set(pids)); groups=[]; gp=[]; gc=[]
    for p in people:
        for c in (0,1):
            ix=np.flatnonzero((pids==p)&(labels==c))
            if len(ix):
                groups.append(ix); gp.append(p); gc.append(c)
    gc=np.array(gc); gp=np.array(gp); sizes=np.array([len(i) for i in groups])
    x=np.stack([d[i].mean(0) for i in groups]); y=np.stack([r[i].mean(0) for i in groups])
    pc=Counter(gp); a=np.array([1/len(people)/pc[p] for p in gp])
    bc=np.array([.5/np.sum(gc==c) for c in gc])
    ua, omega = np.zeros(len(d)), np.zeros(len(d))
    groupindex=np.empty(len(d),dtype=int)
    for j,ix in enumerate(groups):
        ua[ix]=a[j]/len(ix); omega[ix]=bc[j]/len(ix); groupindex[ix]=j
    assert abs(a.sum()-1)<1e-12 and abs(omega.sum()-1)<1e-12
    gd=d.T@(omega[:,None]*d); hd=gd+.1*np.eye(64)
    hr=r.T@(omega[:,None]*r)+.1*np.eye(128)
    public_mu=data['P_TARGET']['target'].reshape(2,64).astype(np.float64)
    s=public_mu.mean(0)
    classmean=np.stack([clipped(x[gc==c],CAPS[c]).mean(0) for c in (0,1)])
    public_target_error=float(np.max(np.abs(classmean-public_mu)))
    assert public_target_error<1e-12
    mu={}
    for release in ('DP1','DP2'):
        delta=hd@data['OLD_SOURCE_HEADS'][release+'_DINO']
        mu[release]=np.stack([s-delta,s+delta])
    # All matrix variants fixed here; none is selected using V.
    variants=['HR','UNCLIPPED','CLIP_BEFORE_MEAN','CLASS_BALANCED_MAP','UNCENTERED','IMAGE_LEVEL_MAP']
    maps={name:[] for name in variants}; mapstats={}; lambdas=[]
    clipstats=[]; checks={'public_target_reconstruction':public_target_error}
    for c,cap in enumerate(CAPS):
        xc=clipped(x,cap)
        image_clipped=clipped(d,cap)
        xbc=np.stack([image_clipped[ix].mean(0) for ix in groups])
        t,b,lam,err=affine(xc,y,a)
        lambdas.append(lam)
        specs={
          'HR':(xc,y,a,True),
          'UNCLIPPED':(x,y,a,True),
          'CLIP_BEFORE_MEAN':(xbc,y,a,True),
          'CLASS_BALANCED_MAP':(xc,y,bc,True),
          'UNCENTERED':(xc,y,a,False),
          'IMAGE_LEVEL_MAP':(image_clipped,r,ua,True),
        }
        for name,(xx,yy,ww,cent) in specs.items():
            tt,bb,ll,ee=affine(xx,yy,ww,lam,cent)
            maps[name].append((tt,bb))
            own_lam=.001*np.trace((xx-ww@xx).T@(ww[:,None]*(xx-ww@xx)))/64 if cent else .001*np.trace(xx.T@(ww[:,None]*xx))/64
            mapstats.setdefault(name,[]).append({
              'cap_index':c,'fixed_lambda':ll,'own_trace_lambda_diagnostic_only':float(own_lam),
              'independent_fit_error':ee,
              'relative_T_difference_from_HR':float(np.linalg.norm(tt-t)/np.linalg.norm(t)),
              'intercept_difference_norm':float(np.linalg.norm(bb-b)),
              'matched_training_weighted_error':float(np.sum(ww[:,None]*(xx@tt+bb-yy)**2)),
              'patient_summary_weighted_error_on_HR_inputs':float(np.sum(a[:,None]*(xc@tt+bb-y)**2)),
            })
        old=data['PUBLIC_MAPS']
        e=max(np.max(np.abs(t-old['ResNet18_T'+str(c)])),np.max(np.abs(b-old['ResNet18_b'+str(c)])))
        checks['map_reproduction_'+str(c)]=float(e); assert e<1e-11,e
        gap=xbc-xc
        clipstats.append({
          'cap':cap,'groups_clipped':int(np.sum(np.linalg.norm(x,axis=1)>cap)),
          'images_clipped':int(np.sum(np.linalg.norm(d,axis=1)>cap)),
          'groups_with_clip_order_gap':int(np.sum(np.linalg.norm(gap,axis=1)>1e-12)),
          'weighted_clip_order_gap_norm':float(np.sqrt(np.sum(a[:,None]*gap**2))),
          'max_clip_order_gap_norm':float(np.max(np.linalg.norm(gap,axis=1))),
          'relative_weighted_clip_order_gap':float(np.sqrt(np.sum(a[:,None]*gap**2)/np.sum(a[:,None]*xc**2))),
        })
    covariance={}
    for name,iw,gw in [('patient_equal',ua,a),('class_balanced',omega,bc)]:
        md=iw@d; mr=iw@r
        di=d-md; ri=r-mr; xb=x-md; yb=y-mr
        gi=di.T@(iw[:,None]*di); gb=xb.T@(gw[:,None]*xb)
        ci=di.T@(iw[:,None]*ri); cb=xb.T@(gw[:,None]*yb)
        residuald=d-x[groupindex]; residualr=r-y[groupindex]
        within=residuald.T@(iw[:,None]*residuald)
        within_cross=residuald.T@(iw[:,None]*residualr)
        e=max(np.max(np.abs(gi-gb-within)),np.max(np.abs(ci-cb-within_cross)))
        assert e<1e-12,e
        covariance[name]={'between_within_identity_error':float(e),
          'within_trace_fraction':float(np.trace(within)/np.trace(gi)),
          'within_cross_fro_fraction':float(np.linalg.norm(within_cross)/np.linalg.norm(ci))}
    targets={}; td={}
    for name in variants:
        td[name]={}
        for release in ('DP1','DP2'):
            mm=np.stack([mu[release][c]@maps[name][c][0]+maps[name][c][1] for c in (0,1)])
            w=np.linalg.solve(hr,.5*(mm[1]-mm[0])); targets[name+'_'+release]=w
            ref=data['OLD_HR_TARGETS'][release+'_HR_RN_ONLY']
            diff=w-ref
            if name=='HR':
                checks['HR_target_reproduction_'+release]=float(np.max(np.abs(diff)))
                assert np.max(np.abs(diff))<1e-11
            td[name][release]={
              'public_image_target_score_difference_rms':weighted_rms(r@diff,omega),
              'relative_public_target_score_difference':weighted_rms(r@diff,omega)/weighted_rms(r@ref,omega),
              'target_coefficient_difference_norm':float(np.linalg.norm(diff)),
            }
    # Public 5-fold patient CV, same frozen fold assignment as historical maps.
    folds=np.array([int(hashlib.sha256(('public-transport-v1'+p).encode()).hexdigest()[:8],16)%5 for p in gp])
    cv={}
    for name in variants:
        cv[name]=[]
        for c,cap in enumerate(CAPS):
            xhr=clipped(x,cap); idata=clipped(d,cap)
            xx={'HR':xhr,'UNCLIPPED':x,'CLIP_BEFORE_MEAN':np.stack([idata[ix].mean(0) for ix in groups]),'CLASS_BALANCED_MAP':xhr,'UNCENTERED':xhr,'IMAGE_LEVEL_MAP':idata}[name]
            yy=r if name=='IMAGE_LEVEL_MAP' else y
            ww=ua if name=='IMAGE_LEVEL_MAP' else (bc if name=='CLASS_BALANCED_MAP' else a)
            ff=folds[groupindex] if name=='IMAGE_LEVEL_MAP' else folds
            pred=np.zeros_like(y); const=np.zeros_like(y)
            for fold in range(5):
                train=ff!=fold; test=folds==fold
                tf,bf,_,_=affine(xx[train],yy[train],ww[train],lambdas[c],name!='UNCENTERED')
                pred[test]=xhr[test]@tf+bf
                const[test]=np.average(y[folds!=fold],axis=0,weights=a[folds!=fold])
            err=float(np.sum(a[:,None]*(pred-y)**2)); base=float(np.sum(a[:,None]*(const-y)**2))
            cv[name].append({'cap_index':c,'patient_weighted_R2_for_HR_input':1-err/base,
              'patient_weighted_MSE':err, 'constant_MSE':base,
              'negative_group_MSE':float(np.mean(np.sum((pred[gc==0]-y[gc==0])**2,axis=1))),
              'positive_group_MSE_six_patients_only':float(np.mean(np.sum((pred[gc==1]-y[gc==1])**2,axis=1)))})
    result={
      'item':'2-1R2','scope':'Public geometry and existing-head postprocessing only; no new labels or downstream efficacy.',
      'computed_utc':datetime.now(timezone.utc).isoformat(),
      'read_log':log,'new_private_queries':0,'new_private_releases':0,'V_reads':0,'new_labels':0,
      'population':{'patients':len(people),'images':len(d),'patient_class_rows':len(groups),'class_rows':Counter(map(int,gc)),
        'mixed_class_patients':int(sum(v>1 for v in pc.values())),'multi_visit_patient_class_groups':int(np.sum(sizes>1)),
        'max_visits_in_patient_class_group':int(sizes.max())},
      'weights':{},
      'clipping_order':clipstats,'covariance_decomposition':covariance,'checks':checks,
      'map_variants':mapstats,'public_target_difference_only':td,'public_patient_cv':cv,
      'note':'All variants use original HR per-cap lambda fixed. CV uses fixed full-P lambda and is a geometry diagnostic, not an independent unbiased method-performance estimate. Positive CV class has only six patients. Public target difference/CV cannot establish AUROC mechanism.'
    }
    for name,ww in [('patient_equal_groups',a),('class_balanced_groups',bc)]:
        patientweights=np.array([ww[gp==p].sum() for p in people])
        result['weights'][name]={'positive_class_mass':float(ww[gc==1].sum()),'group_ESS':float(1/(ww@ww)),
          'patient_ESS':float(1/(patientweights@patientweights)),'max_patient_mass':float(patientweights.max())}
    result['calculation_seconds']=time.perf_counter()-started
    (out/'public_map_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'public_geometry_targets_NOT_LABELS.npz',**targets)
    print(json.dumps({k:result[k] for k in ['population','weights','clipping_order','covariance_decomposition','checks','public_target_difference_only','public_patient_cv','calculation_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rr',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.root,args.rr,args.out)
