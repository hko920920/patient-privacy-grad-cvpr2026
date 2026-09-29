"""2-1 algebra only: P caches and existing DP outputs. No V or new labels."""
import collections, datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent; rr=out.parent; tr=rr.parents[1]; r=tr/'code_working'/'_reports'
sys.path.insert(0,str(rr/'contribution_search_20260924'))
from public_transport_probe import fit
from geometry_diagnostic import moments

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths={
 'P_DINO':r/'receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz',
 'P_RN':r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
 'P_target':r/'receiver_feature_followup_20260924_v1/public_target/target.npz',
 'DP1':r/'receiver_feature_dp8_20260924_v1/one_release/protected_target/target.npz',
 'DP2':r/'receiver_feature_followup_20260924_v1/one_release/protected_target/target.npz',
 'old_targets':rr/'contribution_kd_control_20260928_v1/target_weights.npz',
 'old_maps':rr/'contribution_search_20260924/transport_public_maps_and_dp_weights.npz'}
assert not (out/'operation_algebra.json').exists()
inputs={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}
(out/'algebra_inputs.json').write_text(json.dumps({'contract_sha256':sha(out/'contract.json'),'script_sha256':sha(__file__),'inputs':inputs},indent=2),encoding='utf-8')
def read(k):
 p=paths[k]
 with (out/'access_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'item':'2-1','input':k})+'\n')
 with np.load(p,allow_pickle=False) as z:return {n:z[n] for n in z.files}
start=time.monotonic();src=read('P_DINO');rn=read('P_RN')
assert set(src['roles'].tolist())==set(rn['roles'].tolist())=={'P'}
for k in ['image_ids','patient_ids','labels']:assert np.array_equal(src[k].astype(str),rn[k].astype(str))
z=src['z'].transpose(1,0,2).reshape(-1,64).astype(float)
R=rn['z'].astype(float);pids=src['patient_ids'].astype(str);labs=src['labels']
groups=[];rowpid=[];rowclass=[]
people=sorted(set(pids))
for p in people:
 for c in (0,1):
  ix=np.flatnonzero((pids==p)&(labs==c))
  if len(ix):groups.append(ix);rowpid.append(p);rowclass.append(c)
cnt=collections.Counter(rowpid);a=np.array([1/len(people)/cnt[p] for p in rowpid]);A=np.diag(a)
X=np.stack([z[ix].mean(0) for ix in groups]);Y=np.stack([R[ix].mean(0) for ix in groups])
Cs=[1.9061329951133719,1.5917066731966771];norms=np.linalg.norm(X,axis=1)
T=[];b=[];mx=[];Gc=[];lam=[];Xc=[]
for C in Cs:
 xx=X*np.minimum(1,C/np.maximum(norms,1e-300))[:,None]
 tt,bb,ll=fit(xx,Y,a);T.append(tt);b.append(bb);lam.append(ll);mx.append(a@xx)
 xc=xx-mx[-1];Gc.append(xc.T@(a[:,None]*xc));Xc.append(xc)
old=read('old_maps');maperror=max(float(abs(T[c]-old['ResNet18_T'+str(c)]).max()) for c in (0,1))
maperror=max(maperror,max(float(abs(b[c]-old['ResNet18_b'+str(c)]).max()) for c in (0,1)))
assert maperror<1e-12
omega=np.zeros(len(labs))
for c in (0,1):
 ps=sorted(set(pids[labs==c]))
 for p in ps:
  ix=np.flatnonzero((pids==p)&(labs==c));omega[ix]=.5/len(ps)/len(ix)
assert abs(omega.sum()-1)<1e-14
HD=z.T@(omega[:,None]*z)+.1*np.eye(64)
HR=R.T@(omega[:,None]*R)+.1*np.eye(128)
cov,bp=moments(rn,True);assert np.max(abs(HR-(cov+.1*np.eye(128))))<1e-14
oldtargets=read('old_targets');public=read('P_target')['target'].reshape(2,64);sp=public.mean(0)
case={};err=0.;decomp_err=0.;source_inv_err=0.
for name in ['DP1','DP2']:
 mu=read(name)['target'].reshape(2,64);d=(mu[1]-mu[0])/2;s=mu.mean(0)
 delta=(T[1]+T[0]).T@d/2
 common=(T[1]-T[0]).T@s/2
 intercept=(b[1]-b[0])/2
 direct=((mu[1]@T[1]+b[1])-(mu[0]@T[0]+b[0]))/2
 decomp_err=max(decomp_err,float(abs(direct-delta-common-intercept).max()))
 w=np.linalg.solve(HR,direct);wd=np.linalg.solve(HD,d)
 err=max(err,float(abs(w-oldtargets[name+'_MT_RN']).max()),float(abs(wd-oldtargets[name+'_DINO']).max()))
 alpha=[a+a*(Xc[c]@np.linalg.solve(Gc[c]+lam[c]*np.eye(64),mu[c]-mx[c])) for c in (0,1)]
 v=np.zeros(len(labs))
 for j,ix in enumerate(groups):v[ix]=.5*(alpha[1][j]-alpha[0][j])/len(ix)
 q=v/omega
 wr=np.linalg.solve(HR,R.T@(omega*q));err=max(err,float(abs(w-wr).max()))
 # Exact teacher-weight inversion recovers the class contrast, not the two means.
 source_inv_err=max(source_inv_err,float(abs(HD@wd-d).max()))
 # A source-contrast-preserving diagnostic: replace only the class midpoint by its public counterpart.
 b_contrast=delta+(T[1]-T[0]).T@sp/2+intercept
 w_contrast=np.linalg.solve(HR,b_contrast)
 pub_full=R@w;pub_change=R@(w-w_contrast)
 # Algebraic collision witness: same d under a common perturbation of both class means.
 _,sv,vh=np.linalg.svd((T[1]-T[0]).T,full_matrices=False)
 h=1e-3*vh[0]
 witness_delta=.5*((mu[1]+h)-(mu[0]+h))
 witness_target=.5*((mu[1]+h)@T[1]+b[1]-(mu[0]+h)@T[0]-b[0])
 case[name]={'target_reconstruction_max_abs':float(abs(w-oldtargets[name+'_MT_RN']).max()),
   'signed_public_label_ridge_max_abs':float(abs(w-wr).max()),'public_signed_label_range':[float(q.min()),float(q.max())],
   'class_weight_sums':[float(x.sum()) for x in alpha],
   'midpoint_replacement_public_prediction_RMS_relative':float(np.sqrt(omega@(pub_change**2)/(omega@(pub_full**2)))),
   'midpoint_replacement_target_weight_relative':float(np.linalg.norm(w-w_contrast)/np.linalg.norm(w)),
   'common_shift_witness_source_contrast_max_abs':float(abs(witness_delta-d).max()),
   'common_shift_witness_MT_moment_l2':float(np.linalg.norm(witness_target-direct)),
   'scope':'Existing DP targets only; public prediction geometry, no AUROC/AP, no label solve. Witness is algebraic, not a fresh private-data realization.'}
checks={'map_exact_max_abs':maperror,'MT_and_source_and_signed_label_max_abs':err,
 'class_mean_decomposition_max_abs':decomp_err,'source_head_inversion_max_abs':source_inv_err}
assert max(checks.values())<1e-10
for k,p in paths.items():assert sha(p)==inputs[k]['sha256']
result={'item':'2-1','status':'PASS','P_patients':len(people),'P_images':len(labs),'patient_class_rows':len(groups),
 'public_class_clip_active_rows':[int((norms>C).sum()) for C in Cs],
 'public_map_ridge':lam,'public_map_difference_frobenius':float(np.linalg.norm(T[1]-T[0])),
 'public_map_difference_spectral':float(sv[0]),'cases':case,'checks':checks,'inputs_unchanged':True,
 'new_Q_access':0,'new_releases':0,'new_labels':0,'new_V_evaluations':0,'seconds':time.monotonic()-start}
(out/'operation_algebra.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))
