"""2-1R1 public/head-only algebra. No labels, V scores or fresh DP means."""
import datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
rr=out.parent; tr=rr.parents[1]; reports=tr/'code_working'/'_reports'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paths={
 'P_DINO':reports/'receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz',
 'P_RN':reports/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
 'P_target':reports/'receiver_feature_followup_20260924_v1/public_target/target.npz',
 'old_targets':rr/'contribution_kd_control_20260928_v1/target_weights.npz',
 'public_maps':rr/'contribution_search_20260924/transport_public_maps_and_dp_weights.npz',
 'HR_targets':rr/'contribution_head_reconstruction_20260928_v1/HR_target_weights.npz'}
assert not (out/'public_algebra.json').exists()
meta={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}
dump(out/'algebra_inputs.json',{'script_sha256':sha(__file__),'contract_sha256':sha(out/'contract.json'),'inputs':meta})
def read(k,keys=None):
 with (out/'algebra_access.jsonl').open('a',encoding='utf-8') as f:
  f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input':k,'keys':keys})+'\n')
 with np.load(paths[k],allow_pickle=False) as z:return {n:z[n] for n in (keys or z.files)}
t0=time.monotonic()
pd=read('P_DINO');pr=read('P_RN')
assert set(pd['roles'])==set(pr['roles'])=={'P'}
for k in ['patient_ids','image_ids','labels']:assert np.array_equal(pd[k].astype(str),pr[k].astype(str))
sys.path.insert(0,str(rr/'contribution_search_20260924'))
from fixed_image_compilation import pw
omega=pw(pd);assert np.array_equal(omega,pw(pr))
D=pd['z'].transpose(1,0,2).reshape(813,64).astype(float);R=pr['z'].astype(float)
G=D.T@(omega[:,None]*D);HD=G+.1*np.eye(64);HR=R.T@(omega[:,None]*R)+.1*np.eye(128)
C=R.T@(omega[:,None]*D)
maps=read('public_maps',['ResNet18_T0','ResNet18_T1','ResNet18_b0','ResNet18_b1'])
T=[maps['ResNet18_T'+str(k)] for k in (0,1)];b=[maps['ResNet18_b'+str(k)] for k in (0,1)]
muP=read('P_target')['target'].reshape(2,64).astype(float)
dP=.5*(muP[1]-muP[0]);wDP=np.linalg.solve(HD,dP)
wr_map_P=np.linalg.solve(HR,.5*(muP[1]@T[1]+b[1]-muP[0]@T[0]-b[0]))
wr_real_P=np.linalg.solve(HR,R.T@(omega*(2*pd['labels']-1)))
AK=np.linalg.solve(HR,C)
AM=np.linalg.solve(HR,.5*(T[1]+T[0]).T@HD)
lam=.001*np.trace(G)/64
AC=np.linalg.solve(HR,C@np.linalg.solve(G+lam*np.eye(64),HD))
old=read('old_targets',['DP1_DINO','DP2_DINO','DP1_KD_RN','DP2_KD_RN'])
hr=read('HR_targets')
def rms(x):return float(np.sqrt(omega@(x*x)))
results={};hr_error=0.;kd_error=0.;corr_error=0.
for draw in ['DP1','DP2']:
 wd=old[draw+'_DINO'];delta=wd-wDP
 # Exact current HR = its mapped public anchor plus source-head update.
 wr=wr_map_P+AM@delta
 hr_error=max(hr_error,float(abs(wr-hr[draw+'_HR_RN_ONLY']).max()))
 kd_error=max(kd_error,float(abs(AK@wd-old[draw+'_KD_RN']).max()))
 corrected=AC@wd
 q=D@np.linalg.solve(G+lam*np.eye(64),HD@wd)
 independent=np.linalg.solve(HR,R.T@(omega*q))
 corr_error=max(corr_error,float(abs(corrected-independent).max()))
 # Differences here are PUBLIC prediction geometry only, not task efficacy.
 diff_anchor=R@(wr_map_P-AK@wDP);diff_operator=R@((AM-AK)@delta)
 total=R@(wr-AK@wd)
 assert np.max(abs(total-diff_anchor-diff_operator))<1e-12
 results[draw]={
   'HR_public_prediction_RMS':rms(R@wr),
   'HR_minus_KD_public_prediction_RMS':rms(total),
   'mapped_anchor_correction_RMS':rms(diff_anchor),
   'operator_change_on_update_RMS':rms(diff_operator),
   'signed_cross_term_in_squared_RMS':float(2*omega@(diff_anchor*diff_operator)),
   'source_update_public_prediction_RMS':rms(D@delta),
   'corrected_vs_KD_public_prediction_RMS':rms(R@(corrected-AK@wd)),
   'scope':'No AUROC/AP, no labels or V. Terms are not utility contributions.'}
eig=np.linalg.eigvalsh(G)
checks={'HR_anchor_plus_update_max_abs':hr_error,'old_KD_max_abs':kd_error,'corrected_dual_formula_max_abs':corr_error}
assert max(checks.values())<1e-10
for k,p in paths.items():assert sha(p)==meta[k]['sha256']
res={'item':'2-1R1','status':'PASS','checks':checks,'public_raw_covariance_ridge':float(lam),
 'public_source_covariance_eigenvalue_min_max':[float(eig.min()),float(eig.max())],
 'source_score_shrink_eigenvalue_min_max':[float((eig/(eig+.1)).min()),float((eig/(eig+.1)).max())],
 'corrected_score_filter_min_max':[float((eig/(eig+lam)).min()),float((eig/(eig+lam)).max())],
 'mapped_vs_actual_public_RN_anchor_prediction_RMS':rms(R@(wr_map_P-wr_real_P)),
 'cases':results,'new_labels':0,'new_V_evaluations':0,'new_Q_access':0,'new_DP_releases':0,
 'Expert_Reserved':False,'seconds':time.monotonic()-t0}
dump(out/'public_algebra.json',res)
print(json.dumps(res,ensure_ascii=True,indent=2))

