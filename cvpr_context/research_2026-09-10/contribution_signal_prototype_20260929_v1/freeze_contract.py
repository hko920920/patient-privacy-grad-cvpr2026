"""Freeze 2-4B's public-only protocol before running any method comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
THESIS=ROOT.parent.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def main():
    assert not (OUT/"contract.json").exists(), "Protocol already frozen"
    inputs={
      "selection":ROOT/"contribution_public128_control_20260928_v1/public128_selection.npz",
      "DINO_P":THESIS/"code_working/_reports/receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz",
      "RN_P":THESIS/"code_working/_reports/receiver_resnet18_reuse_20260924_v1/P_projected_features.npz"}
    expected={
      "selection":"7e49a9ba1ab1ea10ecb7ae6f8cd91792ccd458e14cb1da1534b31154370de3c4",
      "DINO_P":"4093700089740598561abe4fef5246c3f1626d9efc9165183872f157f7bfba6f",
      "RN_P":"bd07854ecc436ccea6845698f0087fd465c71ec163dceb7d3d720e9903626274"}
    assert {k:sha(p) for k,p in inputs.items()}==expected
    prior=json.loads((ROOT/"research_state.json").read_text(encoding="utf-8"))
    preserve=dict(prior["public_transfer_2_4A_design_20260929"]["preserved_evidence_sha256"])
    preserve["PUBLIC_TRANSFER_2_4A_SIGNAL_DESIGN_20260929.md"]="8af0452b9ae797334b19fe52e9cad33d3d2dba381f8b2bc4afb922bf2ba0df4d"
    for name,h in preserve.items(): assert sha(ROOT/name)==h, name
    now=datetime.now(timezone.utc).isoformat()
    contract={
      "item":"2-4B","created_utc":now,"start_utc":"2026-09-29T10:06:57+00:00","eta_minutes":[60,90],
      "scope":"Public-input mechanism and label prototype; NOT a private release or clinical utility experiment",
      "inputs":{k:{"path":str(p),"sha256":expected[k]} for k,p in inputs.items()},
      "preserved_evidence":preserve,
      "construction_models":["DINO64","ResNet18_128"],"carrier_n":128,"ridge":0.1,"rank":64,
      "public_geometry":"N=ker(X_DINO.T); M=H_R^.5 B_R N; U=orth(M), H_R=P_R.T Omega P_R+.1I",
      "patient_aggregation":"mean visits within patient/class; mean patients within class; signed contrast (class1-class0)/2",
      "public_center":"One common patient-equal mean of H_R^-.5 patient/class rows; split each patient's mass among observed classes; subtract before clipping, add after class-mean decoding. The common center cancels from the class contrast.",
      "center_rationale":"Pre-outcome strengthening: all controls may use the same public reference; this is a fixed public affine shift, NOT a novel component or a result-selected branch. 2-4A's descriptive uncentered caps are not reused.",
      "methods":{
        "FULL":"128D whitened centered vector -> class cap -> one joint patient numerator query -> project U",
        "COMPAT":"U.T centered vector -> class cap -> one joint patient numerator query",
        "PCA":"top64 eigenvectors of patient-equal centered public covariance -> class cap -> joint query -> lift to128 -> project U"},
      "cap_policies":{
        "shared_full_q95":"Primary: same per-class unweighted .95 quantile of full centered patient/class norms for all methods",
        "own_q95":"Predeclared secondary: each method's own per-class .95 quantile; same quantile rule, no search"},
      "cap_floor":1e-6,"quantile_method":"linear",
      "numerator_query":"t_i=concat(clip_C0(x_i0)/C0,clip_C1(x_i1)/C1)/sqrt(2); absent class zero; roundoff radius 1-16*float64eps; add/remove sensitivity1",
      "illustrative_privacy_only":{
        "numerator_epsilon":4.0,"numerator_delta":5e-6,
        "shared_source_epsilon":4.0,"shared_source_delta":5e-6,
        "shared_counts":"simulate the count coordinates of a normalized .5*[class0,class1,indicator0,indicator1] source release; count noise SD=2*sigma_source",
        "prospective_basic_total":[8.0,1e-5],
        "not_actual_release_contract":True,
        "fixed_public_source_baseline":"No source-noise efficacy modeled; y0 is P-only. Shared-count stress is conditional on fixed y0; does not model source-label/count correlation.",
        "warning":"Cannot append this mechanism to a historical epsilon8 source and call the result epsilon8."},
      "denominator":"One same simulated count vector per draw across all methods; max(1,noisy_count) per class; no extra norm-consistency projection. Conditional true-count analytic risk plus separate noisy-count finite stress.",
      "analytic_primary":"Expected squared retained-target error, conditional on true public counts; exact aggregate bias plus projected Gaussian covariance trace. Also report full-H objective's inaccessible constant. Not AUROC.",
      "simulations":{"draws":512,"seed":2026092901,"generator":"numpy.default_rng PCG64","coupling":"one ambient128 Gaussian per draw/class; project same coins into U/PCA; same shared count coins. Coupling only numerical variance reduction, not a joint-release privacy claim.","public_only_seed_can_be_recorded":True},
      "source_baseline":"Existing bounded DINO Label Solve applied to PUBLIC P target only, once, unchanged solver; no prior DP labels accessed",
      "compiler":"minnorm least squares in source nullspace; no new regularization. ell=alpha*(y0+Nv), alpha=1/max(1,max(abs(y))); report target error before scale and score-scale equality.",
      "generic_label_solve":"Independent equality-constrained/KKT solver with same y0, query, ridge and global scaling; if equivalent, report equivalence, do not manufacture another algorithm.",
      "unconstrained_reference":"Full-target unconstrained receiver Label Solve with same global scaling, noiseless public target only; measures source-preservation tradeoff, not a fair source-preserving competitor",
      "tests":["patient add/remove norm bound including multi-class/extreme/absent cases","independent Gaussian privacy-profile calibration","same-cap no-clipping FULL/COMPAT target-noise-law equality","projection-before-clipping row inequality","source primal/dual head and public-score preservation before scale / proportionality after scale","independent constrained Label Solve equivalence","invalid arrays/caps/counts rejected and noisy negative counts safely floored","input/evidence hashes unchanged"],
      "decision":"Implementation pass and public risk difference reported separately. Practical support requires lower primary analytic retained-target MSE than BOTH full and PCA, and same direction in fixed noisy-count simulation. Own-cap result cannot overwrite a failed primary. Any difference remains conditional on P with only six positive patients. No automatic Q/release/V expansion.",
      "forbidden":["Q raw or feature access","V clinical efficacy","DenseNet features","Expert/Reserved","private summary values","new encoder forwards","new synthetic pixels","public P replication to fake Q size","outcome-driven cap/rank/noise search"],
      "source_dependencies":{
        "label_solve":{"path":str(ROOT/"contribution_public128_control_20260928_v1/construct_labels.py"),"sha256":sha(ROOT/"contribution_public128_control_20260928_v1/construct_labels.py")},
        "gaussian_calibration":{"path":str(THESIS/"code_working/relation_distillation/moments.py"),"sha256":sha(THESIS/"code_working/relation_distillation/moments.py")}}
    }
    save(OUT/"contract.json",contract)
    save(OUT/"contract_seal.json",{"created_utc":now,"sha256":sha(OUT/"contract.json"),"phase":"BEFORE_METHOD_RESULTS"})
    save(OUT/"state_before.json",prior)
    prior["active_execution"]={"status":"running_public_prototype","item":"2-4B","started_utc":contract["start_utc"],"eta_minutes":[60,90],"out_dir":str(OUT),"Q_V_release_Expert_Reserved":False,"latest_clinical_result_remains":"2-2R2"}
    prior["numbered_execution_plan_20260928"]["upper_2_supplements"]["2-4B"]={"status":"running_public_only","eta_minutes":[60,90],"contract":str(OUT/"contract.json")}
    prior["numbered_execution_plan_20260928"]["current_item"]="2-4B public-input prototype and matched controls executing; no Q/V/private release"
    save(ROOT/"research_state.json",prior)
    banner="**2-4B STARTED — 공개 입력 prototype·대조 (2026-09-29)**\n\n사용자 승인과 예상60~90분 안내 후 진행. 고정 public128/P 특징만 사용해 source 보존 라벨·patient query·공유 count 후처리와 full/PCA 대조를 구현한다. 동일 cap 귀무 대조와 각 방식의 같은 공개 cap 선정 규칙을 사전 고정한다. 공개 모의 잡음512회는 실제 DP release·합성 bank·V 평가가 아니다. Q/V/DenseNet/Expert/Reserved/새 forward0. 과거 효용 결과는2-2R2로 유지한다.\n\n"
    for p in [THESIS/"CURRENT_STATUS.md",THESIS/"WORKLOG.md"]:
        text=p.read_text(encoding="utf-8")
        first,sep,rest=text.partition("\n")
        p.write_text(first+"\n\n"+banner+rest,encoding="utf-8")
    print(json.dumps({"contract_sha256":sha(OUT/"contract.json"),"scope":contract["scope"],"created_utc":now},ensure_ascii=False))
if __name__=="__main__": main()

