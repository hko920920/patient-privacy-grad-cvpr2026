"""Post-hoc stage-2 decomposition of saved patient LR scales and coefficients.

No fitting, score-direction selection, GPU, target training, or causal claim.
Two arithmetic paths and two bootstrap/rank paths are checked before reporting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
RESEARCH = CODE.parent / "CVPR 주제 탐색/research_2026-09-10"
SCREEN = CODE / "_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914"
EU = SCREEN / "cdi_e_cohort_v1/eu_analysis_v1"
CONTRACT = RESEARCH / "spec_sources/cdi_eu_scaling_audit_contract_20260915.json"
OUTPUT = EU / "scaling_coefficient_audit_v1"
METHODS = ("patient_mean26_fixed", "patient_mean26_tuned", "patient_meanmax52_fixed", "patient_meanmax52_tuned")
MODELS = ("model_1", "model_2")
COMBOS = (("E", "E"), ("E", "U"), ("U", "E"), ("U", "U"))
EDGES = (
    ("rescale_E_coefficients", ("E", "U"), ("E", "E")),
    ("change_coefficients_at_U_scale", ("U", "U"), ("E", "U")),
    ("change_coefficients_at_E_scale", ("U", "E"), ("E", "E")),
    ("rescale_U_coefficients", ("U", "U"), ("U", "E")),
    ("full_E_to_U_scorer_change", ("U", "U"), ("E", "E")),
)


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def tight(a, b, label):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a, b, rtol=2e-12, atol=2e-12), label)


def rank_changes(a, b):
    i, j = np.triu_indices(len(a), 1)
    return int(np.count_nonzero(np.sign(a[i] - a[j]) != np.sign(b[i] - b[j])))


def prepare():
    need(not CONTRACT.exists(), "Immutable pre-result contract exists")
    c = read(EU / "contract_copy.json")
    files = [EU / n for n in ("contract_copy.json", "analysis.json", "verification.json", "provenance.json", "protocol.json",
                              "fit_parameters.json", "predictions.json", "bootstrap_resamples.json")]
    for scenario in ("E", "U"):
        d = Path(c["inputs"][scenario]["directory"])
        files += [d / "results.json", d / c["inputs"][scenario]["verification_file"]]
    write(CONTRACT, dict(schema="cdi-eu-scaling-coefficient-audit-contract/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        research_stage=2, posthoc=True, motivated_by_verified_EU_comparison=True,
        purpose="Distinguish saved scaler changes from saved standardized coefficient changes in existing patient LR scorers",
        methods=list(METHODS), models=list(MODELS), evaluation_scenarios=["E", "U"],
        combinations=[dict(coefficient_source=a, scale_source=b) for a,b in COMBOS],
        formula="logit=((x-mu_scale_source)/sigma_scale_source) dot coef_coefficient_source + intercept_coefficient_source",
        raw_linear_weight="coef_coefficient_source/sigma_scale_source",
        bias="intercept_coefficient_source - sum(raw_linear_weight * mu_scale_source); constant across patients",
        primary_metric="AUC of logits; original diagonal probability/rank/AUC agreement must be verified",
        contrasts=[dict(id=i,a=list(a),b=list(b)) for i,a,b in EDGES],
        expected_AUC=64, expected_contrasts=80, selection_patients=40, fit_patients_per_saved_scorer=80,
        bootstrap="Exact saved 2000 A20/B20 patient draws, conditional on saved scorers, shared for every combination",
        descriptive_parameters="All feature sigma_U/sigma_E and sqrt(var_U)/sqrt(var_E) ratios, standardized/effective coefficient cosines and norms",
        input_sha256={str(p.resolve()):sha(p) for p in files}, source_sha256=sha(__file__), output_directory=str(OUTPUT),
        constraints=["No new fitting, hyperparameter choice, sign tuning, threshold selection, test/calibration data, or GPU",
            "All four preselected patient-LR methods and all four coefficient/scaler combinations are reported",
            "A scalar threshold shift cannot change AUC ranks; centering/intercept are constant for a fixed patient LR scorer",
            "Hybrid weights are diagnostic off-training-combination interventions, not causal patient participation interventions",
            "U scaler and coefficient estimates retain existing U fit80 auxiliary data/labels; no guarantee of realistic label-free adaptation",
            "Coefficient-source change includes all fitted effects, including the preselected C choice for tuned methods",
            "Same selection40 has been reused; intervals are exploratory, not multiplicity adjusted and omit fitting uncertainty",
            "The decomposition is path dependent; neither path supplies a unique causal attribution or novel attack contribution"],
        new_fitting=False, new_GPU_calls=0, stage2_complete=False))
    print(json.dumps(dict(status="FROZEN_POSTHOC_POLICY_BEFORE_NEW_METRICS", contract_sha256=sha(CONTRACT), source_sha256=sha(__file__))))


def design_two_paths(raw, method):
    primary = raw.mean(1)
    independent = np.asarray([[math.fsum((float(a),float(b)))/2 for a,b in zip(pair[0],pair[1])] for pair in raw])
    if "meanmax52" in method:
        primary = np.c_[primary, raw.max(1)]
        independent = np.c_[independent, np.asarray([[max(float(a),float(b)) for a,b in zip(pair[0],pair[1])] for pair in raw])]
    tight(primary, independent, "Independent raw mean/max representation")
    return primary, independent


def logits_two_paths(x, independent_x, coefficient, scaler):
    coef = np.asarray(coefficient["coef"][0], dtype=float)
    mu, scale = np.asarray(scaler["scaler_mean"]), np.asarray(scaler["scaler_scale"])
    intercept = float(coefficient["intercept"][0])
    need((scale > 0).all(), "Positive stored StandardScaler scales")
    direct = ((x - mu) / scale) @ coef + intercept
    weight = np.asarray([float(a)/float(b) for a,b in zip(coef,scale)])
    bias = intercept - math.fsum(float(w)*float(m) for w,m in zip(weight,mu))
    alternate = np.asarray([math.fsum(float(v)*float(w) for v,w in zip(row,weight))+bias for row in independent_x])
    tight(direct, alternate, "Standardized dot vs independent math.fsum raw-linear score")
    need(rank_changes(direct, alternate) == 0, "Independent arithmetic preserves all patient ranks")
    return direct, weight, bias, float(np.max(np.abs(direct-alternate)))


def metric_two_paths(scores, y, indices, counts):
    p,n = np.flatnonzero(y==1),np.flatnonzero(y==0)
    credit = (scores[p,None] > scores[None,n]).astype(float) + .5*(scores[p,None] == scores[None,n])
    point = float(credit.mean())
    tight(point, roc_auc_score(y,scores), "Independent sklearn point AUC")
    weighted = np.einsum("bi,ij,bj->b", counts[:,p], credit, counts[:,n]) / 400
    boot_y = y[indices]
    need(np.all(boot_y==boot_y[0]), "Shared stratified A/B bootstrap column identities")
    pos = scores[indices[:,boot_y[0]==1]]
    neg = scores[indices[:,boot_y[0]==0]]
    explicit = ((pos[:,:,None] > neg[:,None,:]).astype(float) + .5*(pos[:,:,None] == neg[:,None,:])).mean(axis=(1,2))
    tight(weighted, explicit, "Count-weight vs explicit resampled patient-pair bootstrap")
    return point, weighted, explicit


def cosine(a,b):
    numerator = math.fsum(float(x)*float(y) for x,y in zip(a,b))
    denominator = math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))
    return None if denominator==0 else numerator/denominator


def run(expected_code, expected_contract):
    start=time.perf_counter()
    need(sha(__file__)==expected_code and sha(CONTRACT)==expected_contract, "Externally frozen source and policy")
    policy=read(CONTRACT)
    need(policy["source_sha256"]==expected_code and not OUTPUT.exists(), "Frozen source / immutable output")
    def check_hashes():
        for p,h in policy["input_sha256"].items():need(sha(p)==h,"Changed frozen input "+p)
        need(sha(__file__)==expected_code and sha(CONTRACT)==expected_contract,"Source or contract changed")
    check_hashes()
    v=read(EU/"verification.json")
    need(v["status"]=="PASS_EU_SAVED_PREDICTIONS_FIT_AND_STATISTICS"
         and v["analysis_sha256"]==sha(EU/"analysis.json")
         and v["analysis_provenance_sha256"]==sha(EU/"provenance.json"),"Verified E/U analysis")
    OUTPUT.mkdir()
    write(OUTPUT/"protocol.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=expected_code, contract_sha256=expected_contract, contract=policy,
        execution_policy="All declared outputs; no fitting/GPU; independently checked arithmetic and resampling before release"))
    c=read(EU/"contract_copy.json");patients=[p for p in c["patient_rows"] if p["role"]=="selection"]
    need(len(patients)==40,"40 selection patients")
    par={(r["fit_scenario"],r["model"],r["method"]):r for r in read(EU/"fit_parameters.json")}
    oldpred={(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"],r["patient_id"]):r["score"] for r in read(EU/"predictions.json")}
    oldauc={(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"]):r["selection_patient_AUC"] for r in read(EU/"analysis.json")["results"]}
    boot=read(EU/"bootstrap_resamples.json");indices=np.asarray(boot["indices"],dtype=int)
    need(boot["selection_patient_order"]==[p["patient_id"] for p in patients] and indices.shape==(2000,40),"Original paired bootstrap")
    counts=np.asarray([np.bincount(i,minlength=40) for i in indices],dtype=float)
    raw={};labels={};feature_names=None
    for es in ("E","U"):
        d=Path(c["inputs"][es]["directory"]);rv=read(d/c["inputs"][es]["verification_file"])
        need(rv["status"]==c["inputs"][es]["required_status"] and rv["results_sha256"]==sha(d/"results.json"),"Exact raw features verified")
        ix={(r["model"],r["image_id"]):r for r in read(d/"results.json")}
        for model in MODELS:
            yy=[];rr=[]
            for p in patients:
                label=int(p["group"]==("A" if model=="model_1" else "B"));pair=[]
                for image in p["images"][es]:
                    r=ix[model,image]
                    need(r["patient_id"]==p["patient_id"] and r["scenario"]==es and r["eval_role"]=="selection", "Patient/image identity")
                    need(r["member"]==int(r["patient_training_exposures"]>0)==label,"Patient participation")
                    need(r["image_member"]==int(r["actual_training_exposures"]>0)==(label if es=="E" else 0),"E/U exposure semantics")
                    if feature_names is None:feature_names=r["feature_names"]
                    need(r["feature_names"]==feature_names and len(r["features"])==26,"Exact feature order")
                    pair.append(r["features"])
                need(len(pair)==2,"Two images per patient");rr.append(pair);yy.append(label)
            raw[es,model]=np.asarray(rr,dtype=float);labels[model]=np.asarray(yy)
    scores=[];metrics=[];contrasts=[];parameters=[];checks=[];allboots={};allpoints={}
    for model in MODELS:
        for method in METHODS:
            ep,up=(par[s,model,method] for s in ("E","U"))
            pp={"E":ep["parameters"],"U":up["parameters"]}
            for entry in (ep,up):
                need(entry["fitting_rows"]==entry["parameters"]["scaler_n_samples_seen"]==80
                     and len(entry["fitting_patient_ids"])==80 and not set(entry["fitting_patient_ids"])&set(boot["selection_patient_order"]),"Saved fit80 only")
                need(entry["parameters"]["classes"]==[0,1],"Saved member-positive direction")
            weights={}
            for cs,ss in COMBOS:weights[cs,ss]=np.asarray(pp[cs]["coef"][0])/np.asarray(pp[ss]["scaler_scale"])
            names=["mean:"+n for n in feature_names]+(["max:"+n for n in feature_names] if "meanmax52" in method else [])
            ratios=np.asarray(pp["U"]["scaler_scale"])/np.asarray(pp["E"]["scaler_scale"])
            stdE,stdU=np.sqrt(pp["E"]["scaler_var"]),np.sqrt(pp["U"]["scaler_var"])
            parameters.append(dict(model=model,method=method,E_chosen_C=ep["chosen_C"],U_chosen_C=up["chosen_C"],
                standardized_coefficient_cosine=cosine(pp["E"]["coef"][0],pp["U"]["coef"][0]),
                effective_coefficient_cosines=[dict(a=list(a),b=list(b),cosine=cosine(weights[a],weights[b])) for i,a in enumerate(COMBOS) for b in COMBOS[i+1:]],
                effective_coefficients=[dict(coefficient_source=a,scale_source=b,values=weights[a,b].tolist()) for a,b in COMBOS],
                scale_ratio_U_over_E_summary=dict(minimum=float(ratios.min()),median=float(np.median(ratios)),maximum=float(ratios.max())),
                features=[dict(feature=n,scale_E=float(se),scale_U=float(su),scale_ratio_U_over_E=float(r),
                    fit_std_E=float(ve),fit_std_U=float(vu),fit_std_ratio_U_over_E=None if ve==0 else float(vu/ve))
                    for n,se,su,r,ve,vu in zip(names,pp["E"]["scaler_scale"],pp["U"]["scaler_scale"],ratios,stdE,stdU)]))
            for es in ("E","U"):
                x,altx=design_two_paths(raw[es,model],method);y=labels[model]
                for cs,ss in COMBOS:
                    z,w,bias,error=logits_two_paths(x,altx,pp[cs],pp[ss]);prob=expit(z)
                    point,bw,be=metric_two_paths(z,y,indices,counts)
                    key=(model,method,es,cs,ss);allboots[key]=(bw,be);allpoints[key]=point
                    check=dict(model=model,method=method,evaluation_scenario=es,coefficient_source=cs,scale_source=ss,
                        independent_logit_max_abs_error=error,probability_vs_logit_rank_changes=rank_changes(prob,z),
                        saturated_probabilities=int(np.count_nonzero((prob==0)|(prob==1))),original_diagonal=cs==ss)
                    if cs==ss:
                        old=np.asarray([oldpred[cs,es,model,method,p["patient_id"]] for p in patients])
                        tight(prob,old,"Original diagonal probabilities")
                        need(rank_changes(z,old)==0,"Original diagonal full patient ranks")
                        tight(point,oldauc[cs,es,model,method],"Original diagonal AUC")
                        check["old_probability_max_abs_error"]=float(np.max(np.abs(prob-old)))
                    checks.append(check)
                    metrics.append(dict(model=model,method=method,evaluation_scenario=es,coefficient_source=cs,scale_source=ss,
                        patient_AUC=point,conditional_CI95=np.quantile(bw,[.025,.975]).tolist(),patients=40,
                        bias=bias,primary_score="logit"))
                    for i,p in enumerate(patients):scores.append(dict(model=model,method=method,evaluation_scenario=es,
                        coefficient_source=cs,scale_source=ss,patient_id=p["patient_id"],group=p["group"],member=int(y[i]),
                        logit=float(z[i]),probability=float(prob[i])))
                for name,a,b in EDGES:
                    ka,kb=(model,method,es,*a),(model,method,es,*b)
                    delta=allpoints[ka]-allpoints[kb]
                    bw=allboots[ka][0]-allboots[kb][0];be=allboots[ka][1]-allboots[kb][1]
                    tight(np.quantile(bw,[.025,.975]),np.quantile(be,[.025,.975]),"Independent paired contrast CI")
                    contrasts.append(dict(model=model,method=method,evaluation_scenario=es,id=name,a=list(a),b=list(b),
                        delta_AUC=delta,paired_conditional_CI95=np.quantile(bw,[.025,.975]).tolist()))
                d={(r["id"]):r["delta_AUC"] for r in contrasts if r["model"]==model and r["method"]==method and r["evaluation_scenario"]==es}
                tight(d["rescale_E_coefficients"]+d["change_coefficients_at_U_scale"],d["full_E_to_U_scorer_change"],"First path exact AUC telescope")
                tight(d["change_coefficients_at_E_scale"]+d["rescale_U_coefficients"],d["full_E_to_U_scorer_change"],"Second path exact AUC telescope")
    need(len(metrics)==64 and len(contrasts)==80 and len(scores)==2560 and len(parameters)==8,"Complete predeclared scope")
    check_hashes()
    write(OUTPUT/"patient_scores.json",scores);write(OUTPUT/"parameter_descriptives.json",parameters)
    write(OUTPUT/"analysis.json",dict(schema="cdi-eu-scaling-coefficient-audit/v1",status="PASS_TWO_ARITHMETIC_AND_BOOTSTRAP_PATHS",
        research_stage=2,posthoc=True,results=metrics,contrasts=contrasts,checks=checks,
        interpretation_limits=policy["constraints"],new_fitting=False,new_GPU_calls=0,stage2_complete=False,
        seconds_cpu=time.perf_counter()-start))
    write(OUTPUT/"verification.json",dict(status="PASS_INDEPENDENT_ARITHMETIC_RANK_AND_BOOTSTRAP_PATHS",source_sha256=expected_code,
        contract_sha256=expected_contract,original_diagonal_conditions=32,diagonal_rank_changes=0,
        independent_AUC_checks=64,paired_contrast_checks=80,bootstrap_resamples=2000,patient_score_rows=2560,
        numerical_paths=["NumPy standardized dot","Independent mean/max and raw coefficient division with math.fsum"],
        rank_paths=["sklearn roc_auc_score","400 member/nonmember pair credits"],
        bootstrap_paths=["Patient multiplicity weighted 400-pair matrix","Direct reindexed 20x20 pair comparisons for each saved draw"],
        verification_scope="Separate implementations inside this CPU module; no external agent or independent target-forward rerun"))
    write(OUTPUT/"provenance.json",dict(source_sha256=expected_code,contract_sha256=expected_contract,input_sha256=policy["input_sha256"],
        output_sha256={n:sha(OUTPUT/n) for n in ("protocol.json","analysis.json","patient_scores.json","parameter_descriptives.json","verification.json")}))
    print(json.dumps(dict(status="PASS_TWO_ARITHMETIC_AND_BOOTSTRAP_PATHS",AUC=64,contrasts=80,output=str(OUTPUT),seconds_cpu=time.perf_counter()-start)))


def self_test():
    rng=np.random.default_rng(37);raw=rng.normal(size=(40,2,26));x,alt=design_two_paths(raw,"patient_meanmax52_fixed")
    pp=dict(coef=[rng.normal(size=52).tolist()],intercept=[.2],scaler_mean=rng.normal(size=52).tolist(),scaler_scale=rng.uniform(.5,2,size=52).tolist())
    logits_two_paths(x,alt,pp,pp)
    y=np.r_[np.ones(20,dtype=int),np.zeros(20,dtype=int)]
    indices=np.asarray([np.r_[rng.choice(20,20),20+rng.choice(20,20)] for _ in range(2000)])
    counts=np.asarray([np.bincount(i,minlength=40) for i in indices])
    metric_two_paths(rng.integers(0,7,size=40).astype(float),y,indices,counts)
    metric_two_paths(np.zeros(40),1-y,indices,counts)
    need(rank_changes(np.arange(40),np.arange(40)+17)==0,"Bias cannot change rank")
    print("PASS_SYNTHETIC_LINEAR_IDENTITY_TIES_STRATIFIED_BOOTSTRAP; no actual mixed metrics read")


if __name__=="__main__":
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract",action="store_true");mode.add_argument("--self-test",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--expected-code-sha256");parser.add_argument("--expected-contract-sha256")
    a=parser.parse_args()
    if a.prepare_contract:prepare()
    elif a.self_test:self_test()
    else:run(a.expected_code_sha256,a.expected_contract_sha256)
