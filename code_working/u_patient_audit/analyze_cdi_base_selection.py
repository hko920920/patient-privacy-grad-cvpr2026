"""Frozen-scorer base A/B assignment control. No fitting or membership claim for base."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
TARGET = RUN / "baseline_screen_20260914/cdi_u_cohort_v1"
DEFAULT = RUN / "baseline_screen_20260914/cdi_base_selection_v1"
MODELS = ["model_1", "model_2"]

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def analysis_policy():
    return dict(schema="cdi-base-fixed-scorer-analysis-policy/v1", current_step=2,
        question="Can pretrained base features distinguish the fixed A/B assignment without our additional training?",
        selection_patients=40, images_per_patient=2, frozen_target_fitted_scorers=20,
        representations=["image26", "mean26", "meanmax52", "meanmax54"],
        base_label="A/B assignment in the positive direction of the original fit-target; not base membership",
        incremental_base_membership=0, pretrained_membership="unknown",
        scores="Saved scaler/coef/intercept/C; sigmoid then original image probability mean/max if applicable",
        metrics=["AUC_base_own_assignment", "AUC_own_target", "AUC_common_own_assignment",
                 "own_minus_base_AUC", "common_minus_base_AUC", "Spearman_base_own",
                 "Spearman_base_common", "all_400_AB_pair_exact_half_tie_credit_changes"],
        comparisons="patient_meanmax52 minus cdi_image_mean, fixed/tuned, each fit-target; own gain minus base gain",
        bootstrap="original saved 2000 A20/B20 patient draws, seed260914, conditional frozen-scorer percentile95",
        common_definition="(saved same-scorer target1 probability + target2 probability)/2",
        no_new_fitting=True, no_new_direction=True, no_success_gate=True,
        descriptive_development_only=True, causal_identification=False,
        scorer_information="Scalers and LR were learned from target fit80; only queried features are from base",
        interpretation_limits="Persistence is available A/B information, not training-free entire pipeline; absence may reflect base-target feature/scaling shift; pretraining membership unknown",
        original_primary_unchanged=True)

def representation(x, name):
    if name == "image26":
        return x[:, :, :26].reshape(-1, 26)
    if name == "mean26":
        return x[:, :, :26].mean(1)
    require(name in ("meanmax52", "meanmax54"), "known frozen representation")
    z = x[:, :, :26] if name == "meanmax52" else x
    return np.concatenate((z.mean(1), z.max(1)), 1)

def predict(x, spec, par):
    z = representation(x, spec["representation"])
    p = expit(((z-np.asarray(par["scaler_mean"]))/np.asarray(par["scaler_scale"]))
              @ np.asarray(par["coef"])[0] + par["intercept"][0])
    require(par["classes"] == [0, 1] and np.isfinite(p).all(), "fixed scorer class/finite")
    if spec["representation"] == "image26":
        p = p.reshape(len(x), 2)
        require(spec["patient_probability_pool"] in ("mean", "max"), "original pooling")
        p = p.mean(1) if spec["patient_probability_pool"] == "mean" else p.max(1)
    return p

def credit2(s, pos, neg):
    return 2*(s[pos,None] > s[None,neg]).astype(np.int8)+(s[pos,None] == s[None,neg]).astype(np.int8)

def auc(c):
    return float(c.sum()/800)

def bootstrap(c, counts, pos, neg):
    return np.einsum("bi,ij,bj->b", counts[:,pos], c, counts[:,neg])/800

def ci(x):
    return np.quantile(x, [.025, .975]).tolist()

def change(after, before):
    d = after.astype(np.int16)-before.astype(np.int16)
    return dict(improved_pairs=int((d>0).sum()), worsened_pairs=int((d<0).sum()),
        unchanged_pairs=int((d==0).sum()), twice_credit_change_sum=int(d.sum()),
        AUC_change=float(d.sum()/800),
        transitions={str(a)+"->"+str(b):int(((before==a)&(after==b)).sum())
                     for a in range(3) for b in range(3)})

def correlation(x, y):
    a, b = rankdata(x, method="average"), rankdata(y, method="average")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0,1])

def run(directory, output_tag, expected_code):
    begun = time.perf_counter()
    code_sha = digest(__file__)
    require(code_sha == expected_code, "externally frozen analyzer hash")
    require(Path(output_tag).name == output_tag and output_tag not in (".", ".."), "simple tag")
    out = directory/output_tag
    require(not out.exists(), "immutable output")
    old = TARGET/"analysis_v1"
    paths = [directory/n for n in ("protocol.json","execution.json","results.json")]
    paths += [old/n for n in ("fit_parameters.json","same_fitted_scorer_predictions.json",
                             "contract_copy.json","bootstrap_resamples.json","provenance.json","analysis.json")]
    paths += [TARGET/"verification.json"]
    hashes = {str(p.resolve()):digest(p) for p in paths}
    out.mkdir()
    write(out/"protocol.json", dict(schema="cdi-base-analysis-protocol/v1",
        created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=hashes,
        source_path=str(Path(__file__).resolve()), source_sha256=code_sha,
        policy=analysis_policy(), analysis_status="COMPUTATION_PENDING"))
    # Base values are first opened after all input/source bindings are written.
    protocol, execution = read(directory/"protocol.json"), read(directory/"execution.json")
    require(protocol["contract"]["analysis_policy"] == analysis_policy(), "GPU-precommitted analysis policy")
    require(protocol["contract"]["frozen_sha256"][str(Path(__file__).resolve())] == code_sha,
            "analyzer frozen before extraction")
    require(execution["complete"] is True and execution["records"] == 80, "complete base extraction")
    require(execution["results_sha256"] == digest(directory/"results.json"), "results binding")
    verified = read(TARGET/"verification.json")
    require(verified["status"] == "PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY"
            and verified["analysis_verification"]["status"] == "PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS",
            "verified original target comparison")
    provenance = read(old/"provenance.json")
    for p in paths:
        if p.parent == old and p.name != "provenance.json":
            require(provenance["output_sha256"][p.name] == digest(p), "original artifact hash")
    contract = read(old/"contract_copy.json")
    patients = [p for p in contract["patient_rows"] if p["role"] == "selection"]
    ids = [p["patient_id"] for p in patients]
    require(len(ids) == len(set(ids)) == 40, "selection40")
    methods = {s["id"]:s for s in contract["methods"] if s["kind"] == "logistic_regression"}
    params = {(p["model"],p["method"]):p for p in read(old/"fit_parameters.json")}
    require(len(methods) == 10 and set(params) == {(m,g) for m in MODELS for g in methods}, "all20 original scorers")
    rows = read(directory/"results.json")
    indexed = {r["image_id"]:r for r in rows}
    require(len(rows) == len(indexed) == 80 and set(indexed) == {i for p in patients for i in p["images"]},
            "same80 selection U images")
    for p in patients:
        for iid in p["images"]:
            r = indexed[iid]
            require(r["patient_id"] == p["patient_id"] and r["assignment_group"] == p["group"]
                    and r["model"] == "base" and r["eval_role"] == "selection" and r["scenario"] == "U"
                    and r["incremental_patient_member"] == r["image_member"] == r["actual_training_exposures"] == 0
                    and r["pretrained_membership"] == "unknown", "base labels are not A/B membership")
    x = np.asarray([[indexed[i]["features"]+[indexed[i]["modules"]["noise_optim"]["optimizer"]["fun"]]
                     for i in p["images"]] for p in patients], dtype=np.float64)
    require(x.shape == (40,2,27) and np.isfinite(x).all(), "base26+fixed NOobjective")
    paired = read(old/"same_fitted_scorer_predictions.json")
    paired = {(r["fit_target_scorer"],r["method"],r["patient_id"]):r for r in paired}
    require(len(paired) == 800, "same-scorer original800")
    a = np.asarray([i for i,p in enumerate(patients) if p["group"] == "A"])
    b = np.asarray([i for i,p in enumerate(patients) if p["group"] == "B"])
    require(len(a) == len(b) == 20, "A20/B20")
    old_boot = read(old/"bootstrap_resamples.json")
    require(old_boot["selection_patient_order"] == ids, "saved bootstrap order")
    ix = np.asarray(old_boot["indices"])
    rng = np.random.default_rng(260914)
    require(np.array_equal(ix, [np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(2000)]),
            "original exact bootstrap")
    counts = np.asarray([np.bincount(i,minlength=40) for i in ix], dtype=np.float64)
    summaries, predictions, pairs, contrasts, point, boots = [], [], [], [], {}, {}
    for model in MODELS:
        pos, neg = (a,b) if model == "model_1" else (b,a)
        for method, spec in methods.items():
            par = params[(model,method)]
            base = predict(x, spec, par["parameters"])
            p = [paired[(model,method,pid)] for pid in ids]
            s1 = np.asarray([r["score_model_1_features"] for r in p])
            s2 = np.asarray([r["score_model_2_features"] for r in p])
            own = s1 if model == "model_1" else s2
            common = (s1+s2)/2
            values = {"base":base, "own":own, "common":common}
            credits = {k:credit2(v,pos,neg) for k,v in values.items()}
            bs = {k:bootstrap(v,counts,pos,neg) for k,v in credits.items()}
            av = {k:auc(v) for k,v in credits.items()}
            point[(model,method)], boots[(model,method)] = av, bs
            summaries.append(dict(fit_target_scorer=model, method=method, chosen_C=par["chosen_C"],
                positive_assignment="A" if model=="model_1" else "B",
                AUC_base_own_assignment=av["base"], AUC_own_target=av["own"],
                AUC_common_own_assignment=av["common"],
                AUC_CI95={k:ci(v) for k,v in bs.items()},
                own_minus_base_AUC=av["own"]-av["base"], own_minus_base_CI95=ci(bs["own"]-bs["base"]),
                common_minus_base_AUC=av["common"]-av["base"],
                common_minus_base_CI95=ci(bs["common"]-bs["base"]),
                Spearman_base_own=correlation(base,own), Spearman_base_common=correlation(base,common),
                own_vs_base_rank=change(credits["own"],credits["base"]),
                common_vs_base_rank=change(credits["common"],credits["base"])))
            for i,p in enumerate(patients):
                predictions.append(dict(fit_target_scorer=model,method=method,patient_id=p["patient_id"],
                    assignment_group=p["group"], positive_assignment=int(i in pos),
                    incremental_patient_member=0,pretrained_membership="unknown",
                    score_base=float(base[i]),score_own_target=float(own[i]),score_common=float(common[i])))
            pair_rows = []
            for ip,pi in enumerate(pos):
                for jn,ni in enumerate(neg):
                    pair_rows.append(dict(positive_assignment_patient=ids[pi],negative_assignment_patient=ids[ni],
                        **{k+"_margin":float(v[pi]-v[ni]) for k,v in values.items()},
                        **{k+"_credit_twice":int(v[ip,jn]) for k,v in credits.items()}))
            pairs.append(dict(fit_target_scorer=model,method=method,pairs=pair_rows))
        for policy in ("fixed","tuned"):
            ka,kb = (model,"patient_meanmax52_"+policy),(model,"cdi_image_mean_"+policy)
            gains = {k:point[ka][k]-point[kb][k] for k in ("base","own","common")}
            bg = {k:boots[ka][k]-boots[kb][k] for k in gains}
            contrasts.append(dict(fit_target_scorer=model,C_policy=policy,
                a=ka[1],b=kb[1],base_assignment_AUC_gain=gains["base"],
                own_target_AUC_gain=gains["own"],common_assignment_AUC_gain=gains["common"],
                own_gain_minus_base_gain=gains["own"]-gains["base"],
                gain_CI95={k:ci(v) for k,v in bg.items()},
                own_gain_minus_base_gain_CI95=ci(bg["own"]-bg["base"])))
    for p,h in hashes.items():
        require(digest(p)==h, "unchanged input after analysis")
    require(digest(__file__)==code_sha, "unchanged source")
    write(out/"predictions.json",predictions)
    write(out/"pairs.json",pairs)
    write(out/"analysis.json",dict(status="COMPUTED_PENDING_INDEPENDENT_VERIFICATION",current_step=2,
        interpretation="Pretrained base assignment discrimination under target-fitted scorers; not base membership AUC or causal identification",
        fitted_scorers=20,selection_patients=40,base_images=80,prediction_rows=800,pair_rows=8000,
        summaries=summaries,contrasts=contrasts,bootstrap_resamples=2000,new_fitting=False,
        target_inference_calls=0,original_primary_unchanged=True,success_gate=None,
        seconds_cpu=time.perf_counter()-begun))
    write(out/"provenance.json",dict(source_sha256=code_sha,input_sha256=hashes,
        protocol_sha256=digest(out/"protocol.json"),
        output_sha256={n:digest(out/n) for n in ("predictions.json","pairs.json","analysis.json")}))
    print(json.dumps(dict(status="COMPUTED_PENDING_INDEPENDENT_VERIFICATION",output=str(out),scorers=20)))

def self_test():
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x=np.arange(4*2*27,dtype=float).reshape(4,2,27)/100
    y=np.asarray([0,1,0,1])
    for rep in ("image26","mean26","meanmax52","meanmax54"):
        z=representation(x,rep)
        scaler=StandardScaler().fit(z)
        lr=LogisticRegression(solver="liblinear",C=1).fit(scaler.transform(z),np.repeat(y,2) if rep=="image26" else y)
        par=dict(scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
            coef=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),classes=[0,1])
        for pool in ("mean","max"):
            actual=predict(x,dict(representation=rep,patient_probability_pool=pool),par)
            expected=lr.predict_proba(scaler.transform(z))[:,1]
            if rep=="image26":
                expected=expected.reshape(-1,2)
                expected=expected.mean(1) if pool=="mean" else expected.max(1)
            require(np.allclose(actual,expected,rtol=0,atol=1e-15),"synthetic saved scorer/pool")
    scores=np.arange(40)%7; pos=np.arange(20);neg=np.arange(20,40)
    require(abs(auc(credit2(scores,pos,neg))-roc_auc_score(np.r_[np.ones(20),np.zeros(20)],scores))<1e-15,"exact ties AUC")
    require(change(credit2(scores,pos,neg),credit2(scores,pos,neg))["unchanged_pairs"]==400,"identity rank change")
    print("PASS_SYNTHETIC_FIXED_PARAMETERS_REPRESENTATIONS_POOLING_TIES; no actual fitting or base inputs")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,default=DEFAULT)
    p.add_argument("--output-tag",default="analysis_v1")
    p.add_argument("--expected-code-sha256")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--print-policy",action="store_true")
    a=p.parse_args()
    if a.print_policy: print(json.dumps(analysis_policy(),ensure_ascii=False));return
    if a.self_test:self_test();return
    require(a.expected_code_sha256,"expected source SHA required")
    run(a.run_dir.resolve(),a.output_tag,a.expected_code_sha256.lower())

if __name__=="__main__":
    main()
