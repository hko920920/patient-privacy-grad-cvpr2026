"""Independent saved-data arithmetic/statistics check; no producer imports or fitting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914"
TARGET, BASE = RUN / "cdi_u_cohort_v1", RUN / "cdi_base_selection_v1"
DEFAULT = TARGET / "meanmax_feature_decomposition_v1"
MODELS = ("model_1", "model_2")
METHODS = ("patient_meanmax52_fixed", "patient_meanmax52_tuned")
SOURCES = MODELS + ("base",)
SCORES = ("full", "mean_only", "spread_only")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(value, why):
    if not value:
        raise AssertionError(why)


def checked_index(rows, keys):
    result = {}
    for row in rows:
        k = tuple(row[x] for x in keys)
        require(k not in result, "Duplicate key " + str(k))
        result[k] = row
    return result


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def arithmetic(two, p):
    """Scalar FP64 products and compensated sums, independent of producer arrays."""
    require(np.asarray(two).shape == (2, 26), "Two images and 26 features")
    w, scale, center = p["coef"][0], p["scaler_scale"], p["scaler_mean"]
    require(len(w) == len(scale) == len(center) == 52 and p["classes"] == [0, 1], "Original52 parameters")
    require(all(v > 0 for v in scale), "Positive scaler scales")
    m = [math.fsum((two[0][j], two[1][j])) / 2 for j in range(26)]
    h = [abs(two[0][j] - two[1][j]) / 2 for j in range(26)]
    a = [w[j] / scale[j] + w[j+26] / scale[j+26] for j in range(26)]
    b = [w[j+26] / scale[j+26] for j in range(26)]
    mu_h = [center[j+26] - center[j] for j in range(26)]
    mt = [(m[j]-center[j])*a[j] for j in range(26)]
    ht = [(h[j]-mu_h[j])*b[j] for j in range(26)]
    M, H = math.fsum(mt), math.fsum(ht)
    intercept = p["intercept"][0]
    logits = dict(full=math.fsum((intercept,M,H)), mean_only=math.fsum((intercept,M)),
                  spread_only=math.fsum((intercept,H)))
    original52 = m + [max(two[0][j], two[1][j]) for j in range(26)]
    direct_terms = [(original52[j]-center[j])/scale[j]*w[j] for j in range(52)]
    direct = math.fsum([intercept]+direct_terms)
    # Allow different FP64 reduction orders; separately require all ranks unchanged.
    magnitude = math.fsum([abs(intercept)]+list(map(abs,mt))+list(map(abs,ht))+list(map(abs,direct_terms)))
    tolerance = 512*np.finfo(np.float64).eps*max(1.0,magnitude)
    require(abs(logits["full"]-direct) <= tolerance, "Independent algebra identity")
    return dict(mean_features=m, half_range_features=h, mean_logit_terms=mt,
        spread_logit_terms=ht, mean_logit_component=M, spread_logit_component=H,
        logits=logits, direct=direct, tolerance=float(tolerance),
        coefficients=dict(original_intercept=intercept,mean_coefficient=a,
            spread_coefficient=b,fit_mean_reference=center[:26],fit_half_range_reference=mu_h))


def pair_credits(scores, pos, neg):
    return np.asarray([[2 if scores[i] > scores[j] else 1 if scores[i] == scores[j] else 0
                        for j in neg] for i in pos], dtype=np.int64)


def bootstrap(credits, counts, pos, neg):
    # Patient multiplicities give independent weighted Mann-Whitney calculations.
    numerator = ((counts[:,pos] @ credits) * counts[:,neg]).sum(axis=1)
    denominator = 2*counts[:,pos].sum(axis=1)*counts[:,neg].sum(axis=1)
    return numerator / denominator


def interval(values):
    return np.quantile(values,[.025,.975],method="linear")


def bindings(out):
    protocol, provenance = read(out/"protocol.json"), read(out/"provenance.json")
    producer = Path(protocol["source_path"])
    require(digest(producer) == protocol["source_sha256"] == provenance["source_sha256"], "Frozen producer hash")
    require(digest(out/"protocol.json") == provenance["protocol_sha256"], "Producer protocol binding")
    require(protocol["input_sha256"] == provenance["input_sha256"], "Producer input bindings")
    for path,h in protocol["input_sha256"].items():
        require(digest(path) == h, "Producer input changed: "+path)
    for name,h in provenance["output_sha256"].items():
        require(digest(out/name) == h, "Producer output changed: "+name)
    policy = protocol["policy"]
    require(policy["schema"] == "cdi-meanmax-centered-decomposition-policy/v1", "Policy schema")
    require(policy["feature_sources"] == list(SOURCES) and policy["patients"] == 40
            and policy["images_per_patient"] == 2 and policy["features"] == 26, "Fixed measurement scope")
    for key in ("no_fitting","no_GPU","no_direction_choice","posthoc_descriptive",
                "removed_components_set_to_fit_reference_mean","original_primary_unchanged"):
        require(policy[key] is True, "Policy scope: "+key)
    require(policy["success_gate"] is None and policy["pretrained_membership"] == "unknown", "No efficacy gate")
    require(policy["contrasts"] == ["full-minus-mean_only","full-minus-spread_only"], "Fixed contrasts")
    files = {str(Path(path).resolve()): h for path,h in protocol["input_sha256"].items()}
    for name in ("protocol.json","provenance.json","analysis.json","coefficients.json","patients.json","pairs.json"):
        files[str((out/name).resolve())] = digest(out/name)
    files[str(producer.resolve())] = digest(producer)
    files[str(Path(__file__).resolve())] = digest(__file__)
    return files


def verify(out, frozen):
    started = time.perf_counter()
    errors = Counter()
    def close(name, observed, expected, atol=2e-12, rtol=2e-12):
        x,y=np.asarray(observed,dtype=np.float64),np.asarray(expected,dtype=np.float64)
        require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(), "Shape/finite: "+name)
        err=float(np.max(np.abs(x-y))) if x.size else 0.0
        errors[name]=max(errors[name],err)
        require(np.all(np.abs(x-y)<=atol+rtol*np.abs(y)), "Numeric mismatch: "+name)
    target_ver,base_ver=read(TARGET/"verification.json"),read(BASE/"verification_v2.json")
    require(target_ver["status"]=="PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY", "Verified target features")
    require(base_ver["status"]=="PASS_SAVED_BASE_ARITHMETIC_AND_INTEGRITY", "Verified base features")
    require(target_ver["analysis_verification"]["status"]=="PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS", "Verified target statistics")
    require(base_ver["analysis_verification"]["status"]=="PASS_FIXED_SCORER_BASE_ASSIGNMENT_STATISTICS", "Verified base statistics")
    contract=read(TARGET/"analysis_v1/contract_copy.json")
    patients=[p for p in contract["patient_rows"] if p["role"]=="selection"]
    ids=[p["patient_id"] for p in patients]
    require(len(ids)==len(set(ids))==40,"Selection40")
    groups=[p["group"] for p in patients]
    require(Counter(groups)=={"A":20,"B":20},"Assignment strata")
    a=np.flatnonzero(np.asarray(groups)=="A");b=np.flatnonzero(np.asarray(groups)=="B")
    draws=read(TARGET/"analysis_v1/bootstrap_resamples.json")
    require(draws["selection_patient_order"]==ids,"Original patient bootstrap order")
    indices=np.asarray(draws["indices"],dtype=np.int64)
    rng=np.random.default_rng(260914)
    regenerated=np.asarray([np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(2000)])
    require(np.array_equal(indices,regenerated),"Original2000 draw regeneration")
    counts=np.zeros((2000,40),dtype=np.int64)
    for i,ix in enumerate(indices):
        np.add.at(counts[i],ix,1)
    rows=checked_index(read(TARGET/"results.json")+read(BASE/"results.json"),("model","image_id"))
    features={}
    for source in SOURCES:
        selected=[]
        for p in patients:
            two=[]
            require(len(p["images"])==2,"Patient two images")
            for iid in p["images"]:
                row=rows[(source,iid)]
                require(row["patient_id"]==p["patient_id"] and row["assignment_group"]==p["group"]
                    and row["eval_role"]=="selection" and row["scenario"]=="U"
                    and row["image_member"]==row["actual_training_exposures"]==0,"Exact image/role binding")
                if source=="base":
                    require(row["member"]==0 and row["pretrained_membership"]=="unknown","Base incremental scope")
                two.append(row["features"])
            require(np.asarray(two).shape==(2,26) and np.isfinite(two).all(),"Input26 dimensions")
            selected.append(two)
        features[source]=selected
    parameters=checked_index(read(TARGET/"analysis_v1/fit_parameters.json"),("model","method"))
    old=checked_index(read(TARGET/"analysis_v1/same_fitted_scorer_predictions.json"),("fit_target_scorer","method","patient_id"))
    old_base=checked_index(read(BASE/"analysis_v1/predictions.json"),("fit_target_scorer","method","patient_id"))
    result=read(out/"analysis.json")
    require(result["status"]=="COMPUTED_PENDING_INDEPENDENT_VERIFICATION", "Producer full rank status")
    require(result["scorers"]==4 and result["feature_sources"]==3 and result["patient_records"]==480
        and result["pair_records"]==4800 and result["bootstrap_resamples"]==2000,"Output counts")
    require(result["new_fitting"] is False and result["new_GPU"] is False
        and result["original_primary_unchanged"] is True and result["success_gate"] is None,"Output interpretation scope")
    key=("fit_target_scorer","method","feature_source")
    summaries=checked_index(result["summaries"],key)
    output_rows=checked_index(read(out/"patients.json"),key+("patient_id",))
    pairs=checked_index(read(out/"pairs.json"),key)
    coefs=checked_index(read(out/"coefficients.json"),("fit_target_scorer","method"))
    require(len(summaries)==len(pairs)==12 and len(output_rows)==480 and len(coefs)==4,"Unique output coverage")
    metric_rows=[];contrast_rows=[];rank_mismatches=0;checked_pairs=0
    for model in MODELS:
        pos,neg=(a,b) if model=="model_1" else (b,a)
        y=np.asarray([int(g==("A" if model=="model_1" else "B")) for g in groups])
        for method in METHODS:
            entry=parameters[(model,method)]
            require(entry["representation"]=="meanmax52" and entry["fitting_rows"]==80,"Original fixed scorer")
            require(len(set(entry["fitting_patient_ids"]))==80 and not set(ids).intersection(entry["fitting_patient_ids"]),"Fit/selection separation")
            for source in SOURCES:
                k=(model,method,source); summary=summaries[k]
                require(summary["chosen_C"]==entry["chosen_C"] and summary["positive_assignment"]==("A" if model=="model_1" else "B"),"Fixed C/orientation")
                scores={name:[] for name in SCORES}; saved_probs=[]; reported_scores={name:[] for name in SCORES}
                for i,p in enumerate(patients):
                    calc=arithmetic(features[source][i],entry["parameters"])
                    r=output_rows[k+(p["patient_id"],)]
                    require(r["assignment_group"]==p["group"],"Patient assignment metadata")
                    for name in ("mean_features","half_range_features","mean_logit_terms","spread_logit_terms","mean_logit_component","spread_logit_component"):
                        close(name,r[name],calc[name])
                    for name in SCORES:
                        close("logit_"+name,r["logits"][name],calc["logits"][name],atol=calc["tolerance"],rtol=0)
                        scores[name].append(calc["logits"][name]);reported_scores[name].append(r["logits"][name])
                    close("direct_original52_logit",r["logits"]["full"],calc["direct"],atol=calc["tolerance"],rtol=0)
                    for name,value in calc["coefficients"].items():
                        close("coefficient_"+name,coefs[(model,method)][name],value)
                    require(coefs[(model,method)]["chosen_C"]==entry["chosen_C"],"Coefficient C")
                    old_key=(model,method,p["patient_id"])
                    prob=old_base[old_key]["score_base"] if source=="base" else old[old_key]["score_"+source+"_features"]
                    close("saved_probability_binding",r["saved_full_probability"],prob,atol=0,rtol=0)
                    close("sigmoid_probability",r["reconstructed_full_probability"],sigmoid(calc["logits"]["full"]))
                    close("old_probability_reproduction",sigmoid(calc["direct"]),prob)
                    saved_probs.append(prob)
                credits={n:pair_credits(v,pos,neg) for n,v in scores.items()}
                saved_credit=pair_credits(saved_probs,pos,neg)
                for name in SCORES:
                    require(np.array_equal(credits[name],pair_credits(reported_scores[name],pos,neg)),"Independent reduction changed ranks: "+name)
                mismatch=int(np.count_nonzero(credits["full"]!=saved_credit));rank_mismatches+=mismatch
                require(mismatch==summary["reproduction"]["full_logit_vs_saved_probability_pair_rank_mismatches"],"Saved rank mismatch count")
                bs={}; point={}
                for name in SCORES:
                    point[name]=float(roc_auc_score(y,scores[name]))
                    close("sklearn_AUC",summary["AUC"][name],point[name],atol=1e-15,rtol=0)
                    close("pair_AUC",credits[name].sum()/800,point[name],atol=1e-15,rtol=0)
                    bs[name]=bootstrap(credits[name],counts,pos,neg)
                    close("bootstrap_CI95",summary["AUC_CI95"][name],interval(bs[name]),atol=1e-15,rtol=0)
                    for ix in (0,37,999,1999):
                        draw=indices[ix]
                        close("bootstrap_sklearn_spot",bs[name][ix],roc_auc_score(y[draw],np.asarray(scores[name])[draw]),atol=1e-15,rtol=0)
                    metric_rows.append(dict(fit_target_scorer=model,method=method,feature_source=source,score=name,AUC=point[name],CI95=interval(bs[name]).tolist()))
                cc=checked_index(summary["contrasts"],("comparison",))
                require(len(cc)==2,"Two declared contrasts")
                for name in SCORES[1:]:
                    comparison="full-minus-"+name; d=credits["full"]-credits[name]; c=cc[(comparison,)]
                    close("delta_AUC",c["delta_AUC"],point["full"]-point[name],atol=1e-15,rtol=0)
                    close("paired_delta_CI95",c["paired_CI95"],interval(bs["full"]-bs[name]),atol=1e-15,rtol=0)
                    require(c["improved_pairs"]==int((d>0).sum()) and c["worsened_pairs"]==int((d<0).sum())
                        and c["same_pairs"]==int((d==0).sum()) and c["twice_credit_difference_sum"]==int(d.sum()),"Contrast pair accounting")
                    contrast_rows.append(dict(fit_target_scorer=model,method=method,feature_source=source,**c))
                pp=pairs[k]["pairs"]
                require(len(pp)==400,"Full positive-negative pair grid")
                for index,(ip,jn) in enumerate((i,j) for i in range(20) for j in range(20)):
                    row=pp[index];pi,ni=pos[ip],neg[jn]
                    require(row["positive_assignment_patient"]==ids[pi] and row["negative_assignment_patient"]==ids[ni],"Pair patient order")
                    for name in SCORES:
                        close("pair_margin",row[name+"_margin"],scores[name][pi]-scores[name][ni])
                        require(row[name+"_credit_twice"]==int(credits[name][ip,jn]),"Pair credit")
                    require(row["saved_full_probability_credit_twice"]==int(saved_credit[ip,jn]),"Saved probability pair credit")
                    checked_pairs+=1
                reproduction=summary["reproduction"]
                require(reproduction["saved_probability_saturated_count"]==sum(p in (0,1) for p in saved_probs),"Saturation counts")
                require(reproduction["saved_probability_pair_ties"]==int((saved_credit==1).sum())
                    and reproduction["full_logit_pair_ties"]==int((credits["full"]==1).sum()),"Tie counts")
                close("reproduction_AUC",reproduction["saved_probability_AUC"],saved_credit.sum()/800,atol=1e-15,rtol=0)
                close("logit_probability_AUC_delta",reproduction["full_logit_minus_saved_probability_AUC"],(credits["full"]-saved_credit).sum()/800,atol=1e-15,rtol=0)
    require(rank_mismatches==result["full_rank_mismatches"]==0,"No full-rank interpretation when mismatch exists")
    for path,h in frozen.items():
        require(digest(path)==h,"Input/code changed during verification: "+path)
    return dict(status="PASS_SAVED_CENTERED_MEANMAX_DECOMPOSITION_AND_STATISTICS",current_step=2,
        scorers=4,feature_sources=3,contexts=12,patient_records=480,pairs_checked=checked_pairs,
        AUC_metrics_checked=len(metric_rows),contrast_CIs_checked=len(contrast_rows),
        full_minus_mean_contrasts=12,full_minus_spread_contrasts=12,bootstrap_resamples=2000,
        sklearn_bootstrap_spot_checks=144,full_probability_rank_mismatches=rank_mismatches,
        max_absolute_errors=dict(errors),metrics=metric_rows,contrasts=contrast_rows,
        source_sha256=digest(__file__),seconds_cpu=time.perf_counter()-started,
        independent_refitting=False,new_GPU_calls=0,raw_tensor_reverification=False,
        input_scope="Previously verified source26 scalar packets and frozen coefficients; raw UNet tensors not regenerated",
        interpretation="Saved centered feature accounting and conditional statistics; not a new attack, removal intervention efficacy or causal attribution")


def self_test():
    rng=np.random.default_rng(11)
    x=rng.normal(size=(2,26)).tolist()
    p=dict(coef=[rng.normal(size=52).tolist()],scaler_scale=rng.uniform(.01,3,52).tolist(),
        scaler_mean=rng.normal(size=52).tolist(),intercept=[.39],classes=[0,1])
    result=arithmetic(x,p)
    direct=(np.concatenate((np.mean(x,axis=0),np.max(x,axis=0)))-p["scaler_mean"])/p["scaler_scale"]
    require(abs(result["logits"]["full"]-(direct@p["coef"][0]+.39))<=result["tolerance"],"Synthetic full52 identity")
    y=np.r_[np.ones(20),np.zeros(20)];scores=np.arange(40)%7
    pos,neg=np.arange(20),np.arange(20,40)
    draws=np.asarray([np.r_[rng.choice(pos,20),rng.choice(neg,20)] for _ in range(31)])
    counts=np.asarray([np.bincount(d,minlength=40) for d in draws])
    expected=[roc_auc_score(y[d],scores[d]) for d in draws]
    require(np.allclose(bootstrap(pair_credits(scores,pos,neg),counts,pos,neg),expected,atol=1e-15,rtol=0),"Synthetic ties and weighted bootstrap")
    print("PASS_SYNTHETIC_ARITHMETIC_SKLEARN_WEIGHTED_BOOTSTRAP; no actual data read")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,default=DEFAULT)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--freeze-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return
    out=args.run_dir.resolve()
    frozen=bindings(out)
    protocol_path=out/"verification_protocol.json"
    policy=dict(schema="independent-cdi-meanmax-feature-verification/v1",source_sha256=digest(__file__),
        input_sha256=frozen,arithmetic="scalar FP64 products/math.fsum; original52 direct logit; independent sigmoid",
        algebra_tolerance="512*float64_epsilon*max(1,sum of absolute direct/decomposed terms)",
        scalar_tolerance=dict(atol=2e-12,rtol=2e-12),statistics_atol=1e-15,
        rank_tolerance="zero changed half-credit pairs",bootstrap="original 2000 regenerated draws; weighted patient pair credits",
        new_fitting=False,new_GPU_calls=0)
    if protocol_path.exists():
        require(read(protocol_path)==policy,"Frozen verifier protocol mismatch")
    else:
        write_new(protocol_path,policy)
    if args.freeze_only:
        print("FROZEN_VERIFICATION_PROTOCOL_NO_ARITHMETIC_EXECUTED");return
    require(not (out/"verification.json").exists(),"Immutable verification already exists")
    try:
        result=verify(out,frozen)
        result["verification_protocol_sha256"]=digest(protocol_path)
        write_new(out/"verification.json",result)
        print(json.dumps({k:result[k] for k in ("status","patient_records","pairs_checked","AUC_metrics_checked","contrast_CIs_checked","full_probability_rank_mismatches","max_absolute_errors","seconds_cpu")}))
    except BaseException as exc:
        failure=out/"verification_failure.json"
        if not failure.exists():
            write_new(failure,dict(status="FAILED_INDEPENDENT_VERIFICATION",error=repr(exc),
                source_sha256=digest(__file__),verification_protocol_sha256=digest(protocol_path),
                time_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=="__main__":
    main()
