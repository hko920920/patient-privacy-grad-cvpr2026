"""Post-hoc, descriptive rank accounting for all 20 frozen CDI U scorers.

No target invocation, fitting, direction selection, threshold, or efficacy gate.
Source/inputs are recorded before the actual analysis is read or computed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT/"_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1"
MODEL_ORDER = ("model_1", "model_2")
METHOD_ORDER = (
    "cdi_image_mean_fixed", "cdi_image_mean_tuned",
    "cdi_image_max_fixed", "cdi_image_max_tuned",
    "patient_mean26_fixed", "patient_mean26_tuned",
    "patient_meanmax52_fixed", "patient_meanmax52_tuned",
    "patient_meanmax54_noobjective_fixed", "patient_meanmax54_noobjective_tuned")


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def credit2(positive, negative):
    """Integer twice-rank-credit: lose=0, exact numerical tie=1, win=2."""
    return (2*(positive > negative).astype(np.int16)+(positive == negative).astype(np.int16))


def auc_from_credit(c):
    return float(np.sum(c,dtype=np.int64)/(2*c.size))


def describe_change(after, before):
    delta = after.astype(np.int16)-before
    return dict(improved_pairs=int((delta>0).sum()), worsened_pairs=int((delta<0).sum()),
        same_credit_pairs=int((delta==0).sum()), delta_twice_credit_sum=int(delta.sum()),
        AUC_change=float(delta.sum()/(2*delta.size)),
        transition_counts_before_to_after=[[int(((before==i)&(after==j)).sum()) for j in range(3)] for i in range(3)])


def decompose(s1, s2, a, b, own_model):
    require(s1.shape == s2.shape and np.isfinite(s1).all() and np.isfinite(s2).all(), "finite matched scores")
    c, d = (s1+s2)/2, (s1-s2)/2
    cs1,cs2,cc = [credit2(v[a,None],v[None,b]) for v in (s1,s2,c)]
    own = cs1 if own_model=="model_1" else 2-cs2
    common_own = cc if own_model=="model_1" else 2-cc
    contribution = own-common_own
    require(int(own.sum()) == int(common_own.sum())+int(contribution.sum()), "integer rank accounting")
    dc,dd = c[a,None]-c[None,b],d[a,None]-d[None,b]
    # Preserve ranks from the original saved scores, including exact ties.
    # Floating c +/- d may differ by an ulp; report this, never substitute its ranks.
    reconstruction = dict(score_max_abs_error=float(max(np.max(np.abs(c+d-s1)),np.max(np.abs(c-d-s2)))),
        model1_pair_credit_mismatches=int(np.count_nonzero(credit2(dc+dd,np.zeros_like(dc))-cs1)),
        model2_pair_credit_mismatches=int(np.count_nonzero(credit2(dc-dd,np.zeros_like(dc))-cs2)))
    return dict(c=c,d=d,s1_credit=cs1,s2_credit=cs2,common_credit=cc,own_credit=own,
        common_own_credit=common_own,contribution=contribution,
        common_margin=dc,change_margin=dd,reconstruction=reconstruction)


def conditional_boot(credit, count_a, count_b):
    return np.einsum("bi,ij,bj->b",count_a,credit,count_b)/(2*20*20)


def ci(values):
    return np.quantile(values,[.025,.975],method="linear").tolist()


def run(directory, output_tag):
    begun = time.perf_counter()
    source_hash = digest(__file__)
    require(Path(output_tag).name == output_tag and output_tag not in (".",".."), "simple output directory")
    out = directory/output_tag
    require(not out.exists(), "immutable output already exists")
    analysis = directory/"analysis_v1"
    names = ("same_fitted_scorer_predictions.json","predictions.json","analysis.json",
             "bootstrap_resamples.json","contract_copy.json","provenance.json")
    inputs = [analysis/n for n in names]+[directory/"verification.json",directory/"results.json",directory/"protocol.json"]
    hashes = {str(p.resolve()):digest(p) for p in inputs}
    out.mkdir()
    protocol = dict(schema="cdi-u-rank-decomposition-protocol/v1", current_step=2,
        created_utc=datetime.now(timezone.utc).isoformat(), source_path=str(Path(__file__).resolve()),
        source_sha256=source_hash,input_sha256=hashes,all_scorers=20,
        method_order=list(METHOD_ORDER),target_order=list(MODEL_ORDER),
        score_definition="s1/s2 from the same frozen scorer; c=(s1+s2)/2,d=(s1-s2)/2 in float64",
        ranks="original saved s1/s2 and computed float64 c; exact ties receive half credit; integer twice-credit accounting",
        contrasts="patient_meanmax52 minus cdi_image_mean, fixed and tuned, within each fit-target scorer",
        post_hoc=True,descriptive_only=True,
        common_component_is_identified_nuisance=False,individual_causal_effect=False,
        bootstrap="reuse original 2000 stratified paired patient indices, conditional on frozen fitted scorers",
        new_fitting=False,new_GPU=False,original_primary_analysis_unchanged=True,success_gate=None)
    write(out/"protocol.json",protocol)
    # Actual result reads and all decomposition start only after protocol binding.
    verified = read(directory/"verification.json")
    require(verified["status"]=="PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY"
        and verified["analysis_verification"]["status"]=="PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS", "verified source analysis required")
    provenance=read(analysis/"provenance.json")
    for name in names[:-1]:
        require(provenance["output_sha256"][name]==digest(analysis/name),"bound analysis output "+name)
    contract=read(analysis/"contract_copy.json")
    require(contract["models"]==list(MODEL_ORDER),"original model order")
    require([m["id"] for m in contract["methods"] if m["kind"]=="logistic_regression"]==list(METHOD_ORDER),"all frozen LR methods")
    original=read(analysis/"analysis.json")
    own_reference={(r["model"],r["method"]):r["selection_patient_AUC"] for r in original["results"]}
    patients=[p for p in contract["patient_rows"] if p["role"]=="selection"]
    ids=[p["patient_id"] for p in patients]
    require(len(ids)==len(set(ids))==40,"40 selection patients")
    a=np.asarray([i for i,p in enumerate(patients) if p["group"]=="A"])
    b=np.asarray([i for i,p in enumerate(patients) if p["group"]=="B"])
    require(len(a)==len(b)==20,"20 A/B patients")
    old_boot=read(analysis/"bootstrap_resamples.json")
    require(old_boot["selection_patient_order"]==ids,"original bootstrap patient order")
    indices=np.asarray(old_boot["indices"],dtype=np.int64)
    rng=np.random.default_rng(260914)
    expected=np.asarray([np.r_[rng.choice(a,20,replace=True),rng.choice(b,20,replace=True)] for _ in range(2000)])
    require(np.array_equal(indices,expected),"unchanged bootstrap indices")
    counts=np.asarray([np.bincount(ix,minlength=40) for ix in indices],dtype=float)
    rows=read(analysis/"same_fitted_scorer_predictions.json")
    indexed={(r["fit_target_scorer"],r["method"],r["patient_id"]):r for r in rows}
    require(len(rows)==len(indexed)==800 and set(indexed)=={(m,g,p) for m in MODEL_ORDER for g in METHOD_ORDER for p in ids},"all 800 same-scorer rows")
    summaries,patient_out,pair_out,details,boots=[],[],[],{},{}
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            rr=[indexed[(model,method,p)] for p in ids]
            require([r["group"] for r in rr]==[p["group"] for p in patients],"paired group metadata")
            s1=np.asarray([r["score_model_1_features"] for r in rr],dtype=float)
            s2=np.asarray([r["score_model_2_features"] for r in rr],dtype=float)
            v=decompose(s1,s2,a,b,model)
            details[(model,method)]=v
            own_auc=auc_from_credit(v["own_credit"])
            common_auc=auc_from_credit(v["common_own_credit"])
            response=float(v["contribution"].sum()/800)
            require(abs(own_auc-own_reference[(model,method)])<=2e-15,"original deployed AUC unchanged")
            own_boot=conditional_boot(v["own_credit"],counts[:,a],counts[:,b])
            common_boot=conditional_boot(v["common_own_credit"],counts[:,a],counts[:,b])
            response_boot=conditional_boot(v["contribution"],counts[:,a],counts[:,b])
            require(np.allclose(own_boot,common_boot+response_boot,rtol=0,atol=2e-16),"bootstrap accounting")
            boots[(model,method)]=(own_boot,common_boot,response_boot)
            summaries.append(dict(fit_target_scorer=model,method=method,patient_pairs=400,
                AUC_s1_yA=auc_from_credit(v["s1_credit"]),AUC_s2_yA=auc_from_credit(v["s2_credit"]),
                AUC_common_yA=auc_from_credit(v["common_credit"]),
                same_scorer_target1_actual_membership_AUC=auc_from_credit(v["s1_credit"]),
                same_scorer_target2_actual_membership_AUC=1-auc_from_credit(v["s2_credit"]),
                own_target_AUC=own_auc,common_own_label_AUC=common_auc,own_vs_common_AUC_contribution=response,
                own_target_AUC_CI95=ci(own_boot),common_own_label_AUC_CI95=ci(common_boot),
                own_vs_common_contribution_CI95=ci(response_boot),
                own_twice_credit_sum=int(v["own_credit"].sum()),
                common_own_twice_credit_sum=int(v["common_own_credit"].sum()),
                own_minus_common_twice_credit_sum=int(v["contribution"].sum()),
                own_vs_common=describe_change(v["own_credit"],v["common_own_credit"]),
                s1_vs_s2_under_yA=describe_change(v["s1_credit"],v["s2_credit"]),
                exact_tie_pairs={k:int((v[k]==1).sum()) for k in ("s1_credit","s2_credit","common_credit")},
                floating_reconstruction=v["reconstruction"]))
            for i,p in enumerate(patients):
                patient_out.append(dict(fit_target_scorer=model,method=method,patient_id=p["patient_id"],
                    group=p["group"],s1=float(s1[i]),s2=float(s2[i]),common=float(v["c"][i]),change=float(v["d"][i])))
            pairs=[]
            for ai,pi in enumerate(a):
                for bi,pj in enumerate(b):
                    pairs.append(dict(patient_A=ids[pi],patient_B=ids[pj],
                        s1_margin=float(s1[pi]-s1[pj]),s2_margin=float(s2[pi]-s2[pj]),
                        common_margin=float(v["common_margin"][ai,bi]),change_margin=float(v["change_margin"][ai,bi]),
                        s1_credit_twice=int(v["s1_credit"][ai,bi]),s2_credit_twice=int(v["s2_credit"][ai,bi]),
                        common_yA_credit_twice=int(v["common_credit"][ai,bi]),
                        own_credit_twice=int(v["own_credit"][ai,bi]),common_own_credit_twice=int(v["common_own_credit"][ai,bi]),
                        own_vs_common_delta_twice=int(v["contribution"][ai,bi])))
            pair_out.append(dict(fit_target_scorer=model,method=method,pairs=pairs))
    comparisons=[]
    for model in MODEL_ORDER:
        for policy in ("tuned","fixed"):
            ka,kb=(model,"patient_meanmax52_"+policy),(model,"cdi_image_mean_"+policy)
            va,vb=details[ka],details[kb]
            own=int(va["own_credit"].sum()-vb["own_credit"].sum())
            common=int(va["common_own_credit"].sum()-vb["common_own_credit"].sum())
            response=int(va["contribution"].sum()-vb["contribution"].sum())
            require(own==common+response,"exact between-method rank accounting")
            boot=[boots[ka][j]-boots[kb][j] for j in range(3)]
            comparisons.append(dict(fit_target_scorer=model,policy=policy,method_a=ka[1],method_b=kb[1],
                original_role="primary" if policy=="tuned" else "secondary",
                observed_own_AUC_gain=own/800,common_component_AUC_gain=common/800,
                difference_in_target_change_contribution=response/800,
                own_gain_twice_credit_sum=own,common_gain_twice_credit_sum=common,
                response_contribution_twice_credit_sum=response,
                observed_gain_CI95=ci(boot[0]),common_gain_CI95=ci(boot[1]),response_contribution_CI95=ci(boot[2])))
    for path,value in hashes.items():
        require(digest(path)==value,"input changed during execution")
    require(digest(__file__)==source_hash,"source changed during execution")
    write(out/"patients.json",patient_out)
    write(out/"pairs.json",pair_out)
    summary=dict(schema="cdi-u-rank-decomposition/v1",status="PASS_EXACT_DESCRIPTIVE_RANK_ACCOUNTING",
        current_step=2,post_hoc=True,scorers=20,patient_records=800,pair_records=8000,
        summaries=summaries,comparisons=comparisons,seconds_cpu=time.perf_counter()-begun,
        new_GPU_calls=0,new_fitting=False,identified_nuisance_or_causal_mechanism=False,success_gate=None,
        limitations=["Common uses both targets and is analyst-only, not a deployable baseline or identified nuisance.",
                    "Decomposition is algebra on frozen fitted scores; neither feature/fit causality nor individual membership effect is identified.",
                    "CIs reuse development patients and fixed fitted scorers; no refit uncertainty or multiple-comparison correction.",
                    "Original primary/secondary comparisons are unchanged; no post-hoc direction or winner selection."],
        protocol_sha256=digest(out/"protocol.json"),patients_sha256=digest(out/"patients.json"),
        pairs_sha256=digest(out/"pairs.json"),source_sha256=source_hash)
    write(out/"summary.json",summary)
    print(json.dumps(dict(status=summary["status"],seconds_cpu=summary["seconds_cpu"],comparisons=comparisons),ensure_ascii=False))
    return summary


def self_test():
    a,b=np.asarray([0,1]),np.asarray([2,3])
    s1=np.asarray([.2,.8,.2,.7]);s2=np.asarray([.4,.6,.4,.9])
    for model in MODEL_ORDER:
        x=decompose(s1,s2,a,b,model)
        require(int(x["own_credit"].sum())==int(x["common_own_credit"].sum()+x["contribution"].sum()),"tie accounting")
        require(np.any(x["s1_credit"]==1) and np.any(x["common_credit"]==1),"exact tie fixture")
        changed=describe_change(x["own_credit"],x["common_own_credit"])
        require(changed["improved_pairs"]+changed["worsened_pairs"]+changed["same_credit_pairs"]==4,"rank partitions")
    for model in MODEL_ORDER:
        x=decompose(s1,s1,a,b,model)
        require(np.all(x["contribution"]==0),"identical-target zero contribution")
        shifted=decompose(s1+2,s1,a,b,model)
        require(np.all(shifted["contribution"]==0),"global target offset cannot change ranks")
    a,b=np.arange(20),np.arange(20,40)
    rng=np.random.default_rng(3)
    s1=np.arange(40)%7;s2=np.arange(40)%11
    x=decompose(s1.astype(float),s2.astype(float),a,b,"model_1")
    ix=np.asarray([np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(17)])
    count=np.asarray([np.bincount(i,minlength=40) for i in ix],dtype=float)
    from sklearn.metrics import roc_auc_score
    expected=[roc_auc_score(np.r_[np.ones(20),np.zeros(20)],s1[i]) for i in ix]
    require(np.allclose(conditional_boot(x["own_credit"],count[:,a],count[:,b]),expected,rtol=0,atol=2e-16),"independent bootstrap tie AUC")
    print("PASS_SYNTHETIC_TIES_RANK_IDENTITIES_OFFSET_AND_BOOTSTRAP")


def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test",action="store_true")
    mode.add_argument("--run",action="store_true")
    p.add_argument("--run-dir",type=Path,default=RUN)
    p.add_argument("--output-tag",default="rank_decomposition_v1")
    args=p.parse_args()
    self_test() if args.self_test else run(args.run_dir.resolve(),args.output_tag)


if __name__=="__main__":
    main()
