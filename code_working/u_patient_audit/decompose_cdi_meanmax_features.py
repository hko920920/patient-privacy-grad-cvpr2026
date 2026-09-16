"""Post-hoc centered mean/spread accounting of four frozen meanmax52 scorers.
No fitting, GPU calls, direction choice, new features, or causal attribution.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/"_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914"
TARGET=RUN/"cdi_u_cohort_v1"
BASE=RUN/"cdi_base_selection_v1"
MODELS=("model_1","model_2")
METHODS=("patient_meanmax52_fixed","patient_meanmax52_tuned")
SOURCES=("model_1","model_2","base")
OUTPUT=TARGET/"meanmax_feature_decomposition_v1"

def require(c,m):
    if not c:raise AssertionError(m)

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def write(p,obj):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def policy():
    return dict(schema="cdi-meanmax-centered-decomposition-policy/v1",current_step=2,
        fit_scorers=[dict(model=m,method=g) for m in MODELS for g in METHODS],
        feature_sources=list(SOURCES),patients=40,images_per_patient=2,features=26,
        formula="m=(x1+x2)/2; h=abs(x1-x2)/2; k_m=w[:26]/scale[:26]; k_x=w[26:]/scale[26:]; mu_m=mean[:26]; mu_h=mean[26:]-mu_m; M=(k_m+k_x).(m-mu_m); H=k_x.(h-mu_h)",
        scores="full=original_intercept+M+H; mean_only=original_intercept+M; spread_only=original_intercept+H",
        metric="AUC on logits under the positive A/B assignment of each fit-target; base is not membership",
        reproduction="Check direct standardized full logit and saved sigmoid; report all400 pair rank mismatches/saturation/ties. No completed interpretation if full rankings differ.",
        bootstrap="reuse original 2000 stratified paired A20/B20 patient indices, seed260914, conditional percentiles",
        contrasts=["full-minus-mean_only","full-minus-spread_only"],
        no_fitting=True,no_GPU=True,no_direction_choice=True,no_feature_selection=True,
        posthoc_descriptive=True,off_manifold_feature_intervention_possible=True,
        removed_components_set_to_fit_reference_mean=True,
        narrower_question="Does this already fitted meanmax scorer use within-patient feature spread in its observed ranking?",
        not_identified_cause_of_separately_fitted_mean26_difference=True,
        not_new_attack=True,not_individual_causal_membership=True,
        target_fit80_information_retained=True,pretrained_membership="unknown",
        original_primary_unchanged=True,success_gate=None)

def components(x,par):
    require(x.ndim==3 and x.shape[1:]==(2,26),"patient2x26")
    w=np.asarray(par["coef"],dtype=float)[0]
    scale=np.asarray(par["scaler_scale"],dtype=float)
    center=np.asarray(par["scaler_mean"],dtype=float)
    require(w.shape==scale.shape==center.shape==(52,) and np.all(scale>0)
            and par["classes"]==[0,1],"fixed52 scorer")
    m=x.mean(1);h=np.abs(x[:,0]-x[:,1])/2
    km,kx=w[:26]/scale[:26],w[26:]/scale[26:]
    mu_m,mu_h=center[:26],center[26:]-center[:26]
    a,b=km+kx,kx
    mt=(m-mu_m)*a;ht=(h-mu_h)*b
    M,H=mt.sum(1),ht.sum(1);intercept=par["intercept"][0]
    full=intercept+M+H
    direct=((np.concatenate((m,x.max(1)),1)-center)/scale)@w+intercept
    # Conservative FP64 operation-rounding bound, not a scientific pass threshold.
    magnitude=np.abs(intercept)+np.abs(mt).sum(1)+np.abs(ht).sum(1)
    magnitude+=np.abs(((np.concatenate((m,x.max(1)),1)-center)/scale)*w).sum(1)
    bound=128*np.finfo(np.float64).eps*np.maximum(1,magnitude)
    require(np.all(np.abs(full-direct)<=bound),"full centered algebra numerical identity")
    return dict(mean=m,half_range=h,mean_terms=mt,spread_terms=ht,M=M,H=H,
        full=full,mean_only=intercept+M,spread_only=intercept+H,direct=direct,
        numerical_bound=bound,
        coefficients=dict(original_intercept=float(intercept),mean_coefficient=a.tolist(),
                          spread_coefficient=b.tolist(),fit_mean_reference=mu_m.tolist(),
                          fit_half_range_reference=mu_h.tolist()))

def credit(s,pos,neg):
    return 2*(s[pos,None]>s[None,neg]).astype(np.int8)+(s[pos,None]==s[None,neg]).astype(np.int8)

def boot(c,counts,pos,neg):return np.einsum("bi,ij,bj->b",counts[:,pos],c,counts[:,neg])/800
def ci(x):return np.quantile(x,[.025,.975]).tolist()

def run(out,expected):
    start=time.perf_counter();source=digest(__file__)
    require(expected==source,"externally frozen source SHA")
    require(not out.exists(),"immutable new output")
    ta=TARGET/"analysis_v1";ba=BASE/"analysis_v1"
    inputs=[TARGET/"results.json",TARGET/"verification.json",
            BASE/"results.json",BASE/"verification_v2.json",
            BASE/"verification_v2_protocol.json"]
    inputs += [ta/n for n in ("contract_copy.json","fit_parameters.json","same_fitted_scorer_predictions.json",
                              "bootstrap_resamples.json","provenance.json")]
    inputs += [ba/n for n in ("predictions.json","provenance.json")]
    hashes={str(p.resolve()):digest(p) for p in inputs}
    out.mkdir()
    write(out/"protocol.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(Path(__file__).resolve()),source_sha256=source,input_sha256=hashes,policy=policy()))
    # No new data-derived statistics occur until this immutable protocol is saved.
    tv,bv=read(TARGET/"verification.json"),read(BASE/"verification_v2.json")
    require(tv["status"]=="PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY"
            and tv["analysis_verification"]["status"]=="PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS","verified target")
    require(bv["status"]=="PASS_SAVED_BASE_ARITHMETIC_AND_INTEGRITY"
            and bv["analysis_verification"]["status"]=="PASS_FIXED_SCORER_BASE_ASSIGNMENT_STATISTICS","verified base")
    for directory in (ta,ba):
        provenance=read(directory/"provenance.json")
        for p in inputs:
            if p.parent==directory and p.name!="provenance.json":
                require(provenance["output_sha256"][p.name]==digest(p),"original output binding")
    contract=read(ta/"contract_copy.json")
    patients=[p for p in contract["patient_rows"] if p["role"]=="selection"]
    ids=[p["patient_id"] for p in patients]
    require(len(ids)==len(set(ids))==40,"same40 selection")
    a=np.asarray([i for i,p in enumerate(patients) if p["group"]=="A"])
    b=np.asarray([i for i,p in enumerate(patients) if p["group"]=="B"])
    require(len(a)==len(b)==20,"A20/B20")
    rows=read(TARGET/"results.json")+read(BASE/"results.json")
    rows={(r["model"],r["image_id"]):r for r in rows}
    xs={}
    for model in SOURCES:
        selected=[]
        for p in patients:
            rr=[rows[(model,i)] for i in p["images"]]
            for r in rr:
                require(r["patient_id"]==p["patient_id"] and r["assignment_group"]==p["group"]
                    and r["eval_role"]=="selection" and r["scenario"]=="U" and r["image_member"]==0,"same patient/photo identities")
            selected.append([r["features"] for r in rr])
        xs[model]=np.asarray(selected,dtype=np.float64)
        require(xs[model].shape==(40,2,26) and np.isfinite(xs[model]).all(),"finite80features")
    params={(r["model"],r["method"]):r for r in read(ta/"fit_parameters.json")}
    saved={(r["fit_target_scorer"],r["method"],r["patient_id"]):r
           for r in read(ta/"same_fitted_scorer_predictions.json")}
    base_saved={(r["fit_target_scorer"],r["method"],r["patient_id"]):r
                for r in read(ba/"predictions.json")}
    resamples=read(ta/"bootstrap_resamples.json")
    require(resamples["selection_patient_order"]==ids,"original bootstrap ordering")
    indices=np.asarray(resamples["indices"],dtype=np.int64)
    rng=np.random.default_rng(260914)
    require(np.array_equal(indices,[np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(2000)]),"exact original2000draws")
    counts=np.asarray([np.bincount(ix,minlength=40) for ix in indices],dtype=np.float64)
    summaries,patient_out,pair_out,coefficient_out=[],[],[],[]
    mismatch_total=0
    for model in MODELS:
        pos,neg=(a,b) if model=="model_1" else (b,a)
        for method in METHODS:
            entry=params[(model,method)]
            require(entry["representation"]=="meanmax52","fixed original representation")
            for feature_source in SOURCES:
                c=components(xs[feature_source],entry["parameters"])
                if feature_source=="base":
                    probability=np.asarray([base_saved[(model,method,p)]["score_base"] for p in ids])
                else:
                    field="score_model_1_features" if feature_source=="model_1" else "score_model_2_features"
                    probability=np.asarray([saved[(model,method,p)][field] for p in ids])
                require(np.allclose(expit(c["full"]),probability,rtol=2e-12,atol=2e-12),"saved full probability reproduction")
                scores={n:c[n] for n in ("full","mean_only","spread_only")}
                credits={n:credit(v,pos,neg) for n,v in scores.items()}
                saved_credit=credit(probability,pos,neg)
                mismatches=int((credits["full"]!=saved_credit).sum());mismatch_total+=mismatches
                bs={n:boot(v,counts,pos,neg) for n,v in credits.items()}
                av={n:float(v.sum()/800) for n,v in credits.items()}
                checks=dict(max_abs_logit_algebra_error=float(np.max(np.abs(c["full"]-c["direct"]))),
                    max_abs_probability_error=float(np.max(np.abs(expit(c["full"])-probability))),
                    maximum_logit_rounding_bound=float(c["numerical_bound"].max()),
                    saved_probability_saturated_count=int(((probability==0)|(probability==1)).sum()),
                    saved_probability_pair_ties=int((saved_credit==1).sum()),
                    full_logit_pair_ties=int((credits["full"]==1).sum()),
                    full_logit_vs_saved_probability_pair_rank_mismatches=mismatches,
                    saved_probability_AUC=float(saved_credit.sum()/800),
                    full_logit_minus_saved_probability_AUC=float((credits["full"].astype(int)-saved_credit).sum()/800))
                contrasts=[]
                for n in ("mean_only","spread_only"):
                    d=credits["full"].astype(int)-credits[n]
                    contrasts.append(dict(comparison="full-minus-"+n,delta_AUC=av["full"]-av[n],
                        paired_CI95=ci(bs["full"]-bs[n]),improved_pairs=int((d>0).sum()),
                        worsened_pairs=int((d<0).sum()),same_pairs=int((d==0).sum()),
                        twice_credit_difference_sum=int(d.sum())))
                summaries.append(dict(fit_target_scorer=model,method=method,feature_source=feature_source,
                    positive_assignment="A" if model=="model_1" else "B",chosen_C=entry["chosen_C"],
                    AUC=av,AUC_CI95={n:ci(v) for n,v in bs.items()},contrasts=contrasts,reproduction=checks))
                if feature_source==SOURCES[0]:
                    coefficient_out.append(dict(fit_target_scorer=model,method=method,
                        chosen_C=entry["chosen_C"],**c["coefficients"]))
                for i,p in enumerate(patients):
                    patient_out.append(dict(fit_target_scorer=model,method=method,feature_source=feature_source,
                        patient_id=p["patient_id"],assignment_group=p["group"],
                        mean_features=c["mean"][i].tolist(),half_range_features=c["half_range"][i].tolist(),
                        mean_logit_terms=c["mean_terms"][i].tolist(),spread_logit_terms=c["spread_terms"][i].tolist(),
                        mean_logit_component=float(c["M"][i]),spread_logit_component=float(c["H"][i]),
                        logits={n:float(v[i]) for n,v in scores.items()},
                        saved_full_probability=float(probability[i]),reconstructed_full_probability=float(expit(c["full"][i]))))
                pair_rows=[]
                for ip,pi in enumerate(pos):
                    for jn,ni in enumerate(neg):
                        pair_rows.append(dict(positive_assignment_patient=ids[pi],negative_assignment_patient=ids[ni],
                            **{n+"_margin":float(v[pi]-v[ni]) for n,v in scores.items()},
                            **{n+"_credit_twice":int(v[ip,jn]) for n,v in credits.items()},
                            saved_full_probability_credit_twice=int(saved_credit[ip,jn])))
                pair_out.append(dict(fit_target_scorer=model,method=method,feature_source=feature_source,pairs=pair_rows))
    for p,h in hashes.items():require(digest(p)==h,"unchanged source input")
    require(digest(__file__)==source,"unchanged source code")
    write(out/"coefficients.json",coefficient_out);write(out/"patients.json",patient_out);write(out/"pairs.json",pair_out)
    write(out/"analysis.json",dict(status="COMPUTED_PENDING_INDEPENDENT_VERIFICATION" if mismatch_total==0 else "FULL_RANK_REPRODUCTION_MISMATCH_REQUIRES_REVIEW",
        current_step=2,summaries=summaries,scorers=4,feature_sources=3,patient_records=480,pair_records=4800,
        bootstrap_resamples=2000,full_rank_mismatches=mismatch_total,new_fitting=False,new_GPU=False,
        interpretation="Within-frozen-scorer feature intervention; not the cause of performance difference between separately fitted models, new attack, or individual causal membership",
        original_primary_unchanged=True,success_gate=None,seconds_cpu=time.perf_counter()-start))
    write(out/"provenance.json",dict(source_sha256=source,input_sha256=hashes,
        protocol_sha256=digest(out/"protocol.json"),
        output_sha256={n:digest(out/n) for n in ("analysis.json","coefficients.json","patients.json","pairs.json")}))
    print(json.dumps(dict(output=str(out),full_rank_mismatches=mismatch_total,scorers=4,contexts=12,
        status="COMPUTED_PENDING_INDEPENDENT_VERIFICATION" if mismatch_total==0 else "FULL_RANK_REPRODUCTION_MISMATCH_REQUIRES_REVIEW")))

def self_test():
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(5)
    x=rng.normal(size=(40,2,26));x[:3,1]=x[:3,0]
    center=rng.normal(size=52);scale=rng.uniform(.1,3,size=52);weights=rng.normal(size=52)
    par=dict(coef=[weights.tolist()],scaler_mean=center.tolist(),scaler_scale=scale.tolist(),
             intercept=[.77],classes=[0,1])
    c=components(x,par)
    require(np.allclose(c["full"],c["direct"],rtol=0,atol=1e-12),"centered algebra nontrivial scaler")
    require(np.array_equal(x.max(1),c["mean"]+c["half_range"]) or np.allclose(x.max(1),c["mean"]+c["half_range"],atol=1e-15),"two-image identity")
    # Empirical fitted centers keep zero-spread reference meaningful.
    par["scaler_mean"]=np.r_[x.mean(1).mean(0),x.max(1).mean(0)].tolist()
    c=components(x,par)
    require(np.allclose(c["M"].mean(),0,atol=1e-13) and np.allclose(c["H"].mean(),0,atol=1e-13),"centered components")
    a=np.arange(20);b=np.arange(20,40);s=np.arange(40)%7;y=np.r_[np.ones(20),np.zeros(20)]
    ix=np.asarray([np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(23)])
    counts=np.asarray([np.bincount(i,minlength=40) for i in ix])
    measured=boot(credit(s,a,b),counts,a,b)
    require(np.allclose(measured,[roc_auc_score(y[i],s[i]) for i in ix],rtol=0,atol=1e-15),"half-tie shared bootstrap")
    # Detect saturation rather than equating sigmoid ties with logit rank ties.
    require(np.any(credit(np.arange(40)+1000.,a,b)!=credit(expit(np.arange(40)+1000.),a,b)),"saturation guard")
    print("PASS_CENTERED_LOGIT_IDENTITY_TIES_BOOTSTRAP_SATURATION; no actual result statistics")

def main():
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--print-policy",action="store_true")
    p.add_argument("--output-dir",type=Path,default=OUTPUT)
    p.add_argument("--expected-code-sha256")
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.print_policy:print(json.dumps(policy(),ensure_ascii=False));return
    require(a.expected_code_sha256,"expected frozen source required")
    run(a.output_dir.resolve(),a.expected_code_sha256.lower())

if __name__=="__main__":main()

