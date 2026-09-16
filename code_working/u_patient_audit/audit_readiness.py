"""CPU-only readiness audit. Does not query models or fit attack scores."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, math, statistics as st
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .common import ROOT, RUN, digest, read_csv, write_json

OUT=RUN/"review_20260914"
RESEARCH=ROOT.parent/"CVPR 주제 탐색/research_2026-09-10"

def auc(y,s):
    return float(roc_auc_score(y,s))

def main():
    torch.set_num_threads(2)
    cache_meta=json.loads((RUN/"cache/summary.json").read_text())
    assert digest(RUN/"cache/cache.pt")==cache_meta["cache_sha256"]
    cache=torch.load(RUN/"cache/cache.pt",map_location="cpu",weights_only=True)
    h=cache["hidden"][cache["audit_prompt"]].float()
    mask=cache["token_mask"].bool().reshape(-1)
    assert h.shape[0]==1 and len(mask)==h.shape[1]
    active=h[0,mask,:]
    total_norm=float(h.norm());active_norm=float(active.norm())
    g=torch.Generator().manual_seed(260910)
    active_dims=active.numel()
    q=torch.linalg.qr(torch.randn((active_dims,8),generator=g),mode="reduced").Q
    basis_error=float((q.T@q-torch.eye(8)).abs().max())
    u_path=RUN/"probe_smoke/training_coverage_v2/U_step_1000_n8/results.json"
    rows=json.loads(u_path.read_text())
    folds=[f for r in rows for f in r["folds"]]
    norms=np.array([f["coefficient_norm"] for f in folds])
    missing=[k for k in ["coefficient_vector","support_objective_start","support_objective_end",
                          "optimization_trace","gradient_check"] if not any(k in f for f in folds)]
    evaluations=read_csv(RUN/"cohort/evaluation_images.csv")
    auxiliary=read_csv(RUN/"cohort/auxiliary_images.csv")
    ref_patients={r["patient_id"] for r in auxiliary if r["eval_role"]=="reference"}
    eval_patients={r["patient_id"] for r in evaluations}
    ref_map={r["image_id"]:r["patient_id"] for r in auxiliary if r["eval_role"]=="reference"}
    assert not ref_patients & eval_patients
    for r in evaluations:
        refs=cache["reference_matches"][r["image_id"]]
        assert len(refs)==2 and all(x in ref_map for x in refs)
        assert len({ref_map[x] for x in refs})==2
    t=np.array([st.mean([st.mean(f["own_gain"] for f in row["folds"]) for row in rows if row["patient_id"]==p])
                for p in sorted({r["patient_id"] for r in rows})])
    c=np.array([st.mean([st.mean(f["reference_gain"] for f in row["folds"]) for row in rows if row["patient_id"]==p])
                for p in sorted({r["patient_id"] for r in rows})])
    response=t-c
    cov=float(np.mean((t-t.mean())*(c-c.mean())))
    residual=float(np.var(response)-(np.var(t)+np.var(c)-2*cov))
    assert abs(residual)<1e-15
    e_report=json.loads((RUN/"probe_smoke/training_coverage_v2/E_step_1000_n1/report.json").read_text())
    model_checks={m:json.loads((RUN/"training_coverage_v2"/m/"verification_1000.json").read_text())
                  for m in ["model_1","model_2"]}
    # Pure mathematical counterexamples, not fitted to any patient data.
    toy_y=np.array([0,1,0,1,0,1,0,1]);identity=np.arange(8.)
    toy_s1=identity+.1*toy_y;toy_s2=identity+.1*(1-toy_y)
    assert np.array_equal(np.argsort(toy_s1),np.argsort(toy_s2))
    member_scores=np.where(toy_y,toy_s1,toy_s2)
    nonmember_scores=np.where(toy_y,toy_s2,toy_s1)
    assert np.all(member_scores>nonmember_scores)
    other_y=np.array([0,0,1,1]);toy_own=np.array([-.1,.1,-.1,.1])
    toy_ref=np.array([1.,1.,-1.,-1.]);toy_corrected=toy_own-toy_ref
    assert np.std(toy_corrected)>np.std(toy_own) and auc(other_y,toy_corrected)>auc(other_y,toy_own)
    source_paths=[Path(__file__).with_name(n) for n in [
        "probe.py","models.py","run_probe_smoke.py","verify_probe.py","build_cache.py","train_coverage.py",
        "quality_loss.py","analyze_u_smoke.py"]]
    source_paths += [u_path,RUN/"cohort/lock.json",RUN/"cache/summary.json",
                     RESEARCH/"EXPERIMENT_DESIGN.md",RESEARCH/"U_EXECUTION_PLAN.md"]
    facts=dict(scope="CPU and static readiness audit; no new model calls, patients, training, fitting or threshold selection",
        new_gpu_queries=0,new_patients=0,
        embedding=dict(shape=list(h.shape),active_tokens=int(mask.sum()),active_dimensions=active_dims,
            full_norm=total_norm,active_norm=active_norm,coefficient_radius=.05,
            max_perturbation_over_full_norm=.05,max_perturbation_over_active_norm=.05*total_norm/active_norm,
            basis_dimensions=8,dimension_fraction=8/active_dims,basis_orthogonality_max_error=basis_error),
        optimization_records=dict(folds=len(folds),radius_boundary_count=int(np.isclose(norms,.05,rtol=0,atol=1e-6).sum()),
            norm_min=float(norms.min()),norm_max=float(norms.max()),missing_diagnostics=missing,
            missing_logs_do_not_prove_gradient_bug=True),
        reference_checks=dict(evaluation_images_checked=len(evaluations),reference_patients=len(ref_patients),
                              two_distinct_external_reference_patients_per_image=True,target_score_used_in_selection=False),
        covariance=dict(own_sd=float(t.std()),reference_sd=float(c.std()),response_sd=float(response.std()),
            sd_ratio=float(response.std()/t.std()),variance_ratio=float(response.var()/t.var()),
            own_reference_correlation=float(np.corrcoef(t,c)[0,1]),covariance=cov,identity_residual=residual,
            diagnostic_ratio_is_not_membership_signal_to_noise=True),
        sensitivity_readiness=dict(E_patients=e_report["unique_fit_patients"],E_status=e_report["status"],
            E_discrimination_established=False,input_gradient_nonzero_check_established=False,
            fitted_incremental_attack_comparison_completed=False,medical_utility_established=False),
        training_integrity={m:dict(status=v["status"],steps=v["steps"],training_images=v["images"],
                    exposure_min=v["exposure_min"],exposure_max=v["exposure_max"],U_image_exposures=v["U_image_exposures"])
                    for m,v in model_checks.items()},
        mathematical_counterexamples=dict(
            rank_invariance_with_positive_membership_shift=dict(auc_model1=auc(toy_y,toy_s1),auc_model2=auc(1-toy_y,toy_s2),
                identical_ranks=True,positive_member_shift_patients=8,member_shift=.1),
            variance_increase_with_improved_discrimination=dict(auc_before=auc(other_y,toy_own),auc_after=auc(other_y,toy_corrected),
                sd_before=float(toy_own.std()),sd_after=float(toy_corrected.std())),
            scope="Illustrative algebra only; not actual-data fitting or a proposed attack"),
        source_sha256={str(p):digest(p) for p in source_paths},audit_code_sha256=digest(Path(__file__)))
    write_json(OUT/"review_facts.json",facts)
    print(json.dumps({k:v for k,v in facts.items() if k not in ["source_sha256","audit_code_sha256"]},indent=2))

if __name__=="__main__":main()

