"""Independent sklearn/scipy verification of the eight-patient descriptive analysis."""
import json, math, statistics as st
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .common import ROOT, RUN, digest, read_csv, write_json

def main():
    source=RUN/"probe_smoke/training_coverage_v2/U_step_1000_n8"
    out=RUN/"analysis/U_step1000_n8_v1"
    analysis=json.loads((out/"analysis.json").read_text())
    assert digest(Path(__file__).with_name("analyze_u_smoke.py")) == analysis["analysis_code_sha256"]
    for name,sha in analysis["input_sha256"].items():
        assert digest(ROOT/name)==sha,name
    rows=json.loads((source/"results.json").read_text())
    metrics=list(analysis["metrics"])
    by=defaultdict(dict)
    for r in rows:by[r["patient_id"]][r["model"]]=r
    ids=sorted(by,key=int); aliases={p:f"P{i+1:02d}" for i,p in enumerate(ids)}
    close=lambda a,b: math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-13)
    pair_csv=read_csv(out/"paired_scores.csv")
    score_csv=read_csv(out/"scores.csv")
    assert len(pair_csv)==48 and len(score_csv)==16 and len(ids)==8
    pair_map={(r["patient"],r["metric"]):r for r in pair_csv}
    score_map={(r["patient"],r["model"]):r for r in score_csv}
    assert len(pair_map)==48 and len(score_map)==16
    for key,s in analysis["metrics"].items():
        aucs=[]
        for m in ["model_1","model_2"]:
            subset=[r for r in rows if r["model"]==m]
            value=float(roc_auc_score([r["realized_member"] for r in subset],[r[key] for r in subset]))
            assert close(value,s["models"][m]["auc"])
            aucs.append(value)
        assert close(st.mean(aucs),s["macro_auc"])
        xs=[by[p]["model_1"][key] for p in ids];ys=[by[p]["model_2"][key] for p in ids]
        assert close(float(spearmanr(xs,ys)[0]),s["model_rank_spearman"])
        ds=[]
        for p in ids:
            mem=next(r for r in by[p].values() if r["realized_member"])
            non=next(r for r in by[p].values() if not r["realized_member"])
            delta=mem[key]-non[key];ds.append(delta)
            rec=pair_map[(aliases[p],key)]
            assert close(delta,float(rec["member_minus_nonmember"]))
            assert close(mem[key],float(rec["member_score"])) and close(non[key],float(rec["nonmember_score"]))
            assert rec["member_model"]==mem["model"]
            for m in ["model_1","model_2"]:
                saved=score_map[(aliases[p],m)]
                assert int(saved["member"])==by[p][m]["realized_member"]
                assert close(float(saved[key]),by[p][m][key])
            loo=[]
            for m in ["model_1","model_2"]:
                sub=[r for r in rows if r["model"]==m and r["patient_id"]!=p]
                loo.append(float(roc_auc_score([r["realized_member"] for r in sub],[r[key] for r in sub])))
            assert close(st.mean(loo),s["leave_one_patient_out_macro_auc"][aliases[p]])
        assert sum(d>0 for d in ds)==s["paired_positive"]
        assert sum(d<0 for d in ds)==s["paired_negative"]
        assert close(st.mean(ds),s["paired_mean"]) and close(st.median(ds),s["paired_median"])
    for component,models in analysis["decomposition_diagnostic_auc"].items():
        rawkey="own_gain" if component=="own_gain_mean" else "reference_gain"
        for m,expected in models.items():
            sub=[r for r in rows if r["model"]==m]
            value=float(roc_auc_score([r["realized_member"] for r in sub],
                                      [st.mean(f[rawkey] for f in r["folds"]) for r in sub]))
            assert close(value,expected)
    assert analysis["new_target_queries"]==analysis["new_training_steps"]==analysis["new_patients"]==0
    assert not analysis["thresholds_fitted"] and not analysis["score_signs_fitted"]
    verification=dict(status="PASS",independent_implementation="sklearn AUROC; scipy Spearman; source-derived paired arithmetic",
                      metrics_verified=len(metrics),per_model_auc_values_verified=12,
                      leave_one_patient_out_values_verified=48,paired_score_rows_verified=48,
                      original_score_rows_verified=16,source_hashes_verified=len(analysis["input_sha256"]),
                      analysis_sha256=digest(out/"analysis.json"),verifier_code_sha256=digest(Path(__file__)),
                      scope="descriptive numeric consistency; not hypothesis confirmation")
    write_json(out/"verification.json",verification)
    print(json.dumps(verification))
if __name__=="__main__":main()

