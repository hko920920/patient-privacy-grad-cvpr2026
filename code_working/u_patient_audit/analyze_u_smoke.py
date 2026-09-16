"""CPU-only descriptive analysis of the already completed eight-patient U probe.

No new patients, model queries, score fitting, sign selection or calibration.
The paired-model comparison is an analyst diagnostic, not an attack input.
"""
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .common import RUN, ROOT, digest, read_csv, write_csv, write_json, verify_inputs

SOURCE = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
OUT = RUN / "analysis/U_step1000_n8_v1"
METRICS = [
    ("response", "Candidate response"),
    ("loss_mean", "Mean negative loss"),
    ("loss_max", "Max negative loss"),
    ("base_difference_mean", "Mean base - target loss"),
    ("projected_conditioning_gradient_dot", "Gradient dot product"),
    ("projected_conditioning_gradient_cosine", "Gradient cosine"),
]
MODELS = ("model_1", "model_2")


def auc(rows, key):
    pos = [r[key] for r in rows if r["realized_member"] == 1]
    neg = [r[key] for r in rows if r["realized_member"] == 0]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ranks(values):
    # Average rank for exact ties; rank 1 is the lowest score.
    return [1 + sum(y < x for y in values) + .5 * (sum(y == x for y in values) - 1)
            for x in values]


def corr(xs, ys):
    xm, ym = st.mean(xs), st.mean(ys)
    denominator = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denominator if denominator else None


def analyze():
    if OUT.exists():
        raise RuntimeError("analysis output already exists; do not overwrite a completed analysis")
    verify_inputs()
    results = json.loads((SOURCE / "results.json").read_text())
    report = json.loads((SOURCE / "report.json").read_text())
    check = json.loads((SOURCE / "verification.json").read_text())
    assert check["status"] == "PASS"
    assert digest(SOURCE / "results.json") == check["results_sha256"]
    assert digest(SOURCE / "report.json") == check["report_sha256"]
    assert digest(Path(__file__).with_name("probe.py")) == report["probe_code_sha256"]
    assert report["scenario"] == "U" and report["unique_fit_patients"] == 8
    assert not report["calibration_test_patients_queried"]
    manifest = read_csv(RUN / "cohort/evaluation_images.csv")
    images = {r["image_id"]: r for r in manifest}
    trains = {m: {r["image_id"] for r in read_csv(RUN / "cohort" / (m + "_train.csv"))} for m in MODELS}
    by_patient = defaultdict(dict)
    for r in results:
        assert r["scenario"] == "U" and r["eval_role"] == "fit"
        assert r["weights_unchanged"] and r["forward"] == 312 and r["backward"] == 72
        assert r["declared_member"] == r["realized_member"]
        assert digest(RUN / "training_coverage_v2" / r["model"] / "step_1000.pt") == r["checkpoint_sha256"]
        assert r["model"] not in by_patient[r["patient_id"]]
        by_patient[r["patient_id"]][r["model"]] = r
        for key, _ in METRICS:
            assert math.isfinite(r[key])
        for f in r["folds"]:
            for role in ("support", "query"):
                item = images[f[role]]
                assert item["patient_id"] == r["patient_id"] and item["eval_role"] == "fit"
                assert item["record_role"] == "U_observed" and f[role] not in trains[r["model"]]
        group = images[r["folds"][0]["support"]]["assignment_group"]
        assert r["realized_member"] == int(group == ("A" if r["model"] == "model_1" else "B"))
        assert math.isclose(r["response"], st.mean(f["own_gain"] - f["reference_gain"] for f in r["folds"]), abs_tol=1e-12)
        r["own_gain_mean"] = st.mean(f["own_gain"] for f in r["folds"])
        r["reference_gain_mean"] = st.mean(f["reference_gain"] for f in r["folds"])
    assert len(results) == 16 and len(by_patient) == 8
    for pair in by_patient.values():
        assert set(pair) == set(MODELS)
        assert sorted(r["realized_member"] for r in pair.values()) == [0, 1]
        assert sorted(f["support"] for f in pair["model_1"]["folds"]) == sorted(f["support"] for f in pair["model_2"]["folds"])
    for m in MODELS:
        assert Counter(r["realized_member"] for r in results if r["model"] == m) == Counter({0: 4, 1: 4})
    # Display aliases only; deterministic ordering comes from existing pseudonymous NIH identifiers.
    patient_ids = sorted(by_patient, key=int)
    aliases = {p: f"P{i + 1:02d}" for i, p in enumerate(patient_ids)}
    metrics = {}
    paired_rows = []
    for key, label in METRICS:
        xs = [by_patient[p]["model_1"][key] for p in patient_ids]
        ys = [by_patient[p]["model_2"][key] for p in patient_ids]
        deltas = []
        for p in patient_ids:
            pair = by_patient[p]
            member = next(r for r in pair.values() if r["realized_member"])
            nonmember = next(r for r in pair.values() if not r["realized_member"])
            delta = member[key] - nonmember[key]
            deltas.append(delta)
            paired_rows.append(dict(patient=aliases[p], metric=key, member_model=member["model"],
                                    nonmember_score=nonmember[key], member_score=member[key],
                                    member_minus_nonmember=delta))
        model_stats = {}
        for m in MODELS:
            subset = [r for r in results if r["model"] == m]
            pos = [r[key] for r in subset if r["realized_member"]]
            neg = [r[key] for r in subset if not r["realized_member"]]
            model_stats[m] = dict(auc=auc(subset, key), members=len(pos), nonmembers=len(neg),
                                  member_mean=st.mean(pos), nonmember_mean=st.mean(neg),
                                  member_range=[min(pos), max(pos)], nonmember_range=[min(neg), max(neg)])
        loo = {aliases[p]: st.mean(auc([r for r in results if r["model"] == m and r["patient_id"] != p], key)
                                     for m in MODELS) for p in patient_ids}
        ordering_agreement = sum((xs[i] > xs[j]) == (ys[i] > ys[j]) and (xs[i] == xs[j]) == (ys[i] == ys[j])
                                 for i in range(8) for j in range(i))
        metrics[key] = dict(label=label, orientation="higher score indicates membership; not fitted to these labels",
                            models=model_stats, macro_auc=st.mean(s["auc"] for s in model_stats.values()),
                            paired_positive=sum(d > 0 for d in deltas), paired_negative=sum(d < 0 for d in deltas),
                            paired_ties=sum(d == 0 for d in deltas), paired_mean=st.mean(deltas),
                            paired_median=st.median(deltas), paired_range=[min(deltas), max(deltas)],
                            model_rank_spearman=corr(ranks(xs), ranks(ys)),
                            patient_pair_rank_agreements=ordering_agreement, patient_pair_rank_total=28,
                            leave_one_patient_out_macro_auc=loo,
                            leave_one_patient_out_range=[min(loo.values()), max(loo.values())])
    components = {key: {m: auc([r for r in results if r["model"] == m], key) for m in MODELS}
                  for key in ("own_gain_mean", "reference_gain_mean")}
    # The public base and query noise are shared: base adjustment changes cross-patient ranking,
    # but its within-patient member/nonmember difference cancels exactly (up to roundoff).
    base_cancellation_error = max(abs(a["member_minus_nonmember"] - b["member_minus_nonmember"])
                                  for a, b in zip([x for x in paired_rows if x["metric"] == "loss_mean"],
                                                  [x for x in paired_rows if x["metric"] == "base_difference_mean"]))
    assert base_cancellation_error < 1e-10
    inputs = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in
              (SOURCE / "results.json", SOURCE / "report.json", SOURCE / "verification.json",
               RUN / "cohort/lock.json", RUN / "cohort/evaluation_images.csv")}
    result = dict(scope="eight existing fit patients; descriptive U diagnostic only", patients=8,
                  model_patient_observations=16, input_sha256=inputs,
                  analysis_code_sha256=digest(Path(__file__)),
                  score_signs_fitted=False, classifiers_fitted=False, thresholds_fitted=False,
                  new_target_queries=0, new_training_steps=0, new_patients=0,
                  calibration_or_test_evaluated=False, paired_model_scores_are_analyst_only=True,
                  observations_are_not_16_independent_patients=True, independent_training_seed_count=1,
                  auc_direction="fixed high score => member for every model; no per-model sign reversal",
                  macro_auc_is_descriptive_not_independent_replication=True,
                  metrics=metrics, decomposition_diagnostic_auc=components,
                  base_adjustment_paired_difference_max_error=base_cancellation_error,
                  empirical_fpr_resolution_per_model=.25,
                  low_fpr_performance_estimated=False,
                  uncertainty="leave-one-patient-out sensitivity only; not a confidence interval or a p-value",
                  limitations=["4 members and 4 nonmembers per model; fit data, not confirmatory test",
                               "Two models share initialization/background and switch many patients jointly",
                               "Paired deltas do not identify a single-patient causal or DP-neighbor effect",
                               "No calibration, fitted attack, strong-baseline reproduction or fair runtime comparison",
                               "Small or absent signal here does not establish impossibility of patient membership inference"])
    write_json(OUT / "analysis.json", result)
    write_csv(OUT / "paired_scores.csv", paired_rows)
    rows = [dict(patient=aliases[r["patient_id"]], model=r["model"], member=r["realized_member"],
                 **{k: r[k] for k, _ in METRICS},
                 own_gain_mean=r["own_gain_mean"], reference_gain_mean=r["reference_gain_mean"]) for r in results]
    write_csv(OUT / "scores.csv", rows)
    plot(result, paired_rows)
    print(json.dumps({k: {x: s[x] for x in ["macro_auc", "paired_positive", "paired_negative",
                      "paired_mean", "paired_median", "model_rank_spearman", "patient_pair_rank_agreements",
                      "leave_one_patient_out_range"]} for k, s in metrics.items()}, indent=2))
    print(json.dumps({"decomposition": components, "output": str(OUT)}))


def plot(result, paired_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    for i, (key, label) in enumerate(METRICS):
        for m, color, offset in [("model_1", "#2466a0", -.12), ("model_2", "#cc6831", .12)]:
            axes[0].scatter(result["metrics"][key]["models"][m]["auc"], i + offset,
                            color=color, label=m.replace("_", " ") if i == 0 else None, s=55)
    axes[0].set(yticks=range(len(METRICS)), yticklabels=[v for _, v in METRICS],
                xlim=(-.03, 1.03), xlabel="AUROC (higher score = member)",
                title="4 members + 4 nonmembers in each model")
    axes[0].axvline(.5, color="#666666", linestyle="--", linewidth=1)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    selected = [r for r in paired_rows if r["metric"] == "response"]
    for i, r in enumerate(selected):
        color = "#2466a0" if r["member_model"] == "model_1" else "#cc6831"
        axes[1].plot([0, 1], [r["nonmember_score"], r["member_score"]], "o-", color=color, alpha=.8)
        axes[1].annotate(r["patient"], (1, r["member_score"]), xytext=(5, 0),
                         textcoords="offset points", fontsize=8)
    axes[1].set(xticks=[0, 1], xticklabels=["Not included", "Included"], xlim=(-.2, 1.3),
                ylabel="Candidate response score", title="Same U photos; membership switched")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle("Existing U diagnostic: 8 unique patients, 2 paired models\n"
                 "Descriptive development results; no threshold fitting or new GPU execution", fontsize=13)
    fig.savefig(OUT / "u_eight_patient_diagnostic.png", dpi=180)
    fig.savefig(OUT / "u_eight_patient_diagnostic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    analyze()

