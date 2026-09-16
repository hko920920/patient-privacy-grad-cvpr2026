"""Independent CPU verification of descriptive paired-score arithmetic.

Does not import the paired-score producer or recompute target-model responses.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
DEFAULT_AUDIT = RUN / "baseline_screen_20260914/paired_score_audit_20260914_v1"
SOURCE = RUN / "baseline_screen_20260914/pfami_v1"
SCORES = ("relative", "difference", "negative_loss", "base_relative")
SCENARIOS = ("U", "E")
NUMERICAL_ERRORS = []


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def close(actual, expected, label="", rtol=1e-10, atol=1e-12):
    if expected is None:
        assert actual is None, label
        return
    assert math.isfinite(float(actual)) and math.isfinite(float(expected)), label
    NUMERICAL_ERRORS.append(abs(float(actual)-float(expected)))
    assert math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol), (label, actual, expected)


def sign(value):
    return int(value > 0)-int(value < 0)


def H(value):
    return 1.0 if value > 0 else 0.0 if value < 0 else 0.5


def mean(values):
    values = list(map(float, values))
    return math.fsum(values)/len(values)


def variance(values):
    center = mean(values)
    return math.fsum((float(x)-center)**2 for x in values)/(len(values)-1)


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    da, db = [float(x)-ma for x in a], [float(x)-mb for x in b]
    denom = math.sqrt(math.fsum(x*x for x in da)*math.fsum(x*x for x in db))
    return math.fsum(x*y for x, y in zip(da, db))/denom if denom else None


def percentile(values, probability):
    ordered = sorted(map(float, values))
    index = (len(ordered)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo]+(index-lo)*(ordered[hi]-ordered[lo])


def bootstrap_weights():
    random = np.random.Generator(np.random.PCG64(260914))
    a = random.integers(0, 20, size=(2000, 20), dtype=np.int64)
    b = random.integers(20, 40, size=(2000, 20), dtype=np.int64)
    draws = np.concatenate((a, b), axis=1)
    wa = np.stack([np.bincount(row, minlength=20) for row in a]).astype(float)
    wb = np.stack([np.bincount(row-20, minlength=20) for row in b]).astype(float)
    return wa, wb, hashlib.sha256(draws.tobytes()).hexdigest()


def evaluate(s1, s2, wa, wb):
    """Use original patient order A20 then B20; produce independent statistics."""
    a, b = list(map(float, s1)), list(map(float, s2))
    assert len(a) == len(b) == 40 and all(math.isfinite(x) for x in a+b)
    common = [(x+y)/2 for x, y in zip(a, b)]
    half = [(x-y)/2 for x, y in zip(a, b)]
    aligned = [(1 if i < 20 else -1)*(a[i]-b[i]) for i in range(40)]
    labels = [1]*20+[0]*20
    order_counts = Counter()
    for i in range(40):
        for j in range(i+1, 40):
            first, second = sign(a[i]-a[j]), sign(b[i]-b[j])
            if first == second == 0:
                order_counts["both_tied"] += 1
            elif first == 0 or second == 0:
                order_counts["one_tied"] += 1
            elif first == second:
                order_counts["agreement"] += 1
            else:
                order_counts["reversal"] += 1
    assert sum(order_counts.values()) == 780
    auc1 = float(roc_auc_score(labels, a))
    auc2 = float(roc_auc_score([1-x for x in labels], b))
    common1 = float(roc_auc_score(labels, common))
    common2 = float(roc_auc_score([1-x for x in labels], common))
    excess = (auc1+auc2)/2-.5
    pairs = []
    pair_values = np.empty((20, 20), dtype=float)
    for i in range(20):
        for j in range(20, 40):
            cm, dm = common[i]-common[j], half[i]-half[j]
            direct = .5*(H(a[i]-a[j])+H(b[j]-b[i]))-.5
            identity = .5*(H(cm+dm)-H(cm-dm))
            # Membership pair contribution equality includes fixed half credit for ties.
            close(direct, identity, (i, j, "400-pair identity"), rtol=0, atol=0)
            pair_values[i, j-20] = direct
            pairs.append(dict(a_index=i, b_index=j, common_margin=cm, change_margin=dm,
                model1_margin=a[i]-a[j], model2_A_minus_B_margin=b[i]-b[j],
                contribution=direct, changed=H(a[i]-a[j]) != H(b[i]-b[j])))
    close(mean(p["contribution"] for p in pairs), excess)
    delta_boot = (wa @ np.asarray(aligned[:20])+wb @ np.asarray(aligned[20:]))/40
    excess_boot = np.einsum("bi,ij,bj->b", wa, pair_values, wb, optimize=True)/400
    var1, var2, varc, vard = map(variance, (a, b, common, half))
    close((var1+var2)/2, varc+vard)
    groups = {}
    for group, values in (("all", aligned), ("A", aligned[:20]), ("B", aligned[20:])):
        ordered = sorted(values)
        groups[group] = dict(mean=mean(values), median=(ordered[(len(values)-1)//2]+ordered[len(values)//2])/2,
            positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values), zero=sum(x == 0 for x in values))
    return dict(s1=a, s2=b, common=common, half_change=half, aligned_delta=aligned,
        pearson=pearson(a, b), spearman=pearson(rankdata(a, method="average"), rankdata(b, method="average")),
        rank_counts=dict(order_counts), variance=dict(s1=var1, s2=var2, common=varc, half_change=vard,
            common_share=varc/(varc+vard) if varc+vard else None), delta_groups=groups,
        aucs=dict(model_1=auc1, model_2=auc2, common_model_1=common1, common_model_2=common2,
            model_1_gain=auc1-common1, model_2_gain=auc2-common2, mean_auc_excess=excess),
        pair_counts=dict(changed=sum(p["changed"] for p in pairs), positive=sum(p["contribution"] > 0 for p in pairs),
            negative=sum(p["contribution"] < 0 for p in pairs), zero=sum(p["contribution"] == 0 for p in pairs)),
        pairs=pairs, delta_mean_ci95=[percentile(delta_boot, q) for q in (.025, .975)],
        mean_auc_excess_ci95=[percentile(excess_boot, q) for q in (.025, .975)])


def self_test():
    wa, wb, _ = bootstrap_weights()
    unchanged = evaluate(list(range(40)), list(range(40)), wa, wb)
    close(unchanged["pearson"], 1)
    close(unchanged["spearman"], 1)
    assert unchanged["rank_counts"].get("agreement") == 780
    assert unchanged["delta_mean_ci95"] == unchanged["mean_auc_excess_ci95"] == [0, 0]
    pure = evaluate([1]*20+[0]*20, [0]*20+[1]*20, wa, wb)
    close(pure["pearson"], -1)
    close(pure["aucs"]["mean_auc_excess"], .5)
    assert pure["delta_mean_ci95"] == [1, 1] and pure["mean_auc_excess_ci95"] == [.5, .5]
    assert pure["pair_counts"] == dict(changed=400, positive=400, negative=0, zero=0)
    return dict(status="PASS_SYNTHETIC_CPU_ARITHMETIC", cases=["identical target scores", "opposite perfectly aligned scores with ties"],
        GPU_executions=0, real_derived_statistics_computed=False)


def original_data(audit):
    contract = load(audit / "contract.json")
    assert contract["schema"] == "paired-score-descriptive-audit/v1"
    scope = contract["scope"]
    for key, value in dict(patients=40, role="selection", groups={"A": 20, "B": 20},
        models=["model_1", "model_2"], scenarios=list(SCENARIOS), scores=list(SCORES), primary=["U", "relative"],
        new_gpu_forward=0, new_backward=0, new_training=0, new_images_scored=0).items():
        assert scope[key] == value, key
    assert contract["bootstrap"]["seed"] == 260914 and contract["bootstrap"]["replicates"] == 2000
    assert contract["bootstrap"]["strata"] == {"A": 20, "B": 20}
    assert contract["bootstrap"]["shared_across_scenarios_scores"] is True
    integrity = load(audit / "data_integrity.json")
    assert integrity["status"] == "PASS_LABEL_NOISE_AND_BASE_CANCELLATION"
    for filename, sha in integrity["input_sha256"].items():
        path = Path(filename).resolve()
        assert path.is_relative_to(ROOT.parent.resolve()) and digest(path) == sha, filename
    source = load(SOURCE / "analysis.json")
    prior_verification = load(SOURCE / "verification.json")
    assert prior_verification["status"] == "PASS_INTEGRITY_RESIDUAL_AND_STATISTICAL_RECOMPUTATION"
    for name, sha in prior_verification["input_sha256"].items():
        path = (SOURCE / name).resolve()
        assert path.is_relative_to(SOURCE.resolve()) and digest(path) == sha, name
    original_protocol = load(SOURCE / "protocol.json")
    for name, sha in original_protocol["original_protected_sha256"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and digest(path) == sha, name
    records = source["patient_scores"]
    rows = {(r["model"], r["scenario"], str(r["patient_id"])): r for r in records}
    assert len(rows) == len(records) == 160
    order = source["bootstrap"]["patient_order"]
    assert len(order) == len(set(order)) == 40
    assert order[:20] == sorted(order[:20], key=int) and order[20:] == sorted(order[20:], key=int)
    for model in ("model_1", "model_2"):
        for scenario in SCENARIOS:
            for i, patient in enumerate(order):
                row = rows[model, scenario, patient]
                assert row["eval_role"] == "selection"
                assert row["assignment_group"] == ("A" if i < 20 else "B")
                assert row["member"] == int((i < 20) == (model == "model_1"))
                assert set(row["scores"]) == set(SCORES)
    return contract, source, rows, order


def check_group(actual, expected, n):
    assert actual["n"] == n
    for field in ("positive", "negative", "zero"):
        assert actual[field] == expected[field], field
    close(actual["mean"], expected["mean"])
    close(actual["median"], expected["median"])


def check_interval(actual, expected):
    assert len(actual) == len(expected) == 2
    for x, y in zip(actual, expected):
        close(x, y)


def check_summary(actual, expected):
    assert actual["patients"] == 40
    for field in ("pearson", "spearman"):
        close(actual["correlations"][field], expected[field], field)
    rank = expected["rank_counts"]
    assert actual["rank_pairs"] == dict(total=780, agreement=rank.get("agreement", 0),
        reversal=rank.get("reversal", 0), one_model_tie=rank.get("one_tied", 0), both_model_tie=rank.get("both_tied", 0))
    av, ev = actual["variance"], expected["variance"]
    assert av["ddof"] == 1
    for field, source in (("s1_variance", "s1"), ("s2_variance", "s2"),
                          ("common_variance", "common"), ("half_change_variance", "half_change"),
                          ("common_share", "common_share")):
        close(av[field], ev[source], field)
    close(av["common_sd"], math.sqrt(ev["common"]))
    close(av["half_change_sd"], math.sqrt(ev["half_change"]))
    left, right = (ev["s1"]+ev["s2"])/2, ev["common"]+ev["half_change"]
    close(av["identity_left"], left)
    close(av["identity_right"], right)
    close(av["identity_error"], left-right)
    check_group(actual["aligned_delta"], expected["delta_groups"]["all"], 40)
    assert set(actual["aligned_delta"]["by_group"]) == {"A", "B"}
    for group in ("A", "B"):
        check_group(actual["aligned_delta"]["by_group"][group], expected["delta_groups"][group], 20)
    check_interval(actual["aligned_delta"]["mean_bootstrap95"], expected["delta_mean_ci95"])
    aa, ea = actual["aucs"], expected["aucs"]
    for field, source in (("model_1", "model_1"), ("model_2", "model_2"),
                          ("common_A_labels", "common_model_1"), ("common_B_labels", "common_model_2"),
                          ("model_1_gain_over_common", "model_1_gain"), ("model_2_gain_over_common", "model_2_gain"),
                          ("mean_auc_excess", "mean_auc_excess")):
        close(aa[field], ea[source], field)
    close(aa["common_average"], .5)
    check_interval(aa["mean_auc_excess_bootstrap95"], expected["mean_auc_excess_ci95"])
    counts, provided = expected["pair_counts"], actual["pair_contributions"]
    assert provided["total"] == 400
    for field, source in (("changed", "changed"), ("net_positive", "positive"), ("net_negative", "negative"), ("zero", "zero")):
        assert provided[field] == counts[source], field
    total = math.fsum(row["contribution"] for row in expected["pairs"])
    close(provided["sum"], total)
    close(provided["mean"], total/400)
    close(provided["preserved_auc_excess_error"], total/400-ea["mean_auc_excess"])


def verify(audit):
    began = time.perf_counter()
    NUMERICAL_ERRORS.clear()
    audit = Path(audit).resolve()
    output = audit / "independent_verification.json"
    if output.exists():
        raise FileExistsError(output)
    contract, original, rows, order = original_data(audit)
    protocol = load(audit / "protocol.json")
    actual = load(audit / "analysis.json")
    assert protocol["schema"] == "paired-score-audit-provenance/v1"
    assert protocol["mode"] == "post_hoc_descriptive_cpu_only"
    assert protocol["new_gpu_forward"] == protocol["new_backward"] == 0
    assert actual["schema"] == "paired-score-descriptive-audit-result/v1"
    assert actual["protocol_sha256"] == digest(audit / "protocol.json")
    assert actual["contract_sha256"] == protocol["contract_sha256"] == digest(audit / "contract.json")
    code = Path(__file__).with_name("paired_score_audit.py").resolve()
    assert Path(protocol["analysis_code_path"]).resolve() == code
    assert actual["analysis_code_sha256"] == protocol["analysis_code_sha256"] == digest(code)
    wanted_inputs = {str((SOURCE / name).resolve()) for name in
                     ("analysis.json", "report.json", "verification.json", "protocol.json", "image_scores.json")}
    assert set(protocol["input_sha256"]) == wanted_inputs
    for filename, sha in protocol["input_sha256"].items():
        assert digest(filename) == sha, filename
    for field in ("new_gpu_forward", "new_backward", "new_training"):
        assert actual[field] == 0
    for field in ("groundtruth_aligned_delta_auc_computed", "causal_membership_effect_identified",
                  "attack_available_inputs_changed", "automatic_followup"):
        assert actual[field] is False
    assert actual["independent_patients"] == 40 and actual["rank_pairs_are_not_independent_observations"] is True
    assert actual["constraints"] == contract["constraints"]
    wa, wb, drawsha = bootstrap_weights()
    assert actual["bootstrap"]["patient_order"] == order
    assert actual["bootstrap"]["seed"] == 260914 and actual["bootstrap"]["replicates"] == 2000
    assert actual["bootstrap"]["indices_sha256"] == original["bootstrap"]["indices_sha256"] == drawsha
    assert actual["bootstrap"]["interval"] == contract["bootstrap"]["interval"]
    summaries = actual["summaries"]
    assert [(s["scenario"], s["score"]) for s in summaries] == [(s, k) for s in SCENARIOS for k in SCORES]
    original_auc = {(r["model"], r["scenario"], r["score"]): r["auc"] for r in original["metrics"]}
    expectations = {}
    for summary in summaries:
        scenario, score = summary["scenario"], summary["score"]
        values = [[rows[model, scenario, patient]["scores"][score] for patient in order] for model in ("model_1", "model_2")]
        expected = evaluate(*values, wa, wb)
        expectations[scenario, score] = expected
        check_summary(summary, expected)
        for model in ("model_1", "model_2"):
            close(expected["aucs"][model], original_auc[model, scenario, score])
    patient_values = actual["patient_values"]
    assert len(patient_values) == 320
    keys = [(r["scenario"], r["score"], str(r["patient_id"])) for r in patient_values]
    assert keys == [(s, k, p) for s in SCENARIOS for k in SCORES for p in order]
    position = {p: i for i, p in enumerate(order)}
    for row in patient_values:
        i = position[str(row["patient_id"])]
        assert row["group"] == ("A" if i < 20 else "B")
        expected = expectations[row["scenario"], row["score"]]
        for field in ("s1", "s2", "common", "half_change", "aligned_delta"):
            close(row[field], expected[field][i], field)
    assert actual["primary_patient_values"] == [r for r in patient_values if (r["scenario"], r["score"]) == ("U", "relative")]
    pair_rows = actual["primary_pair_margins"]
    assert len(pair_rows) == 400
    primary = expectations["U", "relative"]
    for row, expected in zip(pair_rows, primary["pairs"]):
        assert row["patient_A"] == order[expected["a_index"]] and row["patient_B"] == order[expected["b_index"]]
        for field in ("common_margin", "change_margin", "model1_margin", "model2_A_minus_B_margin", "contribution"):
            close(row[field], expected[field], field)
        close(row["H_a_plus_b"], H(expected["common_margin"]+expected["change_margin"]), rtol=0, atol=0)
        close(row["H_a_minus_b"], H(expected["common_margin"]-expected["change_margin"]), rtol=0, atol=0)
    assert [r["scenario"] for r in actual["base_relative_identity"]] == list(SCENARIOS)
    for row in actual["base_relative_identity"]:
        original_values = expectations[row["scenario"], "relative"]
        base_values = expectations[row["scenario"], "base_relative"]
        for field, statistic in (("half_change", "half_change_max_abs"), ("aligned_delta", "aligned_delta_max_abs")):
            error = max(abs(x-y) for x, y in zip(original_values[field], base_values[field]))
            close(row[statistic], error)
            assert error < 1e-12
    result = dict(schema="paired-score-independent-verification/v1", status="PASS_INDEPENDENT_CPU_STATISTICAL_RECOMPUTATION",
        scenario_score_conditions=8, patient_value_rows=320, patient_scalar_values_checked=1600,
        unordered_rank_pairs_checked=8*780, AB_pair_contributions_recomputed=8*400, primary_raw_pair_margins_checked=400,
        AUC_values_checked=32, original_AUC_values_reproduced=16, correlation_values_checked=16,
        bootstrap_intervals_checked=16, bootstrap_replicates=2000, indices_sha256=drawsha,
        methods=dict(AUC="sklearn.roc_auc_score", Spearman="scipy rankdata average ranks then direct Pearson",
            moments="math.fsum direct centered sample variance/covariance", bootstrap="A/B resample multiplicity weights; weighted patient sums and 20x20 pair contributions; manual linear percentile"),
        max_absolute_numeric_difference=max(NUMERICAL_ERRORS, default=0),
        input_sha256={str(p.resolve()): digest(p) for p in [audit/"contract.json", audit/"protocol.json", audit/"analysis.json",
            audit/"data_integrity.json", SOURCE/"analysis.json", SOURCE/"verification.json", code]},
        verifier_sha256=digest(Path(__file__)), source_analysis_module_imported=False,
        GPU_executions=0, new_target_scores=0, efficacy_or_causal_claim=False, seconds_cpu=time.perf_counter()-began)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else verify(args.audit_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
