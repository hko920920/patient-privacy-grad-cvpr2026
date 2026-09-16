"""Predeclared CPU analysis for SecMI fit8/selection40 diagnostic.

Primary = mean of the two negative image L2 scores. Maximum is descriptive.
Bootstrap patient draws preserve assignment strata and are shared across targets
and E/U scenarios. These are exploratory marginal intervals, not confirmatory
or simultaneous-coverage confidence statements.
"""
import hashlib
import math

import numpy as np

MODELS = ("model_1", "model_2")
SCENARIOS = ("E", "U")
SCORE_FIELDS = ("score_mean", "score_max")
BOOTSTRAP_SEED = 260914
BOOTSTRAP_REPLICATES = 2000


def auc(scores, members):
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(members, dtype=np.int64)
    assert values.ndim == labels.ndim == 1 and values.shape == labels.shape
    assert np.isfinite(values).all() and set(labels.tolist()) == {0, 1}
    positive, negative = values[labels == 1], values[labels == 0]
    comparisons = positive[:, None]-negative[None, :]
    return float(np.mean((comparisons > 0).astype(np.float64)+.5*(comparisons == 0)))


def bootstrap_indices():
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    a = generator.integers(0, 20, size=(BOOTSTRAP_REPLICATES, 20), dtype=np.int64)
    b = generator.integers(20, 40, size=(BOOTSTRAP_REPLICATES, 20), dtype=np.int64)
    return np.concatenate((a, b), axis=1)


def bootstrap_aucs(scores, members, draws):
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(members, dtype=np.int64)
    assert values.shape == labels.shape == (40,)
    assert draws.shape == (BOOTSTRAP_REPLICATES, 40)
    assert set(labels[:20]) in ({0}, {1}) and set(labels[20:]) == {1-int(labels[0])}
    sample = values[draws]
    positive = sample[:, :20] if labels[0] else sample[:, 20:]
    negative = sample[:, 20:] if labels[0] else sample[:, :20]
    comparisons = positive[:, :, None]-negative[:, None, :]
    return np.mean((comparisons > 0).astype(np.float64)+.5*(comparisons == 0), axis=(1, 2))


def analyse(patient_records, reference_fit_scores=None):
    """Accept normalized model/scenario/role/patient/group/member/score records."""
    rows = {}
    assignments = {}
    role_patients = {"fit": set(), "selection": set()}
    for row in patient_records:
        model, scenario, role = row["model"], row["scenario"], row["eval_role"]
        patient = str(row["patient_id"])
        assert model in MODELS and scenario in SCENARIOS and role in role_patients
        assert row["assignment_group"] in ("A", "B")
        member = int(row["member"])
        assert member == int(row["assignment_group"] == ("A" if model == "model_1" else "B"))
        assert all(math.isfinite(float(row[k])) for k in SCORE_FIELDS)
        key = model, scenario, role, patient
        assert key not in rows
        rows[key] = dict(row, patient_id=patient, member=member)
        previous = assignments.setdefault(patient, (role, row["assignment_group"]))
        assert previous == (role, row["assignment_group"])
        role_patients[role].add(patient)
    assert len(role_patients["fit"]) == 8 and len(role_patients["selection"]) == 40
    assert not role_patients["fit"] & role_patients["selection"]
    assert len(rows) == 192
    order = {}
    for role, ids in role_patients.items():
        groups = {group: sorted(p for p in ids if assignments[p][1] == group) for group in ("A", "B")}
        expected = 4 if role == "fit" else 20
        assert len(groups["A"]) == len(groups["B"]) == expected
        order[role] = groups["A"]+groups["B"]
        for model in MODELS:
            for scenario in SCENARIOS:
                assert all((model, scenario, role, patient) in rows for patient in order[role])
    draws = bootstrap_indices()
    metrics = {}
    for role in ("fit", "selection"):
        role_metrics = {}
        for scenario in SCENARIOS:
            scenario_metrics = {}
            for score in SCORE_FIELDS:
                models = {}
                for model in MODELS:
                    records = [rows[model, scenario, role, p] for p in order[role]]
                    values = [r[score] for r in records]
                    labels = [r["member"] for r in records]
                    result = {
                        "auc": auc(values, labels),
                        "patients": len(records), "members": sum(labels), "nonmembers": len(labels)-sum(labels),
                        "member_score_mean": float(np.mean([v for v, y in zip(values, labels) if y])),
                        "nonmember_score_mean": float(np.mean([v for v, y in zip(values, labels) if not y])),
                    }
                    if role == "selection":
                        boot = bootstrap_aucs(values, labels, draws)
                        result["patient_bootstrap_percentile_95"] = np.quantile(
                            boot, [.025, .975], method="linear").tolist()
                        result["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
                    models[model] = result
                paired = []
                for patient in order[role]:
                    member_model = "model_1" if assignments[patient][1] == "A" else "model_2"
                    other = "model_2" if member_model == "model_1" else "model_1"
                    difference = rows[member_model, scenario, role, patient][score]-rows[other, scenario, role, patient][score]
                    paired.append({"patient_id": patient, "member_model": member_model,
                                   "member_minus_nonmember": difference})
                scenario_metrics[score] = {
                    "models": models,
                    "paired_model_diagnostic_analyst_only": paired,
                    "paired_positive": sum(p["member_minus_nonmember"] > 0 for p in paired),
                    "paired_mean": float(np.mean([p["member_minus_nonmember"] for p in paired])),
                    "primary": score == "score_mean",
                }
            role_metrics[scenario] = scenario_metrics
        metrics[role] = role_metrics
    gates = {}
    for scenario in SCENARIOS:
        model_stats = metrics["selection"][scenario]["score_mean"]["models"]
        lower = {model: model_stats[model]["patient_bootstrap_percentile_95"][0] for model in MODELS}
        supported = all(bound > .5 for bound in lower.values())
        gates[scenario] = {
            "primary_score": "score_mean", "role": "selection", "patients": 40,
            "rule": "Both target models must have primary AUC marginal bootstrap lower95 > 0.5.",
            "per_model_lower95": lower,
            "screening_signal_supported_in_this_setting": supported,
            "decision": "SCREENING_SIGNAL_SUPPORTED_IN_THIS_SETTING" if supported else "HOLD_CURRENT_CANDIDATE_EXPANSION",
            "not_a_no_leakage_or_impossibility_conclusion": True,
            "not_confirmatory_test_performance": True,
        }
    reference_comparison = None
    if reference_fit_scores is not None:
        reference_comparison = {}
        for model in MODELS:
            records = [rows[model, "U", "fit", p] for p in order["fit"]]
            values = [reference_fit_scores[model, p] for p in order["fit"]]
            reference_comparison[model] = {
                "reference_corrected_response_auc": auc(values, [r["member"] for r in records]),
                "SecMI_mean_auc": metrics["fit"]["U"]["score_mean"]["models"][model]["auc"],
                "SecMI_max_auc": metrics["fit"]["U"]["score_max"]["models"][model]["auc"],
                "scope": "same eight fit patients; descriptive, different methods/costs, no superiority claim",
            }
    return {
        "schema": "secmi-diagnostic-analysis/v1",
        "primary_score": "mean of two negative image L2 scores",
        "secondary_score": "maximum of two negative image L2 scores; descriptive only",
        "score_direction": "higher score indicates membership; fixed before this diagnostic",
        "metrics": metrics, "screening_decisions": gates,
        "fit_U_reference_response_comparison": reference_comparison,
        "bootstrap": {
            "unit": "patient", "assignment_strata": {"A": 20, "B": 20},
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "generator": "numpy.default_rng PCG64", "quantile_method": "linear",
            "patient_order": order["selection"],
            "index_dtype": "int64", "indices_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "same_indices_for_both_models_and_E_U_and_scores": True,
            "interval": "marginal percentile 95%; not simultaneous-coverage or confirmatory intervals",
            "max_score_not_used_for_decision": True,
        },
        "fit_and_selection_kept_separate": True,
        "calibration_and_test_evaluated": False,
        "independent_target_training_seed_count": 1,
        "model_pair_changes_many_patients_jointly": True,
        "paired_contrasts_are_not_single_patient_causal_effects": True,
        "low_fpr_performance_or_DP_guarantee_claimed": False,
        "limitations": [
            "Selection40 is exploratory method-development data, not an untouched final test.",
            "Both models share initialization and jointly swap many patient inclusions.",
            "A failed screening gate is not evidence of no leakage or impossibility of patient MIA.",
            "E and U gates are interpreted separately; E success does not prove U success.",
        ],
    }

