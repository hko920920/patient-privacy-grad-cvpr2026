"""Independent CPU verification of fixed E/U loss controls on the original U8.

No target model execution, fitting, sign selection, calibration or new patients.
Numerical agreement with the old batch-one U probe is reported separately from
integrity: a PASS here never silently declares batch-four scores equivalent.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

import torch
from sklearn.metrics import roc_auc_score

from .common import ROOT, RUN, SALT, digest, read_csv, verify_inputs, write_json

MODELS = ("model_1", "model_2")
SCENARIOS = ("E", "U")
PROMPTS = ("generic", "matched_weak_label")
TIMESTEPS = (50, 250, 500, 750, 950)
SCORES = ("loss_mean", "loss_max", "base_difference_mean")
OLD = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b)), (a, b)
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def sign(value):
    return int(value > 0) - int(value < 0)


def identity(row, image=False):
    key = (row["model"], row["scenario"], row["prompt_kind"], str(row["patient_id"]))
    return key + (row["image_id"],) if image else key


def discrepancy(pairs):
    differences = [new - old for old, new in pairs]
    return {
        "n": len(pairs),
        "exact_equal": sum(old == new for old, new in pairs),
        "mean_signed_difference": st.mean(differences),
        "mean_absolute_difference": st.mean(abs(d) for d in differences),
        "maximum_absolute_difference": max(abs(d) for d in differences),
        "rms_difference": math.sqrt(st.mean(d * d for d in differences)),
        "maximum_relative_absolute_difference": max(
            abs(new - old) / max(abs(old), 1e-15) for old, new in pairs
        ),
        "relative_denominator_floor": 1e-15,
    }


def verify(path):
    torch.set_num_threads(2)
    verify_inputs()
    report = load(path / "report.json")
    cells = load(path / "cell_losses.json")
    emitted_images = load(path / "image_scores.json")
    emitted_patients = load(path / "patient_scores.json")
    protocol = load(path / "protocol.json")
    assert report["protocol_sha256"] == digest(path / "protocol.json")
    expected_protocol = {
        "schema": "cvpr-u-basic-controls/v1",
        "query_timesteps": list(TIMESTEPS), "draws": 2, "batch_size": 4,
        "phase": "query",
        "noise_seed_parts": ["audit-noise", "query", "image_id", "timestep", "draw"],
        "generic_prompt": "a frontal chest radiograph",
        "matched_prompt_policy": "cache.train_prompts: original per-image weak-label training prompt",
        "prompt_kinds": list(PROMPTS), "model_mode": "train",
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_recomputation": False,
        "gradient_mode": "inference_mode; no gradient computation",
        "precision": "FP16 model/latents/conditioning, autocast FP16, loss reduction FP32",
        "conditioning_optimization": False, "new_patients": 0,
        "unique_patients": 8, "images_per_model": 32,
        "low_fpr_performance_or_novelty_claim": False,
    }
    for key, expected in expected_protocol.items():
        assert protocol[key] == expected, (key, protocol[key], expected)
    for key in ("code_sha256", "input_sha256", "checkpoint_sha256", "cache_sha256", "cohort_lock_sha256"):
        assert protocol[key] == report[key], key
    assert set(report["output_sha256"]) == {"cell_losses.json", "image_scores.json", "patient_scores.json"}
    for filename, expected in report["output_sha256"].items():
        assert digest(path / filename) == expected, filename
    assert report["status"] == "PASS_EXECUTION_EXPLORATORY_CONTROLS"
    assert report["schema"] == "cvpr-u-basic-controls-report/v1"
    assert report["image_records"] == 128 and report["patient_score_records"] == 64
    assert report["noise_cell_records"] == 1280
    assert report["forward_batch_calls"] == 640 and report["optimizer_steps"] == 0
    assert report["role"] == "fit" and report["calibration_or_test_evaluated"] is False
    assert report["all_weights_unchanged"] is True
    for key in ("model_mode", "gradient_checkpointing_enabled", "gradient_checkpointing_recomputation"):
        assert report[key] == expected_protocol[key], key
    assert set(report["per_model"]) == set(MODELS)
    for model, checks in report["per_model"].items():
        assert checks["forward_model_examples"] == 1280
        assert checks["forward_batch_calls"] == 320 and checks["backward"] == 0
        assert checks["checkpoint_sha256"] == report["checkpoint_sha256"][model]
        for key in ("weights_unchanged", "parameter_gradients_absent", "all_parameters_frozen", "E_membership_matches_actual_exposure"):
            assert checks[key] is True, (model, key)
        assert checks["parameter_versions_checked"] == 942
        assert checks["dropout_nonzero_count"] == checks["U_image_exposures"] == 0
        assert math.isfinite(checks["seconds_including_model_load"]) and checks["seconds_including_model_load"] > 0
        assert math.isfinite(checks["max_cuda_GiB"]) and checks["max_cuda_GiB"] > 0
    assert report["forward_model_examples"] == sum(r["forward_model_examples"] for r in report["per_model"].values())
    assert report["forward_batch_calls"] == sum(r["forward_batch_calls"] for r in report["per_model"].values())
    assert report["cohort_lock_sha256"] == digest(RUN / "cohort/lock.json")
    assert report["cache_sha256"] == digest(RUN / "cache/cache.pt")
    cache_summary = load(RUN / "cache/summary.json")
    assert report["cache_sha256"] == cache_summary["cache_sha256"]
    assert report["cohort_lock_sha256"] == cache_summary["cohort_lock_sha256"]
    assert report["unique_patients"] == 8 and report["new_patients"] == 0
    assert report["forward_model_examples"] == 2560 and report["backward"] == 0
    verified_inputs = {}
    for filename, expected in report["input_sha256"].items():
        source = (ROOT / filename).resolve()
        assert source.is_relative_to(ROOT.resolve()), filename
        assert digest(source) == expected, filename
        verified_inputs[filename] = expected
    assert verified_inputs, "missing run input bindings"
    for filename, expected in report["code_sha256"].items():
        source = (ROOT / filename).resolve()
        assert source.is_relative_to(ROOT.resolve()), filename
        assert digest(source) == expected, filename

    old_report = load(OLD / "report.json")
    old_check = load(OLD / "verification.json")
    old_rows = load(OLD / "results.json")
    assert old_check["status"] == "PASS"
    assert old_check["results_sha256"] == digest(OLD / "results.json")
    assert old_check["report_sha256"] == digest(OLD / "report.json")
    assert old_report["unique_fit_patients"] == 8
    assert old_report["scenario"] == "U"
    old_patient = {(r["model"], str(r["patient_id"])): r for r in old_rows}
    assert len(old_patient) == len(old_rows) == 16
    selected = {p for _, p in old_patient}
    assert len(selected) == 8

    manifest = read_csv(RUN / "cohort/evaluation_images.csv")
    images = {r["image_id"]: r for r in manifest}
    assert len(images) == len(manifest)
    by_patient = defaultdict(list)
    for row in manifest:
        by_patient[row["patient_id"]].append(row)
    labels = {}
    exposure_counts = {}
    for model in MODELS:
        checkpoint = RUN / "training_coverage_v2" / model / "step_1000.pt"
        assert digest(checkpoint) == report["checkpoint_sha256"][model]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["step"] == 1000
        assert state["contract"] == load(checkpoint.parent.parent / "protocol.json")
        exposures = state["exposures"]
        training = {r["image_id"] for r in read_csv(RUN / "cohort" / (model + "_train.csv"))}
        assert set(exposures) == training and min(exposures.values()) > 0
        assert sum(exposures.values()) == 4000
        exposure_counts[model] = dict(exposures)
        for patient in selected:
            rows = by_patient[patient]
            assert len(rows) == 4 and all(r["eval_role"] == "fit" for r in rows)
            declared = int(any(r["image_id"] in training for r in rows))
            realized = int(any(exposures.get(r["image_id"], 0) > 0 for r in rows))
            expected_group = "A" if model == "model_1" else "B"
            assert declared == realized == int(rows[0]["assignment_group"] == expected_group)
            assert old_patient[model, patient]["realized_member"] == realized
            labels[model, patient] = realized
            for row in rows:
                if row["record_role"] == "U_observed":
                    assert exposures.get(row["image_id"], 0) == 0
        del state
    for patient in selected:
        assert sorted(labels[m, patient] for m in MODELS) == [0, 1]

    grouped = defaultdict(list)
    cell_keys = set()
    for row in cells:
        model, scenario, prompt, patient, image_id = identity(row, image=True)
        assert model in MODELS and scenario in SCENARIOS and prompt in PROMPTS
        assert patient in selected and row["eval_role"] == "fit"
        item = images[image_id]
        assert item["patient_id"] == patient and item["eval_role"] == "fit"
        assert item["record_role"] == ("U_observed" if scenario == "U" else "train_candidate")
        assert row["declared_member"] == row["realized_member"] == labels[model, patient]
        assert (exposure_counts[model].get(image_id, 0) > 0) == bool(scenario == "E" and labels[model, patient])
        timestep, draw = int(row["timestep"]), int(row["draw"])
        assert timestep in TIMESTEPS and draw in (0, 1)
        noise_text = "|".join(map(str, (SALT, "audit-noise", "query", image_id, timestep, draw)))
        expected_seed = int(hashlib.sha256(noise_text.encode()).hexdigest()[:15], 16)
        assert int(row["noise_seed"]) == expected_seed
        key = identity(row, image=True) + (timestep, draw)
        assert key not in cell_keys, key
        cell_keys.add(key)
        assert all(math.isfinite(row[name]) and row[name] >= 0 for name in ("target_loss", "base_loss"))
        grouped[identity(row, image=True)].append(row)
    assert len(cells) == 1280 and len(grouped) == 128
    assert report["forward_model_examples"] == 2 * len(cells)
    recomputed_images = {}
    for key, rows in grouped.items():
        assert len(rows) == 10
        assert {(int(r["timestep"]), int(r["draw"])) for r in rows} == set(itertools.product(TIMESTEPS, (0, 1)))
        target = st.mean(r["target_loss"] for r in rows)
        base = st.mean(r["base_loss"] for r in rows)
        recomputed_images[key] = dict(target_loss=target, base_loss=base, base_minus_target=base - target)
    assert len(emitted_images) == 128
    assert len({identity(r, image=True) for r in emitted_images}) == 128
    for row in emitted_images:
        key = identity(row, image=True)
        assert row["declared_member"] == row["realized_member"] == labels[key[0], key[3]]
        for name, value in recomputed_images[key].items():
            close(row[name], value)

    patients = {}
    for model, scenario, prompt, patient in itertools.product(MODELS, SCENARIOS, PROMPTS, sorted(selected)):
        ids = sorted(r["image_id"] for r in by_patient[patient]
                     if r["record_role"] == ("U_observed" if scenario == "U" else "train_candidate"))
        assert len(ids) == 2
        values = [recomputed_images[model, scenario, prompt, patient, image_id] for image_id in ids]
        patients[model, scenario, prompt, patient] = {
            "image_ids": ids, "member": labels[model, patient],
            "loss_mean": -st.mean(v["target_loss"] for v in values),
            "loss_max": max(-v["target_loss"] for v in values),
            "base_difference_mean": st.mean(v["base_minus_target"] for v in values),
        }
    assert len(emitted_patients) == 64
    assert len({identity(r) for r in emitted_patients}) == 64
    for row in emitted_patients:
        expected = patients[identity(row)]
        assert sorted(row["image_ids"]) == expected["image_ids"]
        assert row["declared_member"] == row["realized_member"] == expected["member"]
        for name in SCORES:
            close(row[name], expected[name])

    metrics = {}
    for scenario, prompt, score in itertools.product(SCENARIOS, PROMPTS, SCORES):
        model_stats = {}
        for model in MODELS:
            records = [patients[model, scenario, prompt, p] for p in sorted(selected)]
            classes = [r["member"] for r in records]
            assert Counter(classes) == {0: 4, 1: 4}
            model_stats[model] = {
                "auc": float(roc_auc_score(classes, [r[score] for r in records])),
                "members": 4, "nonmembers": 4,
                "member_mean": st.mean(r[score] for r in records if r["member"]),
                "nonmember_mean": st.mean(r[score] for r in records if not r["member"]),
            }
        paired = []
        for patient in sorted(selected):
            member_model = next(m for m in MODELS if labels[m, patient])
            nonmember_model = next(m for m in MODELS if not labels[m, patient])
            delta = patients[member_model, scenario, prompt, patient][score] - patients[nonmember_model, scenario, prompt, patient][score]
            paired.append(dict(patient_id=patient, member_model=member_model, member_minus_nonmember=delta))
        deltas = [r["member_minus_nonmember"] for r in paired]
        metrics[f"{scenario}/{prompt}/{score}"] = {
            "models": model_stats, "paired_analyst_only": paired,
            "paired_positive": sum(d > 0 for d in deltas),
            "paired_negative": sum(d < 0 for d in deltas),
            "paired_mean": st.mean(deltas), "paired_median": st.median(deltas),
            "paired_is_not_a_single_patient_causal_effect": True,
        }

    # Old folds hold means over the same 5 timesteps x 2 query-noise draws.
    # Compare these means, not nonexistent old per-cell observations.
    image_comparisons = []
    for (model, patient), old in old_patient.items():
        for fold in old["folds"]:
            image_id = fold["query"]
            now = recomputed_images[model, "U", "generic", patient, image_id]
            image_comparisons.append({
                "model": model, "patient_id": patient, "image_id": image_id,
                **{name: {"old": fold[name], "new": now[name], "difference": now[name] - fold[name]}
                   for name in ("target_loss", "base_loss", "base_minus_target")},
            })
    numerical = {name: discrepancy([(r[name]["old"], r[name]["new"]) for r in image_comparisons])
                 for name in ("target_loss", "base_loss", "base_minus_target")}
    ranking_checks = {}
    for score in SCORES:
        checks = {}
        for model in MODELS:
            positive = sorted(p for p in selected if labels[model, p])
            negative = sorted(p for p in selected if not labels[model, p])
            changed = []
            for pos, neg in itertools.product(positive, negative):
                old_difference = old_patient[model, pos][score] - old_patient[model, neg][score]
                new_difference = patients[model, "U", "generic", pos][score] - patients[model, "U", "generic", neg][score]
                if sign(old_difference) != sign(new_difference):
                    changed.append(dict(member_patient=pos, nonmember_patient=neg,
                                        old_order=sign(old_difference), new_order=sign(new_difference)))
            old_auc = float(roc_auc_score([labels[model, p] for p in sorted(selected)],
                                          [old_patient[model, p][score] for p in sorted(selected)]))
            checks[model] = dict(member_nonmember_pairs=16, changed_pairs=changed,
                                 changed_pair_count=len(changed), old_auc=old_auc,
                                 new_auc=metrics[f"U/generic/{score}"]["models"][model]["auc"])
        paired_sign_changes = []
        for patient in sorted(selected):
            member = next(m for m in MODELS if labels[m, patient])
            nonmember = next(m for m in MODELS if not labels[m, patient])
            old_delta = old_patient[member, patient][score] - old_patient[nonmember, patient][score]
            new_delta = patients[member, "U", "generic", patient][score] - patients[nonmember, "U", "generic", patient][score]
            if sign(old_delta) != sign(new_delta):
                paired_sign_changes.append(dict(patient_id=patient, old_delta=old_delta, new_delta=new_delta))
        ranking_checks[score] = dict(models=checks, paired_delta_sign_changes=paired_sign_changes)
    any_rank_change = any(v["changed_pair_count"] for r in ranking_checks.values() for v in r["models"].values())
    any_delta_sign_change = any(r["paired_delta_sign_changes"] for r in ranking_checks.values())
    numerical_difference = any(r["maximum_absolute_difference"] > 0 for r in numerical.values())
    analysis = {
        "scope": "original eight fit patients; fixed-score diagnostic only",
        "unique_patients": 8, "model_patient_pairs": 16,
        "conditioned_patient_scores": 64,
        "score_direction": "higher score => member for every model; not fitted",
        "score_signs_fitted": False, "classifiers_fitted": False, "thresholds_fitted": False,
        "low_fpr_performance_estimated": False, "calibration_or_test_used": False,
        "paired_comparisons_are_analyst_only": True, "independent_training_seed_count": 1,
        "matched_weak_label_is_optimistic_conditioning_control": True,
        "matched_prompt_is_not_the_candidate_generic_auditor_setting": True,
        "metrics": metrics,
        "old_U_generic_numerical_comparison": {
            "comparison_level": "per-image average over five timesteps and two paired query-noise draws",
            "old_per_cell_values_available": False,
            "descriptive_discrepancies": numerical, "per_image": image_comparisons,
            "rank_and_paired_sign_checks": ranking_checks,
            "membership_pair_rank_changed": any_rank_change,
            "paired_membership_delta_sign_changed": any_delta_sign_change,
            "numerical_difference_present": numerical_difference,
            "batch_equivalence_declared": not numerical_difference,
            "decision": ("BATCH1_FALLBACK_REVIEW_REQUIRED" if any_rank_change or any_delta_sign_change
                         else "NUMERICAL_DIFFERENCES_REVIEW_REQUIRED" if numerical_difference else "EXACT_REPRODUCTION"),
            "interpretation": "Integrity PASS does not establish batch equivalence; root review determines need for batch-one fallback.",
        },
        "limitations": [
            "Four members and four nonmembers per target; no power or low-FPR claim.",
            "Shared seed and joint participation swaps; not independent target replications.",
            "No result-selected score direction, coefficient, model choice or threshold.",
            "Matched weak-label conditioning is a separate favorable diagnostic, not proof of patient-only deployability.",
        ],
    }
    verification = {
        "status": "PASS_INTEGRITY_ONLY", "unique_patients": 8,
        "protocol_declarations_and_output_hashes_verified": True,
        "producer_freeze_and_counter_reports_cross_checked": True,
        "freeze_report_is_not_independent_GPU_weight_reinspection": True,
        "cell_losses_checked": len(cells), "image_means_checked": len(recomputed_images),
        "patient_metrics_checked": len(patients), "per_model_auc_recomputed": len(metrics) * 2,
        "labels_roles_noise_and_realized_exposure_verified": True,
        "forward_model_examples": report["forward_model_examples"],
        "forward_batch_calls": report["forward_batch_calls"], "backward": report["backward"],
        "new_patients": 0, "gpu_execution_by_verifier": False,
        "calibration_or_test_used": False, "score_fitting_performed": False,
        "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in
                         [path / "protocol.json", path / "report.json", path / "cell_losses.json",
                          path / "image_scores.json", path / "patient_scores.json", OLD / "results.json",
                          OLD / "verification.json", RUN / "cohort/evaluation_images.csv", RUN / "cohort/lock.json"]},
        "run_source_input_bindings_checked": verified_inputs,
        "verifier_sha256": digest(Path(__file__)),
        "numerical_equivalence_status": analysis["old_U_generic_numerical_comparison"]["decision"],
    }
    write_json(path / "analysis.json", analysis)
    verification["analysis_sha256"] = digest(path / "analysis.json")
    write_json(path / "verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=RUN / "verification_20260914/basic_controls")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
