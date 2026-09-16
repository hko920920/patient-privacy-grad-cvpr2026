"""Assemble verified evidence; do not upgrade scientific efficacy status."""
import json
from datetime import datetime, timezone
from pathlib import Path
from .common import ROOT, RUN, digest, write_json

V = RUN / "verification_20260914"
def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    out = V / "final_summary.json"
    if out.exists():
        raise FileExistsError("final summary already exists")
    foundation = load(V / "foundation/report.json")
    assert foundation["status"] == "PASS_FROZEN_FOUNDATION_RECHECK"
    for name, expected in foundation["protected_sha256"].items():
        assert digest(ROOT / name) == expected, name
    stages = [
        ("numerical_v2", "verify_numerical_diagnostic.py"),
        ("basic_controls", "verify_basic_controls.py"),
        ("traced_U8_v1", "verify_traced_replay.py"),
        ("endpoint_precision_v1", "verify_endpoint_precision.py"),
        ("endpoint_precision_U8_v1", "verify_endpoint_precision_u8.py"),
    ]
    outputs, states = {}, {}
    for stage, verifier in stages:
        folder = V / stage
        check = load(folder / "verification.json")
        assert check["status"].startswith("PASS"), (stage, check["status"])
        assert check["verifier_sha256"] == digest(ROOT / "u_patient_audit" / verifier), verifier
        for name, expected in check.get("input_sha256", {}).items():
            path = folder / name if Path(name).name == name else ROOT / name
            assert digest(path) == expected, (stage, name)
        for name, expected in check.get("detail_sha256", {}).items():
            assert digest(folder / name) == expected, (stage, name)
        if "independent_numeric_helper_sha256" in check:
            assert check["independent_numeric_helper_sha256"] == digest(ROOT / "u_patient_audit/verify_numerical_diagnostic.py")
        if "shared_CPU_arithmetic_checker_sha256" in check:
            assert check["shared_CPU_arithmetic_checker_sha256"] == digest(ROOT / "u_patient_audit/verify_endpoint_precision.py")
        for name in ("analysis", "interpretation"):
            if name+"_sha256" in check:
                assert digest(folder / (name+".json")) == check[name+"_sha256"]
        states[stage] = check["status"]
        for name in ("protocol.json", "report.json", "verification.json", "analysis.json", "interpretation.json"):
            path = folder / name
            if path.exists():
                outputs[path.relative_to(RUN).as_posix()] = digest(path)
    replay = load(V / "traced_U8_v1/report.json")
    numeric = load(V / "numerical_v2/interpretation.json")
    basic = load(V / "basic_controls/analysis.json")
    endpoint = load(V / "endpoint_precision_v1/report.json")
    corrected = load(V / "endpoint_precision_U8_v1/report.json")
    correction_analysis = load(V / "endpoint_precision_U8_v1/analysis.json")
    corrected_metrics = {
        name: {
            "fp16_auc": {m: x["fp16_auc"] for m, x in row["per_model"].items()},
            "fp32_auc": {m: x["fp32_auc"] for m, x in row["per_model"].items()},
            "fp16_paired_positive": row["fp16_paired_positive"],
            "fp32_paired_positive": row["fp32_paired_positive"],
            "paired_sign_changes": row["paired_sign_changes"],
            "rank_changes": {m: x["changed_patient_pair_count"] for m, x in row["per_model"].items()},
        } for name, row in correction_analysis["metrics"].items()
    }
    summary = {
        "status": "PASS_EXISTING_PILOT_VERIFICATION_WITH_PRECISION_CORRECTION",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage_verification_statuses": states,
        "protected_original_files_unchanged": len(foundation["protected_sha256"]),
        "new_patients": 0, "new_training_steps": 0,
        "score_fitting_or_calibration_or_test_evaluation": False,
        "original_six_scalars_exactly_reproduced": replay["all_legacy_scalars_exact"],
        "support_final_gain_positive_count": replay["support_final_gain_positive_count"],
        "support_folds": replay["support_folds"],
        "support_folds_with_intermediate_decrease": replay["support_folds_with_decreasing_steps"],
        "fp32_gradient_direction_fd_relative_error_h001": next(
            r["symmetric_relative_error"] for r in numeric["phases"]["fp32"]["gradient_direction_finite_differences"] if r["h"]==.001),
        "basic_controls_batch4_decision": basic["old_U_generic_numerical_comparison"]["decision"],
        "first_patient_fixed_coefficient_precision_sign_flip": endpoint["paired_diagnostic_analyst_only"]["paired_delta_sign_changed"],
        "precision_correction_scope": "FP32 query/reference endpoint evaluation at saved FP16 optimized coefficients; equal endpoint precision for response and raw-loss U baselines",
        "correction_report_status": corrected["status"],
        "four_score_precision_comparison": corrected_metrics,
        "FP32_U8_endpoint_wall_seconds": corrected["elapsed_seconds"],
        "originals_preserved": True,
        "attack_efficacy_or_novelty_or_DP_guarantee_verified": False,
        "artifact_sha256": outputs,
        "finalizer_sha256": digest(Path(__file__)),
        "limitations": [
            "Eight fit patients and one correlated target pair, no independent evaluation.",
            "FP32 endpoint version is not a full FP32 reoptimization experiment.",
            "Gradient finite-difference validation is local to one preselected image/timestep.",
            "Versioned correction is motivated by precision sensitivity, not chosen for favorable AUC.",
        ],
    }
    write_json(out, summary)
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
