"""Validate and summarize the frozen five-run WISDM protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = tuple(f"confirmatory_run{index:02d}" for index in range(1, 6))
SCENARIO = "wisdm_v2_hardened_overlap50"
DEFAULT_ROOT = Path("reports/wisdm_v4_multirun_001")
METRIC_NAMES = ("test_accuracy", "test_macro_f1", "train_accuracy")
MECHANISM_FIELDS = (
    "mechanism_id",
    "mechanism_version",
    "accountant_id",
    "accountant_version",
    "accounting_unit",
    "accountant_adjacency",
    "sample_rate_numerator",
    "sample_rate_denominator",
    "steps",
    "noise_multiplier",
    "clipping_norm",
    "optimizer_step_size",
    "secure_mode",
    "secure_rng_backend",
    "noise_generation",
    "noise_hardening",
    "gradient_aggregation",
    "update_normalization",
    "adapter_registry_entry_sha256",
)
MANIFEST_FIELDS = (
    "dataset_file_sha256",
    "selected_mapping_sha256",
    "selected_record_count",
    "code_artifact_id",
    "code_sha256",
    "accountant_checker_artifact_id",
    "accountant_checker_sha256",
    "split_id",
)
EXPECTED_DECISIONS = {
    "window": ("ALLOWED", "DIRECT", 1, True),
    "event": ("ALLOWED", "CONVERT", 4, True),
    "owner": ("BLOCKED_VACUOUS", "GROUP", 600, False),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_subset(source: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: source[field] for field in fields}


def describe(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Write summaries outside the evidence root (default: evidence root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    output_root = args.output_root or root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "execution_manifest.json"
    execution = load_json(manifest_path)
    require(execution["status"] == "COMPLETE", "Execution manifest is not complete.")
    require(
        execution["confirmatory_run_ids"] == list(RUN_IDS),
        "Execution manifest has an unexpected run set or order.",
    )
    require(
        execution["pilot_excluded"]
        == "reports/wisdm_v4_multirun_001/noise2_run01",
        "Pilot exclusion is missing or changed.",
    )
    protocol_path = PROJECT_ROOT / execution["protocol_path"]
    require(
        sha256_file(protocol_path)
        == execution["protocol_sha256_before_execution"],
        "Frozen protocol changed after execution began.",
    )

    rows: List[Dict[str, Any]] = []
    static_bindings: List[Dict[str, Any]] = []
    privacy_bindings: List[Dict[str, float]] = []
    decision_records: List[Dict[str, Any]] = []
    released_model_digests: List[str] = []

    timings_by_run = {item["run_id"]: item for item in execution["runs"]}
    require(
        tuple(timings_by_run) == RUN_IDS,
        "Execution timing records have an unexpected run set or order.",
    )

    for run_id in RUN_IDS:
        run_dir = root / run_id
        metrics = load_json(run_dir / "metrics.json")
        verification = load_json(run_dir / "independent_verification.json")
        report = load_json(run_dir / "privacy_report.json")
        assertions = load_json(run_dir / "gold_assertions.json")

        require(verification["passed"], f"{run_id}: verifier did not pass.")
        require(
            verification["check_count"] == 38,
            f"{run_id}: expected 38 verifier checks.",
        )
        require(
            verification["raw_inputs_rescanned"],
            f"{run_id}: verifier did not rescan raw input.",
        )
        require(
            all(item["passed"] for item in assertions.values()),
            f"{run_id}: a gold assertion failed.",
        )
        require(report["scenario"] == SCENARIO, f"{run_id}: wrong scenario.")

        mechanism = stable_subset(report["mechanism"], MECHANISM_FIELDS)
        pipeline_manifest = report["pipeline"]["manifest"]
        binding = {
            "mechanism": mechanism,
            "manifest": stable_subset(pipeline_manifest, MANIFEST_FIELDS),
            "train_records": metrics["train_records"],
            "test_records": metrics["test_records"],
            "train_owners": metrics["train_owners"],
            "test_owners": metrics["test_owners"],
        }
        static_bindings.append(binding)
        privacy_bindings.append(
            {
                "epsilon": report["privacy"]["epsilon"],
                "delta": report["privacy"]["delta"],
            }
        )
        released_model_digests.append(pipeline_manifest["released_model_sha256"])

        decisions: Dict[str, Any] = {}
        for claim, expected in EXPECTED_DECISIONS.items():
            certificate_path = (
                run_dir
                / "certificates"
                / f"certificate_{SCENARIO}_{claim}.json"
            )
            internal_certificate_path = (
                run_dir
                / "certificates"
                / f"certificate_{SCENARIO}_{claim}_internal.json"
            )
            certificate = load_json(certificate_path)
            internal_certificate = load_json(internal_certificate_path)
            actual = (
                certificate["release_status"],
                certificate["unit_path"],
                int(internal_certificate["stability_bound"]),
                bool(certificate["supported_statement"]),
            )
            require(
                actual == expected,
                f"{run_id}: {claim} decision {actual!r} != {expected!r}.",
            )
            decisions[claim] = {
                "release_status": actual[0],
                "unit_path": actual[1],
                "stability_bound": actual[2],
                "supported_statement": actual[3],
                "epsilon": certificate.get("epsilon"),
                "delta": certificate.get("delta"),
            }
        decision_records.append(decisions)

        timing = timings_by_run[run_id]
        require(
            timing["pipeline"]["return_code"] == 0
            and timing["verifier"]["return_code"] == 0,
            f"{run_id}: a recorded command failed.",
        )
        rows.append(
            {
                "run_id": run_id,
                **{name: float(metrics[name]) for name in METRIC_NAMES},
                "pipeline_seconds": float(timing["pipeline"]["elapsed_seconds"]),
                "verifier_seconds": float(timing["verifier"]["elapsed_seconds"]),
                "released_model_sha256": pipeline_manifest[
                    "released_model_sha256"
                ],
            }
        )

    require(
        all(item == static_bindings[0] for item in static_bindings[1:]),
        "Static input/configuration bindings differ across runs.",
    )
    require(
        all(item == privacy_bindings[0] for item in privacy_bindings[1:]),
        "Deterministic base privacy values differ across runs.",
    )
    require(
        all(item == decision_records[0] for item in decision_records[1:]),
        "Decision certificates differ across runs.",
    )
    require(
        len(set(released_model_digests)) == len(RUN_IDS),
        "Released-model digests are not all distinct.",
    )

    utility = {
        name: describe([float(row[name]) for row in rows])
        for name in METRIC_NAMES
    }
    timings = {
        "pipeline_seconds": describe(
            [float(row["pipeline_seconds"]) for row in rows]
        ),
        "verifier_seconds": describe(
            [float(row["verifier_seconds"]) for row in rows]
        ),
    }
    # The execution-time inventory is part of the frozen manifest.  Using it
    # keeps the scientific aggregate invariant when a portable distribution
    # omits non-decision derivative plots or logs.
    total_files = sum(
        int(timings_by_run[run_id]["inventory"]["file_count"])
        for run_id in RUN_IDS
    )
    total_bytes = sum(
        int(timings_by_run[run_id]["inventory"]["total_bytes"])
        for run_id in RUN_IDS
    )

    summary = {
        "protocol_version": execution["protocol_version"],
        "protocol_sha256": execution["protocol_sha256_before_execution"],
        "passed": True,
        "confirmatory_run_count": len(RUN_IDS),
        "pilot_excluded": execution["pilot_excluded"],
        "verifier_result_per_run": "38/38 with raw input rescan",
        "decision_invariance": decision_records[0],
        "base_privacy": privacy_bindings[0],
        "static_binding": static_bindings[0],
        "distinct_released_model_digests": len(set(released_model_digests)),
        "utility_descriptive_statistics": utility,
        "runtime_descriptive_statistics": timings,
        "confirmatory_artifact_inventory": {
            "file_count": total_files,
            "total_bytes": total_bytes,
        },
        "runs": rows,
        "interpretation_boundary": (
            "Five fresh-randomness executions support stochastic robustness "
            "of the registered WISDM path only; they do not establish "
            "cross-dataset or cross-mechanism generality."
        ),
    }
    (output_root / "confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output_root / "confirmatory_utility.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# WISDM five-run confirmatory summary",
        "",
        "- Protocol result: **PASS**",
        "- Confirmatory runs: **5/5** (the earlier pilot is excluded)",
        "- Independent verification: **38/38 with raw-input rescan in every run**",
        "- Released-model digests: **5 distinct**",
        "- Decision paths: **invariant** (`window: ALLOWED/DIRECT/K=1`; "
        "`event: ALLOWED/CONVERT/K=4`; "
        "`owner: BLOCKED_VACUOUS/GROUP/K=600`)",
        "",
        "## Per-run outcomes",
        "",
        "| Run | Test accuracy | Macro-F1 | Train accuracy | Pipeline s | Verifier s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['test_accuracy']:.6f} | "
            f"{row['test_macro_f1']:.6f} | {row['train_accuracy']:.6f} | "
            f"{row['pipeline_seconds']:.3f} | "
            f"{row['verifier_seconds']:.3f} |"
        )
    lines.extend(["", "## Descriptive aggregate", ""])
    for name in METRIC_NAMES:
        item = utility[name]
        lines.append(
            f"- `{name}`: mean {item['mean']:.6f}, sample SD "
            f"{item['sample_standard_deviation']:.6f}, range "
            f"[{item['minimum']:.6f}, {item['maximum']:.6f}]"
        )
    lines.extend(
        [
            "",
            "These results test fresh-randomness robustness for the one "
            "registered WISDM path. They do not establish cross-dataset or "
            "cross-mechanism generality, and no run was selected by utility.",
            "",
        ]
    )
    (output_root / "confirmatory_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("confirmatory_protocol_passed=True")
    print(f"runs={len(RUN_IDS)}")
    print(f"Wrote {output_root / 'confirmatory_summary.json'}")
    print(f"Wrote {output_root / 'confirmatory_utility.csv'}")
    print(f"Wrote {output_root / 'confirmatory_summary.md'}")


if __name__ == "__main__":
    main()
