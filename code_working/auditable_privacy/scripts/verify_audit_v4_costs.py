"""Independently verify the Audit v4 cost record and pilot disclosure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTIRUN = Path("reports/wisdm_v4_multirun_001")
DEFAULT_COST = Path("reports/audit_v4_cost_benchmark_001")
RUN_IDS = tuple(f"confirmatory_run{index:02d}" for index in range(1, 6))
TRIAL_IDS = tuple(f"portable_trial{index:02d}" for index in range(1, 6))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def describe(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "observations": list(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "median": statistics.median(values),
    }


def same_numbers(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            same_numbers(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_numbers(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def directory_inventory(path: Path) -> Dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def resolve_project_file(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes project root: {relative}") from exc
    if not path.is_file():
        raise RuntimeError(f"Missing bound file: {relative}")
    return path


class Gate:
    def __init__(self) -> None:
        self.checks: Dict[str, Dict[str, Any]] = {}

    def add(self, name: str, passed: bool, detail: Any) -> None:
        self.checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            raise RuntimeError(f"{name}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multirun-root", type=Path, default=DEFAULT_MULTIRUN)
    parser.add_argument("--cost-root", type=Path, default=DEFAULT_COST)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--portable-package",
        action="store_true",
        help=(
            "Validate the frozen complete-directory inventory declaration "
            "without requiring optional derivative plots and logs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multirun = args.multirun_root
    if not multirun.is_absolute():
        multirun = PROJECT_ROOT / multirun
    cost_root = args.cost_root
    if not cost_root.is_absolute():
        cost_root = PROJECT_ROOT / cost_root
    output = args.output_json or cost_root / "independent_verification.json"
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    execution = load_json(multirun / "execution_manifest.json")
    confirmatory = load_json(multirun / "confirmatory_summary.json")
    cost = load_json(cost_root / "cost_and_integration_summary.json")
    pilot = load_json(multirun / "pilot_disclosure.json")
    gate = Gate()

    gate.add(
        "confirmatory_source",
        execution["status"] == "COMPLETE"
        and execution["confirmatory_run_ids"] == list(RUN_IDS)
        and confirmatory["passed"]
        and confirmatory["confirmatory_run_count"] == 5,
        "complete five-run source",
    )
    gate.add("cost_record_passed", cost["passed"] is True, cost["passed"])

    protocol = resolve_project_file(cost["protocol"]["path"])
    gate.add(
        "cost_protocol_digest",
        sha256_file(protocol) == cost["protocol"]["sha256_before_trials"]
        and cost["protocol"]["trial_count"] == 5,
        cost["protocol"]["sha256_before_trials"],
    )
    multirun_protocol = resolve_project_file(execution["protocol_path"])
    gate.add(
        "multirun_protocol_digest",
        sha256_file(multirun_protocol)
        == execution["protocol_sha256_before_execution"]
        == confirmatory["protocol_sha256"],
        execution["protocol_sha256_before_execution"],
    )

    trials = cost["portable_trials"]
    gate.add(
        "portable_trial_set",
        [item["trial_id"] for item in trials] == list(TRIAL_IDS)
        and all(
            item["return_code"] == 0
            and item["verification_passed"] is True
            and item["check_count"] == 35
            and item["raw_inputs_rescanned"] is False
            and float(item["elapsed_seconds"]) > 0
            for item in trials
        ),
        f"{len(trials)} trials",
    )

    pipeline_times = [
        float(item["pipeline"]["elapsed_seconds"]) for item in execution["runs"]
    ]
    full_times = [
        float(item["verifier"]["elapsed_seconds"]) for item in execution["runs"]
    ]
    portable_times = [float(item["elapsed_seconds"]) for item in trials]
    timing_expected = {
        "complete_pipeline": describe(pipeline_times),
        "independent_full_raw_rescan_verifier": describe(full_times),
        "independent_portable_verifier": describe(portable_times),
    }
    gate.add(
        "timing_recomputation",
        same_numbers(cost["timings_seconds"], timing_expected),
        timing_expected,
    )

    if args.portable_package:
        declared_inventories = cost["complete_run_directories"]
        gate.add(
            "complete_directory_inventory_declaration",
            tuple(declared_inventories) == RUN_IDS
            and all(
                int(declared_inventories[run_id]["file_count"]) == 36
                and int(declared_inventories[run_id]["total_bytes"]) > 900_000
                for run_id in RUN_IDS
            ),
            "five frozen execution-time inventories; optional derivatives omitted",
        )
    else:
        expected_inventories = {
            run_id: directory_inventory(multirun / run_id) for run_id in RUN_IDS
        }
        gate.add(
            "complete_directory_inventories",
            cost["complete_run_directories"] == expected_inventories,
            expected_inventories,
        )

    for inventory_name in (
        "per_run_required_verifier_inputs",
        "shared_bound_route_files",
    ):
        inventory = cost[inventory_name]
        paths = [resolve_project_file(item["path"]) for item in inventory["files"]]
        actual = [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
        if args.portable_package and inventory_name == "shared_bound_route_files":
            declared = {
                item["path"]: item for item in inventory["files"]
            }
            stable_paths = [
                item
                for item in actual
                if item["path"] != "docs/mechanism_registry_v2_0.json"
            ]
            stable_match = all(
                item == declared.get(item["path"]) for item in stable_paths
            )
            registry = load_json(
                PROJECT_ROOT / "docs/mechanism_registry_v2_0.json"
            )
            report = load_json(
                multirun / RUN_IDS[0] / "privacy_report.json"
            )
            mechanism = report["mechanism"]
            registry_key = (
                f"{mechanism['mechanism_id']}@"
                f"{mechanism['mechanism_version']}"
            )
            entry = registry.get("mechanisms", {}).get(registry_key)
            registry_entry_match = (
                isinstance(entry, Mapping)
                and canonical_json_sha256(entry)
                == mechanism["adapter_registry_entry_sha256"]
            )
            gate.add(
                f"{inventory_name}_hashes",
                inventory["file_count"] == 4
                and len(stable_paths) == 3
                and stable_match
                and registry_entry_match,
                {
                    "historical_inventory_files": inventory["file_count"],
                    "stable_files": len(stable_paths),
                    "registry_key": registry_key,
                    "registry_entry_match": registry_entry_match,
                    "registry_file_extended": (
                        actual
                        != inventory["files"]
                    ),
                },
            )
        else:
            gate.add(
                f"{inventory_name}_hashes",
                inventory["file_count"] == len(actual)
                and inventory["total_bytes"]
                == sum(item["bytes"] for item in actual)
                and inventory["files"] == actual,
                {
                    "files": len(actual),
                    "bytes": sum(item["bytes"] for item in actual),
                },
            )

    parser_evidence = load_json(multirun / RUN_IDS[0] / "parser_evidence.json")
    raw = cost["raw_dataset"]
    gate.add(
        "raw_dataset_declaration",
        raw["bytes"] == 50326282
        and raw["sha256"] == parser_evidence["dataset_file_sha256"]
        and raw["required_for"] == "38-check full raw-input route only",
        {"bytes": raw["bytes"], "sha256": raw["sha256"]},
    )

    pilot_dir = multirun / "noise2_run01"
    pilot_metrics = load_json(pilot_dir / "metrics.json")
    pilot_report = load_json(pilot_dir / "privacy_report.json")
    pilot_verification_path = pilot_dir / "independent_verification.json"
    pilot_verification = load_json(pilot_verification_path)
    gate.add(
        "pilot_exclusion_and_protocol",
        pilot["aggregate_inclusion"] is False
        and pilot["directory"]
        == "reports/wisdm_v4_multirun_001/noise2_run01"
        and pilot["protocol_sha256"]
        == execution["protocol_sha256_before_execution"],
        pilot["exclusion_reason"],
    )
    gate.add(
        "pilot_metrics_and_binding",
        same_numbers(pilot["metrics"], {
            "test_accuracy": pilot_metrics["test_accuracy"],
            "test_macro_f1": pilot_metrics["test_macro_f1"],
            "train_accuracy": pilot_metrics["train_accuracy"],
        })
        and pilot["released_model_sha256"]
        == pilot_report["pipeline"]["manifest"]["released_model_sha256"]
        and pilot["selected_mapping_sha256"]
        == pilot_report["pipeline"]["manifest"]["selected_mapping_sha256"],
        pilot["metrics"],
    )
    gate.add(
        "pilot_independent_verification",
        pilot_verification["passed"] is True
        and pilot_verification["check_count"] == 38
        and pilot_verification["raw_inputs_rescanned"] is True
        and pilot["independent_verification"]["sha256"]
        == sha256_file(pilot_verification_path),
        "38/38 raw-input rescan; excluded by chronology",
    )

    record = {
        "schema_version": "audit_v4_cost_and_pilot_verification_1",
        "verification_passed": True,
        "checks_passed": len(gate.checks),
        "checks_total": len(gate.checks),
        "checks": gate.checks,
        "cost_summary_sha256": sha256_file(
            cost_root / "cost_and_integration_summary.json"
        ),
        "confirmatory_summary_sha256": sha256_file(
            multirun / "confirmatory_summary.json"
        ),
        "portable_package_mode": args.portable_package,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"verification_passed=True "
        f"checks={record['checks_passed']}/{record['checks_total']}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
