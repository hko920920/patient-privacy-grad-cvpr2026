"""Execute the frozen Audit v4 portable-cost benchmark and inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PROJECT_ROOT
    / "paper_notes"
    / "AAAI27_AUDIT_V4_COST_PROTOCOL_2026-07-24.md"
)
DEFAULT_MULTIRUN_ROOT = Path("reports/wisdm_v4_multirun_001")
DEFAULT_OUTPUT_ROOT = Path("reports/audit_v4_cost_benchmark_001")
TRIAL_IDS = tuple(f"portable_trial{index:02d}" for index in range(1, 6))

PER_RUN_REQUIRED = (
    Path("privacy_report.json"),
    Path("mapping.csv"),
    Path("parser_evidence.json"),
    Path("released_model.json"),
    Path("batch_trace.csv"),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_window.json"
    ),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_window_internal.json"
    ),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_event.json"
    ),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_event_internal.json"
    ),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_owner.json"
    ),
    Path(
        "certificates/"
        "certificate_wisdm_v2_hardened_overlap50_owner_internal.json"
    ),
)
SHARED_BOUND = (
    Path("docs/privacy_report_schema_v2_0.json"),
    Path("docs/mechanism_registry_v2_0.json"),
    Path("scripts/wisdm_v2_gold_pipeline.py"),
    Path("scripts/rdp_accountant_lite.py"),
)
RAW_DATA = Path("data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
VERIFIER = Path("scripts/verify_wisdm_v2_gold.py")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "observations": list(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "median": statistics.median(values),
    }


def inventory(paths: Sequence[Path]) -> Dict[str, Any]:
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing inventory input: {path}")
    return {
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "files": [
            {
                "path": path.resolve()
                .relative_to(PROJECT_ROOT.resolve())
                .as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def directory_inventory(path: Path) -> Dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multirun-root",
        type=Path,
        default=DEFAULT_MULTIRUN_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multirun_root = args.multirun_root
    if not multirun_root.is_absolute():
        multirun_root = PROJECT_ROOT / multirun_root
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite benchmark output: {output_root}")
    if not PROTOCOL.is_file():
        raise SystemExit(f"Missing frozen protocol: {PROTOCOL}")

    execution = load_json(multirun_root / "execution_manifest.json")
    confirmatory = load_json(multirun_root / "confirmatory_summary.json")
    if execution["status"] != "COMPLETE" or not confirmatory["passed"]:
        raise SystemExit("The frozen confirmatory execution is not complete.")

    run_ids = tuple(execution["confirmatory_run_ids"])
    if run_ids != tuple(
        f"confirmatory_run{index:02d}" for index in range(1, 6)
    ):
        raise SystemExit("Unexpected confirmatory run set.")
    reference_dir = multirun_root / run_ids[0]

    output_root.mkdir(parents=True)
    portable_times: List[float] = []
    portable_trials: List[Dict[str, Any]] = []
    for trial_id in TRIAL_IDS:
        result_path = output_root / f"{trial_id}.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / VERIFIER),
            "--evidence-dir",
            str(reference_dir),
            "--skip-input-rescan",
            "--output-json",
            str(result_path),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - started
        (output_root / f"{trial_id}_stdout.log").write_text(
            completed.stdout.replace(str(PROJECT_ROOT), "."),
            encoding="utf-8",
        )
        (output_root / f"{trial_id}_stderr.log").write_text(
            completed.stderr.replace(str(PROJECT_ROOT), "."),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise SystemExit(f"{trial_id} returned {completed.returncode}.")
        result = load_json(result_path)
        if (
            not result["passed"]
            or result["check_count"] != 35
            or result["raw_inputs_rescanned"]
        ):
            raise SystemExit(f"{trial_id} did not produce portable 35/35.")
        portable_times.append(elapsed)
        portable_trials.append(
            {
                "trial_id": trial_id,
                "elapsed_seconds": elapsed,
                "return_code": completed.returncode,
                "verification_passed": result["passed"],
                "check_count": result["check_count"],
                "raw_inputs_rescanned": result["raw_inputs_rescanned"],
            }
        )

    pipeline_times = [
        float(item["pipeline"]["elapsed_seconds"])
        for item in execution["runs"]
    ]
    full_verifier_times = [
        float(item["verifier"]["elapsed_seconds"])
        for item in execution["runs"]
    ]
    run_inventories = {
        run_id: directory_inventory(multirun_root / run_id)
        for run_id in run_ids
    }
    required_paths = [reference_dir / item for item in PER_RUN_REQUIRED]
    shared_paths = [PROJECT_ROOT / item for item in SHARED_BOUND]
    raw_path = PROJECT_ROOT / RAW_DATA

    record: Dict[str, Any] = {
        "protocol": {
            "path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(),
            "sha256_before_trials": sha256_file(PROTOCOL),
            "trial_count": len(TRIAL_IDS),
        },
        "passed": True,
        "environment": {
            "description": (
                "CPython 3.11.5; Windows 10 Pro 64-bit; Intel Core "
                "i7-11700 (8 cores, 16 logical processors); 31.7 GiB RAM; CPU"
            ),
            "python_executable_version": sys.version.split()[0],
        },
        "timings_seconds": {
            "complete_pipeline": describe(pipeline_times),
            "independent_full_raw_rescan_verifier": describe(
                full_verifier_times
            ),
            "independent_portable_verifier": describe(portable_times),
        },
        "portable_trials": portable_trials,
        "complete_run_directories": run_inventories,
        "per_run_required_verifier_inputs": inventory(required_paths),
        "shared_bound_route_files": inventory(shared_paths),
        "raw_dataset": {
            "path": RAW_DATA.as_posix(),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
            "required_for": "38-check full raw-input route only",
        },
        "integration_categories": [
            "requested claim and adjacency",
            "actual generated-record mapping and attribution",
            "pipeline, preprocessing, population, schedule, and selection",
            "mechanism and accountant declaration",
            "content-bound runtime trace and model/batch artifacts",
            "versioned registry plus bound source/checker",
            "public and internal typed decision outputs",
        ],
        "scope_note": (
            "Machine-specific descriptive timings. The complete pipeline "
            "includes parsing, mapping, model training, report construction, "
            "and certificate generation; it is not isolated validator "
            "overhead or a throughput benchmark."
        ),
    }
    record_path = output_root / "cost_and_integration_summary.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pipeline = record["timings_seconds"]["complete_pipeline"]
    full = record["timings_seconds"]["independent_full_raw_rescan_verifier"]
    portable = record["timings_seconds"]["independent_portable_verifier"]
    required = record["per_run_required_verifier_inputs"]
    shared = record["shared_bound_route_files"]
    average_run_bytes = statistics.mean(
        item["total_bytes"] for item in run_inventories.values()
    )
    lines = [
        "# Audit v4 cost and integration summary",
        "",
        "- Protocol result: **PASS**",
        "- Platform: CPython 3.11.5, Windows 10 Pro, Intel i7-11700 CPU, "
        "31.7 GiB RAM",
        f"- Complete pipeline: {pipeline['mean']:.3f} s mean "
        f"(sample SD {pipeline['sample_standard_deviation']:.3f}, "
        f"range {pipeline['minimum']:.3f}--{pipeline['maximum']:.3f}; n=5)",
        f"- Independent full raw-input verifier: {full['mean']:.3f} s mean "
        f"(sample SD {full['sample_standard_deviation']:.3f}, "
        f"range {full['minimum']:.3f}--{full['maximum']:.3f}; n=5)",
        f"- Independent portable verifier: {portable['mean']:.3f} s mean "
        f"(sample SD {portable['sample_standard_deviation']:.3f}, "
        f"range {portable['minimum']:.3f}--{portable['maximum']:.3f}; n=5)",
        f"- Complete generated directory: {average_run_bytes / 1_000_000:.3f} "
        "MB mean per run",
        f"- Required per-run verifier inputs: {required['file_count']} files, "
        f"{required['total_bytes'] / 1_000_000:.3f} MB",
        f"- Shared bound route files: {shared['file_count']} files, "
        f"{shared['total_bytes'] / 1_000:.1f} kB",
        f"- Raw WISDM input: {record['raw_dataset']['bytes'] / 1_000_000:.3f} "
        "MB; needed only for the 38-check full route",
        "",
        "The complete-pipeline number includes parsing, mapping, training, "
        "report construction, and certificates. It is not an estimate of "
        "isolated validator overhead. Portable verification omits the three "
        "raw-input reconstruction checks and passes 35/35.",
        "",
    ]
    (output_root / "cost_and_integration_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("cost_protocol_passed=True")
    print(f"portable_trials={len(TRIAL_IDS)}")
    print(f"Wrote {record_path}")


if __name__ == "__main__":
    main()
