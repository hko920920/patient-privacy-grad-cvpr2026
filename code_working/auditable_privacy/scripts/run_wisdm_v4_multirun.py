"""Run the frozen five-execution WISDM stochastic-robustness protocol.

This runner deliberately refuses to overwrite a confirmatory directory or an
existing execution manifest. The pre-result protocol is:
paper_notes/AAAI27_WISDM_V4_MULTIRUN_PROTOCOL_2026-07-24.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "paper_notes"
    / "AAAI27_WISDM_V4_MULTIRUN_PROTOCOL_2026-07-24.md"
)
RUN_IDS = tuple(f"confirmatory_run{index:02d}" for index in range(1, 6))
PIPELINE_REL = Path("scripts/wisdm_v2_gold_pipeline.py")
VERIFIER_REL = Path("scripts/verify_wisdm_v2_gold.py")
DEFAULT_DATA_REL = Path(
    "data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
)
DEFAULT_OUTPUT_REL = Path("reports/wisdm_v4_multirun_001")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def redact_project_path(value: str) -> str:
    redacted = value.replace(str(PROJECT_ROOT), ".")
    return redacted.replace(PROJECT_ROOT.as_posix(), ".")


def write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def directory_inventory(path: Path) -> Dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def run_command(
    command: List[str],
    run_dir: Path,
    label: str,
) -> Dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{label}_stdout.log").write_text(
        redact_project_path(result.stdout),
        encoding="utf-8",
    )
    (run_dir / f"{label}_stderr.log").write_text(
        redact_project_path(result.stderr),
        encoding="utf-8",
    )
    return {
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "return_code": result.returncode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_REL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_REL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    if not PROTOCOL_PATH.is_file():
        raise SystemExit(f"Missing frozen protocol: {PROTOCOL_PATH}")
    if not data_path.is_file():
        raise SystemExit(f"Missing WISDM raw input: {data_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "execution_manifest.json"
    if manifest_path.exists():
        raise SystemExit(
            "Refusing to overwrite an existing execution manifest: "
            f"{manifest_path}"
        )
    existing = [
        output_root / run_id
        for run_id in RUN_IDS
        if (output_root / run_id).exists()
    ]
    if existing:
        raise SystemExit(
            "Refusing to overwrite confirmatory directories: "
            + ", ".join(str(path) for path in existing)
        )

    common_pipeline_args = [
        "--data-path",
        str(data_path),
        "--sample-rate-numerator",
        "1",
        "--sample-rate-denominator",
        "50",
        "--steps",
        "100",
        "--noise-multiplier",
        "2",
        "--clipping-norm",
        "1",
        "--optimizer-step-size",
        "0.006666666666666667",
        "--delta",
        "1e-6",
    ]
    manifest: Dict[str, Any] = {
        "protocol_version": "wisdm_v4_multirun_001",
        "protocol_path": relative_label(PROTOCOL_PATH),
        "protocol_sha256_before_execution": sha256_file(PROTOCOL_PATH),
        "pilot_excluded": "reports/wisdm_v4_multirun_001/noise2_run01",
        "confirmatory_run_ids": list(RUN_IDS),
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "data_path": relative_label(data_path),
        "data_sha256": sha256_file(data_path),
        "fixed_pipeline_arguments": common_pipeline_args[2:],
        "runs": [],
    }
    write_json(manifest_path, manifest)

    for run_id in RUN_IDS:
        run_dir = output_root / run_id
        pipeline_command = [
            sys.executable,
            str(PROJECT_ROOT / PIPELINE_REL),
            "--output-dir",
            str(run_dir),
            *common_pipeline_args,
        ]
        run_record: Dict[str, Any] = {
            "run_id": run_id,
            "output_dir": relative_label(run_dir),
            "pipeline_command": [
                "python",
                PIPELINE_REL.as_posix(),
                "--output-dir",
                relative_label(run_dir),
                "--data-path",
                relative_label(data_path),
                *common_pipeline_args[2:],
            ],
        }
        run_record["pipeline"] = run_command(
            pipeline_command,
            run_dir,
            "pipeline",
        )
        if run_record["pipeline"]["return_code"] != 0:
            run_record["inventory_after_failure"] = directory_inventory(run_dir)
            manifest["runs"].append(run_record)
            manifest["status"] = "FAILED"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(
                f"{run_id} pipeline failed; preserved evidence and stopped."
            )

        verification_output = run_dir / "independent_verification.json"
        verifier_command = [
            sys.executable,
            str(PROJECT_ROOT / VERIFIER_REL),
            "--evidence-dir",
            str(run_dir),
            "--data-path",
            str(data_path),
            "--output-json",
            str(verification_output),
        ]
        run_record["verifier_command"] = [
            "python",
            VERIFIER_REL.as_posix(),
            "--evidence-dir",
            relative_label(run_dir),
            "--data-path",
            relative_label(data_path),
            "--output-json",
            relative_label(verification_output),
        ]
        run_record["verifier"] = run_command(
            verifier_command,
            run_dir,
            "verifier",
        )
        run_record["inventory"] = directory_inventory(run_dir)
        manifest["runs"].append(run_record)
        if run_record["verifier"]["return_code"] != 0:
            manifest["status"] = "FAILED"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(
                f"{run_id} verifier failed; preserved evidence and stopped."
            )
        write_json(manifest_path, manifest)
        print(
            f"{run_id}: pipeline "
            f"{run_record['pipeline']['elapsed_seconds']:.3f}s; verifier "
            f"{run_record['verifier']['elapsed_seconds']:.3f}s"
        )

    manifest["status"] = "COMPLETE"
    manifest["finished_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"Completed {len(RUN_IDS)} confirmatory runs.")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

