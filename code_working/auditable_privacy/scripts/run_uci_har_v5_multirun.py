#!/usr/bin/env python3
"""Execute the five protocol-fixed Audit v5 UCI HAR confirmatory runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper_notes/AAAI27_UCI_HAR_V5_MULTIRUN_PROTOCOL_2026-07-24.md"
PROTOCOL_SHA256 = (
    "ca3c50c575a25cbc996bc363783d1ae10c3dd7352d4861e62e774324aaa91070"
)
PIPELINE = "scripts/uci_har_v5_pipeline.py"
PIPELINE_SHA256 = (
    "0c226ad3acb1d0fcc7a08904288611877269a173dfe30e6d3bcee75fa2b9d0bf"
)
VERIFIER = "scripts/verify_uci_har_v5.py"
VERIFIER_SHA256 = (
    "14193a2f8faffe7d1998ecbb62307556ceb4cc67eb3793161c4e52f567e08b26"
)
REGISTRY = "docs/mechanism_registry_v2_0.json"
REGISTRY_SHA256 = (
    "9344d1549f3db3265b45e4c0f10a19155ea20e77a14fd914c3d20a12a09e2834"
)
DATASET_BUNDLE_SHA256 = (
    "0d10319685c36554d12b88c91d5608805a6bfd95969de59de48c41c4d6ad7487"
)
RUN_IDS = tuple(f"confirmatory_run{index:02d}" for index in range(1, 6))


def file_sha256(path: Path) -> str:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run_command(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def check_frozen_sources() -> None:
    expected = {
        PROTOCOL: PROTOCOL_SHA256,
        PIPELINE: PIPELINE_SHA256,
        VERIFIER: VERIFIER_SHA256,
        REGISTRY: REGISTRY_SHA256,
    }
    drift = {
        path: {
            "actual": file_sha256(PROJECT_ROOT / path),
            "expected": digest,
        }
        for path, digest in expected.items()
        if file_sha256(PROJECT_ROOT / path) != digest
    }
    if drift:
        raise RuntimeError(f"Frozen UCI confirmatory source drift: {drift}")


def artifact_hashes(run_dir: Path) -> Dict[str, str]:
    paths = (
        "input_manifest.json",
        "mapping.csv",
        "released_model.json",
        "batch_trace.csv",
        "metrics.json",
        "privacy_report.json",
        "route_assertions.json",
        "independent_verification.json",
        "independent_verification_portable.json",
        "certificates/certificate_uci_har_v5_generated_window_window.json",
        "certificates/certificate_uci_har_v5_generated_window_event.json",
        "certificates/certificate_uci_har_v5_generated_window_owner.json",
    )
    return {
        relative: file_sha256(run_dir / relative)
        for relative in paths
        if (run_dir / relative).is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="reports/uci_har_v5_multirun_001",
    )
    parser.add_argument(
        "--data-root",
        default="data/uci_har/extracted/UCI HAR Dataset",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    check_frozen_sources()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite nonempty confirmatory root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    pilot_dir = PROJECT_ROOT / "reports" / "uci_har_v5_pilot_002"
    pilot = {
        "path": "reports/uci_har_v5_pilot_002",
        "excluded_from_aggregate": True,
        "exclusion_basis": "completed before confirmatory protocol freeze",
        "privacy_report_sha256": file_sha256(pilot_dir / "privacy_report.json"),
        "released_model_sha256": file_sha256(pilot_dir / "released_model.json"),
        "independent_verification_sha256": file_sha256(
            pilot_dir / "independent_verification.json"
        ),
    }
    manifest: Dict[str, Any] = {
        "schema_version": "uci_har_v5_multirun_manifest_v1",
        "protocol": {"path": PROTOCOL, "sha256": PROTOCOL_SHA256},
        "source_snapshot": {
            "pipeline": {"path": PIPELINE, "sha256": PIPELINE_SHA256},
            "verifier": {"path": VERIFIER, "sha256": VERIFIER_SHA256},
            "registry": {"path": REGISTRY, "sha256": REGISTRY_SHA256},
            "dataset_bundle_sha256": DATASET_BUNDLE_SHA256,
            "runner": {
                "path": "scripts/run_uci_har_v5_multirun.py",
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "parameters": {
            "sample_rate_numerator": 1,
            "sample_rate_denominator": 50,
            "steps": 100,
            "noise_multiplier": 2.0,
            "clipping_norm": 1.0,
            "optimizer_step_size": 1.0 / 200.0,
            "delta": 1e-6,
        },
        "pilot": pilot,
        "run_order": list(RUN_IDS),
        "runs": [],
    }
    failures = []
    for run_id in RUN_IDS:
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        pipeline_result = run_command(
            (
                sys.executable,
                str(PROJECT_ROOT / PIPELINE),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(run_dir),
            ),
            PROJECT_ROOT,
        )
        (run_dir / "pipeline_stdout.log").write_text(
            pipeline_result["stdout"],
            encoding="utf-8",
        )
        (run_dir / "pipeline_stderr.log").write_text(
            pipeline_result["stderr"],
            encoding="utf-8",
        )
        full_result = {
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
        portable_result = {
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
        if pipeline_result["returncode"] == 0:
            full_result = run_command(
                (
                    sys.executable,
                    str(PROJECT_ROOT / VERIFIER),
                    "--evidence-dir",
                    str(run_dir),
                    "--data-root",
                    str(data_root),
                ),
                PROJECT_ROOT,
            )
            (run_dir / "verifier_full_stdout.log").write_text(
                full_result["stdout"],
                encoding="utf-8",
            )
            (run_dir / "verifier_full_stderr.log").write_text(
                full_result["stderr"],
                encoding="utf-8",
            )
            portable_result = run_command(
                (
                    sys.executable,
                    str(PROJECT_ROOT / VERIFIER),
                    "--evidence-dir",
                    str(run_dir),
                    "--data-root",
                    str(data_root),
                    "--skip-input-rescan",
                ),
                PROJECT_ROOT,
            )
            (run_dir / "verifier_portable_stdout.log").write_text(
                portable_result["stdout"],
                encoding="utf-8",
            )
            (run_dir / "verifier_portable_stderr.log").write_text(
                portable_result["stderr"],
                encoding="utf-8",
            )

        record: Dict[str, Any] = {
            "run_id": run_id,
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_returncode": pipeline_result["returncode"],
            "full_verifier_returncode": full_result["returncode"],
            "portable_verifier_returncode": portable_result["returncode"],
            "artifact_hashes": artifact_hashes(run_dir),
        }
        for name in (
            "metrics.json",
            "independent_verification.json",
            "independent_verification_portable.json",
        ):
            path = run_dir / name
            if path.is_file():
                record[name.removesuffix(".json")] = json.loads(
                    path.read_text(encoding="utf-8")
                )
        record["passed"] = (
            record["pipeline_returncode"] == 0
            and record["full_verifier_returncode"] == 0
            and record["portable_verifier_returncode"] == 0
        )
        manifest["runs"].append(record)
        if not record["passed"]:
            failures.append(run_id)
        snapshot = dict(manifest)
        snapshot["content_sha256"] = canonical_json_sha256(snapshot)
        write_json(output_root / "execution_manifest.json", snapshot)

    manifest["content_sha256"] = canonical_json_sha256(manifest)
    write_json(output_root / "execution_manifest.json", manifest)
    print(
        f"UCI confirmatory runs complete: "
        f"{sum(run['passed'] for run in manifest['runs'])}/{len(RUN_IDS)}"
    )
    if failures:
        raise SystemExit(f"Confirmatory failures retained: {failures}")


if __name__ == "__main__":
    main()
