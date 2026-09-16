"""Verify and fully regenerate the two-route mechanism-matched audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_v3_mechanism_matched_reference_audit import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_POISSON_EVIDENCE,
    DEFAULT_SEPSIS_CACHE,
    DEFAULT_SRSWOR_EVIDENCE,
    DEFAULT_UCI_ROOT,
    DEFAULT_WISDM_RAW,
    MechanismMatchedReferenceError,
    _assert_public_report,
    build_report,
    file_sha256,
    load_json,
    payload_sha256,
)


def _without_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "payload_sha256"
    }


def verify_integrity(path: Path) -> dict[str, Any]:
    report = load_json(path)
    _assert_public_report(report)
    payload = report.get("payload_sha256")
    if payload != payload_sha256(_without_payload(report)):
        raise MechanismMatchedReferenceError(
            "Mechanism-matched report payload digest differs"
        )
    for relative, expected in report["source_files_sha256"].items():
        source = ROOT / relative
        if file_sha256(source) != expected:
            raise MechanismMatchedReferenceError(
                f"Reference source hash differs: {relative}"
            )
    totals = report.get("totals", {})
    expected_totals = {
        "datasets": 6,
        "runs": 30,
        "diagnostic_steps_exact": 850,
        "final_models_bitwise": 30,
        "final_tensors_bitwise": 60,
        "public_metrics_exact": 120,
        "maximum_public_metric_gap": 0.0,
    }
    if totals != expected_totals:
        raise MechanismMatchedReferenceError(
            f"Unexpected final totals: {totals}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--uci-root", default=str(DEFAULT_UCI_ROOT))
    parser.add_argument("--wisdm-raw", default=str(DEFAULT_WISDM_RAW))
    parser.add_argument(
        "--sepsis-cache",
        default=str(DEFAULT_SEPSIS_CACHE),
    )
    parser.add_argument(
        "--poisson-evidence",
        default=str(DEFAULT_POISSON_EVIDENCE),
    )
    parser.add_argument(
        "--srswor-evidence",
        default=str(DEFAULT_SRSWOR_EVIDENCE),
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Skip private-input full regeneration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_integrity(Path(args.report))
    if not args.integrity_only:
        regenerated = build_report(args)
        if regenerated != report:
            raise MechanismMatchedReferenceError(
                "Full regeneration differs from the frozen report"
            )
    print(
        "verified mechanism-matched reference audit: "
        f"{report['totals']['runs']} runs, "
        f"{report['totals']['diagnostic_steps_exact']} steps, "
        f"{report['totals']['final_tensors_bitwise']} tensors"
    )
    print(
        "full_regeneration="
        f"{str(not args.integrity_only).lower()}"
    )
    print(
        "payload_sha256="
        f"{report['payload_sha256']}"
    )


if __name__ == "__main__":
    main()
