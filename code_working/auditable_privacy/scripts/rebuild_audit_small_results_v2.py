"""Rebuild and verify the small Audit result families cross-platform."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(label: str, *arguments: str) -> None:
    print(f"\n[{label}]")
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reproduced_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    ablation = output_root / "validator_ablation"
    wisdm = output_root / "wisdm"
    existing = [path for path in (ablation, wisdm) if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite rebuilt output: "
            + ", ".join(str(path) for path in existing)
        )

    raw_wisdm = (
        PROJECT_ROOT
        / "data"
        / "wisdm"
        / "WISDM_ar_v1.1"
        / "WISDM_ar_v1.1_raw.txt"
    )
    if not raw_wisdm.is_file():
        raise SystemExit(
            "WISDM raw input is required at "
            "data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
        )

    run(
        "rebuild controlled probes",
        "scripts/run_validator_ablation_v2.py",
        "--output-dir",
        str(ablation),
    )
    run(
        "verify controlled probes",
        "scripts/verify_validator_ablation_v2.py",
        "--bundle-dir",
        str(ablation),
        "--output-json",
        str(ablation / "independent_verification.json"),
    )
    run(
        "rebuild WISDM execution",
        "scripts/wisdm_v2_gold_pipeline.py",
        "--output-dir",
        str(wisdm),
    )
    run(
        "verify WISDM execution",
        "scripts/verify_wisdm_v2_gold.py",
        "--evidence-dir",
        str(wisdm),
        "--data-path",
        str(raw_wisdm),
        "--output-json",
        str(wisdm / "independent_verification.json"),
    )
    print(f"\nRebuilt small result families under {output_root}")


if __name__ == "__main__":
    main()

