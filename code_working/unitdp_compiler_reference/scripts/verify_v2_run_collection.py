"""Verify all hash, schema, redaction, and summary links in a V2 collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.execution_verifier_v2 import (  # noqa: E402
    verify_collection_artifacts_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection_dir",
        help="Directory containing public_collection_v2.json.",
    )
    parser.add_argument(
        "--source-root",
        default=str(ROOT),
        help="Repository root used to verify current bound source files.",
    )
    args = parser.parse_args()

    verified = verify_collection_artifacts_v2(
        args.collection_dir,
        source_root=args.source_root,
    )
    report = {
        "schema_version": "unitdp.collection_verification_report.v2.1",
        "status": "verified",
        "collection_dir": str(Path(args.collection_dir).resolve()),
        "public_collection_sha256": (
            verified.public_collection_sha256
        ),
        "dataset_count": verified.dataset_count,
        "run_count": verified.run_count,
        "current_source_bindings_verified": (
            verified.current_source_bindings_verified
        ),
        "datasets": [
            {
                "dataset": item.dataset,
                "run_count": item.run_count,
                "public_summary_sha256": item.public_summary_sha256,
            }
            for item in verified.datasets
        ],
    }
    print(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
