from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


EXPECTED_CASE_IDS = {
    "legacy_observed_owner_schedule",
    "legacy_empty_owner_sample_skips_update",
    "legacy_replace_owner_adjacency_accepted",
    "legacy_public_certificate_exposes_execution_fields",
    "legacy_wisdm_mapping_is_stale",
    "legacy_sepsis_mapping_is_stale",
    "legacy_uci_runtime_scaler_crosses_owner_boundary",
    "legacy_fixed_size_claim_uses_epoch_shuffle_loader",
}


class VerificationError(RuntimeError):
    """Raised when a natural-failure report cannot be verified."""


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _require_dict(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{context} must be an object")
    return value


def _verify_repository_witness(value: object, context: str) -> None:
    witness = _require_dict(value, context)
    if set(witness) != {"path", "sha256", "bytes"}:
        raise VerificationError(f"{context} has unexpected fields")
    relative = witness["path"]
    if not isinstance(relative, str) or not relative:
        raise VerificationError(f"{context}.path is invalid")
    candidate = (ROOT / Path(relative)).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise VerificationError(f"{context}.path escapes the repository") from exc
    if not candidate.is_file():
        raise VerificationError(f"{context}.path is missing")
    if candidate.stat().st_size != witness["bytes"]:
        raise VerificationError(f"{context}.bytes mismatch")
    if _file_sha256(candidate) != witness["sha256"]:
        raise VerificationError(f"{context}.sha256 mismatch")


def _verify_external_witness(
    value: object,
    *,
    context: str,
    path: Path,
    expected_reference: str,
) -> None:
    witness = _require_dict(value, context)
    if set(witness) != {"public_reference", "sha256", "bytes"}:
        raise VerificationError(f"{context} has unexpected fields")
    if witness["public_reference"] != expected_reference:
        raise VerificationError(f"{context}.public_reference mismatch")
    if not path.is_file():
        raise VerificationError(f"{context} local public file is missing")
    if path.stat().st_size != witness["bytes"]:
        raise VerificationError(f"{context}.bytes mismatch")
    if _file_sha256(path) != witness["sha256"]:
        raise VerificationError(f"{context}.sha256 mismatch")


def verify_report(
    report_path: Path,
    *,
    uci_train_dir: Path,
) -> dict[str, object]:
    try:
        raw = json.loads(
            report_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Could not load report: {exc}") from exc
    report = _require_dict(raw, "report")
    if report.get("schema_version") != (
        "unitdp.natural_failure_corpus.v2.1"
    ):
        raise VerificationError("Unsupported report schema")
    reported_sha256 = report.get("natural_failure_corpus_sha256")
    if not isinstance(reported_sha256, str):
        raise VerificationError("Missing natural_failure_corpus_sha256")
    payload = dict(report)
    payload.pop("natural_failure_corpus_sha256")
    if _canonical_sha256(payload) != reported_sha256:
        raise VerificationError("Natural-failure payload digest mismatch")

    bindings = _require_dict(report.get("bindings"), "bindings")
    for name in ("builder", "oracle_spec", "oracle_implementation"):
        _verify_repository_witness(bindings.get(name), f"bindings.{name}")
    if (
        bindings.get("production_source_bundle_sha256")
        != execution_source_bundle_sha256_v2()
    ):
        raise VerificationError("Production source bundle mismatch")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise VerificationError("cases must be a list")
    case_objects = [
        _require_dict(case, f"cases[{index}]")
        for index, case in enumerate(cases)
    ]
    case_ids = [case.get("case_id") for case in case_objects]
    if set(case_ids) != EXPECTED_CASE_IDS or len(case_ids) != len(
        EXPECTED_CASE_IDS
    ):
        raise VerificationError("Natural-failure case ids mismatch")
    if any(case.get("status") != "reproduced" for case in case_objects):
        raise VerificationError("At least one case is not reproduced")

    external_by_reference = {
        "UCI HAR Dataset/train/X_train.txt": (
            uci_train_dir / "X_train.txt"
        ),
        "UCI HAR Dataset/train/subject_train.txt": (
            uci_train_dir / "subject_train.txt"
        ),
    }
    for case_index, case in enumerate(case_objects):
        witnesses = case.get("witnesses")
        if not isinstance(witnesses, list) or not witnesses:
            raise VerificationError(
                f"cases[{case_index}].witnesses must be non-empty"
            )
        for witness_index, witness_raw in enumerate(witnesses):
            witness = _require_dict(
                witness_raw,
                f"cases[{case_index}].witnesses[{witness_index}]",
            )
            context = (
                f"cases[{case_index}].witnesses[{witness_index}]"
            )
            if "path" in witness:
                _verify_repository_witness(witness, context)
                continue
            reference = witness.get("public_reference")
            if reference not in external_by_reference:
                raise VerificationError(
                    f"{context} has an unknown public reference"
                )
            _verify_external_witness(
                witness,
                context=context,
                path=external_by_reference[str(reference)],
                expected_reference=str(reference),
            )

    summary = _require_dict(report.get("summary"), "summary")
    if (
        summary.get("case_count") != len(case_objects)
        or summary.get("reproduced_count") != len(case_objects)
        or summary.get("all_cases_reproduced") is not True
        or summary.get(
            "contract_mutation_cases_with_oracle_production_rejection"
        )
        != 5
        or summary.get("mapping_array_binding_cases_rejected") != 2
        or summary.get("public_private_boundary_cases_rejected") != 1
    ):
        raise VerificationError("Summary counts are inconsistent")

    uci_case = next(
        case
        for case in case_objects
        if case["case_id"]
        == "legacy_uci_runtime_scaler_crosses_owner_boundary"
    )
    uci_observed = _require_dict(uci_case.get("observed"), "uci.observed")
    if (
        uci_observed.get("changed_entries") != 3_929_802
        or uci_observed.get("total_entries") != 3_929_805
        or uci_observed.get(
            "registered_fixed_transform_removal_invariant"
        )
        is not True
    ):
        raise VerificationError("UCI removal witness is inconsistent")

    for dataset, old_rows, current_rows in (
        ("wisdm", 6933, 7049),
        ("sepsis", 1628, 1601),
    ):
        case = next(
            item
            for item in case_objects
            if item["case_id"] == f"legacy_{dataset}_mapping_is_stale"
        )
        observed = _require_dict(
            case.get("observed"),
            f"{dataset}.observed",
        )
        if (
            observed.get("old_mapping_rows") != old_rows
            or observed.get("registered_mapping_rows") != current_rows
            or observed.get("prepared_feature_rows") != current_rows
            or observed.get("production_executor_rejected") is not True
        ):
            raise VerificationError(
                f"{dataset} stale-mapping witness is inconsistent"
            )

    return {
        "schema_version": (
            "unitdp.natural_failure_corpus_verification.v2.1"
        ),
        "status": "verified",
        "case_count": len(case_objects),
        "current_source_bindings": "verified",
        "external_public_witnesses": "verified",
        "natural_failure_corpus_sha256": reported_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=(
            ROOT
            / "reports"
            / "v2_natural_failure_corpus_v32r1_20260724"
            / "natural_failure_corpus_v2.json"
        ),
    )
    parser.add_argument(
        "--uci-train-dir",
        type=Path,
        default=(
            DATA_ROOT
            / "uci_har"
            / "extracted"
            / "UCI HAR Dataset"
            / "train"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_report(
        args.report.resolve(),
        uci_train_dir=args.uci_train_dir.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
