"""Independently verify the public V2 boundary-attribution report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "v2_boundary_ablation_v32r1_20260724"
    / "public_boundary_ablation_v2.json"
)
EXPECTED_MUTATION_STAGES = (
    (
        "contract_only",
        {"contract_load", "contract_parse"},
    ),
    ("compiled_route", {"compile"}),
    ("executable_backend_gate", {"require_executable"}),
    (
        "integrity_boundary",
        {"pre_execution_integrity", "mutation_attempt"},
    ),
)
EXPECTED_NATURAL_BOUNDARIES = (
    "contract_only",
    "data_execution_binding",
    "public_artifact_boundary",
)


class BoundaryAblationVerificationError(RuntimeError):
    """Raised when the public boundary report is inconsistent."""


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BoundaryAblationVerificationError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryAblationVerificationError(
            f"Could not load {context}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BoundaryAblationVerificationError(
            f"{context} must be an object"
        )
    return value


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


def _resolve_bound_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BoundaryAblationVerificationError(
            f"{context} is invalid"
        )
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise BoundaryAblationVerificationError(
            f"{context} escapes the repository"
        ) from exc
    if not path.is_file():
        raise BoundaryAblationVerificationError(
            f"{context} does not identify a file"
        )
    return path


def _verify_embedded_digest(
    report: dict[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    reported = report.get(field)
    if not isinstance(reported, str):
        raise BoundaryAblationVerificationError(
            f"{context} is missing {field}"
        )
    payload = dict(report)
    payload.pop(field)
    if _canonical_sha256(payload) != reported:
        raise BoundaryAblationVerificationError(
            f"{context} embedded digest mismatch"
        )
    return reported


def _object_list(
    value: object,
    *,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BoundaryAblationVerificationError(
            f"{context} must be a list"
        )
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BoundaryAblationVerificationError(
                f"{context}[{index}] must be an object"
            )
        result.append(item)
    return result


def _expected_mutation_sections(
    report: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    cases = _object_list(report.get("cases"), context="mutation.cases")
    if (
        report.get("schema_version")
        != "unitdp.compiler_mutation_audit_public.v2.1"
        or len(cases) != 52
        or report.get("mutation_case_count") != 52
        or report.get("all_mutations_rejected") is not True
    ):
        raise BoundaryAblationVerificationError(
            "Bound mutation report is not the frozen passing audit"
        )
    known_stages = {
        stage
        for _, stages in EXPECTED_MUTATION_STAGES
        for stage in stages
    }
    stage_to_boundary = {
        stage: boundary
        for boundary, stages in EXPECTED_MUTATION_STAGES
        for stage in stages
    }
    ids_by_boundary = {
        boundary: [] for boundary, _ in EXPECTED_MUTATION_STAGES
    }
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        stage = case.get("stage")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or stage not in known_stages
            or case.get("expected") != "reject"
            or case.get("observed") != "rejected"
        ):
            raise BoundaryAblationVerificationError(
                "Mutation case cannot be independently attributed"
            )
        seen.add(case_id)
        ids_by_boundary[stage_to_boundary[str(stage)]].append(case_id)

    rows: list[dict[str, object]] = []
    cumulative = 0
    for boundary, _ in EXPECTED_MUTATION_STAGES:
        exclusive = len(ids_by_boundary[boundary])
        cumulative += exclusive
        rows.append(
            {
                "boundary": boundary,
                "newly_rejected": exclusive,
                "cumulative_rejected": cumulative,
                "total_cases": len(cases),
            }
        )
    return rows, ids_by_boundary


def _classify_natural(case: dict[str, Any]) -> str:
    judgment = case.get("v2_judgment")
    if not isinstance(judgment, dict):
        raise BoundaryAblationVerificationError(
            "Natural case lacks a V2 judgment"
        )
    parser = judgment.get("production_parser")
    if isinstance(parser, dict) and parser.get("accepted") is False:
        return "contract_only"
    if (
        judgment.get("production_executor")
        == "rejected_before_training"
        and judgment.get("oracle_mapping_array_binding") == "rejected"
    ):
        return "data_execution_binding"
    if judgment.get("production_public_boundary") == "rejected":
        return "public_artifact_boundary"
    raise BoundaryAblationVerificationError(
        f"Natural case {case.get('case_id')} has no supported boundary"
    )


def _expected_natural_sections(
    report: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    cases = _object_list(report.get("cases"), context="natural.cases")
    summary = report.get("summary")
    if (
        report.get("schema_version")
        != "unitdp.natural_failure_corpus.v2.1"
        or len(cases) != 8
        or not isinstance(summary, dict)
        or summary.get("case_count") != 8
        or summary.get("reproduced_count") != 8
        or summary.get("all_cases_reproduced") is not True
    ):
        raise BoundaryAblationVerificationError(
            "Bound natural-failure report is not the frozen corpus"
        )
    ids_by_boundary = {
        boundary: [] for boundary in EXPECTED_NATURAL_BOUNDARIES
    }
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or case.get("status") != "reproduced"
        ):
            raise BoundaryAblationVerificationError(
                "Natural case ids/status are inconsistent"
            )
        seen.add(case_id)
        ids_by_boundary[_classify_natural(case)].append(case_id)

    rows: list[dict[str, object]] = []
    cumulative = 0
    for boundary in EXPECTED_NATURAL_BOUNDARIES:
        exclusive = len(ids_by_boundary[boundary])
        cumulative += exclusive
        rows.append(
            {
                "boundary": boundary,
                "newly_rejected": exclusive,
                "cumulative_rejected": cumulative,
                "total_cases": len(cases),
            }
        )
    return rows, ids_by_boundary


def verify_report(path: Path) -> dict[str, object]:
    report = _load_json(path, context="boundary report")
    if (
        report.get("schema_version")
        != "unitdp.boundary_ablation_public.v2.1"
        or report.get("visibility") != "public"
        or report.get("status") != "passed"
    ):
        raise BoundaryAblationVerificationError(
            "Unsupported boundary-report header"
        )
    boundary_sha256 = _verify_embedded_digest(
        report,
        "boundary_ablation_sha256",
        context="boundary report",
    )
    bindings = report.get("bindings")
    if not isinstance(bindings, dict):
        raise BoundaryAblationVerificationError(
            "Boundary report lacks bindings"
        )

    builder_path = _resolve_bound_path(
        bindings.get("builder_path"),
        context="bindings.builder_path",
    )
    verifier_path = _resolve_bound_path(
        bindings.get("verifier_path"),
        context="bindings.verifier_path",
    )
    mutation_path = _resolve_bound_path(
        bindings.get("mutation_report_path"),
        context="bindings.mutation_report_path",
    )
    natural_path = _resolve_bound_path(
        bindings.get("natural_failure_report_path"),
        context="bindings.natural_failure_report_path",
    )
    for candidate, key in (
        (builder_path, "builder_sha256"),
        (verifier_path, "verifier_sha256"),
        (mutation_path, "mutation_report_file_sha256"),
        (natural_path, "natural_failure_report_file_sha256"),
    ):
        if _file_sha256(candidate) != bindings.get(key):
            raise BoundaryAblationVerificationError(
                f"{key} does not match the bound file"
            )

    mutation_report = _load_json(
        mutation_path,
        context="bound mutation report",
    )
    natural_report = _load_json(
        natural_path,
        context="bound natural-failure report",
    )
    mutation_payload = _verify_embedded_digest(
        mutation_report,
        "public_mutation_audit_sha256",
        context="bound mutation report",
    )
    natural_payload = _verify_embedded_digest(
        natural_report,
        "natural_failure_corpus_sha256",
        context="bound natural-failure report",
    )
    if (
        mutation_payload
        != bindings.get("mutation_report_payload_sha256")
        or natural_payload
        != bindings.get("natural_failure_payload_sha256")
    ):
        raise BoundaryAblationVerificationError(
            "A bound input payload digest changed"
        )

    mutation_rows, mutation_ids = _expected_mutation_sections(
        mutation_report
    )
    natural_rows, natural_ids = _expected_natural_sections(
        natural_report
    )
    targeted = report.get("targeted_mutations")
    natural = report.get("natural_failures")
    if (
        not isinstance(targeted, dict)
        or targeted.get("rows") != mutation_rows
        or targeted.get("case_ids_by_earliest_boundary")
        != mutation_ids
        or not isinstance(natural, dict)
        or natural.get("rows") != natural_rows
        or natural.get("case_ids_by_earliest_boundary") != natural_ids
    ):
        raise BoundaryAblationVerificationError(
            "Published stage attribution does not recompute"
        )

    summary = {
        "contract_only_targeted_rejected": mutation_rows[0][
            "cumulative_rejected"
        ],
        "full_lifecycle_targeted_rejected": mutation_rows[-1][
            "cumulative_rejected"
        ],
        "targeted_increment_beyond_contract_only": (
            int(mutation_rows[-1]["cumulative_rejected"])
            - int(mutation_rows[0]["cumulative_rejected"])
        ),
        "contract_only_natural_rejected": natural_rows[0][
            "cumulative_rejected"
        ],
        "full_lifecycle_natural_rejected": natural_rows[-1][
            "cumulative_rejected"
        ],
        "natural_increment_beyond_contract_only": (
            int(natural_rows[-1]["cumulative_rejected"])
            - int(natural_rows[0]["cumulative_rejected"])
        ),
    }
    if report.get("summary") != summary:
        raise BoundaryAblationVerificationError(
            "Boundary summary does not recompute"
        )
    if summary != {
        "contract_only_targeted_rejected": 36,
        "full_lifecycle_targeted_rejected": 52,
        "targeted_increment_beyond_contract_only": 16,
        "contract_only_natural_rejected": 5,
        "full_lifecycle_natural_rejected": 8,
        "natural_increment_beyond_contract_only": 3,
    }:
        raise BoundaryAblationVerificationError(
            "Frozen stage-attribution totals changed"
        )
    return {
        "schema_version": "unitdp.boundary_ablation_verification.v2.1",
        "status": "verified",
        "boundary_ablation_sha256": boundary_sha256,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_report(args.report.resolve()),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
