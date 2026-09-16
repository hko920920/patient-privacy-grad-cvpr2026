"""Build a public stage-attribution report for frozen V2 rejection evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v2_mutation_audit_v32_20260724"
    / "public_mutation_audit_v2.json"
)
DEFAULT_NATURAL_FAILURE_REPORT = (
    ROOT
    / "reports"
    / "v2_natural_failure_corpus_v32r1_20260724"
    / "natural_failure_corpus_v2.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v2_boundary_ablation_v32r1_20260724"
    / "public_boundary_ablation_v2.json"
)
VERIFIER = ROOT / "scripts" / "verify_v2_boundary_ablation.py"

MUTATION_STAGE_TO_BOUNDARY = {
    "contract_load": "contract_only",
    "contract_parse": "contract_only",
    "compile": "compiled_route",
    "require_executable": "executable_backend_gate",
    "pre_execution_integrity": "integrity_boundary",
    "mutation_attempt": "integrity_boundary",
}
MUTATION_BOUNDARY_ORDER = (
    "contract_only",
    "compiled_route",
    "executable_backend_gate",
    "integrity_boundary",
)
NATURAL_BOUNDARY_ORDER = (
    "contract_only",
    "data_execution_binding",
    "public_artifact_boundary",
)


class BoundaryAblationBuildError(RuntimeError):
    """Raised when a frozen input cannot support the stage attribution."""


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BoundaryAblationBuildError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryAblationBuildError(
            f"Could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BoundaryAblationBuildError(
            f"{path} must contain a JSON object"
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


def _verify_embedded_digest(
    report: dict[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    reported = report.get(field)
    if not isinstance(reported, str):
        raise BoundaryAblationBuildError(
            f"{context} is missing {field}"
        )
    payload = dict(report)
    payload.pop(field)
    if _canonical_sha256(payload) != reported:
        raise BoundaryAblationBuildError(
            f"{context} embedded digest is inconsistent"
        )
    return reported


def _object_list(
    value: object,
    *,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BoundaryAblationBuildError(
            f"{context} must be a list"
        )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BoundaryAblationBuildError(
                f"{context}[{index}] must be an object"
            )
        rows.append(item)
    return rows


def _mutation_attribution(
    report: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    if report.get("schema_version") != (
        "unitdp.compiler_mutation_audit_public.v2.1"
    ):
        raise BoundaryAblationBuildError(
            "Unexpected mutation-report schema"
        )
    cases = _object_list(report.get("cases"), context="mutation.cases")
    if (
        len(cases) != 52
        or report.get("mutation_case_count") != 52
        or report.get("all_mutations_rejected") is not True
    ):
        raise BoundaryAblationBuildError(
            "Mutation report is not the frozen 52-case passing audit"
        )
    case_ids = [case.get("case_id") for case in cases]
    if (
        any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        raise BoundaryAblationBuildError(
            "Mutation case ids must be unique nonempty strings"
        )

    ids_by_boundary: dict[str, list[str]] = {
        boundary: [] for boundary in MUTATION_BOUNDARY_ORDER
    }
    for case in cases:
        if (
            case.get("expected") != "reject"
            or case.get("observed") != "rejected"
        ):
            raise BoundaryAblationBuildError(
                f"Mutation case {case.get('case_id')} is not rejected"
            )
        stage = case.get("stage")
        if stage not in MUTATION_STAGE_TO_BOUNDARY:
            raise BoundaryAblationBuildError(
                f"Unknown mutation stage: {stage!r}"
            )
        boundary = MUTATION_STAGE_TO_BOUNDARY[str(stage)]
        ids_by_boundary[boundary].append(str(case["case_id"]))

    rows: list[dict[str, object]] = []
    cumulative = 0
    for boundary in MUTATION_BOUNDARY_ORDER:
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
    if cumulative != len(cases):
        raise BoundaryAblationBuildError(
            "Mutation attribution does not cover every case"
        )
    return rows, ids_by_boundary


def _natural_boundary(case: dict[str, Any]) -> str:
    judgment = case.get("v2_judgment")
    if not isinstance(judgment, dict):
        raise BoundaryAblationBuildError(
            f"Natural case {case.get('case_id')} lacks v2_judgment"
        )
    parser = judgment.get("production_parser")
    if isinstance(parser, dict) and parser.get("accepted") is False:
        return "contract_only"
    if (
        judgment.get("oracle_mapping_array_binding") == "rejected"
        and judgment.get("production_executor")
        == "rejected_before_training"
    ):
        return "data_execution_binding"
    if judgment.get("production_public_boundary") == "rejected":
        return "public_artifact_boundary"
    raise BoundaryAblationBuildError(
        f"Could not attribute natural case {case.get('case_id')}"
    )


def _natural_attribution(
    report: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    if report.get("schema_version") != (
        "unitdp.natural_failure_corpus.v2.1"
    ):
        raise BoundaryAblationBuildError(
            "Unexpected natural-failure schema"
        )
    cases = _object_list(report.get("cases"), context="natural.cases")
    summary = report.get("summary")
    if (
        len(cases) != 8
        or not isinstance(summary, dict)
        or summary.get("case_count") != 8
        or summary.get("reproduced_count") != 8
        or summary.get("all_cases_reproduced") is not True
    ):
        raise BoundaryAblationBuildError(
            "Natural-failure report is not the frozen 8-case corpus"
        )
    case_ids = [case.get("case_id") for case in cases]
    if (
        any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or any(case.get("status") != "reproduced" for case in cases)
    ):
        raise BoundaryAblationBuildError(
            "Natural cases must be unique and reproduced"
        )

    ids_by_boundary: dict[str, list[str]] = {
        boundary: [] for boundary in NATURAL_BOUNDARY_ORDER
    }
    for case in cases:
        ids_by_boundary[_natural_boundary(case)].append(
            str(case["case_id"])
        )

    rows: list[dict[str, object]] = []
    cumulative = 0
    for boundary in NATURAL_BOUNDARY_ORDER:
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
    if cumulative != len(cases):
        raise BoundaryAblationBuildError(
            "Natural-failure attribution does not cover every case"
        )
    return rows, ids_by_boundary


def _write_json(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mutation-report",
        type=Path,
        default=DEFAULT_MUTATION_REPORT,
    )
    parser.add_argument(
        "--natural-failure-report",
        type=Path,
        default=DEFAULT_NATURAL_FAILURE_REPORT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mutation_path = args.mutation_report.resolve()
    natural_path = args.natural_failure_report.resolve()
    mutation_report = _load_json(mutation_path)
    natural_report = _load_json(natural_path)
    mutation_payload_sha256 = _verify_embedded_digest(
        mutation_report,
        "public_mutation_audit_sha256",
        context="mutation report",
    )
    natural_payload_sha256 = _verify_embedded_digest(
        natural_report,
        "natural_failure_corpus_sha256",
        context="natural-failure report",
    )
    mutation_rows, mutation_ids = _mutation_attribution(
        mutation_report
    )
    natural_rows, natural_ids = _natural_attribution(natural_report)

    mutation_exclusive = Counter(
        {
            row["boundary"]: int(row["newly_rejected"])
            for row in mutation_rows
        }
    )
    natural_exclusive = Counter(
        {
            row["boundary"]: int(row["newly_rejected"])
            for row in natural_rows
        }
    )
    report: dict[str, Any] = {
        "schema_version": "unitdp.boundary_ablation_public.v2.1",
        "visibility": "public",
        "status": "passed",
        "scope": (
            "stage attribution of the frozen 52 targeted mutations and "
            "8 natural failures; not a completeness or privacy claim"
        ),
        "baseline_definition": (
            "contract_only includes duplicate-key-safe loading and strict "
            "registered contract parsing, but excludes mapping/preprocessor "
            "compilation, executable-backend approval, data/execution "
            "binding, compiled-object integrity, and artifact redaction"
        ),
        "bindings": {
            "builder_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": _file_sha256(Path(__file__).resolve()),
            "verifier_path": str(VERIFIER.relative_to(ROOT)),
            "verifier_sha256": _file_sha256(VERIFIER),
            "mutation_report_path": str(
                mutation_path.relative_to(ROOT.resolve())
            ),
            "mutation_report_file_sha256": _file_sha256(mutation_path),
            "mutation_report_payload_sha256": mutation_payload_sha256,
            "natural_failure_report_path": str(
                natural_path.relative_to(ROOT.resolve())
            ),
            "natural_failure_report_file_sha256": _file_sha256(
                natural_path
            ),
            "natural_failure_payload_sha256": natural_payload_sha256,
        },
        "targeted_mutations": {
            "rows": mutation_rows,
            "case_ids_by_earliest_boundary": mutation_ids,
        },
        "natural_failures": {
            "rows": natural_rows,
            "case_ids_by_earliest_boundary": natural_ids,
        },
        "summary": {
            "contract_only_targeted_rejected": mutation_exclusive[
                "contract_only"
            ],
            "full_lifecycle_targeted_rejected": sum(
                mutation_exclusive.values()
            ),
            "targeted_increment_beyond_contract_only": sum(
                mutation_exclusive.values()
            )
            - mutation_exclusive["contract_only"],
            "contract_only_natural_rejected": natural_exclusive[
                "contract_only"
            ],
            "full_lifecycle_natural_rejected": sum(
                natural_exclusive.values()
            ),
            "natural_increment_beyond_contract_only": sum(
                natural_exclusive.values()
            )
            - natural_exclusive["contract_only"],
        },
    }
    report["boundary_ablation_sha256"] = _canonical_sha256(report)
    _write_json(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "boundary_ablation_sha256": report[
                    "boundary_ablation_sha256"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
