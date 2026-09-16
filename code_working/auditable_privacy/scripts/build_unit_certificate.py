"""Build fail-closed v2.0 privacy-claim certificates.

The mapping audit supplies observed diagnostics. All privacy authorization is
delegated to ``privacy_claim_validator.validate_privacy_claim``. This script
cannot turn an audit-row verdict, a warning, or arbitrary external JSON into a
positive public statement.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from privacy_claim_validator import (
    ClaimValidationResult,
    normalize_unit,
    validate_privacy_claim,
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_audit_row(
    rows: Iterable[Mapping[str, Any]], scenario: str, claim_unit: str
) -> Mapping[str, Any]:
    normalized_claim = normalize_unit(claim_unit)
    matches = [
        row
        for row in rows
        if str(row.get("scenario") or "") == scenario
        and normalize_unit(row.get("claimed_privacy_unit")) == normalized_claim
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one audit row for "
            f"scenario={scenario!r}, claim_unit={claim_unit!r}; found {len(matches)}."
        )
    return matches[0]


def write_csv(row: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: (
            ";".join(str(item) for item in value)
            if isinstance(value, list)
            else "" if value is None else value
        )
        for key, value in row.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable.keys()))
        writer.writeheader()
        writer.writerow(serializable)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json_strict(path: Path) -> Any:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-standard JSON numeric constant: {token}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )


def safe_filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not component:
        raise ValueError("Scenario and claim unit need a safe filename component.")
    return component


def _format_parameter(value: Any) -> str:
    if value is None or value == "":
        return "not available"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_internal_markdown(result: ClaimValidationResult, path: Path) -> None:
    lines = [
        "# Internal Privacy-Claim Audit Record",
        "",
        "This record may contain non-public provenance and diagnostics. It is not",
        "the reviewer-facing public certificate.",
        "",
        "## Decision",
        "",
        f"- Scenario: {result.scenario}",
        f"- Claimed unit: {result.claimed_privacy_unit}",
        f"- Accounting unit: {result.accounting_unit or 'not available'}",
        f"- Raw adjacency: {result.raw_adjacency or 'not available'}",
        f"- Accountant adjacency: {result.accountant_adjacency or 'not available'}",
        f"- Unit path: {result.unit_path}",
        f"- Release status: {result.release_status}",
        f"- Assurance status: {result.assurance_status}",
        f"- Stability method: {result.stability_method or 'not available'}",
        f"- Validated K: {_format_parameter(result.stability_bound)}",
        "",
        "## Privacy Parameters",
        "",
        f"- Base epsilon: {_format_parameter(result.base_epsilon)}",
        f"- Base delta: {_format_parameter(result.base_delta)}",
        f"- Reportable epsilon: {_format_parameter(result.reported_epsilon)}",
        f"- Reportable delta: {_format_parameter(result.reported_delta)}",
        f"- log10(reportable delta): {_format_parameter(result.reported_delta_log10)}",
        f"- Vacuous conversion: {_format_parameter(result.conversion_vacuous)}",
        "",
        "## Observed Mapping Diagnostics",
        "",
        f"- Observed event kappa: {_format_parameter(result.observed_event_kappa)}",
        f"- Observed owner kappa: {_format_parameter(result.observed_owner_kappa)}",
        f"- Selected record count: {_format_parameter(result.selected_record_count)}",
        f"- Maximum attribution arity: {_format_parameter(result.attribution_arity_max)}",
        f"- Generated record unit: {result.generated_record_unit or 'not available'}",
        f"- Mapping-file SHA-256: {result.mapping_file_sha256 or 'not available'}",
        f"- Selected-mapping SHA-256: {result.selected_mapping_sha256 or 'not available'}",
        f"- Pipeline SHA-256: {result.pipeline_sha256 or 'not available'}",
        f"- Runtime status: {result.runtime_status or 'not available'}",
        f"- RNG assurance: {result.rng_assurance or 'not available'}",
        f"- DP-noise seed policy: {result.noise_seed_policy or 'not available'}",
        f"- Metadata classification: {result.metadata_classification or 'not available'}",
        "",
        "Observed kappa values are diagnostics. Only the validated stability bound",
        "in the accountant metric controls the verdict.",
        "",
        "## Validation Issues",
        "",
    ]
    if result.issues:
        lines.extend(
            f"- `{issue.code}` [{issue.category}] `{issue.field}`: {issue.message}"
            for issue in result.issues
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Wording Control",
            "",
            f"- Assurance boundary: {result.assurance_boundary}",
            f"- Supported statement: {result.supported_statement or 'NONE'}",
            f"- Do not say: {result.do_not_say}",
            f"- Next step: {result.next_step}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_public_markdown(record: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Public Privacy-Claim Certificate",
        "",
        f"- Claimed unit: {record.get('claimed_privacy_unit', 'not available')}",
        f"- Accounting unit: {record.get('accounting_unit', 'not available')}",
        f"- Raw adjacency: {record.get('raw_adjacency', 'not available')}",
        f"- Accountant adjacency: {record.get('accountant_adjacency', 'not available')}",
        f"- Unit path: {record.get('unit_path', 'UNDERSPECIFIED')}",
        f"- Release status: {record.get('release_status', 'BLOCKED_UNVERIFIED')}",
        f"- Assurance status: {record.get('assurance_status', 'DIAGNOSTIC_ONLY')}",
        f"- Metadata redacted: {record.get('metadata_redacted', True)}",
        "",
    ]
    if record.get("release_status") == "ALLOWED":
        lines.extend(
            [
                "## Allowed Statement",
                "",
                str(record.get("supported_statement") or ""),
                "",
                f"- Epsilon: {_format_parameter(record.get('epsilon'))}",
                f"- Delta: {_format_parameter(record.get('delta'))}",
                f"- Validated K: {_format_parameter(record.get('stability_bound'))}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## No Positive Statement Authorized",
                "",
                "This certificate is blocked. It contains no copy-ready privacy claim.",
                "",
            ]
        )
    lines.extend(
        [
            "## Wording Control",
            "",
            f"- Assurance boundary: {record.get('assurance_boundary', '')}",
            f"- Do not say: {record.get('do_not_say', '')}",
            f"- Blocking issue codes: {', '.join(record.get('issue_codes', [])) or 'none'}",
            "",
        ]
    )
    if not record.get("metadata_redacted", True):
        lines.extend(
            [
                "## Public-Benchmark Provenance",
                "",
                f"- Scenario: {record.get('scenario', 'not available')}",
                f"- Selected-mapping SHA-256: {record.get('selected_mapping_sha256', 'not available')}",
                f"- Pipeline SHA-256: {record.get('pipeline_sha256', 'not available')}",
                f"- Observed event kappa: {_format_parameter(record.get('observed_event_kappa'))}",
                f"- Observed owner kappa: {_format_parameter(record.get('observed_owner_kappa'))}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument(
        "--mapping-csv",
        required=True,
        help=(
            "Actual mapping export. The validator reopens it, recomputes its "
            "digests, row count, and event/owner incidence."
        ),
    )
    parser.add_argument(
        "--recommendations-csv",
        default=None,
        help=(
            "Accepted only for CLI compatibility. Recommendation prose never "
            "participates in a v2.0 verdict."
        ),
    )
    parser.add_argument("--privacy-report-json", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--claim-unit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--diagnostic-exit-zero",
        action="store_true",
        help=(
            "Write a blocked diagnostic certificate but return zero. This does "
            "not change release_status or create positive wording."
        ),
    )
    return parser.parse_args()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audit_path = _resolve(project_root, args.audit_csv)
    mapping_path = _resolve(project_root, args.mapping_csv)
    privacy_path = _resolve(project_root, args.privacy_report_json)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_row = find_audit_row(
        read_csv(audit_path),
        scenario=args.scenario,
        claim_unit=args.claim_unit,
    )
    privacy_report = read_json_strict(privacy_path)
    if not isinstance(privacy_report, dict):
        raise ValueError("Privacy report JSON must contain one object.")

    result = validate_privacy_claim(
        audit_row=audit_row,
        privacy_report=privacy_report,
        claim_unit=args.claim_unit,
        mapping_path=mapping_path,
    )
    internal_record = result.as_internal_record()
    public_record = result.as_public_record()

    basename = (
        f"certificate_{safe_filename_component(args.scenario)}_"
        f"{safe_filename_component(args.claim_unit)}"
    )
    write_csv(internal_record, output_dir / f"{basename}_internal.csv")
    write_json(
        {
            **internal_record,
            "issues": [issue.as_dict() for issue in result.issues],
        },
        output_dir / f"{basename}_internal.json",
    )
    write_internal_markdown(result, output_dir / f"{basename}_internal.md")

    # The unsuffixed files are deliberately the redacted public certificate.
    write_csv(public_record, output_dir / f"{basename}.csv")
    write_json(public_record, output_dir / f"{basename}.json")
    write_public_markdown(public_record, output_dir / f"{basename}.md")

    print(f"release_status={result.release_status}")
    print(f"unit_path={result.unit_path}")
    print(f"issue_codes={';'.join(result.issue_codes())}")
    print(f"Wrote {output_dir / f'{basename}.md'}")
    print(f"Wrote {output_dir / f'{basename}_internal.md'}")

    if result.release_status != "ALLOWED" and not args.diagnostic_exit_zero:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
