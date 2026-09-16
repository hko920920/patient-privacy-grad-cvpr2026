"""Strictly verify a public V2 compiler mutation-audit report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


BUILD_SCRIPT = ROOT / "scripts" / "build_v2_mutation_audit.py"
UCI_CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
UCI_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)


class MutationAuditVerificationError(RuntimeError):
    """Raised when the public mutation audit fails closed."""


def unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MutationAuditVerificationError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise MutationAuditVerificationError(
            f"{context} fields differ from the strict schema"
        )


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except MutationAuditVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MutationAuditVerificationError(
            f"Could not load mutation audit: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MutationAuditVerificationError(
            "Mutation audit must be a JSON object"
        )
    return value


def verify(path: Path) -> dict[str, Any]:
    report = load_report(path)
    require_keys(
        report,
        {
            "schema_version",
            "visibility",
            "status",
            "scope",
            "handling",
            "source_bindings",
            "baseline",
            "mutation_case_count",
            "all_mutations_rejected",
            "case_count_by_family",
            "mutation_spec_sha256",
            "cases",
            "adjacency_invariance",
            "public_mutation_audit_sha256",
        },
        "mutation_audit",
    )
    if (
        report["schema_version"]
        != "unitdp.compiler_mutation_audit_public.v2.1"
        or report["visibility"] != "public"
        or report["status"] != "passed"
        or report["scope"]
        != "strict_registered_owner_poisson_v2_route"
        or report["handling"]
        != "contains_no_private_mapping_or_research_seed"
    ):
        raise MutationAuditVerificationError(
            "Mutation-audit public header is invalid"
        )
    try:
        assert_public_certificate_redacted(report)
    except ValueError as exc:
        raise MutationAuditVerificationError(
            f"Mutation audit crosses the public/private boundary: {exc}"
        ) from exc

    reported_hash = report["public_mutation_audit_sha256"]
    without_hash = dict(report)
    without_hash.pop("public_mutation_audit_sha256")
    if payload_sha256(without_hash) != reported_hash:
        raise MutationAuditVerificationError(
            "Mutation-audit self-hash mismatch"
        )

    cases = report["cases"]
    if (
        not isinstance(cases, list)
        or report["mutation_case_count"] != len(cases)
        or report["mutation_case_count"] != 52
        or report["all_mutations_rejected"] is not True
    ):
        raise MutationAuditVerificationError(
            "Mutation-audit case count/status is invalid"
        )
    case_ids: list[str] = []
    observed_family_counts: dict[str, int] = {}
    case_spec: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise MutationAuditVerificationError(
                f"Mutation case {index} is not an object"
            )
        require_keys(
            case,
            {
                "case_id",
                "family",
                "stage",
                "expected",
                "observed",
                "exception_class",
            },
            f"mutation case {index}",
        )
        if (
            not isinstance(case["case_id"], str)
            or not isinstance(case["family"], str)
            or not isinstance(case["stage"], str)
            or case["expected"] != "reject"
            or case["observed"] != "rejected"
            or case["exception_class"]
            not in {
                "ContractV2Error",
                "ExecutorNotReadyError",
                "ValueError",
            }
        ):
            raise MutationAuditVerificationError(
                f"Mutation case {index} has an invalid result"
            )
        case_ids.append(case["case_id"])
        observed_family_counts[case["family"]] = (
            observed_family_counts.get(case["family"], 0) + 1
        )
        case_spec.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "stage": case["stage"],
                "expected": case["expected"],
            }
        )
    if len(case_ids) != len(set(case_ids)):
        raise MutationAuditVerificationError(
            "Mutation case ids are not unique"
        )
    if dict(sorted(observed_family_counts.items())) != (
        report["case_count_by_family"]
    ):
        raise MutationAuditVerificationError(
            "Mutation family counts do not match the cases"
        )
    if payload_sha256(case_spec) != report["mutation_spec_sha256"]:
        raise MutationAuditVerificationError(
            "Mutation case-spec hash mismatch"
        )

    adjacency = report["adjacency_invariance"]
    if not isinstance(adjacency, dict):
        raise MutationAuditVerificationError(
            "Adjacency-invariance result must be an object"
        )
    require_keys(
        adjacency,
        {
            "case_id",
            "family",
            "expected",
            "observed",
            "checks",
            "public_plan_sha256",
        },
        "adjacency_invariance",
    )
    checks = adjacency["checks"]
    expected_checks = {
        "public_contract_equal",
        "public_plan_equal",
        "opacus_epsilon_equal",
        "dp_accounting_epsilon_equal",
        "fixed_q_equal",
        "fixed_steps_equal",
        "fixed_denominator_equal",
    }
    if (
        adjacency["case_id"]
        != "add_remove_owner_public_plan_invariance"
        or adjacency["observed"] != "passed"
        or not isinstance(checks, dict)
        or set(checks) != expected_checks
        or any(value is not True for value in checks.values())
    ):
        raise MutationAuditVerificationError(
            "Adjacency-invariance checks did not all pass"
        )

    bindings = report["source_bindings"]
    if not isinstance(bindings, dict):
        raise MutationAuditVerificationError(
            "Mutation-audit source bindings must be an object"
        )
    require_keys(
        bindings,
        {
            "execution_source_bundle_sha256",
            "audit_script_sha256",
            "registered_contract_file_sha256",
            "registered_preprocessor_file_sha256",
        },
        "source_bindings",
    )
    expected_bindings = {
        "execution_source_bundle_sha256": (
            execution_source_bundle_sha256_v2()
        ),
        "audit_script_sha256": file_sha256(BUILD_SCRIPT),
        "registered_contract_file_sha256": file_sha256(UCI_CONFIG),
        "registered_preprocessor_file_sha256": (
            file_sha256(UCI_PREPROCESSOR)
        ),
    }
    if bindings != expected_bindings:
        raise MutationAuditVerificationError(
            "Mutation-audit source bindings differ from current files"
        )

    baseline = report["baseline"]
    if not isinstance(baseline, dict):
        raise MutationAuditVerificationError(
            "Mutation-audit baseline must be an object"
        )
    require_keys(
        baseline,
        {
            "contract_status",
            "execution_ready",
            "public_contract_sha256",
            "public_plan_sha256",
            "accountant_epsilon_opacus",
            "accountant_epsilon_dp_accounting",
        },
        "baseline",
    )
    if (
        baseline["contract_status"]
        != "validated_contract_research_executor_ready"
        or baseline["execution_ready"] is not True
        or adjacency["public_plan_sha256"]
        != baseline["public_plan_sha256"]
    ):
        raise MutationAuditVerificationError(
            "Mutation-audit positive baseline is inconsistent"
        )

    return {
        "schema_version": "unitdp.compiler_mutation_audit_verification.v2.1",
        "status": "verified",
        "mutation_case_count": len(cases),
        "adjacency_invariance": "verified",
        "current_source_bindings": "verified",
        "public_mutation_audit_sha256": reported_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        nargs="?",
        default=str(
            ROOT
            / "reports"
            / "v2_mutation_audit_v32_20260724"
            / "public_mutation_audit_v2.json"
        ),
    )
    args = parser.parse_args()
    result = verify(Path(args.report))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
