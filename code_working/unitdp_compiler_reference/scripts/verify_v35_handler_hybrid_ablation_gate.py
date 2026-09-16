"""Verify the V3.5 handler-hybrid ablation report by re-enumeration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_v4 import (  # noqa: E402
    ALLOCATION_HANDLER_V4,
    POISSON_HANDLER_V4,
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
    SRSWOR_HANDLER_V4,
    CompilerDispatchV4Error,
    RouteHandlerV4,
    validate_route_handler_v4,
)


DEFAULT_REPORT = (
    ROOT / "reports" / "v35_handler_hybrid_ablation_gate_20260725"
    / "handler_hybrid_ablation_gate_v35.json"
)
EXPECTED_TOTALS = {
    "pair_count": 3,
    "endpoint_validations": 6,
    "required_hybrids": 378,
    "weak_plugin_accepted": 378,
    "preseal_accepted": 42,
    "sealed_rejected": 378,
    "sealed_accepted": 0,
}
PAIR_ROWS = {
    "P/F": (POISSON_HANDLER_V4, SRSWOR_HANDLER_V4),
    "P/A": (POISSON_HANDLER_V4, ALLOCATION_HANDLER_V4),
    "F/A": (SRSWOR_HANDLER_V4, ALLOCATION_HANDLER_V4),
}


class HandlerHybridVerificationError(RuntimeError):
    """Raised when the handler-hybrid report cannot be reproduced."""


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


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HandlerHybridVerificationError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except HandlerHybridVerificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandlerHybridVerificationError(
            f"could not load handler-hybrid report: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HandlerHybridVerificationError(
            "handler-hybrid report is not an object"
        )
    return value


def reconstruct_candidate(
    left: RouteHandlerV4,
    right: RouteHandlerV4,
    mask: int,
    surfaces: list[dict[str, Any]],
) -> RouteHandlerV4:
    left_witnesses = {
        value.obligation_id: value
        for value in left.obligation_witnesses
    }
    right_witnesses = {
        value.obligation_id: value
        for value in right.obligation_witnesses
    }
    order = tuple(
        value.obligation_id for value in left.obligation_witnesses
    )
    fields: dict[str, Any] = {}
    witnesses: dict[str, Any] = {}
    for index, surface in enumerate(surfaces):
        choose_right = bool(mask & (1 << index))
        source = right if choose_right else left
        source_witnesses = (
            right_witnesses if choose_right else left_witnesses
        )
        for field in surface["fields"]:
            fields[field] = getattr(source, field)
        for obligation in surface["obligations"]:
            witnesses[obligation] = source_witnesses[obligation]
    if set(witnesses) != REQUIRED_REGISTRATION_OBLIGATIONS_V4:
        raise HandlerHybridVerificationError(
            "reported surfaces do not cover all obligations"
        )
    return replace(
        left,
        **fields,
        obligation_witnesses=tuple(
            witnesses[obligation] for obligation in order
        ),
    )


def verify_report(path: Path) -> dict[str, Any]:
    report = load_report(path)
    candidate = dict(report)
    reported_payload = candidate.pop("payload_sha256", None)
    if (
        not isinstance(reported_payload, str)
        or payload_sha256(candidate) != reported_payload
    ):
        raise HandlerHybridVerificationError(
            "handler-hybrid payload digest mismatch"
        )
    if report.get("status") != "PASS":
        raise HandlerHybridVerificationError(
            "handler-hybrid status is not PASS"
        )
    if report.get("totals") != EXPECTED_TOTALS:
        raise HandlerHybridVerificationError(
            "handler-hybrid exact totals mismatch"
        )
    surfaces = report.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != 7:
        raise HandlerHybridVerificationError(
            "handler-hybrid surface count mismatch"
        )
    field_names: list[str] = []
    obligation_names: list[str] = []
    for surface in surfaces:
        if not isinstance(surface, dict):
            raise HandlerHybridVerificationError(
                "malformed handler surface"
            )
        field_names.extend(surface.get("fields", []))
        obligation_names.extend(surface.get("obligations", []))
    if len(field_names) != len(set(field_names)):
        raise HandlerHybridVerificationError(
            "handler fields overlap between surfaces"
        )
    if (
        len(obligation_names) != len(set(obligation_names))
        or set(obligation_names)
        != REQUIRED_REGISTRATION_OBLIGATIONS_V4
    ):
        raise HandlerHybridVerificationError(
            "handler obligations are not disjoint/exhaustive"
        )

    source_hashes = report.get("claim_source_sha256")
    if not isinstance(source_hashes, dict):
        raise HandlerHybridVerificationError(
            "claim-source map is malformed"
        )
    for relative, expected in source_hashes.items():
        source = ROOT / Path(relative)
        if not source.is_file() or file_sha256(source) != expected:
            raise HandlerHybridVerificationError(
                f"claim source mismatch: {relative}"
            )

    endpoints = report.get("registration_endpoints")
    if not isinstance(endpoints, dict):
        raise HandlerHybridVerificationError(
            "endpoint map is malformed"
        )
    for label, handler in (
        ("P", POISSON_HANDLER_V4),
        ("F", SRSWOR_HANDLER_V4),
        ("A", ALLOCATION_HANDLER_V4),
    ):
        validate_route_handler_v4(handler)
        expected = {
            "route_id": handler.route_id,
            "registration_core_sha256": (
                handler.registration_core_sha256
            ),
            "registration_sha256": handler.registration_sha256,
            "evidence_report_sha256": (
                handler.evidence_report_sha256
            ),
        }
        if endpoints.get(label) != expected:
            raise HandlerHybridVerificationError(
                f"endpoint registration mismatch: {label}"
            )

    rows = report.get("hybrids")
    if not isinstance(rows, list) or len(rows) != 378:
        raise HandlerHybridVerificationError(
            "handler-hybrid row count mismatch"
        )
    by_key = {
        (row.get("pair"), row.get("mask")): row for row in rows
    }
    if len(by_key) != len(rows):
        raise HandlerHybridVerificationError(
            "handler-hybrid pair/mask keys are not unique"
        )
    strict_rejections = 0
    for pair_name, (left, right) in PAIR_ROWS.items():
        pair_hashes: set[str] = set()
        for mask in range(1, 127):
            row = by_key.get((pair_name, mask))
            if row is None:
                raise HandlerHybridVerificationError(
                    f"missing handler hybrid {pair_name}/{mask}"
                )
            hybrid = reconstruct_candidate(
                left,
                right,
                mask,
                surfaces,
            )
            if row.get("registration_sha256") != hybrid.registration_sha256:
                raise HandlerHybridVerificationError(
                    f"registration hash mismatch {pair_name}/{mask}"
                )
            expected_surfaces = [
                surface["id"]
                for index, surface in enumerate(surfaces)
                if mask & (1 << index)
            ]
            if row.get("selected_right_surfaces") != expected_surfaces:
                raise HandlerHybridVerificationError(
                    f"surface mask mismatch {pair_name}/{mask}"
                )
            if (
                row.get("weak_plugin_accept") is not True
                or row.get("sealed_validator_accept") is not False
            ):
                raise HandlerHybridVerificationError(
                    f"reported decision mismatch {pair_name}/{mask}"
                )
            try:
                validate_route_handler_v4(hybrid)
            except CompilerDispatchV4Error as exc:
                observed_error = hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest()
                if observed_error != row.get("sealed_error_sha256"):
                    raise HandlerHybridVerificationError(
                        f"error digest mismatch {pair_name}/{mask}"
                    )
                strict_rejections += 1
            else:
                raise HandlerHybridVerificationError(
                    f"sealed validator accepted {pair_name}/{mask}"
                )
            pair_hashes.add(hybrid.registration_sha256)
        if len(pair_hashes) != 126:
            raise HandlerHybridVerificationError(
                f"hybrids are not unique for {pair_name}"
            )
    if strict_rejections != 378:
        raise HandlerHybridVerificationError(
            "sealed rejection total mismatch"
        )
    return {
        "status": "verified",
        "hybrids_rebuilt": 378,
        "sealed_rejections": strict_rejections,
        "endpoint_validations": 3,
        "payload_sha256": reported_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_report(Path(args.report).resolve())
    result["claim_bearing_hashes"] = {
        "report": file_sha256(Path(args.report).resolve()),
        "verifier": file_sha256(Path(__file__).resolve()),
        "test": file_sha256(
            ROOT / "tests"
            / "test_v35_handler_hybrid_ablation_gate.py"
        )
        if (
            ROOT / "tests"
            / "test_v35_handler_hybrid_ablation_gate.py"
        ).is_file()
        else None,
    }
    result["verification_payload_sha256"] = payload_sha256(result)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
