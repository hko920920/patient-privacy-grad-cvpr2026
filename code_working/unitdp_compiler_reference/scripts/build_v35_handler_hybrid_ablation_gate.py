"""Build the V3.5 exhaustive handler-hybrid and pre-seal ablation gate."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
from dataclasses import asdict, replace
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


OUTPUT = (
    ROOT / "reports" / "v35_handler_hybrid_ablation_gate_20260725"
)
PAIR_ROWS = (
    ("P/F", POISSON_HANDLER_V4, SRSWOR_HANDLER_V4),
    ("P/A", POISSON_HANDLER_V4, ALLOCATION_HANDLER_V4),
    ("F/A", SRSWOR_HANDLER_V4, ALLOCATION_HANDLER_V4),
)
SURFACES = (
    {
        "id": "identity_and_evidence_report",
        "fields": (
            "route_id",
            "schema_version",
            "adjacency",
            "sampler",
            "accountant_family",
            "evidence_report_path",
            "evidence_report_sha256",
        ),
        "obligations": (),
    },
    {
        "id": "contract_and_compiled_types",
        "fields": (
            "contract_type",
            "compiled_type",
            "validation_error_types",
            "integrity_method_name",
        ),
        "obligations": (),
    },
    {
        "id": "strict_parser",
        "fields": ("parser",),
        "obligations": ("strict_parser",),
    },
    {
        "id": "strict_compiler",
        "fields": ("compiler",),
        "obligations": ("strict_compiler",),
    },
    {
        "id": "route_specific_registry",
        "fields": ("specific_registry_entry",),
        "obligations": (),
    },
    {
        "id": "accounting_and_numeric_crosscheck",
        "fields": (),
        "obligations": ("accountant", "numeric_crosscheck"),
    },
    {
        "id": "execution_integrity_and_source",
        "fields": (),
        "obligations": (
            "executor",
            "postcompile_integrity",
            "source_binding",
        ),
    },
)


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


def implementation_id(value: object) -> str:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", "")
    if not module or not name:
        raise ValueError("object has no qualified implementation id")
    return f"{module}.{name}"


def resolve_implementation(value: str) -> object:
    parts = value.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            target: object = importlib.import_module(
                ".".join(parts[:split])
            )
        except ImportError:
            continue
        try:
            for component in parts[split:]:
                target = getattr(target, component)
        except AttributeError:
            continue
        return target
    raise ValueError(f"could not resolve {value!r}")


def source_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    candidate.relative_to(ROOT.resolve())
    if not candidate.is_file():
        raise ValueError(f"missing source: {relative}")
    return candidate


def weak_plugin_accepts(handler: RouteHandlerV4) -> bool:
    """Model a conventional callable/type plugin map without route evidence."""

    return (
        isinstance(handler.route_id, str)
        and bool(handler.route_id)
        and isinstance(handler.schema_version, str)
        and bool(handler.schema_version)
        and inspect.isclass(handler.contract_type)
        and inspect.isclass(handler.compiled_type)
        and callable(handler.parser)
        and callable(handler.compiler)
        and len(handler.obligation_witnesses) == 7
    )


def preseal_validator_accepts(handler: RouteHandlerV4) -> bool:
    """Reproduce the source-bound validator before complete-core sealing."""

    try:
        registry = handler.specific_registry_entry
        for field_name, expected in (
            ("route_id", handler.route_id),
            ("adjacency", handler.adjacency),
            ("sampler", handler.sampler),
        ):
            if getattr(registry, field_name, None) != expected:
                return False
        if getattr(registry, "executor_status", "") not in {
            "implemented_research_validated",
            "implemented_release_validated",
        }:
            return False
        witnesses = {
            witness.obligation_id: witness
            for witness in handler.obligation_witnesses
        }
        if (
            len(witnesses) != len(handler.obligation_witnesses)
            or set(witnesses) != REQUIRED_REGISTRATION_OBLIGATIONS_V4
        ):
            return False
        integrity = getattr(
            handler.compiled_type,
            handler.integrity_method_name,
            None,
        )
        if not callable(integrity):
            return False
        expected_ids = {
            "strict_parser": implementation_id(handler.parser),
            "strict_compiler": implementation_id(handler.compiler),
            "executor": str(
                getattr(registry, "implementation_id", "")
            ),
            "postcompile_integrity": implementation_id(integrity),
        }
        if any(
            witnesses[key].implementation_id != expected
            for key, expected in expected_ids.items()
        ):
            return False
        for witness in handler.obligation_witnesses:
            path = source_path(witness.source_path)
            if file_sha256(path) != witness.source_sha256:
                return False
            implementation = resolve_implementation(
                witness.implementation_id
            )
            if not callable(implementation):
                return False
            observed = Path(
                inspect.getsourcefile(implementation) or ""
            ).resolve()
            if (
                str(observed.relative_to(ROOT.resolve())).replace(
                    "\\", "/"
                )
                != witness.source_path
            ):
                return False
        report_path = source_path(handler.evidence_report_path)
        if file_sha256(report_path) != handler.evidence_report_sha256:
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return isinstance(report, dict) and report.get("status") == "PASS"
    except (
        AttributeError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def hybrid(
    left: RouteHandlerV4,
    right: RouteHandlerV4,
    mask: int,
) -> RouteHandlerV4:
    left_witnesses = {
        value.obligation_id: value
        for value in left.obligation_witnesses
    }
    right_witnesses = {
        value.obligation_id: value
        for value in right.obligation_witnesses
    }
    obligation_order = tuple(
        value.obligation_id for value in left.obligation_witnesses
    )
    fields: dict[str, Any] = {}
    witnesses: dict[str, Any] = {}
    for index, surface in enumerate(SURFACES):
        choose_right = bool(mask & (1 << index))
        source = right if choose_right else left
        source_witnesses = (
            right_witnesses if choose_right else left_witnesses
        )
        for name in surface["fields"]:
            fields[name] = getattr(source, name)
        for obligation in surface["obligations"]:
            witnesses[obligation] = source_witnesses[obligation]
    if set(witnesses) != REQUIRED_REGISTRATION_OBLIGATIONS_V4:
        raise AssertionError("surface partition does not cover obligations")
    return replace(
        left,
        **fields,
        obligation_witnesses=tuple(
            witnesses[obligation] for obligation in obligation_order
        ),
    )


def selected_surfaces(mask: int) -> list[str]:
    return [
        surface["id"]
        for index, surface in enumerate(SURFACES)
        if mask & (1 << index)
    ]


def build_report() -> dict[str, Any]:
    endpoint_checks = 0
    rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for pair_name, left, right in PAIR_ROWS:
        validate_route_handler_v4(left)
        validate_route_handler_v4(right)
        endpoint_checks += 2
        pair_rows: list[dict[str, Any]] = []
        unique: set[str] = set()
        for mask in range(1, 2 ** len(SURFACES) - 1):
            candidate = hybrid(left, right, mask)
            registration_sha256 = candidate.registration_sha256
            unique.add(registration_sha256)
            weak = weak_plugin_accepts(candidate)
            preseal = preseal_validator_accepts(candidate)
            strict_error: str | None = None
            try:
                validate_route_handler_v4(candidate)
            except CompilerDispatchV4Error as exc:
                strict_error = str(exc)
            if not weak:
                raise AssertionError("weak plugin map rejected a hybrid")
            if strict_error is None:
                raise AssertionError(
                    "sealed registration accepted a nonendpoint hybrid"
                )
            row = {
                "pair": pair_name,
                "mask": mask,
                "selected_right_surfaces": selected_surfaces(mask),
                "registration_sha256": registration_sha256,
                "weak_plugin_accept": weak,
                "preseal_validator_accept": preseal,
                "sealed_validator_accept": False,
                "sealed_error_type": "CompilerDispatchV4Error",
                "sealed_error_sha256": hashlib.sha256(
                    strict_error.encode("utf-8")
                ).hexdigest(),
            }
            rows.append(row)
            pair_rows.append(row)
        required = 2 ** len(SURFACES) - 2
        if len(unique) != required:
            raise AssertionError(
                f"{pair_name} handler hybrids are not unique"
            )
        pair_summaries.append(
            {
                "pair": pair_name,
                "required_hybrids": required,
                "unique_hybrids": len(unique),
                "weak_plugin_accepted": sum(
                    row["weak_plugin_accept"] for row in pair_rows
                ),
                "preseal_accepted": sum(
                    row["preseal_validator_accept"]
                    for row in pair_rows
                ),
                "sealed_rejected": sum(
                    not row["sealed_validator_accept"]
                    for row in pair_rows
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": (
            "unitdp.v35_handler_hybrid_ablation_gate.v1"
        ),
        "status": "PASS",
        "scope": (
            "finite exhaustive binary lattice over seven declared handler "
            "surfaces; not open-world plugin safety or privacy proof"
        ),
        "surfaces": [
            {
                "id": surface["id"],
                "fields": list(surface["fields"]),
                "obligations": list(surface["obligations"]),
            }
            for surface in SURFACES
        ],
        "surface_count": len(SURFACES),
        "pairs": pair_summaries,
        "totals": {
            "pair_count": len(PAIR_ROWS),
            "endpoint_validations": endpoint_checks,
            "required_hybrids": len(rows),
            "weak_plugin_accepted": sum(
                row["weak_plugin_accept"] for row in rows
            ),
            "preseal_accepted": sum(
                row["preseal_validator_accept"] for row in rows
            ),
            "sealed_rejected": sum(
                not row["sealed_validator_accept"] for row in rows
            ),
            "sealed_accepted": sum(
                row["sealed_validator_accept"] for row in rows
            ),
        },
        "hybrids": rows,
        "registration_endpoints": {
            label: {
                "route_id": handler.route_id,
                "registration_core_sha256": (
                    handler.registration_core_sha256
                ),
                "registration_sha256": handler.registration_sha256,
                "evidence_report_sha256": (
                    handler.evidence_report_sha256
                ),
            }
            for label, handler in (
                ("P", POISSON_HANDLER_V4),
                ("F", SRSWOR_HANDLER_V4),
                ("A", ALLOCATION_HANDLER_V4),
            )
        },
        "claim_source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(
                path
            )
            for path in (
                ROOT / "src" / "unitdp" / "compiler_v4.py",
                Path(__file__).resolve(),
                ROOT / "scripts"
                / "build_v35_route_handler_evidence.py",
                ROOT / "reports"
                / "v35_route_handler_evidence_20260725"
                / "route_handler_evidence_index_v35.json",
            )
        },
        "verdict": (
            "A weak callable/type plugin map accepts all hybrids; the "
            "pre-seal source-bound validator accepts 42; complete-core "
            "route evidence rejects all 378."
        ),
    }
    report["payload_sha256"] = payload_sha256(report)
    return report


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"refusing to overwrite handler-hybrid gate: {OUTPUT}"
        )
    OUTPUT.mkdir(parents=True)
    report = build_report()
    json_path = OUTPUT / "handler_hybrid_ablation_gate_v35.json"
    json_path.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# V3.5 Handler-Hybrid Ablation Gate",
        "",
        f"Status: **{report['status']}**",
        "",
        "```text",
        f"surfaces                 {report['surface_count']}",
        f"route pairs              {report['totals']['pair_count']}",
        f"nonendpoint hybrids      {report['totals']['required_hybrids']}",
        f"weak plugin accepted     {report['totals']['weak_plugin_accepted']}",
        f"pre-seal accepted        {report['totals']['preseal_accepted']}",
        f"sealed rejected          {report['totals']['sealed_rejected']}",
        f"sealed accepted          {report['totals']['sealed_accepted']}",
        "```",
        "",
        report["verdict"],
        "",
        "This is finite registration-binding evidence, not a privacy proof, "
        "formal refinement, or open-world plugin guarantee.",
        "",
        f"Canonical payload: `{report['payload_sha256']}`",
        "",
    ]
    (OUTPUT / "handler_hybrid_ablation_gate_v35.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps(report["totals"], sort_keys=True))
    print(f"payload_sha256={report['payload_sha256']}")


if __name__ == "__main__":
    main()
