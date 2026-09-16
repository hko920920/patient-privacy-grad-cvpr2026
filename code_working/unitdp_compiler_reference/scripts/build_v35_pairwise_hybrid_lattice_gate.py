#!/usr/bin/env python3
"""Exhaust the V3.5 pairwise route-surface hybrid lattice."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_v4 import (  # noqa: E402
    CompilerDispatchV4Error,
    parse_registered_contract_v4,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v35_pairwise_hybrid_lattice_gate_20260725"
    / "pairwise_hybrid_lattice_gate_v35.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")
SCHEMA_VERSION = "unitdp.v35_pairwise_hybrid_lattice_gate.v1"
REPORT_DATE = "2026-07-25"

DATASETS = ("uci", "wisdm", "sepsis")
ROUTES = {
    "P": ("v2", "owner_poisson_v2"),
    "F": ("v3", "owner_srswor_v3"),
    "A": ("v4", "owner_random_allocation_v4"),
}
ROUTE_PAIRS = (("P", "F"), ("P", "A"), ("F", "A"))

# These eight groups are disjoint and reconstruct the complete contract.  The
# mechanism groups partition every mechanism key used by P, F, or A.
SURFACE_ORDER = (
    "identity",
    "execution",
    "privacy_accountant",
    "data_lineage",
    "schedule_parameters",
    "noise_geometry",
    "update_contribution",
    "optimizer_sampler",
)
SCHEDULE_PARAMETER_KEYS = frozenset(
    {
        "sampling_unit",
        "accounting_unit",
        "parameter_source",
        "owner_sample_rate",
        "source_dataset_size",
        "sample_size",
        "total_steps",
        "num_steps",
        "num_selected_steps_per_owner",
        "num_epochs",
        "assignment_schedule",
    }
)
NOISE_GEOMETRY_KEYS = frozenset(
    {
        "clipping_unit",
        "noising_unit",
        "clip_norm",
        "noise_multiplier",
        "empty_step_behavior",
    }
)
UPDATE_CONTRIBUTION_KEYS = frozenset(
    {
        "update_denominator",
        "per_owner_contribution",
        "owner_aggregation",
    }
)
OPTIMIZER_SAMPLER_KEYS = frozenset({"sampler"})
MECHANISM_KEY_PARTITION = (
    SCHEDULE_PARAMETER_KEYS,
    NOISE_GEOMETRY_KEYS,
    UPDATE_CONTRIBUTION_KEYS,
    OPTIMIZER_SAMPLER_KEYS,
)

FROZEN_BASELINE = {
    "src/unitdp/compiler_v3.py": (
        "9a083ecd83943503ceb0d7b061296084c754e6fb6218b2786af85a2896aaa3d0"
    ),
    "dist/compiler_submission_upload_manifest_v3.json": (
        "d5fa835d92cdbb69c8dc4832c032cbc46f53e9a5f53eae6a5906b27bd5bdecbe"
    ),
    "dist/compiler_code_and_data_supplement_aaai27_v3.zip": (
        "7214060b29c91530c67c8c202647be495c5011e6c131839d9e5c271382d4c3b3"
    ),
}

CLAIM_SOURCES = (
    "scripts/build_v35_pairwise_hybrid_lattice_gate.py",
    "scripts/verify_v35_pairwise_hybrid_lattice_gate.py",
    "src/unitdp/compiler_v4.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_path(route: str, dataset: str) -> Path:
    directory, suffix = ROUTES[route]
    return (
        ROOT
        / "configs"
        / directory
        / f"{dataset}_{suffix}.yaml"
    )


def load_raw(route: str, dataset: str) -> dict[str, Any]:
    raw = yaml.safe_load(
        config_path(route, dataset).read_text(encoding="utf-8")
    )
    if not isinstance(raw, dict):
        raise TypeError("registered contract must be a mapping")
    return raw


def _mechanism_piece(
    raw: dict[str, Any],
    keys: frozenset[str],
) -> dict[str, Any]:
    mechanism = raw.get("mechanism")
    if not isinstance(mechanism, dict):
        raise TypeError("mechanism must be a mapping")
    return {
        key: copy.deepcopy(mechanism[key])
        for key in sorted(keys)
        if key in mechanism
    }


def surface_payload(
    raw: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    if surface == "identity":
        return {
            "schema_version": copy.deepcopy(raw["schema_version"]),
            "route_id": copy.deepcopy(raw["route_id"]),
        }
    if surface == "execution":
        return {"execution": copy.deepcopy(raw["execution"])}
    if surface == "privacy_accountant":
        return {
            "privacy": copy.deepcopy(raw["privacy"]),
            "accountant": copy.deepcopy(raw["accountant"]),
        }
    if surface == "data_lineage":
        return {"data": copy.deepcopy(raw["data"])}
    if surface == "schedule_parameters":
        return {
            "mechanism": _mechanism_piece(
                raw,
                SCHEDULE_PARAMETER_KEYS,
            )
        }
    if surface == "noise_geometry":
        return {
            "mechanism": _mechanism_piece(
                raw,
                NOISE_GEOMETRY_KEYS,
            )
        }
    if surface == "update_contribution":
        return {
            "mechanism": _mechanism_piece(
                raw,
                UPDATE_CONTRIBUTION_KEYS,
            ),
            "contribution_policy": copy.deepcopy(
                raw["contribution_policy"]
            ),
        }
    if surface == "optimizer_sampler":
        return {
            "mechanism": _mechanism_piece(
                raw,
                OPTIMIZER_SAMPLER_KEYS,
            ),
            "optimization": copy.deepcopy(raw["optimization"]),
        }
    raise KeyError(surface)


def merge_surface(
    target: dict[str, Any],
    piece: dict[str, Any],
) -> None:
    for key, value in piece.items():
        if key == "mechanism":
            mechanism = target.setdefault("mechanism", {})
            if not isinstance(mechanism, dict) or not isinstance(value, dict):
                raise TypeError("mechanism surface must be a mapping")
            overlap = set(mechanism) & set(value)
            if overlap:
                raise RuntimeError(
                    f"mechanism surface overlap: {sorted(overlap)}"
                )
            mechanism.update(copy.deepcopy(value))
        else:
            if key in target:
                raise RuntimeError(f"top-level surface overlap: {key}")
            target[key] = copy.deepcopy(value)


def hybrid_contract(
    left: dict[str, Any],
    right: dict[str, Any],
    mask: int,
) -> dict[str, Any]:
    if mask < 0 or mask >= 2 ** len(SURFACE_ORDER):
        raise ValueError("hybrid mask is out of range")
    result: dict[str, Any] = {}
    for bit, surface in enumerate(SURFACE_ORDER):
        source = right if mask & (1 << bit) else left
        merge_surface(result, surface_payload(source, surface))
    return result


def _partition_check() -> dict[str, Any]:
    route_rows: list[dict[str, Any]] = []
    all_mechanism_keys: set[str] = set()
    for route in ROUTES:
        for dataset in DATASETS:
            raw = load_raw(route, dataset)
            mechanism = raw["mechanism"]
            if not isinstance(mechanism, dict):
                raise TypeError("mechanism must be a mapping")
            all_mechanism_keys.update(mechanism)
            left = hybrid_contract(raw, raw, 0)
            checks = {
                "reconstructs_exact_contract": left == raw,
                "top_level_complete": (
                    set(left)
                    == {
                        "schema_version",
                        "route_id",
                        "execution",
                        "privacy",
                        "mechanism",
                        "accountant",
                        "contribution_policy",
                        "optimization",
                        "data",
                    }
                ),
            }
            route_rows.append(
                {
                    "route": route,
                    "dataset": dataset,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    mechanism_union = set().union(*MECHANISM_KEY_PARTITION)
    disjoint = sum(
        len(group) for group in MECHANISM_KEY_PARTITION
    ) == len(mechanism_union)
    return {
        "surface_order": list(SURFACE_ORDER),
        "surface_count": len(SURFACE_ORDER),
        "mechanism_key_groups": [
            sorted(group) for group in MECHANISM_KEY_PARTITION
        ],
        "observed_mechanism_keys": sorted(all_mechanism_keys),
        "partition_mechanism_keys": sorted(mechanism_union),
        "mechanism_groups_disjoint": disjoint,
        "mechanism_union_exact": (
            all_mechanism_keys == mechanism_union
        ),
        "route_rows": route_rows,
        "pass": (
            len(SURFACE_ORDER) == 8
            and disjoint
            and all_mechanism_keys == mechanism_union
            and all(row["pass"] for row in route_rows)
        ),
    }


def _pair_dataset_row(
    left_route: str,
    right_route: str,
    dataset: str,
) -> dict[str, Any]:
    left = load_raw(left_route, dataset)
    right = load_raw(right_route, dataset)
    surface_rows = []
    for surface in SURFACE_ORDER:
        left_piece = surface_payload(left, surface)
        right_piece = surface_payload(right, surface)
        surface_rows.append(
            {
                "surface": surface,
                "left_sha256": canonical_sha256(left_piece),
                "right_sha256": canonical_sha256(right_piece),
                "different": left_piece != right_piece,
            }
        )
    endpoint_left = hybrid_contract(left, right, 0)
    endpoint_right = hybrid_contract(
        left,
        right,
        2 ** len(SURFACE_ORDER) - 1,
    )
    left_contract = parse_registered_contract_v4(endpoint_left)
    right_contract = parse_registered_contract_v4(endpoint_right)

    accepted_masks: list[int] = []
    rejection_types: Counter[str] = Counter()
    transcript: list[str] = []
    payload_hashes: set[str] = set()
    identity_counts: Counter[str] = Counter()
    for mask in range(1, 2 ** len(SURFACE_ORDER) - 1):
        hybrid = hybrid_contract(left, right, mask)
        payload_hashes.add(canonical_sha256(hybrid))
        identity = (
            right_route if mask & 1 else left_route
        )
        identity_counts[identity] += 1
        try:
            parse_registered_contract_v4(hybrid)
        except CompilerDispatchV4Error as exc:
            error_type = type(exc).__name__
            rejection_types[error_type] += 1
            transcript.append(
                f"{mask:02x}:{error_type}:"
                f"{hashlib.sha256(str(exc).encode('utf-8')).hexdigest()}"
            )
        else:
            accepted_masks.append(mask)
            transcript.append(f"{mask:02x}:ACCEPT")
    required = 2 ** len(SURFACE_ORDER) - 2
    checks = {
        "all_surfaces_pairwise_different": all(
            row["different"] for row in surface_rows
        ),
        "left_endpoint_exact": endpoint_left == left,
        "right_endpoint_exact": endpoint_right == right,
        "endpoint_types_distinct": (
            type(left_contract) is not type(right_contract)
        ),
        "all_hybrid_payloads_unique": len(payload_hashes) == required,
        "balanced_identity_ownership": (
            identity_counts[left_route] == required // 2
            and identity_counts[right_route] == required // 2
        ),
        "all_nonendpoints_rejected": not accepted_masks,
        "complete_transcript": len(transcript) == required,
    }
    return {
        "pair": f"{left_route}/{right_route}",
        "dataset": dataset,
        "surface_rows": surface_rows,
        "required_nonendpoint_hybrids": required,
        "rejected_nonendpoint_hybrids": (
            required - len(accepted_masks)
        ),
        "accepted_masks": accepted_masks,
        "unique_hybrid_payloads": len(payload_hashes),
        "identity_counts": dict(identity_counts),
        "rejection_types": dict(rejection_types),
        "rejection_transcript_sha256": canonical_sha256(transcript),
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_report() -> dict[str, Any]:
    partition = _partition_check()
    rows = [
        _pair_dataset_row(left, right, dataset)
        for left, right in ROUTE_PAIRS
        for dataset in DATASETS
    ]
    required = sum(
        row["required_nonendpoint_hybrids"] for row in rows
    )
    rejected = sum(
        row["rejected_nonendpoint_hybrids"] for row in rows
    )
    frozen = [
        {
            "path": path,
            "expected_sha256": expected,
            "observed_sha256": file_sha256(ROOT / path),
            "pass": file_sha256(ROOT / path) == expected,
        }
        for path, expected in FROZEN_BASELINE.items()
    ]
    checks = {
        "surface_partition": partition["pass"],
        "nine_pair_dataset_lattices": (
            len(rows) == 9 and all(row["pass"] for row in rows)
        ),
        "all_2286_nonendpoints_rejected": (
            required == rejected == 2286
        ),
        "all_2286_payloads_unique_within_lattice": all(
            row["unique_hybrid_payloads"] == 254 for row in rows
        ),
        "all_endpoints_parse": all(
            row["checks"]["left_endpoint_exact"]
            and row["checks"]["right_endpoint_exact"]
            for row in rows
        ),
        "frozen_v3": all(row["pass"] for row in frozen),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_date": REPORT_DATE,
        "candidate": "AAAI27 Algorithm V3.5",
        "object": (
            "complete pairwise binary lattice over eight disjoint "
            "route-semantic surfaces"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "partition": partition,
        "route_pairs": [list(pair) for pair in ROUTE_PAIRS],
        "pair_dataset_rows": rows,
        "summary": {
            "route_pairs": len(ROUTE_PAIRS),
            "datasets": len(DATASETS),
            "lattices": len(rows),
            "surfaces": len(SURFACE_ORDER),
            "nonendpoint_hybrids_per_lattice": 254,
            "required_nonendpoint_hybrids": required,
            "rejected_nonendpoint_hybrids": rejected,
            "accepted_nonendpoint_hybrids": required - rejected,
        },
        "frozen_v3_baseline": frozen,
        "claim_source_sha256": {
            path: file_sha256(ROOT / path) for path in CLAIM_SOURCES
        },
        "verdict": {
            "pairwise_hybrid_lattice_gate": (
                "PASS" if all(checks.values()) else "FAIL"
            ),
            "bounded_statement": (
                "For the frozen P/F/A contracts, all 2,286 nonendpoint "
                "pairwise hybrids over the eight prespecified disjoint "
                "surfaces are rejected while all endpoints parse."
            ),
            "does_not_authorize": [
                "all possible leaf-level hybrid partitions",
                "arbitrary unregistered routes",
                "arbitrary-input completeness",
                "formal semantic noninterference",
                "a privacy proof",
                "a manuscript novelty claim",
            ],
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# V3.5 Pairwise Hybrid Lattice Gate",
            "",
            f"- Status: `{report['status']}`",
            f"- Payload SHA-256: `{report['payload_sha256']}`",
            f"- Surfaces: `{summary['surfaces']}`",
            f"- Pair/dataset lattices: `{summary['lattices']}`",
            (
                "- Nonendpoint hybrids rejected: "
                f"`{summary['rejected_nonendpoint_hybrids']}/"
                f"{summary['required_nonendpoint_hybrids']}`"
            ),
            "",
            "The claim is bounded to the frozen surface partition.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(report["payload_sha256"])
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
