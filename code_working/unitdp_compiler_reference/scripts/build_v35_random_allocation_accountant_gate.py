#!/usr/bin/env python3
"""Build the V3.5 random-allocation accountant theorem gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.random_allocation_accountant_v4 import (  # noqa: E402
    RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
    RANDOM_ALLOCATION_RDP_ORDERS_V4,
    RANDOM_ALLOCATION_THEOREM_ID_V4,
    RandomAllocationAccountingV4Error,
    account_random_allocation_v4,
    calibrate_random_allocation_sigma_v4,
)
from unitdp_spec_oracle.random_allocation_oracle_v4 import (  # noqa: E402
    account_random_allocation_oracle_v4,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v35_random_allocation_accountant_gate_20260725"
    / "random_allocation_accountant_gate_v35.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")
SCHEMA_VERSION = "unitdp.v35_random_allocation_accountant_gate.v1"
REPORT_DATE = "2026-07-25"
NUMERIC_TOLERANCE = 1e-10
SIGMA_TOLERANCE = 2e-13

CASES: tuple[dict[str, Any], ...] = (
    {
        "profile": "uci",
        "num_owners": 21,
        "num_steps": 15,
        "num_selected": 6,
        "num_epochs": 1,
        "update_denominator": 8.0,
        "sigma": 1.2295752282782475,
        "expected_optimal_order": 4,
        "external_pld_upper": 6.7189308002345465,
    },
    {
        "profile": "wisdm",
        "num_owners": 25,
        "num_steps": 20,
        "num_selected": 6,
        "num_epochs": 1,
        "update_denominator": 8.0,
        "sigma": 1.090645987511382,
        "expected_optimal_order": 4,
        "external_pld_upper": 6.952359145280982,
    },
    {
        "profile": "sepsis",
        "num_owners": 312,
        "num_steps": 50,
        "num_selected": 5,
        "num_epochs": 1,
        "update_denominator": 32.0,
        "sigma": 0.776172784941656,
        "expected_optimal_order": 3,
        "external_pld_upper": 7.0705960633291065,
    },
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


def _case_result(case: dict[str, Any]) -> dict[str, Any]:
    query = {
        "num_steps": int(case["num_steps"]),
        "num_selected": int(case["num_selected"]),
        "num_epochs": int(case["num_epochs"]),
        "sigma": float(case["sigma"]),
        "delta": 1e-5,
        "orders": RANDOM_ALLOCATION_RDP_ORDERS_V4,
    }
    production = account_random_allocation_v4(**query)
    oracle = account_random_allocation_oracle_v4(**query)
    calibrated = calibrate_random_allocation_sigma_v4(
        target_epsilon=8.0,
        delta=1e-5,
        num_steps=query["num_steps"],
        num_selected=query["num_selected"],
        num_epochs=query["num_epochs"],
        orders=query["orders"],
    )
    rdp_differences = [
        abs(observed - expected)
        for (_, observed), (_, expected) in zip(
            production.remove_rdp_by_order,
            oracle.remove_rdp_by_order,
        )
    ]
    checks = {
        "production_oracle_epsilon": (
            abs(production.epsilon - oracle.epsilon)
            <= NUMERIC_TOLERANCE
        ),
        "production_oracle_remove": (
            abs(
                production.epsilon_remove
                - oracle.epsilon_remove
            )
            <= NUMERIC_TOLERANCE
        ),
        "production_oracle_add": (
            abs(production.epsilon_add - oracle.epsilon_add)
            <= NUMERIC_TOLERANCE
        ),
        "production_oracle_all_rdp_orders": (
            max(rdp_differences, default=0.0)
            <= NUMERIC_TOLERANCE
        ),
        "optimal_order": (
            production.optimal_remove_order
            == oracle.optimal_remove_order
            == int(case["expected_optimal_order"])
        ),
        "bidirectional_max": (
            production.epsilon
            == max(
                production.epsilon_remove,
                production.epsilon_add,
            )
        ),
        "target_epsilon": production.epsilon <= 8.0 + 1e-12,
        "sigma_recalibration": (
            abs(calibrated - query["sigma"]) <= SIGMA_TOLERANCE
        ),
        "external_pld_non_authorizing_comparison": (
            float(case["external_pld_upper"])
            < production.epsilon
        ),
    }
    return {
        "profile": case["profile"],
        "num_owners": case["num_owners"],
        "update_denominator": case["update_denominator"],
        "query": {
            **query,
            "orders": list(query["orders"]),
        },
        "production": asdict(production),
        "oracle": asdict(oracle),
        "calibrated_sigma": calibrated,
        "external_pld_probe": {
            "epsilon_upper": case["external_pld_upper"],
            "authorizes_route": False,
        },
        "maximum_absolute_rdp_difference": max(
            rdp_differences,
            default=0.0,
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _query_identity_checks() -> dict[str, Any]:
    common: dict[str, Any] = {
        "num_steps": 20,
        "num_selected": 6,
        "num_epochs": 1,
        "sigma": 1.090645987511382,
        "delta": 1e-5,
        "orders": RANDOM_ALLOCATION_RDP_ORDERS_V4,
    }
    baseline = account_random_allocation_v4(**common)
    mutations = (
        ("num_steps_floor_equivalent", {"num_steps": 21}),
        ("num_selected", {"num_selected": 5}),
        ("num_epochs", {"num_epochs": 2}),
        (
            "sigma_one_ulp",
            {"sigma": math.nextafter(common["sigma"], math.inf)},
        ),
        ("delta", {"delta": 1.1e-5}),
        ("order_grid", {"orders": tuple(range(2, 16))}),
    )
    rows: list[dict[str, Any]] = []
    for name, mutation in mutations:
        changed = account_random_allocation_v4(
            **{**common, **mutation}
        )
        rows.append(
            {
                "name": name,
                "mutation": {
                    key: (
                        list(value)
                        if isinstance(value, tuple)
                        else value
                    )
                    for key, value in mutation.items()
                },
                "query_identity_changed": changed != baseline,
                "epsilon_changed": changed.epsilon != baseline.epsilon,
                "rdp_rows_changed": (
                    changed.remove_rdp_by_order
                    != baseline.remove_rdp_by_order
                ),
                "floor_reduction_same_numeric_bound": (
                    name == "num_steps_floor_equivalent"
                    and changed.epsilon == baseline.epsilon
                    and changed.remove_rdp_by_order
                    == baseline.remove_rdp_by_order
                    and changed.num_steps != baseline.num_steps
                ),
            }
        )
    return {
        "required": len(rows),
        "passed": sum(row["query_identity_changed"] for row in rows),
        "rows": rows,
        "pass": all(row["query_identity_changed"] for row in rows),
        "interpretation": (
            "The floor(t/k) theorem reduction can map distinct t values "
            "to the same numerical bound; the result therefore retains "
            "the original query identity."
        ),
    }


def _invalid_query_checks() -> dict[str, Any]:
    common: dict[str, Any] = {
        "num_steps": 20,
        "num_selected": 6,
        "num_epochs": 1,
        "sigma": 1.1,
        "delta": 1e-5,
        "orders": RANDOM_ALLOCATION_RDP_ORDERS_V4,
    }
    mutations = (
        ("boolean_steps", {"num_steps": True}),
        ("zero_steps", {"num_steps": 0}),
        ("zero_selected", {"num_selected": 0}),
        ("selected_exceeds_steps", {"num_selected": 21}),
        ("zero_epochs", {"num_epochs": 0}),
        ("zero_sigma", {"sigma": 0.0}),
        ("infinite_sigma", {"sigma": math.inf}),
        ("zero_delta", {"delta": 0.0}),
        ("unit_delta", {"delta": 1.0}),
        ("empty_orders", {"orders": ()}),
        ("descending_orders", {"orders": (3, 2)}),
        ("duplicate_orders", {"orders": (2, 2)}),
        ("order_below_two", {"orders": (1, 2)}),
    )
    rows: list[dict[str, Any]] = []
    for name, mutation in mutations:
        try:
            account_random_allocation_v4(
                **{**common, **mutation}
            )
        except RandomAllocationAccountingV4Error as exc:
            rows.append(
                {
                    "name": name,
                    "rejected": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "rejected": False,
                    "error_type": None,
                    "error": None,
                }
            )
    return {
        "required": len(rows),
        "rejected": sum(row["rejected"] for row in rows),
        "rows": rows,
        "pass": all(row["rejected"] for row in rows),
    }


def build_report() -> dict[str, Any]:
    cases = [_case_result(case) for case in CASES]
    baseline_rows = [
        {
            "path": path,
            "expected_sha256": expected,
            "observed_sha256": file_sha256(ROOT / path),
            "pass": file_sha256(ROOT / path) == expected,
        }
        for path, expected in FROZEN_BASELINE.items()
    ]
    query_identity = _query_identity_checks()
    invalid_queries = _invalid_query_checks()
    source_paths = (
        "src/unitdp/random_allocation_accountant_v4.py",
        "src/unitdp_spec_oracle/random_allocation_oracle_v4.py",
        "tests/test_random_allocation_accountant_v4.py",
        "scripts/build_v35_random_allocation_accountant_gate.py",
    )
    passed = (
        all(case["pass"] for case in cases)
        and all(row["pass"] for row in baseline_rows)
        and query_identity["pass"]
        and invalid_queries["pass"]
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_date": REPORT_DATE,
        "status": "PASS" if passed else "FAIL",
        "accountant": {
            "accountant_id": RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
            "theorem_id": RANDOM_ALLOCATION_THEOREM_ID_V4,
            "orders": list(RANDOM_ALLOCATION_RDP_ORDERS_V4),
            "directions": ["remove", "add"],
            "reported_epsilon": "maximum_of_both_directions",
            "remove_method": (
                "integer-partition direct RDP plus composition"
            ),
            "add_method": (
                "Gaussian direct bound with conservative composed "
                "normalizer shift"
            ),
        },
        "primary_sources": {
            "neurips_2025": (
                "https://papers.nips.cc/paper_files/paper/2025/hash/"
                "8a724af64e891da5c84078af2c308a72-"
                "Abstract-Conference.html"
            ),
            "neurips_2025_pdf_sha256": (
                "bc37ed6c89cbc2caa76909445ef8928368574a0b23b59cce"
                "8804e1abdc880803"
            ),
            "icml_2026_pld": "https://arxiv.org/abs/2602.17284",
            "authors_pld_repository": (
                "https://github.com/moshenfeld/PLD_accounting"
            ),
        },
        "external_probe": {
            "role": "comparison_only_not_route_authorization",
            "pld_accounting_version": "0.5.0",
            "pld_wheel_sha256": (
                "1585d4030e4cda6209b6e9726ad81e46c2fd7305ec3dd2631"
                "d28125059d59804"
            ),
            "authors_repository_commit": (
                "11ed6d14e846de658465fb91309f574ab933cdc9"
            ),
            "random_allocation_version": "1.0.5",
        },
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "sigma_tolerance": SIGMA_TOLERANCE,
        "cases": cases,
        "case_count": len(cases),
        "query_identity_checks": query_identity,
        "invalid_query_checks": invalid_queries,
        "frozen_baseline": baseline_rows,
        "source_sha256": {
            path: file_sha256(ROOT / path) for path in source_paths
        },
        "verdict": {
            "accountant_gate": "PASS" if passed else "FAIL",
            "authorizes_next_step": (
                "A contract/compiler implementation"
                if passed
                else "none"
            ),
            "does_not_authorize": [
                "new random-allocation mechanism",
                "new privacy accountant or theorem",
                "generic DP compiler",
                "privacy for an unimplemented executor",
                "production release",
                "utility superiority",
            ],
        },
        "limits": [
            (
                "The k-out-of-t route uses the floor(t/k) reduction; "
                "different t values can share one conservative bound."
            ),
            (
                "The direct integer-order bound is looser than the "
                "external 2026 PLD probe in all three registered settings."
            ),
            (
                "Arithmetic agreement does not prove that an executor "
                "implements the required hidden allocation law."
            ),
            (
                "All results remain research-only and use no release-grade "
                "randomness claim."
            ),
        ],
    }
    # Normalize tuples from dataclasses to their exact JSON representation so
    # an in-memory rebuild equals the serialized report byte-for-byte.
    report = json.loads(
        json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    report["payload_sha256"] = canonical_sha256(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for case in report["cases"]:
        rows.append(
            "| {profile} | {steps} | {selected} | {sigma:.15f} | "
            "{remove:.12f} | {add:.12f} | {pld:.12f} | {order} |".format(
                profile=case["profile"],
                steps=case["query"]["num_steps"],
                selected=case["query"]["num_selected"],
                sigma=case["query"]["sigma"],
                remove=case["production"]["epsilon_remove"],
                add=case["production"]["epsilon_add"],
                pld=case["external_pld_probe"]["epsilon_upper"],
                order=case["production"]["optimal_remove_order"],
            )
        )
    return f"""# V3.5 Random-Allocation Accountant Gate

Status: **{report["status"]}**

The gate reproduces the conservative direct NeurIPS 2025 bound through a
stable floating-point implementation and an independently structured
high-precision oracle.  Both privacy directions are evaluated and the larger
epsilon is reported.

| Profile | t | k | sigma | remove epsilon | add epsilon | external PLD upper | best alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

```text
profiles                              {report["case_count"]}/{report["case_count"]}
query-identity mutations              {report["query_identity_checks"]["passed"]}/{report["query_identity_checks"]["required"]}
invalid queries rejected              {report["invalid_query_checks"]["rejected"]}/{report["invalid_query_checks"]["required"]}
frozen baseline hashes                {sum(row["pass"] for row in report["frozen_baseline"])}/{len(report["frozen_baseline"])}
numeric tolerance                     {report["numeric_tolerance"]}
sigma tolerance                       {report["sigma_tolerance"]}
payload SHA-256                       {report["payload_sha256"]}
```

The 2026 PLD values are comparison-only.  They do not authorize the route and
are not a runtime dependency.  This gate authorizes only the next
contract/compiler implementation step; it does not establish an executor,
generic compiler, new accountant, production release, or utility advantage.
"""


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
    args.markdown.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
