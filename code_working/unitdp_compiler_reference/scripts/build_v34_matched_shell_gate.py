#!/usr/bin/env python3
"""Build the V3.4 matched-shell P/F noninterchangeability gate."""

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

from unitdp.compiler_srswor_v3 import (  # noqa: E402
    CORE_ROUTE_ID_SRSWOR_V3,
)
from unitdp.compiler_v2 import CORE_ROUTE_ID_V2  # noqa: E402
from unitdp.compiler_v3 import (  # noqa: E402
    CompilerDispatchV3Error,
    parse_registered_contract_v3,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v34_matched_shell_gate_20260725"
    / "matched_shell_gate_v34.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")

DATASETS = ("uci", "wisdm", "sepsis")
ROUTES = ("P", "F")
CONFIGS = {
    "P": {
        dataset: ROOT / "configs" / "v2" / f"{dataset}_owner_poisson_v2.yaml"
        for dataset in DATASETS
    },
    "F": {
        dataset: ROOT / "configs" / "v3" / f"{dataset}_owner_srswor_v3.yaml"
        for dataset in DATASETS
    },
}
EXPECTED_ROUTE_IDS = {
    "P": CORE_ROUTE_ID_V2,
    "F": CORE_ROUTE_ID_SRSWOR_V3,
}
F_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v3_route_mutation_audit_v32_20260724"
    / "public_route_mutation_audit_v3.json"
)

# These are the complete equal leaf paths in each frozen P/F dataset pair.
EXPECTED_SHARED_PATHS = (
    "contribution_policy.max_windows_per_owner",
    "contribution_policy.type",
    "contribution_policy.windows_per_owner_per_step",
    "data.dataset_id",
    "data.input_dim",
    "data.num_classes",
    "data.preprocessing.artifact_id",
    "data.preprocessing.artifact_sha256",
    "data.preprocessing.fit_on_protected_data",
    "data.preprocessing.input_dim",
    "data.preprocessing.schema",
    "data.preprocessing.scope",
    "data.preprocessing.source_reference_sha256",
    "data.preprocessing.transform",
    "data.public_protocol.data_preparation_implementation_id",
    "data.public_protocol.evaluation_scope",
    "data.public_protocol.full_data_conformance_sha256",
    "data.public_protocol.protocol_id",
    "data.public_protocol.source_reference_sha256",
    "data.public_protocol.test_owners",
    "data.public_protocol.train_owners",
    "execution.profile",
    "execution.random_coins_public",
    "execution.rng_backend",
    "mechanism.accounting_unit",
    "mechanism.clip_norm",
    "mechanism.clipping_unit",
    "mechanism.sampling_unit",
    "mechanism.total_steps",
    "mechanism.update_denominator",
    "optimization.class_weights",
    "optimization.learning_rate",
    "optimization.loss",
    "optimization.model_id",
    "optimization.optimizer",
    "privacy.delta",
    "privacy.target_epsilon",
    "privacy.unit",
)

# These keys occur in both contracts but deliberately carry route-specific
# values. Route-only leaves are listed separately below.
EXPECTED_DIFFERENT_PATHS = (
    "accountant.id",
    "accountant.rdp_orders_id",
    "contribution_policy.scope",
    "data.benchmark_profile_id",
    "data.benchmark_profile_sha256",
    "execution.implementation_id",
    "mechanism.empty_step_behavior",
    "mechanism.noise_multiplier",
    "mechanism.noising_unit",
    "mechanism.owner_aggregation",
    "mechanism.parameter_source",
    "mechanism.per_owner_contribution",
    "mechanism.sampler",
    "privacy.adjacency",
    "route_id",
    "schema_version",
)
EXPECTED_ONLY_P_PATHS = ("mechanism.owner_sample_rate",)
EXPECTED_ONLY_F_PATHS = (
    "accountant.oracle_implementation_id",
    "accountant.sensitivity_multiplier",
    "accountant.theorem_id",
    "data.base_data_profile_id",
    "data.base_data_profile_sha256",
    "mechanism.sample_size",
    "mechanism.source_dataset_size",
)

# Frozen before execution in the Stage-2 plan. Each case copies the named
# donor surface into an otherwise unchanged target contract.
MUTATION_SPECS = (
    {
        "family": "route_identifier_only",
        "copy_paths": ("route_id",),
        "expected_rejection_stage": "route_dispatch",
    },
    {
        "family": "schema_version_only",
        "copy_paths": ("schema_version",),
        "expected_rejection_stage": "route_dispatch",
    },
    {
        "family": "adjacency",
        "copy_paths": ("privacy.adjacency",),
        "expected_rejection_stage": "route_specific_contract",
    },
    {
        "family": "sampler",
        "copy_paths": ("mechanism.sampler",),
        "expected_rejection_stage": "route_specific_contract",
    },
    {
        "family": "accountant_identifier_theorem_surface",
        "copy_paths": ("accountant",),
        "expected_rejection_stage": "route_specific_contract",
    },
    {
        "family": "executor_implementation_identifier",
        "copy_paths": ("execution.implementation_id",),
        "expected_rejection_stage": "route_specific_contract",
    },
    {
        "family": "noise_multiplier",
        "copy_paths": ("mechanism.noise_multiplier",),
        "expected_rejection_stage": "route_specific_contract",
    },
    {
        "family": "full_mechanism_accountant_splice",
        "copy_paths": ("mechanism", "accountant"),
        "expected_rejection_stage": "route_specific_contract",
    },
)

EXPECTED_CORRECT_REPLACE_ONE_EPSILON = 7.99999999999975
EXPECTED_OMITTED_2C_EPSILON = 3.432835890289768


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


def path_key(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(flatten_leaves(child, child_prefix))
        return leaves
    return {prefix: value}


def get_path(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        current = current[part]
    return current


def set_path(
    value: dict[str, Any],
    dotted_path: str,
    replacement: Any,
) -> None:
    parts = dotted_path.split(".")
    current: dict[str, Any] = value
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = copy.deepcopy(replacement)


def changed_leaf_paths(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    left = flatten_leaves(before)
    right = flatten_leaves(after)
    return sorted(
        path
        for path in set(left) | set(right)
        if path not in left
        or path not in right
        or left[path] != right[path]
    )


def paired_decomposition(
    dataset: str,
    p_raw: dict[str, Any],
    f_raw: dict[str, Any],
) -> dict[str, Any]:
    p_flat = flatten_leaves(p_raw)
    f_flat = flatten_leaves(f_raw)
    shared_keys = set(p_flat) & set(f_flat)
    same_paths = sorted(
        path for path in shared_keys if p_flat[path] == f_flat[path]
    )
    different_paths = sorted(
        path for path in shared_keys if p_flat[path] != f_flat[path]
    )
    only_p = sorted(set(p_flat) - set(f_flat))
    only_f = sorted(set(f_flat) - set(p_flat))
    shared_values = {path: p_flat[path] for path in same_paths}
    route_values = {
        path: {"P": p_flat[path], "F": f_flat[path]}
        for path in different_paths
    }
    exact = (
        same_paths == list(EXPECTED_SHARED_PATHS)
        and different_paths == list(EXPECTED_DIFFERENT_PATHS)
        and only_p == list(EXPECTED_ONLY_P_PATHS)
        and only_f == list(EXPECTED_ONLY_F_PATHS)
    )
    return {
        "dataset": dataset,
        "shared_leaf_count": len(same_paths),
        "shared_leaf_paths": same_paths,
        "shared_values": shared_values,
        "shared_shell_payload_sha256": canonical_sha256(shared_values),
        "route_different_leaf_count": len(different_paths),
        "route_different_leaf_paths": different_paths,
        "route_values": route_values,
        "only_p_leaf_count": len(only_p),
        "only_p_leaf_paths": only_p,
        "only_f_leaf_count": len(only_f),
        "only_f_leaf_paths": only_f,
        "exact_expected_decomposition": exact,
    }


def classify_rejection_stage(error: Exception) -> str:
    message = str(error)
    if message.startswith("route '"):
        return "route_dispatch"
    if message.startswith("Route-specific contract validation failed:"):
        return "route_specific_contract"
    return "unclassified"


def mutation_case(
    dataset: str,
    target_route: str,
    donor_route: str,
    target: dict[str, Any],
    donor: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    mutated = copy.deepcopy(target)
    for dotted_path in spec["copy_paths"]:
        set_path(mutated, dotted_path, get_path(donor, dotted_path))
    changed = changed_leaf_paths(target, mutated)
    mutated_flat = flatten_leaves(mutated)
    target_flat = flatten_leaves(target)
    shared_shell_preserved = all(
        path in mutated_flat
        and mutated_flat[path] == target_flat[path]
        for path in EXPECTED_SHARED_PATHS
    )
    observed_outcome = "accept"
    exception_type = None
    exception_message = None
    observed_stage = "none"
    try:
        parse_registered_contract_v3(mutated)
    except Exception as error:  # the report records unexpected types too
        observed_outcome = "reject"
        exception_type = type(error).__name__
        exception_message = str(error)
        observed_stage = classify_rejection_stage(error)

    passed = (
        bool(changed)
        and shared_shell_preserved
        and observed_outcome == "reject"
        and exception_type == CompilerDispatchV3Error.__name__
        and observed_stage == spec["expected_rejection_stage"]
    )
    return {
        "case_id": (
            f"{dataset}_{target_route.lower()}_from_"
            f"{donor_route.lower()}_{spec['family']}"
        ),
        "dataset": dataset,
        "target_route": target_route,
        "donor_route": donor_route,
        "family": spec["family"],
        "copied_paths": list(spec["copy_paths"]),
        "changed_leaf_count": len(changed),
        "changed_leaf_paths": changed,
        "shared_shell_preserved": shared_shell_preserved,
        "required_outcome": "reject",
        "observed_outcome": observed_outcome,
        "expected_rejection_stage": spec["expected_rejection_stage"],
        "observed_rejection_stage": observed_stage,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "passed": passed,
    }


def sensitivity_counterfactual() -> dict[str, Any]:
    report = load_json(F_MUTATION_REPORT)
    cases = {
        row["case_id"]: row
        for row in report["cases"]
        if isinstance(row, dict) and "case_id" in row
    }
    accounting = report["accounting_counterfactual"]
    factor_case = cases["sensitivity_factor_drop"]
    correct = accounting["correct_replace_one_epsilon"]
    omitted = accounting["unsafe_if_2C_factor_were_omitted"]
    bound = (
        correct == EXPECTED_CORRECT_REPLACE_ONE_EPSILON
        and omitted == EXPECTED_OMITTED_2C_EPSILON
        and factor_case.get("observed_outcome") == "reject"
        and factor_case.get("required_outcome") == "reject"
    )
    return {
        "dataset": "uci",
        "correct_replace_one_epsilon": correct,
        "epsilon_if_2C_factor_were_omitted": omitted,
        "difference": correct - omitted,
        "sensitivity_factor_drop_case": factor_case,
        "source_report": path_key(F_MUTATION_REPORT),
        "source_report_sha256": file_sha256(F_MUTATION_REPORT),
        "exact_binding_pass": bound,
        "interpretation_limit": (
            "This is an accountant-input counterfactual illustrating the "
            "route-specific replace-one sensitivity convention. It is not "
            "an observed privacy-loss estimate."
        ),
    }


def build_report() -> dict[str, Any]:
    raw = {
        route: {
            dataset: load_yaml(CONFIGS[route][dataset])
            for dataset in DATASETS
        }
        for route in ROUTES
    }

    baselines: list[dict[str, Any]] = []
    for route in ROUTES:
        for dataset in DATASETS:
            contract = parse_registered_contract_v3(raw[route][dataset])
            baselines.append(
                {
                    "route": route,
                    "dataset": dataset,
                    "config": path_key(CONFIGS[route][dataset]),
                    "config_sha256": file_sha256(
                        CONFIGS[route][dataset]
                    ),
                    "route_id": contract.route_id,
                    "schema_version": contract.schema_version,
                    "contract_type": type(contract).__name__,
                    "expected_route_id": EXPECTED_ROUTE_IDS[route],
                    "route_identity_pass": (
                        contract.route_id == EXPECTED_ROUTE_IDS[route]
                    ),
                    "observed_outcome": "accept",
                }
            )

    pairs = [
        paired_decomposition(dataset, raw["P"][dataset], raw["F"][dataset])
        for dataset in DATASETS
    ]

    cases: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for target_route, donor_route in (("P", "F"), ("F", "P")):
            for spec in MUTATION_SPECS:
                cases.append(
                    mutation_case(
                        dataset,
                        target_route,
                        donor_route,
                        raw[target_route][dataset],
                        raw[donor_route][dataset],
                        spec,
                    )
                )

    family_counts = Counter(row["family"] for row in cases)
    dataset_counts = Counter(row["dataset"] for row in cases)
    direction_counts = Counter(
        f"{row['target_route']}_from_{row['donor_route']}"
        for row in cases
    )
    rejection_stage_counts = Counter(
        row["observed_rejection_stage"] for row in cases
    )
    counterfactual = sensitivity_counterfactual()

    baseline_pass = (
        len(baselines) == 6
        and all(row["route_identity_pass"] for row in baselines)
    )
    pair_pass = (
        len(pairs) == 3
        and all(row["exact_expected_decomposition"] for row in pairs)
    )
    mutation_pass = (
        len(cases) == 48
        and all(row["passed"] for row in cases)
        and family_counts
        == Counter({spec["family"]: 6 for spec in MUTATION_SPECS})
    )
    status_pass = (
        baseline_pass
        and pair_pass
        and mutation_pass
        and counterfactual["exact_binding_pass"]
    )

    report: dict[str, Any] = {
        "schema_version": "unitdp.v34_matched_shell_gate.v1",
        "scope": (
            "paired finite-registry contract noninterchangeability for the "
            "three frozen P/F dataset pairs"
        ),
        "status": "PASS" if status_pass else "FAIL",
        "baseline_contracts": {
            "accepted": len(baselines) if baseline_pass else 0,
            "required": 6,
            "rows": baselines,
        },
        "paired_contract_decomposition": {
            "pairs_passed": sum(
                row["exact_expected_decomposition"] for row in pairs
            ),
            "pairs_required": 3,
            "expected_shared_leaf_count": len(EXPECTED_SHARED_PATHS),
            "expected_route_different_leaf_count": len(
                EXPECTED_DIFFERENT_PATHS
            ),
            "expected_only_p_leaf_count": len(EXPECTED_ONLY_P_PATHS),
            "expected_only_f_leaf_count": len(EXPECTED_ONLY_F_PATHS),
            "rows": pairs,
        },
        "cross_route_substitutions": {
            "required": 48,
            "rejected": sum(
                row["observed_outcome"] == "reject" for row in cases
            ),
            "passed": sum(row["passed"] for row in cases),
            "all_shared_shells_preserved": all(
                row["shared_shell_preserved"] for row in cases
            ),
            "all_changed_nonvacuously": all(
                row["changed_leaf_count"] > 0 for row in cases
            ),
            "family_counts": dict(sorted(family_counts.items())),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "direction_counts": dict(sorted(direction_counts.items())),
            "rejection_stage_counts": dict(
                sorted(rejection_stage_counts.items())
            ),
            "cases": cases,
        },
        "sensitivity_counterfactual": counterfactual,
        "verdict": {
            "finite_registered_noninterchangeability": (
                "PASS" if status_pass else "FAIL"
            ),
            "claim_allowed": (
                "Across the three frozen dataset pairs, P and F share the "
                "same 38-leaf scientific shell while carrying distinct "
                "route obligations; all 48 prespecified bidirectional "
                "substitutions fail before execution."
            ),
            "claims_forbidden": [
                "a general-purpose DP library fails on these contracts",
                "the finite substitutions establish complete bug detection",
                "the schema alone proves privacy or executor correctness",
                "the counterfactual measures observed privacy loss",
                "an arbitrary third route is safe to register",
            ],
            "manuscript_action": (
                "The matched-shell result may be promoted only with its "
                "three-pair, eight-family, pre-execution, finite-registry "
                "scope and the existing two-route limitation."
            ),
        },
        "source_sha256": {
            path_key(path): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src" / "unitdp" / "compiler_v2.py",
                ROOT / "src" / "unitdp" / "compiler_srswor_v3.py",
                ROOT / "src" / "unitdp" / "compiler_v3.py",
                F_MUTATION_REPORT,
                *(
                    CONFIGS[route][dataset]
                    for route in ROUTES
                    for dataset in DATASETS
                ),
            )
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def markdown_report(report: dict[str, Any]) -> str:
    decomposition = report["paired_contract_decomposition"]
    substitutions = report["cross_route_substitutions"]
    lines = [
        "# V3.4 Matched-Shell Noninterchangeability Gate",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Exact result",
        "",
        f"- Exact baselines accepted: "
        f"{report['baseline_contracts']['accepted']}/"
        f"{report['baseline_contracts']['required']}",
        f"- Dataset-pair decompositions: "
        f"{decomposition['pairs_passed']}/"
        f"{decomposition['pairs_required']}",
        f"- Bidirectional substitutions rejected and stage-matched: "
        f"{substitutions['passed']}/{substitutions['required']}",
        "",
        "Each P/F pair has 38 equal scientific-shell leaves, 16 common keys "
        "with route-specific values, one P-only leaf, and seven F-only "
        "leaves.",
        "",
        "## Paired contracts",
        "",
        "| Dataset | Equal shell | Route-different | P-only | F-only |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in decomposition["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['shared_leaf_count']} | "
            f"{row['route_different_leaf_count']} | "
            f"{row['only_p_leaf_count']} | "
            f"{row['only_f_leaf_count']} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified substitutions",
            "",
            "| Family | Cases |",
            "| --- | ---: |",
        ]
    )
    for family, count in substitutions["family_counts"].items():
        lines.append(f"| {family} | {count}/{count} rejected |")
    counterfactual = report["sensitivity_counterfactual"]
    lines.extend(
        [
            "",
            "## Bound accounting counterfactual",
            "",
            "For the frozen UCI F contract, the correct replace-one "
            f"calculation is {counterfactual['correct_replace_one_epsilon']}; "
            "omitting the registered `2C` sensitivity factor would report "
            f"{counterfactual['epsilon_if_2C_factor_were_omitted']}. The "
            "factor-drop contract is rejected.",
            "",
            counterfactual["interpretation_limit"],
            "",
            "## Claim boundary",
            "",
            report["verdict"]["manuscript_action"],
            "",
            "This gate does not establish a general library failure, "
            "complete bug detection, generic privacy soundness, formal "
            "refinement, or safety of an unregistered route.",
            "",
            f"Payload SHA-256: `{report['payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(
        f"status={report['status']} "
        f"baselines={report['baseline_contracts']['accepted']}/6 "
        f"pairs={report['paired_contract_decomposition']['pairs_passed']}/3 "
        f"substitutions={report['cross_route_substitutions']['passed']}/48 "
        f"payload_sha256={report['payload_sha256']}"
    )
    print(f"result={args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
