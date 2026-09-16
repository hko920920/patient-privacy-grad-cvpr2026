"""Audit utility-comparator candidates against the strict V3.1 boundary.

This audit prevents legacy utility rows from being reused merely because they
target the same numerical owner epsilon.  A current comparator must also bind
adjacency, sampling law, schedule, exact public data, execution source, and
artifacts.  The output contains only public aggregate metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.accountant_registry import get_accountant_route  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v32_comparator_feasibility_20260724"
    / "public_utility_comparator_feasibility_v32.json"
)
MECHANISM_REFERENCE_REPORT = (
    ROOT
    / "reports"
    / "v3_mechanism_matched_reference_v32_20260724"
    / "public_mechanism_matched_reference_v32.json"
)
LEGACY_POISSON_MOG = (
    ROOT
    / "reports"
    / "mog_owner_baseline_uci_wisdm_5seed_001"
    / "aggregate.json"
)
LEGACY_FIXED_MOG = (
    ROOT
    / "reports"
    / "fixed_size_mog_owner_baseline_uci_wisdm_5seed_001"
    / "aggregate.json"
)
LEGACY_GROUP = (
    ROOT
    / "reports"
    / "conservative_els_owner_5seed_001"
    / "aggregate.json"
)
WINDOW_EXECUTOR = ROOT / "src" / "unitdp" / "window_dpsgd.py"
ACCOUNTANT_REGISTRY = ROOT / "src" / "unitdp" / "accountant_registry.py"
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_DRIVE_PREFIX = "C:" + chr(92)


class ComparatorAuditError(RuntimeError):
    """Raised when the comparator evidence surface is internally inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparatorAuditError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ComparatorAuditError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(value, dict):
        raise ComparatorAuditError(f"Expected YAML object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _registered_profiles() -> dict[str, dict[str, dict[str, Any]]]:
    profiles: dict[str, dict[str, dict[str, Any]]] = {
        "route_p": {},
        "route_f": {},
    }
    for dataset in ("uci", "wisdm", "sepsis"):
        poisson = load_yaml(
            ROOT / "configs" / "v2" / f"{dataset}_owner_poisson_v2.yaml"
        )
        fixed = load_yaml(
            ROOT / "configs" / "v3" / f"{dataset}_owner_srswor_v3.yaml"
        )
        profiles["route_p"][dataset] = {
            "adjacency": poisson["privacy"]["adjacency"],
            "sampler": poisson["mechanism"]["sampler"],
            "sensitivity_multiplier": 1.0,
            "steps": int(poisson["mechanism"]["total_steps"]),
            "update_denominator": float(
                poisson["mechanism"]["update_denominator"]
            ),
            "windows_per_owner_per_step": int(
                poisson["contribution_policy"][
                    "windows_per_owner_per_step"
                ]
            ),
            "train_owners": int(
                poisson["data"]["public_protocol"]["train_owners"]
            ),
            "full_data_conformance_sha256": poisson["data"][
                "public_protocol"
            ]["full_data_conformance_sha256"],
        }
        profiles["route_f"][dataset] = {
            "adjacency": fixed["privacy"]["adjacency"],
            "sampler": fixed["mechanism"]["sampler"],
            "sensitivity_multiplier": float(
                fixed["accountant"]["sensitivity_multiplier"]
            ),
            "steps": int(fixed["mechanism"]["total_steps"]),
            "update_denominator": float(
                fixed["mechanism"]["update_denominator"]
            ),
            "windows_per_owner_per_step": int(
                fixed["contribution_policy"][
                    "windows_per_owner_per_step"
                ]
            ),
            "train_owners": int(
                fixed["data"]["public_protocol"]["train_owners"]
            ),
            "full_data_conformance_sha256": fixed["data"][
                "public_protocol"
            ]["full_data_conformance_sha256"],
        }
    return profiles


def _legacy_schedule(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    return {
        str(row["dataset"]): {
            "runs": int(row["runs"]),
            "steps": int(row["steps"]),
            "selected_windows": int(row["selected_windows"]),
        }
        for row in rows
    }


def _group_snapshot(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        str(row["dataset"]): {
            "runs": int(row["runs"]),
            "selected_windows": int(row["selected_windows"]),
            "selected_owners": int(row["selected_owners"]),
        }
        for row in rows
    }


def _window_executor_findings() -> dict[str, bool]:
    text = WINDOW_EXECUTOR.read_text(encoding="utf-8")
    findings = {
        "epoch_driven_loop": "for _epoch in range(config.epochs)" in text,
        "empty_batch_skipped": (
            "if xb.shape[0] == 0:" in text
            and "continue" in text.split(
                "if xb.shape[0] == 0:", maxsplit=1
            )[1][:80]
        ),
        "shuffled_loader": "shuffle=True" in text,
        "fixed_public_total_steps_argument": (
            "total_steps" in text
            or "fixed_total_steps" in text
        ),
    }
    if findings != {
        "epoch_driven_loop": True,
        "empty_batch_skipped": True,
        "shuffled_loader": True,
        "fixed_public_total_steps_argument": False,
    }:
        raise ComparatorAuditError(
            f"Legacy window executor findings changed: {findings}"
        )
    return findings


def _candidate(
    *,
    candidate_id: str,
    purpose: str,
    same_adjacency: bool | None,
    same_privacy_target: bool,
    same_mechanism: bool,
    exact_current_data_binding: bool,
    fixed_current_schedule: bool,
    execution_source_and_artifact_chain: bool,
    all_three_datasets: bool,
    prior_method_comparator: bool,
    verdict: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "purpose": purpose,
        "criteria": {
            "same_adjacency_explicitly_bound": same_adjacency,
            "same_owner_privacy_target": same_privacy_target,
            "same_dp_mechanism": same_mechanism,
            "exact_current_data_binding": exact_current_data_binding,
            "fixed_current_schedule": fixed_current_schedule,
            "execution_source_and_artifact_chain": (
                execution_source_and_artifact_chain
            ),
            "all_three_datasets": all_three_datasets,
            "prior_method_comparator": prior_method_comparator,
        },
        "verdict": verdict,
        "reasons": reasons,
    }


def _assert_public_report(report: dict[str, Any]) -> None:
    encoded = json.dumps(report, sort_keys=True, ensure_ascii=False)
    for token in (
        "research_seed",
        "seed_13",
        "seed_23",
        "seed_31",
        "seed_37",
        "seed_41",
        LOCAL_IDENTITY_TOKEN,
        WINDOWS_DRIVE_PREFIX,
    ):
        if token in encoded:
            raise ComparatorAuditError(
                f"Public comparator report exposes forbidden token: {token}"
            )
    if report.get("visibility") != "public":
        raise ComparatorAuditError("Comparator report must be public")


def build_report() -> dict[str, Any]:
    profiles = _registered_profiles()
    matched = load_json(MECHANISM_REFERENCE_REPORT)
    if matched["totals"] != {
        "datasets": 6,
        "diagnostic_steps_exact": 850,
        "final_models_bitwise": 30,
        "final_tensors_bitwise": 60,
        "maximum_public_metric_gap": 0.0,
        "public_metrics_exact": 120,
        "runs": 30,
    }:
        raise ComparatorAuditError(
            "Mechanism-matched report no longer has the frozen exact totals"
        )

    poisson_status = get_accountant_route("mog_pld_els_owner").status
    fixed_status = get_accountant_route(
        "fixed_size_mog_pld_els_owner"
    ).status
    group_status = get_accountant_route(
        "window_rdp_group_fallback"
    ).status
    if (
        poisson_status != "legacy_not_execution_bound"
        or fixed_status != "legacy_not_execution_bound"
        or group_status != "implemented_conservative"
    ):
        raise ComparatorAuditError("Comparator route statuses changed")

    poisson_rows = load_json(LEGACY_POISSON_MOG)
    fixed_rows = load_json(LEGACY_FIXED_MOG)
    group_rows = load_json(LEGACY_GROUP)
    if not all(
        isinstance(rows, list)
        for rows in (poisson_rows, fixed_rows, group_rows)
    ):
        raise ComparatorAuditError("Legacy aggregate is not a JSON array")
    poisson_schedule = _legacy_schedule(poisson_rows)
    fixed_schedule = _legacy_schedule(fixed_rows)
    group_snapshot = _group_snapshot(group_rows)
    if set(poisson_schedule) != {"uci", "wisdm"}:
        raise ComparatorAuditError("Legacy Poisson MoG coverage changed")
    if set(fixed_schedule) != {"uci", "wisdm"}:
        raise ComparatorAuditError("Legacy fixed MoG coverage changed")
    if set(group_snapshot) != {"uci", "wisdm", "sepsis"}:
        raise ComparatorAuditError("Legacy group coverage changed")
    if int(group_snapshot["sepsis"]["selected_owners"]) == int(
        profiles["route_p"]["sepsis"]["train_owners"]
    ):
        raise ComparatorAuditError(
            "Legacy group Sepsis owner count unexpectedly matches V3.1"
        )

    current_steps_p = {
        dataset: int(row["steps"])
        for dataset, row in profiles["route_p"].items()
    }
    current_steps_f = {
        dataset: int(row["steps"])
        for dataset, row in profiles["route_f"].items()
    }
    legacy_poisson_steps = {
        dataset: int(row["steps"])
        for dataset, row in poisson_schedule.items()
    }
    legacy_fixed_steps = {
        dataset: int(row["steps"])
        for dataset, row in fixed_schedule.items()
    }
    if legacy_poisson_steps == {
        key: current_steps_p[key] for key in legacy_poisson_steps
    }:
        raise ComparatorAuditError(
            "Legacy Poisson schedule unexpectedly matches V3.1"
        )
    if legacy_fixed_steps == {
        key: current_steps_f[key] for key in legacy_fixed_steps
    }:
        raise ComparatorAuditError(
            "Legacy fixed schedule unexpectedly matches V3.1"
        )

    candidates = [
        _candidate(
            candidate_id="same_route_clean_room_replay",
            purpose="execution correspondence",
            same_adjacency=True,
            same_privacy_target=True,
            same_mechanism=True,
            exact_current_data_binding=True,
            fixed_current_schedule=True,
            execution_source_and_artifact_chain=True,
            all_three_datasets=True,
            prior_method_comparator=False,
            verdict="eligible_execution_correspondence_only",
            reasons=[
                "covers all 30 registered runs",
                "does not compare utility against a different algorithm",
            ],
        ),
        _candidate(
            candidate_id="route_p_versus_route_f",
            purpose="sampling-scheme utility ranking",
            same_adjacency=False,
            same_privacy_target=True,
            same_mechanism=False,
            exact_current_data_binding=True,
            fixed_current_schedule=True,
            execution_source_and_artifact_chain=True,
            all_three_datasets=True,
            prior_method_comparator=False,
            verdict="ineligible_superiority_comparison",
            reasons=[
                "add/remove and replace-one adjacency differ",
                "sensitivity C and 2C calibration differ",
            ],
        ),
        _candidate(
            candidate_id="legacy_poisson_mog_pld_els",
            purpose="owner-private window-sampling utility comparator",
            same_adjacency=None,
            same_privacy_target=True,
            same_mechanism=False,
            exact_current_data_binding=False,
            fixed_current_schedule=False,
            execution_source_and_artifact_chain=False,
            all_three_datasets=False,
            prior_method_comparator=True,
            verdict="ineligible_current_evidence_requires_new_route",
            reasons=[
                f"registry status is {poisson_status}",
                "legacy steps differ from the registered Route-P schedule",
                "legacy epoch executor skips empty batches",
                "no current full-data, source-bundle, or artifact-chain binding",
            ],
        ),
        _candidate(
            candidate_id="legacy_fixed_size_mog_pld_els",
            purpose="owner-private fixed-window utility comparator",
            same_adjacency=None,
            same_privacy_target=True,
            same_mechanism=False,
            exact_current_data_binding=False,
            fixed_current_schedule=False,
            execution_source_and_artifact_chain=False,
            all_three_datasets=False,
            prior_method_comparator=True,
            verdict="ineligible_current_evidence_requires_new_route",
            reasons=[
                f"registry status is {fixed_status}",
                "legacy steps differ from the registered Route-F schedule",
                "epoch-shuffled batches are not the current owner SRSWOR route",
                "no current full-data, source-bundle, or artifact-chain binding",
            ],
        ),
        _candidate(
            candidate_id="legacy_group_privacy_window_fallback",
            purpose="conservative owner-private window baseline",
            same_adjacency=None,
            same_privacy_target=True,
            same_mechanism=False,
            exact_current_data_binding=False,
            fixed_current_schedule=False,
            execution_source_and_artifact_chain=False,
            all_three_datasets=True,
            prior_method_comparator=True,
            verdict="valid_accounting_idea_but_ineligible_legacy_experiment",
            reasons=[
                f"registry status is {group_status}",
                "legacy experiment uses the epoch executor",
                "legacy Sepsis owner count differs from the current public profile",
                "no current source or artifact-chain binding",
            ],
        ),
        _candidate(
            candidate_id="same_route_single_window_policy_ablation",
            purpose="within-owner policy ablation",
            same_adjacency=True,
            same_privacy_target=True,
            same_mechanism=True,
            exact_current_data_binding=True,
            fixed_current_schedule=True,
            execution_source_and_artifact_chain=False,
            all_three_datasets=True,
            prior_method_comparator=False,
            verdict="feasible_new_ablation_not_prior_method_baseline",
            reasons=[
                "privacy mechanism remains valid after owner clipping",
                "requires new registered profiles and evidence chains",
                "tests a policy choice rather than a competing published method",
            ],
        ),
        _candidate(
            candidate_id="new_strict_els_route",
            purpose="owner-private window-sampling prior-method comparator",
            same_adjacency=True,
            same_privacy_target=True,
            same_mechanism=False,
            exact_current_data_binding=True,
            fixed_current_schedule=True,
            execution_source_and_artifact_chain=False,
            all_three_datasets=True,
            prior_method_comparator=True,
            verdict="scientifically_possible_only_after_full_new_route",
            reasons=[
                "requires an explicit adjacency and contribution model",
                "requires a sampler-matched independently checked accountant",
                "requires a new executor, mutation surface, artifacts, and reruns",
            ],
        ),
    ]

    source_files = (
        Path(__file__).resolve(),
        ACCOUNTANT_REGISTRY,
        WINDOW_EXECUTOR,
        MECHANISM_REFERENCE_REPORT,
        LEGACY_POISSON_MOG,
        LEGACY_FIXED_MOG,
        LEGACY_GROUP,
        *(
            ROOT
            / "configs"
            / version
            / f"{dataset}_{suffix}.yaml"
            for version, suffix in (
                ("v2", "owner_poisson_v2"),
                ("v3", "owner_srswor_v3"),
            )
            for dataset in ("uci", "wisdm", "sepsis")
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "unitdp.utility_comparator_feasibility_public.v31",
        "visibility": "public",
        "scope": (
            "eligibility audit for current manuscript evidence; "
            "not a utility experiment"
        ),
        "eligibility_rule": [
            "do not rank routes with different adjacency or sensitivity",
            "do not reuse legacy rows without exact current data and schedule",
            "require sampler/accountant/executor/source/artifact binding",
            "distinguish execution correspondence from utility superiority",
        ],
        "registered_profiles": profiles,
        "legacy_observations": {
            "window_executor": _window_executor_findings(),
            "poisson_mog_registry_status": poisson_status,
            "fixed_mog_registry_status": fixed_status,
            "group_fallback_registry_status": group_status,
            "route_p_steps": current_steps_p,
            "route_f_steps": current_steps_f,
            "legacy_poisson_mog_steps": legacy_poisson_steps,
            "legacy_fixed_mog_steps": legacy_fixed_steps,
            "legacy_group_snapshot": group_snapshot,
        },
        "candidates": candidates,
        "decision": {
            "reuse_legacy_utility_rows": False,
            "rank_route_p_against_route_f": False,
            "current_mechanism_matched_comparator_role": (
                "execution correspondence only"
            ),
            "new_utility_experiment_authorized_by_this_audit": False,
            "manuscript_action": (
                "retain V3.1 results and explicit non-ranking limitation"
            ),
        },
        "source_files_sha256": {
            _relative(path): file_sha256(path) for path in source_files
        },
    }
    _assert_public_report(report)
    report["payload_sha256"] = payload_sha256(report)
    _assert_public_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "utility comparator eligibility audit: "
        f"{len(report['candidates'])} candidates"
    )
    print(f"payload_sha256={report['payload_sha256']}")


if __name__ == "__main__":
    main()
