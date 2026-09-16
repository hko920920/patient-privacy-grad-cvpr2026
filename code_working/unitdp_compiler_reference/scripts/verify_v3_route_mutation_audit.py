"""Verify and partially replay the public V3 route-mutation audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_accounting import dp_event  # noqa: E402
from dp_accounting.privacy_accountant import (  # noqa: E402
    NeighboringRelation,
)
from dp_accounting.rdp import RdpAccountant  # noqa: E402
from unitdp.compiler_srswor_v3 import (  # noqa: E402
    parse_owner_srswor_contract_v3,
)
from unitdp.compiler_v3 import parse_registered_contract_v3  # noqa: E402
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (  # noqa: E402
    fixed_size_srswor_epsilon_v3,
)


CONFIG = ROOT / "configs" / "v3" / "uci_owner_srswor_v3.yaml"
EXPECTED_CASE_IDS = {
    "adjacency_swap",
    "sampler_swap",
    "population_change",
    "sample_size_change",
    "step_count_change",
    "noise_change",
    "denominator_change",
    "empty_behavior_change",
    "sensitivity_factor_drop",
    "poisson_accountant_reuse",
    "theorem_swap",
    "executor_swap",
    "random_coins_public",
    "privacy_target_change",
    "profile_hash_change",
    "base_profile_swap",
    "within_owner_budget_change",
    "learning_rate_change",
    "unknown_top_level_field",
    "missing_accountant_section",
    "route_id_splice",
    "schema_splice",
    "unknown_route",
    "full_mapping_byte_change",
    "preprocessor_state_change",
    "poisson_compiled_object_to_srswor_executor",
    "sensitivity_one_direct_oracle",
}


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


def set_nested(
    raw: dict[str, Any],
    path: tuple[str, ...],
    value: object,
) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    cursor = result
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value
    return result


def must_reject(action: Callable[[], object], name: str) -> None:
    try:
        action()
    except Exception:
        return
    raise AssertionError(f"Critical replay unexpectedly accepted: {name}")


def verify(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert_public_certificate_redacted(report)
    without_hash = dict(report)
    reported_hash = without_hash.pop("payload_sha256")
    if payload_sha256(without_hash) != reported_hash:
        raise AssertionError("Report payload hash mismatch")
    cases = report["cases"]
    case_ids = {case["case_id"] for case in cases}
    if case_ids != EXPECTED_CASE_IDS:
        raise AssertionError("Mutation case set differs from the gate")
    if (
        report["case_count"] != len(EXPECTED_CASE_IDS)
        or report["required_rejections"] != len(EXPECTED_CASE_IDS)
        or report["observed_rejections"] != len(EXPECTED_CASE_IDS)
        or report["all_required_rejections_observed"] is not True
    ):
        raise AssertionError("Mutation rejection totals are inconsistent")
    if any(
        case["required_outcome"] != "reject"
        or case["observed_outcome"] != "reject"
        for case in cases
    ):
        raise AssertionError("At least one case did not fail closed")
    for relative, expected in report["source_files_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise AssertionError(f"Source hash mismatch: {relative}")

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    must_reject(
        lambda: parse_owner_srswor_contract_v3(
            set_nested(
                raw,
                ("privacy", "adjacency"),
                "add_remove_one_owner",
            )
        ),
        "adjacency_swap",
    )
    must_reject(
        lambda: parse_owner_srswor_contract_v3(
            set_nested(
                raw,
                ("mechanism", "sampler"),
                "independent_bernoulli_owner",
            )
        ),
        "sampler_swap",
    )
    must_reject(
        lambda: parse_owner_srswor_contract_v3(
            set_nested(
                raw,
                ("accountant", "sensitivity_multiplier"),
                1.0,
            )
        ),
        "sensitivity_factor_drop",
    )
    must_reject(
        lambda: parse_registered_contract_v3(
            {
                **raw,
                "route_id": "owa_owner_poisson_rdp_add_remove_v2",
            }
        ),
        "route_id_splice",
    )
    must_reject(
        lambda: fixed_size_srswor_epsilon_v3(
            actual_noise_multiplier=3.806632095748225,
            source_dataset_size=21,
            sample_size=8,
            steps=15,
            delta=1e-5,
            sensitivity_multiplier=1.0,
        ),
        "sensitivity_one_direct_oracle",
    )

    orders = (2, 3, 4, 5, 8, 16)
    noise = 3.806632095748225
    recomputed = {}
    for name, normalized in (
        ("correct_replace_one_epsilon", noise / 2.0),
        ("unsafe_if_2C_factor_were_omitted", noise),
    ):
        accountant = RdpAccountant(
            orders=orders,
            neighboring_relation=NeighboringRelation.REPLACE_ONE,
        )
        accountant.compose(
            dp_event.SampledWithoutReplacementDpEvent(
                source_dataset_size=21,
                sample_size=8,
                event=dp_event.GaussianDpEvent(normalized),
            ),
            count=15,
        )
        recomputed[name] = float(accountant.get_epsilon(1e-5))
    for key, value in recomputed.items():
        if abs(report["accounting_counterfactual"][key] - value) > 1e-12:
            raise AssertionError(
                f"Accounting counterfactual mismatch: {key}"
            )
    if not (
        recomputed["unsafe_if_2C_factor_were_omitted"]
        < recomputed["correct_replace_one_epsilon"]
    ):
        raise AssertionError("Sensitivity omission did not under-report")
    return {
        "verified": True,
        "case_count": len(EXPECTED_CASE_IDS),
        "critical_replays": 5,
        "payload_sha256": reported_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(
            ROOT
            / "reports"
            / "v3_route_mutation_audit_v32_20260724"
            / "public_route_mutation_audit_v3.json"
        ),
    )
    args = parser.parse_args()
    result = verify(Path(args.report))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
