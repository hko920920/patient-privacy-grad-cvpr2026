"""Build public evidence that cross-route semantic mutations fail closed."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import tempfile
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
    compile_owner_srswor_contract_v3,
    load_owner_srswor_contract_v3,
    parse_owner_srswor_contract_v3,
)
from unitdp.compiler_v2 import (  # noqa: E402
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
)
from unitdp.compiler_v3 import (  # noqa: E402
    parse_registered_contract_v3,
)
from unitdp.owner_srswor_v3 import (  # noqa: E402
    train_owner_srswor_fixed_v3,
)
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (  # noqa: E402
    fixed_size_srswor_epsilon_v3,
)


CONFIG = ROOT / "configs" / "v3" / "uci_owner_srswor_v3.yaml"
POISSON_CONFIG = (
    ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
)
MAPPING = (
    ROOT
    / "reports"
    / "uci_public_preprocessing_smoke_20260724"
    / "uci_train_mapping.csv"
)
PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)


CONTRACT_MUTATIONS = (
    ("adjacency_swap", ("privacy", "adjacency"), "add_remove_one_owner"),
    (
        "sampler_swap",
        ("mechanism", "sampler"),
        "independent_bernoulli_owner",
    ),
    ("population_change", ("mechanism", "source_dataset_size"), 20),
    ("sample_size_change", ("mechanism", "sample_size"), 7),
    ("step_count_change", ("mechanism", "total_steps"), 14),
    ("noise_change", ("mechanism", "noise_multiplier"), 1.26970999503374),
    ("denominator_change", ("mechanism", "update_denominator"), 7.0),
    ("empty_behavior_change", ("mechanism", "empty_step_behavior"), "skip"),
    (
        "sensitivity_factor_drop",
        ("accountant", "sensitivity_multiplier"),
        1.0,
    ),
    (
        "poisson_accountant_reuse",
        ("accountant", "id"),
        "poisson_gaussian_rdp_add_remove_dual_v1",
    ),
    (
        "theorem_swap",
        ("accountant", "theorem_id"),
        "poisson_sampled_gaussian",
    ),
    (
        "executor_swap",
        ("execution", "implementation_id"),
        "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2",
    ),
    (
        "random_coins_public",
        ("execution", "random_coins_public"),
        True,
    ),
    ("privacy_target_change", ("privacy", "target_epsilon"), 9.0),
    (
        "profile_hash_change",
        ("data", "benchmark_profile_sha256"),
        "0" * 64,
    ),
    (
        "base_profile_swap",
        ("data", "base_data_profile_id"),
        "wisdm_aaai27_v1",
    ),
    (
        "within_owner_budget_change",
        ("contribution_policy", "windows_per_owner_per_step"),
        7,
    ),
    ("learning_rate_change", ("optimization", "learning_rate"), 0.1),
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


def rejection_case(
    case_id: str,
    stage: str,
    action: Callable[[], object],
) -> dict[str, Any]:
    try:
        action()
    except Exception as exc:
        return {
            "case_id": case_id,
            "stage": stage,
            "required_outcome": "reject",
            "observed_outcome": "reject",
            "exception_type": type(exc).__name__,
        }
    return {
        "case_id": case_id,
        "stage": stage,
        "required_outcome": "reject",
        "observed_outcome": "accept",
        "exception_type": None,
    }


def build_cases() -> list[dict[str, Any]]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    cases = [
        rejection_case(
            case_id,
            "strict_contract",
            lambda path=path, value=value: parse_owner_srswor_contract_v3(
                set_nested(raw, path, value)
            ),
        )
        for case_id, path, value in CONTRACT_MUTATIONS
    ]
    unknown = copy.deepcopy(raw)
    unknown["unexpected"] = True
    cases.append(
        rejection_case(
            "unknown_top_level_field",
            "strict_contract",
            lambda: parse_owner_srswor_contract_v3(unknown),
        )
    )
    missing = copy.deepcopy(raw)
    missing.pop("accountant")
    cases.append(
        rejection_case(
            "missing_accountant_section",
            "strict_contract",
            lambda: parse_owner_srswor_contract_v3(missing),
        )
    )
    cases.extend(
        (
            rejection_case(
                "route_id_splice",
                "route_dispatch",
                lambda: parse_registered_contract_v3(
                    {
                        **raw,
                        "route_id": (
                            "owa_owner_poisson_rdp_add_remove_v2"
                        ),
                    }
                ),
            ),
            rejection_case(
                "schema_splice",
                "route_dispatch",
                lambda: parse_registered_contract_v3(
                    {
                        **raw,
                        "schema_version": (
                            "unitdp.owner_poisson_contract.v2"
                        ),
                    }
                ),
            ),
            rejection_case(
                "unknown_route",
                "route_dispatch",
                lambda: parse_registered_contract_v3(
                    {**raw, "route_id": "unregistered"}
                ),
            ),
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        with MAPPING.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
        rows[-1]["scenario"] += "_mutated"
        changed_mapping = temporary / "mapping.csv"
        with changed_mapping.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        contract = load_owner_srswor_contract_v3(CONFIG)
        cases.append(
            rejection_case(
                "full_mapping_byte_change",
                "compile_binding",
                lambda: compile_owner_srswor_contract_v3(
                    contract,
                    mapping_path=changed_mapping,
                    preprocessing_artifact_path=PREPROCESSOR,
                ),
            )
        )

        artifact = json.loads(
            PREPROCESSOR.read_text(encoding="utf-8")
        )
        artifact["mean"][0] = float(artifact["mean"][0]) + 0.25
        changed_preprocessor = temporary / "preprocessor.json"
        changed_preprocessor.write_text(
            json.dumps(artifact),
            encoding="utf-8",
        )
        cases.append(
            rejection_case(
                "preprocessor_state_change",
                "compile_binding",
                lambda: compile_owner_srswor_contract_v3(
                    contract,
                    mapping_path=MAPPING,
                    preprocessing_artifact_path=changed_preprocessor,
                ),
            )
        )

    poisson = compile_owner_poisson_contract_v2(
        load_owner_poisson_contract_v2(POISSON_CONFIG),
        mapping_path=MAPPING,
        preprocessing_artifact_path=PREPROCESSOR,
        require_executable=True,
    )
    cases.append(
        rejection_case(
            "poisson_compiled_object_to_srswor_executor",
            "executor_type",
            lambda: train_owner_srswor_fixed_v3(
                route=poisson,  # type: ignore[arg-type]
                raw_train_x=None,  # type: ignore[arg-type]
                train_y=None,  # type: ignore[arg-type]
                raw_test_x=None,  # type: ignore[arg-type]
                test_y=None,  # type: ignore[arg-type]
                research_seed=0,
            ),
        )
    )
    cases.append(
        rejection_case(
            "sensitivity_one_direct_oracle",
            "accountant_oracle",
            lambda: fixed_size_srswor_epsilon_v3(
                actual_noise_multiplier=3.806632095748225,
                source_dataset_size=21,
                sample_size=8,
                steps=15,
                delta=1e-5,
                sensitivity_multiplier=1.0,
            ),
        )
    )
    return cases


def accounting_counterfactual() -> dict[str, float]:
    noise = 3.806632095748225
    orders = (2, 3, 4, 5, 8, 16)
    correct = RdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.REPLACE_ONE,
    )
    correct.compose(
        dp_event.SampledWithoutReplacementDpEvent(
            source_dataset_size=21,
            sample_size=8,
            event=dp_event.GaussianDpEvent(noise / 2.0),
        ),
        count=15,
    )
    unsafe = RdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.REPLACE_ONE,
    )
    unsafe.compose(
        dp_event.SampledWithoutReplacementDpEvent(
            source_dataset_size=21,
            sample_size=8,
            event=dp_event.GaussianDpEvent(noise),
        ),
        count=15,
    )
    return {
        "correct_replace_one_epsilon": float(correct.get_epsilon(1e-5)),
        "unsafe_if_2C_factor_were_omitted": float(
            unsafe.get_epsilon(1e-5)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "reports"
            / "v3_route_mutation_audit_v32_20260724"
            / "public_route_mutation_audit_v3.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = build_cases()
    failed = [
        case
        for case in cases
        if case["observed_outcome"] != case["required_outcome"]
    ]
    source_paths = (
        ROOT / "scripts" / "build_v3_route_mutation_audit.py",
        ROOT / "scripts" / "verify_v3_route_mutation_audit.py",
        ROOT / "src" / "unitdp" / "compiler_v3.py",
        ROOT / "src" / "unitdp" / "compiler_srswor_v3.py",
        ROOT / "src" / "unitdp" / "owner_srswor_v3.py",
        ROOT
        / "src"
        / "unitdp_spec_oracle"
        / "srswor_rdp_oracle_v3.py",
    )
    report: dict[str, Any] = {
        "schema_version": "unitdp.route_mutation_audit_public.v3",
        "visibility": "public",
        "route_count": 2,
        "route_ids": [
            "owa_owner_poisson_rdp_add_remove_v2",
            "owa_owner_srswor_rdp_replace_one_v3",
        ],
        "case_count": len(cases),
        "required_rejections": len(cases),
        "observed_rejections": len(cases) - len(failed),
        "all_required_rejections_observed": not failed,
        "cases": cases,
        "accounting_counterfactual": accounting_counterfactual(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(
                path
            )
            for path in source_paths
        },
    }
    report["payload_sha256"] = payload_sha256(report)
    assert_public_certificate_redacted(report)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} required rejection(s) were accepted"
        )
    print(
        f"Wrote {output}: {len(cases)}/{len(cases)} rejected",
        flush=True,
    )


if __name__ == "__main__":
    main()
