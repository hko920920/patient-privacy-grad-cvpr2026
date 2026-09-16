#!/usr/bin/env python3
"""Build route-local evidence for the provisional external-PLD handler."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CANDIDATE_MODULES = ROOT / "v36_candidate" / "src" / "unitdp"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unitdp  # noqa: E402

if str(CANDIDATE_MODULES) not in unitdp.__path__:
    unitdp.__path__.append(str(CANDIDATE_MODULES))

from unitdp.benchmark_data_v2 import prepare_uci_har_v2  # noqa: E402
from unitdp.compiler_external_pld_v36 import (  # noqa: E402
    ExternalPldContractV36Error,
    compile_owner_external_pld_contract_v36,
    parse_owner_external_pld_contract_v36,
)
from unitdp.compiler_v4 import (  # noqa: E402
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
)
from unitdp.external_pld_backend_v36 import (  # noqa: E402
    BACKEND_SITE_ENV_V36,
    account_external_pld_v36,
    verify_external_pld_environment_v36,
)
from unitdp.handler_external_pld_v36 import (  # noqa: E402
    build_external_pld_handler_v36,
)
from unitdp.owner_external_pld_v36 import (  # noqa: E402
    train_owner_external_pld_fixed_v36,
)
from unitdp.source_bundle_external_pld_v36 import (  # noqa: E402
    compiler_source_bundle_external_pld_v36,
    compiler_source_bundle_sha256_external_pld_v36,
    execution_source_bundle_external_pld_v36,
    execution_source_bundle_sha256_external_pld_v36,
)


OUTPUT = ROOT / "reports" / "g8_external_handler_evidence_v2_20260725"
DATASETS = ("uci", "wisdm", "sepsis")
CONFIGS = {
    dataset: (
        ROOT / "v36_candidate" / "configs" / f"{dataset}_owner_external_pld_v36.yaml"
    )
    for dataset in DATASETS
}
ARTIFACTS = {
    "uci": {
        "mapping": (
            ROOT / "reports/uci_public_preprocessing_smoke_20260724/"
            "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT / "configs/preprocessing/"
            "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "mapping": (
            ROOT / "reports/wisdm_public_protocol_smoke_20260724/"
            "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT / "configs/preprocessing/"
            "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "mapping": (
            ROOT / "reports/sepsis_public_protocol_smoke_20260724/"
            "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT / "configs/preprocessing/"
            "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}
FROZEN_NESTED_PLANS = {
    "uci": ("f0c6334acbc242d7271e6a5d6436a7406955b187fced24a3463390a277444b20"),
    "wisdm": ("40b9a393510443b5c3b0e712affeaccf10ac0ce7a8ba7c73e97710a9987baba1"),
    "sepsis": ("407a01749c6104969da4f66f3f5ea4211208246b3b0fa397e08c1514d8fa7be6"),
}
DEFAULT_DATA_ROOT = ROOT.parent / "DP-SGD" / "data"
DATA_ROOT = Path(os.environ.get("UNITDP_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
UCI_ROOT = DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
EXECUTION_SEED = 6761


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_raw(dataset: str) -> dict[str, Any]:
    value = yaml.safe_load(CONFIGS[dataset].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{dataset} H config is not a mapping")
    return value


def registration_source_checks(handler: object) -> dict[str, Any]:
    obligations = {
        witness.obligation_id: witness for witness in handler.obligation_witnesses
    }
    if (
        set(obligations) != REQUIRED_REGISTRATION_OBLIGATIONS_V4
        or len(obligations) != 7
    ):
        raise AssertionError("H does not carry exactly seven obligations")
    rows: list[dict[str, Any]] = []
    for obligation in sorted(obligations):
        witness = obligations[obligation]
        path = ROOT / witness.source_path
        observed = file_sha256(path)
        if observed != witness.source_sha256:
            raise AssertionError(f"{obligation} source hash is stale")
        rows.append(
            {
                "obligation_id": obligation,
                "implementation_id": witness.implementation_id,
                "source_path": witness.source_path,
                "source_sha256": witness.source_sha256,
                "source_exists": path.is_file(),
            }
        )
    integrity = getattr(
        handler.compiled_type,
        handler.integrity_method_name,
    )
    if (
        Path(inspect.getsourcefile(integrity) or "").resolve()
        != (ROOT / obligations["postcompile_integrity"].source_path).resolve()
    ):
        raise AssertionError("H integrity witness source is spliced")
    return {
        "exact_obligation_count": len(obligations),
        "exact_obligation_ids": sorted(obligations),
        "rows": rows,
    }


def parser_rejection_rows() -> list[dict[str, Any]]:
    raw = load_raw("uci")
    cases: list[tuple[str, dict[str, Any]]] = []
    changed = copy.deepcopy(raw)
    changed["extra"] = True
    cases.append(("extra_top_level_field", changed))
    changed = copy.deepcopy(raw)
    changed["schema_version"] = "unitdp.unregistered"
    cases.append(("wrong_schema", changed))
    changed = copy.deepcopy(raw)
    changed["accountant"]["accepted_bound_type"] = "IS_DOMINATED"
    cases.append(("optimistic_bound_as_accepted", changed))
    changed = copy.deepcopy(raw)
    changed["accountant"]["loss_discretization"] = 0.01
    cases.append(("changed_discretization", changed))
    changed = copy.deepcopy(raw)
    changed["execution"]["implementation_id"] = (
        "unitdp.owner_random_allocation_v4.train_owner_random_allocation_fixed_v4"
    )
    cases.append(("bypassed_executor_wrapper", changed))
    rows: list[dict[str, Any]] = []
    for name, value in cases:
        error: str | None = None
        try:
            parse_owner_external_pld_contract_v36(value)
        except ExternalPldContractV36Error as exc:
            error = str(exc)
        if error is None:
            raise AssertionError(f"strict H parser accepted {name}")
        rows.append(
            {
                "case": name,
                "rejected": True,
                "error_sha256": hashlib.sha256(error.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def build_evidence() -> tuple[dict[str, Any], dict[str, bytes]]:
    if not os.environ.get(BACKEND_SITE_ENV_V36):
        raise RuntimeError(f"{BACKEND_SITE_ENV_V36} must be set for the evidence run")
    if not (UCI_ROOT / "train/X_train.txt").is_file():
        raise FileNotFoundError("Registered UCI HAR source is unavailable")
    environment = verify_external_pld_environment_v36()
    handler = build_external_pld_handler_v36(evidence_report_sha256="0" * 64)
    source_checks = registration_source_checks(handler)
    compilation_rows: list[dict[str, Any]] = []
    compiled_by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        contract = parse_owner_external_pld_contract_v36(load_raw(dataset))
        compiled = compile_owner_external_pld_contract_v36(
            contract,
            mapping_path=ARTIFACTS[dataset]["mapping"],
            preprocessing_artifact_path=(ARTIFACTS[dataset]["preprocessor"]),
            require_executable=True,
        )
        repeated = account_external_pld_v36(
            sigma=contract.base_contract.noise_multiplier,
            num_steps=contract.base_contract.num_steps,
            num_selected=(contract.base_contract.num_selected_steps_per_owner),
            num_epochs=contract.base_contract.num_epochs,
            delta=contract.base_contract.delta,
            target_epsilon=contract.base_contract.target_epsilon,
            direct_epsilon=compiled.base_route.accountant_epsilon,
        )
        if repeated.payload() != compiled.external_response.payload():
            raise AssertionError(
                f"{dataset} clean external responses are not identical"
            )
        if compiled.base_route.public_plan_sha256 != FROZEN_NESTED_PLANS[dataset]:
            raise AssertionError(f"{dataset} nested A plan changed under H")
        compiled.assert_private_execution_integrity()
        row = {
            "dataset": dataset,
            "H_public_contract_sha256": (compiled.contract.public_contract_sha256),
            "H_public_plan_sha256": compiled.public_plan_sha256,
            "nested_A_public_contract_sha256": (
                compiled.base_route.contract.public_contract_sha256
            ),
            "nested_A_public_plan_sha256": (compiled.base_route.public_plan_sha256),
            "direct_RDP_epsilon": (compiled.base_route.accountant_epsilon),
            "external_epsilon_upper": (compiled.external_response.epsilon_upper),
            "external_epsilon_lower": (compiled.external_response.epsilon_lower),
            "external_bracket_gap": (
                compiled.external_response.epsilon_upper
                - compiled.external_response.epsilon_lower
            ),
            "external_request_sha256": (compiled.external_request.sha256),
            "external_response_sha256": (compiled.external_response.response_sha256),
            "two_clean_responses_byte_identical": True,
            "execution_ready": compiled.execution_ready,
            "pass": True,
        }
        compilation_rows.append(row)
        compiled_by_dataset[dataset] = compiled

    torch.set_num_threads(1)
    prepared = prepare_uci_har_v2(
        dataset_root=UCI_ROOT,
        preprocessing_artifact_path=ARTIFACTS["uci"]["preprocessor"],
    )
    prepared.assert_registered_full_conformance()
    uci_result = train_owner_external_pld_fixed_v36(
        route=compiled_by_dataset["uci"],
        raw_train_x=prepared.raw_train_x,
        train_y=prepared.train_y,
        raw_test_x=prepared.raw_test_x,
        test_y=prepared.test_y,
        research_seed=EXECUTION_SEED,
    )
    public = uci_result.public_payload()
    private = uci_result.private_manifest()
    public_bytes = json_bytes(public)
    private_bytes = json_bytes(private)
    execution_manifest = {
        "schema_version": ("unitdp.external_pld_execution_manifest.v36"),
        "dataset": "uci",
        "research_status": "non_release",
        "chain": {
            "H_public_contract_sha256": (
                compiled_by_dataset["uci"].contract.public_contract_sha256
            ),
            "nested_A_public_plan_sha256": (
                compiled_by_dataset["uci"].base_route.public_plan_sha256
            ),
            "external_request_sha256": (
                compiled_by_dataset["uci"].external_request.sha256
            ),
            "external_response_sha256": (
                compiled_by_dataset["uci"].external_response.response_sha256
            ),
            "H_public_plan_sha256": (compiled_by_dataset["uci"].public_plan_sha256),
            "H_public_execution_sha256": (public["public_execution_sha256"]),
            "nested_A_public_execution_sha256": (
                public["nested_allocation_execution"]["public_execution_sha256"]
            ),
        },
        "files": {
            "public": {
                "path": ("uci_external_pld_execution_public_v36.json"),
                "file_sha256": bytes_sha256(public_bytes),
            },
            "private": {
                "path": ("uci_external_pld_execution_private_v36.json"),
                "file_sha256": bytes_sha256(private_bytes),
                "handling": "do_not_publish",
            },
        },
    }
    execution_manifest["payload_sha256"] = canonical_sha256(execution_manifest)
    manifest_bytes = json_bytes(execution_manifest)
    files = {
        "uci_external_pld_execution_public_v36.json": public_bytes,
        "uci_external_pld_execution_private_v36.json": private_bytes,
        "uci_external_pld_execution_manifest_v36.json": manifest_bytes,
    }
    source_bundle = dict(compiler_source_bundle_external_pld_v36())
    execution_bundle = dict(execution_source_bundle_external_pld_v36())
    report: dict[str, Any] = {
        "schema_version": ("unitdp.g8_external_handler_route_evidence.v1"),
        "status": "PASS",
        "scope": (
            "one externally maintained accountant overlay for the "
            "already registered A mechanism; finite route evidence, not "
            "a new privacy theorem or open-world plugin guarantee"
        ),
        "route_id": handler.route_id,
        "schema_id": handler.schema_version,
        "registration_core": handler.registration_core_payload(),
        "registration_core_sha256": handler.registration_core_sha256,
        "environment": environment,
        "source_witnesses": source_checks,
        "strict_parser_rejections": parser_rejection_rows(),
        "compilations": compilation_rows,
        "execution": {
            "dataset": "uci",
            "research_seed": EXECUTION_SEED,
            "public_execution_sha256": (public["public_execution_sha256"]),
            "private_manifest_sha256": (private["private_manifest_sha256"]),
            "manifest": execution_manifest,
            "manifest_file_sha256": bytes_sha256(manifest_bytes),
            "pass": True,
        },
        "source_bundles": {
            "compiler": source_bundle,
            "compiler_sha256": (
                compiler_source_bundle_sha256_external_pld_v36(source_bundle)
            ),
            "execution": execution_bundle,
            "execution_sha256": (
                execution_source_bundle_sha256_external_pld_v36(execution_bundle)
            ),
        },
        "claim_source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): (file_sha256(path))
            for path in (
                Path(__file__).resolve(),
                ROOT / "v36_candidate/src/unitdp/handler_external_pld_v36.py",
                ROOT / "v36_candidate/tests/test_external_pld_v36.py",
                *CONFIGS.values(),
            )
        },
        "limits": [
            "H reuses A's mechanism and executor",
            "the external package is not an independent privacy certificate",
            "the finite evidence does not establish arbitrary plugin safety",
            "the execution uses a noncryptographic research RNG",
        ],
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report, files


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite route evidence: {OUTPUT}")
    report, files = build_evidence()
    if report["status"] != "PASS":
        raise AssertionError("route evidence did not pass")
    report_bytes = json_bytes(report)
    markdown = "\n".join(
        [
            "# G8 External-Handler Route Evidence",
            "",
            f"Status: **{report['status']}**",
            "",
            "```text",
            f"route                       {report['route_id']}",
            "strict H compilations       3/3",
            "clean-response repeats      3/3",
            "UCI execution chain         PASS",
            "exact obligations           7/7",
            "```",
            "",
            "This registers an external accountant overlay for the existing "
            "A mechanism. It is not a fourth mechanism, a new privacy "
            "theorem, or an open-world plugin guarantee.",
            "",
            f"Canonical payload: `{report['payload_sha256']}`",
            "",
        ]
    ).encode("utf-8")
    OUTPUT.mkdir(parents=True)
    for name, value in files.items():
        (OUTPUT / name).write_bytes(value)
    (OUTPUT / "external_pld_handler_evidence_v36.json").write_bytes(report_bytes)
    (OUTPUT / "external_pld_handler_evidence_v36.md").write_bytes(markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "registration_core_sha256": (report["registration_core_sha256"]),
                "payload_sha256": report["payload_sha256"],
                "compilations": [
                    {
                        "dataset": row["dataset"],
                        "epsilon_upper": row["external_epsilon_upper"],
                        "epsilon_lower": row["external_epsilon_lower"],
                        "H_public_plan_sha256": row["H_public_plan_sha256"],
                    }
                    for row in report["compilations"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
