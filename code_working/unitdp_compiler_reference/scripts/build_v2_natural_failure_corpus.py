from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_data_v2 import (  # noqa: E402
    prepare_sepsis_v2,
    prepare_wisdm_v2,
)
from unitdp.compiler_v2 import (  # noqa: E402
    ContractV2Error,
    compile_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)
from unitdp.contract import load_contract  # noqa: E402
from unitdp.owner_poisson_v2 import (  # noqa: E402
    OwnerPoissonExecutionV2Error,
    train_owner_poisson_fixed_v2,
)
from unitdp.preprocessing import (  # noqa: E402
    load_fixed_affine_preprocessor,
)
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)
from unitdp_spec_oracle import (  # noqa: E402
    load_contract_file,
    load_oracle_spec,
    validate_mapping_array_binding,
    validate_registered_research_contract,
)


SPEC_PATH = (
    ROOT / "specs" / "owner_poisson_v2_contract_oracle_v1.json"
)
UCI_CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
WISDM_CONFIG = (
    ROOT / "configs" / "v2" / "wisdm_owner_poisson_v2.yaml"
)
SEPSIS_CONFIG = (
    ROOT / "configs" / "v2" / "sepsis_owner_poisson_v2.yaml"
)
UCI_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
WISDM_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
)
SEPSIS_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
)
OLD_WISDM_MAPPING = (
    ROOT
    / "reports"
    / "comparison_benchmark_5seed_001"
    / "wisdm"
    / "seed_13"
    / "wisdm_train_mapping.csv"
)
CURRENT_WISDM_MAPPING = (
    ROOT
    / "reports"
    / "wisdm_public_protocol_smoke_20260724"
    / "wisdm_train_mapping.csv"
)
OLD_SEPSIS_MAPPING = (
    ROOT
    / "reports"
    / "comparison_benchmark_5seed_001"
    / "sepsis"
    / "seed_13"
    / "sepsis_train_mapping.csv"
)
CURRENT_SEPSIS_MAPPING = (
    ROOT
    / "reports"
    / "sepsis_public_protocol_smoke_20260724"
    / "sepsis_train_mapping.csv"
)
LEGACY_PUBLIC_CERTIFICATE = (
    ROOT
    / "reports"
    / "comparison_benchmark_3seed_001"
    / "uci"
    / "seed_13"
    / "owner_certificate.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _witness(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _external_public_witness(
    path: Path,
    *,
    public_reference: str,
) -> dict[str, object]:
    return {
        "public_reference": public_reference,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _set_nested(
    value: dict[str, Any],
    path: tuple[str, ...],
    replacement: object,
) -> None:
    cursor: Any = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement


def _contract_rejection(
    *,
    spec: dict[str, Any],
    path: tuple[str, ...],
    replacement: object,
) -> dict[str, object]:
    raw = load_contract_file(UCI_CONFIG)
    mutated = copy.deepcopy(raw)
    _set_nested(mutated, path, replacement)
    oracle = validate_registered_research_contract(mutated, spec)
    try:
        parse_owner_poisson_contract_v2(mutated)
    except ContractV2Error as exc:
        production_accepted = False
        production_reason = str(exc)
    else:
        production_accepted = True
        production_reason = "accepted"
    if oracle.accepted or production_accepted:
        raise RuntimeError(
            f"Expected both judgments to reject mutation at {path!r}"
        )
    return {
        "mutation_path": list(path),
        "replacement": replacement,
        "oracle": {
            "accepted": oracle.accepted,
            "stage": oracle.stage,
            "reason": oracle.reason,
        },
        "production_parser": {
            "accepted": production_accepted,
            "reason": production_reason,
        },
    }


def _require_ordered_fragments(
    path: Path,
    fragments: tuple[str, ...],
) -> dict[str, int]:
    source = path.read_text(encoding="utf-8")
    positions: dict[str, int] = {}
    previous = -1
    for fragment in fragments:
        position = source.find(fragment, previous + 1)
        if position < 0:
            raise RuntimeError(
                f"Required source witness is absent from {path}: {fragment!r}"
            )
        if position <= previous:
            raise RuntimeError(
                f"Source witness order changed in {path}: {fragment!r}"
            )
        positions[fragment] = position
        previous = position
    return positions


def _csv_record_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _legacy_schedule_case(
    spec: dict[str, Any],
) -> dict[str, object]:
    source = ROOT / "src" / "unitdp" / "owa_dpsgd.py"
    fragments = (
        "owner_batch_size = min(config.owner_batch_size, len(owners))",
        "owner_sample_rate = min(1.0, owner_batch_size / len(owners))",
        "steps_per_epoch = math.ceil(len(owners) / owner_batch_size)",
        "total_steps = config.epochs * steps_per_epoch",
    )
    _require_ordered_fragments(source, fragments)

    def schedule(owner_count: int) -> dict[str, object]:
        batch_size = min(8, owner_count)
        return {
            "owners": owner_count,
            "owner_batch_size": batch_size,
            "owner_sample_rate": batch_size / owner_count,
            "total_steps": 5 * math.ceil(owner_count / batch_size),
        }

    full = schedule(25)
    removed = schedule(24)
    if (
        full["owner_sample_rate"] == removed["owner_sample_rate"]
        or full["total_steps"] == removed["total_steps"]
    ):
        raise RuntimeError("Legacy schedule counterexample is not effective")
    return {
        "case_id": "legacy_observed_owner_schedule",
        "origin": "preserved_pre_v2_source",
        "classification": "privacy_mechanism_mismatch",
        "witnesses": [_witness(source)],
        "observed": {
            "full_dataset_schedule": full,
            "remove_one_owner_schedule": removed,
            "q_changed": True,
            "total_steps_changed": True,
        },
        "violated_clause": (
            "mechanism.parameter_source must be registered_benchmark_profile; "
            "q and T cannot be derived from the observed owner count"
        ),
        "v2_judgment": _contract_rejection(
            spec=spec,
            path=("mechanism", "parameter_source"),
            replacement="observed_owner_count",
        ),
        "status": "reproduced",
    }


def _legacy_empty_step_case(
    spec: dict[str, Any],
) -> dict[str, object]:
    source = ROOT / "src" / "unitdp" / "owa_dpsgd.py"
    fragments = (
        "accountant.step(noise_multiplier=noise_multiplier, sample_rate=owner_sample_rate)",
        "if not selected_owners:",
        "empty_steps += 1",
        "continue",
        "noise = rng.normal_tensor(",
        "optimizer.step()",
    )
    positions = _require_ordered_fragments(source, fragments)
    if not (
        positions[fragments[0]]
        < positions[fragments[1]]
        < positions[fragments[4]]
        < positions[fragments[5]]
    ):
        raise RuntimeError("Legacy empty-step control-flow witness changed")
    return {
        "case_id": "legacy_empty_owner_sample_skips_update",
        "origin": "preserved_pre_v2_source",
        "classification": "privacy_mechanism_mismatch",
        "witnesses": [_witness(source)],
        "observed": {
            "accountant_step_precedes_empty_check": True,
            "empty_branch_continues_before_noise": True,
            "empty_branch_continues_before_optimizer_step": True,
        },
        "violated_clause": (
            "mechanism.empty_step_behavior must be apply_gaussian_update"
        ),
        "v2_judgment": _contract_rejection(
            spec=spec,
            path=("mechanism", "empty_step_behavior"),
            replacement="skip_update",
        ),
        "status": "reproduced",
    }


def _legacy_adjacency_case(
    spec: dict[str, Any],
) -> dict[str, object]:
    config = ROOT / "configs" / "synthetic_owner_contract.yaml"
    contract_source = ROOT / "src" / "unitdp" / "contract.py"
    legacy = load_contract(config)
    if legacy.adjacency != "replace_all_owner_contributions":
        raise RuntimeError("Legacy adjacency witness changed")
    return {
        "case_id": "legacy_replace_owner_adjacency_accepted",
        "origin": "preserved_pre_v2_contract",
        "classification": "unsupported_privacy_claim",
        "witnesses": [
            _witness(config),
            _witness(contract_source),
        ],
        "observed": {
            "legacy_validation_accepted": True,
            "legacy_adjacency": legacy.adjacency,
            "registered_v2_adjacency": "add_remove_one_owner",
        },
        "violated_clause": (
            "privacy.adjacency must equal add_remove_one_owner for the "
            "registered sampled-Gaussian route"
        ),
        "v2_judgment": _contract_rejection(
            spec=spec,
            path=("privacy", "adjacency"),
            replacement="replace_all_owner_contributions",
        ),
        "status": "reproduced",
    }


def _legacy_public_artifact_case() -> dict[str, object]:
    raw = json.loads(
        LEGACY_PUBLIC_CERTIFICATE.read_text(encoding="utf-8")
    )
    try:
        assert_public_certificate_redacted(raw)
    except ValueError as exc:
        reason = str(exc)
    else:
        raise RuntimeError("Legacy public certificate was not rejected")
    forbidden = (
        "contract_sha256",
        "selected_mapping_sha256",
        "selected_mapping_stats",
        "selection_diagnostics",
        "source_mapping_sha256",
        "source_mapping_stats",
        "training_result",
    )
    if not all(name in reason for name in forbidden):
        raise RuntimeError("Expected legacy public-field witnesses are absent")
    return {
        "case_id": "legacy_public_certificate_exposes_execution_fields",
        "origin": "preserved_pre_v2_artifact",
        "classification": "public_private_metadata_boundary_failure",
        "witnesses": [_witness(LEGACY_PUBLIC_CERTIFICATE)],
        "observed": {
            "v2_redaction_check_rejected": True,
            "forbidden_top_level_fields": list(forbidden),
            "nested_seed_detected": "training_result.config.seed" in reason,
            "rejection_reason": reason,
        },
        "violated_clause": (
            "public artifacts must exclude private mapping commitments, "
            "execution diagnostics, and training seeds"
        ),
        "v2_judgment": {
            "oracle": "not_applicable_to_release_artifact_schema",
            "production_public_boundary": "rejected",
        },
        "status": "reproduced",
    }


def _stale_mapping_case(
    *,
    dataset: str,
    spec: dict[str, Any],
    config_path: Path,
    preprocessor_path: Path,
    old_mapping_path: Path,
    current_mapping_path: Path,
    prepared: object,
) -> dict[str, object]:
    raw = load_contract_file(config_path)
    profile_id = raw["data"]["benchmark_profile_id"]
    current_binding = validate_mapping_array_binding(
        current_mapping_path,
        profile_id=profile_id,
        feature_shape=prepared.raw_train_x.shape,
        labels=prepared.train_y,
        spec=spec,
        require_registered_train_rows=True,
    )
    if not current_binding.accepted:
        raise RuntimeError(
            f"Current {dataset} mapping failed independent binding: "
            f"{current_binding}"
        )
    old_binding = validate_mapping_array_binding(
        old_mapping_path,
        profile_id=profile_id,
        feature_shape=prepared.raw_train_x.shape,
        labels=prepared.train_y,
        spec=spec,
        require_registered_train_rows=True,
    )
    if old_binding.accepted:
        raise RuntimeError(
            f"Old {dataset} mapping unexpectedly passed independent binding"
        )

    contract = parse_owner_poisson_contract_v2(raw)
    route = compile_owner_poisson_contract_v2(
        contract,
        mapping_path=old_mapping_path,
        preprocessing_artifact_path=preprocessor_path,
    )
    try:
        train_owner_poisson_fixed_v2(
            route=route,
            raw_train_x=prepared.raw_train_x,
            train_y=prepared.train_y,
            raw_test_x=prepared.raw_test_x,
            test_y=prepared.test_y,
            research_seed=13,
        )
    except OwnerPoissonExecutionV2Error as exc:
        execution_reason = str(exc)
    else:
        raise RuntimeError(
            f"Old {dataset} mapping unexpectedly reached training"
        )
    if "row count must equal" not in execution_reason:
        raise RuntimeError(
            f"Unexpected {dataset} rejection reason: {execution_reason}"
        )
    old_rows = _csv_record_count(old_mapping_path)
    current_rows = _csv_record_count(current_mapping_path)
    prepared_rows = int(prepared.raw_train_x.shape[0])
    if old_rows == current_rows or current_rows != prepared_rows:
        raise RuntimeError(f"{dataset} stale-mapping witness is ineffective")
    return {
        "case_id": f"legacy_{dataset}_mapping_is_stale",
        "origin": "preserved_pre_v2_artifact",
        "classification": "data_pipeline_conformance_failure",
        "witnesses": [
            _witness(old_mapping_path),
            _witness(current_mapping_path),
        ],
        "observed": {
            "old_mapping_rows": old_rows,
            "registered_mapping_rows": current_rows,
            "prepared_feature_rows": prepared_rows,
            "structural_compiler_accepted_old_mapping": True,
            "independent_array_binding": {
                "accepted": old_binding.accepted,
                "stage": old_binding.stage,
                "reason": old_binding.reason,
            },
            "production_executor_rejected": True,
            "production_rejection_reason": execution_reason,
        },
        "violated_clause": (
            "mapping row indices and labels must bind the exact feature and "
            "label arrays supplied to the registered executor"
        ),
        "v2_judgment": {
            "oracle_mapping_array_binding": "rejected",
            "production_structural_compiler": "accepted",
            "production_executor": "rejected_before_training",
        },
        "status": "reproduced",
    }


def _uci_runtime_scaler_case(
    *,
    spec: dict[str, Any],
    uci_train_dir: Path,
) -> dict[str, object]:
    x_path = uci_train_dir / "X_train.txt"
    owner_path = uci_train_dir / "subject_train.txt"
    x = np.loadtxt(x_path).astype(np.float32)
    owners = np.loadtxt(owner_path, dtype=int)
    if x.shape != (7352, 561) or owners.shape != (7352,):
        raise RuntimeError("UCI public train arrays have unexpected shapes")
    keep = owners != 1
    full_then_remove = (
        StandardScaler().fit_transform(x).astype(np.float32)[keep]
    )
    remove_then_fit = (
        StandardScaler().fit_transform(x[keep]).astype(np.float32)
    )
    difference = np.abs(full_then_remove - remove_then_fit)
    changed = int(np.count_nonzero(difference))
    total = int(difference.size)
    if changed == 0:
        raise RuntimeError("Runtime-fitted UCI scaler was removal invariant")

    fixed = load_fixed_affine_preprocessor(UCI_PREPROCESSOR)
    fixed_full_then_remove = fixed.transform(x)[keep]
    fixed_remove_then_transform = fixed.transform(x[keep])
    fixed_equal = np.array_equal(
        fixed_full_then_remove,
        fixed_remove_then_transform,
    )
    if not fixed_equal:
        raise RuntimeError("Registered UCI preprocessor is not invariant")
    source_registration = json.loads(
        UCI_PREPROCESSOR.read_text(encoding="utf-8")
    )["source"]
    if _file_sha256(x_path) != source_registration["sha256"]:
        raise RuntimeError("UCI source does not match registered digest")
    return {
        "case_id": "legacy_uci_runtime_scaler_crosses_owner_boundary",
        "origin": "reproduced_public_data_failure",
        "classification": "owner_local_sensitivity_premise_failure",
        "witnesses": [
            _witness(UCI_PREPROCESSOR),
            _external_public_witness(
                x_path,
                public_reference="UCI HAR Dataset/train/X_train.txt",
            ),
            _external_public_witness(
                owner_path,
                public_reference=(
                    "UCI HAR Dataset/train/subject_train.txt"
                ),
            ),
        ],
        "observed": {
            "removed_owner": 1,
            "remaining_rows": int(keep.sum()),
            "changed_entries": changed,
            "total_entries": total,
            "maximum_absolute_change": float(difference.max()),
            "mean_absolute_change": float(difference.mean()),
            "registered_fixed_transform_removal_invariant": fixed_equal,
        },
        "violated_clause": (
            "data.preprocessing.fit_on_protected_data must be false; "
            "preprocessing must not couple unaffected owners"
        ),
        "v2_judgment": _contract_rejection(
            spec=spec,
            path=(
                "data",
                "preprocessing",
                "fit_on_protected_data",
            ),
            replacement=True,
        ),
        "status": "reproduced",
    }


def _legacy_fixed_size_loader_case(
    spec: dict[str, Any],
) -> dict[str, object]:
    route_script = (
        ROOT / "scripts" / "run_fixed_size_mog_owner_baseline.py"
    )
    trainer = ROOT / "src" / "unitdp" / "window_dpsgd.py"
    route_fragments = (
        "poisson_sampling=False",
        "drop_last=True",
        '"loader_law": "fixed_size_without_replacement_drop_last"',
    )
    trainer_fragments = (
        "loader = DataLoader(",
        "shuffle=True",
        "drop_last=config.drop_last",
    )
    _require_ordered_fragments(route_script, route_fragments)
    _require_ordered_fragments(trainer, trainer_fragments)
    route_source = route_script.read_text(encoding="utf-8")
    if '"model": "Hypergeometric(' not in route_source:
        raise RuntimeError("Hypergeometric accounting witness is absent")
    return {
        "case_id": "legacy_fixed_size_claim_uses_epoch_shuffle_loader",
        "origin": "preserved_pre_v2_source",
        "classification": "unsupported_exact_accounting",
        "witnesses": [
            _witness(route_script),
            _witness(trainer),
        ],
        "observed": {
            "opacus_poisson_sampling": False,
            "data_loader_shuffle": True,
            "data_loader_drop_last": True,
            "reported_step_law": (
                "fixed_size_without_replacement_drop_last"
            ),
            "accounting_model": (
                "per-step Hypergeometric mixture over the full population"
            ),
            "exact_law_correspondence_established": False,
        },
        "violated_clause": (
            "the registered owner-Poisson route cannot accept a fixed-size "
            "or shuffled loader under its Poisson accountant"
        ),
        "v2_judgment": _contract_rejection(
            spec=spec,
            path=("mechanism", "sampler"),
            replacement="fixed_size_owner",
        ),
        "status": "reproduced",
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    spec = load_oracle_spec(SPEC_PATH)
    cases: list[dict[str, object]] = [
        _legacy_schedule_case(spec),
        _legacy_empty_step_case(spec),
        _legacy_adjacency_case(spec),
        _legacy_public_artifact_case(),
    ]

    wisdm_prepared = prepare_wisdm_v2(
        raw_path=args.wisdm_raw,
        preprocessing_artifact_path=WISDM_PREPROCESSOR,
    )
    cases.append(
        _stale_mapping_case(
            dataset="wisdm",
            spec=spec,
            config_path=WISDM_CONFIG,
            preprocessor_path=WISDM_PREPROCESSOR,
            old_mapping_path=OLD_WISDM_MAPPING,
            current_mapping_path=CURRENT_WISDM_MAPPING,
            prepared=wisdm_prepared,
        )
    )
    sepsis_prepared = prepare_sepsis_v2(
        cache_dir=args.sepsis_cache,
        preprocessing_artifact_path=SEPSIS_PREPROCESSOR,
    )
    cases.append(
        _stale_mapping_case(
            dataset="sepsis",
            spec=spec,
            config_path=SEPSIS_CONFIG,
            preprocessor_path=SEPSIS_PREPROCESSOR,
            old_mapping_path=OLD_SEPSIS_MAPPING,
            current_mapping_path=CURRENT_SEPSIS_MAPPING,
            prepared=sepsis_prepared,
        )
    )
    cases.extend(
        [
            _uci_runtime_scaler_case(
                spec=spec,
                uci_train_dir=args.uci_train_dir,
            ),
            _legacy_fixed_size_loader_case(spec),
        ]
    )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("Natural failure case ids are not unique")
    if not all(case["status"] == "reproduced" for case in cases):
        raise RuntimeError("At least one natural failure was not reproduced")

    classifications: dict[str, int] = {}
    origins: dict[str, int] = {}
    for case in cases:
        classification = str(case["classification"])
        origin = str(case["origin"])
        classifications[classification] = (
            classifications.get(classification, 0) + 1
        )
        origins[origin] = origins.get(origin, 0) + 1
    report: dict[str, object] = {
        "schema_version": "unitdp.natural_failure_corpus.v2.1",
        "scope": (
            "retrospective preserved-source and public-benchmark "
            "conformance failures"
        ),
        "claim_boundary": {
            "establishes": (
                "reproducible detection of the listed concrete failures"
            ),
            "does_not_establish": [
                "complete DP bug detection",
                "a formal privacy proof",
                "that every listed failure has the same privacy consequence",
                "adversarial execution attestation",
            ],
        },
        "bindings": {
            "builder": _witness(Path(__file__)),
            "oracle_spec": _witness(SPEC_PATH),
            "oracle_implementation": _witness(
                ROOT / "src" / "unitdp_spec_oracle" / "oracle_v1.py"
            ),
            "production_source_bundle_sha256": (
                execution_source_bundle_sha256_v2()
            ),
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "reproduced_count": sum(
                case["status"] == "reproduced" for case in cases
            ),
            "all_cases_reproduced": all(
                case["status"] == "reproduced" for case in cases
            ),
            "case_count_by_classification": dict(
                sorted(classifications.items())
            ),
            "case_count_by_origin": dict(sorted(origins.items())),
            "contract_mutation_cases_with_oracle_production_rejection": 5,
            "mapping_array_binding_cases_rejected": 2,
            "public_private_boundary_cases_rejected": 1,
        },
    }
    report["natural_failure_corpus_sha256"] = _canonical_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "v2_natural_failure_corpus_v32r1_20260724"
            / "natural_failure_corpus_v2.json"
        ),
    )
    parser.add_argument(
        "--uci-train-dir",
        type=Path,
        default=(
            DATA_ROOT
            / "uci_har"
            / "extracted"
            / "UCI HAR Dataset"
            / "train"
        ),
    )
    parser.add_argument(
        "--wisdm-raw",
        type=Path,
        default=(
            DATA_ROOT
            / "wisdm"
            / "WISDM_ar_v1.1"
            / "WISDM_ar_v1.1_raw.txt"
        ),
    )
    parser.add_argument(
        "--sepsis-cache",
        type=Path,
        default=DATA_ROOT / "sepsis2019_cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Output must remain inside the repository") from exc
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing report: {output}"
        )
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "case_count": report["summary"]["case_count"],
                "all_cases_reproduced": report["summary"][
                    "all_cases_reproduced"
                ],
                "natural_failure_corpus_sha256": report[
                    "natural_failure_corpus_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
