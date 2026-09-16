"""Build a claim-bounded Opacus versus UnitDP binding-scope probe.

This probe does not test whether Opacus is private or correct.  It records
which responsibilities Opacus 1.6.0 demonstrably enforces at its public API
boundary and which owner-route identities remain external to that boundary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import opacus
import torch
import yaml
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_v2 import (  # noqa: E402
    ContractV2Error,
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)


SCHEMA = "unitdp.opacus_boundary_probe.v1"
REPORT_DATE = "2026-07-24"
CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
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
WRONG_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
)

EXTERNAL_ROUTE_FIELDS = (
    "adjacency",
    "artifact_chain",
    "owner_mapping",
    "preprocessing_sha256",
    "privacy_unit",
    "route_id",
    "source_bundle_sha256",
)


class TaggedTensorDataset(Dataset):
    """Tensor-equivalent datasets with deliberately different route metadata."""

    def __init__(self, tag: str):
        self.features = torch.tensor(
            (
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 0.0),
            ),
            dtype=torch.float32,
        )
        self.labels = torch.tensor((0, 1, 1, 0), dtype=torch.long)
        self.owner_mapping_id = f"owner-map-{tag}"
        self.preprocessing_sha256 = tag * 64
        self.source_bundle_sha256 = tag.upper() * 64

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def make_engine() -> PrivacyEngine:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PrivacyEngine(accountant="rdp", secure_mode=False)


def make_private(
    dataset: Dataset,
    *,
    poisson_sampling: bool,
    engine: PrivacyEngine | None = None,
) -> tuple[
    PrivacyEngine,
    nn.Module,
    torch.optim.Optimizer,
    DataLoader,
    DataLoader,
    list[warnings.WarningMessage],
]:
    engine = engine or make_engine()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        private_model, private_optimizer, private_loader = (
            engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=original_loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                poisson_sampling=poisson_sampling,
            )
        )
    return (
        engine,
        private_model,
        private_optimizer,
        private_loader,
        original_loader,
        list(caught),
    )


def rejection_observation(
    action: Callable[[], object],
    *,
    expected_type: type[BaseException],
) -> dict[str, object]:
    try:
        action()
    except Exception as exc:
        return {
            "observed_outcome": "reject",
            "exception_type": type(exc).__name__,
            "expected_exception_type": expected_type.__name__,
            "passed": isinstance(exc, expected_type),
        }
    return {
        "observed_outcome": "accept",
        "exception_type": None,
        "expected_exception_type": expected_type.__name__,
        "passed": False,
    }


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


def build_opacus_observations() -> dict[str, object]:
    signature = inspect.signature(PrivacyEngine.make_private)
    signature_parameters = tuple(signature.parameters)
    absent_route_fields = sorted(
        set(EXTERNAL_ROUTE_FIELDS).difference(signature_parameters)
    )
    if absent_route_fields != sorted(EXTERNAL_ROUTE_FIELDS):
        raise RuntimeError("Unexpected Opacus route-field signature change")

    (
        poisson_engine,
        poisson_model,
        poisson_optimizer,
        poisson_loader,
        _,
        poisson_warnings,
    ) = make_private(
        TaggedTensorDataset("a"),
        poisson_sampling=True,
    )
    if not isinstance(poisson_loader, DPDataLoader):
        raise RuntimeError("Opacus did not construct DPDataLoader")

    (
        _,
        _,
        _,
        nonpoisson_loader,
        original_nonpoisson_loader,
        nonpoisson_warnings,
    ) = make_private(
        TaggedTensorDataset("b"),
        poisson_sampling=False,
    )

    (
        _,
        _,
        _,
        second_loader,
        _,
        dataset_reuse_warnings,
    ) = make_private(
        TaggedTensorDataset("c"),
        poisson_sampling=True,
        engine=poisson_engine,
    )
    new_dataset_warning = any(
        str(item.message).startswith(
            "PrivacyEngine detected new dataset object"
        )
        for item in dataset_reuse_warnings
    )

    model_a = nn.Linear(2, 2)
    model_b = nn.Linear(2, 2)
    mismatched_optimizer = torch.optim.SGD(model_b.parameters(), lr=0.1)
    mismatch_loader = DataLoader(
        TaggedTensorDataset("d"),
        batch_size=2,
        shuffle=False,
    )
    model_optimizer_mismatch = rejection_observation(
        lambda: make_engine().make_private(
            module=model_a,
            optimizer=mismatched_optimizer,
            data_loader=mismatch_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        ),
        expected_type=ValueError,
    )

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "opacus-default-checkpoint.pt"
        poisson_engine.save_checkpoint(
            path=checkpoint_path,
            module=poisson_model,
            optimizer=poisson_optimizer,
        )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path)
    checkpoint_keys = sorted(checkpoint)

    make_private_doc = inspect.getdoc(PrivacyEngine.make_private) or ""
    documented_nonpoisson_caveat = (
        "Technically this doesn't fit the assumptions made by"
        in make_private_doc
    )
    if not documented_nonpoisson_caveat:
        raise RuntimeError("Expected Opacus non-Poisson caveat is absent")

    return {
        "version": opacus.__version__,
        "make_private_parameters": list(signature_parameters),
        "route_fields_absent_from_make_private": absent_route_fields,
        "model_optimizer_mismatch": model_optimizer_mismatch,
        "poisson_true": {
            "succeeded": True,
            "returned_model_type": qualified_type(poisson_model),
            "returned_optimizer_type": qualified_type(
                poisson_optimizer
            ),
            "returned_loader_type": qualified_type(poisson_loader),
            "batch_sampler_type": qualified_type(
                poisson_loader.batch_sampler
            ),
            "accountant_type": qualified_type(
                poisson_engine.accountant
            ),
            "optimizer_step_hook_present": hasattr(
                poisson_optimizer,
                "step_hook",
            ),
            "dataset_identity_preserved": (
                poisson_loader.dataset is poisson_engine.dataset
            ),
            "warnings_observed": len(poisson_warnings),
        },
        "poisson_false": {
            "succeeded": True,
            "returned_loader_type": qualified_type(nonpoisson_loader),
            "original_loader_identity_preserved": (
                nonpoisson_loader is original_nonpoisson_loader
            ),
            "documented_accounting_assumption_caveat": (
                documented_nonpoisson_caveat
            ),
            "warnings_observed": len(nonpoisson_warnings),
        },
        "new_dataset_same_engine": {
            "succeeded": isinstance(second_loader, DPDataLoader),
            "warning_observed": new_dataset_warning,
            "rejected": False,
        },
        "default_checkpoint": {
            "keys": checkpoint_keys,
            "caller_extension_parameter": (
                "checkpoint_dict"
                in inspect.signature(
                    PrivacyEngine.save_checkpoint
                ).parameters
            ),
        },
        "interpretation_limit": (
            "These observations establish API binding scope only; "
            "they do not establish a privacy violation or implementation bug."
        ),
    }


def build_unitdp_observations() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Expected a mapping-valued UnitDP contract")

    mutations = (
        (
            "adjacency_swap",
            ("privacy", "adjacency"),
            "replace_one_owner",
        ),
        (
            "privacy_unit_swap",
            ("mechanism", "sampling_unit"),
            "window",
        ),
        (
            "sampler_swap",
            ("mechanism", "sampler"),
            "fixed_size_owner",
        ),
        (
            "protected_preprocessing",
            ("data", "preprocessing", "fit_on_protected_data"),
            True,
        ),
        (
            "executor_swap",
            ("execution", "implementation_id"),
            "unitdp.owa_dpsgd.train_owa_dpsgd:v1",
        ),
        (
            "data_profile_digest_swap",
            ("data", "benchmark_profile_sha256"),
            "0" * 64,
        ),
    )
    contract_rejections = {
        case_id: rejection_observation(
            lambda path=path, value=value: (
                parse_owner_poisson_contract_v2(
                    set_nested(raw, path, value)
                )
            ),
            expected_type=ContractV2Error,
        )
        for case_id, path, value in mutations
    }

    contract = load_owner_poisson_contract_v2(CONFIG)
    compiled = compile_owner_poisson_contract_v2(
        contract,
        mapping_path=MAPPING,
        preprocessing_artifact_path=PREPROCESSOR,
    )
    baseline = {
        "status": compiled.status,
        "execution_ready": compiled.execution_ready,
        "selected_mapping_records": len(
            compiled.selected_mapping.records
        ),
        "private_mapping_hash_bound": (
            len(compiled.private_source_mapping_canonical_sha256) == 64
        ),
        "private_preprocessor_hash_bound": (
            len(compiled.private_preprocessor_state_sha256) == 64
        ),
        "execution_source_bundle_hash_bound": (
            len(compiled.execution_source_bundle_sha256) == 64
        ),
        "private_execution_binding_hash_bound": (
            len(compiled.private_execution_binding_sha256) == 64
        ),
    }

    wrong_preprocessor = rejection_observation(
        lambda: compile_owner_poisson_contract_v2(
            contract,
            mapping_path=MAPPING,
            preprocessing_artifact_path=WRONG_PREPROCESSOR,
        ),
        expected_type=ContractV2Error,
    )

    with tempfile.TemporaryDirectory() as directory:
        malformed_mapping = Path(directory) / "mapping.csv"
        with MAPPING.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
        if not rows:
            raise RuntimeError("Mapping fixture is empty")
        rows[0]["owner_ids"] = (
            f"{rows[0]['owner_id']};second-owner"
        )
        with malformed_mapping.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=list(fieldnames),
            )
            writer.writeheader()
            writer.writerows(rows)
        malformed_owner_mapping = rejection_observation(
            lambda: compile_owner_poisson_contract_v2(
                contract,
                mapping_path=malformed_mapping,
                preprocessing_artifact_path=PREPROCESSOR,
            ),
            expected_type=ContractV2Error,
        )

    source_bundle_mismatch = rejection_observation(
        lambda: replace(
            compiled,
            execution_source_bundle_sha256="0" * 64,
        ).assert_private_execution_integrity(),
        expected_type=ContractV2Error,
    )

    compiled.selected_mapping.records.pop()
    postcompile_mapping_mutation = rejection_observation(
        compiled.assert_private_execution_integrity,
        expected_type=ContractV2Error,
    )

    all_rejections = {
        **contract_rejections,
        "wrong_preprocessor_artifact": wrong_preprocessor,
        "malformed_multi_owner_mapping": malformed_owner_mapping,
        "postcompile_source_bundle_mismatch": source_bundle_mismatch,
        "postcompile_mapping_mutation": postcompile_mapping_mutation,
    }
    if not all(
        bool(observation["passed"])
        for observation in all_rejections.values()
    ):
        raise RuntimeError("One or more required UnitDP rejections failed")

    return {
        "baseline": baseline,
        "controlled_substitution_rejections": all_rejections,
        "interpretation_limit": (
            "Finite rejection cases establish only the registered "
            "implementation boundary, not completeness or privacy."
        ),
    }


def source_hashes() -> dict[str, str]:
    privacy_engine_source = Path(
        inspect.getsourcefile(PrivacyEngine) or ""
    ).resolve()
    data_loader_source = Path(
        inspect.getsourcefile(DPDataLoader) or ""
    ).resolve()
    paths = {
        "opacus/privacy_engine.py": privacy_engine_source,
        "opacus/data_loader.py": data_loader_source,
        "src/unitdp/compiler_v2.py": (
            ROOT / "src" / "unitdp" / "compiler_v2.py"
        ),
        "src/unitdp/compiler_v3.py": (
            ROOT / "src" / "unitdp" / "compiler_v3.py"
        ),
        "scripts/build_opacus_boundary_probe_v1.py": Path(__file__),
    }
    return {
        name: file_sha256(path)
        for name, path in paths.items()
    }


def coverage_rows(
    opacus_result: dict[str, object],
    unitdp_result: dict[str, object],
) -> list[dict[str, str]]:
    del opacus_result, unitdp_result
    return [
        {
            "boundary": "model/optimizer compatibility",
            "opacus_1_6": (
                "rejects parameter mismatch and wraps both objects"
            ),
            "unitdp_registered_p": (
                "binds a registered executor/source bundle; not a "
                "replacement for Opacus model validation"
            ),
            "novelty_effect": "prior overlap; exclude from novelty",
        },
        {
            "boundary": "Poisson loader, clipping/noise, accounting hook",
            "opacus_1_6": (
                "constructs DPDataLoader and DPOptimizer/accountant hook"
            ),
            "unitdp_registered_p": (
                "uses a custom owner-level executor and dual arithmetic"
            ),
            "novelty_effect": "core responsibility is prior",
        },
        {
            "boundary": "non-Poisson alternative",
            "opacus_1_6": (
                "accepts explicit opt-out and documents accounting caveat"
            ),
            "unitdp_registered_p": (
                "rejects sampler substitution; fixed-size semantics require F"
            ),
            "novelty_effect": "strict registered-policy difference",
        },
        {
            "boundary": "privacy unit and adjacency",
            "opacus_1_6": (
                "not explicit make_private inputs; unit follows loader items"
            ),
            "unitdp_registered_p": (
                "owner unit and add/remove tuple are mandatory"
            ),
            "novelty_effect": "meaningful residual integration boundary",
        },
        {
            "boundary": "window-to-owner mapping",
            "opacus_1_6": (
                "external dataset semantics; new object warns but is accepted"
            ),
            "unitdp_registered_p": (
                "validates one owner per window, hashes mapping, "
                "and checks postcompile integrity"
            ),
            "novelty_effect": "meaningful residual integration boundary",
        },
        {
            "boundary": "preprocessing and data profile",
            "opacus_1_6": (
                "external to make_private and absent from default checkpoint"
            ),
            "unitdp_registered_p": (
                "requires registered profile and exact public preprocessor"
            ),
            "novelty_effect": "meaningful but application-specific residual",
        },
        {
            "boundary": "source/executor identity",
            "opacus_1_6": (
                "wraps runtime objects; does not hash an application source bundle"
            ),
            "unitdp_registered_p": (
                "binds implementation id and registered source bundle hash"
            ),
            "novelty_effect": "partial residual; no attestation",
        },
        {
            "boundary": "checkpoint/artifact scope",
            "opacus_1_6": (
                "default checkpoint stores model, optimizer, accountant; "
                "caller can extend checkpoint_dict"
            ),
            "unitdp_registered_p": (
                "automatically records route-specific execution/artifact hashes"
            ),
            "novelty_effect": "scope difference, not a correctness defect",
        },
    ]


def build_report() -> dict[str, object]:
    opacus_result = build_opacus_observations()
    unitdp_result = build_unitdp_observations()
    return {
        "schema": SCHEMA,
        "report_date": REPORT_DATE,
        "purpose": (
            "Compare documented and executable binding scope; "
            "do not test privacy or claim an Opacus bug."
        ),
        "environment": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "torch": torch.__version__,
            "opacus": opacus.__version__,
        },
        "source_sha256": source_hashes(),
        "opacus_observations": opacus_result,
        "unitdp_observations": unitdp_result,
        "coverage": coverage_rows(opacus_result, unitdp_result),
        "scope_limits": [
            "This is not a privacy audit of Opacus.",
            "This is not a mechanism, epsilon, utility, or speed comparison.",
            (
                "Opacus operates over samples represented by the caller's "
                "dataset; an owner-level representation remains the caller's "
                "responsibility."
            ),
            (
                "UnitDP rejection evidence is finite and limited to two "
                "registered route families."
            ),
            (
                "Hashes and checkpoint contents do not prove honest execution "
                "or attestation."
            ),
        ],
        "verdict": {
            "opacus_overlap": (
                "substantial for the Route P training stack"
            ),
            "residual_boundary": (
                "explicit owner/window semantics, adjacency-route tuple, "
                "registered preprocessing/source/executor/artifact binding"
            ),
            "claim_allowed": (
                "narrow route-specific integration beyond the documented "
                "PrivacyEngine boundary"
            ),
            "claims_disallowed": [
                "new DP-SGD execution binding in general",
                "new Poisson sampler/accountant coupling",
                "Opacus privacy bug",
                "formal equivalence, completeness, or attestation",
            ],
        },
    }


def markdown_report(
    report: dict[str, object],
    *,
    json_sha256: str,
) -> str:
    opacus_result = report["opacus_observations"]
    unitdp_result = report["unitdp_observations"]
    assert isinstance(opacus_result, dict)
    assert isinstance(unitdp_result, dict)
    op_poisson = opacus_result["poisson_true"]
    op_nonpoisson = opacus_result["poisson_false"]
    op_reuse = opacus_result["new_dataset_same_engine"]
    op_mismatch = opacus_result["model_optimizer_mismatch"]
    unit_baseline = unitdp_result["baseline"]
    unit_rejections = unitdp_result[
        "controlled_substitution_rejections"
    ]
    assert isinstance(op_poisson, dict)
    assert isinstance(op_nonpoisson, dict)
    assert isinstance(op_reuse, dict)
    assert isinstance(op_mismatch, dict)
    assert isinstance(unit_baseline, dict)
    assert isinstance(unit_rejections, dict)

    coverage_lines = []
    for row in report["coverage"]:
        assert isinstance(row, dict)
        coverage_lines.append(
            "| {boundary} | {opacus_1_6} | {unitdp_registered_p} | "
            "{novelty_effect} |".format(**row)
        )
    rejection_lines = []
    for case_id, observation in sorted(unit_rejections.items()):
        assert isinstance(observation, dict)
        rejection_lines.append(
            f"| `{case_id}` | {observation['observed_outcome']} | "
            f"`{observation['exception_type']}` | "
            f"{observation['passed']} |"
        )

    absent = ", ".join(
        f"`{name}`"
        for name in opacus_result[
            "route_fields_absent_from_make_private"
        ]
    )
    checkpoint_keys = ", ".join(
        f"`{name}`"
        for name in opacus_result["default_checkpoint"]["keys"]
    )

    return f"""# Opacus 1.6.0 Binding-Boundary Probe V1

- Date: {REPORT_DATE}
- JSON SHA-256: `{json_sha256}`
- Purpose: compare executable and documented binding scope, not privacy.

## Hard result

Opacus already owns substantial Route P infrastructure. It rejected a
model/optimizer parameter mismatch, converted the ordinary loader to
`{op_poisson['returned_loader_type']}`, returned
`{op_poisson['returned_optimizer_type']}`, and attached
`{op_poisson['accountant_type']}`. Its returned loader is the documented
Poisson/empty-batch loader. Those responsibilities cannot be claimed as new.

The residual boundary is narrower. The `make_private` signature has no explicit
inputs for {absent}. UnitDP makes those concepts mandatory in its small route
registry and rejected every controlled substitution below.

## Balanced executable observations

| Observation | Result |
|---|---|
| Opacus model/optimizer mismatch | {op_mismatch['observed_outcome']} (`{op_mismatch['exception_type']}`) |
| Wrapped model | `{op_poisson['returned_model_type']}` |
| Wrapped optimizer | `{op_poisson['returned_optimizer_type']}` |
| Accountant | `{op_poisson['accountant_type']}` |
| Optimizer step hook present | {op_poisson['optimizer_step_hook_present']} |
| `poisson_sampling=True` | accepted; `{op_poisson['returned_loader_type']}` |
| Poisson batch sampler | `{op_poisson['batch_sampler_type']}` |
| `poisson_sampling=False` | accepted; original loader retained = {op_nonpoisson['original_loader_identity_preserved']} |
| Non-Poisson accounting caveat present in docstring | {op_nonpoisson['documented_accounting_assumption_caveat']} |
| New dataset object on same PrivacyEngine | accepted with warning = {op_reuse['warning_observed']} |
| Default checkpoint keys | {checkpoint_keys} |
| Caller can extend checkpoint dictionary | {opacus_result['default_checkpoint']['caller_extension_parameter']} |
| UnitDP registered P baseline | `{unit_baseline['status']}`; {unit_baseline['selected_mapping_records']} selected mapping rows |

The non-Poisson result is not a hidden defect: Opacus explicitly documents
that the opt-out does not strictly fit its accounting assumptions. The
distinction is that Opacus permits the documented approximation, whereas the
registered UnitDP P route rejects a sampler substitution and requires a
separate F route.

## Controlled UnitDP substitutions

| Case | Outcome | Exception | Passed |
|---|---|---|---:|
{chr(10).join(rejection_lines)}

## Claim-level scope comparison

| Boundary | Opacus 1.6.0 | UnitDP registered P | Novelty consequence |
|---|---|---|---|
{chr(10).join(coverage_lines)}

## Interpretation

The comparison supports only this bounded claim:

> The implementation adds an explicit owner/window, adjacency-route,
> preprocessing/source, executor, and artifact binding layer beyond the
> documented PrivacyEngine boundary for two registered mechanisms.

It does not support claims that Opacus is incorrect, that sampler/accountant
coupling is new, or that finite hashes and rejections prove privacy,
completeness, formal equivalence, or attestation.

The candidate's Route P does not simply invoke `PrivacyEngine`; it uses a
custom owner-level executor and Opacus plus `dp-accounting` arithmetic checks.
Accordingly this probe is a binding-scope comparison, not a claim that the two
executors implement identical learning algorithms.
"""


def write_report(
    report: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "opacus_boundary_probe_v1.json"
    markdown_path = output_dir / "opacus_boundary_probe_v1.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing probe output in {output_dir}"
        )

    json_bytes = canonical_json_bytes(report)
    json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    markdown = markdown_report(
        report,
        json_sha256=json_sha256,
    )
    json_path.write_bytes(json_bytes)
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "opacus_boundary_probe_v1_20260724"
        ),
    )
    args = parser.parse_args()

    first = build_report()
    second = build_report()
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise RuntimeError("Repeated in-process probe was not deterministic")
    first["determinism_check"] = {
        "independent_in_process_builds_equal": True,
    }

    json_path, markdown_path = write_report(
        first,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "json_path": str(json_path.relative_to(ROOT)),
                "json_sha256": file_sha256(json_path),
                "markdown_path": str(markdown_path.relative_to(ROOT)),
                "markdown_sha256": file_sha256(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
