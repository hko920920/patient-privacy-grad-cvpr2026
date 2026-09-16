"""Compile a unit contract into an executable training route."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from unitdp.contract import OWNER_UNITS, UnitContract
from unitdp.mapping import WindowMapping
from unitdp.policies import ContributionPolicy


@dataclass(frozen=True)
class CompiledRoute:
    contract: UnitContract
    source_mapping: WindowMapping
    selected_mapping: WindowMapping
    policy: ContributionPolicy
    contract_hash: str
    route_status: str
    route_notes: list[str]

    @property
    def owner_sample_rate(self) -> float:
        owners = max(1, len(self.selected_mapping.owners()))
        batch_size = int(self.contract.training.get("owner_batch_size", owners))
        return min(1.0, batch_size / owners)


def compile_contract(contract: UnitContract, base_dir: str | Path | None = None) -> CompiledRoute:
    """Validate and compile a contract into a selected mapping and route."""

    contract.validate()
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    mapping_path = Path(contract.mapping)
    if not mapping_path.is_absolute():
        mapping_path = root / mapping_path
    source_mapping = WindowMapping.from_csv(mapping_path)
    policy = ContributionPolicy.from_dict(contract.owner_policy)
    selected_mapping = policy.select_mapping(source_mapping)

    notes: list[str] = []
    route_status = "compiled"
    if contract.privacy_unit in OWNER_UNITS:
        multi_owner_windows = int(selected_mapping.stats()["multi_owner_windows"])
        if multi_owner_windows > 0:
            raise ValueError(
                "Owner-level OWA route currently requires single-attribution windows; "
                f"selected mapping contains {multi_owner_windows} multi-owner windows. "
                "Use an explicit allocation/hypergraph contribution rule before compiling."
            )
        notes.append("owner route: sampling, clipping, noising, and accounting units are owner")
        notes.append("owner route invariant: selected windows are single-attribution")
        if selected_mapping.owner_kappa() > 1:
            notes.append("owner multiplicity remains >1, but native owner accounting is used")
        else:
            notes.append("selected mapping also has owner_kappa=1")
    elif contract.privacy_unit == "event":
        if selected_mapping.event_kappa() == 1:
            notes.append("event-direct route: selected mapping has event_kappa=1")
        else:
            route_status = "requires_external_event_accountant"
            notes.append("event claim requires event cap, group conversion, or external structured accountant")
    else:
        notes.append("window route: ordinary generated-unit accounting")

    preprocessing = contract.data.get("preprocessing")
    if preprocessing is not None:
        notes.append(
            "preprocessing bound: "
            f"{preprocessing['scope']} artifact {preprocessing['artifact_id']} "
            f"at {preprocessing['artifact_sha256']}"
        )

    contract_hash = hashlib.sha256(contract.canonical_json().encode("utf-8")).hexdigest()
    return CompiledRoute(
        contract=contract,
        source_mapping=source_mapping,
        selected_mapping=selected_mapping,
        policy=policy,
        contract_hash=contract_hash,
        route_status=route_status,
        route_notes=notes,
    )
