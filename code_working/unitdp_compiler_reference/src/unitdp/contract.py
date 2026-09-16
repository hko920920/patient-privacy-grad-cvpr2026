"""Contract schema for unit-aligned DP training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


OWNER_UNITS = {"owner", "user", "patient", "series"}
SUPPORTED_UNITS = {"window", "event", *OWNER_UNITS}


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class UnitContract:
    """Declarative privacy-unit contract for a training run."""

    privacy_unit: str
    mapping: str
    schedule_mode: str = "fixed"
    adjacency: str = "replace_all_owner_contributions"
    owner_policy: dict[str, Any] = field(default_factory=dict)
    mechanism: dict[str, Any] = field(default_factory=dict)
    accountant: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.privacy_unit not in SUPPORTED_UNITS:
            raise ValueError(f"Unsupported privacy_unit: {self.privacy_unit}")
        if self.schedule_mode not in {"fixed", "support_union", "bounded_stochastic"}:
            raise ValueError(f"Unsupported schedule_mode: {self.schedule_mode}")
        if not self.mapping:
            raise ValueError("mapping path is required")

        if self.privacy_unit in OWNER_UNITS:
            for key in ["sampling_unit", "clipping_unit", "noising_unit"]:
                value = self.mechanism.get(key)
                if value not in OWNER_UNITS:
                    raise ValueError(
                        f"Owner-level contract requires mechanism.{key}=owner, got {value!r}"
                    )
            accountant_unit = self.accountant.get("unit")
            if accountant_unit not in OWNER_UNITS:
                raise ValueError(
                    f"Owner-level contract requires accountant.unit=owner, got {accountant_unit!r}"
                )

        label_source = self.owner_policy.get("label_source", "unused")
        policy_type = self.owner_policy.get("type", "")
        if "label" in policy_type and label_source != "public":
            raise ValueError(
                "Label-aware contribution policies require owner_policy.label_source=public "
                "or a separate private selection analysis."
            )

        preprocessing = self.data.get("preprocessing")
        if preprocessing is not None:
            if not isinstance(preprocessing, dict):
                raise ValueError("data.preprocessing must be a mapping")
            scope = preprocessing.get("scope")
            if scope != "public_fixed":
                raise ValueError(
                    "Only data.preprocessing.scope=public_fixed is currently compiled"
                )
            required = {
                "schema",
                "artifact_id",
                "artifact_sha256",
                "source_reference_sha256",
                "input_dim",
                "transform",
                "fit_on_protected_data",
            }
            missing = sorted(required.difference(preprocessing))
            if missing:
                raise ValueError(
                    "Public fixed preprocessing binding is missing fields: "
                    + ", ".join(missing)
                )
            if preprocessing["schema"] != "unitdp.fixed_affine_preprocessor.v1":
                raise ValueError("Unsupported public fixed preprocessing schema")
            if not str(preprocessing["artifact_id"]):
                raise ValueError("Public fixed preprocessing artifact_id is required")
            if not _is_sha256(preprocessing["artifact_sha256"]):
                raise ValueError(
                    "Public fixed preprocessing artifact_sha256 must be a lowercase SHA-256"
                )
            if not _is_sha256(preprocessing["source_reference_sha256"]):
                raise ValueError(
                    "Public fixed preprocessing source_reference_sha256 "
                    "must be a lowercase SHA-256"
                )
            if int(preprocessing["input_dim"]) <= 0:
                raise ValueError("Public fixed preprocessing input_dim must be positive")
            if preprocessing["transform"] != "standard_scaler":
                raise ValueError("Unsupported public fixed preprocessing transform")
            if preprocessing["fit_on_protected_data"] is not False:
                raise ValueError(
                    "Public fixed preprocessing requires fit_on_protected_data=false"
                )

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "privacy_unit": self.privacy_unit,
                "mapping": self.mapping,
                "schedule_mode": self.schedule_mode,
                "adjacency": self.adjacency,
                "owner_policy": self.owner_policy,
                "mechanism": self.mechanism,
                "accountant": self.accountant,
                "training": self.training,
                "data": self.data,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def load_contract(path: str | Path) -> UnitContract:
    """Load a contract from YAML or JSON."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    contract = UnitContract(
        privacy_unit=str(raw["privacy_unit"]),
        mapping=str(raw["mapping"]),
        schedule_mode=str(raw.get("schedule_mode", "fixed")),
        adjacency=str(raw.get("adjacency", "replace_all_owner_contributions")),
        owner_policy=dict(raw.get("owner_policy", {})),
        mechanism=dict(raw.get("mechanism", {})),
        accountant=dict(raw.get("accountant", {})),
        training=dict(raw.get("training", {})),
        data=dict(raw.get("data", {})),
    )
    contract.validate()
    return contract
