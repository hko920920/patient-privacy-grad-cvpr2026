"""Immutable AAAI-27 benchmark profiles for the strict V2 owner route.

These profiles are experiment registrations, not facts inferred from the
protected runtime mapping.  A profile fixes the public preprocessing artifact,
data protocol, privacy schedule, contribution policy, and optimization
constants that must be reproduced by the benchmark configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class BenchmarkProfileV2:
    """Exact public constants for one registered benchmark experiment."""

    profile_id: str
    dataset_id: str
    input_dim: int
    num_classes: int
    preprocessing_artifact_id: str
    preprocessing_artifact_sha256: str
    source_reference_sha256: str
    protocol_id: str
    data_preparation_implementation_id: str
    train_owners: int
    test_owners: int
    mapping_columns: tuple[str, ...]
    mapping_label_to_index: tuple[tuple[str, int], ...]
    train_rows: int
    test_rows: int
    train_transformed_sha256: str
    test_transformed_sha256: str
    train_labels_sha256: str
    test_labels_sha256: str
    train_mapping_canonical_sha256: str
    test_mapping_canonical_sha256: str
    owner_sample_rate: float
    total_steps: int
    clip_norm: float
    noise_multiplier: float
    update_denominator: float
    max_windows_per_owner: int
    windows_per_owner_per_step: int
    learning_rate: float
    model_id: str
    class_weights: tuple[float, ...] | None
    target_epsilon: float = 8.0
    delta: float = 1e-5

    def _base_contract_binding(self) -> dict[str, object]:
        return {
            "privacy": {
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "owner_sample_rate": self.owner_sample_rate,
                "total_steps": self.total_steps,
                "clip_norm": self.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": self.update_denominator,
            },
            "contribution_policy": {
                "max_windows_per_owner": self.max_windows_per_owner,
                "windows_per_owner_per_step": (
                    self.windows_per_owner_per_step
                ),
            },
            "optimization": {
                "learning_rate": self.learning_rate,
                "model_id": self.model_id,
                "class_weights": (
                    list(self.class_weights)
                    if self.class_weights is not None
                    else None
                ),
            },
            "data": {
                "benchmark_profile_id": self.profile_id,
                "dataset_id": self.dataset_id,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "preprocessing": {
                    "scope": "public_fixed",
                    "schema": "unitdp.fixed_affine_preprocessor.v1",
                    "artifact_id": self.preprocessing_artifact_id,
                    "artifact_sha256": self.preprocessing_artifact_sha256,
                    "source_reference_sha256": self.source_reference_sha256,
                    "input_dim": self.input_dim,
                    "transform": "standard_scaler",
                    "fit_on_protected_data": False,
                },
                "public_protocol": {
                    "protocol_id": self.protocol_id,
                    "data_preparation_implementation_id": (
                        self.data_preparation_implementation_id
                    ),
                    "source_reference_sha256": self.source_reference_sha256,
                    "train_owners": self.train_owners,
                    "test_owners": self.test_owners,
                    "evaluation_scope": "public_benchmark",
                    "full_data_conformance_sha256": (
                        self.full_data_conformance_sha256
                    ),
                },
            },
        }

    def full_data_conformance_payload(self) -> dict[str, object]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_transformed_sha256": self.train_transformed_sha256,
            "test_transformed_sha256": self.test_transformed_sha256,
            "train_labels_sha256": self.train_labels_sha256,
            "test_labels_sha256": self.test_labels_sha256,
            "train_mapping_canonical_sha256": (
                self.train_mapping_canonical_sha256
            ),
            "test_mapping_canonical_sha256": (
                self.test_mapping_canonical_sha256
            ),
        }

    @property
    def full_data_conformance_sha256(self) -> str:
        encoded = json.dumps(
            self.full_data_conformance_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def registry_payload(self) -> dict[str, object]:
        """Return the complete versioned profile, including mapping semantics."""

        return {
            "schema_version": "unitdp.benchmark_profile.v2",
            "contract_binding": self._base_contract_binding(),
            "mapping_columns": list(self.mapping_columns),
            "mapping_label_to_index": [
                [label, index]
                for label, index in self.mapping_label_to_index
            ],
            "full_data_conformance": (
                self.full_data_conformance_payload()
            ),
        }

    @property
    def registry_sha256(self) -> str:
        encoded = json.dumps(
            self.registry_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def expected_contract_binding(self) -> dict[str, object]:
        """Return all profile-controlled contract fields."""

        binding = self._base_contract_binding()
        data = binding["data"]
        assert isinstance(data, dict)
        data["benchmark_profile_sha256"] = self.registry_sha256
        return binding


UCI_HAR_AAAI27_V1 = BenchmarkProfileV2(
    profile_id="uci_har_aaai27_v1",
    dataset_id="uci_har_public_561_v1",
    input_dim=561,
    num_classes=6,
    preprocessing_artifact_id=(
        "uci_har_published_train_standard_scaler_v1"
    ),
    preprocessing_artifact_sha256=(
        "774713e3bef48c784113f4097642a59741481e7495f858ed41cdb1788ae8fadf"
    ),
    source_reference_sha256=(
        "9c1246099c8d5463779eec5a3845cc3898df9d8c715441ab2a1232d4421fc42f"
    ),
    protocol_id="uci_har_published_owner_split_v1",
    data_preparation_implementation_id=(
        "unitdp.benchmark_data_v2.prepare_uci_har_v2"
    ),
    train_owners=21,
    test_owners=9,
    mapping_columns=(
        "scenario",
        "window_id",
        "owner_id",
        "owner_ids",
        "start",
        "end",
        "row_index",
        "label",
    ),
    mapping_label_to_index=tuple((str(index), index) for index in range(6)),
    train_rows=7352,
    test_rows=2947,
    train_transformed_sha256=(
        "d08be9be6d73314969e2f4e4ee979ad6a959e7a9abbafeb5c49e02d3b722e7e3"
    ),
    test_transformed_sha256=(
        "46402cc9832bfd272b7612fc961b3d9f7ffc558cee5cdb844a22e644e32ac111"
    ),
    train_labels_sha256=(
        "cc77aaecf2d4a201542a365c2bb4221f2e1671e8d5ea0c9220899fb190874d76"
    ),
    test_labels_sha256=(
        "666cd1b00a531b008992aeba2caad9d543c603131d2c5999fc50bc1836ac55a3"
    ),
    train_mapping_canonical_sha256=(
        "fe0838ed222094a62067d43df234d9a29be105ae008990232b8450e1906f6714"
    ),
    test_mapping_canonical_sha256=(
        "9f502ec9da02e081b324d7be6615aef5fc2b4cee40573d7e93735df9a1059970"
    ),
    owner_sample_rate=8.0 / 21.0,
    total_steps=15,
    clip_norm=1.0,
    noise_multiplier=1.26970999503374,
    update_denominator=8.0,
    max_windows_per_owner=16,
    windows_per_owner_per_step=8,
    learning_rate=0.05,
    model_id="linear_561x6",
    class_weights=None,
)


WISDM_AAAI27_V1 = BenchmarkProfileV2(
    profile_id="wisdm_aaai27_v1",
    dataset_id="wisdm_v1_1_stats24_v1",
    input_dim=24,
    num_classes=6,
    preprocessing_artifact_id=(
        "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1"
    ),
    preprocessing_artifact_sha256=(
        "707ee3abf755a9601263d16e5f1ce8e5e9096f956f42da4d42cf57e29e6bf32f"
    ),
    source_reference_sha256=(
        "9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e"
    ),
    protocol_id="wisdm_v1_1_stats24_public_split_v1",
    data_preparation_implementation_id=(
        "unitdp.benchmark_data_v2.prepare_wisdm_v2"
    ),
    train_owners=25,
    test_owners=11,
    mapping_columns=(
        "scenario",
        "window_id",
        "owner_id",
        "owner_ids",
        "start",
        "end",
        "row_index",
        "label",
    ),
    mapping_label_to_index=(
        ("Downstairs", 0),
        ("Jogging", 1),
        ("Sitting", 2),
        ("Standing", 3),
        ("Upstairs", 4),
        ("Walking", 5),
    ),
    train_rows=7049,
    test_rows=3331,
    train_transformed_sha256=(
        "e8a962fbbf2fbb33e8211915af3f65184adc148821286e206e20afc6ea5e558e"
    ),
    test_transformed_sha256=(
        "77cdfeb09546a0a3a65fa7fb8f9b5e052fa69d7ec64ea78287224c32bfd7f841"
    ),
    train_labels_sha256=(
        "0bef5d610c0e7b698666b474dfb730013e095430debf251461fb5decffcf64de"
    ),
    test_labels_sha256=(
        "2dd4f586eeb8b731cadfb06a26353dc87dec9b31068bd067be66333189912136"
    ),
    train_mapping_canonical_sha256=(
        "2656712974b95b15fedf41d4049d8587089e2a7451d002b73f92166cccab0e67"
    ),
    test_mapping_canonical_sha256=(
        "603bde6cf1a9f5f90498b1c8b2c00a83b2349b1d9178224ce5dd060536f592f5"
    ),
    owner_sample_rate=8.0 / 25.0,
    total_steps=20,
    clip_norm=1.0,
    noise_multiplier=1.2510256308972472,
    update_denominator=8.0,
    max_windows_per_owner=16,
    windows_per_owner_per_step=8,
    learning_rate=0.05,
    model_id="linear_24x6",
    class_weights=None,
)


SEPSIS_AAAI27_V1 = BenchmarkProfileV2(
    profile_id="sepsis_aaai27_v1",
    dataset_id="physionet2019_setA_first400_basic12x6_v1",
    input_dim=240,
    num_classes=2,
    preprocessing_artifact_id=(
        "physionet2019_setA_first400_basic12x6_public_scaler_v1"
    ),
    preprocessing_artifact_sha256=(
        "e2ddd1cbb651e5ef5629bdb3ba0b885e5c9cead7c6b8afd9d5d9490e463b210b"
    ),
    source_reference_sha256=(
        "b73db6944d7ab58d5f716f714564df9c7613e0eac02cea2f6964dd698ccc27ae"
    ),
    protocol_id="physionet2019_setA_first400_basic12x6_v1",
    data_preparation_implementation_id=(
        "unitdp.benchmark_data_v2.prepare_sepsis_v2"
    ),
    train_owners=312,
    test_owners=78,
    mapping_columns=(
        "scenario",
        "window_id",
        "owner_id",
        "owner_ids",
        "start",
        "end",
        "row_index",
        "label",
        "patient_file",
    ),
    mapping_label_to_index=(("0", 0), ("1", 1)),
    train_rows=1601,
    test_rows=398,
    train_transformed_sha256=(
        "485042b00fac0e217f975987f5056332d25912186f5a72cdc595b588c94b5bc5"
    ),
    test_transformed_sha256=(
        "1fbffcc4ac9cd115829b0ca5fa25fa923f11df4adc51bbcf263a2366f2cb721c"
    ),
    train_labels_sha256=(
        "a2ba32dcdb95fb48ec12f3ffb4309c824ea005c5f356a09316d0307ae73f73a0"
    ),
    test_labels_sha256=(
        "dbe01710fdb4edbde341eb1f8070ef5637a65ec72d89176f0e8d1597cd435722"
    ),
    train_mapping_canonical_sha256=(
        "676141e015a63a57db96d4bf55597dd46c8d5fbfe142eb4e66192b25abdccd79"
    ),
    test_mapping_canonical_sha256=(
        "882f47fd914d0f98213bb667378f730e47c71c004d29ceb6c232ba3beb772808"
    ),
    owner_sample_rate=0.10,
    total_steps=50,
    clip_norm=1.0,
    noise_multiplier=0.8573442266487843,
    update_denominator=32.0,
    max_windows_per_owner=8,
    windows_per_owner_per_step=8,
    learning_rate=0.03,
    model_id="linear_240x2",
    class_weights=(
        0.5125850340136054,
        20.364864864864863,
    ),
)


BENCHMARK_PROFILES_V2 = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (
            UCI_HAR_AAAI27_V1,
            WISDM_AAAI27_V1,
            SEPSIS_AAAI27_V1,
        )
    }
)


def get_benchmark_profile_v2(profile_id: str) -> BenchmarkProfileV2:
    """Return an exact registered benchmark profile."""

    try:
        return BENCHMARK_PROFILES_V2[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported V2 benchmark profile: {profile_id!r}") from exc
