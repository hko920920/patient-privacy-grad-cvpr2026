"""Independent interpreter for the owner-Poisson V2 declarative specification.

Independence is deliberate: this module must not import production ``unitdp``
modules, registry objects, parsers, mapping classes, preprocessing classes, or
accountants.  It uses a static JSON Schema plus frozen profile bindings and is
compared against the production compiler by generative differential tests.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator


ORACLE_SCHEMA_VERSION = "unitdp.owner_poisson_contract_oracle.v1"
_INTEGER_TOKEN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_STRICT_INTEGER_PATHS = (
    ("mechanism", "total_steps"),
    ("contribution_policy", "max_windows_per_owner"),
    ("contribution_policy", "windows_per_owner_per_step"),
    ("data", "input_dim"),
    ("data", "num_classes"),
    ("data", "preprocessing", "input_dim"),
    ("data", "public_protocol", "train_owners"),
    ("data", "public_protocol", "test_owners"),
)


class OracleInputError(ValueError):
    """Raised when an oracle input cannot be loaded unambiguously."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader with duplicate-key rejection independent of production."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise OracleInputError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise OracleInputError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OracleInputError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class OracleResult:
    """Result of one independent oracle judgment."""

    accepted: bool
    stage: str
    reason: str
    profile_id: str | None = None
    details: tuple[tuple[str, object], ...] = ()

    @classmethod
    def accept(
        cls,
        *,
        stage: str,
        profile_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> "OracleResult":
        return cls(
            accepted=True,
            stage=stage,
            reason="accepted",
            profile_id=profile_id,
            details=tuple(sorted((details or {}).items())),
        )

    @classmethod
    def reject(
        cls,
        *,
        stage: str,
        reason: str,
        profile_id: str | None = None,
    ) -> "OracleResult":
        return cls(
            accepted=False,
            stage=stage,
            reason=reason,
            profile_id=profile_id,
        )


def canonical_payload_sha256(value: object) -> str:
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


def load_oracle_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OracleInputError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleInputError(f"Could not load oracle spec {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OracleInputError("Oracle spec must be a JSON object")
    Draft202012Validator.check_schema(raw)
    extension = raw.get("x-unitdp-oracle")
    if not isinstance(extension, dict):
        raise OracleInputError("Oracle spec is missing x-unitdp-oracle")
    if extension.get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise OracleInputError("Unsupported oracle schema version")
    profiles = extension.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise OracleInputError("Oracle spec must register profiles")
    return raw


def load_contract_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            raw = json.loads(text, object_pairs_hook=_unique_json_object)
        else:
            raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except OracleInputError:
        raise
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise OracleInputError(f"Could not load contract {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OracleInputError("Contract must be a mapping")
    return raw


def _walk_scalars(value: object, path: str = "$") -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _walk_scalars(value[key], f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_scalars(item, f"{path}[{index}]")
        return
    yield path, value


def _first_nonfinite(value: object) -> str | None:
    for path, item in _walk_scalars(value):
        if isinstance(item, float) and not math.isfinite(item):
            return path
    return None


def _first_noninteger_at_strict_path(
    value: dict[str, Any],
) -> str | None:
    """Enforce parser-level integer types beyond JSON Schema's numeric model."""

    for parts in _STRICT_INTEGER_PATHS:
        current: object = value
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None and (
            isinstance(current, bool) or not isinstance(current, int)
        ):
            return "$." + ".".join(parts)
    return None


def _normalized_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _numeric_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and float(actual) == float(expected)
    return False


def _registered_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _registered_equal(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _registered_equal(left, right)
                for left, right in zip(actual, expected)
            )
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _numeric_equal(actual, expected)
    return type(actual) is type(expected) and actual == expected


def _profile_binding(raw: dict[str, Any]) -> dict[str, Any]:
    privacy = raw["privacy"]
    mechanism = raw["mechanism"]
    policy = raw["contribution_policy"]
    optimization = raw["optimization"]
    data = raw["data"]
    return {
        "privacy": {
            "target_epsilon": _normalized_float(
                privacy["target_epsilon"]
            ),
            "delta": _normalized_float(privacy["delta"]),
        },
        "mechanism": {
            "owner_sample_rate": _normalized_float(
                mechanism["owner_sample_rate"]
            ),
            "total_steps": mechanism["total_steps"],
            "clip_norm": _normalized_float(mechanism["clip_norm"]),
            "noise_multiplier": _normalized_float(
                mechanism["noise_multiplier"]
            ),
            "update_denominator": _normalized_float(
                mechanism["update_denominator"]
            ),
        },
        "contribution_policy": {
            "max_windows_per_owner": policy["max_windows_per_owner"],
            "windows_per_owner_per_step": (
                policy["windows_per_owner_per_step"]
            ),
        },
        "optimization": {
            "learning_rate": _normalized_float(
                optimization["learning_rate"]
            ),
            "model_id": optimization["model_id"],
            "class_weights": (
                None
                if optimization["class_weights"] is None
                else [
                    _normalized_float(value)
                    for value in optimization["class_weights"]
                ]
            ),
        },
        "data": copy.deepcopy(data),
    }


def _normalized_public_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    payload["privacy"]["target_epsilon"] = _normalized_float(
        payload["privacy"]["target_epsilon"]
    )
    payload["privacy"]["delta"] = _normalized_float(
        payload["privacy"]["delta"]
    )
    for key in (
        "owner_sample_rate",
        "clip_norm",
        "noise_multiplier",
        "update_denominator",
    ):
        payload["mechanism"][key] = _normalized_float(
            payload["mechanism"][key]
        )
    payload["optimization"]["learning_rate"] = _normalized_float(
        payload["optimization"]["learning_rate"]
    )
    weights = payload["optimization"]["class_weights"]
    if weights is not None:
        payload["optimization"]["class_weights"] = [
            _normalized_float(value) for value in weights
        ]
    return payload


def validate_registered_research_contract(
    raw: object,
    spec: dict[str, Any],
) -> OracleResult:
    """Judge the exact registered research contract without production code."""

    if not isinstance(raw, dict):
        return OracleResult.reject(stage="contract", reason="not_object")
    nonfinite_path = _first_nonfinite(raw)
    if nonfinite_path is not None:
        return OracleResult.reject(
            stage="contract",
            reason=f"nonfinite:{nonfinite_path}",
        )
    noninteger_path = _first_noninteger_at_strict_path(raw)
    if noninteger_path is not None:
        return OracleResult.reject(
            stage="contract",
            reason=f"noninteger:{noninteger_path}",
        )
    validator = Draft202012Validator(spec)
    errors = sorted(
        validator.iter_errors(raw),
        key=lambda item: (
            tuple(str(part) for part in item.absolute_path),
            item.validator or "",
            item.message,
        ),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "$"
        return OracleResult.reject(
            stage="contract",
            reason=f"schema:{location}:{first.validator}",
        )

    profile_id = raw["data"]["benchmark_profile_id"]
    profiles = spec["x-unitdp-oracle"]["profiles"]
    profile = profiles.get(profile_id)
    if not isinstance(profile, dict):
        return OracleResult.reject(
            stage="profile",
            reason="unregistered_profile",
            profile_id=str(profile_id),
        )

    if raw["execution"] != {
        "profile": "research_benchmark",
        "rng_backend": "research_default",
        "implementation_id": "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2",
        "random_coins_public": False,
    }:
        return OracleResult.reject(
            stage="execution",
            reason="research_executor_not_ready",
            profile_id=profile_id,
        )

    policy = raw["contribution_policy"]
    if (
        policy["windows_per_owner_per_step"]
        > policy["max_windows_per_owner"]
    ):
        return OracleResult.reject(
            stage="contract",
            reason="per_step_exceeds_cap",
            profile_id=profile_id,
        )
    weights = raw["optimization"]["class_weights"]
    if weights is not None and len(weights) != raw["data"]["num_classes"]:
        return OracleResult.reject(
            stage="contract",
            reason="class_weight_length",
            profile_id=profile_id,
        )
    preprocessing = raw["data"]["preprocessing"]
    protocol = raw["data"]["public_protocol"]
    if preprocessing["input_dim"] != raw["data"]["input_dim"]:
        return OracleResult.reject(
            stage="contract",
            reason="preprocessor_input_dim",
            profile_id=profile_id,
        )
    if (
        preprocessing["source_reference_sha256"]
        != protocol["source_reference_sha256"]
    ):
        return OracleResult.reject(
            stage="contract",
            reason="source_reference_mismatch",
            profile_id=profile_id,
        )
    if raw["privacy"]["delta"] >= 1.0 / protocol["train_owners"]:
        return OracleResult.reject(
            stage="contract",
            reason="delta_owner_count",
            profile_id=profile_id,
        )

    actual_binding = _profile_binding(raw)
    expected_binding = profile.get("binding")
    if not _registered_equal(actual_binding, expected_binding):
        return OracleResult.reject(
            stage="profile",
            reason="profile_binding_mismatch",
            profile_id=profile_id,
        )

    normalized = _normalized_public_payload(raw)
    contract_sha256 = canonical_payload_sha256(normalized)
    if contract_sha256 != profile.get("public_contract_sha256"):
        return OracleResult.reject(
            stage="profile",
            reason="public_contract_digest_mismatch",
            profile_id=profile_id,
        )
    return OracleResult.accept(
        stage="contract",
        profile_id=profile_id,
        details={"public_contract_sha256": contract_sha256},
    )


def _parse_mapping_integer(value: object) -> int:
    if not isinstance(value, str) or not _INTEGER_TOKEN.fullmatch(value):
        raise ValueError
    return int(value)


def validate_mapping_file(
    path: str | Path,
    *,
    profile_id: str,
    spec: dict[str, Any],
) -> OracleResult:
    """Independently validate the compiler-visible mapping predicates."""

    profiles = spec["x-unitdp-oracle"]["profiles"]
    profile = profiles.get(profile_id)
    if not isinstance(profile, dict):
        return OracleResult.reject(
            stage="mapping",
            reason="unregistered_profile",
            profile_id=profile_id,
        )
    expected_columns = tuple(profile["mapping_columns"])
    source = Path(path)
    rows: list[dict[str, str]] = []
    try:
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return OracleResult.reject(
                    stage="mapping",
                    reason="missing_header",
                    profile_id=profile_id,
                )
            observed_columns = tuple(reader.fieldnames)
            if observed_columns != expected_columns:
                return OracleResult.reject(
                    stage="mapping",
                    reason="column_mismatch",
                    profile_id=profile_id,
                )
            if len(observed_columns) != len(set(observed_columns)):
                return OracleResult.reject(
                    stage="mapping",
                    reason="duplicate_column",
                    profile_id=profile_id,
                )
            for position, row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"column_count:{position}",
                        profile_id=profile_id,
                    )
                values = {key: str(value) for key, value in row.items()}
                if any(value != value.strip() for value in values.values()):
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"surrounding_whitespace:{position}",
                        profile_id=profile_id,
                    )
                if (
                    not values["scenario"]
                    or not values["window_id"]
                    or not values["owner_id"]
                    or not values["owner_ids"]
                    or not values["label"]
                ):
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"empty_required_value:{position}",
                        profile_id=profile_id,
                    )
                if (
                    values["owner_ids"] != values["owner_id"]
                    or ";" in values["owner_ids"]
                ):
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"owner_attribution:{position}",
                        profile_id=profile_id,
                    )
                for column in expected_columns:
                    if column not in {
                        "scenario",
                        "window_id",
                        "owner_id",
                        "owner_ids",
                        "start",
                        "end",
                        "row_index",
                        "label",
                    } and not values[column]:
                        return OracleResult.reject(
                            stage="mapping",
                            reason=f"empty_extra_value:{position}:{column}",
                            profile_id=profile_id,
                        )
                try:
                    start = _parse_mapping_integer(values["start"])
                    end = _parse_mapping_integer(values["end"])
                    _parse_mapping_integer(values["row_index"])
                except ValueError:
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"integer_token:{position}",
                        profile_id=profile_id,
                    )
                if end <= start:
                    return OracleResult.reject(
                        stage="mapping",
                        reason=f"nonpositive_span:{position}",
                        profile_id=profile_id,
                    )
                rows.append(values)
    except (OSError, csv.Error) as exc:
        return OracleResult.reject(
            stage="mapping",
            reason=f"io:{type(exc).__name__}",
            profile_id=profile_id,
        )
    if not rows:
        return OracleResult.reject(
            stage="mapping",
            reason="empty_mapping",
            profile_id=profile_id,
        )
    window_ids = [row["window_id"] for row in rows]
    row_indices = [int(row["row_index"]) for row in rows]
    if len(window_ids) != len(set(window_ids)):
        return OracleResult.reject(
            stage="mapping",
            reason="duplicate_window_id",
            profile_id=profile_id,
        )
    if len(row_indices) != len(set(row_indices)):
        return OracleResult.reject(
            stage="mapping",
            reason="duplicate_row_index",
            profile_id=profile_id,
        )
    if sorted(row_indices) != list(range(len(row_indices))):
        return OracleResult.reject(
            stage="mapping",
            reason="noncontiguous_row_index",
            profile_id=profile_id,
        )
    return OracleResult.accept(
        stage="mapping",
        profile_id=profile_id,
        details={
            "record_count": len(rows),
            "owner_count": len({row["owner_id"] for row in rows}),
        },
    )


def validate_mapping_array_binding(
    path: str | Path,
    *,
    profile_id: str,
    feature_shape: object,
    labels: object,
    spec: dict[str, Any],
    require_registered_train_rows: bool = False,
) -> OracleResult:
    """Bind a strict mapping to concrete feature rows and integer labels."""

    mapping_result = validate_mapping_file(
        path,
        profile_id=profile_id,
        spec=spec,
    )
    if not mapping_result.accepted:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason=f"invalid_mapping:{mapping_result.reason}",
            profile_id=profile_id,
        )
    profile = spec["x-unitdp-oracle"]["profiles"].get(profile_id)
    assert isinstance(profile, dict)
    if (
        not isinstance(feature_shape, (tuple, list))
        or len(feature_shape) != 2
    ):
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="feature_shape",
            profile_id=profile_id,
        )
    dimensions: list[int] = []
    for value in feature_shape:
        if isinstance(value, bool):
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="feature_shape",
                profile_id=profile_id,
            )
        try:
            dimension = operator.index(value)
        except TypeError:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="feature_shape",
                profile_id=profile_id,
            )
        if dimension < 0:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="feature_shape",
                profile_id=profile_id,
            )
        dimensions.append(dimension)
    row_count, input_dim = dimensions
    expected_input_dim = profile["binding"]["data"]["input_dim"]
    if input_dim != expected_input_dim:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="input_dim",
            profile_id=profile_id,
        )
    mapping_record_count = int(
        dict(mapping_result.details)["record_count"]
    )
    if row_count != mapping_record_count:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="mapping_row_count",
            profile_id=profile_id,
        )
    registered_rows = profile.get("public_benchmark_rows")
    if (
        not isinstance(registered_rows, dict)
        or isinstance(registered_rows.get("train"), bool)
        or not isinstance(registered_rows.get("train"), int)
        or registered_rows["train"] <= 0
    ):
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="invalid_registered_train_rows",
            profile_id=profile_id,
        )
    if (
        require_registered_train_rows
        and row_count != registered_rows["train"]
    ):
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="registered_train_rows",
            profile_id=profile_id,
        )

    if (
        isinstance(labels, (str, bytes, bytearray, dict))
        or getattr(labels, "ndim", 1) != 1
    ):
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="labels_shape",
            profile_id=profile_id,
        )
    try:
        label_values = list(labels)  # type: ignore[arg-type]
    except TypeError:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="labels_shape",
            profile_id=profile_id,
        )
    if len(label_values) != row_count:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="labels_shape",
            profile_id=profile_id,
        )
    normalized_labels: list[int] = []
    num_classes = profile["binding"]["data"]["num_classes"]
    for value in label_values:
        if isinstance(value, bool):
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="labels_dtype",
                profile_id=profile_id,
            )
        try:
            label = operator.index(value)
        except TypeError:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="labels_dtype",
                profile_id=profile_id,
            )
        if label < 0 or label >= num_classes:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason="labels_range",
                profile_id=profile_id,
            )
        normalized_labels.append(label)

    label_to_index = profile.get("mapping_label_to_index")
    if (
        not isinstance(label_to_index, dict)
        or not label_to_index
        or any(
            not isinstance(label_name, str)
            or isinstance(label_index, bool)
            or not isinstance(label_index, int)
            or label_index < 0
            or label_index >= num_classes
            for label_name, label_index in label_to_index.items()
        )
    ):
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason="invalid_label_registration",
            profile_id=profile_id,
        )

    source = Path(path)
    try:
        with source.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return OracleResult.reject(
            stage="mapping_array_binding",
            reason=f"io:{type(exc).__name__}",
            profile_id=profile_id,
        )
    for position, row in enumerate(rows):
        label_name = row["label"]
        if label_name not in label_to_index:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason=f"unregistered_mapping_label:{position}",
                profile_id=profile_id,
            )
        row_index = _parse_mapping_integer(row["row_index"])
        if label_to_index[label_name] != normalized_labels[row_index]:
            return OracleResult.reject(
                stage="mapping_array_binding",
                reason=f"mapping_label_mismatch:{position}",
                profile_id=profile_id,
            )
    return OracleResult.accept(
        stage="mapping_array_binding",
        profile_id=profile_id,
        details={
            "input_dim": input_dim,
            "record_count": row_count,
            "registered_train_rows_enforced": (
                require_registered_train_rows
            ),
        },
    )


def _preprocessor_payload_sha256(raw: dict[str, Any]) -> str:
    payload = dict(raw)
    payload.pop("payload_sha256", None)
    return canonical_payload_sha256(payload)


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_preprocessor_artifact(
    path: str | Path,
    *,
    contract: dict[str, Any],
    spec: dict[str, Any],
) -> OracleResult:
    """Validate the fixed preprocessor without production loader reuse."""

    contract_result = validate_registered_research_contract(contract, spec)
    if not contract_result.accepted:
        return OracleResult.reject(
            stage="preprocessor",
            reason="invalid_contract",
            profile_id=contract_result.profile_id,
        )
    profile_id = contract_result.profile_id
    assert profile_id is not None
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OracleInputError:
        return OracleResult.reject(
            stage="preprocessor",
            reason="duplicate_key",
            profile_id=profile_id,
        )
    except (OSError, json.JSONDecodeError) as exc:
        return OracleResult.reject(
            stage="preprocessor",
            reason=f"io:{type(exc).__name__}",
            profile_id=profile_id,
        )
    if not isinstance(raw, dict):
        return OracleResult.reject(
            stage="preprocessor",
            reason="not_object",
            profile_id=profile_id,
        )
    required = {
        "schema",
        "artifact_id",
        "scope",
        "input_dim",
        "transform",
        "mean",
        "scale",
        "fit_on_protected_data",
        "source",
        "payload_sha256",
    }
    if not required.issubset(raw):
        return OracleResult.reject(
            stage="preprocessor",
            reason="missing_field",
            profile_id=profile_id,
        )
    if (
        raw["schema"] != "unitdp.fixed_affine_preprocessor.v1"
        or raw["scope"] != "public_fixed"
        or raw["transform"] != "standard_scaler"
        or raw["fit_on_protected_data"] is not False
    ):
        return OracleResult.reject(
            stage="preprocessor",
            reason="fixed_transform_semantics",
            profile_id=profile_id,
        )
    if (
        isinstance(raw["input_dim"], bool)
        or not isinstance(raw["input_dim"], int)
        or raw["input_dim"] <= 0
    ):
        return OracleResult.reject(
            stage="preprocessor",
            reason="input_dim",
            profile_id=profile_id,
        )
    input_dim = raw["input_dim"]
    if (
        not isinstance(raw["mean"], list)
        or not isinstance(raw["scale"], list)
        or len(raw["mean"]) != input_dim
        or len(raw["scale"]) != input_dim
        or not all(_is_number(value) for value in raw["mean"])
        or not all(_is_number(value) for value in raw["scale"])
        or any(float(value) <= 0 for value in raw["scale"])
    ):
        return OracleResult.reject(
            stage="preprocessor",
            reason="affine_parameters",
            profile_id=profile_id,
        )
    observed_payload_sha256 = _preprocessor_payload_sha256(raw)
    if raw["payload_sha256"] != observed_payload_sha256:
        return OracleResult.reject(
            stage="preprocessor",
            reason="self_digest",
            profile_id=profile_id,
        )
    binding = contract["data"]["preprocessing"]
    expected_binding = {
        "scope": raw["scope"],
        "schema": raw["schema"],
        "artifact_id": raw["artifact_id"],
        "artifact_sha256": observed_payload_sha256,
        "source_reference_sha256": (
            raw["source"].get("sha256")
            if isinstance(raw["source"], dict)
            else None
        ),
        "input_dim": input_dim,
        "transform": raw["transform"],
        "fit_on_protected_data": raw["fit_on_protected_data"],
    }
    if not _registered_equal(binding, expected_binding):
        return OracleResult.reject(
            stage="preprocessor",
            reason="contract_binding",
            profile_id=profile_id,
        )
    source_metadata = raw["source"]
    if (
        not isinstance(source_metadata, dict)
        or source_metadata.get("is_public_reference") is not True
        or not isinstance(source_metadata.get("reference"), str)
        or not source_metadata["reference"]
        or not isinstance(source_metadata.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_metadata["sha256"])
    ):
        return OracleResult.reject(
            stage="preprocessor",
            reason="public_source",
            profile_id=profile_id,
        )
    return OracleResult.accept(
        stage="preprocessor",
        profile_id=profile_id,
        details={"payload_sha256": observed_payload_sha256},
    )


def validate_registered_source_bundle(
    repository_root: str | Path,
    spec: dict[str, Any],
) -> OracleResult:
    """Recompute the frozen source bundle without production helper reuse."""

    root = Path(repository_root).resolve()
    registration = spec["x-unitdp-oracle"]["execution_source_bundle"]
    files = registration.get("files")
    if not isinstance(files, list) or not files:
        return OracleResult.reject(stage="source", reason="missing_file_list")
    observed: dict[str, str] = {}
    for relative in files:
        if not isinstance(relative, str) or not relative:
            return OracleResult.reject(stage="source", reason="invalid_path")
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return OracleResult.reject(stage="source", reason="path_escape")
        if not candidate.is_file():
            return OracleResult.reject(
                stage="source",
                reason=f"missing:{relative}",
            )
        observed[relative] = _file_sha256(candidate)
    bundle_sha256 = canonical_payload_sha256(observed)
    if bundle_sha256 != registration.get("sha256"):
        return OracleResult.reject(
            stage="source",
            reason="bundle_digest_mismatch",
        )
    return OracleResult.accept(
        stage="source",
        details={
            "file_count": len(observed),
            "source_bundle_sha256": bundle_sha256,
        },
    )


def validate_accountant_environment(
    spec: dict[str, Any],
) -> OracleResult:
    """Check frozen accountant package versions without production registry use."""

    registrations = spec["x-unitdp-oracle"].get("accountant_versions")
    if not isinstance(registrations, dict) or not registrations:
        return OracleResult.reject(
            stage="accountant_environment",
            reason="missing_version_registration",
        )
    observed: dict[str, str] = {}
    for distribution, expected in sorted(registrations.items()):
        if (
            not isinstance(distribution, str)
            or not distribution
            or not isinstance(expected, str)
            or not expected
        ):
            return OracleResult.reject(
                stage="accountant_environment",
                reason="invalid_version_registration",
            )
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            return OracleResult.reject(
                stage="accountant_environment",
                reason=f"missing_distribution:{distribution}",
            )
        observed[distribution] = installed
        if installed != expected:
            return OracleResult.reject(
                stage="accountant_environment",
                reason=(
                    f"version_mismatch:{distribution}:"
                    f"{expected}:{installed}"
                ),
            )
    return OracleResult.accept(
        stage="accountant_environment",
        details={"versions": tuple(sorted(observed.items()))},
    )


def validate_registered_accounting(
    contract: dict[str, Any],
    spec: dict[str, Any],
) -> OracleResult:
    """Recompute the registered privacy cost through both direct library APIs."""

    contract_result = validate_registered_research_contract(contract, spec)
    if not contract_result.accepted:
        return OracleResult.reject(
            stage="accounting",
            reason="invalid_contract",
            profile_id=contract_result.profile_id,
        )
    environment_result = validate_accountant_environment(spec)
    if not environment_result.accepted:
        return OracleResult.reject(
            stage="accounting",
            reason="invalid_accountant_environment",
            profile_id=contract_result.profile_id,
        )
    registration = spec["x-unitdp-oracle"].get("accounting_judgment")
    if not isinstance(registration, dict):
        return OracleResult.reject(
            stage="accounting",
            reason="missing_accounting_judgment",
            profile_id=contract_result.profile_id,
        )
    if (
        registration.get("event") != "poisson_sampled_gaussian"
        or registration.get("neighboring_relation")
        != "add_or_remove_one"
    ):
        return OracleResult.reject(
            stage="accounting",
            reason="unsupported_accounting_judgment",
            profile_id=contract_result.profile_id,
        )
    orders = spec["x-unitdp-oracle"].get("rdp_orders")
    if (
        not isinstance(orders, list)
        or not orders
        or any(
            isinstance(order, bool)
            or not isinstance(order, (int, float))
            or not math.isfinite(float(order))
            or float(order) <= 1
            for order in orders
        )
    ):
        return OracleResult.reject(
            stage="accounting",
            reason="invalid_rdp_orders",
            profile_id=contract_result.profile_id,
        )
    try:
        agreement_tolerance = _normalized_float(
            registration["dual_agreement_tolerance"]
        )
        target_tolerance = _normalized_float(
            registration["target_epsilon_tolerance"]
        )
    except (KeyError, TypeError, ValueError):
        return OracleResult.reject(
            stage="accounting",
            reason="invalid_tolerance",
            profile_id=contract_result.profile_id,
        )
    if agreement_tolerance < 0 or target_tolerance < 0:
        return OracleResult.reject(
            stage="accounting",
            reason="negative_tolerance",
            profile_id=contract_result.profile_id,
        )

    # These are direct third-party API calls.  No production accountant
    # helper, registry object, or compiler constant is reused here.
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant as DpAccountingRdpAccountant
    from opacus.accountants import RDPAccountant

    mechanism = contract["mechanism"]
    privacy = contract["privacy"]
    sample_rate = float(mechanism["owner_sample_rate"])
    noise_multiplier = float(mechanism["noise_multiplier"])
    total_steps = mechanism["total_steps"]
    delta = float(privacy["delta"])

    opacus_accountant = RDPAccountant()
    for _ in range(total_steps):
        opacus_accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )
    epsilon_opacus = float(
        opacus_accountant.get_epsilon(delta=delta, alphas=orders)
    )

    dp_accountant = DpAccountingRdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    event = dp_event.PoissonSampledDpEvent(
        sampling_probability=sample_rate,
        event=dp_event.GaussianDpEvent(noise_multiplier),
    )
    dp_accountant.compose(event, count=total_steps)
    epsilon_dp_accounting = float(dp_accountant.get_epsilon(delta))

    gap = abs(epsilon_opacus - epsilon_dp_accounting)
    if gap > agreement_tolerance:
        return OracleResult.reject(
            stage="accounting",
            reason="dual_accountant_disagreement",
            profile_id=contract_result.profile_id,
        )
    if max(epsilon_opacus, epsilon_dp_accounting) > (
        float(privacy["target_epsilon"]) + target_tolerance
    ):
        return OracleResult.reject(
            stage="accounting",
            reason="target_epsilon_exceeded",
            profile_id=contract_result.profile_id,
        )
    return OracleResult.accept(
        stage="accounting",
        profile_id=contract_result.profile_id,
        details={
            "accountant_epsilon_dp_accounting": epsilon_dp_accounting,
            "accountant_epsilon_opacus": epsilon_opacus,
            "dual_accountant_gap": gap,
        },
    )


def registered_public_projection(
    raw: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Return only data-invariant registered fields after oracle acceptance."""

    result = validate_registered_research_contract(raw, spec)
    if not result.accepted or result.profile_id is None:
        raise OracleInputError(
            f"Cannot project rejected contract: {result.stage}:{result.reason}"
        )
    normalized = _normalized_public_payload(raw)
    source = spec["x-unitdp-oracle"]["execution_source_bundle"]
    return {
        "route_id": normalized["route_id"],
        "public_contract_sha256": canonical_payload_sha256(normalized),
        "profile_id": result.profile_id,
        "mechanism": copy.deepcopy(normalized["mechanism"]),
        "accountant": copy.deepcopy(normalized["accountant"]),
        "execution_source_bundle_sha256": source["sha256"],
    }
