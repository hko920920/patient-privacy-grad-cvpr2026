"""Deterministic public/private artifact writers for V2 research executions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from unitdp.owner_poisson_v2 import (
    OwnerPoissonExecutionResultV2,
    OwnerPoissonExecutionV2Error,
    _model_state_sha256,
    _sha256_payload,
)
from unitdp.release_artifacts import assert_public_certificate_redacted


MODEL_ARTIFACT_SCHEMA_V2 = "unitdp.linear_model_artifact.v2"
PUBLIC_BUNDLE_SCHEMA_V2 = "unitdp.owner_poisson_run_bundle_public.v2"


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OwnerPoissonExecutionV2Error(
                f"Duplicate model-artifact JSON key: {key!r}"
            )
        result[key] = value
    return result


def build_model_artifact_v2(model: nn.Module) -> dict[str, Any]:
    """Encode an exact float32 linear model without pickle."""

    if not isinstance(model, nn.Linear) or model.bias is None:
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact supports a bias-enabled nn.Linear only"
        )
    tensors: list[dict[str, Any]] = []
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if value.dtype != torch.float32:
            raise OwnerPoissonExecutionV2Error(
                "V2 model artifact requires float32 tensors"
            )
        array = value.numpy().astype("<f4", copy=False)
        tensors.append(
            {
                "name": name,
                "dtype": "float32_le",
                "shape": list(array.shape),
                "data_hex": array.tobytes(order="C").hex(),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": MODEL_ARTIFACT_SCHEMA_V2,
        "model_type": "linear",
        "input_dim": int(model.in_features),
        "num_classes": int(model.out_features),
        "bias": True,
        "state_sha256": _model_state_sha256(model),
        "tensors": tensors,
    }
    payload["payload_sha256"] = _sha256_payload(payload)
    return payload


def load_model_artifact_v2(path: str | Path) -> nn.Linear:
    """Load and verify the deterministic non-pickle V2 model format."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OwnerPoissonExecutionV2Error:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnerPoissonExecutionV2Error(
            f"Could not load V2 model artifact {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact must be a JSON object"
        )
    required = {
        "schema_version",
        "model_type",
        "input_dim",
        "num_classes",
        "bias",
        "state_sha256",
        "tensors",
        "payload_sha256",
    }
    if set(raw) != required:
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact fields do not match the schema"
        )
    if raw["schema_version"] != MODEL_ARTIFACT_SCHEMA_V2:
        raise OwnerPoissonExecutionV2Error(
            "Unsupported V2 model artifact schema"
        )
    if raw["model_type"] != "linear" or raw["bias"] is not True:
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact requires a bias-enabled linear model"
        )
    payload_without_hash = dict(raw)
    reported_payload_hash = str(
        payload_without_hash.pop("payload_sha256")
    )
    if _sha256_payload(payload_without_hash) != reported_payload_hash:
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact payload hash mismatch"
        )
    input_dim = raw["input_dim"]
    num_classes = raw["num_classes"]
    if (
        isinstance(input_dim, bool)
        or not isinstance(input_dim, int)
        or input_dim <= 0
        or isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact dimensions are invalid"
        )
    tensors_raw = raw["tensors"]
    if not isinstance(tensors_raw, list):
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact tensors must be a list"
        )
    expected_shapes = {
        "bias": (num_classes,),
        "weight": (num_classes, input_dim),
    }
    decoded: dict[str, np.ndarray] = {}
    for index, item_raw in enumerate(tensors_raw):
        if not isinstance(item_raw, dict) or set(item_raw) != {
            "name",
            "dtype",
            "shape",
            "data_hex",
        }:
            raise OwnerPoissonExecutionV2Error(
                f"Malformed tensor record at index {index}"
            )
        name = item_raw["name"]
        if name not in expected_shapes or name in decoded:
            raise OwnerPoissonExecutionV2Error(
                f"Unexpected or duplicate tensor name: {name!r}"
            )
        if item_raw["dtype"] != "float32_le":
            raise OwnerPoissonExecutionV2Error(
                "V2 model artifact tensor dtype must be float32_le"
            )
        if item_raw["shape"] != list(expected_shapes[name]):
            raise OwnerPoissonExecutionV2Error(
                f"Tensor {name!r} has the wrong shape"
            )
        data_hex = item_raw["data_hex"]
        if not isinstance(data_hex, str):
            raise OwnerPoissonExecutionV2Error(
                f"Tensor {name!r} data_hex must be a string"
            )
        try:
            data = bytes.fromhex(data_hex)
        except ValueError as exc:
            raise OwnerPoissonExecutionV2Error(
                f"Tensor {name!r} contains invalid hexadecimal data"
            ) from exc
        expected_bytes = int(np.prod(expected_shapes[name])) * 4
        if len(data) != expected_bytes:
            raise OwnerPoissonExecutionV2Error(
                f"Tensor {name!r} byte length is invalid"
            )
        values = np.frombuffer(data, dtype="<f4").reshape(
            expected_shapes[name]
        )
        if not np.isfinite(values).all():
            raise OwnerPoissonExecutionV2Error(
                f"Tensor {name!r} contains non-finite values"
            )
        decoded[name] = values.copy()
    if set(decoded) != set(expected_shapes):
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact is missing required tensors"
        )
    model = nn.Linear(
        input_dim,
        num_classes,
        bias=True,
        device="cpu",
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(decoded["weight"]))
        model.bias.copy_(torch.from_numpy(decoded["bias"]))
    if _model_state_sha256(model) != raw["state_sha256"]:
        raise OwnerPoissonExecutionV2Error(
            "V2 model artifact state hash mismatch"
        )
    return model


@dataclass(frozen=True)
class WrittenExecutionArtifactsV2:
    model_path: Path
    public_execution_path: Path
    public_bundle_path: Path
    private_manifest_path: Path | None
    model_file_sha256: str
    public_execution_file_sha256: str
    public_bundle_payload_sha256: str


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_execution_artifacts_v2(
    result: OwnerPoissonExecutionResultV2,
    output_dir: str | Path,
    *,
    include_private_manifest: bool = False,
    overwrite: bool = False,
) -> WrittenExecutionArtifactsV2:
    """Write a public model/run bundle and an opt-in private manifest."""

    if not isinstance(result, OwnerPoissonExecutionResultV2):
        raise TypeError(
            "write_execution_artifacts_v2 requires "
            "OwnerPoissonExecutionResultV2"
        )
    result.assert_model_integrity()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model_v2.json"
    public_execution_path = target / "public_execution_v2.json"
    public_bundle_path = target / "public_bundle_v2.json"
    private_manifest_path = (
        target / "private_execution_manifest_v2.json"
        if include_private_manifest
        else None
    )
    paths = [model_path, public_execution_path, public_bundle_path]
    if private_manifest_path is not None:
        paths.append(private_manifest_path)
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    model_payload = build_model_artifact_v2(result.model)
    public_execution = result.public_payload()
    _write_json(model_path, model_payload, overwrite=overwrite)
    _write_json(
        public_execution_path,
        public_execution,
        overwrite=overwrite,
    )
    model_file_sha256 = _file_sha256(model_path)
    public_execution_file_sha256 = _file_sha256(public_execution_path)
    bundle: dict[str, Any] = {
        "schema_version": PUBLIC_BUNDLE_SCHEMA_V2,
        "visibility": "public",
        "release_status": public_execution["release_status"],
        "model_artifact": {
            "filename": model_path.name,
            "file_sha256": model_file_sha256,
            "payload_sha256": model_payload["payload_sha256"],
            "state_sha256": model_payload["state_sha256"],
        },
        "public_execution": {
            "filename": public_execution_path.name,
            "file_sha256": public_execution_file_sha256,
            "payload_sha256": public_execution[
                "public_execution_sha256"
            ],
        },
    }
    bundle["public_bundle_payload_sha256"] = _sha256_payload(bundle)
    assert_public_certificate_redacted(bundle)
    _write_json(public_bundle_path, bundle, overwrite=overwrite)
    if private_manifest_path is not None:
        _write_json(
            private_manifest_path,
            result.private_manifest(),
            overwrite=overwrite,
        )
    return WrittenExecutionArtifactsV2(
        model_path=model_path,
        public_execution_path=public_execution_path,
        public_bundle_path=public_bundle_path,
        private_manifest_path=private_manifest_path,
        model_file_sha256=model_file_sha256,
        public_execution_file_sha256=public_execution_file_sha256,
        public_bundle_payload_sha256=bundle[
            "public_bundle_payload_sha256"
        ],
    )


def write_private_execution_manifest_v2(
    result: OwnerPoissonExecutionResultV2,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Explicitly write the private V2 record outside a public bundle."""

    if not isinstance(result, OwnerPoissonExecutionResultV2):
        raise TypeError(
            "write_private_execution_manifest_v2 requires "
            "OwnerPoissonExecutionResultV2"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        target,
        result.private_manifest(),
        overwrite=overwrite,
    )
    return target
