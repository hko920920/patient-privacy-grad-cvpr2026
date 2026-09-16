"""Fail-closed loading and application of fixed public preprocessors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FIXED_AFFINE_SCHEMA = "unitdp.fixed_affine_preprocessor.v1"
PUBLIC_FIXED_SCOPE = "public_fixed"


class PreprocessingArtifactError(ValueError):
    """Raised when a preprocessing artifact is missing a required invariant."""


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreprocessingArtifactError(
                f"Duplicate preprocessing JSON key: {key!r}"
            )
        result[key] = value
    return result


def _canonical_payload(raw: dict[str, Any]) -> bytes:
    payload = dict(raw)
    payload.pop("payload_sha256", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def preprocessing_payload_sha256(raw: dict[str, Any]) -> str:
    """Hash an artifact without its self-reported digest field."""

    return hashlib.sha256(_canonical_payload(raw)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class FixedAffinePreprocessor:
    """A fixed affine transform whose parameters are public side information."""

    artifact_id: str
    input_dim: int
    mean: np.ndarray
    scale: np.ndarray
    payload_sha256: str
    source_sha256: str
    source_reference: str
    transform_type: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the registered transform without fitting on the input."""

        array = np.asarray(values)
        if array.ndim != 2:
            raise PreprocessingArtifactError(
                f"Expected a two-dimensional feature matrix, got shape {array.shape}"
            )
        if array.shape[1] != self.input_dim:
            raise PreprocessingArtifactError(
                f"Expected {self.input_dim} features, got {array.shape[1]}"
            )
        if not np.isfinite(array).all():
            raise PreprocessingArtifactError("Input contains non-finite values")

        # The in-place float32 operations reproduce sklearn StandardScaler's
        # transform path for float32 benchmark inputs.
        transformed = array.astype(np.float32, copy=True)
        transformed -= self.mean
        transformed /= self.scale
        if not np.isfinite(transformed).all():
            raise PreprocessingArtifactError("Transform produced non-finite values")
        return transformed

    def contract_binding(self) -> dict[str, Any]:
        """Return the immutable fields that a route contract must bind."""

        return {
            "scope": PUBLIC_FIXED_SCOPE,
            "schema": FIXED_AFFINE_SCHEMA,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.payload_sha256,
            "source_reference_sha256": self.source_sha256,
            "input_dim": self.input_dim,
            "transform": self.transform_type,
            "fit_on_protected_data": False,
        }

    def verify_reference_file(self, path: str | Path) -> None:
        """Verify that a local public reference file matches the manifest."""

        observed = file_sha256(path)
        if observed != self.source_sha256:
            raise PreprocessingArtifactError(
                "Public reference file digest mismatch: "
                f"expected {self.source_sha256}, got {observed}"
            )


def load_fixed_affine_preprocessor(
    path: str | Path,
    *,
    expected_payload_sha256: str | None = None,
) -> FixedAffinePreprocessor:
    """Load a public fixed affine transform and reject malformed artifacts."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except PreprocessingArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingArtifactError(
            f"Could not load preprocessing artifact {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise PreprocessingArtifactError("Preprocessing artifact must be a JSON object")

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
    missing = sorted(required.difference(raw))
    if missing:
        raise PreprocessingArtifactError(
            f"Preprocessing artifact is missing fields: {', '.join(missing)}"
        )
    if raw["schema"] != FIXED_AFFINE_SCHEMA:
        raise PreprocessingArtifactError(
            f"Unsupported preprocessing schema: {raw['schema']!r}"
        )
    if raw["scope"] != PUBLIC_FIXED_SCOPE:
        raise PreprocessingArtifactError(
            f"Fixed affine artifact requires scope={PUBLIC_FIXED_SCOPE!r}"
        )
    if raw["fit_on_protected_data"] is not False:
        raise PreprocessingArtifactError(
            "Public fixed artifact must declare fit_on_protected_data=false"
        )
    if raw["transform"] != "standard_scaler":
        raise PreprocessingArtifactError(
            f"Unsupported fixed affine transform: {raw['transform']!r}"
        )

    observed_digest = preprocessing_payload_sha256(raw)
    reported_digest = str(raw["payload_sha256"])
    if len(reported_digest) != 64 or any(
        char not in "0123456789abcdef" for char in reported_digest
    ):
        raise PreprocessingArtifactError(
            "payload_sha256 must be a lowercase SHA-256"
        )
    if reported_digest != observed_digest:
        raise PreprocessingArtifactError(
            "Preprocessing artifact payload digest mismatch: "
            f"reported {reported_digest}, observed {observed_digest}"
        )
    if expected_payload_sha256 is not None and observed_digest != expected_payload_sha256:
        raise PreprocessingArtifactError(
            "Preprocessing artifact does not match the contract digest: "
            f"expected {expected_payload_sha256}, got {observed_digest}"
        )

    try:
        if isinstance(raw["input_dim"], bool) or not isinstance(
            raw["input_dim"], int
        ):
            raise TypeError
        input_dim = raw["input_dim"]
        mean = np.asarray(raw["mean"], dtype=np.float64)
        scale = np.asarray(raw["scale"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PreprocessingArtifactError(
            "Preprocessing dimensions and parameters must be numeric"
        ) from exc
    if input_dim <= 0:
        raise PreprocessingArtifactError("input_dim must be positive")
    if mean.shape != (input_dim,) or scale.shape != (input_dim,):
        raise PreprocessingArtifactError(
            "mean and scale must each contain exactly input_dim values"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise PreprocessingArtifactError(
            "Preprocessing parameters must all be finite"
        )
    if np.any(scale <= 0):
        raise PreprocessingArtifactError("Every preprocessing scale must be positive")
    mean.setflags(write=False)
    scale.setflags(write=False)

    source_metadata = raw["source"]
    if not isinstance(source_metadata, dict):
        raise PreprocessingArtifactError("source must be a JSON object")
    if source_metadata.get("is_public_reference") is not True:
        raise PreprocessingArtifactError(
            "Public fixed artifact must identify a public reference source"
        )
    source_sha256 = str(source_metadata.get("sha256", ""))
    source_reference = str(source_metadata.get("reference", ""))
    if (
        len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
        or not source_reference
    ):
        raise PreprocessingArtifactError(
            "Public reference source requires a reference and SHA-256 digest"
        )

    artifact_id = str(raw["artifact_id"])
    if not artifact_id:
        raise PreprocessingArtifactError("artifact_id must be non-empty")

    return FixedAffinePreprocessor(
        artifact_id=artifact_id,
        input_dim=input_dim,
        mean=mean,
        scale=scale,
        payload_sha256=observed_digest,
        source_sha256=source_sha256,
        source_reference=source_reference,
        transform_type=str(raw["transform"]),
    )
