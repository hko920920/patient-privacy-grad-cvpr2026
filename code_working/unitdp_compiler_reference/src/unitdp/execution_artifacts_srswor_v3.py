"""Deterministic artifact writer for strict SRSWOR V3 research runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unitdp.execution_artifacts_v2 import build_model_artifact_v2
from unitdp.owner_srswor_v3 import (
    OwnerSrsworExecutionResultV3,
    _sha256_payload,
)
from unitdp.release_artifacts import assert_public_certificate_redacted


PUBLIC_BUNDLE_SCHEMA_SRSWOR_V3 = (
    "unitdp.owner_srswor_run_bundle_public.v3"
)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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


@dataclass(frozen=True)
class WrittenSrsworExecutionArtifactsV3:
    model_path: Path
    public_execution_path: Path
    public_bundle_path: Path
    private_manifest_path: Path | None
    model_file_sha256: str
    public_execution_file_sha256: str
    public_bundle_payload_sha256: str


def write_srswor_execution_artifacts_v3(
    result: OwnerSrsworExecutionResultV3,
    output_dir: str | Path,
    *,
    include_private_manifest: bool = False,
    overwrite: bool = False,
) -> WrittenSrsworExecutionArtifactsV3:
    if not isinstance(result, OwnerSrsworExecutionResultV3):
        raise TypeError(
            "write_srswor_execution_artifacts_v3 requires "
            "OwnerSrsworExecutionResultV3"
        )
    result.assert_model_integrity()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model_srswor_v3.json"
    public_execution_path = (
        target / "public_execution_srswor_v3.json"
    )
    public_bundle_path = target / "public_bundle_srswor_v3.json"
    private_manifest_path = (
        target / "private_execution_srswor_v3.json"
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
    public_execution_file_sha256 = _file_sha256(
        public_execution_path
    )
    bundle: dict[str, Any] = {
        "schema_version": PUBLIC_BUNDLE_SCHEMA_SRSWOR_V3,
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
    return WrittenSrsworExecutionArtifactsV3(
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
