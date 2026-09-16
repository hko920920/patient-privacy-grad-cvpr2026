"""Deterministic source binding for the external-PLD V3.6 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from unitdp.source_bundle_allocation_v4 import (
    COMPILER_SOURCE_FILES_ALLOCATION_V4,
    EXECUTION_SOURCE_FILES_ALLOCATION_V4,
)


COMPILER_SOURCE_FILES_EXTERNAL_PLD_V36 = tuple(
    sorted(
        {
            *COMPILER_SOURCE_FILES_ALLOCATION_V4,
            "src/unitdp/compiler_v4.py",
            "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
            "v36_candidate/src/unitdp/external_pld_backend_v36.py",
            "v36_candidate/src/unitdp/handler_external_pld_v36.py",
            "v36_candidate/src/unitdp/source_bundle_external_pld_v36.py",
        }
    )
)
COMPILER_SOURCE_FILE_SET_EXTERNAL_PLD_V36 = frozenset(
    COMPILER_SOURCE_FILES_EXTERNAL_PLD_V36
)
EXECUTION_SOURCE_FILES_EXTERNAL_PLD_V36 = tuple(
    sorted(
        {
            *EXECUTION_SOURCE_FILES_ALLOCATION_V4,
            *COMPILER_SOURCE_FILES_EXTERNAL_PLD_V36,
            "v36_candidate/src/unitdp/owner_external_pld_v36.py",
        }
    )
)
EXECUTION_SOURCE_FILE_SET_EXTERNAL_PLD_V36 = frozenset(
    EXECUTION_SOURCE_FILES_EXTERNAL_PLD_V36
)


class SourceBundleExternalPldV36Error(RuntimeError):
    """Raised when a V3.6 source bundle cannot be reproduced."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceBundleExternalPldV36Error(
            f"Could not hash registered V3.6 source file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_bundle(
    relative_paths: tuple[str, ...],
    required: frozenset[str],
) -> MappingProxyType[str, str]:
    root = _repo_root().resolve()
    result: dict[str, str] = {}
    for relative in relative_paths:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SourceBundleExternalPldV36Error(
                f"Registered source path escapes the repository: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SourceBundleExternalPldV36Error(
                f"Registered V3.6 source file is missing: {relative}"
            )
        result[relative] = _file_sha256(candidate)
    if set(result) != required:
        raise SourceBundleExternalPldV36Error(
            "Registered V3.6 source-bundle fields are incomplete"
        )
    return MappingProxyType(result)


def compiler_source_bundle_external_pld_v36() -> MappingProxyType[str, str]:
    return _source_bundle(
        COMPILER_SOURCE_FILES_EXTERNAL_PLD_V36,
        COMPILER_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
    )


def compiler_source_bundle_sha256_external_pld_v36(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    value = dict(
        compiler_source_bundle_external_pld_v36() if bundle is None else bundle
    )
    if set(value) != COMPILER_SOURCE_FILE_SET_EXTERNAL_PLD_V36:
        raise SourceBundleExternalPldV36Error(
            "V3.6 compiler source fields do not match the registered set"
        )
    return _payload_sha256(value)


def execution_source_bundle_external_pld_v36() -> MappingProxyType[str, str]:
    return _source_bundle(
        EXECUTION_SOURCE_FILES_EXTERNAL_PLD_V36,
        EXECUTION_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
    )


def execution_source_bundle_sha256_external_pld_v36(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    value = dict(
        execution_source_bundle_external_pld_v36() if bundle is None else bundle
    )
    if set(value) != EXECUTION_SOURCE_FILE_SET_EXTERNAL_PLD_V36:
        raise SourceBundleExternalPldV36Error(
            "V3.6 execution source fields do not match the registered set"
        )
    return _payload_sha256(value)
