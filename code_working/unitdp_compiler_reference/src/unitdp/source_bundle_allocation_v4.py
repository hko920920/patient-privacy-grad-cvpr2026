"""Deterministic source binding for random-allocation V4 compilation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType


COMPILER_SOURCE_FILES_ALLOCATION_V4 = (
    "src/unitdp/benchmark_registry_allocation_v4.py",
    "src/unitdp/benchmark_registry_v2.py",
    "src/unitdp/compiler_allocation_v4.py",
    "src/unitdp/compiler_v2.py",
    "src/unitdp/mapping.py",
    "src/unitdp/policies.py",
    "src/unitdp/preprocessing.py",
    "src/unitdp/random_allocation_accountant_v4.py",
    "src/unitdp/source_bundle_allocation_v4.py",
    "src/unitdp_spec_oracle/random_allocation_oracle_v4.py",
)
COMPILER_SOURCE_FILE_SET_ALLOCATION_V4 = frozenset(
    COMPILER_SOURCE_FILES_ALLOCATION_V4
)
EXECUTION_SOURCE_FILES_ALLOCATION_V4 = tuple(
    sorted(
        {
            *COMPILER_SOURCE_FILES_ALLOCATION_V4,
            "src/unitdp/benchmark_data_v2.py",
            "src/unitdp/owner_poisson_v2.py",
            "src/unitdp/owner_random_allocation_v4.py",
            "src/unitdp/release_artifacts.py",
        }
    )
)
EXECUTION_SOURCE_FILE_SET_ALLOCATION_V4 = frozenset(
    EXECUTION_SOURCE_FILES_ALLOCATION_V4
)


class SourceBundleAllocationV4Error(RuntimeError):
    """Raised when the V4 compilation bundle cannot be reproduced."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise SourceBundleAllocationV4Error(
            f"Could not hash registered V4 source file {path}: {exc}"
        ) from exc
    return hasher.hexdigest()


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compiler_source_bundle_allocation_v4() -> MappingProxyType[str, str]:
    root = _repo_root().resolve()
    result: dict[str, str] = {}
    for relative in COMPILER_SOURCE_FILES_ALLOCATION_V4:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SourceBundleAllocationV4Error(
                f"Registered source path escapes the repository: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SourceBundleAllocationV4Error(
                f"Registered V4 source file is missing: {relative}"
            )
        result[relative] = _file_sha256(candidate)
    if set(result) != COMPILER_SOURCE_FILE_SET_ALLOCATION_V4:
        raise SourceBundleAllocationV4Error(
            "Registered V4 compiler source bundle is incomplete"
        )
    return MappingProxyType(result)


def compiler_source_bundle_sha256_allocation_v4(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    value = dict(
        compiler_source_bundle_allocation_v4()
        if bundle is None
        else bundle
    )
    if set(value) != COMPILER_SOURCE_FILE_SET_ALLOCATION_V4:
        raise SourceBundleAllocationV4Error(
            "V4 source-bundle fields do not match the registered set"
        )
    return _payload_sha256(value)


def execution_source_bundle_allocation_v4() -> MappingProxyType[str, str]:
    root = _repo_root().resolve()
    result: dict[str, str] = {}
    for relative in EXECUTION_SOURCE_FILES_ALLOCATION_V4:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SourceBundleAllocationV4Error(
                f"Registered source path escapes the repository: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SourceBundleAllocationV4Error(
                f"Registered V4 execution source file is missing: {relative}"
            )
        result[relative] = _file_sha256(candidate)
    if set(result) != EXECUTION_SOURCE_FILE_SET_ALLOCATION_V4:
        raise SourceBundleAllocationV4Error(
            "Registered V4 execution source bundle is incomplete"
        )
    return MappingProxyType(result)


def execution_source_bundle_sha256_allocation_v4(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    value = dict(
        execution_source_bundle_allocation_v4()
        if bundle is None
        else bundle
    )
    if set(value) != EXECUTION_SOURCE_FILE_SET_ALLOCATION_V4:
        raise SourceBundleAllocationV4Error(
            "V4 execution source fields do not match the registered set"
        )
    return _payload_sha256(value)
