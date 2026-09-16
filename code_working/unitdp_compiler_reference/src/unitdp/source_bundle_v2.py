"""Deterministic source-code binding for the registered V2 execution path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType


EXECUTION_SOURCE_FILES_V2 = (
    "scripts/run_sepsis_owa.py",
    "scripts/run_uci_owa.py",
    "scripts/run_v2_registered_benchmarks.py",
    "scripts/run_wisdm_owa.py",
    "src/unitdp/accountant.py",
    "src/unitdp/benchmark_data_v2.py",
    "src/unitdp/benchmark_registry_v2.py",
    "src/unitdp/compiler_v2.py",
    "src/unitdp/execution_artifacts_v2.py",
    "src/unitdp/mapping.py",
    "src/unitdp/owa_dpsgd.py",
    "src/unitdp/owner_poisson_v2.py",
    "src/unitdp/policies.py",
    "src/unitdp/preprocessing.py",
    "src/unitdp/release_artifacts.py",
    "src/unitdp/source_bundle_v2.py",
)
EXECUTION_SOURCE_FILE_SET_V2 = frozenset(EXECUTION_SOURCE_FILES_V2)


class SourceBundleV2Error(RuntimeError):
    """Raised when the registered V2 source bundle cannot be reproduced."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise SourceBundleV2Error(
            f"Could not hash registered V2 source file {path}: {exc}"
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


def execution_source_bundle_v2() -> MappingProxyType[str, str]:
    """Return exact repository-relative source paths and file hashes."""

    root = _repo_root().resolve()
    result: dict[str, str] = {}
    for relative in EXECUTION_SOURCE_FILES_V2:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SourceBundleV2Error(
                f"Registered V2 source path escapes the repository: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SourceBundleV2Error(
                f"Registered V2 source file is missing: {relative}"
            )
        result[relative] = _file_sha256(candidate)
    if set(result) != EXECUTION_SOURCE_FILE_SET_V2:
        raise SourceBundleV2Error(
            "Registered V2 source bundle is incomplete"
        )
    return MappingProxyType(result)


def execution_source_bundle_sha256_v2(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    """Hash an exact V2 source bundle using canonical JSON encoding."""

    value = dict(
        execution_source_bundle_v2() if bundle is None else bundle
    )
    if set(value) != EXECUTION_SOURCE_FILE_SET_V2:
        raise SourceBundleV2Error(
            "V2 source-bundle fields do not match the registered file set"
        )
    return _payload_sha256(value)
