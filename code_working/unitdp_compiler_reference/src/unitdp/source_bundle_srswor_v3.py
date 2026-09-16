"""Deterministic source binding for the registered owner-SRSWOR V3 route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType


EXECUTION_SOURCE_FILES_SRSWOR_V3 = (
    "scripts/run_sepsis_owa.py",
    "scripts/run_uci_owa.py",
    "scripts/run_v3_srswor_registered_benchmarks.py",
    "scripts/run_wisdm_owa.py",
    "src/unitdp/accountant.py",
    "src/unitdp/benchmark_data_v2.py",
    "src/unitdp/benchmark_registry_srswor_v3.py",
    "src/unitdp/benchmark_registry_v2.py",
    "src/unitdp/compiler_srswor_v3.py",
    "src/unitdp/compiler_v2.py",
    "src/unitdp/compiler_v3.py",
    "src/unitdp/execution_artifacts_srswor_v3.py",
    "src/unitdp/execution_artifacts_v2.py",
    "src/unitdp/mapping.py",
    "src/unitdp/owner_poisson_v2.py",
    "src/unitdp/owner_srswor_v3.py",
    "src/unitdp/policies.py",
    "src/unitdp/preprocessing.py",
    "src/unitdp/release_artifacts.py",
    "src/unitdp/source_bundle_srswor_v3.py",
    "src/unitdp_spec_oracle/srswor_rdp_oracle_v3.py",
)
EXECUTION_SOURCE_FILE_SET_SRSWOR_V3 = frozenset(
    EXECUTION_SOURCE_FILES_SRSWOR_V3
)


class SourceBundleSrsworV3Error(RuntimeError):
    """Raised when the registered V3 source bundle cannot be reproduced."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise SourceBundleSrsworV3Error(
            f"Could not hash registered SRSWOR V3 source file {path}: {exc}"
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


def execution_source_bundle_srswor_v3() -> MappingProxyType[str, str]:
    root = _repo_root().resolve()
    result: dict[str, str] = {}
    for relative in EXECUTION_SOURCE_FILES_SRSWOR_V3:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SourceBundleSrsworV3Error(
                f"Registered source path escapes the repository: {relative}"
            ) from exc
        if not candidate.is_file():
            raise SourceBundleSrsworV3Error(
                f"Registered SRSWOR V3 source file is missing: {relative}"
            )
        result[relative] = _file_sha256(candidate)
    if set(result) != EXECUTION_SOURCE_FILE_SET_SRSWOR_V3:
        raise SourceBundleSrsworV3Error(
            "Registered SRSWOR V3 source bundle is incomplete"
        )
    return MappingProxyType(result)


def execution_source_bundle_sha256_srswor_v3(
    bundle: dict[str, str] | MappingProxyType[str, str] | None = None,
) -> str:
    value = dict(
        execution_source_bundle_srswor_v3()
        if bundle is None
        else bundle
    )
    if set(value) != EXECUTION_SOURCE_FILE_SET_SRSWOR_V3:
        raise SourceBundleSrsworV3Error(
            "SRSWOR V3 source-bundle fields do not match the registered set"
        )
    return _payload_sha256(value)
