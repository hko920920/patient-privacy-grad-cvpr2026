"""Fail-closed subprocess adapter for the external PLD accountant.

The public adapter imports no external numerical backend in the caller.  A
worker process prepends an explicitly supplied target directory, verifies the
locked distributions, and returns one canonical JSON response.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BACKEND_SITE_ENV_V36 = "UNITDP_G8_PLD_SITE"
BACKEND_SCHEMA_V36 = "unitdp.external_pld_backend.v36"
REQUEST_SCHEMA_V36 = "unitdp.external_pld_request.v36"
RESPONSE_SCHEMA_V36 = "unitdp.external_pld_response.v36"

PLD_DISTRIBUTION_V36 = "PLD_accounting"
PLD_VERSION_V36 = "0.5.0"
PLD_LICENSE_V36 = "MIT"
PLD_REPOSITORY_URL_V36 = "https://github.com/moshenfeld/PLD_accounting"
PLD_PAPER_URL_V36 = "https://arxiv.org/abs/2602.17284"
PLD_WHEEL_SHA256_V36 = (
    "1585d4030e4cda6209b6e9726ad81e46c2fd7305ec3dd2631d28125059d59804"
)
PLD_RECORD_SHA256_V36 = (
    "bb6f56fae220431276dea044225ed29bbadc3c3a26c07cce5aa185ccab56d1e6"
)
PLD_TREE_SHA256_V36 = "50782379259744c44878da0c0164fa590539a80c076c05366c10bb8c77ca90dc"

TRANSITIVE_DISTRIBUTION_V36 = "random-allocation"
TRANSITIVE_VERSION_V36 = "1.0.5"
TRANSITIVE_LICENSE_V36 = "MIT"
TRANSITIVE_REPOSITORY_URL_V36 = "https://github.com/moshenfeld/random_allocation"
TRANSITIVE_WHEEL_SHA256_V36 = (
    "236660af5ae93ae0659f54580ec3cfdfdadb6d8e52a17c7f8a17a71326987c06"
)
TRANSITIVE_RECORD_SHA256_V36 = (
    "a860ae5b86782de3807db072fbce8ca276da226ba884904fcf5bae7e8d46ac4b"
)
TRANSITIVE_TREE_SHA256_V36 = (
    "757c29118555cc7046b9e007239d443a93544fb918c7ac6009112a6ecbd05179"
)

NUMERICAL_LIBRARY_VERSIONS_V36 = (
    ("numpy", "1.26.4"),
    ("scipy", "1.16.1"),
    ("numba", "0.60.0"),
    ("llvmlite", "0.43.0"),
    ("dp-accounting", "0.6.0"),
)

LOSS_DISCRETIZATION_V36 = 0.005
TAIL_TRUNCATION_V36 = 1e-12
CONVOLUTION_METHOD_V36 = "GEOM"
UPPER_BOUND_TYPE_V36 = "DOMINATES"
LOWER_BOUND_TYPE_V36 = "IS_DOMINATED"
MAX_BRACKET_GAP_V36 = 0.01


class ExternalPldBackendV36Error(RuntimeError):
    """Raised when the external backend cannot be accepted."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExternalPldBackendV36Error(f"Value is not canonical JSON: {exc}") from exc


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ExternalPldBackendV36Error(
            f"Could not hash external distribution file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _canonical_distribution_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _find_distribution(
    site: Path,
    distribution_name: str,
) -> metadata.Distribution:
    expected = _canonical_distribution_name(distribution_name)
    matches = [
        distribution
        for distribution in metadata.distributions(path=[str(site)])
        if _canonical_distribution_name(str(distribution.metadata["Name"])) == expected
    ]
    if len(matches) != 1:
        raise ExternalPldBackendV36Error(
            f"Expected one {distribution_name!r} distribution in "
            f"{site}, found {len(matches)}"
        )
    return matches[0]


def _distribution_binding(
    site: Path,
    distribution_name: str,
) -> dict[str, Any]:
    distribution = _find_distribution(site, distribution_name)
    root = site.resolve()
    rows: list[dict[str, Any]] = []
    record_path: Path | None = None
    for entry in distribution.files or ():
        relative = str(entry).replace("\\", "/")
        if "/__pycache__/" in relative or relative.endswith(".pyc"):
            continue
        candidate = Path(distribution.locate_file(entry)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ExternalPldBackendV36Error(
                f"Distribution file escapes the backend site: {relative}"
            ) from exc
        if not candidate.is_file():
            continue
        if relative.endswith(".dist-info/RECORD"):
            record_path = candidate
        rows.append(
            {
                "path": relative,
                "sha256": _file_sha256(candidate),
                "size": candidate.stat().st_size,
            }
        )
    rows.sort(key=lambda value: str(value["path"]))
    if record_path is None:
        raise ExternalPldBackendV36Error(f"{distribution_name} has no installed RECORD")
    return {
        "distribution": str(distribution.metadata["Name"]),
        "version": distribution.version,
        "license": str(distribution.metadata["License"]),
        "file_count": len(rows),
        "record_sha256": _file_sha256(record_path),
        "tree_sha256": _payload_sha256(rows),
    }


def _distribution_version_at_site(
    site: Path,
    distribution_name: str,
) -> str:
    return _find_distribution(site, distribution_name).version


def _global_distribution_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise ExternalPldBackendV36Error(
            f"Required numerical distribution is missing: {distribution_name}"
        ) from exc


def resolve_external_pld_site_v36(
    backend_site: str | Path | None = None,
) -> Path:
    raw = (
        str(backend_site)
        if backend_site is not None
        else os.environ.get(BACKEND_SITE_ENV_V36, "")
    )
    if not raw:
        raise ExternalPldBackendV36Error(
            f"{BACKEND_SITE_ENV_V36} must identify the isolated backend site"
        )
    source = Path(raw)
    if not source.is_absolute():
        raise ExternalPldBackendV36Error(
            "External backend site must be an absolute path"
        )
    resolved = source.resolve()
    if not resolved.is_dir():
        raise ExternalPldBackendV36Error(
            f"External backend site does not exist: {resolved}"
        )
    return resolved


def verify_external_pld_environment_v36(
    backend_site: str | Path | None = None,
) -> dict[str, Any]:
    site = resolve_external_pld_site_v36(backend_site)
    primary = _distribution_binding(site, PLD_DISTRIBUTION_V36)
    transitive = _distribution_binding(
        site,
        TRANSITIVE_DISTRIBUTION_V36,
    )
    expected_primary = {
        "distribution": PLD_DISTRIBUTION_V36,
        "version": PLD_VERSION_V36,
        "license": PLD_LICENSE_V36,
        "file_count": 25,
        "record_sha256": PLD_RECORD_SHA256_V36,
        "tree_sha256": PLD_TREE_SHA256_V36,
    }
    expected_transitive = {
        "distribution": TRANSITIVE_DISTRIBUTION_V36,
        "version": TRANSITIVE_VERSION_V36,
        "license": TRANSITIVE_LICENSE_V36,
        "file_count": 36,
        "record_sha256": TRANSITIVE_RECORD_SHA256_V36,
        "tree_sha256": TRANSITIVE_TREE_SHA256_V36,
    }
    if primary != expected_primary:
        raise ExternalPldBackendV36Error(
            "Primary PLD distribution binding does not match the registry"
        )
    if transitive != expected_transitive:
        raise ExternalPldBackendV36Error(
            "Declared transitive distribution binding does not match the registry"
        )
    observed_versions: dict[str, str] = {}
    for name, expected in NUMERICAL_LIBRARY_VERSIONS_V36:
        if name in {"numba", "llvmlite"}:
            observed = _distribution_version_at_site(site, name)
        else:
            observed = _global_distribution_version(name)
        if observed != expected:
            raise ExternalPldBackendV36Error(
                f"Numerical dependency {name} must be {expected}, got {observed}"
            )
        observed_versions[name] = observed
    return {
        "schema_version": BACKEND_SCHEMA_V36,
        "primary_distribution": {
            **primary,
            "wheel_sha256": PLD_WHEEL_SHA256_V36,
        },
        "declared_transitive_distribution": {
            **transitive,
            "wheel_sha256": TRANSITIVE_WHEEL_SHA256_V36,
        },
        "numerical_library_versions": observed_versions,
    }


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalPldBackendV36Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ExternalPldBackendV36Error(f"{name} must be finite and positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExternalPldBackendV36Error(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExternalPldRequestV36:
    schema_version: str
    sigma: float
    num_steps: int
    num_selected: int
    num_epochs: int
    delta: float
    loss_discretization: float
    tail_truncation: float
    convolution_method: str
    upper_bound_type: str
    lower_bound_type: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _payload_sha256(self.payload())


def build_external_pld_request_v36(
    *,
    sigma: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    delta: float,
) -> ExternalPldRequestV36:
    request = ExternalPldRequestV36(
        schema_version=REQUEST_SCHEMA_V36,
        sigma=_positive_float(sigma, "sigma"),
        num_steps=_positive_int(num_steps, "num_steps"),
        num_selected=_positive_int(num_selected, "num_selected"),
        num_epochs=_positive_int(num_epochs, "num_epochs"),
        delta=_positive_float(delta, "delta"),
        loss_discretization=LOSS_DISCRETIZATION_V36,
        tail_truncation=TAIL_TRUNCATION_V36,
        convolution_method=CONVOLUTION_METHOD_V36,
        upper_bound_type=UPPER_BOUND_TYPE_V36,
        lower_bound_type=LOWER_BOUND_TYPE_V36,
    )
    if request.delta >= 1:
        raise ExternalPldBackendV36Error("delta must be in (0, 1)")
    if request.num_selected > request.num_steps:
        raise ExternalPldBackendV36Error("num_selected must not exceed num_steps")
    return request


def _parse_request_payload(value: object) -> ExternalPldRequestV36:
    if not isinstance(value, dict):
        raise ExternalPldBackendV36Error("Backend request must be an object")
    expected = {
        "schema_version",
        "sigma",
        "num_steps",
        "num_selected",
        "num_epochs",
        "delta",
        "loss_discretization",
        "tail_truncation",
        "convolution_method",
        "upper_bound_type",
        "lower_bound_type",
    }
    if set(value) != expected:
        raise ExternalPldBackendV36Error(
            "Backend request fields do not match the locked schema"
        )
    request = build_external_pld_request_v36(
        sigma=value["sigma"],
        num_steps=value["num_steps"],
        num_selected=value["num_selected"],
        num_epochs=value["num_epochs"],
        delta=value["delta"],
    )
    if request.payload() != value:
        raise ExternalPldBackendV36Error(
            "Backend request changes the locked numerical configuration"
        )
    return request


@dataclass(frozen=True)
class ExternalPldResponseV36:
    schema_version: str
    request_sha256: str
    epsilon_upper: float
    epsilon_lower: float
    environment: dict[str, Any]
    response_sha256: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "epsilon_upper": self.epsilon_upper,
            "epsilon_lower": self.epsilon_lower,
            "environment": self.environment,
        }

    def payload(self) -> dict[str, Any]:
        return {
            **self.payload_without_hash(),
            "response_sha256": self.response_sha256,
        }

    def assert_valid(
        self,
        request: ExternalPldRequestV36,
        *,
        target_epsilon: float,
        direct_epsilon: float,
    ) -> None:
        if self.schema_version != RESPONSE_SCHEMA_V36:
            raise ExternalPldBackendV36Error(
                "External response schema is not registered"
            )
        if self.request_sha256 != request.sha256:
            raise ExternalPldBackendV36Error(
                "External response is bound to a different request"
            )
        if _payload_sha256(self.payload_without_hash()) != self.response_sha256:
            raise ExternalPldBackendV36Error("External response digest is inconsistent")
        for name, value in (
            ("epsilon_upper", self.epsilon_upper),
            ("epsilon_lower", self.epsilon_lower),
        ):
            if not math.isfinite(value) or value < 0:
                raise ExternalPldBackendV36Error(
                    f"{name} must be finite and nonnegative"
                )
        if self.epsilon_lower > self.epsilon_upper:
            raise ExternalPldBackendV36Error("External PLD bracket is reversed")
        if self.epsilon_upper - self.epsilon_lower > MAX_BRACKET_GAP_V36:
            raise ExternalPldBackendV36Error(
                "External PLD bracket exceeds the registered tolerance"
            )
        if self.epsilon_upper > target_epsilon + 1e-10:
            raise ExternalPldBackendV36Error(
                "External dominating epsilon exceeds the target"
            )
        if self.epsilon_upper > direct_epsilon + 1e-10:
            raise ExternalPldBackendV36Error(
                "External dominating epsilon exceeds the frozen direct bound"
            )


def parse_external_pld_response_v36(
    value: object,
) -> ExternalPldResponseV36:
    if not isinstance(value, dict):
        raise ExternalPldBackendV36Error("Backend response must be an object")
    expected = {
        "schema_version",
        "request_sha256",
        "epsilon_upper",
        "epsilon_lower",
        "environment",
        "response_sha256",
    }
    if set(value) != expected:
        raise ExternalPldBackendV36Error(
            "Backend response fields do not match the locked schema"
        )
    if not isinstance(value["environment"], dict):
        raise ExternalPldBackendV36Error(
            "Backend response environment must be an object"
        )
    upper = _positive_float(
        value["epsilon_upper"],
        "epsilon_upper",
    )
    lower_value = value["epsilon_lower"]
    if isinstance(lower_value, bool) or not isinstance(lower_value, (int, float)):
        raise ExternalPldBackendV36Error("epsilon_lower must be numeric")
    lower = float(lower_value)
    if not math.isfinite(lower) or lower < 0:
        raise ExternalPldBackendV36Error("epsilon_lower must be finite and nonnegative")
    response = ExternalPldResponseV36(
        schema_version=str(value["schema_version"]),
        request_sha256=str(value["request_sha256"]),
        epsilon_upper=upper,
        epsilon_lower=lower,
        environment=dict(value["environment"]),
        response_sha256=str(value["response_sha256"]),
    )
    if _payload_sha256(response.payload_without_hash()) != response.response_sha256:
        raise ExternalPldBackendV36Error("External response digest is inconsistent")
    return response


def validate_external_pld_numeric_crosscheck_v36(
    response: ExternalPldResponseV36,
    request: ExternalPldRequestV36,
    *,
    target_epsilon: float,
    direct_epsilon: float,
) -> None:
    """Validate the pessimistic/optimistic bracket and direct-bound check."""

    response.assert_valid(
        request,
        target_epsilon=target_epsilon,
        direct_epsilon=direct_epsilon,
    )


def account_external_pld_v36(
    *,
    sigma: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    delta: float,
    target_epsilon: float,
    direct_epsilon: float,
    backend_site: str | Path | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalPldResponseV36:
    """Run one exact external upper/lower PLD query in a worker process."""

    site = resolve_external_pld_site_v36(backend_site)
    environment = verify_external_pld_environment_v36(site)
    request = build_external_pld_request_v36(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ExternalPldBackendV36Error("timeout_seconds must be finite and positive")
    worker_input = {
        "backend_site": str(site),
        "request": request.payload(),
    }
    worker_env = os.environ.copy()
    worker_env["NUMBA_CACHE_DIR"] = str(site.parent / "numba_cache")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=_canonical_json(worker_input) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=float(timeout_seconds),
            check=False,
            env=worker_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalPldBackendV36Error(
            f"External PLD worker failed to complete: {exc}"
        ) from exc
    _ = time.perf_counter() - started
    if completed.returncode != 0:
        raise ExternalPldBackendV36Error(
            "External PLD worker rejected the request; stderr SHA-256="
            + hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()
        )
    if completed.stderr:
        raise ExternalPldBackendV36Error(
            "External PLD worker emitted unexpected stderr"
        )
    try:
        raw_response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalPldBackendV36Error(
            "External PLD worker returned non-JSON output"
        ) from exc
    response = parse_external_pld_response_v36(raw_response)
    if response.environment != environment:
        raise ExternalPldBackendV36Error(
            "External worker environment differs from the adapter binding"
        )
    validate_external_pld_numeric_crosscheck_v36(
        response,
        request,
        target_epsilon=_positive_float(
            target_epsilon,
            "target_epsilon",
        ),
        direct_epsilon=_positive_float(
            direct_epsilon,
            "direct_epsilon",
        ),
    )
    return response


def _worker_main() -> int:
    try:
        raw = json.loads(sys.stdin.read())
        if not isinstance(raw, dict) or set(raw) != {"backend_site", "request"}:
            raise ExternalPldBackendV36Error(
                "Worker input fields do not match the schema"
            )
        site = resolve_external_pld_site_v36(raw["backend_site"])
        environment = verify_external_pld_environment_v36(site)
        request = _parse_request_payload(raw["request"])
        sys.path.insert(0, str(site))

        import llvmlite  # type: ignore[import-not-found]
        import numba  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
        import scipy  # type: ignore[import-not-found]
        from PLD_accounting import (  # type: ignore[import-not-found]
            AllocationSchemeConfig,
            BoundType,
            ConvolutionMethod,
            PrivacyParams,
            gaussian_allocation_epsilon_configurable,
        )

        imported_versions = {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
            "dp-accounting": _global_distribution_version("dp-accounting"),
        }
        if imported_versions != dict(NUMERICAL_LIBRARY_VERSIONS_V36):
            raise ExternalPldBackendV36Error(
                "Worker imported numerical versions outside the registry"
            )
        pld_module = sys.modules.get("PLD_accounting")
        pld_file = Path(str(getattr(pld_module, "__file__", ""))).resolve()
        try:
            pld_file.relative_to(site)
        except ValueError as exc:
            raise ExternalPldBackendV36Error(
                "Worker imported PLD_accounting outside the backend site"
            ) from exc

        params = PrivacyParams(
            sigma=request.sigma,
            num_steps=request.num_steps,
            num_selected=request.num_selected,
            num_epochs=request.num_epochs,
            delta=request.delta,
        )
        config = AllocationSchemeConfig(
            loss_discretization=request.loss_discretization,
            tail_truncation=request.tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )
        upper = float(
            gaussian_allocation_epsilon_configurable(
                params,
                config,
                BoundType.DOMINATES,
            )
        )
        lower = float(
            gaussian_allocation_epsilon_configurable(
                params,
                config,
                BoundType.IS_DOMINATED,
            )
        )
        response_without_hash = {
            "schema_version": RESPONSE_SCHEMA_V36,
            "request_sha256": request.sha256,
            "epsilon_upper": upper,
            "epsilon_lower": lower,
            "environment": environment,
        }
        response = {
            **response_without_hash,
            "response_sha256": _payload_sha256(response_without_hash),
        }
        sys.stdout.write(_canonical_json(response))
        return 0
    except Exception as exc:  # worker boundary must return no partial result
        sys.stderr.write(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raise SystemExit(_worker_main())
