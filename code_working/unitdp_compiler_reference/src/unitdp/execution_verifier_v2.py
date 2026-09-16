"""Strict, read-only verification for V2 public benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unitdp.execution_artifacts_v2 import (
    MODEL_ARTIFACT_SCHEMA_V2,
    PUBLIC_BUNDLE_SCHEMA_V2,
    load_model_artifact_v2,
)
from unitdp.owner_poisson_v2 import (
    PUBLIC_EXECUTION_SCHEMA_V2,
    OwnerPoissonExecutionV2Error,
    _model_state_sha256,
    _sha256_payload,
)
from unitdp.release_artifacts import assert_public_certificate_redacted
from unitdp.source_bundle_v2 import (
    EXECUTION_SOURCE_FILE_SET_V2,
    execution_source_bundle_sha256_v2,
)


PUBLIC_SUMMARY_SCHEMA_V2 = "unitdp.benchmark_summary_public.v2.1"
PUBLIC_COLLECTION_SCHEMA_V2 = "unitdp.benchmark_collection_public.v2.1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^run_[0-9]{3}$")


class V2ArtifactVerificationError(RuntimeError):
    """Raised when a V2 artifact collection fails closed."""


@dataclass(frozen=True)
class VerifiedRunArtifactsV2:
    run_id: str
    public_execution_sha256: str
    public_bundle_payload_sha256: str
    model_state_sha256: str
    execution_source_bundle_sha256: str
    public_evaluation: dict[str, Any]


@dataclass(frozen=True)
class VerifiedDatasetArtifactsV2:
    dataset: str
    public_summary_sha256: str
    run_count: int
    runs: tuple[VerifiedRunArtifactsV2, ...]


@dataclass(frozen=True)
class VerifiedCollectionArtifactsV2:
    public_collection_sha256: str
    dataset_count: int
    run_count: int
    datasets: tuple[VerifiedDatasetArtifactsV2, ...]
    current_source_bindings_verified: bool


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V2ArtifactVerificationError(
                f"Duplicate JSON key is forbidden: {key!r}"
            )
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except V2ArtifactVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2ArtifactVerificationError(
            f"Could not strictly load JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise V2ArtifactVerificationError(
            f"JSON artifact must be an object: {path}"
        )
    return value


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise V2ArtifactVerificationError(
            f"Could not hash artifact {path}: {exc}"
        ) from exc
    return hasher.hexdigest()


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise V2ArtifactVerificationError(
            f"{context} fields differ from the strict schema; "
            f"missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise V2ArtifactVerificationError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _verify_payload_hash(
    value: dict[str, Any],
    field: str,
    context: str,
) -> str:
    reported = _require_sha256(value.get(field), f"{context}.{field}")
    without_hash = dict(value)
    without_hash.pop(field, None)
    observed = _sha256_payload(without_hash)
    if observed != reported:
        raise V2ArtifactVerificationError(
            f"{context} payload hash mismatch: "
            f"reported={reported}, observed={observed}"
        )
    return reported


def _assert_public_redacted(
    value: dict[str, Any],
    context: str,
) -> None:
    try:
        assert_public_certificate_redacted(value)
    except ValueError as exc:
        raise V2ArtifactVerificationError(
            f"{context} crosses the public/private boundary: {exc}"
        ) from exc


def _require_public_header(
    value: dict[str, Any],
    *,
    schema: str,
    context: str,
) -> None:
    if value.get("schema_version") != schema:
        raise V2ArtifactVerificationError(
            f"{context} has an unsupported schema"
        )
    if value.get("visibility") != "public":
        raise V2ArtifactVerificationError(
            f"{context} is not explicitly public"
        )
    if value.get("release_status") != "research_non_release":
        raise V2ArtifactVerificationError(
            f"{context} must remain research_non_release"
        )


def _require_public_execution_invariants(
    execution: dict[str, Any],
    context: str,
) -> None:
    mechanism = execution.get("mechanism")
    if not isinstance(mechanism, dict):
        raise V2ArtifactVerificationError(
            f"{context}.mechanism must be an object"
        )
    expected_mechanism = {
        "privacy_unit": "owner",
        "adjacency": "add_remove_one_owner",
        "sampler": "independent_bernoulli_owner",
        "sampling_unit": "owner",
        "clipping_unit": "owner",
        "noising_unit": "owner_sum",
        "accounting_unit": "owner",
        "empty_step_behavior": "apply_gaussian_update",
        "per_owner_contribution": "one_clipped_vector",
        "owner_aggregation": "mean",
    }
    for key, expected in expected_mechanism.items():
        if mechanism.get(key) != expected:
            raise V2ArtifactVerificationError(
                f"{context}.mechanism.{key} is not {expected!r}"
            )
    for key in (
        "owner_sample_rate",
        "total_steps",
        "clip_norm",
        "noise_multiplier",
        "update_denominator",
    ):
        value = mechanism.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise V2ArtifactVerificationError(
                f"{context}.mechanism.{key} must be finite and positive"
            )
    if float(mechanism["owner_sample_rate"]) > 1.0:
        raise V2ArtifactVerificationError(
            f"{context}.mechanism.owner_sample_rate exceeds one"
        )
    if (
        isinstance(mechanism["total_steps"], bool)
        or not isinstance(mechanism["total_steps"], int)
    ):
        raise V2ArtifactVerificationError(
            f"{context}.mechanism.total_steps must be an integer"
        )

    checks = execution.get("execution_checks")
    if not isinstance(checks, dict):
        raise V2ArtifactVerificationError(
            f"{context}.execution_checks must be an object"
        )
    required_true = {
        "compiled_object_only",
        "compiled_integrity_before_and_after",
        "fixed_q",
        "fixed_total_steps",
        "fixed_update_denominator",
        "bernoulli_owner_sampling",
        "one_clipped_vector_per_sampled_owner",
        "Gaussian_noise_every_step",
        "optimizer_update_every_step",
        "preprocessing_applied_inside_executor",
        "source_bundle_unchanged_during_execution",
        "random_streams_domain_separated",
    }
    _require_exact_keys(
        checks,
        required_true | {"random_coins_disclosed"},
        f"{context}.execution_checks",
    )
    if any(checks[key] is not True for key in required_true):
        raise V2ArtifactVerificationError(
            f"{context} has an unverified required execution invariant"
        )
    if checks["random_coins_disclosed"] is not False:
        raise V2ArtifactVerificationError(
            f"{context} unexpectedly discloses random coins"
        )

    rng = execution.get("rng")
    if not isinstance(rng, dict):
        raise V2ArtifactVerificationError(f"{context}.rng must be an object")
    _require_exact_keys(
        rng,
        {"backend", "secure_rng", "random_coins_disclosed", "domains"},
        f"{context}.rng",
    )
    if (
        rng["backend"] != "numpy_pcg64_domain_separated_research"
        or rng["secure_rng"] is not False
        or rng["random_coins_disclosed"] is not False
        or rng["domains"]
        != [
            "model_initialization",
            "owner_sampling",
            "within_owner_selection",
            "gaussian_noise",
        ]
    ):
        raise V2ArtifactVerificationError(
            f"{context} has an unexpected research RNG declaration"
        )

    claim = execution.get("privacy_claim")
    if not isinstance(claim, dict):
        raise V2ArtifactVerificationError(
            f"{context}.privacy_claim must be an object"
        )
    _require_exact_keys(
        claim,
        {"status", "reason_codes"},
        f"{context}.privacy_claim",
    )
    if (
        claim["status"] != "accountant_output_only_not_a_release_claim"
        or claim["reason_codes"]
        != [
            "noncryptographic_research_rng",
            "privacy_release_executor_not_audited",
        ]
    ):
        raise V2ArtifactVerificationError(
            f"{context} overstates its privacy-release status"
        )


def _require_public_evaluation(
    value: object,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V2ArtifactVerificationError(f"{context} must be an object")
    _require_exact_keys(
        value,
        {"scope", "accuracy", "macro_f1", "auroc", "auprc"},
        context,
    )
    if value["scope"] != "public_benchmark":
        raise V2ArtifactVerificationError(
            f"{context}.scope must be public_benchmark"
        )
    for metric in ("accuracy", "macro_f1", "auroc", "auprc"):
        observed = value[metric]
        if observed is None and metric in {"auroc", "auprc"}:
            continue
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not np.isfinite(float(observed))
            or float(observed) < 0.0
            or float(observed) > 1.0
        ):
            raise V2ArtifactVerificationError(
                f"{context}.{metric} must lie in [0, 1] or be allowed null"
            )
    return value


def verify_run_artifacts_v2(
    run_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> VerifiedRunArtifactsV2:
    """Verify one public model/execution/bundle directory."""

    target = Path(run_dir)
    run_id = target.name
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise V2ArtifactVerificationError(
            f"Invalid public run directory name: {run_id!r}"
        )
    if not target.is_dir():
        raise V2ArtifactVerificationError(
            f"Public run directory does not exist: {target}"
        )
    expected_files = {
        "model_v2.json",
        "public_execution_v2.json",
        "public_bundle_v2.json",
    }
    observed_files = {path.name for path in target.iterdir()}
    if observed_files != expected_files:
        raise V2ArtifactVerificationError(
            f"{target} does not contain exactly the three public artifacts; "
            f"observed={sorted(observed_files)}"
        )

    model_path = target / "model_v2.json"
    execution_path = target / "public_execution_v2.json"
    bundle_path = target / "public_bundle_v2.json"
    model_raw = _load_json_object(model_path)
    try:
        model = load_model_artifact_v2(model_path)
    except OwnerPoissonExecutionV2Error as exc:
        raise V2ArtifactVerificationError(
            f"Model artifact verification failed for {run_id}: {exc}"
        ) from exc
    if model_raw.get("schema_version") != MODEL_ARTIFACT_SCHEMA_V2:
        raise V2ArtifactVerificationError(
            f"{run_id} has an unsupported model artifact schema"
        )
    model_payload_sha256 = _require_sha256(
        model_raw.get("payload_sha256"),
        f"{run_id}.model.payload_sha256",
    )
    model_state_sha256 = _require_sha256(
        model_raw.get("state_sha256"),
        f"{run_id}.model.state_sha256",
    )
    if _model_state_sha256(model) != model_state_sha256:
        raise V2ArtifactVerificationError(
            f"{run_id} model state hash changed after loading"
        )

    execution = _load_json_object(execution_path)
    _require_exact_keys(
        execution,
        {
            "schema_version",
            "visibility",
            "release_status",
            "privacy_claim",
            "benchmark_profile_id",
            "route",
            "mechanism",
            "accountant",
            "execution_checks",
            "rng",
            "model",
            "public_data_bindings",
            "public_evaluation",
            "model_output",
            "implementation",
            "public_execution_sha256",
        },
        f"{run_id}.public_execution",
    )
    _require_public_header(
        execution,
        schema=PUBLIC_EXECUTION_SCHEMA_V2,
        context=f"{run_id}.public_execution",
    )
    public_execution_sha256 = _verify_payload_hash(
        execution,
        "public_execution_sha256",
        f"{run_id}.public_execution",
    )
    _assert_public_redacted(execution, f"{run_id}.public_execution")
    _require_public_execution_invariants(
        execution,
        f"{run_id}.public_execution",
    )
    evaluation = _require_public_evaluation(
        execution.get("public_evaluation"),
        f"{run_id}.public_execution.public_evaluation",
    )
    model_output = execution.get("model_output")
    if not isinstance(model_output, dict):
        raise V2ArtifactVerificationError(
            f"{run_id}.public_execution.model_output must be an object"
        )
    _require_exact_keys(
        model_output,
        {"state_sha256", "artifact_status"},
        f"{run_id}.public_execution.model_output",
    )
    if (
        model_output["artifact_status"] != "not_serialized_by_executor"
        or model_output["state_sha256"] != model_state_sha256
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} execution record and model artifact disagree"
        )
    implementation = execution.get("implementation")
    if not isinstance(implementation, dict):
        raise V2ArtifactVerificationError(
            f"{run_id}.public_execution.implementation must be an object"
        )
    _require_exact_keys(
        implementation,
        {
            "source_bundle_schema",
            "source_bundle",
            "source_bundle_sha256",
            "library_versions",
        },
        f"{run_id}.public_execution.implementation",
    )
    if (
        implementation["source_bundle_schema"]
        != "unitdp.execution_source_bundle.v2"
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} has an unsupported execution source-bundle schema"
        )
    execution_source_bundle = implementation["source_bundle"]
    if (
        not isinstance(execution_source_bundle, dict)
        or set(execution_source_bundle) != EXECUTION_SOURCE_FILE_SET_V2
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} execution source bundle has the wrong file set"
        )
    for relative, digest in execution_source_bundle.items():
        _require_sha256(
            digest,
            f"{run_id}.implementation.source_bundle[{relative!r}]",
        )
    execution_source_bundle_sha256 = _require_sha256(
        implementation["source_bundle_sha256"],
        f"{run_id}.implementation.source_bundle_sha256",
    )
    if (
        execution_source_bundle_sha256_v2(execution_source_bundle)
        != execution_source_bundle_sha256
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} execution source-bundle hash mismatch"
        )
    if source_root is not None:
        _verify_current_source_bundle(
            execution_source_bundle,
            Path(source_root),
            f"{run_id}.public_execution.implementation.source_bundle",
        )

    bundle = _load_json_object(bundle_path)
    _require_exact_keys(
        bundle,
        {
            "schema_version",
            "visibility",
            "release_status",
            "model_artifact",
            "public_execution",
            "public_bundle_payload_sha256",
        },
        f"{run_id}.public_bundle",
    )
    _require_public_header(
        bundle,
        schema=PUBLIC_BUNDLE_SCHEMA_V2,
        context=f"{run_id}.public_bundle",
    )
    public_bundle_payload_sha256 = _verify_payload_hash(
        bundle,
        "public_bundle_payload_sha256",
        f"{run_id}.public_bundle",
    )
    _assert_public_redacted(bundle, f"{run_id}.public_bundle")
    model_link = bundle.get("model_artifact")
    execution_link = bundle.get("public_execution")
    if not isinstance(model_link, dict) or not isinstance(
        execution_link, dict
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} bundle links must be objects"
        )
    _require_exact_keys(
        model_link,
        {"filename", "file_sha256", "payload_sha256", "state_sha256"},
        f"{run_id}.public_bundle.model_artifact",
    )
    _require_exact_keys(
        execution_link,
        {"filename", "file_sha256", "payload_sha256"},
        f"{run_id}.public_bundle.public_execution",
    )
    if (
        model_link["filename"] != "model_v2.json"
        or execution_link["filename"] != "public_execution_v2.json"
    ):
        raise V2ArtifactVerificationError(
            f"{run_id} bundle contains a non-canonical filename"
        )
    expected_model_file_sha256 = _file_sha256(model_path)
    expected_execution_file_sha256 = _file_sha256(execution_path)
    link_pairs = (
        (
            model_link.get("file_sha256"),
            expected_model_file_sha256,
            "model file",
        ),
        (
            model_link.get("payload_sha256"),
            model_payload_sha256,
            "model payload",
        ),
        (
            model_link.get("state_sha256"),
            model_state_sha256,
            "model state",
        ),
        (
            execution_link.get("file_sha256"),
            expected_execution_file_sha256,
            "public execution file",
        ),
        (
            execution_link.get("payload_sha256"),
            public_execution_sha256,
            "public execution payload",
        ),
    )
    for observed, expected, label in link_pairs:
        _require_sha256(observed, f"{run_id}.bundle.{label}")
        if observed != expected:
            raise V2ArtifactVerificationError(
                f"{run_id} bundle {label} link mismatch"
            )

    return VerifiedRunArtifactsV2(
        run_id=run_id,
        public_execution_sha256=public_execution_sha256,
        public_bundle_payload_sha256=public_bundle_payload_sha256,
        model_state_sha256=model_state_sha256,
        execution_source_bundle_sha256=(
            execution_source_bundle_sha256
        ),
        public_evaluation=dict(evaluation),
    )


def _verify_current_source_bundle(
    bundle: dict[str, Any],
    source_root: Path,
    context: str,
) -> None:
    root = source_root.resolve()
    for relative, expected in bundle.items():
        if not isinstance(relative, str):
            raise V2ArtifactVerificationError(
                f"{context} source path keys must be strings"
            )
        _require_sha256(expected, f"{context}[{relative!r}]")
        candidates = [
            (root / Path(relative)).resolve(),
            (root / "src" / Path(relative)).resolve(),
        ]
        safe_candidates: list[Path] = []
        for candidate in candidates:
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise V2ArtifactVerificationError(
                    f"{context} source path escapes the source root: "
                    f"{relative}"
                ) from exc
            if candidate.is_file() and candidate not in safe_candidates:
                safe_candidates.append(candidate)
        if not safe_candidates:
            raise V2ArtifactVerificationError(
                f"{context} source file is missing: {relative}"
            )
        matching = [
            candidate
            for candidate in safe_candidates
            if _file_sha256(candidate) == expected
        ]
        if len(matching) != 1:
            raise V2ArtifactVerificationError(
                f"{context} must resolve {relative!r} to exactly one "
                "source file with the recorded hash"
            )


def _verify_aggregate(
    summary: dict[str, Any],
    runs: tuple[VerifiedRunArtifactsV2, ...],
    context: str,
) -> None:
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise V2ArtifactVerificationError(
            f"{context}.aggregate must be an object"
        )
    metrics = ("accuracy", "macro_f1", "auroc", "auprc")
    _require_exact_keys(aggregate, set(metrics), f"{context}.aggregate")
    for metric in metrics:
        record = aggregate[metric]
        if not isinstance(record, dict):
            raise V2ArtifactVerificationError(
                f"{context}.aggregate.{metric} must be an object"
            )
        _require_exact_keys(
            record,
            {"count", "mean", "std"},
            f"{context}.aggregate.{metric}",
        )
        values = [
            float(run.public_evaluation[metric])
            for run in runs
            if run.public_evaluation[metric] is not None
        ]
        expected_count = len(values)
        expected_mean = (
            float(np.asarray(values, dtype=np.float64).mean())
            if values
            else None
        )
        expected_std = (
            float(np.asarray(values, dtype=np.float64).std(ddof=0))
            if values
            else None
        )
        if record["count"] != expected_count:
            raise V2ArtifactVerificationError(
                f"{context}.aggregate.{metric}.count mismatch"
            )
        for key, expected in (("mean", expected_mean), ("std", expected_std)):
            observed = record[key]
            if expected is None:
                if observed is not None:
                    raise V2ArtifactVerificationError(
                        f"{context}.aggregate.{metric}.{key} must be null"
                    )
            elif (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not np.isclose(
                    float(observed),
                    expected,
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                raise V2ArtifactVerificationError(
                    f"{context}.aggregate.{metric}.{key} mismatch"
                )


def verify_dataset_artifacts_v2(
    dataset_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> VerifiedDatasetArtifactsV2:
    """Verify one dataset summary and every referenced public run."""

    target = Path(dataset_dir)
    dataset = target.name
    if not target.is_dir():
        raise V2ArtifactVerificationError(
            f"Dataset artifact directory does not exist: {target}"
        )
    summary_path = target / "public_summary_v2.json"
    summary = _load_json_object(summary_path)
    _require_exact_keys(
        summary,
        {
            "schema_version",
            "visibility",
            "release_status",
            "dataset",
            "benchmark_profile_id",
            "benchmark_profile_sha256",
            "full_data_conformance_sha256",
            "data_preparation_implementation_id",
            "data_preparer_source_bundle",
            "data_preparer_source_bundle_sha256",
            "public_contract_sha256",
            "public_plan_sha256",
            "execution_source_bundle_sha256",
            "run_count",
            "runs",
            "aggregate",
            "public_summary_sha256",
        },
        f"{dataset}.public_summary",
    )
    _require_public_header(
        summary,
        schema=PUBLIC_SUMMARY_SCHEMA_V2,
        context=f"{dataset}.public_summary",
    )
    public_summary_sha256 = _verify_payload_hash(
        summary,
        "public_summary_sha256",
        f"{dataset}.public_summary",
    )
    _assert_public_redacted(summary, f"{dataset}.public_summary")
    if summary["dataset"] != dataset:
        raise V2ArtifactVerificationError(
            f"Dataset directory and summary id disagree: {dataset}"
        )
    for field in (
        "benchmark_profile_sha256",
        "full_data_conformance_sha256",
        "data_preparer_source_bundle_sha256",
        "public_contract_sha256",
        "public_plan_sha256",
        "execution_source_bundle_sha256",
    ):
        _require_sha256(summary[field], f"{dataset}.public_summary.{field}")

    source_bundle = summary["data_preparer_source_bundle"]
    if not isinstance(source_bundle, dict) or not source_bundle:
        raise V2ArtifactVerificationError(
            f"{dataset}.public_summary source bundle must be non-empty"
        )
    if (
        _sha256_payload(source_bundle)
        != summary["data_preparer_source_bundle_sha256"]
    ):
        raise V2ArtifactVerificationError(
            f"{dataset}.public_summary source bundle hash mismatch"
        )
    if source_root is not None:
        _verify_current_source_bundle(
            source_bundle,
            Path(source_root),
            f"{dataset}.public_summary.data_preparer_source_bundle",
        )

    run_records = summary["runs"]
    run_count = summary["run_count"]
    if (
        isinstance(run_count, bool)
        or not isinstance(run_count, int)
        or run_count <= 0
        or not isinstance(run_records, list)
        or len(run_records) != run_count
    ):
        raise V2ArtifactVerificationError(
            f"{dataset}.public_summary run_count is invalid"
        )
    expected_run_ids = [
        f"run_{index:03d}" for index in range(1, run_count + 1)
    ]
    observed_run_ids: list[str] = []
    verified_runs: list[VerifiedRunArtifactsV2] = []
    for index, record in enumerate(run_records):
        if not isinstance(record, dict):
            raise V2ArtifactVerificationError(
                f"{dataset}.public_summary.runs[{index}] must be an object"
            )
        _require_exact_keys(
            record,
            {
                "run_id",
                "public_execution_sha256",
                "public_bundle_payload_sha256",
                "model_state_sha256",
                "public_evaluation",
            },
            f"{dataset}.public_summary.runs[{index}]",
        )
        run_id = record["run_id"]
        if not isinstance(run_id, str):
            raise V2ArtifactVerificationError(
                f"{dataset}.public_summary run_id must be a string"
            )
        observed_run_ids.append(run_id)
        verified = verify_run_artifacts_v2(
            target / run_id,
            source_root=source_root,
        )
        for observed, expected, label in (
            (
                record["public_execution_sha256"],
                verified.public_execution_sha256,
                "public execution",
            ),
            (
                record["public_bundle_payload_sha256"],
                verified.public_bundle_payload_sha256,
                "public bundle",
            ),
            (
                record["model_state_sha256"],
                verified.model_state_sha256,
                "model state",
            ),
        ):
            _require_sha256(
                observed,
                f"{dataset}.public_summary.{run_id}.{label}",
            )
            if observed != expected:
                raise V2ArtifactVerificationError(
                    f"{dataset}.{run_id} {label} summary link mismatch"
                )
        recorded_evaluation = _require_public_evaluation(
            record["public_evaluation"],
            f"{dataset}.public_summary.{run_id}.public_evaluation",
        )
        if recorded_evaluation != verified.public_evaluation:
            raise V2ArtifactVerificationError(
                f"{dataset}.{run_id} evaluation summary link mismatch"
            )
        execution = _load_json_object(
            target / run_id / "public_execution_v2.json"
        )
        route = execution.get("route")
        bindings = execution.get("public_data_bindings")
        if not isinstance(route, dict) or not isinstance(bindings, dict):
            raise V2ArtifactVerificationError(
                f"{dataset}.{run_id} route/data bindings are malformed"
            )
        if (
            execution.get("benchmark_profile_id")
            != summary["benchmark_profile_id"]
            or route.get("public_contract_sha256")
            != summary["public_contract_sha256"]
            or route.get("public_plan_sha256")
            != summary["public_plan_sha256"]
            or verified.execution_source_bundle_sha256
            != summary["execution_source_bundle_sha256"]
            or bindings.get("data_preparation_implementation_id")
            != summary["data_preparation_implementation_id"]
        ):
            raise V2ArtifactVerificationError(
                f"{dataset}.{run_id} does not match its dataset summary"
            )
        verified_runs.append(verified)

    if observed_run_ids != expected_run_ids:
        raise V2ArtifactVerificationError(
            f"{dataset}.public_summary run ids are not canonical and sequential"
        )
    public_run_dirs = sorted(
        path.name
        for path in target.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if public_run_dirs != expected_run_ids:
        raise V2ArtifactVerificationError(
            f"{dataset} public run directories and summary disagree"
        )
    extra_public_files = sorted(
        path.name
        for path in target.iterdir()
        if path.is_file() and path.name != "public_summary_v2.json"
    )
    if extra_public_files:
        raise V2ArtifactVerificationError(
            f"{dataset} contains unregistered public files: "
            f"{extra_public_files}"
        )

    verified_tuple = tuple(verified_runs)
    _verify_aggregate(
        summary,
        verified_tuple,
        f"{dataset}.public_summary",
    )
    return VerifiedDatasetArtifactsV2(
        dataset=dataset,
        public_summary_sha256=public_summary_sha256,
        run_count=run_count,
        runs=verified_tuple,
    )


def verify_collection_artifacts_v2(
    collection_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> VerifiedCollectionArtifactsV2:
    """Verify a complete registered V2 public benchmark collection."""

    target = Path(collection_dir)
    if not target.is_dir():
        raise V2ArtifactVerificationError(
            f"Collection directory does not exist: {target}"
        )
    collection_path = target / "public_collection_v2.json"
    collection = _load_json_object(collection_path)
    _require_exact_keys(
        collection,
        {
            "schema_version",
            "visibility",
            "release_status",
            "execution_source_bundle_sha256",
            "dataset_summaries",
            "public_collection_sha256",
        },
        "public_collection",
    )
    _require_public_header(
        collection,
        schema=PUBLIC_COLLECTION_SCHEMA_V2,
        context="public_collection",
    )
    public_collection_sha256 = _verify_payload_hash(
        collection,
        "public_collection_sha256",
        "public_collection",
    )
    _assert_public_redacted(collection, "public_collection")
    collection_source_bundle_sha256 = _require_sha256(
        collection["execution_source_bundle_sha256"],
        "public_collection.execution_source_bundle_sha256",
    )
    records = collection["dataset_summaries"]
    if not isinstance(records, list) or not records:
        raise V2ArtifactVerificationError(
            "public_collection.dataset_summaries must be non-empty"
        )
    verified_datasets: list[VerifiedDatasetArtifactsV2] = []
    dataset_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise V2ArtifactVerificationError(
                f"public_collection.dataset_summaries[{index}] is invalid"
            )
        _require_exact_keys(
            record,
            {"dataset", "public_summary_sha256"},
            f"public_collection.dataset_summaries[{index}]",
        )
        dataset = record["dataset"]
        if not isinstance(dataset, str) or not dataset:
            raise V2ArtifactVerificationError(
                "public_collection dataset id must be a non-empty string"
            )
        dataset_ids.append(dataset)
        verified = verify_dataset_artifacts_v2(
            target / dataset,
            source_root=source_root,
        )
        _require_sha256(
            record["public_summary_sha256"],
            f"public_collection.{dataset}.public_summary_sha256",
        )
        if record["public_summary_sha256"] != verified.public_summary_sha256:
            raise V2ArtifactVerificationError(
                f"public_collection summary link mismatch for {dataset}"
            )
        summary = _load_json_object(
            target / dataset / "public_summary_v2.json"
        )
        if (
            summary.get("execution_source_bundle_sha256")
            != collection_source_bundle_sha256
        ):
            raise V2ArtifactVerificationError(
                f"public_collection source-bundle link mismatch for {dataset}"
            )
        verified_datasets.append(verified)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise V2ArtifactVerificationError(
            "public_collection contains duplicate dataset ids"
        )
    public_dataset_dirs = sorted(
        path.name
        for path in target.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if public_dataset_dirs != sorted(dataset_ids):
        raise V2ArtifactVerificationError(
            "public_collection dataset directories and index disagree"
        )
    extra_public_files = sorted(
        path.name
        for path in target.iterdir()
        if path.is_file() and path.name != "public_collection_v2.json"
    )
    if extra_public_files:
        raise V2ArtifactVerificationError(
            "public_collection contains unregistered public files: "
            f"{extra_public_files}"
        )

    current_source_verified = source_root is not None
    for dataset in verified_datasets:
        for run in dataset.runs:
            if (
                run.execution_source_bundle_sha256
                != collection_source_bundle_sha256
            ):
                raise V2ArtifactVerificationError(
                    f"{dataset.dataset}.{run.run_id} source bundle "
                    "differs from the collection"
                )

    dataset_tuple = tuple(verified_datasets)
    return VerifiedCollectionArtifactsV2(
        public_collection_sha256=public_collection_sha256,
        dataset_count=len(dataset_tuple),
        run_count=sum(item.run_count for item in dataset_tuple),
        datasets=dataset_tuple,
        current_source_bindings_verified=current_source_verified,
    )
