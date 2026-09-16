"""Audit private V2 step diagnostics without releasing realized statistics.

The public artifacts are first checked by the production read-only verifier.
This auditor then checks hash links and eight fixed consistency predicates for
every private step record.  Its output contains only schedule-derived counts;
it never emits seeds, mappings, owner counts, or realized sampling totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.execution_verifier_v2 import (  # noqa: E402
    verify_collection_artifacts_v2,
)


REPORT_SCHEMA = "unitdp.private_step_diagnostic_audit.v1"
PRIVATE_INDEX_SCHEMA = "unitdp.benchmark_run_index_private.v2.1"
PRIVATE_MANIFEST_SCHEMA = "unitdp.owner_poisson_execution_private.v2.1"
PREDICATES = (
    "canonical_step_index",
    "sampled_owner_count_in_range",
    "one_vector_per_sampled_owner",
    "empty_flag_and_zero_norm_consistency",
    "noise_on_every_parameter_tensor",
    "optimizer_update_applied",
    "fixed_update_denominator",
    "finite_nonnegative_clipped_norms",
)
PRIVATE_INDEX_KEYS = {
    "schema_version",
    "visibility",
    "handling",
    "runs",
    "private_index_sha256",
}
PRIVATE_INDEX_RUN_KEYS = {
    "run_id",
    "research_seed",
    "private_manifest",
}
PRIVATE_MANIFEST_KEYS = {
    "schema_version",
    "visibility",
    "handling",
    "research_seed",
    "source_mapping_file_sha256",
    "source_mapping_canonical_sha256",
    "selected_mapping_canonical_sha256",
    "observed_source_owner_count",
    "observed_source_window_count",
    "observed_selected_owner_count",
    "observed_selected_window_count",
    "private_execution_binding_sha256",
    "initial_model_sha256",
    "final_model_sha256",
    "public_execution_sha256",
    "step_diagnostics",
    "private_manifest_sha256",
}
STEP_KEYS = {
    "step",
    "sampled_owner_count",
    "owner_vectors_computed",
    "empty_sample",
    "noise_applied",
    "noise_parameter_tensors",
    "optimizer_step_applied",
    "update_denominator",
    "max_unclipped_owner_norm",
    "max_clipped_owner_norm",
}


class PrivateDiagnosticAuditError(RuntimeError):
    """Raised when private diagnostic evidence fails closed."""


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PrivateDiagnosticAuditError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except PrivateDiagnosticAuditError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateDiagnosticAuditError(
            f"Could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PrivateDiagnosticAuditError(
            f"{path} must contain a JSON object"
        )
    return value


def _canonical_sha256(value: object) -> str:
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


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise PrivateDiagnosticAuditError(
            f"{context} keys mismatch; missing={missing}, extra={extra}"
        )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_self_hash(
    value: dict[str, Any],
    field: str,
    context: str,
) -> str:
    reported = value.get(field)
    if (
        not isinstance(reported, str)
        or len(reported) != 64
        or any(character not in "0123456789abcdef" for character in reported)
    ):
        raise PrivateDiagnosticAuditError(
            f"{context}.{field} is not a lowercase SHA-256"
        )
    payload = dict(value)
    payload.pop(field)
    observed = _canonical_sha256(payload)
    if observed != reported:
        raise PrivateDiagnosticAuditError(
            f"{context}.{field} mismatch"
        )
    return reported


def _safe_manifest_path(
    dataset_dir: Path,
    relative: object,
    run_id: str,
) -> Path:
    expected = (
        Path("_private")
        / run_id
        / "private_execution_manifest_v2.json"
    )
    if not isinstance(relative, str) or Path(relative) != expected:
        raise PrivateDiagnosticAuditError(
            f"{dataset_dir.name}.{run_id} private path is non-canonical"
        )
    candidate = (dataset_dir / expected).resolve()
    try:
        candidate.relative_to(dataset_dir.resolve())
    except ValueError as exc:
        raise PrivateDiagnosticAuditError(
            f"{dataset_dir.name}.{run_id} private path escapes dataset"
        ) from exc
    if not candidate.is_file():
        raise PrivateDiagnosticAuditError(
            f"{dataset_dir.name}.{run_id} private manifest is missing"
        )
    return candidate


def _audit_step(
    step: dict[str, Any],
    *,
    index: int,
    selected_owner_count: int,
    parameter_tensor_count: int,
    update_denominator: float,
    clip_norm: float,
    context: str,
) -> None:
    _require_exact_keys(step, STEP_KEYS, context)
    sampled = step["sampled_owner_count"]
    vectors = step["owner_vectors_computed"]
    noise_tensors = step["noise_parameter_tensors"]
    empty = step["empty_sample"]
    unclipped = step["max_unclipped_owner_norm"]
    clipped = step["max_clipped_owner_norm"]

    checks = {
        "canonical_step_index": (
            _is_int(step["step"]) and step["step"] == index
        ),
        "sampled_owner_count_in_range": (
            _is_int(sampled)
            and 0 <= sampled <= selected_owner_count
        ),
        "one_vector_per_sampled_owner": (
            _is_int(vectors) and vectors == sampled
        ),
        "empty_flag_and_zero_norm_consistency": (
            isinstance(empty, bool)
            and empty == (sampled == 0)
            and (
                (unclipped == 0.0 and clipped == 0.0)
                if empty
                else True
            )
        ),
        "noise_on_every_parameter_tensor": (
            step["noise_applied"] is True
            and _is_int(noise_tensors)
            and noise_tensors == parameter_tensor_count
        ),
        "optimizer_update_applied": (
            step["optimizer_step_applied"] is True
        ),
        "fixed_update_denominator": (
            _is_number(step["update_denominator"])
            and float(step["update_denominator"])
            == update_denominator
        ),
        "finite_nonnegative_clipped_norms": (
            _is_number(unclipped)
            and _is_number(clipped)
            and float(unclipped) >= 0.0
            and float(clipped) >= 0.0
            and float(clipped) <= clip_norm
            and (
                sampled == 0
                or float(clipped) <= float(unclipped) + 1e-12
            )
        ),
    }
    failed = [name for name in PREDICATES if not checks[name]]
    if failed:
        raise PrivateDiagnosticAuditError(
            f"{context} failed predicates: {failed}"
        )


def audit_collection(collection_dir: Path) -> dict[str, Any]:
    verified = verify_collection_artifacts_v2(
        collection_dir,
        source_root=ROOT,
    )
    dataset_reports: list[dict[str, Any]] = []
    total_steps = 0
    total_runs = 0

    for verified_dataset in verified.datasets:
        dataset = verified_dataset.dataset
        dataset_dir = collection_dir / dataset
        index_path = dataset_dir / "_private" / "private_run_index_v2.json"
        private_index = _load_object(index_path)
        _require_exact_keys(
            private_index,
            PRIVATE_INDEX_KEYS,
            f"{dataset}.private_index",
        )
        if (
            private_index["schema_version"] != PRIVATE_INDEX_SCHEMA
            or private_index["visibility"] != "private"
        ):
            raise PrivateDiagnosticAuditError(
                f"{dataset}.private_index metadata mismatch"
            )
        _require_self_hash(
            private_index,
            "private_index_sha256",
            f"{dataset}.private_index",
        )
        index_runs = private_index["runs"]
        if not isinstance(index_runs, list):
            raise PrivateDiagnosticAuditError(
                f"{dataset}.private_index.runs must be a list"
            )
        expected_run_ids = [
            run.run_id for run in verified_dataset.runs
        ]
        if len(index_runs) != len(expected_run_ids):
            raise PrivateDiagnosticAuditError(
                f"{dataset} private/public run counts differ"
            )

        dataset_steps = 0
        for index, (index_record, verified_run) in enumerate(
            zip(index_runs, verified_dataset.runs)
        ):
            context = f"{dataset}.{verified_run.run_id}"
            if not isinstance(index_record, dict):
                raise PrivateDiagnosticAuditError(
                    f"{context} private index entry must be an object"
                )
            _require_exact_keys(
                index_record,
                PRIVATE_INDEX_RUN_KEYS,
                f"{context}.private_index_entry",
            )
            if index_record["run_id"] != verified_run.run_id:
                raise PrivateDiagnosticAuditError(
                    f"{context} private run order mismatch at {index}"
                )
            if not _is_int(index_record["research_seed"]):
                raise PrivateDiagnosticAuditError(
                    f"{context} research seed is invalid"
                )
            manifest_path = _safe_manifest_path(
                dataset_dir,
                index_record["private_manifest"],
                verified_run.run_id,
            )
            manifest = _load_object(manifest_path)
            _require_exact_keys(
                manifest,
                PRIVATE_MANIFEST_KEYS,
                f"{context}.private_manifest",
            )
            if (
                manifest["schema_version"] != PRIVATE_MANIFEST_SCHEMA
                or manifest["visibility"] != "private"
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} private manifest metadata mismatch"
                )
            _require_self_hash(
                manifest,
                "private_manifest_sha256",
                f"{context}.private_manifest",
            )
            if (
                manifest["research_seed"]
                != index_record["research_seed"]
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} private seed link mismatch"
                )

            public_path = (
                dataset_dir
                / verified_run.run_id
                / "public_execution_v2.json"
            )
            model_path = (
                dataset_dir / verified_run.run_id / "model_v2.json"
            )
            public = _load_object(public_path)
            model = _load_object(model_path)
            if (
                manifest["public_execution_sha256"]
                != verified_run.public_execution_sha256
                or manifest["public_execution_sha256"]
                != public.get("public_execution_sha256")
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} public execution link mismatch"
                )
            model_output = public.get("model_output")
            if not isinstance(model_output, dict):
                raise PrivateDiagnosticAuditError(
                    f"{context}.model_output is malformed"
                )
            if (
                manifest["final_model_sha256"]
                != verified_run.model_state_sha256
                or manifest["final_model_sha256"]
                != model_output.get("state_sha256")
                or manifest["final_model_sha256"]
                != model.get("state_sha256")
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} final model link mismatch"
                )
            tensors = model.get("tensors")
            if not isinstance(tensors, list) or not tensors:
                raise PrivateDiagnosticAuditError(
                    f"{context}.model.tensors is malformed"
                )
            selected_owner_count = manifest[
                "observed_selected_owner_count"
            ]
            if (
                not _is_int(selected_owner_count)
                or selected_owner_count <= 0
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} selected owner count is invalid"
                )
            mechanism = public.get("mechanism")
            if not isinstance(mechanism, dict):
                raise PrivateDiagnosticAuditError(
                    f"{context}.mechanism is malformed"
                )
            total_steps_for_run = mechanism.get("total_steps")
            update_denominator = mechanism.get(
                "update_denominator"
            )
            clip_norm = mechanism.get("clip_norm")
            if (
                not _is_int(total_steps_for_run)
                or total_steps_for_run <= 0
                or not _is_number(update_denominator)
                or float(update_denominator) <= 0.0
                or not _is_number(clip_norm)
                or float(clip_norm) <= 0.0
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} public mechanism is invalid"
                )
            diagnostics = manifest["step_diagnostics"]
            if (
                not isinstance(diagnostics, list)
                or len(diagnostics) != total_steps_for_run
            ):
                raise PrivateDiagnosticAuditError(
                    f"{context} diagnostic schedule length mismatch"
                )
            for step_index, step in enumerate(diagnostics):
                if not isinstance(step, dict):
                    raise PrivateDiagnosticAuditError(
                        f"{context}.steps[{step_index}] must be an object"
                    )
                _audit_step(
                    step,
                    index=step_index,
                    selected_owner_count=selected_owner_count,
                    parameter_tensor_count=len(tensors),
                    update_denominator=float(update_denominator),
                    clip_norm=float(clip_norm),
                    context=f"{context}.steps[{step_index}]",
                )
            dataset_steps += total_steps_for_run

        dataset_runs = len(verified_dataset.runs)
        dataset_reports.append(
            {
                "dataset": dataset,
                "run_count": dataset_runs,
                "step_record_count": dataset_steps,
                "predicate_count_per_step": len(PREDICATES),
                "predicate_checks_passed": (
                    dataset_steps * len(PREDICATES)
                ),
            }
        )
        total_runs += dataset_runs
        total_steps += dataset_steps

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "visibility": "public_aggregate_no_realized_sampling_statistics",
        "status": "verified",
        "public_collection_sha256": (
            verified.public_collection_sha256
        ),
        "auditor_sha256": _file_sha256(Path(__file__).resolve()),
        "predicate_definitions": list(PREDICATES),
        "dataset_count": len(dataset_reports),
        "run_count": total_runs,
        "step_record_count": total_steps,
        "predicate_count_per_step": len(PREDICATES),
        "predicate_checks_passed": total_steps * len(PREDICATES),
        "datasets": dataset_reports,
    }
    report["audit_report_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection_dir",
        help="V2 collection containing public and private run records.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the aggregate JSON audit report.",
    )
    args = parser.parse_args()
    collection_dir = Path(args.collection_dir).resolve()
    report = audit_collection(collection_dir)
    encoded = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
