"""Verify public artifacts and private invariants for registered V3 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_registry_srswor_v3 import (  # noqa: E402
    get_srswor_benchmark_profile_v3,
)
from unitdp.execution_artifacts_v2 import (  # noqa: E402
    load_model_artifact_v2,
)
from unitdp.owner_poisson_v2 import _model_state_sha256  # noqa: E402
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp.source_bundle_srswor_v3 import (  # noqa: E402
    execution_source_bundle_sha256_srswor_v3,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (  # noqa: E402
    fixed_size_srswor_epsilon_v3,
)


DATASETS = ("uci", "wisdm", "sepsis")


def payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_hashed_payload(path: Path, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    without_hash = dict(payload)
    reported = without_hash.pop(hash_field)
    if payload_sha256(without_hash) != reported:
        raise AssertionError(f"Payload hash mismatch: {path}")
    return payload


def metric_summary(
    executions: list[dict[str, Any]],
    key: str,
) -> dict[str, float | int | None]:
    values = [
        float(execution["public_evaluation"][key])
        for execution in executions
        if execution["public_evaluation"][key] is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def verify_dataset(
    root: Path,
    dataset: str,
    source_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_dir = root / dataset
    summary = load_hashed_payload(
        dataset_dir / "public_summary_srswor_v3.json",
        "public_summary_sha256",
    )
    assert_public_certificate_redacted(summary)
    profile = get_srswor_benchmark_profile_v3(
        summary["benchmark_profile_id"]
    )
    if summary["benchmark_profile_sha256"] != profile.registry_sha256:
        raise AssertionError("Benchmark profile hash mismatch")
    if (
        summary["execution_source_bundle_sha256"]
        != source_bundle_sha256
    ):
        raise AssertionError("Summary source-bundle hash mismatch")
    run_count = summary["run_count"]
    if (
        isinstance(run_count, bool)
        or not isinstance(run_count, int)
        or run_count <= 0
    ):
        raise AssertionError("Invalid run count")

    public_executions: list[dict[str, Any]] = []
    private_randomizers: set[int] = set()
    exact_sample_steps = 0
    distinct_sample_steps = 0
    bounded_clip_steps = 0
    noised_update_steps = 0
    total_owner_vectors = 0
    model_roundtrips = 0
    for index in range(1, run_count + 1):
        run_id = f"run_{index:03d}"
        run_dir = dataset_dir / run_id
        execution_path = (
            run_dir / "public_execution_srswor_v3.json"
        )
        execution = load_hashed_payload(
            execution_path,
            "public_execution_sha256",
        )
        assert_public_certificate_redacted(execution)
        if (
            execution["execution_source_bundle_sha256"]
            != source_bundle_sha256
        ):
            raise AssertionError("Run source-bundle hash mismatch")
        if execution["benchmark_profile_id"] != profile.profile_id:
            raise AssertionError("Run profile mismatch")
        bundle = load_hashed_payload(
            run_dir / "public_bundle_srswor_v3.json",
            "public_bundle_payload_sha256",
        )
        assert_public_certificate_redacted(bundle)
        if (
            bundle["public_execution"]["file_sha256"]
            != file_sha256(execution_path)
            or bundle["public_execution"]["payload_sha256"]
            != execution["public_execution_sha256"]
        ):
            raise AssertionError("Public execution bundle mismatch")
        model_path = run_dir / "model_srswor_v3.json"
        if (
            bundle["model_artifact"]["file_sha256"]
            != file_sha256(model_path)
        ):
            raise AssertionError("Model file hash mismatch")
        model = load_model_artifact_v2(model_path)
        state_hash = _model_state_sha256(model)
        if (
            state_hash != bundle["model_artifact"]["state_sha256"]
            or state_hash != execution["model_output"]["state_sha256"]
        ):
            raise AssertionError("Model state hash mismatch")
        model_roundtrips += 1

        private = load_hashed_payload(
            dataset_dir
            / "_private"
            / run_id
            / "private_execution_srswor_v3.json",
            "private_manifest_sha256",
        )
        if (
            private["public_execution_sha256"]
            != execution["public_execution_sha256"]
            or private["final_model_sha256"] != state_hash
        ):
            raise AssertionError("Private/public execution link mismatch")
        randomizer = private["research_seed"]
        if (
            isinstance(randomizer, bool)
            or not isinstance(randomizer, int)
            or randomizer in private_randomizers
        ):
            raise AssertionError("Run randomizers must be distinct integers")
        private_randomizers.add(randomizer)
        steps = private["step_diagnostics"]
        expected_steps = profile.base_profile.total_steps
        if len(steps) != expected_steps:
            raise AssertionError("Private step count mismatch")
        for step in steps:
            if (
                step["sampled_owner_count"] == profile.sample_size
                and step["owner_vectors_computed"] == profile.sample_size
            ):
                exact_sample_steps += 1
            positions = step["sampled_owner_positions"]
            if (
                len(positions) == profile.sample_size
                and len(set(positions)) == profile.sample_size
                and all(
                    0 <= int(position) < profile.source_dataset_size
                    for position in positions
                )
            ):
                distinct_sample_steps += 1
            if (
                step["max_clipped_owner_norm"]
                <= profile.base_profile.clip_norm
            ):
                bounded_clip_steps += 1
            if (
                step["noise_parameter_tensors"] == 2
                and step["optimizer_step_applied"] is True
            ):
                noised_update_steps += 1
            total_owner_vectors += step["owner_vectors_computed"]
        public_executions.append(execution)

    expected_total_steps = run_count * profile.base_profile.total_steps
    if not (
        exact_sample_steps
        == distinct_sample_steps
        == bounded_clip_steps
        == noised_update_steps
        == expected_total_steps
    ):
        raise AssertionError("At least one private step invariant failed")
    if len(private_randomizers) != run_count:
        raise AssertionError("Private run-randomizer count mismatch")

    for key in ("accuracy", "macro_f1", "auroc", "auprc"):
        if summary["aggregate"][key] != metric_summary(
            public_executions,
            key,
        ):
            raise AssertionError(f"Aggregate mismatch: {dataset}:{key}")
    for index, execution in enumerate(
        public_executions,
        start=1,
    ):
        registered = summary["runs"][index - 1]
        if (
            registered["run_id"] != f"run_{index:03d}"
            or registered["public_execution_sha256"]
            != execution["public_execution_sha256"]
            or registered["model_state_sha256"]
            != execution["model_output"]["state_sha256"]
        ):
            raise AssertionError("Summary run index mismatch")

    direct = fixed_size_srswor_epsilon_v3(
        actual_noise_multiplier=profile.noise_multiplier,
        source_dataset_size=profile.source_dataset_size,
        sample_size=profile.sample_size,
        steps=profile.base_profile.total_steps,
        delta=profile.delta,
    )
    accountant = summary["accountant"]
    if (
        abs(
            accountant["epsilon_direct_theorem_oracle"]
            - direct.epsilon
        )
        > 1e-12
        or accountant["optimal_order_direct_theorem_oracle"]
        != direct.optimal_order
        or accountant["absolute_gap"] > 1e-10
    ):
        raise AssertionError("Summary accountant verification mismatch")

    public_result = {
        "dataset": dataset,
        "benchmark_profile_id": profile.profile_id,
        "run_count": run_count,
        "verified_steps": expected_total_steps,
        "verified_owner_vectors": total_owner_vectors,
        "exact_sample_size_steps": exact_sample_steps,
        "distinct_without_replacement_steps": distinct_sample_steps,
        "clip_bound_steps": bounded_clip_steps,
        "noised_optimizer_steps": noised_update_steps,
        "model_roundtrips": model_roundtrips,
        "distinct_private_randomizers": len(private_randomizers),
        "public_summary_sha256": summary["public_summary_sha256"],
    }
    return public_result, summary


def verify(root: Path) -> dict[str, Any]:
    source_bundle_sha256 = (
        execution_source_bundle_sha256_srswor_v3()
    )
    collection = load_hashed_payload(
        root / "public_collection_srswor_v3.json",
        "public_collection_sha256",
    )
    assert_public_certificate_redacted(collection)
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        result, summary = verify_dataset(
            root,
            dataset,
            source_bundle_sha256,
        )
        results.append(result)
        summaries.append(summary)
    expected_collection = [
        {
            "dataset": summary["dataset"],
            "public_summary_sha256": summary[
                "public_summary_sha256"
            ],
            "run_count": summary["run_count"],
        }
        for summary in summaries
    ]
    if collection["datasets"] != expected_collection:
        raise AssertionError("Public collection index mismatch")
    report: dict[str, Any] = {
        "schema_version": (
            "unitdp.srswor_benchmark_verification_public.v3"
        ),
        "visibility": "public",
        "verified": True,
        "private_invariants_checked_without_public_trace_disclosure": True,
        "execution_source_bundle_sha256": source_bundle_sha256,
        "dataset_count": len(results),
        "run_count": sum(item["run_count"] for item in results),
        "verified_steps": sum(
            item["verified_steps"] for item in results
        ),
        "verified_owner_vectors": sum(
            item["verified_owner_vectors"] for item in results
        ),
        "model_roundtrips": sum(
            item["model_roundtrips"] for item in results
        ),
        "datasets": results,
        "source_files_sha256": {
            "scripts/verify_v3_srswor_registered_benchmarks.py": (
                file_sha256(Path(__file__))
            ),
            "src/unitdp_spec_oracle/srswor_rdp_oracle_v3.py": (
                file_sha256(
                    ROOT
                    / "src"
                    / "unitdp_spec_oracle"
                    / "srswor_rdp_oracle_v3.py"
                )
            ),
        },
    }
    report["payload_sha256"] = payload_sha256(report)
    assert_public_certificate_redacted(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        default=str(
            ROOT
            / "reports"
            / "v3_srswor_registered_5seed_v32_20260724"
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional public verification JSON path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = verify(Path(args.evidence_dir))
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
