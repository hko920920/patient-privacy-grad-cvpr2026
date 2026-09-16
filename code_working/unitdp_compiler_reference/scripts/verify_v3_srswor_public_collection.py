"""Verify only the redistributable surface of the registered V3 collection.

This verifier deliberately makes no claim about private step diagnostics or
research randomizers.  It checks the public collection, summaries, execution
records, model artifacts, accountant values, source binding, and redaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
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
PUBLIC_RUN_FILES = {
    "model_srswor_v3.json",
    "public_bundle_srswor_v3.json",
    "public_execution_srswor_v3.json",
}


class PublicSrsworVerificationError(RuntimeError):
    """Raised when a redistributable V3 artifact is inconsistent."""


def payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_payload(
    path: Path,
    hash_field: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSrsworVerificationError(
            f"Could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PublicSrsworVerificationError(
            f"Expected a JSON object: {path}"
        )
    without_hash = dict(value)
    reported = without_hash.pop(hash_field, None)
    if reported != payload_sha256(without_hash):
        raise PublicSrsworVerificationError(
            f"Payload hash mismatch: {path}"
        )
    return value


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


def verify_public_dataset(
    evidence_root: Path,
    dataset: str,
    *,
    source_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_dir = evidence_root / dataset
    summary = load_hashed_payload(
        dataset_dir / "public_summary_srswor_v3.json",
        "public_summary_sha256",
    )
    assert_public_certificate_redacted(summary)
    profile = get_srswor_benchmark_profile_v3(
        summary["benchmark_profile_id"]
    )
    if summary["benchmark_profile_sha256"] != profile.registry_sha256:
        raise PublicSrsworVerificationError(
            f"Benchmark profile mismatch: {dataset}"
        )
    if (
        summary["execution_source_bundle_sha256"]
        != source_bundle_sha256
    ):
        raise PublicSrsworVerificationError(
            f"Source bundle mismatch: {dataset}"
        )
    run_count = summary.get("run_count")
    if (
        isinstance(run_count, bool)
        or not isinstance(run_count, int)
        or run_count != 5
    ):
        raise PublicSrsworVerificationError(
            f"Expected five registered runs: {dataset}"
        )

    executions: list[dict[str, Any]] = []
    model_roundtrips = 0
    for index in range(1, run_count + 1):
        run_id = f"run_{index:03d}"
        run_dir = dataset_dir / run_id
        observed_files = {
            path.name for path in run_dir.iterdir() if path.is_file()
        }
        if observed_files != PUBLIC_RUN_FILES:
            raise PublicSrsworVerificationError(
                f"Unexpected public run files: {dataset}/{run_id}"
            )
        execution_path = (
            run_dir / "public_execution_srswor_v3.json"
        )
        execution = load_hashed_payload(
            execution_path,
            "public_execution_sha256",
        )
        bundle = load_hashed_payload(
            run_dir / "public_bundle_srswor_v3.json",
            "public_bundle_payload_sha256",
        )
        assert_public_certificate_redacted(execution)
        assert_public_certificate_redacted(bundle)
        if (
            execution["visibility"] != "public"
            or execution["release_status"] != "research_non_release"
            or execution["execution_source_bundle_sha256"]
            != source_bundle_sha256
            or execution["benchmark_profile_id"] != profile.profile_id
        ):
            raise PublicSrsworVerificationError(
                f"Execution metadata mismatch: {dataset}/{run_id}"
            )
        if (
            bundle["visibility"] != "public"
            or bundle["release_status"] != "research_non_release"
            or bundle["public_execution"]["filename"]
            != execution_path.name
            or bundle["public_execution"]["file_sha256"]
            != file_sha256(execution_path)
            or bundle["public_execution"]["payload_sha256"]
            != execution["public_execution_sha256"]
        ):
            raise PublicSrsworVerificationError(
                f"Execution bundle mismatch: {dataset}/{run_id}"
            )

        model_path = run_dir / "model_srswor_v3.json"
        model_raw = load_hashed_payload(model_path, "payload_sha256")
        model = load_model_artifact_v2(model_path)
        state_sha256 = _model_state_sha256(model)
        model_binding = bundle["model_artifact"]
        if (
            model_binding["filename"] != model_path.name
            or model_binding["file_sha256"] != file_sha256(model_path)
            or model_binding["payload_sha256"]
            != model_raw["payload_sha256"]
            or model_binding["state_sha256"] != state_sha256
            or model_raw["state_sha256"] != state_sha256
            or execution["model_output"]["state_sha256"]
            != state_sha256
        ):
            raise PublicSrsworVerificationError(
                f"Model binding mismatch: {dataset}/{run_id}"
            )

        registered = summary["runs"][index - 1]
        if registered != {
            "run_id": run_id,
            "public_execution_sha256": execution[
                "public_execution_sha256"
            ],
            "public_bundle_payload_sha256": bundle[
                "public_bundle_payload_sha256"
            ],
            "model_state_sha256": state_sha256,
            "public_evaluation": execution["public_evaluation"],
        }:
            raise PublicSrsworVerificationError(
                f"Summary run index mismatch: {dataset}/{run_id}"
            )
        executions.append(execution)
        model_roundtrips += 1

    for metric in ("accuracy", "macro_f1", "auroc", "auprc"):
        if summary["aggregate"][metric] != metric_summary(
            executions,
            metric,
        ):
            raise PublicSrsworVerificationError(
                f"Aggregate mismatch: {dataset}/{metric}"
            )

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
            float(accountant["epsilon_direct_theorem_oracle"])
            - direct.epsilon
        )
        > 1e-12
        or accountant["optimal_order_direct_theorem_oracle"]
        != direct.optimal_order
        or float(accountant["absolute_gap"]) > 1e-10
    ):
        raise PublicSrsworVerificationError(
            f"Accountant mismatch: {dataset}"
        )

    return (
        {
            "dataset": dataset,
            "run_count": run_count,
            "model_roundtrips": model_roundtrips,
            "public_metric_values": run_count * 4,
            "public_summary_sha256": summary[
                "public_summary_sha256"
            ],
        },
        summary,
    )


def verify_public_collection(evidence_root: Path) -> dict[str, Any]:
    root = evidence_root.resolve()
    source_bundle_sha256 = (
        execution_source_bundle_sha256_srswor_v3()
    )
    collection = load_hashed_payload(
        root / "public_collection_srswor_v3.json",
        "public_collection_sha256",
    )
    assert_public_certificate_redacted(collection)
    if (
        collection["visibility"] != "public"
        or collection["release_status"] != "research_non_release"
    ):
        raise PublicSrsworVerificationError(
            "Collection metadata mismatch"
        )

    results: list[dict[str, Any]] = []
    expected_index: list[dict[str, Any]] = []
    for dataset in DATASETS:
        result, summary = verify_public_dataset(
            root,
            dataset,
            source_bundle_sha256=source_bundle_sha256,
        )
        results.append(result)
        expected_index.append(
            {
                "dataset": dataset,
                "public_summary_sha256": summary[
                    "public_summary_sha256"
                ],
                "run_count": 5,
            }
        )
    if collection["datasets"] != expected_index:
        raise PublicSrsworVerificationError(
            "Collection summary index mismatch"
        )
    observed_datasets = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name != "_private"
    }
    if observed_datasets != set(DATASETS):
        raise PublicSrsworVerificationError(
            "Collection dataset directories mismatch"
        )

    report: dict[str, Any] = {
        "schema_version": (
            "unitdp.srswor_public_collection_verification.v3"
        ),
        "visibility": "public",
        "status": "verified",
        "scope": (
            "public hashes, redaction, models, metrics, accountant, "
            "and current source binding; no private-step claim"
        ),
        "execution_source_bundle_sha256": source_bundle_sha256,
        "public_collection_sha256": collection[
            "public_collection_sha256"
        ],
        "dataset_count": len(results),
        "run_count": sum(row["run_count"] for row in results),
        "model_roundtrips": sum(
            row["model_roundtrips"] for row in results
        ),
        "public_metric_values": sum(
            row["public_metric_values"] for row in results
        ),
        "datasets": results,
        "verifier_sha256": file_sha256(Path(__file__).resolve()),
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
    parser.add_argument("--output", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = verify_public_collection(Path(args.evidence_dir))
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
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
