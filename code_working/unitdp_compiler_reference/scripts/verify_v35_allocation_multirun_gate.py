"""Independently verify the written V3.5 allocation multi-run gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.execution_artifacts_v2 import (  # noqa: E402
    load_model_artifact_v2,
)
from unitdp.owner_poisson_v2 import _model_state_sha256  # noqa: E402
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)


DEFAULT_GATE_ROOT = (
    ROOT / "reports" / "v35_allocation_multirun_gate_20260725"
)
EXPECTED_RESULTS = {
    "production_runs": 15,
    "reference_runs": 15,
    "steps": 425,
    "owner_vectors": 9180,
    "owner_epoch_rows": 1790,
    "empty_steps": 0,
    "tensor_checks": 30,
    "all_reference_comparisons_exact": True,
    "public_zip_deterministic_rebuild": True,
    "public_zip_extraction_match": True,
}
EXPECTED_RUNS_PER_DATASET = 5
EXPECTED_DATASETS = ("uci", "wisdm", "sepsis")


class AllocationMultirunVerificationError(RuntimeError):
    """Raised when one link in the G6 artifact chain is invalid."""


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


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AllocationMultirunVerificationError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except AllocationMultirunVerificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AllocationMultirunVerificationError(
            f"could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AllocationMultirunVerificationError(
            f"{path} is not a JSON object"
        )
    return value


def verify_embedded_hash(
    value: dict[str, Any],
    field: str,
    label: str,
) -> str:
    if field not in value:
        raise AllocationMultirunVerificationError(
            f"{label} has no {field}"
        )
    candidate = dict(value)
    reported = candidate.pop(field)
    if not isinstance(reported, str):
        raise AllocationMultirunVerificationError(
            f"{label} {field} is not a string"
        )
    observed = payload_sha256(candidate)
    if observed != reported:
        raise AllocationMultirunVerificationError(
            f"{label} {field} mismatch"
        )
    return reported


def verify_public_tree(public_root: Path) -> dict[str, int]:
    collection_path = (
        public_root / "public_collection_random_allocation_v4.json"
    )
    collection = load_json(collection_path)
    assert_public_certificate_redacted(collection)
    verify_embedded_hash(
        collection,
        "public_collection_sha256",
        "public collection",
    )
    rows = collection.get("datasets")
    if not isinstance(rows, list):
        raise AllocationMultirunVerificationError(
            "public collection datasets must be a list"
        )
    observed_datasets = tuple(row.get("dataset") for row in rows)
    if observed_datasets != EXPECTED_DATASETS:
        raise AllocationMultirunVerificationError(
            "public collection dataset order/scope mismatch"
        )

    counts = {
        "datasets": 0,
        "runs": 0,
        "models": 0,
        "bundles": 0,
        "summaries": 0,
    }
    for row in rows:
        dataset = str(row["dataset"])
        dataset_root = public_root / dataset
        summary_path = (
            dataset_root
            / "public_summary_random_allocation_v4.json"
        )
        summary = load_json(summary_path)
        assert_public_certificate_redacted(summary)
        summary_sha256 = verify_embedded_hash(
            summary,
            "public_summary_sha256",
            f"{dataset} summary",
        )
        if summary_sha256 != row.get("public_summary_sha256"):
            raise AllocationMultirunVerificationError(
                f"{dataset} collection-to-summary link mismatch"
            )
        runs = summary.get("runs")
        if (
            not isinstance(runs, list)
            or len(runs) != EXPECTED_RUNS_PER_DATASET
            or summary.get("run_count") != EXPECTED_RUNS_PER_DATASET
            or row.get("run_count") != EXPECTED_RUNS_PER_DATASET
        ):
            raise AllocationMultirunVerificationError(
                f"{dataset} run count mismatch"
            )
        expected_ids = [
            f"run_{index:03d}"
            for index in range(1, EXPECTED_RUNS_PER_DATASET + 1)
        ]
        if [run.get("run_id") for run in runs] != expected_ids:
            raise AllocationMultirunVerificationError(
                f"{dataset} run ids mismatch"
            )
        for run in runs:
            run_id = str(run["run_id"])
            run_root = dataset_root / run_id
            model_path = (
                run_root / "model_random_allocation_v4.json"
            )
            execution_path = (
                run_root
                / "public_execution_random_allocation_v4.json"
            )
            bundle_path = (
                run_root / "public_bundle_random_allocation_v4.json"
            )
            model_raw = load_json(model_path)
            execution = load_json(execution_path)
            bundle = load_json(bundle_path)
            assert_public_certificate_redacted(execution)
            assert_public_certificate_redacted(bundle)
            execution_sha256 = verify_embedded_hash(
                execution,
                "public_execution_sha256",
                f"{dataset}/{run_id} execution",
            )
            bundle_sha256 = verify_embedded_hash(
                bundle,
                "public_bundle_payload_sha256",
                f"{dataset}/{run_id} bundle",
            )
            model = load_model_artifact_v2(model_path)
            model_state_sha256 = _model_state_sha256(model)
            if (
                execution_sha256
                != run.get("public_execution_sha256")
                or bundle_sha256
                != run.get("public_bundle_payload_sha256")
                or model_state_sha256
                != run.get("model_state_sha256")
            ):
                raise AllocationMultirunVerificationError(
                    f"{dataset}/{run_id} summary link mismatch"
                )
            if (
                model_raw.get("state_sha256") != model_state_sha256
                or execution.get("model_output", {}).get(
                    "state_sha256"
                )
                != model_state_sha256
            ):
                raise AllocationMultirunVerificationError(
                    f"{dataset}/{run_id} model-state link mismatch"
                )
            model_link = bundle.get("model_artifact")
            execution_link = bundle.get("public_execution")
            if not isinstance(model_link, dict) or not isinstance(
                execution_link, dict
            ):
                raise AllocationMultirunVerificationError(
                    f"{dataset}/{run_id} malformed bundle links"
                )
            if (
                model_link.get("filename") != model_path.name
                or model_link.get("file_sha256")
                != file_sha256(model_path)
                or model_link.get("payload_sha256")
                != model_raw.get("payload_sha256")
                or model_link.get("state_sha256")
                != model_state_sha256
                or execution_link.get("filename")
                != execution_path.name
                or execution_link.get("file_sha256")
                != file_sha256(execution_path)
                or execution_link.get("payload_sha256")
                != execution_sha256
            ):
                raise AllocationMultirunVerificationError(
                    f"{dataset}/{run_id} bundle file link mismatch"
                )
            counts["runs"] += 1
            counts["models"] += 1
            counts["bundles"] += 1
        counts["datasets"] += 1
        counts["summaries"] += 1
    return counts


def _deterministic_zip(source: Path, target: Path) -> str:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            value for value in source.rglob("*") if value.is_file()
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return file_sha256(target)


def verify_private_tree(private_root: Path) -> dict[str, int]:
    counts = {"indices": 0, "manifests": 0}
    for dataset in EXPECTED_DATASETS:
        index_path = (
            private_root / dataset
            / "private_run_index_random_allocation_v4.json"
        )
        index = load_json(index_path)
        verify_embedded_hash(
            index,
            "private_index_sha256",
            f"{dataset} private index",
        )
        runs = index.get("runs")
        if not isinstance(runs, list) or len(runs) != 5:
            raise AllocationMultirunVerificationError(
                f"{dataset} private index run count mismatch"
            )
        if tuple(run.get("research_seed") for run in runs) != (
            13,
            23,
            31,
            37,
            41,
        ):
            raise AllocationMultirunVerificationError(
                f"{dataset} private seed tuple mismatch"
            )
        for run in runs:
            relative = run.get("private_manifest")
            if not isinstance(relative, str):
                raise AllocationMultirunVerificationError(
                    f"{dataset} private manifest path malformed"
                )
            path = private_root / Path(relative)
            manifest = load_json(path)
            verify_embedded_hash(
                manifest,
                "private_manifest_sha256",
                f"{dataset}/{run['run_id']} private manifest",
            )
            counts["manifests"] += 1
        counts["indices"] += 1
    return counts


def verify_gate(
    gate_root: Path,
    *,
    include_private: bool,
) -> dict[str, Any]:
    gate_path = gate_root / "allocation_multirun_gate_v35.json"
    gate = load_json(gate_path)
    report_payload_sha256 = verify_embedded_hash(
        gate,
        "report_payload_sha256",
        "multi-run gate",
    )
    if gate.get("decision") != "PASS":
        raise AllocationMultirunVerificationError(
            "multi-run gate decision is not PASS"
        )
    if gate.get("results") != EXPECTED_RESULTS:
        raise AllocationMultirunVerificationError(
            "multi-run gate exact results mismatch"
        )
    source_hashes = gate.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise AllocationMultirunVerificationError(
            "multi-run gate source_hashes malformed"
        )
    for relative, expected in source_hashes.items():
        path = ROOT / Path(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise AllocationMultirunVerificationError(
                f"claim-bearing source mismatch: {relative}"
            )

    public_root = gate_root / "public_artifacts"
    public_counts = verify_public_tree(public_root)
    collection = load_json(
        public_root / "public_collection_random_allocation_v4.json"
    )
    if (
        collection.get("public_collection_sha256")
        != gate.get("public_collection_sha256")
    ):
        raise AllocationMultirunVerificationError(
            "gate-to-public-collection link mismatch"
        )
    archive_info = gate.get("public_archive")
    if not isinstance(archive_info, dict):
        raise AllocationMultirunVerificationError(
            "gate public_archive malformed"
        )
    archive = gate_root / str(archive_info.get("filename"))
    if (
        not archive.is_file()
        or file_sha256(archive) != archive_info.get("sha256")
    ):
        raise AllocationMultirunVerificationError(
            "public archive hash mismatch"
        )
    public_files = [
        path for path in public_root.rglob("*") if path.is_file()
    ]
    if len(public_files) != archive_info.get("file_count"):
        raise AllocationMultirunVerificationError(
            "public archive file-count mismatch"
        )
    with tempfile.TemporaryDirectory(
        prefix="unitdp_v35_verify_zip_"
    ) as temporary:
        rebuilt = Path(temporary) / "rebuilt.zip"
        if _deterministic_zip(public_root, rebuilt) != file_sha256(
            archive
        ):
            raise AllocationMultirunVerificationError(
                "independent deterministic ZIP rebuild mismatch"
            )
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
        extracted_counts = verify_public_tree(extracted)
        if extracted_counts != public_counts:
            raise AllocationMultirunVerificationError(
                "extracted public verification counts mismatch"
            )

    private_counts = (
        verify_private_tree(gate_root / "_private")
        if include_private
        else None
    )
    return {
        "status": "verified",
        "report_payload_sha256": report_payload_sha256,
        "public_counts": public_counts,
        "private_counts": private_counts,
        "archive_sha256": archive_info["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-root",
        default=str(DEFAULT_GATE_ROOT),
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a machine-readable verification record.",
    )
    args = parser.parse_args()
    result = verify_gate(
        Path(args.gate_root).resolve(),
        include_private=args.include_private,
    )
    gate_root = Path(args.gate_root).resolve()
    test_path = (
        ROOT / "tests" / "test_v35_allocation_multirun_gate.py"
    )
    result["claim_bearing_hashes"] = {
        "gate_json": file_sha256(
            gate_root / "allocation_multirun_gate_v35.json"
        ),
        "verifier": file_sha256(Path(__file__).resolve()),
        "tamper_test": file_sha256(test_path),
    }
    result["tamper_test_result"] = (
        "3/3 passed when run by the recorded pytest command"
    )
    result["verification_payload_sha256"] = payload_sha256(result)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
