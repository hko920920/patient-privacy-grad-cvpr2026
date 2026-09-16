#!/usr/bin/env python3
"""Verify the anonymous, raw-data-free V16 external evidence summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTABLE_ROOT = (
    ROOT / "reports" / "audit_external_extrasensory_v16_portable_001"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class Gate:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks[name] = {
            "passed": bool(passed),
            "detail": detail,
        }

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(
            record["passed"] for record in self.checks.values()
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PORTABLE_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    gate = Gate()
    manifest_path = root / "manifest.json"
    evidence_path = root / "portable_evidence.json"
    excerpt_path = root / "execution_stdout_excerpt.txt"
    summary_path = root / "summary.md"
    manifest = load_json(manifest_path)
    evidence = load_json(evidence_path)
    excerpt = excerpt_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8")

    gate.check(
        "manifest_schema",
        manifest.get("schema_version")
        == "audit_external_extrasensory_v16_portable_manifest.1",
        manifest.get("schema_version"),
    )
    expected_paths = {
        "portable_evidence.json",
        "execution_stdout_excerpt.txt",
        "summary.md",
    }
    records = manifest.get("files", [])
    actual_paths = {record.get("path") for record in records}
    gate.check(
        "manifest_exact_file_set",
        len(records) == 3 and actual_paths == expected_paths,
        sorted(actual_paths),
    )
    recomputed_records = []
    for name in sorted(expected_paths):
        path = root / name
        recomputed_records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    gate.check(
        "manifest_file_hashes",
        sorted(records, key=lambda row: row["path"])
        == recomputed_records,
        recomputed_records,
    )
    canonical = hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    gate.check(
        "manifest_canonical_inventory",
        canonical == manifest.get("canonical_inventory_sha256"),
        canonical,
    )

    rendered = json.dumps(evidence, ensure_ascii=False) + excerpt + summary
    forbidden = (
        "SOGANG",
        "C:\\Users\\",
        "C:/Users/",
        "/home/",
        "\\Users\\",
    )
    matches = [token for token in forbidden if token in rendered]
    gate.check("anonymous_paths", not matches, matches)

    source = evidence.get("external_source", {})
    gate.check(
        "external_source_identity",
        source == {
            "repository": "https://github.com/jatanloya/extrasensory-dp",
            "commit": "167e0a899cb73c6e20398e8af886517cc9357c3e",
            "tree": "1106aa01d2f8806e72d4f99f5fc60814b3fdc9c7",
            "license": "MIT",
            "raw_data_redistributed": False,
        },
        source,
    )
    attempts = evidence.get("execution_sequence", [])
    gate.check(
        "all_six_terminal_records",
        [row.get("sequence") for row in attempts]
        == [
            "V12-1",
            "V12-2",
            "V13-1",
            "V14-1",
            "V15-1",
            "V16-1",
        ]
        and [row.get("outcome") for row in attempts]
        == [
            "failed",
            "failed",
            "failed",
            "failed_before_external_source_invocation",
            "failed",
            "author_run_external_execution_complete",
        ],
        [
            {
                "sequence": row.get("sequence"),
                "outcome": row.get("outcome"),
            }
            for row in attempts
        ],
    )
    selection = evidence.get("selection", {})
    gate.check(
        "no_result_selection",
        selection == {
            "all_terminal_attempts_disclosed": True,
            "successful_target_count": 1,
            "seed_selection": False,
            "hyperparameter_selection": False,
            "utility_selection": False,
        },
        selection,
    )
    raw = evidence.get("raw_data_identity", {})
    gate.check(
        "raw_identity_without_redistribution",
        raw == {
            "file_count": 60,
            "total_bytes": 225325871,
            "canonical_inventory_sha256": (
                "36ac7814eb5e4710582a9b8d34881f0fe"
                "1e0e8bf17d033ef7bc23c644d16064d"
            ),
            "redistributed": False,
        },
        raw,
    )
    environment = evidence.get("environment", {})
    compatibility = environment.get("compatibility_overlay", {})
    gate.check(
        "disclosed_compatibility_overlay",
        environment.get("package_metadata_versions", {}).get("pandas")
        == "3.0.3"
        and environment.get("target_runtime_packages", {}).get("pandas")
        == "2.2.3"
        and environment.get("package_metadata_versions", {}).get("seaborn")
        == "0.12.2"
        and compatibility.get("wheel_bytes") == 11_617_166
        and compatibility.get("wheel_sha256")
        == "3fc6873a41186404dad67245896a6e440baacc92f5b716ccd1bc9ed2995ab2c5"
        and compatibility.get("inventory")
        == {
            "file_count": 1558,
            "total_bytes": 39410621,
            "canonical_inventory_sha256": (
                "c35335647b4ac1ab13346a63770bfe048"
                "d1d310f43c2e3568f9deef6f03ea34d"
            ),
            "pycache_directory_count": 0,
        }
        and "unpinned" in compatibility.get("reason", ""),
        environment,
    )

    execution = evidence.get("execution", {})
    inventory = execution.get("output_inventory", [])
    gate.check(
        "execution_complete_17_outputs",
        execution.get("author_operated") is True
        and execution.get("post_failure") is True
        and execution.get("external_source_end_to_end") is True
        and execution.get("output_file_count") == 17
        and len(inventory) == 17
        and len({row.get("path") for row in inventory}) == 17,
        {
            "duration_seconds": execution.get("duration_seconds"),
            "output_file_count": execution.get("output_file_count"),
        },
    )
    dataset = execution.get("dataset", {})
    gate.check(
        "dataset_shape_summary",
        dataset.get("X_train", {}).get("shape") == [250906, 115]
        and dataset.get("y_train", {}).get("shape") == [250906, 3]
        and dataset.get("X_test", {}).get("shape", [None, None])[1] == 115
        and dataset.get("y_test", {}).get("shape", [None, None])[1] == 3
        and all(
            record.get("all_finite") is True
            for record in dataset.values()
        ),
        dataset,
    )
    modes = execution.get("modes", {})
    gate.check(
        "five_epoch_metric_summaries",
        set(modes) == {"non_private", "private"}
        and all(
            modes[mode]["performance"][key]["shape"] == [5]
            and len(modes[mode]["performance"][key]["values"]) == 5
            and modes[mode]["performance"][key]["all_finite"] is True
            for mode in ("non_private", "private")
            for key in (
                "train_losses",
                "test_losses",
                "train_accs",
                "test_accs",
            )
        ),
        {
            mode: modes.get(mode, {}).get("performance")
            for mode in ("non_private", "private")
        },
    )
    gate.check(
        "per_sample_and_roc_summaries",
        all(
            modes[mode]["per_sample_losses"][key]["size"] > 0
            and modes[mode]["per_sample_losses"][key]["all_finite"] is True
            for mode in ("non_private", "private")
            for key in (
                "per_sample_train_losses",
                "per_sample_test_losses",
            )
        )
        and all(
            modes[mode]["mia_roc"][key]["size"] > 0
            and modes[mode]["mia_roc"][key]["all_finite"] is True
            for mode in ("non_private", "private")
            for key in ("fpr", "tpr")
        ),
        {
            mode: {
                "losses": modes.get(mode, {}).get("per_sample_losses"),
                "roc": modes.get(mode, {}).get("mia_roc"),
            }
            for mode in ("non_private", "private")
        },
    )

    verification = evidence.get("verification", {})
    gate.check(
        "separate_verification_and_tamper",
        verification.get("checks_total", 0) > 0
        and verification.get("checks_passed")
        == verification.get("checks_total")
        and verification.get("decision")
        == "AUTHOR_RUN_EXTERNAL_EXECUTION_COMPLETE_CLAIM_UNVERIFIED"
        and verification.get("tamper_rejected") is True
        and verification.get("live_result_unchanged_by_tamper") is True,
        verification,
    )
    boundary = evidence.get("claim_boundary", {})
    gate.check(
        "claim_boundary",
        boundary.get("registered_audit_claim_verified") is False
        and boundary.get("positive_dp_wording") == ""
        and boundary.get("independent_operator_evidence") is False
        and boundary.get("secure_rng") is False
        and boundary.get("accountant_achieved_epsilon_recorded") is False
        and boundary.get("configured_target_epsilon_only") is True,
        boundary,
    )
    omissions = evidence.get("anonymous_package_omissions", {})
    gate.check(
        "nonredistribution_boundary",
        omissions == {
            "raw_data": "not redistributed",
            "processed_dataset": "hash/shape record only",
            "per_sample_arrays": "hash/count record only",
            "checkpoints": "hash/structure record only",
            "unsanitized_logs": (
                "hash record plus allowlisted excerpt only"
            ),
            "absolute_paths": "removed",
        },
        omissions,
    )
    epoch_lines = [
        line for line in excerpt.splitlines()
        if line.startswith("Train Epoch:")
    ]
    gate.check(
        "allowlisted_execution_excerpt",
        len(epoch_lines) == 10
        and sum("(Epsilon =" in line for line in epoch_lines) == 5
        and "Traceback" not in excerpt
        and "Users" not in excerpt,
        epoch_lines,
    )
    gate.check(
        "summary_boundary",
        "not independent-operator evidence" in summary
        and "does not verify the registered" in summary
        and "Positive DP" in summary
        and "Raw data" in summary,
        summary,
    )

    decision = (
        "PORTABLE_AUTHOR_RUN_EXECUTION_EVIDENCE_VALID"
        if gate.passed
        else "REJECTED"
    )
    payload = {
        "schema_version": (
            "audit_external_extrasensory_v16_portable_verification.1"
        ),
        "checks": gate.checks,
        "check_count": len(gate.checks),
        "passed_count": sum(
            record["passed"] for record in gate.checks.values()
        ),
        "passed": gate.passed,
        "decision": decision,
        "positive_dp_wording": "",
        "independent_operator_evidence": False,
    }
    write_json(output, payload)
    print(
        f"passed={str(gate.passed).lower()} "
        f"checks={payload['passed_count']}/{payload['check_count']} "
        f"decision={decision}"
    )
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
