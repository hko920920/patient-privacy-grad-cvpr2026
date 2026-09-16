#!/usr/bin/env python3
"""Independently summarize the five Audit v5 UCI HAR confirmatory runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = (
    "ca3c50c575a25cbc996bc363783d1ae10c3dd7352d4861e62e774324aaa91070"
)
PIPELINE_SHA256 = (
    "0c226ad3acb1d0fcc7a08904288611877269a173dfe30e6d3bcee75fa2b9d0bf"
)
VERIFIER_SHA256 = (
    "14193a2f8faffe7d1998ecbb62307556ceb4cc67eb3793161c4e52f567e08b26"
)
REGISTRY_SHA256 = (
    "9344d1549f3db3265b45e4c0f10a19155ea20e77a14fd914c3d20a12a09e2834"
)
EXPECTED_RUNS = tuple(f"confirmatory_run{index:02d}" for index in range(1, 6))
METRICS = (
    "train_accuracy",
    "test_accuracy",
    "test_macro_f1",
    "pipeline_seconds",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def add_check(
    checks: Dict[str, Dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    manifest_path = root / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = dict(manifest)
    reported_content_hash = content.pop("content_sha256")
    add_check(
        checks,
        "manifest_content_digest",
        canonical_json_sha256(content) == reported_content_hash,
        {
            "actual": canonical_json_sha256(content),
            "reported": reported_content_hash,
        },
    )
    add_check(
        checks,
        "protocol",
        manifest["protocol"]["sha256"] == PROTOCOL_SHA256,
        manifest["protocol"],
    )
    snapshot = manifest["source_snapshot"]
    add_check(
        checks,
        "source_snapshot",
        snapshot["pipeline"]["sha256"] == PIPELINE_SHA256
        and snapshot["verifier"]["sha256"] == VERIFIER_SHA256
        and snapshot["registry"]["sha256"] == REGISTRY_SHA256,
        snapshot,
    )
    add_check(
        checks,
        "run_order",
        tuple(manifest["run_order"]) == EXPECTED_RUNS
        and tuple(run["run_id"] for run in manifest["runs"]) == EXPECTED_RUNS,
        manifest["run_order"],
    )
    add_check(
        checks,
        "pilot_excluded",
        manifest["pilot"]["excluded_from_aggregate"] is True
        and "before confirmatory protocol freeze"
        in manifest["pilot"]["exclusion_basis"],
        manifest["pilot"],
    )

    rows = []
    model_hashes = []
    decision_signatures = []
    for run_record in manifest["runs"]:
        run_id = run_record["run_id"]
        run_dir = root / run_id
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        full = json.loads(
            (run_dir / "independent_verification.json").read_text(encoding="utf-8")
        )
        portable = json.loads(
            (
                run_dir / "independent_verification_portable.json"
            ).read_text(encoding="utf-8")
        )
        certificates = {
            claim: json.loads(
                (
                    run_dir
                    / "certificates"
                    / f"certificate_uci_har_v5_generated_window_{claim}.json"
                ).read_text(encoding="utf-8")
            )
            for claim in ("window", "event", "owner")
        }
        add_check(
            checks,
            f"{run_id}.full_verifier",
            full["all_passed"]
            and full["checks_passed"] == full["checks_total"] == 49
            and full["raw_inputs_rescanned"] is True,
            {
                "passed": full["checks_passed"],
                "total": full["checks_total"],
                "raw": full["raw_inputs_rescanned"],
            },
        )
        add_check(
            checks,
            f"{run_id}.portable_verifier",
            portable["all_passed"]
            and portable["checks_passed"] == portable["checks_total"] == 38
            and portable["raw_inputs_rescanned"] is False,
            {
                "passed": portable["checks_passed"],
                "total": portable["checks_total"],
                "raw": portable["raw_inputs_rescanned"],
            },
        )
        add_check(
            checks,
            f"{run_id}.manifest_hashes",
            all(
                file_sha256(run_dir / relative) == digest
                for relative, digest in run_record["artifact_hashes"].items()
            ),
            len(run_record["artifact_hashes"]),
        )
        signature = tuple(
            (
                claim,
                certificates[claim]["release_status"],
                certificates[claim]["unit_path"],
                certificates[claim].get("stability_bound"),
                bool(certificates[claim]["supported_statement"]),
            )
            for claim in ("window", "event", "owner")
        )
        decision_signatures.append(signature)
        add_check(
            checks,
            f"{run_id}.decision_boundary",
            signature
            == (
                ("window", "ALLOWED", "DIRECT", 1, True),
                ("event", "BLOCKED_UNVERIFIED", "UNDERSPECIFIED", None, False),
                ("owner", "BLOCKED_UNVERIFIED", "UNDERSPECIFIED", None, False),
            ),
            signature,
        )
        model_hash = file_sha256(run_dir / "released_model.json")
        model_hashes.append(model_hash)
        row = {
            "run_id": run_id,
            "model_sha256": model_hash,
            **{metric: float(metrics[metric]) for metric in METRICS},
        }
        rows.append(row)

    add_check(
        checks,
        "distinct_model_hashes",
        len(set(model_hashes)) == len(EXPECTED_RUNS),
        model_hashes,
    )
    add_check(
        checks,
        "invariant_decisions",
        len(set(decision_signatures)) == 1,
        decision_signatures[0],
    )
    stats = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        add_check(
            checks,
            f"{metric}.finite",
            all(math.isfinite(value) for value in values),
            values,
        )
        stats[metric] = {
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    passed = sum(item["passed"] for item in checks.values())
    summary: Dict[str, Any] = {
        "schema_version": "uci_har_v5_confirmatory_summary_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_manifest_sha256": file_sha256(manifest_path),
        "run_ids": list(EXPECTED_RUNS),
        "pilot_excluded": True,
        "model_hashes": model_hashes,
        "decision_signature": decision_signatures[0],
        "statistics": stats,
        "verification": {
            "checks_passed": passed,
            "checks_total": len(checks),
            "all_passed": passed == len(checks),
            "checks": checks,
        },
    }
    summary["content_sha256"] = canonical_json_sha256(summary)
    write_json(root / "confirmatory_summary.json", summary)
    write_csv(root / "confirmatory_metrics.csv", rows)
    lines = [
        "# Audit v5 UCI HAR five-run confirmatory summary",
        "",
        "- Confirmatory runs: 5/5 retained",
        "- Full independent verification: 49/49 for every run",
        "- Portable verification: 38/38 for every run",
        f"- Distinct model hashes: {len(set(model_hashes))}/5",
        "- Decisions: window ALLOWED/DIRECT/K=1; event and owner BLOCKED_UNVERIFIED",
        "",
        "| Metric | Mean | Sample SD | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        item = stats[metric]
        lines.append(
            f"| {metric} | {item['mean']:.6f} | {item['sample_sd']:.6f} | "
            f"{item['min']:.6f} | {item['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The protocol-fixed pilot is excluded chronologically. This result",
            "reduces one-dataset concentration but does not establish",
            "cross-mechanism universality, raw-event or owner privacy for UCI HAR,",
            "state-of-the-art utility, or execution attestation.",
            "",
        ]
    )
    (root / "confirmatory_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="reports/uci_har_v5_multirun_001",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    summary = summarize(root)
    verification = summary["verification"]
    print(
        f"UCI confirmatory summary: "
        f"{verification['checks_passed']}/{verification['checks_total']}"
    )
    if not verification["all_passed"]:
        failed = [
            name
            for name, item in verification["checks"].items()
            if not item["passed"]
        ]
        raise SystemExit(f"Failed summary checks: {failed}")


if __name__ == "__main__":
    main()
