"""Regenerate fail-closed Sepsis mapping evidence for the v2.0 audit.

This program is deliberately a mapping-evidence generator, not a privacy
certificate generator.  It binds a fixed local PhysioNet/CinC 2019 cache,
exports complete per-scenario mappings with the v2 ``generated_unit`` field,
and separates:

* observed raw-event incidence (kappa);
* a candidate stability bound under an explicitly fixed-slot replacement
  adjacency; and
* privacy wording, which remains unavailable without a bound mechanism and
  accountant report.

Label-balanced selection is retained only as a negative/control policy.  Its
selection is private-data-dependent and therefore cannot inherit the observed
event incidence as a validated stability bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mapping_evidence import inspect_mapping_evidence


SCHEMA_VERSION = "sepsis_mapping_evidence_v2.0"
DEFAULT_SPLITS = ("training_setA", "training_setB")
MAPPING_FIELDS = (
    "scenario",
    "window_id",
    "generated_unit",
    "owner_id",
    "start",
    "end",
    "split",
    "source_file",
    "label",
    "selection_classification",
)


@dataclass(frozen=True)
class Patient:
    split: str
    filename: str
    owner_id: str
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Policy:
    scenario: str
    window_length: int
    stride: int
    selection: str
    selection_classification: str
    cap: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_label(value: str, source: Path, row_number: int) -> int:
    try:
        label = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: row {row_number} has invalid SepsisLabel {value!r}"
        ) from exc
    if label not in {0, 1}:
        raise ValueError(
            f"{source}: row {row_number} has non-binary SepsisLabel {label}"
        )
    return label


def load_patients(
    cache_dir: Path,
    splits: Sequence[str],
    max_files_per_split: int | None,
) -> tuple[list[Patient], list[dict[str, object]]]:
    patients: list[Patient] = []
    manifest_rows: list[dict[str, object]] = []
    for split in splits:
        split_dir = cache_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing Sepsis cache split: {split_dir}")
        files = sorted(split_dir.glob("*.psv"), key=lambda path: path.name)
        if max_files_per_split is not None:
            if len(files) < max_files_per_split:
                raise ValueError(
                    f"{split_dir} contains {len(files)} PSV files; "
                    f"{max_files_per_split} were requested"
                )
            files = files[:max_files_per_split]
        if not files:
            raise ValueError(f"No PSV files selected from {split_dir}")

        for source in files:
            raw = source.read_bytes()
            text = raw.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text), delimiter="|")
            if reader.fieldnames is None or "SepsisLabel" not in reader.fieldnames:
                raise ValueError(f"{source} has no SepsisLabel column")
            labels = tuple(
                parse_label(row.get("SepsisLabel", ""), source, row_number)
                for row_number, row in enumerate(reader, start=2)
            )
            if not labels:
                raise ValueError(f"{source} contains no clinical rows")
            owner_id = f"{split}:{source.stem}"
            patients.append(
                Patient(
                    split=split,
                    filename=source.name,
                    owner_id=owner_id,
                    labels=labels,
                )
            )
            manifest_rows.append(
                {
                    "split": split,
                    "filename": source.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "row_count": len(labels),
                }
            )
    return patients, manifest_rows


def label_for_window(labels: Sequence[int], start: int, end: int, mode: str) -> str:
    selected = labels[start:end]
    if mode == "end":
        positive = selected[-1] == 1
    elif mode == "any":
        positive = any(value == 1 for value in selected)
    else:
        raise ValueError(f"Unsupported label mode: {mode}")
    return "sepsis" if positive else "control"


def base_windows(
    labels: Sequence[int],
    window_length: int,
    stride: int,
    label_mode: str,
) -> list[tuple[int, int, str]]:
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if len(labels) < window_length:
        return []
    return [
        (
            start,
            start + window_length,
            label_for_window(labels, start, start + window_length, label_mode),
        )
        for start in range(0, len(labels) - window_length + 1, stride)
    ]


def select_windows(
    windows: Sequence[tuple[int, int, str]],
    selection: str,
    cap: int | None,
) -> list[tuple[int, int, str]]:
    if selection == "all":
        return list(windows)
    if cap is None or cap <= 0:
        raise ValueError(f"Selection {selection} requires a positive cap")
    if selection == "owner_prefix_cap":
        return list(windows[:cap])
    if selection == "owner_label_balanced_cap":
        grouped: dict[str, list[tuple[int, int, str]]] = {}
        for window in windows:
            grouped.setdefault(window[2], []).append(window)
        positions = {label: 0 for label in grouped}
        selected: list[tuple[int, int, str]] = []
        while len(selected) < cap:
            progressed = False
            for label in sorted(grouped):
                position = positions[label]
                if position >= len(grouped[label]):
                    continue
                selected.append(grouped[label][position])
                positions[label] += 1
                progressed = True
                if len(selected) >= cap:
                    break
            if not progressed:
                break
        return selected
    raise ValueError(f"Unsupported selection: {selection}")


def exact_interval_kappa(intervals: Iterable[tuple[int, int]]) -> int:
    deltas: Counter[int] = Counter()
    for start, end in intervals:
        deltas[start] += 1
        deltas[end] -= 1
    current = 0
    maximum = 0
    for position in sorted(deltas):
        current += deltas[position]
        maximum = max(maximum, current)
    return maximum


def policies() -> list[Policy]:
    result: list[Policy] = []
    for window_length in (6, 12, 24):
        result.extend(
            [
                Policy(
                    scenario=f"sepsis_v2_{window_length}h_dense_stride1",
                    window_length=window_length,
                    stride=1,
                    selection="all",
                    selection_classification="fixed_positional_under_replace_one",
                ),
                Policy(
                    scenario=(
                        f"sepsis_v2_{window_length}h_half_stride"
                        f"{window_length // 2}"
                    ),
                    window_length=window_length,
                    stride=window_length // 2,
                    selection="all",
                    selection_classification="fixed_positional_under_replace_one",
                ),
                Policy(
                    scenario=f"sepsis_v2_{window_length}h_non_overlap",
                    window_length=window_length,
                    stride=window_length,
                    selection="all",
                    selection_classification="fixed_positional_under_replace_one",
                ),
            ]
        )
    for cap in (1, 4, 8, 16):
        result.append(
            Policy(
                scenario=f"sepsis_v2_6h_owner_prefix_cap{cap}",
                window_length=6,
                stride=6,
                selection="owner_prefix_cap",
                cap=cap,
                selection_classification="fixed_owner_local_prefix_under_replace_one",
            )
        )
    for cap in (4, 8, 16):
        result.append(
            Policy(
                scenario=f"sepsis_v2_6h_owner_label_balanced_cap{cap}",
                window_length=6,
                stride=6,
                selection="owner_label_balanced_cap",
                cap=cap,
                selection_classification="private_data_dependent",
            )
        )
    return result


def write_input_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "filename", "bytes", "sha256", "row_count"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_policy_mapping(
    path: Path,
    policy: Policy,
    patients: Sequence[Patient],
    label_mode: str,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    active_owners = 0
    owner_kappa = 0
    event_kappa = 0
    label_counts: Counter[str] = Counter()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        for patient in patients:
            windows = base_windows(
                patient.labels,
                policy.window_length,
                policy.stride,
                label_mode,
            )
            selected = select_windows(windows, policy.selection, policy.cap)
            if selected:
                active_owners += 1
                owner_kappa = max(owner_kappa, len(selected))
                event_kappa = max(
                    event_kappa,
                    exact_interval_kappa((start, end) for start, end, _ in selected),
                )
            for start, end, label in selected:
                window_id = (
                    f"{policy.scenario}:{patient.owner_id}:"
                    f"start{start}:end{end}"
                )
                writer.writerow(
                    {
                        "scenario": policy.scenario,
                        "window_id": window_id,
                        "generated_unit": "window",
                        "owner_id": patient.owner_id,
                        "start": start,
                        "end": end,
                        "split": patient.split,
                        "source_file": patient.filename,
                        "label": label,
                        "selection_classification": (
                            policy.selection_classification
                        ),
                    }
                )
                row_count += 1
                label_counts[label] += 1

    inspected = inspect_mapping_evidence(path, policy.scenario)
    if not inspected.valid:
        issues = "; ".join(issue.code for issue in inspected.issues)
        raise RuntimeError(f"Generated mapping failed v2 inspection: {issues}")
    if (
        inspected.selected_record_count != row_count
        or inspected.event_kappa != event_kappa
        or inspected.owner_kappa != owner_kappa
        or inspected.generated_unit != "window"
    ):
        raise RuntimeError(
            f"Independent mapping-module mismatch for {policy.scenario}"
        )

    fixed_event_selection = policy.selection_classification != "private_data_dependent"
    return {
        **asdict(policy),
        "mapping_file": f"mappings/{path.name}",
        "mapping_file_sha256": inspected.mapping_file_sha256,
        "selected_mapping_sha256": inspected.selected_mapping_sha256,
        "generated_unit": inspected.generated_unit,
        "num_windows": row_count,
        "owners_with_windows": active_owners,
        "observed_event_kappa": event_kappa,
        "observed_owner_kappa": owner_kappa,
        "label_counts": dict(sorted(label_counts.items())),
        "event_raw_adjacency": "replace_one_fixed_owner_hour_slot",
        "accountant_metric_for_candidate_bound": "generated_add_remove",
        "candidate_event_stability_k": (
            2 * event_kappa if fixed_event_selection else None
        ),
        "event_stability_status": (
            "candidate_fixed_slot_bound"
            if fixed_event_selection
            else "blocked_private_data_dependent_selection"
        ),
        "owner_raw_adjacency": "owner_add_remove",
        "candidate_owner_stability_k": owner_kappa,
        "owner_stability_status": "candidate_fixed_owner_partition_bound",
        "privacy_authority": "diagnostic_only",
        "privacy_certificate_status": "not_evaluated_no_bound_mechanism",
        "same_number_event_dp_reuse_allowed": False,
    }


def write_policy_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "scenario",
        "window_length",
        "stride",
        "selection",
        "cap",
        "selection_classification",
        "num_windows",
        "owners_with_windows",
        "observed_event_kappa",
        "observed_owner_kappa",
        "candidate_event_stability_k",
        "event_stability_status",
        "candidate_owner_stability_k",
        "privacy_authority",
        "privacy_certificate_status",
        "mapping_file_sha256",
        "selected_mapping_sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    evidence: dict[str, object],
    policy_rows: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# Sepsis 2019 v2 Mapping Evidence",
        "",
        "Status: **DIAGNOSTIC ONLY — NOT A DP CERTIFICATE**",
        "",
        (
            f"Bound cache: {evidence['patient_count']} patients, "
            f"{evidence['raw_event_count']} hourly rows"
        ),
        f"Input-manifest SHA-256: `{evidence['input_manifest_sha256']}`",
        f"Generator SHA-256: `{evidence['generator_sha256']}`",
        "",
        "Adjacency used for candidate event stability: replacement of values in",
        "one fixed patient/hour slot.  With a generated add/remove accountant,",
        "each affected window costs distance two.  Thus observed event kappa=1",
        "does **not** by itself authorize same-number event-level DP wording.",
        "",
        "Label-balanced policies are private-data-dependent negative controls.",
        "Their observed event incidence is not promoted to a stability bound.",
        "",
        "| policy | windows | owners | observed event κ | owner κ | candidate event K (add/remove) | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in policy_rows:
        candidate = row["candidate_event_stability_k"]
        lines.append(
            "| {scenario} | {num_windows} | {owners_with_windows} | "
            "{observed_event_kappa} | {observed_owner_kappa} | {candidate} | "
            "{event_stability_status} |".format(
                candidate="BLOCKED" if candidate is None else candidate,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "No row above contains an epsilon or delta, because no executed",
            "mechanism/accountant report is bound to this evidence bundle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/sepsis2019_cache")
    parser.add_argument(
        "--output-dir",
        default="reports/sepsis2019_v2_mapping_evidence_5k_001",
    )
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--max-files-per-split", type=int, default=2500)
    parser.add_argument("--label-mode", choices=("end", "any"), default="end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits:
        raise ValueError("At least one split is required")
    patients, manifest_rows = load_patients(
        cache_dir,
        splits,
        args.max_files_per_split,
    )
    input_manifest_path = output_dir / "input_manifest.csv"
    write_input_manifest(input_manifest_path, manifest_rows)

    policy_rows: list[dict[str, object]] = []
    for policy in policies():
        print(f"Generating {policy.scenario}")
        mapping_path = mappings_dir / f"{policy.scenario}.csv"
        policy_rows.append(
            write_policy_mapping(mapping_path, policy, patients, args.label_mode)
        )

    policy_summary_path = output_dir / "policy_summary.csv"
    write_policy_summary(policy_summary_path, policy_rows)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "PhysioNet_CinC_2019_Sepsis_training_v1.0.0",
        "source_type": "public_benchmark_local_cache",
        "splits": list(splits),
        "max_files_per_split": args.max_files_per_split,
        "selection_rule": "lexicographically first filenames per split",
        "patient_count": len(patients),
        "raw_event_count": sum(len(patient.labels) for patient in patients),
        "input_manifest_file": "input_manifest.csv",
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_semantic_sha256": stable_json_sha256(manifest_rows),
        "generator_file": "scripts/sepsis2019_v2_evidence.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "label_mode": args.label_mode,
        "record_identity_policy": "fixed_patient_hour_slots",
        "event_adjacency_scope": (
            "replace values/label within one fixed patient-hour slot; "
            "owner and slot identity do not change"
        ),
        "unsupported_event_adjacencies": [
            "row insertion",
            "row deletion",
            "patient reassignment",
        ],
        "owner_adjacency_scope": "add or remove one complete patient partition",
        "privacy_authority": "diagnostic_only",
        "privacy_certificate_status": "not_evaluated_no_bound_mechanism",
        "policies": policy_rows,
    }
    evidence_path = output_dir / "evidence.json"
    write_json(evidence_path, evidence)
    summary_path = output_dir / "summary.md"
    write_summary(summary_path, evidence, policy_rows)

    indexed_paths = [
        Path(__file__).resolve(),
        input_manifest_path,
        policy_summary_path,
        evidence_path,
        summary_path,
        *(mappings_dir / f"{policy.scenario}.csv" for policy in policies()),
    ]
    artifact_index = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "path": (
                    path.relative_to(project_root).as_posix()
                    if path.is_relative_to(project_root)
                    else str(path)
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in indexed_paths
        ],
    }
    write_json(output_dir / "artifact_index.json", artifact_index)
    print(f"Wrote {evidence_path}")
    print(f"Wrote {policy_summary_path}")


if __name__ == "__main__":
    main()
