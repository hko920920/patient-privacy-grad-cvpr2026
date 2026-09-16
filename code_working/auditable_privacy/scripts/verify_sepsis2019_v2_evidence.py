"""Independent verifier for the Sepsis v2 mapping-evidence bundle.

The default mode does not import the evidence generator: it rereads the 5,000
bound PSV files, reconstructs every policy, compares every exported row, and
recomputes all hashes and summaries. ``--skip-input-rescan`` is the portable
review mode. It still scans every bundled mapping row and recomputes all
mapping hashes, multiplicities, and summaries, but does not claim to reconstruct
those rows from the access-controlled raw challenge files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


EXPECTED_POLICIES: dict[str, tuple[int, int, str, int | None, str]] = {}
for _window in (6, 12, 24):
    EXPECTED_POLICIES[f"sepsis_v2_{_window}h_dense_stride1"] = (
        _window,
        1,
        "all",
        None,
        "fixed_positional_under_replace_one",
    )
    EXPECTED_POLICIES[
        f"sepsis_v2_{_window}h_half_stride{_window // 2}"
    ] = (
        _window,
        _window // 2,
        "all",
        None,
        "fixed_positional_under_replace_one",
    )
    EXPECTED_POLICIES[f"sepsis_v2_{_window}h_non_overlap"] = (
        _window,
        _window,
        "all",
        None,
        "fixed_positional_under_replace_one",
    )
for _cap in (1, 4, 8, 16):
    EXPECTED_POLICIES[f"sepsis_v2_6h_owner_prefix_cap{_cap}"] = (
        6,
        6,
        "owner_prefix_cap",
        _cap,
        "fixed_owner_local_prefix_under_replace_one",
    )
for _cap in (4, 8, 16):
    EXPECTED_POLICIES[f"sepsis_v2_6h_owner_label_balanced_cap{_cap}"] = (
        6,
        6,
        "owner_label_balanced_cap",
        _cap,
        "private_data_dependent",
    )


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


def canonical_mapping_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    canonical = [
        {
            "scenario": row["scenario"],
            "window_id": row["window_id"],
            "generated_unit": row["generated_unit"],
            "owner_ids": [row["owner_id"]],
            "start": int(row["start"]),
            "end": int(row["end"]),
        }
        for row in rows
    ]
    canonical.sort(
        key=lambda row: (
            row["scenario"],
            row["window_id"],
            row["generated_unit"],
            ";".join(row["owner_ids"]),
            row["start"],
            row["end"],
        )
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def forbidden_privacy_keys(value: object, prefix: str = "") -> list[str]:
    forbidden = {"epsilon", "delta", "supported_statement"}
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                found.append(path)
            found.extend(forbidden_privacy_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_privacy_keys(child, f"{prefix}[{index}]"))
    return found


def parse_psv_labels(path: Path) -> tuple[int, ...]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="|")
    if reader.fieldnames is None or "SepsisLabel" not in reader.fieldnames:
        raise ValueError(f"{path} is missing SepsisLabel")
    labels: list[int] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            label = int(float(row.get("SepsisLabel", "")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid label at row {row_number}") from exc
        if label not in {0, 1}:
            raise ValueError(f"{path}: non-binary label at row {row_number}")
        labels.append(label)
    return tuple(labels)


def window_label(labels: Sequence[int], start: int, end: int) -> str:
    return "sepsis" if labels[end - 1] == 1 else "control"


def base_windows(
    labels: Sequence[int],
    length: int,
    stride: int,
) -> list[tuple[int, int, str]]:
    if len(labels) < length:
        return []
    return [
        (start, start + length, window_label(labels, start, start + length))
        for start in range(0, len(labels) - length + 1, stride)
    ]


def selected_windows(
    windows: Sequence[tuple[int, int, str]],
    selection: str,
    cap: int | None,
) -> list[tuple[int, int, str]]:
    if selection == "all":
        return list(windows)
    assert cap is not None
    if selection == "owner_prefix_cap":
        return list(windows[:cap])
    if selection == "owner_label_balanced_cap":
        grouped: dict[str, list[tuple[int, int, str]]] = {}
        for window in windows:
            grouped.setdefault(window[2], []).append(window)
        positions = {label: 0 for label in grouped}
        result: list[tuple[int, int, str]] = []
        while len(result) < cap:
            progressed = False
            for label in sorted(grouped):
                position = positions[label]
                if position < len(grouped[label]):
                    result.append(grouped[label][position])
                    positions[label] += 1
                    progressed = True
                    if len(result) == cap:
                        break
            if not progressed:
                break
        return result
    raise ValueError(selection)


def interval_kappa(intervals: Iterable[tuple[int, int]]) -> int:
    deltas: Counter[int] = Counter()
    for start, end in intervals:
        deltas[start] += 1
        deltas[end] -= 1
    current = 0
    maximum = 0
    for point in sorted(deltas):
        current += deltas[point]
        maximum = max(maximum, current)
    return maximum


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def require(self, identifier: str, condition: bool, detail: str) -> None:
        self.rows.append(
            {"id": identifier, "passed": bool(condition), "detail": detail}
        )
        if not condition:
            raise RuntimeError(f"{identifier}: {detail}")


def verify_artifact_index(
    project_root: Path,
    bundle_dir: Path,
    checks: Checks,
) -> None:
    index = json.loads((bundle_dir / "artifact_index.json").read_text(encoding="utf-8"))
    files = index.get("files")
    checks.require(
        "artifact_index_nonempty",
        isinstance(files, list) and bool(files),
        "artifact index must contain files",
    )
    assert isinstance(files, list)
    for entry in files:
        path = project_root / entry["path"]
        checks.require(
            f"artifact_hash:{Path(entry['path']).name}",
            path.is_file()
            and path.stat().st_size == int(entry["bytes"])
            and sha256_file(path) == entry["sha256"],
            str(entry["path"]),
        )


def load_and_verify_inputs(
    cache_dir: Path,
    bundle_dir: Path,
    evidence: Mapping[str, object],
    checks: Checks,
    skip_input_rescan: bool,
) -> list[tuple[str, str, str, tuple[int, ...]]] | None:
    manifest_path = bundle_dir / "input_manifest.csv"
    checks.require(
        "input_manifest_file_hash",
        sha256_file(manifest_path) == evidence["input_manifest_sha256"],
        sha256_file(manifest_path),
    )
    raw_manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    typed_manifest = [
        {
            "split": row["split"],
            "filename": row["filename"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "row_count": int(row["row_count"]),
        }
        for row in raw_manifest
    ]
    checks.require(
        "input_manifest_semantic_hash",
        stable_json_sha256(typed_manifest)
        == evidence["input_manifest_semantic_sha256"],
        "typed manifest digest",
    )
    checks.require(
        "input_manifest_population",
        len(typed_manifest) == 5000
        and Counter(row["split"] for row in typed_manifest)
        == Counter({"training_setA": 2500, "training_setB": 2500}),
        f"rows={len(typed_manifest)}",
    )
    if skip_input_rescan:
        checks.require(
            "raw_input_rescan_skipped",
            True,
            "explicit portable mode; manifest and all bundled mappings are rescanned",
        )
        return None

    patients: list[tuple[str, str, str, tuple[int, ...]]] = []
    total_rows = 0
    for row in typed_manifest:
        source = cache_dir / str(row["split"]) / str(row["filename"])
        raw = source.read_bytes()
        labels = parse_psv_labels(source)
        if (
            len(raw) != row["bytes"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
            or len(labels) != row["row_count"]
        ):
            raise RuntimeError(f"Input drift: {source}")
        owner = f"{row['split']}:{Path(str(row['filename'])).stem}"
        patients.append(
            (str(row["split"]), str(row["filename"]), owner, labels)
        )
        total_rows += len(labels)
    checks.require(
        "all_raw_inputs_reopened",
        total_rows == int(evidence["raw_event_count"]),
        f"raw_event_count={total_rows}",
    )
    return patients


def expected_rows(
    scenario: str,
    config: tuple[int, int, str, int | None, str],
    patients: Sequence[tuple[str, str, str, tuple[int, ...]]],
) -> Iterator[dict[str, str]]:
    length, stride, selection, cap, classification = config
    for split, filename, owner, labels in patients:
        windows = selected_windows(
            base_windows(labels, length, stride),
            selection,
            cap,
        )
        for start, end, label in windows:
            yield {
                "scenario": scenario,
                "window_id": f"{scenario}:{owner}:start{start}:end{end}",
                "generated_unit": "window",
                "owner_id": owner,
                "start": str(start),
                "end": str(end),
                "split": split,
                "source_file": filename,
                "label": label,
                "selection_classification": classification,
            }


def verify_mapping(
    bundle_dir: Path,
    policy: Mapping[str, object],
    patients: Sequence[tuple[str, str, str, tuple[int, ...]]] | None,
    checks: Checks,
) -> None:
    scenario = str(policy["scenario"])
    config = EXPECTED_POLICIES[scenario]
    mapping_path = bundle_dir / str(policy["mapping_file"])
    checks.require(
        f"{scenario}:mapping_hash",
        sha256_file(mapping_path) == policy["mapping_file_sha256"],
        str(policy["mapping_file"]),
    )
    actual_rows: list[dict[str, str]] = []
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
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
        }
        checks.require(
            f"{scenario}:mapping_schema",
            reader.fieldnames is not None
            and required.issubset(set(reader.fieldnames)),
            str(reader.fieldnames),
        )
        if patients is None:
            for index, actual in enumerate(reader, start=1):
                try:
                    start = int(actual["start"])
                    end = int(actual["end"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{scenario}: invalid interval at mapping row {index}"
                    ) from exc
                if (
                    actual["scenario"] != scenario
                    or actual["generated_unit"] != "window"
                    or actual["selection_classification"] != config[4]
                    or actual["split"] not in {"training_setA", "training_setB"}
                    or actual["label"] not in {"control", "sepsis"}
                    or not actual["window_id"]
                    or not actual["owner_id"]
                    or not actual["source_file"]
                    or start < 0
                    or end <= start
                ):
                    raise RuntimeError(
                        f"{scenario}: malformed mapping row {index}"
                    )
                actual_rows.append(actual)
        else:
            for index, (actual, expected) in enumerate(
                zip(reader, expected_rows(scenario, config, patients), strict=True),
                start=1,
            ):
                if any(actual[key] != value for key, value in expected.items()):
                    raise RuntimeError(
                        f"{scenario}: mapping row {index} differs from raw reconstruction"
                    )
                actual_rows.append(actual)

    owner_counts: Counter[str] = Counter()
    owner_intervals: dict[str, list[tuple[int, int]]] = {}
    label_counts: Counter[str] = Counter()
    for row in actual_rows:
        owner_counts[row["owner_id"]] += 1
        owner_intervals.setdefault(row["owner_id"], []).append(
            (int(row["start"]), int(row["end"]))
        )
        label_counts[row["label"]] += 1
    event_kappa = max(
        (interval_kappa(intervals) for intervals in owner_intervals.values()),
        default=0,
    )
    owner_kappa = max(owner_counts.values(), default=0)
    fixed_selection = config[4] != "private_data_dependent"
    checks.require(
        f"{scenario}:mapping_reconstruction",
        len(actual_rows) == int(policy["num_windows"])
        and len(owner_counts) == int(policy["owners_with_windows"])
        and event_kappa == int(policy["observed_event_kappa"])
        and owner_kappa == int(policy["observed_owner_kappa"])
        and dict(sorted(label_counts.items())) == policy["label_counts"],
        f"rows={len(actual_rows)}, event_k={event_kappa}, owner_k={owner_kappa}",
    )
    checks.require(
        f"{scenario}:canonical_hash",
        canonical_mapping_sha256(actual_rows)
        == policy["selected_mapping_sha256"],
        str(policy["selected_mapping_sha256"]),
    )
    expected_event_bound = 2 * event_kappa if fixed_selection else None
    checks.require(
        f"{scenario}:stability_boundary",
        policy["candidate_event_stability_k"] == expected_event_bound
        and int(policy["candidate_owner_stability_k"]) == owner_kappa
        and policy["same_number_event_dp_reuse_allowed"] is False
        and policy["privacy_authority"] == "diagnostic_only"
        and policy["privacy_certificate_status"]
        == "not_evaluated_no_bound_mechanism",
        f"event_bound={expected_event_bound}",
    )


def verify_policy_summary(
    bundle_dir: Path,
    policies: Sequence[Mapping[str, object]],
    checks: Checks,
) -> None:
    rows = list(
        csv.DictReader((bundle_dir / "policy_summary.csv").open(encoding="utf-8"))
    )
    by_scenario = {row["scenario"]: row for row in rows}
    checks.require(
        "policy_summary_scenarios",
        set(by_scenario) == set(EXPECTED_POLICIES),
        f"scenarios={len(by_scenario)}",
    )
    for policy in policies:
        scenario = str(policy["scenario"])
        row = by_scenario[scenario]
        for key in (
            "num_windows",
            "owners_with_windows",
            "observed_event_kappa",
            "observed_owner_kappa",
            "event_stability_status",
            "privacy_authority",
            "privacy_certificate_status",
            "mapping_file_sha256",
            "selected_mapping_sha256",
        ):
            if row[key] != str(policy[key]):
                raise RuntimeError(f"{scenario}: policy summary drift at {key}")
        candidate = policy["candidate_event_stability_k"]
        if row["candidate_event_stability_k"] != (
            "" if candidate is None else str(candidate)
        ):
            raise RuntimeError(f"{scenario}: candidate-event-bound summary drift")
    checks.require(
        "policy_summary_matches_evidence",
        True,
        f"rows={len(rows)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default="reports/sepsis2019_v2_mapping_evidence_5k_001",
    )
    parser.add_argument("--cache-dir", default="data/sepsis2019_cache")
    parser.add_argument(
        "--skip-input-rescan",
        action="store_true",
        help=(
            "Verify the manifest and rescan every bundled mapping without "
            "reopening the raw challenge cache."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle_dir = Path(args.bundle_dir)
    cache_dir = Path(args.cache_dir)
    if not bundle_dir.is_absolute():
        bundle_dir = project_root / bundle_dir
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    output_path = bundle_dir / "independent_verification.json"
    checks = Checks()
    failure = ""
    try:
        evidence = json.loads(
            (bundle_dir / "evidence.json").read_text(encoding="utf-8")
        )
        checks.require(
            "schema_and_authority",
            evidence.get("schema_version") == "sepsis_mapping_evidence_v2.0"
            and evidence.get("privacy_authority") == "diagnostic_only"
            and evidence.get("privacy_certificate_status")
            == "not_evaluated_no_bound_mechanism",
            "v2 diagnostic-only contract",
        )
        checks.require(
            "no_dp_parameters_or_wording",
            not forbidden_privacy_keys(evidence),
            str(forbidden_privacy_keys(evidence)),
        )
        generator = project_root / str(evidence["generator_file"])
        checks.require(
            "generator_source_hash",
            sha256_file(generator) == evidence["generator_sha256"],
            str(evidence["generator_file"]),
        )
        verify_artifact_index(project_root, bundle_dir, checks)
        patients = load_and_verify_inputs(
            cache_dir,
            bundle_dir,
            evidence,
            checks,
            args.skip_input_rescan,
        )
        policies_value = evidence.get("policies")
        checks.require(
            "expected_policy_set",
            isinstance(policies_value, list)
            and {row["scenario"] for row in policies_value}
            == set(EXPECTED_POLICIES),
            f"expected={len(EXPECTED_POLICIES)}",
        )
        assert isinstance(policies_value, list)
        for policy in policies_value:
            scenario = str(policy["scenario"])
            config = EXPECTED_POLICIES[scenario]
            checks.require(
                f"{scenario}:declared_parameters",
                (
                    int(policy["window_length"]),
                    int(policy["stride"]),
                    str(policy["selection"]),
                    policy["cap"],
                    str(policy["selection_classification"]),
                )
                == config,
                str(config),
            )
            verify_mapping(bundle_dir, policy, patients, checks)
        verify_policy_summary(bundle_dir, policies_value, checks)
    except Exception as exc:  # preserve a machine-readable failure record
        failure = f"{type(exc).__name__}: {exc}"

    passed = not failure and all(bool(row["passed"]) for row in checks.rows)
    record = {
        "schema_version": "sepsis_mapping_evidence_independent_verification_v1",
        "verification_passed": passed,
        "checks_total": len(checks.rows),
        "checks_passed": sum(bool(row["passed"]) for row in checks.rows),
        "failure": failure,
        "raw_inputs_rescanned": not args.skip_input_rescan,
        "verifier_file": "scripts/verify_sepsis2019_v2_evidence.py",
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks.rows,
    }
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verification_passed={passed}")
    print(f"checks={record['checks_passed']}/{record['checks_total']}")
    if failure:
        print(f"failure={failure}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
