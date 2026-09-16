"""Independently cross-verify the hardened WISDM evidence directory.

This verifier intentionally does not import the production mapping parser,
claim validator, pipeline builder, or group-conversion helper. It rescans the
raw WISDM text, regenerates the expected mapping schedule, uses Google's
``dp-accounting`` for an accountant cross-check, and authenticates the
registered executable sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


SCENARIO = "wisdm_v2_hardened_overlap50"
ACTIVITIES = (
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking",
)
TRAIN_OWNERS = tuple(range(1, 26))
WINDOW_LENGTH = 200
WINDOW_STRIDE = 100
OWNER_CAP = 600
RAW_DOMAIN_ID = "fixed_owner_slot_payload_domain_v1"
RAW_EVENT_ADJACENCY = "fixed_owner_slot_payload_replace_one_v1"

RUNTIME_EXACT_FIELDS = (
    "mechanism_id",
    "mechanism_version",
    "accountant_id",
    "accountant_version",
    "accounting_unit",
    "accountant_adjacency",
    "sampling_unit",
    "sampler_law",
    "accountant_sampler_law",
    "sample_rate_numerator",
    "sample_rate_denominator",
    "sampling_implementation",
    "steps",
    "secure_mode",
    "secure_rng_backend",
    "noise_generation",
    "noise_hardening",
    "clipping_unit",
    "noising_unit",
    "privacy_convention",
    "gradient_aggregation",
    "update_normalization",
    "adapter_registry_entry_sha256",
    "code_artifact_id",
    "code_sha256",
    "accountant_checker_artifact_id",
    "accountant_checker_sha256",
    "selected_mapping_sha256",
    "pipeline_sha256",
    "delta_convention",
    "runtime_evidence_sha256",
)
RUNTIME_NUMERIC_FIELDS = (
    "sample_rate",
    "noise_multiplier",
    "clipping_norm",
    "optimizer_step_size",
    "epsilon",
    "delta",
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(
    checks: Dict[str, Dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def parse_owner_ids(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("owner_ids") or row.get("owners") or row.get("owner_id")
    values = [
        value.strip()
        for value in str(raw or "").replace("|", ";").split(";")
        if value.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("invalid owner attribution")
    return values


def canonical_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "scenario": str(record["scenario"]),
            "window_id": str(record["window_id"]),
            "generated_unit": str(record["generated_unit"]),
            "owner_ids": sorted(str(owner) for owner in record["owner_ids"]),
            "start": int(record["start"]),
            "end": int(record["end"]),
        }
        for record in records
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
    return canonical_json_sha256(canonical)


def scan_mapping(path: Path, scenario: str) -> Dict[str, Any]:
    errors: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {
            "scenario",
            "window_id",
            "generated_unit",
            "start",
            "end",
            "owner_id",
            "owner_ids",
            "activity",
            "row_index",
        }
        if len(fieldnames) != len(set(fieldnames)):
            errors.append("duplicate columns")
        missing = sorted(required - set(fieldnames))
        if missing:
            errors.append(f"missing columns: {missing}")
        selected_rows = [
            row for row in reader if str(row.get("scenario") or "") == scenario
        ]

    records: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    generated_units: set[str] = set()
    for index, row in enumerate(selected_rows):
        try:
            window_id = str(row["window_id"]).strip()
            if not window_id or window_id in seen_ids:
                raise ValueError("missing or duplicate window_id")
            seen_ids.add(window_id)
            start = int(row["start"])
            end = int(row["end"])
            if end <= start:
                raise ValueError("empty or reversed interval")
            owners = parse_owner_ids(row)
            generated_unit = str(row["generated_unit"])
            if generated_unit not in {"window", "event", "owner"}:
                raise ValueError("invalid generated unit")
            generated_units.add(generated_unit)
            records.append(
                {
                    "scenario": scenario,
                    "window_id": window_id,
                    "generated_unit": generated_unit,
                    "owner_ids": owners,
                    "owner_id": str(row["owner_id"]),
                    "start": start,
                    "end": end,
                    "split": str(row.get("split") or ""),
                    "activity": str(row.get("activity") or ""),
                    "row_index": int(row["row_index"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")

    event_deltas: Dict[str, Counter[int]] = defaultdict(Counter)
    owner_counts: Counter[str] = Counter()
    attribution_arity = 0
    for record in records:
        attribution_arity = max(attribution_arity, len(record["owner_ids"]))
        for owner in record["owner_ids"]:
            owner_counts[owner] += 1
            event_deltas[owner][record["start"]] += 1
            event_deltas[owner][record["end"]] -= 1
    event_kappa = 0
    for deltas in event_deltas.values():
        active = 0
        for position in sorted(deltas):
            active += deltas[position]
            event_kappa = max(event_kappa, active)

    if len(generated_units) != 1:
        errors.append(f"generated units: {sorted(generated_units)}")
    return {
        "errors": errors,
        "records": records,
        "mapping_file_sha256": file_sha256(path),
        "selected_mapping_sha256": canonical_records_sha256(records),
        "selected_record_count": len(selected_rows),
        "event_kappa": event_kappa,
        "owner_kappa": max(owner_counts.values(), default=0),
        "attribution_arity_max": attribution_arity,
    }


def scan_raw_wisdm(path: Path) -> Dict[str, Any]:
    activity_index = {name: index for index, name in enumerate(ACTIVITIES)}
    activities: Dict[int, bytearray] = defaultdict(bytearray)
    counts = {
        "physical_lines": 0,
        "accepted_canonical_events": 0,
        "invalid_column_count": 0,
        "invalid_parse": 0,
        "invalid_activity": 0,
        "nonfinite_payload": 0,
    }
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            counts["physical_lines"] += 1
            parts = [
                part.strip()
                for part in line.strip().rstrip(";").split(",")
            ]
            if len(parts) != 6:
                counts["invalid_column_count"] += 1
                continue
            try:
                owner = int(parts[0])
                activity = parts[1]
                int(parts[2])
                xyz = tuple(float(value) for value in parts[3:6])
            except ValueError:
                counts["invalid_parse"] += 1
                continue
            if activity not in activity_index:
                counts["invalid_activity"] += 1
                continue
            if not all(math.isfinite(value) for value in xyz):
                counts["nonfinite_payload"] += 1
                continue
            activities[owner].append(activity_index[activity])
            counts["accepted_canonical_events"] += 1
    return {
        "dataset_file_sha256": file_sha256(path),
        "counts": counts,
        "activities": activities,
    }


def expected_train_mapping(
    activities: Mapping[int, Sequence[int]],
) -> list[Dict[str, Any]]:
    expected: list[Dict[str, Any]] = []
    for owner in TRAIN_OWNERS:
        owner_activities = activities.get(owner, ())
        offsets = list(
            range(
                0,
                len(owner_activities) - WINDOW_LENGTH + 1,
                WINDOW_STRIDE,
            )
        )[:OWNER_CAP]
        for ordinal, offset in enumerate(offsets):
            counts = Counter(
                owner_activities[offset : offset + WINDOW_LENGTH]
            )
            activity = min(
                ACTIVITIES,
                key=lambda candidate: (
                    -counts[ACTIVITIES.index(candidate)],
                    candidate,
                ),
            )
            expected.append(
                {
                    "scenario": SCENARIO,
                    "window_id": (
                        f"train_{SCENARIO}_u{owner}_w{ordinal}"
                    ),
                    "generated_unit": "window",
                    "owner_ids": [str(owner)],
                    "owner_id": str(owner),
                    "start": offset,
                    "end": offset + WINDOW_LENGTH,
                    "split": "train",
                    "activity": activity,
                    "row_index": len(expected),
                }
            )
    return expected


def group_conversion(
    epsilon: float,
    delta: float,
    k: int,
) -> tuple[float, float, bool]:
    converted_epsilon = k * epsilon
    log_delta = math.log(delta)
    if epsilon == 0.0:
        log_delta += math.log(k)
    else:
        log_delta += (
            math.log(math.expm1(k * epsilon))
            - math.log(math.expm1(epsilon))
        )
    vacuous = log_delta >= 0.0
    return (
        converted_epsilon,
        1.0 if vacuous else math.exp(log_delta),
        vacuous,
    )


def exact_supported_statement(record: Mapping[str, Any]) -> str:
    """Reconstruct the only licensed positive sentence from public fields."""

    return (
        "Conditional on a complete and faithful mapping export for the bound "
        "pipeline, the released mechanism is "
        f"({float(record['epsilon']):.6g}, {float(record['delta']):.6g})-DP "
        f"for one {record['claimed_privacy_unit']} under "
        f"{record['raw_adjacency']} adjacency."
    )


def independent_accountant_epsilon(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    delta: float,
) -> float:
    from dp_accounting import dp_event, rdp

    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sample_rate,
            dp_event.GaussianDpEvent(noise_multiplier),
        ),
        steps,
    )
    return float(accountant.get_epsilon(delta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        default="reports/wisdm_v2_hardened_secure_002",
    )
    parser.add_argument(
        "--data-path",
        default="data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
    )
    parser.add_argument(
        "--skip-input-rescan",
        action="store_true",
        help=(
            "Verify the bundled mapping, parser-evidence digest, execution, "
            "accounting, and certificates without reopening the public raw file."
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = project_root / evidence_dir
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    output_path = (
        Path(args.output_json)
        if args.output_json
        else evidence_dir / "independent_verification.json"
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path

    report = json.loads(
        (evidence_dir / "privacy_report.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            project_root / "docs" / "privacy_report_schema_v2_0.json"
        ).read_text(encoding="utf-8")
    )
    checks: Dict[str, Dict[str, Any]] = {}
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    check(
        checks,
        "privacy_report_schema",
        not schema_errors,
        [error.message for error in schema_errors],
    )
    check(
        checks,
        "scenario",
        report.get("scenario") == SCENARIO,
        report.get("scenario"),
    )

    pipeline = report["pipeline"]
    manifest = pipeline["manifest"]
    mechanism = report["mechanism"]
    privacy = report["privacy"]
    mapping = scan_mapping(evidence_dir / "mapping.csv", SCENARIO)
    check(checks, "mapping_structure", not mapping["errors"], mapping["errors"])
    check(
        checks,
        "mapping_file_digest",
        mapping["mapping_file_sha256"]
        == pipeline["mapping_file_sha256"]
        == manifest["mapping_file_sha256"],
        mapping["mapping_file_sha256"],
    )
    check(
        checks,
        "selected_mapping_digest",
        mapping["selected_mapping_sha256"]
        == pipeline["selected_mapping_sha256"]
        == manifest["selected_mapping_sha256"],
        mapping["selected_mapping_sha256"],
    )
    check(
        checks,
        "selected_record_count",
        mapping["selected_record_count"] == manifest["selected_record_count"],
        mapping["selected_record_count"],
    )
    check(
        checks,
        "mapping_multiplicity",
        mapping["event_kappa"] == 2
        and mapping["owner_kappa"] == 565
        and mapping["attribution_arity_max"] == 1,
        {
            "event_kappa": mapping["event_kappa"],
            "owner_kappa": mapping["owner_kappa"],
            "attribution_arity_max": mapping["attribution_arity_max"],
        },
    )

    parser_evidence_path = evidence_dir / "parser_evidence.json"
    parser_evidence = json.loads(
        parser_evidence_path.read_text(encoding="utf-8")
    )
    if not args.skip_input_rescan:
        raw_scan = scan_raw_wisdm(data_path)
        check(
            checks,
            "raw_dataset_digest",
            raw_scan["dataset_file_sha256"]
            == manifest["dataset_file_sha256"]
            == parser_evidence["dataset_file_sha256"],
            raw_scan["dataset_file_sha256"],
        )
        check(
            checks,
            "independent_parser_counts",
            raw_scan["counts"]
            == {
                key: parser_evidence[key] for key in raw_scan["counts"]
            },
            raw_scan["counts"],
        )
        expected_mapping = expected_train_mapping(raw_scan["activities"])
        check(
            checks,
            "raw_to_mapping_regeneration",
            expected_mapping == mapping["records"],
            {
                "expected_rows": len(expected_mapping),
                "actual_rows": len(mapping["records"]),
            },
        )

    check(
        checks,
        "mechanism_digest",
        canonical_json_sha256(mechanism) == manifest["mechanism_sha256"],
        manifest["mechanism_sha256"],
    )
    check(
        checks,
        "pipeline_digest",
        canonical_json_sha256(manifest) == pipeline["pipeline_sha256"],
        pipeline["pipeline_sha256"],
    )
    for index, step in enumerate(manifest["preprocessing"]):
        check(
            checks,
            f"preprocessing_parameters_{index}",
            canonical_json_sha256(step["parameters"])
            == step["parameters_sha256"],
            step["id"],
        )
    for name in ("raw_domain", "contribution_policy"):
        block = manifest[name]
        check(
            checks,
            f"{name}_parameters",
            canonical_json_sha256(block["parameters"])
            == block["parameters_sha256"],
            block["id"],
        )
    schedule = manifest["schedule"]
    check(
        checks,
        "schedule_parameters",
        canonical_json_sha256(schedule["parameters"])
        == schedule["parameters_sha256"],
        schedule["id"],
    )

    registry_document = json.loads(
        (
            project_root / "docs" / "mechanism_registry_v2_0.json"
        ).read_text(encoding="utf-8")
    )
    registry_key = (
        f"{mechanism['mechanism_id']}@{mechanism['mechanism_version']}"
    )
    registry_entry = registry_document["mechanisms"].get(registry_key)
    check(
        checks,
        "mechanism_registry_entry",
        isinstance(registry_entry, Mapping)
        and canonical_json_sha256(registry_entry)
        == mechanism["adapter_registry_entry_sha256"],
        registry_key,
    )
    if not isinstance(registry_entry, Mapping):
        registry_entry = {}
    semantic_fields = (
        "mechanism_id",
        "mechanism_version",
        "accountant_id",
        "sampler_law",
        "accountant_sampler_law",
        "sampling_implementation",
        "secure_rng_backend",
        "noise_generation",
        "noise_hardening",
        "gradient_aggregation",
        "update_normalization",
    )
    check(
        checks,
        "registered_mechanism_semantics",
        all(
            str(mechanism.get(key)) == str(registry_entry.get(key))
            for key in semantic_fields
        ),
        {key: mechanism.get(key) for key in semantic_fields},
    )
    for prefix, id_key, hash_key in (
        ("pipeline_code", "code_artifact_id", "code_sha256"),
        (
            "accountant_checker",
            "accountant_checker_artifact_id",
            "accountant_checker_sha256",
        ),
    ):
        artifact_id = str(registry_entry.get(id_key) or "")
        artifact_path = (project_root / artifact_id).resolve()
        in_root = False
        try:
            artifact_path.relative_to(project_root.resolve())
            in_root = True
        except ValueError:
            pass
        actual_hash = file_sha256(artifact_path) if in_root else ""
        check(
            checks,
            f"{prefix}_source_digest",
            in_root
            and actual_hash
            == registry_entry.get(hash_key)
            == manifest.get(hash_key)
            and artifact_id == manifest.get(id_key),
            actual_hash,
        )

    check(
        checks,
        "exact_sampling_contract",
        mechanism["sample_rate_numerator"] == 1
        and mechanism["sample_rate_denominator"] == 50
        and mechanism["sample_rate"] == 1 / 50
        and mechanism["sampling_implementation"]
        == "systemrandom_randrange_bernoulli_v1",
        {
            "numerator": mechanism["sample_rate_numerator"],
            "denominator": mechanism["sample_rate_denominator"],
            "sample_rate": mechanism["sample_rate"],
        },
    )
    check(
        checks,
        "public_constant_update_contract",
        mechanism["update_normalization"]
        == "public_constant_step_no_dataset_denominator"
        and mechanism["optimizer_step_size"] == 1 / 150,
        {
            "update_normalization": mechanism["update_normalization"],
            "optimizer_step_size": mechanism["optimizer_step_size"],
        },
    )
    check(
        checks,
        "four_draw_noise_contract",
        mechanism["noise_generation"]
        == "normalvariate_discard1_sum4_div2_v1"
        and mechanism["noise_hardening"]
        == "known_fp_reconstruction_mitigation_four_draw_v1"
        and report["rng_assurance"] == "known_attack_hardened_secure",
        {
            "noise_generation": mechanism["noise_generation"],
            "noise_hardening": mechanism["noise_hardening"],
            "rng_assurance": report["rng_assurance"],
        },
    )
    raw_domain = manifest["raw_domain"]
    event_contract = report["claim_contracts"]["event"]
    check(
        checks,
        "event_raw_domain_contract",
        raw_domain["id"] == RAW_DOMAIN_ID
        and raw_domain["parameters"]["fixed_presence"] is True
        and {"owner_id", "event_slot"}.issubset(
            raw_domain["parameters"]["immutable_fields"]
        )
        and event_contract["raw_adjacency"] == RAW_EVENT_ADJACENCY
        and event_contract["raw_domain_id"] == RAW_DOMAIN_ID
        and event_contract["raw_domain_sha256"]
        == canonical_json_sha256(raw_domain),
        event_contract,
    )
    contribution = manifest["contribution_policy"]
    owner_contract = report["claim_contracts"]["owner"]
    check(
        checks,
        "public_owner_cap_contract",
        contribution["id"] == "public_owner_generated_record_cap_v1"
        and contribution["classification"] == "public_fixed"
        and contribution["parameters"]["cap"] == OWNER_CAP
        and contribution["parameters"]["enforcement"]
        == "deterministic_owner_window_prefix_v1"
        and owner_contract["stability_bound"] == OWNER_CAP
        and mapping["owner_kappa"] <= OWNER_CAP,
        {
            "public_cap": contribution["parameters"]["cap"],
            "observed_max": mapping["owner_kappa"],
        },
    )

    runtime = report["runtime_trace"]
    runtime_payload = {
        key: runtime.get(key)
        for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
    }
    check(
        checks,
        "runtime_evidence_digest",
        canonical_json_sha256(runtime["evidence"])
        == runtime["runtime_evidence_sha256"],
        runtime["runtime_evidence_sha256"],
    )
    check(
        checks,
        "runtime_trace_digest",
        canonical_json_sha256(runtime_payload) == runtime["trace_hash"],
        runtime["trace_hash"],
    )
    check(
        checks,
        "runtime_contract_fields",
        all(
            runtime.get(key)
            == (
                manifest.get(key)
                if key
                in {
                    "code_artifact_id",
                    "code_sha256",
                    "accountant_checker_artifact_id",
                    "accountant_checker_sha256",
                }
                else pipeline["selected_mapping_sha256"]
                if key == "selected_mapping_sha256"
                else pipeline["pipeline_sha256"]
                if key == "pipeline_sha256"
                else privacy.get(key)
                if key in {"epsilon", "delta", "delta_convention"}
                else mechanism.get(key)
            )
            for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
            if key != "runtime_evidence_sha256"
        ),
        "all bound runtime fields match",
    )

    artifact_checks = (
        (
            "released_model",
            evidence_dir / "released_model.json",
            "released_model_sha256",
        ),
        (
            "batch_trace",
            evidence_dir / "batch_trace.csv",
            "batch_trace_sha256",
        ),
        (
            "parser_evidence",
            parser_evidence_path,
            "parser_evidence_sha256",
        ),
    )
    for name, path, hash_key in artifact_checks:
        actual_hash = file_sha256(path)
        check(
            checks,
            f"{name}_digest",
            actual_hash == manifest[hash_key]
            and (
                hash_key not in runtime["evidence"]
                or actual_hash == runtime["evidence"][hash_key]
            ),
            actual_hash,
        )
    with (evidence_dir / "batch_trace.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        batch_rows = list(csv.DictReader(handle))
    check(
        checks,
        "batch_trace_steps",
        len(batch_rows) == mechanism["steps"]
        and [int(row["step"]) for row in batch_rows]
        == list(range(1, mechanism["steps"] + 1))
        and runtime["evidence"]["executed_updates"] == mechanism["steps"],
        len(batch_rows),
    )

    epsilon = independent_accountant_epsilon(
        mechanism["sample_rate"],
        mechanism["noise_multiplier"],
        mechanism["steps"],
        privacy["delta"],
    )
    check(
        checks,
        "independent_dp_accounting_epsilon",
        math.isclose(
            epsilon,
            privacy["epsilon"],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ),
        {
            "dp_accounting_version": importlib.metadata.version(
                "dp-accounting"
            ),
            "epsilon": epsilon,
        },
    )

    certificate_dir = evidence_dir / "certificates"
    public: Dict[str, Dict[str, Any]] = {}
    internal: Dict[str, Dict[str, Any]] = {}
    for claim in ("window", "event", "owner"):
        stem = f"certificate_{SCENARIO}_{claim}"
        public[claim] = json.loads(
            (certificate_dir / f"{stem}.json").read_text(encoding="utf-8")
        )
        internal[claim] = json.loads(
            (certificate_dir / f"{stem}_internal.json").read_text(
                encoding="utf-8"
            )
        )
    expected_window_statement = exact_supported_statement(public["window"])
    check(
        checks,
        "window_certificate",
        public["window"]["release_status"] == "ALLOWED"
        and public["window"]["unit_path"] == "DIRECT"
        and public["window"]["claimed_privacy_unit"] == "window"
        and public["window"]["raw_adjacency"] == "add_remove"
        and public["window"]["supported_statement"]
        == expected_window_statement
        and internal["window"]["supported_statement"]
        == expected_window_statement,
        {
            "release_status": public["window"]["release_status"],
            "statement": public["window"]["supported_statement"],
        },
    )
    event_conversion = group_conversion(
        epsilon,
        privacy["delta"],
        int(internal["event"]["stability_bound"]),
    )
    expected_event_statement = exact_supported_statement(public["event"])
    check(
        checks,
        "event_certificate",
        public["event"]["release_status"] == "ALLOWED"
        and public["event"]["unit_path"] == "CONVERT"
        and public["event"]["claimed_privacy_unit"] == "event"
        and public["event"]["raw_adjacency"]
        == RAW_EVENT_ADJACENCY
        and int(internal["event"]["stability_bound"]) == 4
        and math.isclose(
            public["event"]["epsilon"],
            event_conversion[0],
            rel_tol=1e-10,
        )
        and math.isclose(
            public["event"]["delta"],
            event_conversion[1],
            rel_tol=1e-10,
        )
        and not event_conversion[2]
        and public["event"]["supported_statement"]
        == expected_event_statement
        and internal["event"]["supported_statement"]
        == expected_event_statement,
        {
            "release_status": public["event"]["release_status"],
            "K": internal["event"]["stability_bound"],
            "statement": public["event"]["supported_statement"],
        },
    )
    owner_conversion = group_conversion(
        epsilon,
        privacy["delta"],
        OWNER_CAP,
    )
    check(
        checks,
        "owner_certificate",
        public["owner"]["release_status"] == "BLOCKED_VACUOUS"
        and public["owner"]["unit_path"] == "GROUP"
        and public["owner"]["claimed_privacy_unit"] == "owner"
        and public["owner"]["raw_adjacency"] == "owner_add_remove"
        and int(internal["owner"]["stability_bound"]) == OWNER_CAP
        and owner_conversion[2]
        and not public["owner"]["supported_statement"]
        and not internal["owner"]["supported_statement"],
        {
            "release_status": public["owner"]["release_status"],
            "K": internal["owner"]["stability_bound"],
        },
    )

    passed = all(item["passed"] for item in checks.values())
    record = {
        "verification_type": (
            "independent_cross_accountant_portable"
            if args.skip_input_rescan
            else "independent_raw_rescan_and_cross_accountant"
        ),
        "passed": passed,
        "check_count": len(checks),
        "raw_inputs_rescanned": not args.skip_input_rescan,
        "evidence_dir": evidence_dir.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verification_passed={passed}")
    print(f"checks={len(checks)}")
    print(f"Wrote {output_path}")
    if not passed:
        failed = [
            name for name, item in checks.items() if not item["passed"]
        ]
        print(f"failed_checks={failed}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
