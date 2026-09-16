import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mapping_evidence import canonical_records_sha256, file_sha256, read_mapping


DEFAULT_CLAIM_UNITS = ["window", "event", "user"]


def build_window_records(num_sequences, sequence_length, window_length, stride):
    records = []
    for seq_id in range(num_sequences):
        for start in range(0, sequence_length - window_length + 1, stride):
            end = start + window_length
            records.append(
                {
                    "window_id": f"s{stride}_u{seq_id}_t{start}",
                    "generated_unit": "window",
                    "owner_id": str(seq_id),
                    "owner_ids": [str(seq_id)],
                    "start": start,
                    "end": end,
                    "scenario": f"stride_{stride}",
                    "stride": stride,
                }
            )
    return records


def subsample_records(records, max_windows, rng):
    if max_windows is None or len(records) <= max_windows:
        return list(records)

    idx = rng.choice(len(records), size=max_windows, replace=False)
    idx.sort()
    return [records[i] for i in idx]


def group_by_scenario(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario"]].append(record)
    return dict(grouped)


def event_multiplicity(records):
    counts = Counter()
    for record in records:
        for owner_id in record.get("owner_ids", [record["owner_id"]]):
            for event_idx in range(record["start"], record["end"]):
                counts[(owner_id, event_idx)] += 1

    values = np.array(list(counts.values()), dtype=np.int32)
    if values.size == 0:
        return {
            "kappa_max": 0,
            "kappa_mean": 0.0,
            "kappa_p95": 0.0,
            "kappa_p99": 0.0,
            "covered_events": 0,
        }

    return {
        "kappa_max": int(values.max()),
        "kappa_mean": float(values.mean()),
        "kappa_p95": float(np.percentile(values, 95)),
        "kappa_p99": float(np.percentile(values, 99)),
        "covered_events": int(values.size),
    }


def owner_multiplicity(records):
    counts = Counter()
    for record in records:
        for owner_id in record.get("owner_ids", [record["owner_id"]]):
            counts[owner_id] += 1
    values = np.array(list(counts.values()), dtype=np.int32)
    if values.size == 0:
        return {
            "owner_windows_max": 0,
            "owner_windows_mean": 0.0,
            "owner_windows_p95": 0.0,
            "owner_windows_p99": 0.0,
            "covered_owners": 0,
        }

    return {
        "owner_windows_max": int(values.max()),
        "owner_windows_mean": float(values.mean()),
        "owner_windows_p95": float(np.percentile(values, 95)),
        "owner_windows_p99": float(np.percentile(values, 99)),
        "covered_owners": int(values.size),
    }


def infer_window_length(records):
    if not records:
        return 0
    lengths = [record["end"] - record["start"] for record in records]
    if len(set(lengths)) == 1:
        return lengths[0]
    return int(np.percentile(lengths, 50))


def infer_sequence_length(records):
    if not records:
        return 0
    return max(record["end"] for record in records)


def attribution_stats(records):
    if not records:
        return {
            "attribution_arity_max": 0,
            "attribution_arity_mean": 0.0,
            "multi_owner_windows": 0,
        }
    arities = np.array(
        [len(record.get("owner_ids", [record["owner_id"]])) for record in records],
        dtype=np.int32,
    )
    return {
        "attribution_arity_max": int(arities.max()),
        "attribution_arity_mean": float(arities.mean()),
        "multi_owner_windows": int(np.sum(arities > 1)),
    }


def classify_observed_alignment(claim_unit, event_stats, owner_stats):
    """Classify observed incidence without authorizing a privacy statement.

    The v2.0 certificate validator separately checks transformation stability,
    adjacency, preprocessing, sampler/accountant compatibility, runtime
    binding, and release conditions. Nothing returned here is a DP verdict.
    """
    event_kappa = max(1, event_stats["kappa_max"])
    owner_kappa = max(1, owner_stats["owner_windows_max"])

    if claim_unit == "window":
        return {
            "verdict": "diagnostic_only",
            "observed_alignment": "generated_unit_identity_candidate",
            "conversion_multiplier": 1,
            "recommended_statement": "Observed mapping diagnostic only; run the v2.0 validator before making a window-level DP statement.",
            "recommended_action": "Supply a complete v2.0 mechanism and claim contract.",
        }

    if claim_unit == "event":
        if event_kappa == 1:
            return {
                "verdict": "diagnostic_only",
                "observed_alignment": "observed_event_kappa_one",
                "conversion_multiplier": 1,
                "recommended_statement": "Observed event incidence is one, but DIRECT requires a validated stability bound in the accountant metric.",
                "recommended_action": "Validate raw adjacency, record identity, preprocessing support, and accountant adjacency.",
            }

        return {
            "verdict": "diagnostic_only",
            "observed_alignment": "observed_event_kappa_gt_one",
            "conversion_multiplier": event_kappa,
            "recommended_statement": (
                f"Observed event incidence is {event_kappa}; this is not yet a "
                "validated accountant-metric stability bound."
            ),
            "recommended_action": (
                "Run the v2.0 stability checker and then apply only the conversion "
                "authorized by the resulting K."
            ),
        }

    if claim_unit in {"user", "series", "patient", "owner"}:
        if owner_kappa == 1:
            return {
                "verdict": "diagnostic_only",
                "observed_alignment": "observed_owner_kappa_one",
                "conversion_multiplier": 1,
                "recommended_statement": "Observed owner incidence is one, but DIRECT requires a validated owner-to-accountant stability proof.",
                "recommended_action": "Validate owner adjacency, attribution, cross-owner dependencies, and accountant adjacency.",
            }

        return {
            "verdict": "diagnostic_only",
            "observed_alignment": "observed_owner_kappa_gt_one",
            "conversion_multiplier": owner_kappa,
            "recommended_statement": (
                f"Observed owner incidence is {owner_kappa}; this is not yet a "
                "validated accountant-metric stability bound."
            ),
            "recommended_action": (
                "Run the v2.0 owner stability checker before selecting a group "
                "conversion or native owner path."
            ),
        }

    raise ValueError(f"Unsupported claim unit: {claim_unit}")


def audit_scenario(
    scenario,
    selected_records,
    full_records,
    batch_size,
    epochs,
    claim_units,
    schedule_mode="fixed_mapping",
    event_kappa_bound=None,
    owner_kappa_bound=None,
    mapping_source="",
    mapping_file_sha256="",
    provenance_note="",
):
    observed_event = event_multiplicity(selected_records)
    observed_owner = owner_multiplicity(selected_records)
    observed_attribution = attribution_stats(selected_records)
    selected_event = dict(observed_event)
    selected_owner = dict(observed_owner)
    full_event = event_multiplicity(full_records)
    full_owner = owner_multiplicity(full_records)

    kappa_source = "observed_mapping_only"
    formal_bound_status = "not_validated"
    if schedule_mode == "union":
        kappa_source = "unverified_union_label_over_observed_mapping"
    elif schedule_mode == "stochastic_bound":
        if event_kappa_bound is None or owner_kappa_bound is None:
            raise ValueError(
                "schedule_mode=stochastic_bound requires --event-kappa-bound and --owner-kappa-bound"
            )
        kappa_source = "user_supplied_unverified_stochastic_bound"
    elif schedule_mode == "adaptive_private":
        kappa_source = "observed_private_adaptive_branch_only"

    steps_per_epoch = math.ceil(len(selected_records) / batch_size)
    total_steps = steps_per_epoch * epochs
    q_window = min(1.0, batch_size / max(1, len(selected_records)))
    q_event_upper = min(1.0, q_window * max(1, selected_event["kappa_max"]))
    q_owner_upper = min(1.0, q_window * max(1, selected_owner["owner_windows_max"]))

    stride_values = sorted({str(r.get("stride", "")) for r in selected_records})
    stride = stride_values[0] if len(stride_values) == 1 else ""

    base = {
        "scenario": scenario,
        "stride": stride,
        "num_windows_full": len(full_records),
        "num_windows_used": len(selected_records),
        "window_length": infer_window_length(selected_records),
        "sequence_length_proxy": infer_sequence_length(selected_records),
        "num_owners": selected_owner["covered_owners"],
        "attribution_arity_max": observed_attribution["attribution_arity_max"],
        "attribution_arity_mean": observed_attribution["attribution_arity_mean"],
        "multi_owner_windows": observed_attribution["multi_owner_windows"],
        "mapping_source": mapping_source,
        "mapping_file_sha256": mapping_file_sha256,
        "selected_mapping_sha256": canonical_records_sha256(selected_records),
        "provenance_note": provenance_note,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "q_window": q_window,
        "q_event_upper_proxy": q_event_upper,
        "q_owner_upper_proxy": q_owner_upper,
        "schedule_mode": schedule_mode,
        "kappa_source": kappa_source,
        "formal_bound_status": formal_bound_status,
        "event_kappa_full": full_event["kappa_max"],
        "event_kappa_observed": observed_event["kappa_max"],
        "event_kappa_bound": event_kappa_bound if event_kappa_bound is not None else "",
        "event_kappa_used": selected_event["kappa_max"],
        "event_kappa_mean_used": selected_event["kappa_mean"],
        "event_kappa_p95_used": selected_event["kappa_p95"],
        "event_kappa_p99_used": selected_event["kappa_p99"],
        "owner_windows_max_full": full_owner["owner_windows_max"],
        "owner_windows_max_observed": observed_owner["owner_windows_max"],
        "owner_windows_bound": owner_kappa_bound if owner_kappa_bound is not None else "",
        "owner_windows_max_used": selected_owner["owner_windows_max"],
        "owner_windows_mean_used": selected_owner["owner_windows_mean"],
        "owner_windows_p95_used": selected_owner["owner_windows_p95"],
        "owner_windows_p99_used": selected_owner["owner_windows_p99"],
        "event_reuse_max_over_epochs": selected_event["kappa_max"] * epochs,
        "owner_reuse_max_over_epochs": selected_owner["owner_windows_max"] * epochs,
    }

    rows = []
    for claim_unit in claim_units:
        classification = classify_observed_alignment(
            claim_unit, selected_event, selected_owner
        )
        if schedule_mode == "adaptive_private" and claim_unit != "window":
            classification = {
                "verdict": "diagnostic_only",
                "observed_alignment": "observed_adaptive_branch_only",
                "conversion_multiplier": max(
                    1,
                    selected_event["kappa_max"]
                    if claim_unit == "event"
                    else selected_owner["owner_windows_max"],
                ),
                "recommended_statement": (
                    "Diagnostic only: the observed mapping is not sufficient for this raw-unit claim because "
                    "the schedule may adaptively create or select windows based on private "
                    "data, model state, or previous private-training outcomes."
                ),
                "recommended_action": (
                    "Provide a deterministic support covering every adaptive branch, a valid "
                    "externally justified contribution bound, or narrow the report to the "
                    "accounted generated-unit claim."
                ),
            }
        row = dict(base)
        row.update(
            {
                "accounting_unit": "window",
                "claimed_privacy_unit": claim_unit,
                **classification,
            }
        )
        rows.append(row)
    return rows


def rows_from_config(
    config,
    claim_units,
    save_mapping_path=None,
    schedule_mode="fixed_mapping",
    event_kappa_bound=None,
    owner_kappa_bound=None,
    mapping_source="generated_config",
    mapping_file_sha256="",
):
    rows = []
    mapping_records = []
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])

    for stride in config["strides"]:
        stride = int(stride)
        rng = np.random.default_rng(config["seed"] + stride)
        full_records = build_window_records(
            int(config["num_train_sequences"]),
            int(config["sequence_length"]),
            int(config["window_length"]),
            stride,
        )
        selected_records = subsample_records(
            full_records, config.get("max_train_windows"), rng
        )
        mapping_records.extend(selected_records)
        rows.extend(
            audit_scenario(
                f"stride_{stride}",
                selected_records,
                full_records,
                batch_size,
                epochs,
                claim_units,
                schedule_mode,
                event_kappa_bound,
                owner_kappa_bound,
                mapping_source,
                mapping_file_sha256,
                "Generated from config; selected_mapping_sha256 is computed from the generated mapping records.",
            )
        )

    if save_mapping_path is not None:
        write_mapping(mapping_records, save_mapping_path)

    return rows


def rows_from_mapping(
    mapping_records,
    batch_size,
    epochs,
    claim_units,
    schedule_mode="fixed_mapping",
    event_kappa_bound=None,
    owner_kappa_bound=None,
    mapping_source="",
    mapping_file_sha256="",
):
    rows = []
    grouped = group_by_scenario(mapping_records)
    for scenario, selected_records in sorted(grouped.items()):
        rows.extend(
            audit_scenario(
                scenario,
                selected_records,
                selected_records,
                batch_size,
                epochs,
                claim_units,
                schedule_mode,
                event_kappa_bound,
                owner_kappa_bound,
                mapping_source,
                mapping_file_sha256,
                "Read from mapping CSV; selected_mapping_sha256 is computed from the scenario subset.",
            )
        )
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_mapping(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "stride",
        "window_id",
        "generated_unit",
        "owner_id",
        "owner_ids",
        "start",
        "end",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["owner_ids"] = ";".join(record.get("owner_ids", [record["owner_id"]]))
            writer.writerow(row)


def scenario_sort_key(value):
    text = str(value)
    if text.startswith("stride_"):
        try:
            return (0, int(text.split("_", 1)[1]))
        except ValueError:
            return (1, text)
    return (1, text)


def plot_kappa(rows, output_dir):
    event_rows = [r for r in rows if r["claimed_privacy_unit"] == "event"]
    scenarios = sorted({r["scenario"] for r in event_rows}, key=scenario_sort_key)
    event_kappa = []
    owner_kappa = []
    labels = []
    for scenario in scenarios:
        row = next(r for r in event_rows if r["scenario"] == scenario)
        event_kappa.append(float(row["event_kappa_used"]))
        owner_kappa.append(float(row["owner_windows_max_used"]))
        labels.append(str(row["stride"] or scenario))

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(labels, event_kappa, marker="o", label="max windows per event")
    plt.plot(labels, owner_kappa, marker="s", label="max windows per owner")
    plt.xlabel("stride / scenario")
    plt.ylabel("multiplicity")
    plt.title("Raw-unit multiplicity induced by window generation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "kappa_vs_scenario.png", dpi=180)
    plt.close()


def plot_observed_alignment_heatmap(rows, output_dir):
    scenarios = sorted({r["scenario"] for r in rows}, key=scenario_sort_key)
    claim_units = [unit for unit in DEFAULT_CLAIM_UNITS if any(r["claimed_privacy_unit"] == unit for r in rows)]
    extra_units = sorted(
        {
            r["claimed_privacy_unit"]
            for r in rows
            if r["claimed_privacy_unit"] not in claim_units
        }
    )
    claim_units.extend(extra_units)

    alignment_to_value = {
        "generated_unit_identity_candidate": 0,
        "observed_event_kappa_one": 1,
        "observed_owner_kappa_one": 2,
        "observed_event_kappa_gt_one": 3,
        "observed_owner_kappa_gt_one": 4,
        "observed_adaptive_branch_only": 5,
    }
    value_to_color = {
        0: "#2e7d32",
        1: "#66bb6a",
        2: "#90caf9",
        3: "#ef6c00",
        4: "#c62828",
        5: "#757575",
    }

    matrix = np.zeros((len(claim_units), len(scenarios)), dtype=np.int32)
    labels = [["" for _ in scenarios] for _ in claim_units]
    for i, claim_unit in enumerate(claim_units):
        for j, scenario in enumerate(scenarios):
            match = [
                r
                for r in rows
                if r["claimed_privacy_unit"] == claim_unit and r["scenario"] == scenario
            ]
            alignment = (
                match[0]["observed_alignment"]
                if match
                else "observed_adaptive_branch_only"
            )
            matrix[i, j] = alignment_to_value.get(alignment, 5)
            labels[i][j] = alignment.replace("_", "\n")

    rgb = np.empty(matrix.shape + (3,), dtype=float)
    for value, color in value_to_color.items():
        rgb[matrix == value] = tuple(int(color[k : k + 2], 16) / 255.0 for k in (1, 3, 5))

    plt.figure(figsize=(max(7.5, len(scenarios) * 1.4), 3.6))
    plt.imshow(rgb, aspect="auto")
    plt.xticks(range(len(scenarios)), scenarios, rotation=25, ha="right")
    plt.yticks(range(len(claim_units)), claim_units)
    plt.title("Observed incidence diagnostic by requested unit")

    for i in range(len(claim_units)):
        for j in range(len(scenarios)):
            plt.text(j, i, labels[i][j], ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "observed_alignment_heatmap.png", dpi=180)
    plt.close()


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({row["scenario"] for row in rows}, key=scenario_sort_key)
    event_rows = [r for r in rows if r["claimed_privacy_unit"] == "event"]
    owner_rows = [r for r in rows if r["claimed_privacy_unit"] in {"user", "series", "patient", "owner"}]
    alignment_counts = Counter(row["observed_alignment"] for row in rows)

    lines = [
        "# Unit Audit Report",
        "",
        "This report computes observed mapping and attribution diagnostics. It does not authorize a privacy claim.",
        "",
        "The q-related columns, observed multiplicities, and alignment labels are diagnostics only. They are not privacy parameters or validated transformation-stability bounds.",
        "",
        "## Scope",
        "",
        "```text",
        "raw protected unit -> generated window -> accounting unit -> claimed privacy unit",
        "```",
        "",
        "Run `build_unit_certificate.py` with a complete v2.0 privacy report to obtain a fail-closed privacy decision.",
        "",
        "Mapping schema: provide `owner_id` for single-owner windows or `owner_ids` for multi-attribution windows, separated by semicolons. Multi-owner rows are treated as hyperedges: each listed owner is charged one contribution for that generated example.",
        "",
        "## Summary",
        "",
        f"- Schedule mode: {rows[0].get('schedule_mode', 'fixed_mapping') if rows else 'unknown'}",
        f"- Kappa source: {rows[0].get('kappa_source', 'observed_mapping') if rows else 'unknown'}",
        f"- Scenarios audited: {', '.join(str(s) for s in scenarios)}",
        f"- Claim rows audited: {len(rows)}",
    ]

    for alignment, count in sorted(alignment_counts.items()):
        lines.append(f"- {alignment}: {count}")

    lines.extend(
        [
            "",
            "Generated figures:",
            "",
            "```text",
            "kappa_vs_scenario.png",
            "observed_alignment_heatmap.png",
            "```",
            "",
            "## Event-Level Claims",
            "",
            "| scenario | observed event kappa | q window | event load indicator | observed alignment |",
            "|---|---:|---:|---:|---|",
        ]
    )

    for row in event_rows:
        lines.append(
            "| {scenario} | {event_kappa_used} | {q_window:.4f} | {q_event_upper_proxy:.4f} | {observed_alignment} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Owner/User-Level Claims",
            "",
            "| scenario | observed max windows per owner | q window | owner load indicator | observed alignment |",
            "|---|---:|---:|---:|---|",
        ]
    )

    for row in owner_rows:
        lines.append(
            "| {scenario} | {owner_windows_max_used} | {q_window:.4f} | {q_owner_upper_proxy:.4f} | {observed_alignment} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "A generated-unit identity candidate still requires a valid mechanism report, exact adjacency, matched sampler/accountant, and release checks.",
            "",
            "Observed event kappa is not the accountant-metric stability bound. Replacement-to-add/remove metric conversion can double the required K, and preprocessing or selection can expand it further.",
            "",
            "Observed owner incidence is likewise diagnostic. A validated fixed owner partition and absence of cross-owner dependencies are required before using it as K.",
            "",
            "No sentence in this report is copy-ready privacy wording. Only the v2.0 certificate builder may emit an allowed statement.",
            "",
            "## Recommended Paper Use",
            "",
            "Use this as the first method artifact for the revised direction:",
            "",
            "```text",
            "Unit-aware privacy reporting for window-based sequential learning.",
            "```",
            "",
            "This result supports a diagnostic/reporting method. It should not be described as proof that existing medical DP pipelines are broadly wrong.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_mapping_diagnostics(rows, path):
    """Write non-authoritative mapping diagnostics.

    This function intentionally emits no certificate, safe statement, or
    allowed privacy wording. The v2.0 certificate builder is the sole
    authority for those outputs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({row["scenario"] for row in rows}, key=scenario_sort_key)
    lines = [
        "# Mapping and Multiplicity Diagnostics",
        "",
        "This file is not a privacy certificate.",
        "",
        "Observed incidence does not establish transformation stability,",
        "adjacency compatibility, sampler/accountant validity, or release-grade",
        "randomness. Run the v2.0 certificate validator before making any",
        "privacy statement.",
        "",
        "## Scenarios",
        "",
    ]

    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        event_candidates = [
            row
            for row in scenario_rows
            if row["claimed_privacy_unit"] == "event"
        ]
        owner_candidates = [
            row
            for row in scenario_rows
            if row["claimed_privacy_unit"]
            in {"user", "series", "patient", "owner"}
        ]
        reference = event_candidates[0] if event_candidates else scenario_rows[0]
        lines.extend(
            [
                f"### {scenario}",
                "",
                f"- generated records observed: {reference['num_windows_used']}",
                f"- owners observed: {reference['num_owners']}",
                f"- observed event kappa: {reference['event_kappa_used']}",
                f"- observed owner kappa: {reference['owner_windows_max_used']}",
                f"- schedule label: {reference.get('schedule_mode', 'fixed_mapping')}",
                f"- kappa source: {reference.get('kappa_source', 'observed_mapping_only')}",
                f"- formal bound status: {reference.get('formal_bound_status', 'not_validated')}",
                "",
            ]
        )
        if event_candidates:
            lines.append(
                f"- Event diagnostic: {event_candidates[0]['recommended_statement']}"
            )
        if owner_candidates:
            lines.append(
                f"- Owner diagnostic: {owner_candidates[0]['recommended_statement']}"
            )
        lines.extend(
            [
                "",
                "No copy-ready privacy statement is authorized by this file.",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_claim_units(text):
    if not text:
        return DEFAULT_CLAIM_UNITS
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to pilot config JSON.")
    source.add_argument("--mapping", help="Path to window mapping CSV.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--claim-units", default="window,event,user")
    parser.add_argument(
        "--save-mapping",
        default=None,
        help="Optional path to save generated mapping records when using --config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to reports/unit_audit_002.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=["fixed_mapping", "union", "stochastic_bound", "adaptive_private"],
        default="fixed_mapping",
        help="How to interpret the supplied mapping for training-time window generation.",
    )
    parser.add_argument(
        "--event-kappa-bound",
        type=int,
        default=None,
        help="User-supplied event multiplicity bound for stochastic_bound mode.",
    )
    parser.add_argument(
        "--owner-kappa-bound",
        type=int,
        default=None,
        help="User-supplied owner contribution bound for stochastic_bound mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir) if args.output_dir else Path("reports/unit_audit_002")
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    claim_units = parse_claim_units(args.claim_units)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        save_mapping_path = None
        if args.save_mapping:
            save_mapping_path = Path(args.save_mapping)
            if not save_mapping_path.is_absolute():
                save_mapping_path = output_dir / save_mapping_path
        rows = rows_from_config(
            config,
            claim_units,
            save_mapping_path,
            args.schedule_mode,
            args.event_kappa_bound,
            args.owner_kappa_bound,
            str(config_path),
            file_sha256(config_path),
        )
    else:
        mapping_path = Path(args.mapping)
        if not mapping_path.is_absolute():
            mapping_path = project_root / mapping_path
        batch_size = args.batch_size
        epochs = args.epochs
        if batch_size is None or epochs is None:
            raise ValueError("--mapping requires --batch-size and --epochs")
        rows = rows_from_mapping(
            read_mapping(mapping_path),
            batch_size,
            epochs,
            claim_units,
            args.schedule_mode,
            args.event_kappa_bound,
            args.owner_kappa_bound,
            str(mapping_path),
            file_sha256(mapping_path),
        )

    write_csv(rows, output_dir / "unit_audit.csv")
    plot_kappa(rows, output_dir)
    plot_observed_alignment_heatmap(rows, output_dir)
    write_summary(rows, output_dir / "summary.md")
    write_mapping_diagnostics(rows, output_dir / "mapping_diagnostics.md")

    print(f"Wrote {output_dir / 'unit_audit.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'mapping_diagnostics.md'}")
    print(f"Wrote {output_dir / 'kappa_vs_scenario.png'}")
    print(f"Wrote {output_dir / 'observed_alignment_heatmap.png'}")


if __name__ == "__main__":
    main()
