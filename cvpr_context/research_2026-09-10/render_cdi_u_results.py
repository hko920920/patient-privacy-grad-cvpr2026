"""Render already verified CDI U comparison metrics; never compute new statistics.

Only complete extraction plus combined raw/analysis verification PASS is accepted.
This reporting source is separate from the frozen experiment/analysis source map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESEARCH = Path(__file__).resolve().parent
RUN = RESEARCH.parent.parent / "code_working/_reports/cvpr_u_pilot_v1_001"
DEFAULT_RUN = RUN / "baseline_screen_20260914/cdi_u_cohort_v1"
ARTIFACT_ROOT = RESEARCH / "cdi_u_results_artifacts"
LABELS = {
    "cdi_image_mean_fixed": "CDI image score, mean (C=1)",
    "cdi_image_mean_tuned": "CDI image score, mean (tuned C)",
    "cdi_image_max_fixed": "CDI image score, max (C=1)",
    "cdi_image_max_tuned": "CDI image score, max (tuned C)",
    "patient_mean26_fixed": "Patient mean26 LR (C=1)",
    "patient_mean26_tuned": "Patient mean26 LR (tuned C)",
    "patient_meanmax52_fixed": "Patient mean+max52 LR (C=1)",
    "patient_meanmax52_tuned": "Patient mean+max52 LR (tuned C)",
    "dl_mean": "Denoising loss, mean",
    "dl_max": "Denoising loss, max",
    "secmi_mean": "SecMI, mean",
    "secmi_max": "SecMI, max",
    "pia_mean": "PIA, mean",
    "pia_max": "PIA, max",
    "pian_mean": "PIAN, mean",
    "pian_max": "PIAN, max",
    "patient_meanmax54_noobjective_fixed": "NO27 diagnostic: mean+max54 (C=1)",
    "patient_meanmax54_noobjective_tuned": "NO27 diagnostic: mean+max54 (tuned C)",
    "dino_existing": "DINO encoder-only (reused)",
}
COST_FIELDS = [
    "records", "unique_patients", "unique_images", "reused_records", "new_records",
    "forward", "backward", "reused_forward", "reused_backward", "new_forward", "new_backward",
    "current_attempt_forward", "current_attempt_backward", "current_attempt_new_records",
    "current_attempt_seconds", "full_score_seconds_including_reuse", "new_score_seconds_excluding_kernel",
    "NO_objective_evaluations", "NO_nfev", "NO_termination_status_counts", "model_reports",
    "vae_forward", "text_encoder_forward",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def complete_verified_inputs(run_dir, analysis_tag, verification_file):
    """No analysis metrics are opened before completion/combined PASS checks."""
    execution_path = run_dir / "execution.json"
    execution = load_json(execution_path)
    require(execution.get("complete") is True and execution.get("records") == 480, "Full extraction is not complete")
    require(execution.get("status") == "PASS_COHORT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION", "Unexpected extraction status")
    verified = load_json(verification_file)
    require(verified.get("status") == "PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY", "Raw extraction verification must PASS")
    av = verified.get("analysis_verification", {})
    require(av.get("status") == "PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS", "Combined analysis verification must PASS")
    require(av.get("methods") == 18 and av.get("reused_controls") == 1, "Verified method counts differ")
    verification_protocol_path = verification_file.with_name(verification_file.stem + "_protocol.json")
    require(digest(verification_protocol_path) == verified["verification_protocol_sha256"], "Verification protocol changed")
    verification_protocol = load_json(verification_protocol_path)
    require(verification_protocol["analysis_tag"] == analysis_tag, "Verifier checked another analysis tag")
    require(digest(verification_protocol["verification_source"]) == verified["verification_code_sha256"]
            == verification_protocol["expected_code_sha256"], "Verification source changed")
    for name, field in (("execution.json", "execution_sha256"), ("results.json", "results_sha256"), ("protocol.json", "protocol_sha256")):
        actual = digest(run_dir / name)
        require(actual == verified[field] == verification_protocol["input_sha256"][name], f"Changed verified input {name}")
    out = run_dir / analysis_tag
    provenance_path = out / "provenance.json"
    require(digest(provenance_path) == av["provenance_sha256"], "Analysis provenance changed")
    provenance = load_json(provenance_path)
    for name in ("analysis.json", "contract_copy.json"):
        require(digest(out / name) == provenance["output_sha256"][name], f"Changed analysis output {name}")
    require(digest(provenance["source_path"]) == provenance["source_sha256"], "Frozen analysis source changed")
    analysis, contract = load_json(out / "analysis.json"), load_json(out / "contract_copy.json")
    require(analysis["computed_methods"] == 18 and analysis["reused_controls"] == 1, "Analysis counts")
    require(analysis["overall_stage2_gate"] is None and contract["scenario"] == "U", "Analysis scope")
    bindings = {str(p.resolve()): digest(p) for p in (
        execution_path, run_dir / "results.json", run_dir / "protocol.json", verification_file,
        verification_protocol_path, provenance_path, out / "analysis.json", out / "contract_copy.json")}
    return analysis, contract, execution, bindings


def report_rows(analysis, contract):
    order = [s["id"] for s in contract["methods"]] + [contract["reused_control"]["id"]]
    require(len(order) == 19 and set(order) == set(LABELS), "Expected the fixed 19 methods")
    primary = {m for c in contract["contrasts"] if c["role"] == "primary" for m in (c["a"], c["b"])}
    indexed = {}
    for r in analysis["results"]:
        key = (r["model"], r["method"])
        require(key not in indexed, f"Duplicate stored metric {key}")
        indexed[key] = r
    require(set(indexed) == {(m, s) for m in contract["models"] for s in order}, "Missing/extra stored metrics")
    rows = []
    for method in order:
        for model in contract["models"]:
            r = indexed[(model, method)]
            auc = r["selection_patient_AUC"]
            limits = r["selection_patient_AUC_CI95"]
            require(len(limits) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in [auc, *limits]), "Invalid stored AUC/CI")
            require(limits[0] <= limits[1], "Invalid CI order")
            require(r["selection_patients"] == 40 and r["selection_members"] == r["selection_nonmembers"] == 20, "Unexpected selection cohort")
            rows.append({"method": method, "method_label": LABELS[method], "model": model,
                         "in_predeclared_primary_contrast": method in primary,
                         "selection_patients": r["selection_patients"],
                         "selection_members": r["selection_members"], "selection_nonmembers": r["selection_nonmembers"],
                         "patient_AUC": auc, "CI95_lower": limits[0], "CI95_upper": limits[1]})
    return rows, order, primary


def figure(rows, order, primary, warning_count=0):
    """Fixed contractual ordering and axes; no sorting by outcome or new estimates."""
    indexed = {(r["model"], r["method"]): r for r in rows}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "svg.fonttype": "path", "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 10.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.315, right=.97, top=.858, bottom=.195, wspace=.12)
        fig.suptitle("CDI feature adaptations: patient-level U audit", x=.315, y=.965,
                     ha="left", fontsize=20, fontweight="bold")
        fig.text(.315, .923, "19 fixed comparisons | Selection development cohort: 40 patients", fontsize=12)
        for column, (ax, model) in enumerate(zip(axes, ("model_1", "model_2"))):
            for y, method in enumerate(order):
                row = indexed[(model, method)]
                color = "#12679B" if method in primary else "#6D737A"
                if method in primary:
                    ax.axhspan(y-.43, y+.43, color="#EAF4FA", zorder=0)
                # Direct interval lines also handle percentile CIs that exclude the point estimate.
                ax.hlines(y, row["CI95_lower"], row["CI95_upper"], color=color, linewidth=1.8, zorder=2)
                ax.vlines([row["CI95_lower"], row["CI95_upper"]], y-.10, y+.10, color=color, linewidth=1.2, zorder=2)
                ax.plot(row["patient_AUC"], y, "o", color=color, markersize=5.7, zorder=3)
            for y in (7.5, 15.5, 17.5):
                ax.axhline(y, color="#BFC6CC", linewidth=.65, zorder=1)
            ax.axvline(.5, color="#90959B", linestyle=(0, (4, 4)), linewidth=1, zorder=1)
            ax.set_xlim(-.015, 1.015)
            ax.set_ylim(len(order)-.45, -.65)
            ax.set_xticks([0, .25, .5, .75, 1], ["0.00", "0.25", "0.50", "0.75", "1.00"])
            ax.set_xlabel("Patient ROC AUC")
            ax.set_title("Target 1 (A participates)" if column == 0 else "Target 2 (B participates)", pad=16)
            ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
            ax.tick_params(axis="y", length=0, pad=10)
            ax.grid(axis="x", color="#E4E7E9", linewidth=.7, zorder=0)
            for edge in ("top", "right", "left"):
                ax.spines[edge].set_visible(False)
            ax.spines["bottom"].set_color("#B6BDC4")
        for tick, method in zip(axes[0].get_yticklabels(), order):
            if method in primary:
                tick.set_color("#12679B"); tick.set_fontweight("bold")
        fig.legend(handles=[Line2D([0], [0], marker="o", color="#12679B", label="Methods in predeclared primary contrasts", linewidth=1.5),
                            Line2D([0], [0], marker="o", color="#6D737A", label="Other prespecified methods / reused control", linewidth=1.5)],
                   loc="lower left", bbox_to_anchor=(.05, .105), frameon=False, ncol=2, fontsize=10.5)
        footnote = ("Dots: saved patient AUC. Bars: saved 95% percentile bootstrap intervals (2,000 patient resamples).\n"
                    "Previously used development cohort (n=40); the complementary targets are dependent. Intervals are conditional and unadjusted.\n"
                    "Blue identifies primary-comparison methods, not winners. No new statistics were computed for this figure.")
        if warning_count:
            footnote += f"\nFitting warnings recorded by the analysis: {warning_count}. See the saved analysis and CV records."
        fig.text(.05, .09, footnote, ha="left", va="top", fontsize=9.5, color="#4E555C", linespacing=1.6)
        return fig


def render(run_dir, analysis_tag, verification_file, output_tag):
    require(Path(analysis_tag).name == analysis_tag and analysis_tag not in (".", ".."), "Simple analysis tag required")
    require(Path(output_tag).name == output_tag and output_tag not in (".", ".."), "Simple output tag required")
    output = ARTIFACT_ROOT / output_tag
    require(not output.exists(), f"Refusing to overwrite report artifacts: {output}")
    analysis, contract, execution, bindings = complete_verified_inputs(run_dir, analysis_tag, verification_file)
    rows, order, primary = report_rows(analysis, contract)
    fig = figure(rows, order, primary, analysis["warning_count"])
    # Create output only after complete/verified input and plotting validation.
    output.mkdir(parents=True)
    fig.savefig(output / "selection_auc_all19.png", dpi=220, facecolor="white")
    fig.savefig(output / "selection_auc_all19.svg", facecolor="white")
    plt.close(fig)
    with (output / "selection_auc_all19.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "report_table.json", {
        "schema": "cdi-u-report-table/v1", "research_stage": 2,
        "operation": "Copy verified stored selection metrics and costs; no fitting or new statistics",
        "method_order": order, "rows": rows,
        "predeclared_contrasts_as_saved": analysis["contrasts"],
        "extraction_costs_as_saved": {k: execution[k] for k in COST_FIELDS},
        "analysis_seconds_as_saved": analysis["seconds_cpu_analysis"],
        "analysis_warning_count_as_saved": analysis["warning_count"],
        "inference_scope_as_saved": analysis["bootstrap"],
        "scope_limits_as_saved": analysis["scope_limits"],
        "overall_stage2_gate": analysis["overall_stage2_gate"],
    })
    for path, sha in bindings.items():
        require(digest(path) == sha, f"Input changed while rendering: {path}")
    manifest = {"schema": "cdi-u-report-artifacts/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "renderer_source": str(Path(__file__).resolve()), "renderer_sha256": digest(__file__),
                "matplotlib_version": matplotlib.__version__, "input_sha256": bindings,
                "output_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
                "new_statistics": False, "new_fitting": False, "new_GPU_execution": False,
                "winner_selection": False, "human_conclusion_generated": False}
    write_json(output / "manifest.json", manifest)
    return output


def self_test():
    """In-memory synthetic metrics only; no cohort/analysis files or artifact writes."""
    order = list(LABELS)
    primary = {"cdi_image_mean_tuned", "patient_meanmax52_tuned", "secmi_mean"}
    rows = [{"model": model, "method": method, "patient_AUC": .5,
             "CI95_lower": .3, "CI95_upper": .7} for method in order for model in ("model_1", "model_2")]
    fig = figure(rows, order, primary)
    png, svg = io.BytesIO(), io.BytesIO()
    fig.savefig(png, format="png", dpi=80); fig.savefig(svg, format="svg")
    plt.close(fig)
    require(png.getvalue().startswith(b"\x89PNG"), "PNG renderer")
    require(b"<svg" in svg.getvalue(), "SVG renderer")
    print("PASS_SYNTHETIC_IN_MEMORY_RENDER; no actual inputs read or artifacts written")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--analysis-tag", default="analysis_v1")
    p.add_argument("--verification-file", type=Path, help="Combined raw+analysis PASS JSON, required for rendering")
    p.add_argument("--output-tag", default="v1")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    if args.verification_file is None:
        p.error("--verification-file is required; actual rendering needs combined verification PASS")
    output = render(args.run_dir.resolve(), args.analysis_tag, args.verification_file.resolve(), args.output_tag)
    print(json.dumps({"status": "RENDERED_VERIFIED_STORED_METRICS", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
