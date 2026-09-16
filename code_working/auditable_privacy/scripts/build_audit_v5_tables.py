#!/usr/bin/env python3
"""Generate Audit v5 manuscript tables from frozen machine-readable evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = (
    PROJECT_ROOT.parent
    / "AAAI27_7p_compressed"
    / "01_audit_reporting_paper"
)
OUTPUT_ROOT = PAPER_ROOT / "supplement_v5_generated"

RETROSPECTIVE_LABELS = {
    "R01": (
        "Verdict bypassed mapping, sampler, and schedule gates",
        "Unsafe positive possible",
        "All prerequisites gate the hardened validator",
    ),
    "R02": (
        "External conversion attachment was not validated",
        "Fabricated numeric claim possible",
        "Unregistered attachments fail closed",
    ),
    "R03": (
        "Private globally fitted preprocessing omitted from support",
        "Local-support premise invalid",
        "Private fitted preprocessing blocks wording",
    ),
    "R04": (
        "Update divided by a private-data-dependent denominator",
        "Accountant did not cover the executed update",
        "Registered public-constant summed update",
    ),
    "R05": (
        "One-draw Gaussian labeled release-grade",
        "Randomness assurance overclaimed",
        "Registered discarded plus four-draw construction",
    ),
    "R06": (
        "Observed owner incidence used as global stability",
        "Owner conversion bound unproved",
        "Public cap required; vacuous wording blocked",
    ),
    "R07": (
        "Generic replace-one event domain",
        "Event conversion not licensed",
        "Exact registered raw domain required",
    ),
    "R08": (
        "Registry did not bind executable semantics",
        "Self-consistent source forgery possible",
        "Exact source and semantics are registry-bound",
    ),
    "R09": (
        "Missing runtime still authorized wording",
        "Post-execution claim lacked execution evidence",
        "Matched runtime is mandatory",
    ),
    "R10": (
        "Float-threshold sampler approximated a rational rate",
        "Exact accountant semantics violated",
        "Exact-rational sampling is bound",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tex_escape(value: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    result = value
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    global PROJECT_ROOT, PAPER_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing reports/ and scripts/.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=None,
        help="Paper directory; generated fragments are written below it.",
    )
    args = parser.parse_args()
    PROJECT_ROOT = args.project_root.resolve()
    PAPER_ROOT = (
        args.paper_root.resolve()
        if args.paper_root is not None
        else (
            PROJECT_ROOT.parent
            / "AAAI27_7p_compressed"
            / "01_audit_reporting_paper"
        ).resolve()
    )
    OUTPUT_ROOT = PAPER_ROOT / "supplement_v5_generated"

    retrospective_csv = (
        PROJECT_ROOT
        / "reports/audit_retrospective_failures_v5_001/incidents.csv"
    )
    retrospective_verification_path = (
        PROJECT_ROOT
        / "reports/audit_retrospective_failures_v5_001/"
        "independent_verification.json"
    )
    uci_metrics_path = (
        PROJECT_ROOT
        / "reports/uci_har_v5_multirun_001/confirmatory_metrics.csv"
    )
    uci_summary_path = (
        PROJECT_ROOT
        / "reports/uci_har_v5_multirun_001/confirmatory_summary.json"
    )
    p0_gate_path = (
        PROJECT_ROOT / "reports/audit_v5_p0_gate_001/gate.json"
    )

    with retrospective_csv.open(newline="", encoding="utf-8") as handle:
        incidents = list(csv.DictReader(handle))
    retrospective_verification = load_json(retrospective_verification_path)
    with uci_metrics_path.open(newline="", encoding="utf-8") as handle:
        uci_rows = list(csv.DictReader(handle))
    uci_summary = load_json(uci_summary_path)
    p0_gate = load_json(p0_gate_path)

    if [row["id"] for row in incidents] != list(RETROSPECTIVE_LABELS):
        raise SystemExit("Retrospective incident set/order changed")
    if not (
        retrospective_verification["all_passed"]
        and retrospective_verification["checks_passed"]
        == retrospective_verification["checks_total"]
        == 176
    ):
        raise SystemExit("Retrospective verification is not 176/176")
    if len(uci_rows) != 5:
        raise SystemExit("Expected exactly five UCI confirmatory rows")
    if len({row["model_sha256"] for row in uci_rows}) != 5:
        raise SystemExit("UCI model hashes are not distinct")
    if not (
        uci_summary["verification"]["all_passed"]
        and uci_summary["verification"]["checks_passed"]
        == uci_summary["verification"]["checks_total"]
        == 31
    ):
        raise SystemExit("UCI aggregate verification is not 31/31")
    if p0_gate["status"] != "PASS":
        raise SystemExit("P0 gate is not PASS")

    for metric in ("train_accuracy", "test_accuracy", "test_macro_f1", "pipeline_seconds"):
        values = [float(row[metric]) for row in uci_rows]
        reported = uci_summary["statistics"][metric]
        if not (
            math.isclose(statistics.fmean(values), reported["mean"], rel_tol=1e-15)
            and math.isclose(
                statistics.stdev(values),
                reported["sample_sd"],
                rel_tol=1e-15,
            )
        ):
            raise SystemExit(f"UCI statistic mismatch: {metric}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    retrospective_lines = [
        r"\begin{longtable}{p{0.06\textwidth}p{0.29\textwidth}p{0.25\textwidth}p{0.29\textwidth}}",
        r"\caption{Ten pre-protocol defects from the preserved development lineage. These authenticated system-lineage incidents are separate from the designed 200-case suite; the legacy 31-case surface is retained only as a regression. They are not an independently sampled paper corpus or prevalence estimate.}\label{tab:v5-retrospective}\\",
        r"\hline",
        r"ID & Preserved root cause & Pre-repair consequence & Current control \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"ID & Preserved root cause & Pre-repair consequence & Current control \\",
        r"\hline",
        r"\endhead",
    ]
    for row in incidents:
        label, consequence, control = RETROSPECTIVE_LABELS[row["id"]]
        retrospective_lines.append(
            f"{row['id']} & {tex_escape(label)} & "
            f"{tex_escape(consequence)} & {tex_escape(control)} \\\\"
        )
    retrospective_lines.extend([r"\hline", r"\end{longtable}", ""])
    write(
        OUTPUT_ROOT / "retrospective_incidents.tex",
        "\n".join(retrospective_lines),
    )

    uci_run_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"Run & Train accuracy & Test accuracy & Macro-F1 & Seconds & Model SHA-256 prefix \\",
        r"\hline",
    ]
    for row in uci_rows:
        uci_run_lines.append(
            f"{tex_escape(row['run_id'].replace('confirmatory_', ''))} & "
            f"{float(row['train_accuracy']):.6f} & "
            f"{float(row['test_accuracy']):.6f} & "
            f"{float(row['test_macro_f1']):.6f} & "
            f"{float(row['pipeline_seconds']):.3f} & "
            rf"\texttt{{{row['model_sha256'][:12]}}} \\"
        )
    uci_run_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{All five protocol-fixed UCI HAR confirmatory executions. The earlier complete pilot is chronologically excluded; no confirmatory run was selected or replaced by utility.}",
            r"\label{tab:v5-uci-runs}",
            r"\end{table*}",
            "",
        ]
    )
    write(OUTPUT_ROOT / "uci_confirmatory_runs.tex", "\n".join(uci_run_lines))

    decision_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llll}",
        r"\hline",
        r"Requested unit & Release status & Path & $K$ \\",
        r"\hline",
    ]
    for unit, status, path, bound, has_wording in uci_summary[
        "decision_signature"
    ]:
        decision_lines.append(
            f"{tex_escape(unit)} & "
            f"{tex_escape(status)} & "
            f"{tex_escape(path)} & "
            f"{'--' if bound is None else bound} \\\\"
        )
        if (status == "ALLOWED") != bool(has_wording):
            raise SystemExit(f"Unexpected wording flag for {unit}")
    decision_lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Invariant UCI HAR authorization boundary in all five runs. Only the published generated-window unit is licensed.}",
            r"\label{tab:v5-uci-decisions}",
            r"\end{table*}",
            "",
        ]
    )
    write(OUTPUT_ROOT / "uci_decisions.tex", "\n".join(decision_lines))

    claims = p0_gate["claims"]
    stats = uci_summary["statistics"]
    macros = [
        rf"\newcommand{{\AuditContractTestCount}}{{{claims['contract_tests']}}}",
        rf"\newcommand{{\AuditRetrospectiveIncidentCount}}{{{len(incidents)}}}",
        rf"\newcommand{{\AuditRetrospectiveCheckCount}}{{{retrospective_verification['checks_passed']}}}",
        rf"\newcommand{{\AuditUciRunCount}}{{{len(uci_rows)}}}",
        rf"\newcommand{{\AuditUciFullCheckCount}}{{{claims['uci_full_checks_per_run']}}}",
        rf"\newcommand{{\AuditUciPortableCheckCount}}{{{claims['uci_portable_checks_per_run']}}}",
        rf"\newcommand{{\AuditUciAggregateCheckCount}}{{{claims['uci_aggregate_checks']}}}",
        rf"\newcommand{{\AuditUciAccuracyMean}}{{{stats['test_accuracy']['mean']:.4f}}}",
        rf"\newcommand{{\AuditUciAccuracySD}}{{{stats['test_accuracy']['sample_sd']:.4f}}}",
        rf"\newcommand{{\AuditUciMacroFOneMean}}{{{stats['test_macro_f1']['mean']:.4f}}}",
        rf"\newcommand{{\AuditUciMacroFOneSD}}{{{stats['test_macro_f1']['sample_sd']:.4f}}}",
        rf"\newcommand{{\AuditUciPipelineMean}}{{{stats['pipeline_seconds']['mean']:.3f}}}",
        rf"\newcommand{{\AuditUciPipelineSD}}{{{stats['pipeline_seconds']['sample_sd']:.3f}}}",
        "",
    ]
    write(OUTPUT_ROOT / "v5_macros.tex", "\n".join(macros))

    verification_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Surface & Passing checks \\",
        r"\hline",
        rf"Contract regression & {claims['contract_tests']}/{claims['contract_tests']} \\",
        rf"Controlled probes & {claims['controlled_probe_verifier_checks']}/{claims['controlled_probe_verifier_checks']} \\",
        rf"Retrospective lineage & {claims['retrospective_verifier_checks']}/{claims['retrospective_verifier_checks']} \\",
        rf"UCI full (each of five) & {claims['uci_full_checks_per_run']}/{claims['uci_full_checks_per_run']} \\",
        rf"UCI portable (each of five) & {claims['uci_portable_checks_per_run']}/{claims['uci_portable_checks_per_run']} \\",
        rf"UCI aggregate & {claims['uci_aggregate_checks']}/{claims['uci_aggregate_checks']} \\",
        r"\hline",
        r"\end{tabular}",
        r"\caption{New and re-audited v5 verification surfaces. Counts expose checked predicates; they are not statistical confidence.}",
        r"\label{tab:v5-verification}",
        r"\end{table}",
        "",
    ]
    write(
        OUTPUT_ROOT / "verification_surface.tex",
        "\n".join(verification_lines),
    )

    input_paths = (
        retrospective_csv,
        retrospective_verification_path,
        uci_metrics_path,
        uci_summary_path,
        p0_gate_path,
    )
    output_paths = tuple(
        OUTPUT_ROOT / name
        for name in (
            "retrospective_incidents.tex",
            "uci_confirmatory_runs.tex",
            "uci_decisions.tex",
            "v5_macros.tex",
            "verification_surface.tex",
        )
    )
    provenance: dict[str, Any] = {
        "schema_version": "audit_v5_generated_tables_v1",
        "builder": {
            "path": "scripts/build_audit_v5_tables.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "inputs": {
            path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
            for path in input_paths
        },
        "outputs": {
            path.name: file_sha256(path)
            for path in output_paths
        },
        "claims": {
            "retrospective_incidents": len(incidents),
            "retrospective_checks": retrospective_verification[
                "checks_passed"
            ],
            "uci_runs": len(uci_rows),
            "uci_decision_signature": uci_summary["decision_signature"],
            "uci_statistics": uci_summary["statistics"],
            "p0_claims": claims,
        },
    }
    provenance["content_sha256"] = canonical_json_sha256(provenance)
    write(
        OUTPUT_ROOT / "provenance.json",
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
    print(
        "Audit v5 generated tables: "
        f"{len(output_paths)} files; "
        f"provenance={provenance['content_sha256']}"
    )


if __name__ == "__main__":
    main()
