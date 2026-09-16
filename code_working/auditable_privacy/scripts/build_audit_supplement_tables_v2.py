#!/usr/bin/env python3
"""Build evidence-derived tables for the Audit paper's v2 supplement.

The generated LaTeX is deliberately derived from the frozen machine-readable
artifacts instead of hand-copied manuscript numbers.  The script also writes a
deterministic provenance manifest with SHA-256 digests for every input and
output.  It does not run experiments or upgrade mapping-only evidence into a
privacy claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latex_text(value: object) -> str:
    """Escape prose for LaTeX while keeping the generated file ASCII."""

    text = str(value).replace("`", "")
    text = (
        text.replace("“", "``")
        .replace("”", "''")
        .replace("’", "'")
        .replace("—", "---")
        .replace("–", "--")
        .replace("−", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("→", "->")
        .replace("ε", "epsilon")
        .replace("δ", "delta")
        .replace("κ", "kappa")
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def status_text(value: object) -> str:
    """Typeset machine labels with legal line breaks at underscores."""

    raw = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.-]+(?:_[A-Za-z0-9_.-]+)*", raw):
        pieces = []
        for piece in raw.split("_"):
            escaped_piece = latex_text(piece)
            semantic_breaks = {
                "underspecified": r"UNDER\allowbreak SPECIFIED",
                "systemrandom": r"system\allowbreak random",
            }
            escaped_piece = semantic_breaks.get(piece.lower(), escaped_piece)
            pieces.append(escaped_piece)
        return r"\texttt{" + r"\_\allowbreak ".join(pieces) + "}"
    escaped = latex_text(raw)
    return r"\texttt{" + escaped.replace(r"\_", r"\_\allowbreak ") + "}"


def number(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "--"
    return f"{int(str(value)):,}"


def scientific(value: object, digits: int = 6) -> str:
    if value is None or str(value).strip() == "":
        return "--"
    return f"{float(value):.{digits}g}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_truth_table(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*T\d{2}\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        require(len(cells) == 7, f"Malformed truth-table row: {line}")
        rows.append(cells)
    require(len(rows) == 37, f"Expected 37 truth-table rows, found {len(rows)}")
    require(
        [row[0] for row in rows] == [f"T{index:02d}" for index in range(1, 38)],
        "Truth-table identifiers are not exactly T01--T37",
    )
    return rows


def longtable(
    *,
    caption: str,
    label: str,
    column_spec: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    font: str = r"\scriptsize",
) -> str:
    header = " & ".join(headers) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(
        [
            r"\begingroup",
            font,
            r"\setlength{\tabcolsep}{1.5pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            rf"\begin{{longtable}}{{{column_spec}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{len(headers)}}}{{l}}{{\itshape Table \thetable\ continued}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
            rf"\midrule \multicolumn{{{len(headers)}}}{{r}}{{\itshape Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            body,
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )


def build_truth_table(rows: list[list[str]]) -> str:
    rendered = []
    for row_id, condition, stability, path, release, assurance, action in rows:
        outcome = (
            latex_text(f"K={stability}; ")
            + status_text(path)
            + "; "
            + status_text(release)
            + "; "
            + status_text(assurance)
        )
        rendered.append(
            [
                status_text(row_id),
                latex_text(condition),
                outcome,
                latex_text(action),
            ]
        )
    return longtable(
        caption=(
            "Normative evidence-to-wording truth table. A plugin row is "
            "conditional on a registered checker; absent such a checker, the "
            "validator fails closed."
        ),
        label="tab:supp-truth-table",
        column_spec=(
            r"@{}p{.035\textwidth}p{.310\textwidth}"
            r"p{.310\textwidth}p{.290\textwidth}@{}"
        ),
        headers=["ID", "Evidence condition", "Expected tuple", "Required action"],
        rows=rendered,
        font=r"\fontsize{7.0}{8.0}\selectfont",
    )


def build_conformance_cases(cases: list[dict[str, Any]]) -> str:
    require(len(cases) == 31, f"Expected 31 conformance cases, found {len(cases)}")
    ids = [case["case_id"] for case in cases]
    require(len(ids) == len(set(ids)), "Conformance case identifiers are not unique")
    rendered = []
    for case in cases:
        numeric = case["expected_numeric_rule"]
        if case.get("expected_epsilon") is not None:
            numeric += (
                f"; eps={scientific(case['expected_epsilon'], 7)}, "
                f"delta={scientific(case['expected_delta'], 7)}"
            )
        issue = case.get("expected_issue_code") or "--"
        expected = (
            status_text(case["expected_unit_path"])
            + "; "
            + status_text(case["expected_release_status"])
            + "; "
            + status_text(case["expected_assurance_status"])
        )
        rendered.append(
            [
                status_text(case["case_id"]),
                latex_text(f"{case['truth_row']}; {case['claim_unit']}"),
                expected,
                status_text(issue) if issue != "--" else "--",
                latex_text(numeric),
                latex_text(case["description"]),
            ]
        )
    return longtable(
        caption=(
            "Frozen conformance and repair-regression cases with manually "
            "declared expected tuples. "
            "The descriptions are test conditions, not empirical prevalence estimates."
        ),
        label="tab:supp-conformance-cases",
        column_spec=(
            r"@{}p{.140\textwidth}p{.075\textwidth}p{.245\textwidth}"
            r"p{.170\textwidth}p{.130\textwidth}p{.200\textwidth}@{}"
        ),
        headers=[
            "Case",
            "Truth; unit",
            "Expected path; release; assurance",
            "Required issue",
            "Numeric oracle",
            "Isolated condition",
        ],
        rows=rendered,
        font=r"\fontsize{6.7}{7.7}\selectfont",
    )


def build_ablation(rows: list[dict[str, str]]) -> str:
    require(len(rows) == 8, f"Expected 8 ablation rows, found {len(rows)}")
    names = {
        "B0_FIELD_PRESENCE": "B0 field presence",
        "B1_JSON_SCHEMA": "B1 JSON Schema",
        "B2_MULTIPLICITY_ONLY": "B2 multiplicity only",
        "B3_HASH_ONLY": "B3 hash only",
        "B4_ACCOUNTANT_ONLY": "B4 accountant only",
        "A1_FULL_MINUS_RUNTIME": "A1 FULL minus runtime",
        "A2_FULL_MINUS_VACUITY": "A2 FULL minus vacuity",
        "FULL": "FULL",
    }
    require(
        {row["method"] for row in rows} == set(names),
        "Ablation method set differs from the frozen B0--B4/A1--A2/FULL set",
    )
    rendered = []
    for row in rows:
        blocked = row["blocked_truth_cases"]
        valid = row["valid_truth_cases"]
        cases = row["cases"]
        rendered.append(
            [
                latex_text(names[row["method"]]),
                f"{row['unsafe_positives']}/{blocked}",
                f"{row['valid_retained']}/{valid}",
                f"{row['valid_exact_decisions']}/{valid}",
                f"{row['exact_tuples']}/{cases}",
                row["wrong_same_number"],
                f"{float(row['median_wall_ms']):.4g}",
                f"{float(row['max_wall_ms']):.4g}",
            ]
        )
    return longtable(
        caption=(
            "Controlled obligation probes on the shared 31-case input set. "
            "B0--B4 are deliberately incomplete probes, not competitor "
            "systems. Timing is wall-clock milliseconds and descriptive only."
        ),
        label="tab:supp-ablation",
        column_spec=(
            r"@{}p{.235\textwidth}p{.100\textwidth}p{.095\textwidth}"
            r"p{.105\textwidth}p{.105\textwidth}p{.105\textwidth}"
            r"p{.090\textwidth}p{.090\textwidth}@{}"
        ),
        headers=[
            "Probe / full contract",
            "Non-allow $+$",
            "Valid kept",
            "Valid exact",
            "Exact tuple",
            "Same-number err.",
            "Median ms",
            "Max ms",
        ],
        rows=rendered,
    )


def build_tfprivacy_families(payload: dict[str, Any]) -> str:
    require(payload["cases"] == 23, "TensorFlow Privacy scope comparison must contain 23 cases")
    require(
        payload["valid_claims_retained"] == 7
        and payload["unvalidated_positives"] == 15,
        "TensorFlow Privacy aggregate differs from frozen 7/7 and 15/16 result",
    )
    rows = []
    for family, values in sorted(payload["by_failure_family"].items()):
        rows.append(
            [
                status_text(family),
                str(values["cases"]),
                str(values["finite_statements"]),
                str(values["unvalidated_positives"]),
            ]
        )
    return longtable(
        caption=(
            "TensorFlow Privacy 0.9.0 statement-function outputs by frozen "
            "failure family on the original 23-case scalar-compatible subset. "
            "``Unvalidated'' is relative to this paper's external-evidence "
            "contract, not a defect in the statement function or an overall rank."
        ),
        label="tab:supp-tfp-families",
        column_spec=(
            r"@{}p{.45\textwidth}p{.14\textwidth}p{.18\textwidth}"
            r"p{.18\textwidth}@{}"
        ),
        headers=["Family", "Cases", "Finite statements", "Unvalidated positives"],
        rows=rows,
    )


def build_wisdm(
    metrics: dict[str, Any],
    report: dict[str, Any],
    public_claims: dict[str, dict[str, Any]],
    owner_internal: dict[str, Any],
) -> str:
    require(metrics["train_records"] == 7304, "Unexpected WISDM train record count")
    require(metrics["test_records"] == 3506, "Unexpected WISDM test record count")
    require(report["runtime_trace"]["status"] == "matched", "WISDM runtime is not matched")
    expected = {
        "window": ("DIRECT", "ALLOWED", 1),
        "event": ("CONVERT", "ALLOWED", 4),
        "owner": ("GROUP", "BLOCKED_VACUOUS", 600),
    }
    rows = []
    for unit in ("window", "event", "owner"):
        claim = public_claims[unit]
        internal = owner_internal if unit == "owner" else claim
        path, release, stability = expected[unit]
        require(claim["unit_path"] == path, f"Unexpected WISDM {unit} path")
        require(claim["release_status"] == release, f"Unexpected WISDM {unit} release")
        require(int(internal["stability_bound"]) == stability, f"Unexpected WISDM {unit} K")
        observed = {
            "window": "--",
            "event": str(claim["observed_event_kappa"]),
            "owner": str(claim["observed_owner_kappa"]),
        }[unit]
        if unit == "owner":
            public_pair = (
                "none; diagnostic "
                f"eps_K={float(owner_internal['reported_epsilon']):.6g}, "
                f"log10(delta_K)={float(owner_internal['reported_delta_log10']):.9g}"
            )
        else:
            public_pair = (
                f"({float(claim['epsilon']):.12g}, "
                f"{float(claim['delta']):.12g})"
            )
        rows.append(
            [
                latex_text(unit),
                status_text(claim["raw_adjacency"]),
                observed,
                f"{stability}; " + status_text(path),
                status_text(release),
                latex_text(public_pair),
            ]
        )
    return longtable(
        caption=(
            "Three queries over one digest-bound WISDM execution. The owner "
            "diagnostic is retained internally but no owner privacy pair is public."
        ),
        label="tab:supp-wisdm-claims",
        column_spec=(
            r"@{}p{.08\textwidth}p{.17\textwidth}p{.12\textwidth}"
            r"p{.14\textwidth}p{.18\textwidth}p{.27\textwidth}@{}"
        ),
        headers=[
            "Claim unit",
            "Raw adjacency",
            "Observed incidence",
            "$K$; path",
            "Release",
            "Public pair or diagnostic",
        ],
        rows=rows,
    )


def build_sepsis(rows: list[dict[str, str]]) -> str:
    require(len(rows) == 16, f"Expected 16 Sepsis policies, found {len(rows)}")
    rendered = []
    for row in rows:
        if row["selection"] == "all":
            policy = f"L={row['window_length']}, s={row['stride']}, all"
        else:
            policy = f"L={row['window_length']}, {row['selection']}, cap={row['cap']}"
        event_k = row["candidate_event_stability_k"] or "--"
        selection_class = (
            "private-dependent"
            if row["selection_classification"] == "private_data_dependent"
            else "fixed/public"
        )
        require(row["privacy_authority"] == "diagnostic_only", "Sepsis row is not diagnostic")
        rendered.append(
            [
                latex_text(policy),
                number(row["num_windows"]),
                number(row["owners_with_windows"]),
                row["observed_event_kappa"],
                event_k,
                row["observed_owner_kappa"],
                row["candidate_owner_stability_k"],
                latex_text(selection_class),
            ]
        )
    return longtable(
        caption=(
            "All Sepsis mapping policies. Candidate $K$ values are transformation "
            "diagnostics only; no row binds a trained mechanism or privacy accountant. "
            "The three label-balanced rows have no authorized event $K$."
        ),
        label="tab:supp-sepsis",
        column_spec=(
            r"@{}p{.30\textwidth}p{.10\textwidth}p{.09\textwidth}"
            r"p{.07\textwidth}p{.07\textwidth}p{.08\textwidth}"
            r"p{.08\textwidth}p{.11\textwidth}@{}"
        ),
        headers=[
            "Policy",
            "Windows",
            "Owners",
            "$\\kappa_e$",
            "Cand. $K_e$",
            "$\\kappa_o$",
            "Cand. $K_o$",
            "Selection",
        ],
        rows=rendered,
    )


def build_backblaze(rows: list[dict[str, str]]) -> str:
    require(len(rows) == 22, f"Expected 22 Backblaze policies, found {len(rows)}")
    rendered = []
    for row in rows:
        require(row["privacy_authority"] == "diagnostic_only", "Backblaze row is not diagnostic")
        if row["semantics"] == "owner_snapshot_prefix_cap":
            policy = f"owner prefix cap {row['cap']}"
            semantics = "owner prefix"
        else:
            suffix = row["semantics"].replace("_", " ")
            policy = f"L={row['window_length']}, s={row['stride']}, {row['policy']}"
            semantics = suffix
        rendered.append(
            [
                latex_text(semantics),
                latex_text(policy),
                number(row["num_windows"]),
                number(row["owners_with_windows"]),
                row["observed_event_kappa"],
                row["candidate_event_stability_k"],
                row["observed_owner_kappa"],
                row["candidate_owner_stability_k"],
            ]
        )
    return longtable(
        caption=(
            "All Backblaze Q1 2025 mapping policies. Ordered-snapshot windows "
            "compress missing calendar days; calendar-contiguous windows do not. "
            "Every row is mapping-only and carries no epsilon or delta."
        ),
        label="tab:supp-backblaze",
        column_spec=(
            r"@{}p{.16\textwidth}p{.22\textwidth}p{.12\textwidth}"
            r"p{.10\textwidth}p{.065\textwidth}p{.07\textwidth}"
            r"p{.075\textwidth}p{.075\textwidth}@{}"
        ),
        headers=[
            "Semantics",
            "Policy",
            "Windows",
            "Owners",
            "$\\kappa_e$",
            "Cand. $K_e$",
            "$\\kappa_o$",
            "Cand. $K_o$",
        ],
        rows=rendered,
    )


def verification_count(payload: dict[str, Any]) -> tuple[int, int, bool]:
    passed = bool(payload.get("verification_passed", payload.get("passed", False)))
    actual = int(payload.get("checks_passed", payload.get("check_count", 0)))
    total = int(payload.get("checks_total", payload.get("check_count", 0)))
    return actual, total, passed


def build_verification_surface(payloads: dict[str, dict[str, Any]]) -> str:
    expected = {
        "Conformance and ablations": (394, "full decision artifacts"),
        "TF Privacy scope comparison": (78, "official scalar statement execution"),
        "WISDM hardened execution": (38, "raw rescan--runtime--accountant--report"),
        "Sepsis mapping study": (128, "mapping only"),
        "Backblaze mapping study": (86, "archive rescan; mapping only"),
    }
    rows = []
    for name, (count, authority) in expected.items():
        actual, total, passed = verification_count(payloads[name])
        require((actual, total, passed) == (count, count, True), f"{name} verifier is not {count}/{count}")
        rows.append(
            [
                latex_text(name),
                f"{actual}/{total}",
                latex_text("pass"),
                latex_text(authority),
            ]
        )
    return longtable(
        caption=(
            "Independent verification surfaces. Counts enumerate deterministic "
            "checks and are not statistical confidence measures."
        ),
        label="tab:supp-verification",
        column_spec=(
            r"@{}p{.35\textwidth}p{.14\textwidth}p{.12\textwidth}"
            r"p{.35\textwidth}@{}"
        ),
        headers=["Bundle", "Checks", "Result", "Authority"],
        rows=rows,
    )


def build_artifact_inputs(repo_root: Path, input_paths: list[Path]) -> str:
    rows = []
    for path in sorted(input_paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        rows.append(
            [
                rf"\path{{{relative}}}",
                rf"\texttt{{\seqsplit{{{sha256_file(path)}}}}}",
            ]
        )
    return longtable(
        caption=(
            "Machine-readable sources consumed by the supplement table builder. "
            "The package-level manifest additionally hashes generated outputs."
        ),
        label="tab:supp-artifact-inputs",
        column_spec=r"@{}p{.48\textwidth}p{.48\textwidth}@{}",
        headers=["Repository-relative input", "SHA-256"],
        rows=rows,
        font=r"\fontsize{6.0}{6.9}\selectfont",
    )


def write_generated(path: Path, content: str) -> None:
    path.write_text(
        "% AUTO-GENERATED by scripts/build_audit_supplement_tables_v2.py.\n"
        "% Do not edit by hand; rebuild from the frozen evidence artifacts.\n"
        + content,
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DP-SGD implementation/evidence repository root",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=None,
        help="Audit paper directory (defaults to the adjacent AAAI paper tree)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paper_root = (
        args.paper_root.resolve()
        if args.paper_root is not None
        else (
            repo_root.parent
            / "AAAI27_7p_compressed"
            / "01_audit_reporting_paper"
        ).resolve()
    )
    output_dir = paper_root / "supplement_v2_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "truth": repo_root / "docs/privacy_claim_truth_table_v2_0.md",
        "cases": repo_root / "reports/validator_ablation_hardened_002/cases.json",
        "ablation": repo_root / "reports/validator_ablation_hardened_002/aggregate_metrics.csv",
        "tfprivacy": repo_root / "reports/tfprivacy_statement_baseline_v2_001/aggregate_metrics.json",
        "wisdm_metrics": repo_root / "reports/wisdm_v2_hardened_secure_002/metrics.json",
        "wisdm_report": repo_root / "reports/wisdm_v2_hardened_secure_002/privacy_report.json",
        "wisdm_window": repo_root
        / "reports/wisdm_v2_hardened_secure_002/certificates/certificate_wisdm_v2_hardened_overlap50_window.json",
        "wisdm_event": repo_root
        / "reports/wisdm_v2_hardened_secure_002/certificates/certificate_wisdm_v2_hardened_overlap50_event.json",
        "wisdm_owner": repo_root
        / "reports/wisdm_v2_hardened_secure_002/certificates/certificate_wisdm_v2_hardened_overlap50_owner.json",
        "wisdm_owner_internal": repo_root
        / "reports/wisdm_v2_hardened_secure_002/certificates/certificate_wisdm_v2_hardened_overlap50_owner_internal.json",
        "sepsis": repo_root
        / "reports/sepsis2019_v2_mapping_evidence_5k_001/policy_summary.csv",
        "backblaze": repo_root
        / "reports/backblaze_v2_mapping_evidence_q1_2025_001/policy_summary.csv",
        "verify_ablation": repo_root
        / "reports/validator_ablation_hardened_002/independent_verification.json",
        "verify_tfprivacy": repo_root
        / "reports/tfprivacy_statement_baseline_v2_001/independent_verification.json",
        "verify_wisdm": repo_root
        / "reports/wisdm_v2_hardened_secure_002/independent_verification.json",
        "verify_sepsis": repo_root
        / "reports/sepsis2019_v2_mapping_evidence_5k_001/independent_verification.json",
        "verify_backblaze": repo_root
        / "reports/backblaze_v2_mapping_evidence_q1_2025_001/independent_verification.json",
    }
    for name, path in paths.items():
        require(path.is_file(), f"Missing required input {name}: {path}")

    truth_rows = parse_truth_table(paths["truth"])
    cases_payload = load_json(paths["cases"])
    cases = cases_payload["cases"]
    ablation = load_csv(paths["ablation"])
    tfprivacy = load_json(paths["tfprivacy"])
    metrics = load_json(paths["wisdm_metrics"])
    wisdm_report = load_json(paths["wisdm_report"])
    wisdm_claims = {
        "window": load_json(paths["wisdm_window"]),
        "event": load_json(paths["wisdm_event"]),
        "owner": load_json(paths["wisdm_owner"]),
    }
    owner_internal = load_json(paths["wisdm_owner_internal"])
    sepsis = load_csv(paths["sepsis"])
    backblaze = load_csv(paths["backblaze"])
    verification = {
        "Conformance and ablations": load_json(paths["verify_ablation"]),
        "TF Privacy scope comparison": load_json(paths["verify_tfprivacy"]),
        "WISDM hardened execution": load_json(paths["verify_wisdm"]),
        "Sepsis mapping study": load_json(paths["verify_sepsis"]),
        "Backblaze mapping study": load_json(paths["verify_backblaze"]),
    }

    outputs = {
        "truth_table.tex": build_truth_table(truth_rows),
        "conformance_cases.tex": build_conformance_cases(cases),
        "ablation_metrics.tex": build_ablation(ablation),
        "tfprivacy_families.tex": build_tfprivacy_families(tfprivacy),
        "wisdm_claims.tex": build_wisdm(metrics, wisdm_report, wisdm_claims, owner_internal),
        "sepsis_policies.tex": build_sepsis(sepsis),
        "backblaze_policies.tex": build_backblaze(backblaze),
        "verification_surface.tex": build_verification_surface(verification),
        "artifact_inputs.tex": build_artifact_inputs(repo_root, list(paths.values())),
    }
    for filename, content in outputs.items():
        write_generated(output_dir / filename, content)

    generator_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "audit_supplement_tables_v2.0",
        "generator": {
            "path": generator_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(generator_path),
        },
        "authority": {
            "wisdm": "complete_execution",
            "sepsis": "mapping_diagnostic_only",
            "backblaze": "mapping_diagnostic_only",
        },
        "counts": {
            "truth_rows": len(truth_rows),
            "conformance_cases": len(cases),
            "ablation_methods": len(ablation),
            "sepsis_policies": len(sepsis),
            "backblaze_policies": len(backblaze),
        },
        "inputs": {
            name: {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(paths.items())
        },
        "outputs": {
            filename: {
                "path": (output_dir / filename).relative_to(paper_root).as_posix(),
                "sha256": sha256_file(output_dir / filename),
            }
            for filename in sorted(outputs)
        },
    }
    manifest_path = output_dir / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "generated_tables": len(outputs),
                "truth_rows": len(truth_rows),
                "conformance_cases": len(cases),
                "sepsis_policies": len(sepsis),
                "backblaze_policies": len(backblaze),
                "provenance_sha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
