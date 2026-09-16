#!/usr/bin/env python3
"""Build the Compiler V3.9 Author-Kit format and preservation gate."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "AAAI27_7p_compressed/02_algorithm_compiler_paper"
V38_SOURCE = PAPER / "main_aaai27_v38_candidate.tex"
V38_PDF = V38_SOURCE.with_suffix(".pdf")
V39_SOURCE = PAPER / "main_aaai27_v39_candidate.tex"
V39_PDF = V39_SOURCE.with_suffix(".pdf")
V39_LOG = V39_SOURCE.with_suffix(".log")
SUPP_SOURCE = PAPER / "main_aaai27_supplement_v38.tex"
SUPP_PDF = SUPP_SOURCE.with_suffix(".pdf")
CHECKLIST_SOURCE = PAPER / "main_aaai27_reproducibility_checklist_v38.tex"
CHECKLIST_ANSWERS = PAPER / "main_aaai27_reproducibility_checklist_answers_v38.tex"
CHECKLIST_PDF = CHECKLIST_SOURCE.with_suffix(".pdf")
TABLE_DIR = PAPER / "tables_v38_candidate"
TABLE_STEMS = ("v38_handler_ablation", "v38_profiles", "v38_conformance")
BIB_FILES = tuple(
    PAPER / name
    for name in (
        "references_v28_candidate.bib",
        "references_v30_additions.bib",
        "references_v33_additions.bib",
        "references_v35_additions.bib",
        "references_v37_additions.bib",
    )
)
V38_GATE = ROOT / "reports/v38_manuscript_gate_20260726/manuscript_gate_v38.json"
V38_VERIFICATION = V38_GATE.with_name("manuscript_gate_verification_v38.json")
OUTPUT = ROOT / "reports/compiler_v39_format_gate_20260726"
DEFAULT_OUTPUT = OUTPUT / "format_gate.json"
DEFAULT_MARKDOWN = OUTPUT / "format_gate.md"

FROZEN = {
    V38_SOURCE: "fc5ffe114015ed5f5b8bb04a5c0fd32ef627dd045f49b400e4e1f1fe033558bb",
    V38_PDF: "53c3d13fcaa34736cb7067618881679e2d9941493879471d85cef8c52bcb8ce1",
    V38_GATE: "41fb3fc47bd87ec1ae0f9b311061af6390cf3ca4d9b89139b1c715abd56459f3",
    V38_VERIFICATION: "60ee9d60daa921d2b57c88cc22e93dd9264bfa3a12739fc1661a50d92654a064",
    SUPP_SOURCE: "78b529949bff4dd7864be1fd781426b9ea640997b98b5d48bc484555682f367e",
    SUPP_PDF: "7e703a446bb4f18c74ff5685ee87c257e497ab752c424ebc9e286bea284c9bca",
    CHECKLIST_SOURCE: "37d196507adc7107ccb5b8fa6f4d5d7a27efea3ebce4d4115f9d549db3d3041e",
    CHECKLIST_ANSWERS: "33443c9cab2b71907015292db9690aa6b29bb9cae6818f0dfcc88a2f39ee3beb",
    CHECKLIST_PDF: "00bb410d7f2526a3987e3ec24d8874da512c0e614472ae96af9e8ccdda9ce7ee",
}
STYLE_HASH = "391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623"
BST_HASH = "5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(args), check=False, capture_output=True)


def pdf_info(path: Path) -> dict[str, str]:
    process = run("pdfinfo", str(path))
    if process.returncode:
        return {}
    result: dict[str, str] = {}
    for line in process.stdout.decode("utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def pdf_text(path: Path, page: int | None = None) -> str:
    command = ["pdftotext"]
    if page is not None:
        command.extend(["-f", str(page), "-l", str(page)])
    command.extend([str(path), "-"])
    process = run(*command)
    return process.stdout.decode("utf-8", errors="replace") if not process.returncode else ""


def fonts_ok(path: Path) -> tuple[bool, int]:
    process = run("pdffonts", str(path))
    if process.returncode:
        return False, 0
    rows = [
        line
        for line in process.stdout.decode("utf-8", errors="replace").splitlines()[2:]
        if line.strip()
    ]
    patterns = [
        re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row, re.I)
        for row in rows
    ]
    return (
        bool(rows)
        and all(match is not None and match.group(1).lower() == "yes" for match in patterns)
        and not any("Type 3" in row for row in rows),
        len(rows),
    )


def object_surface(path: Path) -> dict[str, Any]:
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(str(path), strict=True)
        catalog = reader.trailer["/Root"]
        try:
            outlines = len(reader.outline)
        except AttributeError:
            outlines = len(reader.outlines)
        return {
            "available": True,
            "annotations": sum(bool(page.get("/Annots")) for page in reader.pages),
            "outlines": outlines,
            "open_action": "/OpenAction" in catalog,
            "acroform": "/AcroForm" in catalog,
            "embedded_files": "/EmbeddedFiles" in catalog.get("/Names", {}),
            "javascript": "/JavaScript" in catalog.get("/Names", {}),
            "all_letter": all(
                [float(value) for value in page.mediabox] == [0.0, 0.0, 612.0, 792.0]
                and [float(value) for value in page.cropbox] == [0.0, 0.0, 612.0, 792.0]
                and int(page.get("/Rotate", 0) or 0) == 0
                for page in reader.pages
            ),
        }
    except Exception as error:  # pragma: no cover
        return {"available": False, "error": repr(error)}


def expected_v39_source() -> str:
    text = V38_SOURCE.read_text(encoding="utf-8")
    text = text.replace(
        "% File: main_aaai27_v38_candidate.tex",
        "% File: main_aaai27_v39_candidate.tex",
    ).replace(
        "% V3.8 submission-surface consistency repair. All prior manuscripts are preserved.",
        "% V3.9 Author Kit format-only repair. All prior manuscripts are preserved.",
    )
    text = text.replace(r"\footnotesize", r"\small")
    for stem in TABLE_STEMS:
        table = (TABLE_DIR / f"{stem}.tex").read_text(encoding="utf-8").rstrip("\n")
        table = table.replace(r"\footnotesize", r"\small")
        text = text.replace(rf"\input{{tables_v38_candidate/{stem}}}", table)
    return text


def build_report() -> dict[str, Any]:
    required = (
        V38_SOURCE,
        V38_PDF,
        V39_SOURCE,
        V39_PDF,
        V39_LOG,
        V38_GATE,
        V38_VERIFICATION,
        SUPP_SOURCE,
        SUPP_PDF,
        CHECKLIST_SOURCE,
        CHECKLIST_ANSWERS,
        CHECKLIST_PDF,
        PAPER / "aaai2027.sty",
        PAPER / "aaai2027.bst",
        *BIB_FILES,
        *(TABLE_DIR / f"{stem}.tex" for stem in TABLE_STEMS),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required V3.9 gate inputs: " + ", ".join(missing))

    v38_gate = json.loads(V38_GATE.read_text(encoding="utf-8"))
    v38_verification = json.loads(V38_VERIFICATION.read_text(encoding="utf-8"))
    source = V39_SOURCE.read_text(encoding="utf-8")
    uncommented = "\n".join(line.split("%", 1)[0] for line in source.splitlines())
    setlengths = [
        list(row)
        for row in re.findall(
            r"\\setlength\s*\{([^}]+)\}\s*\{([^}]+)\}", uncommented
        )
    ]
    prohibited = re.findall(
        r"\\(?:addtolength|baselinestretch|balance|clearpage|columnsep|newpage|"
        r"nocopyright|pagebreak|pagestyle|tiny|footnotesize|scriptsize)\b|"
        r"\\(?:vspace|vskip)\s*\{\s*-|\\input\s*\{",
        uncommented,
        flags=re.I,
    )
    log = V39_LOG.read_text(encoding="utf-8", errors="replace")
    log_hits = [
        token
        for token in (
            "Overfull ",
            "undefined references",
            "undefined citations",
            "multiply defined",
            "LaTeX Error",
            "Fatal error",
            "Emergency stop",
        )
        if token.lower() in log.lower()
    ]
    info = pdf_info(V39_PDF)
    embedded, font_rows = fonts_ok(V39_PDF)
    objects = object_surface(V39_PDF)
    text_v38 = pdf_text(V38_PDF)
    text_v39 = pdf_text(V39_PDF)
    page6 = pdf_text(V39_PDF, 6)
    pages78 = pdf_text(V39_PDF, 7) + pdf_text(V39_PDF, 8)
    metadata_identity = {
        key: info.get(key, "")
        for key in ("Author", "Title", "Subject", "Keywords")
        if info.get(key, "")
    }
    text_identity_hits = [
        label
        for label, pattern in {
            "email": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            "windows_path": r"(?i)[A-Z]:\\Users\\",
            "workspace_username": r"(?i)\b" + "SO" + "GANG" + r"\b",
            "acknowledgments": r"(?im)^\s*Acknowledg(?:e)?ments?\s*$",
        }.items()
        if re.search(pattern, text_v39)
    ]

    checks = {
        "frozen_v38_and_inherited_surfaces_exact": all(
            path.is_file() and file_sha256(path) == expected
            for path, expected in FROZEN.items()
        ),
        "v38_gate_and_verification_pass": (
            v38_gate.get("status") == "PASS"
            and len(v38_gate.get("checks", {})) == 26
            and all(v38_gate.get("checks", {}).values())
            and v38_verification.get("status") == "PASS"
            and v38_verification.get("passed") == v38_verification.get("required") == 9
        ),
        "official_author_kit_identity": (
            file_sha256(PAPER / "aaai2027.sty") == STYLE_HASH
            and file_sha256(PAPER / "aaai2027.bst") == BST_HASH
            and "/TemplateVersion (2027.1)" in source
        ),
        "exact_allowlisted_v38_to_v39_transformation": source == expected_v39_source(),
        "rendered_scientific_text_byte_exact": text_v38 == text_v39,
        "single_source_and_minimum_nine_point_tables": not prohibited,
        "only_permitted_tabcolsep_setlength": (
            len(setlengths) == 5
            and all(command.strip() == r"\tabcolsep" for command, _ in setlengths)
        ),
        "compile_log_clean": not log_hits,
        "main_pdf_eight_letter_pages": (
            info.get("Pages") == "8"
            and info.get("Page size", "").startswith("612 x 792 pts")
            and info.get("Encrypted", "").startswith("no")
            and info.get("PDF version") == "1.5"
            and objects.get("available") is True
            and objects.get("all_letter") is True
        ),
        "fonts_embedded_and_no_type3": embedded and font_rows > 0,
        "no_active_pdf_objects": (
            objects.get("available") is True
            and objects.get("annotations") == 0
            and objects.get("outlines") == 0
            and objects.get("open_action") is False
            and objects.get("acroform") is False
            and objects.get("embedded_files") is False
            and objects.get("javascript") is False
        ),
        "anonymous_pdf_surface": (
            not metadata_identity
            and not text_identity_hits
            and "anonymous submission" in text_v39.lower()
        ),
        "content_and_disclosure_end_by_page_six": (
            "Limitations" in page6
            and "AI-tool disclosure" in page6
            and re.search(r"(?m)^\s*References\s*$", page6) is not None
            and not re.search(
                r"(?m)^\s*(?:Introduction|Method|Evaluation|Limitations|Conclusion|AI-tool disclosure)\s*$",
                pages78,
            )
        ),
        "bounded_claim_surface_retained": all(
            phrase in " ".join(text_v39.split())
            for phrase in (
                "complete-core registration judgment",
                "378 nonendpoint handler splices",
                "no new sampler",
                "no post-hoc claim-validation report",
                "Fixed-case evidence",
                "not a privacy proof",
            )
        ),
    }
    source_paths = (*required, Path(__file__).resolve())
    report: dict[str, Any] = {
        "schema_version": "unitdp.compiler_v39_format_gate.v1",
        "scope": (
            "exact V3.8-to-V3.9 Author-Kit formatting transformation, rendered-text "
            "preservation, current PDF/source surface, and inherited V3.8 evidence; "
            "not a new scientific result, privacy proof, novelty theorem, venue-policy "
            "authorization, release authorization, or external certification"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnostics": {
            "prohibited_source_hits": prohibited,
            "setlength_rows": setlengths,
            "log_hits": log_hits,
            "pdf_info": info,
            "font_rows": font_rows,
            "objects": objects,
            "metadata_identity": metadata_identity,
            "text_identity_hits": text_identity_hits,
        },
        "source_sha256": {
            (
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else "Documents/" + path.relative_to(ROOT.parent).as_posix()
            ): file_sha256(path)
            for path in source_paths
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    if DEFAULT_OUTPUT.exists() or DEFAULT_MARKDOWN.exists():
        raise FileExistsError("refusing to overwrite the V3.9 format gate")
    report = build_report()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    DEFAULT_MARKDOWN.write_text(
        "\n".join(
            (
                "# Compiler V3.9 Author-Kit Format Gate",
                "",
                f"Status: **{report['status']}**",
                f"Checks: {sum(report['checks'].values())}/{len(report['checks'])}",
                "",
                "- V3.8 scientific rendered text: byte-exact",
                "- Main tables: at least 9pt",
                "- Main source: single TeX file",
                "- Content/disclosure ends on page 6; pages 7--8 are references only",
                "",
                f"Payload SHA-256: `{report['payload_sha256']}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": sum(report["checks"].values()),
                "required": len(report["checks"]),
                "payload_sha256": report["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
