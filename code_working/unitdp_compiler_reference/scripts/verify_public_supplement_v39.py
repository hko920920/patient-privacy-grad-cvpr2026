#!/usr/bin/env python3
"""Verify the clean anonymous V3.9 code-and-data supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import verify_g8_external_handler_gate as g8  # noqa: E402
import verify_public_supplement_v35 as legacy_v35  # noqa: E402
from unitdp.compiler_v4 import (  # noqa: E402
    CompilerDispatchV4Error,
    validate_route_handler_v4,
)


MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v8"
MAIN_SOURCE = ROOT / "paper/main_aaai27_v39_candidate.tex"
SUPPLEMENT_SOURCE = ROOT / "paper/main_aaai27_supplement_v38.tex"
CHECKLIST_SOURCE = ROOT / "paper/main_aaai27_reproducibility_checklist_v38.tex"
CHECKLIST_ANSWERS = ROOT / "paper/main_aaai27_reproducibility_checklist_answers_v38.tex"
V39_FORMAT_GATE = (
    ROOT / "reports/compiler_v39_format_gate_20260726/format_gate.json"
)
V39_FORMAT_VERIFICATION = V39_FORMAT_GATE.with_name(
    "format_gate_verification.json"
)
EVIDENCE = (
    ROOT / "reports/g8_external_handler_evidence_v2_20260725/"
    "external_pld_handler_evidence_v36.json"
)
G8_GATE = (
    ROOT / "reports/g8_external_handler_gate_v2_20260725/"
    "g8_external_handler_gate_v2.json"
)
G8_COST = (
    ROOT / "reports/g8_external_handler_cost_v2_20260725/"
    "g8_external_handler_cost_v2.json"
)
G8_VERIFICATION = G8_GATE.with_name("g8_external_handler_verification_v2.json")
PUBLIC_EXECUTION = EVIDENCE.with_name("uci_external_pld_execution_public_v36.json")
EXECUTION_MANIFEST = EVIDENCE.with_name("uci_external_pld_execution_manifest_v36.json")
PRIVATE_EXECUTION = EVIDENCE.with_name("uci_external_pld_execution_private_v36.json")

EXPECTED_REPORT_HASHES = {
    EVIDENCE: "478b29b7df26aa99236a99818e0766e6553df3ed9165e67b8410db45f4df1b28",
    G8_GATE: "dfc9e242eae912bc5bc4a17328552d78b8c4395a7ed6134a86008eee02604d82",
    G8_COST: "90f64d7af50ace51a0cfda9c6e00d01941823f662f756a34a1c0cc4eadf55dab",
    G8_VERIFICATION: (
        "e7c92067e5dee0747bf8cf74502b6977803524671929665ad48a0be0cc29e11b"
    ),
    PUBLIC_EXECUTION: (
        "0c50a03ec26a1cacabad6acc274b9c7759946a65a406953abf1d6305ce20d989"
    ),
    EXECUTION_MANIFEST: (
        "4e4f10ecb6eea4edc3d3130f42cbdd4c51d5d235faee11f32e0bb1feb5f82dba"
    ),
    V39_FORMAT_GATE: (
        "1e39188cf2cfa37d71fcc8dc97423b7f984453adc1da880837cf456f4f799d3e"
    ),
    V39_FORMAT_VERIFICATION: (
        "51fe62433d0ae689c1ef9356cc9dcca898080fa9c9a8f0bcbbcc82ceb768cdd2"
    ),
}

FORBIDDEN_PATH_PARTS = {
    "_private",
    "__pycache__",
    ".git",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".pyc",
    ".synctex.gz",
    ".whl",
}
PROCESS_PATTERNS = (
    r"virtual[- ]review",
    r"reviewer[- ]score",
    r"weak[- ]accept",
    r"acceptance[- ]probability",
    r"acceptance[- ]optimization",
    r"acceptance[- ]prediction",
    r"anomaly[- ]target",
    r"paper[- ]outcome",
    r"outcome[- ]prediction",
    r"rejection[- ]risk",
    r"review[- ]optimization",
)
WINDOWS_HOME_MARKER = ("C:" + chr(92) + "Users" + chr(92)).lower().encode()
POSIX_WINDOWS_HOME_MARKER = ("C:" + "/" + "Users" + "/").lower().encode()
ACTIVE_PAPER_PATHS = {
    "paper/aaai2027.bst",
    "paper/aaai2027.sty",
    "paper/main_aaai27_v39_candidate.tex",
    "paper/main_aaai27_supplement_v38.tex",
    "paper/main_aaai27_reproducibility_checklist_v38.tex",
    "paper/main_aaai27_reproducibility_checklist_answers_v38.tex",
    "paper/references_v28_candidate.bib",
    "paper/references_v30_additions.bib",
    "paper/references_v33_additions.bib",
    "paper/references_v35_additions.bib",
    "paper/references_v37_additions.bib",
    "paper/tables_v38_candidate/v38_profiles.tex",
    "paper/tables_v38_candidate/v38_handler_ablation.tex",
    "paper/tables_v38_candidate/v38_conformance.tex",
}


class PublicSupplementV37Error(RuntimeError):
    """Raised when the V3.9 package is stale or crosses its public boundary."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicSupplementV37Error(f"Expected JSON object: {path}")
    return value


def verify_payload(value: dict[str, Any], label: str) -> str:
    payload = dict(value)
    reported = payload.pop("payload_sha256", None)
    if reported != canonical_sha256(payload):
        raise PublicSupplementV37Error(f"{label} payload mismatch")
    return str(reported)


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = load_object(path)
    verify_payload(manifest, "package manifest")
    if (
        manifest.get("schema_version") != "unitdp.public_code_data_package.v39"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("visibility") != "public"
        or manifest.get("routes") != ["P", "F", "A"]
        or manifest.get("registered_handler_endpoints")
        != ["P", "F", "A", "H_external_pld_overlay_on_A"]
        or manifest.get("private_randomizers_or_step_traces_redistributed") is not False
        or manifest.get("third_party_wheels_or_backend_tree_redistributed") is not False
    ):
        raise PublicSupplementV37Error("Package manifest metadata mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PublicSupplementV37Error("Manifest files must be a list")
    expected_paths: list[str] = []
    expected_total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise PublicSupplementV37Error("Malformed manifest row")
        relative = str(row["path"])
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicSupplementV37Error(
                f"Manifest path escapes package: {relative}"
            ) from exc
        if (
            "\\" in relative
            or ".." in Path(relative).parts
            or not candidate.is_file()
            or candidate.stat().st_size != int(row["bytes"])
            or file_sha256(candidate) != row["sha256"]
        ):
            raise PublicSupplementV37Error(f"Manifest binding mismatch: {relative}")
        expected_paths.append(relative)
        expected_total += int(row["bytes"])
    observed_paths = sorted(
        item.relative_to(ROOT).as_posix()
        for item in ROOT.rglob("*")
        if (
            item.is_file()
            and item.name != MANIFEST_NAME
            and "__pycache__" not in item.parts
            and ".pytest_cache" not in item.parts
            and item.suffix.lower() != ".pyc"
        )
    )
    if (
        expected_paths != sorted(expected_paths)
        or len(expected_paths) != len(set(expected_paths))
        or expected_paths != observed_paths
        or manifest.get("file_count") != len(expected_paths)
        or manifest.get("total_bytes") != expected_total
    ):
        raise PublicSupplementV37Error("Manifest allowlist/count mismatch")
    paper_paths = {path for path in expected_paths if path.startswith("paper/")}
    if paper_paths != ACTIVE_PAPER_PATHS or any(
        path.startswith("dist/") for path in expected_paths
    ):
        raise PublicSupplementV37Error(
            "Active-paper-only or no-nested-submission boundary mismatch"
        )
    return manifest


def verify_disclosure_boundary() -> dict[str, int]:
    identity_hits = 0
    process_hits = 0
    paper_independence_hits = 0
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_PATH_PARTS:
            raise PublicSupplementV37Error(f"Forbidden path: {relative.as_posix()}")
        if any(path.name.lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise PublicSupplementV37Error(
                f"Forbidden transient/backend file: {relative.as_posix()}"
            )
        raw = path.read_bytes()
        lowered = raw.lower()
        if (
            WINDOWS_HOME_MARKER in lowered
            or POSIX_WINDOWS_HOME_MARKER in lowered
            or ("SO" + "GANG").lower().encode() in lowered
        ):
            identity_hits += 1
        if path.suffix.lower() in {
            ".bib",
            ".json",
            ".md",
            ".tex",
            ".txt",
            ".yaml",
            ".yml",
        }:
            text = raw.decode("utf-8", errors="strict")
            process_hits += sum(
                bool(re.search(pattern, text, flags=re.IGNORECASE))
                for pattern in PROCESS_PATTERNS
            )
        if relative in (
            Path("paper/main_aaai27_v39_candidate.tex"),
            Path("paper/main_aaai27_supplement_v38.tex"),
            Path("paper/main_aaai27_reproducibility_checklist_v38.tex"),
            Path("paper/main_aaai27_reproducibility_checklist_answers_v38.tex"),
            Path("paper/tables_v38_candidate/v38_conformance.tex"),
        ):
            text = raw.decode("utf-8", errors="strict")
            paper_independence_hits += sum(
                bool(re.search(pattern, text, flags=re.IGNORECASE))
                for pattern in (
                    r"independent verifier",
                    r"independently replay",
                    r"independent correspondence",
                    r"independent validation",
                )
            )
    if identity_hits or process_hits or paper_independence_hits:
        raise PublicSupplementV37Error(
            "Identity, process, or ambiguous validation disclosure detected"
        )
    if PRIVATE_EXECUTION.exists():
        raise PublicSupplementV37Error("Private G8 execution trace was packaged")
    return {
        "files_scanned": len(files),
        "identity_hits": identity_hits,
        "archive_process_hits": process_hits,
        "active_paper_ambiguous_independence_hits": paper_independence_hits,
        "historical_submission_paths": sum(
            path.relative_to(ROOT).as_posix().startswith("dist/")
            or bool(
                re.search(
                    r"paper/main_aaai27_v3[5678]_candidate|"
                    r"paper/main_aaai27_supplement_v3[567]",
                    path.relative_to(ROOT).as_posix(),
                )
            )
            for path in files
        ),
        "third_party_wheels": sum(path.suffix == ".whl" for path in files),
    }


def resolve_report_path(relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if normalized.startswith("../AAAI27_7p_compressed/02_algorithm_compiler_paper/"):
        return ROOT / "paper" / Path(normalized).name
    return ROOT / normalized


def verify_source_map(rows: dict[str, Any], label: str) -> bool:
    return bool(rows) and all(
        resolve_report_path(relative).is_file()
        and file_sha256(resolve_report_path(relative)) == expected
        for relative, expected in rows.items()
    )


def verify_frozen_files(rows: list[dict[str, Any]]) -> bool:
    expected_anchors = {
        "../AAAI27_7p_compressed/02_algorithm_compiler_paper/"
        "main_aaai27_v35_candidate.tex",
        "../AAAI27_7p_compressed/02_algorithm_compiler_paper/"
        "main_aaai27_supplement_v35.tex",
        "src/unitdp/compiler_v4.py",
        "src/unitdp/compiler_allocation_v4.py",
        "src/unitdp/random_allocation_accountant_v4.py",
        "src/unitdp/owner_random_allocation_v4.py",
        "dist/compiler_main_aaai27_v4.pdf",
        "dist/compiler_supplementary_document_aaai27_v4.pdf",
        "dist/compiler_reproducibility_checklist_aaai27_v4.pdf",
        "dist/compiler_code_and_data_supplement_aaai27_v4.zip",
        "dist/compiler_submission_upload_manifest_v4.json",
        "reports/v35_handler_registry_gate_20260725/handler_registry_gate_v35.json",
    }
    observed_anchors = {str(row.get("path", "")).replace("\\", "/") for row in rows}
    return observed_anchors == expected_anchors and all(
        row.get("pass") is True
        and row.get("expected_sha256") == row.get("observed_sha256")
        and isinstance(row.get("expected_sha256"), str)
        and len(row["expected_sha256"]) == 64
        for row in rows
    )


def verify_public_execution(evidence: dict[str, Any]) -> dict[str, Any]:
    embedded = evidence["execution"]["manifest"]
    verify_payload(embedded, "embedded execution manifest")
    standalone = load_object(EXECUTION_MANIFEST)
    verify_payload(standalone, "standalone execution manifest")
    if embedded != standalone:
        raise PublicSupplementV37Error("Execution manifest copies differ")
    public_row = embedded["files"]["public"]
    private_row = embedded["files"]["private"]
    if (
        file_sha256(PUBLIC_EXECUTION) != public_row["file_sha256"]
        or file_sha256(EXECUTION_MANIFEST)
        != evidence["execution"]["manifest_file_sha256"]
        or private_row.get("handling") != "do_not_publish"
        or (EVIDENCE.parent / private_row["path"]).exists()
    ):
        raise PublicSupplementV37Error("G8 public/private execution boundary mismatch")
    return {
        "public_file_sha256": file_sha256(PUBLIC_EXECUTION),
        "private_file_redistributed": False,
        "private_file_committed_by_hash_only": True,
    }


def verify_g8() -> dict[str, Any]:
    if not all(
        path.is_file() and file_sha256(path) == expected
        for path, expected in EXPECTED_REPORT_HASHES.items()
    ):
        raise PublicSupplementV37Error("Frozen active report hash mismatch")
    evidence = load_object(EVIDENCE)
    gate = load_object(G8_GATE)
    cost = load_object(G8_COST)
    verification = load_object(G8_VERIFICATION)
    for label, value in (
        ("G8 evidence", evidence),
        ("G8 gate", gate),
        ("G8 cost", cost),
        ("G8 verification", verification),
    ):
        verify_payload(value, label)
        if value.get("status") != "PASS":
            raise PublicSupplementV37Error(f"{label} is not PASS")

    endpoint_valid = True
    for _, handler in g8.HANDLERS:
        try:
            validate_route_handler_v4(handler)
        except CompilerDispatchV4Error:
            endpoint_valid = False
    structural = {
        "four_endpoints_validate": endpoint_valid,
        "four_registry_entries": len(g8.ROUTE_DISPATCH_REGISTRY_V36) == 4,
        "registration_core_exact": (
            g8.EXTERNAL_PLD_HANDLER_V36.registration_core_sha256
            == evidence["registration_core_sha256"]
        ),
        "evidence_claim_sources": verify_source_map(
            evidence["claim_source_sha256"], "evidence"
        ),
        "gate_claim_sources": verify_source_map(gate["claim_source_sha256"], "gate"),
        "source_bundles": g8.verify_evidence_source_bundles(evidence),
        "frozen_files": verify_frozen_files(gate["frozen_files"]),
        "nine_old_outputs": g8.verify_preservation(gate),
        "seven_witness_deletions": g8.verify_witness_deletion(gate),
        "hybrid_lattice_756": g8.verify_hybrid_lattice(gate),
        "cost_source_rows": g8.verify_cost_sources(cost),
    }
    if not all(structural.values()):
        failed = [key for key, value in structural.items() if not value]
        raise PublicSupplementV37Error(
            "G8 structural verification failed: " + ", ".join(failed)
        )
    public_execution = verify_public_execution(evidence)
    input_hashes = verification.get("input_sha256", {})
    if (
        not (verification.get("passed") == verification.get("required") == 27)
        or input_hashes.get(
            "reports/g8_external_handler_evidence_v2_20260725/"
            "external_pld_handler_evidence_v36.json"
        )
        != file_sha256(EVIDENCE)
        or input_hashes.get(
            "reports/g8_external_handler_gate_v2_20260725/"
            "g8_external_handler_gate_v2.json"
        )
        != file_sha256(G8_GATE)
        or input_hashes.get(
            "reports/g8_external_handler_cost_v2_20260725/"
            "g8_external_handler_cost_v2.json"
        )
        != file_sha256(G8_COST)
    ):
        raise PublicSupplementV37Error("Separate G8 verification binding mismatch")
    if not (
        gate["preservation"]["exact"] == 9
        and gate["handler_hybrids"]["pair_count"] == 6
        and gate["handler_hybrids"]["unique"] == 756
        and gate["handler_hybrids"]["sealed_rejected"] == 756
        and gate["witness_deletion"]["rejected"] == 7
        and gate["tamper_rejections"]["total"] == 10
        and cost["source_cost"]["new_production_files"] == 6
        and cost["source_cost"]["new_production_logical_lines"] == 1728
        and cost["environment_cost"]["backend_site_bytes"] == 110_964_413
    ):
        raise PublicSupplementV37Error("G8 reported counts mismatch")
    return {
        **structural,
        **public_execution,
        "separate_checks": verification["passed"],
        "new_production_files": cost["source_cost"]["new_production_files"],
        "new_production_logical_lines": cost["source_cost"][
            "new_production_logical_lines"
        ],
        "backend_site_redistributed": False,
    }


def source_hash_for_suffix(rows: dict[str, Any], suffix: str) -> str | None:
    matches = [
        str(value)
        for key, value in rows.items()
        if key.replace("\\", "/").endswith(suffix)
    ]
    return matches[0] if len(matches) == 1 else None


def verify_v35_science_without_nested_submissions() -> dict[str, Any]:
    """Verify scientific reports while treating old upload bundles as anchors."""
    reports = {
        "g1": legacy_v35.load_object(legacy_v35.G1),
        "g2": legacy_v35.load_object(legacy_v35.G2),
        "g3": legacy_v35.load_object(legacy_v35.G3),
        "g4": legacy_v35.load_object(legacy_v35.G4),
        "g5": legacy_v35.load_object(legacy_v35.G5),
    }
    for label, value in reports.items():
        legacy_v35.verify_embedded_payload(
            value,
            "payload_sha256",
            f"{label.upper()} scientific report",
        )
        if value.get("status") != "PASS":
            raise PublicSupplementV37Error(
                f"{label.upper()} scientific report is not PASS"
            )
    for label in ("g1", "g3", "g4", "g5"):
        value = reports[label]
        source_map = value.get("source_sha256") or value.get("claim_source_sha256")
        legacy_v35.verify_source_map(source_map, label=label.upper())

    g1 = reports["g1"]
    g2 = reports["g2"]
    g3 = reports["g3"]
    g4 = reports["g4"]
    g5 = reports["g5"]
    g6 = legacy_v35.verify_g6_gate(
        legacy_v35.G6_ROOT,
        include_private=False,
    )
    g7 = legacy_v35.verify_g7_report(legacy_v35.G7)
    checks = {
        "g1_three_profiles": (
            g1.get("case_count") == 3 and len(g1.get("cases", [])) == 3
        ),
        "g2_contract_snapshot": (
            g2.get("status") == "PASS" and all(g2.get("checks", {}).values())
        ),
        "g3_executor_correspondence": (
            all(g3.get("checks", {}).values())
            and all(g3.get("correspondence", {}).values())
            and all(g3.get("execution_invariants", {}).values())
        ),
        "g4_registry_and_preservation_counts": (
            len(g4.get("registry_rows", [])) == 3
            and len(g4.get("required_registration_obligations", [])) == 7
            and len(g4.get("compilation_rows", [])) == 9
            and len(g4.get("conservative_extension_rows", [])) == 6
            and all(
                row.get("pass") is True
                for row in g4.get("conservative_extension_rows", [])
            )
        ),
        "g5_contract_lattice_counts": (
            g5.get("summary", {}).get("required_nonendpoint_hybrids") == 2286
            and g5.get("summary", {}).get("rejected_nonendpoint_hybrids") == 2286
            and g5.get("summary", {}).get("accepted_nonendpoint_hybrids") == 0
        ),
        "g6_public_execution_counts": (
            g6.get("public_counts")
            == {
                "datasets": 3,
                "runs": 15,
                "models": 15,
                "bundles": 15,
                "summaries": 3,
            }
            and g6.get("private_counts") is None
        ),
        "g7_handler_lattice_counts": (
            g7.get("hybrids_rebuilt") == 378 and g7.get("sealed_rejections") == 378
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PublicSupplementV37Error(
            "Clean scientific report verification failed: " + ", ".join(failed)
        )
    return {
        "status": "verified_without_nested_historical_submission",
        "checks": checks,
        "passed": sum(checks.values()),
        "required": len(checks),
        "old_pf_outputs_preserved": 6,
        "contract_nonendpoints_rejected": 2286,
        "handler_nonendpoints_rebuilt": 378,
        "handler_nonendpoints_sealed_rejected": 378,
        "public_allocation_runs": 15,
    }


def verify_v39_manuscript() -> dict[str, Any]:
    gate = load_object(V39_FORMAT_GATE)
    payload = verify_payload(gate, "V3.9 format gate")
    verification = load_object(V39_FORMAT_VERIFICATION)
    if (
        gate.get("status") != "PASS"
        or len(gate.get("checks", {})) != 14
        or not all(gate.get("checks", {}).values())
        or verification.get("status") != "PASS"
        or not (verification.get("passed") == verification.get("required") == 6)
    ):
        raise PublicSupplementV37Error("V3.9 format gate mismatch")
    source_hashes = gate.get("source_sha256", {})
    expected = {
        MAIN_SOURCE: "main_aaai27_v39_candidate.tex",
        SUPPLEMENT_SOURCE: "main_aaai27_supplement_v38.tex",
        CHECKLIST_SOURCE: "main_aaai27_reproducibility_checklist_v38.tex",
        CHECKLIST_ANSWERS: ("main_aaai27_reproducibility_checklist_answers_v38.tex"),
        ROOT / "paper/aaai2027.sty": "aaai2027.sty",
        ROOT / "paper/aaai2027.bst": "aaai2027.bst",
        ROOT / "paper/references_v37_additions.bib": ("references_v37_additions.bib"),
        ROOT
        / "paper/tables_v38_candidate/v38_profiles.tex": (
            "tables_v38_candidate/v38_profiles.tex"
        ),
        ROOT
        / "paper/tables_v38_candidate/v38_handler_ablation.tex": (
            "tables_v38_candidate/v38_handler_ablation.tex"
        ),
        ROOT
        / "paper/tables_v38_candidate/v38_conformance.tex": (
            "tables_v38_candidate/v38_conformance.tex"
        ),
    }
    if not all(
        source_hash_for_suffix(source_hashes, suffix) == file_sha256(path)
        for path, suffix in expected.items()
    ):
        raise PublicSupplementV37Error("Packaged V3.9 manuscript source mismatch")
    main = MAIN_SOURCE.read_text(encoding="utf-8")
    supplement = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    checklist = CHECKLIST_ANSWERS.read_text(encoding="utf-8")
    conformance = (ROOT / "paper/tables_v38_candidate/v38_conformance.tex").read_text(
        encoding="utf-8"
    )
    main_flat = re.sub(r"\s+", " ", main)
    supplement_flat = re.sub(r"\s+", " ", supplement)
    if (
        not all(
            fragment in main_flat
            for fragment in (
                "Evidence-Sealed Registration of Owner-Sampled DP-SGD Routes",
                "production-import-free",
                "Sampler/accountant mismatches can materially weaken implementations",
                "\\citep{chua2024private}",
                "We claim none of these results",
                "handlers, binary partitions, and verifiers are author-constructed",
                "redistributes neither the 110,964,413-byte backend nor its wheels",
            )
        )
        or not all(
            fragment in supplement_flat
            for fragment in (
                "separately implemented verifier that does not import the report builder",
                "111 MB environment",
                "does not redistribute or re-execute that numerical environment",
            )
        )
        or not all(
            fragment in checklist
            for fragment in (
                "computing infrastructure used for running experiments",
                "formally describes evaluation metrics",
                "Analysis of experiments goes beyond single-dimensional",
                "\tpartial",
                "\tno",
            )
        )
        or not (
            "Execution and reference correspondence" in conformance
            and "Execution and independent correspondence" not in conformance
        )
    ):
        raise PublicSupplementV37Error("V3.9 bounded claim text mismatch")
    return {
        "status": "verified_format_repair_and_frozen_science_gate",
        "payload_sha256": payload,
        "aggregate_checks": len(gate["checks"]),
        "main_source_sha256": file_sha256(MAIN_SOURCE),
        "supplement_source_sha256": file_sha256(SUPPLEMENT_SOURCE),
        "checklist_source_sha256": file_sha256(CHECKLIST_SOURCE),
        "checklist_answers_sha256": file_sha256(CHECKLIST_ANSWERS),
    }


def verify_all(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path)
    disclosure = verify_disclosure_boundary()
    legacy_pf = legacy_v35.verify_legacy_pf()
    v35_signal = verify_v35_science_without_nested_submissions()
    v36_signal = verify_g8()
    v39_manuscript = verify_v39_manuscript()
    result: dict[str, Any] = {
        "schema_version": "unitdp.public_code_data_package_verification.v39",
        "status": "verified",
        "package_manifest_payload_sha256": manifest["payload_sha256"],
        "package_file_count": manifest["file_count"],
        "disclosure_scan": disclosure,
        "legacy_pf": legacy_pf,
        "v35_signal_gates": v35_signal,
        "v36_signal_gates": v36_signal,
        "v39_manuscript_binding": v39_manuscript,
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / MANIFEST_NAME),
    )
    args = parser.parse_args()
    result = verify_all(Path(args.manifest))
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
