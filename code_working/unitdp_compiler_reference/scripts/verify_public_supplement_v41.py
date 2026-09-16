#!/usr/bin/env python3
"""Verify the anonymous V4.3-matched Compiler V10 code/data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


# This must precede every package-local import.  The verifier is required to be
# safe when invoked exactly as documented from a clean extraction.
sys.dont_write_bytecode = True

try:
    from scripts import verify_public_supplement_v39 as core
except (ImportError, ModuleNotFoundError):
    import verify_public_supplement_v39 as core


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v10"

MAIN_SOURCE = ROOT / "paper/main_aaai27_v43_candidate.tex"
SUPPLEMENT_SOURCE = ROOT / "paper/main_aaai27_supplement_v41.tex"
CHECKLIST_SOURCE = ROOT / "paper/main_aaai27_reproducibility_checklist_v40.tex"
CHECKLIST_ANSWERS = (
    ROOT / "paper/main_aaai27_reproducibility_checklist_answers_v40.tex"
)
FIGURE7 = ROOT / "paper/figures/figure7.png"
REGRESSION_SUMMARY = (
    ROOT
    / "reports/v42_public_regression_summary_20260728/"
    "public_regression_summary_v42.json"
)

EXPECTED_PAPER_HASHES = {
    "aaai2027.bst": "5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c",
    "aaai2027.sty": "391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623",
    "main_aaai27_v43_candidate.tex": "8a621473ceeaee9a8b365607b49282d6837aa37605a07be81403541de2764bd9",
    "main_aaai27_supplement_v41.tex": "b0db21a8cdbf74723a849c04d5cdd4e3f2f05d00764296c33fbb3029fab879ad",
    "main_aaai27_reproducibility_checklist_v40.tex": "5bd1d615400587a272525a295c0667638f93331041e40e6eae692e14ac1962f8",
    "main_aaai27_reproducibility_checklist_answers_v40.tex": "33443c9cab2b71907015292db9690aa6b29bb9cae6818f0dfcc88a2f39ee3beb",
    "references_v28_candidate.bib": "3b0679066b0cfd9bbc3ac26afaa733bc447ce331cf6419f7f3c9c162cf3083fd",
    "references_v30_additions.bib": "f9a585f466149fda9ffc243c9c0e68be4fd2a50fea2d5e95f2942091b8836ffe",
    "references_v33_additions.bib": "703272954fa948c30392fd5d82c2bbf4e2855ad48b4f18263bf8f0d4e9635f66",
    "references_v35_additions.bib": "74dbcc5c2ae798bcb84663b1f97e7fcbcaa498ee89149e521bb497aa422169e7",
    "references_v37_additions.bib": "d68578e96e75c8709b33506d2488c6c433e8922253468bbec29151de9c623252",
    "tables_v38_candidate/v38_profiles.tex": "bc637efbf5c2de124b898cc34195f1a33e1367211ab303e0acfe136b82166773",
    "figures/figure7.png": "9c1cf1b573607d2140d7ae2e3788a9d603b48baac279f0cb09fb361accea471c",
}

ACTIVE_PAPER_PATHS = {f"paper/{name}" for name in EXPECTED_PAPER_HASHES}
EXPECTED_REGRESSION_FILE_SHA256 = (
    "59846e98cb370e399b537f48af99573e86137db5cc9ba8f6665e5ab6875bcac3"
)
EXPECTED_REGRESSION_PAYLOAD_SHA256 = (
    "f3299473ff77ff12dcd3d5ea0bdb2a3486d3c190ca4ed7aecf9b38db956fd1c3"
)


class PublicSupplementV41Error(RuntimeError):
    """Raised when the active package is stale or crosses its public boundary."""


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = core.load_object(path)
    core.verify_payload(manifest, "package manifest")
    if (
        manifest.get("schema_version") != "unitdp.public_code_data_package.v43"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("visibility") != "public"
        or manifest.get("release_status") != "research_non_release"
        or manifest.get("routes") != ["P", "F", "A"]
        or manifest.get("registered_handler_endpoints")
        != ["P", "F", "A", "H_external_pld_overlay_on_A"]
        or manifest.get("raw_third_party_data_redistributed") is not False
        or manifest.get("private_randomizers_or_step_traces_redistributed") is not False
        or manifest.get("third_party_wheels_or_backend_tree_redistributed") is not False
    ):
        raise PublicSupplementV41Error("package manifest metadata mismatch")

    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PublicSupplementV41Error("manifest files must be a list")
    expected_paths: list[str] = []
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PublicSupplementV41Error("malformed manifest row")
        relative = str(row["path"])
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicSupplementV41Error(
                f"manifest path escapes package: {relative}"
            ) from exc
        if (
            "\\" in relative
            or ".." in Path(relative).parts
            or not candidate.is_file()
            or candidate.stat().st_size != int(row["bytes"])
            or core.file_sha256(candidate) != row["sha256"]
        ):
            raise PublicSupplementV41Error(
                f"manifest binding mismatch: {relative}"
            )
        expected_paths.append(relative)
        total_bytes += int(row["bytes"])

    observed_paths = sorted(
        item.relative_to(ROOT).as_posix()
        for item in ROOT.rglob("*")
        if item.is_file() and item.name != MANIFEST_NAME
    )
    if (
        expected_paths != sorted(expected_paths)
        or len(expected_paths) != len(set(expected_paths))
        or expected_paths != observed_paths
        or manifest.get("file_count") != len(expected_paths)
        or manifest.get("total_bytes") != total_bytes
    ):
        raise PublicSupplementV41Error("manifest allowlist/count mismatch")
    paper_paths = {name for name in expected_paths if name.startswith("paper/")}
    if paper_paths != ACTIVE_PAPER_PATHS or any(
        name.startswith("dist/") for name in expected_paths
    ):
        raise PublicSupplementV41Error("active-paper-only boundary mismatch")

    active = manifest.get("active_artifacts", {})
    expected_active = {
        "main_source_sha256": EXPECTED_PAPER_HASHES["main_aaai27_v43_candidate.tex"],
        "supplement_source_sha256": EXPECTED_PAPER_HASHES[
            "main_aaai27_supplement_v41.tex"
        ],
        "checklist_wrapper_sha256": EXPECTED_PAPER_HASHES[
            "main_aaai27_reproducibility_checklist_v40.tex"
        ],
        "checklist_answers_sha256": EXPECTED_PAPER_HASHES[
            "main_aaai27_reproducibility_checklist_answers_v40.tex"
        ],
        "figure7_sha256": EXPECTED_PAPER_HASHES["figures/figure7.png"],
        "public_regression_summary_sha256": EXPECTED_REGRESSION_FILE_SHA256,
    }
    if not all(active.get(key) == value for key, value in expected_active.items()):
        raise PublicSupplementV41Error("active artifact binding mismatch")
    return manifest


def verify_regression_summary() -> dict[str, Any]:
    if core.file_sha256(REGRESSION_SUMMARY) != EXPECTED_REGRESSION_FILE_SHA256:
        raise PublicSupplementV41Error("public regression summary drifted")
    value = core.load_object(REGRESSION_SUMMARY)
    payload = core.verify_payload(value, "public regression summary")
    if (
        payload != EXPECTED_REGRESSION_PAYLOAD_SHA256
        or value.get("status") != "PASS"
        or value.get("counts")
        != {"collected": 161, "passed": 154, "skipped": 7, "warnings": 2}
        or value.get("source_binding", {}).get("stdout_sha256")
        != "1eb680358fd34177764b4d7ac9508c4776d58de64f5dc8a6c351bd098aabdc53"
        or value.get("source_binding", {}).get("raw_transcripts_redistributed")
        is not False
    ):
        raise PublicSupplementV41Error("public regression summary mismatch")
    if any(
        path.name in {"full_pytest_stdout_v35.txt", "full_pytest_stderr_v35.txt"}
        for path in ROOT.rglob("*")
        if path.is_file()
    ):
        raise PublicSupplementV41Error("local raw regression transcript was packaged")
    return {
        "status": "verified",
        "payload_sha256": payload,
        "collected": 161,
        "passed": 154,
        "skipped": 7,
        "raw_transcripts_redistributed": False,
    }


def verify_active_manuscript() -> dict[str, Any]:
    observed = {
        name: core.file_sha256(ROOT / "paper" / name)
        for name in EXPECTED_PAPER_HASHES
    }
    if observed != EXPECTED_PAPER_HASHES:
        raise PublicSupplementV41Error("active paper source hash mismatch")

    main = MAIN_SOURCE.read_text(encoding="utf-8")
    supplement = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    checklist = CHECKLIST_SOURCE.read_text(encoding="utf-8")
    main_flat = re.sub(r"\s+", " ", main)
    supplement_flat = re.sub(r"\s+", " ", supplement)
    required_main = (
        "Evidence-Sealed Registration of Owner-Sampled DP-SGD Routes",
        "route-label registration consistency",
        "Contract-only rejects 36/52 and 5/8",
        "All 48/48 nonvacuous substitutions fail",
        "45/45 fixed-seed integration grid",
        "not an outcome distribution",
        "full frozen regression collects 161 tests",
        "concurrent study",
        "neither pipeline consumes the other's outputs",
    )
    required_supplement = (
        "route-local registry $r_h$",
        "does not mean an external team or independent human reproduction",
        "same 7,352/2,947 rows, 561 features, 21/9 subject split",
        "Neither pipeline consumes the other's outputs",
        "154/154 run; 7/161 skip",
        "173/173 run; 7/180 skip",
        "30/30; 850 steps; 60 tensors",
        "Its trust boundary is the completeness and correctness of those declarations and of the report builder",
        "Signed provenance, independently reproduced builds",
    )
    if not all(fragment in main_flat for fragment in required_main):
        raise PublicSupplementV41Error("active main claim binding mismatch")
    if not all(fragment in supplement_flat for fragment in required_supplement):
        raise PublicSupplementV41Error("active supplement claim binding mismatch")
    if (
        checklist.count(
            r"\input{main_aaai27_reproducibility_checklist_answers_v40.tex}"
        )
        != 1
        or r"\includegraphics[width=0.9\textwidth]{figures/figure7.png}"
        not in main
        or r"\input{tables_v38_candidate/v38_profiles}" not in supplement
    ):
        raise PublicSupplementV41Error("active source dependency mismatch")

    ambiguous = re.compile(
        r"independent\s+(?:verification|validation|third[- ]party|replication)",
        flags=re.IGNORECASE,
    )
    if ambiguous.search(main + "\n" + supplement):
        raise PublicSupplementV41Error("ambiguous external-independence wording")
    return {
        "status": "verified_active_v43_v41_v40_sources",
        "main_source_sha256": observed["main_aaai27_v43_candidate.tex"],
        "supplement_source_sha256": observed["main_aaai27_supplement_v41.tex"],
        "checklist_answers_sha256": observed[
            "main_aaai27_reproducibility_checklist_answers_v40.tex"
        ],
        "figure7_sha256": observed["figures/figure7.png"],
    }


def verify_readme() -> dict[str, Any]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "matched to the active V4.3",
        "python scripts/verify_public_supplement_v41.py",
        "python -B -m pytest -q -p no:cacheprovider",
        "build_compiler_code_data_package_v10.py",
        "161 collected",
        "154 passed",
        "seven raw-data-scoped skips",
    )
    if not all(fragment in text for fragment in required):
        raise PublicSupplementV41Error("README active-version binding mismatch")
    return {"status": "verified", "required_fragments": len(required)}


def verify_all(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path)
    disclosure = core.verify_disclosure_boundary()
    legacy_pf = core.legacy_v35.verify_legacy_pf()
    v35_signal = core.verify_v35_science_without_nested_submissions()
    v36_signal = core.verify_g8()
    active = verify_active_manuscript()
    regression = verify_regression_summary()
    readme = verify_readme()
    result: dict[str, Any] = {
        "schema_version": "unitdp.public_code_data_package_verification.v43",
        "status": "verified",
        "package_manifest_payload_sha256": manifest["payload_sha256"],
        "package_file_count": manifest["file_count"],
        "disclosure_scan": disclosure,
        "legacy_pf": legacy_pf,
        "v35_signal_gates": v35_signal,
        "v36_signal_gates": v36_signal,
        "active_manuscript_binding": active,
        "public_regression_binding": regression,
        "readme_binding": readme,
    }
    result["payload_sha256"] = core.canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / MANIFEST_NAME))
    args = parser.parse_args()
    result = verify_all(Path(args.manifest))
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
