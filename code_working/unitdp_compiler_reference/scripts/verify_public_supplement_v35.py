#!/usr/bin/env python3
"""Verify the anonymous, redistributable V3.5 code-and-data supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import verify_public_supplement_v34 as legacy  # noqa: E402
from build_v35_handler_registry_gate import (  # noqa: E402
    build_report as build_g4_report,
)
from build_v35_pairwise_hybrid_lattice_gate import (  # noqa: E402
    build_report as build_g5_report,
)
from build_v35_random_allocation_accountant_gate import (  # noqa: E402
    build_report as build_g1_report,
)
from verify_v35_allocation_multirun_gate import (  # noqa: E402
    verify_gate as verify_g6_gate,
)
from verify_v35_handler_hybrid_ablation_gate import (  # noqa: E402
    verify_report as verify_g7_report,
)


MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v4"
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_HOME_PREFIX = "C:" + chr(92) + "Users" + chr(92)
POSIX_WINDOWS_HOME = "C:" + "/" + "Users" + "/"
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
}
PROCESS_MARKERS = (
    "virtual review",
    "reviewer score",
    "acceptance probability",
    "acceptance optimization",
    "weak accept",
    "weak reject",
    "maximize review",
    "anomaly point",
    "paper outcome",
    "synthetic review",
    "review trajectory",
    "neurips review",
)

G1 = (
    ROOT
    / "reports"
    / "v35_random_allocation_accountant_gate_20260725"
    / "random_allocation_accountant_gate_v35.json"
)
G2 = (
    ROOT
    / "reports"
    / "v35_allocation_contract_gate_20260725"
    / "allocation_contract_gate_v35.json"
)
G3 = (
    ROOT
    / "reports"
    / "v35_allocation_executor_gate_20260725"
    / "allocation_executor_gate_v35.json"
)
G4 = (
    ROOT
    / "reports"
    / "v35_handler_registry_gate_v2_20260725"
    / "handler_registry_gate_v35_v2.json"
)
G5 = (
    ROOT
    / "reports"
    / "v35_pairwise_hybrid_lattice_gate_v2_20260725"
    / "pairwise_hybrid_lattice_gate_v35_v2.json"
)
G6_ROOT = ROOT / "reports" / "v35_allocation_multirun_gate_20260725"
G7 = (
    ROOT
    / "reports"
    / "v35_handler_hybrid_ablation_gate_20260725"
    / "handler_hybrid_ablation_gate_v35.json"
)
MANUSCRIPT_GATE = (
    ROOT
    / "reports"
    / "v35_manuscript_gate_20260725"
    / "manuscript_gate_v35.json"
)
MAIN_SOURCE = ROOT / "paper" / "main_aaai27_v35_candidate.tex"
SUPPLEMENT_SOURCE = ROOT / "paper" / "main_aaai27_supplement_v35.tex"


class PublicSupplementV35Error(RuntimeError):
    """Raised when the V3.5 public supplement is stale or disclosive."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicSupplementV35Error(f"Could not load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicSupplementV35Error(f"Expected JSON object: {path}")
    return value


def verify_embedded_payload(
    value: dict[str, Any],
    field: str,
    label: str,
) -> str:
    candidate = dict(value)
    reported = candidate.pop(field, None)
    if not isinstance(reported, str) or canonical_sha256(candidate) != reported:
        raise PublicSupplementV35Error(f"{label} payload mismatch")
    return reported


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = load_object(path)
    verify_embedded_payload(manifest, "payload_sha256", "package manifest")
    if (
        manifest.get("schema_version")
        != "unitdp.public_code_data_package.v35"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("visibility") != "public"
        or manifest.get("routes") != ["P", "F", "A"]
        or manifest.get("private_randomizers_or_step_traces_redistributed")
        is not False
    ):
        raise PublicSupplementV35Error("Package manifest metadata mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PublicSupplementV35Error("Manifest files must be a list")
    expected_paths: list[str] = []
    expected_total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PublicSupplementV35Error("Malformed manifest row")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or chr(92) in relative
            or ".." in Path(relative).parts
        ):
            raise PublicSupplementV35Error("Noncanonical manifest path")
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicSupplementV35Error(
                f"Manifest path escapes package: {relative}"
            ) from exc
        if (
            not candidate.is_file()
            or candidate.stat().st_size != row["bytes"]
            or file_sha256(candidate) != row["sha256"]
        ):
            raise PublicSupplementV35Error(
                f"Manifest binding mismatch: {relative}"
            )
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
        or observed_paths != expected_paths
        or manifest.get("file_count") != len(expected_paths)
        or manifest.get("total_bytes") != expected_total
    ):
        raise PublicSupplementV35Error("Manifest allowlist/count mismatch")
    return manifest


def verify_disclosure_boundary() -> dict[str, int]:
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    identity_hits = 0
    paper_process_hits = 0
    for path in files:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts.intersection(FORBIDDEN_PATH_PARTS):
            raise PublicSupplementV35Error(
                f"Forbidden path: {relative.as_posix()}"
            )
        lowered_name = path.name.lower()
        if any(lowered_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise PublicSupplementV35Error(
                f"Forbidden transient file: {relative.as_posix()}"
            )
        raw = path.read_bytes()
        if any(
            marker.lower().encode() in raw.lower()
            for marker in (
                LOCAL_IDENTITY_TOKEN,
                WINDOWS_HOME_PREFIX,
                POSIX_WINDOWS_HOME,
            )
        ):
            identity_hits += 1
        if relative in (
            Path("paper/main_aaai27_v35_candidate.tex"),
            Path("paper/main_aaai27_supplement_v35.tex"),
        ):
            text = raw.decode("utf-8", errors="strict").lower()
            paper_process_hits += sum(marker in text for marker in PROCESS_MARKERS)
    if identity_hits or paper_process_hits:
        raise PublicSupplementV35Error(
            "Identity or manuscript-process disclosure detected"
        )
    return {
        "files_scanned": len(files),
        "identity_hits": identity_hits,
        "manuscript_process_hits": paper_process_hits,
    }


def verify_exact_rebuild(
    path: Path,
    builder: Callable[[], dict[str, Any]],
    *,
    label: str,
    payload_field: str = "payload_sha256",
) -> dict[str, Any]:
    observed = load_object(path)
    verify_embedded_payload(observed, payload_field, label)
    rebuilt = builder()
    if observed != rebuilt:
        raise PublicSupplementV35Error(f"{label} exact rebuild mismatch")
    if observed.get("status") != "PASS":
        raise PublicSupplementV35Error(f"{label} status is not PASS")
    return observed


def verify_source_map(
    source_map: dict[str, Any],
    *,
    label: str,
) -> None:
    if not source_map:
        raise PublicSupplementV35Error(f"{label} source map is empty")
    for relative, expected in source_map.items():
        path = ROOT / Path(relative)
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or file_sha256(path) != expected
        ):
            raise PublicSupplementV35Error(
                f"{label} source mismatch: {relative}"
            )


def verify_v35_signal_gates() -> dict[str, Any]:
    g1 = verify_exact_rebuild(G1, build_g1_report, label="G1 accountant")
    g2 = load_object(G2)
    verify_embedded_payload(g2, "payload_sha256", "G2 contract snapshot")
    if g2.get("status") != "PASS":
        raise PublicSupplementV35Error(
            "G2 historical contract snapshot status is not PASS"
        )
    g3 = load_object(G3)
    verify_embedded_payload(g3, "payload_sha256", "G3 executor")
    if (
        g3.get("status") != "PASS"
        or not all(g3.get("checks", {}).values())
        or not all(g3.get("correspondence", {}).values())
        or not all(g3.get("execution_invariants", {}).values())
    ):
        raise PublicSupplementV35Error("G3 executor report mismatch")
    verify_source_map(
        g3.get("claim_source_sha256", {}),
        label="G3 executor",
    )
    g4 = verify_exact_rebuild(G4, build_g4_report, label="G4 registry")
    g5 = verify_exact_rebuild(G5, build_g5_report, label="G5 lattice")
    g6 = verify_g6_gate(G6_ROOT, include_private=False)
    g7 = verify_g7_report(G7)

    g4_ok = (
        len(g4.get("registry_rows", [])) == 3
        and len(g4.get("required_registration_obligations", [])) == 7
        and len(g4.get("compilation_rows", [])) == 9
        and len(g4.get("conservative_extension_rows", [])) == 6
        and all(
            row.get("pass") is True
            for row in g4.get("conservative_extension_rows", [])
        )
    )
    g5_summary = g5.get("summary", {})
    g5_ok = (
        g5_summary.get("required_nonendpoint_hybrids") == 2286
        and g5_summary.get("rejected_nonendpoint_hybrids") == 2286
        and g5_summary.get("accepted_nonendpoint_hybrids") == 0
    )
    g6_ok = (
        g6.get("public_counts")
        == {
            "datasets": 3,
            "runs": 15,
            "models": 15,
            "bundles": 15,
            "summaries": 3,
        }
        and g6.get("private_counts") is None
    )
    g7_ok = (
        g7.get("hybrids_rebuilt") == 378
        and g7.get("sealed_rejections") == 378
    )
    if not (g4_ok and g5_ok and g6_ok and g7_ok):
        raise PublicSupplementV35Error("V3.5 scientific counts mismatch")
    return {
        "g1_profiles": len(g1.get("cases", [])),
        "g2_pre_executor_contract_snapshot": g2.get("status"),
        "g3_assigned_owner_vectors": g3.get("fixture", {}).get(
            "assigned_owner_vectors"
        ),
        "registered_handlers": len(g4.get("registry_rows", [])),
        "obligations_per_handler": len(
            g4.get("required_registration_obligations", [])
        ),
        "old_pf_outputs_preserved": len(
            g4.get("conservative_extension_rows", [])
        ),
        "contract_nonendpoints_rejected": 2286,
        "public_allocation_runs": g6["public_counts"]["runs"],
        "handler_nonendpoints_rebuilt": g7["hybrids_rebuilt"],
        "handler_nonendpoints_sealed_rejected": g7["sealed_rejections"],
        "private_allocation_traces_used": False,
    }


def _source_hash_for_suffix(
    source_hashes: dict[str, Any],
    suffix: str,
) -> str | None:
    matches = [
        value
        for key, value in source_hashes.items()
        if key.replace("\\", "/").endswith(suffix)
    ]
    return str(matches[0]) if len(matches) == 1 else None


def verify_manuscript_binding() -> dict[str, Any]:
    gate = load_object(MANUSCRIPT_GATE)
    payload = verify_embedded_payload(
        gate, "payload_sha256", "manuscript gate"
    )
    handler = gate.get("handler_ablation_recomputation", {})
    contract = gate.get("contract_lattice_recomputation", {})
    if (
        gate.get("status") != "PASS"
        or not all(gate.get("checks", {}).values())
        or handler.get("required_hybrids") != 378
        or handler.get("preseal_accepted") != 42
        or handler.get("sealed_accepted") != 0
        or contract.get("rejected_nonendpoints") != 2286
    ):
        raise PublicSupplementV35Error("Manuscript gate counts mismatch")
    source_hashes = gate.get("source_sha256", {})
    if (
        _source_hash_for_suffix(
            source_hashes, "main_aaai27_v35_candidate.tex"
        )
        != file_sha256(MAIN_SOURCE)
        or _source_hash_for_suffix(
            source_hashes, "main_aaai27_supplement_v35.tex"
        )
        != file_sha256(SUPPLEMENT_SOURCE)
    ):
        raise PublicSupplementV35Error("Packaged manuscript source mismatch")
    main = MAIN_SOURCE.read_text(encoding="utf-8")
    supplement = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    required = (
        "Evidence-Sealed Registration of Owner-Sampled DP-SGD Routes",
        "complete-core",
        "378",
        "42",
        "2,286",
        "The mechanisms and accountants are prior work.",
        "not a privacy proof",
    )
    if not all(fragment in main for fragment in required):
        raise PublicSupplementV35Error("Main bounded claim text mismatch")
    if not all(
        fragment in supplement
        for fragment in (
            "378/42/0",
            "42-to-0",
            "not transfer mechanism novelty",
            "not attestation",
        )
    ):
        raise PublicSupplementV35Error(
            "Supplement ownership-boundary text mismatch"
        )
    return {
        "status": "verified_frozen_prepackage_gate",
        "payload_sha256": payload,
        "main_source_sha256": file_sha256(MAIN_SOURCE),
        "supplement_source_sha256": file_sha256(SUPPLEMENT_SOURCE),
        "aggregate_checks": len(gate.get("checks", {})),
    }


def verify_legacy_pf() -> dict[str, Any]:
    p_result = legacy.verify_collection_artifacts_v2(
        legacy.P_EVIDENCE,
        source_root=ROOT,
    )
    f_result = legacy.verify_srswor_public_collection(legacy.F_EVIDENCE)
    nonprivate = legacy.verify_nonprivate_public()
    p_mutation = legacy.verify_v2_mutations(legacy.P_MUTATION_REPORT)
    natural = legacy.verify_natural_report_portable()
    boundary = legacy.verify_boundary_report(legacy.BOUNDARY_REPORT)
    f_mutation = legacy.verify_v3_mutations(legacy.F_MUTATION_REPORT)
    mechanism = legacy.verify_mechanism_integrity(legacy.MECHANISM_REPORT)
    comparator_payload = legacy.verify_frozen_regeneration(
        legacy.COMPARATOR_REPORT,
        legacy.build_comparator_report(),
        name="Comparator audit",
    )
    release_payload = legacy.verify_frozen_regeneration(
        legacy.RELEASE_REPORT,
        legacy.build_release_report(),
        name="Release-domain audit",
    )
    v34_signal = legacy.verify_v34_signal_gates()
    opacus = legacy.verify_opacus_probe()
    birrell = legacy.verify_birrell_probe()
    return {
        "route_p_runs": p_result.run_count,
        "route_f_runs": f_result["run_count"],
        "nonprivate_runs": nonprivate["models"],
        "p_mutations": p_mutation["mutation_case_count"],
        "natural_failures": natural["cases"],
        "boundary_full_rejections": boundary["summary"][
            "full_lifecycle_targeted_rejected"
        ],
        "f_cross_route_mutations": f_mutation["case_count"],
        "mechanism_runs": mechanism["totals"]["runs"],
        "comparator_payload_sha256": comparator_payload,
        "release_payload_sha256": release_payload,
        "v34_splice_rejections": v34_signal[
            "bidirectional_splice_rejections"
        ],
        "opacus_controlled_rejections": opacus["controlled_rejections"],
        "birrell_registered_cases": birrell["registered_cases"],
    }


def verify_all(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path)
    disclosure = verify_disclosure_boundary()
    legacy_pf = verify_legacy_pf()
    v35 = verify_v35_signal_gates()
    manuscript = verify_manuscript_binding()
    result: dict[str, Any] = {
        "schema_version": "unitdp.public_code_data_package_verification.v35",
        "status": "verified",
        "package_manifest_payload_sha256": manifest["payload_sha256"],
        "package_file_count": manifest["file_count"],
        "disclosure_scan": disclosure,
        "legacy_pf": legacy_pf,
        "v35_signal_gates": v35,
        "manuscript_binding": manuscript,
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
