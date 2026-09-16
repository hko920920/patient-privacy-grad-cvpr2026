"""Verify the anonymous, redistributable V3.4 code-and-data supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from audit_v31_release_domain_gap import (  # noqa: E402
    DEFAULT_OUTPUT as RELEASE_REPORT,
    build_report as build_release_report,
)
from audit_v31_utility_comparator_candidates import (  # noqa: E402
    DEFAULT_OUTPUT as COMPARATOR_REPORT,
    build_report as build_comparator_report,
)
from build_v34_matched_shell_gate import (  # noqa: E402
    DEFAULT_OUTPUT as V34_MATCHED_SHELL_REPORT,
    build_report as build_v34_matched_shell_report,
)
from build_v34_registry_lemma_gate import (  # noqa: E402
    DEFAULT_OUTPUT as V34_REGISTRY_LEMMA_REPORT,
    build_report as build_v34_registry_lemma_report,
)
from build_v34_route_obligation_gate import (  # noqa: E402
    DEFAULT_OUTPUT as V34_ROUTE_OBLIGATION_REPORT,
    build_report as build_v34_route_obligation_report,
)
from verify_v2_boundary_ablation import (  # noqa: E402
    DEFAULT_REPORT as BOUNDARY_REPORT,
    verify_report as verify_boundary_report,
)
from verify_v2_mutation_audit import verify as verify_v2_mutations  # noqa: E402
from verify_v3_mechanism_matched_reference_audit import (  # noqa: E402
    verify_integrity as verify_mechanism_integrity,
)
from verify_v3_route_mutation_audit import (  # noqa: E402
    verify as verify_v3_mutations,
)
from verify_v3_srswor_public_collection import (  # noqa: E402
    verify_public_collection as verify_srswor_public_collection,
)
from unitdp.execution_artifacts_v2 import (  # noqa: E402
    load_model_artifact_v2,
)
from unitdp.execution_verifier_v2 import (  # noqa: E402
    verify_collection_artifacts_v2,
)
from unitdp.owner_poisson_v2 import _model_state_sha256  # noqa: E402
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_HOME_PREFIX = "C:" + chr(92) + "Users" + chr(92)
POSIX_STYLE_WINDOWS_HOME = "C:" + "/" + "Users" + "/"
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
MANUSCRIPT_PROCESS_MARKERS = (
    "virtual review",
    "reviewer score",
    "acceptance probability",
    "acceptance optimization",
    "weak accept",
    "weak reject",
    "maximize review",
    "anomaly point",
    "paper outcome",
    "가상리뷰",
    "synthetic review",
    "review trajectory",
    "neurips review",
)

P_EVIDENCE = ROOT / "reports" / "v2_registered_5seed_v32_20260724"
F_EVIDENCE = (
    ROOT / "reports" / "v3_srswor_registered_5seed_v32_20260724"
)
NONPRIVATE_REPORT = (
    ROOT
    / "reports"
    / "v2_nonprivate_reference_v32_20260724"
    / "public_nonprivate_reference_v1.json"
)
P_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v2_mutation_audit_v32_20260724"
    / "public_mutation_audit_v2.json"
)
NATURAL_REPORT = (
    ROOT
    / "reports"
    / "v2_natural_failure_corpus_v32r1_20260724"
    / "natural_failure_corpus_v2.json"
)
F_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v3_route_mutation_audit_v32_20260724"
    / "public_route_mutation_audit_v3.json"
)
MECHANISM_REPORT = (
    ROOT
    / "reports"
    / "v3_mechanism_matched_reference_v32_20260724"
    / "public_mechanism_matched_reference_v32.json"
)
OPACUS_PROBE_REPORT = (
    ROOT
    / "reports"
    / "opacus_boundary_probe_v1_20260724"
    / "opacus_boundary_probe_v1.json"
)
BIRRELL_PROBE_REPORT = (
    ROOT
    / "reports"
    / "birrell_fixed_size_accounting_probe_v1_20260724"
    / "birrell_accounting_probe_v1.json"
)
PAPER_SOURCE = ROOT / "paper" / "main_aaai27_v34_candidate.tex"


class PublicSupplementVerificationError(RuntimeError):
    """Raised when the public supplement is stale or disclosive."""


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
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSupplementVerificationError(
            f"Could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PublicSupplementVerificationError(
            f"Expected a JSON object: {path}"
        )
    return value


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = load_object(path)
    reported_payload = manifest.get("payload_sha256")
    payload = dict(manifest)
    payload.pop("payload_sha256", None)
    if reported_payload != canonical_sha256(payload):
        raise PublicSupplementVerificationError(
            "Package manifest payload hash mismatch"
        )
    if (
        manifest.get("schema_version")
        != "unitdp.public_code_data_package.v34"
        or manifest.get("visibility") != "public"
        or manifest.get("package_id")
        != "compiler_code_and_data_supplement_aaai27_v3"
    ):
        raise PublicSupplementVerificationError(
            "Package manifest metadata mismatch"
        )
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PublicSupplementVerificationError(
            "Package manifest files must be a list"
        )
    expected_paths: list[str] = []
    expected_total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise PublicSupplementVerificationError(
                "Malformed package manifest row"
            )
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or chr(92) in relative
        ):
            raise PublicSupplementVerificationError(
                "Manifest paths must be nonempty POSIX-relative paths"
            )
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicSupplementVerificationError(
                f"Manifest path escapes package: {relative}"
            ) from exc
        if not candidate.is_file():
            raise PublicSupplementVerificationError(
                f"Manifest file is missing: {relative}"
            )
        if (
            candidate.stat().st_size != row["bytes"]
            or file_sha256(candidate) != row["sha256"]
        ):
            raise PublicSupplementVerificationError(
                f"Manifest file binding mismatch: {relative}"
            )
        expected_paths.append(relative)
        expected_total += int(row["bytes"])
    if expected_paths != sorted(expected_paths):
        raise PublicSupplementVerificationError(
            "Manifest file paths are not canonical"
        )
    if len(expected_paths) != len(set(expected_paths)):
        raise PublicSupplementVerificationError(
            "Manifest contains duplicate paths"
        )
    observed_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if (
            path.is_file()
            and path.name != MANIFEST_NAME
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix.lower() != ".pyc"
        )
    )
    if observed_paths != expected_paths:
        raise PublicSupplementVerificationError(
            "Package files and manifest allowlist differ"
        )
    if (
        manifest.get("file_count") != len(expected_paths)
        or manifest.get("total_bytes") != expected_total
    ):
        raise PublicSupplementVerificationError(
            "Manifest count or byte total mismatch"
        )
    return manifest


def verify_disclosure_boundary() -> dict[str, int]:
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    identity_hits = 0
    process_hits = 0
    for path in files:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts.intersection({"__pycache__", ".pytest_cache"}):
            continue
        if lowered_parts.intersection({"_private", ".git"}):
            raise PublicSupplementVerificationError(
                f"Forbidden private path: {relative.as_posix()}"
            )
        lowered_name = path.name.lower()
        if lowered_name.endswith(".pyc"):
            continue
        if any(
            lowered_name.endswith(suffix)
            for suffix in FORBIDDEN_SUFFIXES
        ):
            raise PublicSupplementVerificationError(
                f"Forbidden transient file: {relative.as_posix()}"
            )
        raw = path.read_bytes()
        for marker in (
            LOCAL_IDENTITY_TOKEN.encode(),
            WINDOWS_HOME_PREFIX.encode(),
            POSIX_STYLE_WINDOWS_HOME.encode(),
        ):
            if marker.lower() in raw.lower():
                identity_hits += 1
        if relative == Path("paper/main_aaai27_v34_candidate.tex"):
            text = raw.decode("utf-8", errors="strict").lower()
            process_hits += sum(
                marker in text
                for marker in MANUSCRIPT_PROCESS_MARKERS
            )
    if identity_hits:
        raise PublicSupplementVerificationError(
            f"Package contains {identity_hits} local identity markers"
        )
    if process_hits:
        raise PublicSupplementVerificationError(
            "Manuscript contains prohibited process language"
        )
    return {
        "files_scanned": len(files),
        "identity_hits": identity_hits,
        "manuscript_process_hits": process_hits,
    }


def verify_nonprivate_public() -> dict[str, Any]:
    report = load_object(NONPRIVATE_REPORT)
    reported = report.get("public_report_sha256")
    payload = dict(report)
    payload.pop("public_report_sha256", None)
    if reported != canonical_sha256(payload):
        raise PublicSupplementVerificationError(
            "Non-private report payload hash mismatch"
        )
    if (
        report.get("visibility") != "public"
        or report.get("research_seeds_disclosed") is not False
        or report.get("dataset_count") != 3
        or report.get("run_count") != 15
    ):
        raise PublicSupplementVerificationError(
            "Non-private report metadata mismatch"
        )
    expected_sources = {
        "scripts/run_v2_nonprivate_reference.py": (
            ROOT / "scripts" / "run_v2_nonprivate_reference.py"
        ),
        "src/unitdp/nonprivate.py": (
            ROOT / "src" / "unitdp" / "nonprivate.py"
        ),
    }
    source_bundle = report.get("reference_source_bundle")
    if source_bundle != {
        relative: file_sha256(path)
        for relative, path in expected_sources.items()
    }:
        raise PublicSupplementVerificationError(
            "Non-private reference source binding mismatch"
        )
    if (
        canonical_sha256(source_bundle)
        != report["reference_source_bundle_sha256"]
    ):
        raise PublicSupplementVerificationError(
            "Non-private reference source bundle hash mismatch"
        )

    model_count = 0
    metric_count = 0
    current_p_bundle = execution_source_bundle_sha256_v2()
    for dataset in report["datasets"]:
        if dataset["data_binding"]["execution_source_bundle_sha256"] != (
            current_p_bundle
        ):
            raise PublicSupplementVerificationError(
                "Non-private data source binding mismatch"
            )
        if dataset["run_count"] != 5 or len(dataset["runs"]) != 5:
            raise PublicSupplementVerificationError(
                "Non-private run count mismatch"
            )
        for index, run in enumerate(dataset["runs"], start=1):
            if run["run_id"] != f"run_{index:03d}":
                raise PublicSupplementVerificationError(
                    "Non-private run index mismatch"
                )
            model_path = NONPRIVATE_REPORT.parent / run["model_artifact"]
            model_raw = load_object(model_path)
            model_payload = dict(model_raw)
            reported_model_payload = model_payload.pop(
                "payload_sha256",
                None,
            )
            if (
                file_sha256(model_path) != run["model_file_sha256"]
                or canonical_sha256(model_payload)
                != reported_model_payload
                or reported_model_payload != run["model_payload_sha256"]
            ):
                raise PublicSupplementVerificationError(
                    "Non-private model file or payload mismatch"
                )
            model = load_model_artifact_v2(model_path)
            state_sha256 = _model_state_sha256(model)
            if (
                state_sha256 != model_raw["state_sha256"]
                or state_sha256 != run["model_state_sha256"]
            ):
                raise PublicSupplementVerificationError(
                    "Non-private model state mismatch"
                )
            model_count += 1
            metric_count += len(run["evaluation"])
        for metric, summary in dataset["aggregate"].items():
            values = [
                float(run["evaluation"][metric])
                for run in dataset["runs"]
            ]
            expected = {
                "count": len(values),
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }
            if summary != expected:
                raise PublicSupplementVerificationError(
                    f"Non-private aggregate mismatch: {metric}"
                )
    return {
        "status": "verified",
        "models": model_count,
        "metric_values": metric_count,
        "public_report_sha256": reported,
    }


def verify_natural_report_portable() -> dict[str, Any]:
    report = load_object(NATURAL_REPORT)
    payload = dict(report)
    reported = payload.pop("natural_failure_corpus_sha256", None)
    if reported != canonical_sha256(payload):
        raise PublicSupplementVerificationError(
            "Natural-failure report payload hash mismatch"
        )
    if (
        report.get("schema_version")
        != "unitdp.natural_failure_corpus.v2.1"
        or report.get("summary", {}).get("case_count") != 8
        or report["summary"]["reproduced_count"] != 8
        or report["summary"]["all_cases_reproduced"] is not True
        or report["bindings"]["production_source_bundle_sha256"]
        != execution_source_bundle_sha256_v2()
    ):
        raise PublicSupplementVerificationError(
            "Natural-failure report metadata mismatch"
        )
    repository_witnesses = 0
    external_witnesses = 0
    for case in report["cases"]:
        if case.get("status") != "reproduced":
            raise PublicSupplementVerificationError(
                "Natural-failure case is not reproduced"
            )
        for witness in case["witnesses"]:
            if "path" not in witness:
                external_witnesses += 1
                continue
            path = ROOT / witness["path"]
            if (
                not path.is_file()
                or path.stat().st_size != witness["bytes"]
                or file_sha256(path) != witness["sha256"]
            ):
                raise PublicSupplementVerificationError(
                    f"Natural-failure witness mismatch: {witness['path']}"
                )
            repository_witnesses += 1
    if external_witnesses != 2:
        raise PublicSupplementVerificationError(
            "Natural-failure external-witness count mismatch"
        )
    return {
        "status": "verified_with_external_raw_files_not_redistributed",
        "cases": 8,
        "repository_witnesses": repository_witnesses,
        "external_raw_witnesses_not_rechecked": external_witnesses,
        "payload_sha256": reported,
    }


def verify_frozen_regeneration(
    frozen_path: Path,
    regenerated: dict[str, Any],
    *,
    name: str,
) -> str:
    frozen = load_object(frozen_path)
    if regenerated != frozen:
        raise PublicSupplementVerificationError(
            f"{name} regeneration differs from frozen report"
        )
    return str(frozen["payload_sha256"])


def verify_v34_signal_gates() -> dict[str, Any]:
    obligation_payload = verify_frozen_regeneration(
        V34_ROUTE_OBLIGATION_REPORT,
        build_v34_route_obligation_report(),
        name="V3.4 route-obligation gate",
    )
    matched_payload = verify_frozen_regeneration(
        V34_MATCHED_SHELL_REPORT,
        build_v34_matched_shell_report(),
        name="V3.4 matched-shell gate",
    )
    lemma_payload = verify_frozen_regeneration(
        V34_REGISTRY_LEMMA_REPORT,
        build_v34_registry_lemma_report(),
        name="V3.4 registry-lemma gate",
    )

    obligation = load_object(V34_ROUTE_OBLIGATION_REPORT)
    matched = load_object(V34_MATCHED_SHELL_REPORT)
    lemma = load_object(V34_REGISTRY_LEMMA_REPORT)
    obligation_ok = (
        obligation.get("status") == "PASS"
        and obligation.get("baseline_contracts", {}).get("accepted") == 6
        and len(obligation.get("obligations", [])) == 8
        and all(
            row.get("p_evidence_pass")
            and row.get("f_evidence_pass")
            and row.get("both_routes_bound")
            for row in obligation.get("obligations", [])
        )
        and obligation.get("verdict", {}).get(
            "L2_generic_route_privacy_or_correctness"
        )
        == "FORBIDDEN"
    )
    decomposition = matched.get("paired_contract_decomposition", {})
    substitutions = matched.get("cross_route_substitutions", {})
    matched_ok = (
        matched.get("status") == "PASS"
        and matched.get("baseline_contracts", {}).get("accepted") == 6
        and decomposition.get("pairs_passed") == 3
        and decomposition.get("expected_shared_leaf_count") == 38
        and decomposition.get("expected_route_different_leaf_count") == 16
        and decomposition.get("expected_only_p_leaf_count") == 1
        and decomposition.get("expected_only_f_leaf_count") == 7
        and substitutions.get("required") == 48
        and substitutions.get("rejected") == 48
    )
    compilations = lemma.get("successful_compilations", {})
    tamper = lemma.get("postcompile_tamper_checks", {})
    lemma_ok = (
        lemma.get("status") == "PASS"
        and len(lemma.get("proof_obligations", [])) == 7
        and all(
            row.get("pass")
            for row in lemma.get("proof_obligations", [])
        )
        and compilations.get("passed") == 6
        and compilations.get("required") == 6
        and tamper.get("rejected") == 18
        and tamper.get("required") == 18
    )
    if not (obligation_ok and matched_ok and lemma_ok):
        raise PublicSupplementVerificationError(
            "V3.4 signal-gate scientific counts mismatch"
        )
    return {
        "status": "verified_exact_regeneration",
        "route_obligations": 8,
        "exact_baselines": 6,
        "paired_contracts": 3,
        "shared_shell_leaves_per_pair": 38,
        "route_specific_shared_keys_per_pair": 16,
        "bidirectional_splice_rejections": 48,
        "proof_obligations": 7,
        "executable_compilations": 6,
        "postcompile_tamper_rejections": 18,
        "route_obligation_payload_sha256": obligation_payload,
        "matched_shell_payload_sha256": matched_payload,
        "registry_lemma_payload_sha256": lemma_payload,
    }


def verify_opacus_probe() -> dict[str, Any]:
    report = load_object(OPACUS_PROBE_REPORT)
    if (
        report.get("schema") != "unitdp.opacus_boundary_probe.v1"
        or report.get("environment", {}).get("opacus") != "1.6.0"
        or report.get("determinism_check", {}).get(
            "independent_in_process_builds_equal"
        )
        is not True
    ):
        raise PublicSupplementVerificationError(
            "Opacus boundary-probe metadata mismatch"
        )
    observations = report.get("opacus_observations", {})
    if (
        observations.get("model_optimizer_mismatch", {}).get("passed")
        is not True
        or observations.get("poisson_true", {}).get("succeeded") is not True
        or observations.get("poisson_false", {}).get(
            "documented_accounting_assumption_caveat"
        )
        is not True
        or observations.get("new_dataset_same_engine", {}).get(
            "warning_observed"
        )
        is not True
    ):
        raise PublicSupplementVerificationError(
            "Opacus executable observation mismatch"
        )
    absent = set(observations.get("route_fields_absent_from_make_private", []))
    expected_absent = {
        "adjacency",
        "artifact_chain",
        "owner_mapping",
        "preprocessing_sha256",
        "privacy_unit",
        "route_id",
        "source_bundle_sha256",
    }
    if absent != expected_absent:
        raise PublicSupplementVerificationError(
            "Opacus residual-field boundary mismatch"
        )
    unitdp = report.get("unitdp_observations", {})
    cases = unitdp.get("controlled_substitution_rejections", {})
    if (
        not isinstance(cases, dict)
        or len(cases) != 10
        or any(
            case.get("passed") is not True
            or case.get("observed_outcome") != "reject"
            or case.get("exception_type") != "ContractV2Error"
            for case in cases.values()
        )
    ):
        raise PublicSupplementVerificationError(
            "Opacus paired controlled-substitution mismatch"
        )
    source_hashes = report.get("source_sha256", {})
    local_sources = {
        "scripts/build_opacus_boundary_probe_v1.py": (
            ROOT / "scripts" / "build_opacus_boundary_probe_v1.py"
        ),
        "src/unitdp/compiler_v2.py": ROOT / "src" / "unitdp" / "compiler_v2.py",
        "src/unitdp/compiler_v3.py": ROOT / "src" / "unitdp" / "compiler_v3.py",
    }
    for relative, path in local_sources.items():
        if source_hashes.get(relative) != file_sha256(path):
            raise PublicSupplementVerificationError(
                f"Opacus probe source binding mismatch: {relative}"
            )
    return {
        "status": "verified_fixed_scope_report",
        "opacus_version": "1.6.0",
        "coverage_rows": len(report.get("coverage", [])),
        "controlled_rejections": len(cases),
        "report_file_sha256": file_sha256(OPACUS_PROBE_REPORT),
    }


def verify_birrell_probe() -> dict[str, Any]:
    report = load_object(BIRRELL_PROBE_REPORT)
    if (
        report.get("schema")
        != "unitdp.birrell_fixed_size_accounting_probe.v1"
        or report.get("decision", {}).get("route_f_accountant")
        != "retain_frozen_wang_route"
        or report.get("determinism_check", {}).get(
            "independent_in_process_builds_equal"
        )
        is not True
    ):
        raise PublicSupplementVerificationError(
            "Birrell accounting-probe metadata mismatch"
        )
    upstream = report.get("upstream_provenance", {})
    supplement = report.get("supplement_provenance", {})
    if (
        upstream.get("git_commit")
        != "799f7b755e8c792dd4a5f7e77cb878a6d12e7fd6"
        or upstream.get("accountant_sha256")
        != "4847df3dd719e6e6ec6aa1bb11224e73ca30097618b092860682bdaeb5aa6741"
        or supplement.get("supplement_zip_sha256")
        != "bb9d45b0b41c59cddfd250947256d30d47ff8b03a86aaa8106e104702bdf9ce1"
        or supplement.get("matches_linked_github_source") is not False
        or supplement.get("defines_final_replace_one_function_FSwoR_RDP_ro")
        is not False
    ):
        raise PublicSupplementVerificationError(
            "Birrell upstream provenance mismatch"
        )
    expected = {
        "uci_owner_srswor_v3.yaml": 22.03930592881076,
        "wisdm_owner_srswor_v3.yaml": 18.341042032395883,
        "sepsis_owner_srswor_v3.yaml": 14.636324155017116,
    }
    cases = report.get("registered_cases", [])
    if not isinstance(cases, list) or len(cases) != 3:
        raise PublicSupplementVerificationError(
            "Birrell registered-case count mismatch"
        )
    observed: dict[str, float] = {}
    for case in cases:
        name = Path(str(case.get("config", ""))).name
        epsilon = float(
            case.get("best_observed_birrell_upper_bound", {}).get(
                "epsilon",
                float("nan"),
            )
        )
        wang = float(
            case.get("frozen_wang", {}).get(
                "dp_accounting_epsilon",
                float("nan"),
            )
        )
        best = case.get("best_observed_birrell_upper_bound", {})
        if (
            name not in expected
            or abs(epsilon - expected[name]) > 1e-12
            or abs(wang - 8.0) > 1e-10
            or case.get("birrell_improves_registered_plan") is not False
            or best.get("taylor_order_m") != 3
            or float(best.get("optimal_rdp_order", -1)) != 2.0
        ):
            raise PublicSupplementVerificationError(
                f"Birrell registered result mismatch: {name}"
            )
        observed[name] = epsilon
    ratio_range = report.get("paper_regime_sanity_check", {}).get(
        "ratio_range",
        [],
    )
    if (
        len(ratio_range) != 2
        or abs(float(ratio_range[0]) - 3.793410358358147) > 1e-12
        or abs(float(ratio_range[1]) - 3.8638896120022106) > 1e-12
    ):
        raise PublicSupplementVerificationError(
            "Birrell paper-regime sanity mismatch"
        )
    return {
        "status": "verified_fixed_external_source_report",
        "registered_cases": len(observed),
        "retained_wang_cases": sum(
            epsilon > 8.0 for epsilon in observed.values()
        ),
        "paper_regime_ratio_range": ratio_range,
        "report_file_sha256": file_sha256(BIRRELL_PROBE_REPORT),
        "external_upstream_source_redistributed": False,
    }


def verify_all(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path)
    disclosure = verify_disclosure_boundary()

    p_result = verify_collection_artifacts_v2(
        P_EVIDENCE,
        source_root=ROOT,
    )
    f_result = verify_srswor_public_collection(F_EVIDENCE)
    nonprivate = verify_nonprivate_public()
    p_mutation = verify_v2_mutations(P_MUTATION_REPORT)
    natural = verify_natural_report_portable()
    boundary = verify_boundary_report(BOUNDARY_REPORT)
    f_mutation = verify_v3_mutations(F_MUTATION_REPORT)
    mechanism = verify_mechanism_integrity(MECHANISM_REPORT)
    comparator_payload = verify_frozen_regeneration(
        COMPARATOR_REPORT,
        build_comparator_report(),
        name="Comparator audit",
    )
    release_payload = verify_frozen_regeneration(
        RELEASE_REPORT,
        build_release_report(),
        name="Release-domain audit",
    )
    v34_signal_gates = verify_v34_signal_gates()
    opacus_probe = verify_opacus_probe()
    birrell_probe = verify_birrell_probe()

    result: dict[str, Any] = {
        "schema_version": (
            "unitdp.public_code_data_package_verification.v34"
        ),
        "status": "verified",
        "package_manifest_payload_sha256": manifest["payload_sha256"],
        "package_file_count": manifest["file_count"],
        "disclosure_scan": disclosure,
        "route_p": {
            "datasets": p_result.dataset_count,
            "runs": p_result.run_count,
            "collection_sha256": p_result.public_collection_sha256,
        },
        "route_f": {
            "datasets": f_result["dataset_count"],
            "runs": f_result["run_count"],
            "models": f_result["model_roundtrips"],
            "metric_values": f_result["public_metric_values"],
            "collection_sha256": f_result[
                "public_collection_sha256"
            ],
        },
        "nonprivate": nonprivate,
        "p_mutations": p_mutation["mutation_case_count"],
        "natural_failures": natural,
        "boundary": boundary["summary"],
        "f_cross_route_mutations": f_mutation["case_count"],
        "mechanism_matched_totals": mechanism["totals"],
        "comparator_payload_sha256": comparator_payload,
        "release_domain_payload_sha256": release_payload,
        "v34_signal_gates": v34_signal_gates,
        "opacus_boundary_probe": opacus_probe,
        "birrell_accounting_probe": birrell_probe,
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(ROOT / MANIFEST_NAME),
    )
    args = parser.parse_args()
    result = verify_all(Path(args.manifest))
    print(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
