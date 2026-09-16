"""Build the deterministic anonymous AAAI-27 Compiler V3.4 code/data ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v3"
DEFAULT_OUTPUT = ROOT / "dist" / PACKAGE_ID
DEFAULT_ZIP = ROOT / "dist" / f"{PACKAGE_ID}.zip"
MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_HOME_PREFIX = "C:" + chr(92) + "Users" + chr(92)
POSIX_STYLE_WINDOWS_HOME = "C:" + "/" + "Users" + "/"

SCRIPT_ALLOWLIST = (
    "audit_v31_release_domain_gap.py",
    "audit_v31_utility_comparator_candidates.py",
    "build_birrell_accounting_probe_v1.py",
    "build_opacus_boundary_probe_v1.py",
    "build_sepsis_public_preprocessing_artifact.py",
    "build_v2_boundary_ablation.py",
    "build_v2_mutation_audit.py",
    "build_v2_natural_failure_corpus.py",
    "build_v3_mechanism_matched_reference_audit.py",
    "build_v3_route_mutation_audit.py",
    "build_v34_matched_shell_gate.py",
    "build_v34_registry_lemma_gate.py",
    "build_v34_route_obligation_gate.py",
    "build_wisdm_public_preprocessing_artifact.py",
    "run_aggregation_ablation.py",
    "run_conservative_els_owner_baseline.py",
    "run_fixed_size_mog_owner_baseline.py",
    "run_owner_membership_probe.py",
    "run_sepsis_comparison.py",
    "run_sepsis_owa.py",
    "run_uci_comparison.py",
    "run_uci_owa.py",
    "run_v2_nonprivate_reference.py",
    "run_v2_registered_benchmarks.py",
    "run_v3_srswor_registered_benchmarks.py",
    "run_wisdm_comparison.py",
    "run_wisdm_owa.py",
    "verify_public_supplement_v34.py",
    "verify_v2_boundary_ablation.py",
    "verify_v2_mutation_audit.py",
    "verify_v2_natural_failure_corpus.py",
    "verify_v2_nonprivate_reference.py",
    "verify_v2_private_execution_diagnostics.py",
    "verify_v2_run_collection.py",
    "verify_v3_mechanism_matched_reference_audit.py",
    "verify_v3_route_mutation_audit.py",
    "verify_v3_srswor_public_collection.py",
    "verify_v3_srswor_registered_benchmarks.py",
    "verify_v34_matched_shell_gate.py",
    "verify_v34_registry_lemma_gate.py",
    "verify_v34_route_obligation_gate.py",
)

SINGLE_REPORT_FILES = (
    (
        "reports/v2_nonprivate_reference_verification_v32_20260724/"
        "public_verification_v1.json"
    ),
    (
        "reports/v2_mutation_audit_v32_20260724/"
        "public_mutation_audit_v2.json"
    ),
    (
        "reports/v2_natural_failure_corpus_v32r1_20260724/"
        "natural_failure_corpus_v2.json"
    ),
    (
        "reports/v2_boundary_ablation_v32r1_20260724/"
        "public_boundary_ablation_v2.json"
    ),
    (
        "reports/v3_route_mutation_audit_v32_20260724/"
        "public_route_mutation_audit_v3.json"
    ),
    (
        "reports/v3_mechanism_matched_reference_v32_20260724/"
        "public_mechanism_matched_reference_v32.json"
    ),
    (
        "reports/v32_comparator_feasibility_20260724/"
        "public_utility_comparator_feasibility_v32.json"
    ),
    (
        "reports/v32_release_domain_gap_20260724/"
        "public_release_domain_gap_v32.json"
    ),
    (
        "reports/opacus_boundary_probe_v1_20260724/"
        "opacus_boundary_probe_v1.json"
    ),
    (
        "reports/birrell_fixed_size_accounting_probe_v1_20260724/"
        "birrell_accounting_probe_v1.json"
    ),
    "reports/mog_owner_baseline_uci_wisdm_5seed_001/aggregate.json",
    (
        "reports/fixed_size_mog_owner_baseline_uci_wisdm_5seed_001/"
        "aggregate.json"
    ),
    "reports/conservative_els_owner_5seed_001/aggregate.json",
    (
        "reports/comparison_benchmark_3seed_001/uci/seed_13/"
        "owner_certificate.json"
    ),
    (
        "reports/comparison_benchmark_5seed_001/wisdm/seed_13/"
        "wisdm_train_mapping.csv"
    ),
    (
        "reports/comparison_benchmark_5seed_001/sepsis/seed_13/"
        "sepsis_train_mapping.csv"
    ),
    (
        "reports/uci_public_preprocessing_smoke_20260724/"
        "uci_train_mapping.csv"
    ),
    (
        "reports/wisdm_public_protocol_smoke_20260724/"
        "wisdm_train_mapping.csv"
    ),
    (
        "reports/sepsis_public_protocol_smoke_20260724/"
        "sepsis_train_mapping.csv"
    ),
)

PAPER_FILES = (
    "aaai2027.bst",
    "aaai2027.sty",
    "main_aaai27_v34_candidate.tex",
    "references_v28_candidate.bib",
    "references_v30_additions.bib",
    "references_v33_additions.bib",
    "tables_v31_candidate/v31_conformance.tex",
    "tables_v31_candidate/v31_profiles.tex",
    "tables_v31_candidate/v31_utility.tex",
)


class PackageBuildError(RuntimeError):
    """Raised when the public package cannot be built safely."""


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


def ensure_inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PackageBuildError(
            f"Output must remain inside the repository: {resolved}"
        ) from exc
    return resolved


def add_file(
    source: Path,
    relative: str,
    *,
    output: Path,
    seen: set[str],
) -> None:
    normalized = Path(relative).as_posix()
    if (
        not normalized
        or normalized.startswith("/")
        or normalized in seen
        or ".." in Path(normalized).parts
    ):
        raise PackageBuildError(
            f"Invalid or duplicate package path: {relative}"
        )
    if not source.is_file():
        raise PackageBuildError(f"Required source is missing: {source}")
    target = output / Path(normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    seen.add(normalized)


def add_tree(
    source_root: Path,
    destination_root: str,
    *,
    output: Path,
    seen: set[str],
    suffixes: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> None:
    excluded = exclude_names or set()
    for source in sorted(path for path in source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if any(part in {"__pycache__", "_private"} for part in relative.parts):
            continue
        if source.name in excluded:
            continue
        if suffixes is not None and source.suffix.lower() not in suffixes:
            continue
        add_file(
            source,
            (Path(destination_root) / relative).as_posix(),
            output=output,
            seen=seen,
        )


def public_execution_tree(
    relative_root: str,
    *,
    output: Path,
    seen: set[str],
    exclude_top_level: set[str] | None = None,
) -> None:
    source_root = ROOT / relative_root
    excluded = exclude_top_level or set()
    for source in sorted(path for path in source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if "_private" in relative.parts:
            continue
        if len(relative.parts) == 1 and source.name in excluded:
            continue
        add_file(
            source,
            (Path(relative_root) / relative).as_posix(),
            output=output,
            seen=seen,
        )


def check_staging(output: Path) -> None:
    identity_hits: list[str] = []
    forbidden_paths: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts.intersection(
            {"_private", "__pycache__", ".git", ".pytest_cache"}
        ):
            forbidden_paths.append(relative.as_posix())
        raw = path.read_bytes().lower()
        if any(
            marker.lower().encode() in raw
            for marker in (
                LOCAL_IDENTITY_TOKEN,
                WINDOWS_HOME_PREFIX,
                POSIX_STYLE_WINDOWS_HOME,
            )
        ):
            identity_hits.append(relative.as_posix())
    if forbidden_paths:
        raise PackageBuildError(
            "Private/cache paths entered staging: "
            + ", ".join(forbidden_paths)
        )
    if identity_hits:
        raise PackageBuildError(
            "Local identity markers entered staging: "
            + ", ".join(identity_hits)
        )


def build_manifest(output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(output).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    manifest: dict[str, object] = {
        "schema_version": "unitdp.public_code_data_package.v34",
        "package_id": PACKAGE_ID,
        "visibility": "public",
        "release_status": "research_non_release",
        "raw_third_party_data_redistributed": False,
        "private_randomizers_or_step_traces_redistributed": False,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    return manifest


def write_deterministic_zip(
    output: Path,
    zip_path: Path,
) -> None:
    prefix = output.name
    with zipfile.ZipFile(
        zip_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source in sorted(
            path for path in output.rglob("*") if path.is_file()
        ):
            relative = source.relative_to(output).as_posix()
            info = zipfile.ZipInfo(
                filename=f"{prefix}/{relative}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(
                info,
                source.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def require_paper_dir(path: Path) -> Path:
    resolved = path.resolve()
    missing = [
        relative
        for relative in PAPER_FILES
        if not (resolved / relative).is_file()
    ]
    if missing:
        raise PackageBuildError(
            "Paper directory is missing: " + ", ".join(missing)
        )
    return resolved


def build(
    *,
    paper_dir: Path,
    output: Path,
    zip_path: Path,
) -> dict[str, object]:
    output = ensure_inside_repo(output)
    zip_path = ensure_inside_repo(zip_path)
    if output.exists() or zip_path.exists():
        raise PackageBuildError(
            "Refusing to overwrite an existing package or ZIP"
        )
    paper_dir = require_paper_dir(paper_dir)
    output.mkdir(parents=True)
    seen: set[str] = set()

    add_file(
        ROOT / "submission" / "compiler_v34" / "README.md",
        "README.md",
        output=output,
        seen=seen,
    )
    add_file(
        ROOT
        / "submission"
        / "compiler_v34"
        / "requirements-aaai27-v34.txt",
        "requirements-aaai27-v34.txt",
        output=output,
        seen=seen,
    )
    add_file(
        ROOT / "pyproject.toml",
        "pyproject.toml",
        output=output,
        seen=seen,
    )

    add_tree(
        ROOT / "src",
        "src",
        output=output,
        seen=seen,
        suffixes={".py"},
    )
    add_tree(
        ROOT / "configs",
        "configs",
        output=output,
        seen=seen,
        suffixes={".json", ".yaml", ".yml"},
    )
    add_file(
        ROOT / "specs" / "owner_poisson_v2_contract_oracle_v1.json",
        "specs/owner_poisson_v2_contract_oracle_v1.json",
        output=output,
        seen=seen,
    )
    for name in SCRIPT_ALLOWLIST:
        add_file(
            ROOT / "scripts" / name,
            f"scripts/{name}",
            output=output,
            seen=seen,
        )
    add_tree(
        ROOT / "tests",
        "tests",
        output=output,
        seen=seen,
        suffixes={".py"},
        exclude_names={"test_v34_manuscript_gate.py"},
    )

    public_execution_tree(
        "reports/v2_registered_5seed_v32_20260724",
        output=output,
        seen=seen,
    )
    public_execution_tree(
        "reports/v3_srswor_registered_5seed_v32_20260724",
        output=output,
        seen=seen,
        exclude_top_level={
            "public_verification_srswor_v3.json",
            "public_verification_srswor_public_v3.json",
        },
    )
    public_execution_tree(
        "reports/v2_nonprivate_reference_v32_20260724",
        output=output,
        seen=seen,
    )
    public_execution_tree(
        "reports/v34_route_obligation_gate_20260725",
        output=output,
        seen=seen,
    )
    public_execution_tree(
        "reports/v34_matched_shell_gate_20260725",
        output=output,
        seen=seen,
    )
    public_execution_tree(
        "reports/v34_registry_lemma_gate_20260725",
        output=output,
        seen=seen,
    )
    for relative in SINGLE_REPORT_FILES:
        add_file(
            ROOT / relative,
            relative,
            output=output,
            seen=seen,
        )

    for relative in PAPER_FILES:
        destination = (
            "paper/main_aaai27_v34_candidate.tex"
            if relative == "main_aaai27_v34_candidate.tex"
            else f"paper/{relative}"
        )
        add_file(
            paper_dir / relative,
            destination,
            output=output,
            seen=seen,
        )

    check_staging(output)
    manifest = build_manifest(output)
    (output / MANIFEST_NAME).write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_deterministic_zip(output, zip_path)
    return {
        "package_id": PACKAGE_ID,
        "directory": str(output),
        "zip": str(zip_path),
        "zip_sha256": file_sha256(zip_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": file_sha256(output / MANIFEST_NAME),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        paper_dir=args.paper_dir,
        output=args.output,
        zip_path=args.zip_path,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
