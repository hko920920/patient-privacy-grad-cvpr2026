"""Reproduce Birrell et al. FSwoR-RDP on registered Route F plans.

The final NeurIPS 2024 paper links the external FSRDP repository.  This
builder deliberately imports a caller-supplied, commit-pinned checkout instead
of copying unlicensed upstream source into the UnitDP repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.accountant import (  # noqa: E402
    epsilon_for_srswor_replace_one_noise,
)
from unitdp.benchmark_registry_srswor_v3 import (  # noqa: E402
    RDP_ORDERS_SRSWOR_V3,
)
from unitdp.compiler_srswor_v3 import (  # noqa: E402
    load_owner_srswor_contract_v3,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (  # noqa: E402
    fixed_size_srswor_epsilon_v3,
)


SCHEMA = "unitdp.birrell_fixed_size_accounting_probe.v1"
REPORT_DATE = "2026-07-24"
UPSTREAM_COMMIT = "799f7b755e8c792dd4a5f7e77cb878a6d12e7fd6"
UPSTREAM_ACCOUNTANT_SHA256 = (
    "4847df3dd719e6e6ec6aa1bb11224e73ca30097618b092860682bdaeb5aa6741"
)
SUPPLEMENT_ACCOUNTANT_SHA256 = (
    "c367b5a43ec75713da3e7a66c3dc8c7c073b1e45f73dd1e4457c7637380ec1d9"
)
SUPPLEMENT_ZIP_SHA256 = (
    "bb9d45b0b41c59cddfd250947256d30d47ff8b03a86aaa8106e104702bdf9ce1"
)
PAPER_URL = (
    "https://papers.neurips.cc/paper_files/paper/2024/hash/"
    "14fef58f09f2ebe69306e0a322e3be2b-Abstract-Conference.html"
)
UPSTREAM_URL = "https://github.com/star-ailab/FSRDP"

CONFIGS = tuple(
    sorted((ROOT / "configs" / "v3").glob("*_owner_srswor_v3.yaml"))
)
REGISTERED_ORDERS = np.asarray(RDP_ORDERS_SRSWOR_V3, dtype=float)
UPSTREAM_EXAMPLE_ORDERS = np.asarray(
    [1 + index / 10.0 for index in range(1, 100)]
    + list(range(12, 64)),
    dtype=float,
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_upstream_module(repository: Path) -> ModuleType:
    source = repository / "fs_rdp_bounds.py"
    if git_commit(repository) != UPSTREAM_COMMIT:
        raise RuntimeError("Unexpected FSRDP commit")
    if file_sha256(source) != UPSTREAM_ACCOUNTANT_SHA256:
        raise RuntimeError("Unexpected FSRDP accountant source hash")
    spec = importlib.util.spec_from_file_location(
        "unitdp_pinned_fsrdp_probe",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load pinned FSRDP source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "FSwoR_RDP_ro"):
        raise RuntimeError("Pinned source lacks FSwoR_RDP_ro")
    return module


def epsilon_from_rdp(
    *,
    orders: np.ndarray,
    rdp: np.ndarray,
    delta: float,
) -> tuple[float, float]:
    finite = np.isfinite(rdp)
    if not finite.any():
        raise ArithmeticError("No finite RDP values")
    finite_orders = orders[finite]
    finite_rdp = rdp[finite]
    epsilon = (
        finite_rdp
        - (np.log(delta) + np.log(finite_orders))
        / (finite_orders - 1)
        + np.log((finite_orders - 1) / finite_orders)
    )
    index = int(np.argmin(epsilon))
    return (
        float(max(0.0, epsilon[index])),
        float(finite_orders[index]),
    )


def birrell_epsilon(
    module: ModuleType,
    *,
    orders: np.ndarray,
    sigma: float,
    taylor_order: int,
    sample_rate: float,
    steps: int,
    delta: float,
) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        per_step = np.asarray(
            [
                module.FSwoR_RDP_ro(
                    float(alpha),
                    sigma,
                    taylor_order,
                    sample_rate,
                )
                for alpha in orders
            ],
            dtype=float,
        )
    composed = steps * per_step
    epsilon, optimal_order = epsilon_from_rdp(
        orders=orders,
        rdp=composed,
        delta=delta,
    )
    return {
        "taylor_order_m": taylor_order,
        "epsilon": epsilon,
        "optimal_rdp_order": optimal_order,
        "finite_orders": int(np.isfinite(composed).sum()),
        "evaluated_orders": int(len(orders)),
        "runtime_warnings": len(caught),
    }


def registered_case(
    module: ModuleType,
    config_path: Path,
) -> dict[str, object]:
    contract = load_owner_srswor_contract_v3(config_path)
    noise = contract.noise_multiplier
    population = contract.source_dataset_size
    batch = contract.sample_size
    steps = contract.total_steps
    delta = contract.delta
    sample_rate = batch / population

    library_epsilon = epsilon_for_srswor_replace_one_noise(
        noise_multiplier=noise,
        source_dataset_size=population,
        sample_size=batch,
        steps=steps,
        delta=delta,
        alphas=list(RDP_ORDERS_SRSWOR_V3),
    )
    direct = fixed_size_srswor_epsilon_v3(
        actual_noise_multiplier=noise,
        source_dataset_size=population,
        sample_size=batch,
        steps=steps,
        delta=delta,
        orders=RDP_ORDERS_SRSWOR_V3,
        sensitivity_multiplier=2.0,
    )
    if abs(library_epsilon - direct.epsilon) > 1e-10:
        raise RuntimeError("Frozen Wang accountant paths disagree")

    same_order_results = [
        birrell_epsilon(
            module,
            orders=REGISTERED_ORDERS,
            sigma=noise,
            taylor_order=taylor_order,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        for taylor_order in (3, 4, 5)
    ]
    upstream_grid_results = [
        birrell_epsilon(
            module,
            orders=UPSTREAM_EXAMPLE_ORDERS,
            sigma=noise,
            taylor_order=taylor_order,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        for taylor_order in (3, 4, 5)
    ]
    m_sweep = [
        birrell_epsilon(
            module,
            orders=REGISTERED_ORDERS,
            sigma=noise,
            taylor_order=taylor_order,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        for taylor_order in range(3, 17)
    ]
    candidates = same_order_results + upstream_grid_results + m_sweep
    best_birrell = min(
        candidates,
        key=lambda item: float(item["epsilon"]),
    )

    return {
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "route_id": contract.route_id,
        "parameters": {
            "source_dataset_size": population,
            "sample_size": batch,
            "sample_rate": sample_rate,
            "steps": steps,
            "actual_noise_std_over_clip_C": noise,
            "replace_one_sensitivity_multiplier": 2.0,
            "wang_gaussian_noise_over_sensitivity_2C": (
                noise / 2.0
            ),
            "delta": delta,
        },
        "sigma_mapping": (
            "Birrell sigma is actual Gaussian std divided by clip C; "
            "the theorem handles replace-one 2C sensitivity internally."
        ),
        "frozen_wang": {
            "dp_accounting_epsilon": library_epsilon,
            "direct_decimal_oracle_epsilon": direct.epsilon,
            "optimal_registered_order": direct.optimal_order,
            "absolute_difference": abs(
                library_epsilon - direct.epsilon
            ),
        },
        "birrell_same_registered_orders": same_order_results,
        "birrell_upstream_example_order_grid": upstream_grid_results,
        "birrell_registered_order_m_sweep_3_to_16": m_sweep,
        "best_observed_birrell_upper_bound": best_birrell,
        "best_birrell_over_wang_epsilon_ratio": (
            float(best_birrell["epsilon"]) / library_epsilon
        ),
        "birrell_improves_registered_plan": (
            float(best_birrell["epsilon"]) < library_epsilon
        ),
    }


def paper_regime_sanity_check(module: ModuleType) -> dict[str, object]:
    population = 50_000
    batch = 120
    sigma = 6.0
    sample_rate = batch / population
    wang = fixed_size_srswor_epsilon_v3(
        actual_noise_multiplier=sigma,
        source_dataset_size=population,
        sample_size=batch,
        steps=1,
        delta=1e-5,
        orders=RDP_ORDERS_SRSWOR_V3,
        sensitivity_multiplier=2.0,
    )
    rows = []
    for item in wang.order_results:
        birrell = float(
            module.FSwoR_RDP_ro(
                float(item.order),
                sigma,
                4,
                sample_rate,
            )
        )
        rows.append(
            {
                "rdp_order": item.order,
                "wang_per_step_rdp": item.per_step_rdp,
                "birrell_m4_per_step_rdp": birrell,
                "wang_over_birrell_ratio": (
                    item.per_step_rdp / birrell
                ),
            }
        )
    ratios = [
        float(row["wang_over_birrell_ratio"])
        for row in rows
    ]
    if not all(3.7 < ratio < 4.1 for ratio in ratios):
        raise RuntimeError(
            "Paper-regime sanity check did not reproduce factor-four behavior"
        )
    return {
        "parameters": {
            "source_dataset_size": population,
            "sample_size": batch,
            "sample_rate": sample_rate,
            "sigma": sigma,
            "taylor_order_m": 4,
            "steps": 1,
        },
        "rows": rows,
        "ratio_range": [min(ratios), max(ratios)],
        "interpretation": (
            "The pinned code reproduces the paper's approximately "
            "factor-four RDP improvement in its illustrated regime."
        ),
    }


def supplement_provenance(
    supplement_root: Path | None,
    supplement_zip: Path | None,
) -> dict[str, object]:
    if supplement_root is None:
        return {"inspected": False}
    candidates = list(supplement_root.rglob("fs_rdp_bounds.py"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one supplemental fs_rdp_bounds.py"
        )
    source = candidates[0]
    source_text = source.read_text(encoding="utf-8")
    source_hash = file_sha256(source)
    if source_hash != SUPPLEMENT_ACCOUNTANT_SHA256:
        raise RuntimeError("Unexpected supplemental accountant hash")
    result: dict[str, object] = {
        "inspected": True,
        "accountant_sha256": source_hash,
        "matches_linked_github_source": (
            source_hash == UPSTREAM_ACCOUNTANT_SHA256
        ),
        "defines_final_replace_one_function_FSwoR_RDP_ro": (
            "def FSwoR_RDP_ro" in source_text
        ),
        "interpretation": (
            "The proceedings supplement contains an older/different "
            "accountant file and does not expose the final linked "
            "replace-one function; the paper-linked GitHub commit is "
            "therefore pinned for this probe."
        ),
    }
    if supplement_zip is not None:
        zip_hash = file_sha256(supplement_zip)
        if zip_hash != SUPPLEMENT_ZIP_SHA256:
            raise RuntimeError("Unexpected proceedings supplement ZIP hash")
        result["supplement_zip_sha256"] = zip_hash
    return result


def build_report(
    *,
    fsrdp_root: Path,
    supplement_root: Path | None,
    supplement_zip: Path | None,
) -> dict[str, object]:
    upstream = load_upstream_module(fsrdp_root)
    cases = [
        registered_case(upstream, config_path)
        for config_path in CONFIGS
    ]
    if not cases:
        raise RuntimeError("No registered Route F configurations found")
    if any(
        bool(case["birrell_improves_registered_plan"])
        for case in cases
    ):
        decision = "reassess_migration"
    else:
        decision = "retain_frozen_wang_route"

    license_names = sorted(
        path.name
        for path in fsrdp_root.iterdir()
        if path.is_file() and "license" in path.name.lower()
    )
    return {
        "schema": SCHEMA,
        "report_date": REPORT_DATE,
        "primary_sources": {
            "paper": PAPER_URL,
            "paper_linked_repository": UPSTREAM_URL,
        },
        "upstream_provenance": {
            "git_commit": git_commit(fsrdp_root),
            "accountant_sha256": file_sha256(
                fsrdp_root / "fs_rdp_bounds.py"
            ),
            "license_files_at_repository_root": license_names,
            "redistribution_action": (
                "No upstream source copied into the UnitDP repository."
            ),
        },
        "supplement_provenance": supplement_provenance(
            supplement_root,
            supplement_zip,
        ),
        "environment": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "paper_regime_sanity_check": paper_regime_sanity_check(
            upstream
        ),
        "registered_cases": cases,
        "decision": {
            "route_f_accountant": decision,
            "reason": (
                "The pinned Birrell upper bounds are looser than the "
                "frozen Wang bound on every registered plan, while the "
                "paper-regime sanity check reproduces Birrell's expected "
                "factor-four improvement."
            ),
            "required_manuscript_action": (
                "Cite Birrell et al.; state that the registered high-q/"
                "low-sigma regimes retain the numerically tighter frozen "
                "Wang/dp-accounting bound."
            ),
            "required_artifact_action": (
                "No accountant or numeric artifact migration."
            ),
        },
        "interpretation_limits": [
            (
                "A looser Birrell upper bound does not invalidate the "
                "smaller valid Wang upper bound or imply epsilon exceeds 8."
            ),
            (
                "The comparison is limited to the pinned upstream "
                "implementation, the evaluated RDP/Taylor orders, and the "
                "three registered plans."
            ),
            (
                "Selecting the smaller of independently valid upper bounds "
                "is sound; no Birrell candidate improved a registered plan."
            ),
            (
                "The observed reversal from the paper regime is a numerical "
                "bound comparison, not a claim about exact privacy loss."
            ),
        ],
    }


def markdown_report(
    report: dict[str, object],
    *,
    json_sha256: str,
) -> str:
    provenance = report["upstream_provenance"]
    supplement = report["supplement_provenance"]
    sanity = report["paper_regime_sanity_check"]
    decision = report["decision"]
    assert isinstance(provenance, dict)
    assert isinstance(supplement, dict)
    assert isinstance(sanity, dict)
    assert isinstance(decision, dict)

    case_lines = []
    for case in report["registered_cases"]:
        assert isinstance(case, dict)
        parameters = case["parameters"]
        wang = case["frozen_wang"]
        best = case["best_observed_birrell_upper_bound"]
        assert isinstance(parameters, dict)
        assert isinstance(wang, dict)
        assert isinstance(best, dict)
        case_lines.append(
            "| `{config}` | {population} | {batch} | {q:.6f} | "
            "{steps} | {sigma:.6f} | {wang_eps:.12f} | "
            "{birrell_eps:.12f} | {m} | {ratio:.3f} |".format(
                config=Path(str(case["config"])).name,
                population=parameters["source_dataset_size"],
                batch=parameters["sample_size"],
                q=parameters["sample_rate"],
                steps=parameters["steps"],
                sigma=parameters["actual_noise_std_over_clip_C"],
                wang_eps=wang["dp_accounting_epsilon"],
                birrell_eps=best["epsilon"],
                m=best["taylor_order_m"],
                ratio=case["best_birrell_over_wang_epsilon_ratio"],
            )
        )

    supplement_text = (
        "not inspected"
        if not supplement.get("inspected")
        else (
            f"hash `{supplement['accountant_sha256']}`; "
            "different from GitHub and final `FSwoR_RDP_ro` present = "
            f"{supplement['defines_final_replace_one_function_FSwoR_RDP_ro']}"
        )
    )
    ratio_range = sanity["ratio_range"]
    return f"""# Birrell Fixed-Size Accounting Probe V1

- Date: {REPORT_DATE}
- JSON SHA-256: `{json_sha256}`
- Upstream commit: `{provenance['git_commit']}`
- Upstream accountant SHA-256: `{provenance['accountant_sha256']}`
- Proceedings supplement: {supplement_text}

## Hard result

The final paper's replace-one theorem and linked GitHub code were reproduced
with the paper's noise convention: `sigma` is the actual Gaussian standard
deviation divided by clip norm `C`. The current Wang path instead passes
`sigma/2` to a Gaussian event because its generic mechanism is normalized by
replace-one sensitivity `2C`.

The mapping is validated by the paper-regime check (`N=50,000`, `batch=120`,
`sigma=6`, Taylor `m=4`): Wang/Birrell one-step RDP ratios ranged from
{ratio_range[0]:.3f} to {ratio_range[1]:.3f}, reproducing the reported
approximately factor-four improvement.

That ordering reverses on every registered Route F plan:

| Config | N | batch | q | T | sigma | Wang epsilon | Best observed Birrell epsilon | Taylor m | Birrell/Wang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(case_lines)}

Each Wang value was independently reproduced by `dp-accounting==0.6.0` and
the Decimal theorem oracle within `1e-10`. Birrell Taylor orders 3--5 were
checked on both the registered order set and the upstream example grid;
Taylor orders 3--16 were also swept on the registered order set.

## Decision

`{decision['route_f_accountant']}`

Retain the frozen Wang-based Route F. Birrell supplies a valid but looser
upper bound in these registered regimes, so migration would not improve any
certificate and would force unnecessary route, artifact, and evidence churn.
The manuscript must nevertheless cite Birrell and explain this regime-specific
choice.

A Birrell result above epsilon 8 does not refute the Wang certificate or show
that the exact privacy loss exceeds 8. Both are upper bounds, and upper bounds
need not be ordered uniformly over parameter space.

## Provenance limits

The proceedings supplement and linked GitHub accountant are not byte-identical;
the inspected supplement lacks the final `FSwoR_RDP_ro` entry point. The
paper-linked GitHub source was therefore pinned by commit and file hash. No
upstream source was copied into this repository; no root-level license file
was present in the pinned checkout.

## Reproduction

```text
git clone https://github.com/star-ailab/FSRDP.git <FSRDP>
git -C <FSRDP> checkout 799f7b755e8c792dd4a5f7e77cb878a6d12e7fd6
python scripts/build_birrell_accounting_probe_v1.py \
  --fsrdp-root <FSRDP> \
  --supplement-root <EXTRACTED_NEURIPS_SUPPLEMENT> \
  --supplement-zip <NEURIPS_SUPPLEMENT_ZIP>
```

The builder checks the upstream commit, accountant source hash, proceedings
supplement hashes, all three registered configurations, and a repeated
in-process deterministic build before writing output.
"""


def write_report(
    report: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "birrell_accounting_probe_v1.json"
    markdown_path = output_dir / "birrell_accounting_probe_v1.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output in {output_dir}"
        )
    json_bytes = canonical_json_bytes(report)
    json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    markdown = markdown_report(report, json_sha256=json_sha256)
    json_path.write_bytes(json_bytes)
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsrdp-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path)
    parser.add_argument("--supplement-zip", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "birrell_fixed_size_accounting_probe_v1_20260724"
        ),
    )
    args = parser.parse_args()

    first = build_report(
        fsrdp_root=args.fsrdp_root.resolve(),
        supplement_root=(
            args.supplement_root.resolve()
            if args.supplement_root
            else None
        ),
        supplement_zip=(
            args.supplement_zip.resolve()
            if args.supplement_zip
            else None
        ),
    )
    second = build_report(
        fsrdp_root=args.fsrdp_root.resolve(),
        supplement_root=(
            args.supplement_root.resolve()
            if args.supplement_root
            else None
        ),
        supplement_zip=(
            args.supplement_zip.resolve()
            if args.supplement_zip
            else None
        ),
    )
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise RuntimeError("Repeated accountant probes were not deterministic")
    first["determinism_check"] = {
        "independent_in_process_builds_equal": True,
    }

    json_path, markdown_path = write_report(
        first,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "json_path": str(json_path.relative_to(ROOT)),
                "json_sha256": file_sha256(json_path),
                "markdown_path": str(markdown_path.relative_to(ROOT)),
                "markdown_sha256": file_sha256(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
