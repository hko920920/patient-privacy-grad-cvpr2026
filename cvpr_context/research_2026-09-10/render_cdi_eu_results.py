"""Export all predeclared, independently verified CDI E/U metrics; no new statistics.

The standalone figures preserve the contract method order and use fixed [0, 1]
colors. Real results are not opened until the independent verification PASS and
its hash bindings have been checked. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH = Path(__file__).resolve().parent
CODE = RESEARCH.parent.parent / "code_working"
DEFAULT_INPUT = CODE / "_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_e_cohort_v1/eu_analysis_v1"
DEFAULT_OUTPUT = RESEARCH / "cdi_eu_results_artifacts/v1"
DEFAULT_MARKDOWN = RESEARCH / "CDI_EU_COMPARISON_RESULTS.md"
CONTRACT = RESEARCH / "spec_sources/cdi_eu_analysis_contract_20260915.json"
CONTRACT_SHA = "bc2497d17b3110ec8b994eaa541d4eda00891ff5d30215a9aff26b7aa5f1db43"
OLD_CONTRACT = RESEARCH / "spec_sources/cdi_u_analysis_contract.json"
OLD_CONTRACT_SHA = "6f0afa62cf2bbf26790370965c4716f3afc927e0ab2412f8c4e6befbe02efca3"
VERIFIER = CODE / "u_patient_audit/verify_cdi_eu_analysis.py"
CELLS = (("E", "E"), ("E", "U"), ("U", "E"), ("U", "U"))
MODELS = ("model_1", "model_2")
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
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    require(bool(rows), "CSV must contain records")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_inputs(input_dir):
    """Verify status and content bindings before reading analysis metric values."""
    verification = read(input_dir / "verification.json")
    require(verification["status"] == "PASS_EU_SAVED_PREDICTIONS_FIT_AND_STATISTICS", "Independent E/U verification must PASS")
    require(verification["AUC_metrics"] == 144 and verification["within_cell_contrasts"] == 48
            and verification["cross_cell_contrasts"] == 108, "Unexpected verified scope")
    require(verification["U_parameters_CV_predictions_exact"] is True, "Exact original U restoration required")
    require(digest(VERIFIER) == verification["verification_code_sha256"], "Verifier source changed")
    require(digest(input_dir / "analysis.json") == verification["analysis_sha256"], "Verified analysis changed")
    require(digest(input_dir / "provenance.json") == verification["analysis_provenance_sha256"], "Verified provenance changed")
    provenance = read(input_dir / "provenance.json")
    require(digest(CONTRACT) == CONTRACT_SHA == provenance["analysis_contract_sha256"], "Frozen E/U contract changed")
    require(digest(OLD_CONTRACT) == OLD_CONTRACT_SHA, "Original method contract changed")
    require(digest(input_dir / "protocol.json") == provenance["protocol_sha256"], "Analysis protocol changed")
    for name, expected in provenance["output_sha256"].items():
        require(digest(input_dir / name) == expected, "Analysis output changed: " + name)
    contract, old = read(CONTRACT), read(OLD_CONTRACT)
    require(contract == read(input_dir / "contract_copy.json"), "Analysis contract copy differs")
    require(contract["methods"] == old["methods"], "Predeclared original method order changed")
    require([(c["fit_scenario"], c["evaluation_scenario"]) for c in contract["cells"]] == list(CELLS), "Four-cell order changed")
    require(contract["models"] == list(MODELS) and len(contract["methods"]) == 18, "Reporting scope changed")
    sources = [input_dir / n for n in ("verification.json", "provenance.json", "protocol.json", *provenance["output_sha256"])]
    sources += [CONTRACT, OLD_CONTRACT, VERIFIER]
    return contract, {str(p.resolve()): digest(p) for p in sources}


def flatten(analysis, contract):
    methods = [m["id"] for m in contract["methods"]]
    require(set(methods) == set(LABELS) and len(methods) == 18, "Method label inventory differs")
    index = {(r["model"], r["method"], r["fit_scenario"], r["evaluation_scenario"]): r for r in analysis["results"]}
    expected = {(model, method, fs, es) for model in MODELS for method in methods for fs, es in CELLS}
    require(len(analysis["results"]) == len(index) == 144 and set(index) == expected, "Expected all 144 unique AUC rows")
    auc_rows = []
    for model in MODELS:
        for ordinal, method in enumerate(methods, 1):
            for fs, es in CELLS:
                r = index[model, method, fs, es]
                auc, ci = r["selection_patient_AUC"], r["selection_patient_AUC_CI95"]
                require(all(math.isfinite(x) and 0 <= x <= 1 for x in [auc, *ci]) and ci[0] <= ci[1], "Invalid AUC or interval")
                require(r["selection_patients"] == 40 and r["selection_members"] == 20 and r["selection_nonmembers"] == 20, "Patient counts differ")
                auc_rows.append(dict(model=model, method_order=ordinal, method=method, method_label=LABELS[method],
                    fit_scenario=fs, evaluation_scenario=es, selection_patient_AUC=auc,
                    CI95_low=ci[0], CI95_high=ci[1], selection_patients=40, selection_members=20,
                    selection_nonmembers=20, fit_patient_AUC_descriptive_only=r["fit_patient_AUC_descriptive_only"]))
    contrast_rows = []
    for field, definitions, count in (("within_cell_contrasts", contract["original_within_cell_contrasts"], 48),
                                      ("cross_cell_contrasts", contract["cross_cell_contrasts"], 108)):
        definitions = {d["id"]: d for d in definitions}
        rows = analysis[field]
        require(len(rows) == count, "Wrong contrast count")
        seen = set()
        for r in rows:
            d = definitions[r["id"]]
            require(all(r[k] == v for k, v in d.items()), "Contrast differs from predeclared definition")
            if field == "within_cell_contrasts":
                fs, es = r["fit_scenario"], r["evaluation_scenario"]
                am, bm, af, ae, bf, be = r["a"], r["b"], fs, es, fs, es
                key = (r["model"], fs, es, r["id"])
            else:
                am = bm = r["method"]
                af, ae = r["a"]["fit_scenario"], r["a"]["evaluation_scenario"]
                bf, be = r["b"]["fit_scenario"], r["b"]["evaluation_scenario"]
                key = (r["model"], r["method"], r["id"])
            require(key not in seen, "Duplicate contrast")
            seen.add(key)
            delta, ci = r["selection_delta_AUC"], r["paired_CI95"]
            require(all(math.isfinite(x) and -1 <= x <= 1 for x in [delta, *ci]) and ci[0] <= ci[1], "Invalid contrast interval")
            require(abs(delta - (index[r["model"], am, af, ae]["selection_patient_AUC"] - index[r["model"], bm, bf, be]["selection_patient_AUC"])) < 1e-12, "Reported contrast arithmetic differs")
            contrast_rows.append(dict(contrast_type=field, model=r["model"], contrast_id=r["id"],
                role=r.get("role", "predeclared_cross_cell"), a_method=am, a_fit_scenario=af, a_evaluation_scenario=ae,
                b_method=bm, b_fit_scenario=bf, b_evaluation_scenario=be,
                selection_delta_AUC=delta, paired_CI95_low=ci[0], paired_CI95_high=ci[1]))
    require(len(contrast_rows) == 156, "All 156 contrasts required")
    return auc_rows, contrast_rows, methods


def draw_heatmap(auc_rows, methods, output):
    values = {(r["model"], r["method"], r["fit_scenario"], r["evaluation_scenario"]): r["selection_patient_AUC"] for r in auc_rows}
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 10.2), sharey=True)
    fig.subplots_adjust(left=.285, right=.945, top=.88, bottom=.16, wspace=.13)
    cmap = plt.get_cmap("viridis")
    for ax, model in zip(axes, MODELS):
        matrix = np.asarray([[values[model, method, fs, es] for fs, es in CELLS] for method in methods])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(model.replace("_", " ").title(), fontsize=14, pad=13)
        ax.set_xticks(range(4), [f"{fs}-fit\n{es}-eval" for fs, es in CELLS], fontsize=11)
        ax.set_yticks(range(18), [LABELS[m] for m in methods], fontsize=10)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 18, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.6, alpha=.7)
        ax.tick_params(which="minor", length=0)
        for i in range(18):
            for j in range(4):
                rgb = cmap(matrix[i, j])[:3]
                luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=10.5,
                        color="black" if luminance > .54 else "white")
    colorax = fig.add_axes([.965, .225, .012, .565])
    fig.colorbar(im, cax=colorax, ticks=np.arange(0, 1.01, .2), label="Patient AUC")
    fig.suptitle("CDI patient comparison: four fixed fit / evaluation cells", fontsize=17, y=.97)
    fig.text(.5, .925, "All 18 predeclared methods; 40 selection patients per target (20 members / 20 nonmembers)", ha="center", fontsize=11)
    fig.text(.285, .08, "E: designated training-image pair (exposed only in the participating target).\n"
             "U: different same-patient pair, never exposed in either target's additional training.\n"
             "Fixed color scale 0–1. Full conditional 95% bootstrap intervals and all contrasts are in CSV.\n"
             "Development comparison; no selection of best methods, no causal claim, no original CDI set-test claim.", fontsize=10, va="center")
    fig.savefig(output / "patient_auc_four_cells.png", dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(output / "patient_auc_four_cells.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def markdown(auc_rows, methods):
    ix = {(r["model"], r["method"], r["fit_scenario"], r["evaluation_scenario"]): r for r in auc_rows}
    lines = ["# CDI E/U 환자 비교 결과 — 검산된 표", "", "현재 연구 단계는 **2번: 가까운 기존 방법의 실패 조건·원인 확인**이다. 이 문서는 검산된 수치 전체를 제시하는 표 초안이며, 결과의 핵심 해석은 연구 책임자의 별도 검토로 완성한다.", "",
             "fit80에서 학습한 scorer를 동일 selection40 환자(각 target의 member20/nonmember20)에 적용했다. 각 환자는 조건별 두 장이다. E는 해당 target에 환자가 참여하면 실제 추가학습에 노출된 영상이고, U는 두 target의 추가학습에 쓰이지 않은 같은 환자의 다른 영상이다. 기존 U-fit scorer·C·예측은 그대로 복원했다.", "",
             "표의 값은 AUC [조건부 95% CI]이다. 모든 18개 방법을 사전 계약 순서로 표시한다. CI는 동일 2,000회 환자 bootstrap이며, selection 자료 재사용 및 다중 대조를 보정한 확증적 구간이 아니다. scalar 방법은 fitting이 없으므로 같은 평가 조건의 fit 방향 두 표기가 반복된다.", "",
             "![두 target의 네 셀 AUC](cdi_eu_results_artifacts/v1/patient_auc_four_cells.png)", "",
             "[AUC·CI 전체 144행 CSV](cdi_eu_results_artifacts/v1/auc_all.csv) · [차이·paired CI 전체 156행 CSV](cdi_eu_results_artifacts/v1/contrasts_all.csv) · [SVG](cdi_eu_results_artifacts/v1/patient_auc_four_cells.svg) · [해시 manifest](cdi_eu_results_artifacts/v1/manifest.json)", ""]
    for model in MODELS:
        lines += [f"## {model}", "", "| 사전 지정 방법 | E-fit → E-eval | E-fit → U-eval | U-fit → E-eval | U-fit → U-eval |", "|---|---:|---:|---:|---:|"]
        for method in methods:
            cells = []
            for fs, es in CELLS:
                r = ix[model, method, fs, es]
                cells.append(f"{r['selection_patient_AUC']:.4f} [{r['CI95_low']:.4f}, {r['CI95_high']:.4f}]")
            lines.append("| " + LABELS[method] + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 범위와 후속 해석", "", "- E→U 차이는 같은 scorer의 영상 조건 전이이며, U-fit 회복은 fit 환자 자료에서의 scorer 적응을 포함한다.",
              "- 검산 PASS는 저장 특징·예측·통계의 정합성이다. 참여 효과의 인과적 확인이나 새 공격의 기여를 뜻하지 않는다.",
              "- 원형 CDI reference-set 검정, 실제 MoFit 환자 비교와 남는 실패 원인 설명은 별도 작업이다.",
              "- **핵심 해석: 연구 책임자가 검토 후 이 초안을 완성한다.**", ""]
    return "\n".join(lines)


def run(input_dir, output, md_path, expected_sha):
    require(digest(__file__) == expected_sha, "Provide the frozen renderer SHA256")
    require(not output.exists() and not md_path.exists(), "Immutable outputs already exist")
    contract, hashes = verified_inputs(input_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    write_json(output / "protocol.json", dict(schema="cdi-eu-reporting-protocol/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        renderer_path=str(Path(__file__).resolve()), renderer_sha256=expected_sha, input_sha256=hashes,
        output_dir=str(output.resolve()), markdown_path=str(md_path.resolve()), methods=[m["id"] for m in contract["methods"]],
        cells=[dict(fit_scenario=f, evaluation_scenario=e) for f, e in CELLS], models=list(MODELS),
        figure_color_limits=[0, 1], new_statistics=False, fitting=False, GPU_calls=0, best_method_selection=False))
    analysis = read(input_dir / "analysis.json")
    auc_rows, contrasts, methods = flatten(analysis, contract)
    write_csv(output / "auc_all.csv", auc_rows)
    write_csv(output / "contrasts_all.csv", contrasts)
    draw_heatmap(auc_rows, methods, output)
    draft = markdown(auc_rows, methods)
    with (output / "comparison_table_draft.md").open("x", encoding="utf-8") as f:
        f.write(draft)
    with md_path.open("x", encoding="utf-8") as f:
        f.write(draft)
    for p, sha in hashes.items():
        require(digest(p) == sha, "Input changed while reporting: " + p)
    require(digest(__file__) == expected_sha, "Renderer changed while reporting")
    paths = [output / name for name in ("protocol.json", "auc_all.csv", "contrasts_all.csv", "patient_auc_four_cells.png", "patient_auc_four_cells.svg", "comparison_table_draft.md")]
    write_json(output / "manifest.json", dict(schema="cdi-eu-reporting-manifest/v1", status="EXPORTED_VERIFIED_METRICS_WITHOUT_NEW_STATISTICS",
        renderer_sha256=expected_sha, input_sha256=hashes, AUC_rows=144, contrast_rows=156,
        figure_panels=2, methods_per_panel=18, cells_per_method=4, method_order=methods,
        cells=[list(c) for c in CELLS], color_limits=[0, 1],
        output_sha256={p.name:digest(p) for p in paths},
        markdown_handoff=dict(path=str(md_path.resolve()), initial_draft_sha256=digest(md_path),
            editable_by_root=True, immutable_copy="comparison_table_draft.md"),
        stage2_complete=False, causal_claim=False, new_statistics=False))
    print(json.dumps(dict(status="EXPORTED_VERIFIED_METRICS_WITHOUT_NEW_STATISTICS", AUC_rows=144,
        contrast_rows=156, output=str(output), markdown=str(md_path)), ensure_ascii=False))


def self_test():
    # Test order, all-cell coverage, CSV preservation and a real synthetic PNG/SVG.
    contract = read(CONTRACT)
    require(digest(CONTRACT) == CONTRACT_SHA and digest(OLD_CONTRACT) == OLD_CONTRACT_SHA, "Frozen contracts")
    require(contract["methods"] == read(OLD_CONTRACT)["methods"], "Exact original method order")
    methods = [m["id"] for m in contract["methods"]]
    results = []
    for mi, model in enumerate(MODELS):
        for j, method in enumerate(methods):
            for k, (fs, es) in enumerate(CELLS):
                value = (mi * 72 + j * 4 + k) / 143
                results.append(dict(model=model, method=method, fit_scenario=fs, evaluation_scenario=es,
                    selection_patient_AUC=value, selection_patient_AUC_CI95=[max(0,value-.03), min(1,value+.03)],
                    selection_patients=40, selection_members=20, selection_nonmembers=20, fit_patient_AUC_descriptive_only=.5))
    ix = {(r["model"],r["method"],r["fit_scenario"],r["evaluation_scenario"]):r["selection_patient_AUC"] for r in results}
    within, cross = [], []
    for model in MODELS:
        for fs, es in CELLS:
            for d in contract["original_within_cell_contrasts"]:
                delta = ix[model,d["a"],fs,es]-ix[model,d["b"],fs,es]
                within.append(dict(d,model=model,fit_scenario=fs,evaluation_scenario=es,selection_delta_AUC=delta,paired_CI95=[delta,delta]))
        for method in methods:
            for d in contract["cross_cell_contrasts"]:
                a,b=d["a"],d["b"]
                delta=ix[model,method,a["fit_scenario"],a["evaluation_scenario"]]-ix[model,method,b["fit_scenario"],b["evaluation_scenario"]]
                cross.append(dict(d,model=model,method=method,selection_delta_AUC=delta,paired_CI95=[delta,delta]))
    rows, contrasts, order = flatten(dict(results=list(reversed(results)), within_cell_contrasts=within, cross_cell_contrasts=cross), contract)
    require(order == methods and rows[0]["selection_patient_AUC"] == 0 and rows[-1]["selection_patient_AUC"] == 1, "Ordering or endpoints changed")
    with tempfile.TemporaryDirectory(prefix="cdi_eu_renderer_synthetic_") as tmp:
        temp = Path(tmp)
        write_csv(temp / "auc.csv", rows)
        require(len(list(csv.DictReader((temp / "auc.csv").open(encoding="utf-8-sig")))) == 144, "CSV row count")
        draw_heatmap(rows, order, temp)
        require((temp / "patient_auc_four_cells.png").stat().st_size > 10000 and (temp / "patient_auc_four_cells.svg").stat().st_size > 10000, "Real synthetic figure export")
        require(markdown(rows, methods).count("| Patient mean+max52 LR (C=1) |") == 2, "All-model Markdown table")
    print("PASS_SYNTHETIC_144_AUC_156_CONTRASTS_ORDER_CSV_PNG_SVG; no real results opened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--expected-code-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.input_dir, args.output_dir, args.markdown, args.expected_code_sha256)
