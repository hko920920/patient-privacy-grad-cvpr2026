"""CPU-only replay of public MoFit COCO scalar results; no model inference."""
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent
RESEARCH=OUT.parents[1]
SOURCE=RESEARCH/"code_sources/X12/Eval/mia_th_COCO.py"
PREDICTOR=RESEARCH/"code_sources/X12/Eval/calculate_pred.py"
NAMES=("get_ori_data","get_l_clidavg_last3","deal_data_weight_avg","get_th")
FILES={kind:RESEARCH/"code_sources/X12/Results/COCO"/f"COCO_{variant}_500images_{split}_t_[140].txt"
       for kind,variant,split in (("embedding_member","emb","train"),("embedding_nonmember","emb","test"),
                                 ("vlm_member","blip","train"),("vlm_nonmember","blip","test"))}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name,value):
    path=OUT/name
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write("\n")


def source_namespace():
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"),filename=str(SOURCE))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in NAMES]
    assert len(functions)==len(NAMES)
    namespace=dict(np=np,MetricName="---",Dataname="---")
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),"exec"),namespace)
    return namespace


def independent(member,nonmember):
    member=np.asarray(member,dtype=np.float64).ravel()
    nonmember=np.asarray(nonmember,dtype=np.float64).ravel()
    differences=member[:,None]-nonmember[None,:]
    rank_auc=float(np.mean((differences<0)+.5*(differences==0)))
    label=np.r_[np.ones(len(member)),np.zeros(len(nonmember))]
    sklearn_auc=float(roc_auc_score(label,-np.r_[member,nonmember]))
    assert math.isclose(rank_auc,sklearn_auc,abs_tol=1e-12)
    thresholds=np.r_[-np.inf,np.unique(np.r_[member,nonmember])]
    tpr=np.searchsorted(np.sort(member),thresholds,side="right")/len(member)
    fpr=np.searchsorted(np.sort(nonmember),thresholds,side="right")/len(nonmember)
    idx=np.where(fpr<=.01)[0][-1]
    best_accuracy=((tpr+1-fpr)/2).max()
    return dict(rank_auc=rank_auc,sklearn_auc=sklearn_auc,
        max_empirical_tpr_at_fpr_le_01=float(tpr[idx]),actual_fpr_at_that_point=float(fpr[idx]),
        threshold_at_that_point=float(thresholds[idx]) if np.isfinite(thresholds[idx]) else None,
        exact_threshold_best_asr=float(best_accuracy),auc_member_high_score="negative source score",
        convention="all distinct empirical thresholds; include ties; no interpolation")


def literal_grid_details(member,nonmember):
    m,n=np.asarray(member).ravel(),np.asarray(nonmember).ravel()
    lo,hi=min(float(m.min()),float(n.min())),max(float(m.max()),float(n.max()))
    grid=np.arange(lo,hi,(hi-lo)/2000)
    tpr=np.searchsorted(np.sort(m),grid,side="right")/len(m)
    fpr=np.searchsorted(np.sort(n),grid,side="right")/len(n)
    idx=np.where(fpr>=.01)[0]
    idx=int(idx[0]) if len(idx) else None
    return dict(grid_points=len(grid),first_fpr_ge_01_index=idx,
        first_fpr_ge_01_actual_fpr=float(fpr[idx]) if idx is not None else None,
        first_fpr_ge_01_tpr=float(tpr[idx]) if idx is not None else 0.,
        grid_first_roc=[float(fpr[0]),float(tpr[0])],grid_last_roc=[float(fpr[-1]),float(tpr[-1])])


def main():
    for name in ("protocol.json","summary.json","evidence.json","report.md"):
        if (OUT/name).exists():raise FileExistsError(OUT/name)
    began=time.perf_counter()
    input_hashes={str(path.resolve()):digest(path) for path in [*FILES.values(),SOURCE,PREDICTOR]}
    script_hash=digest(__file__)
    protocol=dict(schema="mofit-public-scalar-replay/v1",research_stage=2,
        question="Can the public scalar score/evaluation path be reproduced, and how does its ROC convention differ?",
        repository="https://github.com/JoonsungJeon/MoFit",commit="91e4b5edc153bac84b0b4209f70d1b2b94e653b2",
        original_functions_compiled_unchanged=list(NAMES),input_sha256=input_hashes,script_sha256=script_hash,
        gamma_grid=np.linspace(0,1,21).tolist(),threshold_grid_points_requested=2000,
        source_member_direction="source score lower means member; get_th counts member<=threshold",
        source_selection="maximize ASR over thresholds for each gamma; then maximize ASR, tie-break by grid AUC, first gamma on exact tie",
        source_scaling="RobustScaler fitted to the complete public member+nonmember fixture, separately for the two features",
        no_separate_fixture_test_split=True,patient_data_access=False,GPU_forward=0,backward=0,
        actual_model_inference=False,patient_baseline_success_claimed=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__))
    save("protocol.json",protocol)
    namespace=source_namespace()
    with contextlib.redirect_stdout(io.StringIO()):
        raw=namespace["get_ori_data"](*(str(FILES[k]) for k in FILES))
    assert [a.shape for a in raw]==[(500,3),(500,3),(500,6),(500,6)]
    assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in raw)
    fields=namespace["get_l_clidavg_last3"](*raw)
    m,n,vm,vn=[np.asarray(x,dtype=np.float64) for x in fields]
    scale1=RobustScaler().fit(np.concatenate((m,n),axis=0))
    scale2=RobustScaler().fit(np.concatenate((vm,vn),axis=0))
    m,n=scale1.transform(m),scale1.transform(n)
    vm,vn=scale2.transform(vm),scale2.transform(vn)
    rows=[];warning_messages=set()
    for gamma in np.linspace(0,1,21):
        member,nonmember=namespace["deal_data_weight_avg"](m,n,vm,vn,gamma)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            values=namespace["get_th"](member,nonmember,n_points=2000)
            warning_messages.update(str(w.message) for w in seen)
        threshold,asr,auc,tpr01,tpr001,maximum,minimum=values
        exact=independent(member,nonmember)
        grid=literal_grid_details(member,nonmember)
        assert math.isclose(float(tpr01),grid["first_fpr_ge_01_tpr"],abs_tol=1e-12)
        rows.append(dict(gamma=float(gamma),literal_best_threshold=float(np.asarray(threshold).item()),
            literal_best_asr=float(asr),literal_grid_auc=float(auc),literal_tpr_at_first_fpr_ge_01=float(tpr01),
            literal_tpr_at_first_fpr_ge_001=float(tpr001),**grid,**exact,
            rank_auc_minus_literal_grid_auc=exact["rank_auc"]-float(auc),
            proper_tpr_minus_literal_tpr=exact["max_empirical_tpr_at_fpr_le_01"]-float(tpr01)))
    best_asr=max(r["literal_best_asr"] for r in rows)
    candidates=[r for r in rows if r["literal_best_asr"]==best_asr]
    best=max(candidates,key=lambda r:r["literal_grid_auc"])
    toy_member=np.r_[np.full(30,-1.),np.zeros(40),np.ones(30)]
    toy_nonmember=np.r_[np.zeros(2),np.full(98,2.)]
    toy_grid=literal_grid_details(toy_member,toy_nonmember)
    toy_exact=independent(toy_member,toy_nonmember)
    assert toy_grid["first_fpr_ge_01_actual_fpr"]==.02
    assert toy_grid["first_fpr_ge_01_tpr"]==.7
    assert toy_exact["max_empirical_tpr_at_fpr_le_01"]==.3
    file_records=[]
    for (kind,path),array in zip(FILES.items(),raw):
        header=path.read_text(encoding="utf-8").splitlines()[0]
        file_records.append(dict(kind=kind,path=str(path),rows=array.shape[0],columns=array.shape[1],
            dtype=str(array.dtype),header=header,embedded_patient_or_image_ids=False,
            last_minus_first_vs_penultimate_max_abs=float(np.max(np.abs((array[:,-1]-array[:,0])-array[:,-2])))))
    evidence=dict(files=file_records,
        feature_columns=dict(embedding="column -2 = unconditional - fitted conditional = -h",
            vlm="negative of column -2 = VLM conditional - unconditional = v",
            combined_source_score="gamma*R(-h)+(1-gamma)*R(v); lower=member"),
        scaler_embedding=dict(center=scale1.center_.tolist(),scale=scale1.scale_.tolist()),
        scaler_vlm=dict(center=scale2.center_.tolist(),scale=scale2.scale_.tolist()),
        original_runtime_warnings=sorted(warning_messages),
        fpr_boundary_synthetic=dict(literal=toy_grid,proper=toy_exact),
        header_iteration_discrepancy="Embedding headers say OptimIter1000_iters1000; paper COCO default and shell say 1000+300; exact inference provenance not established.",
        alignment_limitation="No sample IDs in these scalar rows; preserve released row ordering, cannot independently prove cross-file identity alignment.",
        noise_scope="Released evaluator embedding path uses fixed rnd_noise_1.npy; VLM path uses separate global Noise. No inference/noise generation is performed here.")
    for path,value in input_hashes.items():assert digest(path)==value
    assert digest(__file__)==script_hash
    save("evidence.json",evidence)
    summary=dict(schema="mofit-public-scalar-replay-result/v1",status="PASS_PUBLIC_SCALAR_EVALUATOR_REPLAY",
        research_stage=2,protocol_sha256=digest(OUT/"protocol.json"),evidence_sha256=digest(OUT/"evidence.json"),
        input_files_unchanged=True,script_unchanged=True,rows_per_group=500,conditions=21,
        gamma_results=rows,literal_selected=best,source_only_gamma1=rows[-1],aux_only_gamma0=rows[0],
        paper_coco_table2=dict(asr=.88,auc=.9417,tpr_at_01=.47),
        selected_minus_paper=dict(asr=best["literal_best_asr"]-.88,auc=best["literal_grid_auc"]-.9417,
            tpr=best["literal_tpr_at_first_fpr_ge_01"]-.47),
        elapsed_seconds=time.perf_counter()-began,GPU_forward=0,backward=0,patient_data_access=False,
        inference_reproduced=False,patient_attack_success_claimed=False,
        limitations=[evidence["header_iteration_discrepancy"],evidence["alignment_limitation"],
            "Scalers, gamma and threshold use the entire public fixture as the released evaluator; this is not a held-out performance estimate.",
            "Proper empirical TPR<=1% on this fixture is descriptive, not an independently calibrated deployable threshold.",
            "Replaying scalar values does not validate image/gradient/embedding kernels, memory or runtime of a new patient implementation."])
    save("summary.json",summary)
    report=("# MoFit COCO public scalar replay\n\n"
        "Stage 2, public stored results only. No model inference, GPU, or patient data.\n\n"
        f"- Original evaluator selection: gamma={best['gamma']:.2f}, ASR={100*best['literal_best_asr']:.2f}%, "
        f"grid AUC={100*best['literal_grid_auc']:.4f}%, first-FPR>=1% TPR={100*best['literal_tpr_at_first_fpr_ge_01']:.2f}%.\n"
        f"- Independent rank AUC={100*best['rank_auc']:.4f}%; empirical max TPR at FPR<=1%="
        f"{100*best['max_empirical_tpr_at_fpr_le_01']:.2f}% (actual FPR={100*best['actual_fpr_at_that_point']:.2f}%).\n"
        "- Four released files each contain 500 numeric records; all input/source hashes unchanged.\n"
        "- Embedding header says 1000 embedding iterations; paper/shell default is 300. Sample IDs are absent.\n"
        "- Full-fixture parameter selection reproduces the published evaluator procedure; it is not held-out calibration.\n"
        "- This result verifies the stored score path, not original model inference or patient-U performance.\n")
    with (OUT/"report.md").open("x",encoding="utf-8") as f:f.write(report)
    print(json.dumps(dict(status=summary["status"],best=best,seconds=summary["elapsed_seconds"],GPU_forward=0)))


if __name__=="__main__":main()
