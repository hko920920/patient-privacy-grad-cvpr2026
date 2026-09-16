"""Independent CPU tensor, provenance and statistical verification of SecMI screen."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT, RUN, digest, read_csv, verify_inputs, write_json
from .secmi_analysis import analyse

MODELS = ("model_1", "model_2")
OLD = RUN/"probe_smoke/training_coverage_v2/U_step_1000_n8"
REFERENCE = RUN/"verification_20260914/endpoint_precision_U8_v1"
STEPS = [(t,t+10) for t in range(0,100,10)]+[(100,110),(110,100)]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)


def norm64(tensor):
    return float(torch.linalg.vector_norm(tensor.double().reshape(-1)))


def norm_agreement(runtime, reference, size=4096):
    # Conservative standard float32 accumulated-rounding envelope, not a performance tolerance.
    unit=np.finfo(np.float32).eps/2
    gamma=(size+2)*unit/(1-(size+2)*unit)
    bound=(gamma+4*unit)*abs(reference)+8*np.finfo(np.float32).tiny
    assert abs(runtime-reference)<=bound,(runtime,reference,bound)
    return bound


def independent_statistics(analysis, records):
    rows={(r["model"],r["scenario"],r["eval_role"],str(r["patient_id"])):r for r in records}
    order=analysis["bootstrap"]["patient_order"]
    assert len(order)==40
    gen=np.random.Generator(np.random.PCG64(260914))
    a=gen.integers(0,20,size=(2000,20),dtype=np.int64)
    b=gen.integers(20,40,size=(2000,20),dtype=np.int64)
    draws=np.concatenate((a,b),axis=1)
    assert hashlib.sha256(draws.tobytes()).hexdigest()==analysis["bootstrap"]["indices_sha256"]
    ca=np.stack([np.bincount(v,minlength=20) for v in a]).astype(float)
    cb=np.stack([np.bincount(v-20,minlength=20) for v in b]).astype(float)
    for role,n in (("fit",8),("selection",40)):
        ids=sorted({p for m,s,r,p in rows if r==role})
        assert len(ids)==n
        for scenario in ("E","U"):
            for score in ("score_mean","score_max"):
                for model in MODELS:
                    selected=[rows[model,scenario,role,p] for p in ids]
                    values=[r[score] for r in selected]
                    labels=[r["member"] for r in selected]
                    # Average ranks, independently from pairwise helper implementation.
                    ranks=[]
                    for v in values:
                        ranks.append(1+sum(x<v for x in values)+(sum(x==v for x in values)-1)/2)
                    pos=sum(labels);neg=n-pos
                    au=(sum(rank for rank,label in zip(ranks,labels) if label)-pos*(pos+1)/2)/(pos*neg)
                    target=analysis["metrics"][role][scenario][score]["models"][model]
                    close(target["auc"],au)
                    if role=="selection":
                        vals=np.array([rows[model,scenario,role,p][score] for p in order])
                        matrix=(vals[:20,None]>vals[None,20:]).astype(float)+.5*(vals[:20,None]==vals[None,20:])
                        if model=="model_2":matrix=1-matrix
                        # Bootstrap count weights, independently from helper's expanded-resample array.
                        boot=np.einsum("bi,ij,bj->b",ca,matrix,cb,optimize=True)/400
                        boot.sort()
                        ci=[]
                        for q in (.025,.975):
                            index=q*1999;low=math.floor(index);high=math.ceil(index)
                            ci.append(float(boot[low]+(index-low)*(boot[high]-boot[low])))
                        for x,y in zip(ci,target["patient_bootstrap_percentile_95"]):close(x,y)
    for scenario in ("E","U"):
        stats=analysis["metrics"]["selection"][scenario]["score_mean"]["models"]
        expected=all(stats[m]["patient_bootstrap_percentile_95"][0]>.5 for m in MODELS)
        assert analysis["screening_decisions"][scenario]["screening_signal_supported_in_this_setting"]==expected


def verify(directory):
    started=time.perf_counter();torch.set_num_threads(2);verify_inputs()
    protocol=load(directory/"protocol.json");report=load(directory/"report.json")
    images=load(directory/"image_scores.json");patients=load(directory/"patient_scores.json")
    assert report["protocol_sha256"]==digest(directory/"protocol.json")
    assert report["image_scores_sha256"]==digest(directory/"image_scores.json")
    assert report["patient_scores_sha256"]==digest(directory/"patient_scores.json")
    expected=dict(schema="u-secmi-screen/v1",precision="fp32",batch_size=1,t0=0,t=100,step=10,
                  target_only=True,generic_prompt="a frontal chest radiograph",fit_patients=8,
                  selection_patients=40,unique_patients=48,calibration_or_test_used=False,
                  model_training_steps=0,attack_fitting=False,expected_image_records=384,
                  expected_patient_records=192,forward_per_image=12,total_forward=4608,total_backward=0)
    for key,value in expected.items():assert protocol[key]==value,key
    assert protocol["bootstrap"]["seed"]==260914 and protocol["bootstrap"]["replicates"]==2000
    assert protocol["primary_patient_score"]=="mean of two image scores"
    assert protocol["secondary_patient_score"]=="max of two image scores; descriptive only"
    assert protocol["models"]==list(MODELS) and protocol["scenarios"]==["E","U"]
    for filename,sha in protocol["input_sha256"].items():
        path=(ROOT/filename).resolve();assert path.is_relative_to(ROOT.resolve()) and digest(path)==sha,filename
    for filename,sha in protocol["code_sha256"].items():
        path=(ROOT/"u_patient_audit"/filename).resolve()
        assert path.is_relative_to((ROOT/"u_patient_audit").resolve()) and digest(path)==sha,filename
    assert set(protocol["code_sha256"])=={"run_secmi_diagnostic.py","secmi_adapter.py","models.py","common.py"}
    protected=protocol["original_protected_sha256"]
    assert len(protected)==11
    for filename,sha in protected.items():assert digest(ROOT/filename)==sha,filename
    source=protocol["source"];source_root=Path(source["source_root"]).resolve()
    assert source["repository"]=="https://github.com/sprintml/copyrighted_data_identification"
    assert source["commit"]=="dcd62258b0b3fde05d52aaecfade3b5f4c09507a"
    source_hashes={
        "src/attacks/features_extraction/secmi.py":"4e4f36d04f0cfe7f6dcab4812d66be4ed65680fbbb92910927d6b5bce85de9c3",
        "src/attacks/scores_computation/secmi.py":"10f1d642639d30f9f14f15a9d7657d819413a9ce077b68ff7aa7bcc4aee74dc2",
        "conf/attack/secmi_stat.yaml":"aa188200f526bc4f2c5c4fac3e92f3541636d8af1f809de9eb882e95d25fb6ad"}
    assert source["files_sha256"]==source_hashes
    for filename,sha in source_hashes.items():
        path=(source_root/filename).resolve()
        assert path.is_relative_to(source_root) and digest(path)==sha,filename
    assert source["members_lower"] is True and source["raw_discrepancy"]=="Euclidean L2 norm, not squared sum or mean"
    assert protocol["conformance"]["status"]=="PASS_SYNTHETIC_CPU_CONFORMANCE"
    assert protocol["conformance"]["source_bindings"]==source and protocol["conformance"]["GPU_executions"]==0
    assert set(protocol["conformance"]["cases"])=={"epsilon","v_prediction"}
    for case in protocol["conformance"]["cases"].values():
        assert case["forward_score"]==12 and case["backward"]==0 and case["weights_unchanged"] is True
        assert case["stage_pairs"]==[list(p) for p in STEPS]
    cache_path=RUN/"cache/cache.pt"
    assert digest(cache_path)==protocol["cache_sha256"]
    cache_meta=load(RUN/"cache/summary.json")
    assert cache_meta["cache_sha256"]==protocol["cache_sha256"]
    assert cache_meta["cohort_lock_sha256"]==digest(RUN/"cohort/lock.json")
    cache=torch.load(cache_path,map_location="cpu",weights_only=True)
    assert cache["audit_prompt"]==protocol["generic_prompt"]
    # Independent reconstruction of the pinned scheduler's scaled-linear beta schedule.
    config_path=Path.home()/".cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c/scheduler/scheduler_config.json"
    cfg=load(config_path)
    assert cfg["beta_schedule"]=="scaled_linear" and cfg["trained_betas"] is None
    betas=torch.linspace(cfg["beta_start"]**.5,cfg["beta_end"]**.5,cfg["num_train_timesteps"],dtype=torch.float32).square()
    alphas=torch.cumprod(1-betas,dim=0)
    old=load(OLD/"results.json");old_check=load(OLD/"verification.json")
    assert old_check["status"]=="PASS" and old_check["results_sha256"]==digest(OLD/"results.json")
    fit=[str(r["patient_id"]) for r in old if r["model"]=="model_1"]
    manifest=read_csv(RUN/"cohort/evaluation_images.csv")
    by_id={r["image_id"]:r for r in manifest};by_patient=defaultdict(list)
    for row in manifest:by_patient[row["patient_id"]].append(row)
    selection=sorted({r["patient_id"] for r in manifest if r["eval_role"]=="selection"},key=int)
    assert len(set(fit))==8 and len(selection)==40 and not set(fit)&set(selection)
    assert protocol["patient_order_by_role"]=={"fit":fit,"selection":selection}
    expected_order=[]
    for role,ids in (("fit",fit),("selection",selection)):
        assert Counter(by_patient[p][0]["assignment_group"] for p in ids)=={"A":len(ids)//2,"B":len(ids)//2}
        for patient in ids:
            rows=by_patient[patient];assert len(rows)==4 and all(r["eval_role"]==role for r in rows)
            for scenario,record_role in (("E","train_candidate"),("U","U_observed")):
                ids2=sorted(r["image_id"] for r in rows if r["record_role"]==record_role)
                assert len(ids2)==2
                expected_order.extend((scenario,role,patient,image) for image in ids2)
    assert [(r["model"],r["scenario"],r["eval_role"],str(r["patient_id"]),r["image_id"]) for r in images]==[
        (model,*entry) for model in MODELS for entry in expected_order]
    exposure={};training={}
    for model in MODELS:
        path=RUN/"training_coverage_v2"/model/"step_1000.pt"
        assert digest(path)==protocol["checkpoint_sha256"][model]
        state=torch.load(path,map_location="cpu",weights_only=True)
        assert state["step"]==1000 and state["contract"]==load(path.parent.parent/"protocol.json")
        exposure[model]=dict(state["exposures"])
        training[model]={r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
        assert set(exposure[model])==training[model] and min(exposure[model].values())>=1
        assert sum(exposure[model].values())==4000
        del state
    groups=defaultdict(list);norm_differences=[];tensor_hashes={}
    for row in images:
        model,patient,image_id=row["model"],str(row["patient_id"]),row["image_id"]
        original=by_id[image_id]
        for field in ("patient_id","eval_role","assignment_group","record_role"):assert row[field]==original[field]
        assert row["eval_role"] in ("fit","selection")
        assert row["record_role"]==("train_candidate" if row["scenario"]=="E" else "U_observed")
        member=int(original["assignment_group"]==("A" if model=="model_1" else "B"))
        assert row["member"]==member
        assert member==int(any(exposure[model].get(r["image_id"],0)>0 for r in by_patient[patient]))
        image_member=int(member and row["scenario"]=="E")
        assert row["image_member"]==image_member==int(exposure[model].get(image_id,0)>0)
        assert row["actual_training_exposures"]==exposure[model].get(image_id,0)
        assert row["checkpoint_sha256"]==protocol["checkpoint_sha256"][model]
        assert row["forward"]==12 and row["backward"]==0
        path=(directory/row["tensor_path"]).resolve()
        assert path.is_relative_to(directory.resolve()) and digest(path)==row["tensor_sha256"]
        assert path not in tensor_hashes;tensor_hashes[path]=row["tensor_sha256"]
        tensor=torch.load(path,map_location="cpu",weights_only=True)
        assert set(tensor)=={"z_det","z_recon","image_id","model"}
        assert tensor["image_id"]==image_id and tensor["model"]==model
        det,recon=tensor["z_det"],tensor["z_recon"]
        assert det.dtype==recon.dtype==torch.float32 and det.shape==recon.shape==(1,4,32,32)
        assert torch.isfinite(det).all() and torch.isfinite(recon).all()
        independent=norm64(det.double()-recon.double())
        assert row["l2"]>=0 and math.isfinite(row["l2"])
        bound=norm_agreement(row["l2"],independent)
        close(row["score"],-row["l2"])
        stages=row["stages"];assert len(stages)==12
        assert [(s["timestep"],s["target_timestep"]) for s in stages]==STEPS
        for i,stage in enumerate(stages):
            assert stage["stage"]==i
            close(stage["alpha_input"],float(alphas[stage["timestep"]]))
            close(stage["alpha_target"],float(alphas[stage["target_timestep"]]))
            assert stage["prediction_type"]==cfg["prediction_type"]
            assert stage["epsilon_conversion"]==("identity" if cfg["prediction_type"]=="epsilon" else "sqrt(alpha)*v + sqrt(1-alpha)*x_t")
            for key in ("input_l2","output_l2","input_max_abs","output_max_abs","prediction_l2","epsilon_l2"):
                assert math.isfinite(stage[key]) and stage[key]>=0
            if i:
                close(stage["input_l2"],stages[i-1]["output_l2"])
                close(stage["input_max_abs"],stages[i-1]["output_max_abs"])
        norm_agreement(stages[0]["input_l2"],norm64(cache["latents"][image_id]))
        close(stages[0]["input_max_abs"],float(cache["latents"][image_id].abs().max()))
        norm_agreement(stages[9]["output_l2"],norm64(det))
        norm_agreement(stages[11]["output_l2"],norm64(recon))
        close(stages[9]["output_max_abs"],float(det.abs().max()))
        close(stages[11]["output_max_abs"],float(recon.abs().max()))
        norm_differences.append(dict(model=model,image_id=image_id,runtime_l2=row["l2"],CPU_float64_l2=independent,
                                     difference=row["l2"]-independent,rounding_envelope=bound))
        groups[model,row["scenario"],row["eval_role"],patient].append((row,independent))
    del cache
    normalized=[];double_records=[]
    emitted={(r["model"],r["scenario"],r["eval_role"],str(r["patient_id"])):r for r in patients}
    assert len(emitted)==len(patients)==len(groups)==192
    for key,values in groups.items():
        assert len(values)==2
        model,scenario,role,patient=key
        score_mean=math.fsum(v[0]["score"] for v in values)/2
        score_max=max(v[0]["score"] for v in values)
        result=dict(model=model,scenario=scenario,eval_role=role,patient_id=patient,
                    assignment_group=values[0][0]["assignment_group"],member=values[0][0]["member"],
                    image_ids=sorted(v[0]["image_id"] for v in values),score_mean=score_mean,score_max=score_max)
        actual=emitted[key]
        for field in ("assignment_group","member","image_ids"):assert actual[field]==result[field]
        close(actual["score_mean"],score_mean);close(actual["score_max"],score_max)
        normalized.append(result)
        double_records.append(dict(result,score_mean=-math.fsum(v[1] for v in values)/2,score_max=max(-v[1] for v in values)))
    ref_report=load(REFERENCE/"report.json");ref_rows=load(REFERENCE/"results.json")
    assert ref_report["results_sha256"]==digest(REFERENCE/"results.json")
    reference={(r["model"],str(r["patient_id"])):r["fp32_scores"]["response"] for r in ref_rows}
    assert set(reference)=={(m,p) for m in MODELS for p in fit}
    analysis=analyse(normalized,reference)
    independent_statistics(analysis,normalized)
    double_analysis=analyse(double_records,reference)
    sensitivity=[]
    for role in ("fit","selection"):
        for scenario in ("E","U"):
            for score in ("score_mean","score_max"):
                for model in MODELS:
                    before=analysis["metrics"][role][scenario][score]["models"][model]
                    after=double_analysis["metrics"][role][scenario][score]["models"][model]
                    sensitivity.append(dict(role=role,scenario=scenario,score=score,model=model,
                                            runtime_auc=before["auc"],float64_recomputed_auc=after["auc"],
                                            AUC_changed=before["auc"]!=after["auc"]))
    gates_changed={s:analysis["screening_decisions"][s]["screening_signal_supported_in_this_setting"]!=
                      double_analysis["screening_decisions"][s]["screening_signal_supported_in_this_setting"] for s in ("E","U")}
    analysis["raw_norm_precision_check"]=dict(
        formula="CPU float64 Euclidean norm of exact persisted FP32 tensor difference",
        rounding_envelope="gamma_(4098)+4u relative; conservative standard float32 accumulation bound, not an efficacy threshold",
        image_records=len(norm_differences),max_absolute_difference=max(abs(r["difference"]) for r in norm_differences),
        all_within_rounding_envelope=True,per_image=norm_differences,AUC_sensitivity=sensitivity,
        float64_primary_screening_decisions=double_analysis["screening_decisions"],screening_decision_changed=gates_changed)
    analysis["upstream_scope"]="Port of the CDI repository's SecMI-stat implementation; not a full reproduction of the original SecMI authors' experiments."
    assert report["status"]=="COMPLETED_SECMI_EXECUTION_REQUIRES_INDEPENDENT_ANALYSIS"
    assert report["unique_patients"]==48 and report["fit_patients"]==8 and report["selection_patients"]==40
    assert report["image_records"]==len(images)==384 and report["patient_records"]==len(patients)==192
    assert report["total_forward"]==sum(r["forward"] for r in images)==4608 and report["total_backward"]==0
    assert report["training_updates"]==0 and report["calibration_or_test_used"] is False and report["attack_success_declared"] is False
    assert report["original_protected_files_unchanged"]==11
    assert [m["model"] for m in report["model_reports"]]==list(MODELS)
    for model in report["model_reports"]:
        assert model["forward"]==2304 and model["backward"]==0 and model["weights_unchanged"] is True
        assert model["adapter_tensors_checked"]==256 and model["prediction_type"]==cfg["prediction_type"]
        assert model["checkpoint_sha256"]==protocol["checkpoint_sha256"][model["model"]]
    write_json(directory/"analysis.json",analysis)
    verification=dict(status="PASS_INTEGRITY_AND_STATISTICAL_RECOMPUTATION",
        raw_tensor_pairs_checked=384,image_L2_scores_checked=384,patient_scores_checked=192,
        patient_metrics_checked=384,bootstrap_replicates=2000,bootstrap_independent_count_weight_recomputation=True,
        primary_gate_E_U_separate=True,primary_score_mean_only=True,protected_original_files_unchanged=11,
        forward_accounted=4608,backward=0,GPU_execution_by_verifier=False,
        freeze_checks_are_producer_report_checks_not_independent_GPU_reinspection=True,
        source_scope="CDI repository SecMI-stat port",source_bindings=source,
        analysis_sha256=digest(directory/"analysis.json"),verifier_sha256=digest(Path(__file__)),
        statistics_helper_sha256=digest(Path(__file__).with_name("secmi_analysis.py")),
        scheduler_config_sha256=digest(config_path),
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [
            directory/"protocol.json",directory/"report.json",directory/"image_scores.json",directory/"patient_scores.json",
            REFERENCE/"report.json",REFERENCE/"results.json"]},
        raw_tensor_sha256={p.relative_to(directory).as_posix():v for p,v in tensor_hashes.items()},
        numerical_gate_changes=gates_changed,seconds_cpu=time.perf_counter()-started)
    write_json(directory/"verification.json",verification)
    return verification


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,default=RUN/"baseline_screen_20260914/secmi_v1")
    args=parser.parse_args()
    result=verify(args.directory)
    print(json.dumps({k:v for k,v in result.items() if k not in ("raw_tensor_sha256","source_bindings","input_sha256")},indent=2))


if __name__=="__main__":main()

