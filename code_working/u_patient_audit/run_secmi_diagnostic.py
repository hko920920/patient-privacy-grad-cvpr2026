"""Fixed SecMI-stat screen: original fit8 plus selection40; no tuning/training."""
import argparse, gc, json, math, re, time
from pathlib import Path
from collections import Counter, defaultdict
import torch
from .common import ROOT, RUN, PROMPT, digest, read_csv, verify_inputs, write_json
from .models import setup, load_cache, load_unet, scheduler, adapter_state
from .secmi_adapter import SecMIAdapter, source_bindings, cpu_conformance

OLD=RUN/"probe_smoke/training_coverage_v2/U_step_1000_n8"
MODELS=("model_1","model_2")
def load(p): return json.loads(p.read_text(encoding="utf-8"))

def selected_data():
    old=load(OLD/"results.json")
    original_check=load(OLD/"verification.json")
    assert original_check["results_sha256"]==digest(OLD/"results.json")
    fit=[r["patient_id"] for r in old if r["model"]=="model_1"]
    assert len(fit)==len(set(fit))==8
    evaluation=read_csv(RUN/"cohort/evaluation_images.csv")
    selection=sorted({r["patient_id"] for r in evaluation if r["eval_role"]=="selection"},key=int)
    assert len(selection)==40 and not set(selection).intersection(fit)
    by_patient=defaultdict(list)
    for r in evaluation: by_patient[r["patient_id"]].append(r)
    rows=[]
    for role, ids in (("fit",fit),("selection",selection)):
        counts=Counter(by_patient[p][0]["assignment_group"] for p in ids)
        assert counts=={"A":len(ids)//2,"B":len(ids)//2}
        for p in ids:
            candidate=by_patient[p]
            assert len(candidate)==4 and all(r["eval_role"]==role for r in candidate)
            for scenario,record_role in (("E","train_candidate"),("U","U_observed")):
                subset=sorted([r for r in candidate if r["record_role"]==record_role],key=lambda r:r["image_id"])
                assert len(subset)==2
                rows.extend(dict(r,scenario=scenario) for r in subset)
    assert len(rows)==192
    return fit,selection,rows

def aggregate(images):
    groups=defaultdict(list)
    for r in images: groups[(r["model"],r["scenario"],r["eval_role"],r["patient_id"])].append(r)
    results=[]
    for (model,scenario,role,patient),rows in sorted(groups.items()):
        assert len(rows)==2
        values=[r["score"] for r in rows]
        results.append(dict(model=model,scenario=scenario,eval_role=role,patient_id=patient,
            assignment_group=rows[0]["assignment_group"],member=rows[0]["member"],
            image_ids=sorted(r["image_id"] for r in rows),score_mean=math.fsum(values)/2,score_max=max(values)))
    assert len(results)==192
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-tag",default="secmi_v1")
    args=parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",args.output_tag)
    out=RUN/"baseline_screen_20260914"/args.output_tag
    if out.exists(): raise FileExistsError("immutable baseline run exists; inspect before any new tag")
    verify_inputs()
    fit,selection,rows=selected_data()
    conformance=cpu_conformance()
    checkpoints={m:RUN/"training_coverage_v2"/m/"step_1000.pt" for m in MODELS}
    hashes={m:digest(p) for m,p in checkpoints.items()}
    original=load(OLD/"results.json")
    for r in original: assert hashes[r["model"]]==r["checkpoint_sha256"]
    files=[OLD/"results.json",OLD/"verification.json",RUN/"cohort/lock.json",
        RUN/"cohort/evaluation_images.csv",RUN/"cache/summary.json",RUN/"training_coverage_v2/protocol.json"]
    files += [RUN/"cohort"/(m+"_train.csv") for m in MODELS]
    protected=load(RUN/"verification_20260914/foundation/report.json")["protected_sha256"]
    for name,value in protected.items(): assert digest(ROOT/name)==value
    protocol=dict(schema="u-secmi-screen/v1",method="SecMI-stat port of CDI authors' preserved extractor",
        source=source_bindings(),precision="fp32",batch_size=1,t0=0,t=100,step=10,
        generic_prompt=PROMPT,latent_source="existing verified VAE posterior-mode FP16 cache promoted to FP32",
        guidance="single conditional UNet prediction; no classifier-free guidance",target_only=True,
        primary_image_score="negative L2 norm of deterministic t100 latent minus t100 reconstructed via t110",
        primary_patient_score="mean of two image scores",secondary_patient_score="max of two image scores; descriptive only",
        patient_order_by_role=dict(fit=fit,selection=selection),scenarios=["E","U"],models=list(MODELS),
        fit_patients=8,selection_patients=40,unique_patients=48,newly_target_queried_selection_patients=40,
        calibration_or_test_used=False,model_training_steps=0,attack_fitting=False,
        expected_image_records=384,expected_patient_records=192,forward_per_image=12,total_forward=4608,total_backward=0,
        bootstrap=dict(replicates=2000,seed=260914,method="percentile 2.5/97.5; patient-stratified A/B each20; same draws across models and E/U"),
        gate="selection40 primary mean only: both target AUC lower95 strictly above0.5 => conditional ranking signal supported; E/U separate",
        failure_action="hold candidate tuning and further target training; no assertion of absent leakage or impossible research",
        scope="one fixed setting and one correlated model pair; not final test, practical low-FPR validation, or all-SOTA comparison",
        checkpoint_sha256=hashes,cache_sha256=load(RUN/"cache/summary.json")["cache_sha256"],
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in ("run_secmi_diagnostic.py","secmi_adapter.py","models.py","common.py")},
        original_protected_sha256=protected,conformance=conformance)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"protocol.json",protocol)
    setup(); cache=load_cache(); started=time.perf_counter()
    images=[]; model_reports=[]
    for model in MODELS:
        model_start=time.perf_counter()
        state=torch.load(checkpoints[model],map_location="cpu",weights_only=True)
        exposure=state["exposures"]
        assert state["step"]==1000 and state["contract"]==load(RUN/"training_coverage_v2/protocol.json")
        del state
        train={r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
        assert set(exposure)==train and min(exposure.values())>=1 and sum(exposure.values())==4000
        unet=load_unet(checkpoints[model],training=False)
        probe=SecMIAdapter(unet,scheduler(),cache)
        before=adapter_state(unet); torch.cuda.reset_peak_memory_stats()
        for index,row in enumerate(rows):
            member=int(row["assignment_group"]==("A" if model=="model_1" else "B"))
            image_member=int(row["scenario"]=="E" and member)
            assert int(exposure.get(row["image_id"],0)>0)==image_member
            assert int(row["image_id"] in train)==image_member
            measured=probe.score(row["image_id"])
            assert measured["forward"]==12 and measured["backward"]==0
            assert math.isfinite(measured["l2"]) and measured["l2"]>=0
            tensor_rel=Path("states")/model/(row["image_id"]+".pt")
            tensor_path=out/tensor_rel; tensor_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(z_det=measured["z_det"],z_recon=measured["z_recon"],
                image_id=row["image_id"],model=model),tensor_path)
            record=dict(model=model,scenario=row["scenario"],eval_role=row["eval_role"],patient_id=row["patient_id"],
                image_id=row["image_id"],assignment_group=row["assignment_group"],member=member,image_member=image_member,
                record_role=row["record_role"],actual_training_exposures=exposure.get(row["image_id"],0),
                l2=measured["l2"],score=-measured["l2"],forward=12,backward=0,
                stages=measured["stages"],seconds=measured["seconds"],
                tensor_path=tensor_rel.as_posix(),tensor_sha256=digest(tensor_path),checkpoint_sha256=hashes[model])
            images.append(record)
            if (index+1)%4==0:
                progress=dict(status="RUNNING",completed_images=len(images),total_images=384,model=model,
                    patient_id=row["patient_id"],role=row["eval_role"],elapsed_seconds=time.perf_counter()-started)
                write_json(out/"partial_image_scores.json",images); write_json(out/"progress.json",progress)
                print(json.dumps(progress),flush=True)
        probe.assert_weights_unchanged()
        after=adapter_state(unet)
        assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
        assert probe.forward==2304 and probe.backward==0
        model_reports.append(dict(model=model,prediction_type=probe.prediction_type,forward=probe.forward,backward=probe.backward,
            weights_unchanged=True,adapter_tensors_checked=len(before),max_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.perf_counter()-model_start,checkpoint_sha256=hashes[model]))
        del probe,unet,before,after; gc.collect();torch.cuda.empty_cache()
    patients=aggregate(images)
    write_json(out/"image_scores.json",images);write_json(out/"patient_scores.json",patients)
    for name,value in protected.items(): assert digest(ROOT/name)==value
    report=dict(status="COMPLETED_SECMI_EXECUTION_REQUIRES_INDEPENDENT_ANALYSIS",protocol_sha256=digest(out/"protocol.json"),
        unique_patients=48,fit_patients=8,selection_patients=40,image_records=len(images),patient_records=len(patients),
        total_forward=sum(r["forward"] for r in images),total_backward=0,model_reports=model_reports,
        elapsed_seconds=time.perf_counter()-started,original_protected_files_unchanged=11,
        image_scores_sha256=digest(out/"image_scores.json"),patient_scores_sha256=digest(out/"patient_scores.json"),
        attack_success_declared=False,calibration_or_test_used=False,training_updates=0)
    assert report["total_forward"]==4608
    write_json(out/"report.json",report);write_json(out/"progress.json",dict(status="COMPLETE",completed_images=384,total_images=384))
    print(json.dumps(report),flush=True)
if __name__=="__main__": main()

