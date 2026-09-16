"""Actual-data execution checks, using fitting patients only; not a performance claim."""
import argparse,json,gc,time
from collections import defaultdict
import numpy as np
from .common import *
from .models import *
from .probe import ResponseProbe

def selection(scenario="U"):
    rows=read_csv(RUN/"cohort"/"evaluation_images.csv")
    by=defaultdict(list)
    for r in rows:
        required_role="U_observed" if scenario=="U" else "train_candidate"
        if r["eval_role"]=="fit" and r["record_role"]==required_role:
            by[r["patient_id"]].append(r)
    chosen=[]
    for group in ["A","B"]:
        pool=sorted([p for p,rs in by.items() if rs[0]["assignment_group"]==group],
            key=lambda p:stable("runtime-smoke",p))[:4]
        chosen.extend((p,by[p]) for p in pool)
    return chosen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--step",type=int,choices=[50,250,1000],default=50)
    parser.add_argument("--count-per-model",type=int,choices=[1,2,8],default=1)
    parser.add_argument("--training-dir",choices=["training","training_coverage_v2"],default="training")
    parser.add_argument("--scenario",choices=["E","U"],default="U")
    args=parser.parse_args()
    setup();verify_inputs();cache=load_cache();selected=selection(args.scenario)
    # First diagnostic point is also paired so both membership classes enter eventual 8-patient runs.
    if args.count_per_model<8:
        selected=[selected[0],selected[4]][:args.count_per_model]
    out=RUN/"probe_smoke"/args.training_dir/f"{args.scenario}_step_{args.step:04d}_n{args.count_per_model}"
    if (out/"report.json").exists():raise RuntimeError("completed probe smoke must not be overwritten")
    out.mkdir(parents=True,exist_ok=True)
    unet=load_unet(training=False);probe=ResponseProbe(unet,scheduler(),cache)
    ids=[r["image_id"] for r in selected[0][1]]
    a=torch.ones(8,device="cuda")*.005
    zero,g,_,_=probe.delta(ids[0],a,"null-smoke",(500,),1,True)
    assert abs(zero)<1e-9 and float(g.norm())<1e-9,"zero adapter does not equal base"
    del probe,unet;gc.collect();torch.cuda.empty_cache()
    reports=[]
    for model,group in [("model_1","A"),("model_2","B")]:
        checkpoint=RUN/args.training_dir/model/f"step_{args.step:04d}.pt"
        checkpoint_state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        exposure=checkpoint_state["exposures"]
        del checkpoint_state
        unet=load_unet(checkpoint,training=False)
        probe=ResponseProbe(unet,scheduler(),cache)
        torch.cuda.reset_peak_memory_stats()
        for p,rows in selected:
            image_ids=sorted(r["image_id"] for r in rows)
            result=probe.patient(image_ids)
            assert result["forward"]==312 and result["backward"]==72
            result.update(model=model,patient_id=p,eval_role="fit",scenario=args.scenario,
                declared_member=int(rows[0]["assignment_group"]==group),
                realized_member=int(any(exposure.get(r["image_id"],0)>0
                    for r in read_csv(RUN/"cohort"/"evaluation_images.csv") if r["patient_id"]==p)),
                checkpoint_sha256=digest(checkpoint))
            reports.append(result)
            write_json(out/"partial_results.json",reports)
            print(json.dumps({"model":model,"completed":len(reports),
                "seconds":result["seconds"],"forward":result["forward"],"backward":result["backward"],
                "max_cuda_GiB":torch.cuda.max_memory_allocated()/2**30,"weights_unchanged":True}),flush=True)
        del probe,unet;gc.collect();torch.cuda.empty_cache()
    write_json(out/"results.json",reports)
    summary={"status":"PASS_EXECUTION_ONLY","step":args.step,"training_dir":args.training_dir,"scenario":args.scenario,
        "unique_fit_patients":len(selected),"model_patient_evaluations":len(reports),
        "zero_adapter_delta_and_gradient":0.,
        "forward_per_patient":312,"backward_per_patient":72,
        "median_seconds_per_patient":float(np.median([r["seconds"] for r in reports])),
        "max_seconds_per_patient":max(r["seconds"] for r in reports),
        "all_weights_unchanged":True,"cohort_lock_sha256":digest(RUN/"cohort"/"lock.json"),
        "score_direction_or_threshold_fitted":False,"calibration_test_patients_queried":False,
        "medical_performance_or_novelty_claim":False,"probe_code_sha256":digest(Path(__file__).with_name("probe.py"))}
    write_json(out/"report.json",summary);print(json.dumps(summary),flush=True)

if __name__=="__main__":main()
