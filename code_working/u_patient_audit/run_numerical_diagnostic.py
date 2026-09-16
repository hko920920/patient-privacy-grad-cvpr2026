"""Actual fixed-input numerical diagnostics. No score fitting or new patients."""
import argparse, gc, json, time
from pathlib import Path
import torch
from .common import RUN, digest, write_json, verify_inputs
from .models import setup, load_cache, load_unet, scheduler, adapter_state
from .probe import ResponseProbe
from .probe_verified import DiagnosticProbe

def serial(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

def compare_grad(a,b):
    diff=(a-b).float()
    return dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/max(float(a.norm()),1e-12)),
                exact_equal=bool(torch.equal(a,b)))

def run_precision(precision,checkpoint,image_id,cache,out):
    start=time.perf_counter()
    unet=load_unet(checkpoint,training=False)
    probe=DiagnosticProbe(unet,scheduler(),cache,precision=precision)
    initial_adapter=adapter_state(unet)
    torch.cuda.reset_peak_memory_stats()
    trace=probe.trace_support(image_id)
    a0=torch.zeros(8,device="cuda")
    cell0=next(c for c in trace["steps"][0]["cells"] if c["timestep"]==500 and c["draw"]==0)
    ad_gradient=torch.tensor(cell0["gradient"],device="cuda")
    gnorm=float(ad_gradient.norm())
    assert gnorm>0 and torch.isfinite(ad_gradient).all(),"nonzero direction unavailable"
    checkpoint_on_value=cell0["value"]
    repeats=[probe.delta_cells(image_id,a0,"support",(500,),gradient=False)["value"] for _ in range(2)]
    unet.disable_gradient_checkpointing()
    off=probe.delta_cells(image_id,a0,"support",(500,),gradient=True)
    unet.enable_gradient_checkpointing()
    checkpoint_check=dict(on_value=checkpoint_on_value,off_value=off["value"],
                          loss_absolute_difference=abs(checkpoint_on_value-off["value"]),
                          gradient=compare_grad(ad_gradient,off["gradient"]))
    legacy=None;legacy_f=legacy_b=0
    if precision=="fp16":
        frozen=ResponseProbe(unet,scheduler(),cache)
        old_value,old_gradient,old_base,old_target=frozen.delta(image_id,a0,"support",(500,),1,True)
        legacy=dict(value_absolute_difference=abs(old_value-checkpoint_on_value),
                    base_loss_absolute_difference=abs(old_base-cell0["base_loss"]),
                    target_loss_absolute_difference=abs(old_target-cell0["target_loss"]),
                    gradient=compare_grad(old_gradient,ad_gradient))
        legacy_f,legacy_b=frozen.forward,frozen.backward
        assert old_value==checkpoint_on_value and torch.equal(old_gradient,ad_gradient),"new instrumentation changed original cell"
        del frozen
    finite=[]
    directions=[("gradient_direction",ad_gradient/ad_gradient.norm(),[.005,.001,.0002])]
    # In FP32, inspect all eight coefficient coordinates as well as the gradient direction.
    if precision=="fp32":
        directions += [(f"coordinate_{i}",torch.eye(8,device="cuda")[i],[.001,.0002]) for i in range(8)]
    for name,direction,hs in directions:
        expected=float(torch.dot(ad_gradient,direction))
        for h in hs:
            minus=probe.delta_cells(image_id,a0-h*direction,"support",(500,),gradient=False)
            plus=probe.delta_cells(image_id,a0+h*direction,"support",(500,),gradient=False)
            observed=(plus["value"]-minus["value"])/(2*h)
            finite.append(dict(direction=name,h=h,autograd=expected,finite_difference=observed,
                               absolute_error=abs(observed-expected),
                               symmetric_relative_error=abs(observed-expected)/max(abs(observed),abs(expected),1e-12),
                               minus=serial(minus),plus=serial(plus),
                               embedding_change=probe.embedding_change(a0-h*direction,a0+h*direction)))
    probe.assert_weights_unchanged()
    final_adapter=adapter_state(unet)
    assert initial_adapter.keys()==final_adapter.keys()
    assert all(torch.equal(initial_adapter[k],final_adapter[k]) for k in initial_adapter)
    torch.cuda.synchronize()
    result=dict(status="NUMERICAL_MEASUREMENTS_COMPLETE",precision=precision,trace=trace,
        checkpointing_check=checkpoint_check,legacy_cell_reproduction=legacy,
        repeated_values=repeats,repeat_absolute_range=max(repeats)-min(repeats),
        initial_gradient_norm=gnorm,finite_differences=finite,
        forward=probe.forward,backward=probe.backward,legacy_extra_forward=legacy_f,legacy_extra_backward=legacy_b,
        peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.perf_counter()-start,weights_unchanged=True,
        scope="same-input numerical checks; neither attack efficacy nor clinical utility")
    write_json(out/(precision+".json"),result)
    print(json.dumps(dict(precision=precision,objective_gain=trace["objective_gain"],
        checkpoint_gradient=checkpoint_check["gradient"],legacy=legacy,
        finite_gradient_direction=[{k:r[k] for k in ["h","autograd","finite_difference","symmetric_relative_error"]}
                                  for r in finite if r["direction"]=="gradient_direction"],
        forward=probe.forward,backward=probe.backward,seconds=result["seconds"],peak_cuda_GiB=result["peak_cuda_GiB"])),flush=True)
    del probe,unet,initial_adapter,final_adapter
    gc.collect();torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-tag",default="numerical_v1")
    args=parser.parse_args()
    assert args.output_tag.replace("_","").isalnum()
    out=RUN/"verification_20260914"/args.output_tag
    if (out/"report.json").exists():raise RuntimeError("completed numerical run already exists")
    setup();verify_inputs();cache=load_cache()
    source=RUN/"probe_smoke/training_coverage_v2/U_step_1000_n8/results.json"
    rows=json.loads(source.read_text())
    first=next(r for r in rows if r["model"]=="model_1")
    image_id=first["folds"][0]["support"]
    checkpoint=RUN/"training_coverage_v2/model_1/step_1000.pt"
    assert digest(checkpoint)==first["checkpoint_sha256"]
    codepaths=[Path(__file__),Path(__file__).with_name("probe_verified.py"),Path(__file__).with_name("probe.py"),
               Path(__file__).with_name("models.py")]
    contract=dict(schema="cvpr-u-actual-numerical/v1",checkpoint_sha256=digest(checkpoint),
                  source_results_sha256=digest(source),image_id=image_id,original_patient_id=first["patient_id"],
                  selection_rule="first existing model1 record, first support image, not score-dependent",
                  cache_sha256=digest(RUN/"cache/cache.pt"),code_sha256={p.name:digest(p) for p in codepaths},
                  FP16_and_FP32_control=True,finite_steps=[.005,.001,.0002],
                  FP32_coordinate_steps=[.001,.0002],new_patients=0,new_training_steps=0)
    write_json(out/"protocol.json",contract)
    results={}
    for precision in ["fp16","fp32"]:
        if (out/(precision+".json")).exists():
            raise RuntimeError("partial phase exists; review and use separate output tag")
        results[precision]=run_precision(precision,checkpoint,image_id,cache,out)
    f16=results["fp16"];f32=results["fp32"]
    t16=f16["trace"];t32=f32["trace"]
    vector16=torch.tensor(t16["final_coefficient"]);vector32=torch.tensor(t32["final_coefficient"])
    report=dict(status="MEASUREMENTS_COMPLETE_REVIEW_REQUIRED",new_patients=0,new_training_steps=0,
                scenario="existing U input numerical diagnostic",checkpoint_sha256=digest(checkpoint),
                protocol_sha256=digest(out/"protocol.json"),
                phase_hashes={k:digest(out/(k+".json")) for k in results},
                final_coefficient_precision_comparison=compare_grad(vector16,vector32),
                objective_gains={k:v["trace"]["objective_gain"] for k,v in results.items()},
                totals=dict(forward=sum(v["forward"]+v["legacy_extra_forward"] for v in results.values()),
                            backward=sum(v["backward"]+v["legacy_extra_backward"] for v in results.values())),
                no_attack_performance_claim=True)
    write_json(out/"report.json",report);print(json.dumps(report),flush=True)
if __name__=="__main__":main()

