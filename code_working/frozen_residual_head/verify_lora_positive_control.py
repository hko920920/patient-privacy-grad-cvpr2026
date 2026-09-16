"""Independent CPU arithmetic/provenance checks for a saved LoRA sampling bridge.

No diffusion pipeline, producer numerical helper, UNet, text encoder or VAE is
executed here. A saved-packet PASS is not a generation-quality judgment.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key,"1")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"_reports/frozen_residual_lora_positive_control_20260916_v1"
STATUS="PASS_SAVED_LORA_POSITIVE_CONTROL_REPLAY_AND_BRIDGE_ARITHMETIC"
REPLAY_STATUS="PASS_SAVED_LORA_HISTORICAL_REPLAY_AND_ARITHMETIC"
STAGES=("replay_pipeline_fp16","manual_fp16","manual_fp32_cfg75","manual_fp32_cfg1",
        "manual_fp32_cached","manual_fp32_currentnoise")


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path,allow_pickle=False) as packet:
        return {key:packet[key].copy() for key in packet.files}


def save(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")


class Check:
    def __init__(self):
        self.count=0
        self.errors={}

    def require(self,condition,label):
        self.count+=1
        if not bool(condition):
            raise AssertionError(label)

    def equal(self,actual,expected,label):
        actual,expected=np.asarray(actual),np.asarray(expected)
        self.require(actual.shape==expected.shape and actual.dtype==expected.dtype,
                     label+": shape/dtype")
        self.require(np.array_equal(actual,expected),label+": exact values")

    def close(self,actual,expected,label,atol,rtol):
        actual,expected=np.asarray(actual),np.asarray(expected)
        self.require(actual.shape==expected.shape,label+": shape")
        self.require(np.isfinite(actual).all() and np.isfinite(expected).all(),label+": finite")
        difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        self.errors[label]=float(np.max(difference,initial=0.))
        self.require(np.all(difference<=float(atol)+float(rtol)*np.abs(expected.astype(np.float64))),
                     label+": fixed tolerance")

    def bounded(self,actual,reference,bound,label):
        actual,reference,bound=(np.asarray(x,dtype=np.float64) for x in (actual,reference,bound))
        self.require(actual.shape==reference.shape==bound.shape,label+": shape")
        self.require(np.isfinite(actual).all() and np.isfinite(reference).all()
                     and np.isfinite(bound).all() and np.all(bound>=0),label+": finite envelope")
        difference=np.abs(actual-reference)
        self.errors[label] = {"maximum_absolute_error":float(np.max(difference,initial=0.)),
            "maximum_bound_fraction":float(np.max(difference/np.maximum(bound,np.finfo(float).tiny),initial=0.))}
        self.require(np.all(difference<=bound),label+": operation-rounding envelope")


def independent_alphas(config):
    count=int(config["num_train_timesteps"])
    if config.get("trained_betas") is not None:
        betas=torch.tensor(config["trained_betas"],dtype=torch.float32)
    elif config["beta_schedule"]=="scaled_linear":
        betas=torch.linspace(float(config["beta_start"])**.5,float(config["beta_end"])**.5,
                             count,dtype=torch.float32).square()
    elif config["beta_schedule"]=="linear":
        betas=torch.linspace(config["beta_start"],config["beta_end"],count,dtype=torch.float32)
    else:
        raise AssertionError("Unreviewed beta schedule")
    if config.get("rescale_betas_zero_snr",False):
        raise AssertionError("Zero-SNR rescaling is outside the fixed reviewed scheduler")
    return torch.cumprod(1.-betas,dim=0).numpy()


def schedule(config,steps=30):
    total=int(config["num_train_timesteps"])
    spacing=config.get("timestep_spacing","leading")
    if spacing=="leading":
        return (np.arange(steps)*(total//steps))[::-1].copy().astype(np.int64)+int(config.get("steps_offset",0))
    if spacing=="linspace":
        return np.linspace(0,total-1,steps).round()[::-1].copy().astype(np.int64)
    if spacing=="trailing":
        return (np.arange(total,0,-total/steps).round()-1).astype(np.int64)
    raise AssertionError("Unreviewed timestep spacing")


def cfg(raw,guidance):
    """Separate eager CPU operations preserve the actual FP16/FP32 dtype."""
    raw=np.asarray(raw)
    if raw.dtype not in (np.float16,np.float32):
        raise AssertionError("Expected raw epsilon FP16 or FP32")
    if float(guidance)<=1.:
        if raw.shape!=(1,4,32,32):
            raise AssertionError("No-CFG branch must have one conditional example")
        return raw.copy()
    if raw.shape!=(2,4,32,32):
        raise AssertionError("CFG raw order must be unconditional then conditional")
    raw_t=torch.from_numpy(raw.copy())
    uncond,cond=raw_t[0:1],raw_t[1:2]
    difference=cond-uncond
    scaled=float(guidance)*difference
    return (uncond+scaled).numpy()


def ddim(sample,epsilon,t,alphas,config,steps=30):
    """Saved epsilon/eta0 DDIM update; no scheduler implementation is imported."""
    if config["prediction_type"]!="epsilon" or config.get("thresholding",False) or config.get("clip_sample",False):
        raise AssertionError("Unreviewed DDIM prediction or clipping policy")
    sample,epsilon=np.asarray(sample),np.asarray(epsilon)
    if sample.dtype not in (np.float16,np.float32) or epsilon.dtype!=sample.dtype:
        raise AssertionError("Reviewed update expects a single actual arithmetic dtype")
    previous=int(t)-int(config["num_train_timesteps"])//steps
    a=np.float32(alphas[int(t)])
    ap=np.float32(alphas[previous] if previous>=0 else (1. if config.get("set_alpha_to_one",True) else alphas[0]))
    x,e=torch.from_numpy(sample.copy()),torch.from_numpy(epsilon.copy())
    at,apt=torch.tensor(a,dtype=torch.float32),torch.tensor(ap,dtype=torch.float32)
    beta=1.-at
    original=(x-beta.sqrt()*e)/at.sqrt()
    # eta=0 and use_clipped_model_output=False, as fixed by the outer protocol.
    previous_sample=apt.sqrt()*original+(1.-apt).sqrt()*e
    x64,e64=sample.astype(np.float64),epsilon.astype(np.float64)
    a64,ap64=float(a),float(ap)
    reference=np.sqrt(ap64)*(x64-np.sqrt(1.-a64)*e64)/np.sqrt(a64)+np.sqrt(1.-ap64)*e64
    unit=float(np.finfo(sample.dtype).eps)/2.
    gamma=32.*unit/(1.-32.*unit)
    scale=np.sqrt(ap64)*(np.abs(x64)+np.sqrt(1.-a64)*np.abs(e64))/np.sqrt(a64)+np.sqrt(1.-ap64)*np.abs(e64)
    bound=gamma*scale+32.*float(np.nextafter(sample.dtype.type(0),sample.dtype.type(1)))
    return previous_sample.numpy(),reference,bound


def png_pixels(path):
    with Image.open(path) as picture:
        return np.asarray(picture.convert("RGB")).copy()


def historical_seed(prompt_index):
    payload="|".join(map(str,("cvpr-u-pilot-20260914-v1","quality-generation",prompt_index)))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15],16)


def check_packet(check,packet,stage,guidance,config):
    label=stage
    half=stage in ("replay_pipeline_fp16","manual_fp16")
    dtype=np.dtype(np.float16 if half else np.float32)
    batch=2 if float(guidance)>1. else 1
    shapes={"conditioning":(batch,77,1024),"timesteps":(30,),
        "latents":(31,1,4,32,32),"model_inputs":(30,batch,4,32,32),
        "raw_eps":(30,batch,4,32,32),"eps":(30,1,4,32,32),
        "decoded_raw":(1,3,256,256),"image_float":(256,256,3),"initial_from_prepare":(1,4,32,32)}
    check.require(set(packet)==set(shapes),label+": all stored tensor fields")
    for key,shape in shapes.items():
        check.require(packet[key].shape==shape,label+": shape "+key)
        expected=np.dtype(np.int64) if key=="timesteps" else np.dtype(np.float32) if key=="image_float" else dtype
        check.require(packet[key].dtype==expected,label+": actual dtype "+key)
        check.require(np.isfinite(packet[key]).all(),label+": finite "+key)
    times=schedule(config)
    alphas=independent_alphas(config)
    check.equal(packet["timesteps"],times,label+": independently specified DDIM schedule")
    check.equal(packet["initial_from_prepare"],packet["latents"][0],label+": actual prepared initial latent")
    cpu_exact_steps=0
    for i,timestep in enumerate(times):
        current=packet["latents"][i]
        expected_input=np.concatenate([current,current],axis=0) if batch==2 else current
        check.equal(packet["model_inputs"][i],expected_input,label+": DDIM identity scale and input order "+str(i))
        combined=cfg(packet["raw_eps"][i],guidance)
        check.equal(packet["eps"][i],combined,label+": eager CFG arithmetic "+str(i))
        cpu,exact,bound=ddim(current,combined,int(timestep),alphas,config)
        check.bounded(packet["latents"][i+1],exact,bound,label+": DDIM rounding "+str(i))
        check.bounded(cpu,exact,bound,label+": independent CPU DDIM consistency "+str(i))
        cpu_exact_steps+=int(np.array_equal(cpu,packet["latents"][i+1]))
    decoded=torch.from_numpy(packet["decoded_raw"].copy())
    processed=(decoded/2.+.5).clamp(0.,1.).permute(0,2,3,1).float().numpy()[0]
    check.equal(packet["image_float"],processed,label+": independent decoded-to-image conversion")
    return {"steps":30,"cpu_ddim_exact_steps":cpu_exact_steps,"precision":str(dtype),
            "guidance":float(guidance),"decoded_unclamped_minimum":float(packet["decoded_raw"].min()),
            "decoded_unclamped_maximum":float(packet["decoded_raw"].max())}


def image_from_packet(packet):
    return np.clip(packet["image_float"]*np.float32(255.),0.,255.).astype(np.uint8)


def check_bridge_pair(check,earlier,later,transition):
    if transition=="exact_manual_replay":
        check.require(set(earlier)==set(later),"pipeline/manual packet schema identical")
        for key in earlier:
            check.equal(later[key],earlier[key],"pipeline/manual exact full path "+key)
    elif transition=="precision":
        check.equal(later["latents"][0],earlier["latents"][0].astype(np.float32),"precision bridge same initial values")
        check.equal(later["conditioning"],earlier["conditioning"].astype(np.float32),"precision bridge same conditioning values")
    elif transition=="guidance":
        check.equal(later["latents"][0],earlier["latents"][0],"CFG bridge same initial latent")
        check.equal(later["conditioning"],earlier["conditioning"][1:2],"CFG bridge conditional branch unchanged")
    elif transition=="conditioning":
        check.equal(later["latents"][0],earlier["latents"][0],"cached conditioning bridge same initial latent")
    elif transition=="initial_noise":
        check.equal(later["conditioning"],earlier["conditioning"],"noise bridge same conditioning")
    else:
        raise AssertionError("Unknown fixed bridge transition")
    check.equal(later["timesteps"],earlier["timesteps"],"bridge unchanged scheduler timesteps "+transition)


def fixed_file(mapping,suffix):
    found=[Path(path) for path in mapping if str(path).replace("\\","/").endswith(suffix)]
    if len(found)!=1:
        raise AssertionError("Expected one fixed source with suffix "+suffix)
    return found[0]


def verify_contract(check,out,contract):
    check.require(contract["schema"]=="lora-positive-control-bridge/v1","reviewed contract schema")
    sources=contract["source_sha256"]
    check.require(sources[str(Path(__file__).resolve())]==sha(__file__),"verifier prebound before GPU")
    for path,digest in sources.items():
        check.require(sha(path)==digest,"source/input unchanged "+path)
    for suffix in ("u_patient_audit/generate_quality.py","u_patient_audit/models.py","u_patient_audit/common.py",
                   "diffusers/image_processor.py","diffusers/schedulers/scheduling_ddim.py",
                   "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py"):
        fixed_file(sources,suffix)
        check.require(True,"required historical/library source bound "+suffix)
    check.require(tuple(contract["stages"])==STAGES,"all six predefined stages")
    check.require(contract["steps"]==30 and contract["height"]==256 and contract["width"]==256
                  and contract["eta"]==0.,"fixed steps, size and deterministic DDIM")
    check.require(contract["new_residual_head_applied"] is False and contract["new_training"] is False
                  and contract["DP_claim"] is False and contract["whole_research_success_claim"] is False,
                  "diagnostic-only scope")
    check.require(contract["expected_total_UNet_calls"]==360 and contract["expected_total_UNet_batch_examples"]==540
                  and contract["expected_backward"]==0,"fixed total compute scope")
    check.require(contract["historical_initial_latent_saved"] is False,"historical evidence limited to image and source")
    check.require(contract["historical_png_equality_target"]=="exact file and decoded pixels",
                  "unchanged exact historical replay gate")
    check.require(contract["manual_reference_equality_target"]==
        "exact all UNet inputs/outputs, guided epsilon, DDIM latents and decoded image","unchanged exact manual gate")
    config=contract["scheduler_config"]
    check.require(config["prediction_type"]=="epsilon" and not config["clip_sample"]
                  and not config.get("thresholding",False),"epsilon/nonclipped scheduler")
    check.equal(np.asarray(contract["timesteps"],np.int64),schedule(config),"fixed time grid")
    alphas=independent_alphas(config)
    check.require(np.isfinite(alphas).all() and np.all(alphas>0) and np.all(alphas<=1),"valid independent alpha schedule")
    previous_contract_path=fixed_file(sources,"frozen_residual_sampling_integration_20260916_v1/contract.json")
    previous_directory=previous_contract_path.parent
    previous=read(previous_contract_path)
    verification=read(previous_directory/"independent_verification.json")
    check.require(verification["status"]=="PASS_SAVED_SAMPLING_INTEGRATION_RESIDUAL_DDIM_AND_BASE_RESTORATION"
                  and verification["complete"] is True,"previous sampling packet independently passed")
    for name,digest in verification["input_sha256"].items():
        check.require(sha(previous_directory/name)==digest,"previous verified input remains unchanged "+name)
    check.require(contract["snapshot"]==previous["snapshot"] and contract["model_id"]==previous["model_id"]
                  and contract["model_revision"]==previous["model_revision"],"same fixed base artifact provenance")
    check.equal(np.asarray(contract["timesteps"],np.int64),np.asarray(previous["timesteps"],np.int64),
                "same current sampling schedule")
    for key,value in config.items():
        if not key.startswith("_"):
            check.require(previous["scheduler_config"][key]==value,"same prior numerical scheduler config "+key)
    history_path=fixed_file(sources,"quality/training_coverage_v2/step_1000_both/report.json")
    history=read(history_path)
    checkpoint=Path(contract["checkpoint"])
    check.require(checkpoint.resolve()==fixed_file(sources,"training_coverage_v2/model_1/step_1000.pt").resolve(),
                  "predefined existing M1 checkpoint")
    check.require(sha(checkpoint)==contract["checkpoint_sha256"]==history["checkpoint_hashes"]["model_1"],
                  "historical exact checkpoint binding")
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=True)
    adapter=checkpoint_data["adapter"]
    check.require(len(adapter)==256 and sum(value.numel() for value in adapter.values())==1659904,
                  "original r8 saved adapter tensor/count")
    check.require(all(".lora_" in name and value.dtype==torch.float32 and bool(torch.isfinite(value).all())
                      for name,value in adapter.items()),"actual checkpoint LoRA FP32 values")
    cache_path=fixed_file(sources,"cvpr_u_pilot_v1_001/cache/cache.pt")
    check.require(sources[str(cache_path)]==previous["source_sha256"][str(cache_path)],"same cached conditioning artifact")
    cache=torch.load(cache_path,map_location="cpu",weights_only=True)
    check.require([case["case_id"] for case in contract["cases"]]==["normal","effusion"],"two fixed cases")
    previous_manifest=read(previous_directory/"manifest.json")
    for index,case in zip((1,2),contract["cases"]):
        original=next(row for row in history["images"] if row["model"]=="model_1" and row["path"]==f"model_1_{index}.png")
        old_path=history_path.parent/original["path"]
        check.require(Path(case["historical_image_path"]).resolve()==old_path.resolve()
                      and case["historical_image_sha256"]==sha(old_path)==original["sha256"],"unchanged historical image")
        check.require(case["prompt"]==original["prompt"] and case["historical_cuda_seed"]==original["seed"]==historical_seed(index),
                      "historical fixed prompt and CUDA seed")
        prior_case=next(item for item in previous["cases"] if item["case_id"]==case["case_id"])
        check.require(case["prompt"]==prior_case["prompt"] and case["current_cpu_seed"]==prior_case["seed"],
                      "current fixed prompt and CPU seed")
        hidden=cache["hidden"][case["prompt"]].float().numpy()
        check.require(hidden.shape==(1,77,1024) and hidden.dtype==np.float32,"cached current conditioning tensor")
        check.require(hashlib.sha256(hidden.tobytes()).hexdigest()==case["current_conditioning_sha256"]==prior_case["conditioning_sha256"],
                      "current hidden cache and prior conditioning hash")
        entry=next(item for item in previous_manifest["trajectories"] if item["case_id"]==case["case_id"] and item["method"]=="base")
        prior_trace=previous_directory/entry["path"]
        check.require(Path(case["current_base_trace"]).resolve()==prior_trace.resolve()
                      and sha(prior_trace)==entry["sha256"]==sources[str(prior_trace)],"previous base trace exact source")
    return cache,previous_directory


def verify_runtime_and_execution(check,out,contract,phase):
    replay_runtime=read(out/"runtime_replay.json")
    check.require(replay_runtime["vae_dtype"]=="torch.float16" and replay_runtime["tf32_matmul"] is False
                  and replay_runtime["tf32_cudnn"] is False,"actual historical VAE/TF32 settings")
    inventory=replay_runtime["unet_dtype_inventory"]
    check.require(set(inventory["base"])=={"torch.float16"} and sum(inventory["base"].values())>0
                  and inventory["lora"]=={"torch.float32":256},"historical mixed base/LoRA precision")
    for key,value in contract["scheduler_config"].items():
        if not key.startswith("_"):
            check.require(replay_runtime["scheduler_config"][key]==value,"actual replay scheduler "+key)
    replay=read(out/"execution_replay.json")
    check.require(replay["completed"] is True,"replay completed")
    check.require(replay["contract_sha256"]==sha(out/"contract.json")
                  and replay["manifest_sha256"]==sha(out/"manifest_replay.json")
                  and replay["runtime_sha256"]==sha(out/"runtime_replay.json"),"replay output hash links")
    check.require(replay["unet_calls"]==60 and replay["batch_examples"]==120 and replay["backward"]==0
                  and replay["gradients_present"] is False,"replay actual counters and no gradients")
    before=replay["canonical_model_sha256_before"]
    check.require(len(before)==64 and before==replay["canonical_model_sha256_after"],"replay canonical values unchanged")
    if phase=="bridge":
        bridge=read(out/"execution_bridge.json")
        check.require(bridge["completed"] is True,"bridge completed")
        check.require(bridge["contract_sha256"]==sha(out/"contract.json")
                      and bridge["manifest_sha256"]==sha(out/"manifest_bridge.json")
                      and bridge["replay_verification_sha256"]==sha(out/"replay_verification.json"),"bridge output hash links")
        check.require(bridge["unet_calls"]==300 and bridge["batch_examples"]==420 and bridge["backward"]==0
                      and bridge["gradients_present"] is False and bridge["new_head_applied"] is False,
                      "bridge counters/no gradients/no residual head")
        check.require(bridge["canonical_model_sha256_before"]==bridge["canonical_model_sha256_after"]==before,
                      "fresh load and full bridge preserve canonical FP32 weight values")
        check.require(bridge["initial_dtype_inventory"]==inventory,"same mixed initial precision")
        final=bridge["final_dtype_inventory"]
        check.require(final["base"]=={"torch.float32":sum(inventory["base"].values())}
                      and final["lora"]==inventory["lora"],"one-way FP32 promotion only")


def load_records(check,out,records,contract,bound):
    results,packets={},{}
    for row in records:
        case=next(case for case in contract["cases"] if case["case_id"]==row["case_id"])
        stage=row["stage"]
        stem=case["case_id"]+"_"+stage
        check.require(row["path"]=="traces/"+stem+".npz" and row["image_path"]=="images/"+stem+".png",
                      "canonical trace/image paths")
        for path_key,hash_key in (("path","sha256"),("image_path","image_sha256")):
            check.require(sha(out/row[path_key])==row[hash_key],"packet image/file hash "+row[path_key])
            bound[row[path_key]]=row[hash_key]
        guidance=7.5 if stage in STAGES[:3] else 1.
        precision="original_mixed_autocast_fp16" if stage in STAGES[:2] else "fp32_no_autocast"
        check.require(row["guidance"]==guidance and row["precision"]==precision and row["unet_calls"]==30
                      and row["batch_examples"]==(60 if guidance>1 else 30),"actual stage policy "+stem)
        packet=arrays(out/row["path"])
        results[stem]=check_packet(check,packet,stage,guidance,contract["scheduler_config"])
        check.equal(png_pixels(out/row["image_path"]),image_from_packet(packet),"actual historical PNG truncation "+stem)
        packets[(case["case_id"],stage)]=packet
    return results,packets


def verify(out,phase):
    out=Path(out).resolve()
    result_path=out/("replay_verification.json" if phase=="replay" else "independent_verification.json")
    protocol_path=out/("replay_verification_protocol.json" if phase=="replay" else "independent_verification_protocol.json")
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve previous independent verification attempt")
    names=["contract.json","manifest_replay.json","runtime_replay.json","execution_replay.json"]
    if phase=="bridge":
        names.extend(["manifest_bridge.json","execution_bridge.json","replay_verification.json"])
    bound={name:sha(out/name) for name in names}
    source=sha(__file__)
    protocol={"schema":"independent-lora-positive-control-verification/v1","phase":phase,
        "created_utc":datetime.now(timezone.utc).isoformat(),"code_sha256":source,"initial_input_sha256":dict(bound),
        "numeric_policy":"Source-bound before replay. CFG/cast/image and paired pipeline/manual paths exact. Each DDIM update checked against independent FP64 equation using gamma_32 of its actual FP16 or FP32 dtype and 32 smallest subnormal units; CPU dtype replay exactness is separately reported.",
        "DDIM_bound_scope":"Elementwise epsilon eta0 update from saved input/epsilon; not a UNet numerical-error bound or end-to-end model equivalence tolerance.",
        "new_gpu_calls":0,"producer_numerical_imports":False,"quality_or_efficacy_claim":False}
    save(protocol_path,protocol)
    check=Check();started=time.perf_counter()
    try:
        contract=read(out/"contract.json")
        cache,_=verify_contract(check,out,contract)
        verify_runtime_and_execution(check,out,contract,phase)
        replay_records=read(out/"manifest_replay.json")
        check.require([(r["case_id"],r["stage"]) for r in replay_records]==[(c["case_id"],STAGES[0]) for c in contract["cases"]],
                      "all fixed historical replay records")
        results,packets=load_records(check,out,replay_records,contract,bound)
        historical={}
        for case,row in zip(contract["cases"],replay_records):
            check.equal(png_pixels(out/row["image_path"]),png_pixels(case["historical_image_path"]),
                        "historical PNG decoded pixels exact "+case["case_id"])
            check.require(row["image_sha256"]==case["historical_image_sha256"],"historical PNG bytes exact "+case["case_id"])
            historical[case["case_id"]]={"png_file_exact":True,"decoded_pixels_exact":True}
        if phase=="bridge":
            prior=read(out/"replay_verification.json")
            check.require(prior["complete"] is True and prior["status"]==REPLAY_STATUS,"independent replay PASS before bridge")
            for name,digest in prior["input_sha256"].items():
                check.require(sha(out/name)==digest,"verified replay input unchanged "+name)
            records=read(out/"manifest_bridge.json")
            check.require([(r["case_id"],r["stage"]) for r in records]==[(c["case_id"],s) for s in STAGES[1:] for c in contract["cases"]],
                          "all ten bridge records with no image/stage selection")
            more,extra=load_records(check,out,records,contract,bound)
            results.update(more);packets.update(extra)
            transitions=("exact_manual_replay","precision","guidance","conditioning","initial_noise")
            for case in contract["cases"]:
                for first,last,transition in zip(STAGES,STAGES[1:],transitions):
                    check_bridge_pair(check,packets[(case["case_id"],first)],packets[(case["case_id"],last)],transition)
                hidden=cache["hidden"][case["prompt"]].float().numpy()
                for stage in STAGES[4:]:
                    check.equal(packets[(case["case_id"],stage)]["conditioning"],hidden,"current cached conditional input "+stage)
                final=packets[(case["case_id"],STAGES[-1])]
                previous=arrays(case["current_base_trace"])
                noise=torch.randn((1,4,32,32),generator=torch.Generator(device="cpu").manual_seed(case["current_cpu_seed"]),
                                  dtype=torch.float32).numpy()
                check.equal(final["latents"][0],noise,"current noise independently regenerated "+case["case_id"])
                check.equal(final["latents"][0],previous["latents"][0],"M1/current-base exact starting latent "+case["case_id"])
                check.equal(final["timesteps"],previous["timesteps"],"M1/current-base exact timetable "+case["case_id"])
        for name,digest in bound.items():
            check.require(sha(out/name)==digest,"verification input unchanged "+name)
        check.require(sha(out/"contract.json")==bound["contract.json"] and sha(__file__)==source,"code/contract unchanged")
        check.require(not torch.cuda.is_initialized(),"independent verifier never initialized CUDA")
        result={"status":REPLAY_STATUS if phase=="replay" else STATUS,"complete":True,"phase":phase,
            "checks":check.count,"seconds":time.perf_counter()-started,"code_sha256":source,
            "protocol_sha256":sha(protocol_path),"input_sha256":bound,"historical_replay":historical,
            "trajectory_checks":results,"maximum_absolute_errors":check.errors,
            "DDIM_transitions_checked":30*len(results),"new_GPU_forwards":0,"new_VAE_decodes":0,
            "limits":["Checks saved tensors and source/hash evidence; does not independently rerun UNet, text encoder or VAE.",
                "Historical CUDA random draws are represented by captured actual inputs, fixed source/seed and exact PNG replay; CPU RNG is not assumed equivalent to CUDA RNG.",
                "Canonical model-state evidence is a producer-recorded full hash; this verifier separately checks checkpoint identity/dtype but does not reload the full UNet.",
                "Bridge contrasts change the declared sequential settings jointly with their stated numerical precision; they do not identify population or clinical utility effects.",
                "The existing M1 is a private-role non-DP diagnostic model; no new residual head, training, DP guarantee or generated-quality ranking is established."]}
        save(result_path,result)
        return result
    except Exception as error:
        save(result_path,{"status":"FAILED_SAVED_LORA_POSITIVE_CONTROL_VERIFICATION","complete":False,"phase":phase,
            "error":repr(error),"traceback":traceback.format_exc(),"checks":check.count,
            "seconds":time.perf_counter()-started,"code_sha256":source,"protocol_sha256":sha(protocol_path),
            "input_sha256":bound,"maximum_absolute_errors":check.errors})
        raise


def self_test():
    checker=Check()
    generator=np.random.default_rng(26091659)
    config=dict(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule="scaled_linear",
        prediction_type="epsilon",clip_sample=False,thresholding=False,set_alpha_to_one=False,
        steps_offset=1,timestep_spacing="leading")
    alphas=independent_alphas(config)
    for dtype in (np.float16,np.float32):
        for guidance in (1.,7.5):
            raw=generator.normal(size=(2 if guidance>1 else 1,4,32,32)).astype(dtype)
            result=cfg(raw,guidance)
            checker.require(result.shape==(1,4,32,32) and result.dtype==dtype,"synthetic CFG shape/dtype")
            if guidance==1.:
                checker.equal(result,raw,"synthetic conditional bypass")
        for t in schedule(config):
            x=generator.normal(size=(1,4,32,32)).astype(dtype)
            e=generator.normal(size=x.shape).astype(dtype)
            actual,reference,bound=ddim(x,e,int(t),alphas,config)
            checker.bounded(actual,reference,bound,"synthetic DDIM "+str(dtype)+"/"+str(t))
    checker.require(historical_seed(1)==535476699870967460 and historical_seed(2)==594566474961504545,
                    "published historical seeds independently reconstructed")
    checker.require(not torch.cuda.is_initialized(),"CPU-only self test")
    return {"status":"PASS_SYNTHETIC_CFG_DDIM_AND_HISTORICAL_SEEDS","checks":checker.count}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--out",type=Path,default=DEFAULT)
    parser.add_argument("--phase",choices=("replay","bridge"))
    parser.add_argument("--expected-code-sha256")
    args=parser.parse_args()
    if args.expected_code_sha256 and sha(__file__)!=args.expected_code_sha256:
        raise RuntimeError("Verifier code does not match supplied frozen hash")
    if args.self_test:
        result=self_test()
    else:
        if not args.phase:
            parser.error("--phase replay or bridge is required")
        result=verify(args.out,args.phase)
        result={key:result[key] for key in ("status","complete","phase","checks","seconds","code_sha256","DDIM_transitions_checked")}
    print(json.dumps(result,indent=2))
