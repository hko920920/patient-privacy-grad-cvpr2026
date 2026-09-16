"""Independent CPU verification of saved frozen-head sampling integration.

No UNet inference, image decoding or producer numerical function is executed.
Stored residual arithmetic and every DDIM transition are checked separately from
the exact base/zero/restored forward-replay evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable,"1")
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"_reports/frozen_residual_sampling_integration_20260916_v1"
STATUS="PASS_SAVED_SAMPLING_INTEGRATION_RESIDUAL_DDIM_AND_BASE_RESTORATION"
U32=float(np.finfo(np.float32).eps)/2
GAMMA32=32*U32/(1-32*U32)
TINY32=float(np.finfo(np.float32).tiny)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path,allow_pickle=False) as packet:
        return {key:packet[key].copy() for key in packet.files}


def save(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write("\n")


class Check:
    def __init__(self):
        self.count=0
        self.errors={}
    def require(self,value,label):
        self.count+=1
        if not bool(value):
            raise AssertionError(label)
    def equal(self,a,b,label):
        x,y=np.asarray(a),np.asarray(b)
        self.require(x.dtype==y.dtype and x.shape==y.shape,label+": dtype/shape")
        self.require(np.array_equal(x,y),label+": exact")
    def close(self,a,b,label,atol=1e-12,rtol=1e-10):
        x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        self.require(x.shape==y.shape,label+": shape")
        self.require(np.isfinite(x).all() and np.isfinite(y).all(),label+": finite")
        error=np.abs(x-y)
        self.errors[label]=float(error.max(initial=0))
        self.require(np.all(error<=atol+rtol*np.abs(y)),label+": arithmetic")
    def envelope(self,a,b,bound,label):
        x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        bound=np.asarray(bound,dtype=np.float64)
        self.require(x.shape==y.shape==bound.shape,label+": shape")
        self.require(np.isfinite(x).all() and np.isfinite(y).all()
                     and np.isfinite(bound).all() and np.all(bound>=0),label+": finite")
        error=np.abs(x-y)
        self.errors[label]={"max_abs_error":float(error.max(initial=0)),
            "max_rounding_envelope":float(bound.max(initial=0)),
            "max_error_over_envelope":float(np.max(error/np.maximum(bound,np.finfo(float).tiny),initial=0))}
        self.require(np.all(error<=bound),label+": operation-scaled FP32 envelope")


def nchw(flat):
    return np.asarray(flat).reshape(32,32,4).transpose(2,0,1)[None]


def basis(t,alphas,lo,hi):
    a=float(alphas[int(t)])
    logsnr=math.log(a)-math.log1p(-a)
    u=2*(logsnr-lo)/(hi-lo)-1
    return np.array([1.,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2],dtype=np.float64)


def residual(features,b,w):
    """Direct basis-major 64-column NumPy design, separate from GPU adapter."""
    phi=np.concatenate([float(bk)*np.asarray(features,dtype=np.float64) for bk in b],axis=1)
    return phi@np.asarray(w,dtype=np.float64)


def check_residual(check,features,b,w,base,saved_residual,corrected,label):
    check.require(features.shape==(1024,16) and features.dtype==np.float32,label+": projected features")
    check.equal(features[:,0],np.ones(1024,dtype=np.float32),label+": constant feature")
    check.require(np.asarray(b).shape==(4,) and np.asarray(b).dtype==np.float64,label+": basis")
    check.require(w.shape==(64,4) and w.dtype==np.float64,label+": full64 head")
    check.require(base.shape==(1,4,32,32) and base.dtype==np.float32,label+": epsilon base")
    expected=residual(features,b,w)
    check.close(saved_residual,expected,label+": FP64 residual")
    # One explicit FP64->FP32 cast followed by one FP32 addition.
    exact_add=np.add(base,nchw(saved_residual.astype(np.float32)),dtype=np.float32)
    check.equal(corrected,exact_add,label+": cast then add")


def independent_alphas(config):
    if config.get("trained_betas") is not None:
        betas=torch.tensor(config["trained_betas"],dtype=torch.float32)
    elif config["beta_schedule"]=="scaled_linear":
        betas=torch.linspace(math.sqrt(config["beta_start"]),math.sqrt(config["beta_end"]),
                             config["num_train_timesteps"],dtype=torch.float32).square()
    elif config["beta_schedule"]=="linear":
        betas=torch.linspace(config["beta_start"],config["beta_end"],
                             config["num_train_timesteps"],dtype=torch.float32)
    else:
        raise AssertionError("Unsupported unreviewed beta schedule")
    return torch.cumprod(1-betas,dim=0).numpy()


def schedule(config,steps=30):
    count=config["num_train_timesteps"]
    spacing=config.get("timestep_spacing","leading")
    if spacing=="leading":
        return np.array([j*(count//steps)+config.get("steps_offset",0)
                         for j in reversed(range(steps))],dtype=np.int64)
    if spacing=="linspace":
        return np.rint(np.linspace(0,count-1,steps))[::-1].astype(np.int64)
    if spacing=="trailing":
        return np.rint(np.arange(count,0,-count/steps)).astype(np.int64)-1
    raise AssertionError("Unsupported timestep spacing")


def ddim_step(sample,epsilon,timestep,alphas,config,steps=30):
    """Independent eta0/epsilon/no-threshold DDIM equation, CPU FP32 and FP64."""
    if config["prediction_type"]!="epsilon" or config.get("thresholding",False):
        raise AssertionError("Only frozen epsilon/nonthreshold DDIM is reviewed")
    # Reviewed base config has clip_sample=False; fail rather than silently adapt.
    if config.get("clip_sample",True):
        raise AssertionError("Unexpected clipping changes the reviewed integration")
    previous=int(timestep)-int(config["num_train_timesteps"])//steps
    a=np.float32(alphas[int(timestep)])
    ap=np.float32(alphas[previous] if previous>=0 else (1. if config["set_alpha_to_one"] else alphas[0]))
    x=torch.from_numpy(np.asarray(sample,dtype=np.float32))
    e=torch.from_numpy(np.asarray(epsilon,dtype=np.float32))
    at,apt=torch.tensor(a),torch.tensor(ap)
    x0=(x-torch.sqrt(1-at)*e)/torch.sqrt(at)
    cpu=(torch.sqrt(apt)*x0+torch.sqrt(1-apt)*e).numpy()
    xd,ed=np.asarray(sample,dtype=np.float64),np.asarray(epsilon,dtype=np.float64)
    a,ap=float(a),float(ap)
    x0d=(xd-math.sqrt(1-a)*ed)/math.sqrt(a)
    exact=math.sqrt(ap)*x0d+math.sqrt(1-ap)*ed
    # Conservative elementwise rounding envelope for <=32 scalar FP32
    # subtraction/sqrt/multiply/divide/add operations, expressed in magnitudes
    # before cancellation. It is not an error bound for UNet inference.
    magnitude=(math.sqrt(ap)*(np.abs(xd)+math.sqrt(1-a)*np.abs(ed))/math.sqrt(a)
               +math.sqrt(1-ap)*np.abs(ed))
    return cpu,exact,GAMMA32*magnitude+32*TINY32


def noised(clean,noise,timestep,alphas):
    a=torch.tensor(float(alphas[int(timestep)]),dtype=torch.float32)
    return (a.sqrt()*torch.from_numpy(clean)+(1-a).sqrt()*torch.from_numpy(noise)).numpy()


def check_witness(check,packet,old,row,cache,weights,alphas,lo,hi,projection,tolerances):
    label="witness/"+str(row["record_id"])
    clean=cache["latents"][row["image_id"]].float().cpu().numpy()
    hidden=cache["hidden"][row["prompt"]].float().cpu().numpy()
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(row["noise_seed"]),
                      dtype=torch.float32).numpy()
    check.equal(packet["clean"],clean,label+": cached clean latent")
    check.equal(packet["conditioning"],hidden,label+": exact conditional prompt")
    check.equal(packet["noise"],noise,label+": fixed CPU diffusion noise")
    check.equal(noise,nchw(old["target"]),label+": original target is same noise")
    check.equal(packet["noisy"],noised(clean,noise,row["timestep"],alphas),label+": CPU FP32 original noising")
    check.require(int(packet["timestep"])==row["timestep"],label+": source timestep")
    original_base=nchw(old["base"])
    check.close(packet["plain"],original_base,label+": original public base prediction",
                atol=tolerances["saved_base_features_atol"],rtol=tolerances["saved_base_features_rtol"])
    for key in ("zero","restored"):
        check.equal(packet[key],packet["plain"],label+": "+key+" exact base restoration")
    check.equal(packet["repeat_public"],packet["public_corrected"],label+": deterministic second public call")
    h=old["hidden"].astype(np.float64)
    p=projection.astype(np.float32).astype(np.float64)
    product=h@p
    gamma=320*U32/(1-320*U32)
    bound=2*gamma*(np.abs(h)@np.abs(p))+32*TINY32
    check.envelope(old["features"][:,1:],product,bound,label+": original FP32 projection witness")
    expected_basis=basis(row["timestep"],alphas,lo,hi)
    check.close(old["basis"],expected_basis,label+": original public time basis")
    for method in ("public","pooled"):
        check.equal(packet[method+"_base"],packet["plain"],label+": "+method+" unchanged base")
        check.close(packet[method+"_features"],old["features"],label+": "+method+" original projected feature",
                    atol=tolerances["saved_base_features_atol"],rtol=tolerances["saved_base_features_rtol"])
        check.close(packet[method+"_basis"],expected_basis,label+": "+method+" time basis")
        check_residual(check,packet[method+"_features"],packet[method+"_basis"],weights[method],
                       packet[method+"_base"],packet[method+"_residual"],packet[method+"_corrected"],
                       label+"/"+method)
    return {"base_exact_to_original_public_raw":bool(np.array_equal(packet["plain"],original_base)),
            "public_features_exact_to_original":bool(np.array_equal(packet["public_features"],old["features"])),
            "pooled_features_exact_to_original":bool(np.array_equal(packet["pooled_features"],old["features"]))}


def check_trajectory(check,packet,method,case,weights,alphas,config,lo,hi,tolerances):
    label="trajectory/"+case["case_id"]+"/"+method
    times=schedule(config)
    check.equal(packet["timesteps"],times,label+": exact DDIM timestep sequence")
    for field in ("latents","scaled_inputs","base_eps","eps"):
        expected=(31,1,4,32,32) if field=="latents" else (30,1,4,32,32)
        check.require(packet[field].shape==expected and packet[field].dtype==np.float32,
                      label+": "+field+" shape/dtype")
        check.require(np.isfinite(packet[field]).all(),label+": finite "+field)
    initial=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(case["seed"]),
                        dtype=torch.float32).numpy()
    check.equal(packet["latents"][0],initial,label+": fixed CPU initial noise (sigma1)")
    if method in ("zero","public","pooled"):
        for field,shape,dtype in (("features",(30,1024,16),np.float32),
                                  ("basis",(30,4),np.float64),("residual",(30,1024,4),np.float64)):
            check.require(packet[field].shape==shape and packet[field].dtype==dtype,
                          label+": "+field+" capture shape/dtype")
    else:
        check.equal(packet["eps"],packet["base_eps"],label+": unwrapped model output")
    exact_cpu_steps=0
    for index,t in enumerate(times):
        check.equal(packet["scaled_inputs"][index],packet["latents"][index],label+": DDIM input scale identity "+str(index))
        if method in ("zero","public","pooled"):
            expected_basis=basis(int(t),alphas,lo,hi)
            check.close(packet["basis"][index],expected_basis,label+": basis "+str(index))
            check_residual(check,packet["features"][index],packet["basis"][index],weights[method],
                           packet["base_eps"][index],packet["residual"][index],packet["eps"][index],
                           label+": residual "+str(index))
        cpu,exact,bound=ddim_step(packet["latents"][index],packet["eps"][index],int(t),alphas,config)
        actual=packet["latents"][index+1]
        exact_cpu_steps+=int(np.array_equal(actual,cpu))
        check.close(actual,cpu,label+": contracted CPU DDIM "+str(index),
                    atol=tolerances["scheduler_fp32_atol"],rtol=tolerances["scheduler_fp32_rtol"])
        check.envelope(actual,exact,bound,label+": DDIM transition "+str(index))
        # The separately ordered CPU FP32 implementation must itself lie within
        # the same mathematical FP64 rounding envelope.
        check.envelope(cpu,exact,bound,label+": CPU DDIM consistency "+str(index))
    return {"steps":30,"exact_cpu_fp32_steps":exact_cpu_steps,
            "all_steps_within_operation_scaled_envelope":True}


def verify_inputs(out,check):
    c=read(out/"contract.json")
    check.require(c["schema"]=="sampling-integration/v1","frozen integration schema")
    for path,digest in c["source_sha256"].items():
        check.require(sha(path)==digest,"frozen source/input "+path)
    check.require(c["source_sha256"][str(Path(__file__).resolve())]==sha(__file__),"verifier bound before GPU")
    def bound(suffix):
        found=[Path(p) for p in c["source_sha256"] if p.replace("\\","/").endswith(suffix)]
        check.require(len(found)==1,"one bound "+suffix)
        return found[0]
    cap=bound("frozen_residual_capacity_20260916_v1/contract.json").parent
    pub=bound("frozen_residual_public32_20260916_v1/manifest.json").parent
    pool=bound("frozen_residual_pooled_reference_20260916_v1/models.npz").parent
    cache_path=bound("cvpr_u_pilot_v1_001/cache/cache.pt")
    capc,pubc=read(cap/"contract.json"),read(pub/"contract.json")
    check.require(c["model_id"]==capc["model_id"] and c["model_revision"]==capc["model_revision"],
                  "fixed model identity")
    check.require(c["snapshot"]==capc["snapshot"],"fixed artifact snapshot")
    check.require(c["guidance_scale"]==1. and c["eta"]==0. and c["num_inference_steps"]==30,
                  "conditional-only eta0 thirty-step integration")
    check.require(c["height"]==c["width"]==256 and c["prediction_type"]=="epsilon","P256 epsilon model")
    check.require(c["backward"]==c["optimization_steps"]==0 and not c["efficacy_or_clinical_claim"]
                  and not c["final_cohort_evaluation"],"no optimization/efficacy/final cohort use")
    check.require(c["trajectory_methods"]==["base","zero","public","pooled","base_restored"],"fixed five methods")
    check.require([(x["case_id"],x["seed"]) for x in c["cases"]]==[("normal",26091631),("effusion",26091632)],
                  "two fixed cases/no seed search")
    expected_prompts={"normal":"a frontal posteroanterior chest radiograph with no labeled finding",
                      "effusion":"a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion"}
    for case in c["cases"]:
        check.require(case["prompt"]==expected_prompts[case["case_id"]],"fixed conditional prompt")
    check.require(c["logsnr_min"]==capc["time_basis_logsnr_min"]
                  and c["logsnr_max"]==capc["time_basis_logsnr_max"],"original basis bounds")
    projection=arrays(cap/"projection.npz")
    check.require(sha(cap/"projection.npz")==capc["projection_sha256"]==pubc["projection_sha256"],
                  "training/public projection identity")
    alphas=independent_alphas(c["scheduler_config"])
    check.equal(projection["alphas"],alphas,"independently constructed FP32 alpha schedule")
    check.equal(np.asarray(c["timesteps"],dtype=np.int64),schedule(c["scheduler_config"]),"declared DDIM sequence")
    for folder,status in ((pub,"PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE"),
            (pool,"PASS_POOLED_REFERENCE_SAVED_PATIENT_AVERAGES_RIDGE_AND_DEVELOPMENT_LOSSES")):
        prior=read(folder/"independent_verification.json")
        check.require(prior["status"]==status and prior["complete"] is True,"prior independent PASS")
        for name,digest in prior["input_sha256"].items():
            check.require(sha(folder/name)==digest,"prior input unchanged "+name)
    original=read(pub/"manifest.json")
    expected=[original[i] for i in (0,7,16,23)]
    check.require(c["witnesses"]==expected,"exact four predeclared public witnesses")
    for row in expected:
        check.require(row["split"]=="public" and row["original_eval_role"]=="quality","public-only witness")
        check.require(sha(pub/row["raw_path"])==row["raw_sha256"],"original public witness hash")
    check.require(sha(cache_path)==pubc["source_sha256"][str(cache_path)],"original frozen latent/prompt cache")
    cache=torch.load(cache_path,map_location="cpu",weights_only=True)
    for case in c["cases"]:
        check.require(case["prompt"] in cache["hidden"],"sampling condition exists in frozen cache")
        conditioning=cache["hidden"][case["prompt"]].float().cpu().numpy()
        check.require(conditioning.shape==(1,77,1024) and conditioning.dtype==np.float32,
                      "sampling conditioning shape and promotion")
        check.require(hashlib.sha256(conditioning.tobytes()).hexdigest()==case["conditioning_sha256"],
                      "sampling conditioning independently bound to cached tensor")
    weights={}
    expected_keys={"zero":"W_full_base","public":"W_full_public_only","pooled":"W_full_pooled"}
    check.require(set(c["methods"])==set(expected_keys),"three fixed head files")
    for method,item in c["methods"].items():
        check.require(Path(item["path"]).resolve()==(pool/"models.npz").resolve()
                      and item["key"]==expected_keys[method],"fixed head provenance")
        check.require(sha(item["path"])==item["sha256"],"head archive hash")
        weights[method]=arrays(item["path"])[item["key"]]
        check.require(weights[method].shape==(64,4) and weights[method].dtype==np.float64
                      and np.isfinite(weights[method]).all(),"full64 finite saved head")
    check.equal(weights["zero"],np.zeros((64,4),dtype=np.float64),"actual zero model")
    execution,runtime,manifest=read(out/"execution.json"),read(out/"runtime.json"),read(out/"manifest.json")
    check.require(execution["completed"] is True and execution["status"]=="COMPLETED_PENDING_INDEPENDENT_VERIFICATION",
                  "completed producer packet")
    for key,name in (("contract_sha256","contract.json"),("manifest_sha256","manifest.json"),("runtime_sha256","runtime.json")):
        check.require(execution[key]==sha(out/name),"execution binding "+name)
    check.require(execution["forward_calls"]==c["maximum_expected_unet_forward"]==325,
                  "24 witness +300 trajectory +1 warmup forwards")
    check.require(execution["backward"]==0 and execution["parameter_gradients_present"] is False,
                  "no backward or parameter gradients")
    check.require(execution["witness_count"]==4 and execution["trajectories"]==10
                  and execution["generated_preview_images"]==0,"packet scope before decoding")
    before,after=execution["model_state_sha256_before"],execution["model_state_sha256_after"]
    check.require(before==after and len(before)==64,"reported exact model-state before/after")
    check.require(execution["remaining_conv_out_hooks"]==runtime["initial_conv_out_hooks"]==0,"all capture hooks removed")
    check.require(runtime["dtype"]=="torch.float32" and runtime["conditional_only"] is True
                  and runtime["tf32_matmul"] is False and runtime["tf32_cudnn"] is False,"actual declared precision")
    check.require(runtime["scheduler_class"]=="DDIMScheduler" and runtime["timesteps"]==c["timesteps"],
                  "actual scheduler/time grid")
    for key,value in c["scheduler_config"].items():
        if not key.startswith("_"):
            check.require(runtime["scheduler_config"][key]==value,"scheduler runtime config "+key)
    check.require(runtime["alphas_sha256"]==hashlib.sha256(alphas.tobytes()).hexdigest(),"runtime alpha hash")
    check.require(runtime["vae_config"]==read(Path(c["snapshot"])/"vae/config.json"),"bound VAE config for later preview")
    return c,cache,weights,alphas,projection,pub,manifest


def verify(out):
    out=Path(out).resolve()
    result_path=out/"independent_verification.json"
    protocol_path=out/"independent_verification_protocol.json"
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve existing sampling verification attempt")
    if read(out/"execution.json").get("completed") is not True:
        raise RuntimeError("Completed stored integration packet required")
    names=("contract.json","runtime.json","manifest.json","execution.json")
    bound={name:sha(out/name) for name in names}
    source=sha(__file__)
    protocol={"schema":"independent-saved-sampling-integration-verification/v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),"code_sha256":source,"input_sha256":bound,
        "numeric_policy":"Residual FP64 atol1e-12/rtol1e-10; explicit cast+addition exact; zero/restored/repeat exact. Original witness feature/base and CPU DDIM use frozen producer contract tolerances.",
        "DDIM_additional_rounding_check":{"unit_roundoff_float32":U32,"gamma32":GAMMA32,
            "envelope":"gamma32*(sqrt(ap)*(abs(x)+sqrt(1-a)*abs(eps))/sqrt(a)+sqrt(1-ap)*abs(eps))+32*tiny32",
            "scope":"Elementwise eta0 epsilon update only; not an error bound for UNet inference."},
        "independence":"No producer numerical imports, UNet forward or image decode. NumPy basis/residual, separately implemented CPU DDIM equation and seeded CPU input replay.",
        "source_frozen_before_producer_execution":"Checked against producer contract source_sha256",
        "clinical_or_efficacy_claim":False}
    save(protocol_path,protocol)
    started=time.perf_counter()
    check=Check()
    try:
        c,cache,weights,alphas,projection,pub,manifest=verify_inputs(out,check)
        check.require(set(manifest)=={"witnesses","trajectories"},"packet manifest fields")
        check.require([r["record_id"] for r in manifest["witnesses"]]==[0,7,16,23],"four ordered witness rows")
        witness_results={}
        for row,entry in zip(c["witnesses"],manifest["witnesses"]):
            expected_path="witnesses/%06d.npz"%row["record_id"]
            check.require(entry["path"]==expected_path,"canonical witness packet path")
            check.require(sha(out/entry["path"])==entry["sha256"],"saved witness hash")
            check.require(Path(entry["original_raw_path"]).resolve()==(pub/row["raw_path"]).resolve()
                          and entry["original_raw_sha256"]==row["raw_sha256"],"bound original witness")
            check.require(entry["noise_seed"]==row["noise_seed"],"witness seed metadata")
            packet=arrays(out/entry["path"])
            old=arrays(pub/row["raw_path"])
            witness_results[str(row["record_id"])]=check_witness(check,packet,old,row,cache,weights,alphas,
                c["logsnr_min"],c["logsnr_max"],projection["P"],c["tolerances"])
        expected_pairs=[(case["case_id"],method) for case in c["cases"] for method in c["trajectory_methods"]]
        check.require([(r["case_id"],r["method"]) for r in manifest["trajectories"]]==expected_pairs,"ten ordered trajectories")
        packet_cache={}
        trajectory_results={}
        for entry in manifest["trajectories"]:
            case=next(x for x in c["cases"] if x["case_id"]==entry["case_id"])
            method=entry["method"]
            key=case["case_id"]+"/"+method
            check.require(entry["prompt"]==case["prompt"] and entry["seed"]==case["seed"]
                          and entry["conditioning_sha256"]==case["conditioning_sha256"]
                          and entry["unet_forward"]==30,"trajectory condition/seed/count")
            check.require(entry["path"]=="trajectories/"+case["case_id"]+"_"+method+".npz","canonical trajectory path")
            check.require(sha(out/entry["path"])==entry["sha256"],"trajectory file hash")
            packet=arrays(out/entry["path"]);packet_cache[(case["case_id"],method)]=packet
            trajectory_results[key]=check_trajectory(check,packet,method,case,weights,alphas,c["scheduler_config"],
                c["logsnr_min"],c["logsnr_max"],c["tolerances"])
        for case in c["cases"]:
            for method in ("zero","base_restored"):
                for key in ("timesteps","latents","scaled_inputs","base_eps","eps"):
                    check.equal(packet_cache[(case["case_id"],method)][key],packet_cache[(case["case_id"],"base")][key],
                                case["case_id"]+"/"+method+": full exact base trajectory "+key)
        for name,digest in bound.items():
            check.require(sha(out/name)==digest,"unchanged verification input "+name)
        for path,digest in c["source_sha256"].items():
            check.require(sha(path)==digest,"unchanged producer-bound source/input "+path)
        for section in manifest.values():
            for entry in section:
                check.require(sha(out/entry["path"])==entry["sha256"],"unchanged saved packet")
        check.require(not torch.cuda.is_initialized(),"verifier did not initialize CUDA")
        result={"status":STATUS,"complete":True,"checks":check.count,"seconds":time.perf_counter()-started,
            "code_sha256":source,"protocol_sha256":sha(protocol_path),"input_sha256":bound,
            "witnesses":witness_results,"trajectories":trajectory_results,"maximum_absolute_errors":check.errors,
            "DDIM_transitions_checked":300,"new_UNet_forwards":0,"new_VAE_decodes":0,
            "limits":["CPU checks of saved tensors; no independent UNet inference at every trajectory step.",
                "Four original hidden/projection witnesses support the feature tap; hidden320 is not saved for all trajectory steps.",
                "Model-state equality is the producer's hash-bound before/after evidence, not a separately loaded full UNet fingerprint.",
                "Prompt metadata/cache binding is checked; no new text encoder is run.",
                "No generated-image, denoising-efficacy, privacy or clinical-utility conclusion."]}
        save(result_path,result)
        return result
    except Exception as exc:
        save(result_path,{"status":"FAILED_SAVED_SAMPLING_INTEGRATION_VERIFICATION","complete":False,
            "error":repr(exc),"traceback":traceback.format_exc(),"checks":check.count,
            "maximum_absolute_errors":check.errors,"seconds":time.perf_counter()-started,
            "code_sha256":source,"protocol_sha256":sha(protocol_path),"input_sha256":bound})
        raise


def self_test():
    rng=np.random.default_rng(26091642)
    f=rng.normal(size=(1024,16)).astype(np.float32);f[:,0]=1
    b=np.array([1.,.2,-.44,-.28]);w=rng.normal(size=(64,4))*.01
    base=rng.normal(size=(1,4,32,32)).astype(np.float32)
    r=residual(f,b,w)
    corrected=base+nchw(r.astype(np.float32))
    check=Check()
    check_residual(check,f,b,w,base,r,corrected,"synthetic")
    config={"num_train_timesteps":1000,"beta_start":.00085,"beta_end":.012,
            "beta_schedule":"scaled_linear","prediction_type":"epsilon","clip_sample":False,
            "thresholding":False,"set_alpha_to_one":False,"steps_offset":1,"timestep_spacing":"leading"}
    alphas=independent_alphas(config)
    for t in schedule(config):
        cpu,reference,bound=ddim_step(base,corrected,int(t),alphas,config)
        check.envelope(cpu,reference,bound,"synthetic DDIM "+str(t))
    return {"status":"PASS_SYNTHETIC_RESIDUAL_CAST_AND_DDIM_ENVELOPE","checks":check.count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DEFAULT)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--expected-code-sha256")
    args=p.parse_args()
    if args.expected_code_sha256 and sha(__file__)!=args.expected_code_sha256:
        raise AssertionError("Verifier source hash mismatch")
    result=self_test() if args.self_test else verify(args.out)
    print(json.dumps({key:result[key] for key in
          ("status","complete","checks","seconds","code_sha256","DDIM_transitions_checked")
          if key in result},indent=2),flush=True)


if __name__=="__main__":
    main()
