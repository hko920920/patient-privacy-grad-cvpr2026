"""Historical LoRA replay and a fixed sequential bridge to current sampling.

Existing M1 is a private-role, non-DP diagnostic control. No residual head is
attached, no parameters are trained, and no public-backbone privacy claim follows.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_key]="4"
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime,timezone
from functools import wraps
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from u_patient_audit import models as old_models
from diffusers import StableDiffusionPipeline,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"_reports/cvpr_u_pilot_v1_001"
OLD=RUN/"quality/training_coverage_v2/step_1000_both"
CHECKPOINT=RUN/"training_coverage_v2/model_1/step_1000.pt"
PREVIOUS=ROOT/"_reports/frozen_residual_sampling_integration_20260916_v1"
CACHE=RUN/"cache/cache.pt"
OUT=ROOT/"_reports/frozen_residual_lora_positive_control_20260916_v1"
STAGES=("replay_pipeline_fp16","manual_fp16","manual_fp32_cfg75","manual_fp32_cfg1","manual_fp32_cached","manual_fp32_currentnoise")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def arr(x):return x.detach().cpu().numpy().copy()


def save(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def save_npz(path,data):
    with Path(path).open("xb") as f:np.savez_compressed(f,**data)


def canonical_state_sha(unet):
    # Promotion preserves original FP16 and FP32 values; compares across .float().
    h=hashlib.sha256()
    for name,value in unet.state_dict().items():
        a=value.detach().float().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def dtype_inventory(unet):
    return {"base":dict(Counter(str(p.dtype) for n,p in unet.named_parameters() if ".lora_" not in n)),
            "lora":dict(Counter(str(p.dtype) for n,p in unet.named_parameters() if ".lora_" in n))}


def prepare(out):
    require(not out.exists(),"Preserve previous output")
    historic=read(OLD/"report.json");previous=read(PREVIOUS/"contract.json")
    require(sha(CHECKPOINT)==historic["checkpoint_hashes"]["model_1"],"Historical checkpoint mismatch")
    require(read(PREVIOUS/"independent_verification.json")["complete"],"Verified current path required")
    snap=Path(previous["snapshot"])
    scheduler=DDIMScheduler.from_pretrained(snap/"scheduler",local_files_only=True)
    scheduler.set_timesteps(30)
    cases=[]
    sources=[Path(__file__),Path(__file__).with_name("verify_lora_positive_control.py"),
        ROOT/"u_patient_audit/generate_quality.py",ROOT/"u_patient_audit/models.py",ROOT/"u_patient_audit/common.py",
        CHECKPOINT,OLD/"report.json",RUN/"quality/visual_review_pair.json",CACHE,
        PREVIOUS/"contract.json",PREVIOUS/"manifest.json",PREVIOUS/"independent_verification.json"]
    for index,caseid in ((1,"normal"),(2,"effusion")):
        row=next(x for x in historic["images"] if x["model"]=="model_1" and x["path"]==f"model_1_{index}.png")
        require(sha(OLD/row["path"])==row["sha256"],"Historical PNG changed")
        current=next(x for x in previous["cases"] if x["case_id"]==caseid)
        require(row["prompt"]==current["prompt"],"Prompt mismatch")
        cases.append(dict(case_id=caseid,prompt=row["prompt"],historical_cuda_seed=row["seed"],
            historical_image_path=str(OLD/row["path"]),historical_image_sha256=row["sha256"],
            current_cpu_seed=current["seed"],current_conditioning_sha256=current["conditioning_sha256"],
            current_base_trace=str(PREVIOUS/"trajectories"/(caseid+"_base.npz"))))
        sources.extend([OLD/row["path"],Path(cases[-1]["current_base_trace"])])
    for relative in ("unet/config.json","unet/diffusion_pytorch_model.fp16.safetensors","vae/config.json",
                     "vae/diffusion_pytorch_model.fp16.safetensors","scheduler/scheduler_config.json",
                     "text_encoder/config.json","text_encoder/model.fp16.safetensors",
                     "tokenizer/tokenizer_config.json","tokenizer/vocab.json","tokenizer/merges.txt"):
        sources.append(snap/relative)
    sources.append(snap/"tokenizer/special_tokens_map.json")
    sources.append(ROOT.parent/"CVPR 주제 탐색/research_2026-09-10/TRACK1_LORA_POSITIVE_CONTROL_PROTOCOL_20260916.md")
    for cls in (StableDiffusionPipeline,DDIMScheduler,VaeImageProcessor,old_models.UNet2DConditionModel,old_models.cast_mixed_precision_params):
        sources.append(Path(inspect.getfile(cls)))
    out.mkdir(parents=True)
    save(out/"contract.json",dict(schema="lora-positive-control-bridge/v1",created_utc=now(),current_step=2,
        source_sha256={str(p):sha(p) for p in sources},checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
        snapshot=str(snap),model_id=previous["model_id"],model_revision=previous["model_revision"],cases=cases,stages=list(STAGES),
        scheduler_config=dict(scheduler.config),timesteps=scheduler.timesteps.tolist(),height=256,width=256,steps=30,eta=0.,
        original_precision="base FP16 + LoRA FP32, autocast FP16; VAE FP16; original pipeline text encoding",
        stage_changes=["historical replay","manual route only, all actual captured inputs identical",
            "UNet/VAE and inputs promoted FP32, no autocast; no value re-quantization",
            "CFG7.5 to1; conditional branch only","replace conditional embedding by current cache",
            "replace initial noise by exact prior current base input"],
        mixed_to_fp32_state_policy="canonical FP32 values unchanged; no round-trip .half()",
        historical_initial_latent_saved=False,historical_png_equality_target="exact file and decoded pixels",
        manual_reference_equality_target="exact all UNet inputs/outputs, guided epsilon, DDIM latents and decoded image",
        precision_or_guidance_effect_on_quality_prejudged=False,
        new_residual_head_applied=False,new_training=False,DP_claim=False,
        control_scope="M1 used private-role data; diagnostic positive control only, not a DP-safe public backbone",
        image_policy="uint8 truncation identical to historical generator; all12 fixed images retained",
        expected_total_UNet_calls=360,expected_total_UNet_batch_examples=540,expected_backward=0,
        gate="replay independently verified and visually reviewed before bridge; no outcome-driven seed/checkpoint tuning",
        whole_research_success_claim=False))
    print(json.dumps({"phase":"prepared","contract_sha256":sha(out/"contract.json")}),flush=True)


def validate(out):
    c=read(out/"contract.json")
    for path,digest in c["source_sha256"].items():require(sha(path)==digest,"Changed source: "+path)
    return c


def pipeline(c):
    old_models.setup()
    unet=old_models.load_unet(Path(c["checkpoint"]),training=False)
    pipe=StableDiffusionPipeline.from_pretrained(Path(c["snapshot"]),unet=unet,torch_dtype=torch.float16,
        variant="fp16",use_safetensors=True,safety_checker=None,feature_extractor=None,
        requires_safety_checker=False,local_files_only=True).to("cuda")
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    require(pipe.scheduler.config.prediction_type=="epsilon","Expected epsilon")
    require(dict(pipe.scheduler.config)==c["scheduler_config"] or all(dict(pipe.scheduler.config).get(k)==v
        for k,v in c["scheduler_config"].items() if not k.startswith("_")),"Scheduler settings differ")
    require(dtype_inventory(unet)["lora"]=={"torch.float32":len([p for n,p in unet.named_parameters() if '.lora_' in n])},"Original LoRA FP32 required")
    require(set(dtype_inventory(unet)["base"])=={"torch.float16"},"Original base FP16 required")
    return pipe


class Capture:
    def __init__(self,pipe):self.pipe=pipe;self.data={"model_inputs":[],"raw_eps":[],"eps":[],"latents":[],"timesteps":[]}
    def __enter__(self):
        pipe=self.pipe;data=self.data
        self.old_step=pipe.scheduler.step;self.old_decode=pipe.vae.decode;self.old_prepare=pipe.prepare_latents
        @wraps(self.old_prepare)
        def prepare(*args,**kwargs):
            result=self.old_prepare(*args,**kwargs);data["initial_from_prepare"]=arr(result);return result
        @wraps(self.old_step)
        def step(epsilon,timestep,latent,*args,**kwargs):
            if not data["latents"]:data["latents"].append(arr(latent))
            data["eps"].append(arr(epsilon));data["timesteps"].append(int(timestep.item()))
            result=self.old_step(epsilon,timestep,latent,*args,**kwargs)
            data["latents"].append(arr(result[0] if isinstance(result,tuple) else result.prev_sample));return result
        @wraps(self.old_decode)
        def decode(*args,**kwargs):
            result=self.old_decode(*args,**kwargs);raw=result[0] if isinstance(result,tuple) else result.sample
            require(bool(torch.isfinite(raw).all()),"Nonfinite raw VAE")
            data["decoded_raw"]=arr(raw);return result
        def pre(module,args,kwargs):
            data["model_inputs"].append(arr(args[0]))
            embedding=kwargs["encoder_hidden_states"]
            if "conditioning" not in data:data["conditioning"]=arr(embedding)
            else:require(np.array_equal(data["conditioning"],arr(embedding)),"Embedding changed mid sampling")
        def post(module,args,kwargs,output):data["raw_eps"].append(arr(output[0] if isinstance(output,tuple) else output.sample))
        self.handles=[]
        try:
            self.handles.append(pipe.unet.register_forward_pre_hook(pre,with_kwargs=True))
            self.handles.append(pipe.unet.register_forward_hook(post,with_kwargs=True))
            pipe.prepare_latents=prepare;pipe.scheduler.step=step;pipe.vae.decode=decode
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.pipe.prepare_latents=self.old_prepare;self.pipe.scheduler.step=self.old_step;self.pipe.vae.decode=self.old_decode
    def finish(self,image):
        result={k:(np.stack(v) if isinstance(v,list) else v) for k,v in self.data.items()}
        result["image_float"]=np.asarray(image,dtype=np.float32)
        return result


def write_record(out,case,stage,data,seconds,peak,guidance,precision):
    stem=case["case_id"]+"_"+stage
    path=out/"traces"/(stem+".npz");imagepath=out/"images"/(stem+".png")
    require(all(np.isfinite(x).all() for x in data.values()),"Nonfinite trace")
    save_npz(path,data)
    Image.fromarray(np.clip(data["image_float"]*255,0,255).astype(np.uint8)).save(imagepath)
    record=dict(case_id=case["case_id"],stage=stage,path=path.relative_to(out).as_posix(),sha256=sha(path),
        image_path=imagepath.relative_to(out).as_posix(),image_sha256=sha(imagepath),seconds=seconds,peak_memory_bytes=peak,
        guidance=guidance,precision=precision,unet_calls=30,batch_examples=60 if guidance>1 else 30)
    print(json.dumps({"phase":stage,"case":case["case_id"],"seconds":round(seconds,3)}),flush=True)
    return record


def replay(out):
    c=validate(out);require(not(out/"execution_replay.json").exists(),"Replay exists")
    (out/"traces").mkdir();(out/"images").mkdir();start=time.perf_counter()
    pipe=pipeline(c);before=canonical_state_sha(pipe.unet);inventory=dtype_inventory(pipe.unet)
    save(out/"runtime_replay.json",dict(packages={k:version(k) for k in ("torch","diffusers","peft","transformers","numpy")},
        GPU=torch.cuda.get_device_name(),unet_dtype_inventory=inventory,vae_dtype=str(next(pipe.vae.parameters()).dtype),
        scheduler_config=dict(pipe.scheduler.config),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32))
    records=[]
    for case in c["cases"]:
        gen=torch.Generator(device="cuda").manual_seed(case["historical_cuda_seed"])
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16),Capture(pipe) as capture:
            image=pipe(case["prompt"],height=256,width=256,num_inference_steps=30,guidance_scale=7.5,
                generator=gen,output_type="np").images[0]
        torch.cuda.synchronize()
        data=capture.finish(image)
        records.append(write_record(out,case,STAGES[0],data,time.perf_counter()-tick,torch.cuda.max_memory_allocated(),7.5,"original_mixed_autocast_fp16"))
    after=canonical_state_sha(pipe.unet);require(before==after,"Replay mutated weights")
    save(out/"manifest_replay.json",records)
    save(out/"execution_replay.json",dict(completed=True,utc=now(),seconds=time.perf_counter()-start,
        contract_sha256=sha(out/"contract.json"),manifest_sha256=sha(out/"manifest_replay.json"),runtime_sha256=sha(out/"runtime_replay.json"),
        canonical_model_sha256_before=before,canonical_model_sha256_after=after,unet_calls=60,batch_examples=120,backward=0,
        gradients_present=any(p.grad is not None for p in pipe.unet.parameters())))


def manual(pipe,initial,conditioning,guidance,autocast):
    sched=DDIMScheduler.from_config(pipe.scheduler.config);sched.set_timesteps(30,device="cuda")
    latent=initial.clone();data={"latents":[arr(latent)],"model_inputs":[],"raw_eps":[],"eps":[]}
    with torch.inference_mode(),(torch.autocast("cuda",dtype=torch.float16) if autocast else nullcontext()):
        for timestep in sched.timesteps:
            inputs=torch.cat([latent]*2) if guidance>1 else latent
            inputs=sched.scale_model_input(inputs,timestep)
            raw=pipe.unet(inputs,timestep,encoder_hidden_states=conditioning,return_dict=False)[0]
            if guidance>1:
                null,conditional=raw.chunk(2);epsilon=null+guidance*(conditional-null)
            else:epsilon=raw
            data["model_inputs"].append(arr(inputs));data["raw_eps"].append(arr(raw));data["eps"].append(arr(epsilon))
            latent=sched.step(epsilon,timestep,latent,eta=0.,return_dict=False)[0]
            data["latents"].append(arr(latent))
        decoded=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
        require(bool(torch.isfinite(decoded).all()),"Nonfinite unclamped VAE")
        image=pipe.image_processor.postprocess(decoded,output_type="np",do_denormalize=[True])[0]
    result={k:np.stack(v) for k,v in data.items()}
    result.update(conditioning=arr(conditioning),timesteps=arr(sched.timesteps),decoded_raw=arr(decoded),image_float=np.asarray(image,dtype=np.float32),initial_from_prepare=arr(initial))
    return result


def bridge(out):
    c=validate(out);require(not(out/"execution_bridge.json").exists(),"Bridge exists")
    verify=read(out/"replay_verification.json");require(verify["complete"] and verify["status"].startswith("PASS"),"Verified original replay required")
    for name,digest in verify["input_sha256"].items():require(sha(out/name)==digest,"Replay verification input changed")
    start=time.perf_counter();pipe=pipeline(c)
    pipe.text_encoder.to("cpu");torch.cuda.empty_cache()
    before=canonical_state_sha(pipe.unet)
    require(before==read(out/"execution_replay.json")["canonical_model_sha256_before"],"Fresh load differs")
    original={case["case_id"]:dict(np.load(out/"traces"/(case["case_id"]+"_"+STAGES[0]+".npz"))) for case in c["cases"]}
    cache=torch.load(CACHE,map_location="cpu",weights_only=True)
    records=[];initial_inventory=dtype_inventory(pipe.unet)
    for stage in STAGES[1:]:
        if stage=="manual_fp32_cfg75":pipe.unet.float();pipe.vae.float()
        half=stage=="manual_fp16";dtype=torch.float16 if half else torch.float32
        guidance=7.5 if stage in ("manual_fp16","manual_fp32_cfg75") else 1.
        for case in c["cases"]:
            source=original[case["case_id"]]
            initial=torch.from_numpy(source["latents"][0]).to(device="cuda",dtype=dtype)
            embedding=torch.from_numpy(source["conditioning"]).to(device="cuda",dtype=dtype)
            if guidance==1.:embedding=embedding[1:]
            if stage in ("manual_fp32_cached","manual_fp32_currentnoise"):
                embedding=cache["hidden"][case["prompt"]].float().to("cuda")
                require(tensor_sha(embedding)==case["current_conditioning_sha256"],"Current cache embedding mismatch")
            if stage=="manual_fp32_currentnoise":
                previous=np.load(case["current_base_trace"])
                initial=torch.from_numpy(previous["latents"][0]).to("cuda")
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            data=manual(pipe,initial,embedding,guidance,half)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
            if stage=="manual_fp16":
                for key in ("conditioning","timesteps","latents","model_inputs","raw_eps","eps","decoded_raw","image_float"):
                    require(np.array_equal(data[key],source[key]),"Matched manual/pipeline mismatch: "+key)
            records.append(write_record(out,case,stage,data,seconds,peak,guidance,"original_mixed_autocast_fp16" if half else "fp32_no_autocast"))
    after=canonical_state_sha(pipe.unet);require(before==after,"Bridge altered canonical weight values")
    require(all(p.grad is None and not p.requires_grad for p in pipe.unet.parameters()),"Unexpected gradient")
    require(len(pipe.unet._forward_pre_hooks)==len(pipe.unet._forward_hooks)==0,"Unexpected remaining UNet hooks")
    save(out/"manifest_bridge.json",records)
    save(out/"execution_bridge.json",dict(completed=True,utc=now(),seconds=time.perf_counter()-start,
        contract_sha256=sha(out/"contract.json"),manifest_sha256=sha(out/"manifest_bridge.json"),
        replay_verification_sha256=sha(out/"replay_verification.json"),
        canonical_model_sha256_before=before,canonical_model_sha256_after=after,
        initial_dtype_inventory=initial_inventory,final_dtype_inventory=dtype_inventory(pipe.unet),
        unet_calls=300,batch_examples=420,backward=0,gradients_present=False,new_head_applied=False))
    grid=Image.new("RGB",(len(STAGES)*256,2*282),"white");draw=ImageDraw.Draw(grid)
    for ri,case in enumerate(c["cases"]):
        for ci,stage in enumerate(STAGES):
            grid.paste(Image.open(out/"images"/(case["case_id"]+"_"+stage+".png")),(ci*256,ri*282+26))
            draw.text((ci*256+4,ri*282+6),case["case_id"]+" / "+stage,fill="black")
    grid.save(out/"comparison_grid.png")
    print("LORA_POSITIVE_CONTROL_BRIDGE_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","replay","bridge"))
    parser.add_argument("--out",type=Path,default=OUT);args=parser.parse_args()
    {"prepare":prepare,"replay":replay,"bridge":bridge}[args.phase](args.out)
