"""Bounded sampling-path correctness packet, then optional six-image preview.

No optimization, new DP mechanism, efficacy test, or seed selection occurs.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from dp_protocol.calibrate_xray_public_clip_norms import disable_broken_optional_onnx
disable_broken_optional_onnx()
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler, AutoencoderKL
from .sampling_adapter import FrozenResidualAdapter, require

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "_reports/frozen_residual_capacity_20260916_v1"
PUB = ROOT / "_reports/frozen_residual_public32_20260916_v1"
POOL = ROOT / "_reports/frozen_residual_pooled_reference_20260916_v1"
CACHE = ROOT / "_reports/cvpr_u_pilot_v1_001/cache/cache.pt"
OUT = ROOT / "_reports/frozen_residual_sampling_integration_20260916_v1"
METHODS = ("base", "zero", "public", "pooled", "base_restored")
CASES = [dict(case_id="normal", prompt="a frontal posteroanterior chest radiograph with no labeled finding", seed=26091631),
         dict(case_id="effusion", prompt="a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion", seed=26091632)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def npz(path, arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def setup():
    require(torch.cuda.is_available(), "CUDA is required for this fixed run")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def prepare(out):
    require(not out.exists(), "Output already exists")
    cap, public = read(CAP/"contract.json"), read(PUB/"contract.json")
    snap = Path(cap["snapshot"])
    for folder in (PUB, POOL):
        require(read(folder/"independent_verification.json")["status"].startswith("PASS"), "Verified inputs required")
    require(sha(CACHE) == public["source_sha256"][str(CACHE)], "Frozen input cache changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    cases = [dict(case) for case in CASES]
    for case in cases:
        require(case["prompt"] in cache["hidden"], "Use cached conditional embeddings")
        case["conditioning_sha256"] = hashlib.sha256(cache["hidden"][case["prompt"]].float().numpy().tobytes()).hexdigest()
    sched = DDIMScheduler.from_pretrained(snap/"scheduler", local_files_only=True)
    require(sched.config.prediction_type == cap["prediction_type"] == "epsilon", "Unsupported prediction type")
    sched.set_timesteps(30)
    with np.load(CAP/"projection.npz") as p:
        require(np.array_equal(p["alphas"], sched.alphas_cumprod.numpy()), "Training/sampling alpha mismatch")
    manifest = read(PUB/"manifest.json")
    witnesses = [manifest[i] for i in (0,7,16,23)]
    sources = [Path(__file__), Path(__file__).with_name("sampling_adapter.py"),
        Path(__file__).with_name("verify_sampling_integration.py"), CACHE,
        CAP/"contract.json", CAP/"projection.npz", PUB/"contract.json", PUB/"manifest.json",
        PUB/"independent_verification.json", POOL/"contract.json", POOL/"models.npz", POOL/"independent_verification.json"]
    sources.extend(Path(inspect.getfile(cls)) for cls in (DDIMScheduler,DDPMScheduler,UNet2DConditionModel,AutoencoderKL))
    sources.append(ROOT.parent / "CVPR 주제 탐색/research_2026-09-10/TRACK1_SAMPLING_INTEGRATION_PROTOCOL_20260916.md")
    for name in ("unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                 "scheduler/scheduler_config.json", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"):
        sources.append(snap/name)
    for row in witnesses:
        p=PUB/row["raw_path"]
        require(sha(p)==row["raw_sha256"], "Original witness mutated")
        sources.append(p)
    git = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True)
    methods = {"zero":{"path":str(POOL/"models.npz"),"key":"W_full_base"},
               "public":{"path":str(POOL/"models.npz"),"key":"W_full_public_only"},
               "pooled":{"path":str(POOL/"models.npz"),"key":"W_full_pooled"}}
    for item in methods.values():
        item["sha256"]=sha(item["path"])
    out.mkdir(parents=True)
    save(out/"contract.json", dict(schema="sampling-integration/v1", created_utc=now(), current_step=2,
        source_sha256={str(p):sha(p) for p in sources}, snapshot=str(snap),
        model_id=cap["model_id"],model_revision=cap["model_revision"],
        source_commit=git.stdout.strip() if git.returncode==0 else None,
        source_commit_scope="local code_working checkout if available; file SHA binding is authoritative",
        witnesses=witnesses, methods=methods, cases=cases, trajectory_methods=list(METHODS),
        scheduler_config=dict(sched.config),timesteps=sched.timesteps.tolist(),
        num_inference_steps=30,eta=0.,guidance_scale=1.,height=256,width=256,
        prediction_type="epsilon", residual_location="base epsilon + correction BEFORE DDIM step",
        scale_model_input="called once per step before UNet, DDIM identity checked",
        precision="UNet FP16 artifact promoted FP32; projection FP32; basis/head CPU FP64; correction cast FP32 then add",
        conditioning="existing weak-label cached hidden FP16-origin promoted FP32; no unconditional branch",
        seed_policy="two fixed CPU FP32 initial latents shared across all methods, no search",
        time_basis="original1000-step training alpha schedule and fixed logSNR limits; basis-major order",
        logsnr_min=cap["time_basis_logsnr_min"],logsnr_max=cap["time_basis_logsnr_max"],
        tolerances={"zero_and_restore":"exact", "saved_base_features_atol":2e-6,
                    "saved_base_features_rtol":2e-6,"residual_fp64_atol":1e-12,"residual_fp64_rtol":1e-10,
                    "scheduler_fp32_atol":2e-5,"scheduler_fp32_rtol":2e-6},
        backward=0,optimization_steps=0,final_cohort_evaluation=False,
        maximum_expected_unet_forward=325,
        preview_after_independent_pass="six images: two fixed conditions x base/public/pooled; no efficacy metrics",
        efficacy_or_clinical_claim=False,source_and_verifier_frozen_before_execution=True))
    print(json.dumps({"phase":"prepared","contract_sha256":sha(out/"contract.json"),"timesteps":sched.timesteps.tolist()}),flush=True)


def validate(out):
    c=read(out/"contract.json")
    for path,digest in c["source_sha256"].items():
        require(sha(path)==digest,"Source changed: "+path)
    return c


def state_digest(unet):
    h=hashlib.sha256()
    for name,tensor in unet.state_dict().items():
        data=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(data.dtype).encode());h.update(str(data.shape).encode())
        h.update(memoryview(data).cast("B"))
    return h.hexdigest()


def make_adapter(unet, projection, w, c):
    return FrozenResidualAdapter(unet,projection["P"],w,projection["alphas"],c["logsnr_min"],c["logsnr_max"],c["prediction_type"])


def run(out):
    c=validate(out)
    require(not (out/"execution.json").exists(),"Execution exists")
    setup(); start=time.perf_counter();start_utc=now()
    snap=Path(c["snapshot"])
    cache=torch.load(CACHE,map_location="cpu",weights_only=True)
    projection=dict(np.load(CAP/"projection.npz"))
    weights={name:np.load(item["path"])[item["key"]] for name,item in c["methods"].items()}
    unet=UNet2DConditionModel.from_pretrained(snap/"unet",torch_dtype=torch.float16,variant="fp16",use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).to("cuda")
    require(not any("lora" in n.lower() for n,_ in unet.named_parameters()),"Direct base only")
    before=state_digest(unet)
    count=[0]
    def counter(module,inputs):count[0]+=1
    initial_taps=len(unet.conv_out._forward_pre_hooks)
    sched=DDIMScheduler.from_config(c["scheduler_config"]);sched.set_timesteps(30,device="cuda")
    require(sched.timesteps.tolist()==c["timesteps"],"Timesteps changed")
    require(np.array_equal(projection["alphas"],sched.alphas_cumprod.cpu().numpy()),"Alphas changed")
    save(out/"runtime.json",dict(created_utc=now(),scheduler_class=type(sched).__name__,scheduler_config=dict(sched.config),
        timesteps=sched.timesteps.tolist(),alphas_sha256=hashlib.sha256(projection["alphas"].tobytes()).hexdigest(),
        vae_config=read(snap/"vae/config.json"),GPU=torch.cuda.get_device_name(),
        packages={k:version(k) for k in ("torch","diffusers","transformers","numpy")},
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        dtype=str(next(unet.parameters()).dtype),conditional_only=True,
        initial_conv_out_hooks=initial_taps,backward=0))
    manifest={"witnesses":[],"trajectories":[]}
    (out/"witnesses").mkdir();(out/"trajectories").mkdir()
    counter_handle=unet.register_forward_pre_hook(counter)
    try:
        with torch.inference_mode():
            # One outcome-independent warmup, counted explicitly.
            unet(torch.zeros((1,4,32,32),device="cuda"),torch.tensor([958],device="cuda"),cache["hidden"][c["cases"][0]["prompt"]].float().to("cuda")).sample
            for row in c["witnesses"]:
                z=cache["latents"][row["image_id"]].float()
                noise=torch.randn(z.shape,generator=torch.Generator().manual_seed(row["noise_seed"]),dtype=torch.float32)
                training_sched=DDPMScheduler.from_pretrained(snap/"scheduler",local_files_only=True)
                noisy=training_sched.add_noise(z,noise,torch.tensor([row["timestep"]]))
                xx=noisy.to("cuda"); tt=torch.tensor([row["timestep"]],device="cuda")
                hh=cache["hidden"][row["prompt"]].float().to("cuda")
                stored=np.load(PUB/row["raw_path"])
                require(np.array_equal(array(noise)[0].transpose(1,2,0).reshape(-1,4),stored["target"]),"Noise recovery mismatch")
                data={"clean":array(z),"noise":array(noise),"noisy":array(noisy),"conditioning":array(hh),"timestep":np.int64(row["timestep"])}
                data["plain"]=array(unet(xx,tt,hh).sample)
                for name in ("zero","public","pooled"):
                    with make_adapter(unet,projection,weights[name],c) as adapter:
                        data[name if name=="zero" else name+"_corrected"]=array(adapter.predict(xx,tt,hh))
                        if name!="zero":
                            for key in ("base","features","basis","residual"):
                                data[name+"_"+key]=adapter.last[key]
                        if name=="public":
                            data["repeat_public"]=array(adapter.predict(xx,tt,hh))
                data["restored"]=array(unet(xx,tt,hh).sample)
                require(len(unet.conv_out._forward_pre_hooks)==initial_taps,"Leaked hook")
                for name in ("zero","restored"):
                    require(np.array_equal(data["plain"],data[name]),"Zero/restored not exact")
                require(np.array_equal(data["repeat_public"],data["public_corrected"]),"Repeated call changed")
                np.testing.assert_allclose(data["plain"][0].transpose(1,2,0).reshape(-1,4),stored["base"],rtol=2e-6,atol=2e-6)
                np.testing.assert_allclose(data["public_features"],stored["features"],rtol=2e-6,atol=2e-6)
                path=out/"witnesses"/("%06d.npz"%row["record_id"]);npz(path,data)
                manifest["witnesses"].append(dict(record_id=row["record_id"],path=path.relative_to(out).as_posix(),sha256=sha(path),original_raw_path=str(PUB/row["raw_path"]),original_raw_sha256=row["raw_sha256"],noise_seed=row["noise_seed"]))
            print(json.dumps({"phase":"witnesses_complete","unet_forward":count[0]}),flush=True)
            for case in c["cases"]:
                initial=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(case["seed"]),dtype=torch.float32).to("cuda")
                hh=cache["hidden"][case["prompt"]].float().to("cuda")
                require(hashlib.sha256(array(hh).tobytes()).hexdigest()==case["conditioning_sha256"],"Conditioning changed")
                for method in METHODS:
                    scheduler=DDIMScheduler.from_config(c["scheduler_config"]);scheduler.set_timesteps(30,device="cuda")
                    latent=initial.clone()*scheduler.init_noise_sigma
                    trace={"latents":[array(latent)],"scaled_inputs":[],"base_eps":[],"eps":[]}
                    wrapper=make_adapter(unet,projection,weights[method],c) if method in weights else None
                    if wrapper is not None:
                        trace.update(features=[],basis=[],residual=[])
                    try:
                        if wrapper is not None:wrapper.__enter__()
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                        for timestep in scheduler.timesteps:
                            scaled=scheduler.scale_model_input(latent,timestep)
                            trace["scaled_inputs"].append(array(scaled))
                            if wrapper is None:
                                eps=unet(scaled,timestep,hh).sample;trace["base_eps"].append(array(eps))
                            else:
                                eps=wrapper.predict(scaled,timestep,hh)
                                trace["base_eps"].append(wrapper.last["base"])
                                for key in ("features","basis","residual"):
                                    trace[key].append(wrapper.last[key])
                            trace["eps"].append(array(eps))
                            latent=scheduler.step(eps,timestep,latent,eta=0.).prev_sample
                            require(bool(torch.isfinite(latent).all()),"Nonfinite latent")
                            trace["latents"].append(array(latent))
                        torch.cuda.synchronize();seconds=time.perf_counter()-tick
                        peak=torch.cuda.max_memory_allocated()
                    finally:
                        if wrapper is not None:wrapper.__exit__(None,None,None)
                    require(len(unet.conv_out._forward_pre_hooks)==initial_taps,"Leaked trajectory hook")
                    arrays={key:np.stack(values) for key,values in trace.items()}
                    arrays["timesteps"]=np.array(c["timesteps"],dtype=np.int64)
                    path=out/"trajectories"/(case["case_id"]+"_"+method+".npz");npz(path,arrays)
                    manifest["trajectories"].append(dict(case_id=case["case_id"],method=method,path=path.relative_to(out).as_posix(),sha256=sha(path),seconds=seconds,peak_memory_bytes=peak,prompt=case["prompt"],seed=case["seed"],conditioning_sha256=case["conditioning_sha256"],unet_forward=30))
                    print(json.dumps({"phase":"trajectory_complete","case":case["case_id"],"method":method,"seconds":round(seconds,3),"peak_GiB":round(peak/2**30,3)}),flush=True)
    finally:
        counter_handle.remove()
    require(all(not p.requires_grad and p.grad is None for p in unet.parameters()),"Unexpected gradients")
    after=state_digest(unet);require(before==after,"UNet state mutated")
    require(count[0]==325,"Unexpected UNet count")
    save(out/"manifest.json",manifest)
    save(out/"execution.json",dict(completed=True,started_utc=start_utc,completed_utc=now(),seconds=time.perf_counter()-start,
        contract_sha256=sha(out/"contract.json"),manifest_sha256=sha(out/"manifest.json"),runtime_sha256=sha(out/"runtime.json"),
        model_state_sha256_before=before,model_state_sha256_after=after,forward_calls=count[0],backward=0,
        parameter_gradients_present=False,remaining_conv_out_hooks=len(unet.conv_out._forward_pre_hooks),
        witness_count=4,trajectories=10,generated_preview_images=0,status="COMPLETED_PENDING_INDEPENDENT_VERIFICATION"))
    print("INTEGRATION_EXECUTION_COMPLETE",flush=True)


def preview(out):
    c=validate(out)
    v=read(out/"independent_verification.json")
    require(v["status"].startswith("PASS") and v["complete"],"Independent packet PASS required before preview")
    for name,digest in v["input_sha256"].items():
        require(sha(out/name)==digest,"Verified input changed before preview: "+name)
    execution=read(out/"execution.json")
    require(sha(out/"manifest.json")==execution["manifest_sha256"],"Manifest changed after execution")
    for row in read(out/"manifest.json")["trajectories"]:
        require(sha(out/row["path"])==row["sha256"],"Trajectory changed before preview")
    dest=out/"preview";require(not dest.exists(),"Preview already exists");dest.mkdir()
    setup();snap=Path(c["snapshot"])
    vae=AutoencoderKL.from_pretrained(snap/"vae",torch_dtype=torch.float16,variant="fp16",use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).to("cuda")
    scale=float(vae.config.scaling_factor);require(scale==.18215,"Unexpected cached latent scale")
    from PIL import Image,ImageDraw
    grid=Image.new("RGB",(3*256,2*282),"white");draw=ImageDraw.Draw(grid);records=[]
    with torch.inference_mode():
        for ri,case in enumerate(c["cases"]):
            for ci,method in enumerate(("base","public","pooled")):
                source=out/"trajectories"/(case["case_id"]+"_"+method+".npz")
                latent=torch.from_numpy(np.load(source)["latents"][-1]).to("cuda")
                torch.cuda.synchronize();tick=time.perf_counter()
                decoded=vae.decode(latent/scale).sample
                require(bool(torch.isfinite(decoded).all()),"Nonfinite raw decoded tensor")
                raw_range=[float(decoded.min()),float(decoded.max())]
                pixels=array((decoded/2+.5).clamp(0,1))[0].transpose(1,2,0)
                require(np.isfinite(pixels).all(),"Nonfinite pixels")
                image=Image.fromarray((pixels*255).round().astype(np.uint8));path=dest/(case["case_id"]+"_"+method+".png");image.save(path)
                grid.paste(image,(ci*256,ri*282+26));draw.text((ci*256+5,ri*282+6),case["case_id"]+" / "+method,fill="black")
                records.append(dict(case_id=case["case_id"],method=method,path=path.name,sha256=sha(path),source_trajectory_sha256=sha(source),decode_seconds=time.perf_counter()-tick,unclamped_decoded_range=raw_range))
    grid.save(dest/"comparison_grid.png")
    save(dest/"manifest.json",dict(images=records,vae_scale=scale,vae_dtype="torch.float32",images_count=6,
        grid_sha256=sha(dest/"comparison_grid.png"),independent_packet_sha256=sha(out/"independent_verification.json"),
        guidance_scale=1.,selection="all predeclared two cases x three methods, no seed/image selection",
        quality_metric_measured=False,scope="minimal integration preview, not generated-utility validation"))
    print("SIX_PREDECLARED_PREVIEW_IMAGES_SAVED",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run","preview"));parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args();{"prepare":prepare,"run":run,"preview":preview}[args.phase](args.out)
