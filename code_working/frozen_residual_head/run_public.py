"""Public-only auxiliary feature extraction for the fixed-head DP comparison.
Frozen capacity source/data remain unchanged. No private fitting in this runner.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from . import run_capacity as cap
from . import residual_math as rm
from u_patient_audit.common import ROOT, RUN, IMAGES, digest, read_csv
from u_patient_audit.models import load_cache, scheduler, diffusion_input

DEFAULT_OUT = ROOT / "_reports/frozen_residual_public32_20260916_v1"
CAPACITY = ROOT / "_reports/frozen_residual_capacity_20260916_v1"
save_json, read_json, save_npz = cap.save_json, cap.read_json, cap.save_npz

def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    c = cap.validate(CAPACITY)
    cache = load_cache()
    auxiliary = RUN / "cohort/auxiliary_images.csv"
    assert digest(auxiliary) == "4ae1c513099d28b142eef6d9d805eadb529fb05b9895301497393d93d9b14fe2"
    rows = sorted([r for r in read_csv(auxiliary) if r["eval_role"] == "quality"],
                  key=lambda r: (int(r["patient_id"]), r["image_id"]))
    assert len(rows) == 64
    assert Counter(Counter(r["patient_id"] for r in rows).values()) == {2: 32}
    excluded = {r["patient_id"] for r in read_csv(RUN / "cohort/evaluation_images.csv")}
    assert not {r["patient_id"] for r in rows} & excluded
    images = []
    for r in rows:
        iid, path = r["image_id"], IMAGES / r["image_filename"]
        assert digest(path) == r["sha256"]
        prompt = cache["train_prompts"][iid]
        assert tuple(cache["latents"][iid].shape) == (1, 4, 32, 32)
        assert tuple(cache["hidden"][prompt].shape) == (1, 77, 1024)
        images.append(dict(split="public", patient_id=r["patient_id"], image_id=iid,
            original_eval_role="quality", original_record_role=r["record_role"],
            path=str(path), sha256=r["sha256"], prompt=prompt))
    out.mkdir(parents=True)
    shutil.copyfile(CAPACITY / "projection.npz", out / "projection.npz")
    save_json(out / "images.json", images)
    bindings = dict(c["source_sha256"])
    for p in [Path(__file__), auxiliary, CAPACITY / "contract.json"]:
        bindings[str(p)] = digest(p)
    c.update(schema="frozen-residual-public32/v1", created_utc=cap.stamp(),
        scope="fixed independent public calibration only; no private model fitting",
        parent_capacity_directory=str(CAPACITY), expected_records=512,
        expected_patients={"public":32}, expected_images={"public":64},
        images_per_patient=2, source_sha256=bindings,
        images_sha256=digest(out / "images.json"), projection_sha256=digest(out / "projection.npz"),
        witness_image_ids=[images[0]["image_id"],images[2]["image_id"]],
        objective="equal public patients, equal two images, equal eight draws; channel SUM",
        public_data_fixed_before_private_comparison=True, private_fitting=False,
        public_calibration_loses_independent_quality_test_status=True,
        all_evaluation_csv_patients_excluded=True,
        exclude_evaluation_manifest=str(RUN / "cohort/evaluation_images.csv"))
    save_json(out / "contract.json", c)
    print("PREPARED_PUBLIC32_64_IMAGES_512_RECORDS", flush=True)

def validate(out):
    c = read_json(out / "contract.json")
    assert c["schema"] == "frozen-residual-public32/v1"
    assert c["expected_records"] == 512 and c["images_per_patient"] == 2
    for name, expected in c["source_sha256"].items():
        assert digest(name) == expected, name
    for f,key in [("images.json","images_sha256"),("projection.npz","projection_sha256")]:
        assert digest(out/f) == c[key]
    return c

def extract(out):
    c = validate(out)
    if (out/"execution.json").exists():
        raise FileExistsError(out/"execution.json")
    started = time.perf_counter()
    images, cache, sched = read_json(out/"images.json"), load_cache(), scheduler()
    plan = cap.sample_plan(c, images)
    rawdir = out/"raw"
    rawdir.mkdir(exist_ok=False)
    projection = np.load(out/"projection.npz")
    unet = cap.new_model(c)
    p = torch.from_numpy(projection["P"].astype(np.float32)).to("cuda")
    captured = {}
    def tap(module, inputs):
        captured["hidden"] = inputs[0].detach()
    hook = unet.conv_out.register_forward_pre_hook(tap)
    timings, manifest = [], []
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for index,row in enumerate(plan):
                z = cache["latents"][row["image_id"]].float()
                eps = torch.randn(z.shape,generator=torch.Generator().manual_seed(row["noise_seed"]),dtype=torch.float32)
                noisy,target = diffusion_input(z,torch.tensor([row["timestep"]]),eps,sched)
                xx,tt = noisy.to("cuda"),torch.tensor([row["timestep"]],device="cuda")
                hh = cache["hidden"][row["prompt"]].float().to("cuda")
                if index == 0:
                    unet(xx,tt,hh).sample
                    torch.cuda.synchronize()
                tick=time.perf_counter()
                pred=unet(xx,tt,hh).sample
                h=captured.pop("hidden")[0].permute(1,2,0).reshape(-1,320)
                f=torch.cat((torch.ones((len(h),1),device=h.device),h@p),dim=1)
                arrays=dict(features=f.cpu().numpy().copy(),
                    base=pred[0].permute(1,2,0).reshape(-1,4).cpu().numpy().copy(),
                    target=target[0].permute(1,2,0).reshape(-1,4).cpu().numpy().copy(),
                    basis=rm.legendre_time_basis(np.int64(row["timestep"]),projection["alphas"],
                        logsnr_min=c["time_basis_logsnr_min"],logsnr_max=c["time_basis_logsnr_max"]),
                    timestep=np.array(row["timestep"],dtype=np.int64))
                if row["image_id"] in c["witness_image_ids"] and row["draw_id"] in (0,7):
                    arrays["hidden"]=h.cpu().numpy().copy()
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-tick)
                assert all(np.isfinite(v).all() for v in arrays.values())
                path=rawdir/("%06d.npz"%row["record_id"])
                save_npz(path,**arrays)
                manifest.append({**row,"raw_path":path.relative_to(out).as_posix(),"raw_sha256":digest(path)})
                if index%128==0 or index+1==len(plan):
                    print({"phase":"public_extract","completed":index+1,"total":len(plan),
                        "seconds":round(time.perf_counter()-started,3)},flush=True)
    finally:
        hook.remove()
    assert len(manifest)==512 and all(not p.requires_grad and p.grad is None for p in unet.parameters())
    save_json(out/"manifest.json",manifest)
    result=dict(completed=True,utc=cap.stamp(),records=512,unet_forward=513,unet_backward=0,
        warmup_forward=1,new_vae_or_text_encoder_forward=0,
        model_load_and_output_seconds=time.perf_counter()-started,
        mean_record_seconds=float(np.mean(timings)),peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        contract_sha256=digest(out/"contract.json"),manifest_sha256=digest(out/"manifest.json"),
        no_backbone_parameter_gradients=True,tf32=False,
        note="Public auxiliary extraction only. Cache/model/source validation time precedes runner timer.")
    save_json(out/"execution.json",result)
    print(result,flush=True)

def analyze(out):
    c=validate(out)
    started=time.perf_counter()
    if (out/"patient_stats.npz").exists():
        raise FileExistsError(out/"patient_stats.npz")
    manifest=read_json(out/"manifest.json")
    assert digest(out/"manifest.json")==read_json(out/"execution.json")["manifest_sha256"]
    patients=defaultdict(lambda:defaultdict(list))
    for row in manifest:
        path=out/row["raw_path"]
        assert digest(path)==row["raw_sha256"]
        with np.load(path) as raw:
            x=raw["features"].astype(np.float64)
            residual=raw["target"].astype(np.float64)-raw["base"].astype(np.float64)
            b=raw["basis"].astype(np.float64)
        st=rm.sufficient_statistics(x,residual)
        full=dict(A=np.kron(np.outer(b,b),st["A"]),B=np.kron(b[:,None],st["B"]),Q=st["Q"])
        patients[row["patient_id"]][row["image_id"]].append({"static":st,"full":full})
    ids=sorted(patients,key=int)
    assert len(ids)==32
    archive=dict(patient_ids=np.array(ids),splits=np.array(["public"]*32))
    for name in ("static","full"):
        vals=[]
        for pid in ids:
            assert len(patients[pid])==2
            image_stats=[]
            for iid,draws in sorted(patients[pid].items()):
                assert len(draws)==8
                image_stats.append(rm.mean_statistics([r[name] for r in draws]))
            vals.append(rm.mean_statistics(image_stats))
        for key in ("A","B","Q"):
            archive[key+"_"+name]=np.stack([v[key] for v in vals])
    save_npz(out/"patient_stats.npz",**archive)
    result=dict(completed=True,seconds=time.perf_counter()-started,patients=32,images=64,records=512,
        source_sha256={f:digest(out/f) for f in ("contract.json","manifest.json","execution.json","patient_stats.npz")},
        private_fitting=False,utc=cap.stamp())
    save_json(out/"analysis.json",result)
    print(result,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=["prepare","extract","analyze"])
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    {"prepare":prepare,"extract":extract,"analyze":analyze}[args.phase](args.run_dir.resolve())
if __name__=="__main__":
    main()

