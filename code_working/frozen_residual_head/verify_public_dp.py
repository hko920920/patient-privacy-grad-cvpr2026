"""Independent CPU verification of the public32 fixed-feature preparation.

Only independent, frozen verifier algebra is reused; no producer statistics,
fitting, calibration, sampling or target-model routines are imported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np


FROZEN_VERIFIER_SHA256 = "2cc5b9ccd2cb1cc0cb94dd799475a4b034846d5398c33e61c610c8cbd9516573"
FROZEN_VERIFIER_PATH = Path(__file__).with_name("verify_capacity.py")
if hashlib.sha256(FROZEN_VERIFIER_PATH.read_bytes()).hexdigest() != FROZEN_VERIFIER_SHA256:
    raise RuntimeError("Independent frozen capacity-verifier source changed")
from .verify_capacity import (Audit, EPS64, average_moments, independent_basis,
                              independent_statistics, projection_witness)


STATUS = "PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE"
FAMILIES = {"full":"time64", "static":"static16"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_seed(salt: str, split: str, image_id: str, draw: int, label: str) -> int:
    text = "|".join(map(str,(salt,split,image_id,draw,label)))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15],16)


def verify_public_samples(run_dir: Path, audit: Audit, *, projection: np.ndarray,
                          alphas: np.ndarray, images: list[dict], manifest: list[dict],
                          seed_salt: str, witness_record_ids: list[int],
                          stats_path: Path) -> dict:
    """Independently expand every full64 design matrix and average by patient."""
    import torch
    audit.require(not torch.cuda.is_initialized(),"CPU-only verifier before raw checks")
    audit.require(len(images)==64 and len(manifest)==512,"32 patients x2 images x8 draws")
    audit.require(projection.shape==(320,15) and projection.dtype==np.float64,"projection shape/dtype")
    audit.require(alphas.shape==(1000,) and alphas.dtype==np.float32,"scheduler shape/dtype")
    audit.compare(projection.T@projection,np.eye(15),"projection orthonormal",atol=128*EPS64*320)
    declared = [dict(image,draw_id=j) for image in images for j in range(8)]
    image_moments, image_draws = {}, defaultdict(set)
    actual_witnesses = []
    noise_seeds = set()
    started = time.perf_counter()
    for index,row in enumerate(manifest):
        audit.require(row["record_id"]==index,"ordered unique record IDs")
        for field,value in declared[index].items():
            audit.require(row[field]==value,"manifest versus declared image/draw "+field)
        split,pid,iid = str(row["split"]),str(row["patient_id"]),str(row["image_id"])
        audit.require(split=="public","public-only split")
        draw,timestep = int(row["draw_id"]),int(row["timestep"])
        audit.require(0<=draw<8,"draw range")
        expected_t = 125*draw+int(np.random.default_rng(
            fixed_seed(seed_salt,split,iid,draw,"timestep")).integers(0,125))
        noise_seed = fixed_seed(seed_salt,split,iid,draw,"noise")
        audit.require(timestep==expected_t and row["noise_seed"]==noise_seed,"predeclared stratified time/noise")
        audit.require(noise_seed not in noise_seeds,"distinct declared noise stream")
        noise_seeds.add(noise_seed)
        key=(pid,iid)
        audit.require(draw not in image_draws[key],"unique draw per image")
        image_draws[key].add(draw)
        path=(run_dir/row["raw_path"]).resolve()
        audit.require(path.is_relative_to(run_dir.resolve()),"raw path within public run")
        audit.require(digest(path)==row["raw_sha256"],"raw hash")
        with np.load(path,allow_pickle=False) as raw:
            h,base,target,basis=(raw[k].copy() for k in ("features","base","target","basis"))
            audit.require(h.shape==(1024,16) and h.dtype==np.float32,"h16 shape/dtype")
            audit.require(base.shape==target.shape==(1024,4) and base.dtype==target.dtype==np.float32,
                          "base/epsilon shape/dtype")
            audit.require(basis.shape==(4,) and basis.dtype==np.float64,"basis shape/dtype")
            audit.require(int(raw["timestep"])==timestep,"raw timestep")
            audit.require(bool(np.isfinite(base).all() and np.isfinite(target).all()),"finite base/epsilon")
            audit.compare(basis,independent_basis(timestep,alphas),"fixed Legendre basis",atol=1e-13)
            replay=torch.randn((1,4,32,32),dtype=torch.float32,
                generator=torch.Generator().manual_seed(noise_seed)).numpy()[0].transpose(1,2,0).reshape(1024,4)
            audit.require(np.array_equal(replay,target),"exact CPU diffusion-noise replay")
            audit.require(("hidden" in raw.files)==(index in witness_record_ids),"predeclared witness presence")
            if "hidden" in raw.files:
                actual_witnesses.append({"record_id":index,
                    **projection_witness(audit,raw["hidden"],h,projection,"public projection witness")})
        residual=target.astype(np.float64)-base.astype(np.float64)
        stats=independent_statistics(h,residual,basis)
        if key not in image_moments:
            image_moments[key]={name:{"A":np.zeros_like(s["A"]),"B":np.zeros_like(s["B"]),"Q":0.0}
                                for name,s in stats.items()}
        for name,s in stats.items():
            for field in ("A","B","Q"):
                image_moments[key][name][field]+=s[field]/8
        if (index+1)%128==0:
            print(json.dumps({"verified_public_samples":index+1,"total":512,
                              "seconds":time.perf_counter()-started}),flush=True)
    audit.require(len(actual_witnesses)==4,"four projection witnesses")
    by_patient=defaultdict(list)
    for key,draws in image_draws.items():
        audit.require(draws==set(range(8)),"eight draws in every image")
        by_patient[key[0]].append(key)
    audit.require(len(by_patient)==32,"32 public patients")
    moments={}
    for pid,keys in by_patient.items():
        audit.require(len(keys)==2,"two equal-weight images in every public patient")
        moments[pid]={name:average_moments([image_moments[key][name] for key in keys])
                      for name in FAMILIES.values()}
    with np.load(stats_path,allow_pickle=False) as saved:
        ids=[str(x) for x in saved["patient_ids"]]
        splits=[str(x) for x in saved["splits"]]
        audit.require(len(ids)==32 and len(set(ids))==32 and set(ids)==set(moments),"saved public patient IDs")
        audit.require(splits==["public"]*32,"saved public split labels")
        for index,pid in enumerate(ids):
            for family,internal in FAMILIES.items():
                for field in ("A","B","Q"):
                    expected=moments[pid][internal][field]
                    scale=max(1.0,float(np.max(np.abs(expected))))
                    audit.compare(saved[field+"_"+family][index],expected,
                                  "public patient "+pid+" "+family+" "+field,
                                  atol=256*EPS64*(1024+16)*scale)
    audit.require(not torch.cuda.is_initialized(),"CPU-only verifier after raw checks")
    return {"samples":512,"patients":32,"images":64,"projection_witnesses":actual_witnesses,
            "patient_ids":ids,"equal_patient_base_mse":math.fsum(moments[pid]["time64"]["Q"]/4 for pid in ids)/32,
            "statistics_definition":"float64 target-base; spatial channel-sum quadratic; draws/images/patients equally weighted"}


def verify_provenance(run_dir: Path,audit: Audit) -> dict:
    c=read_json(run_dir/"contract.json")
    audit.require(c["schema"]=="frozen-residual-public32/v1","public contract schema")
    audit.require(c["expected_records"]==512 and c["images_per_patient"]==2 and c["draws_per_image"]==8,
                  "fixed public record policy")
    audit.require(c["expected_patients"]=={"public":32} and c["expected_images"]=={"public":64},"public cohort size")
    audit.require(c["private_fitting"] is False and c["dp_applied"] is False,"public extraction is not private fitting")
    audit.require(c["public_data_fixed_before_private_comparison"] is True,"public calibration data policy")
    audit.require(c["public_calibration_loses_independent_quality_test_status"] is True,"quality set repurposing disclosed")
    audit.require(c["all_evaluation_csv_patients_excluded"] is True,"patient exclusion policy")
    audit.require(c["tf32"] is False and c["deterministic_algorithms"] is True,"FP32 deterministic policy")
    audit.require(c["prediction_type"]=="epsilon" and c["latent_shape"]==[1,4,32,32],"epsilon/latent contract")
    audit.require(c["seed_salt"]=="frozen-residual-capacity-20260916-v1","original seed salt with new public split")
    audit.require(c["projection_seed"]==26091601,"same original public projection seed")
    sources=c["source_sha256"]
    audit.require(len(sources)>=16,"frozen source/input bindings")
    for name,expected in sources.items():
        audit.require(digest(Path(name))==expected,"frozen source "+name)
    def source(suffix):
        matches=[Path(name) for name in sources if name.replace("\\","/").endswith(suffix)]
        audit.require(len(matches)==1,"one bound "+suffix)
        return matches[0]
    for suffix in ("frozen_residual_head/run_public.py","frozen_residual_head/run_capacity.py",
                   "frozen_residual_head/residual_math.py","u_patient_audit/models.py",
                   "u_patient_audit/common.py","u_patient_audit/build_cache.py",
                   "data_pipeline/nih_cxr14_model_input.py","cohort/lock.json",
                   "cohort/evaluation_images.csv","cache/cache.pt","cache/summary.json",
                   "unet/config.json","scheduler/scheduler_config.json",
                   "unet/diffusion_pytorch_model.fp16.safetensors"):
        source(suffix)
    auxiliary=source("cohort/auxiliary_images.csv")
    expected_auxiliary="4ae1c513099d28b142eef6d9d805eadb529fb05b9895301497393d93d9b14fe2"
    audit.require(digest(auxiliary)==expected_auxiliary,"original fixed quality auxiliary CSV")
    locked=read_json(source("cohort/lock.json"))
    for filename in ("auxiliary_images.csv","evaluation_images.csv"):
        audit.require(digest(source("cohort/"+filename))==locked["files"][filename],"original cohort-lock "+filename)
    with auxiliary.open(encoding="utf-8-sig",newline="") as handle:
        rows=[r for r in csv.DictReader(handle) if r["eval_role"]=="quality"]
    rows.sort(key=lambda r:(int(r["patient_id"]),r["image_id"]))
    with source("cohort/evaluation_images.csv").open(encoding="utf-8-sig",newline="") as handle:
        evaluation=list(csv.DictReader(handle))
    excluded={r["patient_id"] for r in evaluation}
    audit.require(len(excluded)==400,"all original 400 evaluation-cohort patients")
    images=read_json(run_dir/"images.json")
    audit.require(len(rows)==len(images)==64,"64 publicly designated source images")
    ids={r["patient_id"] for r in images}
    audit.require(len(ids)==32 and not ids.intersection(excluded),"public32 disjoint from all private/development/final patients")
    audit.require(Path(c["exclude_evaluation_manifest"]).resolve()==source("cohort/evaluation_images.csv").resolve(),
                  "contract exclusion source identity")
    audit.require(digest(run_dir/"images.json")==c["images_sha256"],"predeclared images hash")
    audit.require(digest(run_dir/"projection.npz")==c["projection_sha256"],"predeclared projection hash")
    import torch
    audit.require(not torch.cuda.is_initialized(),"no CUDA initialized by provenance verification")
    cache_summary=read_json(source("cache/summary.json"))
    audit.require(digest(source("cache/cache.pt"))==cache_summary["cache_sha256"],"original cache summary binding")
    audit.require(digest(source("cohort/lock.json"))==cache_summary["cohort_lock_sha256"],"original cached cohort binding")
    audit.require(digest(source("unet/diffusion_pytorch_model.fp16.safetensors"))==
        cache_summary["model_hashes"]["unet/diffusion_pytorch_model.fp16.safetensors"]["actual"].lower(),
        "original cached pretrained artifact")
    cache=torch.load(source("cache/cache.pt"),map_location="cpu",weights_only=True)
    for image,original in zip(images,rows):
        for field in ("patient_id","image_id","sha256"):
            audit.require(image[field]==original[field],"declared original public "+field)
        audit.require(image["split"]=="public" and image["original_eval_role"]=="quality","public role mapping")
        audit.require(image["original_record_role"]==original["record_role"],"historical record role only")
        audit.require(digest(Path(image["path"]))==image["sha256"],"original image bytes")
        prompt=cache["train_prompts"][image["image_id"]]
        audit.require(image["prompt"]==prompt,"cached per-image weak-label prompt")
        audit.require(tuple(cache["latents"][image["image_id"]].shape)==(1,4,32,32),"existing cached latent")
        audit.require(tuple(cache["hidden"][prompt].shape)==(1,77,1024),"existing cached text conditioning")
    del cache
    parent=Path(c["parent_capacity_directory"]).resolve()
    parent_contract=read_json(parent/"contract.json")
    audit.require(str(parent/"contract.json") in sources,"bound parent capacity contract")
    audit.require(digest(parent/"projection.npz")==digest(run_dir/"projection.npz"),"byte-identical original capacity projection")
    for field in ("projection_seed","time_basis_logsnr_min","time_basis_logsnr_max","seed_salt", "ridge_lambda"):
        audit.require(c[field]==parent_contract[field],"same private/public fixed definition "+field)
    prior=read_json(parent/"independent_verification.json")
    audit.require(prior["status"]=="PASS_FROZEN_RESIDUAL_CAPACITY_SAVED_ARITHMETIC_AND_PROVENANCE" and prior["complete"] is True,
                  "prior capacity arithmetic verification completed")
    for name,expected in prior["input_sha256"].items():
        audit.require(digest(parent/name)==expected,"prior capacity input remains unchanged "+name)
    with np.load(run_dir/"projection.npz",allow_pickle=False) as p:
        projection,alphas=p["P"].copy(),p["alphas"].copy()
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler.from_config(read_json(source("scheduler/scheduler_config.json")))
    audit.require(scheduler.config.prediction_type=="epsilon","actual epsilon scheduler")
    audit.require(np.array_equal(alphas,scheduler.alphas_cumprod.cpu().numpy()),"exact original scheduler alphas")
    logsnr=np.log(alphas.astype(np.float64))-np.log1p(-alphas.astype(np.float64))
    audit.compare(c["time_basis_logsnr_min"],logsnr.min(),"public minimum logSNR",atol=1e-13)
    audit.compare(c["time_basis_logsnr_max"],logsnr.max(),"public maximum logSNR",atol=1e-13)
    audit.require(c["witness_image_ids"]==[images[0]["image_id"],images[2]["image_id"]],"first images of first two public patients witness policy")
    witness_records=[index for index,row in enumerate([dict(r,draw_id=d) for r in images for d in range(8)])
                     if row["image_id"] in c["witness_image_ids"] and row["draw_id"] in (0,7)]
    audit.require(witness_records==[0,7,16,23],"four predeclared witness record indices")
    execution=read_json(run_dir/"execution.json")
    audit.require(execution["completed"] is True and execution["records"]==512,"completed public extraction")
    audit.require(execution["unet_forward"]==513 and execution["warmup_forward"]==1 and execution["unet_backward"]==0,
                  "512 public forward records plus one warmup, zero backward")
    audit.require(execution["new_vae_or_text_encoder_forward"]==0,"existing latent/text cache reuse")
    audit.require(execution["no_backbone_parameter_gradients"] is True and execution["tf32"] is False,"producer inference guards")
    audit.require(execution["contract_sha256"]==digest(run_dir/"contract.json"),"execution/contract binding")
    audit.require(execution["manifest_sha256"]==digest(run_dir/"manifest.json"),"execution/raw-manifest binding")
    analysis=read_json(run_dir/"analysis.json")
    audit.require(analysis["completed"] is True and analysis["private_fitting"] is False,"public statistic analysis scope")
    audit.require((analysis["patients"],analysis["images"],analysis["records"])==(32,64,512),"public analysis counts")
    audit.require(set(analysis["source_sha256"])=={"contract.json","manifest.json","execution.json","patient_stats.npz"},
                  "complete statistic input/output binding")
    for name,expected in analysis["source_sha256"].items():
        audit.require(digest(run_dir/name)==expected,"public analysis source "+name)
    return {"contract":c,"projection":projection,"alphas":alphas,"images":images,
            "manifest":read_json(run_dir/"manifest.json"),"witness_records":witness_records,
            "input_sha256":{name:digest(run_dir/name) for name in
                ("contract.json","images.json","projection.npz","manifest.json","execution.json","analysis.json","patient_stats.npz")}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        from .verify_capacity import self_test
        print(json.dumps({"frozen_independent_helper":FROZEN_VERIFIER_SHA256,**self_test()}),flush=True)
        return
    if args.run_dir is None:
        parser.error("--run-dir or --self-test required")
    run_dir=args.run_dir.resolve()
    destination=run_dir/"independent_verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    started,audit=time.perf_counter(),Audit()
    result={"schema":"independent-public32-residual-verification/v1",
            "verifier_source_sha256":digest(Path(__file__)),
            "frozen_independent_math_source_sha256":FROZEN_VERIFIER_SHA256,
            "producer_numerical_imports":False,"GPU_forward":0,"GPU_backward":0,
            "timing_scope":"Verifier source and inherited numerical tolerances finalized before this verification; GPU-before-freeze status must use the separately timestamped source record, not this result."}
    try:
        inputs=verify_provenance(run_dir,audit)
        result["input_sha256"]=inputs["input_sha256"]
        result["verification"]=verify_public_samples(run_dir,audit,
            projection=inputs["projection"],alphas=inputs["alphas"],images=inputs["images"],manifest=inputs["manifest"],
            seed_salt=inputs["contract"]["seed_salt"],witness_record_ids=inputs["witness_records"],
            stats_path=run_dir/"patient_stats.npz")
        result.update(status=STATUS,complete=True)
    except Exception as error:
        result.update(status="FAIL_PUBLIC32_SAVED_VERIFICATION",complete=False,
                      failure=repr(error),traceback=traceback.format_exc())
    result.update(seconds=time.perf_counter()-started,checks=audit.count,comparisons=audit.comparisons)
    result["limits"]=[
        "No UNet/encoder forward was independently repeated; saved arithmetic and bound source/input provenance are checked.",
        "Exactly four saved 320-channel projection witnesses are checked; all 512 saved 16-channel designs are checked.",
        "The former quality subset is now public calibration data and is not an untouched quality test.",
        "This does not check or execute private clipping/noising, optimizer calibration or a DP mechanism.",
        "Pixels and diffusion-noise draws are not counted as independent patients."]
    with destination.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({k:result[k] for k in ("status","complete","seconds","checks")}),flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
