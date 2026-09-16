"""Immutable PFAMI-style measurement under an external, fixed contract.

Only this entry point executes the GPU. Dry-run is CPU-only. Benchmark uses the
specified original fit patient; the full measurement uses selection patients.
"""
import argparse
import gc
import json
import math
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

from .common import ROOT, RUN, IMAGES, PROMPT, digest, read_csv, verify_inputs, write_json
from .models import setup, snapshot, load_cache, load_unet, scheduler, adapter_state
from .pfami_adapter import PFAMIAdapter, source_bindings, cpu_conformance, tensor_digest
from .run_secmi_diagnostic import selected_data, OLD


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_contract(c):
    expected = dict(timesteps=list(range(0,500,50)), crop_size=207,
        crop_interpolation="bilinear", crop_antialias=True,
        noise_policy="independent_view_t_shared_models", target_noise_stream="target",
        base_noise_stream="base", original_latent_policy="verified_fp16_cache",
        crop_vae_policy="fp16_variant_mode_batch4", prediction_type="epsilon",
        generic_prompt=PROMPT, include_base=True)
    for key, value in expected.items():
        assert c[key] == value, "unsupported contract: " + key
    assert c["crop_strength"] == 73/90 and c["models"] == ["model_1","model_2"]
    assert c["roles"] == ["selection"] and c["scenarios"] == ["E","U"]


def planned_rows(contract, benchmark_patient=None):
    fit, selection, all_rows = selected_data()
    assert contract["patient_order_by_role"]["selection"] == selection
    if benchmark_patient is not None:
        assert benchmark_patient == contract["benchmark_patient"] == fit[0]
        rows = [r for r in all_rows if r["patient_id"] == benchmark_patient]
        assert len(rows) == 4 and all(r["eval_role"] == "fit" for r in rows)
    else:
        rows = [r for r in all_rows if r["eval_role"] in contract["roles"]]
        assert len(rows) == contract["expected"]["unique_images"]
    return rows


def tensor_difference(a, b):
    delta = a.double()-b.double()
    denom = float(b.double().norm())
    return dict(exact_equal=bool(torch.equal(a,b)), max_abs=float(delta.abs().max()),
                mean_abs=float(delta.abs().mean()), l2=float(delta.norm()),
                relative_l2=float(delta.norm())/denom if denom else None,
                differing_values=int(torch.count_nonzero(delta)), total_values=a.numel())


def encode_crops(rows, cache, output, benchmark):
    from data_pipeline import nih_cxr14_model_input as mi
    from torchvision.transforms import functional as TF, InterpolationMode
    from diffusers import AutoencoderKL
    snap = snapshot()
    vae = AutoencoderKL.from_pretrained(snap/"vae", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True, local_files_only=True).eval().to("cuda")
    vae.requires_grad_(False)
    versions = {n:p._version for n,p in vae.named_parameters()}
    latents, records, replication = {}, [], []
    calls, examples, cuda_ms = 0, 0, 0.0
    began = time.perf_counter()
    for offset in range(0,len(rows),4):
        batch_rows = rows[offset:offset+4]
        assert len(batch_rows) == 4
        originals, crops = [], []
        for row in batch_rows:
            path = IMAGES/row["image_filename"]
            assert digest(path) == row["sha256"], "source image changed"
            image, _ = mi.preprocess_path(path,256)
            original = mi.pil_to_normalized_tensor(image)
            unit = TF.to_tensor(image)
            center = TF.center_crop(unit,[207,207])
            assert torch.equal(center,unit[:,24:231,24:231])
            crop = TF.resize(center,[256,256],interpolation=InterpolationMode.BILINEAR,antialias=True)
            crop = TF.normalize(crop,[.5],[.5])
            assert crop.shape == original.shape == (3,256,256) and bool(torch.isfinite(crop).all())
            originals.append(original); crops.append(crop)
            records.append(dict(image_id=row["image_id"], image_filename=row["image_filename"],
                source_sha256=row["sha256"], original_pixel_tensor_sha256=tensor_digest(original),
                crop_pixel_tensor_sha256=tensor_digest(crop), crop_offsets=[24,24,231,231],
                original_latent_sha256=tensor_digest(cache["latents"][row["image_id"]])))
        views = [("crop",crops)] + ([("original_replication",originals)] if benchmark else [])
        for view, tensors in views:
            values = torch.stack(tensors).to("cuda",torch.float16)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode(), torch.autocast("cuda",enabled=False):
                encoded = vae.encode(values).latent_dist.mode()*float(vae.config.scaling_factor)
            end.record(); end.synchronize(); cuda_ms += float(start.elapsed_time(end))
            assert encoded.dtype == torch.float16 and encoded.shape == (4,4,32,32)
            assert bool(torch.isfinite(encoded).all()), "nonfinite FP16 VAE output"
            calls += 1; examples += 4
            for row,value in zip(batch_rows,encoded.cpu()):
                tensor = value.unsqueeze(0).contiguous()
                if view == "crop":
                    latents[row["image_id"]] = tensor
                else:
                    old = cache["latents"][row["image_id"]]
                    replication.append(dict(image_id=row["image_id"],
                        original_cache_sha256=tensor_digest(old), reencoded_sha256=tensor_digest(tensor),
                        **tensor_difference(tensor,old)))
            del values,encoded
    assert all(p._version == versions[n] and p.grad is None and not p.requires_grad
               for n,p in vae.named_parameters())
    vae_file = snap/"vae/diffusion_pytorch_model.fp16.safetensors"
    assert vae_file.exists()
    for row in records:
        row["crop_latent_sha256"] = tensor_digest(latents[row["image_id"]])
    torch.save(dict(crop_latents=latents),output/"crop_latents.pt")
    write_json(output/"encoding_records.json",records)
    report = dict(policy="fp16_variant_mode_batch4", dtype="float16", batch_size=4,
        crop_images=len(rows), original_replication_images=len(replication), forward_batch_calls=calls,
        forward_model_examples=examples, backward=0, cuda_ms=cuda_ms,
        elapsed_seconds=time.perf_counter()-began, weights_unchanged=True,
        vae_weights_sha256=digest(vae_file), vae_config_sha256=digest(snap/"vae/config.json"),
        scaling_factor=float(vae.config.scaling_factor), original_replication=replication,
        crop_latents_sha256=digest(output/"crop_latents.pt"),
        encoding_records_sha256=digest(output/"encoding_records.json"))
    write_json(output/"encoding_report.json",report)
    del vae; gc.collect();torch.cuda.empty_cache()
    return latents, report


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run",action="store_true")
    modes.add_argument("--benchmark-patient")
    modes.add_argument("--full-run",action="store_true")
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--output-tag",required=True)
    parser.add_argument("--benchmark-report",type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",args.output_tag)
    contract = load(args.contract);validate_contract(contract);verify_inputs()
    rows = planned_rows(contract,args.benchmark_patient)
    output = RUN/"baseline_screen_20260914"/args.output_tag
    if output.exists():
        raise FileExistsError("immutable output directory already exists")
    if args.full_run:
        assert args.benchmark_report is not None, "full-run requires completed fit-only benchmark report"
        prior = load(args.benchmark_report)
        assert prior["mode"] == "benchmark" and prior["contract_sha256"] == digest(args.contract)
        assert prior["status"] == "COMPLETED_PFAMI_EXECUTION_REQUIRES_INDEPENDENT_ANALYSIS"
    checkpoints = {m:RUN/"training_coverage_v2"/m/"step_1000.pt" for m in contract["models"]}
    checkpoint_hashes = {m:digest(p) for m,p in checkpoints.items()}
    for old in load(OLD/"results.json"):
        assert checkpoint_hashes[old["model"]] == old["checkpoint_sha256"]
    protected = load(RUN/"verification_20260914/foundation/report.json")["protected_sha256"]
    for name,value in protected.items():
        assert digest(ROOT/name) == value
    code_names = ["pfami_adapter.py","run_pfami_diagnostic.py","pfami_analysis.py",
        "verify_pfami_diagnostic.py","run_secmi_diagnostic.py","common.py","models.py","build_cache.py"]
    code_hashes = {name:digest(Path(__file__).with_name(name)) for name in code_names}
    inputs = [args.contract,OLD/"results.json",OLD/"verification.json",RUN/"cohort/lock.json",
        RUN/"cohort/evaluation_images.csv",RUN/"cache/summary.json",RUN/"training_coverage_v2/protocol.json"]
    inputs += [RUN/"cohort"/(m+"_train.csv") for m in contract["models"]]
    if args.benchmark_report is not None:
        inputs.append(args.benchmark_report)
    mode = "dry_run" if args.dry_run else "benchmark" if args.benchmark_patient else "full"
    protocol = dict(schema="u-pfami-style-execution/v1",contract=contract,
        contract_sha256=digest(args.contract),mode=mode,benchmark_patient=args.benchmark_patient,
        measurement_patient_ids=list(dict.fromkeys(r["patient_id"] for r in rows)),
        image_order=[r["image_id"] for r in rows],models=contract["models"]+["base"],
        expected_image_records=len(rows)*3,expected_forward=len(rows)*60,expected_backward=0,
        source=source_bindings(),cpu_conformance=cpu_conformance(),code_sha256=code_hashes,
        checkpoint_sha256=checkpoint_hashes,cache_sha256=load(RUN/"cache/summary.json")["cache_sha256"],
        input_sha256={str(p.resolve()):digest(p) for p in inputs},original_protected_sha256=protected,
        residual_axes=["timestep","view(original,crop)","channel","height","width"],
        residual_shape=[10,2,4,32,32],residual_dtype="float32",
        environment=dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
            packages={n:version(n) for n in ("numpy","diffusers","transformers","peft","torchvision")}),
        noise_generation="CPU torch.Generator seeded by SHA256; float32 randn then transfer to UNet device",
        scoring="source mean relative fluctuation; positive=member; no fit or threshold")
    output.mkdir(parents=True,exist_ok=False);write_json(output/"protocol.json",protocol)
    if args.dry_run:
        report = dict(status="PASS_CPU_PREFLIGHT_NO_GPU",mode=mode,contract_sha256=digest(args.contract),
            protocol_sha256=digest(output/"protocol.json"),planned_images=len(rows),
            planned_records=len(rows)*3,planned_forward=len(rows)*60,GPU_executions=0)
        write_json(output/"report.json",report);print(json.dumps(report),flush=True);return
    setup();cache=load_cache();began=time.perf_counter()
    assert cache["audit_prompt"] == contract["generic_prompt"]
    crops,encoding = encode_crops(rows,cache,output,bool(args.benchmark_patient))
    images,model_reports = [],[]
    for model in protocol["models"]:
        model_started=time.perf_counter()
        if model == "base":
            exposure,train = None,None
        else:
            state=torch.load(checkpoints[model],map_location="cpu",weights_only=True)
            exposure=state["exposures"]
            assert state["step"] == 1000 and state["contract"] == load(RUN/"training_coverage_v2/protocol.json")
            del state
            train={r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
            assert set(exposure)==train and min(exposure.values())>=1 and sum(exposure.values())==4000
        unet=load_unet(None if model=="base" else checkpoints[model],training=False)
        before=adapter_state(unet)
        if model == "base":
            bs={n:v for n,v in before.items() if ".lora_B." in n}
            assert bs and all(torch.count_nonzero(v)==0 for v in bs.values()), "base adapter must be zero-effect"
        probe=PFAMIAdapter(unet,scheduler(),cache["hidden"][PROMPT],contract["timesteps"],contract["noise_seed"])
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            member = None if model=="base" else int(row["assignment_group"]==("A" if model=="model_1" else "B"))
            image_member = None if member is None else int(row["scenario"]=="E" and member)
            if model != "base":
                assert int(exposure.get(row["image_id"],0)>0)==image_member
                assert int(row["image_id"] in train)==image_member
            measured=probe.score(cache["latents"][row["image_id"]],crops[row["image_id"]],row["image_id"],
                                 stream="base" if model=="base" else "target")
            assert measured.pop("image_id") == row["image_id"]
            residual_path=Path("residuals")/model/(row["image_id"]+".pt")
            (output/residual_path).parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(residuals=measured.pop("residuals"),image_id=row["image_id"],model=model,
                timesteps=contract["timesteps"],view_order=["original","crop"]),output/residual_path)
            record=dict(model=model,scenario=row["scenario"],eval_role=row["eval_role"],
                patient_id=row["patient_id"],image_id=row["image_id"],assignment_group=row["assignment_group"],
                record_role=row["record_role"],member=member,image_member=image_member,
                actual_training_exposures=None if exposure is None else exposure.get(row["image_id"],0),
                checkpoint_sha256=checkpoint_hashes.get(model),residual_path=residual_path.as_posix(),
                residual_sha256=digest(output/residual_path),**measured)
            images.append(record)
            if len(images)%4==0:
                progress=dict(status="RUNNING",model=model,completed_images=len(images),
                    total_images=len(rows)*3,elapsed_seconds=time.perf_counter()-began)
                write_json(output/"partial_image_scores.json",images);write_json(output/"progress.json",progress)
                print(json.dumps(progress),flush=True)
        probe.assert_weights_unchanged();after=adapter_state(unet)
        assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
        assert probe.forward==len(rows)*20 and probe.backward==0
        model_reports.append(dict(model=model,prediction_type=probe.prediction_type,forward=probe.forward,
            backward=0,cuda_ms=probe.cuda_ms,weights_unchanged=True,parameter_versions_checked=len(probe.versions),
            all_parameters_frozen=True,parameter_gradients_absent=True,adapter_tensors_checked=len(before),
            base_lora_B_zero=(model=="base"),checkpoint_sha256=checkpoint_hashes.get(model),
            max_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.perf_counter()-model_started))
        del probe,unet,before,after;gc.collect();torch.cuda.empty_cache()
    write_json(output/"image_scores.json",images)
    for name,value in protected.items():
        assert digest(ROOT/name)==value
    for name,value in code_hashes.items():
        assert digest(Path(__file__).with_name(name))==value, "code changed during execution"
    report=dict(status="COMPLETED_PFAMI_EXECUTION_REQUIRES_INDEPENDENT_ANALYSIS",mode=mode,
        contract_sha256=digest(args.contract),protocol_sha256=digest(output/"protocol.json"),
        unique_patients=len(protocol["measurement_patient_ids"]),unique_images=len(rows),
        image_records=len(images),target_image_records=len(rows)*2,base_image_records=len(rows),
        total_forward=sum(r["forward"] for r in images),total_backward=0,model_reports=model_reports,
        encoding_report_sha256=digest(output/"encoding_report.json"),image_scores_sha256=digest(output/"image_scores.json"),
        minimum_original_loss=min(c["original_loss"] for r in images for c in r["cells"]),
        device=dict(name=torch.cuda.get_device_name(),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        elapsed_seconds=time.perf_counter()-began,original_protected_files_unchanged=len(protected),
        attack_success_declared=False,calibration_or_test_used=False,training_updates=0,
        benchmark_excluded_from_main_analysis=True)
    assert report["total_forward"]==len(rows)*60
    write_json(output/"report.json",report)
    write_json(output/"progress.json",dict(status="COMPLETE",completed_images=len(images),total_images=len(images)))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
