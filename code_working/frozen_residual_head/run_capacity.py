"""Fixed non-DP capacity check on existing development-only chest X-ray data.

No DP/noise or model-backbone optimization is performed by this runner.
Prepare contract, profile cost without reporting loss, extract, then analyze.
Original cohort/model/cache files are read-only and source-bound.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import residual_math as rm
from u_patient_audit.common import (
    ROOT, RUN, IMAGES, MODEL_ID, MODEL_REVISION, digest, read_csv)
from u_patient_audit.models import (
    setup, snapshot, scheduler, diffusion_input, load_cache, UNet2DConditionModel)

DEFAULT_OUT = ROOT / "_reports/frozen_residual_capacity_20260916_v1"
SALT = "frozen-residual-capacity-20260916-v1"
METHODS = ("full", "static", "time")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seeded(*parts):
    text = "|".join(map(str, (SALT,) + parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15], 16)


def save_npz(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez(handle, **arrays)


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    cache = load_cache()
    snap, sched = snapshot(), scheduler()
    assert sched.config.prediction_type == "epsilon"
    source = RUN / "cohort/evaluation_images.csv"
    all_rows = read_csv(source)
    selected = [r for r in all_rows if r["eval_role"] in ("fit", "selection")]
    selected.sort(key=lambda r: (r["eval_role"] != "fit", int(r["patient_id"]), r["image_id"]))
    groups = defaultdict(list)
    for r in selected:
        groups[(r["eval_role"], r["patient_id"])].append(r)
    assert Counter(role for role, _ in groups) == {"fit": 80, "selection": 40}
    assert all(len(v) == 4 for v in groups.values())
    assert not ({p for role, p in groups if role == "fit"} &
                {p for role, p in groups if role == "selection"})
    excluded = {r["patient_id"] for r in all_rows if r["eval_role"] in ("calibration", "test")}
    assert not {p for _, p in groups} & excluded
    a = sched.alphas_cumprod.cpu().numpy().astype(np.float64)
    logsnr = np.log(a) - np.log1p(-a)
    p = rm.make_channel_projection(320, 15, seed=26091601)
    image_rows = []
    for r in selected:
        path = IMAGES / r["image_filename"]
        assert path.exists() and digest(path) == r["sha256"]
        iid = r["image_id"]
        assert tuple(cache["latents"][iid].shape) == (1, 4, 32, 32)
        prompt = cache["train_prompts"][iid]
        assert tuple(cache["hidden"][prompt].shape) == (1, 77, 1024)
        image_rows.append(dict(
            split="train" if r["eval_role"] == "fit" else "eval",
            patient_id=r["patient_id"], image_id=iid,
            original_eval_role=r["eval_role"], original_record_role=r["record_role"],
            path=str(path), sha256=r["sha256"], prompt=prompt))
    source_files = [
        Path(__file__), Path(rm.__file__), ROOT / "u_patient_audit/models.py",
        ROOT / "u_patient_audit/common.py", ROOT / "u_patient_audit/build_cache.py",
        ROOT / "data_pipeline/nih_cxr14_model_input.py",
        RUN / "cohort/lock.json", source, RUN / "cache/cache.pt",
        RUN / "cache/summary.json", snap / "unet/config.json",
        snap / "scheduler/scheduler_config.json",
        snap / "unet/diffusion_pytorch_model.fp16.safetensors"]
    bindings = {str(path): digest(path) for path in source_files}
    weight = "unet/diffusion_pytorch_model.fp16.safetensors"
    expected = read_json(RUN / "cache/summary.json")["model_hashes"][weight]["actual"].lower()
    assert bindings[str(snap / weight)] == expected
    out.mkdir(parents=True)
    save_npz(out / "projection.npz", P=p, alphas=sched.alphas_cumprod.cpu().numpy())
    save_json(out / "images.json", image_rows)
    contract = dict(
        schema="frozen-residual-capacity/v1", created_utc=stamp(), current_stage=2,
        scope="non-DP restricted-family denoising capacity on previously used development patients",
        model_id=MODEL_ID, model_revision=MODEL_REVISION, snapshot=str(snap),
        model_policy="direct fresh base; no LoRA adapters; fp16 artifact promoted to fp32",
        prediction_type="epsilon", latent_shape=[1, 4, 32, 32], batch_size=1,
        precision="FP32 model/inputs/projection; CPU FP64 residual subtraction/statistics/solve",
        tf32=False, deterministic_algorithms=True,
        tap="unet.conv_out forward_pre_hook input after conv_norm_out and conv_act",
        feature_policy="constant1 first, then 15 Gaussian-QR projected channels; four logSNR Legendre terms",
        projection_seed=26091601, projected_channels=15, spatial_channels=16,
        method_dimensions={"full": 64, "static": 16, "time": 4},
        time_basis_logsnr_min=float(logsnr.min()), time_basis_logsnr_max=float(logsnr.max()),
        timestep_policy="eight equal timestep strata; independent hash-seeded integer jitter per image/draw",
        draws_per_image=8, ridge_lambda=0.001, images_per_patient=4,
        expected_patients={"train": 80, "eval": 40},
        expected_images={"train": 320, "eval": 160}, expected_records=3840,
        expected_backbone_backward=0, noise_rng="torch CPU Generator, FP32 randn",
        prompt_policy="cached per-image weak-label prompt; no identity/clean image/noise features",
        objective="equal patients, equal four images, equal eight draws; spatial mean output-channel SUM",
        metric="MSE = quadratic output-channel SUM / 4; lower is better",
        primary_comparison="full64 residual vs frozen base on patient-disjoint development cohort",
        alternative_comparisons=["static16", "time4"],
        development_only=True, private_release=False, dp_applied=False,
        clipping_applied=False, performance_tuning_allowed=False,
        no_adaptive_rank_lambda_seed_timestep_selection=True,
        bootstrap_seed=26091602, bootstrap_replicates=8000,
        source_sha256=bindings, projection_sha256=digest(out / "projection.npz"),
        images_sha256=digest(out / "images.json"), seed_salt=SALT,
        future_claim_boundaries=[
            "not a final independent test", "not generated-image quality or patient MIA",
            "not a DP model", "not advantage over matched DP-SGD/LoRA",
            "one lambda/random subspace is not all residual-head capacities"])
    save_json(out / "contract.json", contract)
    print(json.dumps(dict(phase="prepared", records=3840, patients=[80, 40],
                         code_and_inputs_bound=True, new_GPU_calls=0)), flush=True)


def validate(out):
    c = read_json(out / "contract.json")
    assert c["schema"] == "frozen-residual-capacity/v1"
    for name, expected in c["source_sha256"].items():
        assert digest(name) == expected, "Changed source: " + name
    assert digest(out / "projection.npz") == c["projection_sha256"]
    assert digest(out / "images.json") == c["images_sha256"]
    assert c["expected_records"] == 3840 and not c["dp_applied"]
    assert c["ridge_lambda"] == 0.001
    return c


def sample_plan(c, image_rows):
    rows = []
    for image in image_rows:
        for j in range(c["draws_per_image"]):
            t_seed = seeded(image["split"], image["image_id"], j, "timestep")
            t = 125 * j + int(np.random.default_rng(t_seed).integers(0, 125))
            rows.append({**image, "record_id": len(rows), "draw_id": j, "timestep": t,
                         "noise_seed": seeded(image["split"], image["image_id"], j, "noise")})
    return rows


def new_model(c):
    setup()
    unet = UNet2DConditionModel.from_pretrained(
        Path(c["snapshot"]) / "unet", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=True).float().eval()
    assert not any("lora" in name.lower() for name, _ in unet.named_parameters())
    unet.requires_grad_(False).to("cuda")
    assert sum(p.numel() for p in unet.parameters() if p.requires_grad) == 0
    assert unet.conv_out.in_channels == 320 and unet.conv_out.out_channels == 4
    return unet


def extract(out, profile=False):
    c = validate(out)
    started = time.perf_counter()
    images, cache, sched = read_json(out / "images.json"), load_cache(), scheduler()
    plan = sample_plan(c, images)
    final_path = out / ("profile.json" if profile else "execution.json")
    if final_path.exists():
        raise FileExistsError(final_path)
    raw_dir = out / ("profile_raw" if profile else "raw")
    raw_dir.mkdir(exist_ok=False)
    projection = np.load(out / "projection.npz")
    unet = new_model(c)
    p = torch.from_numpy(projection["P"].astype(np.float32)).to("cuda")
    captured = {}
    def tap(module, inputs):
        assert len(inputs) == 1
        captured["hidden"] = inputs[0].detach()
    hook = unet.conv_out.register_forward_pre_hook(tap)
    selected = plan[:8] if profile else plan
    first_image = {split: next(r["image_id"] for r in plan if r["split"] == split)
                   for split in ("train", "eval")}
    timings, manifest = [], []
    actual_f = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for index, row in enumerate(selected):
                z = cache["latents"][row["image_id"]].float()
                eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(row["noise_seed"]),
                                  dtype=torch.float32)
                noisy, target = diffusion_input(z, torch.tensor([row["timestep"]]), eps, sched)
                xx, tt = noisy.to("cuda"), torch.tensor([row["timestep"]], device="cuda")
                hh = cache["hidden"][row["prompt"]].float().to("cuda")
                if index == 0:
                    _ = unet(xx, tt, hh).sample
                    actual_f += 1
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                pred = unet(xx, tt, hh).sample
                actual_f += 1
                h = captured.pop("hidden")[0].permute(1, 2, 0).reshape(-1, 320)
                assert h.shape == (1024, 320)
                projected = h @ p
                features = torch.cat((torch.ones((len(h), 1), device=h.device), projected), dim=1)
                f_np = features.cpu().numpy().copy()
                base_np = pred[0].permute(1, 2, 0).reshape(-1, 4).cpu().numpy().copy()
                target_np = target[0].permute(1, 2, 0).reshape(-1, 4).cpu().numpy().copy()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - tick)
                basis = rm.legendre_time_basis(
                    np.int64(row["timestep"]), projection["alphas"],
                    logsnr_min=c["time_basis_logsnr_min"], logsnr_max=c["time_basis_logsnr_max"])
                arrays = dict(features=f_np, base=base_np, target=target_np,
                              basis=basis, timestep=np.array(row["timestep"], dtype=np.int64))
                if row["image_id"] == first_image[row["split"]] and row["draw_id"] in (0, 7):
                    arrays["hidden"] = h.cpu().numpy().copy()
                assert all(np.isfinite(v).all() for v in arrays.values())
                path = raw_dir / ("%06d.npz" % row["record_id"])
                save_npz(path, **arrays)
                manifest.append({**row, "raw_path": path.relative_to(out).as_posix(),
                                 "raw_sha256": digest(path)})
                if index % 128 == 0 or index + 1 == len(selected):
                    print(json.dumps(dict(
                        phase="profile" if profile else "extract", completed=index+1,
                        total=len(selected), seconds=round(time.perf_counter()-started, 3),
                        seconds_per_record=round(float(np.mean(timings)), 5))), flush=True)
    finally:
        hook.remove()
    assert all(p.grad is None and not p.requires_grad for p in unet.parameters())
    result = dict(
        phase="profile" if profile else "extract", completed=True, utc=stamp(),
        records=len(manifest), unet_forward=actual_f, unet_backward=0, warmup_forward=1,
        new_vae_or_text_encoder_forward=0,
        model_load_and_output_seconds=time.perf_counter()-started,
        mean_record_seconds=float(np.mean(timings)), median_record_seconds=float(np.median(timings)),
        max_record_seconds=float(np.max(timings)),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        device=torch.cuda.get_device_name(),
        versions={k: version(k) for k in ("torch", "diffusers", "numpy", "transformers")},
        contract_sha256=digest(out / "contract.json"), no_backbone_parameter_gradients=True,
        tf32=False, no_performance_scores_reported_in_profile=profile)
    if profile:
        result["projected_main_forward_seconds"] = float(np.mean(timings))*c["expected_records"]
        save_json(out / "profile_manifest.json", manifest)
    else:
        assert len(manifest) == c["expected_records"]
        save_json(out / "manifest.json", manifest)
        result["manifest_sha256"] = digest(out / "manifest.json")
    save_json(final_path, result)
    print(json.dumps(result), flush=True)


def summarize_difference(a, b, c):
    """Paired patient bootstrap; pixels and noise are not IID people."""
    delta = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(c["bootstrap_seed"])
    ids = rng.integers(0, len(delta), size=(c["bootstrap_replicates"], len(delta)))
    boots = delta[ids].mean(axis=1)
    return dict(mean=float(delta.mean()), ci95=np.quantile(boots, [.025, .975]).tolist(),
                patients=len(delta), positive_patients=int((delta > 0).sum()),
                scope="pointwise descriptive patient bootstrap on reused development cohort")


def analyze(out):
    c, started = validate(out), time.perf_counter()
    if (out / "analysis.json").exists():
        raise FileExistsError(out / "analysis.json")
    manifest, execution = read_json(out / "manifest.json"), read_json(out / "execution.json")
    assert execution["completed"] and digest(out / "manifest.json") == execution["manifest_sha256"]
    assert len(manifest) == c["expected_records"]
    grouped = defaultdict(list)
    for row in manifest:
        grouped[(row["split"], row["patient_id"])].append(row)
    keys = sorted(grouped, key=lambda k: (k[0] != "train", int(k[1])))
    all_stats = {name: [] for name in METHODS}
    for split, pid in keys:
        by_image = defaultdict(list)
        for row in grouped[(split, pid)]:
            by_image[row["image_id"]].append(row)
        assert len(by_image) == c["images_per_patient"]
        image_stats = {name: [] for name in METHODS}
        for iid, draws in sorted(by_image.items()):
            assert len(draws) == c["draws_per_image"]
            draw_stats = {name: [] for name in METHODS}
            for row in draws:
                path = out / row["raw_path"]
                assert digest(path) == row["raw_sha256"]
                with np.load(path) as raw:
                    x = raw["features"].astype(np.float64)
                    residual = raw["target"].astype(np.float64)-raw["base"].astype(np.float64)
                    b = raw["basis"].astype(np.float64)
                static = rm.sufficient_statistics(x, residual)
                full = dict(A=np.kron(np.outer(b, b), static["A"]),
                            B=np.kron(b[:, None], static["B"]), Q=static["Q"])
                time_only = dict(A=np.outer(b, b),
                                 B=b[:, None]*residual.mean(axis=0)[None, :], Q=static["Q"])
                for name, values in (("full", full), ("static", static), ("time", time_only)):
                    draw_stats[name].append(values)
            for name in METHODS:
                image_stats[name].append(rm.mean_statistics(draw_stats[name]))
        for name in METHODS:
            all_stats[name].append(rm.mean_statistics(image_stats[name]))
    archive = dict(patient_ids=np.array([p for _, p in keys]), splits=np.array([s for s, _ in keys]))
    for name in METHODS:
        for key in ("A", "B", "Q"):
            archive[key+"_"+name] = np.stack([s[key] for s in all_stats[name]])
    save_npz(out / "patient_stats.npz", **archive)
    train_mask = archive["splits"] == "train"
    weights, fits = {}, {}
    for name in METHODS:
        avg = rm.mean_patient_statistics([all_stats[name][i] for i in np.where(train_mask)[0]])
        w = rm.solve_ridge(avg["A"], avg["B"], c["ridge_lambda"])
        weights["W_"+name] = w
        eig = np.linalg.eigvalsh(avg["A"])
        fits[name] = dict(
            feature_dimension=len(w), parameter_count=int(w.size), ridge_lambda=c["ridge_lambda"],
            weight_norm=float(np.linalg.norm(w)), gram_min_eigenvalue=float(eig[0]),
            gram_max_eigenvalue=float(eig[-1]),
            stationarity_norm=float(np.linalg.norm(
                (avg["A"]+c["ridge_lambda"]*np.eye(len(w)))@w-avg["B"])))
    save_npz(out / "models.npz", **weights)
    per_patient = []
    for i, (split, pid) in enumerate(keys):
        row = dict(patient_id=pid, split=split, base_mse=float(all_stats["full"][i]["Q"]/4))
        for name in METHODS:
            st = all_stats[name][i]
            row[name+"_mse"] = float(rm.loss_from_statistics(
                st["A"], st["B"], st["Q"], weights["W_"+name])/4)
        per_patient.append(row)
    summaries = {}
    for split in ("train", "eval"):
        chosen = [r for r in per_patient if r["split"] == split]
        means = {name: float(np.mean([r[name+"_mse"] for r in chosen])) for name in ("base",)+METHODS}
        diffs = {}
        for a, b in (("base", "full"), ("static", "full"), ("time", "full"),
                     ("base", "static"), ("base", "time")):
            diffs[a+"_minus_"+b] = summarize_difference(
                [r[a+"_mse"] for r in chosen], [r[b+"_mse"] for r in chosen], c)
        summaries[split] = dict(
            mean_mse=means, paired_reductions=diffs,
            relative_full_reduction=(means["base"]-means["full"])/means["base"])
    result = dict(
        schema="frozen-residual-capacity-analysis/v1", utc=stamp(), ridge_lambda=c["ridge_lambda"],
        per_patient_results=per_patient, summaries=summaries, fits=fits,
        seconds=time.perf_counter()-started,
        primary="eval base_minus_full", secondary="eval static_minus_full and time_minus_full",
        generative_quality_measured=False, dp_mechanism_executed=False,
        information_boundary="features depend on noisy latent/timestep/declared cached weak-label prompt",
        source_sha256={f: digest(out/f) for f in (
            "contract.json", "manifest.json", "execution.json", "patient_stats.npz", "models.npz")})
    save_json(out / "analysis.json", result)
    print(json.dumps({k: result[k] for k in ("seconds", "summaries", "fits")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "profile", "extract", "analyze"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if args.phase == "prepare":
        prepare(out)
    elif args.phase in ("profile", "extract"):
        extract(out, profile=args.phase == "profile")
    else:
        analyze(out)


if __name__ == "__main__":
    main()
