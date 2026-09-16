"""Fixed E/U loss controls on the existing eight fit patients only.

No fitting, conditioning optimization, new patients, or low-FPR evaluation.
Both prompts and both arms use the original audit query noise for every image.
"""
import argparse
import gc
import json
import math
import re
import time
from collections import defaultdict

from .common import ROOT, RUN, PROMPT, digest, read_csv, seed, verify_inputs, write_json
from .models import diffusion_input, load_cache, load_unet, scheduler, setup, torch

TIMESTEPS = (50, 250, 500, 750, 950)
DRAWS = 2
BATCH_SIZE = 4
PROMPT_KINDS = ("generic", "matched_weak_label")
SOURCE_DIR = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_existing_patients():
    original = read_json(SOURCE_DIR / "results.json")
    per_model = {}
    for model in ("model_1", "model_2"):
        records = [r for r in original if r["model"] == model]
        assert len(records) == 8
        assert all(r["scenario"] == "U" and r["eval_role"] == "fit" for r in records)
        ids = {r["patient_id"] for r in records}
        assert len(ids) == 8
        per_model[model] = ids
    assert per_model["model_1"] == per_model["model_2"]
    patients = sorted(per_model["model_1"])
    grouped = defaultdict(list)
    for row in read_csv(RUN / "cohort/evaluation_images.csv"):
        if row["patient_id"] in patients:
            grouped[row["patient_id"]].append(row)
    rows = []
    for patient in patients:
        selected = grouped[patient]
        assert len(selected) == 4
        assert all(r["eval_role"] == "fit" for r in selected)
        assert len({r["assignment_group"] for r in selected}) == 1
        assert len({r["image_id"] for r in selected}) == 4
        for scenario, record_role in (("E", "train_candidate"), ("U", "U_observed")):
            images = sorted(
                [r for r in selected if r["record_role"] == record_role],
                key=lambda r: r["image_id"],
            )
            assert len(images) == 2
            if scenario == "U":
                for model in per_model:
                    prior = next(r for r in original
                                 if r["model"] == model and r["patient_id"] == patient)
                    assert {r["image_id"] for r in images} == {
                        fold["query"] for fold in prior["folds"]}
            rows.extend(dict(r, scenario=scenario) for r in images)
    assert len(rows) == 32
    assert sum(grouped[p][0]["assignment_group"] == "A" for p in patients) == 4
    return patients, rows, original


def verify_model_membership(model, rows, original, checkpoint, expected_sha):
    assert digest(checkpoint) == expected_sha
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 1000
    protocol = read_json(RUN / "training_coverage_v2/protocol.json")
    assert state["contract"] == protocol
    exposures = state["exposures"]
    manifest = read_csv(RUN / "cohort" / (model + "_train.csv"))
    train_ids = {r["image_id"] for r in manifest}
    assert set(exposures) == train_ids and min(exposures.values()) >= 1
    assert sum(exposures.values()) == 4000
    group = "A" if model == "model_1" else "B"
    declared = {}
    for row in rows:
        member = int(row["assignment_group"] == group)
        expected_image_member = bool(member and row["scenario"] == "E")
        assert (row["image_id"] in train_ids) == expected_image_member
        assert (exposures.get(row["image_id"], 0) > 0) == expected_image_member
        declared[row["patient_id"]] = member
    for patient, member in declared.items():
        patient_rows = [r for r in rows if r["patient_id"] == patient]
        realized = int(any(exposures.get(r["image_id"], 0) > 0 for r in patient_rows))
        assert realized == member
        old = next(r for r in original
                   if r["model"] == model and r["patient_id"] == patient)
        assert old["declared_member"] == old["realized_member"] == member
        assert old["checkpoint_sha256"] == expected_sha
    del state
    gc.collect()
    return declared


def aggregate(cells):
    grouped = defaultdict(list)
    for cell in cells:
        key = (cell["model"], cell["scenario"], cell["prompt_kind"],
               cell["patient_id"], cell["image_id"])
        grouped[key].append(cell)
    image_scores = []
    identity = ("model", "scenario", "prompt_kind", "patient_id", "image_id",
                "eval_role", "declared_member", "realized_member")
    for key, values in sorted(grouped.items()):
        assert len(values) == len(TIMESTEPS) * DRAWS
        assert {(v["timestep"], v["draw"]) for v in values} == {
            (t, d) for t in TIMESTEPS for d in range(DRAWS)}
        result = {field: values[0][field] for field in identity}
        base = math.fsum(v["base_loss"] for v in values) / len(values)
        target = math.fsum(v["target_loss"] for v in values) / len(values)
        result.update(base_loss=base, target_loss=target, base_minus_target=base-target)
        image_scores.append(result)
    patients = defaultdict(list)
    for image in image_scores:
        key = tuple(image[k] for k in ("model", "scenario", "prompt_kind", "patient_id"))
        patients[key].append(image)
    patient_scores = []
    for key, values in sorted(patients.items()):
        assert len(values) == 2
        result = {field: values[0][field] for field in identity if field != "image_id"}
        result.update(
            image_ids=sorted(v["image_id"] for v in values),
            loss_mean=-math.fsum(v["target_loss"] for v in values) / 2,
            loss_max=max(-v["target_loss"] for v in values),
            base_difference_mean=math.fsum(v["base_minus_target"] for v in values) / 2,
        )
        patient_scores.append(result)
    return image_scores, patient_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-tag", default="basic_controls")
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag), "invalid output tag"
    output = RUN / "verification_20260914" / args.output_tag
    if output.exists():
        raise RuntimeError("Immutable diagnostic output already exists; use a new output tag")
    started = time.perf_counter()
    verify_inputs()
    patients, rows, original = select_existing_patients()
    checkpoints = {
        model: RUN / "training_coverage_v2" / model / "step_1000.pt"
        for model in ("model_1", "model_2")
    }
    checkpoint_sha = {
        model: read_json(path.parent / "report_1000.json")["checkpoint_sha256"]
        for model, path in checkpoints.items()
    }
    source_paths = [
        SOURCE_DIR / "results.json", SOURCE_DIR / "report.json",
        SOURCE_DIR / "verification.json", RUN / "cohort/lock.json",
        RUN / "cohort/evaluation_images.csv", RUN / "cohort/auxiliary_images.csv",
        RUN / "cohort/model_1_train.csv", RUN / "cohort/model_2_train.csv",
        RUN / "cache/summary.json", RUN / "training_coverage_v2/protocol.json",
    ]
    input_sha = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}
    code_paths = [
        ROOT / "u_patient_audit" / name
        for name in ("run_basic_controls.py", "models.py", "common.py", "probe.py")
    ]
    code_sha = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths}
    cache_meta = read_json(RUN / "cache/summary.json")
    protocol = {
        "schema": "cvpr-u-basic-controls/v1",
        "scope": "exploratory fixed-loss controls on the same eight fit patients",
        "query_timesteps": list(TIMESTEPS), "draws": DRAWS, "batch_size": BATCH_SIZE,
        "phase": "query", "noise_seed_parts": ["audit-noise", "query", "image_id", "timestep", "draw"],
        "generic_prompt": PROMPT,
        "matched_prompt_policy": "cache.train_prompts: original per-image weak-label training prompt",
        "prompt_kinds": list(PROMPT_KINDS), "model_mode": "train",
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_recomputation": False,
        "gradient_mode": "inference_mode; no gradient computation",
        "precision": "FP16 model/latents/conditioning, autocast FP16, loss reduction FP32",
        "freeze_policy": "requires_grad_(False) after each adapter switch; versions and .grad checked",
        "conditioning_optimization": False, "new_patients": 0,
        "unique_patients": len(patients), "images_per_model": len(rows),
        "code_sha256": code_sha, "input_sha256": input_sha,
        "checkpoint_sha256": checkpoint_sha, "cache_sha256": cache_meta["cache_sha256"],
        "cohort_lock_sha256": digest(RUN / "cohort/lock.json"),
        "low_fpr_performance_or_novelty_claim": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    setup()
    cache = load_cache()
    assert cache["audit_prompt"] == PROMPT
    assert all(r["image_id"] in cache["latents"] for r in rows)
    cells = []
    model_reports = {}
    global_forward_examples = 0
    global_forward_calls = 0
    for model, checkpoint in checkpoints.items():
        model_started = time.perf_counter()
        members = verify_model_membership(model, rows, original, checkpoint, checkpoint_sha[model])
        unet = load_unet(checkpoint, training=False)
        # Match original ResponseProbe mode; all dropout probabilities must be zero.
        unet.train()
        assert not any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0
                       for m in unet.modules())
        unet.requires_grad_(False)
        versions = {name: p._version for name, p in unet.named_parameters()}
        sched = scheduler()
        torch.cuda.reset_peak_memory_stats()
        model_calls = 0
        model_examples = 0
        for prompt_kind in PROMPT_KINDS:
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start+BATCH_SIZE]
                assert len(batch) == BATCH_SIZE
                z = torch.cat([cache["latents"][r["image_id"]].clone() for r in batch]).to(
                    "cuda", torch.float16)
                prompt_texts = [
                    PROMPT if prompt_kind == "generic" else cache["train_prompts"][r["image_id"]]
                    for r in batch
                ]
                hidden = torch.cat([cache["hidden"][p].clone() for p in prompt_texts]).to(
                    "cuda", torch.float16)
                for tvalue in TIMESTEPS:
                    for draw in range(DRAWS):
                        noise_seeds = [
                            seed("audit-noise", "query", r["image_id"], tvalue, draw)
                            for r in batch
                        ]
                        noises = [
                            torch.randn(cache["latents"][r["image_id"]].shape,
                                        generator=torch.Generator(device="cuda").manual_seed(s),
                                        device="cuda", dtype=torch.float16)
                            for r, s in zip(batch, noise_seeds)
                        ]
                        noise = torch.cat(noises)
                        t = torch.full((len(batch),), tvalue, device="cuda", dtype=torch.long)
                        noisy, target = diffusion_input(z, t, noise, sched)
                        values = {}
                        for arm in ("base", "target"):
                            if arm == "base":
                                unet.disable_adapters()
                            else:
                                unet.enable_adapters()
                            unet.requires_grad_(False)
                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                                prediction = unet(noisy, t, hidden).sample
                                losses = (prediction.float() - target.float()).square().mean(
                                    dim=(1, 2, 3))
                            assert torch.isfinite(losses).all()
                            values[arm] = losses.cpu().tolist()
                            model_calls += 1
                            model_examples += len(batch)
                        for index, row in enumerate(batch):
                            cells.append({
                                "model": model, "scenario": row["scenario"],
                                "prompt_kind": prompt_kind, "patient_id": row["patient_id"],
                                "image_id": row["image_id"], "eval_role": "fit",
                                "declared_member": members[row["patient_id"]],
                                "realized_member": members[row["patient_id"]],
                                "timestep": tvalue, "draw": draw, "noise_seed": noise_seeds[index],
                                "target_loss": values["target"][index],
                                "base_loss": values["base"][index],
                            })
                torch.cuda.synchronize()
                assert all(p.grad is None and not p.requires_grad for p in unet.parameters())
                assert all(p._version == versions[name] for name, p in unet.named_parameters())
                progress = {
                    "status": "RUNNING", "model": model, "prompt_kind": prompt_kind,
                    "images_completed_in_prompt": start+len(batch), "images_per_prompt": len(rows),
                    "completed_cells": len(cells), "expected_cells": 1280,
                    "model_forward_examples": model_examples, "model_forward_batch_calls": model_calls,
                    "elapsed_seconds": time.perf_counter()-started, "weights_unchanged": True,
                }
                write_json(output / "partial_cell_losses.json", cells)
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        assert model_examples == 1280 and model_calls == 320
        model_reports[model] = {
            "seconds_including_model_load": time.perf_counter()-model_started,
            "forward_model_examples": model_examples, "forward_batch_calls": model_calls,
            "backward": 0, "max_cuda_GiB": torch.cuda.max_memory_allocated()/2**30,
            "weights_unchanged": True, "parameter_versions_checked": len(versions),
            "parameter_gradients_absent": True, "all_parameters_frozen": True,
            "dropout_nonzero_count": 0, "checkpoint_sha256": checkpoint_sha[model],
            "U_image_exposures": 0, "E_membership_matches_actual_exposure": True,
        }
        global_forward_examples += model_examples
        global_forward_calls += model_calls
        del unet, sched, z, hidden, noisy, target, prediction, losses, noise, noises
        gc.collect()
        torch.cuda.empty_cache()
    assert len(cells) == 1280
    images, scores = aggregate(cells)
    assert len(images) == 128 and len(scores) == 64
    write_json(output / "cell_losses.json", cells)
    write_json(output / "image_scores.json", images)
    write_json(output / "patient_scores.json", scores)
    report = {
        "status": "PASS_EXECUTION_EXPLORATORY_CONTROLS",
        "schema": "cvpr-u-basic-controls-report/v1",
        "protocol_sha256": digest(output / "protocol.json"),
        "code_sha256": code_sha, "input_sha256": input_sha,
        "checkpoint_sha256": checkpoint_sha, "cohort_lock_sha256": protocol["cohort_lock_sha256"],
        "cache_sha256": cache_meta["cache_sha256"], "unique_patients": len(patients),
        "new_patients": 0, "role": "fit", "calibration_or_test_evaluated": False,
        "image_records": len(images), "patient_score_records": len(scores),
        "noise_cell_records": len(cells), "forward_model_examples": global_forward_examples,
        "forward_batch_calls": global_forward_calls, "backward": 0, "optimizer_steps": 0,
        "per_model": model_reports, "all_weights_unchanged": True,
        "model_mode": "train", "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_recomputation": False,
        "seconds_total": time.perf_counter()-started,
        "output_sha256": {
            name: digest(output/name)
            for name in ("cell_losses.json", "image_scores.json", "patient_scores.json")
        },
        "limitations": [
            "Eight existing fit patients and one correlated inclusion-swap model pair only.",
            "Generic versus matched-label prompts have different information access.",
            "E success does not establish U success; E failure does not rule out U leakage.",
            "No fitted attack, low-FPR estimate, clinical utility, DP or novelty conclusion.",
            "Batch4 FP16 may differ numerically from original batch1; independent comparison required.",
        ],
    }
    write_json(output / "report.json", report)
    write_json(output / "progress.json", {"status": "COMPLETE", "report": "report.json",
                                        "seconds_total": report["seconds_total"]})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

