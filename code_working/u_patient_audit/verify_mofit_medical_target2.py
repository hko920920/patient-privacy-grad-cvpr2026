"""CPU provenance/exposure wrapper for target2's fixed MoFit packets.
All numerical verification uses the unchanged, hash-bound first verifier.
E is an image role; both E/U images and their patient are nonmembers of target2.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
CHECKPOINT = RUN / "training_coverage_v2/model_2/step_1000.pt"
CHECKPOINT_SHA = "ec54c5c405de03f2ba4daa361436ab1248acb541303e99291ab32ce3a7d63962"
HELPER_SHA = "cb48ec545eb1c14e0c572cfb4f7815eb0e4512708d6c8eb445d08f91fe2b713f"
IMAGES = ["00014393_002.png", "00014393_001.png"]
COUNTS = dict(unet_forward_calls=2202, unet_forward_examples=3203,
              unet_backward_calls=1300, vae_forward=1003, vae_backward=1000)
ARGUMENTS = dict(stage1_steps=1000, stage2_steps=300, nominal_stage1_steps=1000,
                 mode="literal_cfg2", monitor="literal_every_step", fd_steps=[],
                 endpoint_diagnostics=True, capture_pixel=[0, 1, 99, 499, 999],
                 capture_embedding=[0, 1, 29, 149, 299])


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def numeric_helper():
    need(sha(Path(__file__).with_name("verify_mofit_medical_kernel.py")) == HELPER_SHA,
         "unchanged numerical verifier")
    from .verify_mofit_medical_kernel import verify_packet, thash
    return verify_packet, thash


def actual_target2_exposure():
    need(sha(CHECKPOINT) == CHECKPOINT_SHA, "actual fixed target2 checkpoint identity")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    with (RUN / "cohort/model_2_train.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ledger = state["exposures"]
    need(state["step"] == 1000 and len(rows) == 912, "target2 training step/manifest size")
    need(len({r["image_id"] for r in rows}) == 912 and set(ledger) == {r["image_id"] for r in rows},
         "checkpoint exposure ledger exactly covers manifest")
    need(sum(ledger.values()) == 4000 and min(ledger.values()) > 0, "committed training exposures")
    patient_rows = [r for r in rows if r["patient_id"] == "14393"]
    patient_exposures = sum(int(ledger[r["image_id"]]) for r in patient_rows)
    need(not patient_rows and patient_exposures == 0, "whole patient absent from model2 training")
    need(all(int(ledger.get(image, 0)) == 0 for image in IMAGES), "both selected image exposures zero")
    need(len(state["adapter"]) == 256 and all(t.dtype == torch.float32 for t in state["adapter"].values()),
         "actual saved FP32 LoRA tensors")
    return state, dict(patient_training_images=len(patient_rows), patient_training_exposures=patient_exposures,
                       manifest_images=len(rows), training_exposures_total=sum(ledger.values()))


def verify(directory):
    directory = Path(directory).resolve()
    p, e = read(directory / "protocol.json"), read(directory / "execution.json")
    need(p["schema"] == "mofit-medical-target2-execution/v1" and p["mode"] == "full", "target2 execution schema")
    need(e["status"] == "PASS_TARGET2_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION"
         and e["complete"] is True and e["model"] == "model_2", "actual target2 completion")
    c = p["contract"]
    need(sha(p["contract_path"]) == p["contract_sha256"] and read(p["contract_path"]) == c,
         "external/embedded contract binding")
    expected_meta = dict(schema="mofit-medical-target2-contract/v1", current_step=2, patient_id="14393",
                         eval_role="fit", assignment_group="A", model="model_2", checkpoint_step=1000,
                         member=0, dtype="float32")
    for key, value in expected_meta.items():
        need(c[key] == value, "target2 contract " + key)
    need(Path(c["checkpoint"]).resolve() == CHECKPOINT.resolve() and c["checkpoint_sha256"] == CHECKPOINT_SHA,
         "real target2 checkpoint path/hash, not relabeled model1")
    need(c["image_order"] == IMAGES and c["kernel_arguments"] == ARGUMENTS, "same fixed full-kernel arguments")
    need(c["expected_counts_per_image"] == COUNTS
         and c["expected_total_counts"] == {k: 2*v for k,v in COUNTS.items()}, "fixed computational budget")
    for key in ("new_target_training", "attack_parameter_tuning", "cohort_expansion",
                "perform_membership_performance_evaluation", "stage2_completion_claim"):
        need(c[key] is False and e[key] is False, "unchanged scope " + key)
    frozen = c["frozen_sha256"]
    need(frozen[str(Path(__file__).resolve())] == sha(__file__), "verifier frozen before execution")
    for filename, expected in frozen.items():
        need(sha(filename) == expected, "frozen source/input changed: " + filename)
    for filename, field in (("protocol.json", "protocol_sha256"), ("results.json", "results_sha256"), ("raw.pt", "raw_sha256")):
        need(sha(directory / filename) == e[field], "output hash " + filename)
    state, exposure = actual_target2_exposure()
    need(c["actual_exposure"] == e["actual_exposure"] == exposure, "actual ledger-derived patient absence")
    need(c["image_member_by_image"] == {i: 0 for i in IMAGES}
         and c["actual_training_exposures_by_image"] == {i: 0 for i in IMAGES}, "image absence contract")
    old = read(c["numerical_contract_reference"])
    need(old["inputs_sha256"] == c["inputs_sha256"] and old["model_snapshot"] == c["model_snapshot"],
         "unchanged pixels/text/draw preparation and base artifacts")
    need(old["policy"] == c["kernel_policy"], "unchanged numerical kernel policy")
    alpha = float(c["alpha_cumprod_t140"])
    need(alpha == p["alpha_cumprod_t140"] == old["alpha_cumprod_t140"], "same t140 alpha")
    config = read(Path(c["model_snapshot"]) / "scheduler/scheduler_config.json")
    need(config["beta_schedule"] == "scaled_linear" and config["num_train_timesteps"] == 1000
         and config["prediction_type"] == "epsilon" and config["trained_betas"] is None, "fixed scheduler family")
    betas = torch.linspace(config["beta_start"] ** .5, config["beta_end"] ** .5, 1000, dtype=torch.float32) ** 2
    need(float(torch.cumprod(1 - betas, 0)[140]) == alpha, "independent scheduler reconstruction")
    input_path = Path(c["input_directory"]) / "inputs.pt"
    need(sha(input_path) == c["inputs_sha256"], "same original prepared inputs")
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    need(payload["policy"] == read(Path(c["input_directory"]) / "preparation_protocol.json"),
         "input preparation policy")
    need(e["before_fingerprints"] == e["after_fingerprints"], "unchanged full model/VAE values")
    reference = read(Path(c["pixel_FD_diagnostic_directory"]) / "execution.json")
    need(e["before_fingerprints"]["vae"] == reference["before_fingerprints"]["vae"]
         == reference["after_fingerprints"]["vae"], "same target1/target2 VAE artifact values")
    for key in ("target_and_VAE_fingerprints_equal", "checkpoint_adapter_values_exact",
                "adapter_tensors_exact_equal", "parameter_versions_and_no_weight_gradients", "frozen_files_unchanged"):
        need(e[key] is True, "producer actual-state guard " + key)
    need(e["patients"] == 1 and e["images"] == 2 and e["mode"] == "full", "same fixed patient scope")
    need(e["cfg_control_sha256"] is None and e["cfg_control_report_sha256"] is None, "no new CFG control")
    results = read(directory / "results.json")
    raws = torch.load(directory / "raw.pt", map_location="cpu", weights_only=True)
    need([r["image_id"] for r in results] == IMAGES and set(raws) == set(IMAGES), "full two-image coverage")
    verify_packet, thash = numeric_helper()
    checks, totals = [], {k: 0 for k in COUNTS}
    for report in results:
        image = report["image_id"]
        raw, item = raws[image], payload["images"][image]
        is_e = image == "00014393_002.png"
        source = item["metadata"]
        need(source["patient_id"] == "14393" and source["assignment_group"] == "A"
             and source["eval_role"] == "fit", "fixed original source patient")
        expected = dict(patient_id="14393", eval_role="fit", assignment_group="A", model="model_2",
                        scenario="E" if is_e else "U", record_role="train_candidate" if is_e else "U_observed",
                        member=0, image_member=0, actual_training_exposures=int(state["exposures"].get(image, 0)),
                        patient_training_exposures=exposure["patient_training_exposures"])
        for key, value in expected.items():
            need(report[key] == value, "actual target2 metadata " + key)
        need(report["record_role"] == source["record_role"], "E/U roles preserved independently of membership")
        need(report["caption"] == item["caption"], "same medical caption")
        for key, value in (("initial_pixels", item["pixels"]), ("initial_hidden", payload["hidden"][item["caption"]]),
                           ("null_hidden", payload["null_hidden"])):
            need(torch.equal(raw[key], value), "same prepared tensor " + key)
        for key, value in payload["draws"].items():
            if isinstance(value, torch.Tensor):
                need(torch.equal(raw["draws"][key], value), "same original random draw " + key)
            else:
                need(raw["draws"][key] == value, "same original random policy")
        need(thash(raw["initial_pixels"]) == report["pixels_sha256"] == item["pixels_sha256"], "original pixel hash")
        need(thash(raw["initial_hidden"]) == report["initial_hidden_sha256"], "original hidden hash")
        need(report["policy"] == c["kernel_policy"], "same literal kernel policy")
        for key in ("stage1_steps", "stage2_steps", "nominal_stage1_steps", "mode", "monitor", "fd_steps", "endpoint_diagnostics"):
            need(report[key] == ARGUMENTS[key], "same full numeric argument " + key)
        checked = verify_packet(report, raw, alpha)
        need(checked["counts"] == COUNTS, "per-image numerical counters")
        checks.append(checked)
        for key in totals:
            totals[key] += checked["counts"][key]
    need(totals == e["counts"] == c["expected_total_counts"], "total actual full-kernel budget")
    return dict(status="PASS_TARGET2_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_NONMEMBERSHIP",
                model="model_2", patient_id="14393", patients=1, images=2, counts=totals,
                actual_exposure=exposure, image_membership={i: 0 for i in IMAGES},
                checkpoint_sha256=CHECKPOINT_SHA, original_numerical_verifier_sha256=HELPER_SHA,
                inputs_sha256=c["inputs_sha256"], frozen_files_checked=len(frozen),
                image_verifications=checks, loaded_adapter_equality_is_producer_guard_evidence=True,
                independent_neural_network_backward_replay=False,
                numerical_verification_scope="saved full traces and ten sparse updates per image; no new target2 finite differences",
                membership_performance_claim=False, individual_patient_causal_claim=False,
                membership_label_scope="additional_NIH_training_only", pretrained_membership="unknown",
                paired_comparison_limit="whole training cohorts differ; two images from one patient cannot establish attack AUC",
                protocol_sha256=sha(directory / "protocol.json"), results_sha256=sha(directory / "results.json"),
                raw_sha256=sha(directory / "raw.pt"), execution_sha256=sha(directory / "execution.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-code-sha256", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    need(sha(__file__) == args.expected_code_sha256, "externally frozen target2 verifier")
    out, vp = args.run_dir / "verification.json", args.run_dir / "verification_protocol.json"
    need(not out.exists() and not vp.exists(), "immutable verification outputs")
    started = time.perf_counter()
    new_json(vp, dict(schema="mofit-target2-independent-verification/v1",
             verifier_sha256=sha(__file__), numerical_helper_sha256=HELPER_SHA,
             input_sha256={n: sha(args.run_dir / n) for n in ("protocol.json", "execution.json", "results.json", "raw.pt")},
             actual_nonmembership_source="model2 step1000 checkpoint ledger joined to full model2 training manifest",
             GPU_or_new_fitting=False))
    try:
        result = verify(args.run_dir)
    except BaseException as exc:
        result = dict(status="FAILED_TARGET2_INDEPENDENT_VERIFICATION", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(verifier_sha256=sha(__file__), verification_protocol_sha256=sha(vp),
                      CPU_seconds=time.perf_counter() - started)
        new_json(out, result)
        print(json.dumps({k: result[k] for k in ("status", "CPU_seconds", "verifier_sha256")}), flush=True)


if __name__ == "__main__":
    main()
