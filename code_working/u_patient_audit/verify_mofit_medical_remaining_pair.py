"""CPU outer verification of the fixed remaining E/U pair in both targets.

The original hash-bound verify_packet supplies unchanged numerical checks.
This wrapper derives membership from each actual checkpoint and manifest.
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
MODELS = ["model_1", "model_2"]
IMAGES = ["00014393_006.png", "00014393_004.png"]
CHECKPOINT_SHA = {
    "model_1": "ae789c32ce9ebaca3eb568fda968c28d5b9cbdf39682d7a498615b26cbb08435",
    "model_2": "ec54c5c405de03f2ba4daa361436ab1248acb541303e99291ab32ce3a7d63962",
}
HELPER_SHA = "cb48ec545eb1c14e0c572cfb4f7815eb0e4512708d6c8eb445d08f91fe2b713f"
COUNTS = dict(unet_forward_calls=2202, unet_forward_examples=3203,
              unet_backward_calls=1300, vae_forward=1003, vae_backward=1000)
ARGUMENTS = dict(stage1_steps=1000, stage2_steps=300, nominal_stage1_steps=1000,
                 mode="literal_cfg2", monitor="literal_every_step", fd_steps=[],
                 endpoint_diagnostics=True, capture_pixel=[0, 1, 99, 499, 999],
                 capture_embedding=[0, 1, 29, 149, 299])
SCOPE_FLAGS = ("new_target_training", "attack_parameter_tuning", "cohort_expansion",
               "perform_membership_performance_evaluation", "stage2_completion_claim")


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


def checkpoint(model):
    return RUN / "training_coverage_v2" / model / "step_1000.pt"


def numerical_helper():
    need(sha(Path(__file__).with_name("verify_mofit_medical_kernel.py")) == HELPER_SHA,
         "unchanged numerical verifier")
    from .verify_mofit_medical_kernel import verify_packet, thash
    return verify_packet, thash


def actual_exposures():
    states, summaries, image_exposures = {}, {}, {}
    for model in MODELS:
        need(sha(checkpoint(model)) == CHECKPOINT_SHA[model], "actual fixed checkpoint " + model)
        state = torch.load(checkpoint(model), map_location="cpu", weights_only=True)
        with (RUN / "cohort" / (model + "_train.csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ledger = state["exposures"]
        need(state["step"] == 1000 and len(rows) == 912, "training step and manifest count")
        need(len({r["image_id"] for r in rows}) == 912
             and set(ledger) == {r["image_id"] for r in rows}, "complete manifest/ledger correspondence")
        need(sum(ledger.values()) == 4000 and min(ledger.values()) > 0, "actual committed exposures")
        patient_rows = [r for r in rows if r["patient_id"] == "14393"]
        patient_exposures = sum(int(ledger[r["image_id"]]) for r in patient_rows)
        need({r["image_id"] for r in patient_rows}
             == ({"00014393_002.png", "00014393_006.png"} if model == "model_1" else set()),
             "actual patient training images")
        need(patient_exposures == (8 if model == "model_1" else 0), "actual patient training exposure")
        image_exposures[model] = {i: int(ledger.get(i, 0)) for i in IMAGES}
        need(image_exposures[model] == dict(zip(IMAGES, [4, 0] if model == "model_1" else [0, 0])),
             "actual remaining image exposures")
        need(len(state["adapter"]) == 256
             and all(t.dtype == torch.float32 for t in state["adapter"].values()), "saved FP32 LoRA state")
        summaries[model] = dict(patient_training_images=len(patient_rows),
            patient_training_exposures=patient_exposures, manifest_images=len(rows),
            training_exposures_total=sum(ledger.values()))
        states[model] = state
    return states, summaries, image_exposures


def verify(directory):
    directory = Path(directory).resolve()
    p, e = read(directory / "protocol.json"), read(directory / "execution.json")
    need(p["schema"] == "mofit-medical-remaining-pair-execution/v1" and p["mode"] == "full",
         "remaining-pair execution schema")
    need(e["status"] == "PASS_REMAINING_PAIR_EXECUTION_PENDING_INDEPENDENT_VERIFICATION"
         and e["complete"] is True, "actual complete extraction")
    c = p["contract"]
    need(sha(p["contract_path"]) == p["contract_sha256"] and read(p["contract_path"]) == c,
         "external and embedded contract binding")
    for key, value in dict(schema="mofit-medical-remaining-pair-contract/v1", current_step=2,
            patient_id="14393", eval_role="fit", assignment_group="A", checkpoint_step=1000,
            dtype="float32").items():
        need(c[key] == value, "contract " + key)
    need(c["model_order"] == MODELS and c["image_order"] == IMAGES, "two fixed models and remaining images")
    need(c["kernel_arguments"] == ARGUMENTS and c["expected_counts_per_image"] == COUNTS,
         "unchanged full-kernel schedule")
    need(c["expected_total_counts"] == {k: 4*v for k,v in COUNTS.items()}, "fixed full computational budget")
    need(c["expected_counts_per_model"] == {k: 2*v for k,v in COUNTS.items()}, "fixed per-model budget")
    for key in SCOPE_FLAGS:
        need(c[key] is False and e[key] is False, "bounded diagnostic scope " + key)
    frozen = c["frozen_sha256"]
    need(frozen[str(Path(__file__).resolve())] == sha(__file__), "verifier frozen before extraction")
    for filename, expected in frozen.items():
        need(sha(filename) == expected, "frozen source/input changed: " + filename)
    for filename, field in (("protocol.json", "protocol_sha256"), ("results.json", "results_sha256"),
                            ("raw.pt", "raw_sha256")):
        need(sha(directory / filename) == e[field], "output binding " + filename)
    states, exposure, image_exposures = actual_exposures()
    members = {m: int(exposure[m]["patient_training_images"] > 0) for m in MODELS}
    image_members = {m: {i: int(image_exposures[m][i] > 0) for i in IMAGES} for m in MODELS}
    for model in MODELS:
        need(Path(c["checkpoints"][model]).resolve() == checkpoint(model).resolve()
             and c["checkpoint_sha256"][model] == CHECKPOINT_SHA[model], "actual checkpoint binding " + model)
    need(c["actual_exposure"] == e["actual_exposure"] == exposure, "actual per-model patient exposure")
    need(c["member_by_model"] == members and c["image_member_by_model"] == image_members
         and c["actual_training_exposures_by_model"] == image_exposures, "actual contract membership")
    prior_executions = {}
    for model in MODELS:
        prior_dir = Path(c["prior_directories"][model])
        prior_v = read(prior_dir / "verification.json")
        prior_p = read(prior_dir / "protocol.json")
        prior_e = read(prior_dir / "execution.json")
        expected_status = ("PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE"
            if model == "model_1" else "PASS_TARGET2_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_NONMEMBERSHIP")
        need(prior_v["status"] == expected_status and prior_v["images"] == 2, "previous exact numerical verification")
        for filename, field in (("protocol.json", "protocol_sha256"), ("results.json", "results_sha256"),
                                ("raw.pt", "raw_sha256")):
            need(sha(prior_dir / filename) == prior_v[field] == prior_e[field], "previous verified input binding")
        need(prior_e["model"] == model and prior_e["before_fingerprints"] == prior_e["after_fingerprints"],
             "previous actual fixed model identity")
        prior_executions[model] = prior_e
        if model == "model_1":
            old = prior_p["contract"]
    need(sha(c["analysis_contract"]) == c["analysis_contract_sha256"], "predeclared combined analysis contract")
    need(old["inputs_sha256"] == c["inputs_sha256"] and old["model_snapshot"] == c["model_snapshot"],
         "same prepared inputs and base artifacts")
    need(old["policy"] == c["kernel_policy"], "same original numerical policy")
    alpha = float(c["alpha_cumprod_t140"])
    need(alpha == p["alpha_cumprod_t140"] == old["alpha_cumprod_t140"], "same t140 alpha")
    config = read(Path(c["model_snapshot"]) / "scheduler/scheduler_config.json")
    need(config["beta_schedule"] == "scaled_linear" and config["num_train_timesteps"] == 1000
         and config["prediction_type"] == "epsilon" and config["trained_betas"] is None, "scheduler family")
    betas = torch.linspace(config["beta_start"] ** .5, config["beta_end"] ** .5, 1000, dtype=torch.float32) ** 2
    need(float(torch.cumprod(1 - betas, 0)[140]) == alpha, "independent scheduler reconstruction")
    input_path = Path(c["input_directory"]) / "inputs.pt"
    need(sha(input_path) == c["inputs_sha256"], "same prepared input hash")
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    need(payload["policy"] == read(Path(c["input_directory"]) / "preparation_protocol.json"),
         "input preparation policy")
    need(payload["policy"]["kernel_policy"] == c["kernel_policy"], "preparation and kernel policy")
    results = read(directory / "results.json")
    raws = torch.load(directory / "raw.pt", map_location="cpu", weights_only=True)
    need([(r["model"], r["image_id"]) for r in results] == [(m,i) for m in MODELS for i in IMAGES],
         "four results in the frozen model/image order")
    need(set(raws) == set(MODELS) and all(set(raws[m]) == set(IMAGES) for m in MODELS),
         "four raw packets and no extras")
    model_reports = e["model_reports"]
    need([r["model"] for r in model_reports] == MODELS, "complete per-model execution guards")
    for mr in model_reports:
        model = mr["model"]
        need(mr["before_fingerprints"] == mr["after_fingerprints"], "unchanged model/VAE values")
        need(mr["before_fingerprints"] == prior_executions[model]["before_fingerprints"],
             "identical previous target and VAE values")
        for key in ("target_and_VAE_fingerprints_equal", "checkpoint_adapter_values_exact",
                    "adapter_tensors_exact_equal", "parameter_versions_and_no_weight_gradients"):
            need(mr[key] is True, "actual-state producer guard " + key)
        need(mr["counts"] == {k: 2*v for k,v in COUNTS.items()}, "per-model actual counts")
        need(mr["images"] == 2 and mr["actual_exposure"] == exposure[model]
             and mr["checkpoint_sha256"] == CHECKPOINT_SHA[model], "per-model actual training state")
    need(e["frozen_files_unchanged"] is True, "execution-end source/input binding")
    for key in ("target_and_VAE_fingerprints_equal", "checkpoint_adapter_values_exact",
                "adapter_tensors_exact_equal", "parameter_versions_and_no_weight_gradients"):
        need(e[key] is True, "aggregate execution guard " + key)
    need(e["patients"] == 1 and e["images"] == 4 and e["unique_images"] == 2
         and e["model_order"] == MODELS and e["mode"] == "full", "one-patient four-record scope")
    verify_packet, thash = numerical_helper()
    checks, totals = [], {k: 0 for k in COUNTS}
    for report in results:
        model, image = report["model"], report["image_id"]
        raw, item = raws[model][image], payload["images"][image]
        source, is_e = item["metadata"], image == IMAGES[0]
        need(source["patient_id"] == "14393" and source["assignment_group"] == "A"
             and source["eval_role"] == "fit", "fixed source patient")
        expected = dict(patient_id="14393", eval_role="fit", assignment_group="A",
            scenario="E" if is_e else "U", record_role="train_candidate" if is_e else "U_observed",
            member=members[model], image_member=image_members[model][image],
            actual_training_exposures=image_exposures[model][image],
            patient_training_exposures=exposure[model]["patient_training_exposures"])
        for key, value in expected.items():
            need(report[key] == value, "actual metadata " + model + "/" + key)
        need(report["record_role"] == source["record_role"], "E/U role distinct from membership")
        need(report["caption"] == item["caption"], "fixed medical caption")
        for key, value in (("initial_pixels", item["pixels"]),
                           ("initial_hidden", payload["hidden"][item["caption"]]),
                           ("null_hidden", payload["null_hidden"])):
            need(torch.equal(raw[key], value), "same original prepared tensor " + key)
        for key, value in payload["draws"].items():
            if isinstance(value, torch.Tensor):
                need(torch.equal(raw["draws"][key], value), "same frozen random draw " + key)
            else:
                need(raw["draws"][key] == value, "same frozen random policy")
        need(thash(raw["initial_pixels"]) == report["pixels_sha256"] == item["pixels_sha256"], "pixel binding")
        need(thash(raw["initial_hidden"]) == report["initial_hidden_sha256"], "hidden binding")
        need(report["policy"] == c["kernel_policy"], "unchanged per-image kernel policy")
        for key in ("stage1_steps", "stage2_steps", "nominal_stage1_steps", "mode", "monitor",
                    "fd_steps", "endpoint_diagnostics"):
            need(report[key] == ARGUMENTS[key], "unchanged numerical argument " + key)
        checked = verify_packet(report, raw, alpha)
        need(checked["counts"] == COUNTS, "per-image actual numerical counters")
        checks.append(dict(model=model, **checked))
        for key in totals:
            totals[key] += checked["counts"][key]
    need(totals == e["counts"] == c["expected_total_counts"], "total actual numerical budget")
    return dict(status="PASS_REMAINING_PAIR_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_MEMBERSHIP",
        models=MODELS, patient_id="14393", patients=1, images=4, unique_images=2, counts=totals,
        actual_exposure=exposure, image_exposures=image_exposures, patient_membership=members,
        image_membership=image_members, checkpoint_sha256=CHECKPOINT_SHA,
        original_numerical_verifier_sha256=HELPER_SHA, inputs_sha256=c["inputs_sha256"],
        frozen_files_checked=len(frozen), image_verifications=checks,
        loaded_adapter_equality_is_producer_guard_evidence=True,
        independent_neural_network_backward_replay=False,
        numerical_verification_scope="saved complete scalar traces and ten sparse updates per image; no new finite differences",
        membership_performance_claim=False, individual_patient_causal_claim=False,
        membership_label_scope="additional_NIH_training_only", pretrained_membership="unknown",
        scope_limit="one existing patient's remaining two images, no AUC or confidence interval",
        protocol_sha256=sha(directory / "protocol.json"), results_sha256=sha(directory / "results.json"),
        raw_sha256=sha(directory / "raw.pt"), execution_sha256=sha(directory / "execution.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-code-sha256", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    need(sha(__file__) == args.expected_code_sha256, "externally frozen remaining-pair verifier")
    out, vp = args.run_dir / "verification.json", args.run_dir / "verification_protocol.json"
    need(not out.exists() and not vp.exists(), "immutable verification outputs")
    started = time.perf_counter()
    new_json(vp, dict(schema="mofit-remaining-pair-independent-verification/v1",
        verifier_sha256=sha(__file__), numerical_helper_sha256=HELPER_SHA,
        input_sha256={n: sha(args.run_dir / n) for n in ("protocol.json", "execution.json", "results.json", "raw.pt")},
        actual_membership_source="both step1000 checkpoint ledgers joined to their entire training manifests",
        GPU_or_new_fitting=False))
    try:
        result = verify(args.run_dir)
    except BaseException as exc:
        result = dict(status="FAILED_REMAINING_PAIR_INDEPENDENT_VERIFICATION",
                      error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(verifier_sha256=sha(__file__), verification_protocol_sha256=sha(vp),
                      CPU_seconds=time.perf_counter() - started)
        new_json(out, result)
        print(json.dumps({k: result[k] for k in ("status", "CPU_seconds", "verifier_sha256")}), flush=True)


if __name__ == "__main__":
    main()
