"""Run the fixed stage-2, one-patient CDI feature-path/cost check.

This entry point never fits an attack or evaluates membership performance.
The external contract binds all code before any actual target invocation.
"""
import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import traceback

import torch

from .common import ROOT, RUN, PROMPT, digest, read_csv, write_json
from .models import setup, load_cache, load_unet, scheduler, adapter_state, snapshot
from .cdi_adapter import CDIAdapter, source_bindings, cpu_conformance


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes_match(expected):
    for filename, value in expected.items():
        assert digest(Path(filename)) == value, "Frozen input/code changed: " + filename


def validate(contract):
    expected = dict(current_step=2, patient_id="14393", eval_role="fit",
                    model="model_1", checkpoint_step=1000, master_seed=260914,
                    stream="primary", batch_size=1, prediction_type="epsilon",
                    dtype="float32", prompt=PROMPT, perform_fitting=False,
                    perform_performance_evaluation=False, expected_images=4,
                    code_policy="released_code_literal")
    for key, value in expected.items():
        assert contract[key] == value, key
    wanted = ["00014393_002.png", "00014393_006.png",
              "00014393_001.png", "00014393_004.png"]
    assert contract["image_order"] == wanted
    records = read_csv(RUN / "cohort/evaluation_images.csv")
    rows = {r["image_id"]: r for r in records if r["patient_id"] == "14393"}
    assert set(rows) == set(wanted)
    ordered = [rows[k] for k in wanted]
    assert all(r["eval_role"] == "fit" and r["assignment_group"] == "A" for r in ordered)
    assert [r["record_role"] for r in ordered] == ["train_candidate"] * 2 + ["U_observed"] * 2
    hashes_match(contract["frozen_sha256"])
    checkpoint = RUN / "training_coverage_v2/model_1/step_1000.pt"
    assert digest(checkpoint) == contract["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 1000
    exposures = state["exposures"]
    assert sum(exposures.values()) == 4000
    manifest = {r["image_id"] for r in read_csv(RUN / "cohort/model_1_train.csv")}
    assert set(exposures) == manifest
    for row in ordered:
        n = int(exposures.get(row["image_id"], 0))
        is_e = row["record_role"] == "train_candidate"
        assert (n > 0) == is_e and (row["image_id"] in manifest) == is_e
        row["actual_training_exposures"] = n
    return ordered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_hash = digest(args.contract)
    rows = validate(contract)
    binding = source_bindings()
    if args.dry_run:
        report = cpu_conformance()
        print(json.dumps({"status": "PASS_CDI_CPU_PREFLIGHT", "images": len(rows),
                          "conformance": report}, ensure_ascii=False, allow_nan=False))
        return
    output = RUN / "baseline_screen_20260914/cdi_kernel_v1"
    output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    protocol = dict(schema="cdi-kernel-execution/v1", current_step=2,
                    started_utc=now(), contract=contract,
                    contract_path=str(args.contract), contract_sha256=contract_hash,
                    source_bindings=binding,
                    environment={"python": platform.python_version(),
                                 **{name: version(name) for name in
                                    ("torch", "diffusers", "peft", "numpy", "scipy")}},
                    scope="CDI released-code feature-path and cost on one fit patient; no performance inference")
    write_json(output / "protocol.json", protocol)
    unet = None
    results, raw_packet = [], {}
    try:
        setup()
        cache = load_cache()
        assert cache["audit_prompt"] == PROMPT
        checkpoint = RUN / "training_coverage_v2/model_1/step_1000.pt"
        assert digest(checkpoint) == contract["checkpoint_sha256"]
        unet = load_unet(checkpoint, training=False)
        hidden = cache["hidden"][cache["audit_prompt"]]
        sched = scheduler()
        assert sched.config.prediction_type == "epsilon"
        adapter = CDIAdapter(unet, sched, hidden, master_seed=contract["master_seed"])
        before_adapter = adapter_state(unet)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        print(json.dumps(dict(event="target_loaded", seconds=time.perf_counter()-begun,
                              device=torch.cuda.get_device_name(0))), flush=True)
        for row in rows:
            hashes_match({str(args.contract): contract_hash})
            latent = cache["latents"][row["image_id"]].detach().clone()
            measured = adapter.score(latent, row["image_id"], stream=contract["stream"])
            raw_packet[row["image_id"]] = measured.pop("raw")
            metadata = dict(patient_id=row["patient_id"], image_id=row["image_id"],
                          eval_role="fit", record_role=row["record_role"],
                          scenario="E" if row["record_role"] == "train_candidate" else "U",
                          model="model_1", member=1,
                          image_member=int(row["record_role"] == "train_candidate"),
                          actual_training_exposures=row["actual_training_exposures"])
            for key in set(metadata).intersection(measured):
                assert metadata[key] == measured[key], "Adapter metadata mismatch: " + key
            result = {**metadata, **measured}
            assert len(result["features"]) == len(result["feature_names"]) == 26
            results.append(result)
            write_json(output / "results.partial.json", results)
            print(json.dumps(dict(event="image_complete", image_id=row["image_id"],
                                  forward=result["forward"], backward=result["backward"],
                                  seconds=result["seconds"]), allow_nan=False), flush=True)
        torch.cuda.synchronize()
        measurement_seconds = time.perf_counter() - measured_start
        assert adapter.assert_weights_unchanged()
        after_adapter = adapter_state(unet)
        assert set(before_adapter) == set(after_adapter)
        assert all(torch.equal(before_adapter[k], after_adapter[k]) for k in before_adapter)
        hashes_match(contract["frozen_sha256"])
        assert digest(args.contract) == contract_hash
        write_json(output / "results.json", results)
        torch.save(raw_packet, output / "raw.pt")
        report = dict(schema="cdi-kernel-execution-result/v1",
                      status="PASS_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION",
                      current_step=2, ended_utc=now(),
                      total_seconds=time.perf_counter()-begun,
                      measurement_seconds=measurement_seconds,
                      images=len(results), patients=1, model="model_1",
                      forward=sum(r["forward"] for r in results),
                      backward=sum(r["backward"] for r in results),
                      vae_forward=0, text_encoder_forward=0,
                      encoding_policy="reuse frozen FP16 posterior-mode latent/hidden cache promoted to FP32; standalone encoding cost excluded",
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      cuda_device=torch.cuda.get_device_name(0),
                      adapter_tensors_exact_equal=True,
                      parameter_versions_and_no_weight_gradients=True,
                      frozen_files_unchanged=True,
                      results_sha256=digest(output / "results.json"),
                      raw_sha256=digest(output / "raw.pt"),
                      protocol_sha256=digest(output / "protocol.json"),
                      performance_evaluation=False, attack_fitting=False,
                      claim="Feature arithmetic/path and cost only; not an identified MIA failure cause")
        write_json(output / "execution.json", report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        write_json(output / "failure.json",
                   dict(status="FAILED_INCOMPLETE_KERNEL", ended_utc=now(),
                        seconds=time.perf_counter()-begun,
                        completed_images=len(results), error=repr(exc),
                        traceback=traceback.format_exc(), performance_claim=False))
        if raw_packet:
            torch.save(raw_packet, output / "raw.partial.pt")
        raise
    finally:
        del unet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
