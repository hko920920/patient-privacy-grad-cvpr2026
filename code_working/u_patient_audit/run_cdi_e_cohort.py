"""Fixed E-only CDI feature extraction with immutable per-image CPU packets.

No attack fitting, membership performance analysis, or calibration/test
measurement is performed here. The external contract is bound before GPU use.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import traceback
import uuid

import torch

from .common import ROOT, RUN, PROMPT, digest, read_csv, write_json
from .models import setup, load_cache, load_unet, scheduler, adapter_state
from .cdi_adapter import CDIAdapter, FEATURE_NAMES, source_bindings, tensor_digest

OUTPUT = RUN / "baseline_screen_20260914/cdi_e_cohort_v1"
KERNEL = RUN / "baseline_screen_20260914/cdi_kernel_v1"
ADAPTER_SHA256 = "70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce"
REUSE_IDS = ("00014393_002.png", "00014393_006.png")
MODELS = ("model_1", "model_2")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes_match(expected):
    for filename, value in expected.items():
        assert digest(Path(filename)) == value, "Frozen input/code changed: " + filename


def _new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), "Immutable output already exists: " + str(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    temporary.rename(path)  # Windows rename refuses an existing destination.


def validate(contract):
    required = dict(schema="cdi-e-cohort-contract/v1", current_step=2,
        eval_roles=["fit", "selection"], scenario="E", models=list(MODELS),
        checkpoint_step=1000, master_seed=260914, stream="primary", batch_size=1,
        prediction_type="epsilon", dtype="float32", prompt=PROMPT,
        code_policy="released_code_literal", perform_fitting=False,
        perform_performance_evaluation=False, expected_patients={"fit": 80, "selection": 40},
        expected_unique_images=240, expected_records=480,
        expected_reused_records=2, expected_new_records=478)
    for key, value in required.items():
        assert contract[key] == value, "Contract mismatch: " + key
    frozen = {str(Path(p).resolve()): v for p, v in contract["frozen_sha256"].items()}
    adapter_path = ROOT / "u_patient_audit/cdi_adapter.py"
    assert digest(adapter_path) == ADAPTER_SHA256
    essential = [Path(__file__), adapter_path, ROOT / "u_patient_audit/common.py",
        ROOT / "u_patient_audit/models.py", RUN / "cache/cache.pt", RUN / "cache/summary.json",
        RUN / "cohort/lock.json", RUN / "cohort/evaluation_images.csv",
        RUN / "training_coverage_v2/protocol.json"]
    essential += [KERNEL / f for f in ("protocol.json", "results.json", "raw.pt", "execution.json", "verification.json")]
    essential += [RUN / "cohort" / (m + "_train.csv") for m in MODELS]
    essential += [RUN / "training_coverage_v2" / m / "step_1000.pt" for m in MODELS]
    for path in essential:
        assert str(path.resolve()) in frozen, "Missing required frozen binding: " + str(path)
    hashes_match(frozen)
    lock = load(RUN / "cohort/lock.json")
    for name, expected in lock["files"].items():
        assert digest(RUN / "cohort" / name) == expected
    evaluation = read_csv(RUN / "cohort/evaluation_images.csv")
    by_patient = defaultdict(list)
    for row in evaluation:
        by_patient[row["patient_id"]].append(row)
    order, selected = {}, []
    for role, count in (("fit", 80), ("selection", 40)):
        ids = sorted({r["patient_id"] for r in evaluation if r["eval_role"] == role}, key=int)
        assert len(ids) == count and contract["patient_order"][role] == ids
        assert Counter(by_patient[p][0]["assignment_group"] for p in ids) == {"A": count // 2, "B": count // 2}
        order[role] = ids
        for patient in ids:
            rows = by_patient[patient]
            assert len(rows) == 4 and all(r["eval_role"] == role for r in rows)
            assert len({r["assignment_group"] for r in rows}) == 1
            assert Counter(r["record_role"] for r in rows) == {"train_candidate": 2, "U_observed": 2}
            selected.extend(sorted((r for r in rows if r["record_role"] == "train_candidate"), key=lambda r: r["image_id"]))
    assert len(selected) == len({r["image_id"] for r in selected}) == 240
    assert contract["image_order"] == [r["image_id"] for r in selected]
    records = {}
    for model in MODELS:
        checkpoint = RUN / "training_coverage_v2" / model / "step_1000.pt"
        assert digest(checkpoint) == contract["checkpoint_sha256"][model]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        exposures = state["exposures"]
        manifest_rows = read_csv(RUN / "cohort" / (model + "_train.csv"))
        manifest = {r["image_id"] for r in manifest_rows}
        assert state["step"] == 1000 and len(manifest) == 912
        assert set(exposures) == manifest and sum(exposures.values()) == 4000 and min(exposures.values()) > 0
        records[model] = []
        for row in selected:
            patient = row["patient_id"]
            member = int(row["assignment_group"] == ("A" if model == "model_1" else "B"))
            candidate = [r["image_id"] for r in by_patient[patient] if r["record_role"] == "train_candidate"]
            actual = [int(exposures.get(i, 0)) for i in candidate]
            assert len(candidate) == 2 and all((n > 0) == bool(member) for n in actual)
            assert all((i in manifest) == bool(member) for i in candidate)
            assert (row["image_id"] in manifest) == bool(member)
            assert (exposures.get(row["image_id"], 0) > 0) == bool(member)
            records[model].append(dict(model=model, scenario="E", patient_id=patient,
                image_id=row["image_id"], eval_role=row["eval_role"], assignment_group=row["assignment_group"],
                record_role="train_candidate", member=member, image_member=member,
                actual_training_exposures=int(exposures.get(row["image_id"], 0)),
                patient_training_exposures=sum(actual), checkpoint_sha256=contract["checkpoint_sha256"][model]))
        del state
    return {"records": records, "frozen": frozen, "patient_order": order,
            "source_bindings": source_bindings()}


def prepare_reuse(contract, context, cache, sched):
    execution, protocol = load(KERNEL / "execution.json"), load(KERNEL / "protocol.json")
    verification = load(KERNEL / "verification.json")
    assert execution["status"] == "PASS_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION"
    assert verification["status"] == "PASS_SAVED_KERNEL_ARITHMETIC_AND_INTEGRITY"
    assert execution["adapter_tensors_exact_equal"] and execution["parameter_versions_and_no_weight_gradients"]
    assert execution["frozen_files_unchanged"] and execution["images"] == 4
    for name, field in (("results.json", "results_sha256"), ("raw.pt", "raw_sha256"), ("protocol.json", "protocol_sha256")):
        assert digest(KERNEL / name) == execution[field]
    original = protocol["contract"]
    for key in ("master_seed", "stream", "batch_size", "prediction_type", "dtype", "prompt", "code_policy"):
        assert original[key] == contract[key]
    assert original["checkpoint_sha256"] == contract["checkpoint_sha256"]["model_1"]
    assert protocol["source_bindings"] == context["source_bindings"]
    old_contract = Path(protocol["contract_path"])
    if not old_contract.is_absolute():
        old_contract = ROOT / old_contract
    assert digest(old_contract) == protocol["contract_sha256"]
    assert str(old_contract.resolve()) in context["frozen"]
    for path in (RUN / "cache/cache.pt", RUN / "cache/summary.json", RUN / "cohort/lock.json",
                 ROOT / "u_patient_audit/cdi_adapter.py"):
        assert original["frozen_sha256"][str(path)] == context["frozen"][str(path.resolve())]
    rows = {r["image_id"]: r for r in load(KERNEL / "results.json")}
    packets = torch.load(KERNEL / "raw.pt", map_location="cpu", weights_only=True)
    known = {r["image_id"]: r for r in context["records"]["model_1"]}
    reuse = {}
    hidden_hash = tensor_digest(cache["hidden"][cache["audit_prompt"]].float())
    alphas = sched.alphas_cumprod.detach().cpu().float()
    for iid in REUSE_IDS:
        row, raw = rows[iid], packets[iid]
        assert all(row[k] == known[iid][k] for k in ("model", "scenario", "patient_id", "eval_role", "record_role",
                                                       "member", "image_member", "actual_training_exposures"))
        assert row["stream"] == contract["stream"] and row["feature_names"] == FEATURE_NAMES
        assert raw["image_id"] == iid and raw["stream"] == contract["stream"]
        assert torch.equal(raw["latent"], cache["latents"][iid].float())
        assert torch.equal(raw["alphas"], alphas) and raw["hidden_sha256"] == hidden_hash
        reuse[iid] = (row, raw)
    assert len(reuse) == 2
    provenance = {"directory": str(KERNEL), "protocol_sha256": execution["protocol_sha256"],
                  "results_sha256": execution["results_sha256"], "raw_sha256": execution["raw_sha256"],
                  "execution_sha256": digest(KERNEL / "execution.json"),
                  "verification_sha256": digest(KERNEL / "verification.json")}
    return reuse, provenance


def metadata(row, contract_hash, cache_hash, lock_hash):
    return {**row, "contract_sha256": contract_hash, "cache_sha256": cache_hash,
            "cohort_lock_sha256": lock_hash}


def pair_paths(row):
    stem = Path(row["image_id"]).stem
    assert Path(row["image_id"]).name == row["image_id"] and stem
    return OUTPUT / row["model"] / (stem + ".json"), OUTPUT / row["model"] / (stem + ".pt")


def check_pair(row, expected, cache, alphas, hidden_hash):
    json_path, raw_path = pair_paths(expected)
    assert json_path.is_file() and raw_path.is_file(), "Incomplete immutable JSON/PT pair; preserve and review: " + str(json_path)
    for key, value in expected.items():
        assert row[key] == value, "Saved metadata mismatch: " + key
    assert row["raw_path"] == raw_path.relative_to(OUTPUT).as_posix()
    assert digest(raw_path) == row["raw_sha256"]
    assert row["feature_names"] == FEATURE_NAMES and len(row["features"]) == 26 and row["weights_unchanged"]
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    assert raw["image_id"] == row["image_id"] and raw["stream"] == row["stream"] == "primary"
    assert torch.equal(raw["latent"], cache["latents"][row["image_id"]].float())
    assert torch.equal(raw["alphas"], alphas) and raw["hidden_sha256"] == hidden_hash
    q = raw["no"]["optimizer"]["objective_evaluations"]
    assert q == len(raw["no"]["objective_trace"])
    assert row["forward"] == 51 + q and row["backward"] == 10 + q
    assert row["modules"]["noise_optim"]["optimizer"]["objective_evaluations"] == q
    assert row["reused_from_kernel"] == (row["model"] == "model_1" and row["image_id"] in REUSE_IDS)
    return row


def save_pair(expected, measured, raw, reused, provenance):
    json_path, raw_path = pair_paths(expected)
    assert not json_path.exists() and not raw_path.exists(), "Refusing to overwrite immutable image pair"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("xb") as f:
        torch.save(raw, f)
    raw_hash = digest(temporary)
    temporary.rename(raw_path)
    for key in set(expected).intersection(measured):
        assert measured[key] == expected[key], "Measured metadata mismatch: " + key
    row = {**measured, **expected, "raw_path": raw_path.relative_to(OUTPUT).as_posix(),
           "raw_sha256": raw_hash, "reused_from_kernel": bool(reused),
           "reuse_provenance": provenance if reused else None}
    _new_json(json_path, row)
    return row


def aggregate_cost(rows):
    return {"records": len(rows), "forward": sum(r["forward"] for r in rows),
            "backward": sum(r["backward"] for r in rows),
            "score_seconds": sum(r["seconds"] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    contract = load(args.contract)
    contract_hash = digest(args.contract)
    context = validate(contract)
    cache = load_cache()
    assert cache["audit_prompt"] == PROMPT
    sched = scheduler()
    assert sched.config.prediction_type == "epsilon"
    alphas = sched.alphas_cumprod.detach().cpu().float()
    reuse, provenance = prepare_reuse(contract, context, cache, sched)
    cache_hash = context["frozen"][str((RUN / "cache/cache.pt").resolve())]
    lock_hash = context["frozen"][str((RUN / "cohort/lock.json").resolve())]
    hidden_hash = tensor_digest(cache["hidden"][cache["audit_prompt"]].float())
    if args.dry_run:
        assert not args.resume
        print(json.dumps({"status": "PASS_CPU_COHORT_PREFLIGHT", "records": 480,
                          "patients": 120, "unique_images": 240, "reused_records": 2,
                          "new_records": 478, "roles": {"fit": 80, "selection": 40},
                          "model_2_reuse_patient_membership": next(r["member"] for r in context["records"]["model_2"] if r["patient_id"] == "14393"),
                          "GPU_executions": 0, "contract_sha256": contract_hash}, ensure_ascii=False))
        return
    if OUTPUT.exists():
        assert args.resume, "Output exists; use --resume only for validated immutable pairs"
        existing_protocol = load(OUTPUT / "protocol.json")
        assert existing_protocol["contract_sha256"] == contract_hash and existing_protocol["contract"] == contract
        assert existing_protocol["source_bindings"] == context["source_bindings"]
    else:
        assert not args.resume, "No existing run to resume"
        OUTPUT.mkdir(parents=True, exist_ok=False)
        _new_json(OUTPUT / "protocol.json", {
            "schema": "cdi-e-cohort-execution/v1", "current_step": 2, "started_utc": now(),
            "contract_path": str(args.contract), "contract_sha256": contract_hash, "contract": contract,
            "source_bindings": context["source_bindings"], "measurement_plan": context["records"],
            "environment": {"python": platform.python_version(), **{n: version(n) for n in ("torch", "diffusers", "peft", "numpy", "scipy")}},
            "scope": "fixed E-only fit80 and previously observed selection40; feature extraction, not attack fitting",
        })
    protocol_hash = digest(OUTPUT / "protocol.json")
    existing = {}
    for model in MODELS:
        for row in context["records"][model]:
            expected = metadata(row, contract_hash, cache_hash, lock_hash)
            jp, rp = pair_paths(row)
            assert jp.exists() == rp.exists(), "Dangling immutable image pair; preserve and review: " + str(jp)
            if jp.exists():
                existing[(model, row["image_id"])] = check_pair(load(jp), expected, cache, alphas, hidden_hash)
    assert len(existing) <= 480
    if (OUTPUT / "execution.json").exists():
        previous = load(OUTPUT / "execution.json")
        assert previous["complete"] and len(existing) == 480 and previous["records"] == 480
        assert digest(OUTPUT / "results.json") == previous["results_sha256"]
        assert previous["protocol_sha256"] == protocol_hash
        print(json.dumps({"status": "ALREADY_COMPLETE_VALIDATED_NO_GPU", "records": 480}))
        return
    attempt = OUTPUT / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    _new_json(attempt / "start.json", {"started_utc": now(), "resume": args.resume,
              "preexisting_records": len(existing), "contract_sha256": contract_hash})
    results, model_reports, current_rows = [], [], []
    unet = adapter = None
    begun = time.perf_counter()
    cheap_bindings = {str(args.contract): contract_hash}
    cheap_bindings.update({p: h for p, h in context["frozen"].items() if Path(p).suffix == ".py"})
    try:
        setup()
        for model in MODELS:
            hashes_match(context["frozen"])
            hashes_match({str(args.contract): contract_hash})
            model_start = time.perf_counter()
            planned = context["records"][model]
            missing = [r for r in planned if (model, r["image_id"]) not in existing
                       and not (model == "model_1" and r["image_id"] in REUSE_IDS)]
            before_adapter = None
            model_new = []
            if missing:
                unet = load_unet(RUN / "training_coverage_v2" / model / "step_1000.pt", training=False)
                adapter = CDIAdapter(unet, sched, cache["hidden"][cache["audit_prompt"]], contract["master_seed"])
                before_adapter = adapter_state(unet)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                print(json.dumps({"event": "target_loaded", "model": model, "remaining_new_images": len(missing),
                                  "seconds": time.perf_counter() - begun}), flush=True)
            for source_row in planned:
                iid = source_row["image_id"]
                expected = metadata(source_row, contract_hash, cache_hash, lock_hash)
                if (model, iid) in existing:
                    saved = existing[(model, iid)]
                elif model == "model_1" and iid in REUSE_IDS:
                    measured, raw = reuse[iid]
                    saved = save_pair(expected, measured, raw, True, {**provenance, "image_id": iid})
                else:
                    measured = adapter.score(cache["latents"][iid], iid, stream=contract["stream"])
                    raw = measured.pop("raw")
                    assert torch.equal(raw["alphas"], alphas) and raw["hidden_sha256"] == hidden_hash
                    saved = save_pair(expected, measured, raw, False, None)
                    model_new.append(saved)
                    current_rows.append(saved)
                    del raw, measured
                results.append(saved)
                if len(results) % 10 == 0 or len(results) == 480:
                    hashes_match(cheap_bindings)
                    progress = {"event": "progress", "complete": False, "records": len(results), "expected_records": 480,
                                "current_attempt_new_records": len(current_rows), "model": model,
                                "seconds": time.perf_counter() - begun, "contract_sha256": contract_hash}
                    write_json(OUTPUT / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            exact = None
            peak = None
            if adapter is not None:
                torch.cuda.synchronize()
                assert adapter.assert_weights_unchanged()
                after = adapter_state(unet)
                assert set(after) == set(before_adapter)
                assert all(torch.equal(after[n], before_adapter[n]) for n in before_adapter)
                exact = True
                peak = torch.cuda.max_memory_allocated()
                adapter._forward_handle.remove()
                del adapter, unet, before_adapter, after
                adapter = unet = None
                gc.collect()
                torch.cuda.empty_cache()
            hashes_match(context["frozen"])
            report = {"model": model, "records": 240,
                      "resumed_records": sum((model, r["image_id"]) in existing for r in planned),
                      "new_this_attempt": aggregate_cost(model_new), "current_session_adapter_values_exact_equal": exact,
                      "all_records_parameter_version_guards": True, "peak_cuda_allocated_bytes": peak,
                      "seconds": time.perf_counter() - model_start}
            model_reports.append(report)
            _new_json(attempt / (model + ".json"), report)
        assert len(results) == len({(r["model"], r["image_id"]) for r in results}) == 480
        assert [r["image_id"] for r in results] == contract["image_order"] * 2
        assert Counter(r["eval_role"] for r in results) == {"fit": 320, "selection": 160}
        reused = [r for r in results if r["reused_from_kernel"]]
        fresh = [r for r in results if not r["reused_from_kernel"]]
        assert len(reused) == 2 and len(fresh) == 478
        assert all(r["record_role"] == "train_candidate" and r["image_member"] == r["member"]
                   and (r["actual_training_exposures"] > 0) == bool(r["member"]) for r in results)
        # Require every pair and raw byte hash at completion; no partial success.
        for row in results:
            jp, rp = pair_paths(row)
            assert jp.is_file() and rp.is_file() and digest(rp) == row["raw_sha256"]
        hashes_match(context["frozen"])
        hashes_match({str(args.contract): contract_hash})
        total, rcost, ncost, current = map(aggregate_cost, (results, reused, fresh, current_rows))
        _new_json(OUTPUT / "results.json", results)
        summary = {"schema": "cdi-e-cohort-execution-result/v1",
            "status": "PASS_COHORT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION", "complete": True,
            "current_step": 2, "ended_utc": now(), "records": 480, "unique_patients": 120, "unique_images": 240,
            "reused_records": 2, "new_records": 478, "forward": total["forward"], "backward": total["backward"],
            "reused_forward": rcost["forward"], "reused_backward": rcost["backward"],
            "new_forward": ncost["forward"], "new_backward": ncost["backward"],
            "current_attempt_forward": current["forward"], "current_attempt_backward": current["backward"],
            "current_attempt_new_records": len(current_rows), "current_attempt_seconds": time.perf_counter() - begun,
            "full_score_seconds_including_reuse": total["score_seconds"],
            "new_score_seconds_excluding_kernel": ncost["score_seconds"],
            "NO_objective_evaluations": sum(r["modules"]["noise_optim"]["optimizer"]["objective_evaluations"] for r in results),
            "NO_nfev": sum(r["modules"]["noise_optim"]["optimizer"]["nfev"] for r in results),
            "NO_termination_status_counts": dict(Counter(str(r["modules"]["noise_optim"]["optimizer"]["status"]) for r in results)),
            "model_reports": model_reports, "attempt_path": attempt.relative_to(OUTPUT).as_posix(),
            "protocol_sha256": protocol_hash, "contract_sha256": contract_hash,
            "results_sha256": digest(OUTPUT / "results.json"),
            "frozen_files_unchanged": True, "all_records_parameter_version_guards": True,
            "vae_forward": 0, "text_encoder_forward": 0, "attack_fitting": False, "performance_evaluation": False,
            "scope": "complete fixed E extraction; selection is existing development data; no efficacy or failure-cause conclusion"}
        _new_json(OUTPUT / "execution.json", summary)
        _new_json(attempt / "completion.json", summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        _new_json(attempt / "failure.json", {"status": "FAILED_INCOMPLETE_COHORT", "complete": False,
            "ended_utc": now(), "completed_in_iteration": len(results), "new_this_attempt": len(current_rows),
            "seconds": time.perf_counter() - begun, "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if adapter is not None:
            adapter._forward_handle.remove()
        adapter = unet = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

