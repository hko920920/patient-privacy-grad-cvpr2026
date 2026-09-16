"""Independent CPU residual, provenance and statistics audit for fixed PFAMI adaptation.

No target inference, fitting, training, or imports of executed producer modules.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .common import ROOT, RUN, digest, read_csv, verify_inputs, write_json

MODELS = ("model_1", "model_2")
ALL_MODELS = MODELS + ("base",)
TIMESTEPS = tuple(range(0, 500, 50))
REPLICATES = 2000
SEED = 260914
MSE_RTOL = 2e-6
MSE_ATOL = 1e-8
SOURCE_SHA = "3bf2f4f28fbcfd3c4b4291a932e58a68332da3920c8fbc49fcde6e5414d83117"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected, *, rtol=1e-10, atol=1e-12, label=""):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), label
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (label, actual, expected)


def inside(parent, name):
    parent = Path(parent).resolve()
    result = (parent / name).resolve()
    assert result.is_relative_to(parent), (parent, name)
    return result


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def independent_noise(image_id, view, timestep, stream, master=SEED):
    payload = "|".join(("pfami-v1", str(master), image_id, view, str(timestep), stream))
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63-1)
    value = torch.randn((1, 4, 32, 32), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    return seed, value


def recompute_image(row, residual):
    assert residual.dtype == torch.float32 and tuple(residual.shape) == (10, 2, 4, 32, 32)
    assert residual.device.type == "cpu" and bool(torch.isfinite(residual).all())
    cells = row["cells"]
    assert len(cells) == 10 and tuple(c["timestep"] for c in cells) == TIMESTEPS
    assert row["forward"] == 20 and row["backward"] == 0 and row["finite"] is True
    mse64 = residual.double().square().mean(dim=(2, 3, 4)).tolist()
    stream = "base" if row["model"] == "base" else "target"
    if "stream" in row:
        assert row["stream"] == stream
    errors = []
    for index, cell in enumerate(cells):
        for vi, view in enumerate(("original", "crop")):
            observed = cell[view + "_loss"]
            expected = mse64[index][vi]
            assert observed > 0 and expected > 0
            close(observed, expected, rtol=MSE_RTOL, atol=MSE_ATOL, label=(row["model"], row["image_id"], index, view))
            errors.append(dict(model=row["model"], image_id=row["image_id"], timestep=cell["timestep"], view=view,
                stored_loss=observed, CPU_float64_MSE=expected, difference=observed-expected,
                relative_difference=(observed-expected)/expected))
            seed, noise = independent_noise(row["image_id"], view, cell["timestep"], stream)
            assert cell[view + "_noise_seed"] == seed
            assert cell[view + "_noise_sha256"] == tensor_sha(noise)
            close(cell[view + "_noise_l2"], float(noise.norm()), rtol=2e-6, atol=1e-6)
            assert math.isfinite(cell[view + "_prediction_l2"]) and cell[view + "_prediction_l2"] >= 0
            for suffix in ("_cuda_ms", "_wall_seconds"):
                assert math.isfinite(cell[view + suffix]) and cell[view + suffix] >= 0
        relative = (cell["crop_loss"]-cell["original_loss"])/cell["original_loss"]
        close(cell["relative"], relative)
        assert cell["finite"] is True
        assert cell["original_noise_seed"] != cell["crop_noise_seed"]
    stored = dict(
        relative=math.fsum(c["relative"] for c in cells)/10,
        difference=math.fsum(c["crop_loss"]-c["original_loss"] for c in cells)/10,
        negative_loss=-math.fsum(c["original_loss"] for c in cells)/10)
    recomputed = dict(
        relative=math.fsum((crop-original)/original for original, crop in mse64)/10,
        difference=math.fsum(crop-original for original, crop in mse64)/10,
        negative_loss=-math.fsum(original for original, crop in mse64)/10)
    close(row["score"], stored["relative"])
    return stored, recomputed, errors


def percentile(values, q):
    ordered = sorted(map(float, values))
    position = q*(len(ordered)-1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (position-lo)*(ordered[hi]-ordered[lo])


def independent_statistic_tables(records, order):
    """Use sklearn points and multiplicity-weighted pair comparisons for bootstrap."""
    assert len(order) == len(set(order)) == 40
    rng = np.random.Generator(np.random.PCG64(SEED))
    a = rng.integers(0, 20, size=(REPLICATES, 20), dtype=np.int64)
    b = rng.integers(20, 40, size=(REPLICATES, 20), dtype=np.int64)
    draws = np.concatenate((a, b), axis=1)
    wa = np.stack([np.bincount(row, minlength=20) for row in a]).astype(np.float64)
    wb = np.stack([np.bincount(row-20, minlength=20) for row in b]).astype(np.float64)
    by_key = {(r["model"], r["scenario"], str(r["patient_id"])): r for r in records}
    assert len(by_key) == len(records) == 160
    tables, boot_values = {}, {}
    score_names = tuple(records[0]["scores"])
    for model in MODELS:
        for scenario in ("E", "U"):
            selected = [by_key[model, scenario, patient] for patient in order]
            assert all(r["assignment_group"] == "A" for r in selected[:20])
            assert all(r["assignment_group"] == "B" for r in selected[20:])
            labels = np.array([r["member"] for r in selected], dtype=np.int64)
            assert labels[:20].tolist() == [int(model == "model_1")]*20
            for score in score_names:
                values = np.array([r["scores"][score] for r in selected], dtype=np.float64)
                point = float(roc_auc_score(labels, values))
                pair = (values[:20, None] > values[None, 20:]).astype(float) + .5*(values[:20, None] == values[None, 20:])
                if model == "model_2":
                    pair = 1-pair
                boot = np.einsum("bi,ij,bj->b", wa, pair, wb, optimize=True)/400
                key = (model, scenario, score)
                boot_values[key] = boot
                tables[key] = dict(auc=point, ci95=[percentile(boot, .025), percentile(boot, .975)],
                    member_mean=math.fsum(values[labels == 1])/20, nonmember_mean=math.fsum(values[labels == 0])/20)
    return tables, boot_values, hashlib.sha256(draws.tobytes()).hexdigest()


def cohort_and_exposures(protocol):
    verify_inputs()
    manifest = read_csv(RUN / "cohort/evaluation_images.csv")
    selected = [r for r in manifest if r["eval_role"] == "selection"]
    by_image = {r["image_id"]: r for r in selected}
    by_patient = defaultdict(list)
    for row in selected:
        by_patient[str(row["patient_id"])].append(row)
    assert len(by_patient) == 40 and len(by_image) == len(selected) == 160
    assert Counter(rows[0]["assignment_group"] for rows in by_patient.values()) == {"A": 20, "B": 20}
    for patient, rows in by_patient.items():
        assert len(rows) == 4
        assert Counter(r["record_role"] for r in rows) == {"train_candidate": 2, "U_observed": 2}
        assert len({r["assignment_group"] for r in rows}) == 1
    exposure = {}
    for model in MODELS:
        path = RUN / "training_coverage_v2" / model / "step_1000.pt"
        assert digest(path) == protocol["checkpoint_sha256"][model]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 1000
        assert checkpoint["contract"] == load(RUN / "training_coverage_v2/protocol.json")
        exposure[model] = dict(checkpoint["exposures"])
        training = {r["image_id"] for r in read_csv(RUN / "cohort" / (model+"_train.csv"))}
        assert set(exposure[model]) == training and len(training) == 912
        assert sum(exposure[model].values()) == 4000
        assert set(exposure[model].values()) <= {4, 5}
        for patient, rows in by_patient.items():
            member = int(rows[0]["assignment_group"] == ("A" if model == "model_1" else "B"))
            for row in rows:
                actual = exposure[model].get(row["image_id"], 0)
                assert bool(actual) == bool(member and row["record_role"] == "train_candidate")
        del checkpoint
    return by_image, by_patient, exposure


def verify_image_identity(row, by_image, exposure, protocol):
    model = row["model"]
    assert model in ALL_MODELS and row["scenario"] in ("E", "U")
    source = by_image[row["image_id"]]
    for key in ("patient_id", "eval_role", "assignment_group", "record_role"):
        assert row[key] == source[key], (key, row["image_id"])
    assert row["eval_role"] == "selection"
    assert row["record_role"] == ("train_candidate" if row["scenario"] == "E" else "U_observed")
    if model == "base":
        assert row["member"] is None and row["image_member"] is None
        assert row["actual_training_exposures"] is None
    else:
        expected = int(row["assignment_group"] == ("A" if model == "model_1" else "B"))
        image_member = int(expected and row["scenario"] == "E")
        assert row["member"] == expected and row["image_member"] == image_member
        assert row["actual_training_exposures"] == exposure[model].get(row["image_id"], 0)
        assert row["checkpoint_sha256"] == protocol["checkpoint_sha256"][model]


def reconstruct_patients(images, image_scores):
    groups = defaultdict(list)
    for row in images:
        key = row["model"], row["scenario"], str(row["patient_id"])
        groups[key].append(row)
    assert len(groups) == 240 and all(len(rows) == 2 for rows in groups.values())
    reconstructed = []
    base_rows = {}
    for (model, scenario, patient), rows in groups.items():
        if model == "base":
            base_rows[scenario, patient] = math.fsum(image_scores[model, r["image_id"]]["relative"] for r in rows)/2
    for (model, scenario, patient), rows in groups.items():
        if model == "base":
            continue
        scores = {field: math.fsum(image_scores[model, r["image_id"]][field] for r in rows)/2
                  for field in ("relative", "difference", "negative_loss")}
        scores["base_relative"] = math.fsum(image_scores[model, r["image_id"]]["relative"]-
            image_scores["base", r["image_id"]]["relative"] for r in rows)/2
        reconstructed.append(dict(model=model, scenario=scenario, eval_role="selection", patient_id=patient,
            assignment_group=rows[0]["assignment_group"], member=rows[0]["member"],
            image_ids=sorted(r["image_id"] for r in rows), scores=scores))
    assert len(reconstructed) == 160
    return reconstructed


def verify_statistics(directory, reconstructed, double_reconstructed):
    analysis = load(directory / "analysis.json")
    assert analysis["schema"] == "pfami-style-development-analysis/v1"
    assert analysis["input_image_scores_sha256"] == digest(directory / "image_scores.json")
    assert analysis["protocol_sha256"] == digest(directory / "protocol.json")
    assert analysis["analysis_code_sha256"] == digest(Path(__file__).with_name("pfami_analysis.py"))
    declared = analysis["bootstrap"]
    for key, expected in dict(seed=SEED, replicates=REPLICATES, unit="patient", strata={"A": 20, "B": 20},
                              dtype="int64", shared_across_models_scenarios_scores=True).items():
        assert declared[key] == expected, key
    order = declared["patient_order"]
    expected_order = []
    for group in ("A", "B"):
        expected_order += sorted({r["patient_id"] for r in reconstructed if r["assignment_group"] == group}, key=int)
    assert order == expected_order
    rows = {(r["model"], r["scenario"], r["patient_id"]): r for r in reconstructed}
    reported_patients = analysis["patient_scores"]
    assert len(reported_patients) == len(rows) == 160
    seen = set()
    for actual in reported_patients:
        key = actual["model"], actual["scenario"], str(actual["patient_id"])
        assert key not in seen
        seen.add(key)
        expected = rows[key]
        for field in ("eval_role", "assignment_group", "member", "image_ids"):
            assert actual[field] == expected[field]
        assert set(actual["scores"]) == set(expected["scores"])
        for field, value in expected["scores"].items():
            close(actual["scores"][field], value, label=(key, field))
    stats, boots, drawsha = independent_statistic_tables(reconstructed, order)
    assert drawsha == declared["indices_sha256"]
    double_stats, double_boots, doublesha = independent_statistic_tables(double_reconstructed, order)
    assert doublesha == drawsha
    assert len(analysis["metrics"]) == len(stats) == 16
    seen = set()
    numerical_sensitivity = []
    for metric in analysis["metrics"]:
        key = metric["model"], metric["scenario"], metric["score"]
        assert key not in seen
        seen.add(key)
        expected = stats[key]
        for field in ("auc", "member_mean", "nonmember_mean"):
            close(metric[field], expected[field], label=(key, field))
        for actual, value in zip(metric["patient_bootstrap95"], expected["ci95"]):
            close(actual, value, label=(key, "bootstrap95"))
        assert metric["patients"] == 40 and metric["members"] == metric["nonmembers"] == 20
        numerical_sensitivity.append(dict(model=key[0], scenario=key[1], score=key[2],
            stored_loss_auc=expected["auc"], residual_float64_auc=double_stats[key]["auc"],
            AUC_changed=expected["auc"] != double_stats[key]["auc"],
            stored_loss_bootstrap95=expected["ci95"], residual_float64_bootstrap95=double_stats[key]["ci95"]))
    assert len(analysis["contrasts"]) == 8
    primary = []
    contrast_sensitivity = []
    seen = set()
    for contrast in analysis["contrasts"]:
        model, scenario, name = contrast["model"], contrast["scenario"], contrast["contrast"]
        assert (model, scenario, name) not in seen
        seen.add((model, scenario, name))
        assert name in ("relative_minus_difference", "relative_minus_negative_loss")
        comparator = name.removeprefix("relative_minus_")
        key, other = (model, scenario, "relative"), (model, scenario, comparator)
        point = stats[key]["auc"]-stats[other]["auc"]
        diff = boots[key]-boots[other]
        ci = [percentile(diff, q) for q in (.025, .975)]
        close(contrast["delta_auc"], point)
        for actual, expected in zip(contrast["patient_bootstrap95"], ci):
            close(actual, expected)
        is_primary = scenario == "U" and comparator == "difference"
        assert contrast["primary"] == is_primary
        assert contrast["raw_forward_budget_matched"] == (comparator == "difference")
        if is_primary:
            primary.append(dict(point=point, ci95=ci))
        double_diff = double_boots[key]-double_boots[other]
        contrast_sensitivity.append(dict(model=model, scenario=scenario, contrast=name,
            stored_loss_delta_auc=point, residual_float64_delta_auc=double_stats[key]["auc"]-double_stats[other]["auc"],
            stored_loss_bootstrap95=ci, residual_float64_bootstrap95=[percentile(double_diff, q) for q in (.025, .975)]))
    assert len(primary) == 2
    if all(p["ci95"][0] > 0 for p in primary):
        interpretation = "RELATIVE_GAIN_OBSERVED_IN_THIS_DEVELOPMENT_DATA"
    elif all(p["point"] <= 0 for p in primary):
        interpretation = "NO_POSITIVE_POINT_ESTIMATE_GAIN_FROM_NORMALIZATION_IN_THIS_SETTING"
    else:
        interpretation = "ADDITIONAL_RELATIVE_GAIN_NOT_ESTABLISHED"
    assert analysis["interpretation"] == interpretation
    for field in ("admission_gate_created", "followup_authorized_by_analysis", "automatic_followup",
                  "low_fpr_utility_claimed", "novel_method_success_claimed"):
        assert analysis[field] is False
    assert analysis["primary_scenario"] == "U" and analysis["primary_score"] == "relative"
    assert analysis["primary_contrast"] == "relative_minus_difference"
    assert analysis["useful_identification_must_consider_absolute_auc_levels"] is True
    return dict(auc_checks=16, patient_scores_checked=160, patient_score_values_checked=640,
        paired_contrast_checks=8, bootstrap_replicates=REPLICATES, indices_sha256=drawsha,
        point_AUC_method="independent sklearn.roc_auc_score",
        bootstrap_method="independent A/B sample-count weighted 20x20 pair matrices; manual linear percentiles",
        interpretation_checked=interpretation, metrics=numerical_sensitivity, contrasts=contrast_sensitivity)


def verify_provenance(protocol):
    for filename, expected in protocol["input_sha256"].items():
        assert digest(inside(ROOT.parent, filename)) == expected, filename
    for filename, expected in protocol["code_sha256"].items():
        assert digest(inside(Path(__file__).parent, filename)) == expected, filename
    assert {"run_pfami_diagnostic.py", "pfami_adapter.py", "models.py", "common.py"} <= set(protocol["code_sha256"])
    protected = protocol["original_protected_sha256"]
    assert len(protected) == 11
    assert protected == load(RUN / "verification_20260914/foundation/report.json")["protected_sha256"]
    for filename, expected in protected.items():
        assert digest(inside(ROOT, filename)) == expected, filename
    source = protocol["source"]
    assert source["repository"] == "https://github.com/wjfu99/MIA-Gen"
    assert source["commit"] == "f80df339e11ff49db211a455c0a455d630cf34a7"
    assert source["source_sha256"] == SOURCE_SHA
    assert digest(Path(source["source_path"])) == SOURCE_SHA
    assert source["methods_reused"] == ["AttackModel.ddpm_loss", "AttackModel.feat_prepare"]
    assert source["source_member_label"] == 0 and source["local_member_label"] == 1
    assert source["local_score"] == "positive mean relative fluctuation"
    assert digest(RUN / "cache/cache.pt") == protocol["cache_sha256"]
    summary = load(RUN / "cache/summary.json")
    assert summary["cache_sha256"] == protocol["cache_sha256"]
    assert summary["cohort_lock_sha256"] == digest(RUN / "cohort/lock.json")
    return source, protected


def verify_execution_declarations(directory, protocol, report):
    assert protocol["schema"] == "u-pfami-style-execution/v1" and protocol["mode"] == "full"
    assert protocol["benchmark_patient"] is None
    assert protocol["models"] == list(ALL_MODELS)
    assert protocol["expected_image_records"] == 480 and protocol["expected_forward"] == 9600
    assert protocol["expected_backward"] == 0
    assert protocol["residual_shape"] == [10, 2, 4, 32, 32] and protocol["residual_dtype"] == "float32"
    assert protocol["residual_axes"] == ["timestep", "view(original,crop)", "channel", "height", "width"]
    contract = protocol["contract"]
    expected_contract = dict(schema="u-pfami-style-diagnostic-contract/v1", timesteps=list(TIMESTEPS),
        crop_strength=73/90, crop_size=207, crop_interpolation="bilinear", crop_antialias=True,
        noise_seed=SEED, noise_policy="independent_view_t_shared_models", target_noise_stream="target",
        base_noise_stream="base", original_latent_policy="verified_fp16_cache", crop_vae_policy="fp16_variant_mode_batch4",
        prediction_type="epsilon", generic_prompt="a frontal chest radiograph", roles=["selection"],
        models=list(MODELS), scenarios=["E", "U"], include_base=True, benchmark_role="fit",
        benchmark_excluded_from_analysis=True,
        primary_image_score="mean_t((L_crop-L_original)/L_original); high=member",
        primary_patient_score="arithmetic mean of two primary image scores",
        primary_comparator_image_score="mean_t(L_crop-L_original); high=member",
        primary_comparator_patient_score="arithmetic mean of two comparator image scores",
        score_fields=["relative", "difference", "negative_loss", "base_relative"])
    for key, value in expected_contract.items():
        assert contract[key] == value, key
    assert contract["denominator_policy"] == "no epsilon/clipping; nonpositive or nonfinite original loss is an execution error; report minimum"
    assert contract["bootstrap"]["replicates"] == REPLICATES and contract["bootstrap"]["seed"] == SEED
    assert contract["bootstrap"]["strata"] == {"A": 20, "B": 20}
    assert contract["bootstrap"]["draws_shared_across_models_scenarios_scores"] is True
    assert contract["source"]["stat_crop_index"] == 5 and contract["source"]["chosen_calibration"] is False
    assert contract["interpretation_policy"]["primary_scenario"] == "U"
    for key in ("no_new_admission_gate", "no_confirmatory_significance", "selection_previously_observed", "no_automatic_followup"):
        assert contract["interpretation_policy"][key] is True
    expected_counts = dict(selection_patients=40, unique_images=160, target_image_records=320,
        base_image_records=160, total_image_records=480, patient_target_records=160, target_forward=6400,
        base_forward=3200, total_forward=9600, total_backward=0, crop_vae_images=160, benchmark_counted_separately=True)
    assert contract["expected"] == expected_counts
    matching = [Path(path) for path, sha in protocol["input_sha256"].items() if sha == protocol["contract_sha256"]]
    assert len(matching) == 1 and load(matching[0]) == contract
    selection = [r for r in read_csv(RUN / "cohort/evaluation_images.csv") if r["eval_role"] == "selection"]
    patients = sorted({r["patient_id"] for r in selection}, key=int)
    assert len(patients) == 40 and protocol["measurement_patient_ids"] == patients
    assert contract["patient_order_by_role"] == {"selection": patients}
    assert contract["benchmark_patient"] not in patients
    order = []
    for patient in patients:
        for role in ("train_candidate", "U_observed"):
            ids = sorted(r["image_id"] for r in selection if r["patient_id"] == patient and r["record_role"] == role)
            assert len(ids) == 2
            order.extend(ids)
    assert protocol["image_order"] == order
    expected_report = dict(status="COMPLETED_PFAMI_EXECUTION_REQUIRES_INDEPENDENT_ANALYSIS", mode="full",
        unique_patients=40, unique_images=160, image_records=480, target_image_records=320, base_image_records=160,
        total_forward=9600, total_backward=0, original_protected_files_unchanged=11,
        attack_success_declared=False, calibration_or_test_used=False, training_updates=0,
        benchmark_excluded_from_main_analysis=True)
    for key, value in expected_report.items():
        assert report[key] == value, key
    assert report["contract_sha256"] == protocol["contract_sha256"]
    assert [r["model"] for r in report["model_reports"]] == list(ALL_MODELS)
    for model in report["model_reports"]:
        assert model["forward"] == 3200 and model["backward"] == 0 and model["prediction_type"] == "epsilon"
        for field in ("weights_unchanged", "all_parameters_frozen", "parameter_gradients_absent"):
            assert model[field] is True
        assert model["adapter_tensors_checked"] == 256 and model["parameter_versions_checked"] == 942
        assert model["base_lora_B_zero"] == (model["model"] == "base")
        assert model["checkpoint_sha256"] == protocol["checkpoint_sha256"].get(model["model"])
    conformance = protocol["cpu_conformance"]
    assert conformance["status"] == "PASS_CPU_SOURCE_STATISTIC_AND_SEED_CONFORMANCE"
    assert conformance["GPU_executions"] == 0 and conformance["unique_seed_cases"] == 40
    assert conformance["source"] == protocol["source"]
    close(conformance["source_score"], conformance["independent_score"])
    assert report["encoding_report_sha256"] == digest(directory / "encoding_report.json")
    encoding = load(directory / "encoding_report.json")
    for key, value in dict(policy="fp16_variant_mode_batch4", dtype="float16", batch_size=4,
        crop_images=160, original_replication_images=0, forward_batch_calls=40,
        forward_model_examples=160, backward=0, weights_unchanged=True, original_replication=[]).items():
        assert encoding[key] == value, key
    assert encoding["crop_latents_sha256"] == digest(directory / "crop_latents.pt")
    assert encoding["encoding_records_sha256"] == digest(directory / "encoding_records.json")
    crops = torch.load(directory / "crop_latents.pt", map_location="cpu", weights_only=True)
    assert set(crops) == {"crop_latents"} and set(crops["crop_latents"]) == set(order)
    cached = torch.load(RUN / "cache/cache.pt", map_location="cpu", weights_only=True)
    records = load(directory / "encoding_records.json")
    assert len(records) == 160 and [r["image_id"] for r in records] == order
    by_id = {r["image_id"]: r for r in selection}
    for row in records:
        iid = row["image_id"]
        assert row["source_sha256"] == by_id[iid]["sha256"]
        assert row["image_filename"] == by_id[iid]["image_filename"]
        assert row["crop_offsets"] == [24, 24, 231, 231]
        crop = crops["crop_latents"][iid]
        assert crop.dtype == torch.float16 and tuple(crop.shape) == (1, 4, 32, 32) and bool(torch.isfinite(crop).all())
        assert tensor_sha(crop) == row["crop_latent_sha256"]
        assert tensor_sha(cached["latents"][iid]) == row["original_latent_sha256"]
    assert cached["audit_prompt"] == contract["generic_prompt"]
    del cached, crops


def verify(directory):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    assert directory.is_dir()
    if (directory / "verification.json").exists():
        raise FileExistsError("Completed verification is immutable: " + str(directory / "verification.json"))
    torch.set_num_threads(2)
    protocol = load(directory / "protocol.json")
    report = load(directory / "report.json")
    images = load(directory / "image_scores.json")
    assert report["protocol_sha256"] == digest(directory / "protocol.json")
    assert report["image_scores_sha256"] == digest(directory / "image_scores.json")
    source, protected = verify_provenance(protocol)
    # Exact execution declaration checks are completed against the frozen runner below.
    verify_execution_declarations(directory, protocol, report)
    by_image, by_patient, exposure = cohort_and_exposures(protocol)
    assert len(images) == 480 and Counter(r["model"] for r in images) == dict.fromkeys(ALL_MODELS, 160)
    expected = {(model, image) for model in ALL_MODELS for image in by_image}
    observed = set()
    raw_hashes, errors, stored_scores, double_scores = {}, [], {}, {}
    for row in images:
        verify_image_identity(row, by_image, exposure, protocol)
        key = row["model"], row["image_id"]
        assert key not in observed
        observed.add(key)
        path = inside(directory, row["residual_path"])
        assert path not in raw_hashes and digest(path) == row["residual_sha256"]
        raw_hashes[path] = row["residual_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert set(payload) == {"residuals", "image_id", "model", "timesteps", "view_order"}
        assert payload["image_id"] == row["image_id"] and payload["model"] == row["model"]
        assert payload["timesteps"] == list(TIMESTEPS) and payload["view_order"] == ["original", "crop"]
        original, double, differences = recompute_image(row, payload["residuals"])
        stored_scores[key] = original
        double_scores[key] = double
        errors.extend(differences)
        del payload
    assert observed == expected and len(errors) == 9600
    reconstructed = reconstruct_patients(images, stored_scores)
    double_reconstructed = reconstruct_patients(images, double_scores)
    statistics = verify_statistics(directory, reconstructed, double_reconstructed)
    assert sum(r["forward"] for r in images) == 9600 and sum(r["backward"] for r in images) == 0
    error_report = dict(schema="pfami-residual-recomputation/v1", MSE_relative_tolerance=MSE_RTOL,
        MSE_absolute_tolerance=MSE_ATOL, records=errors,
        residual_dtype="float32 stored; square and mean independently recomputed on CPU float64",
        max_absolute_MSE_difference=max(abs(r["difference"]) for r in errors),
        max_relative_MSE_difference=max(abs(r["relative_difference"]) for r in errors),
        minimum_original_loss=min(c["original_loss"] for r in images for c in r["cells"]),
        minimum_crop_loss=min(c["crop_loss"] for r in images for c in r["cells"]),
        denominator_clipping_or_imputation=False, all_MSEs_within_declared_tolerance=True)
    output = directory / "residual_recomputation.json"
    assert not output.exists()
    write_json(output, error_report)
    result = dict(status="PASS_INTEGRITY_RESIDUAL_AND_STATISTICAL_RECOMPUTATION",
        source_scope="PFAMI-style conditional latent diffusion adaptation; not full original SD experiment reproduction",
        image_records_checked=480, target_image_records=320, base_image_records=160,
        raw_residual_tensors_checked=480, loss_values_checked=9600, image_relative_scores_checked=480,
        unique_selection_patients=40, fit_benchmark_included=False, calibration_or_test_used=False,
        actual_training_exposure_checks_passed=True, base_private_training_membership_unknown=True,
        total_forward=9600, total_backward=0, target_forward=6400, base_forward=3200,
        protected_original_files_unchanged=11, GPU_execution_by_verifier=False,
        freeze_checks_scope="producer record/protocol checks; no independent GPU state or inference replay",
        no_efficacy_or_novelty_claim_from_integrity_PASS=True, statistics=statistics,
        max_absolute_MSE_difference=error_report["max_absolute_MSE_difference"],
        max_relative_MSE_difference=error_report["max_relative_MSE_difference"],
        source=source, verifier_sha256=digest(Path(__file__)),
        analysis_code_sha256=digest(Path(__file__).with_name("pfami_analysis.py")),
        residual_recomputation_sha256=digest(output),
        input_sha256={name: digest(directory / name) for name in
                      ("protocol.json", "report.json", "image_scores.json", "analysis.json")},
        residual_sha256={p.relative_to(directory).as_posix(): sha for p, sha in raw_hashes.items()},
        seconds_cpu=time.perf_counter()-started)
    write_json(directory / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ("residual_sha256", "source", "input_sha256")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
