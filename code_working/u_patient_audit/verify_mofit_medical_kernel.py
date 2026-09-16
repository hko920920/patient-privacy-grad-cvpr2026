"""Independent CPU checks of saved MoFit arithmetic; no model/producer imports.
Sparse snapshots do not independently verify every network backward pass.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch

PIXEL = (1, 3, 256, 256)
EMBEDDING = (1, 77, 1024)
LATENT = (1, 4, 32, 32)
EPS = float(np.finfo(np.float32).eps)
TINY = float(np.finfo(np.float32).tiny)
SOURCE_COMMIT = "91e4b5edc153bac84b0b4209f70d1b2b94e653b2"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def thash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def arr(value, name, shape):
    require(isinstance(value, torch.Tensor), name + ": tensor required")
    require(value.device.type == "cpu" and value.dtype == torch.float32, name + ": CPU FP32")
    require(tuple(value.shape) == tuple(shape), name + ": shape")
    result = value.detach().contiguous().numpy().astype(np.float64)
    require(np.isfinite(result).all(), name + ": finite")
    return result


def exact(a, b, label):
    require(np.array_equal(np.asarray(a), np.asarray(b)), label + ": exact saved values")


class Arithmetic:
    def __init__(self):
        self.checks = []

    def close(self, actual, expected, magnitude, label, units=32):
        actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        require(actual.shape == expected.shape, label + ": shape")
        require(np.isfinite(actual).all() and np.isfinite(expected).all(), label + ": finite")
        error = np.abs(actual - expected)
        bound = units * EPS * np.maximum(np.asarray(magnitude, dtype=np.float64), TINY) + 32 * TINY
        require(np.all(error <= bound), label + ": local FP32 arithmetic mismatch")
        self.checks.append({"check": label, "elements": int(actual.size),
                            "maximum_absolute_error": float(np.max(error)),
                            "maximum_local_rounding_allowance": float(np.max(bound))})

    def scalar(self, actual, expected, label, rtol=2e-6, atol=1e-8):
        actual, expected = float(actual), float(expected)
        require(math.isfinite(actual) and math.isfinite(expected), label + ": finite")
        error = abs(actual - expected)
        bound = atol + rtol * max(abs(actual), abs(expected))
        require(error <= bound, label + ": scalar mismatch")
        self.checks.append({"check": label, "elements": 1, "maximum_absolute_error": error,
                            "maximum_local_rounding_allowance": bound})

    def double(self, actual, expected, label, magnitude=None):
        actual, expected = float(actual), float(expected)
        scale = max(abs(actual), abs(expected), abs(float(magnitude or 0)), 1e-30)
        require(math.isfinite(actual) and math.isfinite(expected), label + ": finite")
        error, bound = abs(actual - expected), 2e-12 * scale + 1e-25
        require(error <= bound, label + ": CPU-double scalar mismatch")
        self.checks.append({"check": label, "elements": 1, "maximum_absolute_error": error,
                            "maximum_local_rounding_allowance": bound})


def verify_draws(raw):
    draws = raw["draws"]
    initial = torch.rand(PIXEL, generator=torch.Generator().manual_seed(0), dtype=torch.float32) * .6 - .3
    diffusion = torch.randn(LATENT, generator=torch.Generator().manual_seed(260915), dtype=torch.float32)
    pg = torch.Generator().manual_seed(0)
    sur = torch.randn(LATENT, generator=pg, dtype=torch.float32)
    ori = torch.randn(LATENT, generator=pg, dtype=torch.float32)
    for key, reference in [("initial_pixel_perturbation", initial), ("diffusion_noise", diffusion),
                           ("posterior_noise_surrogate", sur), ("posterior_noise_original", ori)]:
        arr(draws[key], key, reference.shape)
        require(torch.equal(draws[key], reference), key + ": declared CPU RNG replay")
    require(draws["metadata"]["initial_uniform_seed"] == 0, "initial seed metadata")
    require(draws["metadata"]["diffusion_seed"] == 260915, "diffusion seed metadata")
    require(draws["metadata"]["posterior_seed_reset"] == 0, "posterior seed metadata")
    require(draws["metadata"]["posterior_order"] == ["surrogate", "original"], "posterior order")
    return {key: thash(draws[key]) for key in draws if isinstance(draws[key], torch.Tensor)}


def verify_cell(cell, noise, alpha, arithmetic, label, *, latent=None, pixel=False):
    prediction = arr(cell["prediction"], label + ".prediction", LATENT)
    target = arr(cell["target"], label + ".target", LATENT)
    exact(target, noise, label + ": fixed epsilon")
    z = arr(cell["scaled_latent"], label + ".scaled_latent", LATENT)
    if pixel:
        mean = arr(cell["posterior_mean"], label + ".posterior_mean", LATENT)
        arithmetic.close(z, mean * .18215, np.abs(mean) * .18215, label + ": VAE mean scaling")
        if "conditional_prediction_unused" in cell:
            arr(cell["conditional_prediction_unused"], label + ".unused_caption_prediction", LATENT)
    if latent is not None:
        exact(z, latent, label + ": fixed posterior sample")
    noisy = arr(cell["noised_latent"], label + ".noised_latent", LATENT)
    q1, q2 = math.sqrt(alpha) * z, math.sqrt(1 - alpha) * target
    arithmetic.close(noisy, q1 + q2, np.abs(q1) + np.abs(q2), label + ": q_sample")
    residual = prediction - target
    loss = math.fsum(float(x) ** 2 for x in residual.reshape(-1)) / residual.size
    arithmetic.scalar(cell["loss"], loss, label + ": independent FP64 MSE")
    return loss


def verify_posterior(raw, arithmetic):
    latents = {}
    for role in ("surrogate", "original"):
        item = raw["posterior"][role]
        values = {k: arr(item[k], role + "." + k, LATENT)
                  for k in ("mean", "std", "logvar", "noise", "sample", "scaled_sample")}
        exact(values["noise"], raw["draws"]["posterior_noise_" + role].numpy(), role + ": actual draw")
        require(float(item["scaling_factor"]) == .18215, role + ": scaling factor")
        require(np.all(values["std"] >= 0), role + ": nonnegative std")
        expected_std = np.exp(.5 * values["logvar"])
        arithmetic.close(values["std"], expected_std, expected_std, role + ": posterior std")
        addend = values["std"] * values["noise"]
        arithmetic.close(values["sample"], values["mean"] + addend,
                         np.abs(values["mean"]) + np.abs(addend), role + ": posterior sampling")
        arithmetic.close(values["scaled_sample"], values["sample"] * .18215,
                         np.abs(values["sample"]) * .18215, role + ": sample scaling")
        latents[role] = values["scaled_sample"]
    return latents


def verify_trace(raw, stage, expected_steps, updates, arithmetic):
    rows = raw["traces"][stage]
    require(len(rows) == expected_steps, stage + ": trace count")
    snapshots = {r["iteration"]: r for r in updates}
    require(len(snapshots) == len(updates), stage + ": duplicate snapshot")
    for index, row in enumerate(rows):
        require(row["iteration"] == index and row["gradient_finite"] is True, stage + ": trace index/finite")
        for key in ("loss", "gradient_l2", "gradient_absmax", "step_l2"):
            require(math.isfinite(float(row[key])) and row[key] >= 0, stage + ": trace " + key)
        for key in ("before_sha256", "after_sha256", "gradient_sha256"):
            require(len(row[key]) == 64 and all(c in "0123456789abcdef" for c in row[key]), stage + ": trace hash")
        if index:
            require(rows[index - 1]["after_sha256"] == row["before_sha256"], stage + ": state hash chain")
        if stage == "pixel":
            arithmetic.double(row["actual_step_size"], .15 - (.15 - .0015) * index / 1000,
                              stage + str(index) + ": nominal schedule")
        else:
            arithmetic.double(row["h"], row["h_raw"] - row["u"], stage + str(index) + ": h orientation")
            arithmetic.double(row["v"], row["v_raw"] - row["u"], stage + str(index) + ": v orientation")
            arithmetic.scalar(row["u"], rows[0]["u"], stage + str(index) + ": fixed null monitor")
            arithmetic.scalar(row["v_raw"], rows[0]["v_raw"], stage + str(index) + ": fixed caption monitor")
        if index in snapshots:
            p = snapshots[index]
            for tensor, key in (("before", "before_sha256"), ("gradient", "gradient_sha256"), ("after", "after_sha256")):
                require(thash(p[tensor]) == row[key], stage + ": captured tensor hash")
            arithmetic.double(p["loss"], row["loss"], stage + ": captured loss")
            g = p["gradient"].double().numpy()
            arithmetic.double(row["gradient_l2"], float(np.linalg.norm(g.reshape(-1))), stage + ": gradient norm")
            arithmetic.double(row["gradient_absmax"], float(np.max(np.abs(g))), stage + ": gradient max")
            distance = (p["after"] - p["before"]).double().numpy()
            arithmetic.double(row["step_l2"], float(np.linalg.norm(distance.reshape(-1))), stage + ": step norm")
    endpoint = raw["final"]["surrogate_pixels" if stage == "pixel" else "embedding"]
    require(thash(endpoint) == rows[-1]["after_sha256"], stage + ": endpoint trace hash")


def verify_updates(raw, noise, alpha, latents, arithmetic):
    for p in raw["pixel_updates"]:
        i = p["iteration"]
        before, gradient, after = [arr(p[k], "pixel." + k, PIXEL) for k in ("before", "gradient", "after")]
        step = .15 - (.15 - .0015) * i / 1000
        arithmetic.double(p["actual_step_size"], step, "pixel: step")
        expected = np.clip(before - np.sign(gradient) * step, -1, 1)
        arithmetic.close(after, expected, np.abs(before) + step, "pixel: sign descent and clamp")
        require(np.all((after >= -1) & (after <= 1)), "pixel: projected range")
        expected_f32 = p["before"].numpy() - np.sign(p["gradient"].numpy()) * np.float32(step)
        clamp_count = int(np.count_nonzero(expected_f32 != np.clip(expected_f32, -1, 1)))
        require(raw["traces"]["pixel"][i]["clamped_count"] == clamp_count, "pixel: clamp count")
        verify_cell(p["cell"], noise, alpha, arithmetic, "pixel" + str(i), pixel=True)
        arithmetic.double(p["loss"], p["cell"]["loss"], "pixel: cell loss")
    previous = {}
    for p in raw["embedding_updates"]:
        i = p["iteration"]
        v = {k: arr(p[k], "embedding." + k, EMBEDDING) for k in
             ("before", "gradient", "after", "exp_avg_before", "exp_avg_after", "exp_avg_sq_before", "exp_avg_sq_after")}
        require(p["step_before"] == i and p["step_after"] == i + 1, "Adam: optimizer step")
        require(np.all(v["exp_avg_sq_before"] >= 0) and np.all(v["exp_avg_sq_after"] >= 0), "Adam: second moment")
        if i == 0:
            exact(v["exp_avg_before"], np.zeros(EMBEDDING), "Adam: initial first moment")
            exact(v["exp_avg_sq_before"], np.zeros(EMBEDDING), "Adam: initial second moment")
        if i - 1 in previous:
            q = previous[i - 1]
            for before, after in (("before", "after"), ("exp_avg_before", "exp_avg_after"), ("exp_avg_sq_before", "exp_avg_sq_after")):
                exact(v[before], q[after], "Adam: adjacent captured state")
        m = .9 * v["exp_avg_before"] + .1 * v["gradient"]
        sq = .999 * v["exp_avg_sq_before"] + .001 * v["gradient"] ** 2
        arithmetic.close(v["exp_avg_after"], m, .9 * np.abs(v["exp_avg_before"]) + .1 * np.abs(v["gradient"]), "Adam: first moment")
        arithmetic.close(v["exp_avg_sq_after"], sq, sq, "Adam: second moment")
        update = .06 * (v["exp_avg_after"] / (1 - .9 ** (i + 1))) / (
            np.sqrt(v["exp_avg_sq_after"] / (1 - .999 ** (i + 1))) + 1e-8)
        arithmetic.close(v["after"], v["before"] - update,
                         np.abs(v["before"]) + np.abs(update), "Adam: bias-corrected full embedding step", units=64)
        verify_cell(p["cell"], noise, alpha, arithmetic, "embedding" + str(i), latent=latents["surrogate"])
        arithmetic.double(p["loss"], p["cell"]["loss"], "embedding: cell loss")
        previous[i] = v


def verify_fd(raw, noise, alpha, latents, arithmetic, expected_h):
    evidence = {}
    require(set(raw["fd"]) == ({"pixel", "embedding"} if expected_h else set()), "FD: stage coverage")
    for stage, packet in raw["fd"].items():
        first = raw[stage + "_updates"][0]
        require(first["iteration"] == 0, "FD: first update present")
        shape, seed = (PIXEL, 1731) if stage == "pixel" else (EMBEDDING, 1732)
        point, direction = arr(packet["point"], "FD point", shape), arr(packet["direction"], "FD direction", shape)
        exact(point, first["before"].numpy(), "FD: exact pre-update point")
        generated = (torch.randint(0, 2, shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1).float()
        require(packet["seed"] == seed and torch.equal(generated, packet["direction"]), "FD: fixed RMS1 direction")
        grad = arr(first["gradient"], "FD gradient", shape)
        products = (grad * direction).reshape(-1)
        ad = math.fsum(float(x) for x in products)
        arithmetic.double(packet["ADdot"], ad, "FD: saved-gradient directional derivative", magnitude=np.sum(np.abs(products)))
        require([r["h"] for r in packet["cells"]] == list(expected_h), "FD: two predeclared h values")
        rows = []
        for row in packet["cells"]:
            h = float(row["h"])
            for side in ("plus", "minus"):
                verify_cell(row[side], noise, alpha, arithmetic, "FD." + stage + "." + side,
                            pixel=stage == "pixel", latent=latents["surrogate"] if stage == "embedding" else None)
                arithmetic.double(row[side + "_loss"], row[side]["loss"], "FD: recorded objective")
            plus, minus = float(row["plus_loss"]), float(row["minus_loss"])
            fd = (plus - minus) / (2 * h)
            producer_floor = EPS * (abs(plus) + abs(minus)) / (2 * h)
            arithmetic.double(row["finite_difference"], fd, "FD: central difference")
            arithmetic.double(row["heuristic_rounding_scale"], producer_floor, "FD: producer rounding heuristic")
            arithmetic.double(row["absolute_AD_difference"], abs(fd - ad), "FD: absolute difference", magnitude=abs(fd) + abs(ad))
            for sign, key in ((1, "plus"), (-1, "minus")):
                shifted = packet["point"] + sign * h * packet["direction"]
                distance = float(np.linalg.norm((shifted - packet["point"]).double().numpy().reshape(-1)))
                arithmetic.double(row["actual_" + key + "_displacement_l2"], distance, "FD: actual FP32 displacement")
            floor = 8 * EPS * max(abs(plus), abs(minus), abs(float(first["loss"]))) / (2 * h)
            rows.append({"h": h, "FD": fd, "AD": ad, "absolute_difference": abs(fd - ad),
                         "relative_difference": abs(fd - ad) / max(abs(fd), abs(ad), 1e-300),
                         "loss_subtraction_heuristic": floor, "sign_agreement": bool(fd * ad > 0)})
        floor = max((r["loss_subtraction_heuristic"] for r in rows), default=0)
        if np.count_nonzero(grad) == 0 or abs(ad) <= floor or any(abs(r["FD"]) <= r["loss_subtraction_heuristic"] for r in rows):
            status = "INCONCLUSIVE_SMALL_DERIVATIVE_OR_LOSS_CANCELLATION"
        else:
            status = "FINITE_DIFFERENCE_EVIDENCE_REQUIRES_NUMERICAL_REVIEW"
        evidence[stage] = {"status": status, "AD": ad, "gradient_nonzero_elements": int(np.count_nonzero(grad)),
                           "rows": rows, "FD_change_between_h": abs(rows[0]["FD"] - rows[1]["FD"]) if len(rows) == 2 else None,
                           "derivative_correctness_claim": False,
                           "limit": "Saved arithmetic only; no independent model backward replay or universal FD tolerance."}
    return evidence


def verify_packet(report, raw, alpha_cumprod_t140):
    started, arithmetic = time.perf_counter(), Arithmetic()
    alpha = float(alpha_cumprod_t140)
    require(0 < alpha < 1, "scheduler alpha")
    require(report["nominal_stage1_steps"] == 1000, "source nominal schedule")
    require(report["mode"] == "literal_cfg2" and report["monitor"] == "literal_every_step", "source execution modes")
    require(report["policy"]["source_commit"] == SOURCE_COMMIT, "public source commit")
    require(report["policy"]["timestep"] == 140 and report["policy"]["prediction_type"] == "epsilon", "source time/prediction")
    require(report["weight_versions_and_no_gradients"] is True, "weight guard")
    require(report["unet_training"] is False and report["vae_training"] is False, "eval mode")
    require(report["all_embedding_coordinates_optimized"] is True and report["optimizer_parameter_shape"] == list(EMBEDDING), "whole embedding")
    require(report["membership_performance_claim"] is False, "kernel-only scope")
    n1, n2 = int(report["stage1_steps"]), int(report["stage2_steps"])
    require(0 < n1 <= 1000 and 0 < n2 <= 300, "supported stage sizes")
    fd_h = report["fd_steps"]
    require(fd_h in ([], [.01, .005]), "fixed FD step policy")
    end, k = int(report["endpoint_diagnostics"]), len(fd_h)
    expected = {"unet_forward_calls": n1 + 4 * n2 + 4 * k + 2 * end,
                "unet_forward_examples": 2 * n1 + 4 * n2 + 6 * k + 3 * end,
                "unet_backward_calls": n1 + n2,
                "vae_forward": n1 + 2 + 2 * k + end, "vae_backward": n1}
    require(report["counts"] == expected, "literal schedule and FD/endpoints counters")
    px = arr(raw["initial_pixels"], "initial_pixels", PIXEL)
    hidden = arr(raw["initial_hidden"], "initial_hidden", EMBEDDING)
    arr(raw["null_hidden"], "null_hidden", EMBEDDING)
    require(np.all((px >= -1) & (px <= 1)), "original normalized pixel range")
    draw_hashes = verify_draws(raw)
    noise = arr(raw["draws"]["diffusion_noise"], "fixed diffusion epsilon", LATENT)
    for stage, n, schedule in (("pixel", n1, [0, 1, 99, 499, 999]), ("embedding", n2, [0, 1, 29, 149, 299])):
        indices = [i for i in schedule if i < n]
        require(report["captured_" + stage + "_updates"] == indices, stage + ": declared snapshots")
        require([p["iteration"] for p in raw[stage + "_updates"]] == indices, stage + ": actual snapshots")
    first_pixel = raw["pixel_updates"][0]
    require(torch.equal(first_pixel["before"], raw["initial_pixels"] + raw["draws"]["initial_pixel_perturbation"]), "unclamped initialization")
    exact(raw["embedding_updates"][0]["before"].numpy(), hidden, "initial embedding")
    latents = verify_posterior(raw, arithmetic)
    verify_updates(raw, noise, alpha, latents, arithmetic)
    for stage, n in (("pixel", n1), ("embedding", n2)):
        verify_trace(raw, stage, n, raw[stage + "_updates"], arithmetic)
    final = raw["final"]
    arr(final["surrogate_pixels"], "final pixels", PIXEL)
    arr(final["embedding"], "final embedding", EMBEDDING)
    scores = final["original_scores"]
    for key, cell_key in (("u", "null_cell"), ("h_raw", "optimized_cell"), ("v_raw", "initial_cell")):
        verify_cell(scores[cell_key], noise, alpha, arithmetic, "original." + key, latent=latents["original"])
        arithmetic.double(scores[key], scores[cell_key]["loss"], "original score/cell")
    for key, raw_key in (("h", "h_raw"), ("v", "v_raw")):
        arithmetic.double(scores[key], scores[raw_key] - scores["u"], "final source score orientation")
    for key in ("u", "h", "v"):
        arithmetic.double(report[key], scores[key], "reported final " + key)
        arithmetic.double(raw["traces"]["embedding"][-1][key], scores[key], "last source monitor " + key)
    for stage in ("pixel", "embedding"):
        first_loss = raw[stage + "_updates"][0]["loss"]
        arithmetic.double(final["initial_" + stage + "_objective"], first_loss, "initial objective")
        endpoint = final["surrogate_" + stage + "_endpoint"]
        if end:
            verify_cell(endpoint["cell"], noise, alpha, arithmetic, "endpoint." + stage,
                        pixel=stage == "pixel", latent=latents["surrogate"] if stage == "embedding" else None)
            arithmetic.double(endpoint["loss"], endpoint["cell"]["loss"], "endpoint loss")
            arithmetic.double(endpoint["initial_minus_final"], first_loss - endpoint["loss"], "endpoint objective change")
            arithmetic.double(report[stage + "_initial_minus_final"], endpoint["initial_minus_final"], "reported objective change")
        else:
            require(endpoint is None and report[stage + "_initial_minus_final"] is None, "no unmeasured endpoint claim")
    fd = verify_fd(raw, noise, alpha, latents, arithmetic, fd_h)
    return {"status": "PASS_SAVED_PACKET_ARITHMETIC_FD_REVIEW_SEPARATE", "image_id": report["image_id"],
            "counts": expected, "draw_sha256": draw_hashes, "saved_scalar_and_array_checks": len(arithmetic.checks),
            "captured_pixel_updates_verified": len(raw["pixel_updates"]),
            "captured_embedding_updates_verified": len(raw["embedding_updates"]),
            "trace_rows_checked": n1 + n2, "independent_backward_replay": False,
            "finite_difference": fd, "checks": arithmetic.checks, "seconds": time.perf_counter() - started,
            "membership_performance_claim": False}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_cfg(report, packet, literal_raw, alpha):
    a = Arithmetic()
    literal = literal_raw["pixel_updates"][0]
    require(report["image_id"] == packet["image_id"], "CFG: image identity")
    require(packet["literal_reference"] == "raw[image_id].pixel_updates[0]", "CFG: declared reference")
    require(torch.equal(packet["point"], literal["before"]), "CFG: same original first point")
    noise = arr(literal_raw["draws"]["diffusion_noise"], "CFG epsilon", LATENT)
    verify_cell(packet["cell"], noise, alpha, a, "CFG1 control", pixel=True)
    a.double(report["loss_literal"], literal["loss"], "CFG literal loss")
    a.double(report["loss_unconditional_only"], packet["cell"]["loss"], "CFG control loss")
    a.double(report["loss_difference"], report["loss_unconditional_only"] - report["loss_literal"], "CFG loss difference")
    results = {}
    for key, reference, measured, shape in (
        ("gradient", literal["gradient"], packet["gradient"], PIXEL),
        ("prediction", literal["cell"]["prediction"], packet["cell"]["prediction"], LATENT)):
        x, y = arr(reference, "CFG literal " + key, shape), arr(measured, "CFG control " + key, shape)
        x, y = x.reshape(-1), y.reshape(-1)
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        observed = {"max_absolute_difference": float(np.max(np.abs(x - y))),
                    "l2_difference": float(np.linalg.norm(x - y)), "literal_l2": nx, "control_l2": ny,
                    "cosine": float(np.dot(x, y) / (nx * ny)) if nx and ny else None,
                    "exact_equal": bool(np.array_equal(x, y))}
        for field in ("max_absolute_difference", "l2_difference", "literal_l2", "control_l2"):
            a.double(report[key][field], observed[field], "CFG " + key + "." + field)
        if observed["cosine"] is None:
            require(report[key]["cosine"] is None, "CFG undefined cosine")
        else:
            a.double(report[key]["cosine"], observed["cosine"], "CFG cosine")
        require(report[key]["exact_equal"] is observed["exact_equal"], "CFG exact-equality metadata")
        results[key] = observed
    expected = dict(unet_forward_calls=1, unet_forward_examples=1, unet_backward_calls=1, vae_forward=1, vae_backward=1)
    require(report["counts"] == expected, "CFG separate counters")
    return {"status": "PASS_SAVED_CFG_COMPARISON_ARITHMETIC", "counts": expected,
            "drift": results, "checks": a.checks,
            "scope": "one point; does not establish complete trajectory equivalence"}


def verify_run(run_dir):
    root = Path(__file__).resolve().parent.parent
    run_dir = Path(run_dir).resolve()
    protocol, execution = load_json(run_dir / "protocol.json"), load_json(run_dir / "execution.json")
    require(execution["status"] == "PASS_KERNEL_EXECUTION_PENDING_INDEPENDENT_VERIFICATION"
            and execution["complete"] is True, "producer completion")
    require(protocol["schema"] == "mofit-medical-kernel-execution/v1", "protocol schema")
    contract = protocol["contract"]
    require(contract["schema"] == "mofit-medical-execution-contract/v1", "external contract schema")
    require(digest(protocol["contract_path"]) == protocol["contract_sha256"], "external contract hash")
    require(load_json(protocol["contract_path"]) == contract, "embedded/external contract")
    frozen = contract["frozen_sha256"]
    require(frozen[str(Path(__file__).resolve())] == digest(__file__), "verifier pre-execution binding")
    for filename, expected in frozen.items():
        require(digest(filename) == expected, "frozen input/source changed: " + filename)
    for field, filename in (("protocol_sha256", "protocol.json"), ("results_sha256", "results.json"), ("raw_sha256", "raw.pt")):
        require(execution[field] == digest(run_dir / filename), filename + ": execution hash")
    for key in ("target_and_VAE_fingerprints_equal", "adapter_tensors_exact_equal",
                "parameter_versions_and_no_weight_gradients", "frozen_files_unchanged"):
        require(execution[key] is True, "execution guard: " + key)
    require(execution["before_fingerprints"] == execution["after_fingerprints"], "unchanged full tensor fingerprints")
    require(execution["perform_membership_performance_evaluation"] is False
            and execution["new_target_training"] is False and execution["stage2_completion_claim"] is False,
            "scope limitations")
    config = load_json(Path(contract["model_snapshot"]) / "scheduler/scheduler_config.json")
    require(config["beta_schedule"] == "scaled_linear" and config["trained_betas"] is None
            and config["prediction_type"] == "epsilon" and config["num_train_timesteps"] == 1000,
            "pinned scheduler family")
    betas = torch.linspace(config["beta_start"] ** .5, config["beta_end"] ** .5, 1000, dtype=torch.float32) ** 2
    alpha = float(torch.cumprod(1 - betas, dim=0)[140])
    require(alpha == contract["alpha_cumprod_t140"] == protocol["alpha_cumprod_t140"], "independent scheduler alpha")
    mode = protocol["mode"]
    require(mode in ("benchmark", "full") and execution["mode"] == mode, "measurement mode")
    selected = contract[mode]
    expected_order = ["00014393_002.png"] if mode == "benchmark" else ["00014393_002.png", "00014393_001.png"]
    require(selected["image_order"] == expected_order, "fixed image scope")
    results = load_json(run_dir / "results.json")
    raws = torch.load(run_dir / "raw.pt", map_location="cpu", weights_only=True)
    require([r["image_id"] for r in results] == expected_order and set(raws) == set(expected_order), "result/raw image coverage")
    require(execution["images"] == len(results) and execution["patients"] == 1 and execution["model"] == "model_1", "scope counts")
    inputs_path = Path(contract["input_directory"]) / "inputs.pt"
    require(digest(inputs_path) == contract["inputs_sha256"], "prepared input hash")
    payload = torch.load(inputs_path, map_location="cpu", weights_only=True)
    cp = root / "_reports/cvpr_u_pilot_v1_001/training_coverage_v2/model_1/step_1000.pt"
    require(digest(cp) == contract["checkpoint_sha256"], "checkpoint hash")
    state = torch.load(cp, map_location="cpu", weights_only=True)
    require(state["step"] == 1000 and sum(state["exposures"].values()) == 4000, "actual checkpoint exposure total")
    with (root / "_reports/cvpr_u_pilot_v1_001/cohort/model_1_train.csv").open(encoding="utf-8", newline="") as stream:
        manifest = {r["image_id"] for r in csv.DictReader(stream)}
    require(set(state["exposures"]) == manifest, "checkpoint/manifest binding")
    verified, totals = [], dict(unet_forward_calls=0, unet_forward_examples=0, unet_backward_calls=0, vae_forward=0, vae_backward=0)
    for report in results:
        image = report["image_id"]
        raw, item = raws[image], payload["images"][image]
        is_e = image == "00014393_002.png"
        expected_metadata = {"patient_id": "14393", "eval_role": "fit", "assignment_group": "A",
                             "scenario": "E" if is_e else "U", "model": "model_1", "member": 1,
                             "record_role": item["metadata"]["record_role"],
                             "image_member": int(is_e), "actual_training_exposures": int(state["exposures"].get(image, 0))}
        for key, value in expected_metadata.items():
            require(report[key] == value, "actual E/U metadata: " + key)
        require((report["actual_training_exposures"] > 0) == is_e, "actual E/U exposure")
        require((report["record_role"] == "train_candidate") == is_e, "E/U source role")
        require(report["caption"] == item["caption"], "frozen caption")
        for key, expected in (("initial_pixels", item["pixels"]), ("initial_hidden", payload["hidden"][item["caption"]]),
                              ("null_hidden", payload["null_hidden"])):
            require(torch.equal(raw[key], expected), "prepared input tensor: " + key)
        require(thash(raw["initial_pixels"]) == report["pixels_sha256"] == item["pixels_sha256"], "pixel hash")
        require(thash(raw["initial_hidden"]) == report["initial_hidden_sha256"], "hidden hash")
        for key in ("stage1_steps", "stage2_steps", "fd_steps", "endpoint_diagnostics"):
            require(report[key] == selected[key], "predeclared kernel field: " + key)
        require(report["policy"] == contract["policy"], "predeclared source kernel policy")
        checked = verify_packet(report, raw, alpha)
        verified.append(checked)
        for key in totals:
            totals[key] += checked["counts"][key]
    cfg = None
    if mode == "benchmark":
        require(execution["cfg_control_sha256"] == digest(run_dir / "cfg_control.pt"), "CFG raw hash")
        require(execution["cfg_control_report_sha256"] == digest(run_dir / "cfg_control.json"), "CFG report hash")
        cfg_raw = torch.load(run_dir / "cfg_control.pt", map_location="cpu", weights_only=True)
        cfg = verify_cfg(load_json(run_dir / "cfg_control.json"), cfg_raw, raws[expected_order[0]], alpha)
        for key in totals:
            totals[key] += cfg["counts"][key]
        require(totals == selected["expected_counts"], "benchmark predeclared total cost")
    else:
        require(execution["cfg_control_sha256"] is None and execution["cfg_control_report_sha256"] is None, "no undeclared CFG call")
    require(totals == execution["counts"], "actual total counters")
    return {"status": "PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE",
            "mode": mode, "images": len(results), "patients": 1, "counts": totals,
            "frozen_files_checked": len(frozen), "checkpoint_sha256": contract["checkpoint_sha256"],
            "input_sha256": contract["inputs_sha256"], "protocol_sha256": digest(run_dir / "protocol.json"),
            "results_sha256": digest(run_dir / "results.json"), "raw_sha256": digest(run_dir / "raw.pt"),
            "image_verifications": verified, "CFG_control": cfg,
            "independent_neural_network_backward_replay": False,
            "membership_performance_claim": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-code-sha256")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.run_dir / "verification.json"
    verifier_protocol = args.run_dir / "verification_protocol.json"
    require(not output.exists() and not verifier_protocol.exists(), "immutable verification already exists")
    code_hash = digest(__file__)
    if args.expected_code_sha256:
        require(code_hash == args.expected_code_sha256, "requested verifier code hash")
    started = time.perf_counter()
    save_new(verifier_protocol, {"schema": "mofit-independent-cpu-verification/v1",
             "verifier_sha256": code_hash, "input_sha256": {name: digest(args.run_dir / name)
             for name in ("protocol.json", "execution.json", "results.json", "raw.pt")},
             "method": "independent saved-tensor arithmetic; FD evidence separate; no producer import or GPU"})
    try:
        result = verify_run(args.run_dir)
    except BaseException as exc:
        result = {"status": "FAILED_INDEPENDENT_VERIFICATION", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        result.update(verifier_sha256=code_hash, verifier_protocol_sha256=digest(verifier_protocol),
                      CPU_seconds=time.perf_counter() - started)
        save_new(output, result)
        print(json.dumps({k: result[k] for k in ("status", "CPU_seconds", "verifier_sha256")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
