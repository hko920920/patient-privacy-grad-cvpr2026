"""Independent CPU checks of all predeclared smaller-h pixel FD measurements.

The old benchmark, gradient, point and direction remain fixed. All five new h
values and both original h values are reported. No best-h selection, model
calls, or universal neural-network error bound is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch

HELPER_SHA = "cb48ec545eb1c14e0c572cfb4f7815eb0e4512708d6c8eb445d08f91fe2b713f"
STEPS = [.002, .001, .0005, .00025, .0001]
SHAPE = (1, 3, 256, 256)
LATENT = (1, 4, 32, 32)
EPS = float(np.finfo(np.float32).eps)


def require(value, label):
    if not value:
        raise AssertionError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, result):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def helper():
    path = Path(__file__).with_name("verify_mofit_medical_kernel.py")
    require(sha(path) == HELPER_SHA, "unchanged original arithmetic helper")
    from .verify_mofit_medical_kernel import Arithmetic, arr, thash, verify_cell
    return Arithmetic, arr, thash, verify_cell


def verify_refinement(packet, benchmark_raw, alpha):
    Arithmetic, arr, thash, verify_cell = helper()
    a = Arithmetic()
    original = benchmark_raw["fd"]["pixel"]
    first = benchmark_raw["pixel_updates"][0]
    require(first["iteration"] == 0, "saved original pre-update gradient")
    point = arr(packet["point"], "refined point", SHAPE)
    direction = arr(packet["direction"], "refined direction", SHAPE)
    gradient = arr(first["gradient"], "original saved gradient", SHAPE)
    require(torch.equal(packet["point"], first["before"]) and torch.equal(packet["point"], original["point"]),
            "same frozen first point, no optimization or projection")
    require(torch.equal(packet["direction"], original["direction"]), "same original direction")
    regenerated = (torch.randint(0, 2, SHAPE, generator=torch.Generator().manual_seed(1731)) * 2 - 1).float()
    require(packet["seed"] == original["seed"] == 1731 and torch.equal(packet["direction"], regenerated),
            "independent fixed Rademacher direction replay")
    products = (gradient * direction).reshape(-1)
    ad = math.fsum(float(v) for v in products)
    for value in (packet["ADdot"], original["ADdot"]):
        a.double(value, ad, "same original AD directional derivative", magnitude=np.sum(np.abs(products)))
    require([r["h"] for r in packet["cells"]] == STEPS, "all five predeclared smaller h values in order")
    require([r["h"] for r in original["cells"]] == [.01, .005], "unchanged original large h values")
    noise = arr(benchmark_raw["draws"]["diffusion_noise"], "original fixed epsilon", LATENT)
    initial_loss = float(first["loss"])
    curve = []
    for source, rows in (("original_benchmark", original["cells"]), ("smaller_h_refinement", packet["cells"])):
        for row in rows:
            h = float(row["h"])
            fp64_losses = {}
            for side in ("plus", "minus"):
                fp64_losses[side] = verify_cell(row[side], noise, alpha, a,
                                               source + "." + str(h) + "." + side, pixel=True)
                a.double(row[side + "_loss"], row[side]["loss"], "stored objective/cell agreement")
                require("conditional_prediction_unused" in row[side], "literal CFG2 packet")
            plus, minus = float(row["plus_loss"]), float(row["minus_loss"])
            fd = (plus - minus) / (2 * h)
            mse64_fd = (fp64_losses["plus"] - fp64_losses["minus"]) / (2 * h)
            a.double(row["finite_difference"], fd, "reported central difference")
            a.double(row["absolute_AD_difference"], abs(fd - ad), "reported absolute AD difference",
                     magnitude=abs(fd) + abs(ad))
            raw_scale = EPS * (abs(plus) + abs(minus)) / (2 * h)
            a.double(row["heuristic_rounding_scale"], raw_scale, "original scalar rounding scale")
            for sign, side in ((1, "plus"), (-1, "minus")):
                shifted = packet["point"] + sign * h * packet["direction"]
                displacement = (shifted - packet["point"]).double().numpy().reshape(-1)
                a.double(row["actual_" + side + "_displacement_l2"],
                         float(np.linalg.norm(displacement)), "actual rounded displacement")
            floor = 8 * EPS * max(abs(plus), abs(minus), abs(initial_loss)) / (2 * h)
            resolved = abs(ad) > floor and abs(fd) > floor and np.count_nonzero(gradient) > 0
            curve.append({"source": source, "h": h, "AD": ad, "FD_from_saved_FP32_losses": fd,
                          "FD_from_independent_FP64_MSE": mse64_fd,
                          "absolute_AD_difference": abs(fd - ad),
                          "relative_AD_difference": abs(fd - ad) / max(abs(ad), abs(fd), 1e-300),
                          "difference_divided_by_abs_AD": abs(fd - ad) / abs(ad) if ad else None,
                          "FP64_MSE_relative_AD_difference": abs(mse64_fd - ad) / max(abs(ad), abs(mse64_fd), 1e-300),
                          "FD_loss_reduction_precision_difference": abs(fd - mse64_fd),
                          "raw_scalar_rounding_scale": raw_scale,
                          "conservative_scalar_cancellation_heuristic": floor,
                          "derivative_exceeds_scalar_cancellation_heuristic": bool(resolved),
                          "sign_agreement": bool(ad * fd > 0),
                          "nominal_displacement_l2": h * math.sqrt(math.prod(SHAPE)),
                          "actual_plus_displacement_l2": row["actual_plus_displacement_l2"],
                          "actual_minus_displacement_l2": row["actual_minus_displacement_l2"]})
    changes = []
    for left, right in zip(curve[:-1], curve[1:]):
        changes.append({"larger_h": left["h"], "smaller_h": right["h"],
                        "absolute_error_change": right["absolute_AD_difference"] - left["absolute_AD_difference"],
                        "error_decreased": right["absolute_AD_difference"] < left["absolute_AD_difference"],
                        "both_exceed_scalar_cancellation_heuristic":
                            left["derivative_exceeds_scalar_cancellation_heuristic"] and
                            right["derivative_exceeds_scalar_cancellation_heuristic"]})
    require(len(curve) == 7 and len(changes) == 6, "entire h curve retained")
    return {"status": "PASS_SAVED_PIXEL_FD_REFINEMENT_ARITHMETIC_REVIEW_SEPARATE",
            "point_sha256": thash(packet["point"]), "direction_sha256": thash(packet["direction"]),
            "original_gradient_sha256": thash(first["gradient"]), "AD": ad,
            "original_h": [.01, .005], "refined_h": STEPS, "all_h_results": curve,
            "all_adjacent_h_changes": changes, "checks": a.checks,
            "saved_predictions_MSE_checked": 14, "new_predictions_MSE_checked": 10,
            "relative_AD_difference_definition": "abs(FD-AD)/max(abs(AD),abs(FD)); abs(AD) denominator is also separately reported",
            "FP64_MSE_limit": "CPU double reduction of saved FP32 predictions; not FP64 neural-network inference",
            "best_h_selected": False, "independent_neural_network_backward_replay": False,
            "derivative_correctness_automatically_passed": False,
            "interpretation": (
                "Agreement with the fixed AD derivative across several decreasing, numerically resolved h values "
                "can support finite-scale derivative consistency. Coarse-scale error can reflect curvature, and "
                "small-scale loss subtraction can reach an FP32 floor. Inspect the whole predeclared curve; neither "
                "a single favorable h nor saved-cell arithmetic alone proves gradient correctness. The cancellation "
                "heuristic is not a rigorous error bound for neural-network evaluation."),
            "membership_performance_claim": False}


def verify_run(directory):
    directory = Path(directory).resolve()
    protocol, execution = read(directory / "protocol.json"), read(directory / "execution.json")
    require(protocol["schema"] == "mofit-pixel-fd-refinement-execution/v1", "execution schema")
    require(execution["status"] == "PASS_FD_EXECUTION_PENDING_INDEPENDENT_VERIFICATION"
            and execution["complete"] is True, "complete refinement")
    contract = protocol["contract"]
    require(contract["schema"] == "mofit-pixel-fd-refinement-contract/v1", "external contract schema")
    require(sha(protocol["contract_path"]) == protocol["contract_sha256"]
            and read(protocol["contract_path"]) == contract, "external frozen contract")
    frozen = contract["frozen_sha256"]
    require(frozen[str(Path(__file__).resolve())] == sha(__file__), "verifier frozen before measurement")
    for filename, expected in frozen.items():
        require(sha(filename) == expected, "frozen source/input changed: " + filename)
    for name, field in (("protocol.json", "protocol_sha256"), ("results.json", "results_sha256"), ("raw.pt", "raw_sha256")):
        require(sha(directory / name) == execution[field], "producer output hash: " + name)
    benchmark = Path(contract["benchmark_directory"])
    image = contract["benchmark_image_id"]
    require(image == "00014393_002.png", "same existing E fit image")
    require(contract["h_values"] == STEPS, "fixed refined h values")
    old_protocol = read(benchmark / "protocol.json")
    old_execution = read(benchmark / "execution.json")
    old_verification = read(benchmark / "verification.json")
    require(old_verification["status"] == "PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE",
            "original benchmark arithmetic verification")
    require(old_verification["verifier_sha256"] == HELPER_SHA, "original verifier source")
    require(old_execution["raw_sha256"] == sha(benchmark / "raw.pt"), "unchanged original raw packet")
    alpha = protocol["alpha_cumprod_t140"]
    require(alpha == contract["alpha_cumprod_t140"] == old_protocol["alpha_cumprod_t140"], "same original scheduler alpha")
    require(execution["before_fingerprints"] == execution["after_fingerprints"]
            == old_execution["before_fingerprints"] == old_execution["after_fingerprints"],
            "same original model and VAE tensor values")
    require(execution["weight_versions_and_no_gradients"] is True
            and execution["frozen_files_unchanged"] is True, "unchanged parameters and source")
    counts = {"unet_forward_calls": 10, "unet_forward_examples": 20, "unet_backward_calls": 0,
              "vae_forward": 10, "vae_backward": 0}
    require(execution["counts"] == contract["expected_counts"] == counts, "only five symmetric FD pairs; no backward/optimization")
    for key in ("new_backward", "new_target_training", "attack_parameter_tuning",
                "membership_performance_claim", "stage2_completion_claim"):
        require(contract[key] is False and execution[key] is False, "fixed scope: " + key)
    original = torch.load(benchmark / "raw.pt", map_location="cpu", weights_only=True)[image]
    raw = torch.load(directory / "raw.pt", map_location="cpu", weights_only=True)
    result = verify_refinement(raw, original, float(alpha))
    require(result["point_sha256"] == contract["point_sha256"]
            and result["direction_sha256"] == contract["direction_sha256"]
            and result["original_gradient_sha256"] == contract["gradient_sha256"], "contract's fixed input hashes")
    summary = read(directory / "results.json")
    require(summary["image_id"] == image and summary["counts"] == counts, "result identity/cost")
    require(summary["no_best_h_selection"] is True, "all step sizes retained")
    Arithmetic, _, _, _ = helper()
    a = Arithmetic()
    a.double(summary["cached_AD"], result["AD"], "original saved AD")
    a.double(summary["base_loss"], original["pixel_updates"][0]["loss"], "original first-point loss")
    require(len(summary["rows"]) == 7, "all original and refined result rows")
    for reported, independent in zip(summary["rows"], result["all_h_results"]):
        origin = "benchmark" if independent["source"] == "original_benchmark" else "refinement"
        require(reported["h"] == independent["h"] and reported["origin"] == origin, "ordered complete h curve")
        for field, expected in (
            ("FD", independent["FD_from_saved_FP32_losses"]), ("AD", independent["AD"]),
            ("absolute_difference", independent["absolute_AD_difference"]),
            ("relative_difference", independent["relative_AD_difference"]),
            ("diagnostic_floor", independent["conservative_scalar_cancellation_heuristic"])):
            a.double(reported[field], expected, "reported curve field: " + field)
        require(reported["sign_agreement"] == independent["sign_agreement"], "FD sign")
    result.update(image_id=image, counts=counts, result_summary_checks=a.checks,
                  frozen_files_checked=len(frozen), benchmark_directory=str(benchmark),
                  original_verification_sha256=sha(benchmark / "verification.json"),
                  protocol_sha256=sha(directory / "protocol.json"), raw_sha256=sha(directory / "raw.pt"),
                  results_sha256=sha(directory / "results.json"))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-code-sha256", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    require(sha(__file__) == args.expected_code_sha256, "externally frozen refinement verifier")
    out, vp = args.run_dir / "verification.json", args.run_dir / "verification_protocol.json"
    require(not out.exists() and not vp.exists(), "immutable verification outputs")
    started = time.perf_counter()
    new_json(vp, {"schema": "mofit-pixel-fd-independent-verification/v1",
                 "verifier_sha256": sha(__file__), "original_arithmetic_helper_sha256": HELPER_SHA,
                 "input_sha256": {n: sha(args.run_dir / n) for n in ("protocol.json", "execution.json", "results.json", "raw.pt")},
                 "all_h_reported_no_best_h": True, "no_GPU_or_backward_replay": True})
    try:
        result = verify_run(args.run_dir)
    except BaseException as exc:
        result = {"status": "FAILED_PIXEL_FD_REFINEMENT_VERIFICATION", "error": repr(exc),
                  "traceback": traceback.format_exc()}
        raise
    finally:
        result.update(verifier_sha256=sha(__file__), verification_protocol_sha256=sha(vp),
                      CPU_seconds=time.perf_counter() - started)
        new_json(out, result)
        print(json.dumps({k: result[k] for k in ("status", "CPU_seconds", "verifier_sha256")}), flush=True)


if __name__ == "__main__":
    main()
