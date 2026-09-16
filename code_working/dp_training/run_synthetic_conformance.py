#!/usr/bin/env python3
"""Run deterministic and statistical conformance checks before model training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dp_training.mechanism import (
    MechanismConfig,
    append_public_trace,
    aggregate_noised_update,
    clip_unit_vector,
    make_public_scheduled_event,
    mean_unit_vectors,
    poisson_select,
    tensor_sha256,
    verify_public_trace,
)


SCHEMA = "nih-cxr14-dp-trainer-synthetic-conformance/v1"
PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
CLIP_IMAGE = 0.28448700606156724
CLIP_PATIENT = 0.1997973088974048
RDP_ORDERS = [round(1.0 + index / 10.0, 1) for index in range(1, 100)]
RDP_ORDERS += list(range(12, 64)) + [64, 128, 256, 512]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def protocol_entry(protocol: dict[str, Any], arm: str, cap: int = 10) -> dict[str, Any]:
    matches = [
        item
        for item in protocol["accounting"]["entries"]
        if item["arm"] == arm and int(item["cap"]) == cap
    ]
    require(len(matches) == 1, f"expected one {arm}/K{cap} entry")
    return matches[0]


def config_from_entry(
    entry: dict[str, Any], *, clip_norm: float, rng_backend: str
) -> MechanismConfig:
    return MechanismConfig(
        arm=str(entry["arm"]),
        privacy_unit=str(entry["accounting_unit"]),
        population=int(entry["population"]),
        expected_batch=int(entry["expected_batch"]),
        poisson_sample_rate=float(entry["poisson_sample_rate"]),
        clip_norm=clip_norm,
        noise_multiplier=float(entry["noise_multiplier"]),
        fixed_denominator=float(entry["expected_batch"]),
        max_steps=int(entry["max_steps"]),
        rng_security_mode="TEST_ONLY_DETERMINISTIC",
        rng_backend=rng_backend,
    )


def opacus_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from opacus.accountants.analysis import rdp

    values = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=RDP_ORDERS)
    epsilon, order = rdp.get_privacy_spent(orders=RDP_ORDERS, rdp=values, delta=delta)
    return float(epsilon), float(order)


def google_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    accountant = RdpAccountant(
        orders=RDP_ORDERS,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    accountant.compose(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)),
        count=steps,
    )
    epsilon, order = accountant.get_epsilon_and_optimal_order(delta)
    return float(epsilon), float(order)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    protocol_path = root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json"
    output_dir = root / "_reports" / "nih_cxr14_dp_trainer_synthetic_v1_001"
    output_dir.mkdir(parents=True, exist_ok=True)
    require(sha256_file(protocol_path) == PROTOCOL_SHA256, "frozen protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    image_entry = protocol_entry(protocol, "M1-I8")
    patient_entry = protocol_entry(protocol, "M2-P8")
    image_config = config_from_entry(
        image_entry, clip_norm=CLIP_IMAGE, rng_backend="torch_cpu_seeded_synthetic_only"
    )
    patient_config = config_from_entry(
        patient_entry, clip_norm=CLIP_PATIENT, rng_backend="torch_cpu_seeded_synthetic_only"
    )

    checks: dict[str, Any] = {}

    # Analytic clipping sentinel.
    clipped, clip_stats = clip_unit_vector(torch.tensor([3.0, 4.0]), 2.0)
    require(torch.equal(clipped, torch.tensor([1.2, 1.6])), "analytic clip mismatch")
    checks["analytic_clip"] = {**clip_stats, "output": clipped.tolist(), "status": "PASS"}

    # Patient semantics sentinel: the legal mean-before-clip result must be
    # observably different from clipping each image before averaging.
    per_image = [torch.tensor([4.0, 0.0]), torch.tensor([0.0, 0.0])]
    patient_mean = mean_unit_vectors(per_image)
    legal, _ = clip_unit_vector(patient_mean, 1.0)
    illegal = torch.stack([clip_unit_vector(item, 1.0)[0] for item in per_image]).mean(0)
    require(torch.equal(legal, torch.tensor([1.0, 0.0])), "patient mean-before-clip mismatch")
    require(torch.equal(illegal, torch.tensor([0.5, 0.0])), "illegal comparator mismatch")
    checks["patient_mean_before_clip"] = {
        "patient_mean": patient_mean.tolist(),
        "legal_outer_clipped": legal.tolist(),
        "illegal_image_clip_then_mean": illegal.tolist(),
        "distinct": True,
        "status": "PASS",
    }

    # Empty samples must still consume a Gaussian event and produce a noised
    # gradient.  Exact replay is allowed here only because this is TEST_ONLY.
    template = torch.zeros(262_144, dtype=torch.float32)
    seed = 260_903_101
    empty_update, empty_stats = aggregate_noised_update(
        [], template=template, config=patient_config, generator=torch.Generator().manual_seed(seed)
    )
    replay, replay_stats = aggregate_noised_update(
        [], template=template, config=patient_config, generator=torch.Generator().manual_seed(seed)
    )
    require(float(torch.linalg.vector_norm(empty_update)) > 0.0, "empty sample was skipped")
    require(tensor_sha256(empty_update) == tensor_sha256(replay), "empty noise replay mismatch")
    require(empty_stats["noise_sha256"] == replay_stats["noise_sha256"], "noise digest replay mismatch")
    checks["empty_noise_only_update"] = {
        "update_sha256": tensor_sha256(empty_update),
        "noise_sha256": empty_stats["noise_sha256"],
        "nonzero": True,
        "replay_exact": True,
        "fixed_denominator": empty_stats["fixed_denominator_used"],
        "status": "PASS",
    }

    # Gaussian moment check on the pre-division noise recovered from the empty
    # update. Tolerances were fixed before observing this sample.
    standardized = (
        empty_update.double()
        * patient_config.fixed_denominator
        / patient_config.noise_std_before_division
    ).numpy()
    gaussian_mean = float(standardized.mean())
    gaussian_std = float(standardized.std(ddof=0))
    mean_abs_tolerance = 0.01
    std_relative_tolerance = 0.01
    require(abs(gaussian_mean) <= mean_abs_tolerance, "Gaussian empirical mean outside tolerance")
    require(abs(gaussian_std - 1.0) <= std_relative_tolerance, "Gaussian empirical std outside tolerance")
    checks["gaussian_moments"] = {
        "samples": int(standardized.size),
        "standardized_mean": gaussian_mean,
        "standardized_std": gaussian_std,
        "mean_abs_tolerance": mean_abs_tolerance,
        "std_relative_tolerance": std_relative_tolerance,
        "status": "PASS",
    }

    # Fixed denominator: identical noise cancels, leaving precisely the sum of
    # clipped vectors divided by the expected rather than realized batch.
    small_config = MechanismConfig(
        arm="SYNTHETIC-M1",
        privacy_unit="image",
        population=100,
        expected_batch=4,
        poisson_sample_rate=0.04,
        clip_norm=2.0,
        noise_multiplier=0.5,
        fixed_denominator=4,
        max_steps=100,
        rng_security_mode="TEST_ONLY_DETERMINISTIC",
        rng_backend="torch_cpu_seeded_synthetic_only",
    )
    unit_vectors = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 1.0])]
    mixed, mixed_stats = aggregate_noised_update(
        unit_vectors,
        template=torch.zeros(2),
        config=small_config,
        generator=torch.Generator().manual_seed(91),
    )
    only_noise, _ = aggregate_noised_update(
        [],
        template=torch.zeros(2),
        config=small_config,
        generator=torch.Generator().manual_seed(91),
    )
    expected_signal = torch.tensor([0.3, 0.65])
    torch.testing.assert_close(mixed - only_noise, expected_signal, rtol=1e-6, atol=1e-7)
    require(mixed_stats["fixed_denominator_used"] == 4.0, "realized denominator was used")
    checks["fixed_denominator"] = {
        "realized_units": 2,
        "fixed_denominator": 4,
        "recovered_signal": (mixed - only_noise).tolist(),
        "expected_signal": expected_signal.tolist(),
        "status": "PASS",
    }

    # Independent Bernoulli/Poisson sampling moments.
    sample_ids = [f"u{index:03d}" for index in range(100)]
    sample_q = 0.04
    sample_steps = 20_000
    sampling_generator = torch.Generator().manual_seed(260_903_202)
    counts = np.asarray(
        [len(poisson_select(sample_ids, sample_q, generator=sampling_generator)) for _ in range(sample_steps)],
        dtype=np.float64,
    )
    theoretical_mean = len(sample_ids) * sample_q
    mean_se = math.sqrt(len(sample_ids) * sample_q * (1.0 - sample_q) / sample_steps)
    theoretical_empty = (1.0 - sample_q) ** len(sample_ids)
    empirical_empty = float(np.mean(counts == 0))
    empty_se = math.sqrt(theoretical_empty * (1.0 - theoretical_empty) / sample_steps)
    mean_z = abs(float(counts.mean()) - theoretical_mean) / mean_se
    empty_z = abs(empirical_empty - theoretical_empty) / empty_se
    require(mean_z <= 5.0, "Poisson sample-size mean outside five standard errors")
    require(empty_z <= 5.0, "Poisson empty frequency outside five standard errors")
    checks["poisson_sampling"] = {
        "population": len(sample_ids),
        "sample_rate": sample_q,
        "steps": sample_steps,
        "theoretical_mean": theoretical_mean,
        "empirical_mean": float(counts.mean()),
        "mean_standard_error_z": mean_z,
        "theoretical_empty_frequency": theoretical_empty,
        "empirical_empty_frequency": empirical_empty,
        "empty_frequency_standard_error_z": empty_z,
        "acceptance": "both absolute z scores <= 5",
        "status": "PASS",
    }

    # Public trace commits only to scheduled events. It intentionally excludes
    # sampled identities, realized counts, and DP seeds.
    trace: list[dict[str, Any]] = []
    append_public_trace(
        trace,
        make_public_scheduled_event(image_config, first_step=1, event_count=image_config.max_steps),
    )
    append_public_trace(
        trace,
        make_public_scheduled_event(patient_config, first_step=1, event_count=patient_config.max_steps),
    )
    trace_result = verify_public_trace(trace)
    serialized_trace = json.dumps(trace, sort_keys=True)
    forbidden = [
        key
        for key in ("selected_ids", "realized_batch_size", "realized_unit_count", "sampling_seed", "noise_seed")
        if key in serialized_trace
    ]
    require(not forbidden, f"public trace leaked fields: {forbidden}")
    tampered = copy.deepcopy(trace)
    tampered[0]["event"]["noise_multiplier"] *= 1.01
    tamper_detected = False
    try:
        verify_public_trace(tampered)
    except ValueError:
        tamper_detected = True
    require(tamper_detected, "trace tampering was not detected")
    checks["public_event_trace"] = {
        **trace_result,
        "forbidden_fields_found": forbidden,
        "tamper_detected": tamper_detected,
        "status": "PASS",
    }

    # Recompute both accountant implementations from the executable schedule.
    accountant_rows = []
    for entry, cfg in ((image_entry, image_config), (patient_entry, patient_config)):
        delta = float(entry["target_delta"])
        opacus = opacus_epsilon(cfg.noise_multiplier, cfg.poisson_sample_rate, cfg.max_steps, delta)
        google = google_epsilon(cfg.noise_multiplier, cfg.poisson_sample_rate, cfg.max_steps, delta)
        require(abs(opacus[0] - float(entry["opacus_epsilon"])) <= 1e-12, "Opacus replay drift")
        require(abs(google[0] - float(entry["google_dp_accounting_epsilon"])) <= 1e-12, "Google replay drift")
        accountant_rows.append(
            {
                "arm": cfg.arm,
                "steps": cfg.max_steps,
                "delta": delta,
                "opacus_epsilon": opacus[0],
                "opacus_order": opacus[1],
                "google_epsilon": google[0],
                "google_order": google[1],
                "maximum_absolute_drift": max(
                    abs(opacus[0] - float(entry["opacus_epsilon"])),
                    abs(google[0] - float(entry["google_dp_accounting_epsilon"])),
                ),
                "status": "PASS",
            }
        )
    checks["accountant_replay"] = {
        "rdp_orders": len(RDP_ORDERS),
        "entries": accountant_rows,
        "status": "PASS",
    }

    # Fail-closed configuration sentinels.
    rejections = []
    for label, changes in (
        ("wrong_sampling_rate", {"poisson_sample_rate": patient_config.poisson_sample_rate * 2}),
        ("variable_denominator", {"fixed_denominator": 3}),
        ("wrong_privacy_unit", {"privacy_unit": "record"}),
    ):
        values = patient_config.__dict__.copy()
        values.update(changes)
        try:
            MechanismConfig(**values)
        except ValueError as error:
            rejections.append({"case": label, "rejected": True, "reason": str(error)})
    require(len(rejections) == 3, "a malformed configuration was accepted")
    checks["fail_closed_configuration"] = {"cases": rejections, "status": "PASS"}

    trace_path = output_dir / "public_event_trace.json"
    diagnostics_path = output_dir / "synthetic_private_diagnostics.json"
    write_json(trace_path, {"schema": "nih-cxr14-public-dp-event-trace-bundle/v1", "trace": trace})
    write_json(
        diagnostics_path,
        {
            "schema": "nih-cxr14-synthetic-private-diagnostics/v1",
            "notice": "synthetic only; contains no NIH identifier or image",
            "empty_update": empty_stats,
            "mixed_update": mixed_stats,
        },
    )
    report = {
        "schema": SCHEMA,
        "status": "PASS_SYNTHETIC_DP_MECHANISM_CONFORMANCE",
        "scope": "SYNTHETIC_ONLY_NOT_MODEL_TRAINING_NOT_A_PRIVACY_RELEASE",
        "frozen_protocol_sha256": PROTOCOL_SHA256,
        "mechanism_configs": {
            "M1_I8_K10": image_config.public_dict(),
            "M2_P8_K10": patient_config.public_dict(),
        },
        "checks": checks,
        "artifacts": {
            "public_event_trace": trace_path.name,
            "public_event_trace_sha256": sha256_file(trace_path),
            "synthetic_private_diagnostics": diagnostics_path.name,
            "synthetic_private_diagnostics_sha256": sha256_file(diagnostics_path),
        },
        "claim_limits": [
            "No private image was read and no model parameter was created or updated.",
            "Seeded PyTorch randomness is TEST_ONLY and supplies no release-grade DP evidence.",
            "Passing this gate validates mechanism semantics, not end-to-end private training.",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
