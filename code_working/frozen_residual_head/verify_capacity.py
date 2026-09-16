"""Independent CPU checks for the fixed-feature capacity experiment.

This module does not import the feature producer, its statistics routines, or
its fitting routines. Numerical acceptance is separate from research efficacy.
The run-directory schema is bound explicitly by verify_run below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np


EPS64 = float(np.finfo(np.float64).eps)
EPS32 = float(np.finfo(np.float32).eps)
STATUS = "PASS_FROZEN_RESIDUAL_CAPACITY_SAVED_ARITHMETIC_AND_PROVENANCE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Audit:
    def __init__(self) -> None:
        self.count = 0
        self.comparisons: list[dict[str, Any]] = []

    def require(self, value: bool, name: str) -> None:
        self.count += 1
        if not bool(value):
            raise AssertionError(name)

    def compare(self, actual: Any, expected: Any, name: str,
                *, atol: float, rtol: float = 0.0) -> None:
        x, y = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(x.shape == y.shape, name + ": shape")
        self.require(bool(np.isfinite(x).all() and np.isfinite(y).all()), name + ": finite")
        difference = np.abs(x - y)
        bound = atol + rtol * np.abs(y)
        maximum = float(difference.max(initial=0))
        self.comparisons.append({"name": name, "max_abs_error": maximum,
                                 "atol": atol, "rtol": rtol})
        self.require(bool(np.all(difference <= bound)),
                     f"{name}: maximum error {maximum:.17g}, atol={atol}, rtol={rtol}")


def independent_basis(timestep: int, alphas: np.ndarray) -> np.ndarray:
    """P0..P3 on scheduler-endpoint-normalized log SNR, without fitted scales."""
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.shape != (1000,) or not np.all((alphas > 0) & (alphas < 1)):
        raise AssertionError("scheduler alphas must be 1000 finite values strictly inside (0,1)")
    if not (0 <= int(timestep) < 1000):
        raise AssertionError("timestep range")
    log_snr = np.log(alphas) - np.log1p(-alphas)
    minimum, maximum = float(log_snr.min()), float(log_snr.max())
    if not maximum > minimum:
        raise AssertionError("nonconstant log SNR")
    x = 2 * (float(log_snr[timestep]) - minimum) / (maximum - minimum) - 1
    return np.array([1, x, (3*x*x - 1)/2, (5*x*x*x - 3*x)/2], dtype=np.float64)


def independent_statistics(projected: np.ndarray, residual: np.ndarray,
                           basis: np.ndarray) -> dict[str, dict[str, Any]]:
    """Full design statistics; loss is squared vector norm per pixel.

    The producer lifts spatial moments with a Kronecker product. Here the
    64-column design is explicitly expanded and multiplied, independently.
    Q is the mean squared output-vector norm, so ordinary channel MSE is Q/4.
    """
    h = np.asarray(projected, dtype=np.float64)
    e = np.asarray(residual, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 16 or e.shape != (len(h), 4) or b.shape != (4,):
        raise AssertionError("sample matrix shapes must be (positions,16)/(positions,4)/(4,)")
    if not np.isfinite(h).all() or not np.isfinite(e).all() or not np.isfinite(b).all():
        raise AssertionError("finite sample")
    if not np.array_equal(h[:, 0], np.ones(len(h))) or b[0] != 1:
        raise AssertionError("explicit spatial intercept and P0")
    spatial_a = (h.T @ h) / len(h)
    spatial_b = (h.T @ e) / len(h)
    full = np.concatenate([h * float(coefficient) for coefficient in b], axis=1)
    q = float(math.fsum(float(v) for v in np.square(e).sum(axis=1)) / len(h))
    return {
        "time64": {"A": (full.T @ full) / len(h),
                   "B": (full.T @ e) / len(h), "Q": q},
        "static16": {"A": spatial_a, "B": spatial_b, "Q": q},
        "time4": {"A": np.outer(b, b),
                  "B": b[:, None] * e.mean(axis=0)[None, :], "Q": q},
    }


def direct_mse(projected: np.ndarray, residual: np.ndarray, basis: np.ndarray,
               weight: np.ndarray, family: str) -> float:
    h, e, b, w = (np.asarray(z, dtype=np.float64)
                   for z in (projected, residual, basis, weight))
    if family == "time64":
        correction = np.zeros_like(e)
        for k in range(4):
            correction += b[k] * (h @ w[16*k:16*(k+1), :])
    elif family == "static16":
        correction = h @ w
    elif family == "time4":
        correction = np.broadcast_to(b @ w, e.shape)
    else:
        raise AssertionError(f"unsupported family {family}")
    return float(np.mean(np.square(e - correction)))


def quadratic_mse(stats: dict[str, Any], weight: np.ndarray) -> float:
    a, b, w = (np.asarray(z, dtype=np.float64) for z in (stats["A"], stats["B"], weight))
    return float((stats["Q"] - 2 * np.sum(w * b) + np.sum(w * (a @ w))) / b.shape[1])


def verify_solution(audit: Audit, stats: dict[str, Any], weight: np.ndarray,
                    ridge: float, name: str) -> dict[str, float]:
    """Check the raw quadratic's stationarity, curvature, and minimizer.

    All thresholds below are numerical roundoff checks, not performance gates.
    The independent solution uses the eigensystem of the symmetric Gram.
    """
    a = np.asarray(stats["A"], dtype=np.float64)
    b = np.asarray(stats["B"], dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    audit.require(a.shape == (len(w), len(w)) and b.shape == w.shape, name + ": shapes")
    audit.require(ridge > 0 and math.isfinite(ridge), name + ": positive finite ridge")
    scale = max(1.0, float(np.linalg.norm(a, ord=2)))
    audit.compare(a, a.T, name + ": symmetric", atol=128*EPS64*len(w)*scale)
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    audit.require(float(values.min()) >= -256*EPS64*len(w)*scale, name + ": PSD Gram")
    h = a + ridge * np.eye(len(w))
    residual = h @ w - b
    denominator = float(np.linalg.norm(h, 2) * np.linalg.norm(w) + np.linalg.norm(b))
    relative_stationarity = float(np.linalg.norm(residual)) / max(denominator, np.finfo(float).tiny)
    audit.require(relative_stationarity <= 1024*EPS64*len(w), name + ": stationarity")
    expected_w = vectors @ ((vectors.T @ b) / (values + ridge)[:, None])
    condition = float((values.max()+ridge)/(values.min()+ridge))
    weight_scale = max(1.0, float(np.linalg.norm(expected_w)))
    audit.compare(w, expected_w, name + ": eigensystem solution",
                  atol=1024*EPS64*len(w)*max(1.0,condition)*weight_scale)
    objective = quadratic_mse(stats, w) * b.shape[1] + ridge * float(np.sum(w*w))
    audit.require(objective <= float(stats["Q"]) + 1024*EPS64*len(w)*max(1, abs(float(stats["Q"]))),
                  name + ": training regularized objective no larger than W0")
    return {"relative_stationarity": relative_stationarity, "gram_min_eigenvalue": float(values.min()),
            "regularized_condition_number": condition, "training_mse": quadratic_mse(stats,w),
            "training_vector_loss_plus_ridge": objective}


def self_test() -> dict[str, Any]:
    """Tiny algebra fixture; no target model, data fitting, or GPU imports."""
    audit = Audit()
    rng = np.random.default_rng(30916)
    h = np.column_stack([np.ones(23), rng.normal(size=(23,15))])
    e = rng.normal(size=(23,4))
    b = np.array([1,.3,(3*.3**2-1)/2,(5*.3**3-3*.3)/2])
    moments = independent_statistics(h,e,b)
    phi = np.concatenate([v*h for v in b], axis=1)
    audit.compare(moments["time64"]["A"], phi.T@phi/len(phi), "fixture A", atol=5e-15)
    audit.compare(moments["time64"]["B"], phi.T@e/len(phi), "fixture B", atol=5e-15)
    for family, s in moments.items():
        w = np.linalg.solve(s["A"]+.001*np.eye(len(s["A"])),s["B"])
        verify_solution(audit,s,w,.001,"fixture "+family)
        audit.compare(quadratic_mse(s,w),direct_mse(h,e,b,w,family),"fixture direct "+family,atol=1e-13)
    # A changed solution must fail the stationary-quadratic check.
    corrupted = w.copy(); corrupted[0,0] += .01
    rejected = False
    try:
        verify_solution(Audit(),s,corrupted,.001,"corrupted fixture")
    except AssertionError:
        rejected = True
    audit.require(rejected,"corrupted solution rejected")
    return {"status":"PASS_SYNTHETIC_ALGEBRA_ONLY","checks":audit.count}


FAMILIES = {"full": "time64", "static": "static16", "time": "time4"}


def average_moments(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit unit-weighted average, retaining the Q constant."""
    return {"A": sum(row["A"] for row in units) / len(units),
            "B": sum(row["B"] for row in units) / len(units),
            "Q": math.fsum(float(row["Q"]) for row in units) / len(units)}


def projection_witness(audit: Audit, hidden: np.ndarray, features: np.ndarray,
                       projection: np.ndarray, name: str) -> dict[str, float]:
    audit.require(hidden.shape == (1024,320) and hidden.dtype == np.float32,
                  name + ": witness shape/dtype")
    audit.require(bool(np.isfinite(hidden).all()), name + ": finite hidden")
    # GPU uses P32. Include that conversion before the independent FP64 dot.
    hp = hidden.astype(np.float64)
    pp = projection.astype(np.float32).astype(np.float64)
    exactish = hp @ pp
    envelope = (320*EPS32/(1-320*EPS32)) * (np.abs(hp) @ np.abs(pp))
    envelope += EPS32 * np.abs(exactish) + np.finfo(np.float32).tiny
    difference = np.abs(features[:,1:].astype(np.float64) - exactish)
    audit.require(bool(np.all(difference <= envelope)), name + ": FP32 projection rounding envelope")
    return {"max_absolute_error": float(difference.max()),
            "max_error_over_envelope": float(np.max(difference/envelope)),
            "rounding_model": "gamma_320 FP32 dot of saved hidden and P32; not a GPU re-execution"}


def verify_saved_capacity(run_dir: Path, audit: Audit) -> dict[str, Any]:
    """Read the declared saved tensors; this never loads a target model."""
    run_dir = run_dir.resolve()
    contract = read_json(run_dir / "contract.json")
    manifest = read_json(run_dir / "manifest.json")
    analysis = read_json(run_dir / "analysis.json")
    images = read_json(run_dir / "images.json")
    expected_records = [dict(image,draw_id=draw) for image in images for draw in range(8)]
    audit.require(contract["ridge_lambda"] == .001, "fixed channel-sum ridge .001")
    audit.require(contract["draws_per_image"] == 8, "eight draws per image")
    audit.require(len(manifest) == 3840, "3840 samples: train80/eval40 x4images x8draws")
    with np.load(run_dir / "projection.npz", allow_pickle=False) as z:
        projection, alphas = z["P"].copy(), z["alphas"].copy()
    audit.require(projection.shape == (320,15) and projection.dtype == np.float64, "projection dimensions/dtype")
    audit.require(alphas.shape == (1000,) and alphas.dtype == np.float32, "public scheduler dimensions/dtype")
    random_matrix = np.random.default_rng(contract["projection_seed"]).standard_normal((320,15))
    q, rr = np.linalg.qr(random_matrix, mode="reduced")
    expected_projection = q * np.where(np.diag(rr)<0,-1.0,1.0)
    audit.compare(projection,expected_projection,"seeded public projection",atol=128*EPS64*320)
    audit.compare(projection.T@projection, np.eye(15), "orthonormal projection", atol=128*EPS64*320)
    logsnr = np.log(alphas.astype(np.float64)) - np.log1p(-alphas.astype(np.float64))
    audit.compare(contract["time_basis_logsnr_min"], logsnr.min(), "public logSNR lower endpoint", atol=1e-13)
    audit.compare(contract["time_basis_logsnr_max"], logsnr.max(), "public logSNR upper endpoint", atol=1e-13)
    with np.load(run_dir / "models.npz", allow_pickle=False) as z:
        weights = {family:z["W_"+family].copy() for family in FAMILIES}
    for family, dimension in (("full",64),("static",16),("time",4)):
        audit.require(weights[family].shape == (dimension,4) and weights[family].dtype == np.float64,
                      family + ": W shape/dtype")
    grouped: dict[tuple[str,str,str], dict[str, Any]] = {}
    image_draws: dict[tuple[str,str,str], set[int]] = defaultdict(set)
    patient_roles: dict[str,str] = {}
    seen_sample_keys = set()
    witness_rows = []
    direct_gradients = {family:np.zeros_like(weights[family]) for family in FAMILIES}
    direct_losses: dict[tuple[str,str,str], dict[str,float]] = {}
    started = time.perf_counter()
    import torch
    audit.require(not torch.cuda.is_initialized(), "CPU verifier starts without CUDA initialization")
    for index, row in enumerate(manifest):
        audit.require(row["record_id"] == index, "manifest contiguous record ID")
        split, pid, image_id = str(row["split"]), str(row["patient_id"]), str(row["image_id"])
        for field,value in expected_records[index].items():
            audit.require(row[field] == value,"manifest versus predeclared image/draw "+field)
        audit.require(split in ("train","eval"), "declared split")
        audit.require(pid not in patient_roles or patient_roles[pid] == split, "patient train/eval disjoint")
        patient_roles[pid] = split
        draw, timestep = int(row["draw_id"]), int(row["timestep"])
        seed_parts = (contract["seed_salt"],split,image_id,draw)
        def declared_seed(suffix):
            payload = "|".join(map(str,(*seed_parts,suffix))).encode("utf-8")
            return int(hashlib.sha256(payload).hexdigest()[:15],16)
        audit.require(row["noise_seed"]==declared_seed("noise"),"fixed independent noise seed")
        expected_t = draw*125 + int(np.random.default_rng(declared_seed("timestep")).integers(0,125))
        audit.require(timestep==expected_t,"fixed stratified timestep jitter")
        audit.require(0 <= draw < 8, "draw range")
        key = (split,pid,image_id)
        sample_key = (*key,draw)
        audit.require(sample_key not in seen_sample_keys, "unique image draw")
        seen_sample_keys.add(sample_key)
        image_draws[key].add(draw)
        path = (run_dir / row["raw_path"]).resolve()
        audit.require(path.is_relative_to(run_dir), "raw path in run directory")
        audit.require(sha256(path) == row["raw_sha256"], "sample raw hash")
        with np.load(path, allow_pickle=False) as raw:
            h, base, target, basis = (raw[k].copy() for k in ("features","base","target","basis"))
            audit.require(int(raw["timestep"]) == timestep, "raw/manifest timestep")
            audit.require(h.shape==(1024,16) and h.dtype==np.float32, "raw h16 layout/dtype")
            audit.require(base.shape==target.shape==(1024,4) and base.dtype==target.dtype==np.float32,
                          "raw base/target layout/dtype")
            audit.require(basis.shape==(4,) and basis.dtype==np.float64, "basis layout/dtype")
            audit.require(bool(np.isfinite(base).all() and np.isfinite(target).all()), "finite base/target")
            target_replay = torch.randn((1,4,32,32),dtype=torch.float32,
                generator=torch.Generator().manual_seed(row["noise_seed"])).numpy()[0].transpose(1,2,0).reshape(1024,4)
            audit.require(np.array_equal(target,target_replay),"exact CPU epsilon draw replay")
            audit.compare(basis,independent_basis(timestep,alphas),"sample fixed Legendre basis",atol=1e-13)
            if "hidden" in raw.files:
                witness_rows.append({"record_id":index,"split":split,"image_id":image_id,"draw":draw,
                                     **projection_witness(audit,raw["hidden"],h,projection,"sample projection")})
        residual = target.astype(np.float64) - base.astype(np.float64)
        sample_stats = independent_statistics(h,residual,basis)
        if key not in grouped:
            grouped[key] = {family:{"A":np.zeros_like(s["A"]),"B":np.zeros_like(s["B"]),"Q":0.0}
                            for family,s in sample_stats.items()}
            direct_losses[key] = {"base":0.0,**{family:0.0 for family in FAMILIES}}
        for family,s in sample_stats.items():
            for entry in ("A","B","Q"):
                grouped[key][family][entry] += s[entry] / 8
        direct_losses[key]["base"] += float(np.mean(residual*residual)) / 8
        for family, internal in FAMILIES.items():
            direct_losses[key][family] += direct_mse(h,residual,basis,weights[family],internal) / 8
            if split == "train":
                if family == "full":
                    design = np.concatenate([h.astype(np.float64)*v for v in basis],axis=1)
                elif family == "static":
                    design = h.astype(np.float64)
                else:
                    design = np.broadcast_to(basis,(len(h),4))
                # Derivative of channel-sum loss /2, directly from raw residuals.
                direct_gradients[family] += (design.T@(design@weights[family]-residual)) / (1024*8*4*80)
        if (index+1) % 256 == 0:
            print(json.dumps({"verified_samples":index+1,"total":len(manifest),
                              "seconds":round(time.perf_counter()-started,2)}),flush=True)
    audit.require(len(witness_rows)==4, "four predeclared projection witnesses")
    for split in ("train","eval"):
        selected = [r for r in manifest if r["split"] == split]
        first_image = selected[0]["image_id"]
        actual_witness = {(r["image_id"],r["draw"]) for r in witness_rows if r["split"]==split}
        audit.require(actual_witness=={(first_image,0),(first_image,7)},"predeclared witness identities")
    audit.require(sum(v=="train" for v in patient_roles.values())==80,"80 training patients")
    audit.require(sum(v=="eval" for v in patient_roles.values())==40,"40 development evaluation patients")
    by_patient = defaultdict(list)
    for key,draws in image_draws.items():
        audit.require(draws==set(range(8)), "every image eight unique draws")
        by_patient[(key[0],key[1])].append(key)
    patient_moments, patient_results = {}, []
    for (split,pid), keys in by_patient.items():
        audit.require(len(keys)==4,"four equally weighted images per patient")
        patient_moments[pid] = {family:average_moments([grouped[key][family] for key in keys])
                               for family in FAMILIES.values()}
        metric = {family:math.fsum(direct_losses[key][family] for key in keys)/4
                  for family in ("base",*FAMILIES)}
        patient_results.append({"patient_id":pid,"split":split,
                                **{family+"_mse":value for family,value in metric.items()}})
    with np.load(run_dir/"patient_stats.npz",allow_pickle=False) as saved:
        saved_ids,saved_splits = [str(x) for x in saved["patient_ids"]],[str(x) for x in saved["splits"]]
        audit.require(len(saved_ids)==120 and len(set(saved_ids))==120,"unique 120 saved patient IDs")
        audit.require(set(saved_ids)==set(patient_roles),"saved patient membership")
        for index,pid in enumerate(saved_ids):
            audit.require(saved_splits[index]==patient_roles[pid],"saved patient split")
            for family,internal in FAMILIES.items():
                s = patient_moments[pid][internal]
                for entry in ("A","B","Q"):
                    expected = s[entry]
                    scale = max(1.0,float(np.max(np.abs(expected))))
                    bound = 256*EPS64*(1024+32)*scale
                    audit.compare(saved[entry+"_"+family][index],expected,
                                  "patient "+pid+" "+family+" "+entry,atol=bound)
    reported = analysis["per_patient_results"]
    audit.require(len(reported)==120,"120 reported patient metric rows")
    report_by_id = {str(r["patient_id"]):r for r in reported}
    audit.require(len(report_by_id)==120 and set(report_by_id)==set(patient_roles),"reported patient IDs")
    for row in patient_results:
        pid = row["patient_id"]
        audit.require(report_by_id[pid]["split"]==row["split"],"reported metric split")
        for field in ("base_mse","full_mse","static_mse","time_mse"):
            audit.compare(report_by_id[pid][field],row[field],"raw direct patient "+pid+" "+field,
                          atol=1024*EPS64*(1024+64)*max(1.0,abs(row[field])))
        for family,internal in FAMILIES.items():
            expected = quadratic_mse(patient_moments[pid][internal],weights[family])
            audit.compare(expected,row[family+"_mse"],"raw direct versus quadratic "+pid+" "+family,
                          atol=1024*EPS64*(1024+64)*max(1.0,abs(expected)))
    audit.require(analysis["ridge_lambda"]==.001,"analysis fixed ridge")
    solution_diagnostics = {}
    for family,internal in FAMILIES.items():
        train = [patient_moments[pid][internal] for pid,split in patient_roles.items() if split=="train"]
        stats = average_moments(train)
        solution_diagnostics[family] = verify_solution(audit,stats,weights[family],.001,family)
        direct_gradient = direct_gradients[family]+.001*weights[family]
        scale = max(1.0,float(np.linalg.norm(stats["B"])))
        audit.require(float(np.linalg.norm(direct_gradient)) <= 2048*EPS64*(1024+64)*scale,
                      family + ": raw sample gradient plus ridge stationarity")
        solution_diagnostics[family]["direct_raw_half_gradient_frobenius"] = float(np.linalg.norm(direct_gradient))
    summaries = []
    for split in ("train","eval"):
        rows = [r for r in patient_results if r["split"]==split]
        summary = {"split":split,"patients":len(rows)}
        for family in ("base",*FAMILIES):
            summary[family+"_mse"] = math.fsum(r[family+"_mse"] for r in rows)/len(rows)
        summaries.append(summary)
        declared = analysis["summaries"][split]
        for family in ("base",*FAMILIES):
            audit.compare(declared["mean_mse"][family],summary[family+"_mse"],
                          "equal-patient reported mean "+split+" "+family,atol=1e-10)
        audit.compare(declared["relative_full_reduction"],
                      (summary["base_mse"]-summary["full_mse"])/summary["base_mse"],
                      "relative full reduction "+split,atol=1e-10)
        # The bootstrap is conditional descriptive uncertainty over patients.
        # Preserve the producer's patient order; independently use multinomial
        # counts rather than producer delta[indices].mean(axis=1).
        ordered_ids = [str(r["patient_id"]) for r in reported if r["split"]==split]
        independently_scored = {r["patient_id"]:r for r in rows}
        resamples = np.random.default_rng(contract["bootstrap_seed"]).integers(
            0,len(rows),size=(contract["bootstrap_replicates"],len(rows)))
        count_matrix = np.zeros((len(resamples),len(rows)),dtype=np.int32)
        for sample_index,indices in enumerate(resamples):
            count_matrix[sample_index] = np.bincount(indices,minlength=len(rows))
        contrasts = (("base","full"),("static","full"),("time","full"),("base","static"),("base","time"))
        audit.require(set(declared["paired_reductions"])=={a+"_minus_"+b for a,b in contrasts},
                      "all five predetermined contrasts "+split)
        for left,right in contrasts:
            name = left+"_minus_"+right
            delta = np.array([independently_scored[pid][left+"_mse"]-independently_scored[pid][right+"_mse"]
                              for pid in ordered_ids])
            estimates = (count_matrix @ delta) / len(rows)
            expected_ci = np.quantile(estimates,[.025,.975])
            saved_delta = declared["paired_reductions"][name]
            audit.compare(saved_delta["mean"],math.fsum(delta)/len(delta),"paired mean "+split+" "+name,atol=1e-10)
            audit.compare(saved_delta["ci95"],expected_ci,"paired patient CI "+split+" "+name,atol=1e-10)
            audit.require(saved_delta["patients"]==len(rows),"bootstrap patient unit")
            audit.require(saved_delta["positive_patients"]==int(np.sum(delta>0)),"positive patient count")
            audit.require(saved_delta["scope"]=="pointwise descriptive patient bootstrap on reused development cohort",
                          "bootstrap descriptive scope")
    audit.require(not torch.cuda.is_initialized(),"CPU verifier completed without CUDA initialization")
    return {"projection_witnesses":witness_rows,"solution_diagnostics":solution_diagnostics,
            "independent_patient_results":patient_results,"independent_equal_patient_means":summaries,
            "raw_samples":len(manifest),"patient_count":len(patient_roles)}


def verify_provenance(run_dir: Path, audit: Audit) -> dict[str,str]:
    contract = read_json(run_dir/"contract.json")
    execution = read_json(run_dir/"execution.json")
    analysis = read_json(run_dir/"analysis.json")
    audit.require(contract["schema"]=="frozen-residual-capacity/v1","contract schema")
    audit.require(contract["seed_salt"]=="frozen-residual-capacity-20260916-v1","fixed seed salt")
    audit.require(contract["projection_seed"]==26091601,"fixed projection seed")
    audit.require(contract["bootstrap_seed"]==26091602 and contract["bootstrap_replicates"]==8000,"fixed bootstrap")
    audit.require(contract["expected_patients"]=={"train":80,"eval":40},"fixed patient sample")
    audit.require(contract["expected_images"]=={"train":320,"eval":160},"fixed image sample")
    audit.require(contract["expected_records"]==3840 and contract["images_per_patient"]==4,"fixed record count")
    audit.require(contract["method_dimensions"]=={"full":64,"static":16,"time":4},"fixed head dimensions")
    audit.require(contract["prediction_type"]=="epsilon" and contract["latent_shape"]==[1,4,32,32],"epsilon latent contract")
    audit.require(contract["tf32"] is False and contract["deterministic_algorithms"] is True,"FP32 arithmetic policy")
    audit.require(contract["development_only"] is True and contract["private_release"] is False,"development scope")
    audit.require(contract["dp_applied"] is False and contract["clipping_applied"] is False,"unclipped non-DP capacity")
    audit.require(contract["performance_tuning_allowed"] is False and contract["no_adaptive_rank_lambda_seed_timestep_selection"] is True,
                  "fixed capacity configuration")
    sources = contract["source_sha256"]
    audit.require(len(sources)>=13,"required frozen sources")
    for name,expected in sources.items():
        audit.require(sha256(Path(name))==expected,"frozen source "+name)
    def one_source(suffix):
        matches = [Path(p) for p in sources if p.replace("\\","/").endswith(suffix)]
        audit.require(len(matches)==1,"one bound source "+suffix)
        return matches[0]
    for suffix in ("frozen_residual_head/run_capacity.py","frozen_residual_head/residual_math.py",
                   "u_patient_audit/models.py","u_patient_audit/common.py","u_patient_audit/build_cache.py",
                   "data_pipeline/nih_cxr14_model_input.py","cohort/lock.json","cache/cache.pt",
                   "cache/summary.json","unet/config.json","scheduler/scheduler_config.json",
                   "unet/diffusion_pytorch_model.fp16.safetensors"):
        one_source(suffix)
    csv_path = one_source("cohort/evaluation_images.csv")
    cohort_lock = read_json(one_source("cohort/lock.json"))
    audit.require(sha256(csv_path)==cohort_lock["files"]["evaluation_images.csv"],
                  "evaluation image table matches original cohort lock")
    with csv_path.open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["eval_role"] in ("fit","selection")]
    selected.sort(key=lambda r:(r["eval_role"]!="fit",int(r["patient_id"]),r["image_id"]))
    images = read_json(run_dir/"images.json")
    audit.require(len(images)==len(selected)==480,"480 bound original development images")
    excluded = {r["patient_id"] for r in rows if r["eval_role"] in ("calibration","test")}
    audit.require(not excluded.intersection(r["patient_id"] for r in images),"untouched calibration/test patients")
    for image,original in zip(images,selected):
        for field in ("patient_id","image_id","sha256"):
            audit.require(image[field]==original[field],"original image identity "+field)
        audit.require(image["split"]==("train" if original["eval_role"]=="fit" else "eval"),"original role mapping")
        audit.require(image["original_eval_role"]==original["eval_role"] and
                      image["original_record_role"]==original["record_role"],"historical role annotation")
        audit.require(sha256(Path(image["path"]))==image["sha256"],"bound source image")
    for name in ("projection","images"):
        suffix = ".npz" if name=="projection" else ".json"
        audit.require(sha256(run_dir/(name+suffix))==contract[name+"_sha256"],"predeclared "+name+" hash")
    # The cache stores sensitive per-image labels; no claim that labels are public.
    import torch
    cache = torch.load(one_source("cache/cache.pt"),map_location="cpu",weights_only=True)
    cache_summary = read_json(one_source("cache/summary.json"))
    audit.require(sha256(one_source("cache/cache.pt"))==cache_summary["cache_sha256"],"original cache summary binding")
    audit.require(sha256(one_source("cohort/lock.json"))==cache_summary["cohort_lock_sha256"],"original cache cohort binding")
    weight_hash = sha256(one_source("unet/diffusion_pytorch_model.fp16.safetensors"))
    audit.require(weight_hash==cache_summary["model_hashes"]["unet/diffusion_pytorch_model.fp16.safetensors"]["actual"].lower(),
                  "original cache/base model artifact binding")
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_config(read_json(one_source("scheduler/scheduler_config.json")))
    audit.require(scheduler.config.prediction_type=="epsilon","actual scheduler epsilon target")
    with np.load(run_dir/"projection.npz",allow_pickle=False) as saved:
        audit.require(np.array_equal(saved["alphas"],scheduler.alphas_cumprod.cpu().numpy()),
                      "exact independently reconstructed scheduler alpha schedule")
    for image in images:
        prompt = cache["train_prompts"][image["image_id"]]
        audit.require(image["prompt"]==prompt,"original per-image weak-label condition")
        audit.require(tuple(cache["hidden"][prompt].shape)==(1,77,1024),"cached conditioning shape")
        audit.require(tuple(cache["latents"][image["image_id"]].shape)==(1,4,32,32),"cached latent shape")
    del cache
    audit.require(execution["phase"]=="extract" and execution["completed"] is True,"completed main extraction")
    audit.require(execution["records"]==3840 and execution["unet_forward"]==3841 and execution["warmup_forward"]==1,
                  "3840 measured records plus one forward warmup")
    audit.require(execution["unet_backward"]==0 and execution["new_vae_or_text_encoder_forward"]==0,"no backbone backward or extra encoders")
    audit.require(execution["no_backbone_parameter_gradients"] is True and execution["tf32"] is False,"execution gradient/FP32 guards")
    audit.require(execution["contract_sha256"]==sha256(run_dir/"contract.json"),"execution contract binding")
    audit.require(execution["manifest_sha256"]==sha256(run_dir/"manifest.json"),"execution raw-manifest binding")
    audit.require(analysis["schema"]=="frozen-residual-capacity-analysis/v1","analysis schema")
    audit.require(analysis["generative_quality_measured"] is False and analysis["dp_mechanism_executed"] is False,"analysis claim scope")
    audit.require(set(analysis["source_sha256"])=={"contract.json","manifest.json","execution.json","patient_stats.npz","models.npz"},
                  "complete analysis source bindings")
    for name,expected in analysis["source_sha256"].items():
        audit.require(sha256(run_dir/name)==expected,"analysis input hash "+name)
    return {name:sha256(run_dir/name) for name in
            ("contract.json","projection.npz","images.json","manifest.json","execution.json",
             "patient_stats.npz","models.npz","analysis.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()),flush=True)
        return
    if args.run_dir is None:
        parser.error("--run-dir or --self-test is required")
    run_dir = args.run_dir.resolve()
    destination = run_dir/"independent_verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    audit,started = Audit(),time.perf_counter()
    result = {"schema":"independent-frozen-residual-capacity-verification/v1",
              "verifier_source_sha256":sha256(Path(__file__)),
              "independent_verifier_code_frozen_before_gpu":False,
              "numeric_policy":"fixed before GPU; verifier source bound in this artifact after implementation",
              "producer_numerical_function_imports":False,"new_GPU_forward":0,"new_GPU_backward":0}
    try:
        result["input_sha256"] = verify_provenance(run_dir,audit)
        result["verification"] = verify_saved_capacity(run_dir,audit)
        result.update(status=STATUS,complete=True)
    except Exception as error:
        result.update(status="FAIL_SAVED_CAPACITY_VERIFICATION",complete=False,
                      failure=repr(error),traceback=traceback.format_exc())
    result.update(seconds=time.perf_counter()-started,checks=audit.count,comparisons=audit.comparisons)
    result["limits"] = [
        "Saved arithmetic and provenance checks do not re-execute the UNet.",
        "Projection witnesses check four saved inputs, not every hidden activation.",
        "No generation quality, DP utility, final test performance or paper novelty is established.",
        "Patient bootstrap is conditional and pointwise on an already used development cohort.",
        "Model weights are source-bound and producer gradient guards are checked; no independent weight inference is executed."]
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("status","complete","seconds","checks")}),flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
