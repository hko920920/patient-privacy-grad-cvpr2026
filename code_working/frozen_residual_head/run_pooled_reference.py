"""Fixed public32 + private80 non-DP reference; cached CPU statistics only.

This is a controlled NIH role simulation and reused development evaluation.
No generated utility, DP release, or held-out upper-bound claim is made.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "_reports/frozen_residual_capacity_20260916_v1"
PUB = ROOT / "_reports/frozen_residual_public32_20260916_v1"
CAL = PUB / "dp_calibration_v1"
DP = ROOT / "_reports/frozen_residual_patient_dp_20260916_v1"
COHORT = ROOT / "_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv"
OUT = ROOT / "_reports/frozen_residual_pooled_reference_20260916_v1"
FAMILIES = ("static", "full")
REFERENCES = ("base", "public_only", "private_only", "pooled")
RIDGE = .001


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def npz(path, arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def preflight():
    c, p = read(CAP / "contract.json"), read(PUB / "contract.json")
    fields = ["model_id", "model_revision", "model_policy", "prediction_type",
              "precision", "tf32", "tap", "feature_policy", "projection_seed",
              "method_dimensions", "time_basis_logsnr_min", "time_basis_logsnr_max",
              "timestep_policy", "draws_per_image", "ridge_lambda", "prompt_policy",
              "metric", "projection_sha256", "seed_salt"]
    for key in fields:
        assert c[key] == p[key], key
    assert c["ridge_lambda"] == RIDGE
    assert c["images_per_patient"] == 4 and p["images_per_patient"] == 2
    for folder, contract in ((CAP, c), (PUB, p)):
        v = read(folder / "independent_verification.json")
        assert v["status"].startswith("PASS") and v["complete"]
        for name in ("contract.json", "patient_stats.npz", "projection.npz", "images.json"):
            assert sha(folder / name) == v["input_sha256"][name], name
        assert sha(folder / "projection.npz") == contract["projection_sha256"]
    with np.load(CAP / "patient_stats.npz") as bank, np.load(PUB / "patient_stats.npz") as pub:
        assert set(bank["splits"]) == {"train", "eval"}
        assert set(pub["splits"]) == {"public"}
        ids = {"private": bank["patient_ids"][bank["splits"] == "train"].tolist(),
               "development": bank["patient_ids"][bank["splits"] == "eval"].tolist(),
               "public": pub["patient_ids"].tolist()}
        for role, n in (("private", 80), ("development", 40), ("public", 32)):
            assert len(ids[role]) == n and len(set(ids[role])) == n
        assert len(set(sum(ids.values(), []))) == 152
    with COHORT.open(encoding="utf-8-sig", newline="") as f:
        roles = {}
        for row in csv.DictReader(f):
            roles.setdefault(row["eval_role"], set()).add(row["patient_id"])
    assert set(ids["private"]) == roles["fit"]
    assert set(ids["development"]) == roles["selection"]
    assert not set(ids["public"]) & set().union(*roles.values())
    assert len(roles["calibration"]) == 140 and len(roles["test"]) == 140
    assert not set(sum(ids.values(), [])) & (roles["calibration"] | roles["test"])
    return ids, fields


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    ids, fields = preflight()
    sources = [Path(__file__), Path(__file__).with_name("verify_pooled_reference.py"), COHORT,
               CAP / "models.npz", CAL / "public_only_models.npz",
               CAL / "independent_selection_verification.json", DP / "controls.npz",
               DP / "evaluation_mse.npz", DP / "independent_verification.json"]
    for folder in (CAP, PUB):
        sources.extend(folder / name for name in ("contract.json", "patient_stats.npz",
                       "projection.npz", "images.json", "independent_verification.json"))
    digests = {str(p): sha(p) for p in sources}
    out.mkdir(parents=True)
    save(out / "contract.json", dict(
        schema="fixed-pooled-reference/v1", created_utc=now(), current_stage=2,
        source_sha256=digests, families=list(FAMILIES), references=list(REFERENCES), ridge=RIDGE,
        patient_counts={"public": 32, "private": 80, "development": 40}, patient_ids=ids,
        public_weight=32/112, private_weight=80/112,
        pooling="equal patients; weighted group means (32*public + 80*private)/112",
        objective="patient mean of channel-SUM residual loss + .001*||W||_F^2",
        metric="unregularized development quadratic /4, then equal-patient mean",
        expected_new_models=2, existing_reference_refits=4, saved_weights_including_zero=8,
        matching_source_fields=fields, samples_per_patient={"public": 2, "private": 4, "development": 4},
        draws_per_image=8, tuning_allowed=False, search_over_lambda_rho_patients=False,
        dp_applied=False, generated_images=0, expected_new_backbone_forward=0,
        expected_new_backbone_backward=0, raw_images_opened=False,
        cohort_csv_use="IDs and roles only; no calibration/test image/statistic evaluation",
        development_scope="previously reused development40, not final or new independent test",
        public_private_scope="role simulation within public NIH data",
        reference_scope="pooled is an exact regularized finite-training optimum, not held-out/generation upper bound",
        comparisons=["pooled_vs_public_only", "pooled_vs_private_only", "pooled_vs_base"],
        comparison_sign="reference minus pooled MSE; positive means lower pooled error",
        inference="descriptive paired differences only; no significance or generalization test",
        numeric_tolerances={"statistics_weights_rtol": 1e-10, "statistics_weights_atol": 1e-12,
                            "mse_atol": 1e-12, "stationarity_absolute": 1e-10},
        producer_and_independent_verifier_frozen_before_pooled_execution=True,
        numpy_version=np.__version__, preflight_seconds=time.perf_counter()-started))
    print(json.dumps({"phase": "prepared", "contract_sha256": sha(out / "contract.json")}))


def mse(a, b, q, w):
    return (q - 2*np.einsum("ij,nij->n", w, b)
            + np.einsum("ij,nij->n", w, np.einsum("nij,jk->nik", a, w))) / 4


def objective(a, b, q, w):
    return float(q - 2*np.sum(w*b) + np.sum(w*(a@w)) + RIDGE*np.sum(w*w))


def run(out):
    contract = read(out / "contract.json")
    for name, digest in contract["source_sha256"].items():
        assert sha(name) == digest, name
    for name in ("execution.json", "models.npz", "pooled_statistics.npz", "evaluation_mse.npz", "analysis.json"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    started_utc, started = now(), time.perf_counter()
    weights, statistics, analysis, timings = {}, {}, {}, {}
    with np.load(CAP / "patient_stats.npz") as bank, np.load(PUB / "patient_stats.npz") as pub, \
            np.load(CAP / "models.npz") as old_private, np.load(CAL / "public_only_models.npz") as old_public, \
            np.load(DP / "controls.npz") as controls, np.load(DP / "evaluation_mse.npz") as old_eval:
        train, dev = bank["splits"] == "train", bank["splits"] == "eval"
        losses = {"patient_ids": bank["patient_ids"][dev]}
        assert np.array_equal(losses["patient_ids"], old_eval["patient_ids"])
        for family in FAMILIES:
            aggregates = {}
            for group, source, mask in (("public_only", pub, slice(None)), ("private_only", bank, train)):
                aggregates[group] = tuple(source[k+"_"+family][mask].mean(axis=0) for k in ("A", "B", "Q"))
            aggregates["pooled"] = tuple((32*u + 80*v)/112 for u,v in zip(aggregates["public_only"], aggregates["private_only"]))
            for k, value in zip(("A", "B", "Q"), aggregates["pooled"]):
                statistics[k+"_"+family] = value
            refs = {}
            for name in REFERENCES:
                clock = time.perf_counter()
                if name == "base":
                    w = np.zeros_like(aggregates["pooled"][1])
                else:
                    a, b, q = aggregates[name]
                    assert np.linalg.eigvalsh(a + RIDGE*np.eye(len(a)))[0] > 0
                    w = np.linalg.solve(a + RIDGE*np.eye(len(a)), b)
                timings[family+"_"+name] = time.perf_counter()-clock
                if name in ("public_only", "private_only"):
                    old = old_public["W_"+family] if name == "public_only" else old_private["W_"+family]
                    np.testing.assert_allclose(w, old, rtol=1e-10, atol=1e-12)
                    old_key = family+("_public_only" if name == "public_only" else "_nonprivate")
                    np.testing.assert_allclose(w, controls[old_key], rtol=1e-10, atol=1e-12)
                weights["W_"+family+"_"+name] = w
                values = mse(bank["A_"+family][dev], bank["B_"+family][dev], bank["Q_"+family][dev], w)
                assert np.all(np.isfinite(values)) and np.all(values >= 0)
                losses[family+"_"+name] = values
                if name in ("public_only", "private_only"):
                    np.testing.assert_allclose(values, old_eval["control_"+old_key], rtol=0, atol=1e-12)
                if name == "base":
                    np.testing.assert_allclose(values, old_eval["base_mse"], rtol=0, atol=1e-12)
                refs[name] = {"mean_mse": float(values.mean()),
                              "pooled_training_regularized_objective": objective(*aggregates["pooled"], w)}
                if name != "base":
                    a, b, _ = aggregates[name]
                    refs[name]["own_objective_stationarity_norm"] = float(np.linalg.norm((a+RIDGE*np.eye(len(a)))@w-b))
            comparisons = {}
            pooled = losses[family+"_pooled"]
            for name in ("public_only", "private_only", "base"):
                ref = losses[family+"_"+name]
                delta = ref-pooled
                comparisons["pooled_vs_"+name] = dict(
                    mean_improvement=float(delta.mean()), relative_reduction_percent=float(100*delta.mean()/ref.mean()),
                    patient_improved=int(np.sum(delta>0)), patient_tied=int(np.sum(delta==0)),
                    patient_worse=int(np.sum(delta<0)), median_improvement=float(np.median(delta)),
                    minimum_improvement=float(delta.min()), maximum_improvement=float(delta.max()))
                assert refs["pooled"]["pooled_training_regularized_objective"] <= refs[name]["pooled_training_regularized_objective"]+1e-12
            analysis[family] = {"references": refs, "paired_comparisons": comparisons}
    npz(out / "models.npz", weights)
    npz(out / "pooled_statistics.npz", statistics)
    npz(out / "evaluation_mse.npz", losses)
    save(out / "analysis.json", dict(schema="fixed-pooled-reference-analysis/v1", families=analysis,
         development_patients=40, generated_quality_measured=False, significance_test_performed=False,
         paired_difference_sign="reference minus pooled MSE; positive is improvement"))
    save(out / "execution.json", dict(completed=True, started_utc=started_utc, completed_utc=now(),
         contract_sha256=sha(out / "contract.json"), seconds=time.perf_counter()-started,
         per_reference_linear_solve_seconds=timings, new_pooled_models=2, reference_refits=4,
         backbone_forward=0, backbone_backward=0, generated_images=0,
         output_sha256={n:sha(out/n) for n in ("models.npz", "pooled_statistics.npz", "evaluation_mse.npz", "analysis.json")}))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else run)(args.out)
