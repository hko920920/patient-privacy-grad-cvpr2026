"""Post-hoc CPU diagnostic of all five saved CDI denoising-loss repeats.

No model import/call, new image, fitting, gate, or seed/score selection.
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
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
RESEARCH = next(ROOT.parent.glob("*/research_2026-09-10"))
CONTRACT = RESEARCH / "spec_sources/cdi_saved_dl_repeats_contract_20260915b.json"
OUTPUT = RUN / "baseline_screen_20260915/cdi_saved_dl_repeats_v1"
DIRS = {s: RUN / "baseline_screen_20260914" / ("cdi_" + s.lower() + "_cohort_v1") for s in ("E", "U")}
MODELS, SCENARIOS = ("model_1", "model_2"), ("E", "U")
NORMS, POOLS = ("L2", "MSE"), ("mean", "max")
VIEWS = {**{f"single_{i}": [i] for i in range(5)},
         **{f"prefix_{n}": list(range(n)) for n in (1, 2, 4, 5)}}
SHAPE = (1, 4, 32, 32)


def require(x, message):
    if not x:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                          for k, v in r.items()} for r in rows)


def cohort():
    rows = [r for r in csv_rows(RUN / "cohort/evaluation_images.csv") if r["eval_role"] == "selection"]
    group = {}
    for r in rows:
        require(group.setdefault(r["patient_id"], r["assignment_group"]) == r["assignment_group"], "assignment")
    order = sum([sorted((p for p, g in group.items() if g == a), key=int) for a in ("A", "B")], [])
    require(len(rows) == 160 and len(group) == 40
            and [sum(g == a for g in group.values()) for a in ("A", "B")] == [20, 20], "selection40 only")
    return {r["image_id"]: r for r in rows}, order, group


def source_rows():
    out = []
    for scenario, directory in DIRS.items():
        verify = read(directory / "verification.json")
        require(verify["status"] == f"PASS_SAVED_{scenario}_COHORT_ARITHMETIC_AND_INTEGRITY", "prior raw verification")
        for name, field in (("results.json", "results_sha256"), ("protocol.json", "protocol_sha256"),
                            ("execution.json", "execution_sha256")):
            require(sha(directory / name) == verify[field], "prior verified binding")
        out += [(scenario, directory, r) for r in read(directory / "results.json") if r["eval_role"] == "selection"]
    require(len(out) == 320, "fixed saved 320 records")
    return out


def create_contract():
    require(not CONTRACT.exists() and not OUTPUT.exists(), "new immutable diagnostic")
    metadata, order, group = cohort()
    sources = source_rows()
    frozen = {str(Path(__file__).resolve()): sha(__file__)}
    for p in (RUN / "cohort/lock.json", RUN / "cohort/evaluation_images.csv",
              RUN / "cache/summary.json", ROOT / "u_patient_audit/cdi_adapter.py",
              ROOT / "u_patient_audit/verify_cdi_e_cohort.py", ROOT / "u_patient_audit/verify_cdi_u_cohort.py"):
        frozen[str(p.resolve())] = sha(p)
    for model in MODELS:
        for p in (RUN / "training_coverage_v2" / model / "step_1000.pt", RUN / "cohort" / (model + "_train.csv")):
            frozen[str(p.resolve())] = sha(p)
    for directory in DIRS.values():
        for name in ("protocol.json", "results.json", "execution.json", "verification.json"):
            frozen[str((directory / name).resolve())] = sha(directory / name)
    for _, directory, r in sources:
        p = (directory / r["raw_path"]).resolve()
        require(p.is_relative_to(directory.resolve()), "raw source path boundary")
        require(sha(p) == r["raw_sha256"], "saved packet hash")
        frozen[str(p)] = r["raw_sha256"]
    c = dict(schema="cdi-saved-dl-repeat-diagnostic-contract/v1", current_step=2,
        post_hoc=True, prior_selection_results_observed=True, untouched_test=False,
        patient_order=order, assignment=group, models=list(MODELS), scenarios=list(SCENARIOS),
        images=160, records=320, repeats=5, timestep=100, norms=list(NORMS), pools=list(POOLS),
        views=VIEWS, sign="negative image loss first, then patient mean/max",
        L2_policy="saved original FP32 per-repeat norm; FP32 prefix mean exactly follows source DL mean",
        MSE_policy="CPU FP64 mean squared residual from saved FP32 predictions/noise; FP64 prefix mean",
        precision_control="FP64 residual L2 repeats independently recomputed; report AUC/rank sensitivity",
        comparison="each fixed prefix2/4/5 minus prefix1; paired patient bootstrap; no winner selection",
        bootstrap=dict(seed=260914, replicates=2000, strata={"A": 20, "B": 20},
            shared_patient_indices_across_models_scenarios_norms_views_pools=True,
            intervals="percentile95 conditional on these five noises, exploratory and not multiplicity adjusted"),
        variance=dict(pool="mean only; max is nonlinear and receives no additive component estimate",
            W="mean over patients of sample variance(ddof1) across5 single-repeat patient means",
            B="sample variance(ddof1) across patients of their mean over5 repeats",
            between="B-W/5, signed estimate retained; heteroscedastic independent noise model assumption",
            single_noise_variance="W", five_repeat_noise_variance="W/5",
            strata=["all", "A", "B"], truncated_between_not_used=True,
            limit="between-patient variation includes anatomy/pathology/site/labels; not membership signal variance"),
        expected_metrics=144, expected_paired_contrasts=48, expected_patient_scores=5760,
        new_GPU_calls=0, new_backward_calls=0, new_patients=0, fitting=False,
        new_training=False, seed_selection=False, admission_gate=False,
        frozen_sha256=frozen, output_directory=str(OUTPUT.resolve()))
    new_json(CONTRACT, c)
    return c


def bootstrap(order):
    rng = np.random.default_rng(260914)
    draws = np.concatenate([rng.integers(0, 20, (2000, 20), dtype=np.int64),
                            rng.integers(20, 40, (2000, 20), dtype=np.int64)], axis=1)
    weights = np.stack([np.bincount(row, minlength=40) for row in draws]).astype(np.float64)
    return draws, weights


def auc_stats(values, labels, draws, weights):
    v, y = np.asarray(values, dtype=np.float64), np.asarray(labels)
    pos, neg = np.flatnonzero(y), np.flatnonzero(1-y)
    delta = v[pos, None] - v[None, neg]
    credit = (delta > 0).astype(float) + .5*(delta == 0)
    point = float(credit.mean())
    require(abs(point - roc_auc_score(y, v)) <= 1e-14, "sklearn independent point AUC")
    boot = np.einsum("bi,ij,bj->b", weights[:, pos], credit, weights[:, neg], optimize=True)/400
    sampled = v[draws]
    a, b = (sampled[:, :20], sampled[:, 20:]) if y[0] else (sampled[:, 20:], sampled[:, :20])
    d = a[:, :, None] - b[:, None, :]
    direct = np.mean((d > 0) + .5*(d == 0), axis=(1, 2))
    require(np.array_equal(boot, direct), "weighted versus direct shared patient bootstrap")
    return point, boot, credit


def load_repeats(contract):
    metadata, order, groups = cohort()
    require(order == contract["patient_order"] and groups == contract["assignment"], "fixed patient order")
    cache_meta = read(RUN / "cache/summary.json")
    states, patient_exposures = {}, {}
    for model in MODELS:
        states[model] = torch.load(RUN / "training_coverage_v2" / model / "step_1000.pt",
                                  map_location="cpu", weights_only=True)
        manifest = csv_rows(RUN / "cohort" / (model + "_train.csv"))
        ledger = states[model]["exposures"]
        require(states[model]["step"] == 1000 and len(manifest) == 912
                and set(ledger) == {r["image_id"] for r in manifest}
                and sum(ledger.values()) == 4000 and min(ledger.values()) > 0, "actual training ledger")
        patient_exposures[model] = defaultdict(int)
        for r in manifest:
            patient_exposures[model][r["patient_id"]] += ledger[r["image_id"]]
    images, cells, source_l2_exact, noise_checks = {}, [], 0, 0
    worst = 0.0
    noise_by_image = {}
    for scenario, directory, r in source_rows():
        model, iid, p = r["model"], r["image_id"], r["patient_id"]
        require(model in MODELS and iid in metadata and r["scenario"] == scenario, "source scope")
        mr = metadata[iid]
        for key in ("patient_id", "assignment_group", "record_role", "eval_role"):
            require(r[key] == mr[key], "cohort metadata " + key)
        member = int(groups[p] == ("A" if model == "model_1" else "B"))
        ex = int(states[model]["exposures"].get(iid, 0))
        require(r["member"] == member == int(patient_exposures[model][p] > 0)
                and r["image_member"] == int(scenario == "E" and member) == int(ex > 0),
                "actual patient/image membership")
        require(r["actual_training_exposures"] == ex
                and r["patient_training_exposures"] == patient_exposures[model][p], "actual image/patient exposure")
        require(r["cache_sha256"] == cache_meta["cache_sha256"]
                and r["checkpoint_sha256"] == contract["frozen_sha256"][str((RUN / "training_coverage_v2" / model / "step_1000.pt").resolve())],
                "checkpoint and latent-cache binding")
        path = directory / r["raw_path"]
        require(sha(path) == r["raw_sha256"], "immutable per-image raw packet")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        require(raw["image_id"] == iid and raw["stream"] == r["stream"] == "primary", "raw identity")
        preds, noises = raw["predictions"]["denoising_loss"], raw["noise_draws"]["denoising_loss"]
        saved = raw["module_outputs"]["denoising_loss"]
        require(len(preds) == len(noises) == 5 and saved.shape == (1, 5, 1), "five DL repeats")
        require(float(saved.mean(dim=1).squeeze(1)[0]) == r["features"][0]
                and r["feature_names"][0] == "Denoising Loss", "exact original CDI DL scalar")
        source_l2_exact += 1
        fp64_l2, mse, seed_list = [], [], []
        for j, (pred, noise) in enumerate(zip(preds, noises)):
            seed = int.from_bytes(hashlib.sha256(
                f"cdi-v1|260914|{iid}|primary|denoising_loss|{j}".encode()).digest()[:8], "big") % (2**63-1)
            n = noise["noise"]
            require(noise["seed"] == seed and noise["draw"] == j and n.shape == SHAPE and n.dtype == torch.float32,
                    "source image/repeat seed")
            regenerated = torch.randn(SHAPE, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
            require(torch.equal(regenerated, n) and hashlib.sha256(n.numpy().tobytes()).hexdigest() == noise["sha256"],
                    "exact independent saved noise regeneration")
            require(pred["timestep"] == 100 and pred["alpha"] == float(raw["alphas"][100])
                    and pred["grad_enabled"] is False and pred["epsilon"].shape == SHAPE
                    and pred["epsilon"].dtype == torch.float32
                    and torch.equal(pred["epsilon"], pred["raw_prediction"]), "saved epsilon prediction")
            epsilon = pred["epsilon"].double().numpy()
            residual = epsilon - n.double().numpy()
            square_sum = math.fsum(float(x*x) for x in residual.reshape(-1))
            l2, m = math.sqrt(square_sum), square_sum / 4096
            require(math.isfinite(l2) and math.isfinite(m), "finite recomputed loss")
            err = abs(l2-float(saved[0,j,0])); worst = max(worst, err)
            require(err <= 1e-8 + 2e-6*abs(l2), "independent FP64 norm versus saved source norm")
            z, a = raw["latent"].double().numpy(), float(raw["alphas"][100])
            expected = math.sqrt(a)*z + math.sqrt(1-a)*n.double().numpy()
            bound = 32*np.finfo(np.float32).eps*(math.sqrt(a)*np.abs(z)+math.sqrt(1-a)*np.abs(n.double().numpy()))+1e-37
            require(np.all(np.abs(pred["input"].double().numpy()-expected) <= bound), "saved noising arithmetic")
            fp64_l2.append(l2); mse.append(m); seed_list.append(seed); noise_checks += 1
            cells.append(dict(model=model, scenario=scenario, patient_id=p, image_id=iid,
                repeat=j, seed=seed, source_L2=float(saved[0,j,0]), residual_FP64_L2=l2,
                residual_FP64_MSE=m, source_norm_absolute_error=err))
        require(noise_by_image.setdefault(iid, seed_list) == seed_list, "same noises across target models")
        images[model, scenario, p, iid] = dict(L2=saved.reshape(5).numpy().astype(np.float64),
                                            MSE=np.asarray(mse), L2_FP64=np.asarray(fp64_l2))
    require(len(images) == 320 and len(cells) == 1600, "complete fixed records/repeats")
    return images, cells, dict(original_DL_scalar_exact=source_l2_exact, noises_exactly_regenerated=noise_checks,
        residual_norms_checked=noise_checks, maximum_source_L2_absolute_error=worst,
        new_GPU_calls=0, new_backward_calls=0, new_patient_evaluation=0)


def analyse(images, order, groups):
    draws, weights = bootstrap(order)
    patients, metrics, contrasts, variances, sensitivity = [], [], [], [], []
    for model in MODELS:
        labels = np.array([int(groups[p] == ("A" if model == "model_1" else "B")) for p in order])
        for scenario in SCENARIOS:
            keys = [[k for k in images if k[:3] == (model, scenario, p)] for p in order]
            require(all(len(k) == 2 for k in keys), "two images per patient and scenario")
            keys = [sorted(k, key=lambda x: x[3]) for k in keys]
            for norm in NORMS:
                matrix = np.array([[images[k][norm] for k in ks] for ks in keys])
                alt = np.array([[images[k]["L2_FP64"] for k in ks] for ks in keys]) if norm == "L2" else matrix
                single_patient_mean = -matrix.mean(axis=1)
                for stratum in ("all", "A", "B"):
                    ix = np.arange(40) if stratum == "all" else np.flatnonzero(np.array([groups[p] for p in order]) == stratum)
                    q = single_patient_mean[ix]
                    W, B = float(q.var(axis=1, ddof=1).mean()), float(q.mean(axis=1).var(ddof=1))
                    variances.append(dict(model=model, scenario=scenario, norm=norm, patient_pool="mean",
                        stratum=stratum, patients=len(ix), noise_repeats=5,
                        W_mean_within_patient_noise_variance=W, B_variance_of_five_repeat_patient_means=B,
                        estimated_noise_variance_of_mean5=W/5, signed_between_patient_variance=B-W/5,
                        noise_fraction_of_observed_mean5_variance=(W/5/B if B > 0 else None),
                        between_includes_nonmembership_factors=True, max_pool_variance_decomposition=False))
                for pool in POOLS:
                    stats = {}
                    for name, repeats in VIEWS.items():
                        selected = matrix[:,:,repeats]
                        # The original source averages FP32 L2 repeats on CPU.
                        per_image = -torch.from_numpy(selected.astype(np.float32)).mean(dim=2).numpy().astype(np.float64) if norm == "L2" else -selected.mean(axis=2)
                        per_image_alt = -alt[:,:,repeats].mean(axis=2)
                        values = per_image.mean(axis=1) if pool == "mean" else per_image.max(axis=1)
                        alternative = per_image_alt.mean(axis=1) if pool == "mean" else per_image_alt.max(axis=1)
                        point, boots, credit = auc_stats(values, labels, draws, weights)
                        stats[name] = (point, boots)
                        point_alt, _, credit_alt = auc_stats(alternative, labels, draws, weights)
                        sensitivity.append(dict(model=model, scenario=scenario, norm=norm, pool=pool, view=name,
                            saved_path_AUC=point, FP64_path_AUC=point_alt,
                            changed_member_nonmember_pair_credits=int(np.count_nonzero(credit != credit_alt))))
                        metrics.append(dict(model=model, scenario=scenario, norm=norm, pool=pool, view=name,
                            repeats=repeats, nominal_DL_forward_examples_per_patient=2*len(repeats),
                            AUC=point, patient_bootstrap95=np.quantile(boots,[.025,.975],method="linear").tolist(),
                            member_mean=float(values[labels==1].mean()), nonmember_mean=float(values[labels==0].mean())))
                        for j, p in enumerate(order):
                            patients.append(dict(model=model, scenario=scenario, norm=norm, pool=pool, view=name,
                                patient_id=p, assignment_group=groups[p], member=int(labels[j]),
                                image_ids=[k[3] for k in keys[j]], value=float(values[j])))
                    for count in (2,4,5):
                        a,b = stats[f"prefix_{count}"], stats["prefix_1"]
                        delta = a[1]-b[1]
                        contrasts.append(dict(model=model, scenario=scenario, norm=norm, pool=pool,
                            contrast=f"prefix_{count}_minus_prefix_1", delta_AUC=a[0]-b[0],
                            paired_patient_bootstrap95=np.quantile(delta,[.025,.975],method="linear").tolist(),
                            same_forward_budget=False))
    require(len(metrics)==144 and len(contrasts)==48 and len(patients)==5760 and len(variances)==24, "all fixed analyses")
    return dict(metrics=metrics, contrasts=contrasts, variance_components=variances,
        precision_sensitivity=sensitivity, patient_scores=patients,
        bootstrap=dict(patient_order=order, seed=260914, replicates=2000,
            indices_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
            direct_resampling_and_frequency_weighting_all_144_pass=True),
        interpretation="POST_HOC_REPEAT_VARIABILITY_DIAGNOSTIC_NO_ADMISSION_GATE",
        limitations=[
            "Selection40 was repeatedly used for development; intervals are not confirmatory or adjusted for selection/multiplicity.",
            "Bootstrap resamples patients with shared indices and conditions on the five observed image-specific noises.",
            "No bootstrap over target training seeds; target pair shares initialization and swaps whole A/B cohorts.",
            "AUC between-patient variation includes many image and disease factors, not solely membership.",
            "Prefix averages have different query cost; repeat1 and single0 duplicate by design, no best prefix/seed selected.",
            "B-W/5 assumes independent zero-mean image/noise variation and is signed; negative estimates are not forced to zero.",
            "Additive noise variance decomposition is given only for linear patient mean, never patient max.",
            "L2 and MSE are different aggregate scores; MSE is not the original CDI DL feature.",
            "This is one timestep t100 under generic conditioning and fixed cached VAE mode; not all diffusion MIA or MoFit.",
            "No outcome establishes individual-patient causal training effect, absence of leakage, or a new attack contribution."])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--create-contract",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    require(args.create_contract != args.run, "choose contract preparation or actual CPU analysis")
    torch.set_num_threads(4)
    if args.create_contract:
        c=create_contract()
        print(json.dumps(dict(status="CONTRACT_FROZEN_BEFORE_REPEAT_ANALYSIS",contract_sha256=sha(CONTRACT),
                              files=len(c["frozen_sha256"]),code_sha256=sha(__file__))))
        return
    c=read(CONTRACT)
    require(not OUTPUT.exists(), "immutable new output directory")
    for path,h in c["frozen_sha256"].items():
        require(sha(path)==h,"frozen input/code changed: "+path)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    new_json(OUTPUT/"protocol.json",dict(contract=c,contract_path=str(CONTRACT),contract_sha256=sha(CONTRACT),
                                       code_sha256=sha(__file__),GPU=False))
    try:
        images,cells,checks=load_repeats(c)
        a=analyse(images,c["patient_order"],c["assignment"])
        patient_scores=a.pop("patient_scores")
        a.update(schema="cdi-saved-dl-repeats-analysis/v1",status="PASS_CPU_SAVED_REPEAT_ARITHMETIC_AND_STATISTICS",
            records=320,patients=40,noise_cells=1600,post_hoc=True,checks=checks,
            no_best_seed_selected=True,new_training=False,new_GPU_calls=0,
            current_step2_complete=False,automatic_followup=False)
        new_json(OUTPUT/"analysis.json",a)
        new_json(OUTPUT/"per_repeat_scores.json",cells)
        new_json(OUTPUT/"patient_scores.json",patient_scores)
        for name,key in (("metrics","metrics"),("contrasts","contrasts"),("variance_components","variance_components")):
            write_csv(OUTPUT/(name+".csv"),a[key])
        new_json(OUTPUT/"verification.json",dict(status="PASS_TWO_CPU_AUC_AND_BOOTSTRAP_PATHS",
            scope="same new module; independent arithmetic/sklearn plus frequency-weighted/direct resampling, not separate verifier software",
            raw_DL_exact_checks=320,residual_noise_checks=1600,AUC_checks=144,bootstrap_checks=144,
            precision_changed_pair_credits=sum(q["changed_member_nonmember_pair_credits"] for q in a["precision_sensitivity"]),
            no_neural_backward_verification=True))
        for path,h in c["frozen_sha256"].items():
            require(sha(path)==h,"input/code changed during CPU analysis")
        new_json(OUTPUT/"provenance.json",dict(status="PASS_COMPLETE_CPU_DIAGNOSTIC",
            seconds=time.perf_counter()-start,code_sha256=sha(__file__),contract_sha256=sha(CONTRACT),
            input_files=len(c["frozen_sha256"]),
            output_sha256={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()}))
        print(json.dumps(dict(status="PASS_COMPLETE_CPU_DIAGNOSTIC",seconds=read(OUTPUT/"provenance.json")["seconds"],
            metrics=len(a["metrics"]),contrasts=len(a["contrasts"]),variance_groups=len(a["variance_components"]),
            raw_checks=checks,precision_changed_pair_credits=read(OUTPUT/"verification.json")["precision_changed_pair_credits"])))
    except BaseException as exc:
        new_json(OUTPUT/"failure.json",dict(status="FAILED_PRESERVED_CPU_DIAGNOSTIC",
            error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise


if __name__=="__main__":
    main()
