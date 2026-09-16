"""Public-only calibration for the frozen residual-head patient-DP comparison.

No protected80 or development40 statistics are read. Public train16 are repeated
five times as an 80-slot calibration surrogate, NOT 80 independent patients.
All choices use two public folds and three fixed noise repeats. This is research
calibration, not a DP release. Run prepare -> profile -> run in separate phases.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

if __name__ == "__main__":
    for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_thread_name] = "1"
import numpy as np
from . import dp_mechanisms as dm
from . import residual_math as rm

CODE = Path(__file__).resolve().parents[1]
PUBLIC = CODE / "_reports/frozen_residual_public32_20260916_v1"
DEFAULT_OUT = PUBLIC / "dp_calibration_v1"
RESEARCH = CODE.parent / "CVPR 주제 탐색/research_2026-09-10"
ACCOUNT_SOURCE = CODE / "dp_protocol/build_xray_dp_attack_protocol.py"
ACCOUNT_CACHE = RESEARCH / "spec_sources/track1_dp_accounting_precheck_20260916.json"
EXPECTED_PUBLIC_STATS_SHA = "396575b651bdca3b33aef1f94562154c5b8e3acfb15eb1dfc0bb294ace32e974"
PUBLIC_PASS = "PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE"
HEADS = ("full", "static")
N0, Q, RIDGE = 80, .1, .001
NOISE_REPEATS = (260916101, 260916102, 260916103)
T_CANDIDATES = (500, 2000, 8000)
CLIP_QUANTILES = (.5, .9)
LR_MULTIPLIERS = (.25, .5, 1.)
FLOOR_MULTIPLIERS = (0., 1., 2.)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_npz(path, arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def seed(*parts):
    text = "|".join(map(str, ("public-dp-calibration-v1",) + parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def learning_rates(steps, initial_lr):
    """First floor(.8*T) updates constant; final updates linear to .01*eta0."""
    if isinstance(steps, bool) or int(steps) != steps or steps < 5:
        raise ValueError("integer steps >=5 required")
    if not math.isfinite(initial_lr) or initial_lr <= 0:
        raise ValueError("positive finite initial learning rate required")
    steps = int(steps)
    cut = int(math.floor(.8*steps))
    rates = np.full(steps, float(initial_lr), dtype=np.float64)
    # The first decay update is below eta0; the last is exactly .01*eta0.
    rates[cut:] *= np.linspace(1., .01, steps-cut+1)[1:]
    return rates


def code_bindings():
    return {str(p): digest(p) for p in
            (Path(__file__), Path(dm.__file__), Path(rm.__file__), ACCOUNT_SOURCE)}


def require_bindings(mapping):
    for name, expected in mapping.items():
        if digest(name) != expected:
            raise RuntimeError("Frozen source/input changed: " + name)


def policy():
    return dict(
        schema="frozen-residual-public-dp-selection/v1", created_utc=stamp(),
        public_directory=str(PUBLIC), public_denominator=N0,
        protected_statistics_read=False, development_statistics_read=False,
        adjacency="ADD_OR_REMOVE_ONE", epsilon=8., delta=1e-5, q=Q,
        heads=list(HEADS), ridge=RIDGE, public_patients=32,
        patient_images=2, draws_per_image=8, output_channels=4, public_split_label="public",
        fold_rule="sort patient IDs by SHA256(public-dp-fold-v1|260916|pid); first16 vs last16; two-way",
        surrogate="each public train16 statistic repeated exactly5; 80 PUBLIC slots, not independent patients",
        scale_rule="median patient Frobenius norm for A and B separately; zero/nonfinite abort",
        quantile_method="higher", clip_quantiles=list(CLIP_QUANTILES),
        ssp_floor_multipliers=list(FLOOR_MULTIPLIERS),
        gram_noise_proxy="nu_A=(sigma*C*a0/N0)*sqrt(2*d); scale proxy, NOT a probability bound",
        sgd_gradient_reference_positions=[0., .25, .5, 1.],
        sgd_gradient_reference="four straight-line points from W0=0 to public train-fold ridge; pool patient norms",
        L_public="2*(lambda_max(mean A)+ridge)",
        sgd_lr_multipliers=list(LR_MULTIPLIERS),
        schedule="constant80_linear20_end01",
        schedule_exact="first floor(.8*T) constant; final segment linspace(1,.01,T-floor(.8*T)+1)[1:]",
        convergence_steps=list(T_CANDIDATES), convergence_reference_lr_multiplier=.5,
        convergence="same Poisson q/denominator/optimizer; sigma0, clipping disabled, W0=0",
        convergence_metric="(W-Wstar)'(Abar+lambda I)(W-Wstar)/(Wstar'(Abar+lambda I)Wstar)",
        convergence_threshold=.01,
        convergence_selection="first T for which EACH public fold's three-seed MEAN normalized gap <=.01",
        convergence_failure="preserve all attempted results; affected head unselected; no forced private GO",
        noise_repeats=list(NOISE_REPEATS),
        candidate_count_per_head_method=6,
        candidate_selection="minimum arithmetic mean of 2fold*3repeat validation patient MSE; exact ties candidate order",
        candidate_order={"SSP":"quantile ascending then floor multiplier ascending",
                         "SGD":"quantile ascending then LR multiplier ascending"},
        randomization="same base repeat IDs; independent derived Poisson/Gaussian streams; within-method CRN across candidates",
        validation="held-out16 public patients, equal patients, unregularized channel-sum loss/4",
        final_recompute="selected quantiles/multipliers transferred; scales/C/L/public ridge recomputed from all public32",
        public_only_baseline="all-public32 nonprivate ridge full/static; W0 for main four comparisons remains0",
        disclosure="public data only calibration; seeded simulations are not confidential DP releases",
        max_seed_selection=False, private_optimization=False,
        candidate_failure="retain failure; candidate ineligible unless all6 fold/repeat fits finite",
        thread_limit=1, source_sha256=code_bindings(),
        accountant_cache_path=str(ACCOUNT_CACHE),
        accountant_cache_sha256=digest(ACCOUNT_CACHE),
        expected_public_stats_sha256=EXPECTED_PUBLIC_STATS_SHA,
        profile="one public full-head fold,500 clipped noisy steps, fixed .9 quantile and .5/L; timing only, no candidate change",
        limitations=[
            "public2-photo/16-prototype surrogate does not reproduce private4-photo/80-patient distribution",
            "small public candidate portfolio is not a proof of optimal DP-SGD or SSP",
            "positive PSD floor changes the protected surrogate, not just floating point precision",
            "convergence check is optimization sanity, not a privacy or efficacy certificate",
            "all public noise repeats are averaged; no best-seed selection"])


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    write_json(out/"selection_contract.json", policy())
    print(json.dumps({"status":"PREPARED_PUBLIC_ONLY_SELECTION_CONTRACT",
                      "contract_sha256":digest(out/"selection_contract.json")}), flush=True)


def validate(out):
    c = read_json(out/"selection_contract.json")
    if c["schema"] != "frozen-residual-public-dp-selection/v1":
        raise ValueError("Wrong selection contract")
    require_bindings(c["source_sha256"])
    if digest(ACCOUNT_CACHE) != c["accountant_cache_sha256"]:
        raise RuntimeError("Accountant cache changed")
    if digest(PUBLIC/"patient_stats.npz") != c["expected_public_stats_sha256"]:
        raise RuntimeError("Public statistics hash mismatch")
    verification_path = PUBLIC/"independent_verification.json"
    v = read_json(verification_path)
    if v.get("status") != PUBLIC_PASS:
        raise RuntimeError("Independent public raw/statistics verification has not passed")
    # Bound the exact public verification and every source artifact before values are loaded.
    names = ["patient_stats.npz", "contract.json", "manifest.json", "images.json",
             "analysis.json", "execution.json", "independent_verification.json"]
    bindings = {str(PUBLIC/name): digest(PUBLIC/name) for name in names}
    for name in names:
        if name != "independent_verification.json" and v["input_sha256"][name] != bindings[str(PUBLIC/name)]:
            raise RuntimeError("Public verifier input binding mismatch: " + name)
    return c, bindings


def load_public():
    with np.load(PUBLIC/"patient_stats.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["patient_ids"]]
        splits = [str(x) for x in z["splits"]]
        if len(ids) != 32 or len(set(ids)) != 32 or set(splits) != {"public"}:
            raise ValueError("Expected exactly32 public patients")
        data = {h:{k:np.array(z[k+"_"+h], dtype=np.float64, copy=True)
                   for k in ("A","B","Q")} for h in HEADS}
    for h,d in data.items():
        dim = 64 if h == "full" else 16
        if d["A"].shape != (32,dim,dim) or d["B"].shape != (32,dim,4) or d["Q"].shape != (32,):
            raise ValueError("Public statistic shapes")
        if not all(np.isfinite(v).all() for v in d.values()):
            raise ValueError("Nonfinite public statistics")
        if not np.allclose(d["A"], d["A"].transpose(0,2,1), atol=1e-12, rtol=1e-12):
            raise ValueError("Asymmetric public A")
    ordered = sorted(range(32), key=lambda i:hashlib.sha256(
        ("public-dp-fold-v1|260916|"+ids[i]).encode()).hexdigest())
    folds = [(ordered[:16],ordered[16:]),(ordered[16:],ordered[:16])]
    return ids, data, folds


def subset(data, indices):
    return {k:v[indices] for k,v in data.items()}


def reference(data):
    a,b = data["A"].mean(axis=0), data["B"].mean(axis=0)
    w = rm.solve_ridge(a,b,RIDGE)
    L = 2*(float(np.linalg.eigvalsh(a)[-1])+RIDGE)
    a0 = float(np.median(np.linalg.norm(data["A"],axis=(1,2))))
    b0 = float(np.median(np.linalg.norm(data["B"],axis=(1,2))))
    if min(L,a0,b0) <= 0 or not np.isfinite([L,a0,b0]).all():
        raise ValueError("Nonpositive public calibration scale")
    norms = np.sqrt((np.linalg.norm(data["A"],axis=(1,2))/a0)**2+
                    (np.linalg.norm(data["B"],axis=(1,2))/b0)**2)
    gradient_norms = np.concatenate([
        np.linalg.norm(dm.patient_gradients(data["A"],data["B"],f*w),axis=(1,2))
        for f in (0.,.25,.5,1.)])
    return dict(W=w,L=L,a0=a0,b0=b0,stat_norms=norms,gradient_norms=gradient_norms)


def high_quantile(values, fraction):
    value = float(np.quantile(values, fraction, method="higher"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Nonpositive public clipping quantile")
    return value


def mse(data, W):
    a,b,q = data["A"],data["B"],data["Q"]
    values = (q-2*np.einsum("ij,nij->n",W,b)+
              np.einsum("ij,njk,ki->n",W.T,a,W))/4
    # Explicit shape-independent form is checked in self_test against residual_math.
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite public validation loss")
    return values


def normalized_gap(data,W,Wstar):
    H = data["A"].mean(axis=0)+RIDGE*np.eye(W.shape[0])
    e = W-Wstar
    numerator = float(np.sum(e*(H@e)))
    denominator = float(np.sum(Wstar*(H@Wstar)))
    if denominator <= 0:
        if numerator == 0:
            return 0.
        raise ValueError("Zero public objective improvement prevents normalized gap")
    return numerator/denominator


def ssp_parameters(ref, dim, quantile, floor_multiplier, sigma):
    C = high_quantile(ref["stat_norms"],quantile)
    nu = sigma*C*ref["a0"]/N0*math.sqrt(2*dim)
    return dict(a0=ref["a0"],b0=ref["b0"],clip_norm=C,sigma=sigma,ridge=RIDGE,
                eigenvalue_floor=floor_multiplier*nu,public_denominator=N0,
                clip_quantile=quantile,floor_multiplier=floor_multiplier,nu_A=nu)


def sgd_parameters(ref, quantile, multiplier, sigma, steps):
    return dict(clip_norm=high_quantile(ref["gradient_norms"],quantile),
                sigma=sigma,ridge=RIDGE,q=Q,steps=steps,
                initial_lr=multiplier/ref["L"],lr_multiplier=multiplier,
                L_public=ref["L"],public_denominator=N0,
                schedule="constant80_linear20_end01",clip_quantile=quantile)


def ssp_fit(data, params, repeat, head, fold):
    a,b = np.repeat(data["A"],5,axis=0),np.repeat(data["B"],5,axis=0)
    return dm.ssp_release(a,b,**{k:params[k] for k in
        ("a0","b0","clip_norm","sigma","public_denominator","ridge","eigenvalue_floor")},
        seed=seed("SSP",head,fold,repeat,"gaussian"),include_diagnostics=True)


def sgd_fit(data, params, repeat, head, fold, *, unclipped=False, diagnostics=True):
    a,b = np.repeat(data["A"],5,axis=0),np.repeat(data["B"],5,axis=0)
    return dm.poisson_sgd(a,b,steps=params["steps"],q=Q,public_denominator=N0,
        clip_norm=float("inf") if unclipped else params["clip_norm"],
        sigma=0. if unclipped else params["sigma"],ridge=RIDGE,
        learning_rates=learning_rates(params["steps"],params["initial_lr"]),
        sampling_seed=seed("SGD",head,fold,repeat,"poisson"),
        noise_seed=seed("SGD",head,fold,repeat,"gaussian"),
        include_diagnostics=diagnostics)


def compact_diagnostics(result, data, ref):
    d = result["diagnostics"]
    if "weight_history" not in d:
        return dict(clipped_patients=d["clipped_patients"],
                    patient_count=d["patient_count"],
                    stationarity_norm=d["stationarity_norm"],
                    min_noisy_eigenvalue=float(d["noisy_gram_eigenvalues"][0]),
                    max_noisy_eigenvalue=float(d["noisy_gram_eigenvalues"][-1]),
                    weight_norm=float(np.linalg.norm(result["W"])))
    history = d["weight_history"]
    points = sorted(set([0,1,len(history)-1]+
                        np.linspace(0,len(history)-1,12,dtype=int).tolist()))
    return dict(patient_gradient_evaluations=d["patient_gradient_evaluations"],
                empty_batches=d["empty_batches"],
                clipped_patient_inclusions=int(d["clipped_patients"].sum()),
                max_sampled_gradient_norm=float(d["maximum_sampled_gradient_norm"].max()),
                weight_norm=float(np.linalg.norm(result["W"])),
                monitors=[dict(step=int(t),normalized_public_objective_gap=normalized_gap(
                    data,history[t],ref["W"])) for t in points])


def account(q,steps,cache,records):
    key=f"q{q}_T{steps}"
    if key in records:
        return records[key]
    if cache["original_source"]["sha256_after"] != digest(ACCOUNT_SOURCE):
        raise RuntimeError("Cached accountant source mismatch")
    matches=[r for r in cache["results"] if r["poisson_sample_rate"]==q and r["steps"]==steps]
    if matches:
        row = dict(matches[0], reused_public_accounting_cache=True)
    else:
        spec=importlib.util.spec_from_file_location("_public_dp_original_accountant",ACCOUNT_SOURCE)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        capture=io.StringIO()
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as captured, redirect_stderr(capture):
            warnings.simplefilter("always")
            measured=module.calibrate_dual(8.,q,steps,1e-5)
        row=dict(measured,target_epsilon=8.,target_delta=1e-5,
                 poisson_sample_rate=q,steps=steps,seconds=time.perf_counter()-started,
                 reused_public_accounting_cache=False,
                 python_warnings=[str(w.message) for w in captured],
                 stderr=capture.getvalue(),original_source_sha256=digest(ACCOUNT_SOURCE))
    if row["conservative_epsilon"] > 8.+1e-10:
        raise RuntimeError("Accounted epsilon exceeds fixed budget")
    records[key]=row
    return row


def begin_phase(out,phase):
    c,inputs=validate(out)
    path=out/(phase+"_protocol.json")
    write_json(path,dict(schema="public-dp-calibration-phase/v1",phase=phase,
        started_utc=stamp(),selection_contract_sha256=digest(out/"selection_contract.json"),
        input_sha256=inputs,source_sha256=code_bindings(),
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
                         platform=platform.platform(),thread_environment={k:os.environ.get(k) for k in
                         ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}),
        private_data_read=False,GPU_calls=0))
    return c,inputs


def profile(out):
    c,inputs=begin_phase(out,"profile")
    ids,data,folds=load_public()
    train=subset(data["full"],folds[0][0]); ref=reference(train)
    cache=read_json(ACCOUNT_CACHE); accounts={}
    sig=account(Q,500,cache,accounts)["noise_multiplier"]
    par=sgd_parameters(ref,.9,.5,sig,500)
    t=time.perf_counter()
    # Timed public arithmetic; its objective value is not used to choose parameters.
    result=sgd_fit(train,par,NOISE_REPEATS[0],"full",0)
    seconds=time.perf_counter()-t
    write_json(out/"profile.json",dict(status="PASS_PUBLIC_ONLY_TIMING",
        steps=500,seconds=seconds,seconds_per_step=seconds/500,
        num_public_prototypes=16,num_surrogate_slots=80,
        patient_gradient_evaluations=result["diagnostics"]["patient_gradient_evaluations"],
        empty_batches=result["diagnostics"]["empty_batches"],
        finite_weights=bool(np.isfinite(result["W"]).all()),
        parameter_selection_performed=False,protocol_sha256=digest(out/"profile_protocol.json"),
        input_sha256=inputs))
    require_bindings(inputs)
    print(json.dumps({"phase":"profile","seconds":seconds,"steps":500}),flush=True)


def run(out):
    started=time.perf_counter()
    c,inputs=begin_phase(out,"run")
    ids,data,folds=load_public()
    write_json(out/"folds.json",[dict(fold=i,train_patient_ids=[ids[j] for j in tr],
              validation_patient_ids=[ids[j] for j in va]) for i,(tr,va) in enumerate(folds)])
    references={h:[reference(subset(data[h],tr)) for tr,_ in folds] for h in HEADS}
    final_ref={h:reference(data[h]) for h in HEADS}
    write_npz(out/"public_only_models.npz",{"W_"+h:final_ref[h]["W"] for h in HEADS})
    write_json(out/"public_only.json",dict(
        description="all-public32 nonprivate ridge; main DP comparison W0 remains0",
        patient_ids=ids,ridge=RIDGE,
        train_mse={h:float(mse(data[h],final_ref[h]["W"]).mean()) for h in HEADS},
        independent_validation_claim=False,model_sha256=digest(out/"public_only_models.npz")))
    cache=read_json(ACCOUNT_CACHE); accounts={}
    ssp_sigma=account(1.,1,cache,accounts)["noise_multiplier"]
    convergence=[]; selected_steps={}; weights={}; traces=[]
    for h in HEADS:
        for T in T_CANDIDATES:
            fold_means=[]
            for fold,(tr,va) in enumerate(folds):
                dt=subset(data[h],tr); ref=references[h][fold]; gaps=[]
                for repeat in NOISE_REPEATS:
                    key=f"conv_{h}_T{T}_f{fold}_r{repeat}"
                    par=sgd_parameters(ref,.9,.5,0.,T)
                    tick=time.perf_counter()
                    result=sgd_fit(dt,par,repeat,h,fold,unclipped=True)
                    gap=normalized_gap(dt,result["W"],ref["W"]); gaps.append(gap)
                    weights[key]=result["W"]
                    convergence.append(dict(key=key,head=h,steps=T,fold=fold,repeat=repeat,
                        normalized_gap=gap,seconds=time.perf_counter()-tick,
                        initial_lr=par["initial_lr"],diagnostics=compact_diagnostics(result,dt,ref)))
                fold_means.append(float(np.mean(gaps)))
            passed=all(x<=.01 for x in fold_means)
            print(json.dumps(dict(phase="convergence",head=h,steps=T,
                                  fold_mean_gaps=fold_means,passed=passed)),flush=True)
            if passed:
                selected_steps[h]=T
                break
    write_json(out/"convergence.json",dict(records=convergence,selected_steps=selected_steps,
        criterion=c["convergence_selection"],threshold=.01))
    trials=[]; selected={}; candidate_summaries=[]
    for h in HEADS:
        selected[h]={"SSP":None,"SGD":None}
        if h not in selected_steps:
            continue
        T=selected_steps[h]
        sgdsigma=account(Q,T,cache,accounts)["noise_multiplier"]
        for method in ("SSP","SGD"):
            candidates=[(cq,m) for cq in CLIP_QUANTILES for m in
                        (FLOOR_MULTIPLIERS if method=="SSP" else LR_MULTIPLIERS)]
            summaries=[]
            for candidate_id,(cq,mult) in enumerate(candidates):
                vals=[]; errors=[]
                for fold,(tr,va) in enumerate(folds):
                    dt,dv=subset(data[h],tr),subset(data[h],va); ref=references[h][fold]
                    par=(ssp_parameters(ref,dt["B"].shape[1],cq,mult,ssp_sigma) if method=="SSP"
                         else sgd_parameters(ref,cq,mult,sgdsigma,T))
                    for repeat in NOISE_REPEATS:
                        key=f"cv_{h}_{method}_c{candidate_id}_f{fold}_r{repeat}"
                        tick=time.perf_counter()
                        try:
                            result=(ssp_fit(dt,par,repeat,h,fold) if method=="SSP"
                                    else sgd_fit(dt,par,repeat,h,fold))
                            score=mse(dv,result["W"])
                            value=float(np.mean(score)); vals.append(value); weights[key]=result["W"]
                            trials.append(dict(key=key,head=h,method=method,candidate_id=candidate_id,
                                fold=fold,repeat=repeat,params=par,validation_mse=value,
                                validation_patient_mse=score.tolist(),seconds=time.perf_counter()-tick,
                                status="FINITE",diagnostics=compact_diagnostics(result,dt,ref)))
                        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                            errors.append(str(exc))
                            trials.append(dict(key=key,head=h,method=method,candidate_id=candidate_id,
                                fold=fold,repeat=repeat,params=par,seconds=time.perf_counter()-tick,
                                status="FAILED_CANDIDATE",error=str(exc)))
                row=dict(head=h,method=method,candidate_id=candidate_id,
                         clip_quantile=cq,multiplier=mult,finite_trials=len(vals),
                         eligible=len(vals)==6 and not errors,
                         mean_validation_mse=float(np.mean(vals)) if len(vals)==6 and not errors else None,
                         errors=errors)
                summaries.append(row); candidate_summaries.append(row)
                print(json.dumps(dict(phase="candidate",**row)),flush=True)
            eligible=[r for r in summaries if r["eligible"]]
            if eligible:
                win=min(eligible,key=lambda r:(r["mean_validation_mse"],r["candidate_id"]))
                fr=final_ref[h]
                par=(ssp_parameters(fr,data[h]["B"].shape[1],win["clip_quantile"],win["multiplier"],ssp_sigma)
                     if method=="SSP" else sgd_parameters(fr,win["clip_quantile"],win["multiplier"],sgdsigma,T))
                selected[h][method]=par
                traces.append(dict(head=h,method=method,selected_candidate_id=win["candidate_id"],
                    selection_mean_validation_mse=win["mean_validation_mse"],
                    final_calibration_patient_ids=ids))
    write_json(out/"accounting.json",accounts)
    write_json(out/"trials.json",trials)
    write_json(out/"candidate_summaries.json",candidate_summaries)
    write_json(out/"selection_trace.json",traces)
    write_npz(out/"public_trial_weights.npz",weights)
    complete=all(selected[h][m] is not None for h in HEADS for m in ("SSP","SGD"))
    params=dict(schema="frozen-residual-public-dp-parameters/v1",complete=complete,
        selected=selected,public_only_weights="public_only_models.npz",public_patients=32,
        public_slot_surrogate_not_independent_population=True,private_data_read=False,
        input_sha256=inputs,selection_contract_sha256=digest(out/"selection_contract.json"),
        code_sha256=code_bindings(),selection_trace_sha256=digest(out/"selection_trace.json"),
        accountability="RDP simulation calibration; no confidential DP model released")
    write_json(out/"params.json",params)
    require_bindings(inputs);require_bindings(c["source_sha256"])
    outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="execution.json"}
    write_json(out/"execution.json",dict(
        status="PASS_PUBLIC_CALIBRATION_PENDING_INDEPENDENT_VERIFICATION" if complete else
               "PUBLIC_CALIBRATION_INCOMPLETE_CONVERGENCE_OR_CANDIDATE_FAILURE",
        complete=complete,seconds=time.perf_counter()-started,outputs_sha256=outputs,
        selected_steps=selected_steps,convergence_fits=len(convergence),candidate_fits=len(trials),
        private_patients_read=0,development_patients_read=0,GPU_calls=0,finished_utc=stamp()))
    print(json.dumps({"phase":"run","complete":complete,
                      "seconds":time.perf_counter()-started}),flush=True)


def self_test():
    t=learning_rates(500,.125)
    assert len(t)==500 and np.all(t[:400]==.125) and t[-1]==.00125
    assert np.all(np.diff(t)<=0)
    rng=np.random.default_rng(713)
    xx=rng.normal(size=(4,9,3)); yy=rng.normal(size=(4,9,2))
    stats=[rm.sufficient_statistics(x,y) for x,y in zip(xx,yy)]
    ds={k:np.stack([s[k] for s in stats]) for k in ("A","B","Q")}
    w=rng.normal(size=(3,2))
    actual=mse(ds,w)
    expected=np.array([np.mean(np.sum((y-x@w)**2,axis=1))/4 for x,y in zip(xx,yy)])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    direct=np.array([rm.loss_from_statistics(a,b,q,w)/4 for a,b,q in zip(ds["A"],ds["B"],ds["Q"])])
    np.testing.assert_allclose(actual,direct,rtol=1e-12,atol=1e-12)
    ref=reference(ds)
    assert normalized_gap(ds,ref["W"],ref["W"])==0.
    assert abs(normalized_gap(ds,np.zeros_like(ref["W"]),ref["W"])-1.)<1e-14
    aa=np.repeat(ds["A"],5,axis=0)
    np.testing.assert_allclose(aa.mean(axis=0),ds["A"].mean(axis=0),atol=1e-15)
    n=len(ds["A"]); rates=learning_rates(10,.01)
    manual=np.zeros_like(w)
    for lr in rates:
        manual-=lr*(2*(ds["A"].mean(axis=0)@manual-ds["B"].mean(axis=0))+2*RIDGE*manual)
    result=dm.poisson_sgd(ds["A"],ds["B"],steps=10,q=1.,public_denominator=n,
        clip_norm=float("inf"),sigma=0.,ridge=RIDGE,learning_rates=rates,
        sampling_seed=71,noise_seed=72,include_diagnostics=False)
    np.testing.assert_allclose(result["W"],manual,atol=1e-14,rtol=1e-12)
    print("PASS_SYNTHETIC_SCHEDULE_SURROGATE_RAW_OBJECTIVE_AND_NOCLIP_SGD")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("prepare","profile","run","self-test"),required=True)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    if args.phase=="self-test":
        self_test()
    elif args.phase=="prepare":
        prepare(args.output_dir.resolve())
    elif args.phase=="profile":
        profile(args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())


if __name__=="__main__":
    main()
