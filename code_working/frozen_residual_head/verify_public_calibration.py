"""Independent CPU audit of public-only DP parameter selection.

No numerical producer module is imported. This verifies saved weights, losses,
selection logic, public-derived parameters and privacy accounting; it does not
claim an independent replay of every optimizer step. Private/eval stats are not
opened. Calling verify_calibration writes its own protocol and result once.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import warnings
import numpy as np

CODE = Path(__file__).resolve().parents[1]
DEFAULT = CODE / "_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1"
STATUS = "PASS_PUBLIC_CALIBRATION_INDEPENDENT_SELECTION_ARITHMETIC_AND_ACCOUNTING"
HEADS = ("full", "static")
REPEATS = (260916101, 260916102, 260916103)
TIMES = (500, 2000, 8000)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = {}

    def require(self, condition, name):
        self.count += 1
        if not condition:
            raise AssertionError(name)

    def close(self, actual, expected, name, rtol=2e-9, atol=2e-11):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(a.shape == b.shape, name + ": shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), name + ": finite")
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        category = name.split("/")[0]
        self.errors[category] = max(error, self.errors.get(category, 0.))
        self.require(np.all(np.abs(a-b) <= atol + rtol*np.abs(b)),
                     name + ": arithmetic difference " + repr(error))


def fsummean(values):
    values = list(values)
    return math.fsum(float(x) for x in values)/len(values)


def frobenius(x):
    return math.sqrt(math.fsum(float(y)*float(y) for y in np.ravel(x)))


def independent_mse(data, w):
    result = []
    for a, b, q in zip(data["A"], data["B"], data["Q"]):
        linear = math.fsum(float(x)*float(y) for x, y in zip(w.ravel(), b.ravel()))
        aw = a @ w
        quadratic = math.fsum(float(x)*float(y) for x, y in zip(w.ravel(), aw.ravel()))
        result.append(math.fsum((float(q), -2*linear, quadratic))/4)
    return np.asarray(result)


def public_reference(data):
    a = np.add.reduce(data["A"], axis=0)/len(data["A"])
    b = np.add.reduce(data["B"], axis=0)/len(data["B"])
    a = (a+a.T)/2
    eig, vec = np.linalg.eigh(a)
    # Different solve route from the producer's np.linalg.solve.
    w = vec @ ((vec.T @ b)/(eig[:, None]+.001))
    norm_a = np.asarray([frobenius(x) for x in data["A"]])
    norm_b = np.asarray([frobenius(x) for x in data["B"]])
    a0, b0 = float(np.median(norm_a)), float(np.median(norm_b))
    statnorm = np.hypot(norm_a/a0, norm_b/b0)
    gradnorm = np.asarray([frobenius(2*(au@(fraction*w)-bu))
                          for fraction in (0., .25, .5, 1.)
                          for au, bu in zip(data["A"], data["B"])])
    return dict(W=w, H=a+.001*np.eye(len(a)), L=2*(float(eig[-1])+.001),
                a0=a0, b0=b0, statnorm=statnorm, gradnorm=gradnorm)


def gap(w, ref):
    e = w-ref["W"]
    numerator = math.fsum(float(x)*float(y)
                         for x, y in zip(e.ravel(), (ref["H"]@e).ravel()))
    denominator = math.fsum(float(x)*float(y)
                           for x, y in zip(ref["W"].ravel(), (ref["H"]@ref["W"]).ravel()))
    if denominator <= 0:
        if numerator == 0:
            return 0.
        raise ValueError("No positive reference improvement")
    return numerator/denominator


def higher_quantile(x, q):
    ordered = sorted(float(v) for v in x)
    return ordered[int(math.ceil((len(ordered)-1)*q))]


def expected_parameters(ref, method, quantile, multiplier, sigma, steps):
    if method == "SSP":
        c = higher_quantile(ref["statnorm"], quantile)
        nu = sigma*c*ref["a0"]/80*math.sqrt(2*len(ref["W"]))
        return dict(a0=ref["a0"], b0=ref["b0"], clip_norm=c, sigma=sigma,
                    ridge=.001, eigenvalue_floor=multiplier*nu, public_denominator=80,
                    clip_quantile=quantile, floor_multiplier=multiplier, nu_A=nu)
    return dict(clip_norm=higher_quantile(ref["gradnorm"], quantile), sigma=sigma,
                ridge=.001, q=.1, steps=steps, initial_lr=multiplier/ref["L"],
                lr_multiplier=multiplier, L_public=ref["L"], public_denominator=80,
                schedule="constant80_linear20_end01", clip_quantile=quantile)


def check_parameters(check, actual, expected, label):
    check.require(set(actual) == set(expected), label + " keys")
    for k, value in expected.items():
        if isinstance(value, str):
            check.require(actual[k] == value, label + " " + k)
        else:
            check.close(actual[k], value, "parameters/"+label+"/"+k)


def subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


def verify_accounting(accounts, contract, check):
    # Import accountant libraries only, not the producer/accountant wrapper.
    from opacus.accountants.analysis import rdp
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant
    orders = [round(1.+i/10., 1) for i in range(1, 100)]
    orders += list(range(12, 64)) + [64, 128, 256, 512]
    cache_path = Path(contract["accountant_cache_path"])
    check.require(sha(cache_path) == contract["accountant_cache_sha256"], "accountant cache hash")
    cache = read(cache_path)
    audit = []
    for key, row in accounts.items():
        q, steps, sigma = row["poisson_sample_rate"], row["steps"], row["noise_multiplier"]
        check.require(key == f"q{q}_T{steps}", "account key")
        check.require(row["target_epsilon"] == 8. and row["target_delta"] == 1e-5, "account target")
        check.require(sigma > 0 and 0 < q <= 1 and steps > 0, "account arguments")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            values = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
            eps1, order1 = rdp.get_privacy_spent(orders=orders, rdp=values, delta=1e-5)
            accountant = RdpAccountant(orders=orders,
                neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE)
            accountant.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)),
                               count=steps)
            eps2, order2 = accountant.get_epsilon_and_optimal_order(1e-5)
        check.close(row["opacus_epsilon"], eps1, "accounting/opacus", atol=1e-9)
        check.close(row["google_dp_accounting_epsilon"], eps2, "accounting/google", atol=1e-9)
        check.close(row["opacus_optimal_order"], order1, "accounting/opacus order")
        check.close(row["google_optimal_order"], order2, "accounting/google order")
        check.close(row["conservative_epsilon"], max(eps1, eps2), "accounting/conservative")
        check.close(row["interimplementation_abs_difference"], abs(eps1-eps2), "accounting/disagreement")
        check.require(max(eps1, eps2) <= 8.+1e-9, "account epsilon bound")
        if row["reused_public_accounting_cache"]:
            candidates = [x for x in cache["results"] if
                          x["poisson_sample_rate"] == q and x["steps"] == steps]
            check.require(len(candidates) == 1, "cache unique accounting match")
            for name, value in candidates[0].items():
                check.require(row[name] == value, "cached accounting exact " + name)
        audit.append(dict(key=key, opacus_epsilon=float(eps1), google_epsilon=float(eps2),
                          warnings=[str(x.message) for x in captured]))
    return audit


def _audit(folder, check):
    contract = read(folder/"selection_contract.json")
    execution = read(folder/"execution.json")
    run = read(folder/"run_protocol.json")
    params = read(folder/"params.json")
    check.require(execution["complete"] and params["complete"], "complete public selection")
    check.require(execution["status"] == "PASS_PUBLIC_CALIBRATION_PENDING_INDEPENDENT_VERIFICATION", "producer status")
    for key, value in dict(schema="frozen-residual-public-dp-selection/v1",
        public_denominator=80, adjacency="ADD_OR_REMOVE_ONE", epsilon=8., delta=1e-5,
        q=.1, ridge=.001, public_patients=32, patient_images=2, draws_per_image=8,
        output_channels=4, public_split_label="public", heads=list(HEADS),
        clip_quantiles=[.5,.9], ssp_floor_multipliers=[0.,1.,2.],
        sgd_lr_multipliers=[.25,.5,1.], convergence_steps=list(TIMES),
        noise_repeats=list(REPEATS), convergence_threshold=.01,
        convergence_reference_lr_multiplier=.5, quantile_method="higher",
        sgd_gradient_reference_positions=[0.,.25,.5,1.],
        candidate_count_per_head_method=6, schedule="constant80_linear20_end01",
        protected_statistics_read=False, development_statistics_read=False,
        max_seed_selection=False, private_optimization=False).items():
        check.require(contract[key] == value, "fixed policy " + key)
    check.require(execution["private_patients_read"] == 0 and execution["development_patients_read"] == 0
                  and execution["GPU_calls"] == 0, "public-only execution flags")
    check.require(params["private_data_read"] is False and params["public_patients"] == 32
                  and params["public_slot_surrogate_not_independent_population"] is True, "parameter scope flags")
    check.require(run["private_data_read"] is False and run["GPU_calls"] == 0, "run scope")
    contract_hash = sha(folder/"selection_contract.json")
    check.require(run["selection_contract_sha256"] == contract_hash
                  and params["selection_contract_sha256"] == contract_hash, "contract binding")
    for mapping in (contract["source_sha256"], run["source_sha256"], params["code_sha256"]):
        for path, digest in mapping.items():
            check.require(sha(path) == digest, "code binding " + path)
    check.require(contract["source_sha256"] == run["source_sha256"] == params["code_sha256"], "source sets")
    check.require(run["input_sha256"] == params["input_sha256"], "input sets")
    for path, digest in run["input_sha256"].items():
        check.require(sha(path) == digest, "public input binding " + path)
    for name, digest in execution["outputs_sha256"].items():
        check.require(Path(name).name == name, "output safe basename")
        check.require(sha(folder/name) == digest, "producer output binding " + name)
    check.require(sha(folder/"selection_trace.json") == params["selection_trace_sha256"], "selection trace binding")
    public = Path(contract["public_directory"])
    check.require(public.resolve() == folder.parent.resolve(), "public-only directory")
    check.require(sha(public/"patient_stats.npz") == contract["expected_public_stats_sha256"], "public stats digest")
    verification = read(public/"independent_verification.json")
    check.require(verification["status"] == "PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE",
                  "public raw verifier passed")
    for name in ("patient_stats.npz","contract.json","manifest.json","images.json","analysis.json","execution.json"):
        check.require(sha(public/name) == verification["input_sha256"][name], "public raw provenance "+name)
    with np.load(public/"patient_stats.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["patient_ids"]]
        check.require(len(ids) == len(set(ids)) == 32 and set(map(str,z["splits"])) == {"public"}, "public32 IDs")
        data = {h:{k:np.array(z[k+"_"+h],dtype=np.float64) for k in ("A","B","Q")} for h in HEADS}
    for h in HEADS:
        d = 64 if h == "full" else 16
        check.require(data[h]["A"].shape == (32,d,d) and data[h]["B"].shape == (32,d,4)
                      and data[h]["Q"].shape == (32,), "public statistic shape "+h)
        for v in data[h].values():
            check.require(np.isfinite(v).all(), "public statistics finite")
        check.close(data[h]["A"], data[h]["A"].transpose(0,2,1), "public symmetry", atol=1e-12,rtol=1e-12)
    check.close(data["full"]["Q"],data["static"]["Q"],"public shared Q",atol=0.,rtol=0.)
    order = sorted(range(32), key=lambda j: hashlib.sha256(
        ("public-dp-fold-v1|260916|"+ids[j]).encode()).hexdigest())
    folds = [(order[:16],order[16:]),(order[16:],order[:16])]
    expected_folds = [dict(fold=i,train_patient_ids=[ids[j] for j in tr],
                    validation_patient_ids=[ids[j] for j in va]) for i,(tr,va) in enumerate(folds)]
    check.require(read(folder/"folds.json") == expected_folds, "fold hash-order and disjointness")
    refs = {h:[public_reference(subset(data[h],tr)) for tr,_ in folds] for h in HEADS}
    final_refs = {h:public_reference(data[h]) for h in HEADS}
    public_info = read(folder/"public_only.json")
    check.require(public_info["patient_ids"] == ids and public_info["ridge"] == .001, "public-only fit set")
    check.require(public_info["model_sha256"] == sha(folder/"public_only_models.npz"), "public weights binding")
    with np.load(folder/"public_only_models.npz",allow_pickle=False) as z:
        check.require(set(z.files) == {"W_full","W_static"}, "public weight inventory")
        for h in HEADS:
            check.close(z["W_"+h],final_refs[h]["W"],"public ridge/"+h)
            check.close(public_info["train_mse"][h],fsummean(independent_mse(data[h],z["W_"+h])),
                        "public ridge/MSE "+h)
    accounts = read(folder/"accounting.json")
    accounting_audit = verify_accounting(accounts,contract,check)
    check.require("q1.0_T1" in accounts, "SSP accounting")
    convergence = read(folder/"convergence.json")
    check.require(convergence["threshold"] == .01
                  and convergence["criterion"] == contract["convergence_selection"], "convergence policy")
    convrows = {x["key"]:x for x in convergence["records"]}
    check.require(len(convrows) == len(convergence["records"]), "unique convergence keys")
    trials = read(folder/"trials.json")
    trialmap = {x["key"]:x for x in trials}
    check.require(len(trialmap) == len(trials), "unique trial keys")
    summaries = read(folder/"candidate_summaries.json")
    summarymap = {(x["head"],x["method"],x["candidate_id"]):x for x in summaries}
    check.require(len(summarymap) == len(summaries), "unique candidate summaries")
    selection = read(folder/"selection_trace.json")
    selectionmap = {(x["head"],x["method"]):x for x in selection}
    check.require(len(selectionmap) == len(selection), "unique selected methods")
    selected_steps, conv_expected, trial_expected, weights_expected = {}, set(), set(), set()
    winners, convergence_means = [], []
    with np.load(folder/"public_trial_weights.npz",allow_pickle=False) as weights:
        for h in HEADS:
            for steps in TIMES:
                means = []
                for f,(tr,_) in enumerate(folds):
                    values = []
                    for repeat in REPEATS:
                        key = f"conv_{h}_T{steps}_f{f}_r{repeat}"
                        conv_expected.add(key); weights_expected.add(key)
                        check.require(key in convrows and key in weights.files, "convergence full coverage "+key)
                        row = convrows[key]
                        check.require((row["head"],row["steps"],row["fold"],row["repeat"]) ==
                                      (h,steps,f,repeat), "convergence metadata")
                        check.close(row["initial_lr"],.5/refs[h][f]["L"],"parameters/convergence LR")
                        w = weights[key]
                        value = gap(w,refs[h][f])
                        values.append(value)
                        check.close(row["normalized_gap"],value,"convergence/gap "+key)
                        check.close(row["diagnostics"]["weight_norm"],frobenius(w),"convergence/W norm")
                        monitors = row["diagnostics"]["monitors"]
                        check.require(monitors[0]["step"] == 0 and monitors[-1]["step"] == steps, "convergence monitor endpoints")
                        check.close(monitors[0]["normalized_public_objective_gap"],1.,"convergence/initial")
                        check.close(monitors[-1]["normalized_public_objective_gap"],value,"convergence/final monitor")
                    means.append(fsummean(values))
                passed = all(x <= .01 for x in means)
                convergence_means.append(dict(head=h,steps=steps,independent_fold_mean_gaps=means,passed=passed))
                if passed:
                    selected_steps[h] = steps
                    break
            check.require(h in selected_steps, "required head convergence passed")
        check.require(set(convrows) == conv_expected, "no skipped/extra convergence runs")
        check.require(convergence["selected_steps"] == selected_steps == execution["selected_steps"], "first passing T")
        for h in HEADS:
            steps = selected_steps[h]
            sgkey = f"q0.1_T{steps}"
            check.require(sgkey in accounts, "SGD accounting for selected T")
            for method in ("SSP","SGD"):
                sigma = accounts["q1.0_T1" if method == "SSP" else sgkey]["noise_multiplier"]
                portfolio = [(cq,m) for cq in (.5,.9)
                             for m in ((0.,1.,2.) if method == "SSP" else (.25,.5,1.))]
                independently_eligible = []
                for cid,(cq,multiplier) in enumerate(portfolio):
                    vals, errors = [], []
                    for f,(tr,va) in enumerate(folds):
                        expected = expected_parameters(refs[h][f],method,cq,multiplier,sigma,steps)
                        for repeat in REPEATS:
                            key = f"cv_{h}_{method}_c{cid}_f{f}_r{repeat}"
                            trial_expected.add(key)
                            check.require(key in trialmap, "full public candidate portfolio "+key)
                            row = trialmap[key]
                            check.require((row["head"],row["method"],row["candidate_id"],row["fold"],row["repeat"]) ==
                                          (h,method,cid,f,repeat), "candidate metadata")
                            check_parameters(check,row["params"],expected,key)
                            if row["status"] == "FINITE":
                                check.require(key in weights.files,"finite trial has weights")
                                weights_expected.add(key)
                                w = weights[key]
                                check.require(w.shape == refs[h][f]["W"].shape and np.isfinite(w).all(), "trial W shape/finite")
                                scores = independent_mse(subset(data[h],va),w)
                                value = fsummean(scores)
                                vals.append(value)
                                check.close(row["validation_patient_mse"],scores,"validation/patient losses "+key)
                                check.close(row["validation_mse"],value,"validation/mean "+key)
                                check.close(row["diagnostics"]["weight_norm"],frobenius(w),"validation/W norm")
                                if method == "SGD":
                                    check.close(row["diagnostics"]["monitors"][0]["normalized_public_objective_gap"],1.,
                                                "validation/SGD initial gap")
                                    check.close(row["diagnostics"]["monitors"][-1]["normalized_public_objective_gap"],
                                                gap(w,refs[h][f]),"validation/SGD final gap")
                            else:
                                check.require(row["status"] == "FAILED_CANDIDATE" and key not in weights.files,
                                              "failed candidate inventory")
                                check.require(isinstance(row["error"],str) and bool(row["error"]), "recorded candidate failure")
                                errors.append(row["error"])
                    s = summarymap[(h,method,cid)]
                    check.require(s["clip_quantile"] == cq and s["multiplier"] == multiplier, "candidate ordering")
                    eligible = len(vals) == 6 and not errors
                    check.require(s["finite_trials"] == len(vals) and s["eligible"] == eligible and s["errors"] == errors,
                                  "candidate eligibility")
                    if eligible:
                        average = fsummean(vals)
                        check.close(s["mean_validation_mse"],average,"selection/six-trial mean")
                        independently_eligible.append((average,cid,cq,multiplier))
                    else:
                        check.require(s["mean_validation_mse"] is None, "failed candidate not selected")
                check.require(bool(independently_eligible),"eligible public candidates")
                winner = min(independently_eligible,key=lambda x:(x[0],x[1]))
                trace = selectionmap[(h,method)]
                check.require(trace["selected_candidate_id"] == winner[1], "independent minimum candidate "+h+method)
                check.close(trace["selection_mean_validation_mse"],winner[0],"selection/winner loss")
                check.require(trace["final_calibration_patient_ids"] == ids, "all public32 final recalibration")
                expected = expected_parameters(final_refs[h],method,winner[2],winner[3],sigma,steps)
                check_parameters(check,params["selected"][h][method],expected,"final_"+h+"_"+method)
                ordered_values = sorted(independently_eligible)
                margin = ordered_values[1][0]-ordered_values[0][0] if len(ordered_values)>1 else None
                winners.append(dict(head=h,method=method,selected_candidate_id=winner[1],
                                    independent_validation_mse=winner[0],runner_up_margin=margin))
        check.require(set(weights.files) == weights_expected,"all and only successful weights")
    expected_summary_keys = {(h,m,c) for h in HEADS for m in ("SSP","SGD") for c in range(6)}
    check.require(set(summarymap) == expected_summary_keys,"24 candidate summaries")
    check.require(set(selectionmap) == {(h,m) for h in HEADS for m in ("SSP","SGD")},"four final selections")
    check.require(set(trialmap) == trial_expected,"144 candidate trial inventory")
    check.require(set(accounts) == {"q1.0_T1"} | {f"q0.1_T{x}" for x in selected_steps.values()}, "only used accounting events")
    check.require(execution["convergence_fits"] == len(convrows)
                  and execution["candidate_fits"] == len(trialmap), "execution counters")
    # The producer source and exact schedule declaration are hash-bound. No claim
    # to recover omitted intermediate optimizer states is made.
    return dict(public_patients=32, public_folds=2, surrogate_slots_per_fold=80,
        independent_public_patients_per_training_fold=16, candidate_trials=len(trialmap),
        finite_candidate_trials=sum(x["status"]=="FINITE" for x in trials),
        convergence_fits=len(convrows), candidate_summaries=len(summarymap),
        selected_steps=selected_steps, winners=winners, convergence=convergence_means,
        accountant_rechecks=accounting_audit, private_statistics_read=0,
        development_statistics_read=0, GPU_calls=0,
        numerical_scope="saved final weights and public objectives/selection/calibration; not every optimizer step",
        accountant_scope="independent wrapper with the same two accountant libraries and fixed Renyi orders",
        complete=True)


def verify_calibration(calibration_dir):
    folder = Path(calibration_dir).resolve()
    result_path = folder/"independent_selection_verification.json"
    protocol_path = folder/"independent_selection_verification_protocol.json"
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Independent selection audit already exists; preserve it")
    started = time.perf_counter()
    execution = read(folder/"execution.json")
    contract = read(folder/"selection_contract.json")
    public = Path(contract["public_directory"])
    bound = {str(folder/name):sha(folder/name) for name in execution["outputs_sha256"]}
    for path in (folder/"execution.json",Path(__file__),public/"patient_stats.npz",
                 public/"independent_verification.json",Path(contract["accountant_cache_path"])):
        bound[str(path)] = sha(path)
    bound.update(contract["source_sha256"])
    protocol = dict(schema="independent-public-dp-selection-verification/v1",
        created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),
        input_sha256=bound,selection_contract_sha256=sha(folder/"selection_contract.json"),
        independence="no producer numerical functions imported; public32 only; no optimizer candidate refitting",
        tolerance=dict(rtol=2e-9,atol=2e-11),no_private_or_development_statistic_reads=True)
    write(protocol_path,protocol)
    checks = Checks()
    try:
        summary = _audit(folder,checks)
        for path,digest in bound.items():
            checks.require(sha(path)==digest,"unchanged audit input "+path)
        summary.update(status=STATUS,checks=checks.count,maximum_absolute_errors=checks.errors,
            seconds=time.perf_counter()-started,code_sha256=sha(__file__),
            protocol_sha256=sha(protocol_path),selection_contract_sha256=sha(folder/"selection_contract.json"),
            params_sha256=sha(folder/"params.json"),accounting_sha256=sha(folder/"accounting.json"),
            execution_sha256=sha(folder/"execution.json"),input_sha256=bound)
        write(result_path,summary)
        return summary
    except Exception as exc:
        failure = dict(status="FAILED_PUBLIC_CALIBRATION_INDEPENDENT_VERIFICATION",
            complete=False,error=repr(exc),traceback=traceback.format_exc(),
            checks=checks.count,maximum_absolute_errors=checks.errors,
            seconds=time.perf_counter()-started,protocol_sha256=sha(protocol_path),
            code_sha256=sha(__file__),private_statistics_read=0,development_statistics_read=0,GPU_calls=0)
        write(result_path,failure)
        raise


def self_test():
    rng=np.random.default_rng(915)
    x=rng.normal(size=(4,11,3)); y=rng.normal(size=(4,11,4))
    a=np.stack([xx.T@xx/11 for xx in x])
    b=np.stack([xx.T@yy/11 for xx,yy in zip(x,y)])
    q=np.asarray([np.sum(yy*yy)/11 for yy in y])
    data=dict(A=a,B=b,Q=q)
    w=rng.normal(size=(3,4))
    direct=np.asarray([np.mean((xx@w-yy)**2) for xx,yy in zip(x,y)])
    np.testing.assert_allclose(independent_mse(data,w),direct,rtol=1e-12,atol=1e-12)
    ref=public_reference(data)
    exact=np.linalg.solve(a.mean(0)+.001*np.eye(3),b.mean(0))
    np.testing.assert_allclose(ref["W"],exact,rtol=1e-12,atol=1e-12)
    assert abs(gap(np.zeros((3,4)),ref)-1)<1e-13
    assert gap(ref["W"],ref)==0
    for frac in (.5,.9):
        assert higher_quantile([3.,1.,5.,2.],frac)==float(np.quantile([3.,1.,5.,2.],frac,method="higher"))
    print("PASS_INDEPENDENT_PUBLIC_OBJECTIVE_RIDGE_GAP_QUANTILE_SYNTHETIC")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibration-dir",type=Path,default=DEFAULT)
    ap.add_argument("--expected-code-sha256")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.expected_code_sha256 and sha(__file__)!=args.expected_code_sha256:
        raise RuntimeError("Verifier source hash mismatch")
    if args.self_test:
        self_test()
    else:
        summary=verify_calibration(args.calibration_dir)
        print(json.dumps({k:summary[k] for k in
              ("status","checks","seconds","candidate_trials","convergence_fits","selected_steps",
               "maximum_absolute_errors","code_sha256")},indent=2))

if __name__=="__main__":
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key,"1")
    main()

