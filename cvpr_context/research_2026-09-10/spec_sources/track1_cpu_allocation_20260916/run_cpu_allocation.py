"""Predeclared synthetic allocation filter. CPU only; no neural models."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = np.array([1, 2, 4, 8, 16], dtype=np.int64)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, obj):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def clip(x):
    return x / np.maximum(1., np.linalg.norm(x, axis=-1, keepdims=True))

def draw(rng, n, d, t):
    out = np.zeros((n, d), dtype=np.float64)
    if t["kind"] == "gaussian_radial_isotropic":
        out[:, 0] = rng.normal(size=n) * t["radial_sd"]
        out[:, 1:] = rng.normal(size=(n, d-1)) * t["tangent_total_sd"] / math.sqrt(d-1)
    elif t["kind"] == "gaussian_plane":
        z = rng.normal(size=(n, 2))
        out[:, 0] = t["radial_sd"] * z[:, 0]
        out[:, 1] = t["tangent_sd"] * (t["rho"] * z[:, 0] + math.sqrt(1-t["rho"]**2)*z[:, 1])
    elif t["kind"] == "rademacher_plane":
        z = rng.choice(np.array([-1., 1.]), size=(n, 2))
        out[:, 0] = t["radial_sd"] * z[:, 0]
        out[:, 1] = t["tangent_sd"] * z[:, 1]
    else:
        raise ValueError(t["kind"])
    out[:, 0] += t["mean_norm"]
    return out

def theoretical_a(t):
    tangent = t.get("tangent_total_sd", t.get("tangent_sd", 0.)) ** 2
    raw = t["radial_sd"]**2 + tangent
    if t["mean_norm"] == 1:
        return None
    return raw if t["mean_norm"] < 1 else tangent / t["mean_norm"]**2

def proxies(p1, p2):
    mean, delta = (p1+p2)/2, p1-p2
    norm = np.linalg.norm(mean, axis=1)
    raw = np.sum(delta*delta, axis=1)/2
    guard = np.abs(norm-1) <= np.sqrt(raw)  # fixed kappa=1
    geometry = raw.copy()
    outside = (norm > 1) & (~guard)
    unit = mean[outside] / norm[outside, None]
    residual = delta[outside] - np.sum(delta[outside]*unit, axis=1)[:, None]*unit
    geometry[outside] = np.sum(residual*residual, axis=1)/(2*norm[outside]**2)
    return raw, geometry, guard

def final_means(rng, n, d, t):
    running = np.zeros((n, d), dtype=np.float64)
    means = np.zeros((n, len(LEVELS), d), dtype=np.float64)
    k = 0
    for r in range(1, 17):
        running += draw(rng, n, d, t)
        if r == LEVELS[k]:
            means[:, k, :] = running/r
            k += 1
            if k == len(LEVELS):
                break
    return means

def choose_r(proxy, lam):
    if lam == 0:
        return np.where(proxy > 0, 16, 1)
    desired = np.clip(np.sqrt(proxy/lam), 1, 16)
    return LEVELS[np.searchsorted(LEVELS, desired, side="left")]

def calibrate_lambda(proxy, budget):
    maximum = float(np.mean(choose_r(proxy, 0))+2)
    if maximum <= budget:
        return 0., maximum
    low, high = -30., 30.
    for _ in range(100):
        mid = (low+high)/2
        cost = float(np.mean(choose_r(proxy, 10**mid))+2)
        if cost > budget:
            low = mid
        else:
            high = mid
    lam = 10**high
    return lam, float(np.mean(choose_r(proxy, lam))+2)

def optimal_policy(risk, cap, overhead=0, type_aware=False):
    actions = [[int(a), int(b)] for a in LEVELS for b in LEVELS] if type_aware else [[int(a),int(a)] for a in LEVELS]
    points = []
    for rs in actions:
        cost = float(np.mean(rs)+overhead)
        err = float(sum(risk[j, np.where(LEVELS == rs[j])[0][0]] for j in range(2))/2)
        points.append((cost, err, rs))
    best = None
    def offer(parts):
        nonlocal best
        cost = sum(w*p[0] for w,p in parts)
        err = sum(w*p[1] for w,p in parts)
        if cost <= cap+1e-10 and (best is None or err < best["calibration_mse"]):
            best = {"actions":[{"probability":float(w),"R_by_type":p[2]} for w,p in parts if w>1e-14],
                    "overhead":overhead,"calibration_cost":float(cost),"calibration_mse":float(err)}
    for p in points:
        if p[0] <= cap:
            offer([(1.,p)])
    for a in points:
        for b in points:
            if a[0] < cap < b[0]:
                wb = (cap-a[0])/(b[0]-a[0])
                offer([(1-wb,a),(wb,b)])
    if best is None:
        raise ValueError("No feasible policy")
    return best

def correlation(a,b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a,b)[0,1])

def evaluate_family(family, index, config):
    d, nc, nt = family["dimension"], config["calibration_replicates_per_type"], config["test_replicates_per_type"]
    calibration = {}
    risks, raw_all, geo_all, truea_all = [], [], [], []
    for j,t in enumerate(family["types"]):
        rngp = np.random.default_rng(np.random.SeedSequence([config["master_seed"],index,0,j,0]))
        rngf = np.random.default_rng(np.random.SeedSequence([config["master_seed"],index,0,j,1]))
        raw,geo,guard = proxies(draw(rngp,nc,d,t),draw(rngp,nc,d,t))
        means = final_means(rngf,nc,d,t)
        truth = np.zeros(d); truth[0] = min(t["mean_norm"],1.)
        sq = np.sum((clip(means)-truth)**2,axis=2)
        risks.append(np.mean(sq,axis=0))
        raw_all.append(raw); geo_all.append(geo)
        a = theoretical_a(t)
        truea_all.append(np.full(nc,a if a is not None else np.nan))
        calibration[t["name"]] = {"true_delta_method_a":a,
            "raw_proxy_mean":float(raw.mean()),"geometry_proxy_mean":float(geo.mean()),
            "guard_fraction":float(guard.mean()),"zero_raw_fraction":float(np.mean(raw==0)),
            "zero_geometry_fraction":float(np.mean(geo==0)),
            "fixed_r_mse":{str(r):float(v) for r,v in zip(LEVELS,risks[-1])}}
    risk = np.asarray(risks)
    rawcat,geocat,trueacat = np.concatenate(raw_all),np.concatenate(geo_all),np.concatenate(truea_all)
    calibration["proxy_true_a_correlations"] = {"raw":correlation(rawcat,trueacat) if np.all(np.isfinite(trueacat)) else None,
        "geometry":correlation(geocat,trueacat) if np.all(np.isfinite(trueacat)) else None,
        "note":"Across independent pilot realizations and known synthetic types; equal a or boundary -> undefined."}
    methods = []
    for r in LEVELS:
        methods.append({"name":"fixed_R"+str(r),"kind":"fixed","R":int(r),"overhead":0,"budget":None})
    for budget in config["total_cost_budgets"]:
        for kind,p in [("raw",rawcat),("geometry",geocat)]:
            lam,cost = calibrate_lambda(p,budget)
            name = kind+"_B"+str(budget)
            methods.append({"name":name,"kind":kind,"lambda":lam,"overhead":2,"budget":budget,"calibration_cost":cost})
            matched = optimal_policy(risk,cost,0,False)
            methods.append({"name":"matched_fixed_for_"+name,"kind":"policy","policy":matched,"budget":budget,
                            "matched_to":name,"overhead":0})
        for overhead,label in [(0,"oracle_free"),(2,"oracle_charged")]:
            methods.append({"name":label+"_B"+str(budget),"kind":"policy",
                 "policy":optimal_policy(risk,budget,overhead,True),"budget":budget,"overhead":overhead,
                 "diagnostic_oracle":True})
    accum = {}
    for m in methods:
        accum[m["name"]]={"mse":np.zeros((2,nt)),"cost":np.zeros((2,nt)),
            "pre_sum":np.zeros((2,d)),"pre_sq":np.zeros((2,d)),"r_counts":np.zeros((2,5),dtype=np.int64)}
    test_guard=[0,0]
    for j,t in enumerate(family["types"]):
        rngp = np.random.default_rng(np.random.SeedSequence([config["master_seed"],index,1,j,0]))
        rngf = np.random.default_rng(np.random.SeedSequence([config["master_seed"],index,1,j,1]))
        choices = {m["name"]:np.random.default_rng(np.random.SeedSequence([config["master_seed"],index,2,j,k]))
                   for k,m in enumerate(methods)}
        truth=np.zeros(d);truth[0]=min(t["mean_norm"],1.)
        mu=np.zeros(d);mu[0]=t["mean_norm"]
        for start in range(0,nt,config["test_chunk_size"]):
            n=min(config["test_chunk_size"],nt-start)
            raw,geo,guard=proxies(draw(rngp,n,d,t),draw(rngp,n,d,t))
            test_guard[j]+=int(guard.sum())
            means=final_means(rngf,n,d,t)
            for m in methods:
                if m["kind"]=="fixed":
                    rs=np.full(n,m["R"],dtype=np.int64)
                elif m["kind"] in ("raw","geometry"):
                    rs=choose_r(raw if m["kind"]=="raw" else geo,m["lambda"])
                else:
                    actions=m["policy"]["actions"]
                    u=choices[m["name"]].random(n)
                    probs=np.cumsum([a["probability"] for a in actions])
                    ix=np.minimum(np.searchsorted(probs,u,side="right"),len(actions)-1)
                    rs=np.array([actions[int(k)]["R_by_type"][j] for k in ix],dtype=np.int64)
                picked=means[np.arange(n),np.searchsorted(LEVELS,rs)]
                error=picked-mu
                a=accum[m["name"]]
                a["mse"][j,start:start+n]=np.sum((clip(picked)-truth)**2,axis=1)
                a["cost"][j,start:start+n]=rs+m["overhead"]
                a["pre_sum"][j]+=error.sum(axis=0)
                a["pre_sq"][j]+=(error*error).sum(axis=0)
                a["r_counts"][j]+=np.bincount(np.searchsorted(LEVELS,rs),minlength=5)
    rows,preclip,paired=[],[],[]
    for m in methods:
        a=accum[m["name"]]; vals=a["mse"].mean(axis=0);cost=a["cost"].mean(axis=0)
        rows.append({"family":family["id"],"method":m["name"],"budget":m["budget"],
            "clipped_mse":float(vals.mean()),"mc_se":float(vals.std(ddof=1)/math.sqrt(nt)),
            "actual_mean_cost":float(cost.mean()),"cost_mc_se":float(cost.std(ddof=1)/math.sqrt(nt)),
            "calibration_mean_cost":m.get("calibration_cost",m.get("policy",{}).get("calibration_cost",m.get("R"))),
            "R_frequencies_by_type":(a["r_counts"]/nt).tolist(),
            "mse_by_type":a["mse"].mean(axis=1).tolist(),
            "guard_fraction_by_type":[x/nt for x in test_guard] if m["kind"]=="geometry" else None})
        means=a["pre_sum"]/nt
        variances=np.maximum(0.,(a["pre_sq"]-nt*means**2)/(nt-1))
        se=np.sqrt(variances/nt)
        z=np.divide(means,se,out=np.zeros_like(means),where=se>0)
        preclip.append({"family":family["id"],"method":m["name"],
            "mean_error_vector_by_type":means.tolist(),"coordinate_mc_se_by_type":se.tolist(),
            "max_absolute_coordinate_z":float(np.max(np.abs(z))),
            "mean_error_norm_by_type":np.linalg.norm(means,axis=1).tolist(),
            "claim":"Finite-MC check only; exact conditional expectation follows independent fresh draws. No global hypothesis-test gate."})
        if m["kind"] in ("raw","geometry"):
            fixed=accum["matched_fixed_for_"+m["name"]]
            delta=(a["mse"]-fixed["mse"]).mean(axis=0)
            mean=float(delta.mean());err=1.96*float(delta.std(ddof=1))/math.sqrt(nt)
            cdelta=float((a["cost"]-fixed["cost"]).mean())
            paired.append({"family":family["id"],"method":m["name"],"budget":m["budget"],
                "comparison":"adaptive minus independent-calibration selected matched-cost fixed mixture",
                "delta_mse":mean,"mc_95_interval":[mean-err,mean+err],"test_cost_difference":cdelta,
                "not_medical_or_patient_inference":True})
    return {"family":family,"calibration":calibration,"policies":methods,
            "rows":rows,"paired_comparisons":paired,"preclip_checks":preclip}

def self_test():
    p1=np.array([[2.5,0.],[2.,.1],[.75,0.]])
    p2=np.array([[1.5,0.],[2.,-.1],[.75,0.]])
    raw,geo,guard=proxies(p1,p2)
    assert raw[0]==.5 and geo[0]==0 and not guard[0]
    assert abs(geo[1]-.005)<1e-12
    assert choose_r(np.array([0.,1.,4.]),1.).tolist()==[1,1,2]
    risk=np.tile(np.array([1.,.5,.25,.125,.0625]),(2,1))
    policy=optimal_policy(risk,3.)
    assert abs(policy["calibration_cost"]-3)<1e-12
    assert abs(policy["calibration_mse"]-.375)<1e-12
    assert abs((2-4/math.sqrt(4.01))-.002495322244310705)<1e-15
    print(json.dumps({"self_test":"PASS","no_experiment_results_read":True}))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run:
        raise SystemExit("Use --run or --self-test")
    protocol=HERE/"protocol.json"
    cfg=json.loads(protocol.read_text(encoding="utf-8"))
    if (HERE/"results.json").exists() or (HERE/"execution_protocol.json").exists():
        raise RuntimeError("Immutable run already exists; no overwrite")
    binding={"protocol_sha256":sha(protocol),"script_sha256":sha(__file__),
             "python":platform.python_version(),"numpy":np.__version__,"execution_start_unix":time.time(),
             "all_policy_choices_frozen_before_results":True}
    write_json(HERE/"execution_protocol.json",binding)
    start=time.perf_counter()
    outputs=[]
    for index,family in enumerate(cfg["families"]):
        t=time.perf_counter()
        result=evaluate_family(family,index,cfg)
        outputs.append(result)
        write_json(HERE/(family["id"]+".json"),result)
        print(json.dumps({"completed_family":family["id"],"seconds":time.perf_counter()-t}),flush=True)
    assert sha(protocol)==binding["protocol_sha256"] and sha(__file__)==binding["script_sha256"]
    flat=[r for o in outputs for r in o["rows"]]
    paired=[r for o in outputs for r in o["paired_comparisons"]]
    execution={"status":"PASS_PREDECLARED_SYNTHETIC_MC_COMPLETE","seconds":time.perf_counter()-start,
          "families":len(outputs),"rows":len(flat),"paired_comparisons":len(paired),
          "binding":binding,"preclip_unbiasedness":"E[mean of fresh R(P) iid final draws | pilot P]=mu exactly; clipping is nonlinear and need not be unbiased.",
          "break_even":{"model":"a_i/r_i, continuous oracle, no caps/minimums; p pilot cost",
             "ratio_to_fixed":"B/(B-p)/(1+CV(sqrt(a))^2)",
             "necessary_strict_improvement":"CV(sqrt(a))^2 > p/(B-p)",
             "B4_p2_required_CV":1.,"B8_p2_required_CV":math.sqrt(1/3)},
          "limitations":["Synthetic gradients only","No added DP Gaussian noise","No AdamW update or model quality","Oracle uses known distribution types and separate calibration risk curves","95% intervals are Monte Carlo simulation error"]}
    write_json(HERE/"results.json",{"execution":execution,"rows":flat,"paired_comparisons":paired,
        "family_files":[o["family"]["id"]+".json" for o in outputs]})
    with (HERE/"results.csv").open("x",encoding="utf-8-sig",newline="") as f:
        fields=["family","method","budget","clipped_mse","mc_se","actual_mean_cost","cost_mc_se","calibration_mean_cost"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in flat:w.writerow({k:row[k] for k in fields})
    with (HERE/"paired_comparisons.csv").open("x",encoding="utf-8-sig",newline="") as f:
        fields=["family","method","budget","delta_mse","mc_95_low","mc_95_high","test_cost_difference"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in paired:w.writerow({**{k:r[k] for k in fields if k not in ("mc_95_low","mc_95_high")},
                                   "mc_95_low":r["mc_95_interval"][0],"mc_95_high":r["mc_95_interval"][1]})
    files=sorted(p for p in HERE.iterdir() if p.is_file())
    write_json(HERE/"manifest.json",{"files_sha256":{p.name:sha(p) for p in files},"status":execution["status"],
            "no_gpu_or_training":True,"elapsed_seconds":execution["seconds"]})
    print(json.dumps(execution),flush=True)

if __name__=="__main__":
    main()

