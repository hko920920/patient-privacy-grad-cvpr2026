"""Independent CPU audit of cached equal-patient pooled ridge references.

No producer numerical functions are imported. The verifier checks only completed
saved outputs. Its optional self-test uses synthetic arrays and no study inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "_reports/frozen_residual_pooled_reference_20260916_v1"
STATUS = "PASS_POOLED_REFERENCE_SAVED_PATIENT_AVERAGES_RIDGE_AND_DEVELOPMENT_LOSSES"
FAMILIES = ("static", "full")
REFERENCES = ("base", "public_only", "private_only", "pooled")
RIDGE = .001


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as packet:
        return {name: packet[name].copy() for name in packet.files}


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = {}

    def require(self, value, label):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def close(self, actual, expected, label, *, atol=1e-12, rtol=1e-10):
        x, y = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(x.shape == y.shape, label + ": shape")
        self.require(np.isfinite(x).all() and np.isfinite(y).all(), label + ": finite")
        error = np.abs(x-y)
        self.errors[label] = float(error.max(initial=0))
        self.require(np.all(error <= atol + rtol*np.abs(y)), label + ": numerical agreement")


def patient_average(values):
    """Explicit equal-patient arithmetic, independent of weighted group means."""
    x = np.asarray(values, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    result = np.array([math.fsum(flat[:, j])/len(x) for j in range(flat.shape[1])])
    return result.reshape(x.shape[1:])


def cholesky_ridge(a, b, ridge=RIDGE):
    symmetric = (np.asarray(a, dtype=np.float64) + np.asarray(a, dtype=np.float64).T)/2
    factor = np.linalg.cholesky(symmetric + ridge*np.eye(len(symmetric)))
    # Triangular substitution is explicit instead of the producer solve/eigh.
    d, o = b.shape
    intermediate, w = np.zeros((d, o)), np.zeros((d, o))
    for channel in range(o):
        for i in range(d):
            subtotal = math.fsum(float(factor[i,j])*float(intermediate[j,channel]) for j in range(i))
            intermediate[i,channel] = (float(b[i,channel])-subtotal)/factor[i,i]
        for i in reversed(range(d)):
            subtotal = math.fsum(float(factor[j,i])*float(w[j,channel]) for j in range(i+1,d))
            w[i,channel] = (intermediate[i,channel]-subtotal)/factor[i,i]
    return w


def patient_losses(a, b, q, w):
    losses = []
    for ai, bi, qi in zip(a, b, q):
        linear = math.fsum(float(x)*float(y) for x,y in zip(w.ravel(), bi.ravel()))
        quadratic = math.fsum(float(w[:,k] @ ai @ w[:,k]) for k in range(w.shape[1]))
        losses.append((float(qi) - 2*linear + quadratic)/4)
    return np.asarray(losses)


def regularized_objective(a, b, q, w):
    mse = patient_losses(a[None], b[None], np.array([q]), w)[0]
    penalty = RIDGE*math.fsum(float(v)*float(v) for v in w.ravel())
    return float(4*mse + penalty)


def stationarity(check, a, b, w, label):
    system = a + RIDGE*np.eye(len(a))
    residual = system@w-b
    check.close(residual, np.zeros_like(residual), label + ": normal equation", atol=1e-10, rtol=0)
    scale = float(np.linalg.norm(system, 2)*np.linalg.norm(w) + np.linalg.norm(b))
    check.require(float(np.linalg.norm(residual))/max(scale,np.finfo(float).tiny) <= 1e-10,
                  label + ": normalized stationarity")


def load_inputs(contract, check):
    sources = contract["source_sha256"]
    for filename, digest in sources.items():
        check.require(sha(filename) == digest, "source/input unchanged: " + filename)
    def bound(suffix):
        paths = [Path(p) for p in sources if p.replace("\\","/").endswith(suffix)]
        check.require(len(paths) == 1, "one bound input: " + suffix)
        return paths[0]
    cap = bound("frozen_residual_capacity_20260916_v1/patient_stats.npz").parent
    public = bound("frozen_residual_public32_20260916_v1/patient_stats.npz").parent
    public_models = bound("dp_calibration_v1/public_only_models.npz")
    capacity_models = bound("frozen_residual_capacity_20260916_v1/models.npz")
    cohort_path = bound("cvpr_u_pilot_v1_001/cohort/evaluation_images.csv")
    dp_directory = bound("frozen_residual_patient_dp_20260916_v1/controls.npz").parent
    for folder, status in (
        (cap, "PASS_FROZEN_RESIDUAL_CAPACITY_SAVED_ARITHMETIC_AND_PROVENANCE"),
        (public, "PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE")):
        bound(folder.name + "/contract.json")
        bound(folder.name + "/independent_verification.json")
        verification = read(folder/"independent_verification.json")
        check.require(verification["status"] == status and verification["complete"] is True,
                      "upstream raw-statistics independently verified")
        for name, digest in verification["input_sha256"].items():
            check.require(sha(folder/name) == digest, "upstream verified input: " + name)
    bank, pub = arrays(cap/"patient_stats.npz"), arrays(public/"patient_stats.npz")
    train = bank["splits"] == "train"
    develop = bank["splits"] == "eval"
    check.require(len(bank["patient_ids"]) == 120 and train.sum() == 80 and develop.sum() == 40,
                  "original 80 train and 40 development patients")
    check.require(len(pub["patient_ids"]) == 32 and np.all(pub["splits"] == "public"), "public32 roles")
    groups = [set(pub["patient_ids"].tolist()), set(bank["patient_ids"][train].tolist()),
              set(bank["patient_ids"][develop].tolist())]
    check.require([len(x) for x in groups] == [32,80,40], "unique patients within roles")
    check.require(len(set.union(*groups)) == 152, "public/private/development mutually exclusive")
    check.require(contract["families"] == list(FAMILIES), "two predeclared heads")
    check.require(contract["patient_counts"] == {"public":32,"private":80,"development":40}, "declared roles")
    check.require(contract["ridge"] == RIDGE, "fixed absolute ridge")
    check.require(contract["schema"] == "fixed-pooled-reference/v1", "pooled contract schema")
    check.require(contract["references"] == list(REFERENCES), "four fixed references")
    check.close(contract["public_weight"],32/112,"public equal-patient fraction",atol=0,rtol=0)
    check.close(contract["private_weight"],80/112,"private equal-patient fraction",atol=0,rtol=0)
    check.require(contract["samples_per_patient"] == {"public":2,"private":4,"development":4},
                  "unequal image count does not change patient weight")
    check.require(contract["draws_per_image"] == 8 and not contract["tuning_allowed"]
                  and not contract["search_over_lambda_rho_patients"],"no new noise/parameter search")
    check.require(contract["dp_applied"] is False and contract["generated_images"] == 0
                  and contract["raw_images_opened"] is False,"non-DP cached-only comparison")
    check.require(contract["producer_and_independent_verifier_frozen_before_pooled_execution"] is True,
                  "both source files frozen by producer contract")
    check.require(sha(__file__)==sources[str(Path(__file__).resolve())],"actual verifier bound before execution")
    roles = {}
    with cohort_path.open(encoding="utf-8-sig",newline="") as handle:
        for row in csv.DictReader(handle):
            roles.setdefault(row["eval_role"],set()).add(row["patient_id"])
    check.require(groups[1]==roles["fit"] and groups[2]==roles["selection"],"original cohort role mapping")
    check.require(len(roles["calibration"])==140 and len(roles["test"])==140,"locked cohort counts")
    check.require(not set.union(*groups).intersection(roles["calibration"]|roles["test"]),
                  "no final calibration/test patient inclusion")
    for role, group in zip(("public","private","development"),groups):
        check.require(set(contract["patient_ids"][role])==group and len(contract["patient_ids"][role])==len(group),
                      "contract patient identities: "+role)
    cap_contract,pub_contract = read(cap/"contract.json"),read(public/"contract.json")
    for field in contract["matching_source_fields"]:
        check.require(cap_contract[field]==pub_contract[field],"shared feature definition: "+field)
    check.require(cap_contract["projection_sha256"]==pub_contract["projection_sha256"]
                  ==sha(cap/"projection.npz")==sha(public/"projection.npz"),"exact common projection")
    prior_dp = read(dp_directory/"independent_verification.json")
    check.require(prior_dp["status"]=="PASS_SAVED_PATIENT_DP_MECHANISMS_PUBLIC_CALIBRATION_AND_EVALUATION"
                  and prior_dp["complete"] is True,"previous saved-control arithmetic PASS")
    for filename in ("controls.npz","evaluation_mse.npz"):
        check.require(sha(dp_directory/filename)==prior_dp["input_sha256"][filename],"previous control source: "+filename)
    return {"bank":bank,"public":pub,"train":train,"develop":develop,
            "public_weights":arrays(public_models),"private_weights":arrays(capacity_models),
            "dp_controls":arrays(dp_directory/"controls.npz"),"dp_losses":arrays(dp_directory/"evaluation_mse.npz")}


def verify_math(out, contract, check, inputs):
    bank, pub = inputs["bank"], inputs["public"]
    train, develop = inputs["train"], inputs["develop"]
    weights = arrays(out/"models.npz")
    stats = arrays(out/"pooled_statistics.npz")
    losses = arrays(out/"evaluation_mse.npz")
    check.require(set(weights) == {"W_"+f+"_"+r for f in FAMILIES for r in REFERENCES}, "all eight fixed weights")
    check.require(set(stats) == {m+"_"+f for f in FAMILIES for m in ("A","B","Q")}, "exact pooled moment keys")
    check.require(set(losses) == {"patient_ids"}|{f+"_"+r for f in FAMILIES for r in REFERENCES}, "eight loss arrays")
    check.require(np.array_equal(losses["patient_ids"],bank["patient_ids"][develop]), "exact40 patient order")
    checked = {}
    for family, dimension in (("static",16),("full",64)):
        group = {}
        for name in ("A","B","Q"):
            joined = np.concatenate([pub[name+"_"+family],bank[name+"_"+family][train]],axis=0)
            group[name] = patient_average(joined)
            check.close(stats[name+"_"+family],group[name], family+" equal112 patient "+name)
        own = {}
        for label,source,mask in (("public_only",pub,np.ones(32,dtype=bool)),("private_only",bank,train)):
            own[label] = {name:patient_average(source[name+"_"+family][mask]) for name in ("A","B","Q")}
        own["pooled"] = group
        objectives = {}
        for reference in REFERENCES:
            key = "W_"+family+"_"+reference
            w = weights[key]
            check.require(w.shape == (dimension,4) and w.dtype == np.float64, key+": shape/dtype")
            check.require(np.isfinite(w).all(), key+": finite")
            if reference == "base":
                check.require(np.array_equal(w,np.zeros((dimension,4))), "base head is exactly zero")
            else:
                objective = own[reference]
                independent = cholesky_ridge(objective["A"],objective["B"])
                check.close(w,independent,key+": independent Cholesky solve")
                stationarity(check,objective["A"],objective["B"],w,key)
                if reference in ("public_only","private_only"):
                    previous = inputs["public_weights" if reference=="public_only" else "private_weights"]["W_"+family]
                    check.close(w,previous,key+": refitted previous reference numerical equality")
                    old_key = family+("_public_only" if reference=="public_only" else "_nonprivate")
                    check.close(w,inputs["dp_controls"][old_key],key+": previous DP control equality")
            expected = patient_losses(bank["A_"+family][develop],bank["B_"+family][develop],bank["Q_"+family][develop],w)
            check.close(losses[family+"_"+reference],expected,key+":40 independent patient losses",atol=1e-12,rtol=1e-12)
            if reference in ("public_only","private_only"):
                check.close(expected,inputs["dp_losses"]["control_"+old_key],key+": previous40 control losses",atol=1e-12,rtol=0)
            elif reference=="base":
                check.close(expected,inputs["dp_losses"]["base_mse"],key+": previous40 base losses",atol=1e-12,rtol=0)
            checked[family+"_"+reference] = expected
            objectives[reference] = regularized_objective(group["A"],group["B"],group["Q"],w)
            if reference!="base":
                objective = own[reference]
                checked[family+"_"+reference+"_stationarity"] = float(np.linalg.norm(
                    (objective["A"]+RIDGE*np.eye(dimension))@w-objective["B"]))
        for reference in REFERENCES:
            check.require(objectives["pooled"] <= objectives[reference]+1e-12,
                          family+": pooled regularized training optimum versus "+reference)
        checked[family+"_pooled_training_objectives"] = objectives
    return checked


def verify_summary(out,check,calculated):
    report = read(out/"analysis.json")
    check.require(report["schema"]=="fixed-pooled-reference-analysis/v1","analysis schema")
    check.require(report["development_patients"]==40 and report["generated_quality_measured"] is False
                  and report["significance_test_performed"] is False,"descriptive development-only scope")
    check.require(report["paired_difference_sign"]=="reference minus pooled MSE; positive is improvement",
                  "fixed comparison sign")
    check.require(set(report["families"])==set(FAMILIES),"two summary heads")
    summaries = {}
    for family in FAMILIES:
        result=report["families"][family]
        check.require(set(result["references"])==set(REFERENCES),"four reference summaries")
        for reference in REFERENCES:
            item=result["references"][reference]
            expected_keys={"mean_mse","pooled_training_regularized_objective"}
            if reference!="base":
                expected_keys.add("own_objective_stationarity_norm")
                check.close(item["own_objective_stationarity_norm"],calculated[family+"_"+reference+"_stationarity"],
                            family+"/"+reference+": reported stationarity",atol=1e-12,rtol=1e-10)
            check.require(set(item)==expected_keys,"complete reference summary fields")
            check.close(item["mean_mse"],math.fsum(calculated[family+"_"+reference])/40,
                        family+"/"+reference+": patient mean MSE",atol=1e-12,rtol=0)
            check.close(item["pooled_training_regularized_objective"],
                        calculated[family+"_pooled_training_objectives"][reference],
                        family+"/"+reference+": reported regularized training objective",atol=1e-12,rtol=1e-10)
        check.require(set(result["paired_comparisons"])==
                      {"pooled_vs_"+r for r in ("public_only","private_only","base")},"three predeclared contrasts")
        paired={}
        for reference in ("public_only","private_only","base"):
            values=calculated[family+"_"+reference]
            delta=values-calculated[family+"_pooled"]
            ordered=sorted(delta.tolist())
            mean=math.fsum(delta)/40
            expected={"mean_improvement":mean,"relative_reduction_percent":100*mean/(math.fsum(values)/40),
                "patient_improved":int(sum(float(x)>0 for x in delta)),
                "patient_tied":int(sum(float(x)==0 for x in delta)),
                "patient_worse":int(sum(float(x)<0 for x in delta)),
                "median_improvement":(ordered[19]+ordered[20])/2,
                "minimum_improvement":ordered[0],"maximum_improvement":ordered[-1]}
            actual=result["paired_comparisons"]["pooled_vs_"+reference]
            check.require(set(actual)==set(expected),"complete paired summary fields")
            for key,value in expected.items():
                if isinstance(value,int):
                    check.require(actual[key]==value,family+"/"+reference+": exact "+key)
                else:
                    check.close(actual[key],value,family+"/"+reference+": "+key,atol=1e-12,rtol=1e-10)
            paired["pooled_vs_"+reference]=expected
        summaries[family]=paired
    return summaries


def verify(out):
    out=Path(out).resolve()
    result_path=out/"independent_verification.json"
    protocol_path=out/"independent_verification_protocol.json"
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve existing pooled-reference verification attempt")
    execution=read(out/"execution.json")
    if execution.get("completed") is not True:
        raise RuntimeError("Completed producer execution required")
    names=("contract.json","execution.json","models.npz","pooled_statistics.npz","evaluation_mse.npz","analysis.json")
    bound={name:sha(out/name) for name in names}
    source_hash=sha(__file__)
    protocol={"schema":"independent-pooled-reference-verification/v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),"code_sha256":source_hash,"input_sha256":bound,
        "independence":"No producer numerical imports. Concatenated112-patient moments, explicit sums, Cholesky triangular substitution, patientwise quadratic losses.",
        "tolerances":{"statistics_weights":{"atol":1e-12,"rtol":1e-10},
                      "patient_losses":{"atol":1e-12,"rtol":1e-12},"stationarity_absolute":1e-10},
        "scope":"Saved non-DP references and reused development40 only; no generation quality or significance claim",
        "new_backbone_forward":0,"new_backbone_backward":0}
    write_once(protocol_path,protocol)
    started=time.perf_counter()
    check=Checks()
    try:
        contract=read(out/"contract.json")
        check.require(execution["contract_sha256"]==bound["contract.json"],"execution contract binding")
        check.require(set(execution["output_sha256"])==set(names)-{"contract.json","execution.json"},
                      "all output files bound")
        for name,digest in execution["output_sha256"].items():
            check.require(bound[name]==digest,"execution output binding: "+name)
        check.require(execution["new_pooled_models"]==2 and execution["reference_refits"]==4,
                      "two pooled solves/four fixed reference refits")
        check.require(execution["backbone_forward"]==execution["backbone_backward"]==execution["generated_images"]==0,
                      "cached CPU-only experiment")
        inputs=load_inputs(contract,check)
        computed=verify_math(out,contract,check,inputs)
        summaries=verify_summary(out,check,computed)
        for name,digest in bound.items():
            check.require(sha(out/name)==digest,"unchanged output: "+name)
        for name,digest in contract["source_sha256"].items():
            check.require(sha(name)==digest,"unchanged source/input: "+name)
        check.require(sha(__file__)==source_hash,"verifier unchanged")
        result={"status":STATUS,"complete":True,"checks":check.count,"seconds":time.perf_counter()-started,
            "code_sha256":source_hash,"protocol_sha256":sha(protocol_path),"input_sha256":bound,
            "maximum_absolute_errors":check.errors,"paired_comparisons":summaries,
            "weights_checked":8,"new_pooled_solutions_checked":2,"development_patient_losses_checked":320,
            "scope":"Fixed patient-mean non-DP ridge arithmetic; no inferential test, generated quality or private-data value conclusion.",
            "new_backbone_forward":0,"new_backbone_backward":0}
        write_once(result_path,result)
        return result
    except Exception as exc:
        write_once(result_path,{"status":"FAILED_POOLED_REFERENCE_INDEPENDENT_VERIFICATION","complete":False,
            "error":repr(exc),"traceback":traceback.format_exc(),"checks":check.count,
            "maximum_absolute_errors":check.errors,"seconds":time.perf_counter()-started,
            "code_sha256":source_hash,"protocol_sha256":sha(protocol_path),"input_sha256":bound})
        raise


def self_test():
    rng = np.random.default_rng(269161)
    x = rng.normal(size=(5,9,4)); y = rng.normal(size=(5,9,2))
    a = np.array([z.T@z/9 for z in x])
    b = np.array([xx.T@yy/9 for xx,yy in zip(x,y)])
    q = np.array([np.sum(yy**2)/9 for yy in y])
    check = Checks()
    am,bm,qm = map(patient_average,(a,b,q))
    w = cholesky_ridge(am,bm)
    stationarity(check,am,bm,w,"synthetic")
    check.close(w,np.linalg.solve(am+RIDGE*np.eye(4),bm),"synthetic independent solve")
    direct = np.array([np.mean((xx@w-yy)**2) for xx,yy in zip(x,y)])
    # This helper's fixed four-output convention is checked with duplicated channels.
    check.close(patient_losses(a,np.concatenate([b,b],axis=2),2*q,np.concatenate([w,w],axis=1)),
                direct,"synthetic raw residual loss")
    return {"status":"PASS_SYNTHETIC_POOLED_RIDGE_AND_RAW_LOSS","checks":check.count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--expected-code-sha256")
    args=parser.parse_args()
    if args.expected_code_sha256 and sha(__file__)!=args.expected_code_sha256:
        raise AssertionError("Verifier source hash mismatch")
    result=self_test() if args.self_test else verify(args.out)
    print(json.dumps({key:result[key] for key in
          ("status","complete","checks","seconds","code_sha256","weights_checked","development_patient_losses_checked")
          if key in result},indent=2))


if __name__=="__main__":
    main()
