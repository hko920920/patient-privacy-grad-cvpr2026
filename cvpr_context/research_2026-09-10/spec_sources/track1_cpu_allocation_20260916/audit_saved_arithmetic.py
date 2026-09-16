"""Saved-output and analytic checks; does not import or rerun the experiment."""
import csv,hashlib,json,math,time
from pathlib import Path
P=Path(__file__).resolve().parent
start=time.perf_counter()
def read(n):return json.loads((P/n).read_text(encoding="utf-8"))
def sha(n):return hashlib.sha256((P/n).read_bytes()).hexdigest()
r=read("results.json");protocol=read("protocol.json");manifest=read("manifest.json")
assert r["execution"]["binding"]["script_sha256"]==sha("run_cpu_allocation.py")
assert r["execution"]["binding"]["protocol_sha256"]==sha("protocol.json")
assert all(sha(k)==v for k,v in manifest["files_sha256"].items())
assert len(r["rows"])==161 and len(r["paired_comparisons"])==42
assert len(list(csv.DictReader((P/"results.csv").open(encoding="utf-8-sig"))))==161
assert len(list(csv.DictReader((P/"paired_comparisons.csv").open(encoding="utf-8-sig"))))==42
largest_cost_error=largest_mse_error=0.
for row in r["rows"]:
    hist=row["R_frequencies_by_type"]
    assert all(abs(sum(x)-1)<1e-14 for x in hist)
    pilot=2 if row["method"].startswith(("raw_","geometry_","oracle_charged_")) else 0
    cost=sum(sum(q*n for q,n in zip(x,[1,2,4,8,16])) for x in hist)/2+pilot
    largest_cost_error=max(largest_cost_error,abs(cost-row["actual_mean_cost"]))
    largest_mse_error=max(largest_mse_error,abs(sum(row["mse_by_type"])/2-row["clipped_mse"]))
assert largest_cost_error<1e-12 and largest_mse_error<1e-12
normal_cdf=lambda z:.5*math.erfc(-z/math.sqrt(2))
normal_pdf=lambda z:math.exp(-z*z/2)/math.sqrt(2*math.pi)
def radial_clipped_mse(mu,s):
    m=mu-1
    def moment(b):
        z=(b-mu)/s
        return (m*m+s*s)*normal_cdf(z)-(2*m*s+s*s*z)*normal_pdf(z)
    return max(0.,moment(1)-moment(-1)+4*normal_cdf((-1-mu)/s))
f1=read("F1_exact_discrete_outside.json")
f2=read("F2_gaussian_outside_heterogeneous.json")
f1_rows={x["method"]:x for x in f1["rows"]}
f2_rows={x["method"]:x for x in f2["rows"]}
exact=[]
for n in [1,2,4,8,16]:
    truth=sum(math.comb(n,k)/2**n*(2-4/math.sqrt(4+(.1*(2*k-n)/n)**2)) for k in range(n+1))/2
    observed=f1_rows["fixed_R"+str(n)]
    exact.append({"R":n,"exact_population_mse":truth,"observed":observed["clipped_mse"],
                  "MC_z":(observed["clipped_mse"]-truth)/observed["mc_se"] if observed["mc_se"]>1e-15 else None})
radial={str(n):radial_clipped_mse(2,.5/math.sqrt(n)) for n in [1,2,4,8,16]}
geo=f2_rows["geometry_B8"]
pred_radial=sum(p*radial[str(n)] for p,n in zip(geo["R_frequencies_by_type"][0],[1,2,4,8,16]))
degenerate=[]
for f in r["family_files"]:
    d=read(f)
    aa=[d["calibration"][t["name"]]["true_delta_method_a"] for t in d["family"]["types"]]
    if all(a is not None for a in aa) and aa[0]==aa[1]:
        degenerate.append({"family":d["family"]["id"],"correct_interpretation":"proxy-vs-true-a correlation undefined because true a has zero population variance",
                          "saved_numeric_diagnostic":d["calibration"]["proxy_true_a_correlations"],
                          "source_results_preserved":True})
result={"status":"PASS_SAVED_COST_MSE_BINDINGS_AND_ANALYTIC_CHECKS",
        "rows":161,"paired_rows":42,"input_hashes":{k:sha(k) for k in ["protocol.json","run_cpu_allocation.py","results.json","manifest.json"]},
        "audit_code_sha256":sha(Path(__file__).name),"max_cost_reconstruction_error":largest_cost_error,
        "max_type_mean_mse_reconstruction_error":largest_mse_error,
        "exact_F1_fixed_binomial":exact,
        "F2_radial_exact_mse_by_R":radial,
        "F2_geometry_B8_radial_mse_from_recorded_R_frequencies":pred_radial,
        "F2_geometry_B8_observed_radial_mse":geo["mse_by_type"][0],
        "constant_true_a_correlation_caveat":degenerate,
        "scope":"Independent saved arithmetic and exact special-case expectations, not a second full MC rerun or neural verification.",
        "seconds":time.perf_counter()-start}
with (P/"arithmetic_audit.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(result,ensure_ascii=False))

