"""Eight predeclared patient-cluster reference-set tests of frozen CDI scores.
No target calls, fitting, cutoff choice, or individual-patient membership claim.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Tuple
import warnings
import numpy as np
from scipy.stats import ttest_ind,t as student_t
from scipy.stats._result_classes import TtestResult

ROOT=Path(__file__).resolve().parent.parent
CONTRACT=ROOT.parent/"CVPR 주제 탐색/research_2026-09-10/spec_sources/cdi_reference_set_test_contract_20260915.json"
CONTRACT_SHA="75b1686160ba6fd966227615ea2b16774db4960d153f5bdcd1d60b35361f25da"

def require(c,m):
    if not c:raise AssertionError(m)

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def write(p,x):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def load_source(c):
    path=Path(c["source_function_file"])
    require(digest(path)==c["frozen_sha256"][str(path.resolve())],"unchanged original evaluator")
    tree=ast.parse(path.read_text(encoding="utf-8"))
    names=c["source_functions"]
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require([n.name for n in nodes]==names,"exact pinned source functions")
    namespace=dict(np=np,ttest_ind=ttest_ind,TtestResult=TtestResult,Tuple=Tuple)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["get_p_value"]

def independent_welch(p,r):
    n,m=len(p),len(r)
    pm,rm=math.fsum(map(float,p))/n,math.fsum(map(float,r))/m
    pv=math.fsum((float(x)-pm)**2 for x in p)/(n-1)
    rv=math.fsum((float(x)-rm)**2 for x in r)/(m-1)
    a,b=pv/n,rv/m;se2=a+b
    if se2==0:
        return dict(valid=False,reason="zero estimated standard error",positive_mean=pm,reference_mean=rm,
                    positive_sample_variance=pv,reference_sample_variance=rv)
    t=(pm-rm)/math.sqrt(se2)
    df=se2**2/(a*a/(n-1)+b*b/(m-1))
    return dict(valid=True,n_positive=n,n_reference=m,positive_mean=pm,reference_mean=rm,
        mean_difference=pm-rm,positive_sample_variance=pv,reference_sample_variance=rv,
        standard_error=math.sqrt(se2),t_statistic=t,welch_df=df,
        p_greater=float(student_t.sf(t,df)),p_two_sided=float(2*student_t.sf(abs(t),df)))

def compute(p,r,source):
    p=np.asarray(p,dtype=np.float64);r=np.asarray(r,dtype=np.float64)
    require(p.ndim==r.ndim==1 and p.shape==r.shape and len(p)>1
            and np.isfinite(p).all() and np.isfinite(r).all(),"finite equal independent-unit vectors")
    independent=independent_welch(p,r)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source_p,source_order=source(p,r,members_lower=False)
        paper=ttest_ind(p,r,equal_var=False,alternative="greater")
    warning_messages=[str(w.message) for w in caught]
    if not independent["valid"] or not all(math.isfinite(float(v)) for v in (source_p,paper.statistic,paper.df,paper.pvalue)):
        return dict(status="DEGENERATE_OR_NONFINITE_TEST_NO_INFERENCE",
            independent=independent,warnings=warning_messages)
    expected=(independent["t_statistic"],independent["welch_df"],independent["p_greater"],independent["p_two_sided"])
    actual=(float(paper.statistic),float(paper.df),float(paper.pvalue),float(source_p))
    require(np.allclose(actual,expected,rtol=2e-12,atol=2e-12),"independent Welch arithmetic")
    require(bool(source_order)==(independent["t_statistic"]>0),"source separate direction")
    require(np.isclose(float(source_p)/2 if source_order else 1-float(source_p)/2,
                       float(paper.pvalue),rtol=2e-12,atol=2e-12),"two-sided vs greater relation")
    return dict(status="PASS_SOURCE_AND_INDEPENDENT_WELCH_ARITHMETIC",
        source_two_sided_p=float(source_p),source_correct_order=bool(source_order),
        paper_specification_one_sided_greater_p=float(paper.pvalue),
        scipy_t_statistic=float(paper.statistic),scipy_welch_df=float(paper.df),
        independent=independent,absolute_arithmetic_discrepancies=[abs(a-b) for a,b in zip(actual,expected)],
        warnings=warning_messages)

def validate():
    require(digest(CONTRACT)==CONTRACT_SHA,"predeclared reference-test contract")
    c=read(CONTRACT)
    require(c["schema"]=="cdi-patient-reference-test-contract/v1" and c["expected_tests"]==8
            and c["method"]=="cdi_image_mean_fixed" and c["nominal_p_cutoff"] is None,"fixed narrow scope")
    for p,h in c["frozen_sha256"].items():require(digest(p)==h,"unchanged bound source")
    return c

def run(expected_sha):
    start=time.perf_counter();source_sha=digest(__file__)
    require(expected_sha==source_sha,"externally frozen CPU runner")
    c=validate();directory=Path(c["eu_analysis_directory"]);out=Path(c["output_directory"])
    require(not out.exists(),"immutable reference-test output")
    inputs=[CONTRACT]+[Path(p) for p in c["frozen_sha256"]]
    inputs += [directory/n for n in ("protocol.json","contract_copy.json","analysis.json","predictions.json",
                                    "fit_predictions.json","fit_parameters.json","provenance.json",c["required_verification_file"])]
    hashes={str(p.resolve()):digest(p) for p in inputs}
    out.mkdir()
    write(out/"protocol.json",dict(schema="cdi-patient-reference-test-execution/v1",
        created_utc=datetime.now(timezone.utc).isoformat(),source_path=str(Path(__file__).resolve()),
        source_sha256=source_sha,analysis_contract_sha256=CONTRACT_SHA,input_sha256=hashes,
        policy=c,performance_read_begins_after_this_protocol=True))
    verification=read(directory/c["required_verification_file"])
    require(verification["status"]==c["required_verification_status"],"independent EU statistics PASS required")
    provenance=read(directory/"provenance.json")
    for name,h in provenance["output_sha256"].items():require(digest(directory/name)==h,"EU artifact exact")
    require(provenance["analysis_contract_sha256"]==c["eu_analysis_contract_sha256"],"same four-cell contract")
    eu_contract=read(directory/"contract_copy.json")
    require(eu_contract["patient_rows"] and [p for p in eu_contract["patient_rows"] if p["role"]=="selection"]==c["selection_patients"],"exact patient cohort")
    fit_ids=c["fit_patient_ids"];selection_ids=[p["patient_id"] for p in c["selection_patients"]]
    require(len(fit_ids)==len(set(fit_ids))==80 and len(selection_ids)==40
            and not set(fit_ids)&set(selection_ids),"patient-disjoint control/test")
    params={(r["fit_scenario"],r["model"],r["method"]):r for r in read(directory/"fit_parameters.json")}
    rows=read(directory/"predictions.json")
    rows={(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"],r["patient_id"]):r for r in rows}
    source=load_source(c);tests=[];score_rows=[]
    for fs in c["fit_scenarios"]:
        for es in c["evaluation_scenarios"]:
            for model in c["models"]:
                entry=params[(fs,model,c["method"])]
                require(entry["fitting_patient_ids"]==fit_ids and entry["fitting_rows"]==160
                        and entry["chosen_C"]==1 and entry["C_policy"]=="fixed"
                        and entry["representation"]=="image26" and entry["parameters"]["classes"]==[0,1],"frozen fixed image scorer")
                positive_group="A" if model=="model_1" else "B"
                p,r=[],[];pids,rids=[],[]
                for patient in c["selection_patients"]:
                    pid=patient["patient_id"];row=rows[(fs,es,model,c["method"],pid)]
                    member=int(patient["group"]==positive_group)
                    require(row["member"]==member and row["group"]==patient["group"] and row["role"]=="selection","patient reference labels")
                    value=float(row["score"]);require(math.isfinite(value) and 0<=value<=1,"source probability mean")
                    (p if member else r).append(value);(pids if member else rids).append(pid)
                    score_rows.append(dict(fit_scenario=fs,evaluation_scenario=es,model=model,patient_id=pid,
                        group=patient["group"],patient_member=member,score=value,
                        query_image_member=member if es=="E" else 0,statistical_unit="patient"))
                require(len(p)==len(r)==20,"20 independent patient units per group")
                measured=compute(p,r,source)
                tests.append(dict(fit_scenario=fs,evaluation_scenario=es,model=model,method=c["method"],
                    positive_patient_ids=pids,reference_patient_ids=rids,positive_scores=p,reference_scores=r,
                    control_patients=80,positive_test_patients=20,reference_test_patients=20,**measured))
    require(len(tests)==8 and len(score_rows)==320,"fixed eight test contexts")
    for p,h in hashes.items():require(digest(p)==h,"unchanged source/input after tests")
    require(digest(__file__)==source_sha,"unchanged runner")
    write(out/"scores.json",score_rows)
    write(out/"analysis.json",dict(schema="cdi-patient-reference-test-result/v1",
        status="COMPUTED_WITH_SOURCE_AND_INDEPENDENT_ARITHMETIC_CHECKS_PENDING_ROOT_REVIEW",
        research_stage=2,tests=tests,computed_tests=8,statistical_unit="patient",new_fitting=False,new_GPU_calls=0,
        no_individual_patient_inference=True,no_confirmatory_FPR_or_causal_claim=True,
        unadjusted_reused_development_data=True,original_outputs_unchanged=True,
        overall_stage2_gate=None,seconds_cpu=time.perf_counter()-start))
    write(out/"provenance.json",dict(source_sha256=source_sha,contract_sha256=CONTRACT_SHA,input_sha256=hashes,
        protocol_sha256=digest(out/"protocol.json"),
        output_sha256={n:digest(out/n) for n in ("scores.json","analysis.json")}))
    print(json.dumps(dict(status="COMPUTED_PENDING_ROOT_REVIEW",tests=8,output=str(out),
                          degenerate_tests=sum(t["status"]!="PASS_SOURCE_AND_INDEPENDENT_WELCH_ARITHMETIC" for t in tests))))

def self_test():
    c=validate();source=load_source(c)
    rng=np.random.default_rng(991)
    p=rng.normal(.7,.11,20);r=rng.normal(.4,.23,20)
    for a,b in ((p,r),(r,p),(np.arange(20)/50,np.arange(20)/50)):
        answer=compute(a,b,source)
        require(answer["status"]=="PASS_SOURCE_AND_INDEPENDENT_WELCH_ARITHMETIC","finite unequal-variance Welch")
    require(compute(np.ones(20),np.ones(20),source)["status"]=="DEGENERATE_OR_NONFINITE_TEST_NO_INFERENCE","degenerate score handling")
    import builtins,symtable
    table=symtable.symtable(Path(__file__).read_text(encoding="utf-8"),__file__,"exec")
    missing=[]
    def walk(scope):
        if scope.get_type()=="function":
            for sym in scope.get_symbols():
                if sym.is_global() and sym.is_referenced() and sym.get_name() not in globals() and not hasattr(builtins,sym.get_name()):
                    missing.append((scope.get_name(),sym.get_name()))
        for child in scope.get_children():walk(child)
    walk(table);require(not missing,"global dependencies "+repr(missing))
    print("PASS_UNMODIFIED_SOURCE_TWO_SIDED_ONE_SIDED_INDEPENDENT_WELCH_DEGENERACY; no actual scores read")

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test",action="store_true");mode.add_argument("--run",action="store_true")
    p.add_argument("--expected-code-sha256")
    a=p.parse_args()
    if a.self_test:self_test();return
    require(a.expected_code_sha256,"externally frozen source hash required")
    run(a.expected_code_sha256.lower())

if __name__=="__main__":main()

