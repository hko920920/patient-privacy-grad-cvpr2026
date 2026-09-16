"""Fixed patient-DP four-cell experiment after public-only calibration.

This is a reproducible privacy-mechanism simulation on public NIH data.
Internal source statistics, control models, seeds and traces are NOT DP releases.
"""
import os
for env in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[env]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import dp_mechanisms as dm
from . import public_calibrate_dp as pc

ROOT=Path(__file__).resolve().parents[1]
CAP=ROOT/"_reports/frozen_residual_capacity_20260916_v1"
PUB=ROOT/"_reports/frozen_residual_public32_20260916_v1"
CAL=PUB/"dp_calibration_v1"
OUT=ROOT/"_reports/frozen_residual_patient_dp_20260916_v1"
FAMILIES=("static","full")
REPEATS=16
CONTROL_REPEATS=3

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def save(p,data):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write("\n")

def npz(p,**arr):
    with Path(p).open("xb") as f:
        np.savez_compressed(f,**arr)

def seed(*parts):
    s="fixed-patient-dp-20260916-v1|"+"|".join(map(str,parts))
    return int(hashlib.sha256(s.encode()).hexdigest()[:15],16)

def now():
    return datetime.now(timezone.utc).isoformat()

def mse(a,b,q,w):
    return (q-2*np.einsum("ij,nij->n",w,b)+np.einsum("ij,nij->n",w,np.einsum("nij,jk->nik",a,w)))/4

def compact_params(p,mechanism):
    names=("a0","b0","clip_norm","sigma","public_denominator","ridge","eigenvalue_floor") if mechanism=="SSP" else ("steps","q","public_denominator","clip_norm","sigma","ridge")
    return {name:p[name] for name in names}

def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    params=read(CAL/"params.json")
    assert params["complete"],"public optimizer/calibration is incomplete"
    assert all(name in params["selected"] for name in FAMILIES)
    assert (CAL/"independent_selection_verification.json").exists(),"public selection requires independent verification"
    assert str(read(CAL/"independent_selection_verification.json").get("status","")).startswith("PASS")
    accounts=read(CAL/"accounting.json")
    for family in FAMILIES:
        for mechanism in ("SSP","SGD"):
            par=params["selected"][family][mechanism]
            assert par["public_denominator"]==80 and par["ridge"]==.001
            q,steps=(1.,1) if mechanism=="SSP" else (par["q"],par["steps"])
            if mechanism=="SGD":
                assert q==.1 and steps in (500,2000,8000)
                assert par["schedule"]=="constant80_linear20_end01"
            accountant=accounts[f"q{q}_T{steps}"]
            assert accountant["poisson_sample_rate"]==q and accountant["steps"]==steps
            assert accountant["target_epsilon"]==8. and accountant["target_delta"]==1e-5
            assert accountant["conservative_epsilon"]<=8.+1e-10
            assert abs(par["sigma"]-accountant["noise_multiplier"])<=1e-14
    assert params["selection_contract_sha256"]==sha(CAL/"selection_contract.json")
    for name,value in params["code_sha256"].items():
        assert sha(name)==value,name
    assert (PUB/"independent_verification.json").exists(),"public statistics require independent verification"
    verification=read(PUB/"independent_verification.json")
    assert str(verification.get("status","")).startswith("PASS")
    sources=[Path(__file__),Path(dm.__file__),Path(pc.__file__),
        CAP/"contract.json",CAP/"patient_stats.npz",CAP/"models.npz",
        CAP/"independent_verification.json",PUB/"contract.json",PUB/"patient_stats.npz",
        PUB/"independent_verification.json",CAL/"selection_contract.json",
        CAL/"params.json",CAL/"public_only_models.npz",CAL/"accounting.json",
        CAL/"independent_selection_verification.json"]
    with np.load(CAP/"patient_stats.npz") as st:
        assert (st["splits"]=="train").sum()==80 and (st["splits"]=="eval").sum()==40
        assert len(set(st["patient_ids"].tolist()))==120
    out.mkdir(parents=True)
    c=dict(schema="fixed-patient-dp-comparison/v1",created_utc=now(),
        source_sha256={str(p):sha(p) for p in sources},
        public_parameters=params,patient_counts={"train":80,"development":40,"public":32},
        adjacency="add_remove_one_patient",fixed_public_denominator=80,epsilon=8.,delta=1e-5,
        families=list(FAMILIES),mechanisms=["SSP","SGD"],noise_repeats=REPEATS,
        no_noise_sgd_repeats=CONTROL_REPEATS,initialization="zero for all private SGD",
        control_policy="public-only ridge; nonprivate fit80 ridge; SSP clip-only floor0 and selected floor; SGD sigma0 with and without clipping",
        random_policy="independent hash-derived sampling/noise streams; fixed 16 repeats, no best seed selection",
        seed_salt="fixed-patient-dp-20260916-v1",shared_feature_cache_for_both_methods=True,
        expected_new_backbone_forward=0,expected_new_backbone_backward=0,
        metric="equal-patient denoising MSE, channel-SUM quadratic divided by4",
        primary_comparisons=["each DP method versus frozen base","SSP versus SGD within identical head",
                             "each DP method versus same-head public-only ridge"],
        evaluation_scope="previously used development40; fixed data/noise feature bank; no final-test claim",
        release_scope="internal public-data simulation; entire reproducibility bundle is NOT a DP release",
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        numpy_version=np.__version__,
        privacy_tuning_on_fit80_or_eval40=False,no_adaptive_private_seed_clip_lr_floor_selection=True,
        generative_quality_measured=False)
    save(out/"contract.json",c)
    print(json.dumps({"phase":"private_comparison_prepared","contract_sha256":sha(out/"contract.json")}),flush=True)

def validate(out):
    c=read(out/"contract.json")
    for name,digest in c["source_sha256"].items():
        assert sha(name)==digest,name
    return c

def run(out):
    c=validate(out)
    if (out/"execution.json").exists():
        raise FileExistsError(out/"execution.json")
    params=c["public_parameters"]["selected"]
    bank=np.load(CAP/"patient_stats.npz")
    public_models=np.load(CAL/"public_only_models.npz")
    original_models=np.load(CAP/"models.npz")
    mask=bank["splits"]=="train"
    records=[]
    modeldir=out/"models"
    modeldir.mkdir(exist_ok=False)
    started=time.perf_counter()
    for family in FAMILIES:
        a,b=bank["A_"+family][mask],bank["B_"+family][mask]
        for mechanism in ("SSP","SGD"):
            p=params[family][mechanism]
            kw=compact_params(p,mechanism)
            for rep in range(REPEATS):
                start=time.perf_counter()
                if mechanism=="SSP":
                    result=dm.ssp_release(a,b,**kw,seed=seed(family,mechanism,rep,"noise"),include_diagnostics=True)
                else:
                    result=dm.poisson_sgd(a,b,**kw,learning_rates=pc.learning_rates(p["steps"],p["initial_lr"]),
                        sampling_seed=seed(family,mechanism,rep,"sampling"),noise_seed=seed(family,mechanism,rep,"noise"),
                        include_diagnostics=True)
                elapsed=time.perf_counter()-start
                arrays={"W":result["W"]}
                diag=result["diagnostics"]
                summary={}
                if mechanism=="SSP":
                    arrays.update({k:result[k] for k in ("A_released","A_projected","B_released","released_vector")})
                    summary={"clipped_patients":int(diag["clipped_patients"]),
                        "noisy_gram_min_eigenvalue":float(np.min(diag["noisy_gram_eigenvalues"])),
                        "stationarity_norm":float(diag["stationarity_norm"])}
                else:
                    summary={"empty_batches":int(diag["empty_batches"]),
                        "patient_gradient_evaluations":int(diag["patient_gradient_evaluations"]),
                        "clipped_patient_steps":int(np.sum(diag["clipped_patients"]))}
                if rep==0:
                    arrays.update({"trace_"+k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
                path=modeldir/(family+"_"+mechanism+"_%02d.npz"%rep)
                npz(path,**arrays)
                records.append(dict(family=family,mechanism=mechanism,repeat=rep,path=path.relative_to(out).as_posix(),
                    sha256=sha(path),seconds=elapsed,sampling_seed=seed(family,mechanism,rep,"sampling"),
                    noise_seed=seed(family,mechanism,rep,"noise"),internal_diagnostics=summary))
                if rep in (0,7,15):
                    print(json.dumps(dict(phase="DP_fit",family=family,mechanism=mechanism,completed=rep+1,total=REPEATS,seconds=round(time.perf_counter()-started,3))),flush=True)
    controls={}
    control_info={}
    for family in FAMILIES:
        a,b=bank["A_"+family][mask],bank["B_"+family][mask]
        controls[family+"_public_only"]=public_models["W_"+family]
        controls[family+"_nonprivate"]=original_models["W_"+family]
        sp=params[family]["SSP"]
        for name,floor in [("SSP_clip_only",0.),("SSP_clip_and_floor",sp["eigenvalue_floor"])]:
            kw=compact_params(sp,"SSP")
            kw.update(sigma=0.,eigenvalue_floor=floor)
            controls[family+"_"+name]=dm.ssp_release(a,b,**kw,seed=seed(family,name,0,"noise"))["W"]
        sg=params[family]["SGD"]
        for name,clip in [("SGD_no_noise",sg["clip_norm"]),("SGD_no_noise_no_clip",float("inf"))]:
            for rep in range(CONTROL_REPEATS):
                kw=compact_params(sg,"SGD")
                kw.update(sigma=0.,clip_norm=clip)
                result=dm.poisson_sgd(a,b,**kw,learning_rates=pc.learning_rates(sg["steps"],sg["initial_lr"]),
                    sampling_seed=seed(family,"SGD",rep,"sampling"),noise_seed=seed(family,"SGD",rep,"noise"))
                key=family+"_"+name+"_%02d"%rep
                controls[key]=result["W"]
                control_info[key]={"sampling_seed":seed(family,"SGD",rep,"sampling"),"noise_seed":seed(family,"SGD",rep,"noise")}
    npz(out/"controls.npz",**controls)
    save(out/"models_manifest.json",records)
    save(out/"execution.json",dict(completed=True,utc=now(),total_seconds=time.perf_counter()-started,
        learned_dp_models=len(records),noise_repeats=REPEATS,control_weights=len(controls),
        contract_sha256=sha(out/"contract.json"),manifest_sha256=sha(out/"models_manifest.json"),
        controls_sha256=sha(out/"controls.npz"),control_seed_manifest=control_info,
        backbone_forward=0,backbone_backward=0,public_selection_cost_excluded=True))
    print("DP_EXECUTION_COMPLETED",flush=True)

def analyze(out):
    c=validate(out)
    if (out/"analysis.json").exists():
        raise FileExistsError(out/"analysis.json")
    execution=read(out/"execution.json")
    assert execution["completed"] and sha(out/"models_manifest.json")==execution["manifest_sha256"]
    assert sha(out/"contract.json")==execution["contract_sha256"]
    assert sha(out/"controls.npz")==execution["controls_sha256"]
    bank=np.load(CAP/"patient_stats.npz")
    controls=np.load(out/"controls.npz")
    mask=bank["splits"]=="eval"
    ids=bank["patient_ids"][mask]
    archive={"patient_ids":ids,"base_mse":bank["Q_full"][mask]/4}
    base=archive["base_mse"]
    for name in controls.files:
        family=name.split("_")[0]
        archive["control_"+name]=mse(bank["A_"+family][mask],bank["B_"+family][mask],bank["Q_"+family][mask],controls[name])
    groups={}
    for row in read(out/"models_manifest.json"):
        path=out/row["path"]
        assert sha(path)==row["sha256"]
        family=row["family"]
        w=np.load(path)["W"]
        values=mse(bank["A_"+family][mask],bank["B_"+family][mask],bank["Q_"+family][mask],w)
        groups.setdefault(family+"_"+row["mechanism"],[]).append(values)
    summaries={}
    for name,values in groups.items():
        array=np.stack(values)
        archive["mse_"+name]=array
        seed_means=array.mean(axis=1)
        pub=archive["control_"+name.split("_")[0]+"_public_only"]
        summaries[name]=dict(mean_mse=float(array.mean()),sd_between_noise_seeds=float(seed_means.std(ddof=1)),
            min_seed_mean=float(seed_means.min()),max_seed_mean=float(seed_means.max()),
            relative_reduction_from_base_percent=float(100*(base.mean()-array.mean())/base.mean()),
            repeats_better_than_base=int((seed_means<base.mean()).sum()),
            relative_reduction_from_public_only_percent=float(100*(pub.mean()-array.mean())/pub.mean()),
            repeats_better_than_public_only=int((seed_means<pub.mean()).sum()),
            positive_patients_against_base_for_mean_model_loss=int((array.mean(axis=0)<base).sum()),
            mean_fit_seconds=float(np.mean([r["seconds"] for r in read(out/"models_manifest.json") if r["family"]+"_"+r["mechanism"]==name])))
    npz(out/"evaluation_mse.npz",**archive)
    report=dict(schema="fixed-patient-dp-comparison-analysis/v1",utc=now(),mean_base_mse=float(base.mean()),
        summaries=summaries,controls_mean_mse={name:float(value.mean()) for name,value in archive.items() if name.startswith("control_")},
        same_head_ssp_minus_sgd_mean_mse={family:float(np.mean(groups[family+"_SSP"])-np.mean(groups[family+"_SGD"])) for family in FAMILIES},
        scope="development40; 16 predeclared mechanism random repeats conditional on fixed input bank, not16 datasets; descriptive mean/SD, no best seed",
        public_data_only_baseline_included=True,generation_quality_measured=False,
        source_sha256={f:sha(out/f) for f in ("contract.json","execution.json","models_manifest.json","controls.npz","evaluation_mse.npz")})
    save(out/"analysis.json",report)
    print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["prepare","run","analyze"],required=True)
    p.add_argument("--output-dir",type=Path,default=OUT)
    a=p.parse_args()
    {"prepare":prepare,"run":run,"analyze":analyze}[a.phase](a.output_dir.resolve())
if __name__=="__main__":
    main()
