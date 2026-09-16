"""Independent CPU arithmetic audit of the saved four-cell DP experiment.

No producer mechanism, calibration or fitting function is imported. Replays
verify existing saved runs; they do not introduce new experimental model fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable,"4")
import numpy as np

EPS=float(np.finfo(np.float64).eps)
MECHANISM_SHA="b490512eae6fd38c6553c74c46070532d3f08a056f52430086d184f12a2cdbdd"
STATUS="PASS_SAVED_PATIENT_DP_MECHANISMS_PUBLIC_CALIBRATION_AND_EVALUATION"
CALIBRATION_PASS="PASS_PUBLIC_CALIBRATION_INDEPENDENT_SELECTION_ARITHMETIC_AND_ACCOUNTING"
CALIBRATION_VERIFIER_SHA="4ea4a075d025d63556d3295690550d5c101f4b6e6044238ab0a9e5e1125bb916"
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"_reports/frozen_residual_patient_dp_20260916_v1"
CALIBRATION_DEFAULT=ROOT/"_reports/frozen_residual_public32_20260916_v1/dp_calibration_v1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_seed(family,mechanism,repeat,kind):
    payload="fixed-patient-dp-20260916-v1|"+"|".join(map(str,(family,mechanism,repeat,kind)))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15],16)


class Audit:
    def __init__(self):
        self.checks=0
        self.comparisons=[]

    def require(self,condition,label):
        self.checks+=1
        if not bool(condition):
            raise AssertionError(label)

    def close(self,actual,expected,label,atol=1e-11,rtol=1e-10):
        x,y=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
        self.require(x.shape==y.shape,label+": shape")
        self.require(bool(np.isfinite(x).all() and np.isfinite(y).all()),label+": finite")
        error=np.abs(x-y)
        self.comparisons.append({"label":label,"max_abs_error":float(error.max(initial=0)),"atol":atol,"rtol":rtol})
        self.require(bool(np.all(error<=atol+rtol*np.abs(y))),label+": numerical agreement")


def vector_norm(array):
    return math.sqrt(math.fsum(float(v)*float(v) for v in np.ravel(array)))


def pack_patients(a,b,a0,b0):
    """Manual row-major upper triangle, then row-major B; no producer helper."""
    n,d,o=b.shape
    positions=[(i,j) for i in range(d) for j in range(i,d)]
    rows=[]
    for patient in range(n):
        av=[float(a[patient,i,j])*(math.sqrt(2) if i!=j else 1)/a0 for i,j in positions]
        bv=[float(b[patient,i,j])/b0 for i in range(d) for j in range(o)]
        rows.append(av+bv)
    return np.array(rows,dtype=np.float64),positions


def replay_ssp(a,b,p,noise_seed):
    packed,positions=pack_patients(a,b,p["a0"],p["b0"])
    norms=np.array([vector_norm(row) for row in packed])
    factors=np.array([min(1,p["clip_norm"]/norm) if norm else 1 for norm in norms])
    clipped=packed*factors[:,None]
    total=np.array([math.fsum(clipped[:,j]) for j in range(clipped.shape[1])])
    noise=np.random.default_rng(noise_seed).standard_normal(len(total))
    released=(total+p["sigma"]*p["clip_norm"]*noise)/p["public_denominator"]
    _,d,o=b.shape
    aa=np.zeros((d,d))
    for index,(i,j) in enumerate(positions):
        value=released[index]*p["a0"]/(math.sqrt(2) if i!=j else 1)
        aa[i,j]=aa[j,i]=value
    bb=released[len(positions):].reshape(d,o)*p["b0"]
    eigenvalues,eigenvectors=np.linalg.eigh(aa)
    raised=np.maximum(eigenvalues,p["eigenvalue_floor"])
    projected=eigenvectors@np.diag(raised)@eigenvectors.T
    projected=(projected+projected.T)/2
    w=eigenvectors@((eigenvectors.T@bb)/(raised+p["ridge"])[:,None])
    return {"W":w,"released_vector":released,"A_released":aa,"A_projected":projected,"B_released":bb,
            "patient_norms":norms,"clipping_factors":factors,"clipped_sum":total,"standard_normal":noise,
            "noisy_gram_eigenvalues":eigenvalues,"condition":float((raised.max()+p["ridge"])/(raised.min()+p["ridge"]))}


def sgd_update(a,b,w,mask,noise,p,learning_rate):
    """Explicit per-selected-patient matmul/norm/clip/sum; ridge outside clip."""
    selected=np.flatnonzero(mask)
    gradients=[]
    norms=[]
    clipped_count=0
    for patient in selected:
        gradient=2*(a[patient]@w-b[patient])
        norm=vector_norm(gradient)
        factor=1 if norm<=p["clip_norm"] else p["clip_norm"]/norm
        gradients.append(gradient*factor)
        norms.append(norm)
        clipped_count+=int(factor<1)
    total=np.zeros_like(w)
    for gradient in gradients:
        total+=gradient
    noise_scale=0.0 if p["sigma"]==0 else p["sigma"]*p["clip_norm"]
    update=(total+noise_scale*noise)/(p["q"]*p["public_denominator"])+2*p["ridge"]*w
    return w-learning_rate*update,{
        "sampled_patients":len(selected),"clipped_patients":clipped_count,
        "maximum_sampled_gradient_norm":max(norms,default=0.0)}


def replay_sgd(a,b,p,rates,sampling_seed,noise_seed,*,saved=None,audit=None,label=""):
    tmax=p["steps"]
    sampling=np.random.default_rng(sampling_seed)
    gaussian=np.random.default_rng(noise_seed)
    w=np.zeros(b.shape[1:])
    counts,clips,maxima=[],[],[]
    if saved is not None:
        audit.require(saved["trace_weight_history"].shape==(tmax+1,*w.shape),label+": history dimensions")
        audit.require(np.array_equal(saved["trace_weight_history"][0],w),label+": zero initialization")
        audit.require(saved["trace_sampling_masks"].shape==(tmax,len(a)) and saved["trace_sampling_masks"].dtype==np.bool_,
                      label+": actual bool Poisson masks")
        audit.require(saved["trace_standard_normals"].shape==(tmax,*w.shape),label+": Gaussian trace dimensions")
    for step in range(tmax):
        mask=sampling.random(len(a))<p["q"]
        noise=gaussian.standard_normal(w.shape)
        if saved is not None:
            audit.require(np.array_equal(saved["trace_sampling_masks"][step],mask),label+": exact Poisson stream")
            audit.require(np.array_equal(saved["trace_standard_normals"][step],noise),label+": exact Gaussian stream")
            # Local checks start at the saved previous state to distinguish
            # step arithmetic from accumulated roundoff in a full replay.
            local,diagnostic=sgd_update(a,b,saved["trace_weight_history"][step],mask,noise,p,rates[step])
            audit.close(saved["trace_weight_history"][step+1],local,label+": step "+str(step),atol=1e-12,rtol=1e-11)
            for field in ("sampled_patients","clipped_patients"):
                audit.require(int(saved["trace_"+field][step])==diagnostic[field],label+": exact "+field)
            audit.close(saved["trace_maximum_sampled_gradient_norm"][step],diagnostic["maximum_sampled_gradient_norm"],
                        label+": maximum patient gradient norm",atol=1e-11,rtol=1e-10)
        w,diagnostic=sgd_update(a,b,w,mask,noise,p,rates[step])
        counts.append(diagnostic["sampled_patients"])
        clips.append(diagnostic["clipped_patients"])
        maxima.append(diagnostic["maximum_sampled_gradient_norm"])
    return w,{"empty_batches":sum(x==0 for x in counts),"patient_gradient_evaluations":sum(counts),
              "clipped_patient_steps":sum(clips)}


def patient_mse(a,b,q,w):
    result=[]
    for ai,bi,qi in zip(a,b,q):
        linear=math.fsum(float(x)*float(y) for x,y in zip(w.ravel(),bi.ravel()))
        quadratic=0.0
        for channel in range(w.shape[1]):
            quadratic+=float(w[:,channel]@ai@w[:,channel])
        result.append((float(qi)-2*linear+quadratic)/4)
    return np.array(result)


def fixed_rates(steps,initial_lr):
    """Independent scalar construction of the constant/linear schedule."""
    cut=int(math.floor(.8*steps))
    length=steps-cut
    return np.array([initial_lr if t<cut else initial_lr*(1-.99*(t-cut+1)/length)
                     for t in range(steps)],dtype=np.float64)


def check_ssp(audit,saved,expected,p,label,trace=False):
    for key in ("released_vector","A_released","A_projected","B_released"):
        audit.close(saved[key],expected[key],label+": "+key)
    dimension=len(expected["W"])
    scale=max(1.,float(np.linalg.norm(expected["W"])))
    w_bound=4096*EPS*dimension*max(1.,expected["condition"])*scale
    audit.close(saved["W"],expected["W"],label+": independent eigenbasis solution",atol=w_bound,rtol=1e-10)
    system=expected["A_projected"]+p["ridge"]*np.eye(dimension)
    residual=system@saved["W"]-expected["B_released"]
    denominator=np.linalg.norm(system,2)*np.linalg.norm(saved["W"])+np.linalg.norm(expected["B_released"])
    audit.require(float(np.linalg.norm(residual))/max(float(denominator),np.finfo(float).tiny)
                  <=4096*EPS*dimension,label+": solved projected quadratic stationarity")
    if trace:
        for field in ("patient_norms","clipping_factors","clipped_sum","standard_normal","noisy_gram_eigenvalues"):
            audit.close(saved["trace_"+field],expected[field],label+": trace "+field)
        audit.require(np.array_equal(saved["trace_standard_normal"],expected["standard_normal"]),
                      label+": exact Gaussian draw")


def verify_saved_models(out,audit,context):
    """Replay only after completed producer execution and public-selection gate."""
    c=context["contract"]
    params=c["public_parameters"]["selected"]
    bank=context["bank"]
    mask=bank["splits"]=="train"
    models=read(out/"models_manifest.json")
    audit.require(len(models)==64,"all four cells x16 predetermined repeats")
    expected_keys=[(f,m,r) for f in ("static","full") for m in ("SSP","SGD") for r in range(16)]
    audit.require([(r["family"],r["mechanism"],r["repeat"]) for r in models]==expected_keys,
                  "exact predetermined model order")
    collected={}
    max_replay_error=0.0
    started=time.perf_counter()
    for index,row in enumerate(models):
        family,mechanism,rep=row["family"],row["mechanism"],row["repeat"]
        p=params[family][mechanism]
        a,b=bank["A_"+family][mask],bank["B_"+family][mask]
        expected_path=f"models/{family}_{mechanism}_{rep:02d}.npz"
        audit.require(row["path"]==expected_path,"canonical model path")
        audit.require(digest(out/row["path"])==row["sha256"],"model output hash")
        for kind in ("sampling","noise"):
            audit.require(row[kind+"_seed"]==fixed_seed(family,mechanism,rep,kind),"predeclared "+kind+" seed")
        with np.load(out/row["path"],allow_pickle=False) as z:
            saved={k:z[k].copy() for k in z.files}
        audit.require(saved["W"].shape==b.shape[1:] and saved["W"].dtype==np.float64,"FP64 final head shape")
        audit.require(all(np.isfinite(value).all() for value in saved.values()),"finite saved model and traces")
        label=f"{family}/{mechanism}/{rep}"
        if mechanism=="SSP":
            replay=replay_ssp(a,b,p,row["noise_seed"])
            check_ssp(audit,saved,replay,p,label,trace=rep==0)
            audit.require(row["internal_diagnostics"]["clipped_patients"]==int(np.sum(replay["clipping_factors"]<1)),
                          label+": clipped patient count")
            audit.close(row["internal_diagnostics"]["noisy_gram_min_eigenvalue"],replay["noisy_gram_eigenvalues"].min(),
                        label+": noisy Gram minimum")
            wanted=replay["W"]
        else:
            rates=fixed_rates(p["steps"],p["initial_lr"])
            wanted,counts=replay_sgd(a,b,p,rates,row["sampling_seed"],row["noise_seed"],
                                    saved=saved if rep==0 else None,audit=audit,label=label)
            audit.close(saved["W"],wanted,label+": complete independent SGD replay",atol=1e-10,rtol=1e-9)
            for field,value in counts.items():
                audit.require(row["internal_diagnostics"][field]==value,label+": "+field)
        max_replay_error=max(max_replay_error,float(np.max(np.abs(saved["W"]-wanted))))
        collected[(family,mechanism,rep)]=saved["W"]
        if (index+1)%8==0:
            print(json.dumps({"verified_private_saved_models":index+1,"total":64,
                              "seconds":time.perf_counter()-started}),flush=True)
    with np.load(out/"controls.npz",allow_pickle=False) as z:
        controls={k:z[k].copy() for k in z.files}
    expected_control_keys=[]
    for family in ("static","full"):
        expected_control_keys.extend([family+"_public_only",family+"_nonprivate",family+"_SSP_clip_only",family+"_SSP_clip_and_floor"])
        expected_control_keys.extend([family+"_"+name+f"_{rep:02d}" for name in
                                      ("SGD_no_noise","SGD_no_noise_no_clip") for rep in range(3)])
    audit.require(set(controls)==set(expected_control_keys) and len(controls)==20,"all20 predetermined controls")
    with np.load(context["calibration_dir"]/"public_only_models.npz",allow_pickle=False) as z:
        public_weights={k:z[k].copy() for k in z.files}
    with np.load(context["capacity_dir"]/"models.npz",allow_pickle=False) as z:
        original_weights={k:z[k].copy() for k in z.files}
    for family in ("static","full"):
        a,b=bank["A_"+family][mask],bank["B_"+family][mask]
        audit.require(np.array_equal(controls[family+"_public_only"],public_weights["W_"+family]),"public-only baseline copied exactly")
        audit.require(np.array_equal(controls[family+"_nonprivate"],original_weights["W_"+family]),"existing nonprivate baseline copied exactly")
        for name,floor in (("SSP_clip_only",0.),("SSP_clip_and_floor",params[family]["SSP"]["eigenvalue_floor"])):
            p={**params[family]["SSP"],"sigma":0.,"eigenvalue_floor":floor}
            replay=replay_ssp(a,b,p,fixed_seed(family,name,0,"noise"))
            bound=4096*EPS*len(replay["W"])*max(1.,replay["condition"])*max(1.,float(np.linalg.norm(replay["W"])))
            audit.close(controls[family+"_"+name],replay["W"],family+" "+name+" independent control",atol=bound,rtol=1e-10)
        for name,clip in (("SGD_no_noise",params[family]["SGD"]["clip_norm"]),("SGD_no_noise_no_clip",float("inf"))):
            for rep in range(3):
                key=family+"_"+name+f"_{rep:02d}"
                p={**params[family]["SGD"],"sigma":0.,"clip_norm":clip}
                sampling_seed=fixed_seed(family,"SGD",rep,"sampling")
                noise_seed=fixed_seed(family,"SGD",rep,"noise")
                seed_manifest=context["execution"]["control_seed_manifest"][key]
                audit.require(seed_manifest=={"sampling_seed":sampling_seed,"noise_seed":noise_seed},"control same sampling/noise seed policy")
                expected,_=replay_sgd(a,b,p,fixed_rates(p["steps"],p["initial_lr"]),sampling_seed,noise_seed)
                audit.close(controls[key],expected,key+": independent no-noise control replay",atol=1e-10,rtol=1e-9)
    return {"models":collected,"controls":controls,"manifest":models,
            "diagnostics":{"all64_models_replayed":True,"all20_controls_checked":True,
                           "rep0_SGD_full_step_traces_checked":True,"maximum_final_weight_absolute_difference":max_replay_error,
                           "replay_seconds":time.perf_counter()-started}}


def verify_evaluation(out,audit,context,replayed):
    bank=context["bank"]
    mask=bank["splits"]=="eval"
    ids=bank["patient_ids"][mask]
    base=bank["Q_full"][mask]/4
    analysis=read(out/"analysis.json")
    arrays={"base_mse":base}
    for key,w in replayed["controls"].items():
        family=key.split("_")[0]
        arrays["control_"+key]=patient_mse(bank["A_"+family][mask],bank["B_"+family][mask],bank["Q_"+family][mask],w)
    for family in ("static","full"):
        for mechanism in ("SSP","SGD"):
            values=[patient_mse(bank["A_"+family][mask],bank["B_"+family][mask],bank["Q_"+family][mask],
                                replayed["models"][(family,mechanism,rep)]) for rep in range(16)]
            arrays["mse_"+family+"_"+mechanism]=np.stack(values)
    with np.load(out/"evaluation_mse.npz",allow_pickle=False) as saved:
        audit.require(np.array_equal(saved["patient_ids"],ids),"exact40 development patient order")
        audit.require(set(saved.files)==set(arrays)|{"patient_ids"},"all predeclared evaluation arrays")
        for key,expected in arrays.items():
            audit.close(saved[key],expected,"independent40 patient MSE "+key,atol=1e-10,rtol=1e-10)
    audit.close(analysis["mean_base_mse"],math.fsum(base)/40,"base patient mean")
    summaries={}
    for family in ("static","full"):
        for mechanism in ("SSP","SGD"):
            key=family+"_"+mechanism
            matrix=arrays["mse_"+key]
            means=np.array([math.fsum(row)/40 for row in matrix])
            mean=math.fsum(means)/16
            base_mean=math.fsum(base)/40
            public=arrays["control_"+family+"_public_only"]
            public_mean=math.fsum(public)/40
            expected={"mean_mse":mean,"sd_between_noise_seeds":math.sqrt(math.fsum((means-mean)**2)/15),
                "min_seed_mean":float(means.min()),"max_seed_mean":float(means.max()),
                "relative_reduction_from_base_percent":100*(base_mean-mean)/base_mean,
                "repeats_better_than_base":int(np.sum(means<base_mean)),
                "relative_reduction_from_public_only_percent":100*(public_mean-mean)/public_mean,
                "repeats_better_than_public_only":int(np.sum(means<public_mean)),
                "positive_patients_against_base_for_mean_model_loss":int(np.sum(matrix.mean(axis=0)<base)),
                "mean_fit_seconds":math.fsum(row["seconds"] for row in replayed["manifest"]
                    if row["family"]==family and row["mechanism"]==mechanism)/16}
            audit.require(set(analysis["summaries"][key])==set(expected),"complete16-repeat summary")
            for field,value in expected.items():
                if isinstance(value,int):
                    audit.require(analysis["summaries"][key][field]==value,"exact summary count "+key+" "+field)
                else:
                    audit.close(analysis["summaries"][key][field],value,"16-seed summary "+key+" "+field,atol=1e-9,rtol=1e-10)
            summaries[key]=expected
        audit.close(analysis["same_head_ssp_minus_sgd_mean_mse"][family],
                    summaries[family+"_SSP"]["mean_mse"]-summaries[family+"_SGD"]["mean_mse"],
                    "same-head SSP-SGD contrast "+family)
    expected_controls={key:math.fsum(value)/40 for key,value in arrays.items() if key.startswith("control_")}
    audit.require(set(analysis["controls_mean_mse"])==set(expected_controls),"complete control summary")
    for key,value in expected_controls.items():
        audit.close(analysis["controls_mean_mse"][key],value,"control patient mean "+key)
    return {"patient_count":40,"mechanism_models":64,"controls":20,"scalar_patient_losses_checked":3400,
            "summaries":summaries,"controls_mean_mse":expected_controls}


def verify_provenance(out,audit,calibration_verification_path):
    execution=read(out/"execution.json")
    audit.require(execution["completed"] is True,"completed producer execution required before any private replay")
    audit.require(execution["learned_dp_models"]==64 and execution["noise_repeats"]==16 and execution["control_weights"]==20,
                  "completed predetermined DP/control counts")
    audit.require(execution["backbone_forward"]==execution["backbone_backward"]==0,"zero new backbone work")
    c=read(out/"contract.json")
    audit.require(c["schema"]=="fixed-patient-dp-comparison/v1","private comparison contract schema")
    audit.require(c["adjacency"]=="add_remove_one_patient" and c["fixed_public_denominator"]==80,"patient adjacency/fixed denominator")
    audit.require(c["epsilon"]==8 and c["delta"]==1e-5,"fixed privacy accounting target")
    audit.require(c["families"]==["static","full"] and c["mechanisms"]==["SSP","SGD"],"fixed four cells")
    audit.require(c["noise_repeats"]==16 and c["no_noise_sgd_repeats"]==3,"fixed mechanism repeats")
    audit.require(c["patient_counts"]=={"train":80,"development":40,"public":32},"declared cohort sizes")
    audit.require(c["seed_salt"]=="fixed-patient-dp-20260916-v1","fixed independent stream salt")
    audit.require(c["initialization"]=="zero for all private SGD","fixed private initialization")
    audit.require(c["shared_feature_cache_for_both_methods"] is True,"same sufficient-statistics access")
    audit.require(c["privacy_tuning_on_fit80_or_eval40"] is False and
                  c["no_adaptive_private_seed_clip_lr_floor_selection"] is True,"no private hyperparameter tuning policy")
    audit.require(c["generative_quality_measured"] is False,"denoising-only evaluation")
    audit.require(c["release_scope"]=="internal public-data simulation; entire reproducibility bundle is NOT a DP release",
                  "internal reproducibility bundle is not a DP release")
    sources=c["source_sha256"]
    for filename,expected in sources.items():
        audit.require(digest(filename)==expected,"frozen source/input "+filename)
    def source(suffix):
        found=[Path(name) for name in sources if name.replace("\\","/").endswith(suffix)]
        audit.require(len(found)==1,"one bound "+suffix)
        return found[0]
    mechanism_source=source("frozen_residual_head/dp_mechanisms.py")
    audit.require(digest(mechanism_source)==MECHANISM_SHA,"reviewed numerical mechanism source")
    source("frozen_residual_head/run_patient_dp.py")
    source("frozen_residual_head/public_calibrate_dp.py")
    capacity_dir=source("frozen_residual_capacity_20260916_v1/contract.json").parent
    public_dir=source("frozen_residual_public32_20260916_v1/contract.json").parent
    calibration_dir=source("dp_calibration_v1/params.json").parent
    params=read(calibration_dir/"params.json")
    audit.require(params==c["public_parameters"],"exact public parameter object adopted before private run")
    audit.require(params["complete"] is True and params["private_data_read"] is False,"complete public-only calibration")
    audit.require(params["selection_contract_sha256"]==digest(calibration_dir/"selection_contract.json"),"selection policy binding")
    for filename,expected in params["input_sha256"].items():
        audit.require(digest(filename)==expected,"public calibration input "+filename)
    for filename,expected in params["code_sha256"].items():
        audit.require(digest(filename)==expected,"public calibration source "+filename)
    selection=read(calibration_verification_path)
    # Strict status and input binding are supplied by the separately frozen
    # independent public-selection verifier; no producer numerical function.
    audit.require(selection["status"]==CALIBRATION_PASS and selection["complete"] is True,
                  "independent public parameter-selection verification")
    audit.require(selection["code_sha256"]==CALIBRATION_VERIFIER_SHA,
                  "independent public-selection verifier source")
    audit.require(digest(Path(__file__).with_name("verify_public_calibration.py"))==CALIBRATION_VERIFIER_SHA,
                  "actual independent public-selection verifier remains frozen")
    audit.require(selection["protocol_sha256"]==digest(calibration_dir/"independent_selection_verification_protocol.json"),
                  "independent public-selection protocol binding")
    for key,name in (("params_sha256","params.json"),("accounting_sha256","accounting.json"),
                     ("execution_sha256","execution.json"),("selection_contract_sha256","selection_contract.json")):
        audit.require(selection[key]==digest(calibration_dir/name),"independent selection exact "+name)
    audit.require(source("dp_calibration_v1/independent_selection_verification.json").resolve()
                  ==Path(calibration_verification_path).resolve(),"private contract adopted verified public selection")
    for filename,expected in selection["input_sha256"].items():
        path=Path(filename)
        if not path.is_absolute():
            path=calibration_dir/path
        audit.require(digest(path)==expected,"independent public-selection input "+filename)
    for directory,expected_status in (
            (capacity_dir,"PASS_FROZEN_RESIDUAL_CAPACITY_SAVED_ARITHMETIC_AND_PROVENANCE"),
            (public_dir,"PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE")):
        earlier=read(directory/"independent_verification.json")
        audit.require(earlier["status"]==expected_status and earlier["complete"] is True,"prior raw-statistic verification")
        for filename,expected in earlier["input_sha256"].items():
            audit.require(digest(directory/filename)==expected,"prior verified input remains unchanged "+filename)
    accounts=read(calibration_dir/"accounting.json")
    for family in ("static","full"):
        for mechanism in ("SSP","SGD"):
            p=params["selected"][family][mechanism]
            audit.require(p["public_denominator"]==80 and p["ridge"]==.001,"same N0 and channel-sum ridge")
            audit.require(p["clip_norm"]>0 and math.isfinite(p["clip_norm"]),"finite calibrated patient clipping")
            q,steps=(1.,1) if mechanism=="SSP" else (p["q"],p["steps"])
            record=accounts[f"q{q}_T{steps}"]
            audit.require(record["target_epsilon"]==8 and record["target_delta"]==1e-5,"accountant budget")
            audit.require(record["poisson_sample_rate"]==q and record["steps"]==steps,"accountant actual q/T")
            audit.require(record["conservative_epsilon"]==max(record["opacus_epsilon"],record["google_dp_accounting_epsilon"])
                          and record["conservative_epsilon"]<=8+1e-10,"conservative dual-accountant bound")
            audit.close(p["sigma"],record["noise_multiplier"],"exact calibrated noise multiplier",atol=1e-14,rtol=0)
            if mechanism=="SGD":
                audit.require(q==.1 and steps in (500,2000,8000),"selected public SGD iteration option")
                audit.require(p["schedule"]=="constant80_linear20_end01","public fixed learning-rate schedule")
            else:
                audit.require(p["a0"]>0 and p["b0"]>0 and p["eigenvalue_floor"]>=0,"fixed SSP scaling and PSD rule")
    audit.require(execution["contract_sha256"]==digest(out/"contract.json"),"execution contract binding")
    audit.require(execution["manifest_sha256"]==digest(out/"models_manifest.json"),"execution model-manifest binding")
    audit.require(execution["controls_sha256"]==digest(out/"controls.npz"),"execution control-weight binding")
    analysis=read(out/"analysis.json")
    audit.require(analysis["schema"]=="fixed-patient-dp-comparison-analysis/v1","evaluation analysis schema")
    audit.require(analysis["public_data_only_baseline_included"] is True and analysis["generation_quality_measured"] is False,
                  "public-only baseline and metric scope")
    audit.require(analysis["scope"]=="development40; 16 predeclared mechanism random repeats conditional on fixed input bank, not16 datasets; descriptive mean/SD, no best seed",
                  "no population/seed selection overclaim")
    audit.require(set(analysis["source_sha256"])=={"contract.json","execution.json","models_manifest.json","controls.npz","evaluation_mse.npz"},
                  "complete evaluation input binding")
    for filename,expected in analysis["source_sha256"].items():
        audit.require(digest(out/filename)==expected,"analysis source "+filename)
    with np.load(capacity_dir/"patient_stats.npz",allow_pickle=False) as z:
        bank={k:z[k].copy() for k in z.files}
    audit.require(len(set(bank["patient_ids"].tolist()))==120 and np.sum(bank["splits"]=="train")==80 and
                  np.sum(bank["splits"]=="eval")==40,"disjoint fixed80/40 patients")
    with np.load(public_dir/"patient_stats.npz",allow_pickle=False) as z:
        audit.require(len(z["patient_ids"])==32 and not set(z["patient_ids"].tolist()).intersection(bank["patient_ids"].tolist()),
                      "public32 excluded from private and development patients")
    return {"contract":c,"execution":execution,"capacity_dir":capacity_dir,"calibration_dir":calibration_dir,
            "bank":bank,"public_selection_verification_sha256":digest(calibration_verification_path),
            "input_sha256":{filename:digest(out/filename) for filename in
                ("contract.json","execution.json","models_manifest.json","controls.npz","evaluation_mse.npz","analysis.json")}}


def self_test():
    """Synthetic cost/conformance only, with no study patient data or fitting."""
    rng=np.random.default_rng(30917)
    factors=rng.normal(size=(8,6,4))
    a=np.array([x.T@x/len(x) for x in factors])
    b=rng.normal(size=(8,4,2))*.1
    p={"a0":1.,"b0":1.,"clip_norm":2.,"sigma":.6,"public_denominator":8,"ridge":.01,"eigenvalue_floor":0.}
    result=replay_ssp(a,b,p,4)
    check=Audit()
    check.close((result["A_projected"]+.01*np.eye(4))@result["W"],result["B_released"],"SSP independent solve")
    sp={"steps":4,"q":.1,"public_denominator":8,"clip_norm":.8,"sigma":1.2,"ridge":.01}
    zero=np.zeros((4,2)); noise=np.ones_like(zero)
    got,info=sgd_update(a,b,zero,np.zeros(8,dtype=bool),noise,sp,.03)
    check.close(got,-.03*1.2*.8*noise/.8,"empty batch noise persists")
    check.require(info["sampled_patients"]==0,"empty batch count")
    w,_=replay_sgd(a,b,sp,np.full(4,.03),5,6)
    check.require(np.isfinite(w).all(),"small replay finite")
    # Benchmark uses synthetic d64 moments, never private study statistics.
    aa=np.broadcast_to(np.eye(64)[None,:,:],(80,64,64)).copy()
    bb=rng.normal(size=(80,64,4))*.01
    tick=time.perf_counter()
    replay_sgd(aa,bb,{**sp,"steps":500,"public_denominator":80},np.full(500,.03),7,8)
    return {"status":"PASS_SYNTHETIC_ALGEBRA_AND_COST","checks":check.checks,
            "synthetic_500_step_full64_replay_seconds":time.perf_counter()-tick}


def write_once(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def calibration_phase(directory):
    """Reuse an existing PASS; otherwise invoke only the independent verifier."""
    directory=Path(directory).resolve()
    filename=directory/"independent_selection_verification.json"
    source=Path(__file__).with_name("verify_public_calibration.py")
    if digest(source)!=CALIBRATION_VERIFIER_SHA:
        raise AssertionError("Independent public-selection verifier source changed")
    if filename.exists():
        result=read(filename)
        if result.get("status")!=CALIBRATION_PASS or result.get("complete") is not True:
            raise AssertionError("Existing public-selection verification did not pass; preserve it")
        if result["code_sha256"]!=CALIBRATION_VERIFIER_SHA:
            raise AssertionError("Existing public-selection verification source mismatch")
        for path,value in result["input_sha256"].items():
            if digest(path)!=value:
                raise AssertionError("Changed public-selection input: "+path)
        if digest(directory/"independent_selection_verification_protocol.json")!=result["protocol_sha256"]:
            raise AssertionError("Changed public-selection verification protocol")
        return result
    from .verify_public_calibration import verify_calibration
    return verify_calibration(directory)


def verify_private(directory,calibration_directory):
    """Completed saved experiment replay, never a new candidate-selection run."""
    directory=Path(directory).resolve()
    calibration_directory=Path(calibration_directory).resolve()
    result_path=directory/"independent_verification.json"
    protocol_path=directory/"independent_verification_protocol.json"
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve existing private verification attempt")
    # Refuse incomplete experiment packets before recording a verification attempt.
    execution=read(directory/"execution.json")
    if execution.get("completed") is not True or not (directory/"analysis.json").is_file():
        raise RuntimeError("Completed saved execution and analysis are required")
    selection_path=calibration_directory/"independent_selection_verification.json"
    source_hash=digest(__file__)
    frozen={str(directory/name):digest(directory/name) for name in
            ("contract.json","execution.json","models_manifest.json","controls.npz","analysis.json","evaluation_mse.npz")}
    frozen[str(selection_path)]=digest(selection_path)
    frozen[str(Path(__file__).resolve())]=source_hash
    protocol={
        "schema":"independent-saved-patient-dp-verification/v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "code_sha256":source_hash,
        "input_sha256":frozen,
        "calibration_verifier_sha256":CALIBRATION_VERIFIER_SHA,
        "independence":"No producer numerical function imported. Manual SSP packing/eigenbasis solve and explicit per-patient SGD replay; no experimental candidate fitting.",
        "numeric_policy":"Arithmetic and tolerances fixed in this verifier source before verification. Source finalized during or after producer execution; no pre-experiment tolerance-freeze claim.",
        "tolerances":{"ordinary":{"atol":1e-11,"rtol":1e-10},
            "SGD_step":{"atol":1e-12,"rtol":1e-11},
            "SGD_final":{"atol":1e-10,"rtol":1e-9},
            "SSP_weight_absolute":"4096*eps64*d*max(1,condition)*max(1,Frobenius_norm(W)); relative 1e-10",
            "SSP_normalized_stationarity":"4096*eps64*d",
            "patient_MSE":{"atol":1e-10,"rtol":1e-10},
            "descriptive_summary":{"atol":1e-9,"rtol":1e-10}},
        "scope":"All 64 saved DP model final weights, all 20 controls, full rep0 traces, all 3400 scalar patient losses; fixed public calibration provenance.",
        "new_backbone_forward":0,"new_backbone_backward":0,
        "privacy_scope":"Internal public-data simulation. Raw statistics, nonprivate controls and Gaussian RNG evidence are not a DP release; individual epsilon does not certify a joint 16-model release.",
    }
    write_once(protocol_path,protocol)
    started=time.perf_counter()
    audit=Audit()
    try:
        context=verify_provenance(directory,audit,selection_path)
        replayed=verify_saved_models(directory,audit,context)
        evaluation=verify_evaluation(directory,audit,context,replayed)
        for path,value in frozen.items():
            audit.require(digest(path)==value,"unchanged verification input "+path)
        for path,value in context["contract"]["source_sha256"].items():
            audit.require(digest(path)==value,"unchanged producer-bound input "+path)
        for row in replayed["manifest"]:
            audit.require(digest(directory/row["path"])==row["sha256"],"unchanged saved model")
        result={
            "status":STATUS,"complete":True,"checks":audit.checks,
            "seconds":time.perf_counter()-started,"code_sha256":source_hash,
            "protocol_sha256":digest(protocol_path),"input_sha256":context["input_sha256"],
            "public_selection_verification_sha256":context["public_selection_verification_sha256"],
            "calibration_verifier_sha256":CALIBRATION_VERIFIER_SHA,
            "replay":replayed["diagnostics"],"evaluation":evaluation,
            "comparisons":audit.comparisons,
            "limits":[
                "CPU verification of saved sufficient-statistics mechanisms, not an independent GPU feature extraction.",
                "Public selection audit rechecks saved public candidate weights/objectives/choice; it does not independently refit every public candidate.",
                "DP accountant provenance and q/T/sigma/epsilon/delta connections are checked; this verifier is not a new proof of the accountant.",
                "Fixed input-bank denoising loss, not generated-image quality or a population clinical conclusion.",
                "The reproducibility bundle and 16-model collection are not certified as one epsilon-8 external release.",
                "Numerical tolerances were finalized before this verification, not asserted to have been frozen before the experiment."
            ],
            "new_backbone_forward":0,"new_backbone_backward":0,
        }
        write_once(result_path,result)
        return result
    except Exception as exc:
        failure={"status":"FAILED_SAVED_PATIENT_DP_INDEPENDENT_VERIFICATION","complete":False,
                 "error":repr(exc),"traceback":traceback.format_exc(),"checks":audit.checks,
                 "comparisons":audit.comparisons,"seconds":time.perf_counter()-started,
                 "code_sha256":source_hash,"protocol_sha256":digest(protocol_path)}
        write_once(result_path,failure)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("calibration","private","self-test"),required=True)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT)
    parser.add_argument("--calibration-dir",type=Path,default=CALIBRATION_DEFAULT)
    parser.add_argument("--expected-code-sha256")
    args=parser.parse_args()
    if args.expected_code_sha256 and digest(__file__)!=args.expected_code_sha256:
        raise AssertionError("Verifier source hash mismatch")
    if args.phase=="self-test":
        result=self_test()
    elif args.phase=="calibration":
        result=calibration_phase(args.calibration_dir)
    else:
        result=verify_private(args.run_dir,args.calibration_dir)
    print(json.dumps({k:result[k] for k in
          ("status","complete","checks","seconds","code_sha256","replay","synthetic_500_step_full64_replay_seconds")
          if k in result},indent=2),flush=True)


if __name__=="__main__":
    main()
