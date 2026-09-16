"""Independent CPU verifier for the public-only medical LoRA pilot.

Saved state/provenance and sampling arithmetic, not full optimizer/UNet replay.
Independent CFG/DDIM/image helpers copied from the previously audited
verify_lora_positive_control_v2.py (cdfd8668...); no producer imports.
"""
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name,"1")
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"_reports/public_medical_backbone_20260916_v1"
SALT="public-medical-backbone-execution-20260916-v1"
PHASE_STATUS={
    "training":"PASS_PUBLIC_MEDICAL_TRAINING_SAVED_INPUTS_STATES_AND_EXPOSURES",
    "selection":"PASS_PUBLIC_MEDICAL_SELECTION_SAVED_SAMPLING_AND_PAIRED_INPUTS",
    "confirmation":"PASS_PUBLIC_MEDICAL_CONFIRMATION_SAVED_SAMPLING_AND_SELECTION_BINDING"}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def arrays(path):
    with np.load(path,allow_pickle=False) as packet:
        return {key:packet[key].copy() for key in packet.files}

def save(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")

class Check:
    def __init__(self):
        self.count=0
        self.errors={}

    def require(self,condition,label):
        self.count+=1
        if not bool(condition):
            raise AssertionError(label)

    def equal(self,actual,expected,label):
        actual,expected=np.asarray(actual),np.asarray(expected)
        self.require(actual.shape==expected.shape and actual.dtype==expected.dtype,
                     label+": shape/dtype")
        self.require(np.array_equal(actual,expected),label+": exact values")

    def close(self,actual,expected,label,atol,rtol):
        actual,expected=np.asarray(actual),np.asarray(expected)
        self.require(actual.shape==expected.shape,label+": shape")
        self.require(np.isfinite(actual).all() and np.isfinite(expected).all(),label+": finite")
        difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        self.errors[label]=float(np.max(difference,initial=0.))
        self.require(np.all(difference<=float(atol)+float(rtol)*np.abs(expected.astype(np.float64))),
                     label+": fixed tolerance")

    def bounded(self,actual,reference,bound,label):
        actual,reference,bound=(np.asarray(x,dtype=np.float64) for x in (actual,reference,bound))
        self.require(actual.shape==reference.shape==bound.shape,label+": shape")
        self.require(np.isfinite(actual).all() and np.isfinite(reference).all()
                     and np.isfinite(bound).all() and np.all(bound>=0),label+": finite envelope")
        difference=np.abs(actual-reference)
        self.errors[label] = {"maximum_absolute_error":float(np.max(difference,initial=0.)),
            "maximum_bound_fraction":float(np.max(difference/np.maximum(bound,np.finfo(float).tiny),initial=0.))}
        self.require(np.all(difference<=bound),label+": operation-rounding envelope")

def integer_equal(check,actual,expected,label):
    """Timesteps are integer values, independent of Windows stack storage width."""
    actual,expected=np.asarray(actual),np.asarray(expected)
    check.require(actual.shape==expected.shape,label+": shape")
    for name,value in (("actual",actual),("expected",expected)):
        check.require(value.dtype in (np.dtype(np.int32),np.dtype(np.int64)),label+": "+name+" signed integer dtype")
        check.require(np.all(value>=0) and np.all(value<1000),label+": "+name+" exact timestep range")
    check.require(np.array_equal(actual.astype(np.int64),expected.astype(np.int64)),label+": exact integer values")

def independent_alphas(config):
    count=int(config["num_train_timesteps"])
    if config.get("trained_betas") is not None:
        betas=torch.tensor(config["trained_betas"],dtype=torch.float32)
    elif config["beta_schedule"]=="scaled_linear":
        betas=torch.linspace(float(config["beta_start"])**.5,float(config["beta_end"])**.5,
                             count,dtype=torch.float32).square()
    elif config["beta_schedule"]=="linear":
        betas=torch.linspace(config["beta_start"],config["beta_end"],count,dtype=torch.float32)
    else:
        raise AssertionError("Unreviewed beta schedule")
    if config.get("rescale_betas_zero_snr",False):
        raise AssertionError("Zero-SNR rescaling is outside the fixed reviewed scheduler")
    return torch.cumprod(1.-betas,dim=0).numpy()

def schedule(config,steps=30):
    total=int(config["num_train_timesteps"])
    spacing=config.get("timestep_spacing","leading")
    if spacing=="leading":
        return (np.arange(steps)*(total//steps))[::-1].copy().astype(np.int64)+int(config.get("steps_offset",0))
    if spacing=="linspace":
        return np.linspace(0,total-1,steps).round()[::-1].copy().astype(np.int64)
    if spacing=="trailing":
        return (np.arange(total,0,-total/steps).round()-1).astype(np.int64)
    raise AssertionError("Unreviewed timestep spacing")

def cfg(raw,guidance):
    """Separate eager CPU operations preserve the actual FP16/FP32 dtype."""
    raw=np.asarray(raw)
    if raw.dtype not in (np.float16,np.float32):
        raise AssertionError("Expected raw epsilon FP16 or FP32")
    if float(guidance)<=1.:
        if raw.shape!=(1,4,32,32):
            raise AssertionError("No-CFG branch must have one conditional example")
        return raw.copy()
    if raw.shape!=(2,4,32,32):
        raise AssertionError("CFG raw order must be unconditional then conditional")
    raw_t=torch.from_numpy(raw.copy())
    uncond,cond=raw_t[0:1],raw_t[1:2]
    difference=cond-uncond
    scaled=float(guidance)*difference
    return (uncond+scaled).numpy()

def ddim(sample,epsilon,t,alphas,config,steps=30):
    """Saved epsilon/eta0 DDIM update; no scheduler implementation is imported."""
    if config["prediction_type"]!="epsilon" or config.get("thresholding",False) or config.get("clip_sample",False):
        raise AssertionError("Unreviewed DDIM prediction or clipping policy")
    sample,epsilon=np.asarray(sample),np.asarray(epsilon)
    if sample.dtype not in (np.float16,np.float32) or epsilon.dtype!=sample.dtype:
        raise AssertionError("Reviewed update expects a single actual arithmetic dtype")
    previous=int(t)-int(config["num_train_timesteps"])//steps
    a=np.float32(alphas[int(t)])
    ap=np.float32(alphas[previous] if previous>=0 else (1. if config.get("set_alpha_to_one",True) else alphas[0]))
    x,e=torch.from_numpy(sample.copy()),torch.from_numpy(epsilon.copy())
    at,apt=torch.tensor(a,dtype=torch.float32),torch.tensor(ap,dtype=torch.float32)
    beta=1.-at
    original=(x-beta.sqrt()*e)/at.sqrt()
    # eta=0 and use_clipped_model_output=False, as fixed by the outer protocol.
    previous_sample=apt.sqrt()*original+(1.-apt).sqrt()*e
    x64,e64=sample.astype(np.float64),epsilon.astype(np.float64)
    a64,ap64=float(a),float(ap)
    reference=np.sqrt(ap64)*(x64-np.sqrt(1.-a64)*e64)/np.sqrt(a64)+np.sqrt(1.-ap64)*e64
    unit=float(np.finfo(sample.dtype).eps)/2.
    gamma=32.*unit/(1.-32.*unit)
    scale=np.sqrt(ap64)*(np.abs(x64)+np.sqrt(1.-a64)*np.abs(e64))/np.sqrt(a64)+np.sqrt(1.-ap64)*np.abs(e64)
    bound=gamma*scale+32.*float(np.nextafter(sample.dtype.type(0),sample.dtype.type(1)))
    return previous_sample.numpy(),reference,bound

def check_packet(check,packet,stage,guidance,config):
    label=stage
    half=stage in ("replay_pipeline_fp16","manual_fp16")
    dtype=np.dtype(np.float16 if half else np.float32)
    batch=2 if float(guidance)>1. else 1
    shapes={"conditioning":(batch,77,1024),"timesteps":(30,),
        "latents":(31,1,4,32,32),"model_inputs":(30,batch,4,32,32),
        "raw_eps":(30,batch,4,32,32),"eps":(30,1,4,32,32),
        "decoded_raw":(1,3,256,256),"image_float":(256,256,3),"initial_from_prepare":(1,4,32,32)}
    check.require(set(packet)==set(shapes),label+": all stored tensor fields")
    for key,shape in shapes.items():
        check.require(packet[key].shape==shape,label+": shape "+key)
        if key=="timesteps":
            check.require(packet[key].dtype in (np.dtype(np.int32),np.dtype(np.int64)),label+": signed integer timestep dtype")
            check.require(np.all(packet[key]>=0) and np.all(packet[key]<int(config["num_train_timesteps"])),label+": timestep range")
        else:
            expected=np.dtype(np.float32) if key=="image_float" else dtype
            check.require(packet[key].dtype==expected,label+": actual dtype "+key)
        check.require(np.isfinite(packet[key]).all(),label+": finite "+key)
    times=schedule(config)
    alphas=independent_alphas(config)
    integer_equal(check,packet["timesteps"],times,label+": independently specified DDIM schedule")
    check.equal(packet["initial_from_prepare"],packet["latents"][0],label+": actual prepared initial latent")
    cpu_exact_steps=0
    for i,timestep in enumerate(times):
        current=packet["latents"][i]
        expected_input=np.concatenate([current,current],axis=0) if batch==2 else current
        check.equal(packet["model_inputs"][i],expected_input,label+": DDIM identity scale and input order "+str(i))
        combined=cfg(packet["raw_eps"][i],guidance)
        check.equal(packet["eps"][i],combined,label+": eager CFG arithmetic "+str(i))
        cpu,exact,bound=ddim(current,combined,int(timestep),alphas,config)
        check.bounded(packet["latents"][i+1],exact,bound,label+": DDIM rounding "+str(i))
        check.bounded(cpu,exact,bound,label+": independent CPU DDIM consistency "+str(i))
        cpu_exact_steps+=int(np.array_equal(cpu,packet["latents"][i+1]))
    decoded=torch.from_numpy(packet["decoded_raw"].copy())
    processed=(decoded/2.+.5).clamp(0.,1.).permute(0,2,3,1).float().numpy()[0]
    check.equal(packet["image_float"],processed,label+": independent decoded-to-image conversion")
    return {"steps":30,"cpu_ddim_exact_steps":cpu_exact_steps,"precision":str(dtype),"timestep_dtype":str(packet["timesteps"].dtype),
            "guidance":float(guidance),"decoded_unclamped_minimum":float(packet["decoded_raw"].min()),
            "decoded_unclamped_maximum":float(packet["decoded_raw"].max())}

def png_pixels(path):
    with Image.open(path) as picture:
        return np.asarray(picture.convert("RGB")).copy()

def image_from_packet(packet):
    return np.clip(packet["image_float"]*np.float32(255.),0.,255.).astype(np.uint8)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def file_with_suffix(paths,suffix):
    matches=[Path(path) for path in paths if str(path).replace("\\","/").endswith(suffix)]
    if len(matches)!=1:
        raise AssertionError("Expected one bound file: "+suffix)
    return matches[0]


def planned_hash(*parts):
    return hashlib.sha256("|".join(map(str,(SALT,)+parts)).encode()).hexdigest()


def planned_seed(*parts):
    return int(planned_hash(*parts)[:15],16)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_plan(check,out,contract):
    check.require(contract["schema"]=="public-medical-backbone-runtime/v1"
                  and contract["non_DP"] is True and contract["protected_role_training"] is False
                  and contract["private_checkpoints_loaded"] is False and contract["plan_changes"] is False,
                  "fixed fresh public-only execution scope")
    sources=contract["source_sha256"]
    check.require(sources[str(Path(__file__).resolve())]==sha(__file__),"verifier source fixed before experiment")
    for path,digest in sources.items():
        check.require(sha(path)==digest,"bound source/input "+path)
    plan=Path(contract["plan_directory"]).resolve()
    spec=read(plan/"execution_spec.json")
    check.require(sha(plan/"execution_spec.json")==contract["execution_spec_sha256"],"frozen execution spec")
    check.require(sha(plan/"plan_lock.json")==contract["plan_lock_sha256"],"frozen plan lock")
    research=plan.parent.parent
    for relative,digest in read(plan/"plan_lock.json")["files"].items():
        check.require(sha(research/relative)==digest,"preserved plan file "+relative)
    for section in ("input_sha256","output_sha256"):
        for name,digest in spec[section].items():
            check.require(sha(plan/name)==digest,"original execution plan binding "+name)
    check.require(spec["schema"]=="public-medical-backbone-execution-plan/v1"
                  and spec["one_new_training_run"] is True,"single approved trajectory")
    people=csv_rows(plan/"patients.csv");images=csv_rows(plan/"images.csv")
    persons={row["patient_id"]:row for row in people}
    check.require(len(persons)==len(people)==905,"905 distinct public-role patients")
    check.require(Counter(row["backbone_role"] for row in people)==dict(train=640,selection=128,confirmation=128,reserve=9),
                  "fixed patient roles")
    check.require(Counter(row["backbone_role"] for row in images)==dict(train=749,selection=128,confirmation=128,reserve=11),
                  "fixed image roles")
    check.require(len({row["image_id"] for row in images})==1016,"unique planned images")
    for row in images:
        check.require(row["partition"]=="public_development" and row["view"]=="PA"
                      and row["backbone_role"]==persons[row["patient_id"]]["backbone_role"],"public source/role consistency")
    exclusion=read(plan/"exclusion_calculation.json")
    inputs=exclusion["input_sha256"]
    excluded=set()
    for suffix in ("cohort/evaluation_images.csv","cohort/auxiliary_images.csv","real_evaluator_tasks_local.csv"):
        relative=file_with_suffix(inputs,suffix)
        path=ROOT.parent/relative
        check.require(sha(path)==inputs[str(relative).replace("\\","/")],"bound exclusion metadata "+suffix)
        excluded.update(row["patient_id"] for row in csv_rows(path))
    check.require(not set(persons).intersection(excluded),"all 905 disjoint from protected/evaluation/old auxiliary patients")
    train=sorted((row for row in images if row["backbone_role"]=="train"),key=lambda row:row["image_id"])
    by_image={row["image_id"]:row for row in train}
    check.require(len(by_image)==749 and len({row["patient_id"] for row in train})==640,"actual training whitelist")
    schedule_rows=csv_rows(plan/"training_schedule.csv")
    check.require(len(schedule_rows)==5992,"fixed 5992 committed image presentations")
    sequence=[]
    for epoch in range(8):
        sequence.extend((epoch,image_id) for image_id in sorted(by_image,key=lambda image_id:planned_hash("epoch",epoch,image_id)))
    for i,((epoch,image_id),row) in enumerate(zip(sequence,schedule_rows)):
        step,position=divmod(i,4)
        expected=dict(step=str(step+1),batch_position=str(position),epoch=str(epoch+1),image_id=image_id,
            patient_id=by_image[image_id]["patient_id"],noise_seed=str(planned_seed("noise",step)),
            timestep=str(planned_seed("timestep",step,position)%1000))
        check.require(row==expected,"independent epoch/batch/t/noise schedule "+str(i))
    for exposure,step in ((4,749),(8,1498)):
        seen=Counter(row["image_id"] for row in schedule_rows[:step*4])
        check.require(seen=={iid:exposure for iid in by_image},"exact E"+str(exposure)+" committed exposure")
    tasks=csv_rows(plan/"generation_tasks.csv")
    prompts={"generic":"a frontal chest radiograph",
        "normal":"a frontal posteroanterior chest radiograph with no labeled finding",
        "effusion":"a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion",
        "cardiomegaly":"a frontal posteroanterior chest radiograph with radiographic findings of cardiomegaly"}
    check.require(len(tasks)==32 and len({row["task_id"] for row in tasks})==32,"all fixed generation task identities")
    expected_tasks=[]
    for phase in ("selection","confirmation"):
        for prompt_id,prompt in prompts.items():
            for index in range(4):
                expected_tasks.append(dict(task_id=f"{phase}_{prompt_id}_{index}",phase=phase,prompt_id=prompt_id,prompt=prompt,
                    seed=str(planned_seed("generation",phase,prompt_id,index)),generator_device="cpu",initial_dtype="float32",
                    guidance="1.0",steps="30",width="256",height="256"))
    check.require(tasks==expected_tasks,"independent generation prompt/seed schedule")
    check.require(len({row["seed"] for row in tasks})==32 and not {row["seed"] for row in tasks}&{"26091631","26091632"},
                  "selection/confirmation/old diagnostic seeds disjoint")
    return spec,train,schedule_rows,tasks


def verify_cache(check,out,train):
    report=read(out/"cache_report.json")
    check.require(report["cache_sha256"]==sha(out/"cache.pt"),"cache report identity")
    expected={row["image_id"]:row for row in train}
    ids=sorted(expected)
    check.require(report["train_ids"]==ids and report["train_patients"]==sorted({row["patient_id"] for row in train})
                  and report["vae_records"]==749,
                  "reported train-only encoding counts")
    check.require(report["complete"] is True and report["protected_data_cache_loaded"] is False
                  and report["contract_sha256"]==sha(out/"contract.json") and report["vae_batch_calls"]==188,
                  "actual public-only cache scope")
    check.require(report["source_images_sha256"]=={iid:expected[iid]["sha256"] for iid in ids},"source image hash whitelist")
    for iid,row in expected.items():
        check.require(sha(ROOT.parent/row["image_path"])==row["sha256"],"public training source bytes "+iid)
    cache=torch.load(out/"cache.pt",map_location="cpu",weights_only=True)
    check.require(set(cache)=={"latents","hidden","train_prompts"},"public-only cache schema")
    check.require(set(cache["latents"])==set(cache["train_prompts"])==set(ids),"exact training image ID whitelist")
    for iid,latent in cache["latents"].items():
        check.require(latent.shape==(1,4,32,32) and latent.dtype==torch.float16 and bool(torch.isfinite(latent).all()),
                      "finite public FP16 latent "+iid)
        check.require(cache["train_prompts"][iid] in cache["hidden"],"training conditioning present "+iid)
        check.require(cache["train_prompts"][iid]==weak_prompt(expected[iid]["finding_labels"]),"fixed weak-label prompt "+iid)
    for prompt,hidden in cache["hidden"].items():
        check.require(isinstance(prompt,str) and hidden.shape==(1,77,1024) and hidden.dtype==torch.float16
                      and bool(torch.isfinite(hidden).all()),"finite cached frozen embedding")
    check.require(report["embeddings_count"]==len(cache["hidden"]),"reported encoding count")
    return cache,report


def weak_prompt(raw_labels):
    labels=set(raw_labels.split("|"))
    base="a frontal posteroanterior chest radiograph"
    if labels=={"No Finding"}:
        return base+" with no labeled finding"
    names=("Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax",
           "Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia")
    descriptions=("atelectasis","cardiomegaly","pleural effusion","infiltration","mass opacity","nodule opacity",
                  "pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thickening","hernia")
    selected=[text for name,text in zip(names,descriptions) if name in labels]
    if len(selected)!=len(labels) or not selected:
        raise AssertionError("Unknown weak labels")
    joined=selected[0] if len(selected)==1 else " and ".join(selected) if len(selected)==2 else ", ".join(selected[:-1])+", and "+selected[-1]
    return base+" with radiographic findings of "+joined


def verify_adapter(check,adapter,initial=None,zero_b=False):
    check.require(isinstance(adapter,dict) and len(adapter)==256,"256 named LoRA tensors")
    check.require(sum(value.numel() for value in adapter.values())==1659904,"rank8 LoRA parameter count")
    keys=set(adapter)
    check.require(sum(".lora_A." in name for name in keys)==128 and sum(".lora_B." in name for name in keys)==128,
                  "128 A/B factor pairs")
    if initial is not None:
        check.require(keys==set(initial),"checkpoint exact initial adapter keys")
    for name,value in adapter.items():
        check.require(value.dtype==torch.float32 and value.ndim==2 and bool(torch.isfinite(value).all()),"finite FP32 LoRA tensor "+name)
        if initial is not None:
            check.require(value.shape==initial[name].shape,"unchanged adapter tensor shape "+name)
        if ".lora_A." in name:
            partner=name.replace(".lora_A.",".lora_B.")
            check.require(value.shape[0]==8 and partner in adapter and adapter[partner].shape[1]==8,"rank8 paired factor "+name)
            if zero_b:
                check.require(bool((value!=0).any()),"fresh random A nonzero "+name)
        elif zero_b:
            check.require(bool((value==0).all()),"fresh B exactly zero "+name)


def verify_preflight(check,out,cache,contract):
    meta=read(out/"preflight.json")
    initial=torch.load(out/"initial_adapter.pt",map_location="cpu",weights_only=True)
    verify_adapter(check,initial,zero_b=True)
    check.require(meta["initial_adapter_sha256"]==sha(out/"initial_adapter.pt")
                  and meta["zero_B"] is True and meta["enabled_disabled_exact"] is True and meta["forward_calls"]==2,
                  "fresh initialization preflight claims")
    packet=arrays(out/"preflight.npz")
    check.require(set(packet)=={"latents","hidden","noise","timesteps","noisy","pred_enabled","pred_disabled"},
                  "preflight tensor schema")
    ids=meta["image_ids"]
    check.require(len(ids)==4 and all(iid in cache["latents"] for iid in ids),"preflight uses train-only inputs")
    check.equal(packet["latents"],torch.cat([cache["latents"][iid] for iid in ids]).numpy(),"preflight source latents")
    check.equal(packet["hidden"],torch.cat([cache["hidden"][cache["train_prompts"][iid]] for iid in ids]).numpy(),"preflight source conditioning")
    check.require(packet["timesteps"].shape==(4,) and packet["timesteps"].dtype in (np.int32,np.int64)
                  and np.all(packet["timesteps"]>=0) and np.all(packet["timesteps"]<1000),"preflight integer t range")
    for key in ("latents","noise","noisy","pred_enabled","pred_disabled"):
        check.require(packet[key].shape==(4,4,32,32) and packet[key].dtype==np.float16 and np.isfinite(packet[key]).all(),
                      "preflight actual FP16 shape "+key)
    config=read(Path(contract["snapshot"])/"scheduler/scheduler_config.json")
    alphas=independent_alphas(config)
    alpha=torch.from_numpy(alphas).half()[torch.from_numpy(packet["timesteps"].astype(np.int64))].reshape(4,1,1,1)
    clean=torch.from_numpy(packet["latents"].copy());noise=torch.from_numpy(packet["noise"].copy())
    cpu=alpha.sqrt()*clean+(1.-alpha).sqrt()*noise
    a=alpha.double().numpy();x=packet["latents"].astype(float);e=packet["noise"].astype(float)
    reference=np.sqrt(a)*x+np.sqrt(1.-a)*e
    unit=np.finfo(np.float16).eps/2;gamma=16*unit/(1-16*unit)
    bound=gamma*(np.sqrt(a)*np.abs(x)+np.sqrt(1.-a)*np.abs(e))+16*float(np.nextafter(np.float16(0),np.float16(1)))
    check.bounded(packet["noisy"],reference,bound,"independent FP16 DDPM noising")
    check.bounded(cpu.numpy(),reference,bound,"CPU noising consistency")
    check.equal(packet["pred_enabled"],packet["pred_disabled"],"actual zero-adapter UNet prediction equality")
    return initial,meta


def verify_training(check,out,cache,contract,schedule_rows,initial,preflight):
    report=read(out/"training_report.json")
    check.require(report["complete"] is True and report["steps"]==1498 and report["non_DP_public_only"] is True
                  and report["new_DP_claim"] is False,"complete planned public non-DP training")
    check.require(report["trace_sha256"]==sha(out/"trace.jsonl") and report["cache_sha256"]==sha(out/"cache.pt")
                  and report["contract_sha256"]==sha(out/"contract.json")
                  and report["initial_adapter_sha256"]==sha(out/"initial_adapter.pt"),"training source/output hash links")
    check.require(report["base_sha256_before"]==report["base_sha256_after"]==preflight["base_sha256"]
                  and len(report["base_sha256_before"])==64,"reported frozen base before/after")
    with (out/"trace.jsonl").open(encoding="utf-8") as stream:
        trace=[json.loads(line) for line in stream if line.strip()]
    check.require(len(trace)==report["attempts"] and 1498<=len(trace)<=1598,"all attempts including bounded retries")
    check.require(report["unet_forward_calls"]==report["backward_calls"]==len(trace)
                  and report["preflight_forward_calls"]==2
                  and report["committed_image_presentations"]==5992
                  and report["attempted_image_presentations"]==4*len(trace),"actual logical forward/backward/presentation counts")
    successful=[];committed=0;scale=1024.;seconds=-1.;overflow=0
    actual_counts=Counter()
    for attempt,row in enumerate(trace,1):
        check.require(committed<1498 and row["step"]==committed+1 and row["attempt"]==attempt,
                      "trace attempted/committed step order")
        batch=schedule_rows[committed*4:(committed+1)*4]
        check.require(row["image_ids"]==[r["image_id"] for r in batch]
                      and row["noise_seed"]==int(batch[0]["noise_seed"])
                      and row["timesteps"]==[int(r["timestep"]) for r in batch],"actual attempt uses exact planned batch/time/noise")
        actual_counts.update(row["image_ids"])
        check.require(isinstance(row["committed"],bool) and np.isfinite(row["loss"]) and row["loss"]>=0,
                      "finite attempted loss and commit indicator")
        check.require(np.isfinite(row["seconds"]) and row["seconds"]>=seconds,"monotone attempted elapsed time")
        seconds=row["seconds"]
        check.require(row["scale_before"]==scale and np.isfinite(scale) and scale>0,"continuous positive scaler history")
        if row["committed"]:
            check.require(row["scale_after"]==scale,"fewer than2000 successes: no scheduled scaler growth")
            check.require(row["gradient_norm"] is not None and np.isfinite(row["gradient_norm"])
                          and row["gradient_norm"]>=0,"finite committed preclip gradient norm")
            committed+=1;successful.append(row)
        else:
            overflow+=1
            check.require(row["scale_after"]==scale*.5 and row["gradient_norm"] is None,"overflow backoff and nonfinite gradient log")
        scale=row["scale_after"]
    check.require(committed==1498 and overflow==len(trace)-1498 and overflow<=100,"complete successes and failures counted separately")
    check.require(set(report["checkpoints"])=={"E4","E8"},"exactly two fixed checkpoints")
    checkpoints={}
    for name,step,exposure in (("E4",749,4),("E8",1498,8)):
        meta=report["checkpoints"][name]
        check.require(meta["path"]=="checkpoints/"+name+".pt" and meta["sha256"]==sha(out/meta["path"]),"checkpoint path/hash "+name)
        saved=torch.load(out/meta["path"],map_location="cpu",weights_only=True)
        checkpoints[name]=saved
        check.require(saved["step"]==step and saved["attempt"]==successful[step-1]["attempt"],"checkpoint actual update/attempt "+name)
        check.require(saved["contract_sha256"]==sha(out/"contract.json") and saved["cache_sha256"]==sha(out/"cache.pt"),
                      "checkpoint immutable training inputs "+name)
        check.require(saved["exposures"]=={iid:exposure for iid in cache["latents"]},"exact committed image exposure "+name)
        check.require(saved["losses"]==[row["loss"] for row in successful[:step]],"checkpoint successful loss history "+name)
        verify_adapter(check,saved["adapter"],initial=initial)
        opt=saved["optimizer"]
        check.require(len(opt["param_groups"])==1 and len(opt["state"])==256,"single Adam group/all parameter states "+name)
        group=opt["param_groups"][0]
        for key,value in dict(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False).items():
            actual=tuple(group[key]) if key=="betas" else group[key]
            check.require(actual==value,"fixed optimizer "+key+" "+name)
        check.require(len(group["params"])==256 and set(group["params"])==set(opt["state"]),"optimizer state/parameter correspondence")
        for parameter_id,adapter_key in zip(group["params"],initial):
            state=opt["state"][parameter_id]
            check.require(float(state["step"])==step,"successful Adam step count")
            for key in ("exp_avg","exp_avg_sq"):
                value=state[key]
                check.require(value.shape==initial[adapter_key].shape and value.dtype==torch.float32
                              and bool(torch.isfinite(value).all()),"finite Adam moment and adapter shape")
            check.require(bool((state["exp_avg_sq"]>=0).all()),"nonnegative second Adam moment")
        scaler=saved["scaler"]
        check.require(scaler["scale"]==successful[step-1]["scale_after"] and scaler["growth_factor"]==2.
                      and scaler["backoff_factor"]==.5 and scaler["growth_interval"]==2000,"checkpoint scaler state")
        prior_trace=trace[:saved["attempt"]]
        last_overflow=max((i for i,row in enumerate(prior_trace) if not row["committed"]),default=-1)
        check.require(scaler["_growth_tracker"]==sum(row["committed"] for row in prior_trace[last_overflow+1:]),
                      "scaler consecutive-success growth tracker")
    return report,checkpoints,{"committed_updates":1498,"attempts":len(trace),"overflow_attempts":overflow,
        "committed_presentations":5992,"attempted_presentations":sum(actual_counts.values()),
        "attempted_exposure_min":min(actual_counts.values()),"attempted_exposure_max":max(actual_counts.values()),
        "optimizer_gradient_replay_performed":False}


def prior_verification(check,out,phase):
    path=out/(phase+"_verification.json")
    saved=read(path)
    check.require(saved["complete"] is True and saved["status"]==PHASE_STATUS[phase],"prior independent phase PASS "+phase)
    for name,digest in saved["input_sha256"].items():
        check.require(sha(out/name)==digest,"unchanged independently checked phase input "+name)
    return saved


def verify_decision(check,out):
    prior_verification(check,out,"selection")
    decision=read(out/"selection_decision.json")
    check.require(decision["complete"] is True and decision["phase"]=="selection"
                  and decision["nonclinical_heuristic_only"] is True and decision["private_utility_or_DP_claim"] is False,
                  "bounded nonclinical reviewer decision")
    check.require(decision["selection_verification_sha256"]==sha(out/"selection_verification.json"),"selection decision bound to checked tensors")
    check.require(decision["manifest_sha256"]==sha(out/"manifest_selection.json"),"decision refers to complete selection manifest")
    records=[row for row in read(out/"manifest_selection.json") if row["phase"]=="selection"]
    expected_ids={row["image_id"] for row in records}
    ballots=[]
    for reviewer in ("root","independent"):
        name="review_selection_"+reviewer+".json"
        check.require(sha(out/name)==decision["review_sha256"][name],"review ballot hash "+reviewer)
        review=read(out/name)
        check.require(review["reviewer"]==reviewer and review["phase"]=="selection"
                      and review["checkpoint_blinded"] is True,"declared review identity/blinding")
        ballot={item["image_id"]:item for item in review["images"]}
        check.require(set(ballot)==expected_ids and len(ballot)==len(review["images"]),"complete unique selection votes")
        check.require(all(type(item["pass"]) is bool and isinstance(item["note"],str) and bool(item["note"].strip())
                          for item in ballot.values()),"explicit binary decisions with reasons")
        ballots.append(ballot)
    joint={iid:ballots[0][iid]["pass"] and ballots[1][iid]["pass"] for iid in expected_ids}
    check.require(decision["joint_pass_by_image"]==joint and decision["reviewer_disagreements"]==
                  sum(ballots[0][iid]["pass"]!=ballots[1][iid]["pass"] for iid in expected_ids),
                  "disagreement-fail aggregation")
    outcomes=decision["checkpoint_results"]
    check.require(set(outcomes)=={"E4","E8"},"both checkpoints reported before selection")
    passing={}
    for name in ("E4","E8"):
        item=outcomes[name];counts=item["per_prompt_pass"]
        check.require(set(counts)=={"generic","normal","effusion","cardiomegaly"},"four selection prompt groups")
        check.require(all(isinstance(value,int) and not isinstance(value,bool) and 0<=value<=4 for value in counts.values()),
                      "per-prompt reviewer count range")
        check.require(item["total_pass"]==sum(counts.values()),"reviewer aggregation arithmetic")
        expected_counts={prompt:sum(joint[row["image_id"]] for row in records
            if row["checkpoint"]==name and row["task_id"].split("_")[1]==prompt) for prompt in counts}
        check.require(counts==expected_counts,"per-prompt counts independently reconstructed from ballots")
        passing[name]=item["total_pass"]>=12 and min(counts.values())>=2
        check.require(item["passes_gate"]==passing[name],"fixed threshold boolean")
    expected="E4" if passing["E4"] else "E8" if passing["E8"] else None
    check.require(decision["selected_checkpoint"]==expected,"earliest qualifying checkpoint; no maximum-score choice")
    check.require(expected in ("E4","E8"),"confirmation requires a selection pass")
    return decision,expected


def verify_generation(check,out,phase,contract,cache,tasks,training_report,bound):
    prior_verification(check,out,"training")
    choices=["E4","E8"]
    if phase=="confirmation":
        decision,chosen=verify_decision(check,out)
        choices=[chosen]
        bound["selection_decision.json"]=sha(out/"selection_decision.json")
        bound["selection_verification.json"]=sha(out/"selection_verification.json")
    report=read(out/f"generation_{phase}.json")
    manifest=read(out/f"manifest_{phase}.json")
    check.require(report["complete"] is True and report["checkpoint_choices"]==choices,"fixed generation choices")
    check.require(report["selection_decision_sha256"]==(sha(out/"selection_decision.json") if phase=="confirmation" else None),
                  "confirmation decision bound before generation")
    for key,name in (("manifest_sha256",f"manifest_{phase}.json"),("contract_sha256","contract.json"),
                     ("training_report_sha256","training_report.json"),("generation_inputs_sha256",f"generation_inputs_{phase}.pt")):
        check.require(report[key]==sha(out/name),"generation report hash binding "+name)
    inputs=torch.load(out/f"generation_inputs_{phase}.pt",map_location="cpu",weights_only=True)
    target_tasks=[dict(row) for row in tasks if row["phase"]==phase]
    if phase=="selection":
        old_contract_path=file_with_suffix(contract["source_sha256"],"frozen_residual_sampling_integration_20260916_v1/contract.json")
        old_contract=read(old_contract_path)
        for case in old_contract["cases"]:
            target_tasks.append(dict(task_id="diagnostic_"+case["case_id"],phase="diagnostic",prompt=case["prompt"],
                old_path=old_contract_path.parent/"trajectories"/(case["case_id"]+"_base.npz")))
    task_map={task["task_id"]:task for task in target_tasks}
    check.require(set(inputs)==set(task_map),"all prebound actual generation inputs")
    for task_id,task in task_map.items():
        item=inputs[task_id]
        check.require(set(item)=={"initial","conditioning"},"self-contained generation input schema")
        check.require(item["initial"].dtype==torch.float32 and tuple(item["initial"].shape)==(1,4,32,32)
                      and bool(torch.isfinite(item["initial"]).all()),"actual FP32 initial noise")
        check.equal(item["conditioning"].numpy(),cache["hidden"][task["prompt"]].float().numpy(),"actual current cache conditioning")
        if task["phase"]=="diagnostic":
            path=task["old_path"]
            check.require(sha(path)==contract["source_sha256"][str(path)],"source diagnostic trace binding")
            expected=arrays(path)["latents"][0]
        else:
            expected=torch.randn((1,4,32,32),generator=torch.Generator(device="cpu").manual_seed(int(task["seed"])),
                                  dtype=torch.float32).numpy()
        check.equal(item["initial"].numpy(),expected,"independent fixed initial noise "+task_id)
    expected_pairs=[(choice,task["task_id"]) for choice in choices for task in target_tasks]
    check.require([(row["checkpoint"],row["task_id"]) for row in manifest]==expected_pairs,"all fixed generated cases/order")
    expected_count=36 if phase=="selection" else 16
    check.require(len(manifest)==report["records"]==expected_count and report["unet_forward_calls"]==30*expected_count
                  and report["vae_decodes"]==expected_count,"actual generation count")
    config=report["scheduler_config"]
    original_config=read(Path(contract["snapshot"])/"scheduler/scheduler_config.json")
    for key in ("beta_start","beta_end","beta_schedule","num_train_timesteps","prediction_type","clip_sample","set_alpha_to_one","steps_offset"):
        check.require(config[key]==original_config[key],"actual fixed generation scheduler "+key)
    check.require(config["prediction_type"]=="epsilon" and not config.get("thresholding",False)
                  and not config.get("clip_sample",False),"same reviewed DDIM numerical policy")
    guards=report["model_guards"]
    check.require(set(guards)==set(choices),"actual checkpoint model guards")
    base_hashes=set()
    for choice in choices:
        guard=guards[choice]
        check.require(guard["adapter_loaded_exact"] is True and guard["adapter_after_exact"] is True
                      and guard["adapter_dtype"]=="float32","reported loaded/unchanged exact checkpoint adapter "+choice)
        check.require(guard["base_sha256_before"]==guard["base_sha256_after"] and len(guard["base_sha256_before"])==64,
                      "generation base weights unchanged "+choice)
        base_hashes.add(guard["base_sha256_before"])
    check.require(len(base_hashes)==1,"same frozen FP32 base across generated checkpoints")
    seen=set();details={};paired={}
    for row in manifest:
        task=task_map[row["task_id"]];choice=row["checkpoint"]
        blind=hashlib.sha256(("public-medical-blind-v1|"+choice+"|"+row["task_id"]).encode()).hexdigest()[:12]
        check.require(row["image_id"]==blind and blind not in seen,"fixed blind identity without collisions")
        seen.add(blind)
        check.require(row["phase"]==task["phase"] and row["path"]==f"traces/{blind}.npz"
                      and row["image_path"]==f"images/{blind}.png","phase and canonical output paths")
        check.require(row["guidance"]==1. and row["precision"]=="fp32_no_autocast"
                      and row["unet_calls"]==row["batch_examples"]==30,"actual CFG1 FP32 sampling scope")
        check.require(row["checkpoint_sha256"]==training_report["checkpoints"][choice]["sha256"],"exact training checkpoint linked")
        for path_key,hash_key in (("path","sha256"),("image_path","image_sha256")):
            check.require(sha(out/row[path_key])==row[hash_key],"actual generated packet hash")
            bound[row[path_key]]=row[hash_key]
        packet=arrays(out/row["path"])
        details[blind]=check_packet(check,packet,"public_generation_fp32",1.,config)
        check.equal(packet["initial_from_prepare"],inputs[row["task_id"]]["initial"].numpy(),"saved actual initial latent")
        check.equal(packet["conditioning"],inputs[row["task_id"]]["conditioning"].numpy(),"saved actual fixed conditioning")
        check.equal(png_pixels(out/row["image_path"]),image_from_packet(packet),"independent PNG truncation "+blind)
        if row["task_id"] in paired:
            old=paired[row["task_id"]]
            for key in ("initial_from_prepare","conditioning"):
                check.equal(packet[key],old[key],"E4/E8 paired actual input "+key)
            integer_equal(check,packet["timesteps"],old["timesteps"],"E4/E8 same integer time grid")
        else:
            paired[row["task_id"]]={key:packet[key].copy() for key in ("initial_from_prepare","conditioning","timesteps")}
    return {"records":len(manifest),"checkpoint_choices":choices,"DDIM_transitions_checked":30*len(manifest),
            "paired_checkpoint_tasks":len(paired) if phase=="selection" else 0,"trajectory_checks":details,
            "visual_quality_assessed_by_this_verifier":False,
            "diagnostic_conditioning_scope":"Same new cache across E4/E8; historical embedding exactness is not assumed."}


def verify(out,phase):
    out=Path(out).resolve()
    result_path=out/(phase+"_verification.json")
    protocol_path=out/(phase+"_verification_protocol.json")
    if result_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve previous independent verification attempt")
    names=["contract.json","cache.pt","cache_report.json","training_report.json"]
    if phase=="training":
        names.extend(["initial_adapter.pt","preflight.json","preflight.npz","trace.jsonl","checkpoints/E4.pt","checkpoints/E8.pt"])
    else:
        names.extend(["training_verification.json",f"generation_{phase}.json",f"manifest_{phase}.json",f"generation_inputs_{phase}.pt"])
    bound={name:sha(out/name) for name in names}
    source=sha(__file__)
    save(protocol_path,{"schema":"independent-public-medical-pilot-verification/v1","phase":phase,
        "created_utc":datetime.now(timezone.utc).isoformat(),"code_sha256":source,"initial_input_sha256":dict(bound),
        "numerical_policy":"Frozen verifier source is checked against runtime contract. Generation CFG/cast/PNG and paired inputs exact; integer timestep representations int32/int64 accepted only with exact range/values. DDIM saved input/epsilon updates use an independent FP64 equation and dtype gamma32 envelope. Training preflight FP16 noising uses gamma16 envelope. No tolerance fitted to outputs.",
        "independence":"No producer numerical imports or GPU initialization. No full Adam/UNet/VAE/text-encoder replay.",
        "quality_or_privacy_certification":False})
    check=Check();start=time.perf_counter()
    try:
        contract=read(out/"contract.json")
        spec,train,schedule_rows,tasks=verify_plan(check,out,contract)
        if phase=="training":
            cache,_=verify_cache(check,out,train)
            check.require(set(cache["hidden"])==set(cache["train_prompts"].values())|{r["prompt"] for r in tasks},
                          "no extra conditioning cache inputs")
            initial,preflight=verify_preflight(check,out,cache,contract)
            first=schedule_rows[:4]
            check.require(preflight["image_ids"]==[row["image_id"] for row in first]
                          and preflight["noise_seed"]==int(first[0]["noise_seed"]),"preflight first planned batch/seed")
            integer_equal(check,arrays(out/"preflight.npz")["timesteps"],np.asarray([int(row["timestep"]) for row in first],np.int64),
                          "preflight actual first planned time values")
            _,_,details=verify_training(check,out,cache,contract,schedule_rows,initial,preflight)
        else:
            prior_verification(check,out,"training")
            cache=torch.load(out/"cache.pt",map_location="cpu",weights_only=True)
            report=read(out/"training_report.json")
            details=verify_generation(check,out,phase,contract,cache,tasks,report,bound)
        for name,digest in bound.items():
            check.require(sha(out/name)==digest,"verification input unchanged "+name)
        check.require(sha(__file__)==source,"independent source unchanged")
        check.require(not torch.cuda.is_initialized(),"verifier did not initialize CUDA")
        result={"status":PHASE_STATUS[phase],"complete":True,"phase":phase,"checks":check.count,
            "seconds":time.perf_counter()-start,"code_sha256":source,"protocol_sha256":sha(protocol_path),
            "input_sha256":bound,"details":details,"maximum_absolute_errors":check.errors,
            "new_UNet_forwards":0,"new_VAE_encodes_or_decodes":0,"new_optimizer_steps":0,
            "limits":["Saved training inputs, successful/attempted update accounting, optimizer state and exposure validation; all1498 gradient/Adam operations are not independently replayed.",
                "Frozen base and loaded generation adapter equality are producer-recorded hash/exact-comparison evidence, linked to immutable code and checkpoints.",
                "Cache image IDs, bytes, shapes and fixed prompt policy are checked; VAE/text numerical encoding is not independently rerun.",
                "CUDA training noise is not treated as equal to CPU RNG output; source seeds/retry schedule and saved preflight noising are checked.",
                "Generation checks saved CFG/DDIM/image arithmetic and fixed paired inputs, not all UNet/VAE operations or visual/clinical correctness.",
                "Confirmation decision checks fixed count thresholds and earliest-checkpoint rule; reviewer visual judgments are external inputs.",
                "No private-incremental utility, patient-DP protection, clinical quality or population efficacy claim."]}
        save(result_path,result)
        return result
    except Exception as error:
        save(result_path,{"status":"FAILED_PUBLIC_MEDICAL_PILOT_VERIFICATION","complete":False,"phase":phase,
            "error":repr(error),"traceback":traceback.format_exc(),"checks":check.count,"seconds":time.perf_counter()-start,
            "code_sha256":source,"protocol_sha256":sha(protocol_path),"input_sha256":bound,
            "maximum_absolute_errors":check.errors})
        raise


def self_test():
    check=Check()
    config=dict(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule="scaled_linear",
        prediction_type="epsilon",clip_sample=False,thresholding=False,set_alpha_to_one=False,steps_offset=1,timestep_spacing="leading")
    alphas=independent_alphas(config);rng=np.random.default_rng(26091688)
    latents=[rng.normal(size=(1,4,32,32)).astype(np.float32)]
    inputs=[];raw=[];eps=[]
    for t in schedule(config):
        epsilon=rng.normal(0,.01,size=(1,4,32,32)).astype(np.float32)
        inputs.append(latents[-1].copy());raw.append(epsilon);eps.append(epsilon.copy())
        latents.append(ddim(latents[-1],epsilon,int(t),alphas,config)[0])
    decoded=rng.normal(0,.4,size=(1,3,256,256)).astype(np.float32)
    image=(torch.from_numpy(decoded)/2.+.5).clamp(0,1).permute(0,2,3,1).numpy()[0]
    packet=dict(conditioning=np.zeros((1,77,1024),np.float32),timesteps=schedule(config).astype(np.int32),
        latents=np.stack(latents),model_inputs=np.stack(inputs),raw_eps=np.stack(raw),eps=np.stack(eps),
        decoded_raw=decoded,image_float=image,initial_from_prepare=latents[0].copy())
    check_packet(check,packet,"public_generation_fp32",1.,config)
    integer_equal(check,packet["timesteps"],schedule(config),"signed integer widths have same values")
    check.require(planned_seed("noise",0)==135200119256219742 and planned_seed("timestep",0,0)%1000==994,
                  "known frozen first schedule seeds")
    check.require(weak_prompt("Effusion")=="a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion",
                  "weak-label mapping")
    bad={key:value.copy() for key,value in packet.items()};bad["eps"][0,0,0,0,0]+=.125
    rejected=False
    try:
        check_packet(Check(),bad,"public_generation_fp32",1.,config)
    except AssertionError:
        rejected=True
    check.require(rejected,"altered epsilon packet is rejected")
    check.require(not torch.cuda.is_initialized(),"CPU-only synthetic self-test")
    return {"status":"PASS_SYNTHETIC_PUBLIC_SAMPLING_SEEDS_AND_MUTATION_REJECTION","checks":check.count}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=DEFAULT)
    parser.add_argument("--phase",choices=tuple(PHASE_STATUS))
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.expected_code_sha256 and sha(__file__)!=args.expected_code_sha256:
        raise RuntimeError("Independent verifier source hash mismatch")
    if args.self_test:
        result=self_test()
    else:
        if not args.phase:
            parser.error("--phase training, selection or confirmation is required")
        result=verify(args.out,args.phase)
        result={key:result[key] for key in ("status","complete","phase","checks","seconds","code_sha256")}
    print(json.dumps(result,ensure_ascii=False,indent=2))
