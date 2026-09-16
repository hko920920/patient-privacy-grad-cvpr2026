"""Fixed conditioning-only diffusion loss diagnostic; no fitting or target training."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback
import uuid
import torch
from .common import ROOT,RUN,PROMPT,MODEL_ID,MODEL_REVISION,digest,read_csv
from .models import setup,snapshot,load_cache,load_unet,UNet2DConditionModel,DDPMScheduler,adapter_state
from .run_cdi_base_selection import state_fingerprint

OUTPUT=RUN/"baseline_screen_20260914/prompt_loss_t140_v1"
KINDS=("null","generic","weak_label")
MODELS=("base","model_1","model_2")
T=140

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def now():return datetime.now(timezone.utc).isoformat()

def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),"Immutable output exists"
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".pending")
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    tmp.rename(path)

def tensor_sha(x):
    assert x.device.type=="cpu"
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()

def epsilon_for(iid):
    payload=f"promptloss-v1|260915|{iid}|140".encode()
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")%(2**63-1)
    eps=torch.randn((1,4,32,32),generator=torch.Generator(device="cpu").manual_seed(seed),
                    dtype=torch.float32,device="cpu")
    return seed,eps

def check_hashes(frozen):
    for p,h in frozen.items():assert digest(Path(p))==h,"Changed frozen input: "+p

def preflight(path):
    c=read(path)
    expected=dict(schema="prompt-loss-t140-contract/v1",current_step=2,
        models=list(MODELS),prompt_kinds=list(KINDS),timestep=140,batch_size=1,dtype="float32",
        prediction_type="epsilon",master_seed=260915,expected_patients={"fit":80,"selection":40},
        expected_unique_images=480,expected_records=1440,expected_cells=4320,forward=4320,backward=0,
        new_target_training=False,perform_fitting=False,perform_analysis=False)
    for k,v in expected.items():assert c[k]==v,"Contract "+k
    assert Path(c["output_directory"]).resolve()==OUTPUT.resolve()
    frozen=c["frozen_sha256"];check_hashes(frozen)
    assert frozen[str(Path(__file__).resolve())]==digest(__file__)
    snap=snapshot().resolve()
    assert snap.name==MODEL_REVISION and str(snap)==c["base_snapshot_path"]
    weight=snap/"unet/diffusion_pytorch_model.fp16.safetensors"
    assert str(weight)==c["base_weight_path"] and digest(weight)==c["base_weight_sha256"]
    cache=load_cache();assert cache["audit_prompt"]==PROMPT
    lock=read(RUN/"cohort/lock.json")
    for n,h in lock["files"].items():assert digest(RUN/"cohort"/n)==h
    allrows=read_csv(RUN/"cohort/evaluation_images.csv")
    groups=defaultdict(list)
    for r in allrows:
        if r["eval_role"] in ("fit","selection"):groups[r["patient_id"]].append(r)
    rows=[]
    for role in ("fit","selection"):
        patients=sorted((p for p,rs in groups.items() if rs[0]["eval_role"]==role),key=int)
        assert patients==c["patient_order"][role] and len(patients)==c["expected_patients"][role]
        assert Counter(groups[p][0]["assignment_group"] for p in patients)=={"A":len(patients)//2,"B":len(patients)//2}
        for p in patients:
            rs=groups[p];assert len(rs)==4 and len({r["assignment_group"] for r in rs})==1
            for scenario,rr in (("E","train_candidate"),("U","U_observed")):
                images=sorted((r for r in rs if r["record_role"]==rr),key=lambda r:r["image_id"])
                assert len(images)==2
                for r in images:rows.append(dict(r,scenario=scenario))
    assert len(rows)==len({r["image_id"] for r in rows})==480
    assert [r["image_id"] for r in rows]==c["image_order"]
    ledgers={}
    for model in MODELS[1:]:
        cp=Path(c["checkpoint_paths"][model]);assert digest(cp)==c["checkpoint_sha256"][model]
        state=torch.load(cp,map_location="cpu",weights_only=True)
        train=read_csv(RUN/"cohort"/(model+"_train.csv"))
        exposure=state["exposures"];assert state["step"]==1000
        assert set(exposure)=={r["image_id"] for r in train} and sum(exposure.values())==4000 and min(exposure.values())>0
        bypatient=defaultdict(int)
        for r in train:bypatient[r["patient_id"]]+=int(exposure[r["image_id"]])
        ledgers[model]=dict(exposures=exposure,patients=dict(bypatient),adapter=state["adapter"])
        for r in rows:
            member=int(r["assignment_group"]==("A" if model=="model_1" else "B"))
            assert int(bypatient.get(r["patient_id"],0)>0)==member
            assert int(exposure.get(r["image_id"],0)>0)==int(member and r["scenario"]=="E")
    prompts={}
    for r in rows:
        iid=r["image_id"];z=cache["latents"][iid]
        assert z.shape==(1,4,32,32) and z.dtype==torch.float16 and bool(torch.isfinite(z).all())
        prompts[iid]=dict(null="",generic=PROMPT,weak_label=cache["train_prompts"][iid])
        for prompt in prompts[iid].values():
            h=cache["hidden"][prompt]
            assert h.shape==(1,77,1024) and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
        seed,eps=epsilon_for(iid);assert eps.shape==(1,4,32,32) and bool(torch.isfinite(eps).all())
    sched=DDPMScheduler.from_pretrained(snap/"scheduler",local_files_only=True)
    assert sched.config.prediction_type=="epsilon" and len(sched.alphas_cumprod)==1000
    original_raw=torch.load(RUN/"baseline_screen_20260914/cdi_kernel_v1/raw.pt",map_location="cpu",weights_only=True)
    assert all(torch.equal(v["alphas"],sched.alphas_cumprod.float()) for v in original_raw.values())
    del original_raw
    return dict(contract=c,rows=rows,cache=cache,ledgers=ledgers,prompts=prompts,snapshot=snap,
                weight=weight,scheduler=sched)

def label(r,model,ctx):
    if model=="base":member=image_member=patient_exposures=exposures=0
    else:
        ledger=ctx["ledgers"][model]
        exposures=int(ledger["exposures"].get(r["image_id"],0))
        patient_exposures=int(ledger["patients"].get(r["patient_id"],0))
        member=int(patient_exposures>0);image_member=int(exposures>0)
    return dict(model=model,patient_id=r["patient_id"],image_id=r["image_id"],eval_role=r["eval_role"],
        scenario=r["scenario"],record_role=r["record_role"],assignment_group=r["assignment_group"],
        member=member,incremental_patient_member=member,image_member=image_member,
        actual_training_exposures=exposures,patient_training_exposures=patient_exposures,
        membership_label_scope="additional_NIH_training_only",pretrained_membership="unknown",
        assignment_group_is_base_membership=False)

def guards(unet,versions):
    ps=dict(unet.named_parameters())
    assert ps.keys()==versions.keys() and not unet.training
    assert all(p._version==versions[n] and p.grad is None and not p.requires_grad for n,p in ps.items())

def save_packet(model,iid,raw,row):
    directory=OUTPUT/model;directory.mkdir(exist_ok=True)
    path=directory/(Path(iid).stem+".pt");assert not path.exists()
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".pending")
    with tmp.open("xb") as f:torch.save(raw,f)
    h=digest(tmp);tmp.rename(path)
    row=dict(row,raw_path=path.relative_to(OUTPUT).as_posix(),raw_sha256=h)
    new_json(path.with_suffix(".json"),row)
    return row

def run(path,ctx):
    c=ctx["contract"];contract_sha=digest(path)
    assert not OUTPUT.exists(),"Immutable diagnostic output exists"
    OUTPUT.mkdir(parents=True,exist_ok=False)
    new_json(OUTPUT/"protocol.json",dict(schema="prompt-loss-t140-execution/v1",current_step=2,
        started_utc=now(),contract_path=str(path),contract_sha256=contract_sha,contract=c,
        environment={n:version(n) for n in ("torch","numpy","diffusers","peft","safetensors")},
        source_policy="Cached FP16 VAE posterior mode and text embeddings promoted to FP32; three fixed conditions",
        noise_policy="CPU FP32 one epsilon per image, shared across all three models and prompts",
        weak_label_policy="Cached image-specific template; for unseen U this is a hypothetical caption, not a prompt used to train that U image",
        measurement_plan=[{k:r[k] for k in ("patient_id","image_id","eval_role","scenario","record_role","assignment_group")} for r in ctx["rows"]],
        downstream_analysis="Separate frozen analysis/verification contract required after complete raw extraction; no performance analysis in producer"))
    protocol_sha=digest(OUTPUT/"protocol.json")
    setup();started=time.perf_counter();allrows=[];cells=[];model_reports=[]
    cheap={str(path):contract_sha,**{p:h for p,h in c["frozen_sha256"].items() if Path(p).suffix==".py"}}
    try:
        for model in MODELS:
            model_start=time.perf_counter();unet=None;handles=[]
            try:
                if model=="base":
                    unet=UNet2DConditionModel.from_pretrained(ctx["snapshot"]/"unet",torch_dtype=torch.float16,
                        variant="fp16",use_safetensors=True,local_files_only=True).to("cuda",dtype=torch.float32)
                    assert not any("lora" in n.lower() for n,_ in unet.named_parameters()) and not getattr(unet,"peft_config",None)
                else:
                    unet=load_unet(Path(c["checkpoint_paths"][model]),training=False).to(dtype=torch.float32)
                    unet.enable_adapters()
                unet.eval().requires_grad_(False);unet.disable_gradient_checkpointing()
                assert all(p.dtype==torch.float32 for p in unet.parameters())
                if model!="base":
                    current=adapter_state(unet);expected=ctx["ledgers"][model]["adapter"]
                    assert current.keys()==expected.keys()
                    assert all(torch.equal(current[n],expected[n].float()) for n in current)
                    del current
                before=state_fingerprint(unet,ctx["weight"] if model=="base" else None)
                versions={n:p._version for n,p in unet.named_parameters()};guards(unet,versions)
                counter={"forward":0,"backward":0}
                def forward_hook(module,args,output):counter["forward"]+=1
                def backward_hook(module,grad_input,grad_output):counter["backward"]+=1
                handles=[unet.register_forward_hook(forward_hook),unet.register_full_backward_hook(backward_hook)]
                hidden={prompt:ctx["cache"]["hidden"][prompt].float().to("cuda") for prompt in
                        {s for d in ctx["prompts"].values() for s in d.values()}}
                hidden_hash={p:tensor_sha(h.detach().cpu()) for p,h in hidden.items()}
                torch.cuda.reset_peak_memory_stats();model_rows=[];cuda_total=0.
                for r in ctx["rows"]:
                    iid=r["image_id"];seed,eps=epsilon_for(iid)
                    z=ctx["cache"]["latents"][iid].float().to("cuda");noise=eps.to("cuda")
                    timesteps=torch.tensor([T],device="cuda",dtype=torch.long)
                    noisy=ctx["scheduler"].add_noise(z,noise,timesteps)
                    predictions,losses,timings={},{},{}
                    before_f,before_b=counter["forward"],counter["backward"]
                    image_start=time.perf_counter()
                    for kind in KINDS:
                        start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                        with torch.inference_mode():
                            start_event.record()
                            pred=unet(noisy,timesteps,encoder_hidden_states=hidden[ctx["prompts"][iid][kind]],return_dict=False)[0]
                            end_event.record()
                            assert pred.shape==(1,4,32,32) and pred.dtype==torch.float32 and bool(torch.isfinite(pred).all())
                            loss=(pred-noise).square().mean()
                            assert bool(torch.isfinite(loss))
                            predictions[kind]=pred.cpu()
                            losses[kind]=float(loss)
                        end_event.synchronize();timings[kind]=float(start_event.elapsed_time(end_event))
                    assert counter["forward"]-before_f==3 and counter["backward"]==before_b==0
                    guards(unet,versions)
                    noise_hash=tensor_sha(eps)
                    raw=dict(model=model,image_id=iid,timestep=T,latent=z.cpu(),epsilon=eps,
                        noised_latent=noisy.cpu(),alphas=ctx["scheduler"].alphas_cumprod.detach().cpu().float(),
                        predictions=predictions,noise_seed=seed,noise_sha256=noise_hash,
                        prompt_hidden_sha256={k:hidden_hash[ctx["prompts"][iid][k]] for k in KINDS})
                    cp_sha=c["base_weight_sha256"] if model=="base" else c["checkpoint_sha256"][model]
                    row=dict(**label(r,model,ctx),timestep=T,batch_size=1,dtype="float32",prediction_type="epsilon",
                        prompt_texts=ctx["prompts"][iid],prompt_hidden_sha256=raw["prompt_hidden_sha256"],
                        noise_seed=seed,noise_sha256=noise_hash,losses=losses,
                        cuda_event_forward_ms=timings,forward=3,backward=0,
                        seconds=time.perf_counter()-image_start,weights_unchanged=True,unet_training=False,
                        checkpoint_sha256=cp_sha,base_weight_sha256=c["base_weight_sha256"],
                        cache_sha256=c["frozen_sha256"][str((RUN/"cache/cache.pt").resolve())],
                        cohort_lock_sha256=c["frozen_sha256"][str((RUN/"cohort/lock.json").resolve())],
                        contract_sha256=contract_sha)
                    row=save_packet(model,iid,raw,row);allrows.append(row);model_rows.append(row)
                    cuda_total+=sum(timings.values())
                    for kind in KINDS:
                        cells.append(dict(**{k:v for k,v in row.items() if k not in ("losses","prompt_texts","cuda_event_forward_ms","forward","backward")},
                            prompt_kind=kind,prompt_text=ctx["prompts"][iid][kind],loss_mse=losses[kind],
                            negative_loss=-losses[kind],cuda_event_forward_ms=timings[kind],forward=1,backward=0))
                    del raw,predictions,pred,z,noise,noisy,eps
                    if len(model_rows)%80==0:
                        check_hashes(cheap)
                        print(json.dumps(dict(event="progress",model=model,images=len(model_rows),total_images=480,
                            forward=counter["forward"],backward=counter["backward"],seconds=time.perf_counter()-model_start)),flush=True)
                guards(unet,versions)
                after=state_fingerprint(unet)
                assert before["sha256"]==after["sha256"]
                assert len(model_rows)==480 and counter=={"forward":1440,"backward":0}
                if model!="base":
                    current=adapter_state(unet);expected=ctx["ledgers"][model]["adapter"]
                    assert current.keys()==expected.keys() and all(torch.equal(current[n],expected[n].float()) for n in current)
                    del current
                model_reports.append(dict(model=model,records=480,forward=1440,backward=0,
                    initial_state=before,final_state=after,state_values_exactly_unchanged=True,
                    parameter_versions_checked=len(versions),parameter_versions_and_no_gradients=True,
                    all_parameters_frozen=True,unet_training=False,gradient_checkpointing_enabled=False,
                    no_lora_modules=model=="base",adapter_matches_checkpoint=True if model!="base" else None,
                    cuda_event_forward_ms_total=cuda_total,seconds=time.perf_counter()-model_start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
                del hidden
            finally:
                for h in handles:h.remove()
                unet=None;gc.collect();torch.cuda.empty_cache()
        assert len(allrows)==1440 and len(cells)==4320
        for row in allrows:
            p=OUTPUT/row["raw_path"]
            assert digest(p)==row["raw_sha256"] and read(p.with_suffix(".json"))==row
        check_hashes(c["frozen_sha256"]);assert digest(path)==contract_sha
        new_json(OUTPUT/"results.json",allrows);new_json(OUTPUT/"cells.json",cells)
        new_json(OUTPUT/"execution.json",dict(schema="prompt-loss-t140-result/v1",
            status="COMPLETE_RAW_PENDING_INDEPENDENT_VERIFICATION",complete=True,current_step=2,
            records=1440,cells=4320,unique_images=480,unique_patients=120,fit_patients=80,selection_patients=40,
            forward=4320,backward=0,vae_forward=0,text_encoder_forward=0,model_reports=model_reports,
            checkpoint_sha256=c["checkpoint_sha256"],base_weight_sha256=c["base_weight_sha256"],
            contract_sha256=contract_sha,protocol_sha256=protocol_sha,
            results_sha256=digest(OUTPUT/"results.json"),cells_sha256=digest(OUTPUT/"cells.json"),
            frozen_files_unchanged=True,performance_analysis=False,attack_fitting=False,new_target_training=False,
            pretrained_membership="unknown",seconds=time.perf_counter()-started,completed_utc=now()))
        print(json.dumps(dict(status="COMPLETE_RAW_PENDING_INDEPENDENT_VERIFICATION",forward=4320,backward=0,
                              seconds=time.perf_counter()-started,output=str(OUTPUT))),flush=True)
    except BaseException as exc:
        new_json(OUTPUT/"failure.json",dict(status="FAILED_INCOMPLETE_PROMPT_LOSS_DIAGNOSTIC",complete=False,
            records_saved=len(allrows),error=repr(exc),traceback=traceback.format_exc(),failed_utc=now()))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true");mode.add_argument("--run",action="store_true")
    a=p.parse_args();a.contract=a.contract.resolve()
    ctx=preflight(a.contract)
    if a.dry_run:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status="PASS_CPU_PREFLIGHT",unique_images=480,records=1440,cells=4320,
            forward_plan=4320,backward_plan=0,GPU_initialized=False,contract_sha256=digest(a.contract))),flush=True)
    else:run(a.contract,ctx)

if __name__=="__main__":main()
