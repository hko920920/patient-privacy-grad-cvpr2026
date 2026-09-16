"""Controlled non-DP LoRA pair on new public U manifests; never enters a thesis runner."""
import argparse, time, json, gc
from collections import Counter
from .common import *
from .models import *

def protocol():
    return {"schema":"cvpr-u-training/v1","cohort_lock_sha256":digest(RUN/"cohort"/"lock.json"),
        "model_revision":MODEL_REVISION,"non_DP":True,"rank":8,
        "max_steps":1000,"microbatch":1,"accumulation":4,"optimizer":"AdamW",
        "learning_rate":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":.01,
        "gradient_norm_cap":1.0,"clip_is_not_a_DP_claim":True,"init_seed":260914,
        "noise_policy":"same per-step ordinal draws across pair, independent per image evaluation",
        "prompt_policy":"existing weak-label training prompts; generic audit prompt",
        "snapshot_steps":[50,250,1000],"sampler":"uniform image draws with replacement",
        "code_sha256":digest(Path(__file__)),"models_code_sha256":digest(Path(__file__).with_name("models.py"))}

def save(unet,opt,scaler,step,attempt,exposures,losses,path,contract):
    payload={"adapter":adapter_state(unet),"optimizer":opt.state_dict(),
        "scaler":scaler.state_dict(),"step":step,"attempt":attempt,
        "exposures":dict(exposures),"losses":losses,"contract":contract}
    temp=path.with_suffix(".tmp")
    torch.save(payload,temp);temp.replace(path)

def run(model,cache,contract,stop):
    out=RUN/"training"/model;out.mkdir(parents=True,exist_ok=True)
    final=out/f"step_{stop:04d}.pt"
    if final.exists():
        print(json.dumps({"model":model,"status":"checkpoint_exists","step":stop}),flush=True);return
    checkpoints=sorted(out.glob("step_*.pt"))
    checkpoint=checkpoints[-1] if checkpoints else None
    if checkpoint:
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert saved["contract"]==contract,"training code/contract drift; explicitly version first"
        assert saved["step"]<stop
    else:saved=None
    unet=load_unet(checkpoint,training=True)
    params=[p for p in unet.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024.)
    step=0;attempt=0;exposures=Counter();losses=[]
    if saved:
        opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
        step=saved["step"];attempt=saved["attempt"]
        exposures.update(saved["exposures"]);losses=saved["losses"]
        del saved
    rows=read_csv(RUN/"cohort"/(model+"_train.csv"))
    # Same group ordering rules, no U or reference image can enter.
    rows.sort(key=lambda r:(r["eval_role"]!="background",stable("training-order",r["patient_id"]),r["image_id"]))
    assert len(rows)==912
    sched=scheduler();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();initial_step=step
    log=(out/"training_trace.jsonl").open("a",encoding="utf-8")
    while step<stop:
        attempt+=1;opt.zero_grad(set_to_none=True);total_loss=0.
        g=torch.Generator().manual_seed(seed("training-index",attempt))
        indices=torch.randint(len(rows),(4,),generator=g).tolist()
        for ordinal,ix in enumerate(indices):
            row=rows[ix];image_id=row["image_id"];exposures[image_id]+=1
            z=cache["latents"][image_id].clone().to("cuda",torch.float16)
            hidden=cache["hidden"][cache["train_prompts"][image_id]].clone().to("cuda",torch.float16)
            ng=torch.Generator(device="cuda").manual_seed(seed("training-noise",attempt,ordinal))
            noise=torch.randn(z.shape,generator=ng,device="cuda",dtype=torch.float16)
            t=torch.tensor([seed("training-timestep",attempt,ordinal)%1000],device="cuda")
            noisy,target=diffusion_input(z,t,noise,sched)
            with torch.autocast("cuda",dtype=torch.float16):
                pred=unet(noisy,t,hidden).sample
                loss=torch.nn.functional.mse_loss(pred.float(),target.float())
            assert torch.isfinite(loss),"nonfinite training loss"
            scaler.scale(loss/4).backward();total_loss+=float(loss.detach())/4
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        previous=scaler.get_scale()
        scaler.step(opt);scaler.update()
        if scaler.get_scale()<previous:
            print(json.dumps({"model":model,"phase":"scaled_gradient_overflow","attempt":attempt,"scale":scaler.get_scale()}),flush=True)
            assert attempt<stop+100
            continue
        assert torch.isfinite(norm)
        step+=1;losses.append(total_loss)
        entry={"step":step,"attempt":attempt,"loss":total_loss,"gradient_norm":float(norm),
            "elapsed_sec":time.perf_counter()-started}
        log.write(json.dumps(entry)+"\n");log.flush()
        if step%25==0 or step==1:
            torch.cuda.synchronize()
            progress={"model":model,"step":step,"target":stop,
                "loss_last25":sum(losses[-25:])/len(losses[-25:]),
                "seconds_per_step":(time.perf_counter()-started)/max(step-initial_step,1),
                "max_cuda_GiB":torch.cuda.max_memory_allocated()/2**30}
            write_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
        if step in contract["snapshot_steps"] or step==stop:
            save(unet,opt,scaler,step,attempt,exposures,losses,out/f"step_{step:04d}.pt",contract)
    log.close()
    report={"status":"PASS_TRAINING_EXECUTION","model":model,"step":step,"attempts":attempt,
        "training_images":len(rows),"unique_exposed_images":len(exposures),
        "exposure_min":min(exposures.get(r["image_id"],0) for r in rows),
        "exposure_max":max(exposures.values()),"exposure_total":sum(exposures.values()),
        "loss_first50":sum(losses[:50])/min(50,len(losses)),
        "loss_last50":sum(losses[-50:])/min(50,len(losses)),
        "checkpoint_sha256":digest(final),"seconds_this_segment":time.perf_counter()-started,
        "max_cuda_GiB":torch.cuda.max_memory_allocated()/2**30,
        "quality_gate":"NOT_YET_EVALUATED","medical_MIA":"NOT_YET_EVALUATED"}
    write_json(out/f"report_{step:04d}.json",report)
    print(json.dumps(report),flush=True)
    del unet,opt,scaler,params;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stop",type=int,choices=[50,250,1000],default=50)
    parser.add_argument("--model",choices=["model_1","model_2","both"],default="both")
    args=parser.parse_args()
    setup();verify_inputs();cache=load_cache();contract=protocol()
    path=RUN/"training"/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==contract
    else:write_json(path,contract)
    for model in (["model_1","model_2"] if args.model=="both" else [args.model]):
        run(model,cache,contract,args.stop)

if __name__=="__main__":main()

