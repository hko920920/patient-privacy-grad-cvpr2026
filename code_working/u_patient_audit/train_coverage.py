"""Coverage-controlled non-DP pilot v2. The earlier 50-step run remains a separate smoke."""
import argparse,time,json,gc
from collections import Counter
from .common import *
from .models import *
from .train_pair import save

def epoch_indices(n,step):
    assert n%4==0
    epoch=(step*4)//n;offset=(step*4)%n
    g=torch.Generator().manual_seed(seed("coverage-epoch",epoch))
    return torch.randperm(n,generator=g)[offset:offset+4].tolist()

def contract():
    bench=json.loads((RUN/"batch_benchmark.json").read_text())
    assert bench["status"]=="PASS"
    return {"schema":"cvpr-u-training/v2","cohort_lock_sha256":digest(RUN/"cohort/lock.json"),
        "batch_benchmark_sha256":digest(RUN/"batch_benchmark.json"),"model_revision":MODEL_REVISION,
        "non_DP":True,"rank":8,"max_steps":1000,"microbatch":4,"accumulation":1,
        "learning_rate":1e-4,"optimizer":"AdamW","betas":[.9,.999],"eps":1e-8,
        "weight_decay":.01,"gradient_norm_cap":1.0,"initial_gradient_scale":1024.,
        "sampler":"deterministic epoch permutations without replacement",
        "coverage":"every one of 912 training images used at least once by step 228",
        "steps":1000,"save_steps":[250,1000],"init_seed":260914,
        "prompt_policy":"existing weak-label training prompts; generic audit prompt",
        "code_sha256":digest(Path(__file__)),"models_code_sha256":digest(Path(__file__).with_name("models.py")),
        "save_helper_code_sha256":digest(Path(__file__).with_name("train_pair.py"))}

def run(model,cache,config,stop):
    out=RUN/"training_coverage_v2"/model;out.mkdir(parents=True,exist_ok=True)
    final=out/f"step_{stop:04d}.pt"
    if final.exists():raise RuntimeError("completed checkpoint exists; verify it instead")
    checkpoints=sorted(out.glob("step_*.pt"));prior=checkpoints[-1] if checkpoints else None
    saved=torch.load(prior,map_location="cpu",weights_only=True) if prior else None
    if saved:assert saved["contract"]==config and saved["step"]<stop
    unet=load_unet(prior,training=True)
    params=[p for p in unet.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024.)
    step=0;attempt=0;exposures=Counter();losses=[]
    if saved:
        opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
        step=saved["step"];attempt=saved["attempt"];exposures.update(saved["exposures"]);losses=saved["losses"]
        del saved
    rows=read_csv(RUN/"cohort"/(model+"_train.csv"))
    rows.sort(key=lambda r:(r["eval_role"]!="background",stable("training-order",r["patient_id"]),r["image_id"]))
    assert len(rows)==912
    sched=scheduler();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();start_step=step
    log=(out/"training_trace.jsonl").open("a",encoding="utf-8")
    while step<stop:
        attempt+=1;opt.zero_grad(set_to_none=True)
        batch_rows=[rows[i] for i in epoch_indices(len(rows),step)]
        z=torch.cat([cache["latents"][r["image_id"]].clone() for r in batch_rows]).to("cuda",torch.float16)
        h=torch.cat([cache["hidden"][cache["train_prompts"][r["image_id"]]].clone() for r in batch_rows]).to("cuda",torch.float16)
        g=torch.Generator(device="cuda").manual_seed(seed("coverage-noise",step))
        noise=torch.randn(z.shape,generator=g,device="cuda",dtype=torch.float16)
        t=torch.tensor([seed("coverage-timestep",step,i)%1000 for i in range(4)],device="cuda")
        noisy,target=diffusion_input(z,t,noise,sched)
        with torch.autocast("cuda",dtype=torch.float16):
            pred=unet(noisy,t,h).sample
            loss=(pred.float()-target.float()).square().mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        old=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<old:
            print(json.dumps({"model":model,"overflow_retry_step":step,"scale":scaler.get_scale()}),flush=True)
            assert attempt<stop+100
            continue
        assert torch.isfinite(norm)
        exposures.update(r["image_id"] for r in batch_rows)
        step+=1;losses.append(float(loss.detach()))
        entry={"step":step,"attempt":attempt,"loss":losses[-1],"gradient_norm":float(norm),
            "seconds":time.perf_counter()-started}
        log.write(json.dumps(entry)+"\n");log.flush()
        if step%50==0 or step==1:
            torch.cuda.synchronize()
            progress={"model":model,"step":step,"target":stop,
                "loss_last50":sum(losses[-50:])/len(losses[-50:]),
                "seconds_per_step":(time.perf_counter()-started)/max(step-start_step,1),
                "max_cuda_GiB":torch.cuda.max_memory_allocated()/2**30}
            write_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
        if step in config["save_steps"] or step==stop:
            save(unet,opt,scaler,step,attempt,exposures,losses,out/f"step_{step:04d}.pt",config)
    log.close()
    values=[exposures.get(r["image_id"],0) for r in rows]
    assert step<228 or min(values)>=1
    report={"status":"PASS_TRAINING_EXECUTION_AND_COVERAGE","model":model,"step":step,
        "attempts":attempt,"training_images":len(rows),"unique_exposed_images":len(exposures),
        "exposure_min":min(values),"exposure_max":max(values),"committed_exposure_total":sum(values),
        "all_declared_member_patients_actually_used":min(values)>=1,
        "checkpoint_sha256":digest(final),"contract_sha256":digest(RUN/"training_coverage_v2/protocol.json"),
        "seconds_this_segment":time.perf_counter()-started,
        "loss_first50":sum(losses[:50])/min(50,len(losses)),
        "loss_last50":sum(losses[-50:])/min(50,len(losses)),
        "max_cuda_GiB":torch.cuda.max_memory_allocated()/2**30,
        "quality_gate":"NOT_YET_EVALUATED","medical_MIA":"NOT_YET_EVALUATED"}
    write_json(out/f"report_{step:04d}.json",report);print(json.dumps(report),flush=True)
    del unet,opt,scaler,params;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["model_1","model_2","both"],default="both")
    parser.add_argument("--stop",type=int,choices=[250,1000],default=1000)
    args=parser.parse_args()
    setup();verify_inputs();cache=load_cache();config=contract()
    path=RUN/"training_coverage_v2/protocol.json"
    if path.exists():assert json.loads(path.read_text())==config
    else:write_json(path,config)
    for model in (["model_1","model_2"] if args.model=="both" else [args.model]):
        run(model,cache,config,args.stop)
if __name__=="__main__":main()

