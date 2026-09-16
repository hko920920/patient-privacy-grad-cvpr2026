import json,time,gc
from .common import *
from .models import *

def main():
    setup();verify_inputs();cache=load_cache()
    unet=load_unet(RUN/"training/model_1/step_0050.pt",training=True)
    rows=read_csv(RUN/"cohort/model_1_train.csv")[:4]
    zs=[];hs=[];noises=[];ts=[]
    for i,r in enumerate(rows):
        z=cache["latents"][r["image_id"]].clone()
        zs.append(z);hs.append(cache["hidden"][cache["train_prompts"][r["image_id"]]].clone())
        g=torch.Generator().manual_seed(seed("batch-check",i))
        noises.append(torch.randn(z.shape,generator=g,dtype=torch.float16));ts.append(250+i*100)
    z=torch.cat(zs).to("cuda");h=torch.cat(hs).to("cuda")
    noise=torch.cat(noises).to("cuda");t=torch.tensor(ts,device="cuda")
    noisy,target=diffusion_input(z,t,noise,scheduler())
    outputs=[];grads=[]
    params=[p for p in unet.parameters() if p.requires_grad]
    for batch in [1,4]:
        times=[];loss_value=0.
        for repeat in range(3):
            unet.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();started=time.perf_counter();loss_value=0.
            for i in range(0,4,batch):
                with torch.autocast("cuda",dtype=torch.float16):
                    pred=unet(noisy[i:i+batch],t[i:i+batch],h[i:i+batch]).sample
                    loss=(pred.float()-target[i:i+batch].float()).square().mean()*batch/4
                # Match the GradScaler scale in the real trainer; unscaled FP16
                # backward was an invalid proxy and showed a large discrepancy.
                (loss*1024.).backward();loss_value+=float(loss.detach())
            torch.cuda.synchronize();times.append(time.perf_counter()-started)
        grads.append(torch.cat([p.grad.detach().float().flatten() for p in params]).cpu()/1024.)
        outputs.append({"microbatch":batch,"effective_batch":4,"seconds_median_last2":sum(times[-2:])/2,
            "loss":loss_value,"peak_GiB":torch.cuda.max_memory_allocated()/2**30})
    rel=float((grads[0]-grads[1]).norm()/grads[0].norm())
    result={"scope":"disposable forward/backward equivalence and timing; zero optimizer updates",
        "variants":outputs,"gradient_relative_L2":rel,
        "loss_relative_error":abs(outputs[0]["loss"]-outputs[1]["loss"])/outputs[0]["loss"],
        "batch4_is_finite":bool(torch.isfinite(grads[1]).all()),
        "gradient_scale":1024.0,
        "prior_unscaled_backward_diagnostic":{"gradient_relative_L2":0.3286754786968231,
            "loss_relative_error":1.5862601012564923e-05,
            "status":"FAILED_EQUIVALENCE; did not use actual training gradient scale"},
        "status":"PASS" if rel<.03 else "FAILED_EQUIVALENCE"}
    write_json(RUN/"batch_benchmark.json",result)
    assert result["batch4_is_finite"] and rel<.03,result
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
