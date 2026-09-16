"""Paired loss on 32 withheld quality patients, distinct from MIA selection/calibration/test."""
import argparse,json,time,gc
from collections import defaultdict
import numpy as np
from .common import *
from .models import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",choices=["model_1","model_2"],required=True)
    args=parser.parse_args();setup();verify_inputs();cache=load_cache()
    out=RUN/"quality"/"training_coverage_v2"/args.model
    if (out/"loss_report.json").exists():raise RuntimeError("completed quality losses exist")
    out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read_csv(RUN/"cohort/auxiliary_images.csv") if r["eval_role"]=="quality"]
    checkpoint=RUN/"training_coverage_v2"/args.model/"step_1000.pt"
    unet=load_unet(checkpoint,training=False);sched=scheduler();outputs=[]
    started=time.perf_counter()
    for start in range(0,len(rows),4):
        batch=rows[start:start+4];values={"base":np.zeros(len(batch)),"target":np.zeros(len(batch))}
        z=torch.cat([cache["latents"][r["image_id"]].clone() for r in batch]).to("cuda",torch.float16)
        h=torch.cat([cache["hidden"][cache["train_prompts"][r["image_id"]]].clone() for r in batch]).to("cuda",torch.float16)
        for tvalue in [50,250,500,750,950]:
            noises=[]
            for r in batch:
                gen=torch.Generator(device="cuda").manual_seed(seed("quality-loss",r["image_id"],tvalue))
                noises.append(torch.randn((1,4,32,32),generator=gen,device="cuda",dtype=torch.float16))
            noise=torch.cat(noises);t=torch.full((len(batch),),tvalue,device="cuda",dtype=torch.long)
            noisy,target=diffusion_input(z,t,noise,sched)
            for arm in ["base","target"]:
                if arm=="base":unet.disable_adapters()
                else:unet.enable_adapters()
                unet.requires_grad_(False)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    pred=unet(noisy,t,h).sample
                    loss=(pred.float()-target.float()).square().mean(dim=(1,2,3)).cpu().numpy()
                assert np.isfinite(loss).all()
                values[arm]+=loss/5
        for i,r in enumerate(batch):
            outputs.append({"patient_id":r["patient_id"],"image_id":r["image_id"],
                "base":float(values["base"][i]),"target":float(values["target"][i])})
    grouped=defaultdict(list)
    for r in outputs:grouped[r["patient_id"]].append(r)
    pairs=np.array([[np.mean([r[k] for r in rs]) for k in ["base","target"]] for rs in grouped.values()])
    rng=np.random.default_rng(260914);ratios=[]
    for _ in range(2000):
        draw=pairs[rng.integers(len(pairs),size=len(pairs))].mean(0)
        ratios.append(draw[1]/draw[0])
    means=pairs.mean(0)
    report={"status":"COMPUTED","model":args.model,"patients":len(pairs),"images":len(rows),
        "base_mean_loss":float(means[0]),"target_mean_loss":float(means[1]),
        "target_over_base":float(means[1]/means[0]),
        "patient_bootstrap_95_ratio_interval":np.quantile(ratios,[.025,.975]).tolist(),
        "patient_win_fraction":float((pairs[:,1]<pairs[:,0]).mean()),
        "seconds":time.perf_counter()-started,"checkpoint_sha256":digest(checkpoint),
        "scope":"held-out diffusion-loss domain diagnostic; not clinical utility or MIA performance",
        "prompt_policy":"existing weak-label prompt; auditor prompt remains generic"}
    write_json(out/"loss_rows.json",outputs);write_json(out/"loss_report.json",report)
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()

