"""Fixed public-prompt/seed visual domain diagnostic, not clinical validation."""
import argparse,json,gc,time
from PIL import Image,ImageDraw
import numpy as np
from .common import *
from .models import *

PROMPTS=[PROMPT,"a frontal posteroanterior chest radiograph with no labeled finding",
    "a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion",
    "a frontal posteroanterior chest radiograph with radiographic findings of cardiomegaly"]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--step",type=int,choices=[50,250,1000],default=1000)
    parser.add_argument("--training-dir",choices=["training","training_coverage_v2"],default="training_coverage_v2")
    parser.add_argument("--models",choices=["first","both"],default="both")
    args=parser.parse_args()
    setup();verify_inputs()
    from diffusers import StableDiffusionPipeline,DDIMScheduler
    out=RUN/"quality"/args.training_dir/f"step_{args.step:04d}_{args.models}"
    if (out/"report.json").exists():raise RuntimeError("quality result exists")
    out.mkdir(parents=True,exist_ok=True)
    selected_models=["base","model_1"] if args.models=="first" else ["base","model_1","model_2"]
    canvas=Image.new("RGB",(256*4,280*len(selected_models)),"white");draw=ImageDraw.Draw(canvas)
    rows=[]
    for y,model in enumerate(selected_models):
        checkpoint=None if model=="base" else RUN/args.training_dir/model/f"step_{args.step:04d}.pt"
        unet=load_unet(checkpoint,training=False)
        pipe=StableDiffusionPipeline.from_pretrained(snapshot(),unet=unet,torch_dtype=torch.float16,
            variant="fp16",use_safetensors=True,safety_checker=None,feature_extractor=None,
            requires_safety_checker=False,local_files_only=True).to("cuda")
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        for i,prompt in enumerate(PROMPTS):
            started=time.perf_counter()
            gen=torch.Generator(device="cuda").manual_seed(seed("quality-generation",i))
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                arr=pipe(prompt,height=256,width=256,num_inference_steps=30,guidance_scale=7.5,
                    generator=gen,output_type="np").images[0]
            assert np.isfinite(arr).all()
            image=Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8))
            path=out/f"{model}_{i}.png";image.save(path)
            canvas.paste(image,(i*256,y*280+24))
            draw.text((i*256+4,y*280+4),f"{model} / prompt {i}",fill="black")
            rows.append({"model":model,"prompt":prompt,"seed":seed("quality-generation",i),
                "path":path.name,"sha256":digest(path),"seconds":time.perf_counter()-started})
            print(json.dumps({"phase":"generation","model":model,"image":i}),flush=True)
        del pipe,unet;gc.collect();torch.cuda.empty_cache()
    canvas.save(out/"comparison_grid.png")
    write_json(out/"report.json",{"status":"GENERATED_PENDING_VISUAL_REVIEW","step":args.step,"training_dir":args.training_dir,
        "images":rows,"prompt_selection":"predeclared four prompts","clinical_validation":False,
        "model_revision":MODEL_REVISION,
        "checkpoint_hashes":{model:digest(RUN/args.training_dir/model/f"step_{args.step:04d}.pt")
            for model in selected_models if model!="base"},
        "scheduler":"DDIM from the pinned base scheduler, matching the existing generation loader",
        "loader_note":"Initial load stopped before generation because optional image processor is absent locally; existing k5_generation loader flags restored."})

if __name__=="__main__":main()
