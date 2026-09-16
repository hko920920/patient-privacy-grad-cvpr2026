"""Reuse the exact SD2.1 preprocessing and public DINO encoder; no target training."""
import gc, json, time
import numpy as np
from .common import *
from .models import setup, snapshot, torch
from dp_protocol.calibrate_xray_public_clip_norms import verify_snapshot
from data_pipeline import nih_cxr14_model_input as mi

def main():
    setup();verify_inputs()
    output=RUN/"cache"
    if (output/"summary.json").exists():raise RuntimeError("cache already complete")
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    snap=snapshot();model_hashes=verify_snapshot(snap)
    source_records={r.image_id:r for r in mi.read_manifest(SOURCE,partitions={"public_development"})}
    rows=read_csv(RUN/"cohort"/"evaluation_images.csv")+read_csv(RUN/"cohort"/"auxiliary_images.csv")
    rows=sorted(rows,key=lambda r:r["image_id"])
    prompts={r["image_id"]:mi.prompt_for_record(source_records[r["image_id"]]) for r in rows}
    from transformers import CLIPTokenizer,CLIPTextModel
    tokenizer=CLIPTokenizer.from_pretrained(snap/"tokenizer",local_files_only=True)
    text_model=CLIPTextModel.from_pretrained(snap/"text_encoder",torch_dtype=torch.float16,
        variant="fp16",use_safetensors=True,local_files_only=True).eval().to("cuda")
    all_prompts=sorted(set(prompts.values())|{PROMPT,""})
    hidden={}
    with torch.inference_mode():
        for i in range(0,len(all_prompts),8):
            selected=all_prompts[i:i+8]
            tokens=tokenizer(selected,padding="max_length",max_length=tokenizer.model_max_length,
                truncation=True,return_tensors="pt")
            h=text_model(tokens.input_ids.to("cuda"))[0].cpu()
            for prompt,value in zip(selected,h):hidden[prompt]=value.unsqueeze(0).contiguous()
    tokens=tokenizer(PROMPT,padding="max_length",max_length=tokenizer.model_max_length,
        truncation=True,return_tensors="pt")
    token_mask=tokens.attention_mask.float()
    del text_model;gc.collect();torch.cuda.empty_cache()
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained(snap/"vae",torch_dtype=torch.float16,
        variant="fp16",use_safetensors=True,local_files_only=True).eval().to("cuda")
    latents={}
    with torch.inference_mode():
        for i in range(0,len(rows),4):
            selected=rows[i:i+4]
            batch=torch.stack([mi.pil_to_normalized_tensor(mi.preprocess_path(IMAGES/r["image_filename"],256)[0])
                for r in selected]).to("cuda",torch.float16)
            z=vae.encode(batch).latent_dist.mode()*float(vae.config.scaling_factor)
            assert torch.isfinite(z).all()
            for r,value in zip(selected,z.cpu()):latents[r["image_id"]]=value.unsqueeze(0).contiguous()
            if i%128==0:print(json.dumps({"phase":"VAE_cache","done":i,"total":len(rows)}),flush=True)
    del vae;gc.collect();torch.cuda.empty_cache()
    from patient_set_diffusion.run_nih_cxr14_patient_set_premise import load_dino,preprocess_image,DINO_SHA256
    weights=Path.home()/".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
    assert digest(weights).upper()==DINO_SHA256
    dino,dino_meta=load_dino(weights)
    feature_rows=[r for r in rows if r["eval_role"]!="background"]
    features={}
    with torch.inference_mode():
        for i in range(0,len(feature_rows),4):
            selected=feature_rows[i:i+4]
            batch=torch.stack([preprocess_image(IMAGES/r["image_filename"]) for r in selected]).to("cuda")
            with torch.autocast("cuda",dtype=torch.float16):
                f=dino(batch)
            f=torch.nn.functional.normalize(f.float(),dim=1).cpu()
            for r,value in zip(selected,f):features[r["image_id"]]=value.contiguous()
            if i%128==0:print(json.dumps({"phase":"DINO_cache","done":i,"total":len(feature_rows)}),flush=True)
    del dino;gc.collect();torch.cuda.empty_cache()
    refs=[r for r in feature_rows if r["eval_role"]=="reference"]
    ref_features=torch.stack([features[r["image_id"]] for r in refs])
    matches={}
    for row in rows:
        if row["eval_role"] in {"reference","quality","background"}:continue
        scores=ref_features@features[row["image_id"]]
        chosen=[];used=set()
        for ix in torch.argsort(scores,descending=True,stable=True).tolist():
            ref=refs[ix]
            if ref["patient_id"] in used:continue
            chosen.append(ref["image_id"]);used.add(ref["patient_id"])
            if len(chosen)==2:break
        matches[row["image_id"]]=chosen
    # These are public proxies stored locally; do not use this cache as a medical release artifact.
    payload={"latents":latents,"hidden":hidden,"train_prompts":prompts,
             "audit_prompt":PROMPT,"token_mask":token_mask,
             "encoder_features":features,"reference_matches":matches}
    torch.save(payload,output/"cache.pt")
    summary={"status":"PASS","cache_sha256":digest(output/"cache.pt"),
        "cohort_lock_sha256":digest(RUN/"cohort"/"lock.json"),"images":len(latents),
        "encoder_images":len(features),"reference_matches":len(matches),
        "model_revision":MODEL_REVISION,"model_hashes":model_hashes,
        "vae_policy":"posterior mode; exact existing grayscale full-field P256 preprocessing",
        "training_prompt_policy":mi.PROMPT_POLICY,"audit_prompt":PROMPT,
        "dino":dino_meta,"dino_sha256":DINO_SHA256,"seconds":time.perf_counter()-started,
        "target_training_started":False}
    write_json(output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="model_hashes"}),flush=True)

if __name__=="__main__":main()

