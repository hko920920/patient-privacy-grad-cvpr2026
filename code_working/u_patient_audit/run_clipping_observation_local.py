"""Bounded public-triplet clipping/Adam applicability check, not DP/MIA efficacy."""
import argparse, copy, gc, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
import torch
from .common import ROOT, RUN, PROMPT, MODEL_ID, MODEL_REVISION, digest, write_json, read_csv
from .models import setup, snapshot, load_unet, scheduler, diffusion_input, load_cache
from data_pipeline import nih_cxr14_model_input as mi

TAG = "clipping_observation_local_v1"
C = 0.28448700606156724
SALT = "cvpr-clipping-observation-local-v1-20260915"
T = 500

def fixed_seed(image_id):
    return int(hashlib.sha256((SALT+"|"+image_id).encode()).hexdigest()[:15],16)

def norm(x):
    return float(torch.linalg.vector_norm(x.double()))

def dot(x,y):
    return float(torch.dot(x.double(),y.double()))

def cosine(x,y):
    return dot(x,y)/max(norm(x)*norm(y),1e-300)

def clip(x,c):
    a=min(1.,c/norm(x)) if norm(x)>0 else 1.
    return x*a,a

def flatten(params):
    return torch.cat([p.detach().reshape(-1).cpu() for p in params])

def restore(params, vec):
    at=0
    with torch.no_grad():
        for p in params:
            n=p.numel()
            p.copy_(vec[at:at+n].reshape(p.shape).to(p.device))
            at+=n
    assert at==vec.numel()

def optimizer_step(params, opt, source, theta, gradient):
    restore(params,theta)
    opt.load_state_dict(copy.deepcopy(source))
    at=0
    for p in params:
        n=p.numel()
        p.grad=gradient[at:at+n].reshape(p.shape).to(p.device).clone()
        at+=n
    opt.step()
    result=flatten(params)
    opt.zero_grad(set_to_none=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",required=True)
    args=parser.parse_args()
    fixture_path=Path(args.fixture)
    fixture=json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    inventory_path=fixture_path.with_name("optimizer_state_inventory.json")
    inventory=json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    out=RUN/"protection_observation_20260915"/TAG
    if out.exists(): raise FileExistsError(out)
    started=time.perf_counter()
    groups=defaultdict(list)
    for row in fixture["records"]:
        assert digest(row["path"])==row["sha256"]
        groups[row["patient_id"]].append(row)
    assert len(groups)==8 and all(len(v)==3 for v in groups.values())
    source_rows={r["image_id"]:r for r in read_csv(fixture["source"])}
    assert digest(fixture["source"])==fixture["source_sha256"]
    for row in fixture["records"]:
        assert source_rows[row["image_id"]]["partition"]=="public_development"
        assert source_rows[row["image_id"]]["patient_id"]==row["patient_id"]
    for binding in fixture["excluded_manifest_bindings"]:
        assert digest(binding["path"])==binding["sha256"]
        assert not set(groups)&{r["patient_id"] for r in read_csv(binding["path"])}
    for binding in inventory["checkpoints"]:
        assert digest(binding["path"])==binding["sha256"]
    code_files=[Path(__file__),Path(__file__).with_name("models.py"),Path(__file__).with_name("common.py"),
                ROOT/"data_pipeline/nih_cxr14_model_input.py"]
    protocol=dict(schema="clipping-observation-local/v1", scope="local noiseless clipping/Adam response; not privacy performance",
        fixture_sha256=digest(fixture_path), optimizer_inventory_sha256=digest(inventory_path),
        patient_order=list(groups), images=24, checkpoint_steps=[250,1000],
        model_id=MODEL_ID, model_revision=MODEL_REVISION, clip_norm=C, timestep=T,
        noise_policy="independent image hash seed; same fixed FP32 epsilon across E/U role, branches and states",
        seed_salt=SALT, audit_and_update_prompt=PROMPT,
        A="mean(clip_C(g_E1),clip_C(g_E2))",B="clip_C(mean(g_E1,g_E2))",
        outer_denominator=1., gradient_noise_sigma=0., other_patients_in_local_batch=0,
        clipping_inactive_control="Cinf with exact A=B; both optimizer branches checked with same original state",
        primary="score_A-score_B versus -gradient_query dot (theta_A-theta_B)",
        precision="promote existing FP16 base values to FP32, FP32 LoRA/Adam/noisy inputs; no autocast; saved prediction MSE reduced in FP64",
        historical_AMP_replay=False, optimizer_state="original learned moments, step and param_groups reset independently per branch and patient",
        optimizer="AdamW foreach=False fused=False; source lr/betas/eps/weight_decay",
        E_U_meaning="E2 used only in this local update; U1 unused. All 24 excluded from pilot train/aux/evaluation.",
        primary_forward=144, primary_backward=48,
        numerical_controls="first triplet per state: 3 baseline replay F, 2 Emean F/B, 6 half-displacement F",
        expected_forward=166, expected_backward=52,
        extra_VAE_images=24, expected_parameter_optimizer_steps=36,
        outcome_policy="report all fixed patients; no clipping/timestep/seed selection using observed result; no MIA metrics",
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_files},
        input_sha256={str(fixture_path):digest(fixture_path),str(inventory_path):digest(inventory_path)},
        checkpoint_bindings=inventory["checkpoints"])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"protocol.json",protocol)
    setup()
    base_cache=load_cache()
    hidden=base_cache["hidden"][PROMPT].clone().float()
    del base_cache
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained(snapshot()/"vae",torch_dtype=torch.float16,
        variant="fp16",use_safetensors=True,local_files_only=True).eval().to("cuda")
    latents={}
    with torch.no_grad():
        for row in fixture["records"]:
            image=mi.pil_to_normalized_tensor(mi.preprocess_path(Path(row["path"]),256)[0])
            batch=image.unsqueeze(0).to("cuda",torch.float16)
            z=vae.encode(batch).latent_dist.mode()*float(vae.config.scaling_factor)
            assert torch.isfinite(z).all()
            latents[row["image_id"]]=z.cpu().clone()
    del vae
    gc.collect();torch.cuda.empty_cache()
    sched=scheduler()
    queries={}
    for row in fixture["records"]:
        z=latents[row["image_id"]].float()
        epsilon=torch.randn(z.shape,generator=torch.Generator().manual_seed(fixed_seed(row["image_id"])),dtype=torch.float32)
        noisy,target=diffusion_input(z,torch.tensor([T]),epsilon,sched)
        queries[row["image_id"]]=dict(noisy=noisy,target=target)
    torch.save(dict(latents=latents,hidden=hidden,queries=queries),out/"inputs.pt")
    write_json(out/"input_report.json",dict(inputs_sha256=digest(out/"inputs.pt"),seconds=time.perf_counter()-started))
    print(json.dumps(dict(phase="inputs_prepared",seconds=time.perf_counter()-started)),flush=True)
    forward=0;backward=0;results=[]
    model_seconds_start=time.perf_counter()
    for binding in inventory["checkpoints"]:
        step=binding["step"]
        saved=torch.load(binding["path"],map_location="cpu",weights_only=True)
        unet=load_unet(Path(binding["path"]),training=True).float()
        named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
        names=[n for n,p in named];params=[p for n,p in named]
        assert all(p.dtype==torch.float32 for p in params)
        assert all(torch.equal(p.detach().cpu(),saved["adapter"][n]) for n,p in named)
        theta=flatten(params)
        assert sum(p.numel() for p in params)==1659904
        original_opt=copy.deepcopy(saved["optimizer"])
        assert all(float(v["step"])==step for v in original_opt["state"].values())
        opt=torch.optim.AdamW(params,lr=1e-4,foreach=False,fused=False)
        hh=hidden.to("cuda")
        tt=torch.tensor([T],device="cuda")
        torch.save(dict(theta=theta,parameter_names=names,parameter_shapes=[list(p.shape) for p in params],
                        optimizer=original_opt),out/f"state_{step}.pt")
        def evaluate(row,need_grad):
            nonlocal forward,backward
            if need_grad: opt.zero_grad(set_to_none=True)
            q=queries[row["image_id"]]
            xx=q["noisy"].to("cuda"); yy=q["target"].to("cuda")
            with torch.set_grad_enabled(need_grad):
                pred=unet(xx,tt,hh).sample
                loss=(pred-yy).square().mean()
                assert torch.isfinite(loss)
                if need_grad:
                    loss.backward();backward+=1
            forward+=1
            g=None
            if need_grad:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                g=torch.cat([p.grad.detach().reshape(-1).cpu() for p in params])
                opt.zero_grad(set_to_none=True)
            pp=pred.detach().cpu()
            return pp,g,float((pp.double()-q["target"].double()).square().mean())
        for index,(patient,records) in enumerate(groups.items()):
            patient_started=time.perf_counter()
            records=sorted(records,key=lambda r:r["role"])
            assert [r["role"] for r in records]==["E_local_1","E_local_2","U_local"]
            restore(params,theta)
            base=[];grads=[];base_losses=[]
            for row in records:
                pp,gg,ll=evaluate(row,True)
                base.append(pp);grads.append(gg);base_losses.append(ll)
            g1,g2,gu=grads
            gbar=(g1+g2)/2
            c1,a1=clip(g1,C);c2,a2=clip(g2,C)
            va=(c1+c2)/2
            vb,b=clip(gbar,C)
            cterm=((a1-(a1+a2)/2)*(g1-gbar)+(a2-(a1+a2)/2)*(g2-gbar))/2
            cperp=cterm-gbar*(dot(cterm,gbar)/max(dot(gbar,gbar),1e-300))
            algebra_error=norm(va-((a1+a2)/2*gbar+cterm))
            assert algebra_error<1e-6
            ta=optimizer_step(params,opt,original_opt,theta,va)
            assert all(float(v["step"])==step+1 for v in opt.state.values())
            pa=[];la=[]
            for row in records:
                pp,_,ll=evaluate(row,False);pa.append(pp);la.append(ll)
            tb=optimizer_step(params,opt,original_opt,theta,vb)
            pb=[];lb=[]
            for row in records:
                pp,_,ll=evaluate(row,False);pb.append(pp);lb.append(ll)
            delta_a=ta-theta;delta_b=tb-theta;delta_ab=ta-tb
            actual=[float(((pb[i].double()-pa[i].double())*
                           (pb[i].double()+pa[i].double()-2*queries[row["image_id"]]["target"].double())).mean())
                    for i,row in enumerate(records)]
            predicted=[-dot(g,delta_ab) for g in grads]
            controls={}
            if index==0:
                restore(params,theta)
                replay=[evaluate(row,False)[0] for row in records]
                controls["baseline_replay_exact"]=[torch.equal(x,y) for x,y in zip(base,replay)]
                assert all(controls["baseline_replay_exact"])
                # Same two fixed objectives, independent backward of their arithmetic mean.
                opt.zero_grad(set_to_none=True)
                for row in records[:2]:
                    q=queries[row["image_id"]]
                    pred=unet(q["noisy"].to("cuda"),tt,hh).sample
                    loss=(pred-q["target"].to("cuda")).square().mean()/2
                    loss.backward();forward+=1;backward+=1
                gm=torch.cat([p.grad.detach().reshape(-1).cpu() for p in params])
                opt.zero_grad(set_to_none=True)
                controls["mean_gradient_max_abs"]=float((gm-gbar).abs().max())
                controls["mean_gradient_relative_l2"]=norm(gm-gbar)/max(norm(gbar),1e-300)
                assert controls["mean_gradient_relative_l2"]<2e-5
                vna=(clip(g1,float("inf"))[0]+clip(g2,float("inf"))[0])/2
                vnb=clip(gbar,float("inf"))[0]
                assert torch.equal(vna,vnb)
                tn_a=optimizer_step(params,opt,original_opt,theta,vna)
                tn_b=optimizer_step(params,opt,original_opt,theta,vnb)
                controls["inactive_gradient_exact"]=torch.equal(vna,vnb)
                controls["inactive_optimizer_exact"]=torch.equal(tn_a,tn_b)
                assert controls["inactive_optimizer_exact"]
                half_preds={}
                for branch,delta in (("A",delta_a),("B",delta_b)):
                    restore(params,theta+delta*.5)
                    half_preds[branch]=[evaluate(row,False)[0] for row in records]
                half_actual=[float(((half_preds["B"][i].double()-half_preds["A"][i].double())*
                           (half_preds["B"][i].double()+half_preds["A"][i].double()-2*queries[row["image_id"]]["target"].double())).mean())
                    for i,row in enumerate(records)]
                controls["half_displacement_score_A_minus_B"]=half_actual
                controls["half_displacement_note"]="interpolation check, not a second Adam update or refit"
                controls["half_predictions"]=half_preds
                controls["mean_gradient"]=gm
            data=dict(patient_id=patient,step=step,records=records,gradients=torch.stack(grads),
                      delta_A=delta_a,delta_B=delta_b,prediction_base=base,prediction_A=pa,prediction_B=pb,controls=controls)
            path=out/f"step_{step}_patient_{patient}.pt"
            torch.save(data,path)
            public_controls={k:v for k,v in controls.items() if k not in ("half_predictions","mean_gradient")}
            result=dict(patient_id=patient,step=step,gradient_norms=[norm(g) for g in grads],
                mean_gradient_norm=norm(gbar),clip_factors=[a1,a2],mean_clip_factor=b,
                both_gradients_unclipped=a1==a2==1.,gradient_cosine_E1_E2=cosine(g1,g2),
                gradient_cosine_Emean_U=cosine(gbar,gu),clipped_A_B_cosine=cosine(va,vb),
                clipped_A_B_norm_difference=norm(va-vb),c_perpendicular_norm=norm(cperp),
                c_algebra_l2_error=algebra_error,delta_A_norm=norm(delta_a),delta_B_norm=norm(delta_b),
                delta_A_B_norm=norm(delta_ab),delta_A_B_cosine=cosine(delta_a,delta_b),
                base_losses=base_losses,loss_A=la,loss_B=lb,
                score_A_minus_B=actual,linear_prediction_A_minus_B=predicted,
                raw_SGD_gradient_contrast=[dot(g,va-vb) for g in grads],
                Emean_score_A_minus_B=sum(actual[:2])/2,U_score_A_minus_B=actual[2],
                Emean_linear_prediction=sum(predicted[:2])/2,U_linear_prediction=predicted[2],
                E_U_raw_sign_opposition=(sum(actual[:2])/2)*actual[2]<0,
                E_U_linear_sign_opposition=(sum(predicted[:2])/2)*predicted[2]<0,
                controls=public_controls,raw_file=path.name,raw_sha256=digest(path),
                seconds=time.perf_counter()-patient_started)
            results.append(result)
            write_json(out/"partial_results.json",results)
            print(json.dumps({k:result[k] for k in ("patient_id","step","clip_factors",
                 "delta_A_B_norm","Emean_score_A_minus_B","U_score_A_minus_B","seconds")}),flush=True)
        del unet,opt,params,named,saved,hh
        gc.collect();torch.cuda.empty_cache()
    assert forward==166 and backward==52,(forward,backward)
    for binding in inventory["checkpoints"]:
        assert digest(binding["path"])==binding["sha256"]
    write_json(out/"results.json",results)
    report=dict(status="COMPLETED_LOCAL_MEASUREMENTS_PENDING_INDEPENDENT_VERIFICATION",
        patients=8,state_conditions=16,forward=forward,backward=backward,vae_images=24,
        elapsed_seconds=time.perf_counter()-started,model_measurement_seconds=time.perf_counter()-model_seconds_start,
        max_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
        results_sha256=digest(out/"results.json"),protocol_sha256=digest(out/"protocol.json"),
        inputs_sha256=digest(out/"inputs.pt"),source_checkpoints_unchanged=True,
        hypothesis_verdict="pending analysis",scope="noiseless FP32 local bridge; not DP efficacy, MIA or an exact historical AMP training replay")
    write_json(out/"report.json",report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()

