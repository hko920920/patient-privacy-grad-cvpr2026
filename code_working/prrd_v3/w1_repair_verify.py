"""Bounded old-objective W1 repair verification; no new PRRD learner integration."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import setup,OUT,sha,save,source_hashes,require
from .datasets import manifests,PixelAccess,public_templates
from .encoders import Encoder,state_hash,pil_tensor
from .render import Renderer,augment,augment_parameters
from .resampling import resize
from .objectives import matching,regularization,one_pass,two_pass
from .profile import target_public
from .w1_boundary_debug import error,grads,grad_error,tensor_hash


def numpy_matching(z,target,arm):
    z=np.asarray(z,dtype=np.float64); q=len(z)//4; relation=arm in ("C","D","R_joint")
    ni=np.arange(q) if relation else np.arange(2*q)
    pi=np.arange(2*q,3*q) if relation else np.arange(2*q,4*q)
    g=np.zeros_like(z);value=0.
    for c,ids in enumerate((ni,pi)):
        x=z[ids];m=x.mean(0);a=x.T@x/len(x)
        dm=m-target["m"][c]; da=a-target["A"][c]
        value+=.25*((dm*dm).sum()+(da*da).sum())
        g[ids]+=(.5*dm+.5*x@(da+da.T))/len(x)
    if relation:
        ip=np.arange(3*q,4*q);ineg=np.arange(q,2*q)
        x=(z[ip]-z[ineg])/2 if arm!="R_joint" else np.concatenate([z[ip],z[ineg]],axis=1)/np.sqrt(2)
        mk,ak=("delta","C") if arm!="R_joint" else ("u","U")
        dm=x.mean(0)-target[mk]; da=x.T@x/len(x)-target[ak]
        value+=.5*((dm*dm).sum()+(da*da).sum())
        gx=(dm+x@(da+da.T))/len(x)
        if arm!="R_joint":g[ip]+=gx/2;g[ineg]-=gx/2
        else:
            d=z.shape[1];g[ip]+=gx[:,:d]/np.sqrt(2);g[ineg]+=gx[:,d:]/np.sqrt(2)
    return value,g


def optimizer_step(model,initial):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0,betas=(.9,.999),eps=1e-8,foreach=False)
    gs=grads(model)
    expected=[x.double()-.01*g.double()/(g.double().abs()+1e-8) for x,g in zip(initial,gs)]
    optimizer.step()
    after=[p.detach().cpu().clone() for p in model.parameters()]
    return after,grad_error(after,expected)


def renderer_trace(model,micro):
    out={f"resize_{i}":[] for i in range(model.active)}
    out.update({f"sum_{i}":[] for i in range(model.active)})
    out["base"]=[];out["pixels"]=[]
    with torch.no_grad():
        for start in range(0,model.n,micro):
            ids=list(range(start,min(start+micro,model.n)))
            value=model.base[ids];out["base"].append(value.cpu())
            for i,p in enumerate(model.residuals[:model.active]):
                # The original failing case is A, without paired parameter mixing.
                level=resize(p[ids],(224,224),antialias=False)
                out[f"resize_{i}"].append(level.cpu())
                value=value+level;out[f"sum_{i}"].append(value.cpu())
            out["pixels"].append(value.sigmoid().cpu())
    return {k:torch.cat(v) for k,v in out.items()}


def pixel_vjp(encoder,x,upstream,micro,replay):
    leaf=x.cuda().detach().requires_grad_(True)
    if replay:
        for i in range(0,len(x),micro):
            (encoder(leaf[i:i+micro])*upstream[i:i+micro].cuda()).sum().backward()
    else:
        z=torch.cat([encoder(t) for t in leaf.split(micro)])
        (z*upstream.cuda()).sum().backward()
    return leaf.grad.detach().cpu()


def renderer_vjp(model,gx,ga,seed,micro,replay):
    model.zero_grad(set_to_none=True)
    params=augment_parameters(model.n,seed,500,model.relation,model.base.device)
    accumulated=0.
    for start in range(0,model.n,micro):
        ids=list(range(start,min(start+micro,model.n)));x=model(ids);xa=augment(x,params,ids)
        v=(x*gx[ids].cuda()).sum()+(xa*ga[ids].cuda()).sum()
        v=v+regularization(x,model.templates[ids])*len(ids)/model.n
        if replay:v.backward()
        else:accumulated=accumulated+v
    if not replay:accumulated.backward()
    return grads(model)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--original",type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),"Verification output already exists")
    args.output.mkdir(parents=True);setup();tick=time.perf_counter()
    groups=manifests();access=PixelAccess(groups,("P",)).install()
    encoder=Encoder("source").use_projection(OUT/"source_P_projection.npz")
    before=state_hash(encoder.model);target,joint=target_public(encoder,groups["P"])
    counts={"forward_calls":0,"forward_images":0,"backward_calls":0,"backward_images":0,"probe_optimizer_updates":0}
    def model_hook(module,inputs,output):
        n=len(inputs[0]);counts["forward_calls"]+=1;counts["forward_images"]+=n
        f=output.projected_global_embedding
        if f.requires_grad:
            def back(g):
                counts["backward_calls"]+=1;counts["backward_images"]+=n
                return g
            f.register_hook(back)
    handle=encoder.model.register_forward_hook(model_hook)
    rows=[];fixtures=[];moment_errors=[];optimizer_errors=[];boundary={}
    for seed in (101,202):
        chosen=public_templates(groups["P"],seed);selected=[chosen[i] for i in (0,32,64,96)]
        templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in selected]).cuda()
        fixture={"seed":seed,"images":[{k:r[k] for k in ("image_id","patient_id","sha256")} for r in selected],"templates_sha256":tensor_hash(templates)}
        if seed==101:
            saved=torch.load(args.original/"fixture.pt",map_location="cpu",weights_only=False)
            require(torch.equal(templates.cpu(),saved["renderer"]["templates"]),"Original public pixels changed")
        fixtures.append(fixture)
        for arm in ("A","C","R_joint"):
            t=joint if arm=="R_joint" else target
            model=Renderer(templates,arm,seed);model.set_step(500)
            initial_state=copy.deepcopy(model.state_dict())
            if seed==101 and arm=="A":
                require(all(torch.equal(v.cpu(),saved["renderer"][k]) for k,v in initial_state.items()),"Original renderer state changed")
                full=renderer_trace(model,4);single=renderer_trace(model,1)
                boundary["renderer_first_difference"]={k:error(single[k],full[k]) for k in full}
                bt=torch.load(args.original/"boundary_tensors.pt",map_location="cpu",weights_only=False)["first"]
                gx1=pixel_vjp(encoder,bt["x"],bt["gz"],1,False)
                gx2=pixel_vjp(encoder,bt["x"],bt["gz"],1,True)
                ga1=pixel_vjp(encoder,bt["xa"],bt["gza"],1,False)
                ga2=pixel_vjp(encoder,bt["xa"],bt["gza"],1,True)
                boundary["fixed_upstream_pixel_clean"]=error(gx2,gx1)
                boundary["fixed_upstream_pixel_augmented"]=error(ga2,ga1)
                pg1=renderer_vjp(model,gx1,ga1,seed,1,False)
                pg2=renderer_vjp(model,gx1,ga1,seed,1,True)
                boundary["fixed_pixel_upstream_renderer"]=grad_error(pg2,pg1)
                require(torch.equal(gx1,gx2) and torch.equal(ga1,ga2),"Fixed-input encoder VJP changed")
            for micro in (1,2,4):
                model.load_state_dict(initial_state);model.zero_grad(set_to_none=True)
                loss=one_pass(model,encoder,t,500,seed,microbatch=micro);loss.backward()
                ref_grads=grads(model);loss_value=float(loss.detach())
                initial_params=[p.detach().cpu().clone() for p in model.parameters()]
                ref_update,ref_math=optimizer_step(model,initial_params);counts["probe_optimizer_updates"]+=1
                model.load_state_dict(initial_state)
                trace=two_pass(model,encoder,t,500,seed,micro)
                replay_grads=grads(model)
                replay_update,replay_math=optimizer_step(model,initial_params);counts["probe_optimizer_updates"]+=1
                gerr=grad_error(replay_grads,ref_grads);uerr=grad_error(replay_update,ref_update)
                # Existing W1 thresholds are unchanged; update errors are also reported.
                ok=gerr["max_abs"]<=2e-7+5e-4*gerr["reference_peak"]
                lok=abs(trace["loss"]-loss_value)<=2e-6
                require(ok and lok,"Existing W1 gradient/loss tolerance failed")
                # Independent float64 Adam first-step formula (arithmetic check).
                require(max(ref_math["max_abs"],replay_math["max_abs"])<2e-7,"Adam implementation/formula mismatch")
                row={"seed":seed,"arm":arm,"microbatch":micro,"gradient":gerr,
                     "loss_abs":abs(trace["loss"]-loss_value),"parameter_update":uerr,
                     "reference_Adam_formula":ref_math,"replay_Adam_formula":replay_math,
                     "gradient_pass":ok,"loss_pass":lok}
                rows.append(row);optimizer_errors.append(uerr["max_abs"]);print(json.dumps(row),flush=True)
                del loss,ref_grads,replay_grads,ref_update,replay_update
            model.load_state_dict(initial_state)
            with torch.no_grad():f=torch.cat([encoder(model([i])) for i in range(4)])
            zd=f.detach().double().requires_grad_(True);mval=matching(zd,t,arm);gz,=torch.autograd.grad(mval,zd)
            nv,ng=numpy_matching(zd.detach().cpu().numpy(),t,arm)
            er={"seed":seed,"arm":arm,"FP64_loss_abs":abs(nv-float(mval.detach())),
                "FP64_gradient_max_abs":float(np.max(abs(ng-gz.detach().cpu().numpy())))}
            require(er["FP64_loss_abs"]<1e-12 and er["FP64_gradient_max_abs"]<1e-12,"Independent moment gradient mismatch")
            moment_errors.append(er)
            del model,initial_state;torch.cuda.empty_cache()
        torch.save({"templates":templates.cpu(),"seed":seed},args.output/f"public_fixture_{seed}.pt")
        del templates
    handle.remove()
    require(state_hash(encoder.model)==before,"Frozen source parameters/buffers changed")
    result={"status":"PASS_EXISTING_OBJECTIVE_ACTUAL_MODEL_GRADIENT_REPAIR_ONLY",
            "rows":rows,"fixtures":fixtures,"boundary_errors":boundary,"independent_moments":moment_errors,
            "max_gradient_abs":max(r["gradient"]["max_abs"] for r in rows),
            "max_gradient_relative_l2":max(r["gradient"]["relative_l2"] for r in rows),
            "max_loss_abs":max(r["loss_abs"] for r in rows),
            "max_parameter_update_abs":max(optimizer_errors),
            "source_parameters_and_buffers_unchanged":True,"source_state_sha256":before,
            "counts":counts,"access":access.report(),"seconds":time.perf_counter()-tick,
            "source_sha256":source_hashes(),"original_reproducer_sha256":sha(args.original/"w1_reproducer.json"),
            "old_failure_preserved":True,"gradient_tolerance_unchanged":True,"objective_unchanged":True,
            "new_guard_or_functional_loss_integrated":False,"full_W1_runtime_passed":False,
            "full128_profile_performed":False,"new_W2_banks":0,"new_DP_releases":0,
            "expert_reserved_access":False}
    save(args.output/"w1_repair_verification.json",result)
    print(json.dumps({k:result[k] for k in ("status","max_gradient_abs","max_gradient_relative_l2",
                                         "max_parameter_update_abs","counts","access","seconds")}),flush=True)


if __name__=="__main__":main()
