"""Reproduce the original W1 objective on fixed public-only fixtures."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import setup,OUT,sha,save,source_hashes,require
from .datasets import manifests,PixelAccess,public_templates
from .encoders import Encoder,state_hash,pil_tensor,preprocess
from .render import Renderer,augment,augment_parameters
from .objectives import matching,regularization,two_pass
from .profile import target_public


def error(a,b):
    a=a.detach().double().cpu().reshape(-1); b=b.detach().double().cpu().reshape(-1)
    d=a-b
    return {"max_abs":float(d.abs().max()),"relative_l2":float(d.norm()/b.norm().clamp_min(1e-30)),
            "reference_peak":float(b.abs().max()),"exact":bool(torch.equal(a,b))}


def tensor_hash(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def grads(model):
    return [p.grad.detach().cpu().clone() for p in model.parameters()]


def grad_error(a,b):
    return error(torch.cat([x.reshape(-1) for x in a]),torch.cat([x.reshape(-1) for x in b]))


def retained(renderer,encoder,target,step,seed,micro,full_renderer):
    renderer.zero_grad(set_to_none=True)
    n=renderer.n; ids=list(range(n))
    params=augment_parameters(n,seed,step,renderer.relation,renderer.base.device)
    xfull=renderer() if full_renderer else None
    afull=augment(xfull,params,ids) if full_renderer else None
    xs=[]; axs=[]; zs=[]; zas=[]; pres=[]; apres=[]; reg=0.
    # Retain the full objective graph, but optionally partition EVERY operation.
    for start in range(0,n,micro):
        sub=ids[start:start+micro]
        x=xfull[start:start+micro] if full_renderer else renderer(sub)
        xa=afull[start:start+micro] if full_renderer else augment(x,params,sub)
        pre=preprocess(x,encoder.kind); apre=preprocess(xa,encoder.kind)
        z=encoder.project(encoder.raw(pre)); za=encoder.project(encoder.raw(apre))
        xs.append(x);axs.append(xa);pres.append(pre);apres.append(apre);zs.append(z);zas.append(za)
        reg=reg+regularization(x,renderer.templates[sub])*len(sub)/n
    z=torch.cat(zs);za=torch.cat(zas);z.retain_grad();za.retain_grad()
    loss=matching(z,target,renderer.arm)+.1*matching(za,target,renderer.arm)+reg
    loss.backward()
    result={"loss":float(loss.detach()),"gradient":grads(renderer),
            "x":torch.cat(xs).detach().cpu(),"xa":torch.cat(axs).detach().cpu(),
            "pre":torch.cat(pres).detach().cpu(),"apre":torch.cat(apres).detach().cpu(),
            "z":z.detach().cpu(),"za":za.detach().cpu(),
            "gz":z.grad.detach().cpu(),"gza":za.grad.detach().cpu()}
    return result


def first_pass(renderer,encoder,target,step,seed,micro):
    n=renderer.n;params=augment_parameters(n,seed,step,renderer.relation,renderer.base.device)
    out={k:[] for k in ("x","xa","pre","apre","z","za")}
    with torch.no_grad():
        for start in range(0,n,micro):
            ids=list(range(start,min(start+micro,n)));x=renderer(ids);xa=augment(x,params,ids)
            pre=preprocess(x,encoder.kind);apre=preprocess(xa,encoder.kind)
            z=encoder.project(encoder.raw(pre));za=encoder.project(encoder.raw(apre))
            for k,v in zip(out,(x,xa,pre,apre,z,za)):out[k].append(v.cpu())
    out={k:torch.cat(v) for k,v in out.items()}
    z=out["z"].cuda().requires_grad_(True);za=out["za"].cuda().requires_grad_(True)
    value=matching(z,target,renderer.arm)+.1*matching(za,target,renderer.arm)
    gz,ga=torch.autograd.grad(value,(z,za))
    out.update(gz=gz.detach().cpu(),gza=ga.detach().cpu())
    return out


def run():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=101);args=ap.parse_args()
    require(not args.output.exists(),"Diagnostic output already exists")
    args.output.mkdir(parents=True);setup()
    groups=manifests();access=PixelAccess(groups,("P",)).install()
    chosen=public_templates(groups["P"],args.seed)
    subset=[chosen[i] for i in (0,32,64,96)]
    templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in subset]).cuda()
    encoder=Encoder("source").use_projection(OUT/"source_P_projection.npz")
    before=state_hash(encoder.model);target,joint=target_public(encoder,groups["P"])
    model=Renderer(templates,"A",args.seed);model.set_step(500)
    initial=copy.deepcopy(model.state_dict());params=augment_parameters(4,args.seed,500,False,"cuda")
    torch.save({"renderer":{k:v.cpu() for k,v in initial.items()},
                "augmentation":[x.cpu() for x in params],"target":target},args.output/"fixture.pt")
    record={"status":"FIXED_PUBLIC_W1_REPRODUCER","arm":"A","seed":args.seed,"step":500,
            "template_indices":[0,32,64,96],"public_images":[{k:r[k] for k in ("image_id","patient_id","sha256","path")} for r in subset],
            "fixture_sha256":sha(args.output/"fixture.pt"),"model_checkpoint_sha256":sha(encoder.model.pretrained_model_path) if hasattr(encoder.model,"pretrained_model_path") else "b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77",
            "source_projection_sha256":sha(OUT/"source_P_projection.npz"),
            "source_public_cache_sha256":sha(OUT/"source_P_features.npz"),
            "initial_source_hashes":source_hashes(),"dtype":"FP32 pixels/model, FP64 moments",
            "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
            "TF32":False,"deterministic":True,"gradient_tolerance":"2e-7+5e-4*reference_peak",
            "loss_tolerance":2e-6,"new_losses":False,"new_private_pixels":0}
    save(args.output/"w1_reproducer.json",record)
    tick=time.perf_counter();rows=[]
    for micro in (1,2,4):
        f=first_pass(model,encoder,target,500,args.seed,micro)
        original=retained(model,encoder,target,500,args.seed,micro,True)
        same=retained(model,encoder,target,500,args.seed,micro,False)
        replay=two_pass(model,encoder,target,500,args.seed,micro);actual=grads(model)
        old=grad_error(actual,original["gradient"]);fixed=grad_error(actual,same["gradient"])
        row={"microbatch":micro,
             "original_reference_vs_replay":old,"all_operations_partitioned_vs_replay":fixed,
             "old_pass":old["max_abs"]<=2e-7+5e-4*old["reference_peak"],
             "partitioned_pass":fixed["max_abs"]<=2e-7+5e-4*fixed["reference_peak"],
             "original_loss_abs":abs(replay["loss"]-original["loss"]),
             "partitioned_loss_abs":abs(replay["loss"]-same["loss"]),
             "original_to_first":{k:error(f[k],original[k]) for k in f},
             "partitioned_to_first":{k:error(f[k],same[k]) for k in f},
             "hashes":{"original_x":tensor_hash(original["x"]),"first_x":tensor_hash(f["x"]),
                       "partitioned_x":tensor_hash(same["x"])}}
        rows.append(row);print(json.dumps(row),flush=True)
        if micro==1:torch.save({"first":f,"original":original,"partitioned":same,"replay_gradients":actual},args.output/"boundary_tensors.pt")
        del f,original,same,actual;torch.cuda.empty_cache()
    require(state_hash(encoder.model)==before,"Source parameters/buffers changed")
    save(args.output/"w1_boundary_errors.json",{"status":"BOUNDARY_DIAGNOSTIC_ONLY","rows":rows,
          "seconds":time.perf_counter()-tick,"access":access.report(),"source_unchanged":True,
          "step2_or_new_objective_started":False})
    print(json.dumps({"status":"BOUNDARY_DIAGNOSTIC_COMPLETE","seconds":time.perf_counter()-tick}),flush=True)


if __name__=="__main__":run()
