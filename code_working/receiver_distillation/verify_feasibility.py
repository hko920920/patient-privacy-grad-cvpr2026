"""Actual public two-source gradient check with a memory-bounded reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import setup, require, save, sha, now
from prrd_v3.datasets import manifests, PixelAccess
from prrd_v3.encoders import Encoder, pil_tensor, state_hash
from .schedule import ScheduledRenderer
from .second_source import DinoEncoder, ENCODER_ID, load_combined
from .verify_public import fixture
from .signals import bind_objective, objective_digest
from .objectives import collect, prior, two_pass
from .runtime import seed_all


def gradients(renderer):
    return [p.grad.detach().clone() for p in renderer.parameters() if p.requires_grad]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--a1-handoff',type=Path,required=True)
    ap.add_argument('--second-handoff',type=Path,required=True)
    args=ap.parse_args()
    output=args.root/'changed_path_verification'; output.mkdir(exist_ok=False)
    contract={'created':now(),'pixel_role':'P','images':4,'conditions':4,'encoders':['biovil',ENCODER_ID],
              'microbatches':[1,4],'renderer_logical_step':200,'optimizer_updates':0,
              'reference':'retained global graph per encoder, half matching each, prior once; then aggregate gradients',
              'criteria':{'loss_atol':2e-6,'gradient_atol':2e-7,'gradient_rtol_peak':5e-4},
              'A1_handoff_sha256':sha(args.a1_handoff),'A2_handoff_sha256':sha(args.second_handoff),
              'source_files':{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}}
    save(output/'contract.json',contract)
    setup(); seed_all(101); started=time.monotonic()
    groups=manifests(); access=PixelAccess(groups,('P',)).install()
    rows=fixture(groups['P'])
    templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in rows]).cuda()
    objective,rules=load_combined(args.a1_handoff,args.second_handoff,[0,0,1,1],'P','cuda')
    target_before=objective_digest(objective)
    encoders={'biovil':Encoder('source').use_projection(rules['biovil']['projection']),
              ENCODER_ID:DinoEncoder().use_projection(rules[ENCODER_ID]['projection'])}
    states={n:state_hash(e) for n,e in encoders.items()}
    counters={n:{'forward':0,'backward':0} for n in encoders}; handles=[]
    for name,e in encoders.items():
        def hook(module,inputs,output,name=name):
            n=len(inputs[0]);counters[name]['forward']+=n
            if output.requires_grad:
                def backward(g):
                    counters[name]['backward']+=n
                    return g
                output.register_hook(backward)
        handles.append(e.register_forward_hook(hook))
    checks=[]; torch.cuda.reset_peak_memory_stats()
    for mb in (1,4):
        renderer=ScheduledRenderer(templates,'B',101); renderer.set_step(200)
        renderer.zero_grad(set_to_none=True)
        reference_loss=0.
        for name,encoder in encoders.items():
            single=bind_objective(objective.conditions,{name:objective.heads[name]},
                                  {name:objective.targets[name]},objective.labels)
            values=collect(renderer,{name:encoder},single,mb)
            loss,_=single.matching(values)
            reference_loss+=float(loss.detach().cpu())/2
            (loss/2).backward()
            del values,loss
        reg=prior(renderer,mb); reference_loss+=float(reg.detach().cpu()); reg.backward()
        ref=gradients(renderer)
        actual=two_pass(renderer,encoders,objective,mb)
        got=gradients(renderer)
        abs_err=max(float((a-b).abs().max()) for a,b in zip(ref,got))
        peak=max(float(a.abs().max()) for a in ref)
        relative=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(ref,got)))/
                       torch.sqrt(sum(a.square().sum() for a in ref)).clamp_min(1e-30))
        loss_err=abs(reference_loss-actual['loss'])
        require(loss_err<=2e-6 and abs_err<=2e-7+5e-4*peak,'Actual two-source gradient mismatch')
        require(all(torch.isfinite(g).all() for g in got) and peak>0,'Missing actual image gradient')
        checks.append({'microbatch':mb,'loss_abs':loss_err,'gradient_max_abs':abs_err,
                       'reference_peak':peak,'relative_L2':relative,'PASS':True})
        print(json.dumps(checks[-1]),flush=True)
        del renderer,ref,got,reg
        torch.cuda.empty_cache()
    for h in handles:h.remove()
    require(all(state_hash(e)==states[n] for n,e in encoders.items()),'Fixed model changed')
    require(objective_digest(objective)==target_before,'Target changed')
    require(all(v=={'forward':96,'backward':64} for v in counters.values()),'Gradient check count')
    result={'status':'PASS_ACTUAL_PUBLIC_TWO_SOURCE_GRADIENTS','checks':checks,'counts':counters,
            'access':access.report(),'seconds':time.monotonic()-started,
            'models_and_targets_fixed':True,'optimizer_updates':0,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'V_DP_final':False}
    save(output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
