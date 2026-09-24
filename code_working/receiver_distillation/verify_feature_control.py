"""Only new feature-query/loss boundaries; public GPU fixture, no training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import setup,save,require,now
from prrd_v3.datasets import manifests,PixelAccess
from prrd_v3.encoders import pil_tensor,state_hash
from . import feature_control as f
from .second_source import DinoEncoder,ENCODER_ID
from .schedule import ScheduledRenderer
from .verify_public import fixture
from .objectives import one_pass,two_pass
from .signals import objective_digest
from .runtime import seed_all


CRITERIA={'array_atol':1e-12,'loss_atol':2e-6,'gradient_atol':2e-7,'gradient_rtol_peak':5e-4}


def independent(p,limits):
    sums=np.zeros((2,4,16));counts=np.zeros(2)
    for i in range(len(p.ids)):
        for c in (0,1):
            if p.present[i,c]:
                v=p.means[i,c,0].copy();length=np.sqrt(sum(float(a)**2 for a in v.ravel()))
                if length>limits.class_bounds[c]:v*=limits.class_bounds[c]/length
                sums[c]+=v;counts[c]+=1
    return sums,counts


def check_population(p,meta,z,limits):
    p.validate();err=0.
    for j,person in enumerate(p.ids):
        for c in (0,1):
            ids=np.flatnonzero((meta['patient_ids']==person)&(meta['labels']==c))
            if len(ids):
                ref=np.stack([np.array([sum(float(z[k,i,d]) for i in ids)/len(ids)
                                      for d in range(16)]) for k in range(4)])
                err=max(err,float(abs(ref-p.means[j,c,0]).max()))
    got,counts,_=f.bounded_sums(p,limits);ref,rc=independent(p,limits)
    relative=float(np.max(abs(got-ref)/np.maximum(1.,abs(ref))))
    require(err<=1e-12 and relative<=1e-12 and np.allclose(counts,rc,atol=1e-11,rtol=0),
            'Independent patient-class mean/clip aggregation failed')
    return {'images':len(meta['labels']),'patients':len(p.ids),'class_patients':rc.tolist(),
            'visit_mean_max_abs':err,'bounded_sum_scaled_max_abs':relative,'status':'PASS'}


def cpu_checks():
    rng=np.random.default_rng(41);a=rng.normal(size=(7,1,4,16));y=np.array([0,0,1,1,0,1,1]);ids=np.array([1,1,1,2,3,3,3])
    p=f.aggregate_image_signals(a,y,ids);lim=f.Limits((2.,3.))
    meta={'labels':y,'patient_ids':ids};pop=check_population(p,meta,a[:,0].transpose(1,0,2),lim)
    rows,_=f.clipped_contributions(p,lim);total=f.old.sum_patient_axis(rows)
    err=0.
    for i in range(len(rows)):
        diff=total-f.old.sum_patient_axis(np.delete(rows,i,0))
        err=max(err,float(abs(diff-rows[i]).max()))
        require(np.linalg.norm(diff)<=1+1e-12,'Patient add/remove sensitivity violated')
    sums,counts,_=f.bounded_sums(p,lim)
    target,info=f.decode_target(total,lim,sums,counts)
    ref=sums/counts[:,None,None]
    require(np.max(abs(target-ref))<1e-12,'No-noise reconstruction mismatch')
    huge=f.Limits((100.,100.));rr,_=f.clipped_contributions(p,huge)
    raw=p.means[:, :,0].sum(0)/p.present.sum(0)[:,None,None]
    reproduced,_=f.decode_target(f.old.sum_patient_axis(rr),huge,np.zeros((2,4,16)),np.zeros(2))
    require(np.max(abs(reproduced-raw))<1e-12,'Unclipped no-noise mean mismatch')
    zeros=np.zeros(130);zeros[-2:]=-100.
    empty,_=f.decode_target(zeros,lim,np.zeros((2,4,16)),np.zeros(2))
    require(np.array_equal(empty,np.zeros_like(empty)),'Empty/negative noisy class mass')
    bad=0
    for q in (np.zeros(129),np.full(130,np.nan)):
        try:f.decode_target(q,lim,sums,counts)
        except ValueError:bad+=1
    require(bad==2,'Invalid release shape/nonfinite value accepted')
    gradchecks=[]
    for amplitude in (0.,.01,3.):
        z=torch.tensor(rng.normal(size=(4,4,16))*amplitude,dtype=torch.float64,requires_grad=True)
        obj=f.objective(target,lim.class_bounds,[0,0,1,1])
        fn=lambda v:obj.matching({ENCODER_ID:v})[0]
        require(torch.autograd.gradcheck(fn,(z,),eps=1e-6,atol=1e-5,rtol=1e-3),'Feature loss gradcheck')
        value=fn(z);g=torch.autograd.grad(value,z)[0]
        require(bool(torch.isfinite(g).all()) and 0<=float(value)<=2+1e-12,'Bounded loss/gradient')
        gradchecks.append({'amplitude':amplitude,'loss':float(value.detach()),'PASS':True})
    return {'status':'PASS','population':pop,'dimension':130,'max_patient_norm':float(np.linalg.norm(rows,axis=1).max()),
            'sensitivity_cancellation_error':err,'bad_inputs_rejected':bad,'gradcheck':gradchecks,
            'new_private_releases':0,'new_optimizer_updates':0}


def gpu(root,handoff):
    started=time.monotonic();setup();seed_all(101);groups=manifests();access=PixelAccess(groups,('P',)).install()
    rows=fixture(groups['P']);templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in rows]).cuda()
    obj,rules=f.load_feature(handoff,[0,0,1,1],device='cuda');digest=objective_digest(obj)
    encoder=DinoEncoder().use_projection(rules[ENCODER_ID]['projection']);state=state_hash(encoder)
    counts={'forward':0,'backward':0}
    def hook(module,inputs,output):
        n=len(inputs[0]);counts['forward']+=n
        if output.requires_grad:
            def back(g):counts['backward']+=n;return g
            output.register_hook(back)
    handle=encoder.register_forward_hook(hook);checks=[];torch.cuda.reset_peak_memory_stats()
    for mb in (1,4):
        renderer=ScheduledRenderer(templates,'B',101);renderer.set_step(200)
        value=one_pass(renderer,{ENCODER_ID:encoder},obj,mb);value['loss'].backward()
        refs=[p.grad.clone() for p in renderer.parameters() if p.requires_grad]
        expected={k:float(value[k].detach()) for k in ('loss','matching','prior')}
        del value
        actual=two_pass(renderer,{ENCODER_ID:encoder},obj,mb)
        got=[p.grad.clone() for p in renderer.parameters() if p.requires_grad]
        peak=max(float(g.abs().max()) for g in refs);abs_error=max(float((a-b).abs().max()) for a,b in zip(refs,got))
        rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(refs,got)))/torch.sqrt(sum(a.square().sum() for a in refs)).clamp_min(1e-30))
        loss_error=abs(expected['loss']-actual['loss'])
        require(loss_error<=2e-6 and abs_error<=2e-7+5e-4*peak,'Feature actual-model gradient mismatch')
        require(peak>0 and all(bool(torch.isfinite(g).all()) for g in got),'Invalid actual image gradient')
        checks.append({'microbatch':mb,'loss_max_abs':loss_error,'gradient_max_abs':abs_error,
                       'gradient_reference_peak':peak,'gradient_relative_L2':rel,'components':expected,'PASS':True})
        print(json.dumps(checks[-1]),flush=True);del renderer,refs,got;torch.cuda.empty_cache()
    handle.remove();require(counts=={'forward':96,'backward':64},'Public check count mismatch')
    require(state_hash(encoder)==state and objective_digest(obj)==digest,'Fixed encoder/target mutated')
    result={'status':'PASS_ACTUAL_PUBLIC_FEATURE_GRADIENT','criteria':CRITERIA,'checks':checks,
        'encoder_counts':counts,'pixel_access':access.report(),'optimizer_updates':0,'new_private_releases':0,
        'frozen_encoder_and_target':True,'seconds':time.monotonic()-started,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20}
    save(root/'public_gradient_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--handoff',type=Path,required=True)
    a=p.parse_args();gpu(a.root,a.handoff)
