"""Whole-bank global moments and exact replay chain rule for a fixed encoder."""
import torch
from .contracts import require
from .render import augment,augment_parameters

def bank_moments(z,arm):
    n=len(z); require(n%4==0,'Bank must contain balanced point/pair blocks')
    q=n//4; relation=arm in ('C','D','R_joint')
    neg,pos=(z[:q],z[2*q:3*q]) if relation else (z[:2*q],z[2*q:])
    result={'m':torch.stack([neg.mean(0),pos.mean(0)]),
            'A':torch.stack([neg.T@neg/len(neg),pos.T@pos/len(pos)])}
    if relation:
        delta=(z[3*q:]-z[q:2*q])/2
        u=torch.cat([z[3*q:],z[q:2*q]],dim=1)/(2**.5)
        result.update(delta=delta.mean(0),C=delta.T@delta/len(delta),u=u.mean(0),U=u.T@u/len(u))
    return result

def matching(z,target,arm):
    # Numerically stable small moments; encoder/images and optimizer remain FP32.
    m=bank_moments(z.double(),arm)
    terms={k:((v-torch.as_tensor(target[k],dtype=v.dtype,device=v.device))**2).sum() for k,v in m.items() if k in target}
    loss=(terms['m']+terms['A'])/4
    if arm in ('C','D'): loss=loss+(terms['delta']+terms['C'])/2
    if arm=='R_joint': loss=loss+(terms['u']+terms['U'])/2
    return loss

def regularization(x,templates):
    anchor=(x-templates).square().mean()
    tv=(x[:,:,1:]-x[:,:,:-1]).abs().mean()+(x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
    return .01*anchor+1e-4*tv

def two_pass(renderer,encoder,target,step,bank_seed,microbatch):
    """Return exact loss and accumulate gradients; do not step the optimizer.

    First pass obtains global clean/augmented feature gradients. Second pass
    replays the same images and augmentation, never per-microbatch moments.
    """
    n=renderer.n; params=augment_parameters(n,bank_seed,step,renderer.relation,renderer.base.device)
    first=[]; first_aug=[]; reg=0.
    with torch.no_grad():
        for start in range(0,n,microbatch):
            ids=list(range(start,min(n,start+microbatch))); x=renderer(ids)
            first.append(encoder(x)); first_aug.append(encoder(augment(x,params,ids)))
            reg+=float(regularization(x,renderer.templates[ids]))*len(ids)/n
    z=torch.cat(first).detach().requires_grad_(True)
    za=torch.cat(first_aug).detach().requires_grad_(True)
    clean=matching(z,target,renderer.arm); noisy=matching(za,target,renderer.arm)
    value=clean+.1*noisy
    gz,ga=torch.autograd.grad(value,(z,za))
    renderer.zero_grad(set_to_none=True)
    for start in range(0,n,microbatch):
        ids=list(range(start,min(n,start+microbatch))); x=renderer(ids)
        response=(encoder(x)*gz[ids]).sum()+regularization(x,renderer.templates[ids])*len(ids)/n
        response.backward()
        # Release the clean encoder graph before augmented replay (same global
        # gradients and same parameters; no optimizer update in between).
        xa=renderer(ids)
        (encoder(augment(xa,params,ids))*ga[ids]).sum().backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in renderer.parameters()),'Nonfinite synthesis gradient')
    return {'loss':float(value.detach())+reg,'point_relation_loss':float(clean.detach()),
            'augmentation_matching':float(noisy.detach()),'regularization':reg,
            'source_image_forwards':4*n,'source_image_backwards':2*n}

def one_pass(renderer,encoder,target,step,bank_seed,microbatch=None):
    """Retain the whole graph while matching the replay's full batch partition.

    Renderer/augmentation kernels also depend numerically on batch shape.
    Partition ALL per-image operations, then compute each global moment loss
    once. A full-batch reference remains available with microbatch=None, but
    cross-batch FP32 differences are not a replay chain-rule correctness test.
    """
    n=renderer.n; microbatch=n if microbatch is None else microbatch
    require(isinstance(microbatch,int) and microbatch>0,'Invalid reference microbatch')
    params=augment_parameters(n,bank_seed,step,renderer.relation,renderer.base.device)
    clean=[];augmented=[];reg=0.
    for start in range(0,n,microbatch):
        ids=list(range(start,min(n,start+microbatch)));x=renderer(ids)
        clean.append(encoder(x));augmented.append(encoder(augment(x,params,ids)))
        reg=reg+regularization(x,renderer.templates[ids])*len(ids)/n
    return matching(torch.cat(clean),target,renderer.arm)+.1*matching(torch.cat(augmented),target,renderer.arm)+reg
