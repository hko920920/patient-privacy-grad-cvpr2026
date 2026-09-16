"""Cross-image conditioning-response transfer. All target weights stay frozen."""
import time
import torch
from .common import seed
from .models import diffusion_input

SUPPORT=(250,500,750)
QUERY=(50,250,500,750,950)

def make_basis(hidden,mask,dimension=8):
    generator=torch.Generator().manual_seed(260910)
    active=mask.reshape(-1,1).expand(hidden.shape[1],hidden.shape[2]).reshape(-1).bool()
    values=torch.randn((int(active.sum()),dimension),generator=generator)
    q=torch.linalg.qr(values,mode="reduced").Q
    matrix=torch.zeros(hidden.numel(),dimension)
    matrix[active]=q
    return matrix

def projected_step(a,grad,step=.01,radius=.05):
    updated=a+step*grad/(grad.norm()+1e-12)
    return updated*torch.clamp(radius/(updated.norm()+1e-12),max=1.)

class ResponseProbe:
    def __init__(self,unet,scheduler,cache):
        self.unet=unet
        self.scheduler=scheduler
        self.cache=cache
        self.hidden=cache["hidden"][cache["audit_prompt"]].clone().to("cuda",torch.float32)
        self.basis=make_basis(self.hidden.cpu(),cache["token_mask"]).to("cuda")
        self.scale=self.hidden.norm().detach()
        self.forward=0;self.backward=0
        self.initial_versions={n:p._version for n,p in unet.named_parameters()}
        # Dropout=0 in the pinned UNet and adapter; train mode activates non-reentrant checkpointing.
        self.unet.train()
        self.unet.requires_grad_(False)

    def _switch(self,base):
        if base:self.unet.disable_adapters()
        else:self.unet.enable_adapters()
        # PEFT enable_adapters may re-enable parameter gradients. Audit never trains weights.
        self.unet.requires_grad_(False)

    def loss(self,image_id,a,base,phase,ts,draws,gradient=False):
        self._switch(base)
        grad_sum=torch.zeros_like(a);loss_sum=0.
        z=self.cache["latents"][image_id].clone().to("cuda",torch.float16)
        count=len(ts)*draws
        for tval in ts:
            for draw in range(draws):
                ng=torch.Generator(device="cuda").manual_seed(seed("audit-noise",phase,image_id,tval,draw))
                noise=torch.randn(z.shape,generator=ng,device="cuda",dtype=torch.float16)
                t=torch.tensor([tval],device="cuda")
                noisy,target=diffusion_input(z,t,noise,self.scheduler)
                with torch.set_grad_enabled(gradient):
                    coeff=a.detach().clone().requires_grad_(gradient)
                    hidden=self.hidden+(self.scale*(self.basis@coeff)).reshape_as(self.hidden)
                    with torch.autocast("cuda",dtype=torch.float16):
                        prediction=self.unet(noisy,t,hidden.to(torch.float16)).sample
                        value=(prediction.float()-target.float()).square().mean()
                    self.forward+=1
                    if gradient:
                        # Different adapter states are differentiated BEFORE switching state,
                        # so checkpoint recomputation cannot silently use the other model.
                        grad=torch.autograd.grad(value*1024.,coeff)[0]/1024.
                        self.backward+=1
                        assert torch.isfinite(grad).all()
                        grad_sum+=grad.detach()/count
                assert torch.isfinite(value)
                loss_sum+=float(value.detach())/count
        return loss_sum,grad_sum

    def delta(self,image_id,a,phase,ts,draws,gradient=False):
        b,bg=self.loss(image_id,a,True,phase,ts,draws,gradient)
        t,tg=self.loss(image_id,a,False,phase,ts,draws,gradient)
        return b-t,bg-tg,b,t

    def patient(self,image_ids,steps=6,support=SUPPORT,query=QUERY,draws=2):
        assert len(image_ids)==2 and image_ids[0]!=image_ids[1]
        started=time.perf_counter();beforeF=self.forward;beforeB=self.backward
        initial=[];records=[]
        for support_id,query_id in [image_ids,image_ids[::-1]]:
            a=torch.zeros(8,device="cuda")
            g0=None
            for _ in range(steps):
                _,grad,_,_=self.delta(support_id,a,"support",support,1,True)
                if g0 is None:g0=grad.detach().clone()
                a=projected_step(a,grad).detach()
            initial.append(g0)
            shifted,_,_,_=self.delta(query_id,a,"query",query,draws)
            zero,_,base0,target0=self.delta(query_id,torch.zeros_like(a),"query",query,draws)
            own_gain=shifted-zero
            refs=self.cache["reference_matches"][query_id]
            ref_gains=[]
            for ref in refs:
                shifted_ref,_,_,_=self.delta(ref,a,"query",query,draws)
                zero_ref,_,_,_=self.delta(ref,torch.zeros_like(a),"query",query,draws)
                ref_gains.append(shifted_ref-zero_ref)
            records.append({"support":support_id,"query":query_id,
                "coefficient_norm":float(a.norm()),"own_gain":own_gain,
                "reference_gain":sum(ref_gains)/len(ref_gains),
                "response":own_gain-sum(ref_gains)/len(ref_gains),
                "base_loss":base0,"target_loss":target0,"base_minus_target":zero})
        torch.cuda.synchronize()
        assert all(p.grad is None for p in self.unet.parameters())
        assert all(p._version==self.initial_versions[n] for n,p in self.unet.named_parameters())
        dot=float(torch.dot(initial[0],initial[1]))
        return {"response":sum(r["response"] for r in records)/2,
            "loss_mean":-sum(r["target_loss"] for r in records)/2,
            "loss_max":max(-r["target_loss"] for r in records),
            "base_difference_mean":sum(r["base_minus_target"] for r in records)/2,
            "projected_conditioning_gradient_dot":dot,
            "projected_conditioning_gradient_cosine":dot/max(float(initial[0].norm()*initial[1].norm()),1e-12),
            "folds":records,"forward":self.forward-beforeF,"backward":self.backward-beforeB,
            "seconds":time.perf_counter()-started,"weights_unchanged":True}
