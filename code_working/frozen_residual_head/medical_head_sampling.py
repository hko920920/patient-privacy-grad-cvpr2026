"""Conditional-only FP64 small head in a fixed FP32 CFG7.5 sampler."""
from contextlib import nullcontext
import numpy as np
import torch
from diffusers import DDIMScheduler

def require(ok,message):
    if not ok:raise RuntimeError(message)

def arr(t):return t.detach().cpu().numpy().copy()

class GuidedHead:
    def __init__(self,unet,P,W,alphas,lo,hi):
        self.unet=unet;self.P=torch.from_numpy(P.astype(np.float32)).cuda();self.W=np.array(W,dtype=np.float64,copy=True)
        self.alphas=np.asarray(alphas,np.float64);self.lo=lo;self.hi=hi;self.handle=None;self.hidden=None;self.last=None
        require(P.shape==(320,15) and W.shape==(64,4),'Head dimensions')
    def __enter__(self):
        require(self.handle is None,'Already installed')
        def capture(module,inputs):
            require(self.hidden is None,'Stale captured feature');self.hidden=inputs[0].detach()
        self.handle=self.unet.conv_out.register_forward_pre_hook(capture);return self
    def __exit__(self,*args):
        if self.handle is not None:self.handle.remove()
        self.handle=None;self.hidden=None;self.last=None
    def predict(self,x,t,h):
        require(self.handle is not None and not torch.is_grad_enabled(),'Inference context required')
        require(tuple(x.shape)==(2,4,32,32) and tuple(h.shape)==(2,77,1024),'Fixed null/conditional batch')
        require(x.dtype==h.dtype==torch.float32,'FP32 inputs required')
        try:
            base=self.unet(x,t,encoder_hidden_states=h,return_dict=False)[0]
            require(self.hidden is not None and self.hidden.shape==(2,320,32,32),'Conditional tap')
            hidden=self.hidden[-1].permute(1,2,0).reshape(-1,320)
            f=arr(torch.cat([torch.ones((1024,1),device=x.device),hidden@self.P],dim=1))
            tv=int(t.item()) if isinstance(t,torch.Tensor) else int(t);a=self.alphas[tv]
            u=2*((np.log(a)-np.log1p(-a))-self.lo)/(self.hi-self.lo)-1
            require(-1-1e-12<=u<=1+1e-12,'Basis time bounds')
            b=np.array([1,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2],np.float64)
            phi=(b[None,:,None]*f.astype(np.float64)[:,None,:]).reshape(1024,64);residual=phi@self.W
            delta=torch.from_numpy(residual.astype(np.float32).reshape(32,32,4).transpose(2,0,1).copy())[None].to(x.device)
            raw=torch.cat([base[:1],base[1:]+delta],dim=0)
            guided=raw[:1]+7.5*(raw[1:]-raw[:1]);guided_base=base[:1]+7.5*(base[1:]-base[:1])
            require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(guided).all()),'Nonfinite correction')
            self.last=dict(features=f,basis=b,residual=residual,base_raw=arr(base),guided_base=arr(guided_base),delta=arr(delta))
            return raw,guided
        finally:self.hidden=None

def sample(pipe,initial,conditioning,adapter):
    sched=DDIMScheduler.from_config(pipe.scheduler.config);sched.set_timesteps(30,device='cuda')
    latent=initial.clone();data={'latents':[arr(latent)],'model_inputs':[],'raw_eps':[],'eps':[],
        'head_features':[],'head_basis':[],'head_residual':[],'base_raw_eps':[],'guided_base':[],'conditional_delta':[]}
    with torch.inference_mode(),adapter:
        for t in sched.timesteps:
            x=sched.scale_model_input(torch.cat([latent,latent],0),t);raw,eps=adapter.predict(x,t,conditioning)
            data['model_inputs'].append(arr(x));data['raw_eps'].append(arr(raw));data['eps'].append(arr(eps))
            for target,source in [('head_features','features'),('head_basis','basis'),('head_residual','residual'),('base_raw_eps','base_raw'),('guided_base','guided_base'),('conditional_delta','delta')]:data[target].append(adapter.last[source])
            latent=sched.step(eps,t,latent,eta=0.,return_dict=False)[0];require(bool(torch.isfinite(latent).all()),'Nonfinite trajectory')
            data['latents'].append(arr(latent))
        decoded=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
        require(bool(torch.isfinite(decoded).all()),'Nonfinite VAE output')
        im=pipe.image_processor.postprocess(decoded,output_type='np',do_denormalize=[True])[0]
    result={k:np.stack(v) for k,v in data.items()}
    result.update(conditioning=arr(conditioning),timesteps=arr(sched.timesteps),decoded_raw=arr(decoded),image_float=np.asarray(im,np.float32),initial_from_prepare=arr(initial))
    return result
