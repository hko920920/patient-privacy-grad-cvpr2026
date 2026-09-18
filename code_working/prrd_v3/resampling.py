"""Deterministic linear resampling; preserve the tensor resize/affine operator."""
from functools import lru_cache
import torch
from torch.nn import functional as F

@lru_cache(maxsize=64)
def _cpu_weights(source,target,antialias):
    # Response of the EXACT torchvision/PyTorch linear interpolation to basis
    # vectors. Images are not involved. Rows are destination, columns source.
    basis=torch.eye(source,dtype=torch.float32).reshape(source,1,1,source)
    return F.interpolate(basis,size=(1,target),mode='bilinear',align_corners=False,
                         antialias=antialias)[:,0,0,:].T.contiguous()

@lru_cache(maxsize=128)
def weights(source,target,antialias,device,dtype):
    return _cpu_weights(source,target,antialias).to(device=device,dtype=dtype)

def resize(x,size,antialias=False):
    h,w=x.shape[-2:]; oh,ow=size
    if (h,w)==(oh,ow): return x
    wy=weights(h,oh,antialias,str(x.device),x.dtype)
    wx=weights(w,ow,antialias,str(x.device),x.dtype)
    # CUBLAS_WORKSPACE_CONFIG and disabled TF32 make forward/backward GEMMs
    # deterministic, unlike CUDA atomic interpolation backward in torch2.6.
    return torch.matmul(wy,torch.matmul(x,wx.T))

def affine_sample(x,grid):
    """Bilinear zero-padding grid_sample equivalent, deterministic gather grad."""
    n,c,h,w=x.shape; oh,ow=grid.shape[1:3]
    px=((grid[...,0]+1)*w-1)/2; py=((grid[...,1]+1)*h-1)/2
    x0=px.floor(); y0=py.floor(); dx=px-x0; dy=py-y0
    flat=x.reshape(n,c,h*w); result=torch.zeros(n,c,oh*ow,device=x.device,dtype=x.dtype)
    for ax,ay,weight in ((x0,y0,(1-dx)*(1-dy)),(x0+1,y0,dx*(1-dy)),
                         (x0,y0+1,(1-dx)*dy),(x0+1,y0+1,dx*dy)):
        valid=(ax>=0)&(ax<w)&(ay>=0)&(ay<h)
        indices=(ay.clamp(0,h-1).long()*w+ax.clamp(0,w-1).long()).reshape(n,1,-1).expand(-1,c,-1)
        value=torch.gather(flat,2,indices)
        result=result+value*(weight*valid).reshape(n,1,-1)
    return result.reshape(n,c,oh,ow)
