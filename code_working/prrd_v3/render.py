"""Public-template bounded grayscale pyramid; pair metadata is purely virtual."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import require,seed
from .resampling import resize,affine_sample

LEVELS=(8,16,32,64,112,224)
ACTIVATE=(1,81,161,241,321,401)

class Renderer(nn.Module):
    def __init__(self,templates,arm,bank_seed):
        super().__init__(); n=len(templates)
        require(n%4==0 and templates.shape[1:]==(1,224,224),'Template shape')
        self.arm=arm; self.relation=arm in ('C','D','R_joint'); self.n=n
        self.register_buffer('templates',templates.clone())
        self.register_buffer('base',torch.logit(templates.clamp(1e-4,1-1e-4)))
        generator=torch.Generator(device='cpu').manual_seed(seed('prrd-render',bank_seed))
        self.residuals=nn.ParameterList([nn.Parameter((torch.randn(n,1,r,r,generator=generator)*.001).to(templates.device)) for r in LEVELS])
        self.active=1
    def set_step(self,step):
        self.active=sum(step>=s for s in ACTIVATE)
        for j,p in enumerate(self.residuals): p.requires_grad_(j<self.active)
    def forward(self,indices=None):
        ids=torch.arange(self.n,device=self.base.device) if indices is None else torch.as_tensor(indices,device=self.base.device)
        value=self.base[ids]
        for p in list(self.residuals)[:self.active]:
            z=p[ids]
            if self.relation:
                quarter=self.n//4; neg=(ids>=quarter)&(ids<2*quarter); pos=ids>=3*quarter
                # Pair u at negative index; v at positive index. Marginal slots unchanged.
                uids=torch.where(pos,ids-2*quarter,ids)
                vids=torch.where(neg,ids+2*quarter,ids)
                paired=p[uids]+torch.where(neg[:,None,None,None],-p[vids],p[vids])
                z=torch.where((neg|pos)[:,None,None,None],paired,z)
            value=value+resize(z,(224,224),antialias=False)
        return value.sigmoid()

def augment_parameters(n,bank_seed,step,relation,device):
    g=torch.Generator().manual_seed(seed('prrd-augmentation',bank_seed,step))
    u=torch.rand(n,5,generator=g)
    if relation:
        q=n//4; u[3*q:4*q]=u[q:2*q]
    angle=(u[:,0]*2-1)*3*math.pi/180; scale=.98+.04*u[:,3]
    theta=torch.zeros(n,2,3)
    theta[:,0,0]=angle.cos()/scale; theta[:,0,1]=-angle.sin()/scale
    theta[:,1,0]=angle.sin()/scale; theta[:,1,1]=angle.cos()/scale
    theta[:,0,2]=(u[:,1]*2-1)*.02; theta[:,1,2]=(u[:,2]*2-1)*.02
    return theta.to(device),(.98+.04*u[:,4]).to(device)

def augment(x,params,indices):
    theta,brightness=params; ids=torch.as_tensor(indices,device=x.device)
    grid=F.affine_grid(theta[ids],x.shape,align_corners=False)
    return (affine_sample(x,grid)*brightness[ids,None,None,None]).clamp(0,1)
