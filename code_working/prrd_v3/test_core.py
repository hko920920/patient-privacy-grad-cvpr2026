"""Small mathematical/gradient tests; no patient pixels or medical utility."""
import copy,importlib.util,unittest
import numpy as np
import torch
from torch import nn
from .contracts import SUPPLIED
from .patient_moments import *
from .datasets import derangement,png_roundtrip
from .render import Renderer,augment,augment_parameters
from .objectives import bank_moments,matching,two_pass,one_pass
from .resampling import resize,affine_sample

class ToyEncoder(nn.Module):
    def forward(self,x):
        # Smooth, genuinely image-dependent features with bounded norm.
        f=torch.nn.functional.adaptive_avg_pool2d(x,(2,2)).flatten(1)
        return torch.cat([f,f.square()],1).tanh()/3

class CoreTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(18)
        z=rng.normal(size=(16,4)); z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1)
        self.z=z; self.y=np.tile([0,1],8); self.ids=np.repeat(np.arange(8).astype(str),2)
        self.ps=patient_contributions(z,self.y,self.ids)
    def test_reference_risk_and_empty(self):
        spec=importlib.util.spec_from_file_location('supplied_reference',SUPPLIED/'reference_math.py')
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        for ys in (self.y,np.zeros(16,int),np.ones(16,int)):
            ps=patient_contributions(self.z,ys,self.ids);t=aggregate(ps)
            w=np.arange(4)/3;k,v,c,_=system(t)
            raw=[(self.z[self.ids==i],ys[self.ids==i]) for i in sorted(set(self.ids))]
            self.assertAlmostEqual(ref.direct_risk(raw,w,1,.1),.5*w@k@w-w@v+c,places=12)
    def test_dimensions_sensitivity_joint(self):
        for mode,bound in [('point',6**.5),('relation',3),('joint_pair',3)]:
            q=query(self.ps,mode);self.assertEqual(len(q),dimension(4,mode))
            for i,p in enumerate(self.ps):
                np.testing.assert_allclose(q-query(self.ps[:i]+self.ps[i+1:],mode),vector(p,mode),atol=1e-12)
                self.assertLessEqual(np.linalg.norm(vector(p,mode)),bound+1e-12)
        r=aggregate(self.ps);j=aggregate(self.ps,'joint_pair')
        np.testing.assert_allclose(r['C'],j['C'],atol=1e-12)
        np.testing.assert_allclose(r['delta'],j['delta'],atol=1e-12)
        self.assertEqual([dimension(16,k) for k in ('point','relation','joint_pair')],[306,459,867])
    def test_shuffle_only_private(self):
        public,private=self.ps[:2],self.ps[2:]
        t=aggregate(self.ps);s=shuffle_target(public,private,derangement(6,101))
        for k in ('counts','m','A','delta'):np.testing.assert_allclose(t[k],s[k],atol=1e-12)
        self.assertGreater(np.linalg.norm(t['C']-s['C']),1e-6)
    def test_global_replay_and_resume(self):
        torch.manual_seed(2);templates=torch.rand(8,1,224,224)*.8+.1;encoder=ToyEncoder()
        for arm in ('A','B','C','D','R_joint'):
            r=Renderer(templates,arm,101);r.set_step(1)
            with torch.no_grad():target={k:v.numpy()+.001 for k,v in bank_moments(encoder(templates),arm).items()}
            r.zero_grad();one_pass(r,encoder,target,1,101).backward()
            expected=[None if p.grad is None else p.grad.clone() for p in r.parameters()]
            for micro in (1,3,4):
                two_pass(r,encoder,target,1,101,micro)
                for p,g in zip(r.parameters(),expected):
                    if g is not None:torch.testing.assert_close(p.grad,g,atol=2e-8,rtol=2e-5)
            # Renderer and optimizer serialize exact successful-update state.
            opt=torch.optim.AdamW(r.parameters(),lr=.01,weight_decay=0,foreach=False);opt.step()
            clone=Renderer(templates,arm,999);clone.load_state_dict(copy.deepcopy(r.state_dict()));clone.set_step(2)
            other=torch.optim.AdamW(clone.parameters(),lr=.01,weight_decay=0,foreach=False)
            other.load_state_dict(copy.deepcopy(opt.state_dict()))
            for model,optimizer in ((r,opt),(clone,other)):
                model.set_step(2);two_pass(model,encoder,target,2,101,3);optimizer.step()
            for a,b in zip(r.parameters(),clone.parameters()):self.assertTrue(torch.equal(a,b))
    def test_deterministic_resampling_operator(self):
        torch.manual_seed(1);x=torch.rand(2,1,17,19,requires_grad=True)
        for aa in (False,True):
            target=torch.nn.functional.interpolate(x,size=(23,29),mode='bilinear',align_corners=False,antialias=aa)
            y=resize(x,(23,29),aa)
            torch.testing.assert_close(y,target,atol=4e-7,rtol=1e-6)
            g1,=torch.autograd.grad(y.square().sum(),x,retain_graph=True)
            g2,=torch.autograd.grad(target.square().sum(),x,retain_graph=True)
            torch.testing.assert_close(g1,g2,atol=2e-6,rtol=1e-6)
        theta=torch.tensor([[[.99,-.01,.005],[.01,.99,.005]]]).repeat(2,1,1)
        grid=torch.nn.functional.affine_grid(theta,x.shape,align_corners=False)
        a=affine_sample(x,grid);b=torch.nn.functional.grid_sample(x,grid,align_corners=False)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        ga,=torch.autograd.grad(a.square().sum(),x,retain_graph=True)
        gb,=torch.autograd.grad(b.square().sum(),x)
        torch.testing.assert_close(ga,gb,atol=4e-6,rtol=4e-6)
    def test_png_quantization(self):
        x=np.random.default_rng(1).random((224,224));y,_=png_roundtrip(x)
        self.assertLessEqual(np.max(abs(x-y/255)),.5/255+1e-12)

if __name__=='__main__':unittest.main()
