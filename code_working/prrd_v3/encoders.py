"""Bound image encoders. Receiver is never passed to image optimization."""
from __future__ import annotations
import hashlib, importlib.metadata, json, re, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF, InterpolationMode
from .contracts import OUT, sha, save, require
from .resampling import resize

BIO_REV='692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23'
BIO_PATH=Path.home()/('.cache/huggingface/hub/models--microsoft--BiomedVLP-BioViL-T/snapshots/'+BIO_REV+'/biovil_t_image_model_proj_size_128.pt')
BIO_SHA='b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77'
DENSE_PATH=Path.home()/'.cache/torch/hub/checkpoints/densenet121-a639ec97.pth'

def state_hash(module):
    h=hashlib.sha256()
    for key,v in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def preprocess(x, kind):
    """Tensor implementation of pinned official geometry; no PIL gradient break.

    PIL uint8 bilinear intermediate rounding differs from tensor interpolation.
    W1 reports that difference. Real and synthetic use THIS same tensor path.
    """
    require(x.ndim in (3,4) and x.shape[-3]==1, 'Expected grayscale CHW/NCHW')
    short=512 if kind=='source' else 256
    if x.is_cuda:
        h,w=x.shape[-2:]
        size=(short,int(short*w/h)) if h<=w else (int(short*h/w),short)
        x=resize(x,size,antialias=True)
    else:
        x=TF.resize(x,short,interpolation=InterpolationMode.BILINEAR,antialias=True)
    x=TF.center_crop(x,448 if kind=='source' else 224)
    x=x.repeat(*([1] if x.ndim==4 else []),3,1,1)
    if kind!='source': x=TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
    return x

def pil_tensor(im): return TF.pil_to_tensor(im).to(torch.float32)/255

class Encoder(nn.Module):
    def __init__(self, kind, device='cuda'):
        super().__init__(); require(kind in ('source','recipient'), 'Unknown encoder role')
        self.kind=kind; self.device_name=device
        if kind=='source':
            from health_multimodal.image import ImageModel, ImageEncoderType
            require(sha(BIO_PATH)==BIO_SHA,'BioViL IMAGE weights changed')
            # freeze_encoder=True would disable image gradients inside official forward.
            self.model=ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                                 joint_feature_size=128,freeze_encoder=False,pretrained_model_path=BIO_PATH)
            self.raw_dim=128
        else:
            from torchvision.models import densenet121
            self.model=densenet121(weights=None)
            weights=torch.load(DENSE_PATH,map_location='cpu',weights_only=True)
            # Exact torchvision legacy checkpoint key migration.
            pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(weights):
                match=pattern.match(key)
                if match: weights[match.group(1)+match.group(2)]=weights.pop(key)
            self.model.load_state_dict(weights,strict=True)
            self.model.classifier=nn.Identity(); self.raw_dim=1024
        self.model.eval().requires_grad_(False).to(device)
        self.register_buffer('center',torch.empty(0,device=device))
        self.register_buffer('basis',torch.empty(0,device=device))
        self.register_buffer('scale',torch.ones((),device=device))
        self.eval()
    def raw(self, prepared):
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()), 'Encoder not fixed')
        y=self.model(prepared)
        if self.kind=='source': y=torch.nn.functional.normalize(y.projected_global_embedding,dim=-1)
        return y
    def project(self,y):
        require(self.center.numel()>0,'Public projection not bound')
        a=(y-self.center)@self.basis
        return a/torch.maximum(a.norm(dim=-1,keepdim=True),self.scale.clamp_min(1e-12))
    def forward(self,x): return self.project(self.raw(preprocess(x,self.kind)))
    def use_projection(self,path):
        d=np.load(path,allow_pickle=False)
        require(str(d['role'])==self.kind,'Projection/encoder mismatch')
        for key in ('center','basis','scale'):
            setattr(self,key,torch.as_tensor(d[key],dtype=torch.float32,device=self.device_name))
        return self

def extract_public(encoder, group, access, microbatch=4):
    all_features=[]; started=time.perf_counter()
    for start in range(0,len(group),microbatch):
        batch=torch.stack([preprocess(pil_tensor(access.image(r)),encoder.kind) for r in group[start:start+microbatch]])
        with torch.no_grad(): f=encoder.raw(batch.to(encoder.device_name))
        all_features.append(f.cpu().numpy())
        if start%128==0: print(json.dumps({'phase':'public_features','encoder':encoder.kind,'images':start+len(batch)}),flush=True)
    return np.concatenate(all_features),time.perf_counter()-started

def fit_projection(features, group, dimension, role, path):
    """P-only patient-uniform PCA and weighted q95, no label-dependent selection."""
    counts={pid:sum(r['patient_id']==pid for r in group) for pid in {r['patient_id'] for r in group}}
    weights=np.array([1/(len(counts)*counts[r['patient_id']]) for r in group])
    f=np.asarray(features,dtype=np.float64)
    center=(f*weights[:,None]).sum(0); centered=f-center
    vals,vectors=np.linalg.eigh((centered*weights[:,None]).T@centered)
    basis=vectors[:,np.argsort(vals)[::-1][:dimension]]
    for j in range(dimension):
        if basis[np.argmax(abs(basis[:,j])),j]<0: basis[:,j]*=-1
    norms=np.linalg.norm(centered@basis,axis=1); order=np.argsort(norms,kind='stable')
    q95=norms[order[min(np.searchsorted(np.cumsum(weights[order]),.95),len(order)-1)]]
    scale=max(float(q95),1e-12)
    require(not Path(path).exists(),'Projection overwrite prohibited')
    np.savez(path,center=center,basis=basis,scale=np.array(scale),role=np.array(role),
             explained_eigenvalues=np.sort(vals)[::-1],patient_weights=weights,
             image_ids=np.array([r['image_id'] for r in group]))
    return {'dimension':dimension,'raw_dimension':f.shape[1],'scale':scale,'sha256':sha(path),
            'fit_role':'P','labels_used':False,'PCA_and_q95_weighting':'equal patient, uniform images within patient',
            'retained_variance':float(np.sort(vals)[::-1][:dimension].sum()/vals.sum())}
