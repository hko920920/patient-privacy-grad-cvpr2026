"""Frozen public DINOv2 source. DenseNet is not imported by this adapter."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF
import timm
from prrd_v3.contracts import require, sha
from prrd_v3.resampling import resize
from prrd_v3.bank_runtime import digest
from .signals import conditions, heads, bind_objective
from .target_io import reconstruct, load_target

ENCODER_ID = 'dinov2_vitb14'
WEIGHTS = Path.home()/'.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
PREPROCESS = {
    'input': 'native or synthetic grayscale FP32 [0,1]',
    'geometry': 'tensor bilinear antialias resize short256 then center224',
    'RGB': 'repeat grayscale; ImageNet mean/std',
    'output': 'timm normalized CLS token, external L2 then P-only PCA16/q95',
    'positional_embeddings': 'timm dynamic interpolation from pretrained 518 grid to 224',
    'attention': 'explicit deterministic attention; fused_attn=False',
    'whitening': False,
}


class DinoEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device_name = device
        self.model = timm.create_model('vit_base_patch14_dinov2', pretrained=False,
                                      num_classes=0, dynamic_img_size=True)
        weights = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        incompatible = self.model.load_state_dict(weights, strict=False)
        require(not incompatible.missing_keys and incompatible.unexpected_keys == ['mask_token'],
                'DINO checkpoint architecture mismatch')
        for block in self.model.blocks:
            block.attn.fused_attn = False
        self.model.eval().requires_grad_(False).to(device)
        for key, value in [('center',torch.empty(0)),('basis',torch.empty(0)),('scale',torch.ones(()))]:
            self.register_buffer(key,value.to(device))
        self.eval()

    def raw(self, x):
        require(x.ndim == 4 and x.shape[1] == 1, 'DINO expects grayscale NCHW')
        h, w = x.shape[-2:]
        size = (256,int(256*w/h)) if h <= w else (int(256*h/w),256)
        x = TF.center_crop(resize(x,size,antialias=True),224).repeat(1,3,1,1)
        x = TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()),
                'DINO must remain fixed')
        return torch.nn.functional.normalize(self.model(x),dim=-1)

    def forward(self, x):
        require(self.center.numel() == 768, 'DINO public projection missing')
        a = (self.raw(x)-self.center)@self.basis
        return a/torch.maximum(a.norm(dim=-1,keepdim=True),self.scale.clamp_min(1e-12))

    def use_projection(self, path):
        with np.load(path,allow_pickle=False) as d:
            require(str(d['role']) == ENCODER_ID,'Wrong DINO projection role')
            for key in ('center','basis','scale'):
                setattr(self,key,torch.as_tensor(d[key],dtype=torch.float32,device=self.device_name))
        return self


def descriptor(projection):
    cs, hs = conditions(), heads(ENCODER_ID,16)
    return {
        'schema':'receiver.fixed-condition-linear-ce/v1',
        'encoder_id':ENCODER_ID,'conditions':4,'dimension':16,'public_seed':101,
        'checkpoint':str(WEIGHTS),'checkpoint_sha256':sha(WEIGHTS),
        'projection':str(projection),'projection_sha256':sha(projection),
        'timm_version':timm.__version__,'preprocessing':PREPROCESS,
        'feature_path':'FP32 bounded P-only PCA16/q95; no whitening',
        'placement':'native shared relative affine then resize256 crop224; synthetic224 same affine',
        'gradient_layout':'vec(W[2,16]) then bias[2]; float64 CE derivative',
        'patient_weight':'visit mean within patient/class then present-patient mean then 0.5 each class',
        'empty_class':'zero contribution; do not renormalize remaining class',
        'loss':'mean over K of cosine distance; equal encoder mean; anchor0.01 TV0.0001 once',
        'tuples':[{'index':c.index,'theta':c.theta.tolist(),'brightness':c.brightness.tolist(),
                   'weight':h.weight.tolist(),'bias':h.bias.tolist()} for c,h in zip(cs,hs)],
    }


def load_second(handoff, population='pooled', device='cpu'):
    if isinstance(handoff,(str,Path)):
        handoff = json.loads(Path(handoff).read_text(encoding='utf-8'))
    require(handoff['schema']=='receiver.target-handoff/v1','Wrong second handoff')
    require(sha(handoff['target_file']) == handoff['target_sha256'],'Second target bytes changed')
    require(sha(handoff['rule_file']) == handoff['rule_file_sha256'],'Second rule bytes changed')
    rule = json.loads(Path(handoff['rule_file']).read_text(encoding='utf-8'))
    require(digest(rule)==handoff['rule_sha256']==digest(descriptor(Path(rule['projection']))),
            'DINO rule/checkpoint/projection binding changed')
    with np.load(handoff['target_file'],allow_pickle=False) as d:
        require(str(d['schema'])=='receiver.separate-condition-target/v1'
                and str(d['rule_sha256'])==digest(rule),'DINO target schema/conditions')
        value = d[population+'_gradient'].copy()
    require(value.shape==(4,34) and np.isfinite(value).all(),'DINO target shape/finite')
    cs, hs = reconstruct(rule,device)
    return cs, hs, torch.as_tensor(value,device=device), rule


def load_combined(a1_handoff, second_handoff, labels, population='pooled', device='cpu'):
    first, first_rule = load_target(a1_handoff,labels,population=population,device=device)
    if second_handoff is None:
        return first, {'biovil':first_rule}
    cs, hs, target, rule = load_second(second_handoff,population,device)
    require(all(torch.equal(a.theta,b.theta) and torch.equal(a.brightness,b.brightness)
                for a,b in zip(first.conditions,cs)), 'Encoder augmentation conditions differ')
    objective = bind_objective(first.conditions,dict(first.heads,**{ENCODER_ID:hs}),
                               dict(first.targets,**{ENCODER_ID:target}),labels)
    require(torch.equal(first.targets['biovil'],objective.targets['biovil']),
            'Adding second encoder changed BioViL target')
    return objective, {'biovil':first_rule,ENCODER_ID:rule}
