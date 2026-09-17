"""Corrected shared training kernel: disjoint real/synthetic pools. Original failed profile preserved in train.py."""
import copy,time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from .common import *
from .data_v2 import batch_records,preprocess_batch,letterbox

def initial_state(run_seed):
    require(sha(MODEL_FILE)==MODEL_SHA,'ImageNet weight digest')
    torch.manual_seed(run_seed);m=resnet18(weights=None)
    m.load_state_dict(torch.load(MODEL_FILE,map_location='cpu',weights_only=True),strict=True)
    m.fc=torch.nn.Linear(m.fc.in_features,1)
    return {k:v.detach().clone() for k,v in m.state_dict().items()}

def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train_steps(state,arm,source_pools,image_arrays,steps,run_seed=11,lr=1e-4,weight_decay=1e-4):
    m=resnet18(weights=None);m.fc=torch.nn.Linear(m.fc.in_features,1);m.load_state_dict(state,strict=True)
    m=m.cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=weight_decay,foreach=False)
    lossfn=torch.nn.BCEWithLogitsLoss();trace=[];torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for step in range(steps):
        torch.cuda.synchronize();tick=time.perf_counter()
        selected=batch_records(source_pools,arm,run_seed,step)
        require(len(selected)==32,'Fixed batch')
        tick_io=time.perf_counter();a=[image_arrays[r['array_key']][int(r['index'])] for r in selected]
        x=preprocess_batch(a,arm,run_seed,step,'cuda');y=torch.tensor([int(r['label']) for r in selected],device='cuda',dtype=torch.float32)
        torch.cuda.synchronize();batch_seconds=time.perf_counter()-tick_io
        opt.zero_grad(set_to_none=True);logits=m(x).flatten();loss=lossfn(logits,y)
        require(bool(torch.isfinite(loss)),'Nonfinite classifier loss');loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()),'Nonfinite gradient')
        opt.step();torch.cuda.synchronize()
        trace.append(dict(step=step+1,loss=float(loss.detach()),logits=logits.detach().cpu().tolist(),
            labels=y.detach().cpu().int().tolist(),input_sha256=tensor_sha(x),
            image_ids=[r['image_id'] for r in selected],patient_ids=[r['patient_id'] for r in selected],
            roles=[r['role'] for r in selected],array_keys=[r['array_key'] for r in selected],
            batch_seconds=batch_seconds,step_seconds=time.perf_counter()-tick))
    elapsed=time.perf_counter()-begin
    final={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    result=dict(steps=steps,seconds=elapsed,trace=trace,final_state_sha256=state_hash(final),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    del m,opt;torch.cuda.empty_cache()
    return final,result

