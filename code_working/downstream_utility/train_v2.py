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

def train_steps(state,arm,source_pools,image_arrays,steps,run_seed=11,lr=1e-4,weight_decay=1e-4,trace_directory=None):
    m=resnet18(weights=None);m.fc=torch.nn.Linear(m.fc.in_features,1);m.load_state_dict(state,strict=True)
    m=m.cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=weight_decay,foreach=False)
    require(len(opt.state)==0,'Fresh AdamW state required')
    starting_state_hash=state_hash(m.state_dict())
    require(starting_state_hash==state_hash(state),'Actual common initialization')
    parameter_names=[name for name,_ in m.named_parameters()]
    lossfn=torch.nn.BCEWithLogitsLoss();trace=[];torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for step in range(steps):
        torch.cuda.synchronize();tick=time.perf_counter()
        selected=batch_records(source_pools,arm,run_seed,step)
        require(len(selected)==32,'Fixed batch')
        tick_io=time.perf_counter();a=[image_arrays[r['array_key']][int(r['index'])] for r in selected]
        pixel_hashes=[tensor_sha(z) for z in a]
        require(all(h==r['array_sha256'] for h,r in zip(pixel_hashes,selected)),'Actual source array pixel binding')
        x=preprocess_batch(a,arm,run_seed,step,'cuda');y=torch.tensor([int(r['label']) for r in selected],device='cuda',dtype=torch.float32)
        torch.cuda.synchronize();batch_seconds=time.perf_counter()-tick_io
        fb_start=time.perf_counter()
        opt.zero_grad(set_to_none=True);logits=m(x).flatten();loss=lossfn(logits,y)
        require(bool(torch.isfinite(loss)),'Nonfinite classifier loss');loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()),'Nonfinite gradient')
        torch.cuda.synchronize();forward_backward_seconds=time.perf_counter()-fb_start;opt_start=time.perf_counter()
        opt.step();torch.cuda.synchronize()
        optimizer_seconds=time.perf_counter()-opt_start
        optimizer_steps=[int(v['step'].item()) for v in opt.state.values()]
        require(optimizer_steps and set(optimizer_steps)=={step+1},'AdamW step state')
        if trace_directory is not None:
            np.save(Path(trace_directory)/f'input_{step+1:06d}.npy',x.detach().cpu().numpy(),allow_pickle=False)
        trace.append(dict(step=step+1,loss=float(loss.detach()),logits=logits.detach().cpu().tolist(),
            labels=y.detach().cpu().int().tolist(),input_sha256=tensor_sha(x),
            image_ids=[r['image_id'] for r in selected],patient_ids=[r['patient_id'] for r in selected],
            roles=[r['role'] for r in selected],array_keys=[r['array_key'] for r in selected],
            array_indices=[int(r['index']) for r in selected],source_namespaces=[r['namespace'] for r in selected],
            source_file_sha256=[r['source_file_sha256'] for r in selected],source_pixel_sha256=pixel_hashes,
            finite_loss=True,finite_gradients=True,optimizer_steps=optimizer_steps,
            batch_seconds=batch_seconds,forward_backward_seconds=forward_backward_seconds,optimizer_seconds=optimizer_seconds,
            step_seconds=time.perf_counter()-tick))
    elapsed=time.perf_counter()-begin
    final={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    changed=[name for name in parameter_names if not torch.equal(state[name],final[name])]
    require(bool(changed),'Actual trainable parameter update')
    result=dict(steps=steps,seconds=elapsed,trace=trace,final_state_sha256=state_hash(final),
        initial_state_sha256=starting_state_hash,fresh_optimizer_state_count=0,
        optimizer=dict(name='AdamW',lr=lr,weight_decay=weight_decay,betas=[.9,.999],eps=1e-8,foreach=False),
        parameter_names=parameter_names,changed_trainable_parameters=changed,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    del m,opt;torch.cuda.empty_cache()
    return final,result

