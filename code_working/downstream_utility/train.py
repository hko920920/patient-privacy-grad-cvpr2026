"""Shared training kernel used by every pilot arm; no expert-final reader."""
import copy,time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from .common import *
from .data import pools,batch_records,preprocess_batch,letterbox

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

def profile_classifier(out=OUT):
    from PIL import Image
    setup();c=check(out);require(read(out/'data.json')['complete'],'Verified cache production required')
    require(read(out/'generation.json')['complete'],'Generation required')
    require(not (out/'classifier_runs').exists(),'No implicit rerun')
    (out/'classifier_runs').mkdir();real=rows(out/'real_manifest_private.csv');generated=read(out/'generation_manifest.json')
    arr_real=np.load(out/'real224.npy',mmap_mode='r');synthetic=[]
    for i,r in enumerate(generated):
        with Image.open(out/r['image_path']) as im:synthetic.append(letterbox(im))
        r.update(index=i,array_key='synthetic',patient_id=r['cell'],label=r['label'],role='synthetic_'+r['method'])
    image_arrays={'real':arr_real,'synthetic':np.stack(synthetic)}
    for r in real:r['array_key']='real'
    source_pools={role:pools([r for r in real if r['role']==role]) for role in ('public','private')}
    source_pools.update({method:pools([r for r in generated if r['method']==method]) for method in METHODS})
    init=initial_state(11);torch.save(init,out/'classifier_initial.pt');initial_sha=state_hash(init)
    all_runs=[];last_final=None
    for arm in ARMS:
        for repeat in (0,1):
            final,rec=train_steps(init,arm,source_pools,image_arrays,7)
            rec.update(arm=arm,repeat=repeat,initial_state_sha256=initial_sha,run_seed=11,lr=1e-4)
            p=out/'classifier_runs'/f'{arm}_{repeat}.json';save(p,rec)
            torch.save(final,out/'classifier_runs'/f'{arm}_{repeat}.pt')
            all_runs.append({k:v for k,v in rec.items() if k!='trace'})
            if repeat==0:first=rec
            else:
                require(rec['final_state_sha256']==first['final_state_sha256'],'Classifier replay weights '+arm)
                require(all(a['loss']==b['loss'] and a['logits']==b['logits'] and a['input_sha256']==b['input_sha256'] for a,b in zip(first['trace'],rec['trace'])),'Classifier replay trace '+arm)
            if arm=='R1':last_final=final
            print(json.dumps({'phase':'classifier','arm':arm,'repeat':repeat,'seconds':rec['seconds']}),flush=True)
    # Cost-only validation forward: no AUC/AP or selection based on these profile scores.
    dev=sorted([r for r in real if r['role'] in ('classifier_selection','method_development')],key=lambda r:seed('forward',r['image_id']))[:128]
    m=resnet18(weights=None);m.fc=torch.nn.Linear(m.fc.in_features,1);m.load_state_dict(last_final);m=m.cuda().eval()
    torch.cuda.synchronize();tick=time.perf_counter();outputs=[]
    with torch.inference_mode():
        for start in range(0,len(dev),32):
            rr=dev[start:start+32];x=preprocess_batch([arr_real[int(r['index'])] for r in rr],'R0',0,0,'cuda')
            outputs.extend(m(x).flatten().cpu().tolist())
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-tick
    save(out/'profile_inference_private.json',[dict(image_id=r['image_id'],patient_id=r['patient_id'],role=r['role'],logit=z) for r,z in zip(dev,outputs)])
    require(sum(r['steps'] for r in all_runs)==98,'Optimizer cap')
    save(out/'classifier.json',dict(complete=True,runs=all_runs,optimizer_updates=98,steps_per_arm_per_repeat=7,
        determinism='exact saved inputs, logits, losses and final model state for all seven replays',
        actual_AUROC_AP_computed=False,model_checkpoint_sha256=MODEL_SHA,initial_state_sha256=initial_sha,
        inference_images=len(dev),inference_seconds=inference_seconds,expert_images=0,reserved_images=0,
        image_arrays_source_sha256=sha(out/'real224.npy')))
    print('CLASSIFIER_PROFILE_COMPLETE_98_UPDATES',flush=True)
