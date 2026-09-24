"""Public-only finite-difference operators. No private feature or V read, no optimizer."""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

def run(root,out):
    sys.path.insert(0,str(root))
    import torch
    from torchvision.models import resnet18
    from torchvision.transforms import functional as TF
    from prrd_v3.datasets import TRAIN,PixelAccess
    from prrd_v3.encoders import Encoder,pil_tensor,preprocess
    from receiver_distillation.second_source import DinoEncoder
    from receiver_distillation.signals import conditions
    from receiver_resnet_eval_20260924.run import CHECKPOINT,project
    start=time.monotonic()
    def event(**kw):
        print(json.dumps(dict(elapsed=time.monotonic()-start,**kw)),flush=True)
        assert time.monotonic()-start<600,'Public probe timeout'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with TRAIN.open(encoding='utf-8',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['planned_namespace']=='real/public']
    access=PixelAccess({'P':rows},('P',)).install()
    chosen=[]
    for cl in (0,1):
        ordered=sorted((r for r in rows if int(r['label'])==cl),
            key=lambda r:hashlib.sha256(('public-inverse-probe-v1'+r['image_id']).encode()).hexdigest())
        seen=set()
        for r in ordered:
            if r['patient_id'] in seen:continue
            chosen.append(r);seen.add(r['patient_id'])
            if len(seen)==4:break
        assert len(seen)==4
    assert len({r['patient_id'] for r in chosen})==8
    images=[]
    for r in chosen:
        with access.image(r) as im:
            t=pil_tensor(im)
            images.append(TF.resize(t,[224,224],antialias=True))
    x=torch.stack(images).cuda().clamp(.02,.98)
    logits=torch.logit(x)
    coords=torch.arange(224,device='cuda',dtype=torch.float32)+.5
    patterns=[]
    for ky,kx in ((0,0),(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1)):
        pat=torch.cos(np.pi*ky*coords[:,None]/224)*torch.cos(np.pi*kx*coords[None,:]/224)
        patterns.append(pat/pat.square().mean().sqrt())
    basis=torch.zeros((16,8,1,224,224),device='cuda')
    for j in range(16):basis[j,4*(j//8):4*(j//8+1),0]=patterns[j%8]
    y=np.array([0]*4+[1]*4)
    h=.01
    configs=[('base',None,0.)]
    for j in range(16):
        configs += [(f'p{j}',j,h),(f'm{j}',j,-h)]
    for j in (0,8,15):
        configs += [(f'hp{j}',j,h/2),(f'hm{j}',j,-h/2)]
    out.mkdir(parents=True,exist_ok=True)
    r=root/'_reports'
    public_bounds=np.array([1.9061329951133719,1.5917066731966771])
    source=DinoEncoder().use_projection(r/'receiver_a1a2_s200_20260922_v1/second_targets/public/dinov2_P_projection.npz')
    cs=conditions();source_values={}
    with torch.inference_mode():
        for key,j,step in configs:
            img=torch.sigmoid(logits if j is None else logits+step*basis[j])
            z=torch.stack([source(c.apply(img)) for c in cs]).double()
            values=[]
            for cl in (0,1):
                zc=z[:,4*cl:4*(cl+1)].permute(1,0,2)
                norm=zc.flatten(1).norm(dim=1)
                bounded=zc*(public_bounds[cl]/norm.clamp_min(1e-300)).clamp_max(1)[:,None,None]
                values.append((.5*bounded.mean(0)/public_bounds[cl]).flatten())
            source_values[key]=torch.cat(values).cpu().numpy()
            if key=='base' or key.startswith('p') or key=='hm15':event(model='DINO',point=key)
    del source;torch.cuda.empty_cache()
    A=np.column_stack([(source_values[f'p{j}']-source_values[f'm{j}'])/(2*h) for j in range(16)])
    saved={'A':A,'source_base':source_values['base']}
    derivative_checks={'source':{}}
    for j in (0,8,15):
        half=(source_values[f'hp{j}']-source_values[f'hm{j}'])/h
        derivative_checks['source'][str(j)]=float(np.linalg.norm(half-A[:,j])/max(1e-12,np.linalg.norm(half)))
    for rec in ('DenseNet','ResNet18'):
        if rec=='DenseNet':
            model=Encoder('recipient').use_projection(r/'prrd_v3_20260918_w01/recipient_P_projection.npz')
            ppath=r/'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz'
        else:
            model=resnet18(weights=None)
            model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
            model.fc=torch.nn.Identity();model=model.cuda().eval().requires_grad_(False)
            ppath=r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz'
        with np.load(ppath,allow_pickle=False) as pd:
            assert set(pd['roles'].tolist())=={'P'}
            zp=pd['z'].astype(np.float64);pid=pd['patient_ids']
        counts=Counter(pid.tolist())
        probe_weights=np.array([1/len(counts)/counts[i] for i in pid])
        vals={}
        with torch.inference_mode():
            for key,j,step in configs:
                img=torch.sigmoid(logits if j is None else logits+step*basis[j])
                if rec=='DenseNet':z=model(img).cpu().numpy().astype(np.float64)
                else:
                    raw=model(preprocess(img,'recipient')).cpu().numpy()
                    z,_=project(raw,r/'receiver_resnet18_reuse_20260924_v1/P_projection.npz')
                    z=z.astype(np.float64)
                w=np.linalg.solve(z.T@z/8+.1*np.eye(z.shape[1]),z.T@(2*y-1)/8)
                vals[key]=np.sqrt(probe_weights)*(zp@w)
                if key=='base' or key=='m15' or key=='hm15':event(model=rec,point=key)
        B=np.column_stack([(vals[f'p{j}']-vals[f'm{j}'])/(2*h) for j in range(16)])
        saved[rec+'_B']=B
        saved[rec+'_base']=vals['base']
        derivative_checks[rec]={}
        for j in (0,8,15):
            half=(vals[f'hp{j}']-vals[f'hm{j}'])/h
            derivative_checks[rec][str(j)]=float(np.linalg.norm(half-B[:,j])/max(1e-12,np.linalg.norm(half)))
        del model;torch.cuda.empty_cache()
    np.savez_compressed(out/'public_operators.npz',**saved)
    result={'status':'PUBLIC_OPERATOR_PROBE_COMPLETE','seconds':time.monotonic()-start,
       'scope':'local finite differences, public P only, no utility evaluation',
       'private_queries':0,'synthesis_updates':0,'V_Expert_Reserved_access':False,
       'image_count':8,'coefficient_dimension':16,'source_dimension':128,
       'finite_difference_step':h,'half_step_checks':derivative_checks,
       'source_singular_values':np.linalg.svd(A,compute_uv=False).tolist(),
       'pixel_access':access.report(),
       'source_noise':'no noise generated here; downstream mathematical analysis uses illustrative noise',
       'plan_sha256':hashlib.sha256((out.parent/'PUBLIC_PROBE_PLAN.md').read_bytes()).hexdigest(),
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'probe_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    event(result=result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

