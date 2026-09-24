"""Held-out perturbations, public pixels only, fixed maps from preceding public geometry."""
import argparse,csv,hashlib,json,sys,time
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
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with TRAIN.open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['planned_namespace']=='real/public']
    access=PixelAccess({'P':rows},('P',)).install();chosen=[]
    for cl in (0,1):
        ordered=sorted((r for r in rows if int(r['label'])==cl),key=lambda r:hashlib.sha256(('public-inverse-probe-v1'+r['image_id']).encode()).hexdigest())
        seen=set()
        for r in ordered:
            if r['patient_id'] in seen:continue
            chosen.append(r);seen.add(r['patient_id'])
            if len(seen)==4:break
    xs=[]
    for r in chosen:
        with access.image(r) as im:xs.append(TF.resize(pil_tensor(im),[224,224],antialias=True))
    logits=torch.logit(torch.stack(xs).cuda().clamp(.02,.98))
    coords=torch.arange(224,device='cuda',dtype=torch.float32)+.5
    patterns=[]
    for ky,kx in ((0,0),(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1)):
        pat=torch.cos(np.pi*ky*coords[:,None]/224)*torch.cos(np.pi*kx*coords[None,:]/224)
        patterns.append(pat/pat.square().mean().sqrt())
    basis=torch.zeros((16,8,1,224,224),device='cuda')
    for j in range(16):basis[j,4*(j//8):4*(j//8+1),0]=patterns[j%8]
    def image(a):
        t=torch.as_tensor(a,dtype=torch.float32,device='cuda')
        return torch.sigmoid(logits+torch.einsum('j,jnhwk->nhwk',t,basis))
    r=root/'_reports'
    src=DinoEncoder().use_projection(r/'receiver_a1a2_s200_20260922_v1/second_targets/public/dinov2_P_projection.npz')
    cs=conditions();bounds=np.array([1.9061329951133719,1.5917066731966771])
    def signal(img):
        z=torch.stack([src(c.apply(img)) for c in cs]).double();res=[]
        for cl in (0,1):
            zc=z[:,4*cl:4*(cl+1)].permute(1,0,2);norm=zc.flatten(1).norm(dim=1)
            zc=zc*(bounds[cl]/norm.clamp_min(1e-300)).clamp_max(1)[:,None,None]
            res.append((.5*zc.mean(0)/bounds[cl]).flatten())
        return torch.cat(res).cpu().numpy()
    with np.load(out/'public_probe/candidate_maps.npz') as d:
        maps={k:d[k] for k in ('all16','public_signal_PCA8','source_singular_top8','receiver_risk_top8')}
    with np.load(out/'public_probe/public_operators.npz') as d:t0=d['source_base']
    config=json.loads((out/'public_probe/risk_results.json').read_text())
    tau=config['coefficient_prior_sd'];sigma=config['illustrative_query_noise_sd']
    rng=np.random.default_rng(872413);a_true=rng.normal(size=(32,16))*tau;noise=rng.normal(size=(32,128))*sigma
    coeff={name:[] for name in maps}
    with torch.inference_mode():
        baseline_error=float(np.max(np.abs(signal(image(np.zeros(16)))-t0)));assert baseline_error<1e-7
        for i,a in enumerate(a_true):
            target=signal(image(a))+noise[i]-t0
            for name,K in maps.items():coeff[name].append(K@target)
    del src;torch.cuda.empty_cache()
    print(json.dumps({'phase':'public_noisy_signals_complete','seconds':time.monotonic()-start}),flush=True)
    results={}
    for rec in ('DenseNet','ResNet18'):
        if rec=='DenseNet':
            model=Encoder('recipient').use_projection(r/'prrd_v3_20260918_w01/recipient_P_projection.npz')
            pp=r/'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz'
        else:
            model=resnet18(weights=None);model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
            model.fc=torch.nn.Identity();model=model.cuda().eval().requires_grad_(False)
            pp=r/'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz'
        with np.load(pp,allow_pickle=False) as d:
            assert set(d['roles'].tolist())=={'P'}
            zp=d['z'].astype(np.float64);pid=d['patient_ids']
        count=Counter(pid.tolist());weights=np.array([1/len(count)/count[x] for x in pid])
        def scores(img):
            if rec=='DenseNet':z=model(img).cpu().numpy().astype(np.float64)
            else:
                z,_=project(model(preprocess(img,'recipient')).cpu().numpy(),r/'receiver_resnet18_reuse_20260924_v1/P_projection.npz');z=z.astype(np.float64)
            y=np.array([-1.]*4+[1.]*4)
            w=np.linalg.solve(z.T@z/8+.1*np.eye(z.shape[1]),z.T@y/8)
            return np.sqrt(weights)*(zp@w)
        errors={name:[] for name in maps}
        with torch.inference_mode():
            for i,a in enumerate(a_true):
                clean=scores(image(a))
                for name in maps:errors[name].append(float(np.sum((scores(image(coeff[name][i]))-clean)**2)))
                if i%8==0:print(json.dumps({'model':rec,'pair':i,'seconds':time.monotonic()-start}),flush=True)
                assert time.monotonic()-start<600
        results[rec]={name:{'mean_squared_prediction_difference':float(np.mean(es)),
                    'standard_error':float(np.std(es,ddof=1)/np.sqrt(len(es))),
                    'individual_errors':es} for name,es in errors.items()}
        del model;torch.cuda.empty_cache()
    result={'scope':'PUBLIC_LOCAL_NONLINEAR_PREDICTION_CHECK_NOT_CLINICAL_UTILITY',
            'pairs':32,'rng_seed':872413,'tau':tau,'illustrative_sigma':sigma,
            'seconds':time.monotonic()-start,'source_baseline_reproduction_error':baseline_error,
            'private_queries':0,'synthesis_updates':0,'V_Expert_Reserved_access':False,
            'pixel_access':access.report(),'results':results,
            'plan_sha256':hashlib.sha256((out/'PUBLIC_NONLINEAR_PLAN.md').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'public_probe/nonlinear_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({r:{n:{k:v for k,v in d.items() if k!='individual_errors'} for n,d in rows.items()} for r,rows in results.items()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

