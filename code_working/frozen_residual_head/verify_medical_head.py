"""Rebuild CFG-aware statistics from saved raw arrays, independently of producer math."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from public_medical_backbone.verify_pilot import Check,cfg,sha,read,save

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/medical_head_20260916_v1'

def verify(out):
    start=time.perf_counter();torch.set_num_threads(1);ch=Check();c=read(out/'contract.json');ex=read(out/'extraction.json');fit=read(out/'fit.json')
    for p,h in c['source_sha256'].items():ch.require(sha(p)==h,'source binding '+p)
    for fn,key in [('patient_statistics.npz','patient_statistics_sha256'),('patients.json','patients_sha256'),('weights.npz','weights_sha256')]:ch.require(sha(out/fn)==fit[key],'fit binding '+fn)
    ch.require(ex['complete'] and ex['records']==4352 and ex['unet_calls']==4353 and ex['unet_examples']==8706 and ex['backward']==0,'extraction budget')
    ch.require(ex['base_before']==ex['base_after'] and ex['adapter_exact'],'model immutable')
    ch.require(ex['contract_sha256']==fit['contract_sha256']==sha(out/'contract.json'),'contract binding')
    ch.require(ex['manifest_sha256']==sha(out/'manifest.json'),'manifest binding')
    with np.load(out/'projection.npz') as p:P=p['P'].copy();alphas=p['alphas'].astype(np.float64)
    ch.require(sha(out/'projection.npz')==c['projection_sha256'],'projection frozen')
    manifest=read(out/'manifest.json');tasks=read(out/'extraction_tasks.json')
    ch.require(len(manifest)==len(tasks)==4352,'complete raw records')
    with np.load(out/'weights.npz') as w:weights={k:w[k].copy() for k in w.files}
    ch.require(set(weights)=={'public','pooled','backbone'} and np.array_equal(weights['backbone'],np.zeros((64,4))),'fixed methods')
    groups=defaultdict(list);direct_losses=defaultdict(lambda:defaultdict(list));witnesses=0
    for i,(row,task) in enumerate(zip(manifest,tasks)):
        ch.require({k:row[k] for k in task}==task and row['record_id']==i,'task order')
        path=out/row['raw_path'];ch.require(sha(path)==row['raw_sha256'],'raw hash')
        salt='frozen-residual-capacity-20260916-v1'
        seed=lambda suffix:int(hashlib.sha256(f"{salt}|{row['split']}|{row['image_id']}|{row['draw_id']}|{suffix}".encode()).hexdigest()[:15],16)
        t=125*row['draw_id']+int(np.random.default_rng(seed('timestep')).integers(0,125))
        ch.require(t==row['timestep'] and seed('noise')==row['noise_seed'],'independent draw schedule')
        with np.load(path) as d:
            f=d['features'].astype(np.float64);raw=d['raw_eps'];base=d['base'].astype(np.float64);target=d['target'].astype(np.float64)
            ch.require(f.shape==(1024,16) and raw.shape==(2,4,32,32) and np.all(f[:,0]==1),'raw shapes and constant')
            guided=cfg(raw,7.5)[0].transpose(1,2,0).reshape(-1,4)
            ch.equal(d['base'],guided,'independent guided arithmetic')
            expected_noise=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed('noise')),dtype=torch.float32).numpy()[0].transpose(1,2,0).reshape(-1,4)
            ch.equal(d['target'],expected_noise,'noise target regeneration')
            u=2*((np.log(alphas[t])-np.log1p(-alphas[t]))-c['time_basis_logsnr_min'])/(c['time_basis_logsnr_max']-c['time_basis_logsnr_min'])-1
            basis=np.array([1.,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2]);ch.close(d['basis'],basis,'time basis',atol=1e-14,rtol=1e-14)
            x=7.5*f;r=target-base;n=len(x)
            aa=np.einsum('i,j,ab->iajb',basis,basis,x.T@x/n).reshape(64,64)
            bb=np.einsum('i,ac->iac',basis,x.T@r/n).reshape(64,4);qq=float(np.mean(np.einsum('ij,ij->i',r,r)))
            key=(row['split'],row['patient_id']);groups[key].append((aa,bb,qq))
            if 'hidden' in d:
                witnesses+=1;h=d['hidden'].astype(np.float64);p=P.astype(np.float32).astype(np.float64)
                expected=h@p;eps=np.finfo(np.float32).eps;gamma=320*eps/(1-320*eps)
                bound=gamma*(np.abs(h)@np.abs(p))+2*np.finfo(np.float32).tiny
                ch.require(np.all(np.abs(f[:,1:]-expected)<=bound),'FP32 projection error envelope')
                ch.require(d['conditioning'].shape==(2,77,1024) and d['noisy'].shape==(1,4,32,32),'new witness shapes')
            if row['split']=='eval':
                phi=(basis[None,:,None]*x[:,None,:]).reshape(1024,64)
                for name,w in weights.items():direct_losses[name][row['patient_id']].append(float(np.mean((r-phi@w)**2)))
    ch.require(witnesses==6,'six new backbone witnesses')
    patients=read(out/'patients.json');keys=[(x['split'],x['patient_id']) for x in patients]
    ch.require(set(keys)==set(groups) and len(keys)==len(set(keys))==152,'all disjoint patient units')
    independent=[]
    for key in keys:
        ch.require(len(groups[key])==(16 if key[0]=='public' else 32),'records per patient')
        independent.append(tuple(np.mean(np.stack([x[j] for x in groups[key]]),axis=0) for j in range(3)))
    A=np.stack([x[0] for x in independent]);B=np.stack([x[1] for x in independent]);Q=np.array([x[2] for x in independent])
    with np.load(out/'patient_statistics.npz') as stats:
        for k,value in [('A',A),('B',B),('Q',Q)]:ch.close(stats[k],value,'patient '+k,atol=2e-10,rtol=2e-10)
    for name,roles in [('public',{'public'}),('pooled',{'public','train'})]:
        ii=[i for i,key in enumerate(keys) if key[0] in roles];a=A[ii].mean(0);b=B[ii].mean(0)
        lower=np.linalg.cholesky(a+.001*np.eye(64));w=np.linalg.solve(lower.T,np.linalg.solve(lower,b))
        ch.close(weights[name],w,'independent Cholesky '+name,atol=2e-9,rtol=2e-9)
        ch.close((a+.001*np.eye(64))@weights[name],b,'normal-equation '+name,atol=2e-10,rtol=2e-10)
    for name in weights:
        wanted=[float(np.mean(direct_losses[name][pid])) for pid in fit['development_patient_ids']]
        ch.close(fit['development_losses'][name],wanted,'direct pixel dev MSE '+name,atol=2e-10,rtol=2e-10)
    ch.require(Counter(k[0] for k in keys)=={'public':32,'train':80,'eval':40},'patient counts')
    for p,h in c['source_sha256'].items():ch.require(sha(p)==h,'no source mutation')
    result=dict(status='PASS_NEW_BACKBONE_GUIDED_HEAD_SAVED_ARITHMETIC',complete=True,checks=ch.count,seconds=time.perf_counter()-start,records=4352,
        witnesses=witnesses,patients=152,independent_direct_pixel_losses=True,weights_sha256=sha(out/'weights.npz'),fit_sha256=sha(out/'fit.json'),
        contract_sha256=sha(out/'contract.json'),manifest_sha256=sha(out/'manifest.json'),code_sha256=sha(__file__),max_errors=ch.errors,
        model_forward_replay=False,actual_generation_utility_assessed=False,DP=False)
    save(out/'head_verification.json',result);print({k:result[k] for k in ['status','checks','seconds','records']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();verify(a.out)
