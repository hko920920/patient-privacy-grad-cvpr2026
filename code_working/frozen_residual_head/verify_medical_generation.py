"""Independent CFG/DDIM/head arithmetic and provenance for paired generation."""
import argparse,hashlib,time
from pathlib import Path
import numpy as np
import torch
from public_medical_backbone.verify_pilot import Check,check_packet,arrays,png_pixels,image_from_packet,sha,read,save,cfg

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/medical_head_20260916_v1'
CORE=('conditioning','timesteps','latents','model_inputs','raw_eps','eps','decoded_raw','image_float','initial_from_prepare')

def head(check,label,f,b,res,base,delta,raw,eps,W,t,alphas,c):
    a=float(alphas[t]);u=2*((np.log(a)-np.log1p(-a))-c['time_basis_logsnr_min'])/(c['time_basis_logsnr_max']-c['time_basis_logsnr_min'])-1
    basis=np.array([1,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2]);check.close(b,basis,label+' basis',atol=1e-14,rtol=1e-14)
    phi=np.einsum('b,nk->nbk',basis,f.astype(np.float64)).reshape(1024,64)
    expected=np.einsum('nk,kc->nc',phi,W,optimize=False)
    check.close(res,expected,label+' offline residual',atol=2e-11,rtol=2e-11)
    d=res.astype(np.float32).reshape(32,32,4).transpose(2,0,1)[None].copy()
    check.equal(delta,d,label+' residual dtype/shape')
    expected_raw=np.concatenate([base[:1],base[1:]+d],0)
    check.equal(raw,expected_raw,label+' conditional-only correction')
    check.equal(eps,cfg(raw,7.5),label+' guided equation')
    g0=cfg(base,7.5)
    observed=eps.astype(np.float64)-g0.astype(np.float64);wanted=7.5*d.astype(np.float64)
    bound=64*np.finfo(np.float32).eps*(1+np.abs(base[:1].astype(np.float64))+15*np.abs(base[1:].astype(np.float64))+15*np.abs(d.astype(np.float64)))
    check.bounded(observed,wanted,bound,label+' effective7.5Delta rounding')

def verify(out):
    started=time.perf_counter();torch.set_num_threads(1);ch=Check();c=read(out/'contract.json');g=read(out/'generation_contract.json');ex=read(out/'generation.json');integ=read(out/'integration.json')
    for mapping in [c['source_sha256'],g['source_sha256']]:
        for path,h in mapping.items():ch.require(sha(path)==h,'frozen source '+path)
    for fn,key in [('generation_contract.json','generation_contract_sha256'),('generation_manifest.json','manifest_sha256'),('generation_inputs.pt','inputs_sha256'),('integration.json','integration_sha256')]:ch.require(sha(out/fn)==ex[key],'execution binding '+fn)
    ch.require(ex['complete'] and ex['images']==192 and ex['unet_calls']==5862 and ex['unet_examples']==11724 and ex['backward']==0 and ex['vae_decodes']==195,'executed budget')
    ch.require(ex['base_before']==ex['base_after']==read(out/'extraction.json')['base_before'] and ex['adapter_exact'],'same immutable model')
    config=read(Path(c['snapshot'])/'scheduler/scheduler_config.json')
    with np.load(out/'projection.npz') as p:alphas=p['alphas'].astype(np.float64)
    with np.load(out/'weights.npz') as w:weights={k:w[k].copy() for k in w.files}
    inputs=torch.load(out/'generation_inputs.pt',map_location='cpu',weights_only=True)
    old=torch.load(Path(c['confirmation_directory'])/'inputs.pt',map_location='cpu',weights_only=True)
    ch.require(set(inputs['initials'])==set(c['generation_seeds']) and set(inputs['conditional'])==set(c['prompts']),'input identities')
    for sid,seed in c['generation_seeds'].items():
        expected_seed=int(hashlib.sha256(f'medical-head-generation-20260916-v1|seed|{int(sid[1:])}'.encode()).hexdigest()[:15],16)
        ch.require(seed==expected_seed,'fixed seed derivation')
        z=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
        ch.equal(inputs['initials'][sid].numpy(),z.numpy(),'actual shared latent '+sid)
    for pid in c['prompts']:ch.equal(inputs['conditional'][pid].numpy(),old['conditional'][pid].numpy(),'unchanged prompt '+pid)
    ch.equal(inputs['null'].numpy(),old['null'].numpy(),'unchanged null')
    identity={};witnesses=0
    for row in integ['records']:
        ch.require(sha(out/row['path'])==row['sha256'],'integration raw')
        packet=arrays(out/row['path'])
        if row['kind']=='witness':
            witnesses+=1;ch.require(sha(out/row['original_path'])==row['original_sha256'],'original witness binding')
            original=arrays(out/row['original_path']);t=int(packet['timestep'])
            for method in ('public','pooled'):
                get=lambda key:packet[method+'_'+key]
                ch.close(get('base_raw'),original['raw_eps'],method+' actual prediction replay',atol=2e-6,rtol=2e-6)
                ch.close(get('features'),original['features'],method+' feature replay',atol=2e-6,rtol=2e-6)
                head(ch,str(row['path'])+method,get('features'),get('basis'),get('residual'),get('base_raw'),get('delta'),get('raw'),get('eps'),weights[method],t,alphas,c)
        else:
            identity[row['kind']]=packet
            check_packet(ch,{k:packet[k] for k in CORE},'identity_'+row['kind'],7.5,config)
    ch.require(witnesses==6 and set(identity)=={'plain','zero','restored'} and integ['unet_calls']==102,'complete integration gate')
    for key in CORE:
        ch.equal(identity['zero'][key],identity['plain'][key],'zero identity '+key)
        ch.equal(identity['restored'][key],identity['plain'][key],'restored identity '+key)
    manifest=read(out/'generation_manifest.json');ch.require(len(manifest)==len(g['tasks'])==192,'all generated images')
    ch.require(len({r['image_id'] for r in manifest})==192,'unique IDs')
    details={};raw_hashes={}
    for i,(row,task) in enumerate(zip(manifest,g['tasks'])):
        ch.require({k:row[k] for k in task}==task,'fixed generation cells')
        iid=row['image_id'];pid=row['prompt_id'];sid=row['seed_id'];method=row['method']
        ch.require(iid==hashlib.sha256(f'medical-head-image-v1|{sid}|{pid}|{method}'.encode()).hexdigest()[:12],'blind identifier')
        for key,hkey in [('path','sha256'),('image_path','image_sha256')]:ch.require(sha(out/row[key])==row[hkey],'generation file binding');raw_hashes[row[key]]=row[hkey]
        packet=arrays(out/row['path']);details[iid]=check_packet(ch,{k:packet[k] for k in CORE},iid,7.5,config)
        h=torch.cat([inputs['null'],inputs['conditional'][pid]],0).numpy()
        ch.equal(packet['conditioning'],h,'actual prompt/null '+iid);ch.equal(packet['initial_from_prepare'],inputs['initials'][sid].numpy(),'actual initial '+iid)
        ch.equal(png_pixels(out/row['image_path']),image_from_packet(packet),'PNG '+iid)
        if method!='backbone':
            ch.require(packet['head_features'].shape==(30,1024,16) and np.all(packet['head_features'][:,:,0]==1),'conditional features')
            for j,t in enumerate(packet['timesteps']):
                head(ch,iid+'/'+str(j),packet['head_features'][j],packet['head_basis'][j],packet['head_residual'][j],packet['base_raw_eps'][j],packet['conditional_delta'][j],packet['raw_eps'][j],packet['eps'][j],weights[method],int(t),alphas,c)
                ch.equal(packet['guided_base'][j],cfg(packet['base_raw_eps'][j],7.5),'guided base '+iid+'/'+str(j))
        if (i+1)%48==0:print('VERIFIED_GENERATION '+str(i+1)+'/192',flush=True)
    for mapping in [c['source_sha256'],g['source_sha256']]:
        for path,h in mapping.items():ch.require(sha(path)==h,'no final source mutation')
    result=dict(status='PASS_GUIDED_MEDICAL_HEAD_GENERATION_SAVED_ARITHMETIC',complete=True,checks=ch.count,seconds=time.perf_counter()-started,
        images=192,witness_replays=6,full_model_replay=False,zero_and_restored_exact=True,head_applied_only_to_conditional=True,
        generation_contract_sha256=sha(out/'generation_contract.json'),manifest_sha256=sha(out/'generation_manifest.json'),weights_sha256=sha(out/'weights.npz'),
        code_sha256=sha(__file__),raw_files_sha256=raw_hashes,trajectory_checks=details,errors=ch.errors,scientific_quality_assessed=False)
    save(out/'generation_verification.json',result);print({k:result[k] for k in ['status','checks','seconds','images']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();verify(a.out)
