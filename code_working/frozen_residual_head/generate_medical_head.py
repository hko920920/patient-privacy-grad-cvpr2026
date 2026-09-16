"""Fixed integration gates then three methods on64 paired generation cells."""
import argparse,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from .run_medical_head import OUT,check
from .medical_head_sampling import GuidedHead,sample,arr
from public_medical_backbone.run_pilot import models,manual,require,sha,read,save,save_torch,base_sha,now

CORE=('conditioning','timesteps','latents','model_inputs','raw_eps','eps','decoded_raw','image_float','initial_from_prepare')

def compressed(path,data):
    with Path(path).open('xb') as f:np.savez_compressed(f,**data)

def prepare(out):
    c=check(out);v=read(out/'head_verification.json');require(v['complete'] and v['status'].startswith('PASS'),'Verified new head required')
    require(v['weights_sha256']==sha(out/'weights.npz') and v['fit_sha256']==sha(out/'fit.json'),'Verified weights unchanged')
    files=[Path(__file__),Path(__file__).with_name('medical_head_sampling.py'),Path(__file__).with_name('verify_medical_generation.py'),
        out/'contract.json',out/'head_verification.json',out/'weights.npz',out/'fit.json',out/'manifest.json',out/'projection.npz']
    sources={str(p):sha(p) for p in files};tasks=[]
    for sid in c['generation_seeds']:
        for pid in c['prompts']:
            for method in c['generation_methods']:
                iid=hashlib.sha256(f'medical-head-image-v1|{sid}|{pid}|{method}'.encode()).hexdigest()[:12]
                tasks.append(dict(image_id=iid,method=method,seed_id=sid,prompt_id=pid))
    witnesses=[r for r in read(out/'manifest.json') if r['image_id']==c['witness_images'][r['split']] and r['draw_id'] in (0,7)]
    require(len(witnesses)==6,'Six witnesses')
    save(out/'generation_contract.json',dict(schema='medical-head-generation/v1',created_utc=now(),source_sha256=sources,tasks=tasks,witnesses=witnesses,
        expected_images=192,guidance=7.5,steps=30,eta=0.,precision='FP32',integration_replay_tolerance=dict(atol=2e-6,rtol=2e-6),
        zero_identity='all core packet arrays exact',effective_correction_bound='64*eps32*(1+abs(eps_u)+15*abs(eps_c)+15*abs(delta))',
        no_more_seed_or_scale_search=True,expected_generation_calls=5760,expected_integration_calls=102))
    print('GENERATION_PREPARED '+sha(out/'generation_contract.json'),flush=True)

def run(out):
    c=check(out);g=read(out/'generation_contract.json')
    for p,h in g['source_sha256'].items():require(sha(p)==h,'Generation source changed '+p)
    require(not(out/'generation_inputs.pt').exists(),'No implicit generation repeat')
    models.setup();start=time.perf_counter();snap=Path(c['snapshot']);cp=Path(c['backbone_directory'])/'checkpoints/E4.pt'
    require(sha(cp)==c['checkpoint_sha256'],'Fixed checkpoint')
    unet=models.load_unet(cp,training=False).float();expected=torch.load(cp,map_location='cpu',weights_only=True)['adapter'];actual=models.adapter_state(unet)
    require(set(actual)==set(expected) and all(torch.equal(actual[k],expected[k]) for k in expected),'Actual FP32 adapter')
    before=base_sha(unet);hooks_before=len(unet.conv_out._forward_pre_hooks);calls=[0]
    count_handle=unet.register_forward_pre_hook(lambda module,args:calls.__setitem__(0,calls[0]+1))
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).cuda()
    sched=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True)
    pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=sched,image_processor=VaeImageProcessor(vae_scale_factor=8))
    saved=torch.load(Path(c['confirmation_directory'])/'inputs.pt',map_location='cpu',weights_only=True)
    initials={sid:torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32) for sid,seed in c['generation_seeds'].items()}
    inputs=dict(initials=initials,conditional=saved['conditional'],null=saved['null']);save_torch(out/'generation_inputs.pt',inputs)
    with np.load(out/'projection.npz') as p:P=p['P'].copy();alphas=p['alphas'].copy()
    with np.load(out/'weights.npz') as w:weights={k:w[k].copy() for k in w.files}
    make=lambda method:GuidedHead(unet,P,weights[method],alphas,c['time_basis_logsnr_min'],c['time_basis_logsnr_max'])
    (out/'integration').mkdir();integration=[];cache=models.load_cache();train_sched=models.scheduler()
    with torch.inference_mode():
        for row in g['witnesses']:
            d=dict(np.load(out/row['raw_path']));z=cache['latents'][row['image_id']].float()
            noise=torch.randn(z.shape,generator=torch.Generator(device='cpu').manual_seed(row['noise_seed']),dtype=torch.float32)
            noisy,target=models.diffusion_input(z,torch.tensor([row['timestep']]),noise,train_sched)
            cond=torch.cat([saved['null'],cache['hidden'][row['prompt']].float()],0)
            require(np.array_equal(noisy.numpy(),d['noisy']) and np.array_equal(cond.numpy(),d['conditioning']),'Original cache/noise witness exact')
            data=dict(noisy=d['noisy'],conditioning=d['conditioning'],timestep=d['timestep'],stored_features=d['features'],stored_base_raw=d['raw_eps'])
            for method in ('public','pooled'):
                with make(method) as adapter:
                    raw,eps=adapter.predict(torch.cat([noisy,noisy],0).cuda(),torch.tensor([row['timestep']],device='cuda'),cond.cuda())
                    require(np.allclose(adapter.last['base_raw'],d['raw_eps'],atol=2e-6,rtol=2e-6),'Witness prediction replay')
                    require(np.allclose(adapter.last['features'],d['features'],atol=2e-6,rtol=2e-6),'Witness feature replay')
                    data[method+'_raw']=arr(raw);data[method+'_eps']=arr(eps)
                    for k,v in adapter.last.items():data[method+'_'+k]=v
            path=out/'integration'/f"witness_{row['record_id']:06d}.npz";compressed(path,data)
            integration.append(dict(kind='witness',path=path.relative_to(out).as_posix(),sha256=sha(path),original_path=row['raw_path'],original_sha256=row['raw_sha256']))
    sid=next(iter(initials));pid=next(iter(c['prompts']));z=initials[sid].cuda();h=torch.cat([saved['null'],saved['conditional'][pid]],0).cuda()
    plain=manual(pipe,z,h,7.5,False);zero=sample(pipe,z,h,make('backbone'));restored=manual(pipe,z,h,7.5,False)
    for k in CORE:require(np.array_equal(plain[k],zero[k]) and np.array_equal(plain[k],restored[k]),'Zero/restored full path '+k)
    for name,packet in [('plain',plain),('zero',zero),('restored',restored)]:
        path=out/'integration'/(name+'.npz');compressed(path,packet);integration.append(dict(kind=name,path=path.relative_to(out).as_posix(),sha256=sha(path)))
    require(calls[0]==102 and len(unet.conv_out._forward_pre_hooks)==hooks_before,'Integration call/hook counts')
    save(out/'integration.json',dict(complete=True,unet_calls=calls[0],vae_decodes=3,zero_and_restored_exact=True,witnesses=6,records=integration,
        generation_contract_sha256=sha(out/'generation_contract.json'),model_forward_replay='six saved real-noise witnesses only; not full extraction'))
    print('INTEGRATION_GATE_PASS 102F',flush=True)
    (out/'generation_traces').mkdir();(out/'images').mkdir();manifest=[]
    for i,task in enumerate(g['tasks']):
        z=initials[task['seed_id']].cuda();h=torch.cat([saved['null'],saved['conditional'][task['prompt_id']]],0).cuda()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        packet=manual(pipe,z,h,7.5,False) if task['method']=='backbone' else sample(pipe,z,h,make(task['method']))
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        require(all(np.isfinite(v).all() for v in packet.values()),'Nonfinite generation')
        trace=out/'generation_traces'/(task['image_id']+'.npz');png=out/'images'/(task['image_id']+'.png')
        compressed(trace,packet);require(not png.exists(),'No PNG overwrite');Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(png)
        manifest.append(dict(**task,path=trace.relative_to(out).as_posix(),sha256=sha(trace),image_path=png.relative_to(out).as_posix(),image_sha256=sha(png),
            seconds=seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(),weights_sha256=sha(out/'weights.npz')))
        if (i+1)%12==0:print({'phase':'generate','done':i+1,'total':192,'seconds':round(time.perf_counter()-start,2)},flush=True)
    require(calls[0]==5862 and len(unet.conv_out._forward_pre_hooks)==hooks_before,'Total call/hook counts')
    count_handle.remove();after=base_sha(unet);actual=models.adapter_state(unet)
    require(before==after and all(torch.equal(actual[k],expected[k]) for k in expected),'Model unchanged')
    require(all(p.grad is None and not p.requires_grad for p in unet.parameters()),'No gradients')
    save(out/'generation_manifest.json',manifest)
    save(out/'generation.json',dict(complete=True,seconds=time.perf_counter()-start,images=192,unet_calls=calls[0],unet_examples=2*calls[0],backward=0,
        integration_calls=102,generation_calls=5760,vae_decodes=195,base_before=before,base_after=after,adapter_exact=True,
        scheduler_config=dict(sched.config),manifest_sha256=sha(out/'generation_manifest.json'),inputs_sha256=sha(out/'generation_inputs.pt'),
        generation_contract_sha256=sha(out/'generation_contract.json'),integration_sha256=sha(out/'integration.json')))
    # One contact sheet per8 prompt-latent cells. Method ordering is label-hidden,
    # hash-determined separately per cell; no image selection or retouching.
    for offset in range(0,64,8):
        grid=Image.new('RGB',(768,8*280),'white');draw=ImageDraw.Draw(grid)
        cells=[(sid,pid) for sid in c['generation_seeds'] for pid in c['prompts']][offset:offset+8]
        for i,(sid,pid) in enumerate(cells):
            rows=sorted([x for x in manifest if x['seed_id']==sid and x['prompt_id']==pid],key=lambda x:x['image_id'])
            for j,row in enumerate(rows):
                x=j*256;y=i*280;draw.text((x+3,y+4),row['image_id'],fill='black')
                with Image.open(out/row['image_path']) as im:grid.paste(im,(x,y+24))
        grid.save(out/f'blind_comparison_{offset//8+1}.png')
    print('MEDICAL_HEAD_GENERATION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a.out)
