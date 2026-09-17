"""28 paired profile images plus three exact-replay decodes, never a utility test."""
import gc,time
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from .common import *
from frozen_residual_head.medical_head_sampling import GuidedHead,sample
from public_medical_backbone.run_pilot import models,manual,base_sha

CORE=('conditioning','timesteps','latents','model_inputs','raw_eps','eps','decoded_raw','image_float','initial_from_prepare')
def npz(p,values):
    with p.open('xb') as f:np.savez_compressed(f,**values)

def generate(out=OUT):
    setup();c=check(out);require(not(out/'generation_inputs.pt').exists(),'No implicit generation rerun')
    tick_total=time.perf_counter();med=c['medical_config']
    old=torch.load(Path(med['confirmation_directory'])/'inputs.pt',map_location='cpu',weights_only=True)
    conditions={k:old['conditional'][k].clone() for k in ('normal','effusion','cardiomegaly')};null=old['null'].clone()
    witness=dict(np.load(MEDICAL/c['conditioning_witness']['raw_path']))
    public_cache=torch.load(c['public_conditioning_cache'],map_location='cpu',weights_only=True)
    prompts=c['master_plan']['synthetic']['prompts']
    conditions['pneumothorax']=public_cache['hidden'][prompts['pneumothorax']].float().clone()
    require(all(torch.equal(conditions[k],public_cache['hidden'][prompts[k]].float()) for k in ('normal','effusion','cardiomegaly')),'Same frozen public conditioning source')
    del public_cache
    require(torch.equal(null,torch.from_numpy(witness['conditioning'][:1])),'Null branch exact')
    initials={sid:torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(s),dtype=torch.float32) for sid,s in c['generation_seeds'].items()}
    previous=torch.load(MEDICAL/'generation_inputs.pt',map_location='cpu',weights_only=True)['initials']
    require(all(not torch.equal(a,b) for a in initials.values() for b in [*previous.values(),*old['initials'].values()]),'Fresh actual latent tensors')
    torch.save(dict(initials=initials,conditional=conditions,null=null),out/'generation_inputs.pt')
    snap=Path(c['snapshot']);unet=models.load_unet(Path(c['checkpoint']),training=False).float()
    expected=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)['adapter']
    loaded_adapter=models.adapter_state(unet)
    require(all(torch.equal(v,loaded_adapter[k]) for k,v in expected.items()),'Exact E4 load')
    before=base_sha(unet);calls=[0];counter=unet.register_forward_pre_hook(lambda m,a:calls.__setitem__(0,calls[0]+1))
    hooks_before=len(unet.conv_out._forward_pre_hooks)
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).cuda()
    sched=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True)
    require(sched.config.prediction_type=='epsilon','Frozen scheduler parameterization')
    pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=sched,image_processor=VaeImageProcessor(vae_scale_factor=8))
    p=dict(np.load(MEDICAL/'projection.npz'));w=dict(np.load(out/'weights.npz'))
    make=lambda method:GuidedHead(unet,p['P'],w[method],p['alphas'],med['time_basis_logsnr_min'],med['time_basis_logsnr_max'])
    # A saved real-noise witness verifies the previously unsampled private-only head.
    with torch.inference_mode(),make('private_only') as adapter:
        raw,eps=adapter.predict(torch.from_numpy(np.concatenate([witness['noisy']]*2)).cuda(),torch.tensor([int(witness['timestep'])],device='cuda'),torch.from_numpy(witness['conditioning']).cuda())
        require(np.allclose(adapter.last['features'],witness['features'],atol=2e-6,rtol=2e-6),'Offline/online feature replay')
        phi=(witness['basis'][None,:,None]*witness['features'].astype(np.float64)[:,None,:]).reshape(1024,64)
        residual=phi@w['private_only']
        require(np.allclose(residual,adapter.last['residual'],atol=2e-6,rtol=2e-6),'Offline/online private correction')
        packet=dict(original_features=witness['features'],original_basis=witness['basis'],offline_residual=residual,
            timestep=witness['timestep'],raw=raw.cpu().numpy(),eps=eps.cpu().numpy(),**adapter.last)
        npz(out/'private_witness.npz',packet)
    (out/'traces').mkdir();(out/'images').mkdir();manifest=[];reference_plain=None;reference_private=None
    for cell in c['cells']:
        for method in METHODS:
            z=initials[cell['latent']].cuda();h=torch.cat([null,conditions[cell['prompt']]],0).cuda()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            packet=manual(pipe,z,h,7.5,False) if method=='backbone' else sample(pipe,z,h,make(method))
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            require(all(np.isfinite(a).all() for a in packet.values()),'Finite generation')
            require(len(unet.conv_out._forward_pre_hooks)==hooks_before,'No hook leakage')
            stem=cell['cell']+'_'+method;trace=out/'traces'/(stem+'.npz');png=out/'images'/(stem+'.png')
            io_start=time.perf_counter();npz(trace,packet)
            Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(png)
            item=dict(**cell,method=method,image_id=stem,path=trace.relative_to(out).as_posix(),image_path=png.relative_to(out).as_posix(),
                sha256=sha(trace),image_sha256=sha(png),initial_sha256=tensor_sha(initials[cell['latent']]),conditioning_sha256=tensor_sha(h),
                seconds=elapsed,output_IO_seconds=time.perf_counter()-io_start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            manifest.append(item)
            if cell==c['cells'][0] and method=='backbone':reference_plain=packet
            if cell==c['cells'][0] and method=='private_only':reference_private=packet
        print(json.dumps({'phase':'generation','done':len(manifest),'total':28}),flush=True)
    first=c['cells'][0];z=initials[first['latent']].cuda();h=torch.cat([null,conditions[first['prompt']]],0).cuda()
    replays=[]
    for name,method in [('zero','backbone'),('restored',None),('private_replay','private_only')]:
        packet=manual(pipe,z,h,7.5,False) if method is None else sample(pipe,z,h,make(method))
        reference=reference_private if name=='private_replay' else reference_plain
        keys=reference.keys() if name=='private_replay' else CORE
        require(all(np.array_equal(packet[k],reference[k]) for k in keys),'Exact full-trajectory replay '+name)
        path=out/'traces'/(name+'.npz');npz(path,packet);replays.append(dict(kind=name,path=path.relative_to(out).as_posix(),sha256=sha(path)))
    after=base_sha(unet);actual=models.adapter_state(unet)
    require(before==after and all(torch.equal(expected[k],actual[k]) for k in expected),'Frozen model unchanged')
    require(calls[0]==931 and len(unet.conv_out._forward_pre_hooks)==hooks_before,'31 decodes plus one witness')
    require(all(p.grad is None and not p.requires_grad for p in unet.parameters()),'No generator backward')
    counter.remove();save(out/'generation_manifest.json',manifest)
    save(out/'generation.json',dict(complete=True,images=28,total_decodes=31,replays=replays,unet_calls=calls[0],unet_batch_examples=2*calls[0],
        backward=0,seconds=time.perf_counter()-tick_total,base_before=before,base_after=after,adapter_exact=True,
        private_offline_online=True,zero_identity=True,state_restoration=True,private_replay_exact=True,
        scheduler_config=dict(sched.config),weights_sha256=sha(out/'weights.npz'),manifest_sha256=sha(out/'generation_manifest.json'),
        inputs_sha256=sha(out/'generation_inputs.pt'),source_checkpoint_sha256=sha(c['checkpoint'])))
    grid=Image.new('RGB',(4*256,7*276),'white');d=ImageDraw.Draw(grid)
    for i,cell in enumerate(c['cells']):
        for j,method in enumerate(METHODS):
            r=next(x for x in manifest if x['cell']==cell['cell'] and x['method']==method)
            d.text((j*256+3,i*276+2),cell['cell']+' '+method,fill='black')
            with Image.open(out/r['image_path']) as im:grid.paste(im,(j*256,i*276+20))
    grid.save(out/'profile_grid.png')
    print('GENERATION_PROFILE_COMPLETE_28_IMAGES_31_DECODES',flush=True)
