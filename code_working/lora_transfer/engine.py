from .common import *
import gc,hashlib
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from downstream_utility import data_v2,train_v2

def prepare():
    require(not OUT.exists(),'No implicit overwrite')
    plan=read(PLAN);old=read(OLD/'contract.json')
    for relative,digest in plan['source_sha256'].items():require(sha(CODE.parent/relative)==digest,'Plan dependency changed: '+relative)
    from downstream_utility.development_v1 import allowed_real
    rr=allowed_real();train=read(PLAN_DIR/'training_images_private.json');schedule=rows(PLAN_DIR/'training_schedule_private.csv')
    real_by_id={r['image_id']:r for r in rr}
    for r in train:
        role='public' if r['split']=='public' else 'private'
        require(r['split'] in ('public','train') and r['image_id'] in real_by_id and real_by_id[r['image_id']]['role']==role,'Training role binding')
        require(r['sha256']==real_by_id[r['image_id']]['sha256'],'Training file SHA binding')
    exposures={}
    for arm in ARMS:
        ss=[r for r in schedule if r['arm']==arm];ec=Counter(r['image_id'] for r in ss);pc=Counter(r['patient_id'] for r in ss)
        require(len(ss)==1792 and len(pc)==(32 if arm=='L_public' else 112),'Schedule allocation')
        require(set(pc.values())==({56} if arm=='L_public' else {16}),'Patient mass')
        for r in train:
            want=(28 if r['split']=='public' else 0) if arm=='L_public' else (8 if r['split']=='public' else 4)
            require(ec[r['image_id']]==want,'Image exposure')
        for step in range(1,449):
            b=ss[(step-1)*4:step*4];require([int(r['step']) for r in b]==[step]*4 and [int(r['slot']) for r in b]==list(range(4)),'Ordered four-slot batch')
            require(len({r['noise_seed'] for r in b})==1,'One explicit batch noise seed')
        exposures[arm]=dict(images=dict(ec),patients=dict(pc))
    pp=[r for r in schedule if r['arm']=='L_public'];po=[r for r in schedule if r['arm']=='L_pooled']
    shared=0
    for a,b in zip(pp,po):
        require(a['noise_seed']==b['noise_seed'] and a['timestep']==b['timestep'],'Paired training random inputs')
        if b['role']=='public':
            require(a['image_id']==b['image_id'],'Shared public slot');shared+=1
    require(shared==512,'Shared public presentations')
    dependencies={str(CODE.parent/p):h for p,h in plan['source_sha256'].items()}
    dependencies.update(old['source_sha256'])
    dependencies.update({str(x):sha(x) for x in Path(__file__).parent.glob('*.py')})
    for x in [PLAN,PROFILE/'real224.npy',PROFILE/'real_manifest_private.csv',PROFILE/'image_audit.json',CODE/'downstream_utility/run_v2.py',CODE/'downstream_utility/common.py',MODEL_FILE]:dependencies[str(x)]=sha(x)
    # Hash-bind every old run/prediction/PNG/trace used as a fixed comparison.
    for d in ('runs','evaluation','generation'):
        for x in (OLD/d).rglob('*'):
            if x.is_file():dependencies[str(x)]=sha(x)
    OUT.mkdir()
    save(OUT/'contract.json',dict(schema='lora-transfer-runtime/v1',created_utc=now(),plan_sha256=sha(PLAN),plan=plan,source_sha256=dependencies,
        checkpoint=old['checkpoint'],snapshot=old['snapshot'],expected_exposure=exposures,shared_public_slots=shared,
        extra_generation_decodes=2,classifier_preflight_updates=6,
        preflight_definition='Two E4 baseline image replays; one old S1 update through original provider and one through new provider; two new LoRA one-step updates per arm with exact replay. Six extra classifier updates total.',
        training_resume='Completed arms/cells/runs only with SHA seals. Partial optimizer trajectories require an explicit amendment; never silently restart.',
        namespaces=['real/public','synthetic/lora_public','synthetic/lora_pooled'],expert_final_allowed=False,reserved_allowed=False,DP_allowed=False,final_ready=False,runtime_modules=modules()))
    log(phase='runtime_contract_frozen',schedule_rows=len(schedule),shared_public_slots=shared)

def cache(role,c):
    folder=OUT/('cache_'+role)
    if not fresh(folder):return
    from u_patient_audit import models
    from data_pipeline import nih_cxr14_model_input as mi
    from transformers import CLIPTokenizer,CLIPTextModel
    from diffusers import AutoencoderKL
    start=time.perf_counter();snap=Path(c['snapshot']);rr=[r for r in read(PLAN_DIR/'training_images_private.json') if r['split']==('public' if role=='public' else 'train')]
    require(len(rr)==(64 if role=='public' else 320),'Cache exact selected source count')
    for r in rr:require(sha(r['path'])==r['sha256'],'Selected raw source changed')
    tokenizer=CLIPTokenizer.from_pretrained(snap/'tokenizer',local_files_only=True)
    text=CLIPTextModel.from_pretrained(snap/'text_encoder',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).eval().cuda().requires_grad_(False)
    hidden={};prompts={r['image_id']:r['prompt'] for r in rr};torch.cuda.reset_peak_memory_stats()
    # A fixed one-prompt batch makes shared prompt encoding identical across role caches.
    with torch.inference_mode():
        for prompt in sorted(set(prompts.values())):
            tok=tokenizer([prompt],padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
            hidden[prompt]=text(tok.input_ids.cuda())[0].cpu().contiguous()
    text_peak=torch.cuda.max_memory_allocated();del text;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).eval().cuda().requires_grad_(False)
    latents={};pixels={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(0,len(rr),4):
            batch=rr[i:i+4];a=[mi.pil_to_normalized_tensor(mi.preprocess_path(Path(r['path']),256)[0]) for r in batch]
            for r,x in zip(batch,a):pixels[r['image_id']]=tensor_sha(x)
            z=vae.encode(torch.stack(a).to('cuda',torch.float16)).latent_dist.mode()*float(vae.config.scaling_factor)
            require(bool(torch.isfinite(z).all()),'Finite VAE cache')
            for r,v in zip(batch,z.cpu()):latents[r['image_id']]=v.unsqueeze(0).contiguous()
    tsave(dict(latents=latents,hidden=hidden,train_prompts=prompts),folder/'cache.pt')
    save(folder/'report.json',dict(role=role,images=len(rr),patients=len({r['patient_id'] for r in rr}),seconds=time.perf_counter()-start,
        source_sha256={r['image_id']:r['sha256'] for r in rr},pixel_sha256=pixels,latent_sha256={k:tensor_sha(v) for k,v in latents.items()},
        hidden_sha256={k:tensor_sha(v) for k,v in hidden.items()},peak_allocated=max(text_peak,torch.cuda.max_memory_allocated()),peak_reserved=torch.cuda.max_memory_reserved(),access=audit_record(),broad_cache_loaded=False))
    seal(folder,list(folder.iterdir()));del vae;gc.collect();torch.cuda.empty_cache();log(phase='cache_complete',role=role,images=len(rr),seconds=time.perf_counter()-start)

def pipeline(checkpoint,c):
    from u_patient_audit import models
    from diffusers import AutoencoderKL,DDIMScheduler
    from diffusers.image_processor import VaeImageProcessor
    unet=models.load_unet(checkpoint,training=False).float().eval();unet.disable_gradient_checkpointing()
    require(not any(m._forward_hooks or m._forward_pre_hooks for m in unet.modules()),'Unexpected full64/other forward hook')
    vae=AutoencoderKL.from_pretrained(Path(c['snapshot'])/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().cuda().eval().requires_grad_(False)
    scheduler=DDIMScheduler.from_pretrained(Path(c['snapshot'])/'scheduler',local_files_only=True)
    require(scheduler.config.prediction_type=='epsilon','Fixed epsilon scheduler')
    return SimpleNamespace(unet=unet,vae=vae,scheduler=scheduler,image_processor=VaeImageProcessor(vae_scale_factor=8))

def e4_replay(c):
    folder=OUT/'e4_replay'
    if not fresh(folder):return
    from u_patient_audit import models
    from frozen_residual_head.run_lora_positive_control import manual
    inputs=torch.load(OLD/'generation_inputs.pt',map_location='cpu',weights_only=True);cell=c['plan']['generation']['cells'][0]
    old=np.load(OLD/'generation'/f"{cell['cell']}_backbone"/'trace.npz');records=[]
    state=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)['adapter']
    for arm in ARMS:
        pipe=pipeline(c['checkpoint'],c);actual=models.adapter_state(pipe.unet)
        require(set(actual)==set(state) and all(torch.equal(actual[n],state[n]) for n in state),'Exact learned E4 A/B initialization')
        require(any(torch.count_nonzero(x)>0 for n,x in state.items() if 'lora_B' in n),'Restored learned B, not reset zero')
        conditioning=torch.cat([inputs['null'],inputs['conditional'][cell['prompt']]]).float().cuda()
        tick=time.perf_counter();packet=manual(pipe,inputs['initials'][cell['latent']].float().cuda(),conditioning,7.5,False)
        for k in old.files:require(np.array_equal(packet[k],old[k]),'E4 baseline parity failed: '+k)
        npz(folder/(arm+'.npz'),**packet)
        records.append(dict(arm=arm,adapter_state_sha256=model_state_hash(actual),all_trace_arrays_exact=True,seconds=time.perf_counter()-tick))
        del pipe,actual,packet;gc.collect();torch.cuda.empty_cache()
    save(folder/'report.json',dict(status='PASS_E4_INITIAL_RESTORATION',records=records,decodes=2,access=audit_record()))
    seal(folder,list(folder.iterdir()));log(phase='E4_initial_replay_pass',independent_loads=2,exact=True)

def train(arm,c):
    folder=OUT/'training'/arm
    if not fresh(folder):return
    require(completed(OUT/'e4_replay'),'E4 initialization replay required')
    from u_patient_audit import models
    from public_medical_backbone.run_pilot import base_sha
    roles=['public'] if arm=='L_public' else ['public','private'];cache_data=dict(latents={},hidden={},train_prompts={})
    for role in roles:
        cf=OUT/('cache_'+role);require(completed(cf),'Sealed role cache')
        z=torch.load(cf/'cache.pt',map_location='cpu',weights_only=True)
        for k in cache_data:
            for name,value in z[k].items():
                if name in cache_data[k]:require(torch.equal(value,cache_data[k][name]) if torch.is_tensor(value) else value==cache_data[k][name],'Shared cache value changed')
                cache_data[k][name]=value
    ss=[r for r in rows(PLAN_DIR/'training_schedule_private.csv') if r['arm']==arm]
    require(set(cache_data['latents'])=={r['image_id'] for r in ss},'Cache schedule boundary')
    unet=models.load_unet(c['checkpoint'],training=True);params=[p for p in unet.parameters() if p.requires_grad]
    names=[n for n,p in unet.named_parameters() if p.requires_grad]
    require(sum(p.numel() for p in params)==1659904 and all('.lora_' in n for n in names),'Allowed A/B only')
    require(all(p.dtype==torch.float32 for p in params),'Trainable FP32')
    require(all(not p.requires_grad and p.dtype==torch.float16 for n,p in unet.named_parameters() if '.lora_' not in n),'Frozen base FP16')
    initial=models.adapter_state(unet);expected=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)['adapter']
    require(set(initial)==set(expected) and all(torch.equal(initial[n],expected[n]) for n in expected),'Arm starts independently from exact E4')
    before=base_sha(unet);sched=models.scheduler();require(sched.config.prediction_type=='epsilon' and sched.config.num_train_timesteps==1000,'Fixed training scheduler')
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.);require(len(opt.state)==0,'Fresh optimizer')
    step_calls=[0]
    def after_step(*_):step_calls[0]+=1
    handle=opt.register_step_post_hook(after_step)
    save(folder/'initial.json',dict(adapter_state_sha256=model_state_hash(initial),base_sha256=before,trainable_names=names,parameters=1659904,
        fresh_optimizer_state=0,scaler_initial=scaler.get_scale(),cache_roles=roles,cache_sha256={r:sha(OUT/('cache_'+r)/'cache.pt') for r in roles},runtime_modules=modules()))
    step=0;attempt=0;exposures=Counter();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (folder/'trace.jsonl').open('x',encoding='utf-8') as logfile:
        while step<448:
            batch=ss[step*4:(step+1)*4];attempt+=1;tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            z=torch.cat([cache_data['latents'][r['image_id']] for r in batch]).to('cuda',torch.float16)
            h=torch.cat([cache_data['hidden'][cache_data['train_prompts'][r['image_id']]] for r in batch]).to('cuda',torch.float16)
            generator=torch.Generator(device='cuda').manual_seed(int(batch[0]['noise_seed']))
            noise=torch.randn(z.shape,generator=generator,device='cuda',dtype=torch.float16)
            t=torch.tensor([int(r['timestep']) for r in batch],device='cuda');noisy=sched.add_noise(z,noise,t)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=unet(noisy,t,encoder_hidden_states=h).sample;loss=(pred.float()-noise.float()).square().mean()
            require(bool(torch.isfinite(loss)),'Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);scale_before=scaler.get_scale();calls_before=step_calls[0]
            scaler.step(opt);scaler.update();scale_after=scaler.get_scale();committed=step_calls[0]==calls_before+1
            require(committed==(scale_after>=scale_before),'AMP skip and optimizer post-hook disagree')
            actual_steps={int(v['step'].item()) for v in opt.state.values()}
            require(actual_steps=={step+int(committed)} if opt.state else not committed,'AdamW successful counter')
            if committed:require(bool(torch.isfinite(norm)),'Committed nonfinite gradient')
            torch.cuda.synchronize()
            entry=dict(step=step+1,attempt=attempt,committed=committed,image_ids=[r['image_id'] for r in batch],patient_ids=[r['patient_id'] for r in batch],roles=[r['role'] for r in batch],
                timesteps=[int(r['timestep']) for r in batch],noise_seed=int(batch[0]['noise_seed']),latent_sha256=tensor_sha(z),hidden_sha256=tensor_sha(h),noise_sha256=tensor_sha(noise),
                loss=float(loss.detach()),gradient_norm=float(norm) if torch.isfinite(norm) else None,scale_before=scale_before,scale_after=scale_after,
                optimizer_step_calls=step_calls[0],optimizer_state_steps=sorted(actual_steps),step_seconds=time.perf_counter()-tick)
            logfile.write(json.dumps(entry,allow_nan=False)+'\n');logfile.flush()
            if not committed:
                require(attempt-step<=16,'Skipped attempt cap');continue
            exposures.update(entry['image_ids']);step+=1
            if step in (224,448):
                tsave(dict(adapter=models.adapter_state(unet),optimizer=opt.state_dict(),scaler=scaler.state_dict(),step=step,attempt=attempt,exposures=dict(exposures),contract_sha256=sha(OUT/'contract.json')),folder/f'step_{step}.pt')
            if step in (1,10,20,100,200,224,300,400,448):log(phase='lora_train',arm=arm,step=step,total=448,attempts=attempt,seconds=time.perf_counter()-start,loss=entry['loss'],estimated_remaining=(time.perf_counter()-start)/step*(448-step))
    after=base_sha(unet);require(before==after,'Base parameters changed')
    require(dict(exposures)==c['expected_exposure'][arm]['images'],'Committed exposure totals')
    final=models.adapter_state(unet);changed=[n for n in final if not torch.equal(initial[n],final[n])];require(changed,'No adapter update')
    handle.remove();save(folder/'report.json',dict(arm=arm,successful_updates=step,attempts=attempt,skipped=attempt-step,optimizer_post_hook_calls=step_calls[0],seconds=time.perf_counter()-start,
        base_before=before,base_after=after,initial_adapter_sha256=model_state_hash(initial),final_adapter_sha256=model_state_hash(final),changed_tensors=changed,
        exposure=dict(exposures),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),access=audit_record()))
    seal(folder,list(folder.iterdir()));del unet,opt,params,cache_data;gc.collect();torch.cuda.empty_cache();log(phase='lora_arm_complete',arm=arm,steps=step)

def generate(c):
    from u_patient_audit import models
    from frozen_residual_head.run_lora_positive_control import manual
    require(all(completed(OUT/'training'/a) for a in ARMS),'Both frozen448 checkpoints required')
    if completed(OUT/'generation_done'):return
    inputs=torch.load(OLD/'generation_inputs.pt',map_location='cpu',weights_only=True);records=[];start=time.perf_counter()
    for arm in ARMS:
        pipe=pipeline(OUT/'training'/arm/'step_448.pt',c);before=model_state_hash(models.adapter_state(pipe.unet))
        for i,cell in enumerate(c['plan']['generation']['cells']):
            folder=OUT/'generation'/f"{cell['cell']}_{METHODS[arm]}"
            if completed(folder):records.append(read(folder/'record.json'));continue
            fresh(folder);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            z=inputs['initials'][cell['latent']].float().cuda();conditioning=torch.cat([inputs['null'],inputs['conditional'][cell['prompt']]]).float().cuda()
            packet=manual(pipe,z,conditioning,7.5,False);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            require(all(np.isfinite(v).all() for v in packet.values()),'Finite full trajectory')
            io=time.perf_counter();npz(folder/'trace.npz',**packet);Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(folder/'image.png')
            rec=dict(**cell,arm=arm,method=METHODS[arm],image_id=folder.name,path=str((folder/'trace.npz').relative_to(OUT)),image_path=str((folder/'image.png').relative_to(OUT)),
                sha256=sha(folder/'trace.npz'),image_sha256=sha(folder/'image.png'),initial_sha256=tensor_sha(z),conditioning_sha256=tensor_sha(conditioning),seconds=elapsed,output_IO_seconds=time.perf_counter()-io,
                adapter_state_sha256=before,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            save(folder/'record.json',rec);seal(folder,list(folder.iterdir()));records.append(rec)
            if (i+1)%16==0:log(phase='generation',arm=arm,done=i+1,total=128,seconds=time.perf_counter()-start)
        require(before==model_state_hash(models.adapter_state(pipe.unet)),'Inference mutated adapter')
        require(not any(p.grad is not None for p in pipe.unet.parameters()),'Unexpected inference gradient')
        del pipe;gc.collect();torch.cuda.empty_cache()
    save(OUT/'generation_manifest.json',records);done=OUT/'generation_done';done.mkdir();save(done/'report.json',dict(images=len(records),seconds=time.perf_counter()-start,full64_head=False,access=audit_record()))
    seal(done,[done/'report.json',OUT/'generation_manifest.json']);log(phase='generation_complete',images=len(records))

def load_classifier_inputs():
    from downstream_utility.development_v1 import allowed_real
    rr=allowed_real();audit={r['image_id']:r for r in read(PROFILE/'image_audit.json')};real=np.load(PROFILE/'real224.npy',mmap_mode='r')
    # Real classifier training and method-development only, no private/selection source pool.
    rr=[r for r in rr if r['role'] in ('public','method_development')]
    for r in rr:r.update(array_key='real',namespace='real/'+r['role'],source_file_sha256=r['sha256'],array_sha256=audit[r['image_id']]['tensor_sha256'])
    gm=read(OUT/'generation_manifest.json');images=[]
    for i,r in enumerate(gm):
        p=OUT/r['image_path'];require(sha(p)==r['image_sha256'],'New generated file SHA')
        with Image.open(p) as im:a=data_v2.letterbox(im)
        images.append(a);r.update(index=i,array_key='synthetic',namespace='synthetic/'+r['method'],patient_id=r['cell'],role='synthetic_'+r['method'],source_file_sha256=r['image_sha256'],array_sha256=tensor_sha(a))
    source={'real/public':data_v2.pools([r for r in rr if r['role']=='public'])}
    source.update({'synthetic/'+m:data_v2.pools([r for r in gm if r['method']==m]) for m in METHODS.values()})
    return rr,gm,dict(real=real,synthetic=np.stack(images)),source

def provider(source,arm,run_seed,step):
    if arm not in ARMS:return data_v2.batch_records(source,arm,run_seed,step)
    result=data_v2.draw(source['real/public'],16,seed(run_seed,step,'public-shared'))+data_v2.draw(source['synthetic/'+METHODS[arm]],16,seed(run_seed,step,'synthetic-extra'))
    require([r['namespace'] for r in result[:16]]==['real/public']*16,'Actual real public half')
    require([r['namespace'] for r in result[16:]]==['synthetic/'+METHODS[arm]]*16,'Actual new synthetic half')
    require(Counter(int(r['label']) for r in result)=={0:16,1:16},'Balanced batch')
    return result

@contextmanager
def injected_provider():
    original=train_v2.batch_records;require(original is data_v2.batch_records,'Unexpected starting provider')
    train_v2.batch_records=provider
    try:yield
    finally:train_v2.batch_records=original

def classifier_replay(c):
    folder=OUT/'classifier_replay'
    if not fresh(folder):return
    require(completed(OUT/'generation_done'),'Actual LoRA PNG bank required')
    from downstream_utility.development_v1 import load_inputs
    rr_old,gm_old,arrays_old,source_old=load_inputs(profile=True)
    rr,gm,arrays,source=load_classifier_inputs();evidence=[];init=train_v2.initial_state(11)
    # Two updates verify unchanged old computation when the new provider is installed.
    for repeat in range(2):
        sub=folder/f'old_S1_{repeat}';sub.mkdir()
        context=injected_provider() if repeat else __import__('contextlib').nullcontext()
        with context:state,trace=train_v2.train_steps(init,'S1',source_old,arrays_old,1,trace_directory=sub)
        save(sub/'trace.json',trace);tsave(state,sub/'state.pt');evidence.append(trace)
    reference=read(CODE/'_reports/downstream_all_arm_replay_20260917_v1/runs/S1_0/trace.json')
    fields=['loss','logits','labels','input_sha256','image_ids','patient_ids','roles','array_keys','array_indices','source_namespaces','source_file_sha256','source_pixel_sha256']
    for t in evidence:
        require(t['final_state_sha256']==reference['final_state_sha256'],'Old exact state parity')
        for k in fields:require(t['trace'][0][k]==reference['trace'][0][k],'Old exact runtime parity '+k)
    # Actual new source PNG/raw -> array -> GPU input, independently decoded below.
    new=[];image_lookup={r['image_id']:r for r in rr+gm}
    for arm in ARMS:
        traces=[]
        for repeat in range(2):
            sub=folder/f'{arm}_{repeat}';sub.mkdir()
            with injected_provider():state,trace=train_v2.train_steps(init,arm,source,arrays,1,trace_directory=sub)
            save(sub/'trace.json',trace);tsave(state,sub/'state.pt');traces.append(trace)
            for image_id,ph in zip(trace['trace'][0]['image_ids'],trace['trace'][0]['source_pixel_sha256']):
                r=image_lookup[image_id];path=Path(r['path']) if r['role']=='public' else OUT/r['image_path']
                require(sha(path)==r['source_file_sha256'],'Independent raw/PNG source file')
                with Image.open(path) as im:
                    im=im.convert('L');w,h=im.size;scale=224/max(w,h);nw=max(1,round(w*scale));nh=max(1,round(h*scale));canvas=Image.new('L',(224,224));canvas.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),((224-nw)//2,(224-nh)//2));a=np.asarray(canvas,dtype=np.uint8)
                require(tensor_sha(a)==ph,'Independent letterbox source binding')
        require(traces[0]['final_state_sha256']==traces[1]['final_state_sha256'],'New arm exact update replay')
        for k in fields:require(traces[0]['trace'][0][k]==traces[1]['trace'][0][k],'New arm exact '+k)
        new.append(dict(arm=arm,final_state_sha256=traces[0]['final_state_sha256'],changed_tensors=len(traces[0]['changed_trainable_parameters'])))
    # All 2400 planned batches pair exactly with old S1 public stream and cells.
    _,oldgm,oldarrays,oldsource=load_inputs(profile=False);paired=0
    for run_seed in SEEDS:
        for step in range(400):
            baseline=data_v2.batch_records(oldsource,'S1',run_seed,step)
            for arm in ARMS:
                selected=provider(source,arm,run_seed,step)
                require([r['image_id'] for r in selected[:16]]==[r['image_id'] for r in baseline[:16]],'All-step real draw parity')
                require([r['patient_id'] for r in selected[16:]]==[r['patient_id'] for r in baseline[16:]],'All-step synthetic cell parity')
                require([int(r['label']) for r in selected]==[int(r['label']) for r in baseline],'All-step label parity');paired+=1
    save(folder/'report.json',dict(status='PASS_OLD_KERNEL_AND_NEW_SOURCE_REPLAY',extra_updates=6,old_exact=True,new=new,paired_batch_plans=paired,access=audit_record(),runtime_modules=modules()))
    seal(folder,[p for p in folder.rglob('*') if p.is_file()]);log(phase='classifier_connection_pass',extra_updates=6,paired_plans=paired)

def classify(c):
    require(completed(OUT/'classifier_replay'),'Classifier integration required')
    from downstream_utility.development_v1 import observer,exposure
    rr,gm,arrays,source=load_classifier_inputs();records=[r for r in rr if r['role']=='public']+gm
    for run_seed in SEEDS:
        for arm in ARMS:
            folder=OUT/'runs'/f'{arm}_{run_seed}'
            if not fresh(folder):continue
            init=train_v2.initial_state(run_seed)
            old_initial=read(OLD/'runs'/f'S1_{run_seed}'/'started.json')['initial_state_sha256']
            require(train_v2.state_hash(init)==old_initial,'Identical classifier initial state')
            save(folder/'started.json',dict(arm=arm,seed=run_seed,initial_state_sha256=old_initial,steps=400,lr=.0001,created_utc=now(),runtime_modules=modules()))
            with injected_provider(),observer(folder,folder.name,(),400):state,trace=train_v2.train_steps(init,arm,source,arrays,400,run_seed=run_seed,lr=1e-4)
            trace.update(arm=arm,run_seed=run_seed);tsave(state,folder/'state_400.pt');save(folder/'trace.json',trace);save(folder/'exposure.json',exposure(trace['trace'],records))
            seal(folder,list(folder.iterdir()));log(phase='classifier_run_complete',run=folder.name,steps=400,seconds=trace['seconds'])

def evaluate(c):
    require(all(completed(OUT/'runs'/f'{a}_{s}') for a in ARMS for s in SEEDS),'All six complete before efficacy outputs')
    if (OUT/'result.json').exists():return
    from downstream_utility.development_v1 import predict,metrics
    rr,gm,arrays,source=load_classifier_inputs();evaluation=OUT/'evaluation';evaluation.mkdir(exist_ok=True);scores={}
    for arm in ARMS:
        scores[arm]=[]
        for s in SEEDS:
            path=evaluation/f'{arm}_{s}.npz'
            if path.exists():packet=dict(np.load(path));seconds=None
            else:
                packet,seconds=predict(OUT/'runs'/f'{arm}_{s}'/'state_400.pt',rr,arrays,'method_development');npz(path,**packet)
            scores[arm].append(dict(seed=s,**metrics(packet),inference_seconds=seconds,prediction_sha256=sha(path)))
        log(phase='development_inference_complete',arm=arm)
    old=read(OLD/'result.json');scores.update(old['scores'])
    means={a:{m:float(np.mean([r[m] for r in rr])) for m in ('AUROC','AP')} for a,rr in scores.items()}
    comparisons={}
    for b in ('L_public','R1','R0','S0'):
        ds=[a['AUROC']-z['AUROC'] for a,z in zip(scores['L_pooled'],scores[b])]
        comparisons[b]=dict(AUROC_delta=means['L_pooled']['AUROC']-means[b]['AUROC'],AP_delta=means['L_pooled']['AP']-means[b]['AP'],seed_AUROC_deltas=ds,positive_seeds=sum(x>0 for x in ds))
    passed=all(comparisons[b]['AUROC_delta']>=.01 and comparisons[b]['positive_seeds']>=2 and comparisons[b]['AP_delta']>=0 for b in ('L_public','R1','R0'))
    save(OUT/'result.json',dict(status='DEVELOPMENT_EFFICACY_COMPLETE_PENDING_INDEPENDENT_VERIFY',created_utc=now(),scores=scores,means=means,comparisons=comparisons,engineering_gate_passed=passed,
        contract_sha256=sha(OUT/'contract.json'),new_generator_updates=896,new_images=256,new_classifier_updates=2400,extra_classifier_updates=6,extra_generation_decodes=2,
        new_DP=0,expert_pixels=0,reserved_pixels=0,final_ready=False,current_full64_branch_closed=True,one_training_trajectory_per_arm=True,one_bank_per_arm=True,classifier_seeds=3,weak_label_development_only=True))
    log(phase='efficacy_result',means=means,comparisons=comparisons,gate=passed)

def dispatch(phase):
    if phase=='prepare':prepare();return
    c=validate_sources();setup();access_guard(phase)
    if phase.startswith('cache_'):cache(phase.removeprefix('cache_'),c)
    elif phase.startswith('train_'):train('L_'+phase.removeprefix('train_'),c)
    else:globals()[phase](c)
