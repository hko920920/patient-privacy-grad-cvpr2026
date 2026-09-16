"""Execute the frozen public640/749 E4/E8 plan without changing prior artifacts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import argparse,csv,gc,hashlib,inspect,json,time
from collections import Counter
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from u_patient_audit import models as models
from data_pipeline import nih_cxr14_model_input as mi
from frozen_residual_head.run_lora_positive_control import manual
import torch
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer,CLIPTextModel

CODE=Path(__file__).resolve().parents[1]
RESEARCH=CODE.parent/'CVPR 주제 탐색/research_2026-09-10'
PLAN=RESEARCH/'spec_sources/public_medical_backbone_plan_20260916_v1'
OUT=CODE/'_reports/public_medical_backbone_20260916_v1'
OLD=CODE/'_reports/frozen_residual_sampling_integration_20260916_v1'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_torch(p,value):
    require(not p.exists(),'Existing tensor artifact '+str(p))
    temp=p.with_suffix(p.suffix+'.tmp');require(not temp.exists(),'Existing temporary artifact')
    torch.save(value,temp);temp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def base_sha(unet):
    h=hashlib.sha256()
    for n,p in unet.named_parameters():
        if '.lora_' in n:continue
        a=p.detach().cpu().contiguous().numpy()
        h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def check_sources(out):
    c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Changed bound source: '+p)
    lock=read(PLAN/'plan_lock.json')
    for p,h in lock['files'].items():require(sha(RESEARCH/p)==h,'Changed frozen plan: '+p)
    return c
def train_rows():return sorted((r for r in rows(PLAN/'images.csv') if r['backbone_role']=='train'),key=lambda r:r['image_id'])
def fresh_cache(out):
    report=read(out/'cache_report.json');require(sha(out/'cache.pt')==report['cache_sha256'],'Cache changed')
    cache=torch.load(out/'cache.pt',map_location='cpu',weights_only=True)
    ids={r['image_id'] for r in train_rows()}
    require(set(cache['latents'])==set(cache['train_prompts'])==ids,'Training cache whitelist mismatch')
    return cache

def prepare(out):
    require(not out.exists(),'Preserve previous experiment directory')
    spec=read(PLAN/'execution_spec.json');require(spec['status']=='PLANNED_NOT_EXECUTED','Unexpected plan')
    snap=models.snapshot()
    files=[Path(__file__),Path(__file__).with_name('verify_pilot.py'),Path(__file__).with_name('review_gate.py'),CODE/'u_patient_audit/models.py',
        CODE/'u_patient_audit/common.py',CODE/'data_pipeline/nih_cxr14_model_input.py',
        CODE/'frozen_residual_head/run_lora_positive_control.py',PLAN/'plan_lock.json',
        PLAN/'execution_spec.json',PLAN/'training_schedule.csv',PLAN/'generation_tasks.csv',PLAN/'images.csv',
        OLD/'contract.json',OLD/'trajectories/normal_base.npz',OLD/'trajectories/effusion_base.npz']
    for sub in ['unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors','vae/config.json',
        'vae/diffusion_pytorch_model.fp16.safetensors','text_encoder/config.json','text_encoder/model.fp16.safetensors',
        'tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/special_tokens_map.json','scheduler/scheduler_config.json']:
        files.append(snap/sub)
    files.extend(Path(inspect.getfile(cls)) for cls in (DDIMScheduler,models.DDPMScheduler,models.UNet2DConditionModel,AutoencoderKL,CLIPTextModel,VaeImageProcessor,models.cast_mixed_precision_params))
    for p in files:require(p.is_file(),'Missing required source '+str(p))
    out.mkdir(parents=True)
    save(out/'contract.json',dict(schema='public-medical-backbone-runtime/v1',created_utc=now(),
        plan_lock_sha256=sha(PLAN/'plan_lock.json'),source_sha256={str(p):sha(p) for p in files},
        snapshot=str(snap),plan_directory=str(PLAN),execution_spec_sha256=sha(PLAN/'execution_spec.json'),
        stage=2,track=1,non_DP=True,protected_role_training=False,private_checkpoints_loaded=False,
        generation_uses_frozen_verified_manual_function=True,plan_changes=False))
    check_sources(out)
    print(json.dumps({'phase':'prepared','contract_sha256':sha(out/'contract.json')}),flush=True)

def cache_phase(out):
    c=check_sources(out);require(not(out/'cache.pt').exists(),'Existing cache')
    models.setup();start=time.perf_counter();snap=Path(c['snapshot']);rr=train_rows()
    require(len(rr)==749 and len({r['patient_id'] for r in rr})==640,'Training allocation mismatch')
    all_records={r.image_id:r for r in mi.read_manifest(PLAN/'images.csv',partitions={'public_development'})}
    prompts={r['image_id']:mi.prompt_for_record(all_records[r['image_id']]) for r in rr}
    image_hashes={}
    for r in rr:
        path=CODE.parent/r['image_path'];h=sha(path);require(h==r['sha256'],'Image content changed')
        image_hashes[r['image_id']]=h
    tokenizer=CLIPTokenizer.from_pretrained(snap/'tokenizer',local_files_only=True)
    text=CLIPTextModel.from_pretrained(snap/'text_encoder',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).eval().to('cuda')
    all_prompts=sorted(set(prompts.values())|{r['prompt'] for r in rows(PLAN/'generation_tasks.csv')})
    hidden={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(0,len(all_prompts),8):
            batch=all_prompts[i:i+8]
            tokens=tokenizer(batch,padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
            h=text(tokens.input_ids.to('cuda'))[0].cpu()
            for prompt,value in zip(batch,h):hidden[prompt]=value.unsqueeze(0).contiguous()
    text_peak=torch.cuda.max_memory_allocated();del text;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).eval().to('cuda')
    latents={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(0,len(rr),4):
            subset=rr[i:i+4]
            pixels=torch.stack([mi.pil_to_normalized_tensor(mi.preprocess_path(CODE.parent/r['image_path'],256)[0]) for r in subset]).to('cuda',torch.float16)
            z=vae.encode(pixels).latent_dist.mode()*float(vae.config.scaling_factor)
            require(bool(torch.isfinite(z).all()),'Nonfinite VAE cache')
            for r,value in zip(subset,z.cpu()):latents[r['image_id']]=value.unsqueeze(0).contiguous()
            if i%128==0:print(json.dumps({'phase':'cache','images_done':min(i+4,749),'total':749}),flush=True)
    require(len(latents)==749,'Missing tail image')
    save_torch(out/'cache.pt',dict(latents=latents,hidden=hidden,train_prompts=prompts))
    save(out/'cache_report.json',dict(complete=True,cache_sha256=sha(out/'cache.pt'),train_ids=sorted(latents),
        train_patients=sorted({r['patient_id'] for r in rr}),seconds=time.perf_counter()-start,
        source_images_sha256=image_hashes,embeddings_count=len(hidden),vae_records=749,vae_batch_calls=188,
        peak_memory_bytes=max(text_peak,torch.cuda.max_memory_allocated()),contract_sha256=sha(out/'contract.json'),
        protected_data_cache_loaded=False))
    print('PUBLIC_CACHE_COMPLETE',flush=True)

def training(out):
    c=check_sources(out);require(not(out/'trace.jsonl').exists(),'No implicit resume or overwrite')
    models.setup();cache=fresh_cache(out);schedule=rows(PLAN/'training_schedule.csv')
    require(len(schedule)==5992,'Schedule length')
    unet=models.load_unet(None,training=True)
    params=[p for p in unet.parameters() if p.requires_grad]
    require(sum(p.numel() for p in params)==1659904 and all(p.dtype==torch.float32 for p in params),'LoRA dtype/count')
    require(all((not p.requires_grad and p.dtype==torch.float16) for n,p in unet.named_parameters() if '.lora_' not in n),'Frozen base required')
    initial=models.adapter_state(unet)
    require(all(torch.count_nonzero(p)==0 for n,p in initial.items() if 'lora_B' in n),'Fresh B must be zero')
    save_torch(out/'initial_adapter.pt',initial);before=base_sha(unet)
    sched=models.scheduler();require(sched.config.prediction_type=='epsilon','Frozen epsilon objective required')
    first=schedule[:4]
    def inputs(batch):
        z=torch.cat([cache['latents'][r['image_id']] for r in batch]).to('cuda',torch.float16)
        h=torch.cat([cache['hidden'][cache['train_prompts'][r['image_id']]] for r in batch]).to('cuda',torch.float16)
        g=torch.Generator(device='cuda').manual_seed(int(batch[0]['noise_seed']))
        noise=torch.randn(z.shape,generator=g,device='cuda',dtype=torch.float16)
        t=torch.tensor([int(r['timestep']) for r in batch],device='cuda')
        return z,h,noise,t,sched.add_noise(z,noise,t)
    z,h,noise,t,noisy=inputs(first);unet.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        enabled=unet(noisy,t,encoder_hidden_states=h).sample
        unet.disable_adapters()
        try:disabled=unet(noisy,t,encoder_hidden_states=h).sample
        finally:unet.enable_adapters()
    require(torch.equal(enabled,disabled),'Zero-adapter prediction mismatch')
    with (out/'preflight.npz').open('xb') as f:np.savez_compressed(f,latents=z.cpu().numpy(),hidden=h.cpu().numpy(),noise=noise.cpu().numpy(),timesteps=t.cpu().numpy(),noisy=noisy.cpu().numpy(),pred_enabled=enabled.cpu().numpy(),pred_disabled=disabled.cpu().numpy())
    save(out/'preflight.json',dict(zero_B=True,enabled_disabled_exact=True,base_sha256=before,initial_adapter_sha256=sha(out/'initial_adapter.pt'),forward_calls=2,
        image_ids=[r['image_id'] for r in first],noise_seed=int(first[0]['noise_seed'])))
    del z,h,noise,t,noisy,enabled,disabled,initial;unet.train()
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    (out/'checkpoints').mkdir();exposures=Counter();losses=[];step=0;attempt=0;checkpoints={}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with (out/'trace.jsonl').open('x',encoding='utf-8') as log:
        while step<1498:
            batch=schedule[step*4:(step+1)*4];attempt+=1;opt.zero_grad(set_to_none=True)
            z,h,noise,t,noisy=inputs(batch)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=unet(noisy,t,encoder_hidden_states=h).sample
                loss=(pred.float()-noise.float()).square().mean()
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            scale_before=scaler.get_scale();scaler.step(opt);scaler.update();scale_after=scaler.get_scale()
            committed=scale_after>=scale_before
            norm_value=float(norm) if torch.isfinite(norm) else None
            entry=dict(step=step+1,attempt=attempt,committed=committed,image_ids=[r['image_id'] for r in batch],
                noise_seed=int(batch[0]['noise_seed']),timesteps=[int(r['timestep']) for r in batch],loss=float(loss.detach()),
                gradient_norm=norm_value,scale_before=scale_before,scale_after=scale_after,seconds=time.perf_counter()-start)
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if not committed:
                require(attempt-step<=100,'Too many failed optimizer attempts');continue
            require(norm_value is not None,'Nonfinite committed gradient')
            exposures.update(entry['image_ids']);losses.append(entry['loss']);step+=1
            if step in (749,1498):
                name='E4' if step==749 else 'E8';path=out/'checkpoints'/(name+'.pt')
                expected=4 if name=='E4' else 8
                require(len(exposures)==749 and set(exposures.values())=={expected},'Committed coverage mismatch')
                save_torch(path,dict(adapter=models.adapter_state(unet),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    step=step,attempt=attempt,exposures=dict(exposures),losses=list(losses),
                    contract_sha256=sha(out/'contract.json'),cache_sha256=sha(out/'cache.pt')))
                checkpoints[name]=dict(path=path.relative_to(out).as_posix(),sha256=sha(path))
            if step%100==0 or step in (1,749,1498):
                torch.cuda.synchronize()
                progress=dict(step=step,steps=1498,attempts=attempt,seconds=time.perf_counter()-start,
                    loss_last50=sum(losses[-50:])/len(losses[-50:]),peak_memory_bytes=torch.cuda.max_memory_allocated())
                # Progress is deliberately mutable; all scientific evidence uses append-only trace and snapshots.
                (out/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(json.dumps(progress),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start;after=base_sha(unet)
    require(before==after,'Frozen base mutated')
    save(out/'training_report.json',dict(complete=True,utc=now(),steps=step,attempts=attempt,
        unet_forward_calls=attempt,backward_calls=attempt,preflight_forward_calls=2,
        committed_image_presentations=4*step,attempted_image_presentations=4*attempt,
        seconds=seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(),base_sha256_before=before,base_sha256_after=after,
        initial_adapter_sha256=sha(out/'initial_adapter.pt'),checkpoints=checkpoints,trace_sha256=sha(out/'trace.jsonl'),
        cache_sha256=sha(out/'cache.pt'),contract_sha256=sha(out/'contract.json'),
        packages={k:version(k) for k in ('torch','diffusers','peft','transformers','numpy')},GPU=torch.cuda.get_device_name(),
        new_DP_claim=False,non_DP_public_only=True))
    print('PUBLIC_TRAINING_COMPLETE',flush=True)

def make_blind_grids(out,records,phase):
    relevant=sorted((r for r in records if r['phase']==phase),key=lambda r:r['image_id'])
    for offset in range(0,len(relevant),16):
        group=relevant[offset:offset+16];grid=Image.new('RGB',(1024,4*280),'white');draw=ImageDraw.Draw(grid)
        for i,r in enumerate(group):
            x=(i%4)*256;y=(i//4)*280
            grid.paste(Image.open(out/r['image_path']),(x,y+24));draw.text((x+4,y+4),r['image_id'],fill='black')
        grid.save(out/f'blind_{phase}_{offset//16+1}.png')

def generate(out,phase):
    c=check_sources(out);require(phase in ('selection','confirmation'),'Generation phase')
    require(not(out/f'manifest_{phase}.json').exists(),'Existing generated phase')
    training_report=read(out/'training_report.json')
    tv=read(out/'training_verification.json');require(tv['complete'] and tv['status'].startswith('PASS'),'Training verification required')
    if phase=='confirmation':
        decision=read(out/'selection_decision.json');require(decision['selected_checkpoint'] in ('E4','E8'),'No selected checkpoint')
        sv=read(out/'selection_verification.json')
        require(sv['complete'] and sv['status'].startswith('PASS'),'Verified selection required')
        require(decision['selection_verification_sha256']==sha(out/'selection_verification.json'),'Decision verification binding')
        require(decision['manifest_sha256']==sha(out/'manifest_selection.json'),'Selection manifest binding')
        for name,digest in decision['review_sha256'].items():require(sha(out/name)==digest,'Selection vote binding')
        choices=[decision['selected_checkpoint']]
    else:choices=['E4','E8']
    models.setup();cache=fresh_cache(out);snap=Path(c['snapshot'])
    raw_tasks=[r for r in rows(PLAN/'generation_tasks.csv') if r['phase']==phase]
    inputs={}
    for task in raw_tasks:
        gen=torch.Generator(device='cpu').manual_seed(int(task['seed']))
        inputs[task['task_id']]=torch.randn((1,4,32,32),generator=gen,dtype=torch.float32)
    if phase=='selection':
        old=read(OLD/'contract.json')
        for case in old['cases']:
            tid='diagnostic_'+case['case_id'];raw_tasks.append(dict(task_id=tid,phase='diagnostic',prompt=case['prompt']))
            inputs[tid]=torch.from_numpy(np.load(OLD/'trajectories'/(case['case_id']+'_base.npz'))['latents'][0]).clone()
    # Bind actual inputs once before seeing model outputs, including the diagnostic inputs.
    saved_inputs={tid:dict(initial=x,conditioning=cache['hidden'][next(t['prompt'] for t in raw_tasks if t['task_id']==tid)].float().clone()) for tid,x in inputs.items()}
    save_torch(out/f'generation_inputs_{phase}.pt',saved_inputs)
    (out/'traces').mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True)
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().to('cuda')
    scheduler=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True)
    processor=VaeImageProcessor(vae_scale_factor=8);records=[];model_guards={};start=time.perf_counter()
    for checkpoint in choices:
        cp=training_report['checkpoints'][checkpoint];path=out/cp['path'];require(sha(path)==cp['sha256'],'Checkpoint changed')
        unet=models.load_unet(path,training=False).float()
        expected_adapter=torch.load(path,map_location='cpu',weights_only=True)['adapter']
        loaded_adapter=models.adapter_state(unet)
        require(set(loaded_adapter)==set(expected_adapter) and all(torch.equal(loaded_adapter[k],expected_adapter[k]) for k in expected_adapter),'Actual generation adapter mismatch')
        require(all(v.dtype==torch.float32 for v in loaded_adapter.values()),'Generation adapter must remain FP32')
        frozen_before=base_sha(unet)
        pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=scheduler,image_processor=processor)
        for task in raw_tasks:
            tid=task['task_id'];saved=saved_inputs[tid]
            initial=saved['initial'].to('cuda');conditioning=saved['conditioning'].to('cuda')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            data=manual(pipe,initial,conditioning,1.,False)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            require(all(np.isfinite(a).all() for a in data.values()),'Nonfinite generation tensor')
            blind=hashlib.sha256(('public-medical-blind-v1|'+checkpoint+'|'+tid).encode()).hexdigest()[:12]
            trace=out/'traces'/(blind+'.npz');png=out/'images'/(blind+'.png')
            with trace.open('xb') as f:np.savez_compressed(f,**data)
            Image.fromarray(np.clip(data['image_float']*255,0,255).astype(np.uint8)).save(png)
            records.append(dict(image_id=blind,task_id=tid,checkpoint=checkpoint,phase=task['phase'],path=trace.relative_to(out).as_posix(),
                sha256=sha(trace),image_path=png.relative_to(out).as_posix(),image_sha256=sha(png),seconds=seconds,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=cp['sha256'],guidance=1.,precision='fp32_no_autocast',unet_calls=30,batch_examples=30))
            print(json.dumps({'phase':phase,'checkpoint':checkpoint,'task':tid,'seconds':round(seconds,3)}),flush=True)
        frozen_after=base_sha(unet);adapter_after=models.adapter_state(unet)
        require(frozen_before==frozen_after,'Generation mutated frozen base')
        require(all(torch.equal(adapter_after[k],expected_adapter[k]) for k in expected_adapter),'Generation mutated adapter')
        model_guards[checkpoint]=dict(adapter_loaded_exact=True,adapter_after_exact=True,adapter_dtype='float32',
            base_sha256_before=frozen_before,base_sha256_after=frozen_after)
        del pipe,unet,expected_adapter,loaded_adapter,adapter_after;gc.collect();torch.cuda.empty_cache()
    save(out/f'manifest_{phase}.json',records);make_blind_grids(out,records,phase)
    save(out/f'generation_{phase}.json',dict(complete=True,manifest_sha256=sha(out/f'manifest_{phase}.json'),
        contract_sha256=sha(out/'contract.json'),training_report_sha256=sha(out/'training_report.json'),
        records=len(records),unet_forward_calls=30*len(records),vae_decodes=len(records),seconds=time.perf_counter()-start,
        generation_inputs_sha256=sha(out/f'generation_inputs_{phase}.pt'),checkpoint_choices=choices,
        scheduler_config=dict(scheduler.config),model_guards=model_guards,
        selection_decision_sha256=sha(out/'selection_decision.json') if phase=='confirmation' else None))
    print('PUBLIC_GENERATION_'+phase.upper()+'_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','cache','train','selection','confirmation'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    if args.phase=='prepare':prepare(args.out)
    elif args.phase=='cache':cache_phase(args.out)
    elif args.phase=='train':training(args.out)
    else:generate(args.out,args.phase)
