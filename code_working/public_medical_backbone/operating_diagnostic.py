"""One fixed 2 checkpoint x 2 CFG x 4 latent x 4 prompt development experiment."""
import argparse,csv,gc,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .run_pilot import CODE,RESEARCH,models,manual,require,sha,read,save,save_torch,base_sha,make_blind_grids,now
import torch
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer,CLIPTextModel

PREVIOUS=CODE/'_reports/public_medical_backbone_20260916_v1'
OUT=CODE/'_reports/public_operating_diagnostic_20260916_v1'
SALT='public-operating-diagnostic-20260916-v1'
PROTOCOL=RESEARCH/'TRACK1_PUBLIC_OPERATING_PROTOCOL_20260916.md'
def seed(phase,index):return int(hashlib.sha256(f'{SALT}|{phase}|{index}'.encode()).hexdigest()[:15],16)
def blind(task):return hashlib.sha256(f"{SALT}|{task['checkpoint']}|{task['guidance']}|{task['seed_id']}|{task['prompt_id']}".encode()).hexdigest()[:12]
def prepare(out):
    require(not out.exists(),'Preserve an existing experiment')
    previous=read(PREVIOUS/'contract.json');require(read(PREVIOUS/'confirmation_decision.json')['passes_gate'] is False,'Original failed pilot must remain closed')
    sources=dict(previous['source_sha256'])
    for p,digest in sources.items():require(sha(p)==digest,'Old source changed: '+p)
    files=[Path(__file__),Path(__file__).with_name('verify_operating.py'),PROTOCOL]
    files += [PREVIOUS/n for n in ('contract.json','cache.pt','cache_report.json','training_report.json','training_verification.json',
        'checkpoints/E4.pt','checkpoints/E8.pt','confirmation_decision.json','adoption_status.json')]
    for p in files:require(p.is_file(),'Missing input '+str(p));sources[str(p)]=sha(p)
    with (Path(previous['plan_directory'])/'generation_tasks.csv').open(encoding='utf-8-sig',newline='') as f:old_tasks=list(csv.DictReader(f))
    prompt_ids=['generic','normal','effusion','cardiomegaly']
    prompts={k:next(t['prompt'] for t in old_tasks if t['prompt_id']==k) for k in prompt_ids}
    development={f'S{i}':seed('development',i) for i in range(4)}
    reserved={f'C{i}':seed('confirmation',i) for i in range(4)}
    all_new=list(development.values())+list(reserved.values())
    require(len(set(all_new))==8 and not set(all_new)&{int(t['seed']) for t in old_tasks},'Seed overlap')
    tasks=[]
    for cp in ('E4','E8'):
        for guidance in (1.,7.5):
            for sid in development:
                for pid in prompt_ids:
                    t=dict(checkpoint=cp,guidance=guidance,seed_id=sid,prompt_id=pid,phase='operating')
                    t['image_id']=blind(t);tasks.append(t)
    out.mkdir(parents=True)
    save(out/'contract.json',dict(schema='public-operating-diagnostic/v1',created_utc=now(),previous_directory=str(PREVIOUS),
        snapshot=previous['snapshot'],source_sha256=sources,prompts=prompts,development_seeds=development,
        reserved_confirmation_seeds=reserved,tasks=tasks,new_training=False,new_private_head=False,
        original_pilot_failure_unchanged=True,diagnostic_only=True,automatic_backbone_adoption=False,
        numerical_policy=dict(precision='float32',autocast=False,steps=30,eta=0.,width=256,height=256,
            conditional_first_step_batch_tolerance=dict(atol=1e-4,rtol=1e-4)),
        expected_UNet_calls=1920,expected_UNet_examples=2880,expected_images=64,expected_backward=0,
        review='one assistant; labels hidden until votes fixed; morphology only; candidate nomination, not adoption',
        candidate_priority=[['E4',7.5],['E8',7.5]],candidate_minimum_total=12,candidate_minimum_per_prompt=2))
    print(json.dumps(dict(phase='prepared',images=64,new_training=0,contract_sha256=sha(out/'contract.json'))),flush=True)
def check(out):
    c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Bound source changed: '+p)
    return c
def run(out):
    c=check(out);require(not(out/'inputs.pt').exists(),'No implicit repeat of development inputs')
    models.setup();snap=Path(c['snapshot']);start=time.perf_counter()
    cache=torch.load(PREVIOUS/'cache.pt',map_location='cpu',weights_only=True)
    conditional={pid:cache['hidden'][prompt].float().clone() for pid,prompt in c['prompts'].items()}
    tokenizer=CLIPTokenizer.from_pretrained(snap/'tokenizer',local_files_only=True)
    encoder=CLIPTextModel.from_pretrained(snap/'text_encoder',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).eval().requires_grad_(False).to('cuda')
    tokens=tokenizer([''],padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
    with torch.inference_mode():null=encoder(tokens.input_ids.to('cuda'))[0].float().cpu().clone()
    del encoder,cache;gc.collect();torch.cuda.empty_cache()
    initials={sid:torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(value),dtype=torch.float32) for sid,value in c['development_seeds'].items()}
    save_torch(out/'inputs.pt',dict(initials=initials,conditional=conditional,null=null,empty_token_ids=tokens.input_ids))
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).to('cuda')
    sched=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True)
    require(sched.config.prediction_type=='epsilon','Fixed epsilon path')
    processor=VaeImageProcessor(vae_scale_factor=8)
    (out/'traces').mkdir();(out/'images').mkdir();manifest=[];guards={};training=read(PREVIOUS/'training_report.json')
    for cp in ('E4','E8'):
        path=PREVIOUS/training['checkpoints'][cp]['path'];require(sha(path)==training['checkpoints'][cp]['sha256'],'Checkpoint hash')
        unet=models.load_unet(path,training=False).float();expected=torch.load(path,map_location='cpu',weights_only=True)['adapter']
        actual=models.adapter_state(unet);require(set(actual)==set(expected) and all(torch.equal(actual[k],expected[k]) for k in expected),'FP32 adapter exact')
        before=base_sha(unet);pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=sched,image_processor=processor)
        for t in (x for x in c['tasks'] if x['checkpoint']==cp):
            h=conditional[t['prompt_id']]
            if t['guidance']>1:h=torch.cat([null,h],dim=0)
            initial=initials[t['seed_id']].to('cuda');h=h.to('cuda')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            packet=manual(pipe,initial,h,t['guidance'],False)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            require(all(np.isfinite(a).all() for a in packet.values()),'Nonfinite generated tensor')
            trace=out/'traces'/(t['image_id']+'.npz');png=out/'images'/(t['image_id']+'.png')
            with trace.open('xb') as f:np.savez_compressed(f,**packet)
            require(not png.exists(),'Existing PNG');Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(png)
            manifest.append(dict(**t,path=trace.relative_to(out).as_posix(),sha256=sha(trace),image_path=png.relative_to(out).as_posix(),
                image_sha256=sha(png),seconds=seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(path)))
            if len(manifest)%8==0:print(json.dumps(dict(phase='generation',images_done=len(manifest),images=64)),flush=True)
        after=base_sha(unet);actual=models.adapter_state(unet)
        require(before==after and all(torch.equal(actual[k],expected[k]) for k in expected),'Model changed during generation')
        guards[cp]=dict(base_before=before,base_after=after,adapter_exact=True,adapter_dtype='float32',checkpoint_sha256=sha(path))
        del pipe,unet,actual,expected;gc.collect();torch.cuda.empty_cache()
    save(out/'manifest.json',manifest);make_blind_grids(out,manifest,'operating')
    save(out/'execution.json',dict(complete=True,utc=now(),seconds=time.perf_counter()-start,images=64,
        unet_calls=1920,unet_examples=2880,backward=0,vae_decodes=64,new_empty_text_embedding_calls=1,
        contract_sha256=sha(out/'contract.json'),inputs_sha256=sha(out/'inputs.pt'),manifest_sha256=sha(out/'manifest.json'),
        scheduler_config=dict(sched.config),model_guards=guards,new_training=False,automatic_adoption=False))
    print('OPERATING_DIAGNOSTIC_GENERATION_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run'));p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();prepare(args.out) if args.phase=='prepare' else run(args.out)
