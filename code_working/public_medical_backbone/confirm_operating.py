"""Once-only reserved-latent confirmation of the already nominated configuration."""
import argparse,gc,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from .run_pilot import CODE,RESEARCH,models,manual,require,sha,read,save,save_torch,base_sha,make_blind_grids,now

DEV=CODE/'_reports/public_operating_diagnostic_20260916_v1'
OUT=CODE/'_reports/public_operating_confirmation_20260916_v1'
PROTOCOL=RESEARCH/'TRACK1_PUBLIC_OPERATING_CONFIRMATION_PROTOCOL_20260916.md'
SALT='public-operating-confirmation-20260916-v1'

def prepare(out):
    require(not out.exists(),'Preserve existing confirmation')
    d=read(DEV/'contract.json');s=read(DEV/'development_summary.json');v=read(DEV/'verification.json')
    require(s['nominated_for_separate_confirmation']==['E4',7.5] and not s['confirmation_executed'],'Fixed candidate')
    require(v['complete'] and v['status'].startswith('PASS'),'Verified development')
    for name,key in [('manifest.json','manifest_sha256'),('review_root.json','review_sha256'),('verification.json','verification_sha256')]:
        require(sha(DEV/name)==s[key],'Development decision binding '+name)
    sources=dict(d['source_sha256'])
    for p,h in sources.items():require(sha(p)==h,'Original source changed '+p)
    for p in [Path(__file__),Path(__file__).with_name('verify_confirmation.py'),PROTOCOL]+[DEV/n for n in ('contract.json','development_summary.json','verification.json','inputs.pt','manifest.json','review_root.json')]:
        sources[str(p)]=sha(p)
    tasks=[]
    for sid in d['reserved_confirmation_seeds']:
        for pid in d['prompts']:
            iid=hashlib.sha256(f'{SALT}|{sid}|{pid}'.encode()).hexdigest()[:12]
            tasks.append(dict(image_id=iid,checkpoint='E4',guidance=7.5,seed_id=sid,prompt_id=pid,phase='confirmation'))
    out.mkdir(parents=True)
    save(out/'contract.json',dict(schema='public-operating-confirmation/v1',created_utc=now(),source_sha256=sources,
        development_directory=str(DEV),previous_directory=d['previous_directory'],snapshot=d['snapshot'],
        checkpoint='E4',guidance=7.5,prompts=d['prompts'],confirmation_seeds=d['reserved_confirmation_seeds'],tasks=tasks,
        numerical_policy=d['numerical_policy'],new_training=False,new_private_head=False,backward=0,
        expected_images=16,expected_UNet_calls=480,expected_UNet_examples=960,
        review=dict(reviewers=['root','independent'],image_pass='both pass',minimum_total=12,minimum_per_prompt=2,
            root_knows_candidate=True,labels_hidden=True,clinical=False,ambiguous_large_artifact='fail'),
        original_failure_preserved=True,adoption_requires_numerical_and_two_reviews=True))
    print('CONFIRMATION_PREPARED '+sha(out/'contract.json'),flush=True)

def run(out):
    c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Bound source changed '+p)
    require(not(out/'inputs.pt').exists(),'No implicit rerun')
    start=time.perf_counter();models.setup();snap=Path(c['snapshot']);previous=Path(c['previous_directory'])
    old=torch.load(DEV/'inputs.pt',map_location='cpu',weights_only=True)
    initials={sid:torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32) for sid,seed in c['confirmation_seeds'].items()}
    require(all(not torch.equal(a,b) for a in initials.values() for b in old['initials'].values()),'New actual latent tensors')
    conditional=old['conditional'];null=old['null']
    save_torch(out/'inputs.pt',dict(initials=initials,conditional=conditional,null=null,empty_token_ids=old['empty_token_ids']))
    vae=AutoencoderKL.from_pretrained(snap/'vae',torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True).float().eval().requires_grad_(False).to('cuda')
    scheduler=DDIMScheduler.from_pretrained(snap/'scheduler',local_files_only=True)
    require(scheduler.config.prediction_type=='epsilon','Fixed epsilon path')
    training=read(previous/'training_report.json');cp=training['checkpoints']['E4'];path=previous/cp['path']
    require(sha(path)==cp['sha256'],'Checkpoint binding')
    unet=models.load_unet(path,training=False).float();expected=torch.load(path,map_location='cpu',weights_only=True)['adapter']
    actual=models.adapter_state(unet)
    require(set(actual)==set(expected) and all(torch.equal(actual[k],expected[k]) and actual[k].dtype==torch.float32 for k in expected),'FP32 checkpoint exact')
    before=base_sha(unet);pipe=SimpleNamespace(unet=unet,vae=vae,scheduler=scheduler,image_processor=VaeImageProcessor(vae_scale_factor=8))
    (out/'traces').mkdir();(out/'images').mkdir();manifest=[]
    for task in c['tasks']:
        h=torch.cat([null,conditional[task['prompt_id']]],dim=0).to('cuda');z=initials[task['seed_id']].to('cuda')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        packet=manual(pipe,z,h,7.5,False)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        require(all(np.isfinite(a).all() for a in packet.values()),'Nonfinite packet')
        trace=out/'traces'/(task['image_id']+'.npz');png=out/'images'/(task['image_id']+'.png')
        with trace.open('xb') as f:np.savez_compressed(f,**packet)
        require(not png.exists(),'No PNG overwrite');Image.fromarray(np.clip(packet['image_float']*255,0,255).astype(np.uint8)).save(png)
        manifest.append(dict(**task,path=trace.relative_to(out).as_posix(),sha256=sha(trace),image_path=png.relative_to(out).as_posix(),
            image_sha256=sha(png),seconds=seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=cp['sha256']))
        if len(manifest)%4==0:print('CONFIRMATION_IMAGES '+str(len(manifest))+'/16',flush=True)
    after=base_sha(unet);actual=models.adapter_state(unet)
    require(before==after and all(torch.equal(actual[k],expected[k]) for k in expected),'Model mutation')
    save(out/'manifest.json',manifest);make_blind_grids(out,manifest,'confirmation')
    save(out/'blind_ids.json',sorted(x['image_id'] for x in manifest))
    save(out/'execution.json',dict(complete=True,utc=now(),seconds=time.perf_counter()-start,images=16,unet_calls=480,
        unet_examples=960,backward=0,vae_decodes=16,new_text_encoding_calls=0,new_training=False,
        contract_sha256=sha(out/'contract.json'),inputs_sha256=sha(out/'inputs.pt'),manifest_sha256=sha(out/'manifest.json'),
        scheduler_config=dict(scheduler.config),model_guard=dict(base_before=before,base_after=after,adapter_exact=True,
        adapter_dtype='float32',checkpoint_sha256=cp['sha256']),automatic_adoption=False))
    print('CONFIRMATION_GENERATION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    prepare(a.out) if a.phase=='prepare' else run(a.out)
