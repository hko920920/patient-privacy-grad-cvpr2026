"""Freeze bounded profile before patient pixels or new model outputs are consumed."""
from collections import Counter
from importlib.metadata import version
import numpy as np
from .common import *

def seed_numbers(obj,context=False):
    out=set()
    if isinstance(obj,dict):
        for k,v in obj.items():out|=seed_numbers(v,context or 'seed' in k.lower())
    elif isinstance(obj,list):
        for v in obj:out|=seed_numbers(v,context)
    elif context and isinstance(obj,int) and not isinstance(obj,bool):out.add(obj)
    return out

def prepare(out=OUT):
    require(not out.exists(),'Preserve any prior profile')
    plan=read(RESEARCH/'spec_sources/downstream_master_plan_20260917.json')
    resources=read(MASTER/'resources.json');require(not plan['final_ready'],'Final must remain closed')
    for key,p in resources['source_paths'].items():require(sha(p)==resources['source_sha256'][key],'Prior resource source '+key)
    inv={r['image_id']:r for r in rows(resources['source_paths']['inventory'])}
    public={r['image_id'] for r in rows(MASTER/'public_real_images_private.csv')}
    head=read(MEDICAL/'images.json');private={r['image_id'] for r in head if r['split']=='train'}
    assignments={r['patient_id']:r['role'] for r in rows(MASTER/'provisional_development_private.csv')}
    ledger=rows(CODE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv')
    deny={r['patient_id'] for r in ledger if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    reserved={r['patient_id'] for r in rows(resources['source_paths']['split']) if r['provisional_role']=='reserved_confirmation'}
    deny|=reserved;manifest=[]
    for image_id,r in sorted(inv.items()):
        pid=str(int(r['patient_id']))
        role='public' if image_id in public else 'private' if image_id in private else assignments.get(pid)
        if role is None:continue
        require(pid not in deny,'Locked or reserved patient')
        require(r['partition'] not in ('final_test','official_test_census_only'),'Locked image')
        manifest.append(dict(index=len(manifest),image_id=image_id,patient_id=pid,role=role,
            label=int('Pneumothorax' in r['finding_labels'].split('|')),path=str(RAW/image_id),
            sha256=r['sha256'].lower(),bytes=int(r['bytes']),width=int(r['width']),height=int(r['height']),mode=r['mode']))
    require(Counter(r['role'] for r in manifest)==dict(public=813,private=320,classifier_selection=5097,method_development=5047),'Role sizes')
    require(len(manifest)==11277,'Image inventory')
    med=read(MEDICAL/'contract.json');cp=Path(med['backbone_directory'])/'checkpoints/E4.pt'
    require(sha(cp)==med['checkpoint_sha256'],'E4 checkpoint')
    require(sha(MODEL_FILE)==MODEL_SHA,'ResNet18 official checkpoint')
    require(read(MEDICAL/'head_verification.json')['complete'] and read(SIGNAL/'verification.json')['complete'],'Verified existing heads')
    analysis=read(SIGNAL/'analysis.json')
    require(sha(SIGNAL/'analysis.json')==read(SIGNAL/'verification.json')['analysis_sha256'],'Private analysis verified')
    require(sha(SIGNAL/'analysis_weights.npz')==analysis['weights_sha256'],'Private weight archive binding')
    with np.load(MEDICAL/'weights.npz') as a,np.load(SIGNAL/'analysis_weights.npz') as z:
        # Independent CPU re-solves differ at ~1e-15; keep original production controls.
        weights={m:a[m].copy() for m in ('backbone','public','pooled')}
        weights['private_only']=z['private'].copy()
    targets=plan['synthetic']['prompts']
    witness=next(r for r in read(MEDICAL/'manifest.json') if r['split']=='train' and r['image_id']==med['witness_images']['train'] and r['draw_id']==0)
    require(sha(MEDICAL/witness['raw_path'])==witness['raw_sha256'],'Saved P conditioning/witness')
    with np.load(MEDICAL/witness['raw_path']) as z:
        require({'noisy','conditioning','features','basis'} <= set(z.files),'Complete witness schema')
    public_cache=Path(med['backbone_directory'])/'cache.pt'
    require(sha(public_cache)==read(Path(med['backbone_directory'])/'cache_report.json')['cache_sha256'],'Public cache binding')
    old_seed_sources=[MEDICAL/'contract.json',CODE/'_reports/public_operating_diagnostic_20260916_v1/contract.json',
        CODE/'_reports/public_operating_confirmation_20260916_v1/contract.json',
        CODE/'_reports/frozen_residual_sampling_integration_20260916_v1/contract.json',
        CODE/'_reports/frozen_residual_lora_positive_control_20260916_v1/contract.json',
        RESEARCH/'spec_sources/public_medical_backbone_plan_20260916_v1/execution_spec.json']
    old_seeds=set().union(*(seed_numbers(read(p)) for p in old_seed_sources))
    seeds={f'L{i}':seed('generation',i) for i in range(4)}
    require(not set(seeds.values())&old_seeds and len(set(seeds.values()))==4,'Fresh seeds')
    cells=[dict(cell=f'L{i}_pneumothorax',latent=f'L{i}',prompt='pneumothorax',label=1) for i in range(4)]
    cells += [dict(cell=f'L{i}_{p}',latent=f'L{i}',prompt=p,label=0) for i,p in enumerate(('normal','effusion','cardiomegaly'))]
    out.mkdir(parents=True)
    save_rows(out/'real_manifest_private.csv',manifest)
    with (out/'weights.npz').open('xb') as f:np.savez_compressed(f,**weights)
    sources=[*Path(__file__).parent.glob('*.py'),CODE/'_reports/downstream_profile_20260917_v1/failure_record.json',CODE/'_reports/downstream_profile_20260917_v2/failure_record.json',public_cache,Path(med['backbone_directory'])/'cache_report.json',RESEARCH/'spec_sources/downstream_master_plan_20260917.json',
        RESEARCH/'TRACK1_DOWNSTREAM_PROFILE_PROTOCOL_20260917.md',MASTER/'resources.json',MASTER/'verification.json',
        MASTER/'public_real_images_private.csv',MASTER/'provisional_development_private.csv',
        Path(resources['source_paths']['inventory']),Path(resources['source_paths']['remaining']),
        CODE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv',Path(resources['source_paths']['split']),
        MEDICAL/'contract.json',MEDICAL/'projection.npz',MEDICAL/'weights.npz',MEDICAL/'head_verification.json',
        SIGNAL/'analysis_weights.npz',SIGNAL/'analysis.json',SIGNAL/'verification.json',MEDICAL/witness['raw_path'],
        Path(med['confirmation_directory'])/'inputs.pt',MEDICAL/'generation_inputs.pt',
        CODE/'frozen_residual_head/medical_head_sampling.py',CODE/'frozen_residual_head/run_lora_positive_control.py',
        CODE/'public_medical_backbone/run_pilot.py',CODE/'u_patient_audit/models.py',CODE/'u_patient_audit/common.py',
        MODEL_FILE,cp,out/'real_manifest_private.csv',out/'weights.npz',*old_seed_sources]
    snap=Path(med['snapshot'])
    sources.extend(snap/part for part in ['unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors',
        'vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors','scheduler/scheduler_config.json'])
    save(out/'contract.json',dict(schema='bounded-downstream-profile/v1',created_utc=now(),stage=2,track=1,
        source_sha256={str(p):sha(p) for p in sources},master_plan=plan,manifest_images=len(manifest),
        expert_final_allowed=False,reserved_confirmation_allowed=False,final_ready=False,
        generation_seeds=seeds,previous_numeric_seeds=sorted(old_seeds),cells=cells,methods=list(METHODS),
        distinct_generation_images=28,replay_decodes=3,total_generation_decodes=31,maximum_generation_decodes=32,
        classifier_arms=list(ARMS),classifier_steps_per_run=7,classifier_repeats=2,total_optimizer_updates=98,maximum_optimizer_updates=100,
        classifier_seed=11,profile_lr=1e-4,profile_is_not_hyperparameter_selection=True,
        snapshot=med['snapshot'],checkpoint=str(cp),checkpoint_sha256=sha(cp),medical_config=med,
        conditioning_witness=witness,public_conditioning_cache=str(public_cache),conditioning_policy='Existing FP16-encoded FP32 conditions; P from frozen public backbone cache using exact predeclared prompt; complete designated train witness for replay',
        precision='FP32,no autocast,no TF32,deterministic algorithms',
        tolerance=dict(witness_atol=2e-6,witness_rtol=2e-6,zero='exact',classifier_replay='exact',guided='64*eps32*(1+abs(u)+15*abs(c)+15*abs(delta))'),
        packages={k:version(k) for k in ('torch','torchvision','numpy','Pillow','diffusers','transformers','peft')},
        output_interpretation='Execution/cost profile, no AUROC/AP, no utility gate and no DP'))
    print(json.dumps({'prepared':str(out),'manifest_images':len(manifest),'contract_sha256':sha(out/'contract.json')}),flush=True)
