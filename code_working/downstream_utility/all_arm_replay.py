"""Frozen14-update corrected integration replay. No generator/evaluation access."""
import sys,time,json
import numpy as np
import torch
from PIL import Image
from .common import CODE,OUT as PRIOR,RESEARCH,MODEL_FILE,MODEL_SHA,ARMS,METHODS,read,rows,save,sha,require,now,setup,tensor_sha
from . import data_v2,train_v2

OUT=CODE/'_reports/downstream_all_arm_replay_20260917_v1'
NAMES=['real/public','real/private']+['synthetic/'+m for m in METHODS]

def runtime():
    require(not {'downstream_utility.data','downstream_utility.train'}&set(sys.modules),'No legacy import')
    require(any(type(f).__name__=='RejectLegacy' for f in sys.meta_path),'Safe runner import guard required')
    modules={n:str(m.__file__) for n,m in sys.modules.items() if n.startswith('downstream_utility') and getattr(m,'__file__',None)}
    modules['entrypoint']=str(Path(sys.modules['__main__'].__file__).resolve())
    require(Path(modules['entrypoint']).name=='run_v2.py','Corrected entrypoint only')
    return modules

from pathlib import Path

def prepare():
    modules=runtime();require(not OUT.exists(),'No contract overwrite')
    previous=read(PRIOR/'verification_v2.json');require(previous['generator_gate']=='PASS' and previous['original_classifier_gate']=='FAIL','Prior outcome preserved')
    archive=PRIOR/'frozen_source_before_safe_entry_20260917/archive_manifest.json'
    archived=read(archive)['files'];resolved=[]
    for contract in [PRIOR/'contract.json',PRIOR/'classifier_repair_v2/contract.json']:
        for name,h in read(contract)['source_sha256'].items():
            p=Path(name)
            if sha(p)!=h:
                require(name in archived and archived[name]['sha256']==h,'Historical source archive '+name)
                require(sha(Path(archived[name]['archive']))==h,'Archived bytes match')
                resolved.append(dict(original=name,archive=archived[name]['archive'],sha256=h))
    current=Path(__file__).parent
    sources=[current/n for n in ('run.py','run_v2.py','common.py','data_v2.py','train_v2.py','all_arm_replay.py','verify_all_arm_replay.py')]
    sources += [MODEL_FILE,archive,RESEARCH/'TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_PROTOCOL_20260917.md',
        PRIOR/'contract.json',PRIOR/'verification_v2.json',PRIOR/'data.json',PRIOR/'real_manifest_private.csv',
        PRIOR/'real224.npy',PRIOR/'image_audit.json',PRIOR/'generation_manifest.json',PRIOR/'generation.json',
        PRIOR/'classifier_verification_failure.json',PRIOR/'classifier_initial.pt',
        CODE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv',
        CODE/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv']
    for row in read(PRIOR/'generation_manifest.json'):
        p=PRIOR/row['image_path'];require(sha(p)==row['image_sha256'],'Profile generated image binding');sources.append(p)
    require(sha(PRIOR/'real224.npy')==read(PRIOR/'data.json')['cache_sha256'],'Raw cache binding')
    require(sha(MODEL_FILE)==MODEL_SHA,'Fixed ImageNet weights')
    OUT.mkdir()
    save(OUT/'contract.json',dict(schema='corrected-all-arm-one-step-integration/v1',created_utc=now(),
        source_sha256={str(p):sha(p) for p in sources},historical_source_resolution=resolved,
        runtime_modules=modules,namespaces=NAMES,arms=list(ARMS),classifier_seed=11,steps=1,repeats=2,
        total_optimizer_updates=14,prior_separate_profile_updates=100,new_generation=0,
        lr=1e-4,weight_decay=1e-4,precision='FP32 deterministic no TF32',
        exact_replay_required=True,independent_BCE_atol=2e-6,input_normalization='ImageNet mean/std',
        expert_pixels_allowed=False,reserved_pixels_allowed=False,AUROC_AP_allowed=False,DP_allowed=False,
        scope='All branches one optimizer step twice; not long-run convergence or efficacy validation'))
    print('CORRECTED_ALL_ARM_CONTRACT_FROZEN_14_UPDATES',flush=True)

def execute():
    setup();modules=runtime();c=read(OUT/'contract.json')
    amendment=read(OUT/'logging_fix_amendment.json') if (OUT/'logging_fix_amendment.json').exists() else None
    for p,h in c['source_sha256'].items():
        if amendment and p in amendment['source_corrections']:
            a=amendment['source_corrections'][p];require(a['old_sha256']==h and sha(Path(a['archive']))==h,'Preserved pre-fix source')
            require(sha(Path(p))==a['new_sha256'],'Amended source exact')
        else:require(sha(Path(p))==h,'Frozen source changed '+p)
    if amendment:
        require({p.name for p in (OUT/'runs').iterdir()}=={'R0_0'},'Resume only the single bound completed run')
        for name,h in amendment['completed_artifact_sha256'].items():require(sha(OUT/name)==h,'Completed artifact changed')
    else:
        require(not(OUT/'runs').exists(),'No implicit replay repeat');(OUT/'runs').mkdir()
    begin=time.perf_counter();real=rows(PRIOR/'real_manifest_private.csv');gm=read(PRIOR/'generation_manifest.json')
    cache=np.load(PRIOR/'real224.npy',mmap_mode='r');audits={r['image_id']:r for r in read(PRIOR/'image_audit.json')};synthetic=[]
    real=[r for r in real if r['role'] in ('public','private')]
    for r in real:r.update(array_key='real',namespace='real/'+r['role'],source_file_sha256=r['sha256'],array_sha256=audits[r['image_id']]['tensor_sha256'])
    for i,r in enumerate(gm):
        p=PRIOR/r['image_path'];require(sha(p)==r['image_sha256'],'Synthetic PNG SHA')
        with Image.open(p) as im:x=data_v2.letterbox(im)
        synthetic.append(x);r.update(index=i,array_key='synthetic',namespace='synthetic/'+r['method'],patient_id=r['cell'],role='synthetic_'+r['method'],source_file_sha256=r['image_sha256'],array_sha256=tensor_sha(x))
    arrays={'real':cache,'synthetic':np.stack(synthetic)};pools=data_v2.source_pools(real,gm)
    require(set(pools)==set(NAMES),'Exactly six pool namespaces')
    planned={arm:data_v2.batch_records(pools,arm,11,0) for arm in ARMS}
    selected={r['image_id']:r for batch in planned.values() for r in batch};binding=[]
    for r in selected.values():
        p=Path(r['path']) if r['array_key']=='real' else PRIOR/r['image_path']
        require(sha(p)==r['source_file_sha256'],'Selected raw/PNG file SHA')
        with Image.open(p) as im:x=data_v2.letterbox(im)
        actual=arrays[r['array_key']][int(r['index'])]
        require(np.array_equal(x,actual) and tensor_sha(actual)==r['array_sha256'],'Actual raw/PNG to array index binding')
        binding.append(dict(image_id=r['image_id'],patient_id=r['patient_id'],role=r['role'],namespace=r['namespace'],
            label=int(r['label']),array_key=r['array_key'],index=int(r['index']),path=str(p),
            source_file_sha256=r['source_file_sha256'],source_pixel_sha256=tensor_sha(actual),
            method=r.get('method'),cell=r.get('cell')))
    plans={arm:[r['image_id'] for r in batch] for arm,batch in planned.items()}
    if amendment:
        require(read(OUT/'selected_input_bindings_private.json')==binding and read(OUT/'batch_plans_private.json')==plans,'Same source/array plans on resume')
    else:
        save(OUT/'selected_input_bindings_private.json',binding);save(OUT/'batch_plans_private.json',plans)
    initial=train_v2.initial_state(11);require(train_v2.state_hash(initial)==train_v2.state_hash(torch.load(PRIOR/'classifier_initial.pt',map_location='cpu',weights_only=True)),'Same historical ImageNet init')
    if amendment:require(train_v2.state_hash(initial)==train_v2.state_hash(torch.load(OUT/'initial.pt',map_location='cpu',weights_only=True)),'Same initial state on resume')
    else:torch.save(initial,OUT/'initial.pt')
    summaries=[];updates=0
    for arm in ARMS:
        for rep in (0,1):
            folder=OUT/'runs'/f'{arm}_{rep}'
            if amendment and arm=='R0' and rep==0:
                result=read(folder/'trace.json');first=result;updates+=1
                summaries.append({k:v for k,v in result.items() if k not in ('trace','parameter_names','changed_trainable_parameters')})
                continue
            folder.mkdir()
            final,result=train_v2.train_steps(initial,arm,pools,arrays,1,trace_directory=folder)
            updates+=1;result.update(arm=arm,repeat=rep,run_seed=11,runtime_modules=runtime())
            require(result['trace'][0]['image_ids']==[r['image_id'] for r in planned[arm]],'Actual selections equal bound plan')
            save(folder/'trace.json',result);torch.save(final,folder/'state.pt')
            summaries.append({k:v for k,v in result.items() if k not in ('trace','parameter_names','changed_trainable_parameters')})
            if rep==0:first=result
            else:
                require(first['final_state_sha256']==result['final_state_sha256'],'Exact state replay '+arm)
                for key in ('image_ids','patient_ids','roles','labels','input_sha256','logits','loss','source_namespaces','array_indices','source_pixel_sha256','source_file_sha256'):
                    require(first['trace'][0][key]==result['trace'][0][key],'Exact trace replay '+arm+'/'+key)
            print(json.dumps(dict(arm=arm,repeat=rep,updates=updates,seconds=result['seconds'])),flush=True)
    require(updates==14,'14 update cap')
    save(OUT/'result.json',dict(complete=True,status='PASS_PRODUCTION_CORRECTED_ALL_ARM_ONE_STEP_REPLAY',
        optimizer_updates=updates,backward_calls=updates,forward_calls=updates,new_generation=0,
        images_per_batch=32,patient_selection_policy='within class patient-uniform then image-uniform',
        runs=summaries,selected_source_images=len(binding),seconds=time.perf_counter()-begin,
        runtime_modules=modules,legacy_modules_imported=False,namespaces=NAMES,
        expert_pixels=0,reserved_pixels=0,AUROC_AP_computed=False,new_DP=0,final_ready=False,
        long_run_convergence_validated=False,private_utility_established=False,
        contract_sha256=sha(OUT/'contract.json'),logging_resume=bool(amendment),
        first_segment_optimizer_updates=1 if amendment else 0,resumed_optimizer_updates=13 if amendment else 0,
        logging_fix_amendment_sha256=sha(OUT/'logging_fix_amendment.json') if amendment else None))
    print('ALL14_CORRECTED_UPDATES_COMPLETE',flush=True)
