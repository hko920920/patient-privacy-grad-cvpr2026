"""Budget-limited repair: all-arm batch audit and two single-update S1 replays."""
import sys,time
import numpy as np
import torch
from PIL import Image
from .common import *
from .data_v2 import source_pools,batch_records,letterbox
from .train_v2 import train_steps,initial_state,state_hash

REPAIR=OUT/'classifier_repair_v2'

def prepare_repair():
    c=check();require(not REPAIR.exists(),'Preserve repair attempt')
    failed=read(OUT/'classifier_verification_failure.json')
    require(failed['remaining_user_update_budget']==2,'Original98 consumed')
    REPAIR.mkdir()
    paths=[Path(__file__),Path(__file__).with_name('data_v2.py'),Path(__file__).with_name('train_v2.py'),
           Path(__file__).with_name('verify_profile_v2.py'),OUT/'classifier_verification_failure.json',
           OUT/'classifier.json',OUT/'data.json',OUT/'generation.json',OUT/'generation_manifest.json']
    sources={**c['source_sha256'],**{str(p):sha(p) for p in paths}}
    save(REPAIR/'contract.json',dict(created_utc=now(),source_sha256=sources,original_updates=98,
        additional_updates=2,total_update_cap=100,arm='S1',repeats=2,steps_per_repeat=1,
        all_arm_CPU_batch_audit_steps=7,original_classifier_gate='FAILED',
        planned_conclusion='Limited repair verification only; not a complete corrected seven-arm training replay.',
        tolerances_unchanged=True,efficacy_metrics=False,expert_access=False,reserved_access=False))
    print('REPAIR_CONTRACT_FROZEN_TWO_UPDATES',flush=True)

def execute_repair():
    setup();c=read(REPAIR/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Repair source mutation '+p)
    require(not (REPAIR/'initial.pt').exists(),'No repair rerun')
    real=rows(OUT/'real_manifest_private.csv');gm=read(OUT/'generation_manifest.json');synthetic=[]
    for r in real:r['array_key']='real'
    for i,r in enumerate(gm):
        with Image.open(OUT/r['image_path']) as im:synthetic.append(letterbox(im))
        r.update(index=i,array_key='synthetic',patient_id=r['cell'],role='synthetic_'+r['method'])
    sp=source_pools(real,gm);planned=[]
    for arm in ARMS:
        for step in range(7):
            batch=batch_records(sp,arm,11,step)
            require([r['role'] for r in batch[:16]]==['public']*16,'Correct real public half')
            other='public' if arm in ('R0','R1') else 'private' if arm=='Dreal' else 'synthetic_'+dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]
            require([r['role'] for r in batch[16:]]==[other]*16,'Correct second source')
            require(sum(int(r['label']) for r in batch)==16 and sum(int(r['label']) for r in batch[:16])==8,'Class balance')
            planned.append(dict(arm=arm,step=step,image_ids=[r['image_id'] for r in batch],roles=[r['role'] for r in batch],
                patient_ids=[r['patient_id'] for r in batch],labels=[int(r['label']) for r in batch]))
    save(REPAIR/'all_arm_batch_plan.json',planned)
    arrays={'real':np.load(OUT/'real224.npy',mmap_mode='r'),'synthetic':np.stack(synthetic)}
    init=initial_state(11);torch.save(init,REPAIR/'initial.pt');runs=[]
    for rep in (0,1):
        final,rec=train_steps(init,'S1',sp,arrays,1)
        rec.update(arm='S1',repeat=rep,initial_state_sha256=state_hash(init))
        save(REPAIR/f'S1_{rep}.json',rec);torch.save(final,REPAIR/f'S1_{rep}.pt');runs.append(rec)
    require(runs[0]['final_state_sha256']==runs[1]['final_state_sha256'],'Corrected repeat states exact')
    a,b=runs[0]['trace'][0],runs[1]['trace'][0]
    require(all(a[k]==b[k] for k in ('input_sha256','logits','loss','roles','image_ids')),'Corrected repeat exact')
    save(REPAIR/'result.json',dict(complete=True,corrected_optimizer_updates=2,total_optimizer_updates=100,
        all_arm_CPU_batch_audit=True,batch_plans=49,corrected_training_arm='S1',corrected_training_steps_per_repeat=1,
        corrected_exact_replay=True,original98_valid_as_requested_comparison=False,
        full_corrected_seven_arm_training_replay=False,expert_pixels=0,reserved_pixels=0,
        AUROC_AP_computed=False,new_generation_decodes=0,new_DP=False,contract_sha256=sha(REPAIR/'contract.json')))
    print('CORRECTED_TWO_UPDATES_EXACT_REPLAY_TOTAL100',flush=True)

if __name__=='__main__':
    (prepare_repair if sys.argv[1]=='prepare' else execute_repair)()
