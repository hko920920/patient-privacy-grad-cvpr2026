"""Independent CPU audit of a saved eight-slot patient4092 intervention.

The unchanged saved-prediction mask arithmetic helper is hash-pinned. No
producer is imported and no model training or full UNet backward is rerun.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from .verify_masked_patient_training import (
    ROOT, RUN, PRIOR, read, sha, th, new_json, digest_values, rng_seed,
    exact, expected_frozen_base, mask_arithmetic,
)

HELPER = ROOT/'u_patient_audit/verify_masked_patient_training.py'
HELPER_SHA = 'a174b0a482de8598492df3ecb34a920354c5481554d0bd293519bb9e68ec19bf'
OUT = RUN/'baseline_screen_20260915/masked_patient_4092_training_v1'
REFERENCE = RUN/'baseline_screen_20260915/masked_patient_training_v1'
PATIENT = '4092'
EXPECTED_UPDATES = [37,61,271,444,553,632,837,845]
CHECKPOINT_STEPS = [36,250,1000]
STATUS = 'PASS_MASKED_PATIENT_4092_SAVED_TRAINING_AND_SOURCE_GATES'


def independent_batches():
    with (RUN/'cohort/model_1_train.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    rows.sort(key=lambda r:(r['eval_role']!='background',digest_values('training-order',r['patient_id']),r['image_id']))
    assert len(rows)==912
    batches=[];slots=[];exposures=Counter()
    for step in range(1000):
        epoch,offset=divmod(step*4,912)
        perm=torch.randperm(912,generator=torch.Generator().manual_seed(rng_seed('coverage-epoch',epoch)))
        batch=[rows[int(j)] for j in perm[offset:offset+4]]
        ts=[rng_seed('coverage-timestep',step,j)%1000 for j in range(4)]
        mask=[int(r['patient_id']!=PATIENT) for r in batch]
        ids=[r['image_id'] for r in batch]
        batches.append(dict(update=step+1,image_ids=ids,timesteps=ts,
            noise_seed=rng_seed('coverage-noise',step),control_mask=mask))
        exposures.update(ids)
        slots.extend((step+1,j,r['image_id'],ts[j]) for j,r in enumerate(batch) if not mask[j])
    assert [s[0] for s in slots]==EXPECTED_UPDATES and len(slots)==8
    assert slots[0][1]==1 and slots[0][3]==478
    assert Counter(s[2] for s in slots)=={r['image_id']:4 for r in rows if r['patient_id']==PATIENT}
    assert sum(sum(b['control_mask']) for b in batches)==3992
    return rows,batches,slots,exposures


def reference_evidence():
    v=read(REFERENCE/'verification.json');e=read(REFERENCE/'execution.json');p=read(REFERENCE/'protocol.json')
    assert v['status']=='PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC'
    assert v['complete'] is True and e['complete'] is True
    assert v['execution_sha256']==sha(REFERENCE/'execution.json')
    assert v['protocol_sha256']==sha(REFERENCE/'protocol.json')
    assert v['contract_sha256']==p['contract_sha256']==sha(p['contract_path'])
    report=read(REFERENCE/'treatment/execution.json')
    exact(report,e['branches'][0],'verified treatment report')
    for key,name in [('initial_state_sha256','initial_state.pt'),('committed_steps_sha256','committed_steps.json'),
                     ('attempts_sha256','attempts.json')]:
        assert report[key]==sha(REFERENCE/'treatment'/name)
    paths=[REFERENCE/n for n in ['verification.json','execution.json','protocol.json','conformance.json']]
    paths += [REFERENCE/'treatment'/n for n in ['initial_state.pt','execution.json','committed_steps.json','attempts.json',
                                             'step_0043.pt','step_0250.pt','step_1000.pt']]
    for step in [43,250,1000]:
        assert sha(REFERENCE/'treatment'/f'step_{step:04d}.pt')==v['checkpoint_sha256']['treatment'][str(step)]
    original=torch.load(PRIOR/'step_1000.pt',map_location='cpu',weights_only=True)
    replay=torch.load(REFERENCE/'treatment/step_1000.pt',map_location='cpu',weights_only=True)
    exact(original,replay,'previously verified treatment full1000')
    for step in [250,1000]:paths.append(PRIOR/f'step_{step:04d}.pt')
    paths += [PRIOR/'training_trace.jsonl',Path(p['contract_path'])]
    return v,original,paths


def check_state(state,step,contract,original_groups,batches):
    assert state['step']==step and len(state['losses'])==step
    assert all(math.isfinite(float(v)) for v in state['losses'])
    assert math.isfinite(float(state['scaler']['scale'])) and state['scaler']['scale']>0
    exact(state['contract'],contract['original_training_contract'],'original numerical contract')
    exact(state['optimizer']['param_groups'],original_groups,'original optimizer hyperparameters')
    assert len(state['adapter'])==len(state['optimizer']['state'])==256
    assert sum(v.numel() for v in state['adapter'].values())==1659904
    for value in state['adapter'].values():
        assert value.dtype==torch.float32 and bool(torch.isfinite(value).all())
    for value in state['optimizer']['state'].values():
        assert float(value['step'])==step
        assert value['exp_avg'].dtype==value['exp_avg_sq'].dtype==torch.float32
        assert value['exp_avg'].shape==value['exp_avg_sq'].shape
        assert bool(torch.isfinite(value['exp_avg']).all()) and bool(torch.isfinite(value['exp_avg_sq']).all())
        assert bool((value['exp_avg_sq']>=0).all())
    effective=Counter()
    for batch in batches[:step]:
        effective.update(i for j,i in enumerate(batch['image_ids']) if batch['control_mask'][j])
    exact(state['exposures'],dict(effective),f'actual effective exposure ledger at{step}')
    return dict(effective)


def check_attempts(report,attempts,commits,trace,states,batches,cache,old_commits,original_trace):
    assert len(commits)==1000 and len(attempts)==report['attempts']
    assert report['root_forward_calls']==report['backward_calls']==len(attempts)
    assert report['root_forward_examples']==4*len(attempts)
    assert len(trace)==1000
    count=0;overflows=0
    for ai,a in enumerate(attempts,1):
        assert a['attempt']==ai and a['update']==count+1 and a['step_zero_based']==count
        batch=batches[count]
        for key in ['image_ids','timesteps','noise_seed']:exact(a[key],batch[key])
        exact(a['mask'],batch['control_mask'])
        assert a['gradient_scale_before']==a['scale_before_step']
        assert a['overflow_retry']==(a['scale_after_step']<a['scale_before_step'])
        z=torch.cat([cache['latents'][i] for i in batch['image_ids']]).to(torch.float16)
        h=torch.cat([cache['hidden'][cache['train_prompts'][i]] for i in batch['image_ids']]).to(torch.float16)
        ts=torch.tensor(batch['timesteps'],dtype=torch.long)
        for key,value in [('latents',z),('hidden',h),('timesteps',ts)]:assert th(value)==a['input_sha256'][key]
        assert a['input_sha256']['noise']==a['input_sha256']['target']
        exact(a['input_sha256'],old_commits[count]['input_sha256'],'same treatment inputs')
        if a['overflow_retry']:
            assert a['scale_after_step']==.5*a['scale_before_step'];overflows+=1
            continue
        c=commits[count]
        for key,value in a.items():exact(value,c[key],f'committed{ai}/{key}')
        assert c['gradient_norm_finite'] is True and c['gradient_norm']>=0
        assert c['loss']==states[1000]['losses'][count]
        assert trace[count]['step']==count+1
        for key in ['attempt','loss','gradient_norm']:assert c[key]==trace[count][key]
        if count<36:
            for key in ['attempt','loss','gradient_norm','gradient_scale_before','scale_after_step']:
                exact(c[key],old_commits[count][key],'prefix36 trace')
            for key in ['attempt','loss','gradient_norm']:assert c[key]==original_trace[count][key]
        count+=1
    assert count==1000
    for step in CHECKPOINT_STEPS:
        state=states[step]
        assert state['attempt']==commits[step-1]['attempt']
        assert state['scaler']['scale']==commits[step-1]['scale_after_step']
    assert states[1000]['attempt']==report['attempts']
    return dict(committed_updates=count,actual_attempts=len(attempts),overflow_retries=overflows,
                input_hash_pairs_checked=1000,cached_latent_hidden_timestep_reconstructions=len(attempts),
                original_prefix_trace_exact_steps=36,old_full_checkpoint_at36_available=False)


def check_mask_packets(raw,attempts,commits,slots):
    updates={s[0] for s in slots}
    expected={str(a['attempt']) for a in attempts if a['update'] in updates}
    assert set(raw)==expected
    checks=[]
    for key,packet in raw.items():
        a=attempts[int(key)-1]
        assert packet['attempt']==a['attempt'] and packet['update']==a['update']
        exact(packet['mask'],a['mask'])
        assert th(packet['target'])==a['input_sha256']['target']
        assert packet['gradient_scale']==a['gradient_scale_before']
        if not a['overflow_retry']:assert packet['loss']==commits[packet['update']-1]['loss']
        checks.append(mask_arithmetic(packet,'control'))
    return checks


def self_test():
    assert sha(HELPER)==HELPER_SHA
    p=torch.linspace(-2,2,16384,dtype=torch.float16).reshape(4,4,32,32)
    t=torch.linspace(1,-1,16384,dtype=torch.float16).reshape_as(p)
    q=p.clone().requires_grad_();mask=torch.tensor([1,0,1,1]).reshape(4,1,1,1)
    loss=((q.float()-t.float()).square()*mask).mean();(loss*1024.).backward()
    mask_arithmetic(dict(prediction=p,target=t,mask=[1,0,1,1],loss=float(loss.detach()),
        gradient_scale=1024.,prediction_gradient=q.grad,update=37,attempt=37),'control')
    _,batches,slots,_=independent_batches()
    assert all(all(b['control_mask']) for b in batches[:36]) and batches[36]['control_mask']==[1,0,1,1]
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_CPU_4092_INDEPENDENT_SCHEDULE_AND_MASK_CHAIN',mask_updates=[s[0] for s in slots],
                masked_slots=8,effective_nonzero_exposures=3992,CUDA_initialized=False)


def check_source_contract(contract):
    old_contract=read(REFERENCE/'protocol.json')['contract']
    previous=Path(old_contract['preparation_dir'])/'transformed_training_kernel.py'
    current=Path(contract['preparation_dir'])/'transformed_training_kernel.py'
    old=ast.parse(previous.read_text('utf-8'));new=ast.parse(current.read_text('utf-8'))
    changes=Counter()
    class Restore(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==PATIENT:
                changes['patient']+=1
                return ast.copy_location(ast.Constant('14393'),node)
            if type(node.value) is int and node.value==36:
                changes['checkpoint']+=1
                return ast.copy_location(ast.Constant(43),node)
            return node
    restored=Restore().visit(new)
    assert changes=={'patient':1,'checkpoint':1}
    assert ast.dump(restored,include_attributes=False)==ast.dump(old,include_attributes=False)
    assert sha(current)==contract['transformed_kernel_sha256']
    assert sha(ROOT/'u_patient_audit/run_masked_patient_training.py')=='717960d8e6a4daedf6819dc100e2bcc7ce43577b1cddfedfd82d20951fdd43e8'
    policy_path=Path(contract['analysis_policy_path']);policy=read(policy_path)
    assert sha(policy_path)==contract['analysis_policy_sha256']
    assert policy['focal_patients']=={'p':'14393','q':PATIENT}
    assert contract['patient_id']==PATIENT and contract['branch_order']==['control']
    assert contract['steps_per_branch']==1000 and contract['checkpoints']==CHECKPOINT_STEPS
    assert contract['new_treatment_training'] is contract['resume'] is contract['prefix_full_checkpoint_exact_claim'] is False
    assert contract['original_treatment_reused'] is True and contract['control_prefix_gate']==36
    assert contract['input_hash_gate_updates']==1000
    assert contract['effective_exposures']=={'control':3992}
    assert contract['effective_patient_exposures']=={PATIENT:0,'14393':8}
    return dict(previous_rendered_kernel_sha256=sha(previous),current_rendered_kernel_sha256=sha(current),
                reverse_patient_and_checkpoint_literals_full_AST_exact=True,
                unchanged_numerical_kernel_scope='Saved transformed training loop and frozen mask helper; producer metadata/audit wrapper is separate',
                policy_sha256=sha(policy_path))


def run(args):
    started=time.perf_counter();out=args.run_dir
    assert sha(Path(__file__))==args.expected_code_sha256 and sha(HELPER)==HELPER_SHA
    execution=read(out/'execution.json');protocol=read(out/'protocol.json')
    assert execution['complete'] is True and execution['patient_id']==PATIENT
    assert execution['status']=='PASS_MASKED_PATIENT_4092_TRAINING_PENDING_INDEPENDENT_VERIFICATION'
    contract=protocol['contract'];contract_path=Path(protocol['contract_path'])
    assert sha(contract_path)==protocol['contract_sha256']==execution['contract_sha256']
    exact(contract,read(contract_path),'contract')
    assert sha(out/'protocol.json')==execution['protocol_sha256']
    assert sha(out/'conformance.json')==execution['conformance_sha256']
    assert protocol['code_sha256']==contract['frozen_sha256'][str((ROOT/'u_patient_audit/run_masked_patient_4092_training.py').resolve())]
    inputs=dict(contract['frozen_sha256'])
    files=['protocol.json','execution.json','conformance.json']
    files += ['control/'+n for n in ['execution.json','initial_state.pt','attempts.json','committed_steps.json',
        'masked_update_raw.pt','training_trace.jsonl']]
    files += [f'control/{prefix}{step:04d}{suffix}' for step in CHECKPOINT_STEPS
              for prefix,suffix in [('step_','.pt'),('conformance_at_','.json')]]
    for name in files:inputs[str((out/name).resolve())]=sha(out/name)
    rv,original,reference_paths=reference_evidence()
    for path in reference_paths+[contract_path,Path(__file__),HELPER]:inputs[str(path.resolve())]=sha(path)
    for path,value in inputs.items():assert sha(path)==value,path
    dest=out/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    new_json(dest/'protocol.json',dict(schema='masked-patient4092-independent-verification/v1',
        created_utc=datetime.now(timezone.utc).isoformat(),expected_code_sha256=args.expected_code_sha256,
        arithmetic_helper_sha256=HELPER_SHA,input_sha256=inputs,
        scope='Saved state, original1000 inputs and prediction-level mask derivative; CPU only',
        prefix36_scope='Trace agreement only; original full step36 checkpoint unavailable',
        loss_bound='Previously frozen conservative floating point bound; no data-chosen tolerance'))
    torch.set_num_threads(4)
    source_check=check_source_contract(contract)
    manifest,batches,slots,scheduled=independent_batches()
    exact(batches,read(Path(contract['preparation_dir'])/'scheduled_batches.json'),'independent schedule')
    exact(dict(scheduled),contract['scheduled_exposure_counts'])
    exact([dict(update=u,ordinal=j,image_id=i,timestep=t) for u,j,i,t in slots],contract['mask_slots'])
    expected_base=expected_frozen_base(contract)
    directory=out/'control';report=read(directory/'execution.json')
    assert len(execution['branches'])==1;exact(report,execution['branches'][0])
    assert report['status']=='PASS_BRANCH_EXECUTION_AND_CONFORMANCE' and report['branch']=='control'
    assert report['patient_id']==PATIENT and report['step']==1000
    for key,name in [('initial_state_sha256','initial_state.pt'),('attempts_sha256','attempts.json'),
                     ('committed_steps_sha256','committed_steps.json'),('masked_update_raw_sha256','masked_update_raw.pt')]:
        assert report[key]==sha(directory/name)
    exact(report['frozen_before'],expected_base,'base before')
    exact(report['frozen_after'],expected_base,'base after')
    assert report['frozen_parameter_versions_unchanged'] is report['frozen_parameter_gradients_none'] is True
    initial=torch.load(directory/'initial_state.pt',map_location='cpu',weights_only=True)
    old_initial=torch.load(REFERENCE/'treatment/initial_state.pt',map_location='cpu',weights_only=True)
    initial_counts=exact(initial,old_initial,'same adapter optimizer scaler initialization')
    assert report['initial_state_exact'] is True
    states={};exposures={}
    for step in CHECKPOINT_STEPS:
        path=directory/f'step_{step:04d}.pt'
        assert sha(path)==report['checkpoint_sha256'][str(step)]==execution['checkpoint_sha256'][str(step)]
        state=torch.load(path,map_location='cpu',weights_only=True);states[step]=state
        exposures[step]=check_state(state,step,contract,original['optimizer']['param_groups'],batches)
        saved=read(directory/f'conformance_at_{step:04d}.json')
        assert saved['saved_step']==step and saved['saved_checkpoint_sha256']==sha(path)
        assert saved['full_original_checkpoint_comparison'] is False
        assert saved['gates']==report['conformance']
    commits=read(directory/'committed_steps.json');attempts=read(directory/'attempts.json')
    trace=[json.loads(s) for s in (directory/'training_trace.jsonl').read_text('utf-8').splitlines()]
    old_commits=read(REFERENCE/'treatment/committed_steps.json')
    original_trace=[json.loads(s) for s in (PRIOR/'training_trace.jsonl').read_text('utf-8').splitlines()]
    cache=torch.load(RUN/'cache/cache.pt',map_location='cpu',weights_only=True)
    input_check=check_attempts(report,attempts,commits,trace,states,batches,cache,old_commits,original_trace)
    raw=torch.load(directory/'masked_update_raw.pt',map_location='cpu',weights_only=True)
    mask_checks=check_mask_packets(raw,attempts,commits,slots)
    q_images=[r['image_id'] for r in manifest if r['patient_id']==PATIENT]
    p_images=[r['image_id'] for r in manifest if r['patient_id']=='14393']
    assert len(q_images)==len(p_images)==2
    final=states[1000]['exposures']
    assert all(final.get(i,0)==0 for i in q_images) and sum(final.get(i,0) for i in p_images)==8
    expected=dict(original['exposures']);assert [expected.pop(i) for i in q_images]==[4,4]
    exact(final,expected,'q only removed; other patients unchanged exposure')
    assert sum(final.values())==report['effective_nonzero_exposures']==execution['effective_nonzero_exposures']==3992
    assert len(final)==report['effective_exposed_images']==910
    assert report['scheduled_committed_slots']==execution['scheduled_committed_slots']==4000
    assert report['patient_exposures']==0 and report['other_focal_patient_exposures']=={'14393':8}
    assert execution['effective_patient_exposures']=={PATIENT:0,'14393':8}
    assert execution['committed_updates']==1000 and execution['new_treatment_training'] is False
    assert execution['forward_examples']==report['root_forward_examples']
    assert execution['backward_calls']==report['backward_calls']
    assert report['input_hash_pairs_exact']==1000 and report['prefix_trace_exact_updates']==36
    assert report['prefix_full_checkpoint_exact_claim'] is False
    gates=read(out/'conformance.json')
    assert gates['status']=='PASS_REUSED_TREATMENT_INITIAL_STATE_AND_Q_PREFIX_TRACE'
    assert gates['gates']==report['conformance']
    assert [g['gate'] for g in gates['gates']]==['initial_adapter_optimizer_scaler_and_base','first36_committed_trace_exact']
    assert all(g['exact'] is True for g in gates['gates'])
    assert gates['prefix_full_checkpoint_exact_claim'] is False
    assert gates['gates'][1]['full_checkpoint_comparison'] is gates['gates'][1]['original_step36_checkpoint_available'] is False
    assert gates['reused_treatment_verification_sha256']==execution['reused_treatment_verification_sha256']==sha(REFERENCE/'verification.json')
    for path,value in inputs.items():assert sha(path)==value,path
    result=dict(status=STATUS,complete=True,current_step=2,patient_id=PATIENT,seconds=time.perf_counter()-started,
        source_code_sha256=args.expected_code_sha256,arithmetic_helper_sha256=HELPER_SHA,
        verification_protocol_sha256=sha(dest/'protocol.json'),execution_sha256=sha(out/'execution.json'),
        protocol_sha256=sha(out/'protocol.json'),contract_sha256=sha(contract_path),
        checkpoint_sha256=report['checkpoint_sha256'],source_contract_check=source_check,
        exact_initial_state_check=initial_counts,input_check=input_check,mask_arithmetic_checks=mask_checks,
        scheduled_exposures=4000,effective_exposures=3992,patient_exposures={PATIENT:0,'14393':8},
        forward_examples=execution['forward_examples'],backward_calls=execution['backward_calls'],
        expected_frozen_base_from_artifact=expected_base,CUDA_initialized=torch.cuda.is_initialized(),
        limitations=['No independently recomputed full UNet gradient or1000-step control training.',
            'Initial full state is exact; prefix36 agreement is only saved per-step traces. No old full step36 checkpoint exists.',
            'CUDA FP16 noise is not independently regenerated on CPU; all1000 input hashes match verified treatment and8 mask targets are bound.',
            'Runtime base fingerprints and versions are producer observations checked against the original base artifact.',
            'Single fixed intervention; no endpoint membership performance, population causal claim or privacy guarantee.'])
    assert result['CUDA_initialized'] is False
    new_json(dest/'verification.json',result);new_json(out/'verification.json',result)
    print(json.dumps(dict(status=STATUS,seconds=result['seconds'],mask_cells=len(mask_checks),
                         input_hash_pairs_checked=1000,actual_attempts=input_check['actual_attempts'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--output-tag',default='verification_v1');parser.add_argument('--expected-code-sha256')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--source-preflight',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return
    if args.source_preflight:
        assert sha(HELPER)==HELPER_SHA
        contract=read(OUT.parent/'masked_patient_4092_training_contract_v1.json')
        checked=check_source_contract(contract)
        _,batches,slots,counts=independent_batches()
        exact(batches,read(Path(contract['preparation_dir'])/'scheduled_batches.json'))
        exact(dict(counts),contract['scheduled_exposure_counts'])
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_CPU_4092_SOURCE_AND_INDEPENDENT_SCHEDULE',source=checked,
                              masked_slots=len(slots),CUDA_initialized=False)));return
    assert args.expected_code_sha256,'Freeze verifier SHA before actual execution'
    run(args)


if __name__=='__main__':
    main()
