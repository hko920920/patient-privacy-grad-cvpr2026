"""DINO-only control. Main loop retained from frozen feasibility_runtime.
Only objective/model selection and the one-bank scope validator change.
Shared export/prune/checkpoint/schedule/math remain the previously verified code.
"""
from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import require, sha, setup, now
from prrd_v3.datasets import manifests, PixelAccess
from prrd_v3.encoders import pil_tensor, state_hash
from prrd_v3.bank_runtime import (digest, tree_digest, atomic_json, append_event,
    file_lock, save_checkpoint, load_checkpoint, checkpoint_record)
from .runtime import OPTIMIZER, seed_all, finite_trace
from .schedule import ScheduledRenderer, ACTIVATIONS
from .second_source import DinoEncoder, ENCODER_ID, load_second
from .signals import bind_objective, objective_digest
from .objectives import two_pass
from .feasibility_runtime import export, prune, code_bindings as prior_code_bindings


def code_bindings():
    result=prior_code_bindings()
    result[str(Path(__file__))]=sha(__file__)
    return result


def load_dino(handoff,labels,population='pooled',device='cpu'):
    cs,hs,target,rule=load_second(handoff,population,device)
    objective=bind_objective(cs,{ENCODER_ID:hs},{ENCODER_ID:target},labels)
    require(set(objective.targets)=={ENCODER_ID},'DINO-only source selection')
    return objective,{ENCODER_ID:rule}


def validate_job(job):
    require(job['schema']=='receiver.dino-only-s200-bank/v1','Wrong control schema')
    require(job['id']=='dev_DINO_condk4_101_s200','Wrong bank')
    require(job['seed']==101 and job['images']==128 and job['microbatch']==16,'Frozen recipe changed')
    require(job['activation_updates']==list(ACTIVATIONS) and job['optimizer']==OPTIMIZER,'Schedule/optimizer changed')
    require(job['artifact_kind']=='MAIN_BANK' and job['updates']==200 and job['population']=='pooled','Invalid main bank')
    require(job['maximum_seconds']==3600,'One-bank time cap changed')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Bound code changed')
    require(sha(job['second_handoff'])==job['second_handoff_sha256'],'DINO handoff changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Contract changed')
    campaign=json.loads(Path(job['campaign_contract']).read_text(encoding='utf-8'))
    require(campaign['bank_id']==job['id'] and campaign['maximum_seconds']==job['maximum_seconds'],'Scope changed')
    require(all(sha(p)==h for p,h in campaign['historical_dependencies'].items()),'Historical binding changed')
    old=json.loads(Path(job['A2_job']).read_text(encoding='utf-8'))
    for key in ('seed','images','updates','microbatch','population','activation_updates','optimizer','templates','second_handoff','second_handoff_sha256'):
        require(job[key]==old[key],'Matched A2 contract changed: '+key)
    return (0,)*64+(1,)*64

def execute(job,directory,*,resume=False):
    started=time.monotonic(); wall_started=time.time()
    labels=validate_job(job)
    profile=job['artifact_kind']=='COST_ONLY'
    require(not(profile and resume),'No repeated cost run')
    directory=Path(directory)
    if resume: require(directory.is_dir(),'Missing resume directory')
    else: directory.mkdir(parents=True,exist_ok=False)
    ledger_path=directory/'time_budget.json'
    with file_lock(directory/'.process.lock'):
        require(not(directory/'COMPLETED.json').exists(),'Completed bank cannot rerun')
        charged=0.
        if resume:
            ledger=json.loads(ledger_path.read_text())
            charged=ledger['charged_seconds']
            if ledger['active']:
                # Conservative after abrupt termination: never reset elapsed allowance.
                charged+=max(0.,wall_started-ledger['attempt_wall_started'])
        cap=job.get('maximum_seconds',7200.)
        require(charged<cap,'Cumulative bank time cap exhausted')
        atomic_json(ledger_path,{'charged_seconds':charged,'active':True,
                    'attempt_wall_started':wall_started,'cap_seconds':cap},replace=resume)
        try:
            result=_run(job,directory,labels,profile,resume,started,started+cap-charged)
            result['worker_seconds']=time.monotonic()-started
            result['cumulative_worker_seconds']=charged+result['worker_seconds']
            require(result['cumulative_worker_seconds']<=cap,'Bank worker exceeded declared time cap')
            atomic_json(directory/('attempt_'+uuid.uuid4().hex+'.json'),result)
            return result
        finally:
            atomic_json(ledger_path,{'charged_seconds':charged+time.monotonic()-started,
                        'active':False,'attempt_wall_started':wall_started,'cap_seconds':cap},replace=True)


def _run(job,directory,labels,profile,resume,started,deadline):
    setup()
    objective,rules=load_dino(job['second_handoff'],labels,job['population'],'cuda')
    groups=manifests(); access=PixelAccess(groups,('P',)).install()
    lookup={r['image_id']:r for r in groups['P']}
    rows=[]
    for item in job['templates']:
        require(item['image_id'] in lookup and lookup[item['image_id']]['sha256']==item['sha256'],
                'Template is not the fixed public P image')
        rows.append(lookup[item['image_id']])
    require(len(rows)==128,'Template count')
    decoded={}
    for row in rows:
        if row['image_id'] not in decoded:
            decoded[row['image_id']]=TF.resize(pil_tensor(access.image(row)),[224,224],antialias=True)
    templates=torch.stack([decoded[r['image_id']] for r in rows]).cuda()
    encoders={ENCODER_ID:DinoEncoder().use_projection(rules[ENCODER_ID]['projection'])}
    fixed={name:state_hash(model) for name,model in encoders.items()}
    target_hash=objective_digest(objective)
    renderer=ScheduledRenderer(templates,'B',101)
    options={k:v for k,v in OPTIMIZER.items() if k!='name'}; options['betas']=tuple(options['betas'])
    optimizer=torch.optim.AdamW(renderer.parameters(),**options)
    seed_all(101)
    spec=dict(job,arm='B',template_tensor_digest=tree_digest(templates),
              fixed_encoder_sha256=fixed,objective_digest=target_hash,
              encoder_rules={n:digest(r) for n,r in rules.items()})
    require(spec['template_tensor_digest']==job['A2_template_tensor_digest'],'A2 template tensor changed')
    signature=digest(spec)
    if resume:
        require(json.loads((directory/'run_spec.json').read_text(encoding='utf-8'))==spec,'Resume dependencies changed')
        done,restored=load_checkpoint(directory,renderer,optimizer,signature)
    else:
        renderer.set_step(200 if profile else 1)
        done,restored=0,None
        atomic_json(directory/'run_spec.json',spec)
        if not profile:
            initial_record=save_checkpoint(directory,renderer,optimizer,0,signature)
            old_record=json.loads(Path(job['A2_initial_receipt']).read_text(encoding='utf-8'))
            old_path=Path(job['A2_bank'])/old_record['path']
            require(sha(old_path)==old_record['sha256'],'A2 initialization checkpoint changed')
            old_state=torch.load(old_path,map_location='cpu',weights_only=False)
            new_state=torch.load(directory/initial_record['path'],map_location='cpu',weights_only=False)
            for key in ('renderer','optimizer','rng','active_levels','requires_grad'):
                require(tree_digest(old_state[key])==tree_digest(new_state[key]),'Initialization differs: '+key)
            del old_state,new_state
            atomic_json(directory/'initialization_parity.json',{'status':'PASS_EXACT_A2_INITIALIZATION',
                'A2_checkpoint_sha256':old_record['sha256'],'fields':['renderer','optimizer','rng','active_levels','requires_grad']})
    prior_done=done; initial=state_hash(renderer)
    counts={n:{'forward_images':0,'backward_images':0} for n in encoders}
    handles=[]
    for name,model in encoders.items():
        def hook(module,inputs,output,name=name):
            n=len(inputs[0]); counts[name]['forward_images']+=n
            if output.requires_grad:
                def backward(g):
                    counts[name]['backward_images']+=n
                    return g
                output.register_hook(backward)
        handles.append(model.register_forward_hook(hook))
    torch.cuda.reset_peak_memory_stats()
    durations=[]; capped=False
    while done<job['updates']:
        reserve=max(60.,(max(durations[-3:]) if durations else 45.)*1.5)
        if time.monotonic()+reserve>=deadline:
            capped=True; break
        step=done+1
        renderer.set_step(200 if profile else step)
        torch.cuda.synchronize(); tick=time.perf_counter()
        trace=two_pass(renderer,encoders,objective,16)
        require(finite_trace(trace),'Nonfinite matching objective')
        active=[p for p in renderer.parameters() if p.requires_grad]
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in active),'Invalid image gradient')
        optimizer.step()
        require(all(torch.isfinite(p).all() for p in renderer.parameters()),'Nonfinite parameters')
        require(all(torch.isfinite(v).all() for s in optimizer.state.values()
                    for v in s.values() if isinstance(v,torch.Tensor)),'Nonfinite optimizer')
        torch.cuda.synchronize()
        duration=time.perf_counter()-tick; durations.append(duration); done=step
        append_event(directory/'updates.jsonl',{'at':now(),'successful_step':step,'active_levels':renderer.active,
                     'trace':trace,'seconds':duration,'worker_elapsed':time.monotonic()-started})
        if not profile and step%25==0:
            save_checkpoint(directory,renderer,optimizer,step,signature); prune(directory)
        if profile or step<=3 or step%5==0:
            print(json.dumps({'bank':job['id'],'completed':step,'active_levels':renderer.active,
                              'seconds':duration,'worker_elapsed':time.monotonic()-started}),flush=True)
    for handle in handles: handle.remove()
    require(all(state_hash(model)==fixed[n] for n,model in encoders.items()),'Frozen model changed')
    require(all(not p.requires_grad and p.grad is None for model in encoders.values() for p in model.parameters()),
            'Encoder parameter gradient')
    require(objective_digest(objective)==target_hash,'Target changed')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Runtime changed during training')
    expected={'forward_images':1024*(done-prior_done),'backward_images':512*(done-prior_done)}
    require(all(v==expected for v in counts.values()),'Actual per-encoder image counts differ')
    changed=state_hash(renderer)!=initial
    require(done==prior_done or changed,'Optimizer did not change images')
    png=None
    if not profile:
        record,_=checkpoint_record(directory)
        if record['completed_updates']!=done:
            record=save_checkpoint(directory,renderer,optimizer,done,signature); prune(directory)
        if done==200:
            png=export(directory,renderer,job,spec,signature,record)
            atomic_json(directory/'COMPLETED.json',{'status':'COMPLETE_MAIN_BANK','signature':signature,
                'completed_updates':200,'checkpoint_sha256':record['sha256'],
                'artifact_seal_sha256':sha(directory/'artifact/artifact_seal.json'),'main_reuse_allowed':True})
    return {'status':'COST_ONLY_COMPLETE' if profile and done==4 else ('COMPLETE' if done==200 else 'PAUSED'),
            'artifact_kind':job['artifact_kind'],'updates_this_process':done-prior_done,'completed_updates':done,
            'next_step':done+1,'active_levels':renderer.active,'restoration':restored,
            'source_unchanged':True,'target_unchanged':True,'parameter_updated':changed,
            'counts':counts,'access':access.report(),'PNG':png,'duration_seconds':durations,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'stopped_for_budget':capped,'signature':signature}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    result=execute(json.loads(args.job.read_text(encoding='utf-8')),args.output,resume=args.resume)
    atomic_json(args.result,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

