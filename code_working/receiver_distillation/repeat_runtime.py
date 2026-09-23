"""Explicit synthesis-seed adapter; frozen seed101 runtimes remain untouched.
The original execute/_run loop and export math are reused as a separate module.
Target condition/head seed is fixed independently of synthesis/template seed.
"""
from __future__ import annotations
import argparse,json,os,time,uuid
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import require,sha,setup,now
from prrd_v3.datasets import manifests,PixelAccess,png_roundtrip
from prrd_v3.encoders import Encoder,pil_tensor,state_hash
from prrd_v3.bank_runtime import (digest,tree_digest,atomic_bytes,atomic_json,append_event,file_lock,
    save_checkpoint,load_checkpoint,checkpoint_record,csv_bytes,image_layout,verify_artifact,capture)
from .runtime import OPTIMIZER,seed_all,finite_trace
from .schedule import ScheduledRenderer,ACTIVATIONS
from .second_source import DinoEncoder,ENCODER_ID,load_combined
from .signals import objective_digest
from .objectives import two_pass
from .dino_control_runtime import load_dino
from .feasibility_runtime import prune,code_bindings as previous_bindings


def code_bindings():
    result=previous_bindings()
    for name in ('dino_control_runtime.py','repeat_runtime.py'):
        p=Path(__file__).with_name(name);result[str(p)]=sha(p)
    return result


def initialize_parameters(templates,synthesis_seed):
    require(type(synthesis_seed) is int,'Explicit integer synthesis seed required')
    renderer=ScheduledRenderer(templates,'B',synthesis_seed)
    options={k:v for k,v in OPTIMIZER.items() if k!='name'};options['betas']=tuple(options['betas'])
    optimizer=torch.optim.AdamW(renderer.parameters(),**options)
    seed_all(synthesis_seed)
    return renderer,optimizer


def initial_fingerprint(renderer,optimizer):
    value=capture(renderer,optimizer,0,'initialization-only')
    return {'fields':{k:tree_digest(value[k]) for k in
            ('renderer','optimizer','rng','active_levels','requires_grad')}}


def load_objective(job,labels,device='cpu'):
    require(job['condition_seed']==101,'Never regenerate targets with synthesis seed')
    if job['source_set']=='DINO':
        objective,rules=load_dino(job['second_handoff'],labels,job['population'],device)
    else:
        require(job['source_set']=='A2','Unknown source set')
        objective,rules=load_combined(job['a1_handoff'],job['second_handoff'],labels,job['population'],device)
    require(all(rule['public_seed']==101 for rule in rules.values()),'Prepared condition seed changed')
    return objective,rules


def validate_job(job):
    require(job['schema']=='receiver.seed-repeat-s200-bank/v1','Wrong repeat schema')
    require(job['seed']==202 and job['template_seed']==202 and job['condition_seed']==101,'Seed domains changed')
    require(job['source_set'] in ('DINO','A2'),'Outside two-source comparison')
    require(job['id']=='dev_'+job['source_set']+'_condk4_202_s200','Wrong ID')
    require(job['artifact_kind']=='MAIN_BANK' and job['images']==128 and job['updates']==200
            and job['microbatch']==16 and job['population']=='pooled','Recipe changed')
    require(job['activation_updates']==list(ACTIVATIONS) and job['optimizer']==OPTIMIZER,'Schedule/optimizer changed')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Runtime changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Contract changed')
    contract=json.loads(Path(job['campaign_contract']).read_text(encoding='utf-8'))
    require(job['id'] in contract['bank_ids'],'Bank not authorized')
    require(job['maximum_seconds']==contract['bank_caps_seconds'][job['source_set']],'Time cap changed')
    require(job['templates']==contract['templates'],'Template selection changed')
    for key in ('a1_handoff','second_handoff'):
        require(sha(job[key])==job[key+'_sha256']==contract[key+'_sha256'],'Prepared target handoff changed')
    old=json.loads(Path(job['previous_A2_job']).read_text(encoding='utf-8'))
    for key in ('images','updates','microbatch','population','activation_updates','optimizer',
                'a1_handoff','second_handoff'):
        require(job[key]==old[key],'Historical recipe changed: '+key)
    require(all(sha(p)==h for p,h in contract['historical_dependencies'].items()),'Historical evidence changed')
    return (0,)*64+(1,)*64


def export(directory,renderer,job,spec,signature,checkpoint):
    require(checkpoint['completed_updates']==200,'Never export intermediate bank')
    final=directory/'artifact'
    with torch.no_grad():
        images=torch.cat([renderer(list(range(i,min(i+16,128)))).cpu() for i in range(0,128,16)])
    require(torch.isfinite(images).all(),'Nonfinite PNG input')
    expected=np.rint(np.clip(images[:,0].numpy(),0,1)*255).astype(np.uint8)
    if final.exists():
        result=verify_artifact(final,spec,signature,expected)
        seal=json.loads((final/'artifact_seal.json').read_text(encoding='utf-8'))
        require(seal['checkpoint_sha256']==checkpoint['sha256'],'Export checkpoint differs')
        return result
    staging=directory/('.export_'+uuid.uuid4().hex); (staging/'images').mkdir(parents=True)
    rows,pairs=image_layout(spec)
    for i,row in enumerate(rows):
        _,payload=png_roundtrip(images[i,0].numpy())
        atomic_bytes(staging/row['file'],payload); row['png_sha256']=sha(staging/row['file'])
    atomic_bytes(staging/'images.csv',csv_bytes(rows,list(rows[0])))
    atomic_bytes(staging/'pairs.csv',csv_bytes(pairs,['virtual_pair_id','positive_id','negative_id']))
    atomic_json(staging/'learning_contract.json',{
        'schema':'receiver.synthetic-learning/v2','source_objective':'condition CE gradient cosine',
        'encoder_rules':spec['encoder_rules'],'encoder_aggregation':'equal mean','conditions':4,
        'labels':'64 negative / 64 positive synthetic labels',
        'receiver_evaluation':'fixed point-only ridge0.1 with each receiver P-only projection',
        'real_patient_ids_exported':False,'relation_loss':False,'functional_loss':False,
        'artifact_kind':'MAIN_BANK','main_reuse_allowed':True,'DP_applied':False,
    })
    atomic_json(staging/'provenance_public.json',{
        'schema':'receiver.synthetic-provenance/v2','bank_id':job['id'],'seed':job['seed'],
        'initialization':'PUBLIC_P_ONLY','successful_updates':200,
        'activation_updates':list(ACTIVATIONS),'artifact_kind':'MAIN_BANK',
        'main_reuse_allowed':True,'DP_applied':False,'development_only':True,
    })
    hashes={p.relative_to(staging).as_posix():sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    atomic_json(staging/'artifact_seal.json',{'signature':signature,'bank_id':job['id'],
        'successful_updates':200,'artifact_kind':'MAIN_BANK','checkpoint_sha256':checkpoint['sha256'],
        'files_sha256':hashes})
    result=verify_artifact(staging,spec,signature,expected)
    require(not final.exists(),'Completed artifact overwrite')
    os.rename(staging,final)
    return result


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
    objective,rules=load_objective(job,labels,'cuda')
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
    encoders={}
    if 'biovil' in rules: encoders['biovil']=Encoder('source').use_projection(rules['biovil']['projection'])
    encoders[ENCODER_ID]=DinoEncoder().use_projection(rules[ENCODER_ID]['projection'])
    fixed={name:state_hash(model) for name,model in encoders.items()}
    target_hash=objective_digest(objective)
    renderer,optimizer=initialize_parameters(templates,job['seed'])
    spec=dict(job,arm='B',template_tensor_digest=tree_digest(templates),
              fixed_encoder_sha256=fixed,objective_digest=target_hash,
              encoder_rules={n:digest(r) for n,r in rules.items()})
    signature=digest(spec)
    if resume:
        require(json.loads((directory/'run_spec.json').read_text(encoding='utf-8'))==spec,'Resume dependencies changed')
        done,restored=load_checkpoint(directory,renderer,optimizer,signature)
    else:
        renderer.set_step(200 if profile else 1)
        done,restored=0,None
        atomic_json(directory/'run_spec.json',spec)
        if not profile:
            record=save_checkpoint(directory,renderer,optimizer,0,signature)
            evidence=initial_fingerprint(renderer,optimizer)
            evidence.update(seed=job['seed'],template_tensor_digest=spec['template_tensor_digest'],
                            checkpoint_sha256=record['sha256'],condition_seed=job['condition_seed'])
            if job.get('paired_initialization'):
                paired=json.loads(Path(job['paired_initialization']).read_text(encoding='utf-8'))
                require(all(evidence[k]==paired[k] for k in ('seed','condition_seed','template_tensor_digest','fields')),
                        'Paired seed202 initial states differ')
                evidence['paired_initialization_sha256']=sha(job['paired_initialization'])
            atomic_json(directory/'initialization.json',evidence)
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
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    job=json.loads(args.job.read_text(encoding='utf-8'))
    require(args.seed==job['seed'],'CLI/job synthesis seed mismatch')
    result=execute(job,args.output,resume=args.resume)
    atomic_json(args.result,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

