"""CPU verification of saved replay/masked training evidence; no producer import."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT/'_reports/cvpr_u_pilot_v1_001'
OUT = RUN/'baseline_screen_20260915/masked_patient_training_v1'
PRIOR = RUN/'training_coverage_v2/model_1'
SALT = 'cvpr-u-pilot-20260914-v1'
EXPECTED_SLOTS = [(44,3,'00014393_006.png',172),(83,0,'00014393_002.png',281),
    (380,3,'00014393_006.png',108),(399,2,'00014393_002.png',122),
    (474,3,'00014393_006.png',452),(592,2,'00014393_002.png',475),
    (742,1,'00014393_006.png',305),(904,2,'00014393_002.png',922)]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def th(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def new_json(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def digest_values(*args):
    return hashlib.sha256('|'.join(str(v) for v in (SALT,)+args).encode()).hexdigest()


def rng_seed(*args):
    return int(digest_values(*args)[:15],16)


def exact(left,right,path='root',counts=None):
    counts={} if counts is None else counts
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape,path
        assert torch.equal(left.cpu(),right.cpu()),path
        counts['tensor_pairs']=counts.get('tensor_pairs',0)+1
        counts['tensor_coordinates']=counts.get('tensor_coordinates',0)+left.numel()
    elif isinstance(left,dict):
        assert isinstance(right,dict) and set(left)==set(right),path
        for k in left: exact(left[k],right[k],path+'/'+str(k),counts)
    elif isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right),path
        for j in range(len(left)): exact(left[j],right[j],path+'/'+str(j),counts)
    else:
        assert type(left)==type(right) and left==right,path
        counts['scalar_pairs']=counts.get('scalar_pairs',0)+1
    return counts


def independent_batches():
    with (RUN/'cohort/model_1_train.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:(r['eval_role']!='background',
                digest_values('training-order',r['patient_id']),r['image_id']))
    assert len(rows)==912
    batches=[];slots=[]
    for step in range(1000):
        epoch,offset=divmod(step*4,912)
        perm=torch.randperm(912,generator=torch.Generator().manual_seed(rng_seed('coverage-epoch',epoch)))
        batch=[rows[int(j)] for j in perm[offset:offset+4]]
        ts=[rng_seed('coverage-timestep',step,j)%1000 for j in range(4)]
        ids=[r['image_id'] for r in batch]
        mask=[int(r['patient_id']!='14393') for r in batch]
        batches.append(dict(update=step+1,image_ids=ids,timesteps=ts,
            noise_seed=rng_seed('coverage-noise',step),control_mask=mask))
        slots.extend((step+1,j,r['image_id'],ts[j]) for j,r in enumerate(batch) if not mask[j])
    assert slots==EXPECTED_SLOTS
    return batches


def expected_frozen_base(contract):
    paths=[Path(p) for p in contract['frozen_sha256']
           if Path(p).parent.name=='unet' and Path(p).name=='diffusion_pytorch_model.fp16.safetensors']
    assert len(paths)==1
    state=load_file(str(paths[0]),device='cpu')
    renamed={}
    for name,tensor in state.items():
        key=re.sub(r'\.(to_q|to_k|to_v|to_out\.0)\.(weight|bias)$',r'.\1.base_layer.\2',name)
        assert key not in renamed
        renamed[key]=tensor
    h=hashlib.sha256()
    for name in sorted(renamed):
        h.update(name.encode('utf-8'));h.update(th(renamed[name]).encode('ascii'))
    result=dict(sha256=h.hexdigest(),tensors=len(renamed))
    del state,renamed
    return result


def mask_arithmetic(packet,kind):
    p,t=packet['prediction'],packet['target']
    assert p.dtype==t.dtype==torch.float16 and p.shape==t.shape==(4,4,32,32)
    assert bool(torch.isfinite(p).all()) and bool(torch.isfinite(t).all())
    mask=torch.tensor(packet['mask'],dtype=torch.float64).reshape(4,1,1,1)
    assert set(packet['mask'])<={0,1} and sum(packet['mask'])==(3 if kind=='control' else 4)
    residual=p.double()-t.double()
    # Alternative float64, explicit sum/total-element denominator path.
    truth=float((residual.square()*mask).sum()/p.numel())
    measured=packet['loss']
    n=p.numel();u=2.**-24
    # Conservative forward-error bound for subtraction/square and an arbitrary
    # sequential positive FP32 reduction. This is not an empirical MIA gate.
    gamma=(n+4)*u/(1-(n+4)*u)
    bound=gamma*float(((p.double().abs()+t.double().abs()).square()*mask).sum()/n)+1e-35
    assert abs(measured-truth)<=bound
    # Replay only the saved prediction-to-scaled-loss chain on CPU, independent
    # of the GPU UNet and its parameter gradients.
    candidate=p.clone().requires_grad_()
    sq=(candidate.float()-t.float()).square()
    if kind=='control': sq=sq*mask.float()
    loss=sq.mean()
    (loss*packet['gradient_scale']).backward()
    saved=packet['prediction_gradient']
    assert saved.dtype==torch.float16 and saved.shape==p.shape
    assert torch.equal(candidate.grad,saved),'FP16 prediction-gradient chain mismatch'
    # Closed-form double calculation also checks scale and total denominator.
    closed=(2.*residual*mask*packet['gradient_scale']/n).to(torch.float16)
    assert torch.equal(closed,saved),'Closed-form scaled derivative mismatch'
    for j,value in enumerate(packet['mask']):
        if not value: assert torch.count_nonzero(saved[j]).item()==0
    return dict(update=packet['update'],attempt=packet['attempt'],loss_fp64=truth,
        absolute_loss_error=abs(measured-truth),conservative_fp32_forward_error_bound=bound,
        cpu_autograd_prediction_gradient_exact=True,closed_form_fp16_gradient_exact=True,
        masked_gradient_zero=True,denominator=n,gradient_scale=packet['gradient_scale'])


def self_test():
    p=torch.linspace(-2,2,16384,dtype=torch.float16).reshape(4,4,32,32)
    t=torch.linspace(1,-1,16384,dtype=torch.float16).reshape_as(p)
    for kind,mask in [('treatment',[1,1,1,1]),('control',[1,0,1,1])]:
        q=p.clone().requires_grad_();m=torch.tensor(mask).reshape(4,1,1,1)
        loss=((q.float()-t.float()).square()*m).mean()
        (loss*1024.).backward()
        mask_arithmetic(dict(prediction=p,target=t,mask=mask,loss=float(loss.detach()),
            gradient_scale=1024.,prediction_gradient=q.grad,update=44,attempt=44),kind)
    batches=independent_batches()
    assert sum(sum(b['control_mask']) for b in batches)==3992
    assert re.sub(r'\.(to_q|to_k|to_v|to_out\.0)\.(weight|bias)$',r'.\1.base_layer.\2',
                  'down.attn.to_out.0.weight')=='down.attn.to_out.0.base_layer.weight'
    return dict(status='PASS_CPU_INDEPENDENT_MASK_CHAIN_AND_SCHEDULE',CUDA_initialized=torch.cuda.is_initialized())


def run(args):
    started=time.perf_counter();out=args.run_dir
    assert sha(Path(__file__))==args.expected_code_sha256
    execution=read(out/'execution.json');protocol=read(out/'protocol.json')
    assert execution['complete'] is True
    assert execution['status']=='PASS_REPLAY_AND_MASKED_CONTRIBUTION_EXECUTION_PENDING_INDEPENDENT_VERIFICATION'
    contract=protocol['contract'];contract_path=Path(protocol['contract_path'])
    assert sha(contract_path)==protocol['contract_sha256']==execution['contract_sha256']
    exact(contract,read(contract_path),'contract')
    assert sha(out/'protocol.json')==execution['protocol_sha256']
    assert sha(out/'conformance.json')==execution['conformance_sha256']
    inputs=dict(contract['frozen_sha256'])
    files=['protocol.json','execution.json','conformance.json']
    for kind in ['treatment','control']:
        files += [kind+'/'+n for n in ['execution.json','initial_state.pt','attempts.json',
            'committed_steps.json','masked_update_raw.pt','training_trace.jsonl',
            'step_0043.pt','step_0250.pt','step_1000.pt',
            'conformance_at_0043.json','conformance_at_0250.json','conformance_at_1000.json']]
    for n in files: inputs[str((out/n).resolve())]=sha(out/n)
    inputs[str(contract_path.resolve())]=sha(contract_path)
    inputs[str(Path(__file__).resolve())]=sha(Path(__file__))
    for p,h in inputs.items(): assert sha(p)==h,p
    dest=out/args.output_tag;dest.mkdir(parents=True,exist_ok=False)
    new_json(dest/'protocol.json',dict(schema='masked-training-independent-verification/v1',
        expected_code_sha256=args.expected_code_sha256,input_sha256=inputs,
        scope='Saved state/input integrity and prediction-level mask derivative; CPU only',
        loss_bound='Conservative floating point forward-error bound; no fitted tolerance'))
    torch.set_num_threads(4)
    batches=independent_batches()
    exact(batches,read(Path(contract['preparation_dir'])/'scheduled_batches.json'))
    expected_base=expected_frozen_base(contract)
    cache=torch.load(RUN/'cache/cache.pt',map_location='cpu',weights_only=True)
    checkpoints={};reports={};commits={};attempts={};state_checks=[];mask_checks=[]
    original_trace=[json.loads(s) for s in (PRIOR/'training_trace.jsonl').read_text().splitlines()]
    original_groups=torch.load(PRIOR/'step_1000.pt',map_location='cpu',weights_only=True)['optimizer']['param_groups']
    for kind in ['treatment','control']:
        directory=out/kind
        report=read(directory/'execution.json');reports[kind]=report
        exact(report,execution['branches'][0 if kind=='treatment' else 1])
        assert report['status']=='PASS_BRANCH_EXECUTION_AND_CONFORMANCE' and report['step']==1000
        for field,name in [('initial_state_sha256','initial_state.pt'),('attempts_sha256','attempts.json'),
            ('committed_steps_sha256','committed_steps.json'),('masked_update_raw_sha256','masked_update_raw.pt')]:
            assert report[field]==sha(directory/name)
        exact(report['frozen_before'],expected_base,'base artifact before')
        exact(report['frozen_after'],expected_base,'base artifact after')
        assert report['frozen_parameter_versions_unchanged'] is True
        checkpoints[kind]={}
        for step in [43,250,1000]:
            f=directory/f'step_{step:04d}.pt'
            assert report['checkpoint_sha256'][str(step)]==sha(f)
            q=torch.load(f,map_location='cpu',weights_only=True);checkpoints[kind][step]=q
            assert q['step']==step and len(q['losses'])==step
            exact(q['contract'],contract['original_training_contract'])
            exact(q['optimizer']['param_groups'],original_groups,'optimizer hyperparameters')
            assert len(q['adapter'])==len(q['optimizer']['state'])==256
            assert sum(v.numel() for v in q['adapter'].values())==1659904
            for value in q['adapter'].values():
                assert value.dtype==torch.float32 and bool(torch.isfinite(value).all())
            for state in q['optimizer']['state'].values():
                assert float(state['step'])==step
                assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
                assert state['exp_avg'].shape==state['exp_avg_sq'].shape
                assert bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all())
                assert bool((state['exp_avg_sq']>=0).all())
            effective=Counter()
            for batch in batches[:step]:
                for j,i in enumerate(batch['image_ids']):
                    if kind=='treatment' or batch['control_mask'][j]: effective[i]+=1
            exact(q['exposures'],dict(effective),f'{kind}{step} exposures')
            if kind=='treatment' and step in [250,1000]:
                old=torch.load(PRIOR/f'step_{step:04d}.pt',map_location='cpu',weights_only=True)
                counts=exact(q,old,f'original checkpoint {step}')
                state_checks.append(dict(kind=kind,step=step,exact=True,**counts))
        commits[kind]=read(directory/'committed_steps.json');attempts[kind]=read(directory/'attempts.json')
        assert len(commits[kind])==1000 and len(attempts[kind])==report['attempts']
        assert report['root_forward_calls']==report['backward_calls']==len(attempts[kind])
        assert report['root_forward_examples']==4*len(attempts[kind])
        trace=[json.loads(s) for s in (directory/'training_trace.jsonl').read_text().splitlines()]
        assert len(trace)==1000
        committed_count=0
        for attempt_index,a in enumerate(attempts[kind],1):
            assert a['attempt']==attempt_index and a['update']==committed_count+1
            assert a['step_zero_based']==committed_count
            b=batches[committed_count]
            for key in ['image_ids','timesteps','noise_seed']: exact(a[key],b[key])
            expected_mask=b['control_mask'] if kind=='control' else [1,1,1,1]
            exact(a['mask'],expected_mask)
            assert a['gradient_scale_before']==a['scale_before_step']
            assert a['overflow_retry']==(a['scale_after_step']<a['scale_before_step'])
            z=torch.cat([cache['latents'][i] for i in b['image_ids']]).to(torch.float16)
            h=torch.cat([cache['hidden'][cache['train_prompts'][i]] for i in b['image_ids']]).to(torch.float16)
            ts=torch.tensor(b['timesteps'],dtype=torch.long)
            for key,value in [('latents',z),('hidden',h),('timesteps',ts)]: assert th(value)==a['input_sha256'][key]
            assert a['input_sha256']['noise']==a['input_sha256']['target']
            if kind=='control': exact(a['input_sha256'],commits['treatment'][committed_count]['input_sha256'])
            if a['overflow_retry']:
                assert a['scale_after_step']==.5*a['scale_before_step']
                continue
            c=commits[kind][committed_count]
            for key,value in a.items(): exact(value,c[key],f'committed {attempt_index}/{key}')
            assert c['gradient_norm_finite'] is True and c['gradient_norm']>=0
            assert c['loss']==checkpoints[kind][1000]['losses'][committed_count]
            for key in ['attempt','loss','gradient_norm']: assert c[key]==trace[committed_count][key]
            assert trace[committed_count]['step']==committed_count+1
            if kind=='treatment':
                for key in ['attempt','loss','gradient_norm']: assert c[key]==original_trace[committed_count][key]
            if kind=='control':
                if committed_count<43:
                    for key in ['attempt','loss','gradient_norm','gradient_scale_before','scale_after_step']:
                        exact(c[key],commits['treatment'][committed_count][key])
            committed_count+=1
        assert committed_count==1000
        for saved_step in [43,250,1000]:
            q=checkpoints[kind][saved_step]
            assert q['attempt']==commits[kind][saved_step-1]['attempt']
            assert q['scaler']['scale']==commits[kind][saved_step-1]['scale_after_step']
        raw=torch.load(directory/'masked_update_raw.pt',map_location='cpu',weights_only=True)
        expected_attempts={str(a['attempt']) for a in attempts[kind] if a['update'] in {s[0] for s in EXPECTED_SLOTS}}
        assert set(raw)==expected_attempts
        for key,q in raw.items():
            a=attempts[kind][int(key)-1]
            assert q['attempt']==a['attempt'] and q['update']==a['update']
            exact(q['mask'],a['mask'])
            assert th(q['target'])==a['input_sha256']['target']
            assert q['gradient_scale']==a['gradient_scale_before']
            if not a['overflow_retry']: assert q['loss']==commits[kind][q['update']-1]['loss']
            mask_checks.append(dict(branch=kind,**mask_arithmetic(q,kind)))
        assert report['scheduled_committed_slots']==4000
        assert report['effective_nonzero_exposures']==(4000 if kind=='treatment' else 3992)
        assert report['effective_exposed_images']==(912 if kind=='treatment' else 910)
    state_checks.append(dict(kind='shared_prefix',step=43,exact=True,
        **exact(checkpoints['control'][43],checkpoints['treatment'][43],'shared prefix43')))
    exact(torch.load(out/'treatment/initial_state.pt',map_location='cpu',weights_only=True),
          torch.load(out/'control/initial_state.pt',map_location='cpu',weights_only=True),'same initial')
    assert checkpoints['treatment'][1000]['attempt']==reports['treatment']['attempts']==1000
    for kind in ['treatment','control']:
        assert checkpoints[kind][1000]['attempt']==reports[kind]['attempts']
        effective=checkpoints[kind][1000]['exposures']
        for image in ['00014393_002.png','00014393_006.png']:
            assert effective.get(image,0)==(4 if kind=='treatment' else 0)
    assert execution['committed_updates']==2000 and execution['scheduled_committed_slots']==8000
    assert execution['effective_nonzero_exposures']==7992
    assert execution['forward_examples']==sum(r['root_forward_examples'] for r in reports.values())
    assert execution['backward_calls']==sum(r['backward_calls'] for r in reports.values())
    gates=read(out/'conformance.json')
    assert gates['status']=='PASS_EXACT_TREATMENT_REPLAY_AND_SHARED_PREFIX'
    assert len(gates['gates'])==3 and all(g['exact'] for g in gates['gates'])
    for p,h in inputs.items(): assert sha(p)==h,p
    result=dict(status='PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC',
        complete=True,current_step=2,seconds=time.perf_counter()-started,
        source_code_sha256=args.expected_code_sha256,verification_protocol_sha256=sha(dest/'protocol.json'),
        execution_sha256=sha(out/'execution.json'),protocol_sha256=sha(out/'protocol.json'),
        contract_sha256=sha(contract_path),checkpoint_sha256={k:reports[k]['checkpoint_sha256'] for k in reports},
        exact_state_checks=state_checks,input_hash_pairs_checked=1000,
        independent_cached_latent_hidden_timestep_reconstructions=2000,
        effective_exposures={'treatment':4000,'control':3992},mask_arithmetic_checks=mask_checks,
        expected_frozen_base_from_artifact=expected_base,
        limitations=['No independent full UNet parameter-gradient or 1000-step control refit on CPU',
            'CUDA FP16 RNG not independently regenerated on CPU; all committed branch noise hashes match and eight saved noise tensors bind the mask cells',
            'Runtime frozen-base hashes and versions are producer observations checked against the original artifact',
            'No endpoint attack score, population membership inference, or individual privacy guarantee'],
        CUDA_initialized=torch.cuda.is_initialized())
    assert result['CUDA_initialized'] is False
    new_json(dest/'verification.json',result)
    new_json(out/'verification.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],mask_cells=len(mask_checks))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=OUT)
    p.add_argument('--output-tag',default='verification_v1');p.add_argument('--expected-code-sha256')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test: print(json.dumps(self_test()));return
    assert a.expected_code_sha256,'Freeze verifier SHA before actual execution'
    run(a)


if __name__=='__main__': main()
