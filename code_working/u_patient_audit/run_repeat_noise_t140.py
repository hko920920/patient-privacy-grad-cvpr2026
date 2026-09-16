"""Frozen t140 generic-condition noise replication; new seven draws, old draw0 reused."""
import argparse, gc, hashlib, json, time, traceback
from pathlib import Path
from importlib.metadata import version
import torch
from .common import ROOT,RUN,PROMPT,digest
from .models import setup,load_unet,adapter_state
from .run_prompt_loss_t140 import (preflight as old_preflight, epsilon_for as original_epsilon,
    new_json,now,tensor_sha,check_hashes,guards)
from .run_cdi_base_selection import state_fingerprint

OLD=RUN/'baseline_screen_20260914/prompt_loss_t140_v1'
OUTPUT=RUN/'baseline_screen_20260915/repeat_noise_t140_v1'
MODELS=('model_1','model_2')
R=8
T=140

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def epsilon_for(iid,repeat):
    if repeat==0:return original_epsilon(iid)
    assert repeat in range(1,R)
    payload=f'promptloss-repeat-v1|260915|{iid}|140|{repeat}'.encode()
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**63-1)
    eps=torch.randn((1,4,32,32),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
    return seed,eps

def prepare(path):
    from .analyze_verify_repeat_noise_t140 import analysis_policy
    assert not Path(path).exists()
    old=read(OLD/'protocol.json')
    assert read(OLD/'verification.json')['status']=='PASS_PROMPT_LOSS_RAW_ARITHMETIC_AND_COHORT'
    ctx=old_preflight(Path(old['contract_path']))
    rows=[r for r in ctx['rows'] if r['eval_role']=='selection']
    assert len(rows)==160
    oldrows={(r['model'],r['image_id']):r for r in read(OLD/'results.json')
             if r['eval_role']=='selection' and r['model'] in MODELS}
    assert len(oldrows)==320
    frozen=dict(ctx['contract']['frozen_sha256'])
    paths=[Path(__file__),ROOT/'u_patient_audit/analyze_verify_repeat_noise_t140.py',
           Path(old['contract_path'])]
    paths += [OLD/n for n in ('protocol.json','execution.json','results.json','verification.json',
                              'analysis_v1/analysis.json','analysis_v1/bootstrap.json')]
    paths += [OLD/r['raw_path'] for r in oldrows.values()]
    for p in paths:frozen[str(p.resolve())]=digest(p)
    c=dict(schema='repeat-noise-t140-contract/v1',current_step=2,created_utc=now(),
        models=list(MODELS),prompt_kind='generic',prompt=PROMPT,timestep=T,repeats=list(range(R)),
        reported_prefix_means=[1,2,4,8],batch_size=1,dtype='float32',prediction_type='epsilon',
        master_seed=260915,patient_order=ctx['contract']['patient_order']['selection'],
        image_order=[r['image_id'] for r in rows],original_directory=str(OLD),
        original_contract_path=old['contract_path'],original_contract_sha256=old['contract_sha256'],
        output_directory=str(OUTPUT),checkpoint_paths=ctx['contract']['checkpoint_paths'],
        checkpoint_sha256=ctx['contract']['checkpoint_sha256'],
        expected_patients=40,expected_unique_images=160,expected_records=320,
        expected_stored_cells=2560,reused_cells=320,new_forward=2240,backward=0,
        original_score0='exact saved generic FP32 MSE and saved prediction; no new target query',
        l2_policy='CPU float64 sqrt(sum((saved FP32 prediction - FP32 epsilon)^2)) for every draw including0',
        noise_policy='one independent CPU draw per image/repeat, shared across both targets; image-specific seeds',
        analysis_policy=analysis_policy(),analysis_code_sha256=digest(ROOT/'u_patient_audit/analyze_verify_repeat_noise_t140.py'),
        frozen_sha256=frozen,new_target_training=False,new_fitting=False,stage2_completion_claim=False)
    new_json(path,c)
    return c

def validate(path):
    c=read(path)
    for k,v in dict(schema='repeat-noise-t140-contract/v1',current_step=2,models=list(MODELS),
        prompt_kind='generic',prompt=PROMPT,timestep=T,repeats=list(range(R)),reported_prefix_means=[1,2,4,8],
        batch_size=1,dtype='float32',prediction_type='epsilon',master_seed=260915,
        expected_patients=40,expected_unique_images=160,expected_records=320,expected_stored_cells=2560,
        reused_cells=320,new_forward=2240,backward=0,new_target_training=False,new_fitting=False,
        stage2_completion_claim=False).items():assert c[k]==v,k
    check_hashes(c['frozen_sha256'])
    assert Path(c['output_directory']).resolve()==OUTPUT.resolve()
    old=read(OLD/'protocol.json')
    assert digest(c['original_contract_path'])==c['original_contract_sha256']==old['contract_sha256']
    ctx=old_preflight(Path(c['original_contract_path']))
    rows=[r for r in ctx['rows'] if r['eval_role']=='selection']
    assert [r['image_id'] for r in rows]==c['image_order']
    assert sorted({r['patient_id'] for r in rows},key=int)==c['patient_order']
    assert len(rows)==160
    source={(r['model'],r['image_id']):r for r in read(OLD/'results.json')
            if r['eval_role']=='selection' and r['model'] in MODELS}
    assert len(source)==320
    for (m,i),row in source.items():
        assert digest(OLD/row['raw_path'])==row['raw_sha256']
        assert row['checkpoint_sha256']==c['checkpoint_sha256'][m]
    return c,ctx,rows,source

def run(path,c,ctx,rows,source):
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    cs=digest(path)
    new_json(OUTPUT/'protocol.json',dict(schema='repeat-noise-t140-execution/v1',
        started_utc=now(),contract=c,contract_path=str(Path(path).resolve()),contract_sha256=cs,
        environment={n:version(n) for n in ('torch','numpy','diffusers','peft')}))
    results=[];reports=[];total=0;unet=None;handles=[]
    setup();started=time.perf_counter()
    try:
        for model in MODELS:
            mst=time.perf_counter()
            directory=OUTPUT/model;directory.mkdir()
            unet=load_unet(Path(c['checkpoint_paths'][model]),training=False).float()
            unet.enable_adapters();unet.eval().requires_grad_(False);unet.disable_gradient_checkpointing()
            assert all(p.dtype==torch.float32 for p in unet.parameters())
            loaded=adapter_state(unet);expected=ctx['ledgers'][model]['adapter']
            assert loaded.keys()==expected.keys() and all(torch.equal(loaded[k],expected[k].float()) for k in loaded)
            before=state_fingerprint(unet)
            prior=next(q for q in read(OLD/'execution.json')['model_reports'] if q['model']==model)
            assert before['sha256']==prior['initial_state']['sha256']
            versions={n:p._version for n,p in unet.named_parameters()};guards(unet,versions)
            count={'forward':0,'backward':0}
            def fh(module,args,output):count['forward']+=1
            def bh(module,gin,gout):count['backward']+=1
            handles=[unet.register_forward_hook(fh),unet.register_full_backward_hook(bh)]
            hidden=ctx['cache']['hidden'][PROMPT].float().to('cuda')
            hh=tensor_sha(hidden.cpu())
            torch.cuda.reset_peak_memory_stats()
            for num,r in enumerate(rows):
                iid=r['image_id'];oldrow=source[model,iid]
                oldraw=torch.load(OLD/oldrow['raw_path'],map_location='cpu',weights_only=True)
                assert oldraw['prompt_hidden_sha256']['generic']==hh
                z=ctx['cache']['latents'][iid].float().to('cuda')
                assert torch.equal(z.cpu(),oldraw['latent'])
                ts=torch.tensor([T],device='cuda',dtype=torch.long)
                draws=[];before_f=count['forward']
                for repeat in range(R):
                    seed,eps=epsilon_for(iid,repeat)
                    if repeat==0:
                        assert seed==oldraw['noise_seed'] and torch.equal(eps,oldraw['epsilon'])
                        pred=oldraw['predictions']['generic']
                        noisy=oldraw['noised_latent']
                        mse=float(oldrow['losses']['generic'])
                    else:
                        noise=eps.to('cuda');noisy=ctx['scheduler'].add_noise(z,noise,ts)
                        with torch.inference_mode():
                            pred=unet(noisy,ts,encoder_hidden_states=hidden,return_dict=False)[0]
                            mse=float((pred-noise).square().mean())
                        pred=pred.cpu();noisy=noisy.cpu()
                    assert pred.dtype==torch.float32 and torch.isfinite(pred).all()
                    resid=pred.double()-eps.double()
                    l2=float(torch.sqrt((resid*resid).sum()))
                    draws.append(dict(repeat=repeat,seed=seed,epsilon=eps,noise_sha256=tensor_sha(eps),
                        noised_latent=noisy,prediction=pred,loss_mse_fp32=mse,loss_l2_fp64=l2,
                        reused_original=repeat==0))
                assert count['forward']-before_f==7 and count['backward']==0
                guards(unet,versions)
                raw=dict(model=model,image_id=iid,timestep=T,latent=z.cpu(),
                    alphas=ctx['scheduler'].alphas_cumprod.cpu().float(),hidden_sha256=hh,draws=draws)
                rp=directory/(Path(iid).stem+'.pt');torch.save(raw,rp)
                row={k:oldrow[k] for k in ('model','patient_id','image_id','eval_role','scenario','record_role',
                    'assignment_group','member','image_member','actual_training_exposures','patient_training_exposures',
                    'checkpoint_sha256','cache_sha256','cohort_lock_sha256')}
                row.update(prompt=PROMPT,hidden_sha256=hh,repeats=R,forward=7,backward=0,
                    original_raw_path=str(OLD/oldrow['raw_path']),original_raw_sha256=oldrow['raw_sha256'],
                    raw_path=rp.relative_to(OUTPUT).as_posix(),raw_sha256=digest(rp),contract_sha256=cs,
                    draws=[{k:v for k,v in d.items() if not isinstance(v,torch.Tensor)} for d in draws])
                new_json(rp.with_suffix('.json'),row);results.append(row);total+=7
                del raw,draws,pred,z,noisy,oldraw
                if (num+1)%20==0:
                    print(json.dumps(dict(event='repeat_noise_progress',model=model,images=num+1,total_images=160,
                        new_forward=count['forward'],seconds=time.perf_counter()-mst)),flush=True)
            guards(unet,versions);after=state_fingerprint(unet)
            assert before['sha256']==after['sha256'] and count=={'forward':1120,'backward':0}
            current=adapter_state(unet)
            assert all(torch.equal(current[k],expected[k].float()) for k in current)
            reports.append(dict(model=model,records=160,forward=1120,backward=0,initial_state=before,final_state=after,
                state_values_exactly_unchanged=True,adapter_matches_checkpoint=True,
                parameter_versions_and_no_gradients=True,all_parameters_frozen=True,unet_training=False,
                seconds=time.perf_counter()-mst,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            for h in handles:h.remove()
            handles=[];unet=None
            del hidden,current,loaded
            gc.collect();torch.cuda.empty_cache()
        check_hashes(c['frozen_sha256']);assert digest(path)==cs
        assert total==2240 and len(results)==320
        new_json(OUTPUT/'results.json',results)
        e=dict(schema='repeat-noise-t140-result/v1',status='COMPLETE_PENDING_SAVED_RAW_VERIFICATION',
            complete=True,records=320,stored_cells=2560,reused_cells=320,forward=2240,backward=0,
            model_reports=reports,seconds=time.perf_counter()-started,ended_utc=now(),
            results_sha256=digest(OUTPUT/'results.json'),protocol_sha256=digest(OUTPUT/'protocol.json'),
            contract_sha256=cs,frozen_files_unchanged=True,new_fitting=False,new_target_training=False,
            stage2_completion_claim=False)
        new_json(OUTPUT/'execution.json',e);print(json.dumps(e),flush=True)
    except BaseException as exc:
        new_json(OUTPUT/'failure.json',dict(status='FAILED_INCOMPLETE',error=repr(exc),traceback=traceback.format_exc(),
            completed_records=len(results),completed_forward=total,seconds=time.perf_counter()-started,
            automatic_retry=False))
        raise
    finally:
        for h in handles:h.remove()
        unet=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-contract',action='store_true');mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--run',action='store_true');a=p.parse_args()
    if a.create_contract:
        c=prepare(a.contract);print(json.dumps(dict(status='CPU_CONTRACT_CREATED',sha256=digest(a.contract),files=len(c['frozen_sha256']))))
        return
    c,ctx,rows,source=validate(a.contract)
    if a.dry_run:
        print(json.dumps(dict(status='PASS_CPU_PREFLIGHT_NO_GPU',records=320,new_forward=2240,
            CUDA_initialized=torch.cuda.is_initialized(),sha256=digest(a.contract))));return
    run(a.contract,c,ctx,rows,source)

if __name__=='__main__':main()
