"""One public C128 bank: bounded timing, no main synthesis artifact or utility.

Predeclared microbatch4, five warm-up + ten timed FULL optimizer updates.
Reuse the stage3 objective unchanged. No microbatch search or automatic retry.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import CODE, OUT, RESEARCH, setup, read, save, sha, now, require, source_hashes
from .datasets import manifests, PixelAccess, public_templates
from .encoders import Encoder, state_hash, pil_tensor
from .render import Renderer, LEVELS
from .guarded_objectives import LearnerPolicy, bind_objective, two_pass
from .w1_boundary_debug import tensor_hash

WARMUP = 5
TIMED = 10
MICROBATCH = 4
IMAGES = 128
SEED = 101


def cpu_parameters(renderer):
    return [p.detach().cpu().clone() for p in renderer.parameters()]


def verify_update(renderer, previous, optimizer):
    """Outside the timed region: check every parameter/gradient/state tensor."""
    peak = 0.; changed = 0; delta2 = 0.; grad2 = 0.; gradpeak = 0.
    for p, old in zip(renderer.parameters(), previous):
        require(p.grad is not None, 'Missing active renderer gradient')
        value=p.detach().cpu();g=p.grad.detach().cpu()
        require(bool(torch.isfinite(value).all() and torch.isfinite(g).all()),
                'Nonfinite renderer parameter or gradient')
        delta=value.double()-old.double()
        peak=max(peak,float(delta.abs().max()));changed+=int(torch.count_nonzero(delta))
        delta2+=float(delta.square().sum());grad2+=float(g.double().square().sum())
        gradpeak=max(gradpeak,float(g.abs().max()))
    require(peak>0 and changed>0 and grad2>0,'Optimizer did not change synthetic parameters')
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value,torch.Tensor):require(bool(torch.isfinite(value).all()),'Nonfinite optimizer state')
    return {'finite_parameters_and_gradients':True,'finite_optimizer_state':True,
            'parameter_change_max_abs':peak,'parameter_change_l2':math.sqrt(delta2),
            'changed_parameter_elements':changed,'gradient_l2':math.sqrt(grad2),
            'gradient_max_abs':gradpeak,'actual_parameter_update':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve previous profile attempt')
    args.output.mkdir(parents=True)
    step3=read(RESEARCH/'spec_sources/prrd_step3_guarded_objective_record_20260918.json')
    require(step3['status']=='PASS_NEW_OBJECTIVE_ACTUAL_IMAGE_GRADIENT_ONLY','Stage3 not passed')
    require(sha(step3['verification'])==step3['verification_sha256'],'Stage3 verification changed')
    require(sha(step3['contract'])==step3['contract_sha256'],'Stage3 contract changed')
    cfg3=read(step3['contract'])
    # All previously verified source files stay exact. This profiler is new.
    for path,expected in cfg3['source_sha256'].items():
        require(sha(path)==expected,'Verified source changed: '+path)
    for path,expected in cfg3['inputs'].items():require(sha(path)==expected,'Bound input changed: '+path)
    step2=CODE/'_reports/prrd_step2_paths_20260918_v1/public_paths'
    scales=read(step2/'source_P_relation_scales.json')['scales']
    groups=manifests();chosen=public_templates(groups['P'],SEED)
    require(len(chosen)==IMAGES,'Wrong template bank size')
    policy=LearnerPolicy(1.,.1,kappa=.1)
    optimizer_cfg={'name':'AdamW','lr':.01,'weight_decay':0.,'betas':[.9,.999],'eps':1e-8,'foreach':False}
    config={'schema':'prrd.public-C128-cost-profile/v1','created_before_GPU':now(),
            'scope':'ONE_PUBLIC_C128_BANK_COST_ONLY','arm':'C','images':IMAGES,
            'layout':'64 marginal images +32 virtual pairs (64 pair images)',
            'microbatch':MICROBATCH,'warmup_updates':WARMUP,'timed_updates':TIMED,
            'total_profile_updates':WARMUP+TIMED,'seed':SEED,
            'learner_policy':asdict(policy),'eta':1.,'coefficients_numerical_test_only':True,
            'optimizer':optimizer_cfg,'renderer_levels':list(LEVELS),'all_renderer_levels_active':True,
            'activation_setting':500,'augmentation_nonces':list(range(500,500+WARMUP+TIMED)),
            'activation_setting_is_not_training_updates':True,'source_scales':scales,
            'timing_boundary':'synchronized wall time from full-bank two_pass entry through optimizer.step completion',
            'includes':'all128 clean+aug forwards, global statistics/guard/function loss, full replay backward, optimizer update',
            'excludes':'model/target preparation, input IO, CPU parameter copies/checks, source hashes, report IO',
            'memory':'CUDA allocated/reserved peak per complete update; CPU verification copies excluded',
            'microbatch_search':False,'automatic_retry':False,'performance_or_coefficient_selection':False,
            'source_sha256':source_hashes(),'inputs':cfg3['inputs'],
            'stage3_verification_sha256':step3['verification_sha256'],
            'templates':[{'slot':i,**{k:row[k] for k in ('image_id','patient_id','sha256')}} for i,row in enumerate(chosen)],
            'target_population':'Existing P-only patient-uniform point and mixed-patient relation statistics',
            'Q_V_pixels':False,'recipient_models':False,'DP':False,'expert_reserved':False,
            'main500_training':False,'other_arms':False,'remote_upload':False,
            'artifact_reuse':'PROFILE_ONLY_DO_NOT_USE_AS_MAIN_BANK; no final image or trained-parameter export'}
    save(args.output/'profile_contract.json',config)
    (args.output/'PROFILE_ONLY_DO_NOT_USE_AS_MAIN_BANK.txt').write_text(
        'Cost measurement only. No main bank, trained parameter checkpoint, PNG export or performance candidate.\n',encoding='utf-8')
    setup();start=time.perf_counter()
    access=PixelAccess(groups,('P',)).install()
    samples=[];counts={'forward_calls':0,'forward_images':0,'backward_calls':0,'backward_images':0}
    completed=0;handle=None
    try:
        prep_start=time.perf_counter()
        encoder=Encoder('source').use_projection(OUT/'source_P_projection.npz')
        encoder.use_relation_projection(step2/'source_P_relation_projection.npz')
        source_before=state_hash(encoder)
        with np.load(step2/'source_P_relation_targets.npz',allow_pickle=False) as archive:
            target={key:archive['C_'+key].copy() for key in ('m','A','delta','C')}
        objective=bind_objective(target,'C',scales,policy,eta=1.,device='cuda')
        fixed_target={k:tensor_hash(v) for k,v in objective.moments.items()}
        fixed_target.update(weight=tensor_hash(objective.functional_target.weight),matrix=tensor_hash(objective.functional_target.matrix))
        preparation_seconds=time.perf_counter()-prep_start
        io_start=time.perf_counter();decoded={}
        for row in chosen:
            if row['image_id'] not in decoded:
                decoded[row['image_id']]=TF.resize(pil_tensor(access.image(row)),[224,224],antialias=True)
        template_cpu=torch.stack([decoded[row['image_id']] for row in chosen])
        templates=template_cpu.cuda();torch.cuda.synchronize()
        template_io_seconds=time.perf_counter()-io_start
        model=Renderer(templates,'C',SEED);model.set_step(500)
        require(all(p.requires_grad for p in model.parameters()),'Not all pyramid levels active')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0,betas=(.9,.999),eps=1e-8,foreach=False)
        initial_hash=state_hash(model)
        def count_forward(module,inputs,output):
            n=len(inputs[0]);counts['forward_calls']+=1;counts['forward_images']+=n
            if output.projected_global_embedding.requires_grad:
                def count_backward(g):
                    counts['backward_calls']+=1;counts['backward_images']+=n
                    return g
                output.projected_global_embedding.register_hook(count_backward)
        handle=encoder.model.register_forward_hook(count_forward)
        verification_seconds=0.;progress_io_seconds=0.
        for index in range(WARMUP+TIMED):
            check_start=time.perf_counter();previous=cpu_parameters(model)
            torch.cuda.synchronize();verification_seconds+=time.perf_counter()-check_start
            torch.cuda.reset_peak_memory_stats();before_counts=counts.copy()
            start_update=time.perf_counter()
            trace=two_pass(model,encoder,objective,500+index,SEED,MICROBATCH)
            optimizer.step()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start_update;completed+=1
            memory={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
            check_start=time.perf_counter()
            require(all(math.isfinite(v) for v in trace.values()),'Nonfinite profile loss/learner state')
            changed=verify_update(model,previous,optimizer)
            delta_counts={k:counts[k]-before_counts[k] for k in counts}
            require(delta_counts=={'forward_calls':4*IMAGES//MICROBATCH,'forward_images':4*IMAGES,
                                   'backward_calls':2*IMAGES//MICROBATCH,'backward_images':2*IMAGES},
                    'A reported update did not process the complete128-image bank')
            for st in optimizer.state.values():require(int(st['step'])==completed,'Optimizer update count mismatch')
            torch.cuda.synchronize();verification_seconds+=time.perf_counter()-check_start
            del previous
            sample={'update':completed,'warmup':index<WARMUP,'seconds_full_update':seconds,
                    'memory':memory,'counts':delta_counts,'trace':trace,**changed}
            samples.append(sample)
            io_tick=time.perf_counter()
            with (args.output/'progress.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(sample,allow_nan=False)+'\n')
            print(json.dumps({'update':completed,'warmup':sample['warmup'],'whole_bank_seconds':seconds,
                              'peak_allocated_MiB':memory['allocated_bytes']/2**20,'finite_and_updated':True}),flush=True)
            progress_io_seconds+=time.perf_counter()-io_tick
        handle.remove();handle=None
        source_after=state_hash(encoder);final_hash=state_hash(model)
        require(source_after==source_before,'Fixed source weights, buffers or projections changed')
        require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()),'Source gradient leakage')
        require(final_hash!=initial_hash,'Synthetic parameters did not change')
        target_after={k:tensor_hash(v) for k,v in objective.moments.items()}
        target_after.update(weight=tensor_hash(objective.functional_target.weight),matrix=tensor_hash(objective.functional_target.matrix))
        require(fixed_target==target_after,'Target moments, matrix or classifier changed')
        require(source_hashes()==config['source_sha256'],'Source changed during profile')
        timed=np.array([row['seconds_full_update'] for row in samples if not row['warmup']])
        require(len(timed)==TIMED and completed==WARMUP+TIMED,'Incomplete timing run')
        record={'status':'PASS_PUBLIC_C128_FULL_UPDATE_COST_ONLY','contract_sha256':sha(args.output/'profile_contract.json'),
                'arm':'C','images_per_update':IMAGES,'microbatch':MICROBATCH,
                'warmup_updates':WARMUP,'timed_updates':TIMED,'profile_optimizer_updates':completed,
                'seconds_per_full_update':{'mean':float(timed.mean()),'median':float(np.median(timed)),
                     'std_population':float(timed.std()),'min':float(timed.min()),'max':float(timed.max()),
                     'p10':float(np.quantile(timed,.1)),'p90':float(np.quantile(timed,.9))},
                'peak_bytes_all_updates':{k:max(row['memory'][k] for row in samples) for k in samples[0]['memory']},
                'peak_bytes_timed_updates':{k:max(row['memory'][k] for row in samples if not row['warmup']) for k in samples[0]['memory']},
                'samples':samples,'counts':counts,'access':access.report(),
                'source_weights_buffers_projections_unchanged':True,'source_state_sha256':source_after,
                'source_parameter_gradients_absent':True,'target_fixed':True,'finite_all_updates':True,
                'synthetic_parameter_update_each_iteration':True,'initial_renderer_sha256':initial_hash,
                'final_diagnostic_renderer_sha256':final_hash,'templates_sha256':tensor_hash(template_cpu),
                'trained_renderer_exported':False,'PNG_exported':False,'main_bank_reuse_allowed':False,
                'preparation_seconds':preparation_seconds,'template_decode_and_transfer_seconds':template_io_seconds,
                'verification_seconds_excluded_from_update_timing':verification_seconds,
                'progress_IO_seconds_excluded':progress_io_seconds,'total_runner_seconds':time.perf_counter()-start,
                'GPU':torch.cuda.get_device_name(),'torch_version':torch.__version__,
                'coefficient_selection':False,'coefficients_numerical_test_only':True,
                'main_synthesis_updates':0,'new_main_banks':0,'new_patient_utility':False,
                'Q_V_pixel_access':0,'recipient_model_runs':0,'new_DP_releases':0,'expert_reserved_access':False,
                'full_W1_runtime_freeze':False,'main_budget_frozen':False,'other_arms_run':False,'remote_upload':False}
        save(args.output/'full128_profile.json',record)
        print(json.dumps({k:record[k] for k in ('status','seconds_per_full_update','peak_bytes_all_updates','counts','access','total_runner_seconds')}),flush=True)
    except BaseException as exc:
        if handle is not None:handle.remove()
        save(args.output/'failure.json',{'status':'PROFILE_FAILED_OR_INTERRUPTED','exception':repr(exc),
             'completed_profile_updates':completed,'saved_samples':len(samples),'counts':counts,'access':access.report(),
             'automatic_retry':False,'main_bank_reuse_allowed':False,'seconds':time.perf_counter()-start})
        raise


if __name__=='__main__':main()
