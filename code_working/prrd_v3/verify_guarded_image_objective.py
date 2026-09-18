"""Public four-image fixtures only: stage3 new-objective gradient connection.

No optimizer updates, full-bank profile, Q/V pixels, recipients or DP. Bound
test coefficients exercise arithmetic, and are NOT main-study selection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import setup, OUT, sha, save, read, source_hashes, require, now
from .datasets import manifests, PixelAccess, public_templates
from .encoders import Encoder, state_hash, pil_tensor, BIO_PATH
from .render import Renderer
from .guarded_objectives import (ARMS, LearnerPolicy, bind_objective, one_pass, two_pass,
                                bank_moments)
from .guarded_readout import functional_loss
from .w1_boundary_debug import error, grads, grad_error, tensor_hash


class PixelTrace:
    """Identity views isolate each clean/augmented encoder-input pixel VJP."""
    def __init__(self, encoder):
        self.encoder=encoder;self.pixel_gradients=[]

    def forward_paths(self, x):
        view=x.view_as(x)
        if view.requires_grad:
            index=len(self.pixel_gradients);self.pixel_gradients.append(None)
            def capture(g):
                self.pixel_gradients[index]=g.detach().cpu().clone()
                return g
            view.register_hook(capture)
        return self.encoder.forward_paths(view)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--step2',type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),'Preserve previous verification attempt')
    args.output.mkdir(parents=True)
    stage2=read(args.step2/'public_feature_paths_result.json')
    independent=read(args.step2/'independent_feature_statistics_verification.json')
    require(independent['status'].startswith('PASS'),'Step2 independent statistics not verified')
    inputs={str(args.step2/k):v for k,v in stage2['artifacts'].items()}
    previous=read(args.step2/'feature_path_contract.json')
    inputs.update(previous['inputs'])
    inputs[str(BIO_PATH)]=previous['checkpoint_sha256']
    scale_path=args.step2/'source_P_relation_scales.json'
    scales=read(scale_path)['scales'];inputs[str(scale_path)]=sha(scale_path)
    for path,expected in inputs.items():require(sha(path)==expected,'Changed stage2 bound input: '+path)
    # Verify unchanged runtime prerequisites, not the entire historical source tree.
    for name in ('encoders.py','relation_paths.py','guarded_readout.py','objectives.py','resampling.py','datasets.py'):
        path=Path(__file__).with_name(name)
        require(sha(path)==previous['source_sha256'][str(path)],'Changed stage2 prerequisite: '+name)
    policy=LearnerPolicy(1.,.1,kappa=.1)
    cases=[{'seed':101,'arm':arm,'microbatch':micro,'policy':asdict(policy)}
           for arm in ARMS for micro in (1,3)]
    cases += [{'seed':202,'arm':arm,'microbatch':2,'policy':asdict(policy)} for arm in ('A','C','R_joint')]
    cases += [{'seed':101,'arm':'C','microbatch':2,'policy':asdict(LearnerPolicy(1.,.1,rho=rho))}
              for rho in (0.,100.)]
    cfg={'schema':'prrd.step3.public-objective-gradient/v1','created':now(),
         'scope':'FOUR_IMAGE_PUBLIC_FIXTURES_NEW_LOSS_ONLY','cases':cases,
         'eta':1.,'coefficients_numerical_checks_only':True,'coefficient_selection':False,
         'renderer_step':500,'optimizer_steps':0,'public_fixture_indices':[0,32,64,96],
         'criteria':{'loss_abs':2e-6,'gradient_atol':2e-7,'gradient_rtol_peak':5e-4},
         'criteria_unchanged_from_stage1':True,'functional_placement':'clean once; augmented statistics only',
         'functional_metric':'fixed target M_T; learner cap uses own M',
         'inputs':inputs,'source_sha256':source_hashes(),
         'Q_V_pixels':False,'recipient_execution':False,'DP':False,'expert_reserved':False,
         'full128_profile':False,'synthesis_training':False,'remote_upload':False}
    save(args.output/'public_gradient_contract.json',cfg)
    setup();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    groups=manifests();access=PixelAccess(groups,('P',)).install()
    encoder=Encoder('source').use_projection(OUT/'source_P_projection.npz')
    encoder.use_relation_projection(args.step2/'source_P_relation_projection.npz')
    before=state_hash(encoder);counts={'forward_calls':0,'forward_images':0,'backward_calls':0,'backward_images':0}
    def model_hook(module,inputs,output):
        n=len(inputs[0]);counts['forward_calls']+=1;counts['forward_images']+=n
        f=output.projected_global_embedding
        if f.requires_grad:
            def backward(g):
                counts['backward_calls']+=1;counts['backward_images']+=n
                return g
            f.register_hook(backward)
    handle=encoder.model.register_forward_hook(model_hook)
    targets={}
    with np.load(args.step2/'source_P_relation_targets.npz',allow_pickle=False) as archive:
        for arm in ARMS:
            source='C' if arm in ('A','B') else arm
            keys=['m','A'] + ([] if arm in ('A','B') else ['delta','C'])
            if arm=='R_joint':keys+=['u','U']
            targets[arm]={key:archive[source+'_'+key].copy() for key in keys}
    fixtures={};templates={}
    for seed in (101,202):
        chosen=public_templates(groups['P'],seed);selected=[chosen[i] for i in cfg['public_fixture_indices']]
        x=torch.stack([TF.resize(pil_tensor(access.image(row)),[224,224],antialias=True) for row in selected])
        fixtures[str(seed)]={'images':[{k:row[k] for k in ('image_id','patient_id','sha256')} for row in selected],
                             'templates_sha256':tensor_hash(x)}
        torch.save({'seed':seed,'templates':x},args.output/f'fixture_{seed}.pt')
        templates[seed]=x.cuda()
    save(args.output/'fixtures.json',fixtures)
    rows=[];functional_probe=None
    for index,case in enumerate(cases):
        seed,arm,micro=case['seed'],case['arm'],case['microbatch']
        policy=LearnerPolicy(**case['policy'])
        target=bind_objective(targets[arm],arm,scales,policy,eta=cfg['eta'],device='cuda')
        fixed_w=target.functional_target.weight.clone();fixed_m=target.functional_target.matrix.clone()
        model=Renderer(templates[seed],arm,seed);model.set_step(500)
        initial_hash=state_hash(model)
        ref=PixelTrace(encoder)
        result=one_pass(model,ref,target,500,seed,micro);result['loss'].backward()
        reference=grads(model)
        values={k:float(result[k].detach()) for k in ('loss','statistics','augmentation_matching','functional','regularization')}
        alpha=float(result['readout'].alpha.detach())
        require(all(v is not None and bool(torch.isfinite(v).all()) for v in ref.pixel_gradients),
                'Missing/nonfinite reference image gradient')
        del result
        replay=PixelTrace(encoder)
        trace=two_pass(model,replay,target,500,seed,micro)
        gradient=grad_error(grads(model),reference)
        require(len(ref.pixel_gradients)==len(replay.pixel_gradients),'Pixel trace length mismatch')
        pixels=grad_error(replay.pixel_gradients,ref.pixel_gradients)
        loss_abs=abs(trace['loss']-values['loss'])
        gp=gradient['max_abs']<=2e-7+5e-4*gradient['reference_peak']
        pp=pixels['max_abs']<=2e-7+5e-4*pixels['reference_peak']
        lp=loss_abs<=2e-6
        fixed=bool(torch.equal(fixed_w,target.functional_target.weight) and torch.equal(fixed_m,target.functional_target.matrix))
        unchanged=state_hash(model)==initial_hash
        row=dict(case,loss=values,replay=trace,gradient=gradient,pixel_gradient=pixels,
                 loss_abs=loss_abs,reference_alpha=alpha,gradient_pass=gp,pixel_gradient_pass=pp,
                 loss_pass=lp,target_fixed=fixed,renderer_unchanged=unchanged)
        rows.append(row);save(args.output/f'case_{index:02d}.json',row)
        print(json.dumps({'case':index,'arm':arm,'microbatch':micro,'seed':seed,
                          'gradient_max_abs':gradient['max_abs'],'pixel_max_abs':pixels['max_abs'],
                          'loss_abs':loss_abs,'alpha':alpha,'passed':gp and pp and lp and fixed and unchanged}),flush=True)
        require(gp and pp and lp and fixed and unchanged,'New objective gradient or fixed-state check failed')
        if seed==101 and arm=='C' and micro==1:
            # A FUNCTION-ONLY image derivative is stronger evidence than a
            # nonzero total gradient that could come solely from priors/stats.
            x=templates[seed].detach().clone().requires_grad_(True)
            paths=[encoder.forward_paths(v) for v in x.split(1)]
            paths={k:torch.cat([p[k] for p in paths]) for k in ('z','a')}
            moments=bank_moments(paths,arm,scales);student=policy.solve(moments)
            functional=functional_loss(student.weight,target.functional_target)
            gx,=torch.autograd.grad(functional,x)
            functional_probe={'loss':float(functional.detach()),'gradient_norm':float(gx.norm()),
                              'gradient_max_abs':float(gx.abs().max()),'finite':bool(torch.isfinite(gx).all()),
                              'images':len(x),'includes_statistics_or_image_priors':False}
            require(functional_probe['finite'] and functional_probe['gradient_norm']>0,
                    'Functional loss is detached from image')
            torch.save({'pixels':x.detach().cpu(),'functional_pixel_gradient':gx.cpu()},args.output/'functional_image_probe.pt')
            del x,paths,moments,student,functional,gx
        del ref,replay,model,reference,target;torch.cuda.empty_cache()
    handle.remove();torch.cuda.synchronize()
    require(state_hash(encoder)==before,'Frozen source model/buffers/projections changed')
    require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()),'Source parameter gradient leakage')
    require(source_hashes()==cfg['source_sha256'],'Code changed during declared verification')
    summary={'status':'PASS_NEW_OBJECTIVE_ACTUAL_IMAGE_GRADIENT_ONLY','conditions':len(rows),'rows':rows,
             'max_gradient_abs':max(r['gradient']['max_abs'] for r in rows),
             'max_gradient_relative_l2':max(r['gradient']['relative_l2'] for r in rows),
             'max_pixel_gradient_abs':max(r['pixel_gradient']['max_abs'] for r in rows),
             'max_pixel_gradient_relative_l2':max(r['pixel_gradient']['relative_l2'] for r in rows),
             'max_loss_abs':max(r['loss_abs'] for r in rows),'functional_only_image_probe':functional_probe,
             'source_state_unchanged':True,'source_state_sha256':before,'model_parameters_frozen':True,
             'target_fixed_all_cases':True,'renderer_updated':False,'counts':counts,'access':access.report(),
             'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'GPU':torch.cuda.get_device_name(),
             'torch_version':torch.__version__,'contract_sha256':sha(args.output/'public_gradient_contract.json'),
             'coefficients_numerical_checks_only':True,'full_W1_runtime_passed':False,'full128_profile':False,
             'new_synthesis_updates':0,'Q_V_pixel_access':0,'recipient_model_runs':0,'new_DP_releases':0,
             'expert_reserved_access':False,'new_patient_utility':False,'remote_upload':False}
    save(args.output/'image_gradient_verification.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','conditions','max_gradient_abs','max_gradient_relative_l2',
                      'max_pixel_gradient_abs','max_loss_abs','functional_only_image_probe','counts','access','seconds')}),flush=True)


if __name__=='__main__':main()
