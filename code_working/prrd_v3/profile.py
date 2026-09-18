"""Public-only runtime parity and full128 clean+augmented two-pass timing."""
from __future__ import annotations
import copy,json,time
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from .contracts import *
from .datasets import manifests,PixelAccess,public_templates,png_roundtrip
from .encoders import Encoder,state_hash,pil_tensor
from .patient_moments import patient_contributions,aggregate
from .render import Renderer,augment,augment_parameters
from .objectives import one_pass,two_pass,bank_moments
from .resampling import resize,affine_sample
from .transfer_rn18 import public_replay

def target_public(encoder,group):
    cache=np.load(OUT/'source_P_features.npz',allow_pickle=False)
    require(list(cache['image_ids'])==[r['image_id'] for r in group],'Public feature order mismatch')
    with torch.no_grad(): z=encoder.project(torch.from_numpy(cache['features']).cuda()).cpu().double().numpy()
    patients=patient_contributions(z,[int(r['label']) for r in group],[r['patient_id'] for r in group])
    return aggregate(patients),aggregate(patients,'joint_pair')

def run():
    setup();cfg=read(OUT/'w0_bindings.json');check_bindings(cfg)
    require(not (OUT/'public_profile.json').exists(),'Profile already complete')
    groups=manifests();access=PixelAccess(groups,('P',)).install()
    # Check forbidden known Q/V and unknown raw paths without reading their pixels.
    for r in (groups['Q'][0],groups['V'][0],{'path':str(Path(groups['P'][0]['path']).parent/'not_allowlisted.png')}):
        try: access.check(r['path'])
        except PermissionError: pass
        else: raise RuntimeError('Pixel boundary did not reject path')
    encoder=Encoder('source').use_projection(OUT/'source_P_projection.npz')
    before=state_hash(encoder.model); target,joint=target_public(encoder,groups['P'])
    chosen=public_templates(groups['P'],101)
    tick=time.perf_counter()
    templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in chosen]).cuda()
    io_seconds=time.perf_counter()-tick
    parity=[]
    # Four PUBLIC images, one per point/pair class slot; full one-pass graph fits.
    tiny=templates[[0,32,64,96]].clone()
    for arm in ('A','C','R_joint'):
        model=Renderer(tiny,arm,101);model.set_step(500)
        t=joint if arm=='R_joint' else target
        model.zero_grad();torch.cuda.reset_peak_memory_stats()
        loss=one_pass(model,encoder,t,500,101);loss.backward()
        direct=[p.grad.clone() for p in model.parameters()]
        for micro in (1,2,4):
            # Renderer and augmentation must use the same batch partition too.
            # Encoder-only partitioning left different FP32 pixels in the two
            # paths. Keep full-batch discrepancy as a separate diagnostic.
            model.zero_grad();matched_loss=one_pass(model,encoder,t,500,101,microbatch=micro);matched_loss.backward()
            matched=[p.grad.clone() for p in model.parameters()]
            trace=two_pass(model,encoder,t,500,101,micro)
            absolute=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),matched))
            scale=max(float(g.abs().max()) for g in matched)
            relative=absolute/max(scale,1e-12)
            cross_batch=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),direct))
            print(json.dumps({'phase':'gradient_parity','arm':arm,'microbatch':micro,'gradient_abs':absolute,
                              'gradient_peak':scale,'relative':relative,'cross_batch_gradient_abs':cross_batch,
                              'loss_abs':abs(trace['loss']-float(matched_loss.detach()))}),flush=True)
            require(absolute<=2e-7+5e-4*scale,'Medical encoder one/two-pass gradient mismatch')
            require(abs(trace['loss']-float(matched_loss.detach()))<=2e-6,'Medical encoder one/two-pass loss mismatch')
            parity.append({'arm':arm,'images':4,'microbatch':micro,'loss_abs_difference':abs(trace['loss']-float(matched_loss.detach())),
                           'gradient_max_abs':absolute,'gradient_max_relative_to_peak':relative,
                           'cross_batch_FP32_gradient_max_abs':cross_batch,
                           'comparison':'one-pass retained graph vs two-pass replay with identical renderer/augmentation/encoder microbatch partition',
                           'cross_batch_bitwise_equality_claimed':False})
        del model,direct,loss;torch.cuda.empty_cache()
    # GPU sampling operation equals reference CPU gradient (CUDA official backward
    # is unsupported in deterministic mode). This checks the actual substitute.
    torch.manual_seed(4);x=torch.rand(2,1,16,16,device='cuda',requires_grad=True)
    params=augment_parameters(2,101,1,False,'cuda')
    ids=list(range(2));out=augment(x,params,ids);g,=torch.autograd.grad(out.square().sum(),x)
    xc=x.detach().cpu().requires_grad_(True)
    grid=torch.nn.functional.affine_grid(params[0].cpu(),xc.shape,align_corners=False)
    ref=(torch.nn.functional.grid_sample(xc,grid,align_corners=False)*params[1].cpu()[:,None,None,None]).clamp(0,1)
    gc,=torch.autograd.grad(ref.square().sum(),xc)
    torch.testing.assert_close(g.cpu(),gc,rtol=1e-5,atol=5e-6)
    cuda_affine_error=float((g.cpu()-gc).abs().max())
    del x,xc,out,g,gc,ref
    candidates=[];selected=None
    for micro in (4,2,1):
        model=Renderer(templates,'C',101);model.set_step(500)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0,betas=(.9,.999),foreach=False)
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            trace=two_pass(model,encoder,target,500,101,micro);optimizer.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick
            candidates.append({'microbatch':micro,'seconds':elapsed,'peak_allocated':torch.cuda.max_memory_allocated(),
                               'peak_reserved':torch.cuda.max_memory_reserved(),'status':'PASS_FULL128_UPDATE'})
            selected=micro
            del model,optimizer;torch.cuda.empty_cache();break
        except torch.cuda.OutOfMemoryError:
            candidates.append({'microbatch':micro,'status':'CUDA_OOM_NO_MAIN_UPDATES'})
            del model,optimizer;torch.cuda.empty_cache()
    require(selected is not None,'No valid microbatch')
    print(json.dumps({'phase':'public_profile_selected','microbatch':selected,'candidate':candidates[-1]}),flush=True)
    # Profile max renderer complexity throughout, fresh optimizer, no best checkpoint.
    model=Renderer(templates,'C',101);model.set_step(500)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0,betas=(.9,.999),foreach=False)
    samples=[];torch.cuda.reset_peak_memory_stats()
    for i in range(50):
        torch.cuda.synchronize();tick=time.perf_counter()
        trace=two_pass(model,encoder,target,500+i,101,selected);optimizer.step();torch.cuda.synchronize()
        seconds=time.perf_counter()-tick
        require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite public profile state')
        sample=dict(update=i+1,warmup=i<20,seconds=seconds,**trace);samples.append(sample)
        # Append-only progress can survive an interruption without implying PASS.
        with (OUT/'public_profile_progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
        print(json.dumps({'phase':'public_profile','update':i+1,'seconds':seconds,'warmup':i<20}),flush=True)
    torch.cuda.synchronize()
    peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    witness=OUT/'public_png_witness';witness.mkdir(exist_ok=False)
    errors=[]
    with torch.no_grad():
        floats=model([0,32,64,96]); original_features=encoder(floats)
        decoded=[]
        for i,x in enumerate(floats.cpu().numpy()[:,0]):
            a,contents=png_roundtrip(x);path=witness/f'{i}.png';path.write_bytes(contents)
            # Re-open actual deployment-format bytes from disk.
            with Image.open(path) as im: decoded.append(np.asarray(im,dtype=np.float32)/255)
            errors.append(float(np.max(abs(x-a/255))))
        png_features=encoder(torch.from_numpy(np.stack(decoded))[:,None].cuda())
    require(state_hash(encoder.model)==before,'Source weights or BN buffers changed')
    quantization={'max_pixel_difference':max(errors),'max_source_feature_difference':float((original_features-png_features).abs().max()),
                  'images':4,'scope':'public profile witness only, not W2 bank'}
    del model,optimizer,encoder;torch.cuda.empty_cache()
    public_replay(groups['P'],access)
    timed=np.array([x['seconds'] for x in samples if not x['warmup']])
    record={'status':'PASS_PUBLIC_PROFILE_NOT_EFFICACY','microbatch':selected,'candidate_attempts':candidates,
            'warmup_updates':20,'timed_updates':30,'public_profile_optimizer_updates':51,
            'whole_bank_images':128,'augmentation_included':True,'renderer_all_levels_active':True,
            'seconds_per_update':{'mean':float(timed.mean()),'median':float(np.median(timed)),'p10':float(np.quantile(timed,.1)),
                                  'p90':float(np.quantile(timed,.9))},'samples':samples,
            'peak_bytes':peak,'template_decode_seconds':io_seconds,'medical_gradient_parity':parity,
            'cuda_affine_gradient_abs_error':cuda_affine_error,'quantization':quantization,
            'source_model_unchanged':True,'source_state_sha256':before,'access':access.report(),
            'new_W2_banks':0,'new_Q_or_V_pixels':0,'DP_expert_reserved_access':0}
    save(OUT/'public_profile.json',record)
    print(json.dumps({k:record[k] for k in ('status','microbatch','seconds_per_update','peak_bytes')}),flush=True)

if __name__=='__main__':run()
