"""A1 adapter using established atomic checkpoint/RNG/PNG primitives.

Targets are immutable condition signals, not historical PRRD moments.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import CODE, OUT, require, sha, setup, source_hashes
from prrd_v3.datasets import manifests, PixelAccess, png_roundtrip
from prrd_v3.encoders import Encoder, pil_tensor, state_hash
from prrd_v3.render import Renderer
from prrd_v3.bank_runtime import (digest, tree_digest, atomic_bytes, atomic_json, append_event, file_lock,
                                 save_checkpoint, load_checkpoint, checkpoint_record, csv_bytes,
                                 image_layout, verify_artifact)
from .signals import objective_digest
from .target_io import load_target
from .objectives import two_pass


OPTIMIZER = {'name':'AdamW', 'lr':.01, 'weight_decay':0., 'betas':[.9,.999],
             'eps':1e-8, 'foreach':False}


def code_bindings():
    result = source_hashes()
    for name in ('signals.py','objectives.py','target_io.py','runtime.py'):
        p = Path(__file__).with_name(name)
        result[str(p)] = sha(p)
    return result


def seed_all(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def validate_job(job, *, authorization=None):
    require(job['schema'] == 'receiver.A1-bank-job/v1', 'Wrong A1 job schema')
    require(job['seed'] == 101 and job['encoder_id'] == 'biovil', 'A1 recipe changed')
    require(job['microbatch'] in (4,16) and job['images'] in (4,128), 'Unverified partition/size')
    require(job['optimizer'] == OPTIMIZER, 'Optimizer differs from fixed starting recipe')
    require(job['rule_sha256'] == json.loads(Path(job['handoff']).read_text())['rule_sha256'],
            'Job/target rule mismatch')
    require(sha(job['handoff']) == job['handoff_sha256'], 'Handoff bytes changed')
    require(all(sha(p) == h for p,h in job['code_bindings'].items()), 'Runtime dependency changed')
    kind = job['artifact_kind']
    if kind == 'TECHNICAL_TEST_ONLY':
        require(job['images'] == 4 and job['updates'] == 3 and job['microbatch'] == 4
                and job['population'] == 'P', 'Public resume fixture expanded')
    elif kind == 'COST_ONLY':
        require(job['images'] == 128 and job['updates'] == 4 and job['microbatch'] == 16
                and job['population'] == 'P', 'Public full-bank measurement expanded')
    else:
        require(kind == 'MAIN_BANK' and job['images'] == 128 and job['updates'] == 500
                and job['microbatch'] == 16 and job['population'] == 'pooled', 'Invalid A1 main job')
        require(authorization is not None and authorization.get('job_sha256') == digest(job)
                and job['id'] in authorization.get('allowed_bank_ids', [])
                and authorization.get('maximum_seconds', 0) > 0,
                'A1 main execution is outside this preparation task')
    labels = tuple([0]*(job['images']//2)+[1]*(job['images']//2))
    return labels


def export(directory, renderer, job, spec, signature, checkpoint):
    require(checkpoint['completed_updates'] == job['updates'], 'Cannot export an intermediate bank')
    final = directory/'artifact'
    with torch.no_grad():
        images = torch.cat([renderer(list(range(i,min(i+job['microbatch'],renderer.n)))).cpu()
                            for i in range(0,renderer.n,job['microbatch'])])
    require(bool(torch.isfinite(images).all()), 'Nonfinite export')
    expected = np.rint(np.clip(images[:,0].numpy(),0,1)*255).astype(np.uint8)
    if final.exists():
        report = verify_artifact(final, spec, signature, expected)
        seal = json.loads((final/'artifact_seal.json').read_text())
        require(seal['checkpoint_sha256'] == checkpoint['sha256'], 'Export checkpoint changed')
        return report
    staging = directory/('.export_'+uuid.uuid4().hex)
    (staging/'images').mkdir(parents=True)
    rows, pairs = image_layout(spec)  # arm B => marginal images, no invented patient pairs
    for i,row in enumerate(rows):
        _, data = png_roundtrip(images[i,0].numpy())
        atomic_bytes(staging/row['file'], data)
        row['png_sha256'] = sha(staging/row['file'])
    atomic_bytes(staging/'images.csv', csv_bytes(rows,list(rows[0])))
    atomic_bytes(staging/'pairs.csv', csv_bytes(pairs,['virtual_pair_id','positive_id','negative_id']))
    atomic_json(staging/'learning_contract.json', {
        'schema':'receiver.synthetic-learning/v1', 'source_objective':'fixed-condition linear CE gradient cosine',
        'conditions':4, 'condition_rule_sha256':job['rule_sha256'],
        'labels':'synthetic class targets; 64/64 for main bank', 'real_patient_ids_exported':False,
        'receiver_evaluation':'existing point-only patient/class-weighted ridge0.1 development readout',
        'relation_loss':False, 'functional_matching':False,
        'artifact_kind':job['artifact_kind'], 'main_reuse_allowed':job['artifact_kind']=='MAIN_BANK',
        'privacy_receipt':None, 'DP_applied':False,
    })
    atomic_json(staging/'provenance_public.json', {
        'schema':'receiver.synthetic-provenance/v1','bank_id':job['id'], 'seed':job['seed'],
        'initialization':'PUBLIC_P_ONLY', 'successful_updates':job['updates'],
        'artifact_kind':job['artifact_kind'],'main_reuse_allowed':job['artifact_kind']=='MAIN_BANK',
        'condition_rule_sha256':job['rule_sha256'],'DP_applied':False,
    })
    hashes = {p.relative_to(staging).as_posix():sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    atomic_json(staging/'artifact_seal.json', {
        'signature':signature, 'bank_id':job['id'], 'successful_updates':job['updates'],
        'artifact_kind':job['artifact_kind'], 'checkpoint_sha256':checkpoint['sha256'],
        'files_sha256':hashes})
    report = verify_artifact(staging, spec, signature, expected)
    require(not final.exists(), 'Completed artifact overwrite')
    os.rename(staging, final)
    return report


def finite_trace(trace):
    return all(finite_trace(v) if isinstance(v,dict) else math.isfinite(v) for v in trace.values())


def execute(job, directory, *, resume=False, stop_after=None, perturb_rng=False, authorization=None):
    labels = validate_job(job, authorization=authorization)
    profile = job['artifact_kind'] == 'COST_ONLY'
    require(not (profile and (resume or stop_after is not None)), 'Cost fixture cannot resume')
    objective, rule = load_target(job['handoff'], labels, population=job['population'], device='cuda')
    setup()
    directory = Path(directory)
    if resume:
        require(directory.is_dir(), 'No checkpoint directory')
    else:
        directory.mkdir(parents=True,exist_ok=False)
    groups = manifests()
    access = PixelAccess(groups,('P',)).install()
    lookup = {r['image_id']:r for r in groups['P']}
    chosen = []
    for row in job['templates']:
        require(row['image_id'] in lookup and lookup[row['image_id']]['sha256'] == row['sha256'],
                'Template not in public P')
        chosen.append(lookup[row['image_id']])
    require(len(chosen) == job['images'], 'Template count changed')
    decoded = {}
    for row in chosen:
        if row['image_id'] not in decoded:
            decoded[row['image_id']] = TF.resize(pil_tensor(access.image(row)),[224,224],antialias=True)
    templates = torch.stack([decoded[r['image_id']] for r in chosen]).cuda()
    encoder = Encoder('source').use_projection(rule['projection'])
    fixed_encoder = state_hash(encoder)
    fixed_target = objective_digest(objective)
    renderer = Renderer(templates,'B',job['seed'])
    options = {k:v for k,v in job['optimizer'].items() if k != 'name'}
    options['betas'] = tuple(options['betas'])
    optimizer = torch.optim.AdamW(renderer.parameters(),**options)
    seed_all(job['seed'])
    spec = dict(job, arm='B', template_tensor_digest=tree_digest(templates),
                fixed_encoder_sha256=fixed_encoder, objective_digest=fixed_target)
    signature = digest(spec)
    counts = {'forward_images':0,'backward_images':0}
    def count_forward(module, inputs, output):
        n = len(inputs[0])
        counts['forward_images'] += n
        if output.projected_global_embedding.requires_grad:
            def count_backward(g):
                counts['backward_images'] += n
                return g
            output.projected_global_embedding.register_hook(count_backward)
    handle = encoder.model.register_forward_hook(count_forward)
    start = time.monotonic()
    deadline = start+authorization['maximum_seconds'] if authorization else None
    stop = job['updates'] if stop_after is None else stop_after
    require(0 <= stop <= job['updates'], 'Stop outside job')
    torch.cuda.reset_peak_memory_stats()
    with file_lock(directory/'.process.lock'):
        require(not (directory/'COMPLETED.json').exists(), 'Completed bank cannot resume or overwrite')
        if resume:
            require(json.loads((directory/'run_spec.json').read_text()) == spec, 'Resume signature changed')
            if perturb_rng:
                seed_all(654321)
                random.random(); np.random.rand(3); torch.rand(3); torch.rand(3,device='cuda')
            done, restoration = load_checkpoint(directory, renderer, optimizer, signature)
        else:
            renderer.set_step(1)
            done, restoration = 0, None
            atomic_json(directory/'run_spec.json',spec)
            if not profile:
                save_checkpoint(directory,renderer,optimizer,0,signature)
        require(done <= stop, 'Stop precedes checkpoint')
        initial_state = state_hash(renderer)
        prior_done = done
        durations = []
        attempt = uuid.uuid4().hex
        stopped_for_budget = False
        while done < stop:
            if deadline is not None and time.monotonic()+max(15.,(durations[-1] if durations else 0)*1.25) >= deadline:
                stopped_for_budget = True
                break
            step = done+1
            renderer.set_step(500 if profile else step)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            trace = two_pass(renderer, {'biovil':encoder}, objective, job['microbatch'])
            require(finite_trace(trace), 'Nonfinite condition loss')
            active = [p for p in renderer.parameters() if p.requires_grad]
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in active),
                    'Missing/nonfinite image parameter gradient')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in renderer.parameters()), 'Nonfinite parameter')
            require(all(bool(torch.isfinite(v).all()) for s in optimizer.state.values()
                        for v in s.values() if isinstance(v,torch.Tensor)), 'Nonfinite optimizer state')
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-tick)
            done = step
            append_event(directory/'updates.jsonl',{'attempt':attempt,'pid':os.getpid(),'successful_step':step,
                          'pyramid_position':500 if profile else step,'active_levels':renderer.active,
                          'trace':trace,'seconds':durations[-1]})
            if step % 25 == 0 and not profile:
                save_checkpoint(directory,renderer,optimizer,done,signature)
            if profile or step in (1,2,3) or step % 25 == 0:
                print(json.dumps({'bank':job['id'],'completed':done,'seconds':durations[-1]}),flush=True)
        png = None
        if not profile:
            record,_ = checkpoint_record(directory)
            if record['completed_updates'] != done:
                record = save_checkpoint(directory,renderer,optimizer,done,signature)
            if done == job['updates']:
                png = export(directory,renderer,job,spec,signature,record)
                atomic_json(directory/'COMPLETED.json',{'status':'COMPLETE_'+job['artifact_kind'],
                    'signature':signature,'completed_updates':done,'checkpoint_sha256':record['sha256'],
                    'artifact_seal_sha256':sha(directory/'artifact/artifact_seal.json'),
                    'main_reuse_allowed':job['artifact_kind']=='MAIN_BANK'})
        handle.remove()
        require(state_hash(encoder) == fixed_encoder, 'Frozen encoder changed')
        require(all(p.grad is None and not p.requires_grad for p in encoder.parameters()), 'Encoder gradient leak')
        require(objective_digest(objective) == fixed_target, 'Condition target changed')
        require(all(sha(p) == h for p,h in job['code_bindings'].items()), 'Runtime code changed')
        require(counts == {'forward_images':8*job['images']*(done-prior_done),
                           'backward_images':4*job['images']*(done-prior_done)}, 'Condition execution counts')
        changed = state_hash(renderer) != initial_state
        require(done == prior_done or changed, 'Optimizer did not update image parameters')
        result = {'status':'COST_ONLY_COMPLETE' if profile else ('COMPLETE' if done==job['updates'] else 'PAUSED'),
                  'artifact_kind':job['artifact_kind'],'pid':os.getpid(),'updates_this_process':done-prior_done,
                  'completed_updates':done,'next_step':done+1,'active_levels':renderer.active,
                  'restoration':restoration,'source_unchanged':True,'target_unchanged':True,
                  'parameter_updated':changed,'signature':signature,'counts':counts,'access':access.report(),
                  'PNG':png,'duration_seconds':durations,'bank_seconds':time.monotonic()-start,
                  'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
                  'stopped_for_budget':stopped_for_budget,'main_reuse_allowed':job['artifact_kind']=='MAIN_BANK'}
        atomic_json(directory/('attempt_'+attempt+'.json'),result)
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--stop-after',type=int)
    ap.add_argument('--perturb-rng',action='store_true')
    args = ap.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    require(job['artifact_kind'] != 'MAIN_BANK', 'This preparation CLI cannot launch main synthesis')
    result = execute(job,args.output,resume=args.resume,stop_after=args.stop_after,perturb_rng=args.perturb_rng)
    atomic_json(args.result,result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
