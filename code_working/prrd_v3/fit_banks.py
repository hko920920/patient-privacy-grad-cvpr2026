"""Bind frozen targets/public templates to the reusable guarded bank runtime."""
from __future__ import annotations
from dataclasses import asdict
import json
import random
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import CODE, OUT, require, sha, setup, source_hashes
from .datasets import manifests, PixelAccess
from .encoders import Encoder, pil_tensor, state_hash
from .render import Renderer
from .guarded_objectives import LearnerPolicy, bind_objective
from .bank_runtime import digest, tree_digest, run_bank, atomic_json

PATHS = CODE/'_reports/prrd_step2_paths_20260918_v1/public_paths'


def contract_document(path):
    text = Path(path).read_text(encoding='utf-8')
    fence = chr(96)*3
    return json.loads(text.split(fence+'json\n', 1)[1].split('\n'+fence, 1)[0])


def bound_path(name):
    if name.startswith('CODE/'):
        return CODE/name[5:]
    if name.startswith('RESEARCH/'):
        from .contracts import RESEARCH
        return RESEARCH/name[9:]
    return Path(name)


def check_frozen_inputs(base):
    for group in ('source_sha256', 'dependency_sha256', 'input_and_evidence_sha256'):
        for name, expected in base[group].items():
            require(sha(bound_path(name)) == expected, 'Frozen input/source changed: '+name)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def execute_bound_job(job, bank_directory, *, resume=False, stop_after=None, deadline=None,
                      perturb_rng_before_resume=False):
    """Both technical verification and authorized main runs call this path."""
    require(job['artifact_kind'] in ('MAIN_BANK', 'TECHNICAL_TEST_ONLY'), 'Unknown job scope')
    microbatch = job.get('microbatch', 4)
    require(type(microbatch) is int and microbatch in (4, 8, 16), 'Unverified microbatch')
    require(sha(job['base_contract']) == job['base_contract_sha256'], 'Base contract changed')
    base = contract_document(job['base_contract'])
    check_frozen_inputs(base)
    require(source_hashes() == job['implementation_source_sha256'], 'Implementation changed')
    require(sha(job['target_file']) == job['target_sha256'], 'Target file changed')
    if job['artifact_kind'] == 'TECHNICAL_TEST_ONLY':
        require(microbatch == 4, 'Historical small resume fixture uses microbatch4')
        require(job['images'] == 4 and job['updates'] == 84 and job['arm'] == 'C',
                'Bounded public verification expanded')
        require(Path(job['target_file']).resolve() == (PATHS/'source_P_relation_targets.npz').resolve()
                and job['target_prefix'] == 'C_', 'Technical target is not the existing public C target')
    setup()
    groups = manifests()
    access = PixelAccess(groups, ('P',)).install()
    by_id = {r['image_id']: r for r in groups['P']}
    chosen = []
    for template in job['templates']:
        require(template['image_id'] in by_id, 'Template is outside P')
        row = by_id[template['image_id']]
        require(row['sha256'] == template['sha256'], 'Template hash differs from P manifest')
        chosen.append(row)
    require(len(chosen) == job['images'], 'Wrong template count')
    encoder = Encoder('source').use_projection(OUT/'source_P_projection.npz')
    encoder.use_relation_projection(PATHS/'source_P_relation_projection.npz')
    decoded = {}
    for row in chosen:
        if row['image_id'] not in decoded:
            decoded[row['image_id']] = TF.resize(pil_tensor(access.image(row)), [224, 224], antialias=True)
    template = torch.stack([decoded[row['image_id']] for row in chosen]).cuda()
    policy = LearnerPolicy(**job['learner_policy'])
    keys = ['m', 'A'] + ([] if job['arm'] in ('A', 'B') else ['delta', 'C'])
    if job['arm'] == 'R_joint':
        keys += ['u', 'U']
    with np.load(job['target_file'], allow_pickle=False) as data:
        target = {k: data[job['target_prefix']+k].copy() for k in keys}
    require(target['m'].shape == (2, 16) and target['A'].shape == (2, 16, 16),
            'Unexpected target coordinate system')
    scales = json.loads((PATHS/'source_P_relation_scales.json').read_text(encoding='utf-8'))['scales']
    objective = bind_objective(target, job['arm'], scales, policy, eta=job['eta'], device='cuda')
    renderer = Renderer(template, job['arm'], job['seed'])
    optimizer = torch.optim.AdamW(renderer.parameters(), lr=.01, weight_decay=0.,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    spec = {
        'schema': 'prrd.bank-run/v1', 'id': job['id'], 'arm': job['arm'], 'seed': job['seed'],
        'images': job['images'], 'updates': job['updates'], 'microbatch': microbatch,
        'checkpoint_every': 25, 'artifact_kind': job['artifact_kind'],
        'contract_sha256': job['base_contract_sha256'],
        'implementation_sha256': digest(job['implementation_source_sha256']),
        'job_binding_sha256': digest(job), 'target_sha256': job['target_sha256'],
        'templates_tensor_digest': tree_digest(template),
        'source_encoder_state_sha256': state_hash(encoder),
        'learner_policy': asdict(policy), 'eta': job['eta'],
        'scales': scales,
    }
    seed_all(job['seed'])
    # Deliberately different startup RNG proves the resumed state is restored.
    if resume and perturb_rng_before_resume:
        seed_all(987654321)
        random.random(); np.random.rand(7); torch.rand(11); torch.rand(13, device='cuda')
    counts = {'forward_calls': 0, 'forward_images': 0, 'backward_calls': 0, 'backward_images': 0}
    def forward(module, inputs, output):
        n = len(inputs[0]); counts['forward_calls'] += 1; counts['forward_images'] += n
        if output.projected_global_embedding.requires_grad:
            def backward(g):
                counts['backward_calls'] += 1; counts['backward_images'] += n
                return g
            output.projected_global_embedding.register_hook(backward)
    handle = encoder.model.register_forward_hook(forward)
    def progress(step, trace):
        if step % 20 == 0 or step in (1, 81, 82, 84):
            print(json.dumps({'bank': job['id'], 'successful_step': step,
                              'technical_only': job['artifact_kind'] != 'MAIN_BANK'}), flush=True)
    try:
        result = run_bank(bank_directory, renderer, encoder, objective, optimizer, spec,
                          resume=resume, stop_after=stop_after, deadline=deadline, on_step=progress)
    finally:
        handle.remove()
    updates = result['updates_this_process']
    require(counts == {'forward_calls': 4*updates, 'forward_images': 4*job['images']*updates,
                       'backward_calls': 2*updates, 'backward_images': 2*job['images']*updates}
            if job['images'] == 4 else
            counts['forward_images'] == 4*job['images']*updates and
            counts['backward_images'] == 2*job['images']*updates, 'Execution count mismatch')
    require(source_hashes() == job['implementation_source_sha256'], 'Source changed during bank')
    result.update(counts=counts, access=access.report(), fixed_source=True,
                  main_bank_reuse_allowed=job['artifact_kind'] == 'MAIN_BANK')
    return result
