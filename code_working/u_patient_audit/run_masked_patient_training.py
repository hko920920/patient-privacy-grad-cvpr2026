"""Replay frozen M1 training, then mask exactly eight patient loss slots.

The numerical training loop is extracted from the original frozen source. Only
the stated control slots change their contribution to the batch-mean loss.
"""
import argparse
import ast
from collections import Counter
import copy
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace

import torch

from .common import ROOT, RUN, digest, read_csv, seed, stable, verify_inputs
from .models import setup, load_cache, adapter_state
from . import train_coverage as original
from .train_pair import save as original_save
from .mofit_medical_adapter import tensor_hash
from .run_mofit_medical_kernel import load, new_json, now, assert_hashes

BASE = RUN / 'baseline_screen_20260915'
CONTRACT = BASE / 'masked_patient_training_contract_v1.json'
PREPARATION = BASE / 'masked_patient_training_preparation_v1'
OUTPUT = BASE / 'masked_patient_training_v1'
PRIOR = RUN / 'training_coverage_v2/model_1'
PID = '14393'
EXPECTED_SLOTS = [(44, 3, '00014393_006.png', 172), (83, 0, '00014393_002.png', 281),
    (380, 3, '00014393_006.png', 108), (399, 2, '00014393_002.png', 122),
    (474, 3, '00014393_006.png', 452), (592, 2, '00014393_002.png', 475),
    (742, 1, '00014393_006.png', 305), (904, 2, '00014393_002.png', 922)]


def schedule():
    rows = read_csv(RUN / 'cohort/model_1_train.csv')
    rows.sort(key=lambda r: (r['eval_role'] != 'background',
              stable('training-order', r['patient_id']), r['image_id']))
    assert len(rows) == 912
    batches, slots, counts = [], [], Counter()
    for step in range(1000):
        batch = [rows[i] for i in original.epoch_indices(len(rows), step)]
        images = [r['image_id'] for r in batch]
        ts = [seed('coverage-timestep', step, i) % 1000 for i in range(4)]
        mask = [int(r['patient_id'] != PID) for r in batch]
        batches.append(dict(update=step+1, image_ids=images, timesteps=ts,
                            noise_seed=seed('coverage-noise', step), control_mask=mask))
        counts.update(images)
        slots.extend((step+1, j, r['image_id'], ts[j]) for j, r in enumerate(batch)
                     if r['patient_id'] == PID)
    assert slots == EXPECTED_SLOTS
    old = torch.load(PRIOR / 'step_1000.pt', map_location='cpu', weights_only=True)
    assert dict(counts) == old['exposures'] and sum(counts.values()) == 4000
    return rows, batches, counts


def masked_loss(pred, target, masked_ordinals):
    # Zero whole per-example terms, retaining the original full-tensor mean and
    # its denominator 4*C*H*W. No renormalization of the three remaining terms.
    mask = torch.ones((4, 1, 1, 1), device=pred.device, dtype=torch.float32)
    mask[masked_ordinals] = 0.
    return ((pred.float()-target.float()).square()*mask).mean()


def nodes(text):
    return ast.parse(text).body


def transformed_kernel():
    source = inspect.getsource(original.run)
    module = ast.parse(source)
    fn = module.body[0]
    edits = Counter()
    newbody = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'out' for t in node.targets):
            node.value = ast.Name(id='branch_output', ctx=ast.Load())
            edits['output_path'] += 1
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sched' for t in node.targets):
            newbody += nodes('auditor.start_branch(unet,opt,scaler,rows,config)')
            edits['initial_state_audit'] += 1
        if isinstance(node, ast.While):
            body = []
            for item in node.body:
                if isinstance(item, ast.With):
                    body += nodes('auditor.capture_inputs(step,attempt,batch_rows,z,h,noise,t,noisy,target,scaler.get_scale())')
                    assert len(item.body) == 2 and isinstance(item.body[1], ast.Assign)
                    assert ast.unparse(item.body[1]) == 'loss = (pred.float() - target.float()).square().mean()'
                    exact_original_loss = copy.deepcopy(item.body[1])
                    change = nodes('if branch_kind == "control" and step in auditor.mask_steps:\n    loss = masked_loss(pred,target,auditor.mask_steps[step])\nelse:\n    pass')[0]
                    change.orelse = [exact_original_loss]
                    item.body[1] = change
                    body.append(item)
                    body += nodes('auditor.observe_prediction(step,attempt,pred,target,loss)')
                    edits['masked_control_loss_else_original_exact'] += 1
                    continue
                if isinstance(item, ast.Expr) and ast.unparse(item).startswith('exposures.update('):
                    change = nodes('if branch_kind == "control":\n    exposures.update(r["image_id"] for r in batch_rows if r["patient_id"] != "14393")\nelse:\n    pass')[0]
                    change.orelse = [copy.deepcopy(item)]
                    item = change
                    edits['control_effective_exposures'] += 1
                if isinstance(item, ast.If) and 'save_steps' in ast.unparse(item.test):
                    item.test = ast.BoolOp(op=ast.Or(), values=[ast.Compare(left=ast.Name(id='step', ctx=ast.Load()),
                        ops=[ast.Eq()], comparators=[ast.Constant(43)]), item.test])
                    edits['prefix43_extra_save'] += 1
                body.append(item)
                text = ast.unparse(item)
                if text == 'scaler.update()':
                    body += nodes('auditor.finish_attempt(step,attempt,old,scaler.get_scale(),norm)')
                    edits['overflow_audit'] += 1
                if text.startswith('losses.append('):
                    body += nodes('auditor.commit(step,attempt,losses[-1],norm)')
                    edits['committed_step_audit'] += 1
            node.body = body
        if isinstance(node, ast.Expr) and ast.unparse(node) == 'log.close()':
            newbody.append(node)
            newbody += nodes('auditor.finish_branch(unet,opt,scaler,step,attempt,exposures,losses)\ndel unet,opt,scaler,params\ngc.collect()\ntorch.cuda.empty_cache()')
            edits['replace_report_after_training_only'] += 1
            break
        newbody.append(node)
    fn.body = newbody
    expected = ['output_path', 'initial_state_audit', 'masked_control_loss_else_original_exact',
        'control_effective_exposures', 'prefix43_extra_save', 'overflow_audit',
        'committed_step_audit', 'replace_report_after_training_only']
    assert dict(edits) == {k: 1 for k in expected}, dict(edits)
    ast.fix_missing_locations(module)
    rendered = ast.unparse(module) + '\n'
    return module, rendered, dict(edits)


def same(a, b, prefix='root'):
    """Exact, dtype-aware recursive comparison; no tolerance or score selection."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, prefix
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), prefix
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), prefix
        for k in a:
            same(a[k], b[k], prefix + '.' + str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), prefix
        for k, (x, y) in enumerate(zip(a, b)):
            same(x, y, prefix + '.' + str(k))
    else:
        assert type(a) is type(b) and a == b, prefix


def frozen_fingerprint(unet):
    h = hashlib.sha256()
    n = 0
    for name, value in sorted(unet.state_dict().items()):
        if '.lora_' not in name:
            h.update(name.encode('utf-8'))
            h.update(tensor_hash(value.detach().cpu()).encode('ascii'))
            n += 1
    return dict(sha256=h.hexdigest(), tensors=n)


def prepare(path):
    assert not path.exists() and not PREPARATION.exists()
    config = original.contract()
    same(config, load(RUN / 'training_coverage_v2/protocol.json'))
    _, batches, counts = schedule()
    for step in [250, 1000]:
        q = torch.load(PRIOR / f'step_{step:04d}.pt', map_location='cpu', weights_only=True)
        same(q['contract'], config)
        assert q['step'] == q['attempt'] == step
    _, rendered, edits = transformed_kernel()
    PREPARATION.mkdir(parents=True, exist_ok=False)
    (PREPARATION / 'transformed_training_kernel.py').write_text(rendered, encoding='utf-8')
    new_json(PREPARATION / 'scheduled_batches.json', batches)
    frozen = dict(load(BASE / 'mofit_medical_remaining_pair_contract_v1.json')['frozen_sha256'])
    assert_hashes(frozen)
    for p in [Path(__file__), PREPARATION / 'transformed_training_kernel.py',
              PREPARATION / 'scheduled_batches.json', RUN / 'training_coverage_v2/protocol.json',
              PRIOR / 'step_0250.pt', PRIOR / 'step_1000.pt', PRIOR / 'training_trace.jsonl',
              RUN / 'cache/cache.pt', RUN / 'cache/summary.json', RUN / 'cohort/model_1_train.csv']:
        frozen[str(p.resolve())] = digest(p)
    c = dict(schema='masked-patient-training-contract/v1', created_utc=now(), current_step=2,
        source_model='model_1', patient_id=PID, original_training_contract=config,
        branch_order=['treatment', 'control'], steps_per_branch=1000,
        checkpoints=[43, 250, 1000], treatment_gate_steps=[250, 1000], control_prefix_gate=43,
        gate='Exact adapter/optimizer/scaler/step/attempt/exposures/losses/contract; control starts only after both treatment gates',
        numerical_source='AST extraction of frozen train_coverage.run; original treatment operations retained',
        source_transform_edits=edits, transformed_kernel_sha256=digest(PREPARATION/'transformed_training_kernel.py'),
        mask_slots=[dict(update=u, ordinal=o, image_id=i, timestep=t) for u,o,i,t in EXPECTED_SLOTS],
        mask_loss='Only designated control slots: ((pred.float()-target.float()).square()*mask4x1x1x1).mean(); original total-element denominator retained',
        scheduled_images=912, scheduled_slots_per_branch=4000,
        effective_exposures={'treatment': 4000, 'control': 3992},
        effective_images={'treatment': 912, 'control': 910},
        scheduled_exposure_counts=dict(counts),
        same_input_policy='Same original epoch indices, cached latents, weak captions, CUDA FP16 noise seeded by committed step and per-ordinal timestep',
        optimizer_policy='Original AdamW/GradScaler/autocast/clip/weight-decay; keep optimizer steps when masking',
        overflow_policy='Original retry policy retained; retries logged, scheduled/effective committed exposure separated from actual calls',
        counts=dict(logical_min_forward_examples=8000, logical_min_backward_calls=2000,
                    gradient_checkpoint_recomputation='Not included in root UNet forward-example counter'),
        estimand='Effect of zeroing this patient E loss contributions along one fixed 1000-update training path',
        limits=['Scheduled inputs still include p images with zero loss contribution; not a conventional retrain on a resized manifest',
                'Effective nonzero exposures differ by eight; not equal effective-data exposure',
                'Downstream Adam state and all later gradients can change after the first mask',
                'One patient and one training RNG path; no population membership or privacy guarantee',
                'Endpoint baseline evaluation is separately frozen; no new attack, R optimization, or increased exposure'],
        existing_files_immutable=True, new_attack=False, endpoint_evaluation=False,
        preparation_dir=str(PREPARATION), frozen_sha256=frozen)
    new_json(path, c)
    return c


def validate(c):
    assert c['schema'] == 'masked-patient-training-contract/v1'
    assert c['patient_id'] == PID and c['branch_order'] == ['treatment', 'control']
    assert c['steps_per_branch'] == 1000 and c['checkpoints'] == [43, 250, 1000]
    assert_hashes(c['frozen_sha256'])
    assert digest(Path(__file__)) == c['frozen_sha256'][str(Path(__file__).resolve())]
    config = original.contract()
    same(config, c['original_training_contract'])
    same(config, load(RUN / 'training_coverage_v2/protocol.json'))
    _, rendered, edits = transformed_kernel()
    assert rendered == (PREPARATION/'transformed_training_kernel.py').read_text(encoding='utf-8')
    assert edits == c['source_transform_edits']
    assert digest(PREPARATION/'transformed_training_kernel.py') == c['transformed_kernel_sha256']
    rows, batches, counts = schedule()
    same(batches, load(PREPARATION/'scheduled_batches.json'))
    same(dict(counts), c['scheduled_exposure_counts'])
    verify_inputs()
    cache = load_cache()
    assert all(cache['latents'][r['image_id']].dtype == torch.float16 for r in rows)
    return config, cache, batches


class Auditor:
    def __init__(self, kind, out, c, batches, treatment=None):
        self.kind, self.out, self.c, self.batches = kind, out, c, batches
        self.treatment = treatment
        self.mask_steps = {u-1: [o] for u,o,_,_ in EXPECTED_SLOTS}
        self.attempts, self.commits, self.gates, self.raw = [], [], [], {}
        self.counter = {'root_forward_calls': 0, 'root_forward_examples': 0}
        self.oldtrace = [json.loads(x) for x in (PRIOR/'training_trace.jsonl').read_text().splitlines()]
        assert len(self.oldtrace) == 1000
        self.started = time.perf_counter()

    def start_branch(self, unet, opt, scaler, rows, config):
        assert unet.training and len(rows) == 912
        assert sum(p.numel() for p in unet.parameters() if p.requires_grad) == 1659904
        assert all(('.lora_' in n) == p.requires_grad for n,p in unet.named_parameters())
        assert all(p.dtype == (torch.float32 if p.requires_grad else torch.float16)
                   for p in unet.parameters())
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in unet.modules())
        assert not any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in unet.modules())
        self.initial = dict(adapter=adapter_state(unet), optimizer=copy.deepcopy(opt.state_dict()),
                            scaler=copy.deepcopy(scaler.state_dict()))
        self.frozen_before = frozen_fingerprint(unet)
        self.frozen_versions = {n: p._version for n,p in unet.named_parameters() if not p.requires_grad}
        if self.kind == 'control':
            same(self.initial, self.treatment.initial, 'initial_state')
            same(self.frozen_before, self.treatment.frozen_before, 'frozen_initial')
        torch.save(self.initial, self.out/'initial_state.pt')
        def hook(_module, inputs, _output):
            self.counter['root_forward_calls'] += 1
            self.counter['root_forward_examples'] += inputs[0].shape[0]
        self.hook = unet.register_forward_hook(hook)

    def capture_inputs(self, step, attempt, rows, z, h, noise, t, noisy, target, scale):
        planned = self.batches[step]
        assert [r['image_id'] for r in rows] == planned['image_ids']
        assert t.detach().cpu().tolist() == planned['timesteps']
        hashes = {n: tensor_hash(v.detach().cpu()) for n,v in
                  [('latents',z),('hidden',h),('noise',noise),('timesteps',t),('noised',noisy),('target',target)]}
        rec = dict(step_zero_based=step, update=step+1, attempt=attempt,
            image_ids=planned['image_ids'], timesteps=planned['timesteps'], noise_seed=planned['noise_seed'],
            mask=planned['control_mask'] if self.kind == 'control' else [1,1,1,1],
            input_sha256=hashes, gradient_scale_before=float(scale))
        if self.kind == 'control':
            same(hashes, self.treatment.commits[step]['input_sha256'], f'input_update_{step+1}')
        self.attempts.append(rec)
        self.current = rec

    def observe_prediction(self, step, attempt, pred, target, loss):
        if step in self.mask_steps:
            key = str(attempt)
            self.raw[key] = dict(update=step+1, attempt=attempt, prediction=pred.detach().cpu(),
                target=target.detach().cpu(), loss=float(loss.detach()), mask=self.current['mask'],
                gradient_scale=self.current['gradient_scale_before'])
            def gradient_hook(grad):
                self.raw[key]['prediction_gradient'] = grad.detach().cpu()
                if self.kind == 'control':
                    assert torch.count_nonzero(grad[self.mask_steps[step]]).item() == 0
                return None
            pred.register_hook(gradient_hook)

    def finish_attempt(self, step, attempt, old_scale, new_scale, norm):
        self.current.update(scale_before_step=float(old_scale), scale_after_step=float(new_scale),
            overflow_retry=bool(new_scale < old_scale),
            gradient_norm_finite=bool(torch.isfinite(norm)),
            gradient_norm=float(norm) if torch.isfinite(norm) else None)

    def commit(self, step, attempt, loss, norm):
        assert self.current['update'] == step and self.current['attempt'] == attempt
        rec = dict(self.current, loss=loss, gradient_norm=float(norm))
        assert rec['overflow_retry'] is False
        if self.kind == 'treatment':
            old = self.oldtrace[step-1]
            for k in ['step', 'attempt', 'loss', 'gradient_norm']:
                actual = step if k == 'step' else rec[k]
                assert actual == old[k], f'Original training trace mismatch at {step}: {k}'
        elif step <= 43:
            for k in ['loss','gradient_norm','attempt','gradient_scale_before','scale_after_step']:
                assert rec[k] == self.treatment.commits[step-1][k], f'Prefix trace mismatch {step}: {k}'
        self.commits.append(rec)

    def save(self, unet, opt, scaler, step, attempt, exposures, losses, path, config):
        assert not path.exists(), 'Never overwrite a branch checkpoint'
        original_save(unet,opt,scaler,step,attempt,exposures,losses,path,config)
        current = torch.load(path,map_location='cpu',weights_only=True)
        if self.kind == 'treatment' and step in [250,1000]:
            prior = torch.load(PRIOR/f'step_{step:04d}.pt',map_location='cpu',weights_only=True)
            same(current,prior,f'original_checkpoint_{step}')
            self.gates.append(dict(gate=f'treatment_vs_original_{step}', exact=True,
                current_sha256=digest(path), reference_sha256=digest(PRIOR/f'step_{step:04d}.pt')))
        elif self.kind == 'control' and step == 43:
            ref = self.treatment.out/'step_0043.pt'
            prior = torch.load(ref,map_location='cpu',weights_only=True)
            same(current,prior,'common_prefix43')
            self.gates.append(dict(gate='control_vs_treatment_prefix43', exact=True,
                current_sha256=digest(path), reference_sha256=digest(ref)))
        new_json(self.out/f'conformance_at_{step:04d}.json', self.gates)

    def finish_branch(self, unet,opt,scaler,step,attempt,exposures,losses):
        self.hook.remove()
        assert step == len(self.commits) == len(losses) == 1000
        assert self.counter == dict(root_forward_calls=attempt,root_forward_examples=4*attempt)
        if self.kind == 'treatment':
            assert [g['gate'] for g in self.gates] == ['treatment_vs_original_250','treatment_vs_original_1000']
        else:
            assert [g['gate'] for g in self.gates] == ['control_vs_treatment_prefix43']
        expected = dict(self.c['scheduled_exposure_counts'])
        if self.kind == 'control':
            for i in ['00014393_002.png','00014393_006.png']:
                del expected[i]
        same(dict(exposures),expected,'effective_exposures')
        assert sum(exposures.values()) == self.c['effective_exposures'][self.kind]
        assert len(exposures) == self.c['effective_images'][self.kind]
        assert all(p.grad is None and p._version == self.frozen_versions[n]
                   for n,p in unet.named_parameters() if not p.requires_grad)
        after = frozen_fingerprint(unet)
        same(self.frozen_before,after,'frozen_base_unchanged')
        assert all('prediction_gradient' in q for q in self.raw.values())
        new_json(self.out/'attempts.json',self.attempts)
        new_json(self.out/'committed_steps.json',self.commits)
        torch.save(self.raw,self.out/'masked_update_raw.pt')
        self.report = dict(status='PASS_BRANCH_EXECUTION_AND_CONFORMANCE',branch=self.kind,step=step,
            attempts=attempt,scheduled_committed_slots=4000,effective_nonzero_exposures=sum(exposures.values()),
            effective_exposed_images=len(exposures),patient_exposures=8 if self.kind=='treatment' else 0,
            root_forward_calls=attempt,root_forward_examples=4*attempt,backward_calls=attempt,
            checkpointing_internal_forward_recomputations='not included in root counters',
            frozen_before=self.frozen_before,frozen_after=after,frozen_parameter_versions_unchanged=True,
            conformance=self.gates, seconds=time.perf_counter()-self.started,
            checkpoint_sha256={str(s):digest(self.out/f'step_{s:04d}.pt') for s in [43,250,1000]},
            initial_state_sha256=digest(self.out/'initial_state.pt'),
            attempts_sha256=digest(self.out/'attempts.json'),committed_steps_sha256=digest(self.out/'committed_steps.json'),
            masked_update_raw_sha256=digest(self.out/'masked_update_raw.pt'))
        new_json(self.out/'execution.json',self.report)


def self_test():
    _, rendered, edits = transformed_kernel()
    compile(rendered,'<reviewed_training_transform>','exec')
    pred = torch.arange(16,dtype=torch.float32).reshape(4,1,2,2).requires_grad_()
    target = torch.zeros_like(pred)
    loss = masked_loss(pred,target,[1])
    loss.backward()
    assert torch.count_nonzero(pred.grad[1]) == 0
    expected = 2*pred.detach()/pred.numel();expected[1]=0
    assert torch.equal(pred.grad,expected)
    assert loss.item() == (pred.detach().square().sum()-pred.detach()[1].square().sum()).item()/16
    a={'tensor':torch.tensor([1.,2.]),'nested':[1,{'k':3.}]}
    same(a,copy.deepcopy(a))
    detected=False
    try: same(a,{'tensor':torch.tensor([1.,3.]),'nested':[1,{'k':3.}]})
    except AssertionError: detected=True
    assert detected
    _, batches, counts = schedule()
    assert sum(sum(b['control_mask']) for b in batches) == 3992
    # Execute the actual checkpoint writer three times so its immutable sidecar
    # lifecycle is checked before a long GPU run (no model inference involved).
    with tempfile.TemporaryDirectory(prefix='masked_training_cpu_') as temp:
        audit = Auditor.__new__(Auditor)
        audit.kind, audit.out, audit.gates = 'writer_self_test', Path(temp), []
        tiny = torch.nn.Linear(1,1)
        opt = torch.optim.AdamW(tiny.parameters())
        scaler = SimpleNamespace(state_dict=lambda: {'scale':1024.})
        for step in [43,250,1000]:
            audit.save(tiny,opt,scaler,step,step,Counter(),[],
                       audit.out/f'step_{step:04d}.pt',{'synthetic':True})
            assert (audit.out/f'conformance_at_{step:04d}.json').is_file()
    return dict(status='PASS_CPU_SOURCE_TRANSFORM_MASK_GRADIENT_AND_SCHEDULE',
        source_transform_edits=edits, CUDA_initialized=torch.cuda.is_initialized())


def execute(args,c,config,cache,batches):
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    ch=digest(args.contract)
    new_json(out/'protocol.json',dict(schema='masked-patient-training-execution/v1',
        started_utc=now(),contract_path=str(args.contract.resolve()),contract_sha256=ch,contract=c,
        code_sha256=digest(Path(__file__)),
        environment={n:version(n) for n in ['torch','diffusers','transformers','peft','numpy']}))
    auditors=[]
    try:
        setup()
        for kind in ['treatment','control']:
            if kind=='control':
                assert len(auditors[0].gates)==2 and all(g['exact'] for g in auditors[0].gates)
                assert (out/'treatment/execution.json').is_file()
            auditor=Auditor(kind,out/kind,c,batches,auditors[0] if auditors else None)
            module,rendered,_=transformed_kernel()
            assert rendered==(PREPARATION/'transformed_training_kernel.py').read_text(encoding='utf-8')
            namespace=dict(original.__dict__)
            namespace.update(branch_kind=kind,branch_output=out/kind,auditor=auditor,
                             masked_loss=masked_loss,save=auditor.save)
            exec(compile(module,str(PREPARATION/'transformed_training_kernel.py'),'exec'),namespace)
            namespace['run']('model_1',cache,config,1000)
            auditors.append(auditor)
        assert_hashes(c['frozen_sha256'])
        assert digest(args.contract)==ch
        gates=[g for a in auditors for g in a.gates]
        new_json(out/'conformance.json',dict(status='PASS_EXACT_TREATMENT_REPLAY_AND_SHARED_PREFIX',gates=gates))
        new_json(out/'execution.json',dict(
            status='PASS_REPLAY_AND_MASKED_CONTRIBUTION_EXECUTION_PENDING_INDEPENDENT_VERIFICATION',
            complete=True,current_step=2,branches=[a.report for a in auditors],
            committed_updates=2000,scheduled_committed_slots=8000,effective_nonzero_exposures=7992,
            forward_examples=sum(a.report['root_forward_examples'] for a in auditors),
            backward_calls=sum(a.report['backward_calls'] for a in auditors),
            seconds=time.perf_counter()-started,ended_utc=now(),original_files_unchanged=True,
            contract_sha256=ch,protocol_sha256=digest(out/'protocol.json'),
            conformance_sha256=digest(out/'conformance.json'),endpoint_evaluation=False,
            population_membership_or_privacy_claim=False))
    except BaseException:
        new_json(out/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_CONTINUATION',
            traceback=traceback.format_exc(),completed_branches=[a.kind for a in auditors],
            seconds=time.perf_counter()-started))
        raise


def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-contract',action='store_true')
    g.add_argument('--dry-run',action='store_true')
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--run',action='store_true')
    p.add_argument('--contract',type=Path,default=CONTRACT)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--expected-code-sha256')
    a=p.parse_args()
    if a.expected_code_sha256: assert digest(Path(__file__))==a.expected_code_sha256
    if a.self_test:
        print(json.dumps(self_test()));return
    if a.prepare_contract:
        c=prepare(a.contract)
        print(json.dumps(dict(contract=str(a.contract),sha256=digest(a.contract),
            bound_files=len(c['frozen_sha256']),CUDA_initialized=torch.cuda.is_initialized())));return
    c=load(a.contract)
    config,cache,batches=validate(c)
    if a.dry_run:
        print(json.dumps(dict(status='PASS_CPU_MASKED_TRAINING_PREFLIGHT',scheduled_slots=4000,
            masked_slots=8,control_effective_exposures=3992,CUDA_initialized=torch.cuda.is_initialized())));return
    execute(a,c,config,cache,batches)


if __name__=='__main__':
    main()
