"""One new q=4092 masked control; reuse the already verified original treatment.

The frozen numerical training AST changes only its masked patient and extra
checkpoint index. No checkpoint resume, treatment refit, or endpoint evaluation.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace

import torch

from . import run_masked_patient_training as previous
from .common import ROOT, RUN, digest, read_csv, seed, stable, verify_inputs
from .models import setup, load_cache
from .run_mofit_medical_kernel import load, new_json, now, assert_hashes

BASE = RUN / 'baseline_screen_20260915'
CONTRACT = BASE / 'masked_patient_4092_training_contract_v1.json'
PREPARATION = BASE / 'masked_patient_4092_training_preparation_v1'
OUTPUT = BASE / 'masked_patient_4092_training_v1'
OLD = BASE / 'masked_patient_training_v1'
TREATMENT = OLD / 'treatment'
POLICY = next(ROOT.parent.glob('*/research_2026-09-10')) / 'spec_sources/patient_cross_intervention_analysis_contract_20260915.json'
POLICY_SHA = 'b0855a8bdec2e1a5f738d35ee12930bf0dfd7a21c603cf55f343ba99ccf5bc70'
PREVIOUS_SHA = '717960d8e6a4daedf6819dc100e2bcc7ce43577b1cddfedfd82d20951fdd43e8'
PID = '4092'
E_IMAGES = ['00004092_001.png', '00004092_002.png']
P_IMAGES = ['00014393_002.png', '00014393_006.png']
EXPECTED_SLOTS = [(37,1,E_IMAGES[0],478), (61,0,E_IMAGES[1],947),
    (271,2,E_IMAGES[1],990), (444,1,E_IMAGES[0],327),
    (553,3,E_IMAGES[0],711), (632,2,E_IMAGES[1],451),
    (837,0,E_IMAGES[1],526), (845,1,E_IMAGES[0],965)]
same = previous.same


def schedule():
    rows = read_csv(RUN / 'cohort/model_1_train.csv')
    rows.sort(key=lambda r: (r['eval_role'] != 'background',
                            stable('training-order', r['patient_id']), r['image_id']))
    assert len(rows) == 912
    batches, slots, counts = [], [], Counter()
    for step in range(1000):
        batch = [rows[i] for i in previous.original.epoch_indices(len(rows), step)]
        images = [r['image_id'] for r in batch]
        ts = [seed('coverage-timestep', step, i) % 1000 for i in range(4)]
        batches.append(dict(update=step+1, image_ids=images, timesteps=ts,
            noise_seed=seed('coverage-noise', step),
            control_mask=[int(r['patient_id'] != PID) for r in batch]))
        counts.update(images)
        slots.extend((step+1,j,r['image_id'],ts[j]) for j,r in enumerate(batch) if r['patient_id'] == PID)
    assert slots == EXPECTED_SLOTS
    cp = torch.load(previous.PRIOR/'step_1000.pt', map_location='cpu', weights_only=True)
    same(dict(counts), cp['exposures'])
    assert sum(counts.values()) == 4000 and all(counts[i] == 4 for i in E_IMAGES+P_IMAGES)
    qrows = [r for r in rows if r['patient_id'] == PID]
    assert len(qrows) == 2 and all(r['eval_role'] == 'selection' for r in qrows)
    eligible = sorted({r['patient_id'] for r in rows if r['eval_role'] == 'selection'
        and sum(counts[x['image_id']] for x in rows if x['patient_id'] == r['patient_id']) == 8}, key=int)
    assert eligible[0] == PID, 'Metadata-only selection rule changed'
    return rows, batches, counts


def transformed_kernel():
    assert digest(Path(previous.__file__)) == PREVIOUS_SHA
    module, prior_rendered, prior_edits = previous.transformed_kernel()
    changes = Counter()
    class Retarget(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == '14393':
                changes['effective_exposure_patient_14393_to_4092'] += 1
                return ast.copy_location(ast.Constant(PID), node)
            if type(node.value) is int and node.value == 43:
                changes['extra_checkpoint43_to36'] += 1
                return ast.copy_location(ast.Constant(36), node)
            return node
    module = Retarget().visit(module)
    assert dict(changes) == {'effective_exposure_patient_14393_to_4092':1,'extra_checkpoint43_to36':1}
    ast.fix_missing_locations(module)
    rendered = ast.unparse(module)+'\n'
    # Reverse these two literal replacements and require the whole AST exact.
    class Restore(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value == PID: return ast.copy_location(ast.Constant('14393'),node)
            if type(node.value) is int and node.value == 36: return ast.copy_location(ast.Constant(43),node)
            return node
    restored = Restore().visit(copy.deepcopy(module))
    assert ast.dump(restored,include_attributes=False) == ast.dump(ast.parse(prior_rendered),include_attributes=False)
    return module, rendered, dict(original_transform=prior_edits, retarget=dict(changes),
        reverse_retarget_full_AST_exact=True,
        previous_rendered_sha256=hashlib.sha256(prior_rendered.encode()).hexdigest())


def prior_evidence():
    v = load(OLD/'verification.json')
    assert v['status'] == 'PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC' and v['complete'] is True
    protocol = load(OLD/'protocol.json')
    for key,path in [('execution_sha256',OLD/'execution.json'),('protocol_sha256',OLD/'protocol.json'),
                     ('contract_sha256',Path(protocol['contract_path']))]:
        assert digest(path) == v[key]
    execution = load(OLD/'execution.json')
    assert execution['complete'] is True
    report = load(TREATMENT/'execution.json')
    same(report, execution['branches'][0])
    assert report['branch'] == 'treatment' and report['step'] == 1000
    for name,key in [('initial_state.pt','initial_state_sha256'),('committed_steps.json','committed_steps_sha256'),
                     ('attempts.json','attempts_sha256')]:
        assert digest(TREATMENT/name) == report[key]
    for step in [250,1000]:
        path = TREATMENT/f'step_{step:04d}.pt'
        assert digest(path) == v['checkpoint_sha256']['treatment'][str(step)] == report['checkpoint_sha256'][str(step)]
        same(torch.load(path,map_location='cpu',weights_only=True),
             torch.load(previous.PRIOR/f'step_{step:04d}.pt',map_location='cpu',weights_only=True),f'reused_treatment_{step}')
    commits = load(TREATMENT/'committed_steps.json')
    assert len(commits) == 1000 and all(r['update'] == i+1 for i,r in enumerate(commits))
    proxy = SimpleNamespace(initial=torch.load(TREATMENT/'initial_state.pt',map_location='cpu',weights_only=True),
        frozen_before=report['frozen_before'], commits=commits, out=TREATMENT)
    return v, report, proxy


def check_policy():
    assert digest(POLICY) == POLICY_SHA
    p = load(POLICY)
    assert p['focal_patients'] == {'p':'14393','q':PID}
    t = p['training']
    assert t['source_model'] == 'model_1' and t['updates'] == 1000
    assert t['q_first_contribution_update'] == 37 and t['expected_q_masked_updates'] == [x[0] for x in EXPECTED_SLOTS]
    assert t['scheduled_slots'] == 4000 and t['effective_slots'] == 3992
    assert t['p_exposures_in_q_control'] == 8 and t['q_exposures_in_q_control'] == 0
    assert t['same_schedule_noise_optimizer_and_original_mean_denominator'] is True
    assert t['original_treatment_reused_after_exact_prior_verification'] is True
    return p


def prepare(path):
    assert not path.exists() and not PREPARATION.exists()
    policy = check_policy()
    config = previous.original.contract()
    same(config,load(RUN/'training_coverage_v2/protocol.json'))
    _,batches,counts = schedule()
    _,report,_ = prior_evidence()
    _,rendered,edits = transformed_kernel()
    PREPARATION.mkdir(parents=True,exist_ok=False)
    (PREPARATION/'transformed_training_kernel.py').write_text(rendered,encoding='utf8')
    new_json(PREPARATION/'scheduled_batches.json',batches)
    frozen = dict(load(BASE/'masked_patient_training_contract_v1.json')['frozen_sha256'])
    frozen.update(policy['frozen_sha256'])
    assert_hashes(frozen)
    for path0 in [Path(__file__),POLICY,Path(previous.__file__),
            BASE/'masked_patient_training_contract_v1.json',OLD/'verification.json',OLD/'execution.json',
            OLD/'protocol.json',TREATMENT/'execution.json',TREATMENT/'initial_state.pt',
            TREATMENT/'committed_steps.json',TREATMENT/'attempts.json',
            TREATMENT/'step_0250.pt',TREATMENT/'step_1000.pt',
            PREPARATION/'transformed_training_kernel.py',PREPARATION/'scheduled_batches.json']:
        frozen[str(path0.resolve())] = digest(path0)
    c = dict(schema='masked-patient-4092-training-contract/v1',created_utc=now(),current_step=2,
        source_model='model_1',patient_id=PID,branch_order=['control'],steps_per_branch=1000,
        checkpoints=[36,250,1000],original_training_contract=config,
        analysis_policy_path=str(POLICY),analysis_policy_sha256=POLICY_SHA,
        treatment_directory=str(TREATMENT),treatment_verification_sha256=digest(OLD/'verification.json'),
        original_treatment_reused=True,new_treatment_training=False,resume=False,
        initial_state_gate='Exact saved adapter, optimizer, scaler and frozen-base fingerprint against verified treatment initialization',
        control_prefix_gate=36,prefix_gate_scope='First 36 committed traces only; no original step36 full checkpoint exists',
        prefix_full_checkpoint_exact_claim=False,input_hash_gate_updates=1000,
        source_transform_edits=edits,transformed_kernel_sha256=digest(PREPARATION/'transformed_training_kernel.py'),
        mask_slots=[dict(update=u,ordinal=o,image_id=i,timestep=t) for u,o,i,t in EXPECTED_SLOTS],
        mask_loss='Frozen masked_loss: zero q examples, retain original full tensor mean denominator; keep optimizer steps',
        scheduled_images=912,scheduled_slots_per_branch=4000,scheduled_exposure_counts=dict(counts),
        effective_exposures={'control':3992},effective_images={'control':910},
        effective_patient_exposures={PID:0,'14393':8},
        same_input_policy='Original epoch order, cached latents and weak hidden, CUDA FP16 step-seeded noise, original per-ordinal timesteps',
        overflow_policy='Original GradScaler retry preserved; raw attempt counts may exceed committed updates',
        minimum_counts={'forward_examples':4000,'backward_calls':1000},
        frozen_base_reference=report['frozen_before'],
        preparation_dir=str(PREPARATION),output_directory=str(OUTPUT),
        verification_binding='Execution policy and producer frozen before GPU; independent verifier code bound later in its own protocol',
        limits=['Scheduled q inputs remain with zero contribution; not resized-manifest retraining',
                'Effective contributions 3992 versus original4000; denominator and update count retained',
                'No resume from43: q first contributes at37',
                'First36 trace agreement is not full-state checkpoint agreement',
                'One fixed second intervention; no population privacy or attack efficacy claim'],
        endpoint_evaluation=False,new_attack=False,frozen_sha256=frozen)
    new_json(path,c)
    return c


def validate(c):
    assert c['schema'] == 'masked-patient-4092-training-contract/v1'
    assert c['patient_id'] == PID and c['branch_order'] == ['control']
    assert c['checkpoints'] == [36,250,1000] and c['steps_per_branch'] == 1000 and c['resume'] is False
    assert_hashes(c['frozen_sha256'])
    assert digest(Path(__file__)) == c['frozen_sha256'][str(Path(__file__).resolve())]
    check_policy()
    config = previous.original.contract()
    same(config,c['original_training_contract'])
    same(config,load(RUN/'training_coverage_v2/protocol.json'))
    _,rendered,edits = transformed_kernel()
    assert rendered == (PREPARATION/'transformed_training_kernel.py').read_text(encoding='utf8')
    assert edits == c['source_transform_edits']
    assert digest(PREPARATION/'transformed_training_kernel.py') == c['transformed_kernel_sha256']
    rows,batches,counts = schedule()
    same(batches,load(PREPARATION/'scheduled_batches.json'))
    same(dict(counts),c['scheduled_exposure_counts'])
    _,_,treatment = prior_evidence()
    for i,b in enumerate(batches):
        prior = treatment.commits[i]
        for key in ['update','image_ids','timesteps','noise_seed']: same(b[key],prior[key])
    verify_inputs()
    cache = load_cache()
    assert all(cache['latents'][r['image_id']].dtype == torch.float16 for r in rows)
    return config,cache,batches,treatment


class Auditor(previous.Auditor):
    def __init__(self,out,c,batches,treatment):
        super().__init__('control',out,c,batches,treatment)
        self.mask_steps = {u-1:[o] for u,o,_,_ in EXPECTED_SLOTS}

    def start_branch(self,unet,opt,scaler,rows,config):
        super().start_branch(unet,opt,scaler,rows,config)
        self.gates.append(dict(gate='initial_adapter_optimizer_scaler_and_base',exact=True,
            reference_initial_sha256=digest(TREATMENT/'initial_state.pt'),
            base=self.frozen_before))

    def commit(self,step,attempt,loss,norm):
        assert self.current['update'] == step and self.current['attempt'] == attempt
        rec = dict(self.current,loss=loss,gradient_norm=float(norm))
        assert rec['overflow_retry'] is False
        if step <= 36:
            for key in ['loss','gradient_norm','attempt','gradient_scale_before','scale_after_step']:
                assert rec[key] == self.treatment.commits[step-1][key], f'Prefix trace {step}: {key}'
        self.commits.append(rec)
        if step == 36:
            self.gates.append(dict(gate='first36_committed_trace_exact',exact=True,
                updates=36,fields=['loss','gradient_norm','attempt','gradient_scale_before','scale_after_step'],
                full_checkpoint_comparison=False,original_step36_checkpoint_available=False))

    def save(self,unet,opt,scaler,step,attempt,exposures,losses,path,config):
        assert not path.exists()
        assert step in [36,250,1000]
        previous.original_save(unet,opt,scaler,step,attempt,exposures,losses,path,config)
        new_json(self.out/f'conformance_at_{step:04d}.json',dict(gates=self.gates,
            saved_step=step,saved_checkpoint_sha256=digest(path),
            full_original_checkpoint_comparison=False))

    def finish_branch(self,unet,opt,scaler,step,attempt,exposures,losses):
        self.hook.remove()
        assert step == len(self.commits) == len(losses) == 1000
        assert len(self.attempts) == attempt
        assert self.counter == dict(root_forward_calls=attempt,root_forward_examples=4*attempt)
        assert [g['gate'] for g in self.gates] == ['initial_adapter_optimizer_scaler_and_base','first36_committed_trace_exact']
        expected = dict(self.c['scheduled_exposure_counts'])
        for image in E_IMAGES: del expected[image]
        same(dict(exposures),expected,'effective_exposures')
        assert sum(exposures.values()) == 3992 and len(exposures) == 910
        assert all(exposures.get(i,0) == 0 for i in E_IMAGES)
        assert sum(exposures.get(i,0) for i in P_IMAGES) == 8
        assert all(p.grad is None and p._version == self.frozen_versions[n]
                   for n,p in unet.named_parameters() if not p.requires_grad)
        after = previous.frozen_fingerprint(unet)
        same(self.frozen_before,after,'frozen_base_unchanged')
        assert all('prediction_gradient' in r for r in self.raw.values())
        assert {r['update'] for r in self.raw.values()} == {x[0] for x in EXPECTED_SLOTS}
        for r in self.raw.values():
            assert torch.count_nonzero(r['prediction_gradient'][self.mask_steps[r['update']-1]]).item() == 0
        new_json(self.out/'attempts.json',self.attempts)
        new_json(self.out/'committed_steps.json',self.commits)
        torch.save(self.raw,self.out/'masked_update_raw.pt')
        self.report = dict(status='PASS_BRANCH_EXECUTION_AND_CONFORMANCE',branch='control',patient_id=PID,
            step=step,attempts=attempt,scheduled_committed_slots=4000,effective_nonzero_exposures=3992,
            effective_exposed_images=910,patient_exposures=0,other_focal_patient_exposures={'14393':8},
            root_forward_calls=attempt,root_forward_examples=4*attempt,backward_calls=attempt,
            checkpointing_internal_forward_recomputations='not included in root counters',
            frozen_before=self.frozen_before,frozen_after=after,frozen_parameter_versions_unchanged=True,
            frozen_parameter_gradients_none=True,initial_state_exact=True,input_hash_pairs_exact=1000,
            prefix_trace_exact_updates=36,prefix_full_checkpoint_exact_claim=False,
            conformance=self.gates,seconds=time.perf_counter()-self.started,
            checkpoint_sha256={str(s):digest(self.out/f'step_{s:04d}.pt') for s in [36,250,1000]},
            initial_state_sha256=digest(self.out/'initial_state.pt'),
            attempts_sha256=digest(self.out/'attempts.json'),committed_steps_sha256=digest(self.out/'committed_steps.json'),
            masked_update_raw_sha256=digest(self.out/'masked_update_raw.pt'))
        new_json(self.out/'execution.json',self.report)


def self_test():
    module,rendered,edits = transformed_kernel()
    compile(module,'<q_control_kernel>','exec')
    pred = torch.arange(16,dtype=torch.float32).reshape(4,1,2,2).requires_grad_()
    loss = previous.masked_loss(pred,torch.zeros_like(pred),[1])
    loss.backward()
    expected = 2*pred.detach()/pred.numel(); expected[1]=0
    assert torch.equal(pred.grad,expected)
    assert loss.item() == (pred.detach().square().sum()-pred.detach()[1].square().sum()).item()/16
    _,batches,_ = schedule()
    assert sum(sum(b['control_mask']) for b in batches) == 3992
    with tempfile.TemporaryDirectory(prefix='q_training_cpu_') as temp:
        auditor = Auditor.__new__(Auditor)
        auditor.out,auditor.gates = Path(temp),[]
        tiny = torch.nn.Linear(1,1)
        opt = torch.optim.AdamW(tiny.parameters())
        scaler = SimpleNamespace(state_dict=lambda:{'scale':1024.})
        for step in [36,250,1000]:
            auditor.save(tiny,opt,scaler,step,step,Counter(),[],auditor.out/f'step_{step:04d}.pt',{'synthetic':True})
    return dict(status='PASS_CPU_Q_SOURCE_MASK_SCHEDULE_AND_CHECKPOINT_WRITER',
        AST_edits=edits,CUDA_initialized=torch.cuda.is_initialized())


def execute(args,c,config,cache,batches,treatment):
    out = args.output_dir
    assert out.resolve() == Path(c['output_directory']).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter(); contract_sha = digest(args.contract)
    new_json(out/'protocol.json',dict(schema='masked-patient-4092-training-execution/v1',
        started_utc=now(),contract_path=str(args.contract.resolve()),contract_sha256=contract_sha,contract=c,
        code_sha256=digest(Path(__file__)),analysis_policy_sha256=POLICY_SHA,
        independent_verifier_code_binding='Later own verification protocol; not claimed frozen before execution',
        environment={n:version(n) for n in ['torch','diffusers','transformers','peft','numpy']}))
    try:
        setup()
        auditor = Auditor(out/'control',c,batches,treatment)
        module,rendered,_ = transformed_kernel()
        assert rendered == (PREPARATION/'transformed_training_kernel.py').read_text(encoding='utf8')
        namespace = dict(previous.original.__dict__)
        namespace.update(branch_kind='control',branch_output=out/'control',auditor=auditor,
            masked_loss=previous.masked_loss,save=auditor.save)
        exec(compile(module,str(PREPARATION/'transformed_training_kernel.py'),'exec'),namespace)
        namespace['run']('model_1',cache,config,1000)
        assert_hashes(c['frozen_sha256'])
        assert digest(args.contract) == contract_sha
        new_json(out/'conformance.json',dict(status='PASS_REUSED_TREATMENT_INITIAL_STATE_AND_Q_PREFIX_TRACE',
            gates=auditor.gates,reused_treatment_verification_sha256=digest(OLD/'verification.json'),
            prefix_full_checkpoint_exact_claim=False))
        new_json(out/'execution.json',dict(status='PASS_MASKED_PATIENT_4092_TRAINING_PENDING_INDEPENDENT_VERIFICATION',
            complete=True,current_step=2,patient_id=PID,branches=[auditor.report],
            committed_updates=1000,scheduled_committed_slots=4000,effective_nonzero_exposures=3992,
            effective_patient_exposures={PID:0,'14393':8},new_treatment_training=False,
            forward_examples=auditor.report['root_forward_examples'],backward_calls=auditor.report['backward_calls'],
            checkpoint_sha256=auditor.report['checkpoint_sha256'],seconds=time.perf_counter()-started,
            ended_utc=now(),original_files_unchanged=True,contract_sha256=contract_sha,
            protocol_sha256=digest(out/'protocol.json'),conformance_sha256=digest(out/'conformance.json'),
            reused_treatment_directory=str(TREATMENT),reused_treatment_verification_sha256=digest(OLD/'verification.json'),
            endpoint_evaluation=False,population_membership_or_privacy_claim=False))
    except BaseException:
        new_json(out/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_CONTINUATION',
            traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
        raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ['prepare-contract','dry-run','self-test','run']: group.add_argument('--'+name,action='store_true')
    parser.add_argument('--contract',type=Path,default=CONTRACT)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--expected-code-sha256')
    args = parser.parse_args()
    if args.expected_code_sha256: assert digest(Path(__file__)) == args.expected_code_sha256
    if args.self_test:
        print(json.dumps(self_test()));return
    if args.prepare_contract:
        c = prepare(args.contract)
        print(json.dumps(dict(status='PREPARED_CPU_ONLY',contract=str(args.contract),sha256=digest(args.contract),
            bound_files=len(c['frozen_sha256']),CUDA_initialized=torch.cuda.is_initialized())));return
    c = load(args.contract)
    config,cache,batches,treatment = validate(c)
    if args.dry_run:
        print(json.dumps(dict(status='PASS_CPU_MASKED_PATIENT_4092_PREFLIGHT',scheduled_slots=4000,
            masked_slots=8,effective_exposures=3992,q_exposures=0,p_exposures=8,prefix_trace_updates=36,
            prefix_checkpoint_comparison=False,CUDA_initialized=torch.cuda.is_initialized())));return
    execute(args,c,config,cache,batches,treatment)


if __name__ == '__main__':
    main()
