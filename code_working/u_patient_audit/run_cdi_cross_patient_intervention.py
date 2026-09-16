"""Ninety new literal CDI kernels for two frozen patient-removal interventions.

This producer performs no score fitting or performance analysis.  Training must
already have passed its separate saved-state audits before contract preparation.
"""
import argparse
from collections import Counter
import gc
from importlib.metadata import version
from pathlib import Path
import time
import traceback
import json

import torch

from .common import ROOT, RUN, PROMPT, digest, read_csv, write_json
from .models import setup, load_unet, load_cache, scheduler, adapter_state
from .cdi_adapter import CDIAdapter, FEATURE_NAMES, source_bindings, tensor_digest
from .run_cdi_base_selection import new_json, now, state_fingerprint, hashes_match
from .run_cdi_masked_u82 import source_and_training as original_context, same_random_inputs

BASE = RUN / 'baseline_screen_20260915'
RESEARCH = next(ROOT.parent.glob('*/research_2026-09-10'))
OUTPUT = BASE / 'cdi_cross_patient_intervention_v1'
CONTRACT = BASE / 'cdi_cross_patient_intervention_contract_v1.json'
POLICY = RESEARCH / 'spec_sources/patient_cross_intervention_analysis_contract_20260915.json'
U_SOURCE = RUN / 'baseline_screen_20260914/cdi_u_cohort_v1'
E_SOURCE = RUN / 'baseline_screen_20260914/cdi_e_cohort_v1'
P_SOURCE = BASE / 'cdi_masked_u82_v1'
P_TRAINING = BASE / 'masked_patient_training_v1'
Q_TRAINING = BASE / 'masked_patient_4092_training_v1'
P, Q = '14393', '4092'
BRANCHES = ['q_control', 'p_control']
CHECKPOINTS = {'treatment': RUN/'training_coverage_v2/model_1/step_1000.pt',
              'p_control': P_TRAINING/'control/step_1000.pt',
              'q_control': Q_TRAINING/'control/step_1000.pt'}
ADAPTER_SHA = '70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce'
Q_VERIFICATION_STATUS = 'PASS_MASKED_PATIENT_4092_SAVED_TRAINING_AND_SOURCE_GATES'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def directory_evidence(directory, status):
    execution = read(directory/'execution.json')
    protocol = read(directory/'protocol.json')
    verified = read(directory/'verification.json')
    assert execution['complete'] is True and verified['status'] == status
    for key, name in [('results_sha256', 'results.json'), ('protocol_sha256', 'protocol.json')]:
        assert digest(directory/name) == execution[key] == verified[key]
    assert digest(directory/'execution.json') == verified['execution_sha256']
    assert protocol['source_bindings'] == source_bindings()
    paths = [directory/n for n in ['execution.json', 'protocol.json', 'verification.json', 'results.json']]
    return execution, protocol, verified, paths


def source_record(branch, scenario, directory, row):
    raw_path = directory/row['raw_path']
    row_path = raw_path.with_suffix('.json')
    assert raw_path.is_file() and digest(raw_path) == row['raw_sha256']
    assert read(row_path) == row and row['scenario'] == scenario
    return dict(branch=branch, scenario=scenario, patient_id=row['patient_id'], image_id=row['image_id'],
                directory=str(directory), raw_path=str(raw_path.resolve()), raw_sha256=digest(raw_path),
                row_path=str(row_path.resolve()), row_sha256=digest(row_path))


def old_evidence():
    states, training_paths, training, up, u_rows, _, order = original_context()
    ee, ep, ev, e_paths = directory_evidence(E_SOURCE, 'PASS_SAVED_E_COHORT_ARITHMETIC_AND_INTEGRITY')
    pe, pp, pv, p_paths = directory_evidence(P_SOURCE, 'PASS_MASKED_CDI_U82_SAVED_ARITHMETIC_PROVENANCE_AND_DL_REPLAY')
    assert ee['records'] == 480 and pe['records'] == 82
    assert digest(ROOT/'u_patient_audit/cdi_adapter.py') == ADAPTER_SHA
    metadata = read_csv(RUN/'cohort/evaluation_images.csv')
    e_rows = []
    for pid in [P, Q]:
        selected = sorted((r for r in metadata if r['patient_id'] == pid and r['record_role'] == 'train_candidate'),
                          key=lambda r:r['image_id'])
        assert len(selected) == 2 and all(r['assignment_group'] == 'A' for r in selected)
        e_rows.extend(selected)
    assert len(u_rows) == 82 and Q in order and len(e_rows) == 4
    assert next(r for r in e_rows if r['patient_id'] == Q)['eval_role'] == 'selection'
    u = {r['image_id']:r for r in read(U_SOURCE/'results.json') if r['model'] == 'model_1'}
    e = {r['image_id']:r for r in read(E_SOURCE/'results.json') if r['model'] == 'model_1'}
    p = {r['image_id']:r for r in read(P_SOURCE/'results.json')}
    refs, priors = [], {}
    for rows, scenario, directory, lookup in [(u_rows,'U',U_SOURCE,u), (e_rows,'E',E_SOURCE,e)]:
        for row in rows:
            iid = row['image_id']; prior = lookup[iid]
            assert all(row[k] == prior[k] for k in ['patient_id','image_id','eval_role','assignment_group','record_role'])
            refs.append(source_record('treatment',scenario,directory,prior))
            priors[iid] = prior
    for row in u_rows:
        prior = p[row['image_id']]
        assert prior['checkpoint_sha256'] == digest(CHECKPOINTS['p_control'])
        refs.append(source_record('p_control','U',P_SOURCE,prior))
    assert len(refs) == 168
    paths = training_paths+e_paths+p_paths+[U_SOURCE/n for n in ['execution.json','protocol.json','verification.json','results.json']]
    for s in refs:
        paths += [Path(s['row_path']),Path(s['raw_path'])]
    inherited = {}
    for proto in [up,ep,pp]:
        for path, value in proto['contract']['frozen_sha256'].items():
            assert path not in inherited or inherited[path] == value
            inherited[path] = value
    hashes_match(inherited)
    return dict(states={'treatment':states['treatment'],'p_control':states['control']},
                training=training, paths=paths, inherited=inherited, rows_U=u_rows, rows_E=e_rows,
                patient_order=order, reused_sources=refs, priors=priors,
                source_protocols={'U':up,'E':ep,'p_control_U':pp})


def q_evidence(treatment, e_rows):
    v = read(Q_TRAINING/'verification.json')
    e = read(Q_TRAINING/'execution.json'); p = read(Q_TRAINING/'protocol.json')
    assert v['status'] == Q_VERIFICATION_STATUS
    assert e['complete'] is True and str(e['patient_id']) == Q
    assert e['status'] == 'PASS_MASKED_PATIENT_4092_TRAINING_PENDING_INDEPENDENT_VERIFICATION'
    for key,name in [('execution_sha256','execution.json'),('protocol_sha256','protocol.json')]:
        assert v[key] == digest(Q_TRAINING/name)
    assert v['contract_sha256'] == p['contract_sha256'] == digest(p['contract_path'])
    paths = [Q_TRAINING/n for n in ['verification.json','execution.json','protocol.json']]+[Path(p['contract_path'])]
    checkpoints = v['checkpoint_sha256']
    for step in [36,250,1000]:
        path = Q_TRAINING/'control'/f'step_{step:04d}.pt'
        assert digest(path) == checkpoints[str(step)]
        paths.append(path)
    state = torch.load(CHECKPOINTS['q_control'],map_location='cpu',weights_only=True)
    assert state['step'] == state['attempt'] == 1000 and len(state['losses']) == 1000
    expected = dict(treatment['exposures'])
    q_images = [r['image_id'] for r in e_rows if r['patient_id'] == Q]
    assert len(q_images) == 2 and [expected.pop(i) for i in q_images] == [4,4]
    assert state['exposures'] == expected and sum(expected.values()) == 3992
    assert sum(expected[r['image_id']] for r in e_rows if r['patient_id'] == P) == 8
    assert state['adapter'].keys() == treatment['adapter'].keys()
    for path,value in p['contract']['frozen_sha256'].items():
        assert digest(path) == value
        paths.append(Path(path))
    evidence = dict(status=v['status'],verification_sha256=digest(Q_TRAINING/'verification.json'),
                    execution_sha256=digest(Q_TRAINING/'execution.json'),protocol_sha256=digest(Q_TRAINING/'protocol.json'),
                    contract_sha256=p['contract_sha256'],checkpoint_sha256=checkpoints,
                    patient_id=Q,removed_effective_slots=8,effective_exposures=3992,
                    prefix36_scope='Original per-step trace agreement; no claim of an old saved full checkpoint at36')
    return state,paths,evidence


def plan(context, states):
    manifest = read_csv(RUN/'cohort/model_1_train.csv')
    image_patients = {r['image_id']:r['patient_id'] for r in manifest}
    totals = {branch:Counter() for branch in states}
    for branch,state in states.items():
        for iid,n in state['exposures'].items():
            totals[branch][image_patients[iid]] += n
    rows = []
    for branch in BRANCHES:
        candidates = context['rows_U']+context['rows_E'] if branch=='q_control' else context['rows_E']
        removed = Q if branch=='q_control' else P
        for r in candidates:
            iid,pid = r['image_id'],r['patient_id']; prior=context['priors'][iid]
            scenario = 'U' if r['record_role']=='U_observed' else 'E'
            count = states[branch]['exposures'].get(iid,0); patient_count=totals[branch][pid]
            before = totals['treatment'][pid]
            assert patient_count == (0 if pid==removed else before)
            assert int(before>0) == prior['member'] and before == prior['patient_training_exposures']
            assert count == (0 if scenario=='U' or pid==removed else states['treatment']['exposures'].get(iid,0))
            if scenario=='E':assert before==8 and (count==0 if pid==removed else count==4)
            rows.append(dict(model=branch,branch=branch,removed_patient_id=removed,scenario=scenario,
                patient_id=pid,image_id=iid,eval_role=r['eval_role'],assignment_group=r['assignment_group'],
                record_role=r['record_role'],member=int(patient_count>0),image_member=int(count>0),
                actual_training_exposures=count,patient_training_exposures=patient_count,
                effective_patient_member=int(patient_count>0),effective_image_member=int(count>0),
                effective_image_exposures=count,effective_patient_exposures=patient_count,
                treatment_member=prior['member'],treatment_image_member=prior['image_member'],
                treatment_patient_training_exposures=before,pretrained_membership='unknown',
                membership_label_scope='NIH effective loss contributions only',checkpoint_sha256=digest(CHECKPOINTS[branch])))
    assert len(rows)==90 and Counter(r['branch'] for r in rows)=={'q_control':86,'p_control':4}
    assert Counter(r['scenario'] for r in rows)=={'U':82,'E':8}
    assert len({r['image_id'] for r in rows})==86 and len({r['patient_id'] for r in rows})==41
    return rows


def context_ready():
    context=old_evidence()
    q_state,q_paths,q_gate=q_evidence(context['states']['treatment'],context['rows_E'])
    context['states']['q_control']=q_state
    context['paths'] += q_paths
    context['q_training']=q_gate
    context['measurement_plan']=plan(context,context['states'])
    return context


def prepare(path):
    assert not path.exists() and not OUTPUT.exists() and POLICY.is_file()
    context=context_ready()
    frozen=dict(context['inherited'])
    paths=context['paths']+[Path(__file__),POLICY,ROOT/'u_patient_audit/run_cdi_masked_u82.py',
        ROOT/'u_patient_audit/run_cdi_base_selection.py',ROOT/'u_patient_audit/run_masked_slot_endpoints_v2.py']
    for p in paths:
        key=str(p.resolve());value=digest(p)
        assert key not in frozen or frozen[key]==value
        frozen[key]=value
    c=dict(schema='cdi-cross-patient-intervention-contract/v1',created_utc=now(),current_step=2,
        focal_patient=P,second_patient=Q,branch_order=BRANCHES,measurement_plan=context['measurement_plan'],
        patient_order=context['patient_order'],expected_records=90,expected_unique_images=86,expected_patients=41,
        expected_branch_records={'q_control':86,'p_control':4},expected_scenario_records={'U':82,'E':8},
        reused_sources=context['reused_sources'],expected_reused_records=168,
        reused_counts={'treatment_U':82,'treatment_E':4,'p_control_U':82},
        master_seed=260914,stream='primary',batch_size=1,prediction_type='epsilon',dtype='float32',
        prompt=PROMPT,feature_names=FEATURE_NAMES,code_policy='released_code_literal',
        checkpoint_paths={b:str(p) for b,p in CHECKPOINTS.items()},
        checkpoint_sha256={b:digest(p) for b,p in CHECKPOINTS.items()},
        training_evidence={'p_control':context['training'],'q_control':context['q_training']},
        cache_sha256=digest(RUN/'cache/cache.pt'),cohort_lock_sha256=digest(RUN/'cohort/lock.json'),
        source_bindings=source_bindings(),analysis_policy_path=str(POLICY),analysis_policy_sha256=digest(POLICY),
        output_directory=str(OUTPUT),frozen_sha256=frozen,
        logical_cost='New records only: per-image51+q forward,10+q backward; q actual NO objective evaluations',
        perform_fitting=False,perform_performance_evaluation=False,new_target_training=False,
        limits=['Two fixed patient-removal paths, not a population effect or a new attack.',
                'Counterfactual checkpoints are analyst-only; no limited-access deployment claim.',
                'E is the designated training-candidate role; its effective membership follows each actual control ledger.',
                'New input gradients/noise optimization are released CDI feature kernels, not target training.',
                'Independent endpoint verifier is separately frozen and hash-bound before its own execution.'])
    new_json(path,c);return c


def validate(c):
    assert c['schema']=='cdi-cross-patient-intervention-contract/v1'
    assert c['focal_patient']==P and c['second_patient']==Q and c['branch_order']==BRANCHES
    assert c['expected_records']==90 and c['feature_names']==FEATURE_NAMES
    assert c['master_seed']==260914 and c['stream']=='primary' and c['batch_size']==1
    assert c['dtype']=='float32' and c['prediction_type']=='epsilon' and c['prompt']==PROMPT
    assert c['perform_fitting'] is c['perform_performance_evaluation'] is c['new_target_training'] is False
    assert Path(c['output_directory']).resolve()==OUTPUT.resolve()
    hashes_match(c['frozen_sha256'])
    assert c['frozen_sha256'][str(Path(__file__).resolve())]==digest(Path(__file__))
    context=context_ready()
    assert context['measurement_plan']==c['measurement_plan'] and context['reused_sources']==c['reused_sources']
    assert c['training_evidence']=={'p_control':context['training'],'q_control':context['q_training']}
    for b,p in CHECKPOINTS.items():assert digest(p)==c['checkpoint_sha256'][b]
    cache=load_cache();sched=scheduler();hidden=cache['hidden'][PROMPT].float()
    assert cache['audit_prompt']==PROMPT and sched.config.prediction_type=='epsilon'
    for s in c['reused_sources']:
        raw=torch.load(s['raw_path'],map_location='cpu',weights_only=True)
        assert raw['image_id']==s['image_id'] and raw['stream']=='primary'
        assert torch.equal(raw['latent'],cache['latents'][s['image_id']].float())
        assert torch.equal(raw['alphas'],sched.alphas_cumprod.float())
        assert raw['hidden_sha256']==tensor_digest(hidden)
    return context,cache,sched,hidden


def execute(args,c,context,cache,sched,hidden):
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();contract_hash=digest(args.contract)
    new_json(OUTPUT/'protocol.json',dict(schema='cdi-cross-patient-intervention-execution/v1',started_utc=now(),
        contract_path=str(args.contract.resolve()),contract_sha256=contract_hash,contract=c,
        source_bindings=source_bindings(),measurement_plan=c['measurement_plan'],reused_sources=c['reused_sources'],
        environment={n:version(n) for n in ['torch','diffusers','peft','numpy','scipy','safetensors']}))
    rows=[];reports=[];unet=adapter=None
    source={s['image_id']:s for s in c['reused_sources'] if s['branch']=='treatment'}
    try:
        setup()
        for branch in BRANCHES:
            branch_start=time.perf_counter();(OUTPUT/branch).mkdir()
            unet=load_unet(CHECKPOINTS[branch],training=False).float()
            unet.enable_adapters();unet.eval().requires_grad_(False)
            loaded=adapter_state(unet);expected=context['states'][branch]['adapter']
            assert loaded.keys()==expected.keys() and all(torch.equal(loaded[k],expected[k].float()) for k in loaded)
            adapter=CDIAdapter(unet,sched,hidden,master_seed=260914)
            before=state_fingerprint(unet);versions={n:p._version for n,p in unet.named_parameters()}
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            current=[]
            for metadata in (r for r in c['measurement_plan'] if r['branch']==branch):
                iid=metadata['image_id'];measured=adapter.score(cache['latents'][iid],iid,stream='primary')
                raw=measured.pop('raw');old=torch.load(source[iid]['raw_path'],map_location='cpu',weights_only=True)
                same_random_inputs(raw,old)
                q=raw['no']['optimizer']['objective_evaluations']
                assert measured['feature_names']==FEATURE_NAMES and measured['forward']==51+q and measured['backward']==10+q
                path=OUTPUT/branch/(Path(iid).stem+'.pt');assert not path.exists();torch.save(raw,path)
                for key in set(metadata)&set(measured):assert metadata[key]==measured[key]
                row={**measured,**metadata,'raw_path':path.relative_to(OUTPUT).as_posix(),'raw_sha256':digest(path),
                    'contract_sha256':contract_hash,'cache_sha256':c['cache_sha256'],'cohort_lock_sha256':c['cohort_lock_sha256'],
                    'treatment_source':source[iid],'source_noise_and_input_exact':True,'reused_record':False,'reused_from_kernel':False}
                new_json(path.with_suffix('.json'),row);rows.append(row);current.append(row)
                assert adapter.assert_weights_unchanged();del old,raw,measured
                if len(rows)%10==0 or len(rows)==90:
                    progress=dict(event='cross_patient_cdi_progress',records=len(rows),expected_records=90,branch=branch,
                        seconds=time.perf_counter()-start,forward=sum(r['forward'] for r in rows),backward=sum(r['backward'] for r in rows))
                    write_json(OUTPUT/'progress.json',progress);print(json.dumps(progress),flush=True)
            after=state_fingerprint(unet);assert before==after and adapter.assert_weights_unchanged()
            assert all(p._version==versions[n] and p.grad is None and not p.requires_grad for n,p in unet.named_parameters())
            final=adapter_state(unet);assert all(torch.equal(loaded[k],final[k]) for k in loaded)
            assert adapter.forward==sum(r['forward'] for r in current) and adapter.backward==sum(r['backward'] for r in current)
            assert adapter.forward-adapter.backward==len(current)*41
            report=dict(branch=branch,records=len(current),forward=adapter.forward,backward=adapter.backward,
                seconds=time.perf_counter()-branch_start,initial_state=before,final_state=after,all_parameters_frozen=True,
                no_parameter_gradients=True,parameter_versions_unchanged=True,actual_checkpoint_adapter_exact=True,
                adapter_tensors_unchanged=True,checkpoint_sha256=c['checkpoint_sha256'][branch],
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            new_json(OUTPUT/branch/'execution.json',report);reports.append(report)
            adapter=unet=None;gc.collect();torch.cuda.empty_cache()
        assert [(r['branch'],r['image_id']) for r in rows]==[(r['branch'],r['image_id']) for r in c['measurement_plan']]
        assert len(rows)==90 and Counter(r['branch'] for r in rows)==c['expected_branch_records']
        hashes_match(c['frozen_sha256']);assert digest(args.contract)==contract_hash
        for r in rows:assert digest(OUTPUT/r['raw_path'])==r['raw_sha256']
        new_json(OUTPUT/'results.json',rows)
        new_json(OUTPUT/'execution.json',dict(schema='cdi-cross-patient-intervention-result/v1',
            status='PASS_CDI_CROSS_PATIENT_INTERVENTION_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION',complete=True,
            current_step=2,records=90,unique_images=86,unique_patients=41,reused_records=168,
            forward=sum(r['forward'] for r in rows),backward=sum(r['backward'] for r in rows),vae_forward=0,
            NO_objective_evaluations=sum(r['modules']['noise_optim']['optimizer']['objective_evaluations'] for r in rows),
            runner_seconds=time.perf_counter()-start,score_seconds=sum(r['seconds'] for r in rows),branch_reports=reports,
            source_noise_and_input_exact_all90=True,results_sha256=digest(OUTPUT/'results.json'),
            protocol_sha256=digest(OUTPUT/'protocol.json'),contract_sha256=contract_hash,frozen_inputs_unchanged=True,
            reused_sources_unchanged=True,new_target_training=False,perform_fitting=False,perform_performance_evaluation=False))
    except BaseException:
        new_json(OUTPUT/'failure.json',dict(status='FAILED_PRESERVED_CDI_CROSS_PATIENT_INTERVENTION',
            traceback=traceback.format_exc(),completed_records=len(rows),seconds=time.perf_counter()-start))
        raise
    finally:
        adapter=unet=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--static-preflight',action='store_true')
    modes.add_argument('--prepare-contract',action='store_true')
    modes.add_argument('--dry-run',action='store_true');modes.add_argument('--run',action='store_true')
    parser.add_argument('--contract',type=Path,default=CONTRACT);parser.add_argument('--expected-code-sha256')
    args=parser.parse_args()
    if args.expected_code_sha256:assert digest(Path(__file__))==args.expected_code_sha256
    if args.static_preflight:
        context=old_evidence();assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_STATIC_OLD_SOURCES_PENDING_Q_TRAINING',U_images=len(context['rows_U']),
            E_images=len(context['rows_E']),reused_records=len(context['reused_sources']),expected_new_records=90,
            q_training_checked=False,new_GPU_calls=0)));return
    if args.prepare_contract:
        c=prepare(args.contract);assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_CDI_CROSS_PATIENT_CONTRACT_PREPARED',contract_sha256=digest(args.contract),
            frozen_files=len(c['frozen_sha256']),new_GPU_calls=0)));return
    c=read(args.contract);context=validate(c)
    if args.dry_run:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_CPU_CDI_CROSS_PATIENT_PREFLIGHT',records=90,reused_records=168,new_GPU_calls=0)));return
    execute(args,c,*context)


if __name__=='__main__':
    main()
