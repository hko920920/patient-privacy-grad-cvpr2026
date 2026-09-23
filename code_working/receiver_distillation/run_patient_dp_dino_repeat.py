"""Exactly one independent DINO DP target and one fresh matched seed101 bank."""
import argparse,copy,json,shutil,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from .patient_dp_dino_io import release_from_contract
from .patient_dp_dino_runtime import code_bindings,validate_job
from .dino_control_runtime import load_dino
from .run_feasibility import worker
from .evaluate_patient_dp_dino_repeat import dependencies,PREVIOUS

ROOT=CODE/'_reports/receiver_dino_dp8_noise_repeat_20260924_v1'
START='2026-09-23T16:31:15+00:00'
DEADLINE='2026-09-23T17:46:15+00:00'
BANK='DINO_dp8_noise02_101'
HUMAN=RESEARCH/'RECEIVER_DINO_DP_NOISE_REPEAT_CONTRACT_20260924.md'


def before_deadline():
    require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole75-minute cap reached')


def state_update(root,status,**fields):
    path=RESEARCH/'research_state.json';s=read(path)
    s.setdefault('receiver_DINO_patient_dp_noise_repeat',{}).update(status=status,updated=now(),
        output_directory=str(root),absolute_deadline_utc=DEADLINE,**fields)
    s.setdefault('active_execution',{}).update(status='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_DINO_patient_dp_noise_repeat',
        current_scope='One DINO independent noise02; fixed seed101; compare both DINO and both A2 DP outcomes',output_directory=str(root))
    s.update(current_step=2,stage2_status='in_progress',updated_utc=now(),step2_active_task='DINO patient-DP noise repetition: '+status)
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def ledger(root,status):
    value=copy.deepcopy(read(PREVIOUS/'privacy_release_ledger.json'))
    require(len(value['entries'])==3,'Expected three previous summaries')
    value.update(created=now(),scope='Four frozen protected summaries and their postprocessing over the same Q patients')
    value['entries'].append({'release_id':'DINO_DP8_release02','status':status,'epsilon':8.,'delta':1e-5,
        'protected_targets':str(root/'one_release/protected_targets'),'protected_png':str(root/'protected_png')})
    old=value.pop('if_all_three_outputs_disclosed')
    value['if_all_four_outputs_disclosed']={**old,'epsilon_upper_bound':32.,'delta_upper_bound':4e-5}
    atomic_json(root/'privacy_release_ledger.json',value,replace=(root/'privacy_release_ledger.json').exists())


def initialize(root):
    before_deadline();tick=time.monotonic();root.mkdir(exist_ok=False);(root/'records_before').mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:shutil.copy2(p,root/'records_before'/p.name)
    old=read(PREVIOUS/'campaign_contract.json');binding=read(PREVIOUS/'execution_bindings.json')
    require(read(PREVIOUS/'campaign_result.json')['status']=='COMPLETE_DINO_PATIENT_DP_CONTROL','Prior incomplete')
    history={**old['historical_dependencies'],**old['input_code_bindings'],**binding['implementation']}
    require(all(sha(p)==h for p,h in history.items()),'Frozen previous input/code changed')
    for file in (HUMAN,PREVIOUS/'campaign_contract.json',PREVIOUS/'one_release_contract.json',
                 PREVIOUS/'campaign_result.json',PREVIOUS/'DP8_result.json',PREVIOUS/'public_calibration.json',
                 PREVIOUS/'job.json',PREVIOUS/'bank_seal.json',PREVIOUS/'evaluation/result.json',
                 PREVIOUS/'privacy_release_ledger.json',PREVIOUS/'protected_png/release_manifest.json'):
        history[str(file)]=sha(file)
    c=copy.deepcopy(old)
    c.update(schema='receiver.dino-patient-dp-independent-repeat/v1',created=now(),
        authorization='Latest user explicitly authorizes DINO independent noise02, one fresh seed101 bank and comparison retaining all prior outcomes',
        scope_started_utc=START,absolute_deadline_utc=DEADLINE,maximum_seconds_per_bank=3600,
        maximum_package_seconds=4500,estimated_total_minutes=[40,50],synthesis_reference_minutes=35.11,
        bank_ids={'DP8':BANK},historical_dependencies=history,
        primary='Both DINO and both A2 DP AUROC/AP; all four A2-minus-DINO contrasts and DINO noise02-minus-noise01',
        descriptive_rule='Report every fixed-bank result and patient CI, plus arithmetic per-method two-noise point means; no best-bank choice or equivalence/significance claim about noise distributions',
        inference_limit='Two noises per method, fixed synth seed101, adaptive V/DenseNet; patient CI conditional on fixed banks; noise indices are not coupled/matched pairs',
        original_DINO_contract=str(PREVIOUS/'campaign_contract.json'),original_DINO_contract_sha256=sha(PREVIOUS/'campaign_contract.json'),
        new_release_id='DINO_DP8_release02',same_Q_release_count_after_completion=4,
        joint_basic_composition={'epsilon':32.,'delta':4e-5,'optimal_accounting_claimed':False},
        free_bytes_before=shutil.disk_usage(CODE).free)
    save(root/'campaign_contract.json',c)
    state_update(root,'PREPARING_TARGET',new_private_releases=0,development_evaluation_opened=False)
    plan=copy.deepcopy(read(PREVIOUS/'one_release_contract.json'))
    plan.update(output_directory=str(root/'one_release'),campaign_authority=str(root/'campaign_contract.json'),
        campaign_authority_sha256=sha(root/'campaign_contract.json'),release_id='DINO_DP8_release02',independent_OS_randomness=True)
    save(root/'one_release_contract.json',plan);ledger(root,'RESERVED_NOT_REDRAWABLE')
    protected=release_from_contract(root/'one_release_contract.json');ledger(root,'TARGET_COMPLETE')
    target={'variant':'DP8',**protected,**{k+'_sha256':sha(v) for k,v in protected.items()}}
    save(root/'DP8_target_binding.json',target)
    job=copy.deepcopy(read(PREVIOUS/'job.json'))
    job.update(id=BANK,**protected,**{k+'_sha256':sha(v) for k,v in protected.items()},
        campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
        target_binding=str(root/'DP8_target_binding.json'),target_binding_sha256=sha(root/'DP8_target_binding.json'),
        absolute_deadline_utc=DEADLINE,code_bindings=code_bindings())
    save(root/'job.json',job);validate_job(job)
    previous_job=read(PREVIOUS/'job.json');new_h=read(job['second_handoff']);old_h=read(previous_job['second_handoff'])
    require(read(new_h['contract_file'])==read(old_h['contract_file']),'Mechanism changed')
    require(sha(new_h['rule_file'])==sha(old_h['rule_file']),'Condition/head/projection changed')
    with np.load(new_h['target_file'],allow_pickle=False) as f:new=f['pooled_gradient'].copy()
    with np.load(old_h['target_file'],allow_pickle=False) as f:prior=f['pooled_gradient'].copy()
    objective,_=load_dino(job['second_handoff'],(0,)*64+(1,)*64)
    require(set(objective.targets)=={'dinov2_vitb14'} and np.isfinite(new).all() and np.all(np.linalg.norm(new,axis=-1)>1e-12),'Invalid target')
    verify={'status':'PASS_REPEAT_TARGET_JOB_BINDING','same_protection_metadata':True,'same_condition_head_projection':True,
        'same_training_runtime':job['code_bindings']==previous_job['code_bindings'],
        'target_equal_to_previous':bool(np.array_equal(new,prior)),
        'target_difference_is_not_selection_or_redraw_criterion':True,'new_optimizer_updates':0,'new_GPU_checks':0}
    require(verify['same_training_runtime'],'Training code changed')
    save(root/'changed_path_verification.json',verify)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    result={'created':now(),'job':str(root/'job.json'),'job_sha256':sha(root/'job.json'),
        'implementation':sources,'runtime':code_bindings(),'evaluation_dependencies':dependencies(),
        'campaign_contract_sha256':sha(root/'campaign_contract.json')}
    save(root/'execution_bindings.json',result);(root/'implementation_snapshot').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,root/'implementation_snapshot'/p.name)
    save(root/'preparation_result.json',{'status':'COMPLETE_ONE_FRESH_DINO_TARGET','seconds':time.monotonic()-tick,
        'new_private_releases':1,'new_model_forwards':0,'new_Q_V_pixels':0,'old_calibration_reused':True,
        'no_noise_seed_or_realized_noise_saved':True})
    print(json.dumps({'phase':'PREPARED','seconds':time.monotonic()-tick,'new_private_releases':1}),flush=True)
    return result


def protected_package(root):
    out=root/'protected_png';out.mkdir(exist_ok=False);artifact=root/'banks'/BANK/'artifact'
    shutil.copytree(artifact/'images',out/'images')
    for name in ('images.csv','pairs.csv'):shutil.copy2(artifact/name,out/name)
    rule=copy.deepcopy(read(PREVIOUS/'protected_png/learning_rule.json'))
    rule.update(release_id='DINO_DP8_release02',DINO_family_release_count=2)
    rule.pop('if_jointly_disclosed_with_both_A2_DP_summaries_basic_bound')
    rule['if_all_four_A2_DINO_summaries_disclosed_basic_bound']={'epsilon':32.,'delta':4e-5}
    save(out/'learning_rule.json',rule)
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'release_manifest.json',{'schema':'receiver.protected-png-files/v1','files_sha256':files})
    require(len(list((out/'images').glob('*.png')))==128,'Wrong PNG count')


def run(root,resume=False):
    tick=time.monotonic()
    try:
        before_deadline();binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen code changed')
        require(sha(binding['job'])==binding['job_sha256'],'Job changed')
        result_path=root/'DP8_result.json';bank=root/'banks'/BANK
        if not result_path.exists():
            cmd=['-m','receiver_distillation.patient_dp_dino_runtime','--job',binding['job'],
                '--output',str(bank),'--result',str(result_path)]
            if bank.exists():cmd.append('--resume')
            state_update(root,'RUNNING_BANK',new_private_releases=1)
            worker(root,'DINO_DP8_noise02_main',cmd)
        result=read(result_path);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json');verify_artifact(bank/'artifact',spec,done['signature'])
        require(read(bank/'initialization_parity.json')['status']=='PASS_EXACT_A2_INITIALIZATION','Initial state changed')
        require(read(bank/'artifact/learning_contract.json')['DP_applied'] is True,'Missing privacy metadata')
        if not(root/'bank_seal.json').exists():save(root/'bank_seal.json',{'created':now(),
            'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        if not(root/'protected_png').exists():protected_package(root)
        before_deadline();state_update(root,'EVALUATING',development_evaluation_opened=True)
        if not(root/'evaluation/result.json').exists():
            worker(root,'DINO_DP_noise_repeat_comparison',
                ['-m','receiver_distillation.evaluate_patient_dp_dino_repeat','--root',str(root)])
        c=read(root/'campaign_contract.json')
        require(all(sha(p)==h for p,h in {**binding['implementation'],**c['historical_dependencies'],**c['input_code_bindings']}.items()),'Prior evidence changed')
        before_deadline();e=read(root/'evaluation/result.json');ledger(root,'COMPLETE')
        final={'status':'COMPLETE_DINO_DP_NOISE_REPEAT','finished':now(),'controller_seconds':time.monotonic()-tick,
            'new_private_releases':1,'same_Q_recorded_releases':4,'new_banks':1,'new_updates':200,
            'evaluation_sha256':sha(root/'evaluation/result.json'),'privacy_ledger_sha256':sha(root/'privacy_release_ledger.json'),
            'historical_evidence_unchanged':True,'free_bytes':shutil.disk_usage(CODE).free,
            'development_judgment':e['development_judgment'],'automatic_followup':False}
        save(root/'campaign_result.json',final);atomic_json(root/'campaign_status.json',dict(final,phase='COMPLETE'),replace=True)
        state_update(root,'COMPLETE',result=str(root/'evaluation/result.json'),automatic_followup=False)
        print(json.dumps(final),flush=True)
    except BaseException:
        if root.exists():
            atomic_json(root/'campaign_status.json',{'phase':'STOPPED_ERROR','updated':now(),
                'traceback':traceback.format_exc(),'no_automatic_noise_redraw':True},replace=(root/'campaign_status.json').exists())
            state_update(root,'STOPPED_ERROR',automatic_followup=False)
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();run(args.root.resolve(),args.resume)


if __name__=='__main__':main()
