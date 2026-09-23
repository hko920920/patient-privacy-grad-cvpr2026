"""Exactly one fresh Q Gaussian release and one matched seed101 A2 bank."""
import argparse,copy,json,shutil,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from .patient_dp_io import release_from_contract,HANDOFF_KEYS
from .patient_dp_runtime import code_bindings,validate_job,_unchanged_run,execute
from . import repeat_runtime as base
from .run_feasibility import worker
from .evaluate_patient_dp_noise import dependencies,PREVIOUS

ROOT=CODE/'_reports/receiver_dp8_noise_repeat_20260923_v1'
START='2026-09-23T14:00:15+00:00'
DEADLINE='2026-09-23T17:00:15+00:00'
BANK='A2_dp8_noise02_101'
HUMAN=RESEARCH/'RECEIVER_PATIENT_DP_NOISE_REPEAT_CONTRACT_20260923.md'

def before_deadline():
    require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole package three-hour cap reached')

def state_update(root,status,**fields):
    path=RESEARCH/'research_state.json';state=read(path)
    item=state.setdefault('receiver_patient_dp_noise_repeat',{})
    item.update(status=status,updated=now(),output_directory=str(root),absolute_deadline_utc=DEADLINE,**fields)
    active=state.setdefault('active_execution',{})
    active.update(status='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_patient_dp_noise_repeat',
                  current_scope='one fresh Q DP8 release02; one A2 seed101 bank; fixed V comparison',
                  output_directory=str(root))
    state.update(current_step=2,stage2_status='in_progress',updated_utc=now(),
        step2_active_task='Authorized A2 patient-DP independent-noise repeat, fixed synthesis seed101',
        receiver_feasibility_next_action='Complete only release02 and one bank; no automatic third release/seed/final access')
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def write_ledger(root,status):
    ledger={'schema':'receiver.A2-family-privacy-ledger/v1','created':now(),
        'scope':'Two fixed-A2 patient-DP protected releases and their postprocessing only',
        'excludes':'Historical nonDP Q/V selection, CLIP/nonDP controls, utility evaluation/report disclosure and unrelated experiments',
        'adjacency':'add_remove_one_patient','new_release_authorized_count':1,
        'entries':[
            {'release_id':'A2_DP8_release01','status':'COMPLETE','epsilon':8.,'delta':1e-5,
             'protected_targets':str(PREVIOUS/'one_release/protected_targets'),'protected_png':str(PREVIOUS/'protected_png')},
            {'release_id':'A2_DP8_release02','status':status,'epsilon':8.,'delta':1e-5,
             'protected_targets':str(root/'one_release/protected_targets'),'protected_png':str(root/'protected_png')}],
        'if_both_outputs_are_disclosed':{'method':'basic sequential composition','epsilon_upper_bound':16.,
            'delta_upper_bound':2e-5,'optimal_accounting_claimed':False,
            'source':'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf','theorem':'3.16'},
        'same_release_postprocessing_additional_cost':0,'external_publication_performed':False}
    atomic_json(root/'privacy_release_ledger.json',ledger,replace=(root/'privacy_release_ledger.json').exists())

def initialize(root):
    before_deadline();started=time.monotonic();root.mkdir(exist_ok=False)
    before=root/'records_before';before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('RESEARCH_FRAMEWORK.md','research_state.json')]:shutil.copy2(p,before/p.name)
    old=read(PREVIOUS/'campaign_contract.json');oldbinding=read(PREVIOUS/'execution_bindings.json')
    require(read(PREVIOUS/'campaign_result.json')['status']=='COMPLETE_FIRST_DP_COMPARISON','Prior comparison incomplete')
    require(all(sha(p)==h for p,h in oldbinding['implementation'].items()),'Prior implementation changed')
    require(all(sha(p)==h for p,h in old['historical_dependencies'].items()),'Prior input/evidence changed')
    reference=read(old['reference_job'])
    history=dict(old['historical_dependencies']);history.update(oldbinding['implementation'])
    for p in (HUMAN,PREVIOUS/'campaign_contract.json',PREVIOUS/'one_release_contract.json',
              PREVIOUS/'campaign_result.json',PREVIOUS/'DP8_result.json',PREVIOUS/'two_bank_seal.json',
              PREVIOUS/'evaluation/result.json',PREVIOUS/'protected_png/release_manifest.json'):
        history[str(p)]=sha(p)
    c={'schema':'receiver.patient-dp-independent-noise-repeat/v1','status':'AUTHORIZED_FROZEN','created':now(),
        'authorization':'Latest user explicitly authorizes exactly one fresh independent DP noise and one matched A2 bank plus evaluation',
        'scope_started_utc':START,'absolute_deadline_utc':DEADLINE,'maximum_seconds_per_bank':7200,
        'maximum_package_seconds':10800,'estimated_total_minutes':[80,95],'synthesis_reference_minutes':75.38,
        'bank_ids':{'DP8':BANK},'max_private_releases':1,'prior_A2_private_releases':1,'A2_total_after_completion':2,
        'reference_job':old['reference_job'],'initialization_witness':old['initialization_witness'],
        'initialization_sha256':old['initialization_sha256'],'historical_dependencies':history,
        'public_calibration':old['public_calibration'],'public_calibration_sha256':old['public_calibration_sha256'],
        'mechanism':old['mechanism'],'recipe':old['recipe'],
        'class_patient_weight':old['class_patient_weight'],'postprocessing':old['postprocessing'],
        'privacy_boundary':old['privacy_boundary'],
        'joint_basic_composition':{'epsilon':16.,'delta':2e-5,'optimal_accounting_claimed':False},
        'primary':'DenseNet release02 minus public-only A2 seed101 AUROC and AP',
        'development_rule':old['development_rule'],
        'evaluation':'Only after final200 PNG sealed; prior four-cell scores and same2000 patient draws reused',
        'no_redraw_no_HPO_no_checkpoint_selection':True,'automatic_followup':False,
        'Expert_Reserved_final_receiver':False}
    save(root/'campaign_contract.json',c)
    state_update(root,'PREPARING',started=now(),new_private_releases=0,new_banks=1,development_evaluation_opened=False)
    plan=copy.deepcopy(read(PREVIOUS/'one_release_contract.json'))
    plan.update(status='FROZEN_FOR_EXECUTION',output_directory=str(root/'one_release'),
                campaign_authority=str(root/'campaign_contract.json'),campaign_authority_sha256=sha(root/'campaign_contract.json'),
                release_id='A2_DP8_release02',prior_release_count=1,independent_OS_randomness=True)
    save(root/'one_release_contract.json',plan)
    write_ledger(root,'RESERVED_NOT_REDRAWABLE')
    protected=release_from_contract(root/'one_release_contract.json')
    write_ledger(root,'TARGET_COMPLETE')
    state_update(root,'TARGET_READY',new_private_releases=1)
    binding={'variant':'DP8',**protected,**{k+'_sha256':sha(v) for k,v in protected.items()}}
    target_binding=root/'DP8_target_binding.json';save(target_binding,binding)
    job=dict(reference,schema='receiver.patient-dp-s200-bank/v1',id=BANK,variant='DP8',population='pooled',
        seed=101,template_seed=101,condition_seed=101,source_set='A2',**protected,
        **{k+'_sha256':sha(v) for k,v in protected.items()},target_binding=str(target_binding),
        target_binding_sha256=sha(target_binding),campaign_contract=str(root/'campaign_contract.json'),
        campaign_contract_sha256=sha(root/'campaign_contract.json'),maximum_seconds=7200,
        absolute_deadline_utc=DEADLINE,paired_initialization=old['initialization_witness'],code_bindings=code_bindings())
    (root/'jobs').mkdir();jobfile=root/'jobs/DP8.json';save(jobfile,job);validate_job(job)
    require(_unchanged_run.__code__ is base._run.__code__ and execute.__code__ is base.execute.__code__,'Training/replay changed')
    oldjob=read(PREVIOUS/'jobs/DP8.json');checks={}
    for key in HANDOFF_KEYS:
        newh=read(job[key]);oldh=read(oldjob[key])
        require(read(newh['contract_file'])==read(oldh['contract_file']),'Protection mechanism changed')
        require(sha(newh['rule_file'])==sha(oldh['rule_file']),'Condition/head rule changed')
        with np.load(newh['target_file'],allow_pickle=False) as z:new=z['pooled_gradient'].copy()
        with np.load(oldh['target_file'],allow_pickle=False) as z:previous=z['pooled_gradient'].copy()
        require(new.shape==(4,34) and np.isfinite(new).all() and np.all(np.linalg.norm(new,axis=-1)>1e-12),'Invalid fresh target')
        checks[key]={'mechanism_and_public_rule_equal':True,'finite_nonzero_target':True,
                     'same_value_as_previous_target':bool(np.array_equal(new,previous))}
    verification={'status':'PASS_CHANGED_REPEAT_BINDING','targets':checks,'unchanged_training_replay_bytecode':True,
                  'encoder_forwards':0,'new_synthesis_updates':0,'no_repeated_GPU_profile_or_full_test':True}
    save(root/'changed_path_verification.json',verification)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    execution={'created':now(),'job':str(jobfile),'job_sha256':sha(jobfile),'implementation':sources,
        'runtime':code_bindings(),'evaluation_dependencies':dependencies(),
        'campaign_contract_sha256':sha(root/'campaign_contract.json'),
        'changed_path_verification_sha256':sha(root/'changed_path_verification.json')}
    save(root/'execution_bindings.json',execution);snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    save(root/'preparation_result.json',{'status':'COMPLETE_ONE_FRESH_RELEASE_ONE_JOB','seconds':time.monotonic()-started,
        'new_private_releases':1,'A2_family_releases':2,'new_encoder_forwards':0,'new_Q_V_pixels':0,
        'private_noisy_target_reused':False,'raw_cache_reused':True,'no_noise_seed_or_raw_noise_stored':True})
    print(json.dumps({'phase':'PREPARED','new_private_releases':1,'checks':verification['status'],
                     'seconds':time.monotonic()-started}),flush=True)
    return execution

def protected_package(root):
    directory=root/'protected_png'
    if directory.exists():
        manifest=read(directory/'release_manifest.json')
        require(all(sha(directory/p)==h for p,h in manifest['files_sha256'].items()),'Protected package changed')
        return
    directory.mkdir();artifact=root/'banks'/BANK/'artifact'
    shutil.copytree(artifact/'images',directory/'images')
    for name in ('images.csv','pairs.csv'):shutil.copy2(artifact/name,directory/name)
    save(directory/'learning_rule.json',{
        'schema':'receiver.patient-dp-png-release/v1','release_id':'A2_DP8_release02','epsilon':8.,'delta':1e-5,
        'adjacency':'add_remove_one_patient','private_signal_releases_for_this_bank':1,'DP_applied':True,
        'A2_family_release_count':2,'if_both_A2_releases_are_disclosed_basic_bound':{'epsilon':16.,'delta':2e-5},
        'training_data_protection':'Q given fixed public P and frozen source/condition recipe',
        'development_selection_protected':False,'images':128,'updates':200,'synthesis_seed':101,
        'source_encoders':['BioViL','DINOv2 ViT-B14'],'condition_count':4,'source_aggregation':'equal mean',
        'receiver_rule':'class-balanced point-only ridge0.1 with receiver public-only projection',
        'raw_Q_metadata_included':False,'raw_evaluation_predictions_included':False,
        'external_publication_performed':False})
    payload={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    save(directory/'release_manifest.json',{'schema':'receiver.protected-png-files/v1','files_sha256':payload})
    require(len(list((directory/'images').glob('*.png')))==128,'Protected PNG count mismatch')

def run(root,resume=False):
    started=time.monotonic()
    try:
        before_deadline();binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Job changed')
        durations={};bank=root/'banks'/BANK;resultfile=root/'DP8_result.json'
        if not resultfile.exists():
            command=['-m','receiver_distillation.patient_dp_runtime','--job',binding['job'],
                     '--output',str(bank),'--result',str(resultfile)]
            if bank.exists():command.append('--resume')
            state_update(root,'RUNNING_BANK',new_private_releases=1)
            durations['DP8']=worker(root,'DP8_noise02_main',command)
        result=read(resultfile)
        require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank; no evaluation')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        verify_artifact(bank/'artifact',spec,done['signature'])
        witness=read(read(binding['job'])['paired_initialization']);initial=read(bank/'initialization.json')
        require(all(initial[k]==witness[k] for k in ('seed','condition_seed','fields','template_tensor_digest')),'Historical initialization mismatch')
        require(read(bank/'artifact/learning_contract.json')['DP_applied'] is True,'Privacy label missing')
        if not(root/'bank_seal.json').exists():
            save(root/'bank_seal.json',{'created':now(),'completion_sha256':sha(bank/'COMPLETED.json'),
                'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        protected_package(root);before_deadline()
        state_update(root,'EVALUATING',development_evaluation_opened=True)
        if not(root/'evaluation/result.json').exists():
            durations['evaluation']=worker(root,'noise_repeat_evaluation',
                ['-m','receiver_distillation.evaluate_patient_dp_noise','--root',str(root)])
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen code changed')
        history=read(root/'campaign_contract.json')['historical_dependencies']
        require(all(sha(p)==h for p,h in history.items()),'Prior code/data/results changed')
        before_deadline();evaluation=read(root/'evaluation/result.json');write_ledger(root,'COMPLETE')
        output={'status':'COMPLETE_DP_NOISE_REPEAT','finished':now(),'controller_seconds':time.monotonic()-started,
            'durations_this_controller':durations,'new_private_releases':1,'A2_family_private_releases':2,
            'new_banks':1,'new_updates':200,'automatic_followup':False,
            'evaluation_result_sha256':sha(root/'evaluation/result.json'),
            'privacy_ledger_sha256':sha(root/'privacy_release_ledger.json'),
            'protected_package':str(root/'protected_png'),'free_bytes':shutil.disk_usage(CODE).free,
            'historical_code_inputs_and_results_preserved':True,
            'development_judgment':evaluation['development_judgment'],
            'both_DP_positive':evaluation['both_DP_positive']}
        save(root/'campaign_result.json',output)
        atomic_json(root/'campaign_status.json',dict(output,phase='COMPLETE'),replace=True)
        state_update(root,'COMPLETE',result=str(root/'evaluation/result.json'),automatic_followup=False)
        print(json.dumps(output),flush=True)
    except BaseException:
        if root.exists():
            atomic_json(root/'campaign_status.json',{'phase':'STOPPED_ERROR','updated':now(),
                'seconds':time.monotonic()-started,'traceback':traceback.format_exc(),'no_automatic_noise_redraw':True},
                replace=(root/'campaign_status.json').exists())
            state_update(root,'STOPPED_ERROR',automatic_followup=False)
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args();run(args.root.resolve(),args.resume)

if __name__=='__main__':main()

