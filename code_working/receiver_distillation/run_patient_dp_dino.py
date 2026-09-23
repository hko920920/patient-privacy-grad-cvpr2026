"""Prepare one DINO-only patient query, draw once, synthesize once, compare both A2 DPs."""
import argparse,json,shutil,time,traceback,copy
from pathlib import Path
from datetime import datetime
import numpy as np
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from . import patient_dp_dino as dp
from .patient_dp_dino_io import cached_paths,read_cached_population,release_from_contract
from .verify_patient_dp_dino import prepare_check
from .patient_dp_dino_runtime import code_bindings,validate_job,_unchanged_run,execute
from . import dino_control_runtime as base
from .run_feasibility import worker
from .evaluate_patient_dp_dino import dependencies,PREVIOUS

ROOT=CODE/'_reports/receiver_dino_dp8_20260924_v1'
DINO=CODE/'_reports/receiver_dino_only_s200_20260923_v1'
FIRST=CODE/'_reports/receiver_dp8_first_20260923_v1'
START='2026-09-23T15:38:08+00:00'
DEADLINE='2026-09-23T17:08:08+00:00'
BANK='DINO_dp8_noise01_101'
HUMAN=RESEARCH/'RECEIVER_DINO_PATIENT_DP_CONTRACT_20260924.md'


def before_deadline():
    require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole package90-minute cap reached')


def state_update(root,status,**fields):
    path=RESEARCH/'research_state.json';s=read(path)
    entry=s.setdefault('receiver_DINO_patient_dp_control',{})
    entry.update(status=status,updated=now(),output_directory=str(root),absolute_deadline_utc=DEADLINE,**fields)
    s.setdefault('active_execution',{}).update(status='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_DINO_patient_dp_control',
        current_scope='One DINO-only DP8 query and seed101 bank; compare both A2 DP outcomes',output_directory=str(root))
    s.update(current_step=2,stage2_status='in_progress',updated_utc=now(),
             step2_active_task='DINO-only patient-DP matched control: '+status)
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def ledger(root,status):
    prior=read(PREVIOUS/'privacy_release_ledger.json')
    entries=copy.deepcopy(prior['entries']);entries.append({'release_id':'DINO_DP8_release01','status':status,
        'epsilon':8.,'delta':1e-5,'protected_targets':str(root/'one_release/protected_targets'),
        'protected_png':str(root/'protected_png')})
    value={'schema':'receiver.same-Q-patient-DP-ledger/v1','created':now(),'entries':entries,
        'scope':'These three frozen protected summaries and their postprocessing, over the same Q patients',
        'excludes':'Past nonDP development/selection, raw Q diagnostics, nonDP controls, V predictions and utility reports',
        'adjacency':'add_remove_one_patient','new_release_authorized_count':1,
        'per_method_budget':{'epsilon':8.,'delta':1e-5},
        'if_all_three_outputs_disclosed':{'method':'basic sequential composition','epsilon_upper_bound':24.,
            'delta_upper_bound':3e-5,'optimal_accounting_claimed':False,
            'source':'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf','theorem':'3.16'},
        'same_summary_postprocessing_additional_cost':0,'external_publication_performed':False}
    atomic_json(root/'privacy_release_ledger.json',value,replace=(root/'privacy_release_ledger.json').exists())


def initialize(root):
    before_deadline();tick=time.monotonic();root.mkdir(exist_ok=False)
    (root/'records_before').mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:shutil.copy2(p,root/'records_before'/p.name)
    prior=read(PREVIOUS/'execution_bindings.json');pc=read(PREVIOUS/'campaign_contract.json')
    require(all(sha(p)==h for p,h in prior['implementation'].items()),'Frozen prior code changed')
    require(all(sha(p)==h for p,h in pc['historical_dependencies'].items()),'Prior evidence changed')
    history={**pc['historical_dependencies'],**prior['implementation']}
    for p in (HUMAN,DINO/'job.json',DINO/'campaign_result.json',DINO/'DINO_result.json',
              DINO/'evaluation/result.json',DINO/'evaluation/predictions_private.npz',DINO/'bank_seal.json',
              PREVIOUS/'campaign_result.json',PREVIOUS/'evaluation/result.json',PREVIOUS/'privacy_release_ledger.json'):
        history[str(p)]=sha(p)
    reference=DINO/'job.json';old=read(reference)
    require(all(sha(p)==h for p,h in old['code_bindings'].items()),'DINO runtime changed')
    paths=cached_paths(reference)
    # Calibrate strictly on P before loading Q. No A2 protected target is used.
    p,pref=read_cached_population(paths,'P');limits,details=dp.public_limits(p)
    calibration={'schema':'receiver.public-patient-gradient-calibration/v1','created':now(),'population':'P_ONLY',
        'encoders':['dinov2_vitb14'],'class_bounds':list(limits.class_bounds),'details':details,
        'rule':'class-present DINO K4-vector norm q95 linear; floor1e-6',
        'public_feature_hashes':pref['feature_hashes'],'chosen_before_loading_Q_values':True,'Q_or_V_used_for_choice':False}
    save(root/'public_calibration.json',calibration)
    print(json.dumps({'phase':'P_DINO_CALIBRATION_SEALED','class_bounds':limits.class_bounds}),flush=True)
    checks,qref=prepare_check(root,paths,p,pref,limits)
    input_bindings={}
    for item in paths.values():
        for key in ('handoff','target','rule_file','P','Q'):input_bindings[item[key]]=sha(item[key])
        for key in ('checkpoint','projection'):input_bindings[item['rule'][key]]=sha(item['rule'][key])
    input_bindings[str(reference)]=sha(reference)
    for file in Path(__file__).parent.glob('patient_dp*.py'):input_bindings[str(file)]=sha(file)
    mechanism=dp.mechanism_metadata(limits)
    c={'schema':'receiver.dino-patient-dp-control/v1','status':'AUTHORIZED_FROZEN','created':now(),
        'authorization':'Latest user: own DINO query/clipping; one fresh DP8 release and one matched seed101 bank; evaluate both A2 DP results',
        'scope_started_utc':START,'absolute_deadline_utc':DEADLINE,'maximum_seconds_per_bank':3600,
        'maximum_package_seconds':5400,'estimated_total_minutes':[50,70],'synthesis_reference_minutes':34.71,
        'bank_ids':{'DP8':BANK},'max_private_releases':1,'reference_job':str(reference),
        'historical_dependencies':history,'input_code_bindings':input_bindings,
        'public_calibration':str(root/'public_calibration.json'),'public_calibration_sha256':sha(root/'public_calibration.json'),
        'mechanism':mechanism,'recipe':{k:old[k] for k in ('seed','images','updates','microbatch','activation_updates','optimizer')},
        'objective':'Single DINO full-weight mean K4 CE-gradient cosine; anchor/TV once; no BioViL matching',
        'patient_class_weight':'Per patient/class visit mean; class present-patient equal mean; class weights0.5 each; P counts exact, Q masses protected',
        'query':'0.5 concat(clip(g_p0,C0)/C0,clip(g_p1,C1)/C1,b_p0,b_p1); g_pc is DINO K4 x34',
        'dimension':274,'sensitivity_proof':'Squared patient norm <=0.25*(1+1+1+1)=1; one-patient add/remove sum sensitivity1',
        'postprocessing':'same as A2: nonnegative noisy class masses, sum projected to Cc*mass, pooled P+Q denominator floor1, zero-target public fallback',
        'privacy_boundary':'Fixed public recipe protects Q for this release only; historical nonDP development and utility reports are excluded',
        'sampler':'SystemRandom; two independent normalvariate draws per coordinate; no seed/realized-noise save or redraw; no separate finite-precision proof',
        'primary':'Each A2 DP noise1/noise2 minus fresh DINO DP DenseNet AUROC; AP corroboration; report both, never pick best',
        'descriptive_rule':'A2 advantage in both fixed comparisons iff each AUROC delta>0, patient-CI lower>0, AP delta>=0; otherwise mixed/no clear advantage',
        'inference_limit':'One DINO noise versus two A2 noises, one synthesis seed; conditional patient CIs, not noise/seed-population superiority',
        'evaluation':'Final200 PNG seal first; existing V features and2000 paired patient draws; no new Q/V feature forwards',
        'equal_updates_not_equal_compute':True,'no_redraw_no_HPO_no_checkpoint_selection':True,
        'automatic_followup':False,'Expert_Reserved_final_receiver':False,
        'free_bytes_before':shutil.disk_usage(CODE).free}
    save(root/'campaign_contract.json',c)
    state_update(root,'PREPARING_TARGET',new_private_releases=0,new_banks=1,development_evaluation_opened=False)
    plan={'schema':'receiver.patient-dp-release-plan/v1','status':'FROZEN_FOR_EXECUTION','max_private_releases':1,
        'synthesis_authorized':False,'output_directory':str(root/'one_release'),'input_code_bindings':input_bindings,
        'public_calibration':str(root/'public_calibration.json'),'public_calibration_sha256':sha(root/'public_calibration.json'),
        'reference_job':str(reference),'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient',
        'campaign_authority':str(root/'campaign_contract.json'),'campaign_authority_sha256':sha(root/'campaign_contract.json'),
        'release_id':'DINO_DP8_release01','independent_OS_randomness':True}
    save(root/'one_release_contract.json',plan);ledger(root,'RESERVED_NOT_REDRAWABLE')
    protected=release_from_contract(root/'one_release_contract.json');ledger(root,'TARGET_COMPLETE')
    tb={'variant':'DP8',**protected,**{k+'_sha256':sha(v) for k,v in protected.items()}}
    save(root/'DP8_target_binding.json',tb)
    job=dict(old,schema='receiver.dino-patient-dp-s200-bank/v1',id=BANK,variant='DP8',source_set='DINO',
        template_seed=101,condition_seed=101,**protected,**{k+'_sha256':sha(v) for k,v in protected.items()},
        campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
        target_binding=str(root/'DP8_target_binding.json'),target_binding_sha256=sha(root/'DP8_target_binding.json'),
        absolute_deadline_utc=DEADLINE,code_bindings=code_bindings())
    save(root/'job.json',job);validate_job(job)
    require(_unchanged_run.__code__ is base._run.__code__ and execute.__code__ is base.execute.__code__,'DINO training bytecode changed')
    # No GPU re-profile or replay revalidation: target loading is the only new input.
    objective,_=base.load_dino(job['second_handoff'],(0,)*64+(1,)*64)
    require(set(objective.targets)=={'dinov2_vitb14'} and not any(v.requires_grad for v in objective.targets.values()),'Wrong objective/target gradients')
    for key,value in (('updates',199),('source_set','A2')):
        bad=dict(job);bad[key]=value
        try:validate_job(bad)
        except RuntimeError:pass
        else:raise RuntimeError('Bad job accepted')
    save(root/'job_verification.json',{'status':'PASS','same_training_replay_bytecode':True,'only_DINO_objective':True,
        'bad_jobs_rejected':2,'fresh_initialization_required':True,'new_GPU_checks':0})
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'job':str(root/'job.json'),'job_sha256':sha(root/'job.json'),'implementation':sources,
        'runtime':code_bindings(),'evaluation_dependencies':dependencies(),'campaign_contract_sha256':sha(root/'campaign_contract.json')}
    save(root/'execution_bindings.json',binding);(root/'implementation_snapshot').mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,root/'implementation_snapshot'/file.name)
    save(root/'preparation_result.json',{'status':'COMPLETE','seconds':time.monotonic()-tick,'new_private_releases':1,
        'query_dimension':274,'protection_check':checks['status'],'new_Q_V_pixels':0,'new_encoder_forwards':0})
    print(json.dumps({'phase':'PREPARED_ONE_DINO_QUERY','seconds':time.monotonic()-tick,'dimension':274}),flush=True)
    return binding


def protected_package(root):
    directory=root/'protected_png';directory.mkdir(exist_ok=False);artifact=root/'banks'/BANK/'artifact'
    shutil.copytree(artifact/'images',directory/'images')
    for name in ('images.csv','pairs.csv'):shutil.copy2(artifact/name,directory/name)
    save(directory/'learning_rule.json',{'schema':'receiver.patient-dp-png-release/v1','release_id':'DINO_DP8_release01',
        'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient','protected_query_count':1,'query_dimension':274,
        'DP_applied':True,'training_data_protection':'Q under fixed public P and recipe','development_selection_protected':False,
        'images':128,'updates':200,'synthesis_seed':101,'source_encoders':['DINOv2 ViT-B14'],'conditions':4,
        'receiver_rule':'class-balanced point-only ridge0.1 with receiver public-only projection',
        'raw_Q_metadata_included':False,'evaluation_predictions_included':False,'external_publication_performed':False,
        'if_jointly_disclosed_with_both_A2_DP_summaries_basic_bound':{'epsilon':24.,'delta':3e-5}})
    files={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    save(directory/'release_manifest.json',{'schema':'receiver.protected-png-files/v1','files_sha256':files})
    require(len(list((directory/'images').glob('*.png')))==128,'Wrong protected PNG count')


def run(root,resume=False):
    tick=time.monotonic()
    try:
        before_deadline();binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Job changed')
        file=root/'DP8_result.json';bank=root/'banks'/BANK
        if not file.exists():
            cmd=['-m','receiver_distillation.patient_dp_dino_runtime','--job',binding['job'],'--output',str(bank),'--result',str(file)]
            if bank.exists():cmd.append('--resume')
            state_update(root,'RUNNING_BANK',new_private_releases=1)
            worker(root,'DINO_DP8_main',cmd)
        result=read(file);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json');verify_artifact(bank/'artifact',spec,done['signature'])
        require(read(bank/'initialization_parity.json')['status']=='PASS_EXACT_A2_INITIALIZATION','Initial state differs')
        require(read(bank/'artifact/learning_contract.json')['DP_applied'] is True,'Wrong privacy metadata')
        if not(root/'bank_seal.json').exists():save(root/'bank_seal.json',{'created':now(),
            'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        if not(root/'protected_png').exists():protected_package(root)
        before_deadline();state_update(root,'EVALUATING',development_evaluation_opened=True)
        if not(root/'evaluation/result.json').exists():
            worker(root,'DINO_DP_comparison',['-m','receiver_distillation.evaluate_patient_dp_dino','--root',str(root)])
        c=read(root/'campaign_contract.json')
        require(all(sha(p)==h for p,h in {**binding['implementation'],**c['historical_dependencies'],**c['input_code_bindings']}.items()),'Frozen evidence changed')
        before_deadline();e=read(root/'evaluation/result.json');ledger(root,'COMPLETE')
        final={'status':'COMPLETE_DINO_PATIENT_DP_CONTROL','finished':now(),'controller_seconds':time.monotonic()-tick,
            'new_private_releases':1,'A2_releases_reused':2,'new_banks':1,'new_updates':200,
            'evaluation_sha256':sha(root/'evaluation/result.json'),'privacy_ledger_sha256':sha(root/'privacy_release_ledger.json'),
            'historical_evidence_unchanged':True,'free_bytes':shutil.disk_usage(CODE).free,
            'development_judgment':e['development_judgment'],'automatic_followup':False}
        save(root/'campaign_result.json',final);atomic_json(root/'campaign_status.json',dict(final,phase='COMPLETE'),replace=True)
        state_update(root,'COMPLETE',result=str(root/'evaluation/result.json'),automatic_followup=False)
        print(json.dumps(final),flush=True)
    except BaseException:
        if root.exists():
            atomic_json(root/'campaign_status.json',{'phase':'STOPPED_ERROR','updated':now(),'traceback':traceback.format_exc(),
                'no_automatic_noise_redraw':True},replace=(root/'campaign_status.json').exists())
            state_update(root,'STOPPED_ERROR',automatic_followup=False)
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();run(a.root.resolve(),a.resume)


if __name__=='__main__':main()
