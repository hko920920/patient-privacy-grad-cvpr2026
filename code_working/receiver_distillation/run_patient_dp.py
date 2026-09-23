"""One authorized Q release, CLIP/DP8 seed101 banks, then one fixed evaluation."""
import argparse,json,shutil,time,traceback,copy
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact,digest
from .patient_dp import Limits,clipped_contributions,sum_patient_axis,decode_target,mechanism_metadata
from .patient_dp_io import cached_paths,read_cached_population,release_from_contract,ENCODERS,HANDOFF_KEYS
from .patient_dp_runtime import code_bindings,validate_job,annotate_artifact_metadata,_unchanged_run,execute
from . import repeat_runtime as base
from .second_source import load_combined
from .signals import objective_digest
from .run_feasibility import worker
from .evaluate_patient_dp import dependencies,PUBLIC,FIRST

DESIGN=CODE/'_reports/receiver_patient_dp_design_20260923_v2'
ROOT=CODE/'_reports/receiver_dp8_first_20260923_v1'
BANKS={'CLIP':'A2_clip_101','DP8':'A2_dp8_101'}
DEADLINE='2026-09-23T16:08:39+00:00'


def before_deadline():
    require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole campaign five-hour cap reached')


def state_update(root,status,**fields):
    path=RESEARCH/'research_state.json';s=read(path)
    item=s.setdefault('receiver_first_patient_dp_comparison',{})
    item.update(status=status,updated=now(),output_directory=str(root),absolute_deadline_utc=DEADLINE,**fields)
    s['active_execution']['status']='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_patient_dp_first_comparison'
    s['active_execution']['current_scope']='one Q DP8 release; one CLIP and one DP8 bank; V after both final PNGs'
    s['active_execution']['output_directory']=str(root)
    s['current_step']=2;s['stage2_status']='in_progress';s['updated_utc']=now()
    s['receiver_feasibility_next_action']='Complete only the authorized first DP comparison; no automatic additional release/seed/final access.'
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def clipping_handoffs(root,paths,limits):
    p,_=read_cached_population(paths,'P');q,_=read_cached_population(paths,'Q')
    require(not(set(p.ids)&set(q.ids)),'P/Q patient overlap')
    ps,pc=p.sums_counts();contrib,_=clipped_contributions(q,limits)
    targets,post=decode_target(sum_patient_axis(contrib),limits,ps,pc)
    folder=root/'clipping_targets_INTERNAL';folder.mkdir()
    descriptor=folder/'mechanism.json'
    save(descriptor,{'scope':'NONDP_CLIPPING_CONTROL','noise_applied':False,
        'same_aggregation_and_postprocessing_as_DP8':True,'class_bounds':list(limits.class_bounds)})
    result={}
    for name,key in zip(ENCODERS,HANDOFF_KEYS):
        d=folder/name;d.mkdir();rule=paths[name]['rule'];rulefile=d/'condition_rule.json';save(rulefile,rule)
        target=d/'condition_targets.npz'
        np.savez(target,schema=np.array('receiver.separate-condition-target/v1'),
            rule_sha256=np.array(digest(rule)),condition_ids=np.arange(4),pooled_gradient=targets[name])
        handoff={'schema':'receiver.target-handoff/v1','target_file':str(target),'target_sha256':sha(target),
            'rule_file':str(rulefile),'rule_file_sha256':sha(rulefile),'rule_sha256':digest(rule),
            'contract_file':str(descriptor),'contract_sha256':sha(descriptor),'population':'pooled',
            'input_population':'P + clipped Q; no noise','privacy':'NONDP_CLIPPING_CONTROL',
            'not_an_execution_authorization':True}
        f=d/'target_handoff.json';save(f,handoff);result[key]=str(f)
    save(folder/'internal_postprocessing.json',post)
    return result


def changed_path_checks(root,jobs):
    """New bindings/metadata/target values only; no repeated GPU/profile/resume test."""
    require(_unchanged_run.__code__ is base._run.__code__ and execute.__code__ is base.execute.__code__,
            'Unexpected change to frozen training/replay loop')
    labels=(0,)*64+(1,)*64;results={}
    for arm,path in jobs.items():
        job=read(path);validate_job(job)
        objective,_=load_combined(job['a1_handoff'],job['second_handoff'],labels)
        g=torch.Generator().manual_seed(443)
        x={n:torch.randn((4,128,16),generator=g,dtype=torch.float64,requires_grad=True) for n in ENCODERS}
        loss,_=objective.matching(x);grad=torch.autograd.grad(loss,tuple(x.values()))
        require(torch.isfinite(loss) and all(torch.isfinite(v).all() and v.norm()>0 for v in grad),'New target feature gradient invalid')
        require(not any(t.requires_grad for t in objective.targets.values()),'Private target must be detached')
        metadata=annotate_artifact_metadata('learning_contract.json',{'DP_applied':False},job)
        require(metadata['DP_applied']==(arm=='DP8'),'Artifact privacy metadata wrong')
        bad=copy.deepcopy(job);bad['updates']=201
        try:validate_job(bad)
        except RuntimeError:pass
        else:raise RuntimeError('Changed update schedule accepted')
        bad=copy.deepcopy(job);bad['variant']='DP8' if arm=='CLIP' else 'CLIP'
        try:validate_job(bad)
        except RuntimeError:pass
        else:raise RuntimeError('Cross-arm target accepted')
        results[arm]={'status':'PASS','objective_digest':objective_digest(objective),'loss':float(loss),
            'feature_gradient_norms':[float(v.norm()) for v in grad],
            'wrong_schedule_and_target_rejected':True,'detached_target':True,'DP_artifact_flag':metadata['DP_applied']}
    report={'status':'PASS_CHANGED_DP_JOB_TARGET_METADATA_CONNECTION','cases':results,
        'unchanged_training_and_replay_bytecode':True,'public_or_constructed_features_only':True,
        'encoder_forwards':0,'synthesis_updates':0,'profile_or_full_preparation_repeated':False}
    save(root/'changed_path_verification.json',report);return report


def initialize(root):
    started=time.monotonic();before_deadline();root.mkdir(exist_ok=False)
    before=root/'records_before';before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
              RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:shutil.copy2(p,before/p.name)
    draft=read(DESIGN/'first_dp_comparison_DRAFT.json');verified=read(DESIGN/'verification.json')
    require(verified['status']=='PASS_TARGET_PATH_NO_Q_RELEASE_OR_TRAINING','Target verification incomplete')
    require(all(sha(p)==h for p,h in draft['input_code_bindings'].items()),'Verified target code/inputs changed')
    reference_job=FIRST/'jobs/A2.json';reference=read(reference_job)
    witness=PUBLIC/'reference_initialization.json'
    require(sha(witness)==read(PUBLIC/'campaign_contract.json')['initialization_sha256'],'Seed101 initialization witness changed')
    calibration=read(draft['public_calibration']);limits=Limits(tuple(calibration['class_bounds']))
    require(sha(draft['public_calibration'])==draft['public_calibration_sha256'],'Public calibration changed')
    history=dict(draft['input_code_bindings'])
    for f in (reference_job,witness,PUBLIC/'campaign_contract.json',DESIGN/'verification.json',
              DESIGN/'unit_tests_first.json',RESEARCH/'RECEIVER_PATIENT_DP_TARGET_SPEC_20260923.md',
              RESEARCH/'RECEIVER_PATIENT_DP_FIRST_COMPARISON_CONTRACT_20260923.md'):
        history[str(f)]=sha(f)
    c={'schema':'receiver.first-patient-dp-comparison/v1','status':'AUTHORIZED_FROZEN','created':now(),
       'authorization':'Latest user: bind contract/jobs then execute one clipping-only and one DP bank through joint V evaluation',
       'scope_started_utc':'2026-09-23T11:08:39+00:00','absolute_deadline_utc':DEADLINE,
       'maximum_seconds_per_bank':7200,'maximum_package_seconds':18000,
       'estimated_total_minutes':[155,180],'synthesis_reference_minutes':148,
       'bank_ids':BANKS,'reference_job':str(reference_job),'initialization_witness':str(witness),
       'initialization_sha256':sha(witness),'historical_dependencies':history,
       'public_calibration':draft['public_calibration'],'public_calibration_sha256':draft['public_calibration_sha256'],
       'mechanism':mechanism_metadata(limits),'max_private_releases':1,
       'class_patient_weight':'visit mean per patient/class, clipped class signal sum plus protected class counts; 0.5 each class',
       'postprocessing':'nonnegative noisy Q mass; class sum ball projection; pooled P+protected Q denominator; zero-signal public fallback',
       'clipping_control':'same clipping/aggregation/decoding; Gaussian noise only omitted',
       'privacy_boundary':'fixed algorithm Q-to-summary-to-DP PNG; internal nonDP control/evaluation/selection not protected',
       'public_release_package':'protected_png only; no internal signatures, checkpoint, Q hash/count/seed/noise',
       'recipe':{k:reference[k] for k in ('images','updates','microbatch','seed','activation_updates','optimizer','templates')},
       'evaluation':'Both final200 PNGs sealed first; same fixed V caches/readout and2000 paired patient bootstrap draws',
       'primary':'DenseNet DP8 minus public-only A2 seed101 AUROC',
       'development_rule':{'positive':'AUROC delta>0 and patient CI lower>0 and AP point delta>=0',
                           'mixed':'AUROC point positive but either other check fails',
                           'not_positive':'AUROC point delta<=0'},
       'source_evaluation':'BioViL secondary; no source-based selection',
       'no_redraw_no_HPO_no_checkpoint_selection':True,'automatic_followup':False,
       'Expert_Reserved_final_receiver':False}
    save(root/'campaign_contract.json',c)
    state_update(root,'PREPARING',started=now(),new_private_releases=0,new_banks=2,development_evaluation_opened=False)
    # The nested release command authorizes target creation only. The enclosing
    # campaign separately authorizes two synthesis jobs with a single DP query.
    release_plan=dict(draft,status='FROZEN_FOR_EXECUTION',output_directory=str(root/'one_release'),
        campaign_authority=str(root/'campaign_contract.json'),campaign_authority_sha256=sha(root/'campaign_contract.json'))
    save(root/'one_release_contract.json',release_plan)
    paths=cached_paths(draft['reference_job'])
    clip=clipping_handoffs(root,paths,limits)
    before_deadline()
    protected=release_from_contract(root/'one_release_contract.json')
    state_update(root,'TARGETS_READY',new_private_releases=1)
    handoffs={'CLIP':clip,'DP8':protected};jobs={};(root/'jobs').mkdir()
    for arm,rid in BANKS.items():
        target_binding={'variant':arm,**handoffs[arm],**{k+'_sha256':sha(v) for k,v in handoffs[arm].items()}}
        file=root/(arm+'_target_binding.json');save(file,target_binding)
        job=dict(reference,schema='receiver.patient-dp-s200-bank/v1',id=rid,variant=arm,population='pooled',
            seed=101,template_seed=101,condition_seed=101,source_set='A2',**handoffs[arm],
            **{k+'_sha256':sha(v) for k,v in handoffs[arm].items()},target_binding=str(file),target_binding_sha256=sha(file),
            campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
            maximum_seconds=7200,absolute_deadline_utc=DEADLINE,paired_initialization=str(witness),code_bindings=code_bindings())
        path=root/'jobs'/(arm+'.json');save(path,job);jobs[arm]=str(path)
    check=changed_path_checks(root,jobs)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'jobs':jobs,'job_sha256':{k:sha(v) for k,v in jobs.items()},
        'implementation':sources,'runtime':code_bindings(),'evaluation_dependencies':dependencies(),
        'campaign_contract_sha256':sha(root/'campaign_contract.json'),'changed_path_verification_sha256':sha(root/'changed_path_verification.json')}
    save(root/'execution_bindings.json',binding);snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    save(root/'preparation_result.json',{'status':'COMPLETE_ONE_RELEASE_TWO_BOUND_JOBS','seconds':time.monotonic()-started,
        'private_releases':1,'new_features_extracted':0,'new_encoder_forwards':0,'synthesis_updates':0,
        'changed_path_checks':check,'no_noise_seed_or_raw_noise_stored':True})
    state_update(root,'RUNNING_BANKS',contract=str(root/'campaign_contract.json'),new_private_releases=1)
    print(json.dumps({'phase':'PREPARED','private_releases':1,'checks':check['status'],'seconds':time.monotonic()-started}),flush=True)
    return binding


def protected_png_package(root):
    """Copy only DP-derived pixels and fixed public rules, no raw-dependent hashes."""
    directory=root/'protected_png'
    if directory.exists():
        manifest=read(directory/'release_manifest.json')
        require(all(sha(directory/f)==h for f,h in manifest['files_sha256'].items()),'Protected PNG package changed')
        return
    directory.mkdir();artifact=root/'banks'/BANKS['DP8']/'artifact'
    shutil.copytree(artifact/'images',directory/'images')
    for name in ('images.csv','pairs.csv'):shutil.copy2(artifact/name,directory/name)
    rule={'schema':'receiver.patient-dp-png-release/v1','epsilon':8.,'delta':1e-5,
        'adjacency':'add_remove_one_patient','private_signal_releases':1,'DP_applied':True,
        'training_data_protection':'Q given fixed public P, fixed source/condition recipe',
        'development_selection_protected':False,'images':128,'updates':200,'synthesis_seed':101,
        'source_encoders':['BioViL','DINOv2 ViT-B14'],'condition_count':4,'source_aggregation':'equal mean',
        'receiver_rule':'class-balanced point-only ridge0.1 with receiver public-only projection',
        'raw_Q_metadata_included':False,'raw_evaluation_predictions_included':False,
        'external_publication_performed':False}
    save(directory/'learning_rule.json',rule)
    payload={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    save(directory/'release_manifest.json',{'schema':'receiver.protected-png-files/v1','files_sha256':payload})
    require(len(list((directory/'images').glob('*.png')))==128,'Protected export count mismatch')


def run(root,resume=False):
    start=time.monotonic()
    try:
        before_deadline()
        binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed after binding')
        durations={}
        for arm,rid in BANKS.items():
            before_deadline();jobfile=binding['jobs'][arm]
            require(sha(jobfile)==binding['job_sha256'][arm],'Job changed')
            bank=root/'banks'/rid;resultfile=root/(arm+'_result.json')
            if not resultfile.exists():
                command=['-m','receiver_distillation.patient_dp_runtime','--job',jobfile,
                         '--output',str(bank),'--result',str(resultfile)]
                if bank.exists():command.append('--resume')
                durations[arm]=worker(root,arm+'_main',command)
            result=read(resultfile);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank; no evaluation')
            done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
            verify_artifact(bank/'artifact',spec,done['signature'])
            witness=read(read(jobfile)['paired_initialization']);initial=read(bank/'initialization.json')
            require(all(initial[k]==witness[k] for k in ('seed','condition_seed','fields','template_tensor_digest')),'Historical initialization mismatch')
            require(read(bank/'artifact/learning_contract.json')['DP_applied']==(arm=='DP8'),'Wrong artifact privacy label')
            state_update(root,'RUNNING_BANKS',last_completed_bank=arm)
        sealed={}
        for arm,rid in BANKS.items():
            bank=root/'banks'/rid
            sealed[rid]={'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')}
        if not(root/'two_bank_seal.json').exists():save(root/'two_bank_seal.json',{'created':now(),'banks':sealed})
        protected_png_package(root)
        before_deadline();state_update(root,'EVALUATING',development_evaluation_opened=True)
        if not(root/'evaluation/result.json').exists():
            durations['evaluation']=worker(root,'four_cell_evaluation',['-m','receiver_distillation.evaluate_patient_dp','--root',str(root)])
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen code changed')
        history=read(root/'campaign_contract.json')['historical_dependencies']
        require(all(sha(p)==h for p,h in history.items()),'Historical evidence changed')
        before_deadline();evaluation=read(root/'evaluation/result.json')
        result={'status':'COMPLETE_FIRST_DP_COMPARISON','finished':now(),'controller_seconds':time.monotonic()-start,
            'durations_this_controller':durations,'private_releases':1,'new_banks':2,'new_updates':400,
            'evaluation_result_sha256':sha(root/'evaluation/result.json'),'automatic_followup':False,
            'protected_package':str(root/'protected_png'),'free_bytes':shutil.disk_usage(CODE).free,
            'historical_code_inputs_and_results_preserved':True,'development_judgment':evaluation['development_judgment']}
        save(root/'campaign_result.json',result)
        atomic_json(root/'campaign_status.json',dict(result,phase='COMPLETE'),replace=True)
        state_update(root,'COMPLETE',result=str(root/'evaluation/result.json'),automatic_followup=False)
        print(json.dumps(result),flush=True)
    except BaseException:
        if root.exists():
            atomic_json(root/'campaign_status.json',{'status':'STOPPED_ERROR','at':now(),
                'seconds':time.monotonic()-start,'traceback':traceback.format_exc(),'no_automatic_noise_redraw':True},
                replace=(root/'campaign_status.json').exists())
            state_update(root,'STOPPED_ERROR',error_log=str(root/'campaign_status.json'))
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();run(a.root.resolve(),a.resume)


if __name__=='__main__':main()
