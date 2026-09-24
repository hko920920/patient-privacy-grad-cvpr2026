"""One fresh protected feature summary, one final200 bank, fixed evaluation."""
import argparse,copy,json,os,shutil,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.datasets import manifests
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from . import feature_control as f
from .feature_runtime import code_bindings,validate_job
from .patient_dp_dino_io import cached_paths
from .verify_feature_control import cpu_checks,check_population,CRITERIA
from .run_feasibility import worker
from .evaluate_feature_control import dependencies,PRIOR,RESNET

ROOT=CODE/'_reports/receiver_feature_dp8_20260924_v1'
REFERENCE=CODE/'_reports/receiver_dino_only_s200_20260923_v1/job.json'
START='2026-09-24T03:26:46+00:00'
DEADLINE='2026-09-24T05:26:46+00:00'
BANK='DINO_feature_mean_dp8_noise01_101'
RELEASE='DINO_FEATURE_DP8_release01'
HUMAN=RESEARCH/'RECEIVER_FEATURE_PATIENT_DP_CONTRACT_20260924.md'


def on_time():require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole package120-minute cap reached')


def state(status,**fields):
    path=RESEARCH/'research_state.json';s=read(path)
    s.setdefault('receiver_feature_patient_dp_control',{}).update(status=status,updated=now(),output_directory=str(ROOT),
        absolute_deadline_utc=DEADLINE,**fields)
    s.setdefault('active_execution',{}).update(status='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_feature_patient_dp_control',
        current_scope='One patient-DP feature-mean/L2 control; compare both frozen DINO gradient-DP banks',output_directory=str(ROOT))
    s.update(current_step=2,stage2_status='in_progress',updated_utc=now(),step2_active_task='Feature-mean patient-DP comparison: '+status)
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def ledger(status):
    prior=read(PRIOR/'privacy_release_ledger.json');entries=copy.deepcopy(prior['entries'])
    require(len(entries)==4,'Expected four historical protected releases')
    entries.append({'release_id':RELEASE,'status':status,'epsilon':8.,'delta':1e-5,
                    'protected_target':str(ROOT/'one_release/protected_target'),'protected_png':str(ROOT/'protected_png')})
    value={'schema':'receiver.same-Q-patient-DP-ledger/v1','created':now(),'entries':entries,
        'scope':'Five fixed protected summaries and their postprocessing, if jointly disclosed over the same Q patients',
        'excludes':'Historical nonDP development, raw-Q diagnostics, V predictions, utility reports and method-selection history',
        'new_release_authorized_count':1,'adjacency':'add_remove_one_patient',
        'per_release_budget':{'epsilon':8.,'delta':1e-5},
        'if_all_five_outputs_disclosed':{'method':'basic sequential composition','epsilon_upper_bound':40.,'delta_upper_bound':5e-5,'optimal_accounting_claimed':False},
        'same_summary_postprocessing_additional_cost':0,'external_publication_performed':False}
    atomic_json(ROOT/'privacy_release_ledger.json',value,replace=(ROOT/'privacy_release_ledger.json').exists())


def initialize():
    on_time();tick=time.monotonic();ROOT.mkdir(exist_ok=False);(ROOT/'records_before').mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:
        shutil.copy2(p,ROOT/'records_before'/p.name)
    state('PREPARING',new_private_releases=0)
    text='''# Patient-DP feature-mean control: frozen first comparison (2026-09-24)

One new DINO feature-mean/L2 bank against both existing DINO gradient/cosine DP banks.
This tests the complete signal, metric, class pooling and clipping recipe; not a single-factor causal effect or a full Dosser reproduction.

- Same DINO checkpoint, P-only PCA16/q95, four separate conditions, P/Q roles, patient/class visit means, seed101 public templates, 128 images, final200 updates, microbatch16, AdamW and resolution schedule.
- Feature vectors keep K4 separate. No old aug-mean target reuse. Heads are retained in provenance but do not enter a feature signal.
- P-only class-present patient full-K4 feature norm q95 (linear, floor1e-6) defines C0,C1 before Q is read.
- Clip patient/class feature vectors at Cc for P and Q; synthetic images are one-class virtual patients and use the identical map. This bounded P/synthetic rule differs from the gradient baseline and is part of the full recipe comparison.
- Query u=.5[clip(v0,C0)/C0,clip(v1,C1)/C1,present0,present1] is 130-dimensional. Norm<=1, add/remove-patient sum sensitivity1. Existing audited analytic Gaussian calibration uses epsilon8/delta1e-5. One OS-random draw, no saved coins or redraw.
- Decode nonnegative noisy class counts and project class sums into radius Cc*count; pool bounded P sums with protected Q sums using denominator max(1,Pcount+noisyQcount). Preserve class and condition axes.
- Loss=.25 sum_class,condition,coordinate ((synthetic_class_mean-target)/Cc)^2. This averages two half-squared normalized L2 values. No coordinate/condition averaging. Nominal range0..2, without claiming gradient-scale equivalence to cosine. Prior anchor.01 and TV.0001 occur once.
- Public changed-path checks use the existing loss2e-6 and gradient2e-7+5e-4*peak criteria. No GPU profile or coefficient search. CPU tests cover aggregation, sensitivity, invalid/empty counts and active/inactive/zero clipping derivatives.
- Final PNG only. Freeze the bank before scoring. Reuse cached BioViL/DenseNet/ResNet18 V features and the existing2000 patient bootstrap draws. DenseNet is primary, ResNet18 supportive and BioViL diagnostic; all are existing development readouts. No new receiver, Expert or Reserved.
- Compare feature minus EACH existing gradient-DP draw, never just the worse/better draw. Positive descriptive support requires positive DenseNet AUROC differences with both conditional CI lower bounds>0 and AP point nondecrease; otherwise report mixed/no advantage. A worse feature result without sufficient target optimization does not establish an intrinsic feature limitation.
- One feature noise, two historical gradient noises and fixed synthesis seed do not estimate population-wide noise/seed superiority. No equal-time claim. No automated repeats or changes based on utility.
- Planned total70-100min, hard total120min from03:26:46UTC; deadline05:26:46UTC. Bank worker cap3600sec. Historical DINO synthesis~35min is a reference, not a new speed measurement.
- Estimated main encoder work:200*128*4*2=204800 forward images and102400 backward images. Public verification96 forwards/64 backwards; final evaluation128 PNGs per existing readout. No new real-Q/V pixel forwards.
- One new protected summary; five summaries if jointly disclosed have basic bound(40,5e-5). This does not protect past nonDP development or evaluation reports. Nothing is externally published here.
'''
    require(not HUMAN.exists(),'Contract exists; do not overwrite prior declaration');HUMAN.write_text(text,encoding='utf-8')
    prior=read(PRIOR/'execution_bindings.json');pc=read(PRIOR/'campaign_contract.json')
    history={**pc['historical_dependencies'],**prior['implementation']}
    for p in (HUMAN,REFERENCE,PRIOR/'campaign_result.json',PRIOR/'evaluation/result.json',PRIOR/'privacy_release_ledger.json',RESNET/'result.json',RESNET/'contract.json'):
        history[str(p)]=sha(p)
    require(all(sha(p)==h for p,h in history.items()),'Historical source/evidence changed')
    old=read(REFERENCE);require(all(sha(p)==h for p,h in old['code_bindings'].items()),'Frozen DINO loop changed')
    item=cached_paths(REFERENCE)[f.ENCODER_ID]
    p,pm,pz=f.read_population(item,'P');limits,details=f.public_limits(p)
    save(ROOT/'public_calibration.json',{'created':now(),'population':'P_ONLY','bounds':list(limits.class_bounds),'details':details,
        'rule':'patient/class fullK4 feature norm q95 linear floor1e-6','P_features_sha256':sha(item['P']),
        'chosen_before_Q_read':True,'Q_V_used_for_selection':False})
    groups=manifests()
    for k,field in (('image_ids','image_id'),('patient_ids','patient_id'),('labels','label')):
        require([str(x) for x in pm[k]]==[str(r[field]) for r in groups['P']],'P cache/manifest differs')
    checks=cpu_checks();checks['public_population']=check_population(p,pm,pz,limits)
    save(ROOT/'array_verification.json',checks)
    ps,counts,_=f.bounded_sums(p,limits);public_target=ps/counts[:,None,None]
    mech=f.mechanism_metadata(limits)
    public_handoff=f.write_target(ROOT/'public_fixture',public_target,limits.class_bounds,item['rule'],mech,'PUBLIC_SIMULATION_ONLY')
    save(ROOT/'public_gradient_contract.json',{'created':now(),'criteria':CRITERIA,'pixel_role':'P','images':4,'conditions':4,
        'microbatches':[1,4],'updates':0,'public_handoff_sha256':sha(public_handoff),'loss_definition':f.LOSS_RULE})
    worker(ROOT,'public_feature_gradient',['-m','receiver_distillation.verify_feature_control','--root',str(ROOT),'--handoff',public_handoff])
    require(read(ROOT/'public_gradient_result.json')['status']=='PASS_ACTUAL_PUBLIC_FEATURE_GRADIENT','Public gradient gate')
    q,qm,qz=f.read_population(item,'Q')
    for k,field in (('image_ids','image_id'),('patient_ids','patient_id'),('labels','label')):
        require([str(x) for x in qm[k]]==[str(r[field]) for r in groups['Q']],'Q cache/manifest differs')
    qcheck=check_population(q,qm,qz,limits);save(ROOT/'Q_verification_INTERNAL_NONDP.json',qcheck)
    inputs={str(REFERENCE):sha(REFERENCE)}
    for key in ('P','Q','handoff','target','rule_file'):inputs[item[key]]=sha(item[key])
    for key in ('checkpoint','projection'):inputs[item['rule'][key]]=sha(item['rule'][key])
    c={'schema':'receiver.feature-patient-dp-control/v1','status':'AUTHORIZED_FROZEN','created':now(),
       'authorization':'User approved one feature-mean/L2 control, same DP budget and fixed synthesis/evaluation recipe',
       'scope_started_utc':START,'absolute_deadline_utc':DEADLINE,'maximum_package_seconds':7200,'estimated_minutes':[70,100],
       'bank_ids':{'DP8':BANK},'max_private_releases':1,'reference_job':str(REFERENCE),
       'historical_dependencies':history,'input_bindings':inputs,'mechanism':mech,'loss_definition':f.LOSS_RULE,
       'recipe':{k:old[k] for k in ('seed','images','updates','microbatch','activation_updates','optimizer')},
       'question':'Whole gradient/cosine/pooled-class recipe versus class-feature/L2/bounded-mean recipe',
       'description':str(HUMAN),'description_sha256':sha(HUMAN),'evaluation_receivers':['DenseNet','BioViL','ResNet18'],
       'evaluation_dependencies':dependencies(),'no_redraw_no_HPO_no_checkpoint_selection':True,'automatic_followup':False}
    save(ROOT/'campaign_contract.json',c);on_time()
    release=ROOT/'one_release';release.mkdir(exist_ok=False)
    with (release/'RELEASE_RESERVED.json').open('x',encoding='utf-8') as stream:
        json.dump({'created':now(),'release_id':RELEASE,'max_draws':1,'contract_sha256':sha(ROOT/'campaign_contract.json'),
                   'retry_randomization_forbidden':True},stream);stream.flush();os.fsync(stream.fileno())
    return finish_preparation(item,limits,ps,counts,q,c,old,tick,len(pm['labels']),len(qm['labels']))


def finish_preparation(item,limits,ps,counts,q,c,old,tick,pn,qn):
    release=ROOT/'one_release'
    ledger('RESERVED_NOT_REDRAWABLE');state('CREATING_ONE_PROTECTED_TARGET',new_private_releases=1)
    noisy,metadata=f.privatize_patients(q,limits)
    target,post=f.decode_target(noisy,limits,ps,counts)
    require(metadata==c['mechanism'],'Mechanism mismatch')
    handoff=f.write_target(release/'protected_target',target,limits.class_bounds,item['rule'],metadata,'PATIENT_DP_SINGLE_QUERY_INTERNAL')
    save(release/'postprocessing.json',post)
    save(release/'RELEASE_COMPLETE.json',{'release_id':RELEASE,'handoff_sha256':sha(handoff),'draw_count':1,'created':now()})
    del q,noisy;ledger('TARGET_COMPLETE')
    job=dict(old,schema='receiver.feature-patient-dp-s200-bank/v1',id=BANK,variant='DP8',source_set='DINO_FEATURE',
        second_handoff=handoff,second_handoff_sha256=sha(handoff),campaign_contract=str(ROOT/'campaign_contract.json'),
        campaign_contract_sha256=sha(ROOT/'campaign_contract.json'),absolute_deadline_utc=DEADLINE,maximum_seconds=3600,
        code_bindings=code_bindings())
    save(ROOT/'job.json',job);validate_job(job)
    for key,value in (('updates',199),('source_set','DINO')):
        bad=dict(job);bad[key]=value
        try:validate_job(bad)
        except RuntimeError:pass
        else:raise RuntimeError('Invalid feature job accepted')
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'job':str(ROOT/'job.json'),'job_sha256':sha(ROOT/'job.json'),'implementation':sources,
             'evaluation_dependencies':c['evaluation_dependencies'],'runtime':code_bindings()}
    save(ROOT/'execution_bindings.json',binding);(ROOT/'implementation_snapshot').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'implementation_snapshot'/p.name)
    save(ROOT/'preparation_result.json',{'status':'PASS_READY','seconds':time.monotonic()-tick,'query_dimension':130,
        'new_private_releases':1,'new_Q_V_pixel_forwards':0,'new_private_feature_extraction':0,
        'cached_P_images':pn,'cached_Q_images':qn,'public_gradient_check':True,'bad_jobs_rejected':2})
    return binding


def continue_before_first_draw():
    tick=time.monotonic();on_time();failure=read(ROOT/'pre_sampling_failure/campaign_status.json')
    require("TypeError: privatize_patients() missing 3 required keyword-only arguments" in failure['traceback'],
            'Recovery only valid for proven pre-function argument failure')
    require(not(ROOT/'one_release/protected_target').exists() and not(ROOT/'execution_bindings.json').exists(),
            'Already sampled/prepared; randomization retry forbidden')
    require(not(ROOT/'pre_sampling_recovery.json').exists(),'Recovery already attempted')
    c=read(ROOT/'campaign_contract.json');old=read(REFERENCE)
    require(all(sha(p)==h for p,h in {**c['historical_dependencies'],**c['input_bindings']}.items()),'Frozen inputs changed')
    item=cached_paths(REFERENCE)[f.ENCODER_ID];p,pm,_=f.read_population(item,'P')
    limits=f.Limits(tuple(read(ROOT/'public_calibration.json')['bounds']))
    require(f.mechanism_metadata(limits)==c['mechanism'],'Frozen mechanism changed')
    ps,counts,_=f.bounded_sums(p,limits);q,qm,_=f.read_population(item,'Q')
    # Exercise the repaired keyword-default forwarding with a constructed public sampler.
    class FixedRNG:
        def normalvariate(self,*args):return 0.
    class FakeRandom:
        SystemRandom=FixedRNG
    sampler=f.clone(f.privatize_patients,random=FakeRandom)
    got,meta=sampler(p,limits);rows,_=f.clipped_contributions(p,limits)
    require(np.array_equal(got,f.old.sum_patient_axis(rows)) and meta==c['mechanism'],'Sampler adapter default test')
    require(read(ROOT/'public_gradient_result.json')['status']=='PASS_ACTUAL_PUBLIC_FEATURE_GRADIENT','Public gradient missing')
    save(ROOT/'pre_sampling_recovery.json',{'created':now(),'prior_actual_private_draws':0,
        'change':'Preserve Python keyword-only defaults when adapting immutable functions',
        'constructed_sampler_test':'PASS','reuse_actual_public_gradient_check':True,
        'loss_clipping_calibration_tolerance_unchanged':True,'authorized_actual_draws_remaining':1})
    return finish_preparation(item,limits,ps,counts,q,c,old,tick,len(pm['labels']),len(qm['labels']))


def package():
    directory=ROOT/'protected_png';directory.mkdir(exist_ok=False);artifact=ROOT/'banks'/BANK/'artifact'
    shutil.copytree(artifact/'images',directory/'images')
    for name in ('images.csv','pairs.csv'):shutil.copy2(artifact/name,directory/name)
    save(directory/'learning_rule.json',{'schema':'receiver.patient-dp-png-release/v1','release_id':RELEASE,
        'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient','protected_query_count':1,'query_dimension':130,
        'DP_applied':True,'training_data_protection':'Q given fixed public P and recipe','development_selection_protected':False,
        'images':128,'updates':200,'synthesis_seed':101,'source_encoders':['DINOv2 ViT-B14'],'conditions':4,
        'objective':f.LOSS_RULE,'raw_Q_metadata_included':False,'evaluation_predictions_included':False,
        'external_publication_performed':False,'if_all_five_summaries_jointly_disclosed_basic_bound':{'epsilon':40.,'delta':5e-5}})
    files={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    save(directory/'release_manifest.json',{'files_sha256':files})
    require(len(list((directory/'images').glob('*.png')))==128,'PNG count')


def run(resume=False,pre_sampling=False):
    tick=time.monotonic()
    try:
        binding=(continue_before_first_draw() if pre_sampling else (read(ROOT/'execution_bindings.json') if resume else initialize()))
        on_time();require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Frozen job changed')
        bank=ROOT/'banks'/BANK;file=ROOT/'DP8_result.json'
        if not file.exists():
            state('RUNNING_BANK',new_private_releases=1)
            cmd=['-m','receiver_distillation.feature_runtime','--job',binding['job'],'--output',str(bank),'--result',str(file)]
            if bank.exists():cmd.append('--resume')
            worker(ROOT,'feature_DP8_main',cmd)
        result=read(file);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Bank incomplete; no automatic extension')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json');verify_artifact(bank/'artifact',spec,done['signature'])
        require(read(bank/'initialization_parity.json')['status']=='PASS_EXACT_A2_INITIALIZATION','Initial parity differs')
        require(read(bank/'artifact/learning_contract.json')['DP_applied'] is True,'Privacy metadata')
        if not(ROOT/'bank_seal.json').exists():save(ROOT/'bank_seal.json',{'created':now(),'completion_sha256':sha(bank/'COMPLETED.json'),
            'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        if not(ROOT/'protected_png').exists():package()
        on_time();state('EVALUATING',development_evaluation_opened=True)
        if not(ROOT/'evaluation/result.json').exists():worker(ROOT,'feature_evaluation',['-m','receiver_distillation.evaluate_feature_control','--root',str(ROOT)])
        c=read(ROOT/'campaign_contract.json')
        require(all(sha(p)==h for p,h in {**binding['implementation'],**c['historical_dependencies'],**c['input_bindings']}.items()),'Frozen code/evidence changed')
        on_time();ledger('COMPLETE')
        final={'status':'COMPLETE_FEATURE_PATIENT_DP_CONTROL','finished':now(),'controller_seconds':time.monotonic()-tick,
            'new_private_releases':1,'new_banks':1,'new_updates':200,'evaluation_sha256':sha(ROOT/'evaluation/result.json'),
            'historical_evidence_unchanged':True,'automatic_followup':False,'free_bytes':shutil.disk_usage(CODE).free}
        save(ROOT/'campaign_result.json',final);atomic_json(ROOT/'campaign_status.json',dict(final,phase='COMPLETE'),replace=True)
        state('COMPLETE',result=str(ROOT/'evaluation/result.json'),automatic_followup=False);print(json.dumps(final),flush=True)
    except BaseException:
        if ROOT.exists():
            atomic_json(ROOT/'campaign_status.json',{'status':'STOPPED_ERROR','at':now(),'traceback':traceback.format_exc(),
                'no_automatic_noise_redraw':True},replace=(ROOT/'campaign_status.json').exists())
            state('STOPPED_ERROR',automatic_followup=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--continue-before-first-draw',action='store_true')
    a=p.parse_args();run(a.resume,a.continue_before_first_draw)
