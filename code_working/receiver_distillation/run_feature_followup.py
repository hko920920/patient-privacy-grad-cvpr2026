"""Exactly one public-only bank and one new feature-DP noise bank."""
import argparse,copy,json,os,shutil,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from . import feature_control as f
from .feature_followup_runtime import code_bindings,validate_job,load_feature
from .patient_dp_dino_io import cached_paths
from .run_feasibility import worker
from .evaluate_feature_followup import dependencies,PREVIOUS
ROOT=CODE/'_reports/receiver_feature_followup_20260924_v1'
START='2026-09-24T08:30:40+00:00'
DEADLINE='2026-09-24T10:30:40+00:00'
RELEASE='DINO_FEATURE_DP8_release02'
BANKS={'PUBLIC':'DINO_feature_mean_public_101','DP8':'DINO_feature_mean_dp8_noise02_101'}
HUMAN=RESEARCH/'RECEIVER_FEATURE_FOLLOWUP_CONTRACT_20260924.md'

def on_time():
    require(time.time()<datetime.fromisoformat(DEADLINE).timestamp(),'Whole120-minute cap reached')
    require(shutil.disk_usage(CODE).free>1024**3,'Less than1GiB free: safe storage stop')

def state(status,**fields):
    path=RESEARCH/'research_state.json';s=read(path)
    s.setdefault('receiver_feature_followup',{}).update(status=status,updated=now(),output_directory=str(ROOT),
        absolute_deadline_utc=DEADLINE,**fields)
    s.setdefault('active_execution',{}).update(status='no_running_execution' if status in ('COMPLETE','STOPPED_ERROR') else 'running_feature_followup',
        current_scope='Feature public-only and one independent DP2, seed101, two final200 banks',output_directory=str(ROOT))
    s.update(current_step=2,stage2_status='in_progress',updated_utc=now(),step2_active_task='Feature public/noise follow-up: '+status)
    path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def ledger(status):
    value=copy.deepcopy(read(PREVIOUS/'privacy_release_ledger.json'))
    require(len(value['entries'])==5,'Expected five preserved releases')
    value.update(created=now(),scope='Six protected summaries if jointly disclosed over the same Q patients')
    value['entries'].append({'release_id':RELEASE,'status':status,'epsilon':8.,'delta':1e-5,
        'protected_target':str(ROOT/'one_release/protected_target'),'protected_png':str(ROOT/'protected_png')})
    value.pop('if_all_five_outputs_disclosed',None)
    value['if_all_six_outputs_disclosed']={'method':'basic sequential composition','epsilon_upper_bound':48.,
        'delta_upper_bound':6e-5,'optimal_accounting_claimed':False}
    value['public_only_bank_new_private_releases']=0
    value['new_release_authorized_count']=1
    atomic_json(ROOT/'privacy_release_ledger.json',value,replace=(ROOT/'privacy_release_ledger.json').exists())

def prepare():
    on_time();tick=time.monotonic();ROOT.mkdir(exist_ok=False);(ROOT/'records_before').mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:
        shutil.copy2(p,ROOT/'records_before'/p.name)
    state('PREPARING',new_private_releases=0)
    pc=read(PREVIOUS/'campaign_contract.json');pb=read(PREVIOUS/'execution_bindings.json')
    require(read(PREVIOUS/'campaign_result.json')['status']=='COMPLETE_FEATURE_PATIENT_DP_CONTROL','Prior feature incomplete')
    history={**pc['historical_dependencies'],**pc['input_bindings'],**pb['implementation']}
    for n in ('campaign_result.json','campaign_contract.json','execution_bindings.json','job.json','public_calibration.json',
              'public_gradient_result.json','array_verification.json','evaluation/result.json','bank_seal.json',
              'privacy_release_ledger.json','protected_png/release_manifest.json'):
        history[str(PREVIOUS/n)]=sha(PREVIOUS/n)
    require(all(sha(p)==h for p,h in history.items()),'Frozen evidence changed')
    reference=PREVIOUS/'job.json';old=read(reference)
    item=cached_paths(Path(pc['reference_job']))[f.ENCODER_ID]
    limits=f.Limits(tuple(read(PREVIOUS/'public_calibration.json')['bounds']))
    mechanism=f.mechanism_metadata(limits);require(mechanism==pc['mechanism'],'Mechanism changed')
    require(read(PREVIOUS/'public_gradient_result.json')['status']=='PASS_ACTUAL_PUBLIC_FEATURE_GRADIENT','Missing inherited image-gradient verification')
    c={'schema':'receiver.feature-public-and-noise-followup/v1','status':'AUTHORIZED_FROZEN','created':now(),
       'authorization':'User explicitly said execute after bounded feature-public plus independent DP2 decision',
       'scope_started_utc':START,'absolute_deadline_utc':DEADLINE,'maximum_package_seconds':7200,
       'estimated_minutes':[90,110],'maximum_seconds_per_bank':3600,'max_banks':2,'max_private_releases':1,
       'bank_ids':BANKS,'reference_job':str(reference),'historical_dependencies':history,'mechanism':mechanism,
       'loss_definition':f.LOSS_RULE,'description':str(HUMAN),'description_sha256':sha(HUMAN),
       'recipe':{k:old[k] for k in ('seed','images','updates','microbatch','activation_updates','optimizer')},
       'evaluation_dependencies':dependencies(),'receivers':['DenseNet','ResNet18'],'bootstrap_repeats':2000,
       'questions':['Feature DP versus same-recipe public-only Q increment','Independent second feature DP noise observation'],
       'decision':'Report all points and conditional intervals, no equivalence or automatic contribution claim',
       'no_redraw_no_HPO_no_checkpoint_selection':True,'automatic_followup':False}
    save(ROOT/'campaign_contract.json',c)
    p,pm,_=f.read_population(item,'P');ps,counts,_=f.bounded_sums(p,limits)
    require(np.array_equal(counts,[669.,6.]),'Public class denominator differs')
    public_target=ps/counts[:,None,None]
    with np.load(PREVIOUS/'public_fixture/target.npz',allow_pickle=False) as d:
        require(np.array_equal(public_target,d['target']),'P-only target differs from verified original fixture')
    public_meta={'schema':f.SCHEMA,'private_queries':0,'Q_used':False,'class_bounds':list(limits.class_bounds),
                 'population':'P','DP_applied':False,'signal_dimension':128}
    ph=f.write_target(ROOT/'public_target',public_target,limits.class_bounds,item['rule'],public_meta,'PUBLIC_ONLY_NO_Q')
    handoff=read(ph);handoff['population']='P';atomic_json(ph,handoff,replace=True)
    labels=(0,)*64+(1,)*64
    direct=f.objective(public_target,limits.class_bounds,labels)
    loaded,_=load_feature(ph,labels,'P')
    generator=torch.Generator().manual_seed(7351)
    z=torch.randn(4,128,16,dtype=torch.float64,generator=generator).requires_grad_(True)
    a=direct.matching({f.ENCODER_ID:z})[0];ga=torch.autograd.grad(a,z)[0]
    b=loaded.matching({f.ENCODER_ID:z})[0];gb=torch.autograd.grad(b,z)[0]
    require(torch.equal(a,b) and torch.equal(ga,gb),'Public target adapter changes loss/gradient')
    save(ROOT/'public_target_verification.json',{'status':'PASS_UNCHANGED_PUBLIC_TARGET_ADAPTER','counts':counts.tolist(),
        'matches_existing_verified_fixture_exactly':True,'loss_exact':True,'feature_gradient_exact':True,
        'Q_used_in_public_target':False,'new_GPU_gradient_checks':0,'image_gradient_evidence_reused':str(PREVIOUS/'public_gradient_result.json')})
    on_time();q,qm,_=f.read_population(item,'Q')
    # Reserve before drawing. A surviving reservation can never trigger another draw.
    release=ROOT/'one_release';release.mkdir(exist_ok=False)
    save(release/'RELEASE_RESERVED.json',{'created':now(),'release_id':RELEASE,'max_draws':1,
        'contract_sha256':sha(ROOT/'campaign_contract.json'),'retry_randomization_forbidden':True})
    ledger('RESERVED_NOT_REDRAWABLE');state('CREATING_ONE_PROTECTED_TARGET',new_private_releases=1)
    noisy,metadata=f.privatize_patients(q,limits)
    # Persist only the protected query, never random coins or the unnoised sum.
    temp=release/'protected_query.tmp'
    with temp.open('xb') as stream:
        np.savez(stream,noisy_query=noisy);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,release/'protected_query.npz')
    require(metadata==mechanism,'Noise mechanism changed')
    target,post=f.decode_target(noisy,limits,ps,counts)
    dh=f.write_target(release/'protected_target',target,limits.class_bounds,item['rule'],metadata,'PATIENT_DP_SINGLE_QUERY_INTERNAL')
    save(release/'postprocessing.json',post)
    save(release/'RELEASE_COMPLETE.json',{'created':now(),'release_id':RELEASE,'draw_count':1,
        'protected_query_sha256':sha(release/'protected_query.npz'),'handoff_sha256':sha(dh)})
    del q,noisy;ledger('TARGET_COMPLETE')
    jobs={}
    for variant,h in (('PUBLIC',ph),('DP8',dh)):
        job=dict(old,schema='receiver.feature-followup-s200-bank/v1',id=BANKS[variant],variant=variant,
            population='P' if variant=='PUBLIC' else 'pooled',second_handoff=h,second_handoff_sha256=sha(h),
            campaign_contract=str(ROOT/'campaign_contract.json'),campaign_contract_sha256=sha(ROOT/'campaign_contract.json'),
            absolute_deadline_utc=DEADLINE,maximum_seconds=3600,code_bindings=code_bindings())
        path=ROOT/(variant+'_job.json');save(path,job);validate_job(job);jobs[variant]=str(path)
        # Only changed target/role wiring is tested; frozen full suites are reused.
        for key,value in (('population','pooled' if variant=='PUBLIC' else 'P'),('updates',199),('id','unauthorized')):
            bad=dict(job);bad[key]=value
            try:validate_job(bad)
            except RuntimeError:pass
            else:raise RuntimeError('Invalid follow-up job accepted')
        adapted,_=load_feature(h,labels,job['population'])
        expected,_=f.load_feature(h,labels,'pooled')
        require(torch.equal(adapted.targets[f.ENCODER_ID],expected.targets[f.ENCODER_ID]),'DP/public loader parity')
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    bindings={'created':now(),'jobs':jobs,'job_sha256':{k:sha(v) for k,v in jobs.items()},
        'implementation':sources,'evaluation_dependencies':c['evaluation_dependencies'],'runtime':code_bindings()}
    save(ROOT/'execution_bindings.json',bindings);(ROOT/'implementation_snapshot').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'implementation_snapshot'/p.name)
    save(ROOT/'preparation_result.json',{'status':'PASS_READY_TWO_BANKS','seconds':time.monotonic()-tick,
        'new_private_releases':1,'new_real_feature_forwards':0,'cached_P_images':len(pm['labels']),
        'cached_Q_images':len(qm['labels']),'bad_jobs_rejected':6,'old_image_gradient_check_reused':True})
    return bindings

def package(variant):
    dp=variant=='DP8';folder=ROOT/('protected_png' if dp else 'public_png')
    if folder.exists():
        m=read(folder/'release_manifest.json')
        require(all(sha(folder/p)==h for p,h in m['files_sha256'].items()),'Existing package changed');return
    folder.mkdir();artifact=ROOT/'banks'/BANKS[variant]/'artifact'
    shutil.copytree(artifact/'images',folder/'images')
    for n in ('images.csv','pairs.csv'):shutil.copy2(artifact/n,folder/n)
    save(folder/'learning_rule.json',{'schema':'receiver.feature-PNG-release/v1','release_id':RELEASE if dp else None,
        'DP_applied':dp,'epsilon':8. if dp else None,'delta':1e-5 if dp else None,
        'adjacency':'add_remove_one_patient' if dp else None,'protected_query_count':int(dp),'query_dimension':130 if dp else 0,
        'training_data_protection':'Q given fixed public P and recipe' if dp else 'No Q data used',
        'development_selection_protected':False,'images':128,'updates':200,'synthesis_seed':101,
        'source_encoders':['DINOv2 ViT-B14'],'conditions':4,'objective':f.LOSS_RULE,
        'raw_Q_metadata_included':False,'evaluation_predictions_included':False,'external_publication_performed':False,
        'if_all_six_summaries_jointly_disclosed_basic_bound':{'epsilon':48.,'delta':6e-5} if dp else None})
    save(folder/'release_manifest.json',{'files_sha256':{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}})
    require(len(list((folder/'images').glob('*.png')))==128,'PNG count')

def run(resume=False):
    tick=time.monotonic()
    try:
        binding=read(ROOT/'execution_bindings.json') if resume else prepare()
        on_time();require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        seals={}
        for variant in ('PUBLIC','DP8'):
            on_time();job=binding['jobs'][variant]
            require(sha(job)==binding['job_sha256'][variant],'Bound job changed')
            bank=ROOT/'banks'/BANKS[variant];result_path=ROOT/(variant+'_result.json')
            if not result_path.exists():
                state('RUNNING_'+variant,new_private_releases=1)
                args=['-m','receiver_distillation.feature_followup_runtime','--job',job,'--output',str(bank),'--result',str(result_path)]
                if bank.exists():args.append('--resume')
                worker(ROOT,'feature_'+variant+'_main',args)
            r=read(result_path);require(r['status']=='COMPLETE' and r['completed_updates']==200,'Bank incomplete; no extension')
            done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
            verify_artifact(bank/'artifact',spec,done['signature'])
            require(read(bank/'initialization_parity.json')['status']=='PASS_EXACT_A2_INITIALIZATION','Initial state mismatch')
            require(read(bank/'artifact/learning_contract.json')['DP_applied']==(variant=='DP8'),'Incorrect privacy metadata')
            seals[variant]={'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')}
            package(variant)
        if not(ROOT/'both_banks_sealed.json').exists():save(ROOT/'both_banks_sealed.json',dict(seals,created=now()))
        state('EVALUATING',development_evaluation_opened=True)
        if not(ROOT/'evaluation/result.json').exists():
            worker(ROOT,'feature_followup_evaluation',['-m','receiver_distillation.evaluate_feature_followup','--root',str(ROOT)])
        c=read(ROOT/'campaign_contract.json')
        require(all(sha(p)==h for p,h in {**c['historical_dependencies'],**binding['implementation']}.items()),'Frozen evidence/code changed')
        on_time();ledger('COMPLETE')
        result={'status':'COMPLETE_FEATURE_PUBLIC_AND_NOISE_REPEAT','finished':now(),'controller_seconds':time.monotonic()-tick,
            'new_private_releases':1,'new_banks':2,'new_updates':400,'evaluation_sha256':sha(ROOT/'evaluation/result.json'),
            'historical_evidence_unchanged':True,'automatic_followup':False,'free_bytes':shutil.disk_usage(CODE).free}
        save(ROOT/'campaign_result.json',result)
        atomic_json(ROOT/'campaign_status.json',dict(result,phase='COMPLETE'),replace=True)
        state('COMPLETE',result=str(ROOT/'evaluation/result.json'),automatic_followup=False)
        print(json.dumps(result),flush=True)
    except BaseException:
        if ROOT.exists():
            atomic_json(ROOT/'campaign_status.json',{'status':'STOPPED_ERROR','at':now(),'traceback':traceback.format_exc(),
                'no_automatic_noise_redraw':True},replace=(ROOT/'campaign_status.json').exists())
            state('STOPPED_ERROR',automatic_followup=False)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.resume)

