"""Authorized serial A1/A2 campaign: one cost measurement, two banks, one evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from prrd_v3.contracts import CODE, OUT, read, save, sha, require, now
from prrd_v3.bank_runtime import atomic_json, verify_artifact
from .runtime import OPTIMIZER
from .schedule import ACTIVATIONS
from .feasibility_runtime import code_bindings
from .evaluate_feasibility import dependencies


def worker(root,name,arguments):
    logs=root/'logs'; logs.mkdir(exist_ok=True)
    logfile=logs/(name+'_'+str(time.time_ns())+'.log')
    started=time.monotonic()
    command=[sys.executable,'-B','-X','utf8']+arguments
    print(json.dumps({'phase':'launch','worker':name,'at':now()}),flush=True)
    with logfile.open('x',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',bufsize=1,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atomic_json(root/'campaign_status.json',{'phase':name,'status':'RUNNING','pid':proc.pid,
                    'updated':now(),'log':str(logfile)},replace=(root/'campaign_status.json').exists())
        for line in proc.stdout:
            log.write(line); log.flush()
            print(line,end='',flush=True)
        code=proc.wait()
    require(code==0,'Worker failed: '+name+'; evidence preserved in '+str(logfile))
    elapsed=time.monotonic()-started
    print(json.dumps({'phase':'worker_complete','worker':name,'seconds':elapsed}),flush=True)
    return elapsed


def bind(root):
    require(read(root/'changed_path_verification/result.json')['status']=='PASS_ACTUAL_PUBLIC_TWO_SOURCE_GRADIENTS',
            'Actual two-source gradient check missing')
    require(read(root/'schedule_verification.json')['status']=='PASS','Schedule/restore check missing')
    second=root/'second_targets/private/target_handoff.json'
    first=CODE/'_reports/receiver_a1_targets_runtime_20260922_v1/targets/private/target_handoff.json'
    contract=root/'campaign_contract.json'; campaign=read(contract)
    require(sha(first)==campaign['A1_handoff_sha256'],'A1 target changed')
    require(read(root/'second_targets/result.json')['status']=='PASS_DINO_P_ONLY_PROJECTION_PQ_CONDITION_TARGETS',
            'Second source target incomplete')
    templates=read(CODE/'_reports/receiver_a1_targets_runtime_20260922_v1/runtime_attempt1/main_proposal_job.json')['templates']
    historical=[{'image_id':r['image_id'],'sha256':r['sha256']} for r in read(OUT/'templates_P_private.json')['101']]
    require(templates==historical,'Public template recipe changed')
    jobs={}
    folder=root/'jobs'; folder.mkdir()
    for arm in ('A1','A2','profile_A2'):
        profile=arm=='profile_A2'; two=arm!='A1'
        rid='profile_A2_public128' if profile else campaign['budget']['bank_ids'][int(two)]
        job={'schema':'receiver.matched-s200-bank/v1','id':rid,
             'artifact_kind':'COST_ONLY' if profile else 'MAIN_BANK',
             'seed':101,'images':128,'updates':4 if profile else 200,'microbatch':16,
             'population':'P' if profile else 'pooled','activation_updates':list(ACTIVATIONS),
             'optimizer':OPTIMIZER,'a1_handoff':str(first),'a1_handoff_sha256':sha(first),
             'second_handoff':str(second) if two else None,
             'second_handoff_sha256':sha(second) if two else None,
             'campaign_contract':str(contract),'campaign_contract_sha256':sha(contract),
             'maximum_seconds':900 if profile else (7200 if two else 3600),
             'templates':templates,'code_bindings':code_bindings()}
        path=folder/(arm+'.json'); save(path,job); jobs[arm]=str(path)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    bindings={'created':now(),'jobs':jobs,'job_sha256':{k:sha(v) for k,v in jobs.items()},
              'implementation':sources,'evaluation_dependencies':dependencies(),
              'campaign_contract_sha256':sha(contract),'runtime':code_bindings()}
    save(root/'execution_bindings.json',bindings)
    snapshot=root/'implementation_snapshot'; snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copy2(p,snapshot/p.name)
    return bindings


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    root=args.root.resolve(); started=time.monotonic()
    try:
        binding=read(root/'execution_bindings.json') if args.resume else bind(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Campaign implementation changed')
        durations={}
        profile_result=root/'profile_A2_result.json'
        if not profile_result.exists():
            durations['profile_A2']=worker(root,'profile_A2',[
                '-m','receiver_distillation.feasibility_runtime','--job',binding['jobs']['profile_A2'],
                '--output',str(root/'profile_A2'),'--result',str(profile_result)])
        profile=read(profile_result)
        require(profile['status']=='COST_ONLY_COMPLETE','A2 cost measurement incomplete')
        measured=profile['duration_seconds'][1:]
        require(len(measured)==3,'Not the declared warmup1/measure3 profile')
        estimate=sum(measured)/len(measured)*200
        print(json.dumps({'phase':'A2_measured_cost','seconds_per_update':sum(measured)/3,
                          'synthesis_200_estimated_seconds':estimate,
                          'peak_allocated_MiB':profile['peak_allocated_MiB']}),flush=True)
        require(estimate+180<7200,'A2 estimate does not fit authorized time cap')
        if not(root/'cost_estimate.json').exists():
            save(root/'cost_estimate.json',{'A2_measured_full128_seconds':measured,
                 'A2_estimated_synthesis_seconds':estimate,'A2_profile_worker_seconds':profile['worker_seconds'],
                 'A2_cap_seconds':7200,'A1_historical_estimated_synthesis_seconds':2356.262566661462,
                 'estimate_is_not_guaranteed_total':True})
        for arm in ('A1','A2'):
            job=read(binding['jobs'][arm]); bank=root/'banks'/job['id']; result=root/(arm+'_result.json')
            if (bank/'COMPLETED.json').exists():
                require(result.exists() and read(result)['status']=='COMPLETE','Completed bank lacks receipt')
                verify_artifact(bank/'artifact',read(bank/'run_spec.json'),read(bank/'COMPLETED.json')['signature'])
                continue
            command=['-m','receiver_distillation.feasibility_runtime','--job',binding['jobs'][arm],
                     '--output',str(bank),'--result',str(result)]
            if bank.exists(): command.append('--resume')
            durations[arm]=worker(root,arm,command)
            require(read(result)['status']=='COMPLETE','Bank stopped before200; no evaluation or automatic extension')
        sealed={}
        for arm in ('A1','A2'):
            job=read(binding['jobs'][arm]); bank=root/'banks'/job['id']
            done=read(bank/'COMPLETED.json')
            verify_artifact(bank/'artifact',read(bank/'run_spec.json'),done['signature'])
            sealed[job['id']]={'completion_sha256':sha(bank/'COMPLETED.json'),
                              'artifact_sha256':sha(bank/'artifact/artifact_seal.json')}
        if not(root/'two_bank_seal.json').exists():
            save(root/'two_bank_seal.json',{'at':now(),'campaign_contract_sha256':binding['campaign_contract_sha256'],
                                          'banks':sealed})
        if not(root/'evaluation/result.json').exists():
            durations['evaluation']=worker(root,'fixed_evaluation',[
                '-m','receiver_distillation.evaluate_feasibility','--root',str(root)])
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Frozen code changed')
        result={'status':'COMPLETE_FIRST_A1_A2_S200_COMPARISON','finished':now(),
                'durations_this_controller':durations,'controller_seconds':time.monotonic()-started,
                'evaluation_result_sha256':sha(root/'evaluation/result.json'),
                'free_bytes':shutil.disk_usage(CODE).free,'automatic_followup':False}
        save(root/'campaign_result.json',result)
        atomic_json(root/'campaign_status.json',dict(result,phase='COMPLETE'),replace=True)
        print(json.dumps(result),flush=True)
    except BaseException:
        atomic_json(root/'campaign_status.json',{'status':'STOPPED_ERROR','at':now(),
                    'seconds':time.monotonic()-started,'traceback':traceback.format_exc()},replace=True)
        raise


if __name__=='__main__':main()
