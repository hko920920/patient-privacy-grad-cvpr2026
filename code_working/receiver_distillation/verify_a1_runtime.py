"""Bounded continuation: CPU handoff checks, P-only restart and full128 cost."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from prrd_v3.contracts import CODE, RESEARCH, read, require, save, sha, now
from prrd_v3.datasets import manifests, public_templates
from prrd_v3.bank_runtime import digest, tree_digest, checkpoint_record, verify_artifact
from .runtime import OPTIMIZER, code_bindings


def worker(root,name,arguments,timeout=600):
    command = [sys.executable,'-B','-X','utf8']+arguments
    print(json.dumps({'phase':'launch','worker':name}),flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command,cwd=CODE,capture_output=True,text=True,encoding='utf-8',
                               timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (root/(name+'_stdout.txt')).write_text(completed.stdout,encoding='utf-8')
    (root/(name+'_stderr.txt')).write_text(completed.stderr,encoding='utf-8')
    require(completed.returncode == 0,'Worker failed: '+name+'; logs preserved')
    print(json.dumps({'phase':'exited','worker':name,'seconds':time.perf_counter()-started}),flush=True)
    return time.perf_counter()-started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--handoff',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    require(not args.output.exists(),'Preserve earlier runtime attempt')
    args.output.mkdir(parents=True)
    root = args.output.resolve()
    handoff = args.handoff.resolve()
    binding = read(handoff)
    require(read(handoff.parent.parent/'result.json')['status'] == 'PASS_A1_P_Q_CONDITION_TARGETS_READY',
            'Targets not ready')
    groups = manifests()
    templates = public_templates(groups['P'],101)
    previous = read(CODE/'_reports/receiver_condition_step1_20260922_v1/public_attempt1/public_contract.json')
    small = previous['public_fixture']
    code = code_bindings()
    jobs = {}
    for name,kind,n,steps,pop,micro in (
        ('technical','TECHNICAL_TEST_ONLY',4,3,'P',4),
        ('cost','COST_ONLY',128,4,'P',16),
        ('main_proposal','MAIN_BANK',128,500,'pooled',16)):
        selected = small if n == 4 else templates
        job = {
            'schema':'receiver.A1-bank-job/v1','id':name+'_A1_condition_k4_101',
            'encoder_id':'biovil','artifact_kind':kind,'seed':101,'images':n,'updates':steps,
            'microbatch':micro,'population':pop,'optimizer':OPTIMIZER,
            'handoff':str(handoff),'handoff_sha256':sha(handoff),'rule_sha256':binding['rule_sha256'],
            'code_bindings':code,
            'templates':[{'image_id':r['image_id'],'sha256':r['sha256']} for r in selected],
            'Q_V_pixels':False,'V_receiver_evaluation':False,'DP_expert_reserved':False,
        }
        jobs[name] = job
        save(root/(name+'_job.json'),job)
    cfg = {
        'scope':'A1_PUBLIC_CHANGED_PATH_RESUME_AND_FULL128_COST_ONLY',
        'created':now(),'contract':str(RESEARCH/'RECEIVER_A1_TARGET_RUNTIME_CONTRACT_20260922.md'),
        'contract_sha256':sha(RESEARCH/'RECEIVER_A1_TARGET_RUNTIME_CONTRACT_20260922.md'),
        'target_handoff_sha256':sha(handoff),'code':code,
        'continuous_updates':3,'split_after':2,'total_small_updates':6,
        'full128_warmup':1,'full128_measured':3,
        'expected_source_F':4288,'expected_source_B':2144,'total_optimizer_updates':10,
        'criteria':{'checkpoint_state_digest_exact':True,'loss_atol':2e-6,
                    'PNG_label_hash_exact':True,'fixed_encoder_target':True},
        'main_proposal_not_executed':True,
    }
    save(root/'contract.json',cfg)
    snapshot = root/'implementation_snapshot'
    snapshot.mkdir()
    for name in ('runtime.py','target_io.py','verify_a1_runtime.py','test_target_runtime.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    started = time.perf_counter()
    worker(root,'cpu',['-m','receiver_distillation.test_target_runtime','--handoff',str(handoff),
                      '--output',str(root/'cpu_result.json')])
    require(read(root/'cpu_result.json')['status'] == 'PASS','CPU check failed')
    results = []
    for name in ('continuous','prefix','resume','cost'):
        cost = name == 'cost'
        folder = name if name in ('continuous','cost') else 'split'
        command = ['-m','receiver_distillation.runtime','--job',str(root/('cost_job.json' if cost else 'technical_job.json')),
                   '--output',str(root/folder),'--result',str(root/(name+'_result.json'))]
        if name == 'prefix':
            command += ['--stop-after','2']
        if name == 'resume':
            command += ['--resume','--perturb-rng']
        worker(root,name,command)
        result = read(root/(name+'_result.json'))
        results.append(result)
        if name == 'prefix':
            require(result['status'] == 'PAUSED' and result['next_step'] == 3,'Wrong prefix boundary')
            require(not (root/'split/COMPLETED.json').exists() and not (root/'split/artifact').exists(),
                    'Intermediate checkpoint exported as final')
        if name == 'resume':
            require(result['restoration']['restored_exact'] and result['restoration']['next_step'] == 3,
                    'Checkpoint restoration failed')
    records, payloads = [], []
    for name in ('continuous','split'):
        record,path = checkpoint_record(root/name)
        records.append(record)
        payloads.append(torch.load(path,map_location='cpu',weights_only=True))
    require(tree_digest(payloads[0]) == tree_digest(payloads[1]),'Continuous/restarted whole state differs')
    require(len({r['pid'] for r in results[:3]}) == 3,'Restart did not use separate processes')
    logs = [[json.loads(line) for line in (root/name/'updates.jsonl').read_text().splitlines()]
            for name in ('continuous','split')]
    require(all([r['successful_step'] for r in rows] == [1,2,3] for rows in logs),'Skipped/repeated update')
    loss_error = max(abs(a['trace']['loss']-b['trace']['loss']) for a,b in zip(*logs))
    require(loss_error <= 2e-6,'Restart loss differs')
    png_checks = []
    for name,payload in zip(('continuous','split'),payloads):
        spec = read(root/name/'run_spec.json')
        png_checks.append(verify_artifact(root/name/'artifact',spec,payload['signature']))
    for name in ('images.csv','pairs.csv','learning_contract.json'):
        require((root/'continuous/artifact'/name).read_bytes() == (root/'split/artifact'/name).read_bytes(),
                'Final image labels/hash/learning contract differs')
    cost = results[-1]
    measured = cost['duration_seconds'][1:]
    require(len(measured) == 3 and all(x > 0 and np.isfinite(x) for x in measured),'Invalid timing')
    estimate = float(np.mean(measured))*500
    cost_report = {
        'full128_update_seconds':measured,'mean_seconds':float(np.mean(measured)),
        'min_seconds':min(measured),'max_seconds':max(measured),'warmup_seconds':cost['duration_seconds'][0],
        'peak_allocated_MiB':cost['peak_allocated_MiB'],'peak_reserved_MiB':cost['peak_reserved_MiB'],
        '500_update_synthesis_only_estimate_seconds':estimate,
        'planning_range_seconds':[estimate*.8,estimate*1.25],
        'planning_range_is_not_statistical_CI':True,
        'includes':'full128 renderer, K4 source forward, global signal loss, replay backward, optimizer, finite checks',
        'excludes':'P/Q preparation, model startup, periodic checkpoint IO, PNG export, receiver evaluation',
        'pyramid':'all six levels for every measured update; actual schedule starts with one',
        'no_main_bank_created':True,
    }
    save(root/'cost_report.json',cost_report)
    total = {k:sum(r['counts'][k] for r in results) for k in ('forward_images','backward_images')}
    require(total == {'forward_images':4288,'backward_images':2144},'Technical model totals changed')
    require(sum(r['updates_this_process'] for r in results) == 10,'Technical update count changed')
    require(all(sha(p) == h for p,h in code.items()),'Code changed during runtime verification')
    summary = {
        'status':'PASS_A1_TARGET_RUNTIME_CONNECTION_AND_PUBLIC_COST_ONLY',
        'whole_state_digest_exact':True,'loss_max_abs':loss_error,
        'resumed_next_step':results[2]['restoration']['next_step'],
        'PNG_checks':png_checks,'counts':total,'technical_optimizer_updates':10,
        'Q_V_pixels':0,'recipient_A2_DP_expert_reserved':False,
        'public_unique_templates_technical':4,'public_unique_templates_cost':64,
        'cost':cost_report,'elapsed_seconds':time.perf_counter()-started,
        'main_job_digest':digest(jobs['main_proposal']),'main_started':False,
        'pending':['formal main execution scope and numeric time cap',
                   'freeze the development interpretation rule before utility evaluation',
                   'A2 second encoder implementation and matched inputs remain separate'],
        'stored_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
    }
    save(root/'verification.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()

