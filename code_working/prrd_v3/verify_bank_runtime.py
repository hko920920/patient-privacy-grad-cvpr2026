"""Predeclared public four-image, real-process checkpoint/restart verification.

Does not run main500, profile, Q/V images, recipients, DP or final evaluation.
Run CPU file checks first. This driver stops on the first worker failure.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

from .contracts import CODE, OUT, RESEARCH, now, read, sha, require, source_hashes
from .bank_runtime import (atomic_json, checkpoint_record, digest, tree_digest,
                           verify_artifact, rng_state)
from .fit_banks import contract_document, check_frozen_inputs
from .run import validate_main


def load_payload(bank):
    record, path = checkpoint_record(bank)
    return torch.load(path, map_location='cpu', weights_only=True), record


def compare_tree(a, b, atol, rtol):
    stats = {'tensor_leaves': 0, 'scalar_leaves': 0, 'max_abs': 0.,
             'max_relative_l2': 0., 'all_bitwise_equal': True}
    def rec(x, y, path):
        require(type(x) is type(y), 'Tree type differs: '+path)
        if isinstance(x, torch.Tensor):
            require(x.dtype == y.dtype and x.shape == y.shape, 'Tensor metadata differs: '+path)
            same = torch.equal(x, y)
            stats['all_bitwise_equal'] &= same; stats['tensor_leaves'] += 1
            if x.is_floating_point():
                delta = (x.double()-y.double()).abs()
                peak = float(delta.max()) if delta.numel() else 0.
                denom = float(x.double().norm())
                rel = float(delta.norm())/max(denom, 1e-30)
                stats['max_abs'] = max(stats['max_abs'], peak)
                stats['max_relative_l2'] = max(stats['max_relative_l2'], rel)
                require(torch.allclose(x, y, atol=atol, rtol=rtol), 'Tensor tolerance exceeded: '+path)
            else:
                require(same, 'Integer/RNG tensor differs: '+path)
        elif isinstance(x, dict):
            require(x.keys() == y.keys(), 'Tree keys differ: '+path)
            for k in x:
                rec(x[k], y[k], path+'/'+str(k))
        elif isinstance(x, (list, tuple)):
            require(len(x) == len(y), 'Sequence length differs: '+path)
            for i, (xx, yy) in enumerate(zip(x, y)):
                rec(xx, yy, path+'/'+str(i))
        else:
            stats['scalar_leaves'] += 1
            require(x == y, 'Scalar/step/state differs: '+path)
    rec(a, b, '')
    return stats


def logs(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    plan_path = root/'predeclared_verification_plan.json'
    plan = read(plan_path)
    require(plan['continuous_updates'] == 84 and plan['split_after'] == 82 and
            plan['total_expected_actual_updates'] == 168, 'Predeclared lengths changed')
    require(not (root/'verification_job.json').exists(), 'Preserve this attempt; do not overwrite')
    base = contract_document(plan['base_contract'])
    require(sha(plan['base_contract']) == plan['base_contract_sha256'], 'Base contract changed')
    check_frozen_inputs(base)
    oldprofile = read(CODE/'_reports/prrd_c128_profile_20260918_v1/run/profile_contract.json')
    chosen = [{'image_id': row['image_id'], 'sha256': row['sha256']}
              for row in oldprofile['templates'][:4]]
    target = CODE/'_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_targets.npz'
    implementation = source_hashes()
    job = {
        'id': 'technical_C_101', 'arm': 'C', 'seed': 101, 'images': 4, 'updates': 84,
        'artifact_kind': 'TECHNICAL_TEST_ONLY', 'base_contract': plan['base_contract'],
        'base_contract_sha256': plan['base_contract_sha256'],
        'implementation_source_sha256': implementation,
        'target_file': str(target), 'target_sha256': sha(target), 'target_prefix': 'C_',
        'templates': chosen, 'learner_policy': {k: plan['learner'][k]
            for k in ('beta', 'ridge', 'rho', 'kappa', 'guard')}, 'eta': plan['learner']['eta'],
    }
    packet = {'scope': 'PUBLIC_SMALL_PROCESS_RESUME_PNG_ONLY', 'created_before_workers': now(),
              'predeclared_plan': str(plan_path), 'predeclared_plan_sha256': sha(plan_path),
              'output': str(root/'run'), 'split_after': 82, 'job': job,
              'source_sha256': implementation, 'criteria': plan['criteria']}
    atomic_json(root/'verification_job.json', packet)
    run = root/'run'; run.mkdir()
    atomic_json(run/'TEST_ONLY_DO_NOT_REUSE_AS_MAIN.json', {
        'purpose': 'Public process-resume/PNG connection verification; not main synthesis or efficacy',
        'main_reuse_allowed': False})
    workers = []
    try:
        for leg in ('continuous', 'prefix', 'resume'):
            cmd = [sys.executable, '-B', '-m', 'prrd_v3.run', 'technical-worker',
                   '--job', str(root/'verification_job.json'), '--leg', leg]
            print(json.dumps({'phase': 'launch', 'leg': leg}), flush=True)
            started = time.monotonic()
            result = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True,
                                    encoding='utf-8', timeout=600, check=False)
            (root/(leg+'_stdout.txt')).write_text(result.stdout, encoding='utf-8')
            (root/(leg+'_stderr.txt')).write_text(result.stderr, encoding='utf-8')
            require(result.returncode == 0, 'Worker failed: '+leg+'; see preserved stderr')
            item = read(run/(leg+'_process.json'))
            workers.append({'leg': leg, 'pid': item['pid'], 'returncode': result.returncode,
                            'updates': item['updates_this_process'],
                            'process_exited': True, 'elapsed_seconds': time.monotonic()-started})
            print(json.dumps({'phase': 'exited', **workers[-1]}), flush=True)
            if leg == 'prefix':
                require(item['status'] == 'PAUSED' and item['completed_updates'] == 82,
                        'Prefix did not stop at the predeclared checkpoint')
                require(not (run/'split_bank/COMPLETED.json').exists() and
                        not (run/'split_bank/artifact').exists(), 'Intermediate bank was exported/sealed')
        continuous, split = run/'continuous_bank', run/'split_bank'
        x, xc = load_payload(continuous); y, yc = load_payload(split)
        require(x['signature'] == y['signature'], 'Continuous and resumed contracts differ')
        criteria = plan['criteria']
        state_comparison = compare_tree(x, y, criteria['resumed_parameter_optimizer_atol'],
                                        criteria['resumed_parameter_optimizer_rtol'])
        require(tree_digest(x['rng']) == tree_digest(y['rng']), 'Final RNG states differ')
        for payload in (x, y):
            require(payload['completed_updates'] == 84 and payload['next_step'] == 85 and
                    payload['active_levels'] == 2 and
                    payload['requires_grad'] == [True, True, False, False, False, False],
                    'Final step or pyramid activation incorrect')
            states = payload['optimizer']['state']
            require(set(states) == {0, 1} and int(states[0]['step']) == 84 and
                    int(states[1]['step']) == 4, 'Optimizer per-level counters incorrect')
        prefix = read(run/'prefix_process.json')
        resumed = read(run/'resume_process.json')
        require(resumed['restoration']['restored_exact'] and
                resumed['restoration']['checkpoint_sha256'] == prefix['checkpoint_sha256'] and
                resumed['restoration']['next_step'] == 83 and
                resumed['restoration']['active_levels'] == 2, 'Wrong checkpoint restored')
        require(len({x['pid'] for x in workers}) == 3, 'Workers did not use distinct processes')
        a, b = logs(continuous/'updates.jsonl'), logs(split/'updates.jsonl')
        require([x['successful_step'] for x in a] == list(range(1, 85)) ==
                [x['successful_step'] for x in b], 'Updates were skipped or repeated')
        max_loss_difference = 0.
        for first, second in zip(a, b):
            require(first['active_levels'] == second['active_levels'], 'Pyramid history differs')
            for k in first['trace']:
                v, w = first['trace'][k], second['trace'][k]
                require(math.isclose(v, w, abs_tol=criteria['resumed_loss_atol'],
                                     rel_tol=criteria['resumed_loss_rtol']), 'Trace tolerance exceeded: '+k)
                max_loss_difference = max(max_loss_difference, abs(v-w))
        spec = read(continuous/'run_spec.json')
        png_a = verify_artifact(continuous/'artifact', spec, x['signature'])
        png_b = verify_artifact(split/'artifact', spec, y['signature'])
        require(read(continuous/'COMPLETED.json')['not_for_main_study'] is True and
                read(split/'COMPLETED.json')['not_for_main_study'] is True, 'Technical artifact mislabeled')
        with (continuous/'artifact/images.csv').open(encoding='utf-8', newline='') as f:
            aa = list(csv.DictReader(f))
        with (split/'artifact/images.csv').open(encoding='utf-8', newline='') as f:
            bb = list(csv.DictReader(f))
        require(aa == bb, 'Continuous/restarted PNG labels/hashes differ')
        require((continuous/'artifact/pairs.csv').read_bytes() ==
                (split/'artifact/pairs.csv').read_bytes(), 'Virtual pairs differ')
        initial_files = sorted((continuous/'checkpoints').glob('step_000000_*.pt'))
        initial = torch.load(initial_files[0], map_location='cpu', weights_only=True)
        require(tree_digest(initial['renderer']) != tree_digest(x['renderer']),
                'Synthetic parameters never changed')
        # This check must reject before any further model or image execution.
        auth_path = root/'denied_main_authorization.json'
        atomic_json(auth_path, {'schema': 'prrd.main-authorization/v1', 'user_authorization': False})
        try:
            validate_main(plan['base_contract'], auth_path, 'main_C_101', root/'not_created.json')
        except RuntimeError as err:
            require('authority is absent' in str(err), 'Wrong authorization denial')
        else:
            raise RuntimeError('Unapproved main execution accepted')
        require(source_hashes() == implementation, 'Code changed during verification')
        check_frozen_inputs(base)
        outputs = [read(run/(leg+'_process.json')) for leg in ('continuous', 'prefix', 'resume')]
        report = {
            'status': 'PASS_PUBLIC_PROCESS_RESUME_AND_PNG_ONLY',
            'predeclared_plan_sha256': sha(plan_path),
            'job_sha256': sha(root/'verification_job.json'),
            'base_contract_sha256': plan['base_contract_sha256'],
            'implementation_source_sha256': implementation,
            'workers': workers, 'actual_GPU_worker_processes': 3,
            'images_per_technical_bank': 4, 'total_actual_optimizer_updates': 168,
            'comparison': state_comparison, 'maximum_loss_trace_difference': max_loss_difference,
            'restored_parameter_optimizer_RNG_step_pyramid_exact': True,
            'deliberately_perturbed_resume_startup_RNG_restored': True,
            'optimizer_per_level_steps': [84, 4, 0, 0, 0, 0],
            'PNG_continuous': png_a, 'PNG_resumed': png_b,
            'all4_PNG_pixels_and_hashes_identical': True,
            'intermediate_prefix_not_exported_or_completed': True,
            'unapproved_main_rejected_before_model': True,
            'source_state_unchanged': all(x['source_unchanged'] for x in outputs),
            'target_unchanged': all(x['target_unchanged'] for x in outputs),
            'counts': {k: sum(x['counts'][k] for x in outputs) for k in outputs[0]['counts']},
            'P_decodes': sum(x['access']['decoded'].get('P', 0) for x in outputs),
            'unique_P_images': len({x['image_id'] for x in chosen}),
            'Q_V_pixels': 0, 'recipient_runs': 0, 'DP_releases': 0, 'expert_reserved_access': False,
            'new_main_banks': 0, 'new_profile': False, 'efficacy_evaluated': False,
            'main_reuse_allowed': False, 'main_execution_authorized': False,
            'remaining_launch_requirements': [
                'Allowed bank IDs and numeric time cap',
                'Separate authority for Q preparation and actual Q feature/target hash receipts',
                'Main authorization bound to this verification/code and frozen recipe'],
        }
        atomic_json(root/'verification.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'comparison', 'maximum_loss_trace_difference',
                         'counts', 'P_decodes', 'new_main_banks')}), flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json', {'status': 'TECHNICAL_CHECK_FAILED_OR_INTERRUPTED',
                    'exception': repr(exc), 'completed_workers': workers,
                    'automatic_retry': False, 'main_reuse_allowed': False})
        raise


if __name__ == '__main__':
    main()

