"""Single-bank CLI. Main execution requires a separate bound authorization.

This is an operational allow-list, not an OS boundary against malicious code.
No Q/V image loader, recipient, DP or final-evaluation route is provided.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

from .contracts import CODE, OUT, require, read, sha, source_hashes
from .bank_runtime import atomic_json, digest, file_lock
from .fit_banks import contract_document, check_frozen_inputs, execute_bound_job


def validate_main(base_path, authority_path, bank_id, receipt_path):
    base_path, authority_path, receipt_path = map(Path, (base_path, authority_path, receipt_path))
    base = contract_document(base_path)
    auth = read(authority_path)
    require(auth.get('schema') == 'prrd.main-authorization/v1' and
            auth.get('user_authorization') is True, 'Explicit main execution authority is absent')
    require(auth.get('contract_sha256') == sha(base_path), 'Authorization contract mismatch')
    require(bank_id in auth.get('allowed_run_ids', []), 'Bank is not authorized')
    require(isinstance(auth.get('maximum_seconds'), (int, float)) and
            math.isfinite(auth['maximum_seconds']) and auth['maximum_seconds'] > 0,
            'Numeric time cap is absent')
    for key in ('Q_V_pixel_access', 'recipient_execution', 'DP', 'expert_reserved'):
        require(auth.get(key) is False, 'Main synthesis does not authorize '+key)
    require(auth.get('verification_sha256') == sha(receipt_path), 'Verification receipt is not bound')
    receipt = read(receipt_path)
    require(receipt.get('status') in ('PASS_PUBLIC_PROCESS_RESUME_AND_PNG_ONLY',
                                    'PASS_PUBLIC_MICROBATCH_WITH_INHERITED_RESUME'),
            'Execution connection verification has not passed')
    microbatch = 4
    if receipt['status'] == 'PASS_PUBLIC_MICROBATCH_WITH_INHERITED_RESUME':
        prior = receipt['prior_resume_receipt']
        require(sha(prior['path']) == prior['sha256'] and
                read(prior['path'])['status'] == 'PASS_PUBLIC_PROCESS_RESUME_AND_PNG_ONLY',
                'Inherited process-resume evidence changed')
        runtime = receipt['runtime_amendment']
        require(sha(runtime['path']) == runtime['sha256'], 'Runtime amendment changed')
        amendment = read(runtime['path'])
        require(amendment['base_contract_sha256'] == sha(base_path) and
                amendment['images'] == 128 and amendment['updates'] == 500,
                'Runtime amendment changed the frozen experiment')
        microbatch = amendment['microbatch']
        require(type(microbatch) is int and microbatch in (4, 8, 16), 'Unverified partition')
        for evidence in amendment['verification_evidence']:
            require(sha(evidence['path']) == evidence['sha256'], 'Speed/parity evidence changed')
    implementation = source_hashes()
    require(receipt.get('implementation_source_sha256') == implementation and
            auth.get('implementation_sha256') == digest(implementation),
            'Unverified main implementation')
    check_frozen_inputs(base)
    candidates = [r for r in base['runs'] if r['id'] == bank_id]
    require(len(candidates) == 1, 'Bank is outside the frozen30-run list')
    run = candidates[0]
    target_ref = auth.get('target_receipts', {}).get(bank_id)
    require(isinstance(target_ref, dict), 'Prepared target receipt missing')
    target_path = Path(target_ref['path'])
    require(sha(target_path) == target_ref['sha256'], 'Target receipt changed')
    target = read(target_path)
    require(target.get('schema') == 'prrd.prepared-nondp-target/v1' and
            target.get('contract_sha256') == sha(base_path) and
            target.get('target_generation_rule_sha256') == base['target_generation_rule_sha256'],
            'Target generation contract differs')
    require(target.get('arm') == run['arm'], 'Wrong arm target')
    require(target.get('population') == ('P' if run['arm'] == 'A' else 'P+Q'),
            'Target population differs')
    if run['arm'] != 'A':
        require(auth.get('allow_private_prepared_targets') is True and
                target.get('Q_preparation_authorization_sha256') and
                target.get('Q_feature_sha256'), 'Private preparation is not bound/authorized')
    if run['arm'] == 'D':
        require(target.get('permutation_seed') == run['seed'], 'Wrong D pairing seed')
    require(sha(target['npz_path']) == target['npz_sha256'], 'Prepared target file changed')
    for path, expected in target.get('input_sha256', {}).items():
        require(sha(path) == expected, 'Prepared target input changed')
    templates = read(OUT/'templates_P_private.json')[str(run['seed'])]
    # The existing template manifest contains only selected P source metadata.
    chosen = [{'image_id': t['image_id'], 'sha256': t['sha256']} for t in templates]
    job = {
        'id': bank_id, 'arm': run['arm'], 'seed': run['seed'],
        'images': 128, 'updates': 500, 'artifact_kind': 'MAIN_BANK',
        'microbatch': microbatch,
        'base_contract': str(base_path.resolve()), 'base_contract_sha256': sha(base_path),
        'implementation_source_sha256': implementation,
        'target_file': target['npz_path'], 'target_sha256': target['npz_sha256'],
        'target_prefix': target.get('prefix', ''),
        'target_receipt_sha256': target_ref['sha256'],
        'templates': chosen,
        'learner_policy': {'beta': run['beta'], 'ridge': .1, 'rho': None,
                           'kappa': .1, 'guard': run['guard']},
        'eta': run['eta'],
    }
    return job, auth


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='mode', required=True)
    t = s.add_parser('technical-worker')
    t.add_argument('--job', type=Path, required=True)
    t.add_argument('--leg', choices=('continuous', 'prefix', 'resume'), required=True)
    m = s.add_parser('main-bank')
    m.add_argument('--contract', type=Path, required=True)
    m.add_argument('--authorization', type=Path, required=True)
    m.add_argument('--verification', type=Path, required=True)
    m.add_argument('--bank', required=True)
    m.add_argument('--output-root', type=Path, required=True)
    m.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.mode == 'technical-worker':
        packet = read(args.job)
        require(packet.get('scope') == 'PUBLIC_SMALL_PROCESS_RESUME_PNG_ONLY',
                'Not a bounded public verification job')
        require(sha(packet['predeclared_plan']) == packet['predeclared_plan_sha256'],
                'Predeclared comparison changed')
        require(packet['job']['artifact_kind'] == 'TECHNICAL_TEST_ONLY' and
                packet['job']['updates'] == 84 and packet['split_after'] == 82,
                'Technical length changed')
        root = Path(packet['output'])
        outcome = root/(args.leg+'_process.json')
        require(not outcome.exists(), 'Technical worker already completed; preserve evidence')
        bank = root/('continuous_bank' if args.leg == 'continuous' else 'split_bank')
        result = execute_bound_job(packet['job'], bank, resume=args.leg == 'resume',
                                   stop_after=82 if args.leg == 'prefix' else None,
                                   perturb_rng_before_resume=args.leg == 'resume')
        atomic_json(outcome, result)
        print(json.dumps({'leg': args.leg, 'status': result['status'], 'pid': result['pid'],
                          'updates': result['updates_this_process']}), flush=True)
        return
    # Validate ALL authority/binding fields before model construction or pixel opens.
    job, auth = validate_main(args.contract, args.authorization, args.bank, args.verification)
    root = args.output_root.resolve()
    require(str(root) == str(Path(auth['output_root']).resolve()), 'Output root is not authorized')
    budget = root/'authorization_budgets'/sha(args.authorization)
    budget.mkdir(parents=True, exist_ok=True)
    with file_lock(budget/'.lease.lock'):
        ledger_path = budget/'ledger.json'
        ledger = read(ledger_path) if ledger_path.exists() else {
            'authorization_sha256': sha(args.authorization), 'spent_seconds': 0., 'open_reservation': False}
        require(not ledger['open_reservation'],
                'Interrupted budget reservation remains consumed; reconcile or bind further time before resuming')
        remaining = float(auth['maximum_seconds']) - ledger['spent_seconds']
        require(remaining > 15, 'Authorized cumulative time budget exhausted')
        # Reserve remaining time before starting. An unclean exit cannot reset the budget.
        started = time.monotonic()
        ledger.update(open_reservation=True, reserved_seconds=remaining, bank=args.bank)
        atomic_json(ledger_path, ledger, replace=True)
        try:
            result = execute_bound_job(job, root/'banks'/args.bank, resume=args.resume,
                                       deadline=started+remaining)
            receipt_name = args.bank+'_'+str(time.time_ns())+'.json'
            atomic_json(budget/receipt_name, result)
            print(json.dumps(result), flush=True)
        finally:
            ledger['spent_seconds'] += time.monotonic()-started
            ledger.update(open_reservation=False, reserved_seconds=0.)
            atomic_json(ledger_path, ledger, replace=True)


if __name__ == '__main__':
    main()
