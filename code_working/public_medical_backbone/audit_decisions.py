"""Post-run independent decision audit. Standard library only; no image judgement."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def audit(out):
    start = time.perf_counter()
    destination = out / 'decision_verification.json'
    if destination.exists():
        raise FileExistsError(destination)
    hashes = {}
    checks = 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    def digest(path):
        path = Path(path).resolve()
        key = str(path)
        if key not in hashes:
            value = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1048576), b''):
                    value.update(block)
            hashes[key] = value.hexdigest()
        return hashes[key]

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def bound(path, expected):
        require(digest(path) == expected, 'SHA256 mismatch: ' + str(path))

    result = dict(schema='public-medical-decision-independent-audit/v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256=digest(Path(__file__)),
                  audit_code_finalized_after_original_decisions=True,
                  new_GPU_calls=0, new_image_reviews=0, original_files_modified=False)
    try:
        contract = read(out / 'contract.json')
        plan = Path(contract['plan_directory'])
        bound(plan / 'plan_lock.json', contract['plan_lock_sha256'])
        lock = read(plan / 'plan_lock.json')
        for relative, expected in lock['files'].items():
            bound(plan.parents[1] / relative, expected)
        for path, expected in contract['source_sha256'].items():
            if Path(path).suffix == '.py':
                bound(path, expected)
        spec = read(plan / 'execution_spec.json')
        bound(plan / 'execution_spec.json', contract['execution_spec_sha256'])
        bound(plan / 'generation_tasks.csv', spec['output_sha256']['generation_tasks.csv'])
        with (plan / 'generation_tasks.csv').open(encoding='utf-8-sig', newline='') as handle:
            task_rows = list(csv.DictReader(handle))
        tasks = {row['task_id']: row for row in task_rows}
        require(len(tasks) == len(task_rows) == 32, 'Fixed 32 generation tasks')
        prompts = ('generic', 'normal', 'effusion', 'cardiomegaly')
        require('>=12/16' in spec['gate']['selection'] and '>=2/4' in spec['gate']['selection'], 'Original thresholds')
        require(spec['gate']['uncertain_or_reviewer_disagreement'] == 'fail', 'Disagreement/uncertainty policy')
        require('earliest passing E4 then E8' in spec['gate']['selection'], 'Original earliest policy')
        require('Failure does not permit switching checkpoint' in spec['gate']['confirmation'], 'No confirmation rescue')
        salt = spec['training']['sampler_seed_rule']
        for task in task_rows:
            tid = task['task_id']
            phase, prompt_id, index = tid.split('_')
            require(phase == task['phase'] and prompt_id == task['prompt_id'], 'Task identity')
            expected_seed = int(hashlib.sha256(f'{salt}|generation|{phase}|{prompt_id}|{index}'.encode()).hexdigest()[:15], 16)
            require(int(task['seed']) == expected_seed, 'Preselected generation seed')
        require(len({int(row['seed']) for row in task_rows}) == 32, 'Disjoint selection/confirmation seeds')
        summaries = {}
        selected = None
        for phase, expected_status in (
            ('selection', 'PASS_PUBLIC_MEDICAL_SELECTION_SAVED_SAMPLING_AND_PAIRED_INPUTS'),
            ('confirmation', 'PASS_PUBLIC_MEDICAL_CONFIRMATION_SAVED_SAMPLING_AND_SELECTION_BINDING')):
            manifest_path = out / f'manifest_{phase}.json'
            manifest = read(manifest_path)
            decision = read(out / f'{phase}_decision.json')
            verification_path = out / f'{phase}_verification.json'
            verified = read(verification_path)
            report_path = out / f'generation_{phase}.json'
            report = read(report_path)
            require(verified['status'] == expected_status and verified['complete'] is True, 'Prior numerical PASS')
            bound(out / f'{phase}_verification_protocol.json', verified['protocol_sha256'])
            bound(Path(__file__).with_name('verify_pilot.py'), verified['code_sha256'])
            for path in (out / 'contract.json', manifest_path, report_path):
                bound(path, verified['input_sha256'][str(path.resolve())])
            bound(manifest_path, report['manifest_sha256'])
            bound(out / 'contract.json', report['contract_sha256'])
            bound(out / 'training_report.json', report['training_report_sha256'])
            bound(manifest_path, decision['manifest_sha256'])
            bound(verification_path, decision[f'{phase}_verification_sha256'])
            require(report['complete'] is True and decision['complete'] is True and decision['phase'] == phase, 'Complete phase identity')
            expected_cps = ['E4', 'E8'] if phase == 'selection' else [selected]
            require(report['checkpoint_choices'] == expected_cps, 'Fixed generated checkpoints')
            require(len(manifest) == report['records'] == (36 if phase == 'selection' else 16), 'Manifest size')
            active = [row for row in manifest if row['phase'] == phase]
            diagnostic = [row for row in manifest if row['phase'] == 'diagnostic']
            require(len(active) == 16 * len(expected_cps), 'Counted images')
            require(len(diagnostic) == (4 if phase == 'selection' else 0), 'Diagnostic exclusion')
            require(len(active) + len(diagnostic) == len(manifest), 'No unknown image phase')
            require(len({row['image_id'] for row in manifest}) == len(manifest), 'Unique image IDs')
            ids = {row['image_id'] for row in active}
            expected_tasks = {row['task_id'] for row in task_rows if row['phase'] == phase}
            for cp in expected_cps:
                subset = [row for row in active if row['checkpoint'] == cp]
                require(len(subset) == 16 and {row['task_id'] for row in subset} == expected_tasks, 'Same complete fixed tasks per checkpoint')
                for prompt in prompts:
                    require(sum(tasks[row['task_id']]['prompt_id'] == prompt for row in subset) == 4, 'Four inputs per prompt')
            for row in manifest:
                expected_id = hashlib.sha256(f"public-medical-blind-v1|{row['checkpoint']}|{row['task_id']}".encode()).hexdigest()[:12]
                require(row['image_id'] == expected_id, 'Blind ID mapping')
                for key, sha_key in [('path', 'sha256'), ('image_path', 'image_sha256')]:
                    artifact = out / row[key]
                    bound(artifact, row[sha_key])
                    bound(artifact, verified['input_sha256'][str(artifact.resolve())])
            ballots = {}
            scopes = {}
            for reviewer in ('root', 'independent'):
                ballot_path = out / f'review_{phase}_{reviewer}.json'
                ballot = read(ballot_path)
                bound(ballot_path, decision['review_sha256'][ballot_path.name])
                require(ballot['reviewer'] == reviewer and ballot['phase'] == phase, 'Ballot identity')
                require(ballot['checkpoint_blinded'] is True, 'Stored figure-label blinding flag')
                votes = ballot['images']
                mapping = {vote['image_id']: vote for vote in votes}
                require(len(mapping) == len(votes) == len(ids) and set(mapping) == ids, 'Complete unique votes')
                for vote in votes:
                    require(type(vote['pass']) is bool and isinstance(vote['note'], str) and bool(vote['note'].strip()), 'Explicit binary vote and reason')
                ballots[reviewer] = mapping
                scopes[reviewer] = ballot.get('reviewer_scope')
            both = {iid: ballots['root'][iid]['pass'] and ballots['independent'][iid]['pass'] for iid in ids}
            disagreements = sum(ballots['root'][iid]['pass'] != ballots['independent'][iid]['pass'] for iid in ids)
            require(decision['joint_pass_by_image'] == both, 'Per-image AND reproduced')
            require(decision['reviewer_disagreements'] == disagreements, 'Disagreements reproduced')
            counts = {}
            individual = {}
            for cp in expected_cps:
                subset = [row for row in active if row['checkpoint'] == cp]
                per_prompt = {prompt: sum(both[row['image_id']] for row in subset if tasks[row['task_id']]['prompt_id'] == prompt) for prompt in prompts}
                total = sum(per_prompt.values())
                passed = total >= 12 and all(value >= 2 for value in per_prompt.values())
                counts[cp] = dict(total_pass=total, per_prompt_pass=per_prompt, passes_gate=passed)
                individual[cp] = {reviewer: sum(ballots[reviewer][row['image_id']]['pass'] for row in subset) for reviewer in ballots}
            require(decision['checkpoint_results'] == counts, 'Checkpoint totals and gates reproduced')
            require(decision['nonclinical_heuristic_only'] is True and decision['private_utility_or_DP_claim'] is False, 'Declared narrow scope')
            if phase == 'selection':
                selected = 'E4' if counts['E4']['passes_gate'] else ('E8' if counts['E8']['passes_gate'] else None)
                require(decision['selected_checkpoint'] == selected, 'Earliest passing checkpoint')
            else:
                require(decision['selected_checkpoint'] == selected, 'Fixed confirmation checkpoint')
                require(decision['passes_gate'] == counts[selected]['passes_gate'], 'Confirmation result')
                for obj in (decision, report):
                    bound(out / 'selection_decision.json', obj['selection_decision_sha256'])
                bound(out / 'selection_decision.json', verified['input_sha256'][str((out / 'selection_decision.json').resolve())])
                require('known to the coordinator' in (scopes['root'] or ''), 'Confirmation unblinding limitation preserved')
            summaries[phase] = dict(checkpoints=counts, individual_reviewer_passes=individual,
                reviewer_disagreements=disagreements, excluded_diagnostic_images=len(diagnostic), reviewer_scopes=scopes)
        result.update(status='PASS_INDEPENDENT_PUBLIC_MEDICAL_SELECTION_CONFIRMATION_DECISION_ARITHMETIC',
            complete=True, summaries=summaries, selected_checkpoint=selected,
            confirmation_gate_pass=summaries['confirmation']['checkpoints'][selected]['passes_gate'],
            limits=['Audit PASS verifies saved aggregation and provenance, not a passing image gate.',
                'No independent image re-reading, medical judgement or clinical endpoint validation was performed.',
                'Confirmation root reviewer knew the selected checkpoint; only figure labels were hidden. Two fully checkpoint-blinded reviewers cannot be claimed.',
                'Ambiguous-fail is a reviewer instruction; binary votes are checked without inferring uncertainty from free-text notes.',
                'Selection equality 13/16 is not evidence of equivalent quality. The confirmation gate failed for this fixed pilot.',
                'The unit is a generated input, not the reserved real-reference patient cohorts. This is not private utility or DP evidence.'])
    except Exception as exc:
        result.update(status='FAIL_INDEPENDENT_DECISION_AUDIT', complete=False,
                      error_type=type(exc).__name__, error=str(exc))
    result.update(checks=checks, seconds=time.perf_counter()-start, input_sha256=hashes)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'checks', 'seconds', 'selected_checkpoint', 'confirmation_gate_pass', 'error')}, ensure_ascii=False))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / '_reports/public_medical_backbone_20260916_v1')
    args = parser.parse_args()
    audit(args.out.resolve())
