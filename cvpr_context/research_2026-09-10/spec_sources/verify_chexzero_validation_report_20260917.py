"""Verify the public CheXzero validation bundle without private _reports artifacts.

The private execution packet, selected-image CSVs, and model weights are not in
this GitHub context bundle. This checker validates the public evidence packet,
state-file consistency, result/protocol/figure hashes, and local HTML links.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
P = R / 'spec_sources' / 'chexzero_validation_20260917_v1'
V = R / 'spec_sources' / 'chexzero_validation_report_verification_20260917.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def need(condition, message):
    if not condition:
        raise AssertionError(message)


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for item in attrs:
            if len(item) < 2:
                continue
            key, value = item[0], item[1]
            if key in ('href', 'src') and value:
                self.links.append(value)


def check_local_links(filename):
    parser = LinkCollector()
    body = (R / filename).read_text(encoding='utf-8-sig')
    need('\ufffd' not in body, f'Invalid replacement character in {filename}')
    parser.feed(body)
    checked = 0
    private_refs = 0
    for raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        normalized = unquote(parsed.path).replace('\\', '/')
        if '_reports/' in normalized or '_models/' in normalized or normalized.startswith('../../code_working/_'):
            private_refs += 1
            continue
        target = (R / normalized).resolve()
        need(target.exists(), f'Broken local link in {filename}: {raw}')
        checked += 1
    return checked, private_refs


def state_validation_record(state_name):
    state = read_json(R / state_name)
    record = state.get('active_execution', {}).get('chexzero_validation')
    if record is None:
        record = state.get('completed_chexzero_validation_20260917')
    need(isinstance(record, dict), f'Missing CheXzero validation record in {state_name}')
    return state, record


def main():
    verification_record = read_json(V)
    aggregate = read_json(P / 'aggregate_record.json')
    result = read_json(P / 'result.json')
    metrics = read_json(P / 'metrics.json')
    independent = read_json(P / 'verification.json')
    parity = read_json(P / 'parity.json')

    need(verification_record['status'] == 'PASS_CHEXZERO_ACTUAL_RESULT_REPORT_STATE_AND_HISTORY', 'Unexpected stored verification status')
    need(verification_record['actual_patients'] == 80, 'Actual patient count changed')
    need(verification_record['checkpoints'] == 10, 'Checkpoint count changed')
    need(verification_record['joint_gate_passed'] is False, 'Joint gate should remain failed')
    need(verification_record['classifier_search_closed'] is True, 'Classifier search should be closed')
    need(verification_record['reserved_confirmation_consumed'] is False, 'Reserved confirmation should be unused')

    for name, expected in aggregate['evidence_files'].items():
        need(sha256(P / name) == expected, f'Public evidence hash mismatch: {name}')
    need(independent['status'] == 'PASS_INDEPENDENT_CHEXZERO_VALIDATION', 'Independent validation status mismatch')
    need(independent['result_sha256'] == sha256(P / 'result.json'), 'Result hash mismatch in verification packet')
    need(independent['metrics_sha256'] == sha256(P / 'metrics.json'), 'Metrics hash mismatch in verification packet')
    need(aggregate['contract_sha256'] == result['contract_sha256'] == independent['contract_sha256'], 'Contract hash alignment mismatch')
    need(aggregate['selected_manifest_sha256'] == aggregate['original_plan_manifest_sha256'], 'Prospective cohort manifest changed')
    need(parity.get('passed') is True and len(parity.get('per_checkpoint', [])) == 10, 'Parity packet did not pass')

    need(result['patients'] == 80, 'Result patient count mismatch')
    need(result['checkpoints'] == 10, 'Result checkpoint count mismatch')
    need(result['joint_pass'] is False and metrics['joint_pass'] is False, 'Joint gate should fail in public metrics')
    need(result['new_generation'] == 0 and result['new_DP'] == 0 and result['backward_calls'] == 0, 'Unexpected training/generation scope')
    need(result['reserved_confirmation_consumed'] is False, 'Result consumed reserved confirmation')

    for target in ['Emphysema', 'Pneumothorax']:
        target_metric = metrics['metrics'][target + '__rest']
        opposite_metric = metrics['metrics'][target + '__opposite']
        criterion_pass = target_metric['auc'] >= 0.7 and target_metric['auc_ci95'][0] > 0.5 and opposite_metric['auc'] >= 0.6 and opposite_metric['auc_ci95'][0] > 0.5
        need(criterion_pass is False, f'{target} criterion unexpectedly passes')
        need(metrics['criterion'][target]['pass'] is False, f'{target} public criterion flag changed')

    states_and_records = [state_validation_record('research_state.json'), state_validation_record('patient_baseline_spec.json')]
    for state_name, (state, record) in zip(['research_state.json', 'patient_baseline_spec.json'], states_and_records):
        need(sha256(R / state_name) == verification_record['state_sha256'][state_name], f'State hash mismatch: {state_name}')
        need(state['current_step'] == 2, f'Research step changed in {state_name}')
        need(state['last_actual_model_result_report'] == state['current_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Latest result pointer mismatch in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')
        need(record == state['completed_chexzero_validation_20260917'], f'Completed/active validation mismatch in {state_name}')
        need(record['status'] == 'NEGATIVE_CHEXZERO_JOINT_MEASUREMENT_GATE_CLASSIFIER_SEARCH_CLOSED', 'Unexpected validation decision')
        need(record['actual_validation_patients'] == 80, 'State validation patient count mismatch')
        need(record['official_checkpoint_count'] == 10, 'State checkpoint count mismatch')
        need(record['evaluator_validated'] is False, 'Evaluator should not be validated')
        need(record['joint_gate_passed'] is False and record['both_target_gates_failed'] is True, 'Gate status mismatch')
        need(record['classifier_search_closed'] is True and record['targeted_classifier_path_paused'] is True, 'Stop rule mismatch')
        need(record['private_generation_utility_established'] is False, 'Private utility overclaim')
        need(record['DP_allowed_to_expand'] is False, 'DP expansion overclaim')
        need(record['automatic_targeted_generation_allowed'] is False, 'Automatic generation overclaim')
        need(record['reserved_confirmation_consumed'] is False, 'Reserved confirmation mismatch')
        need(record['remaining_target_development_patients'] == 4053, 'Remaining development mismatch')
        need(record['remaining_exclusive_emphysema'] == 44, 'Remaining exclusive emphysema mismatch')
        need(record['new_generation'] == 0 and record['new_DP'] == 0, 'Unexpected new generation/DP')

    need(sha256(R / 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md') == verification_record['report_sha256'], 'Report hash mismatch')
    need(sha256(R / 'TRACK1_CHEXZERO_VALIDATION_PROTOCOL_20260917.md') == verification_record['protocol_sha256'], 'Protocol hash mismatch')
    for ext, expected in verification_record['figure_sha256'].items():
        need(sha256(R / 'assets' / f'chexzero_validation_auc_20260917.{ext}') == expected, f'Figure hash mismatch: {ext}')
    need(sha256(P / 'aggregate_record.json') == verification_record['public_aggregate_sha256'], 'Aggregate record hash mismatch')
    need(sha256(R / 'TRACK1_CHEXZERO_REVIEW_20260917.md') == 'f9b64d582a2fec76ba0ea333ecbaa777963b8270a26714e389569c4004728154', 'Previous CheXzero review changed')
    need(sha256(R / 'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md') == '8837f86cbb422272ba13d08d60c7ea979539e907aaad4e0c442d125a569a1a4f', 'Previous PadChest report changed')
    need(not list(R.rglob('*.pt')), 'Model weight files must not be included in public bundle')

    checked_links = 0
    private_refs = 0
    for page in ['track1_chexzero_validation_results.html', 'track1_chexzero_validation_protocol.html']:
        public_count, private_count = check_local_links(page)
        checked_links += public_count
        private_refs += private_count
    need('track1_chexzero_validation_results.html' in (R / 'index.html').read_text(encoding='utf-8-sig'), 'Index is missing CheXzero validation result link')

    public_result = {
        'status': 'PASS_PUBLIC_CHEXZERO_VALIDATION_BUNDLE',
        'checked_links': checked_links,
        'private_report_refs': private_refs,
        'actual_patients': 80,
        'checkpoints': 10,
        'joint_gate_passed': False,
        'classifier_search_closed': True,
        'state_files_checked': ['research_state.json', 'patient_baseline_spec.json'],
        'report_sha256': verification_record['report_sha256'],
        'protocol_sha256': verification_record['protocol_sha256'],
        'public_aggregate_sha256': verification_record['public_aggregate_sha256'],
    }
    print(json.dumps(public_result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

