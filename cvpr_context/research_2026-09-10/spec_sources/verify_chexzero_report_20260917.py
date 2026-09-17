"""Verify the public CheXzero review bundle without private _reports artifacts.

The private execution planning directory and model weights are intentionally
excluded from the GitHub context bundle. This checker validates the public
record that remains: report/protocol hashes, public source snapshot hashes,
state-file consistency, local HTML links, and explicit no-inference/no-weight
scope constraints.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
P = R / 'spec_sources' / 'chexzero_review_20260917_v1'
V = R / 'spec_sources' / 'chexzero_report_verification_20260917.json'


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

def state_chexzero_record(state_name):
    state = read_json(R / state_name)
    record = state.get('active_execution', {}).get('chexzero_review')
    if record is None:
        record = state.get('completed_chexzero_review_20260917')
    need(isinstance(record, dict), f'Missing CheXzero record in {state_name}')
    return state, record


def check_public_source_snapshot():
    acquisition = read_json(P / 'acquisition.json')
    checked = 0
    omitted = []
    for item in acquisition.get('files', []):
        rel = Path(item['path'])
        local = P / rel
        if local.exists():
            need(sha256(local) == item['sha256'], f'Public source hash mismatch: {rel.as_posix()}')
            checked += 1
        else:
            omitted.append(rel.as_posix())
    additional = read_json(P / 'additional_sources.json')
    for item in additional:
        rel = Path(item['path'])
        local = P / rel
        if local.exists() and 'sha256' in item:
            need(sha256(local) == item['sha256'], f'Additional source hash mismatch: {rel.as_posix()}')
            checked += 1
    legacy = read_json(P / 'legacy_torchvision_source.json')
    need(sha256(P / 'legacy_torchvision_transforms.py') == legacy['sha256'], 'Legacy torchvision source hash mismatch')
    checked += 1
    return checked, omitted


def main():
    verification = read_json(V)
    need(verification['status'] == 'PASS_CHEXZERO_REVIEW_REPORT_STATE_AND_HISTORY', 'Unexpected verification status')
    need(verification['evaluator_validated'] is False, 'Evaluator should not be validated')
    need(verification['actual_CheXzero_NIH_inference'] == 0, 'Public record should not claim new CheXzero inference')
    need(verification['fresh_evaluator_patients_reserved'] == 80, 'Reserved evaluator count changed')
    need(verification['remaining_development'] == 4053, 'Remaining development count changed')
    need(verification['reserved_confirmation_used'] is False, 'Reserved confirmation set should be unused')

    need(sha256(R / 'TRACK1_CHEXZERO_REVIEW_20260917.md') == verification['report_sha256'], 'CheXzero report hash mismatch')
    need(sha256(R / 'TRACK1_CHEXZERO_REVIEW_PROTOCOL_20260917.md') == verification['protocol_sha256'], 'CheXzero protocol hash mismatch')

    states_and_records = [state_chexzero_record('research_state.json'), state_chexzero_record('patient_baseline_spec.json')]
    records = []
    for state_name, (state, record) in zip(['research_state.json', 'patient_baseline_spec.json'], states_and_records):
        records.append(record)
        need(sha256(R / state_name) == verification['state_sha256'][state_name], f'State hash mismatch: {state_name}')
        need(state['current_step'] == 2, f'Research step changed in {state_name}')
        need(state['current_result_report'] == 'TRACK1_CHEXZERO_REVIEW_20260917.md', f'Current report mismatch in {state_name}')
        need(state['last_actual_model_result_report'] == 'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md', f'Actual model result pointer changed in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')

    for record in records:
        need(record['status'] == 'POSITIVE_SINGLE_ALTERNATIVE_READINESS_NOT_EFFICACY', 'Unexpected CheXzero decision')
        need(record['candidate'] == 'CheXzero_official_ten_checkpoint_ensemble', 'Candidate mismatch')
        need(record['checkpoints'] == 10, 'Checkpoint count mismatch')
        need(record['checkpoint_bytes'] == 3535487090, 'Checkpoint byte count mismatch')
        need(record['weights_strict_load_passed'] is True, 'Weight load readiness mismatch')
        for key in ['evaluator_validated', 'selected_model_results_consumed', 'reserved_confirmation_used', 'original_roles_changed', 'previous_padchest_gate_passed', 'private_generation_utility_established', 'DP_allowed_to_expand', 'reference_power_established']:
            need(record[key] is False, f'Unsupported conclusion changed: {key}')
        for key in ['new_NIH_pixel_access', 'new_model_inference', 'new_GPU', 'new_generation', 'new_DP']:
            need(record[key] == 0, f'Unexpected new experiment count: {key}')
        need(record['fresh_reserved_evaluator_patients'] == verification['fresh_evaluator_patients_reserved'], 'Fresh evaluator count mismatch')
        need(record['remaining_method_development'] == verification['remaining_development'], 'Remaining development mismatch')
        need(record['remaining_exclusive_emphysema'] == 44, 'Remaining exclusive emphysema count mismatch')
        need(record['later_reference_plan_per_condition'] == 32, 'Reference budget mismatch')
        need(record['final_allowed_classifier_alternative'] is True, 'Classifier alternative readiness should be allowed')

    public_sources_checked, omitted_sources = check_public_source_snapshot()
    checkpoint = read_json(P / 'checkpoint_acquisition.json')
    need(checkpoint['completed'] == 10, 'Checkpoint metadata count mismatch')
    need(checkpoint['new_model_inference'] == 0, 'Checkpoint acquisition should not include inference')
    need(not list(R.rglob('*.pt')), 'Model weight files must not be included in public bundle')

    checked_links = 0
    private_report_refs = 0
    for page in ['track1_chexzero_review.html', 'track1_chexzero_review_protocol.html']:
        public_count, private_count = check_local_links(page)
        checked_links += public_count
        private_report_refs += private_count
    need('track1_chexzero_review.html' in (R / 'index.html').read_text(encoding='utf-8-sig'), 'Index is missing CheXzero review link')

    result = {
        'status': 'PASS_PUBLIC_CHEXZERO_REVIEW_BUNDLE',
        'checked_links': checked_links,
        'private_report_refs': private_report_refs,
        'public_sources_checked': public_sources_checked,
        'omitted_source_entries': omitted_sources,
        'state_files_checked': ['research_state.json', 'patient_baseline_spec.json'],
        'report_sha256': verification['report_sha256'],
        'protocol_sha256': verification['protocol_sha256'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


