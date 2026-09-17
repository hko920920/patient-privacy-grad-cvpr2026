"""Verify the public Track 1 downstream profile bundle.

The private execution packet, generated tensors, model checkpoints, and raw
image inventories are intentionally excluded from the GitHub context bundle.
This checker validates the public report/protocol, profile record, verification
summary, state pointers, HTML links, and no-final/no-DP scope.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
RECORD = R / 'spec_sources' / 'downstream_profile_record_20260917.json'
VERIFY = R / 'spec_sources' / 'downstream_profile_report_verification_20260917.json'
REPORT = R / 'TRACK1_DOWNSTREAM_PROFILE_RESULTS_20260917.md'
PROTOCOL = R / 'TRACK1_DOWNSTREAM_PROFILE_PROTOCOL_20260917.md'


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
    omitted = 0
    for raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        normalized = unquote(parsed.path).replace('\\', '/')
        if '_reports/' in normalized or '_data/' in normalized or normalized.endswith(('.csv', '.pt', '.pth', '.npz', '.png')):
            omitted += 1
            continue
        target = (R / normalized).resolve()
        need(target.exists(), f'Broken local link in {filename}: {raw}')
        checked += 1
    return checked, omitted


def main():
    record = read_json(RECORD)
    verification = read_json(VERIFY)

    need(verification['status'] == 'PASS_REPORT_BINDINGS_NOT_EFFICACY', 'Stored verification status changed')
    need(verification['profile_judgment'] == record['judgment'], 'Profile judgment mismatch')
    need(verification['all_profile_gates_passed'] is False, 'Profile gate flag should remain false')
    need(verification['source_contracts_and_archived_failure_sources_unchanged'] is True, 'Source binding flag changed')
    need(verification['old_four_reports_unchanged'] is True, 'Historical report binding flag changed')
    need(verification['model_reruns'] == 0 and verification['new_training_updates'] == 0, 'Public report verification should not rerun models')
    need(sha256(RECORD) == verification['record_sha256'], 'Record hash mismatch')
    need(sha256(R / 'track1_downstream_profile_results.html') == verification['results_html_sha256'], 'Results HTML hash mismatch')

    need(record['report'] == 'TRACK1_DOWNSTREAM_PROFILE_RESULTS_20260917.md', 'Report pointer mismatch')
    need(record['judgment'] == 'mixed_generator_pass_original_classifier_fail_limited_repair', 'Unexpected profile judgment')
    need(record['profile_fully_passed'] is False, 'Profile should remain not fully passed')
    need(record['raw_images_verified'] == 11277 and record['raw_patients'] == 4805, 'Raw metadata verification counts changed')
    need(record['generated_distinct_images'] == 28 and record['generation_decodes'] == 31 and record['UNet_calls'] == 931, 'Generation profile counts changed')
    need(record['generator_backward'] == 0, 'Generator backward count changed')
    need(record['original_invalid_classifier_updates'] == 98, 'Invalid classifier update count changed')
    need(record['corrected_classifier_updates'] == 2 and record['total_classifier_updates'] == 100, 'Corrected/total update counts changed')
    need(record['corrected_complete_seven_arm_training_replay'] is False, 'Seven-arm replay flag changed')
    need(record['final_ready'] is False, 'Final ready flag changed')
    need(record['expert_final_pixels'] == 0 and record['expert_final_predictions'] == 0, 'Expert final should remain unopened')
    need(record['reserved_pixels'] == 0 and record['reserved_predictions'] == 0, 'Reserved set should remain unopened')
    need(record['AUROC_AP_computed'] is False, 'AUROC/AP should not be computed')
    need(record['private_incremental_utility_established'] is False, 'Private utility overclaim')
    need(record['new_DP'] == 0, 'DP should not run')
    need(record['full512_generation_executed'] is False and record['full21classifier_runs_executed'] is False, 'Full package should remain unexecuted')
    need(record['previous_efficacy_result'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Previous efficacy pointer changed')
    need(record['original_roles_changed'] is False, 'Original roles should not change')
    need(record['remote_push'] is False, 'Remote push flag should be false in record')
    need(record['source_sha256']['report'] == sha256(REPORT), 'Report SHA mismatch')
    need(record['source_sha256']['protocol'] == sha256(PROTOCOL), 'Protocol SHA mismatch')

    for filename, expected in {
        'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md': 'a2f1556f6dcfac42f40fbed445afba14f86db0891d851a47d40ad6165abe7b9b',
        'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md': '8837f86cbb422272ba13d08d60c7ea979539e907aaad4e0c442d125a569a1a4f',
        'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md': '08f6b02f71498bb454654aa94a432be4ffc24bf7c06501806d49d199b9f5cb47',
        'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md': 'c150ec73417981db15fa3199dd26d65f2710bedd6daf31362539c0546f9d63e1',
    }.items():
        need(sha256(R / filename) == expected, f'Historical report hash changed: {filename}')

    for state_name in ['research_state.json', 'patient_baseline_spec.json']:
        state = read_json(R / state_name)
        need(state['current_step'] == 2, f'Stage changed in {state_name}')
        need(state['current_result_report'] == state['last_actual_model_result_report'] == 'TRACK1_DOWNSTREAM_PROFILE_RESULTS_20260917.md', f'Current profile result pointer mismatch in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')
        need(state['completed_downstream_profile_20260917'] == record, f'Completed profile record mismatch in {state_name}')
        need(state['active_execution']['downstream_profile'] == record, f'Active profile record mismatch in {state_name}')
        need(state['pre_downstream_profile_actual_result'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Previous actual result pointer mismatch in {state_name}')

    checked_links = 0
    omitted_links = 0
    for page in ['track1_downstream_profile_protocol.html', 'track1_downstream_profile_results.html']:
        public_count, omitted_count = check_local_links(page)
        checked_links += public_count
        omitted_links += omitted_count
    index = (R / 'index.html').read_text(encoding='utf-8-sig')
    need('track1_downstream_profile_results.html' in index, 'Index missing downstream profile result link')
    result_page = (R / 'track1_downstream_profile_results.html').read_text(encoding='utf-8-sig')
    need('track1_downstream_profile_protocol.html' in result_page, 'Result page missing downstream profile protocol link')

    result = {
        'status': 'PASS_PUBLIC_DOWNSTREAM_PROFILE_BUNDLE',
        'checked_links': checked_links,
        'omitted_private_artifact_links': omitted_links,
        'profile_fully_passed': record['profile_fully_passed'],
        'generated_distinct_images': record['generated_distinct_images'],
        'generation_decodes': record['generation_decodes'],
        'total_classifier_updates': record['total_classifier_updates'],
        'expert_final_pixels': record['expert_final_pixels'],
        'reserved_pixels': record['reserved_pixels'],
        'AUROC_AP_computed': record['AUROC_AP_computed'],
        'new_DP': record['new_DP'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
