"""Verify the public NIH expert-label patient-overlap audit bundle.

The private CSVs, patient images, local inventories, and _reports execution
packet are intentionally excluded from the GitHub context bundle. This checker
validates the public audit record, report hash, state pointers, and local HTML
links while preserving the no-new-model/no-new-pixel-read scope.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
P = R / 'spec_sources' / 'nih_expert_overlap_20260917'
REPORT = R / 'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md'
HTML = R / 'nih_expert_patient_overlap_results.html'


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
        if '_reports/' in normalized or '_data/' in normalized or normalized.endswith(('.csv', '.parquet', '.docx')):
            omitted += 1
            continue
        target = (R / normalized).resolve()
        need(target.exists(), f'Broken local link in {filename}: {raw}')
        checked += 1
    return checked, omitted


def main():
    record = read_json(P / 'record.json')
    verification = read_json(P / 'record_verification.json')
    status = record['status']

    need(verification['status'] == 'PASS', 'Stored verification did not pass')
    for key, value in verification['checks'].items():
        need(value is True, f'Stored verification check failed: {key}')
    need(status['report'] == 'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md', 'Report pointer mismatch')
    need(status['type'] == 'patient_metadata_and_recorded_usage_audit_not_model_experiment', 'Unexpected audit type')
    need(status['judgment'] == 'positive_for_disjoint_final_candidate_not_new_development_data', 'Unexpected judgment')
    need(status['patients'] == 532 and status['images'] == 810, 'Expert cohort size changed')
    need(status['expert_positive_patients'] == {'Emphysema': 7, 'Pneumothorax': 86}, 'Expert positive patient counts changed')
    need(status['recorded_model_or_result_consumption_overlap'] == 0, 'Recorded model/result overlap changed')
    need(status['current_generator_training_overlap'] == 0, 'Current generator training overlap changed')
    need(status['reserved_cvpr_confirmation_overlap'] == 0, 'Reserved confirmation overlap changed')
    need(status['all_patients_in_existing_official_test'] is True, 'Official-test role finding changed')
    need(status['original_roles'] == {'final_test': 466, 'official_test_census_only': 66}, 'Original role summary changed')
    need(status['new_nonlocked_development_candidates'] == 0, 'New development candidate count changed')
    need(status['existing_roles_changed'] is False, 'Existing roles should not change')
    need(status['final_evaluation_adopted_or_executed'] is False, 'Final evaluation should not be adopted/executed')
    need(status['official_original_label_row_match_verified'] is False, 'Official row-match flag changed')
    need(status['new_inference'] == status['new_generation'] == status['new_training'] == status['new_DP'] == 0, 'Audit should not add model experiments')
    need(status['new_patient_pixel_reads'] == 0, 'Audit should not read patient pixels')
    need(status['all810_files_exist_and_sizes_match'] is True, 'File availability summary changed')
    need(status['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Actual result pointer changed')
    need(sha256(REPORT) == status['report_sha256'], 'Report hash mismatch')

    for filename, expected in {
        'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md': 'a2f1556f6dcfac42f40fbed445afba14f86db0891d851a47d40ad6165abe7b9b',
        'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md': '8837f86cbb422272ba13d08d60c7ea979539e907aaad4e0c442d125a569a1a4f',
        'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md': '08f6b02f71498bb454654aa94a432be4ffc24bf7c06501806d49d199b9f5cb47',
    }.items():
        need(sha256(R / filename) == expected, f'Historical report hash changed: {filename}')

    for state_name in ['research_state.json', 'patient_baseline_spec.json']:
        state = read_json(R / state_name)
        need(state['current_step'] == 2, f'Stage changed in {state_name}')
        need(state['current_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Actual result pointer mismatch in {state_name}')
        need(state['current_planning_report'] == 'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md', f'Planning pointer mismatch in {state_name}')
        need(state['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Last actual pointer mismatch in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')
        need(state['completed_nih_expert_overlap_20260917'] == status, f'Completed overlap record mismatch in {state_name}')
        need(state['active_execution']['nih_expert_patient_overlap_audit'] == status, f'Active overlap record mismatch in {state_name}')
        previous = record['previous_planning'][state_name]
        need(previous['current_planning_report'] == 'NIH_EXPERT_LABEL_PUBLIC_COPY_20260917.md', f'Previous planning pointer mismatch in {state_name}')

    checked_links, omitted_links = check_local_links('nih_expert_patient_overlap_results.html')
    need('nih_expert_patient_overlap_results.html' in (R / 'index.html').read_text(encoding='utf-8-sig'), 'Index missing overlap result link')
    need(not list(P.glob('*.csv')), 'Private/source CSVs should not be included in public overlap bundle')
    need(not list(P.glob('*.parquet')), 'Private/source parquet files should not be included in public overlap bundle')

    result = {
        'status': 'PASS_PUBLIC_NIH_EXPERT_OVERLAP_BUNDLE',
        'checked_links': checked_links,
        'omitted_data_links': omitted_links,
        'patients': status['patients'],
        'images': status['images'],
        'expert_positive_patients': status['expert_positive_patients'],
        'recorded_model_or_result_consumption_overlap': status['recorded_model_or_result_consumption_overlap'],
        'current_generator_training_overlap': status['current_generator_training_overlap'],
        'new_patient_pixel_reads': status['new_patient_pixel_reads'],
        'last_actual_model_result_report': status['last_actual_model_result_report'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
