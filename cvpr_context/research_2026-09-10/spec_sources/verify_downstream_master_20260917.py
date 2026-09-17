"""Verify the public Track 1 downstream master protocol bundle.

The private resource manifests and development allocation CSVs live under
code_working/_reports and are intentionally excluded from the public GitHub
bundle. This checker validates the public plan, record, verification summary,
state pointers, historical report hashes, and local HTML links.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
PLAN = R / 'spec_sources' / 'downstream_master_plan_20260917.json'
RECORD = R / 'spec_sources' / 'downstream_master_record_20260917.json'
VERIFY = R / 'spec_sources' / 'downstream_master_verification_20260917.json'
REPORT = R / 'TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md'
HTML = R / 'track1_downstream_master_protocol.html'


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
    plan = read_json(PLAN)
    record = read_json(RECORD)
    verification = read_json(VERIFY)
    html_verification = read_json(R / 'spec_sources' / 'downstream_master_html_verification_20260917.json')

    need(plan['schema'] == 'pneumothorax-downstream-master/v1', 'Unexpected schema')
    need(plan['status'] == 'planning_and_metadata_snapshot_complete_no_model_execution', 'Unexpected plan status')
    need(plan['stage'] == 2 and plan['track'] == 1, 'Unexpected stage/track')
    need(plan['report'] == 'TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md', 'Plan report pointer mismatch')
    need(plan['final_ready'] is False and plan['expert_final_performance_opened'] is False, 'Final performance should remain unopened')
    need(plan['original_roles_changed'] is False, 'Original roles should not change')
    need(plan['synthetic']['total_images'] == 512 and plan['synthetic']['images_per_method'] == 128, 'Synthetic arithmetic changed')
    need(plan['synthetic']['shared_across_methods'] is True and plan['synthetic']['no_outcome_filtering'] is True, 'Synthetic design guard changed')
    need(plan['classifier']['paired_seeds'] == [11, 23, 37], 'Classifier seeds changed')
    need(plan['development_investment_rule']['candidate_vs'] == 'S1 and R1', 'Development comparison changed')
    need(plan['DP']['unit'] == 'add_remove_one_patient', 'DP unit changed')
    need(plan['final']['patients'] == 532 and plan['final']['images'] == 810, 'Final expert set size changed')
    need(plan['next_profile']['performed'] is False, 'Profile should not be marked performed')

    need(record['judgment'] == 'positive_design_and_development_resources_not_method_efficacy', 'Unexpected record judgment')
    need(record['report_sha256'] == sha256(REPORT), 'Report hash mismatch')
    need(record['plan_sha256'] == sha256(PLAN), 'Plan hash mismatch')
    need(record['final_ready'] is False and record['expert_final_performance_opened'] is False, 'Record final flags changed')
    need(record['new_model_forwards'] == record['new_training'] == record['new_generation'] == record['new_DP'] == 0, 'Record should not add execution')
    need(record['new_patient_pixel_reads'] == 0, 'Record should not add patient-pixel reads')
    need(record['original_roles_changed'] is False, 'Record role flag changed')
    need(record['actual_result_unchanged'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Actual result pointer changed')
    need(record['public_real'] == {'patients': 672, 'images': 813, 'positive_images': 6, 'positive_patients': 6}, 'Public real summary changed')
    need(record['private_real_diagnostic'] == {'patients': 80, 'images': 320, 'positive_images': 33, 'positive_patients': 17}, 'Private diagnostic summary changed')
    need(record['development']['classifier_selection']['patients'] == 2027, 'Classifier-selection patients changed')
    need(record['development']['method_development']['patients'] == 2026, 'Method-development patients changed')

    need(verification['status'] == 'PASS_METADATA_COUNTS_PARTITIONS_RECORDED_DUPLICATES_AND_PLAN_BINDINGS', 'Verification status changed')
    need(verification['allocation_patients'] == 4053, 'Allocation patient count changed')
    need(verification['public_images'] == 813 and verification['private_images'] == 320, 'Public/private image counts changed')
    need(verification['source_files_checked'] == 6, 'Source-file check count changed')
    need(verification['frozen_reports_checked'] == 4, 'Frozen report check count changed')
    need(verification['raw_image_file_hashes_recomputed'] is False and verification['pixels_read'] == 0 and verification['model_inference'] == 0, 'Verification scope changed')
    need(verification['statistical_power_established'] is False, 'Power claim changed')
    for key, overlap in verification['overlaps'].items():
        need(overlap == {'patient': 0, 'image': 0, 'recorded_file_sha': 0}, f'Overlap changed: {key}')
    need(record['metadata_verification_sha256'] == sha256(VERIFY), 'Verification file hash mismatch')
    need(html_verification['status'].startswith('PASS') or html_verification.get('broken_links') in ([], 0), 'HTML verification packet did not pass')

    for filename, expected in {
        'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md': 'a2f1556f6dcfac42f40fbed445afba14f86db0891d851a47d40ad6165abe7b9b',
        'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md': '8837f86cbb422272ba13d08d60c7ea979539e907aaad4e0c442d125a569a1a4f',
        'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md': '08f6b02f71498bb454654aa94a432be4ffc24bf7c06501806d49d199b9f5cb47',
        'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md': 'c150ec73417981db15fa3199dd26d65f2710bedd6daf31362539c0546f9d63e1',
    }.items():
        need(sha256(R / filename) == expected, f'Frozen report hash changed: {filename}')

    for state_name in ['research_state.json', 'patient_baseline_spec.json']:
        state = read_json(R / state_name)
        need(state['current_step'] == 2, f'Stage changed in {state_name}')
        need(state['current_planning_report'] == plan['report'], f'Planning pointer mismatch in {state_name}')
        need(state['current_result_report'] == state['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Actual result pointer mismatch in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')
        need(state['completed_downstream_master_20260917'] == record, f'Completed downstream record mismatch in {state_name}')
        need(state['active_execution']['downstream_master_design'] == record, f'Active downstream record mismatch in {state_name}')

    checked_links, omitted_links = check_local_links('track1_downstream_master_protocol.html')
    need('track1_downstream_master_protocol.html' in (R / 'index.html').read_text(encoding='utf-8-sig'), 'Index missing downstream protocol link')

    result = {
        'status': 'PASS_PUBLIC_DOWNSTREAM_MASTER_BUNDLE',
        'checked_links': checked_links,
        'omitted_data_links': omitted_links,
        'allocation_patients': verification['allocation_patients'],
        'public_images': verification['public_images'],
        'private_images': verification['private_images'],
        'final_ready': record['final_ready'],
        'expert_final_performance_opened': record['expert_final_performance_opened'],
        'new_model_forwards': record['new_model_forwards'],
        'new_generation': record['new_generation'],
        'new_DP': record['new_DP'],
        'actual_result_unchanged': record['actual_result_unchanged'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
