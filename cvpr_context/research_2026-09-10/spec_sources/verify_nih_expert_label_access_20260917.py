"""Verify the public NIH expert-label access/public-copy bundle.

This bundle records access status and public-derived label metadata. It does not
include private patient pixels, official CSV downloads, parquet/CSV mirrors, or
new model outputs. The author inquiry is a prepared draft only; no external
message is recorded as sent.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
ACCESS = R / 'spec_sources' / 'nih_expert_label_access_20260917'
PUBLIC = R / 'spec_sources' / 'nih_expert_label_public_copy_20260917'


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


def state_records(state_name):
    state = read_json(R / state_name)
    access = state.get('completed_nih_expert_label_access_20260917')
    public = state.get('completed_nih_expert_public_copy_20260917')
    active = state.get('active_execution', {})
    need(access == active.get('nih_expert_label_access_review'), f'Access active/completed mismatch in {state_name}')
    need(public == active.get('nih_expert_label_public_copy'), f'Public-copy active/completed mismatch in {state_name}')
    return state, access, public


def main():
    access_record = read_json(ACCESS / 'access_record.json')
    access_verification = read_json(ACCESS / 'record_verification.json')
    public_record = read_json(PUBLIC / 'record.json')
    public_verification = read_json(PUBLIC / 'record_verification.json')

    access = access_record['status']
    public = public_record['status']
    recovery = public['recovery']

    need(access_verification['status'] == 'PASS', 'Access record verification did not pass')
    need(public_verification['status'] == 'PASS', 'Public-copy record verification did not pass')
    need(access['type'] == 'access_and_document_inventory_not_model_experiment', 'Unexpected access record type')
    need(public['type'] == 'public_derived_label_acquisition_not_model_experiment', 'Unexpected public-copy record type')
    need(access['report'] == 'NIH_EXPERT_LABEL_ACCESS_20260917.md', 'Access report pointer mismatch')
    need(public['report'] == 'NIH_EXPERT_LABEL_PUBLIC_COPY_20260917.md', 'Public-copy report pointer mismatch')
    need(access['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Access actual-result pointer mismatch')
    need(public['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Public-copy actual-result pointer mismatch')

    need(access['expert_csv_acquired'] is False, 'Official expert CSV should not be marked acquired')
    need(access['patient_overlap_computed'] is False, 'Patient overlap should remain uncomputed')
    need(access['anonymous_object_get_http_status'] == 403, 'Expected anonymous object 403 record')
    need(access['user_reports_authenticated_browser_url_http403'] is True, 'Authenticated browser 403 report changed')
    need(access['official_download_form_submissions'] == 1, 'Official form-submission count changed')
    need(access['identity_or_contact_submitted'] is False, 'Identity/contact should not be marked submitted')
    need(access['author_emails_sent'] == 0, 'Author email should not be marked sent')
    need(access['specific_label_file_uniquely_required'] is False, 'Specific file uniqueness claim changed')
    need(access['published_positive_image_counts'] == {'Emphysema': 7, 'Pneumothorax': 136, 'Effusion': 226}, 'Published count summary changed')
    need(access['new_model_inference'] == access['new_generation'] == access['new_training'] == access['new_DP'] == 0, 'Access record should not add experiments')
    need(access['reserved_confirmation_model_or_pixel_consumption'] is False, 'Reserved confirmation should remain unconsumed')
    need(sha256(R / access['report']) == access['report_sha256'], 'Access report hash mismatch')
    need((R / access['author_email_draft']).exists(), 'Prepared inquiry draft missing')

    need(public['author_contact_only_claim_retracted'] is True, 'Author-contact-only claim should be retracted')
    need(public['patient_role_overlap_pending'] is True, 'Patient-role overlap should remain pending')
    need(recovery['status'] == 'PUBLIC_DERIVED_COPY_ACQUIRED_AND_CROSSCHECKED', 'Recovery status changed')
    need(recovery['images'] == 810 and recovery['unique_patients'] == 532, 'Public-copy image/patient counts changed')
    need(recovery['published_all14_positive_counts_match'] is True, 'Published count cross-check changed')
    need(recovery['positive_image_counts']['Emphysema'] == 7, 'Recovered emphysema image count changed')
    need(recovery['positive_image_counts']['Pneumothorax'] == 136, 'Recovered pneumothorax image count changed')
    need(recovery['positive_patient_counts']['Emphysema'] == 7, 'Recovered emphysema patient count changed')
    need(recovery['positive_patient_counts']['Pneumothorax'] == 86, 'Recovered pneumothorax patient count changed')
    need(recovery['author_repository_810_image_ID_set_match'] is True, '810 image-id set match changed')
    need(recovery['original_source_columns_only'] is True, 'Original column constraint changed')
    need(recovery['merged_optimal_columns_used'] is False, 'Merged optimal columns should not be used')
    need(recovery['official_original_csv_downloaded'] is False, 'Official CSV should not be marked downloaded')
    need(recovery['official_byte_identical_verification'] is False, 'Byte-identical official verification should remain false')
    need(recovery['all14_individual_reader_labels_acquired'] is False, 'Individual reader labels should not be acquired')
    need(recovery['existing_cohort_overlap_computed'] is False, 'Existing cohort overlap should remain pending')
    need(recovery['patient_images_opened'] is False, 'Patient images should not be opened')
    need(recovery['new_model_inference'] == recovery['new_training'] == recovery['new_generation'] == recovery['new_DP'] == 0, 'Public-copy recovery should not add experiments')
    need(recovery['author_contact_required_for_this_copy'] is False and recovery['author_emails_sent'] == 0, 'Author contact/email flags changed')
    need(sha256(R / public['report']) == public_record['report_sha256'], 'Public-copy report hash mismatch')

    for source in recovery['sources']:
        need(source['status'] == 200, 'Public mirror source status changed')
        need(source['bytes'] > 0 and len(source['sha256']) == 64, 'Public mirror source metadata malformed')
        need(not (PUBLIC / source['local_file']).exists(), 'Downloaded mirror data file should not be included')
    need(not (PUBLIC / recovery['export_file']).exists(), 'Derived export CSV should not be included')

    frozen = access_record['preserved_historical_report_hashes']
    for filename, expected in frozen.items():
        need(sha256(R / filename) == expected, f'Historical report hash changed: {filename}')

    for state_name in ['research_state.json', 'patient_baseline_spec.json']:
        state, state_access, state_public = state_records(state_name)
        need(state_access == access, f'Access record mismatch in {state_name}')
        need(state_public == public, f'Public-copy record mismatch in {state_name}')
        need(state['current_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Actual result pointer mismatch in {state_name}')
        need(state['current_planning_report'] == 'NIH_EXPERT_LABEL_PUBLIC_COPY_20260917.md', f'Planning pointer mismatch in {state_name}')
        need(state['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Last actual pointer mismatch in {state_name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {state_name}')

    checked_links = 0
    omitted_links = 0
    for page in ['nih_expert_label_access.html', 'nih_expert_label_public_copy.html']:
        public_count, omitted_count = check_local_links(page)
        checked_links += public_count
        omitted_links += omitted_count
    index = (R / 'index.html').read_text(encoding='utf-8-sig')
    need('nih_expert_label_access.html' in index and 'nih_expert_label_public_copy.html' in index, 'Index missing NIH expert-label links')

    result = {
        'status': 'PASS_PUBLIC_NIH_EXPERT_LABEL_ACCESS_BUNDLE',
        'checked_links': checked_links,
        'omitted_data_links': omitted_links,
        'official_csv_acquired': access['expert_csv_acquired'],
        'author_emails_sent': access['author_emails_sent'],
        'public_copy_images': recovery['images'],
        'public_copy_unique_patients': recovery['unique_patients'],
        'last_actual_model_result_report': access['last_actual_model_result_report'],
        'new_model_inference': access['new_model_inference'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
