"""Verify the public measurement-claim review bundle.

This review is a source/claim interpretation correction, not a new model run.
The downloaded paper PDF and private execution artifacts are intentionally not
included in the public GitHub bundle. This checker validates the public report,
HTML page, state pointers, and review record.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
P = R / 'spec_sources' / 'measurement_claim_review_20260917'
REPORT = R / 'MEASUREMENT_CLAIM_REVIEW_20260917.md'
HTML = R / 'measurement_claim_review.html'


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


def check_local_links(path):
    parser = LinkCollector()
    body = Path(path).read_text(encoding='utf-8-sig')
    need('\ufffd' not in body, f'Invalid replacement character in {path.name}')
    parser.feed(body)
    checked = 0
    omitted_private_or_pdf = 0
    for raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        normalized = unquote(parsed.path).replace('\\', '/')
        if '_reports/' in normalized or '_models/' in normalized or normalized.endswith('.pdf'):
            omitted_private_or_pdf += 1
            continue
        target = (R / normalized).resolve()
        need(target.exists(), f'Broken local link: {raw}')
        checked += 1
    return checked, omitted_private_or_pdf


def main():
    review_record = read_json(P / 'review_record.json')
    verification = read_json(P / 'record_verification.json')
    pre_bindings = read_json(P / 'pre_review_bindings.json')
    review = review_record['review']

    need(verification['status'] == 'PASS', 'Stored record verification did not pass')
    need(review['type'] == 'source_and_measurement_claim_review_not_model_experiment', 'Unexpected review type')
    need(review['judgment'] == 'positive_for_existing_measurement_routes_not_private_efficacy', 'Unexpected judgment')
    need(review['current_step'] == 2, 'Unexpected stage')
    need(review['track'] == 1, 'Unexpected track')
    need(review['report'] == 'MEASUREMENT_CLAIM_REVIEW_20260917.md', 'Report pointer mismatch')
    need(review['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', 'Actual result pointer changed')
    need(review['new_model_inference'] == 0 and review['new_generation'] == 0 and review['new_training'] == 0 and review['new_DP'] == 0, 'Review should not add experiments')
    need(review['reserved_confirmation_consumed'] is False, 'Reserved confirmation should remain unused')
    need(review['external_messages_or_access_form_submissions'] == 0, 'External submission count changed')
    need(review['universal_measurement_impossibility_established'] is False, 'Measurement impossibility overclaim')
    need(review['weak_label_error_as_main_cause_established'] is False, 'Weak-label cause overclaim')
    need(review['private_generation_utility_established'] is False, 'Private utility overclaim')
    need(review['clinical_generation_correctness_established'] is False, 'Clinical correctness overclaim')
    need(review['expert_label_files_acquired'] is False, 'Expert labels should not be marked acquired')
    need(review['NIH_all14_expert_images_documented'] == 810, 'NIH all14 count changed')
    need(review['VinDr_majority_label_counts_paper_table2']['Emphysema'] == {'train': 14, 'test': 3}, 'VinDr emphysema counts changed')
    need(review['VinDr_majority_label_counts_paper_table2']['Pneumothorax'] == {'train': 58, 'test': 18}, 'VinDr pneumothorax counts changed')

    need(sha256(REPORT) == review['report_sha256'] == verification['new_review_sha256'], 'Report hash mismatch')
    need(sha256(HTML) == verification['new_html_sha256'], 'HTML hash mismatch')
    need((P / 'roentgen_2211.12737.pdf').exists() is False, 'Downloaded paper PDF should not be included')

    for name in ['research_state.json', 'patient_baseline_spec.json']:
        state = read_json(R / name)
        need(state['current_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Actual result pointer mismatch in {name}')
        need(state['current_planning_report'] == 'MEASUREMENT_CLAIM_REVIEW_20260917.md', f'Planning pointer mismatch in {name}')
        need(state['last_actual_model_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Last actual pointer mismatch in {name}')
        need(state['active_execution']['status'] == 'no_running_execution', f'Unexpected active run in {name}')
        need(state['completed_measurement_claim_review_20260917'] == review, f'Review record mismatch in {name}')
        need(state['active_execution']['measurement_claim_review'] == review, f'Active review record mismatch in {name}')
        before = review_record['pre_review_state_pointers'][name]
        need(before['current_result_report'] == 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md', f'Pre-review actual pointer mismatch in {name}')

    checked, omitted = check_local_links(HTML)
    need('measurement_claim_review.html' in (R / 'index.html').read_text(encoding='utf-8-sig'), 'Index is missing measurement review link')
    need('MEASUREMENT_CLAIM_REVIEW_20260917.md' in pre_bindings.get('expected_new_report', 'MEASUREMENT_CLAIM_REVIEW_20260917.md'), 'Pre-binding packet malformed')

    result = {
        'status': 'PASS_PUBLIC_MEASUREMENT_CLAIM_REVIEW_BUNDLE',
        'checked_links': checked,
        'omitted_private_or_pdf_links': omitted,
        'report_sha256': review['report_sha256'],
        'html_sha256': verification['new_html_sha256'],
        'last_actual_model_result_report': review['last_actual_model_result_report'],
        'new_model_inference': review['new_model_inference'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
