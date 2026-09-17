"""Verify the public PadChest validation bundle without private _reports artifacts.

The original execution directory under code_working/_reports is intentionally
excluded from the GitHub context bundle. This checker validates the public
record that remains in the bundle: report hashes, figure hashes, state-file
consistency, and local HTML links.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

R = Path(__file__).resolve().parents[1]
V = R / 'spec_sources' / 'padchest_report_verification_20260917.json'


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
        attrs = dict(attrs)
        for key in ('href', 'src'):
            value = attrs.get(key)
            if value:
                self.links.append(value)


def check_local_links(filename):
    parser = LinkCollector()
    parser.feed((R / filename).read_text(encoding='utf-8'))
    checked = 0
    for raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (R / unquote(parsed.path)).resolve()
        need(target.exists(), f'Broken local link in {filename}: {raw}')
        checked += 1
    return checked


def state_padchest_record(state_name):
    state = read_json(R / state_name)
    record = state.get('active_execution', {}).get('padchest_validation')
    if record is None:
        record = state.get('completed_padchest_validation_20260917')
    need(isinstance(record, dict), f'Missing PadChest record in {state_name}')
    return record


def main():
    verification = read_json(V)
    need(verification['status'] == 'PASS_PADCHEST_REPORT_STATE_AND_HISTORY', 'Unexpected verification status')
    need(verification['measurement_gate_passed'] is False, 'Measurement gate should remain failed')
    need(verification['primary_patients'] == 80, 'Primary patient count changed')
    need(verification['total_patients'] == 309, 'Total validation patient count changed')
    need(verification['remaining_development_patients'] == 4133, 'Remaining development count changed')
    need(verification['reserved_confirmation_consumed'] is False, 'Reserved confirmation set should be unused')

    report = R / 'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md'
    need(sha256(report) == verification['report_sha256'], 'PadChest report hash mismatch')
    for ext, expected in verification['figure_sha256'].items():
        figure = R / 'assets' / f'padchest_validation_scores_20260917.{ext}'
        need(sha256(figure) == expected, f'PadChest figure hash mismatch: {ext}')

    records = [state_padchest_record('research_state.json'), state_padchest_record('patient_baseline_spec.json')]
    for record in records:
        need(record['status'] == 'NEGATIVE_PADCHEST_CONDITION_DISCRIMINATION', 'Unexpected PadChest decision')
        need(record['report'] == 'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md', 'State report mismatch')
        need(record['protocol'] == 'TRACK1_PADCHEST_VALIDATION_PROTOCOL_20260917.md', 'State protocol mismatch')
        need(record['actual_validation_patients'] == verification['total_patients'], 'State total count mismatch')
        need(record['primary_new_development_patients'] == verification['primary_patients'], 'State primary count mismatch')
        need(record['remaining_target_development_patients'] == verification['remaining_development_patients'], 'State remaining count mismatch')
        need(record['evaluator_validated'] is False, 'Evaluator should not be validated')
        need(record['private_generation_utility_established'] is False, 'Private generation utility should not be established')
        need(record['DP_allowed_to_expand'] is False, 'DP expansion should remain blocked')
        need(record['automatic_targeted_generation_allowed'] is False, 'Automatic targeted generation should remain blocked')
        need(record['reserved_confirmation_consumed'] is False, 'Reserved confirmation flag mismatch')

    checked_links = 0
    for page in ['track1_padchest_validation_protocol.html', 'track1_padchest_validation_results.html']:
        checked_links += check_local_links(page)
    need('track1_padchest_validation_results.html' in (R / 'index.html').read_text(encoding='utf-8'), 'Index is missing PadChest result link')

    result = {
        'status': 'PASS_PUBLIC_PADCHEST_BUNDLE',
        'checked_links': checked_links,
        'report_sha256': verification['report_sha256'],
        'figure_sha256': verification['figure_sha256'],
        'state_files_checked': ['research_state.json', 'patient_baseline_spec.json'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
