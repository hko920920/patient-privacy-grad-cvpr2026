from __future__ import annotations
import hashlib, json, re
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
SPEC = R / 'spec_sources'

def text(path: Path) -> str:
    return path.read_text(encoding='utf-8-sig')

def data(path: Path):
    return json.loads(text(path))

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = R / 'TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md'
    protocol = R / 'TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md'
    report_html = R / 'track1_downstream_development_results.html'
    protocol_html = R / 'track1_downstream_development_protocol.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    record_path = SPEC / 'downstream_development_record_20260917.json'
    verification_path = SPEC / 'downstream_development_report_verification_20260917.json'

    for path in (report, protocol, report_html, protocol_html, index, state_path, record_path, verification_path):
        need(path.exists(), f'missing public file: {path}')

    record = data(record_path)
    verification = data(verification_path)
    state = data(state_path)
    report_text = text(report)
    protocol_text = text(protocol)
    html_text = text(report_html)
    index_text = text(index)

    need(record['report'] == report.name, 'record report mismatch')
    need(record['judgment'] == 'negative_private_synthetic_development_gate', 'judgment changed')
    need(record['engineering_gate_passed'] is False, 'gate unexpectedly passed')
    need(record['generated_images'] == 512, 'generated image count changed')
    need(record['calibration_updates'] == 1600, 'calibration update count changed')
    need(record['development_updates'] == 8400, 'development update count changed')
    need(record['instrumentation_updates'] == 4, 'instrumentation update count changed')
    need(record['development_runs'] == 21, 'development run count changed')
    need(record['new_DP'] == 0, 'new DP scope changed')
    need(record['expert_pixels'] == 0 and record['reserved_pixels'] == 0, 'expert/reserved scope changed')
    need(record['final_ready'] is False, 'final-ready scope changed')

    means = record['means']
    for arm in ('R0', 'R1', 'S0', 'S1', 'S2', 'S3', 'Dreal'):
        need(arm in means, f'missing mean arm {arm}')
        need('AUROC' in means[arm] and 'AP' in means[arm], f'missing metrics for {arm}')
    need(means['Dreal']['AUROC'] > means['S2']['AUROC'], 'Dreal AUROC relation changed')
    need(means['Dreal']['AP'] > means['S2']['AP'], 'Dreal AP relation changed')
    need(record['gate']['S2']['passed'] is False and record['gate']['S3']['passed'] is False, 'S2/S3 gate status changed')

    need(verification['status'] == 'PASS', 'stored report verification did not pass')
    need(verification['states_synced'] is True, 'stored state sync changed')
    need(verification['old192_PNG_and_trace_hashes_preserved'] is True, 'old192 preservation flag changed')
    need(verification['new_model_forward'] == 0, 'report verification model-forward scope changed')
    need(verification['new_training'] == 0, 'report verification training scope changed')
    need(verification['expert_pixels'] == 0 and verification['reserved_pixels'] == 0, 'stored expert/reserved scope changed')
    need(verification['broken_links'] == [], 'stored report verification has broken links')

    need(state['current_result_report'] == report.name, 'state current report mismatch')
    need(state['last_actual_model_result_report'] == report.name, 'state last actual result mismatch')
    need('completed_downstream_development_20260917' in state, 'state missing development record')
    state_record = state['completed_downstream_development_20260917']
    need(state_record['judgment'] == record['judgment'], 'state record judgment mismatch')
    need(state_record['generated_images'] == record['generated_images'], 'state generated image count mismatch')
    need(state_record['development_updates'] == record['development_updates'], 'state development update count mismatch')

    need('track1_downstream_development_results.html' in index_text, 'index missing development result link')
    need('track1_downstream_development_protocol.html' in html_text, 'result page missing development protocol link')
    need('TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md' in report_text, 'markdown report missing protocol link')
    need('512' in report_text and '8400' in report_text, 'report missing development scale')
    need('negative_private_synthetic_development_gate' in report_text or '실패' in report_text or '미통과' in report_text, 'report missing negative gate judgment')
    need('DP' in report_text and ('0' in report_text or '없' in report_text), 'report missing DP scope')
    need('expert' in report_text.lower() and ('0' in report_text or '미사용' in report_text), 'report missing expert scope')
    need('21' in protocol_text and ('classifier' in protocol_text.lower() or 'runs' in protocol_text.lower() or '군' in protocol_text), 'protocol missing run structure')

    for rel in ('development_v1.py', 'run_development_v1.py', 'verify_development_v1.py'):
        need((REPO / 'code_working' / 'downstream_utility' / rel).exists(), f'missing development code file {rel}')

    private_refs = len(re.findall(r'\.\./\.\./code_working/_reports/downstream_development_20260917_v1/[^)"\s<]+', report_text + '\n' + html_text))
    need(private_refs >= 4, 'expected omitted private artifact references not found')
    need(not (REPO / 'code_working' / '_reports' / 'downstream_development_20260917_v1').exists(), 'private _reports artifact directory should not be in public bundle')

    result = {
        'status': 'PASS_PUBLIC_DOWNSTREAM_DEVELOPMENT_BUNDLE',
        'judgment': record['judgment'],
        'engineering_gate_passed': record['engineering_gate_passed'],
        'generated_images': record['generated_images'],
        'calibration_updates': record['calibration_updates'],
        'development_updates': record['development_updates'],
        'development_runs': record['development_runs'],
        'S2_passed': record['gate']['S2']['passed'],
        'S3_passed': record['gate']['S3']['passed'],
        'Dreal_AUROC': record['means']['Dreal']['AUROC'],
        'S2_AUROC': record['means']['S2']['AUROC'],
        'expert_pixels': record['expert_pixels'],
        'reserved_pixels': record['reserved_pixels'],
        'new_DP': record['new_DP'],
        'final_ready': record['final_ready'],
        'omitted_private_artifact_refs': private_refs,
        'record_sha256': sha(record_path),
        'report_sha256': sha(report),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

