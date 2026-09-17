from __future__ import annotations
import hashlib, json, re
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
SPEC = R / 'spec_sources'

def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8-sig')

def read_json(path: Path):
    return json.loads(read_text(path))

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = R / 'TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_RESULTS_20260917.md'
    protocol = R / 'TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_PROTOCOL_20260917.md'
    report_html = R / 'track1_downstream_all_arm_replay_results.html'
    protocol_html = R / 'track1_downstream_all_arm_replay_protocol.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    record_path = SPEC / 'corrected_all_arm_replay_record_20260917.json'
    verification_path = SPEC / 'corrected_all_arm_replay_report_verification_20260917.json'

    for path in (report, protocol, report_html, protocol_html, index, state_path, record_path, verification_path):
        need(path.exists(), f'missing {path}')

    record = read_json(record_path)
    verification = read_json(verification_path)
    state = read_json(state_path)
    report_text = read_text(report)
    protocol_text = read_text(protocol)
    html_text = read_text(report_html)
    index_text = read_text(index)

    need(record['report'] == report.name, 'record report mismatch')
    need(record['judgment'] == 'positive_corrected_all_arm_one_step_integration_not_efficacy', 'judgment changed')
    need(record['integration_gate_passed'] is True, 'integration gate not passed')
    need(record['optimizer_updates'] == 14, 'optimizer update count changed')
    need(record['prior_separate_profile_updates'] == 100, 'prior profile update count changed')
    need(record['old98_still_invalid'] is True, 'old 98 invalid flag changed')
    need(record['arms'] == 7 and record['repeats'] == 2 and record['steps_per_run'] == 1, '7 arm x 2 x 1step shape changed')
    need(record['selected_source_files'] == 61, 'selected source file count changed')
    need(record['independent_GPU_preprocessing_exact_arms'] == 7, 'GPU preprocessing replay count changed')
    need(record['exact_final_state_replays'] == 7, 'final-state replay count changed')
    need(record['changed_trainable_parameter_tensors_each_run'] == 62, 'trainable tensor count changed')
    need(record['actual_source_array_binding'] is True, 'actual source array binding changed')
    need(record['legacy_classifier_CLI_blocked'] is True, 'legacy CLI block changed')
    need(record['legacy_data_train_imports_blocked'] is True, 'legacy import block changed')
    need(record['new_generation'] == 0, 'new generation count changed')
    need(record['AUROC_AP_computed'] is False, 'AUROC/AP scope changed')
    need(record['new_DP'] == 0, 'new DP scope changed')
    need(record['expert_pixels'] == 0 and record['expert_predictions'] == 0, 'expert final scope changed')
    need(record['reserved_pixels'] == 0 and record['reserved_predictions'] == 0, 'reserved scope changed')
    need(record['long_run_convergence_validated'] is False, 'long-run convergence scope changed')
    need(record['private_utility_established'] is False, 'private utility scope changed')
    need(record['final_ready'] is False, 'final-ready scope changed')
    need(record['full512_bank_executed'] is False, 'full512 scope changed')
    need(record['full21_classifier_runs_executed'] is False, 'full21 classifier scope changed')

    need(verification['status'] == 'PASS_REPORT_STATE_SOURCE_ARCHIVE_AND_LINK_BINDINGS', 'stored verification status changed')
    need(verification['corrected_integration_gate'] == 'PASS', 'stored integration gate changed')
    need(verification['new_optimizer_updates'] == 14, 'stored update count changed')
    need(verification['old98_still_invalid'] is True, 'stored old98 flag changed')
    need(verification['actual_corrected_module_files_verified'] is True, 'stored source verification changed')
    need(verification['state_synchronized'] is True, 'stored state sync changed')
    need(verification['new_model_runs_during_this_report_check'] == 0, 'report check model-run scope changed')
    need(verification['expert_reserved_access'] == 0, 'expert/reserved access changed')

    need(state['current_result_report'] == report.name, 'state current report mismatch')
    need(state['last_actual_model_result_report'] == report.name, 'state last actual result mismatch')
    need('completed_corrected_all_arm_replay_20260917' in state, 'state missing completed replay record')
    state_record = state['completed_corrected_all_arm_replay_20260917']
    need(state_record['optimizer_updates'] == record['optimizer_updates'], 'state record update count mismatch')
    need(state_record['judgment'] == record['judgment'], 'state record judgment mismatch')

    need('track1_downstream_all_arm_replay_results.html' in index_text, 'index missing all-arm result link')
    need('track1_downstream_all_arm_replay_protocol.html' in html_text, 'result page missing all-arm protocol link')
    need('TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_PROTOCOL_20260917.md' in report_text, 'markdown report missing protocol link')
    need('새 생성0회' in report_text or '새 생성0' in report_text, 'report missing no-new-generation statement')
    need('14optimizer update' in report_text or '14회' in report_text, 'report missing 14-update statement')
    need('잘못된98회' in report_text or '잘못된 98회' in report_text, 'report missing invalid-old98 statement')
    need('사적 효용' in report_text and '미판정' in report_text, 'report missing utility-not-established scope')
    need('DP' in report_text and ('0' in report_text or '없' in report_text), 'report missing DP scope statement')
    need('7군' in protocol_text and '1step' in protocol_text, 'protocol missing execution shape')

    for rel in ('all_arm_replay.py', 'run_v2.py', 'verify_all_arm_replay.py', 'train_v2.py', 'data_v2.py'):
        need((REPO / 'code_working' / 'downstream_utility' / rel).exists(), f'missing corrected code file {rel}')

    omitted_private_links = len(re.findall(r'\.\./\.\./code_working/_reports/downstream_all_arm_replay_20260917_v1/[^)"\s<]+', report_text + '\n' + html_text))
    need(omitted_private_links >= 3, 'expected private artifact links were not detected')
    need(not (REPO / 'code_working' / '_reports' / 'downstream_all_arm_replay_20260917_v1').exists(), 'private _reports artifact directory should not be in public bundle')

    result = {
        'status': 'PASS_PUBLIC_DOWNSTREAM_ALL_ARM_REPLAY_BUNDLE',
        'optimizer_updates': record['optimizer_updates'],
        'arms': record['arms'],
        'repeats': record['repeats'],
        'steps_per_run': record['steps_per_run'],
        'old98_still_invalid': record['old98_still_invalid'],
        'new_generation': record['new_generation'],
        'AUROC_AP_computed': record['AUROC_AP_computed'],
        'new_DP': record['new_DP'],
        'expert_pixels': record['expert_pixels'],
        'reserved_pixels': record['reserved_pixels'],
        'private_utility_established': record['private_utility_established'],
        'omitted_private_artifact_links': omitted_private_links,
        'record_sha256': sha(record_path),
        'report_sha256': sha(report),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
