from __future__ import annotations
import hashlib, json
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
    report = R / 'TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md'
    report_html = R / 'track1_current_head_branch_closure.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    patient_spec_path = R / 'patient_baseline_spec.json'
    record_path = SPEC / 'current_head_branch_closure_20260917.json'
    verification_path = SPEC / 'current_head_branch_closure_verification_20260917.json'
    development_report = R / 'TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md'
    development_protocol = R / 'TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md'

    for path in (report, report_html, index, state_path, patient_spec_path, record_path, verification_path, development_report, development_protocol):
        need(path.exists(), f'missing public file: {path}')

    record = read_json(record_path)
    verification = read_json(verification_path)
    state = read_json(state_path)
    patient_spec = read_json(patient_spec_path)
    report_text = read_text(report)
    html_text = read_text(report_html)
    index_text = read_text(index)

    need(record['schema'] == 'current-head-branch-closure/v1', 'schema changed')
    need(record['report'] == report.name, 'record report mismatch')
    need(record['report_sha256'] == sha(report), 'record report sha mismatch')
    need(record['decision'] == 'closed_negative_development_gate', 'closure decision changed')
    need(record['actual_result_report'] == development_report.name, 'actual result report changed')
    need(record['closed_scope'] == 'E4/full64/current-objective/128-per-method/one-bank/fixed-compute-ResNet18', 'closed scope changed')
    need(record['head_capacity_isolated_as_cause'] is False, 'head-only cause flag changed')
    need(record['pooled_dilution_mechanism_established'] is False, 'pooled dilution flag changed')
    need(record['DP_expansion'] is False, 'DP expansion flag changed')
    need(record['final_ready'] is False, 'final-ready flag changed')
    need(record['next_experiment_selected'] is False, 'next experiment flag changed')
    need(record['new_model_forwards'] == 0, 'new model-forward count changed')
    need(record['new_training'] == 0, 'new training count changed')
    need(record['new_generation'] == 0, 'new generation count changed')
    need(record['new_patient_pixel_reads'] == 0, 'new patient pixel count changed')
    need(record['expert_or_reserved_opened'] is False, 'expert/reserved flag changed')

    counts = record['role_counts']
    need(counts['public_head']['patients'] == 32 and counts['public_head']['images'] == 64, 'public head counts changed')
    need(counts['downstream_public']['patients'] == 672 and counts['downstream_public']['images'] == 813, 'downstream public counts changed')
    need(counts['private_head']['patients'] == 80 and counts['private_head']['images'] == 320, 'private head counts changed')
    need(abs(record['pooled_patient_weights']['public'] - 2/7) < 1e-12, 'public pooled weight changed')
    need(abs(record['pooled_patient_weights']['private'] - 5/7) < 1e-12, 'private pooled weight changed')

    deltas = record['recomputed_mean_deltas']
    need(deltas['Dreal-R1']['AUROC'] > 0.05, 'Dreal-R1 AUROC delta changed')
    need(deltas['S2-R1']['AUROC'] < 0.01, 'S2-R1 gate relation changed')
    need(deltas['S3-R1']['AUROC'] < 0, 'S3-R1 relation changed')
    need(deltas['S2-R0']['AUROC'] < 0, 'S2-R0 relation changed')

    need(verification['status'] == 'PASS_DOCUMENT_METADATA_AND_STATE_BINDING', 'stored verification status changed')
    need(verification['report_sha256'] == sha(report), 'stored report sha mismatch')
    need(verification['new_model_forwards'] == 0, 'stored model-forward scope changed')
    need(verification['new_training'] == 0, 'stored training scope changed')
    need(verification['new_generation'] == 0, 'stored generation scope changed')
    need(verification['new_patient_pixel_reads'] == 0, 'stored patient-pixel scope changed')
    need(verification['expert_or_reserved_opened'] is False, 'stored expert/reserved flag changed')
    need(verification['remote_push'] is False, 'stored remote push flag should be historical false')

    need(state['current_result_report'] == development_report.name, 'state should keep downstream development as current result')
    need(state['last_actual_model_result_report'] == development_report.name, 'state should keep downstream development as last actual model result')
    need('current_head_branch_closure_20260917' in state, 'state missing closure record')
    need(state['current_head_branch_closure_20260917']['status'] == record['decision'], 'state closure decision mismatch')
    need(state['current_head_branch_closure_20260917']['DP_expansion'] is False and state['current_head_branch_closure_20260917']['final_ready'] is False, 'state closure scope changed')
    need('current_head_branch_closure_20260917' in patient_spec, 'patient spec missing closure record')

    need('track1_current_head_branch_closure.html' in index_text, 'index missing closure page link')
    need('TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md' in report_text, 'markdown report missing development result link')
    need('track1_downstream_development_results.html' in html_text, 'HTML report missing development result link')
    need('DP 확대를 종료' in report_text, 'report missing DP expansion closure statement')
    need('새 모델 추론' in report_text and '없다' in report_text, 'report missing no-new-execution statement')
    need('Expert final' in report_text and 'reserved' in report_text, 'report missing expert/reserved preservation statement')
    need('후속 방법이나 다른 논문 주제를 자동 선택하지 않았다' in report_text, 'report missing no-next-topic statement')

    need(not (REPO / 'code_working' / '_reports').exists(), 'private _reports directory should not be present in public bundle')

    result = {
        'status': 'PASS_PUBLIC_CURRENT_HEAD_BRANCH_CLOSURE_BUNDLE',
        'decision': record['decision'],
        'actual_result_report': record['actual_result_report'],
        'closed_scope': record['closed_scope'],
        'DP_expansion': record['DP_expansion'],
        'final_ready': record['final_ready'],
        'next_experiment_selected': record['next_experiment_selected'],
        'new_model_forwards': record['new_model_forwards'],
        'new_training': record['new_training'],
        'new_generation': record['new_generation'],
        'new_patient_pixel_reads': record['new_patient_pixel_reads'],
        'expert_or_reserved_opened': record['expert_or_reserved_opened'],
        'Dreal_R1_AUROC_delta': deltas['Dreal-R1']['AUROC'],
        'S2_R1_AUROC_delta': deltas['S2-R1']['AUROC'],
        'report_sha256': sha(report),
        'record_sha256': sha(record_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

