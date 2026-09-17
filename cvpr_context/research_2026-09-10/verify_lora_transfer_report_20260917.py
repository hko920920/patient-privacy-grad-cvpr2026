from __future__ import annotations
import hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parent
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

def mean_score(record, arm, metric):
    vals = [row[metric] for row in record['result']['scores'][arm]]
    return sum(vals) / len(vals)

def main():
    report = R / 'TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'
    protocol = R / 'TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md'
    html = R / 'track1_lora_transfer_comparison_results.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    patient_spec_path = R / 'patient_baseline_spec.json'
    record_path = SPEC / 'lora_transfer_execution_record_20260917.json'
    verification_path = SPEC / 'lora_transfer_report_verification_20260917.json'
    manifest_path = SPEC / 'lora_transfer_generation_manifest_20260917.json'
    metrics_png = SPEC / 'lora_transfer_development_metrics_20260917.png'
    loss_png = SPEC / 'lora_transfer_loss_20260917.png'
    grid_png = SPEC / 'lora_transfer_first4blocks_20260917.png'

    required = (report, protocol, html, index, state_path, patient_spec_path, record_path, verification_path, manifest_path, metrics_png, loss_png, grid_png)
    for path in required:
        need(path.exists(), f'missing public file: {path}')

    record = read_json(record_path)
    verification = read_json(verification_path)
    manifest = read_json(manifest_path)
    state = read_json(state_path)
    patient_spec = read_json(patient_spec_path)
    report_text = read_text(report)
    html_text = read_text(html)
    index_text = read_text(index)

    need(record['schema'] == 'lora-transfer-public-record/v1', 'record schema changed')
    need(record['status'] == 'completed_verified', 'record status changed')
    result = record['result']
    need(result['status'] == 'DEVELOPMENT_EFFICACY_COMPLETE_PENDING_INDEPENDENT_VERIFY', 'result status changed')
    need(result['engineering_gate_passed'] is False, 'engineering gate unexpectedly passed')
    need(result['new_generator_updates'] == 896, 'generator update count changed')
    need(result['new_images'] == 256, 'generated image count changed')
    need(result['new_classifier_updates'] == 2400, 'classifier update count changed')
    need(result['extra_classifier_updates'] == 6, 'extra classifier update count changed')
    need(result['extra_generation_decodes'] == 2, 'extra generation decode count changed')
    need(result['new_DP'] == 0, 'new DP scope changed')
    need(result['expert_pixels'] == 0 and result['reserved_pixels'] == 0, 'expert/reserved scope changed')
    need(result['final_ready'] is False, 'final-ready flag changed')
    need(result['current_full64_branch_closed'] is True, 'closed branch flag changed')
    need(result['one_training_trajectory_per_arm'] is True, 'training trajectory scope changed')
    need(result['one_bank_per_arm'] is True, 'bank scope changed')
    need(result['classifier_seeds'] == 3, 'classifier seed count changed')
    need(result['weak_label_development_only'] is True, 'development label scope changed')

    scores = result['scores']
    for arm in ('L_public', 'L_pooled', 'R0', 'R1', 'S0', 'S1', 'S2', 'S3', 'Dreal'):
        need(arm in scores, f'missing score arm {arm}')
        need(len(scores[arm]) == 3, f'expected three seeds for {arm}')
        need({row['seed'] for row in scores[arm]} == {11, 23, 37}, f'seed set changed for {arm}')
    l_public_auc = mean_score(record, 'L_public', 'AUROC')
    l_pooled_auc = mean_score(record, 'L_pooled', 'AUROC')
    l_public_ap = mean_score(record, 'L_public', 'AP')
    l_pooled_ap = mean_score(record, 'L_pooled', 'AP')
    need(abs(l_public_auc - 0.5582343018341279) < 1e-12, 'L_public AUROC mean changed')
    need(abs(l_pooled_auc - 0.5435871836631491) < 1e-12, 'L_pooled AUROC mean changed')
    need(abs(l_public_ap - 0.052514726103239984) < 1e-12, 'L_public AP mean changed')
    need(abs(l_pooled_ap - 0.05238390702193102) < 1e-12, 'L_pooled AP mean changed')

    comps = result['comparisons']
    need(comps['L_public']['AUROC_delta'] < 0, 'L_pooled-L_public AUROC relation changed')
    need(comps['L_public']['positive_seeds'] == 1, 'L_pooled-L_public positive seed count changed')
    need(comps['R1']['AUROC_delta'] < 0 and comps['R1']['positive_seeds'] == 1, 'L_pooled-R1 relation changed')
    need(comps['R0']['AUROC_delta'] < 0 and comps['R0']['positive_seeds'] == 1, 'L_pooled-R0 relation changed')

    v_inner = record['verification']
    need(v_inner['status'] == 'PASS_INTEGRITY_NOT_AUTOMATIC_EFFICACY', 'inner verification status changed')
    need(v_inner['checks'] == 46849, 'inner verification check count changed')
    need(v_inner['engineering_gate_passed'] is False, 'inner engineering gate changed')
    need(v_inner['bootstrap_repeats'] == 2000, 'bootstrap repeat count changed')
    need(v_inner['expert_pixels'] == 0 and v_inner['reserved_pixels'] == 0, 'inner expert/reserved scope changed')
    need(v_inner['new_DP'] == 0 and v_inner['final_ready'] is False, 'inner DP/final scope changed')
    need(v_inner['patient_cluster_intervals_95']['L_pooled-L_public']['AUROC'][0] < 0, 'bootstrap lower bound relation changed')
    need(v_inner['patient_cluster_intervals_95']['L_pooled-L_public']['AUROC'][1] > 0, 'bootstrap upper bound relation changed')

    need(record['additional_input_verification']['status'] == 'PASS_INDEPENDENT_NEW_LORA_RAW_PNG_TO_GPU_INPUT', 'additional input verification status changed')
    need(record['additional_input_verification']['selected_files'] == 43, 'additional input selected file count changed')
    need(record['additional_input_verification']['new_model_forwards'] == 0, 'additional input model-forward scope changed')
    need(record['additional_input_verification']['new_training'] == 0 and record['additional_input_verification']['new_generation'] == 0, 'additional input execution scope changed')
    need(record['training']['L_public']['successful_updates'] == 448, 'L_public training update count changed')
    need(record['training']['L_pooled']['successful_updates'] == 448, 'L_pooled training update count changed')
    need(record['training_changed_tensors']['L_public'] == 256 and record['training_changed_tensors']['L_pooled'] == 256, 'changed tensor counts changed')
    need(record['unchanged_old_failure'] is True, 'old failure preservation flag changed')
    need(record['expert_final_opened'] is False and record['reserved_opened'] is False, 'record expert/reserved flag changed')
    need(record['new_DP'] == 0 and record['final_ready'] is False, 'record DP/final scope changed')

    need(verification['status'] == 'PASS_REPORT_STATE_AND_FROZEN_SOURCE_BINDING', 'stored report verification status changed')
    need(verification['engineering_gate_passed'] is False, 'stored engineering gate changed')
    need(verification['report_sha256'] == sha(report), 'stored report sha mismatch')
    need(verification['html_sha256'] == sha(html), 'stored HTML sha mismatch')
    need(verification['record_sha256'] == sha(record_path), 'stored record sha mismatch')
    need(verification['new_model_inference'] == 0, 'stored new model inference scope changed')
    need(verification['prior_results_preserved'] is True, 'prior result preservation flag changed')
    need(verification['expert_opened'] is False and verification['reserved_opened'] is False, 'stored expert/reserved flag changed')
    need(verification['new_DP'] == 0, 'stored DP scope changed')

    need(isinstance(manifest, list), 'generation manifest should be a list')
    need(len(manifest) == 256, 'manifest image count changed')
    arms = {row['arm'] for row in manifest}
    labels = [row['label'] for row in manifest]
    need(arms == {'L_public', 'L_pooled'}, 'manifest arms changed')
    need(sum(1 for row in manifest if row['arm'] == 'L_public') == 128, 'L_public manifest count changed')
    need(sum(1 for row in manifest if row['arm'] == 'L_pooled') == 128, 'L_pooled manifest count changed')
    need(labels.count(1) == 128 and labels.count(0) == 128, 'manifest label balance changed')
    need(all('sha256' in row and 'image_sha256' in row for row in manifest), 'manifest missing hashes')

    state_record = state['completed_lora_transfer_comparison_20260917']
    need(state_record['status'] == 'completed_verified_gate_failed', 'state status changed')
    need(state_record['engineering_gate_passed'] is False, 'state engineering gate changed')
    need(state_record['images'] == 256, 'state image count changed')
    need(state_record['generator_updates'] == 896, 'state generator update count changed')
    need(state_record['classifier_updates'] == 2400, 'state classifier update count changed')
    need(state_record['classifier_runs'] == 6, 'state classifier run count changed')
    need(abs(state_record['means']['L_public']['AUROC'] - l_public_auc) < 1e-12, 'state L_public mean mismatch')
    need(abs(state_record['means']['L_pooled']['AUROC'] - l_pooled_auc) < 1e-12, 'state L_pooled mean mismatch')
    need(state_record['DP_executed'] is False and state_record['final_ready'] is False, 'state DP/final scope changed')
    need(state_record['expert_opened'] is False and state_record['reserved_opened'] is False, 'state expert/reserved scope changed')
    need(state_record['alternative_selected'] is False, 'state alternative selection flag changed')
    need(patient_spec['completed_lora_transfer_comparison_20260917']['status'] == state_record['status'], 'patient spec lora transfer status mismatch')

    for rel in ('common.py', 'engine.py', 'run.py', 'verify.py', 'README.md', '__init__.py'):
        need((REPO / 'code_working' / 'lora_transfer' / rel).exists(), f'missing public code file {rel}')

    need('track1_lora_transfer_comparison_results.html' in index_text, 'index missing results page link')
    need('track1_lora_transfer_comparison_protocol.html' in html_text or 'TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md' in report_text, 'result missing protocol link')
    need('L_public' in report_text and 'L_pooled' in report_text, 'report missing LoRA arms')
    need('0.558234' in report_text and '0.543587' in report_text, 'report missing headline AUROC values')
    need('관문을 통과하지 못했다' in report_text or '미통과' in report_text, 'report missing failed gate statement')
    need('새 DP 실행0' in report_text or 'new_DP' in report_text or 'DP' in report_text, 'report missing DP scope')
    need('expert final' in report_text and 'reserved' in report_text, 'report missing expert/reserved scope')
    need('lora_transfer_development_metrics_20260917.png' in report_text, 'report missing metrics figure')
    need('lora_transfer_generation_manifest_20260917.json' in report_text, 'report missing public manifest link')
    need(metrics_png.stat().st_size > 0 and loss_png.stat().st_size > 0 and grid_png.stat().st_size > 0, 'public PNG artifact missing content')
    need(not (REPO / 'code_working' / '_reports' / 'lora_transfer_20260917_v1').exists(), 'private _reports directory should not be in public bundle')

    result_out = {
        'status': 'PASS_PUBLIC_LORA_TRANSFER_RESULTS_BUNDLE',
        'record_status': record['status'],
        'state_status': state_record['status'],
        'engineering_gate_passed': result['engineering_gate_passed'],
        'L_public_AUROC': l_public_auc,
        'L_pooled_AUROC': l_pooled_auc,
        'L_pooled_minus_L_public_AUROC': comps['L_public']['AUROC_delta'],
        'positive_seeds_vs_L_public': comps['L_public']['positive_seeds'],
        'generator_updates': result['new_generator_updates'],
        'generated_images': result['new_images'],
        'classifier_updates': result['new_classifier_updates'],
        'classifier_runs': state_record['classifier_runs'],
        'verification_checks': v_inner['checks'],
        'manifest_images': len(manifest),
        'new_DP': result['new_DP'],
        'expert_opened': state_record['expert_opened'],
        'reserved_opened': state_record['reserved_opened'],
        'final_ready': result['final_ready'],
        'report_sha256': sha(report),
        'record_sha256': sha(record_path),
        'manifest_sha256': sha(manifest_path),
    }
    print(json.dumps(result_out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
