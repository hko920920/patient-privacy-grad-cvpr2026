from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parent
REPO = R.parents[1]
SPEC = R / 'spec_sources'
FIG = R / 'figures'

def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8-sig')

def read_json(path: Path):
    return json.loads(read_text(path))

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def f(x) -> float:
    return float(x)

def main():
    report = R / 'TRACK1_CLASSIFIER_GENERALIZATION_RESULTS_20260917.md'
    protocol = R / 'TRACK1_CLASSIFIER_GENERALIZATION_PROTOCOL_20260917.md'
    html = R / 'track1_classifier_generalization_results.html'
    protocol_html = R / 'track1_classifier_generalization_protocol.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    patient_spec_path = R / 'patient_baseline_spec.json'
    record_path = SPEC / 'classifier_generalization_record_20260917.json'
    verification_path = SPEC / 'classifier_generalization_report_verification_20260917.json'
    csv_path = SPEC / 'classifier_generalization_seed_metrics_20260917.csv'
    auroc_fig = FIG / 'classifier_generalization_auroc_20260917.png'
    logits_fig = FIG / 'classifier_generalization_dev_logits_20260917.png'

    for path in (report, protocol, html, protocol_html, index, state_path, patient_spec_path, record_path, verification_path, csv_path, auroc_fig, logits_fig):
        need(path.exists(), f'missing public file: {path}')

    record = read_json(record_path)
    verification = read_json(verification_path)
    state = read_json(state_path)
    patient_spec = read_json(patient_spec_path)
    report_text = read_text(report)
    html_text = read_text(html)
    index_text = read_text(index)
    with csv_path.open(newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))

    need(record['scope'] == 'Frozen-checkpoint diagnostic, not a new method efficacy test', 'record scope changed')
    need(record['budget']['unique_training_images'] == 1389, 'unique training image count changed')
    need(record['budget']['witness_images'] == 64, 'witness image count changed')
    need(record['budget']['raw_or_PNG_decode'] == 1453, 'decode count changed')
    need(record['budget']['training_image_presentations'] == 11201, 'training image presentation count changed')
    need(record['budget']['witness_presentations'] == 768, 'witness presentation count changed')
    need(record['budget']['models'] == 12, 'model count changed')
    need(record['budget']['optimizer_updates'] == 0, 'optimizer update scope changed')
    need(record['budget']['new_generation'] == 0, 'new generation scope changed')
    need(record['budget']['expert_pixels'] == 0 and record['budget']['reserved_pixels'] == 0, 'expert/reserved pixel scope changed')
    need(record['budget']['DP'] == 0, 'DP scope changed')

    means = record['means']
    for arm in ('R1', 'Dreal', 'L_public', 'L_pooled'):
        need(arm in means, f'missing arm {arm}')
        need('development' in means[arm], f'missing development metrics for {arm}')
    need(abs(means['R1']['development']['AUROC'] - 0.5446095692439646) < 1e-12, 'R1 development AUROC changed')
    need(abs(means['Dreal']['development']['AUROC'] - 0.5961724139640613) < 1e-12, 'Dreal development AUROC changed')
    need(abs(means['L_public']['development']['AUROC'] - 0.5582343018341279) < 1e-12, 'L_public development AUROC changed')
    need(abs(means['L_pooled']['development']['AUROC'] - 0.5435871836631491) < 1e-12, 'L_pooled development AUROC changed')
    need(means['R1']['real/public']['AUROC'] == 1.0 and means['R1']['real/public']['AP'] == 1.0, 'R1 seen-public fit changed')
    need(means['Dreal']['real/private']['AUROC'] == 1.0 and means['Dreal']['real/public']['AUROC'] == 1.0, 'Dreal seen-real fit changed')
    need(means['L_public']['synthetic/lora_public']['AUROC'] == 1.0, 'L_public synthetic fit changed')
    need(abs(means['L_pooled']['synthetic/lora_pooled']['AUROC'] - 0.9996744791666666) < 1e-12, 'L_pooled synthetic fit changed')
    need(means['R1']['development']['AUROC'] < 0.60, 'R1 development relation changed')
    need(means['L_pooled']['development']['AUROC'] < means['L_public']['development']['AUROC'], 'LoRA pooled/public development relation changed')

    v_inner = record['verification']
    need(v_inner['status'] == 'PASS_FIXED_INFERENCE_DIAGNOSTIC_INTEGRITY', 'inner verification status changed')
    need(v_inner['checks'] == 325916, 'inner check count changed')
    need(v_inner['maximum_metric_difference'] <= 1.2e-16, 'metric difference changed')
    need(v_inner['independent_raw_PNG_decodes'] == 1453, 'independent decode count changed')
    need(v_inner['independent_GPU_input_arithmetic'] is True, 'GPU input arithmetic flag changed')
    need(v_inner['new_model_forward'] == 0, 'inner new model-forward scope changed')
    need(v_inner['no_production_metric_import'] is True, 'production metric import flag changed')
    need(v_inner['method_efficacy_verified'] is False, 'method efficacy retest flag changed')
    need(len(v_inner['models']) == 12, 'inner model count changed')
    need(all(m['state_unchanged'] is True and m['witness_max_absolute_difference'] == 0.0 for m in v_inner['models']), 'model state/witness changed')

    need(record['actual_inference_seconds'] > 0, 'inference seconds missing')
    need(record['all_model_states_unchanged'] is True, 'model state flag changed')
    need(record['maximum_witness_logit_difference'] == 0.0, 'witness logit difference changed')
    need(record['method_efficacy_retested'] is False, 'method efficacy retest changed')
    need(record['expert_opened'] is False and record['reserved_opened'] is False, 'expert/reserved scope changed')
    need(record['DP'] is False and record['final_ready'] is False, 'DP/final scope changed')

    need(verification['status'] == 'PASS_REPORT_STATE_AND_PAST_EVIDENCE_BINDING', 'stored report verification status changed')
    need(verification['checks'] == 828, 'stored verification check count changed')
    need(verification['new_model_inference'] == 0, 'stored new model inference scope changed')
    need(verification['report_sha256'] == sha(report), 'stored report sha mismatch')
    need(verification['html_sha256'] == sha(html), 'stored HTML sha mismatch')
    need(verification['public_record_sha256'] == sha(record_path), 'stored record sha mismatch')
    need(verification['last_efficacy_result_unchanged'] is True, 'last efficacy preservation flag changed')
    need(verification['expert_reserved_closed'] is True, 'expert/reserved closed flag changed')

    need(len(rows) == 33, 'CSV row count changed')
    need(set(rows[0].keys()) == {'arm', 'seed', 'source', 'images', 'positives', 'AUROC', 'AP', 'BCE', 'balanced_BCE'}, 'CSV columns changed')
    csv_arms = {r['arm'] for r in rows}
    need(csv_arms == {'R1', 'Dreal', 'L_public', 'L_pooled'}, 'CSV arm set changed')
    need({int(r['seed']) for r in rows} == {11, 23, 37}, 'CSV seed set changed')
    need(any(r['arm'] == 'R1' and r['source'] == 'real/public' and f(r['AUROC']) == 1.0 for r in rows), 'CSV missing R1 seen-public perfect AUROC')
    need(any(r['arm'] == 'L_pooled' and r['source'] == 'method_development' and abs(f(r['AUROC']) - 0.5657103124140137) < 1e-12 for r in rows), 'CSV missing L_pooled seed37 development metric')

    state_record = state['completed_classifier_generalization_20260917']
    need(state_record['status'] == 'completed_verified_generalization_gap_observed', 'state status changed')
    need(state_record['new_training'] == 0 and state_record['new_generation'] == 0, 'state training/generation scope changed')
    need(state_record['classifiers'] == 12, 'state classifier count changed')
    need(state_record['unique_training_images'] == 1389, 'state unique training image count changed')
    need(state_record['new_training_image_forward'] == 11201, 'state training forward count changed')
    need(state_record['development_witness_forward'] == 768, 'state witness forward count changed')
    need(state_record['model_states_unchanged'] is True and state_record['BN_buffers_unchanged'] is True, 'state model/BN flag changed')
    need(state_record['development_witness_exact'] is True, 'state witness exact flag changed')
    need(state_record['fixed_LoRA_utility_gate_passed'] is False, 'state LoRA utility gate changed')
    need(state_record['method_efficacy_retested'] is False, 'state efficacy retest flag changed')
    need(state_record['causal_failure_source_established'] is False, 'state causal source flag changed')
    need(state_record['DP_executed'] is False, 'state DP execution flag changed')
    need(state_record['expert_opened'] is False and state_record['reserved_opened'] is False, 'state expert/reserved flag changed')
    need(state_record['final_ready'] is False, 'state final-ready flag changed')
    need(patient_spec['completed_classifier_generalization_20260917']['status'] == state_record['status'], 'patient spec classifier status mismatch')

    for rel in ('common.py', 'run.py', 'verify.py', 'README.md', '__init__.py'):
        need((REPO / 'code_working' / 'classifier_generalization' / rel).exists(), f'missing code file {rel}')

    need('track1_classifier_generalization_results.html' in index_text, 'index missing classifier generalization result link')
    need('TRACK1_CLASSIFIER_GENERALIZATION_PROTOCOL_20260917.md' in report_text, 'report missing protocol link')
    need('classifier_generalization_seed_metrics_20260917.csv' in report_text, 'report missing CSV link')
    need('classifier_generalization_auroc_20260917.png' in report_text, 'report missing AUROC figure link')
    need('classifier_generalization_dev_logits_20260917.png' in report_text, 'report missing logit figure link')
    need('AUROC/AP **1.0**' in report_text, 'report missing seen-data fit statement')
    need('0.5436' in report_text and '0.5962' in report_text, 'report missing development AUROC range')
    need('새 학습·생성·DP는 모두0' in report_text, 'report missing no-new-training/generation/DP statement')
    need('Expert final' in report_text and 'reserved' in report_text, 'report missing expert/reserved statement')
    need('일반화' in html_text or 'generalization' in html_text.lower(), 'HTML report missing generalization content')
    need(auroc_fig.stat().st_size > 0 and logits_fig.stat().st_size > 0, 'public figures are empty')
    need(not (REPO / 'code_working' / '_reports' / 'classifier_generalization_20260917_v1').exists(), 'private _reports directory should not be present')

    result = {
        'status': 'PASS_PUBLIC_CLASSIFIER_GENERALIZATION_BUNDLE',
        'state_status': state_record['status'],
        'classifiers': state_record['classifiers'],
        'unique_training_images': state_record['unique_training_images'],
        'training_image_forwards': state_record['new_training_image_forward'],
        'development_witness_forward': state_record['development_witness_forward'],
        'verification_checks': v_inner['checks'],
        'raw_or_PNG_decodes': record['budget']['raw_or_PNG_decode'],
        'R1_development_AUROC': means['R1']['development']['AUROC'],
        'Dreal_development_AUROC': means['Dreal']['development']['AUROC'],
        'L_public_development_AUROC': means['L_public']['development']['AUROC'],
        'L_pooled_development_AUROC': means['L_pooled']['development']['AUROC'],
        'seen_real_AUROC': 1.0,
        'new_training': state_record['new_training'],
        'new_generation': state_record['new_generation'],
        'DP_executed': state_record['DP_executed'],
        'expert_opened': state_record['expert_opened'],
        'reserved_opened': state_record['reserved_opened'],
        'final_ready': state_record['final_ready'],
        'report_sha256': sha(report),
        'record_sha256': sha(record_path),
        'csv_sha256': sha(csv_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
