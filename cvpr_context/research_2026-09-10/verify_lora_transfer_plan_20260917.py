from __future__ import annotations
import hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parent
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
    protocol = R / 'TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md'
    html = R / 'track1_lora_transfer_comparison_protocol.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    patient_spec_path = R / 'patient_baseline_spec.json'
    plan_path = SPEC / 'lora_transfer_comparison_plan_20260917.json'
    record_path = SPEC / 'lora_transfer_plan_record_20260917.json'
    verification_path = SPEC / 'lora_transfer_plan_verification_20260917.json'
    review_path = SPEC / 'lora_transfer_stored_review_20260917.json'
    figure_path = SPEC / 'lora_transfer_saved_loss_20260917.png'

    for path in (protocol, html, index, state_path, patient_spec_path, plan_path, record_path, verification_path, review_path, figure_path):
        need(path.exists(), f'missing public file: {path}')

    plan = read_json(plan_path)
    record = read_json(record_path)
    verification = read_json(verification_path)
    review = read_json(review_path)
    state = read_json(state_path)
    patient_spec = read_json(patient_spec_path)
    protocol_text = read_text(protocol)
    html_text = read_text(html)
    index_text = read_text(index)

    need(plan['schema'] == 'lora-private-transfer-comparison/v1', 'plan schema changed')
    need(plan['status'] == 'FROZEN_DESIGN_NOT_EXECUTED', 'plan status changed')
    need(plan['current_full64_branch_closed'] is True, 'closed branch prerequisite changed')
    need(plan['new_method_claim'] is False, 'new-method claim flag changed')
    need(plan['DP_allowed'] is False, 'DP allowed flag changed')
    need(plan['final_ready'] is False, 'final-ready flag changed')
    need(plan['expert_allowed'] is False and plan['reserved_allowed'] is False, 'expert/reserved allowed flag changed')
    need(plan['arms'] == ['L_public', 'L_pooled'], 'planned arms changed')

    adaptation = plan['adaptation']
    need(adaptation['rank'] == 8 and adaptation['alpha'] == 8, 'LoRA rank/alpha changed')
    need(adaptation['trainable_parameters'] == 1659904, 'trainable parameter count changed')
    need(adaptation['training_resolution'] == 256, 'training resolution changed')
    need(adaptation['batch_size'] == 4, 'batch size changed')
    need(adaptation['committed_steps_per_arm'] == 448, 'steps per arm changed')
    need(adaptation['total_committed_steps'] == 896, 'total committed steps changed')
    need(adaptation['checkpoint_primary'] == 448, 'primary checkpoint changed')
    need(adaptation['diagnostic_checkpoint_only'] == 224, 'diagnostic checkpoint changed')
    need(adaptation['training_repeats'] == 1, 'training repeat count changed')
    need(set(adaptation['target_modules']) == {'to_q', 'to_k', 'to_v', 'to_out.0'}, 'target modules changed')

    exp = plan['expected_exposure']
    need(exp['L_public']['unique_patients'] == 32 and exp['L_public']['unique_images'] == 64, 'L_public exposure changed')
    need(exp['L_public']['presentations'] == 1792, 'L_public presentations changed')
    need(exp['L_pooled']['unique_patients'] == 112 and exp['L_pooled']['unique_images'] == 384, 'L_pooled exposure changed')
    need(exp['L_pooled']['presentations'] == 1792, 'L_pooled presentations changed')
    need(abs(plan['pooled_patient_mass']['public'] - 2/7) < 1e-12, 'public patient mass changed')
    need(abs(plan['pooled_patient_mass']['private'] - 5/7) < 1e-12, 'private patient mass changed')

    generation = plan['generation']
    need(generation['new_images'] == 256 and generation['images_per_arm'] == 128, 'generation scale changed')
    need(generation['positive'] == 64 and generation['negative'] == 64, 'generation label balance changed')
    need(generation['negative_prompts'] == {'normal': 22, 'effusion': 21, 'cardiomegaly': 21}, 'negative prompt mix changed')
    need(len(generation['cells']) == 128, 'generation cell count changed')

    evaluation = plan['evaluation']
    need(evaluation['role'] == 'method_development', 'evaluation role changed')
    need(evaluation['patients'] == 2026 and evaluation['images'] == 5047, 'evaluation cohort changed')
    need(evaluation['weak_positive_images'] == 223, 'weak-positive count changed')
    need(evaluation['primary'] == 'L_pooled minus L_public', 'primary contrast changed')
    need(evaluation['gate_comparators'] == ['L_public', 'R1', 'R0'], 'gate comparators changed')
    need(evaluation['mean_AUROC_min_delta'] == 0.01, 'AUROC gate changed')
    need(evaluation['positive_seed_minimum'] == 2, 'positive seed gate changed')
    need(evaluation['mean_AP_min_delta'] == 0.0, 'AP gate changed')
    need(evaluation['cluster_bootstrap'] == 2000, 'bootstrap count changed')
    need(evaluation['causal_head_capacity_identification'] is False, 'causal capacity flag changed')

    need(record['status'] == 'PLAN_COMPLETE_NOT_EXECUTED', 'record status changed')
    need(record['report'] == protocol.name, 'record report mismatch')
    need(record['plan_sha256'] == sha(plan_path), 'record plan sha mismatch')
    need(record['new_model_forwards'] == 0, 'record model-forward scope changed')
    need(record['new_training'] == 0, 'record training scope changed')
    need(record['new_generation'] == 0, 'record generation scope changed')
    need(record['new_patient_pixels'] == 0, 'record patient-pixel scope changed')
    need(record['remote_push'] is False, 'record remote push flag should be historical false')

    need(verification['status'] == 'PASS_PLAN_METADATA_AND_SAVED_RESULT_BINDING', 'stored verification status changed')
    need(verification['protocol_sha256'] == sha(protocol), 'stored protocol sha mismatch')
    need(verification['plan_sha256'] == sha(plan_path), 'stored plan sha mismatch')
    if verification.get('figure_sha256'):
        need(verification['figure_sha256'] == sha(figure_path), 'stored figure sha mismatch')
    need(verification['new_model_forwards'] == 0, 'stored model-forward scope changed')
    need(verification['new_training'] == 0, 'stored training scope changed')
    need(verification['new_generation'] == 0, 'stored generation scope changed')
    need(verification['new_patient_pixels'] == 0, 'stored patient-pixel scope changed')

    need(review['status'] == 'STORED_OUTPUT_REVIEW_ONLY', 'stored review status changed')
    need(review['new_model_forwards'] == 0, 'review model-forward scope changed')
    need(review['new_training'] == 0, 'review training scope changed')
    need(review['new_generation'] == 0, 'review generation scope changed')
    need(review['new_patient_pixels'] == 0, 'review patient-pixel scope changed')

    state_record = state['completed_lora_transfer_plan_20260917']
    need(state_record['status'] == record['status'], 'state record status mismatch')
    need(state_record['plan_sha256'] == sha(plan_path), 'state plan sha mismatch')
    need(state_record['new_model_forwards'] == 0 and state_record['new_training'] == 0, 'state execution scope changed')
    need(state_record['new_generation'] == 0 and state_record['new_patient_pixels'] == 0, 'state generation/pixel scope changed')
    need(state_record['DP_allowed'] is False, 'state DP flag changed')
    need(state_record['expert_allowed'] is False and state_record['reserved_allowed'] is False, 'state expert/reserved flag changed')
    need(state_record['final_ready'] is False, 'state final-ready flag changed')
    need(state_record['runtime_implemented'] is False, 'runtime implementation flag changed')
    need(state_record['generator_updates_planned'] == 896, 'state planned update count changed')
    need(state_record['images_planned'] == 256, 'state planned image count changed')
    need(state_record['classifier_runs_planned'] == 6, 'state classifier run count changed')
    need(patient_spec['completed_lora_transfer_plan_20260917']['status'] == record['status'], 'patient spec record mismatch')

    need('track1_lora_transfer_comparison_protocol.html' in index_text, 'index missing LoRA transfer protocol link')
    need('lora_transfer_comparison_plan_20260917.json' in protocol_text, 'protocol missing plan JSON link')
    need('lora_transfer_saved_loss_20260917.png' in protocol_text, 'protocol missing saved loss figure link')
    need('LoRA' in protocol_text and 'L_public' in protocol_text and 'L_pooled' in protocol_text, 'protocol missing planned arms')
    need('새 GPU' in protocol_text and '0' in protocol_text, 'protocol missing no-new-execution statement')
    need('FROZEN_DESIGN_NOT_EXECUTED' in html_text or 'LoRA' in html_text, 'HTML page missing LoRA protocol content')

    result = {
        'status': 'PASS_PUBLIC_LORA_TRANSFER_PLAN_BUNDLE',
        'plan_status': plan['status'],
        'record_status': record['status'],
        'arms': plan['arms'],
        'committed_steps_per_arm': adaptation['committed_steps_per_arm'],
        'total_committed_steps': adaptation['total_committed_steps'],
        'planned_images': generation['new_images'],
        'classifier_runs_planned': state_record['classifier_runs_planned'],
        'DP_allowed': plan['DP_allowed'],
        'final_ready': plan['final_ready'],
        'new_model_forwards': record['new_model_forwards'],
        'new_training': record['new_training'],
        'new_generation': record['new_generation'],
        'new_patient_pixels': record['new_patient_pixels'],
        'protocol_sha256': sha(protocol),
        'plan_sha256': sha(plan_path),
        'figure_sha256': sha(figure_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
