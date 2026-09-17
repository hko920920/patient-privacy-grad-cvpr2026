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

def main():
    protocol = R / 'TRACK1_REAL_SUPPORT_DIAGNOSTIC_PROTOCOL_20260917.md'
    review = R / 'TRACK1_REAL_SUPPORT_DIAGNOSTIC_PLAN_REVIEW_20260917.md'
    protocol_html = R / 'track1_real_support_diagnostic_protocol.html'
    review_html = R / 'track1_real_support_diagnostic_plan_review.html'
    index = R / 'index.html'
    state_path = R / 'research_state.json'
    patient_spec_path = R / 'patient_baseline_spec.json'
    plan_path = SPEC / 'real_support_diagnostic_plan_20260917.json'
    plan_ver_path = SPEC / 'real_support_diagnostic_plan_verification_20260917.json'
    report_ver_path = SPEC / 'real_support_plan_report_verification_20260917.json'

    for path in (protocol, review, protocol_html, review_html, index, state_path, patient_spec_path, plan_path, plan_ver_path, report_ver_path):
        need(path.exists(), f'missing public file: {path}')

    plan = read_json(plan_path)
    plan_ver = read_json(plan_ver_path)
    report_ver = read_json(report_ver_path)
    state = read_json(state_path)
    patient_spec = read_json(patient_spec_path)
    protocol_text = read_text(protocol)
    review_text = read_text(review)
    protocol_html_text = read_text(protocol_html)
    review_html_text = read_text(review_html)
    index_text = read_text(index)

    need(plan['schema'] == 'real-support-generalization-plan/v1', 'plan schema changed')
    need(plan['status'] == 'design_complete_no_new_model_execution', 'plan status changed')
    need(plan['current_step'] == 2, 'current step changed')
    need(plan['protocol'] == protocol.name, 'protocol binding changed')
    need(plan['arm'] == 'Rwide_real_diagnostic_NONDP', 'planned arm changed')
    need(plan['new_model_execution'] == 0, 'new model execution scope changed')
    need(plan['new_pixel_decode'] == 0, 'new pixel decode scope changed')
    need(plan['runtime_implemented'] is False, 'runtime flag changed')
    need(plan['final_ready'] is False, 'final-ready flag changed')
    need(plan['expert_opened'] is False and plan['reserved_opened'] is False, 'expert/reserved flag changed')
    need(plan['DP'] is False, 'DP flag changed')

    counts = plan['counts']
    need(counts['public']['patients'] == 672 and counts['public']['images'] == 813, 'public count changed')
    need(counts['public']['positive_patients'] == 6 and counts['public']['positive_images'] == 6, 'public positive count changed')
    need(counts['former_classifier_selection']['patients'] == 2027 and counts['former_classifier_selection']['images'] == 5097, 'former selection count changed')
    need(counts['former_classifier_selection']['positive_patients'] == 114 and counts['former_classifier_selection']['positive_images'] == 230, 'former selection positive count changed')
    need(counts['expanded']['patients'] == 2699 and counts['expanded']['images'] == 5910, 'expanded count changed')
    need(counts['expanded']['positive_patients'] == 120 and counts['expanded']['positive_images'] == 236, 'expanded positive count changed')
    need(counts['method_development']['patients'] == 2026 and counts['method_development']['images'] == 5047, 'method-development count changed')

    overlaps = plan['overlaps']
    need(overlaps['method_development']['patient_id'] == 0 and overlaps['method_development']['image_id'] == 0 and overlaps['method_development']['sha256'] == 0, 'method-development overlap changed')
    need(overlaps['private80']['patient_id'] == 0 and overlaps['private80']['image_id'] == 0 and overlaps['private80']['sha256'] == 0, 'private80 overlap changed')
    need(overlaps['expert_patients'] == 0 and overlaps['reserved_patients'] == 0 and overlaps['locked_patients'] == 0, 'locked/expert/reserved overlap changed')
    need(plan['additional_source_original_role'] == 'private_train', 'additional source role changed')
    need(plan['original_role_files_changed'] is False, 'original role files changed flag changed')
    need(plan['role_overlay_status'] == 'planned_not_yet_optimizer_consumed', 'role overlay status changed')

    sampling = plan['sampling']
    need(sampling['class_balanced'] is True, 'class-balanced flag changed')
    need(sampling['half_batch_size'] == 16, 'half batch size changed')
    need(sampling['labels_per_half'].count(0) == 8 and sampling['labels_per_half'].count(1) == 8, 'label balance changed')
    need(sampling['stream_names'] == ['public-shared', 'public-extra'], 'stream names changed')
    need(sampling['same_random_streams_not_same_image_ids'] is True, 'random stream statement changed')

    need(plan['seeds'] == [11, 23, 37], 'seed list changed')
    need(plan['steps'] == 400, 'step count changed')
    need(plan['batch'] == 32, 'batch size changed')
    need(plan['learning_rate'] == 0.0001 and plan['weight_decay'] == 0.0001, 'optimizer hyperparameters changed')
    need(plan['scheduled_presentations'] == 38400, 'scheduled presentation count changed')
    need(plan['primary_comparison'] == 'Rwide-R1 mean per-seed AUROC', 'primary comparison changed')
    need(plan['secondary_comparison'] == 'Rwide-R0 AUROC/AP', 'secondary comparison changed')
    need(plan['bootstrap']['iterations'] == 2000 and plan['bootstrap']['patient_clusters'] == 2026, 'bootstrap plan changed')
    need(plan['improvement_candidate']['mean_AUROC_gain_min'] == 0.01, 'improvement AUROC threshold changed')
    need(plan['improvement_candidate']['positive_seeds_min'] == 2, 'positive seed threshold changed')
    need(plan['improvement_candidate']['mean_AP_non_decrease'] is True, 'AP non-decrease rule changed')
    need(plan['improvement_candidate']['implies_DP_entry'] is False, 'DP implication flag changed')

    exp = plan['planned_exposure']
    for seed, images, patients in (('11', 3439, 2466), ('23', 3459, 2468), ('37', 3476, 2472)):
        need(seed in exp, f'missing exposure seed {seed}')
        need(exp[seed]['actual_not_executed'] is True, f'exposure seed {seed} execution flag changed')
        need(exp[seed]['planned_unique_images'] == images, f'exposure seed {seed} image count changed')
        need(exp[seed]['planned_unique_patients'] == patients, f'exposure seed {seed} patient count changed')
        need(exp[seed]['sources']['public/1']['seen_images'] == 6, f'public positive coverage changed for seed {seed}')
        need(exp[seed]['sources']['classifier_selection/1']['seen_images'] == 230, f'extra positive coverage changed for seed {seed}')
        need(exp[seed]['sources']['classifier_selection/1']['seen_patients'] == 114, f'extra positive patient coverage changed for seed {seed}')

    budget = plan['execution_budget']
    need(budget['main_updates'] == 1200, 'main update budget changed')
    need(budget['independent_probe_updates'] == 4, 'probe update budget changed')
    need(budget['new_generated_images'] == 0, 'generated image budget changed')
    need(budget['existing_R1_path_witness_images'] == 192, 'R1 witness budget changed')
    need(budget['method_development_image_forward'] == 15141, 'development forward budget changed')
    need(budget['training_eval_image_forward'] == 10374, 'training eval forward budget changed')
    need(budget['DP'] == 0 and budget['expert_pixels'] == 0 and budget['reserved_pixels'] == 0, 'DP/expert/reserved budget changed')

    need(plan_ver['status'] == 'PASS_METADATA_ROLE_BOUNDARIES_AND_INDEPENDENT_SCHEDULE', 'plan verification status changed')
    need(plan_ver['checks'] == 124175, 'plan verification check count changed')
    need(plan_ver['original_R1_batches_reproduced'] == 1200, 'R1 batch reproduction count changed')
    need(plan_ver['new_planned_batches'] == 1200, 'planned batch count changed')
    need(plan_ver['new_planned_slots'] == 38400, 'planned slot count changed')
    need(plan_ver['metadata_only'] is True, 'metadata-only flag changed')
    need(plan_ver['new_model_inference'] == 0 and plan_ver['new_training'] == 0, 'plan verification execution scope changed')
    need(plan_ver['new_pixel_decode'] == 0 and plan_ver['new_generation'] == 0, 'plan verification pixel/generation scope changed')
    need(plan_ver['expert_reserved_opened'] is False, 'plan verification expert/reserved flag changed')
    need(plan_ver['original_roles_changed'] is False, 'plan verification role flag changed')
    need(plan_ver['runtime_implemented'] is False, 'plan verification runtime flag changed')
    need(plan_ver['plan_sha256'] == sha(plan_path), 'plan verification plan sha mismatch')
    need(plan_ver['protocol_sha256'] == sha(protocol), 'plan verification protocol sha mismatch')

    need(report_ver['status'] == 'PASS_PLAN_REPORT_STATE_AND_PRIOR_RESULT_BINDING', 'report verification status changed')
    need(report_ver['checks'] == 664, 'report verification check count changed')
    need(report_ver['current_model_result_unchanged'] is True, 'current model result changed flag changed')
    need(report_ver['current_efficacy_result_unchanged'] is True, 'current efficacy result changed flag changed')
    need(report_ver['new_model_execution'] == 0 and report_ver['new_training'] == 0 and report_ver['new_pixel_decode'] == 0, 'report verification execution scope changed')
    need(report_ver['plan_sha256'] == sha(plan_path), 'report verification plan sha mismatch')
    need(report_ver['verification_sha256'] == sha(plan_ver_path), 'report verification verification sha mismatch')
    need(report_ver['protocol_sha256'] == sha(protocol), 'report verification protocol sha mismatch')
    need(report_ver['review_sha256'] == sha(review), 'report verification review sha mismatch')
    need(report_ver['review_html_sha256'] == sha(review_html), 'report verification review HTML sha mismatch')

    state_record = state['completed_real_support_plan_20260917']
    need(state_record['status'] == 'metadata_verified_runtime_not_implemented', 'state status changed')
    need(state_record['review'] == review.name and state_record['protocol'] == protocol.name, 'state report/protocol binding changed')
    need(state_record['public_plan'] == 'spec_sources/real_support_diagnostic_plan_20260917.json', 'state public plan path changed')
    need(state_record['counts']['expanded']['images'] == 5910 and state_record['counts']['expanded']['patients'] == 2699, 'state expanded count changed')
    need(state_record['planned_main_updates'] == 1200 and state_record['planned_probe_updates'] == 4, 'state planned update count changed')
    need(state_record['planned_probe_seed'] == 11, 'state probe seed changed')
    need(state_record['new_model_execution'] == 0 and state_record['new_training'] == 0, 'state execution scope changed')
    need(state_record['new_generation'] == 0 and state_record['new_pixel_decode'] == 0, 'state generation/pixel scope changed')
    need(state_record['original_roles_changed'] is False, 'state role-change flag changed')
    need(state_record['runtime_implemented'] is False, 'state runtime flag changed')
    need(state_record['DP_executed'] is False, 'state DP flag changed')
    need(state_record['expert_opened'] is False and state_record['reserved_opened'] is False, 'state expert/reserved flag changed')
    need(state_record['final_ready'] is False, 'state final-ready flag changed')
    need(state_record['improved_performance_established'] is False, 'state improved performance flag changed')
    need(state_record['cause_established'] is False, 'state cause flag changed')
    need(patient_spec['completed_real_support_plan_20260917']['status'] == state_record['status'], 'patient spec state mismatch')

    for rel in ('prepare.py', 'verify.py', '__init__.py'):
        need((REPO / 'code_working' / 'real_support_plan' / rel).exists(), f'missing real support code file {rel}')

    need('track1_real_support_diagnostic_plan_review.html' in index_text, 'index missing review link')
    need('track1_real_support_diagnostic_protocol.html' in review_html_text, 'review HTML missing protocol link')
    need('real_support_diagnostic_plan_20260917.json' in review_text, 'review missing plan JSON link')
    need('성능은 아직 미실행' in review_text or '성능 결과는 없다' in protocol_text, 'reports missing not-executed statement')
    need('새 학습·추론·생성·raw pixel decode는 모두0' in review_text, 'review missing no-new-execution statement')
    need('Rwide' in protocol_text and 'R1' in protocol_text, 'protocol missing Rwide/R1 comparison')
    need(('5910' in review_text or '5,910' in review_text) and ('2699' in review_text or '2,699' in review_text), 'review missing expanded cohort count')
    need('Expert final' in review_text and 'reserved' in review_text, 'review missing expert/reserved statement')
    need('DP' in review_text and 'final_ready=false' in review_text, 'review missing DP/final statement')
    need('새 runtime은 미구현' in protocol_html_text, 'protocol HTML missing runtime-not-implemented statement')
    need(not (REPO / 'code_working' / '_reports' / 'real_support_plan_20260917_v1').exists(), 'private _reports directory should not be present')

    result = {
        'status': 'PASS_PUBLIC_REAL_SUPPORT_PLAN_BUNDLE',
        'plan_status': plan['status'],
        'arm': plan['arm'],
        'expanded_patients': counts['expanded']['patients'],
        'expanded_images': counts['expanded']['images'],
        'expanded_positive_patients': counts['expanded']['positive_patients'],
        'expanded_positive_images': counts['expanded']['positive_images'],
        'planned_main_updates': state_record['planned_main_updates'],
        'planned_probe_updates': state_record['planned_probe_updates'],
        'scheduled_presentations': plan['scheduled_presentations'],
        'verification_checks': plan_ver['checks'],
        'new_model_execution': plan['new_model_execution'],
        'new_training': state_record['new_training'],
        'new_generation': state_record['new_generation'],
        'new_pixel_decode': state_record['new_pixel_decode'],
        'runtime_implemented': plan['runtime_implemented'],
        'DP_executed': state_record['DP_executed'],
        'expert_opened': state_record['expert_opened'],
        'reserved_opened': state_record['reserved_opened'],
        'final_ready': state_record['final_ready'],
        'plan_sha256': sha(plan_path),
        'protocol_sha256': sha(protocol),
        'review_sha256': sha(review),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()



