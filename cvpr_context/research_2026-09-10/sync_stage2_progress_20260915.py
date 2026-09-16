"""Synchronize mutable stage-2 status from verified, immutable run artifacts."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent.parent
RUN = THESIS / 'code_working/_reports/cvpr_u_pilot_v1_001'
SCREEN = RUN / 'baseline_screen_20260914'
MO = RUN / 'baseline_screen_20260915'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def compact_execution(folder):
    d = read(folder / 'execution.json')
    v = read(folder / 'verification.json')
    assert v['status'].startswith('PASS_')
    return {k: d[k] for k in ['ended_utc', 'total_seconds', 'measurement_seconds', 'counts', 'results_sha256', 'protocol_sha256'] if k in d} | {
        'status': 'completed_and_saved_arithmetic_verified',
        'producer_status_as_saved': d['status'],
        'verification_status': v['status'], 'directory': str(folder),
        'verification_scope': 'Saved arithmetic and provenance; not a full independent network/backward replay',
    }

def main():
    complete_m2 = (MO / 'mofit_medical_target2_full_v1/verification.json').exists()
    m1 = compact_execution(MO / 'mofit_medical_full_v1')
    m2 = compact_execution(MO / 'mofit_medical_target2_full_v1') if complete_m2 else None
    remaining_dir = MO / 'mofit_medical_remaining_pair_v1'
    remaining_complete = (remaining_dir / 'verification.json').exists()
    remaining = compact_execution(remaining_dir) if remaining_complete else None
    pooling_path = MO / 'mofit_patient14393_two_image_analysis_v1/analysis.json'
    pooling_complete = pooling_path.exists()
    if pooling_complete:
        assert read(pooling_path)['status'] == 'PASS_FIXED_SCALAR_AND_POOLING_ARITHMETIC'
    eu = {
        'status': 'completed_and_independently_verified',
        'report': 'CDI_EU_COMPARISON_RESULTS.md',
        'patients': {'fit': 80, 'selection': 40},
        'E_records': 480, 'E_new_records': 478, 'U_records_reused': 480,
        'E_runner_seconds': 3066.0602696, 'E_forward': 28064, 'E_backward': 8384,
        'E_features_verified': 12480, 'E_prior_SecMI_exact_records': 192,
        'four_cells': ['Efit_Eeval', 'Efit_Ueval', 'Ufit_Eeval', 'Ufit_Ueval'],
        'methods': 18, 'AUC_metrics': 144, 'paired_contrasts': 156,
        'U_scorers_restored_unchanged': True, 'new_E_scorers': 20,
        'reference_set_test': {'status': 'completed_patient_cluster_adaptation', 'contexts': 8,
            'candidate_patients': 20, 'reference_patients': 20,
            'source_two_sided_and_paper_one_sided_reported': True,
            'original_full_5fold_1000subsets_reproduced': False,
            'individual_patient_inference_or_FPR': False},
        'scaler_coefficient_audit': {'status': 'completed_posthoc', 'AUC_metrics': 64,
            'contrasts': 80, 'new_fitting': False, 'new_GPU': 0,
            'verification_scope': 'Two arithmetic/rank/bootstrap paths in the same module; no separate verifier'},
        'interpretation': 'U-fit standard scorer recovery under auxiliary U patient labels; original26D E-to-U32 contrasts all include0. Neither general transfer failure nor individual participation mechanism established.',
        'calibration_test_used': False, 'stage2_complete': False,
    }
    prompt = compact_execution(SCREEN / 'prompt_loss_t140_v1') | {
        'report': 'PROMPT_CONDITION_RESULTS.md', 'unique_images': 480,
        'models': 3, 'conditions': ['null', 'generic', 'weak_label'], 'forward': 4320, 'backward': 0,
        'target_AUC_metrics': 24, 'base_assignment_AUC_metrics': 6, 'prompt_contrasts': 24,
        'all_prompt_contrast_CIs_include_zero': True,
        'scope': 'One t140/noise-per-image, cached mode-latent fixed loss; not full MoFit',
    }
    for name in ['research_state.json', 'patient_baseline_spec.json']:
        path = ROOT / name
        state = read(path)
        state['current_step'] = 2
        state['stage2_status'] = 'in_progress'
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        state['completed_eu_comparison'] = eu
        state['completed_prompt_diagnostic'] = prompt
        state['mofit_medical_execution'] = {
            'report': 'MOFIT_MEDICAL_RESULTS.md', 'patient_id': '14393', 'patient_count': 1,
            'images_per_target': 4 if remaining_complete else 2, 'steps_per_image': [1000, 300],
            'model1_first_EU_pair': m1, 'model2_first_EU_pair': m2,
            'remaining_E006_U004_both_targets': remaining,
            'patient_two_image_aggregation': 'completed_fixed_h_and_three_basic_scores_mean_max' if pooling_complete else 'fixed_before_new_results_pending_execution',
            'medical_population_performance_comparison_complete': False,
            'medical_fusion_parameters_established': False,
            'individual_patient_causal_claim': False, 'new_target_training': False,
        }
        state['active_execution'] = {
            'status': ('no_running_execution' if pooling_complete else
                       'remaining_pair_extracted_pending_analysis' if remaining_complete else
                       'MoFit_remaining_E006_U004_full_running' if (remaining_dir / 'protocol.json').exists() else
                       'MoFit_remaining_E006_U004_preparing'),
            'current_step': 2, 'output': 'mofit_medical_remaining_pair_v1',
            'new_records': 4, 'new_patients': 0,
            'initial_estimated_followup_including_preparation_minutes': [35, 40],
            'revised_estimated_followup_including_preparation_minutes': [45, 50],
            'estimate_revision_reason': 'Actual pre-execution wrapper/provenance verification took about15 minutes; measured four-record GPU estimate remains about31 minutes',
            'stage2_completion_claim': False,
            'new_target_training': False, 'calibration_test_used': False,
        }
        state.pop('active_comparison', None)
        state['active_comparison_state_source'] = 'active_execution'
        state['current_result_report'] = 'CDI_EU_COMPARISON_RESULTS.md'
        state['step2_active_task'] = 'Synthesize verified E/U, conditioning and actual two-image MoFit evidence' if pooling_complete else 'Complete the same patient E2/U2 with four remaining full MoFit runs and fixed aggregation'
        state['next_task'] = 'Fix the remaining MoFit patient-population and fusion comparison scope and participation-specific contrast using the measured cost; do not count one patient as population performance' if pooling_complete else 'Execute and verify E006/U004 under both targets, combine with four immutable prior records for h and basic mean/max comparisons'
        state['next_task_output'] = 'A bounded, evidence-backed comparison; verified limitation/remedy/evidence connection remains unestablished'
        state['learned_baseline_inventory_correction'] = 'U CDI and E/U four-cell comparison completed. Patient-cluster reference Welch8 completed as adaptation, not original full experiment. MoFit actual kernel completed only for the stated one-patient scope.'
        state['step2_open_items'] = [
            'Patient-population MoFit core/fusion comparison under fixed medical score calibration and feasible cost',
            'Separate patient-group associations from patient participation; two whole-cohort targets are not individual causal interventions',
            'A verified limitation that remains after strong existing U adaptations under the same auxiliary-data access',
            'Normal-operation and failure-condition evidence; E-to-U32 original26D contrasts remain uncertain',
            'GM/NO paper-versus-code differences and their performance effects remain separate from literal-source arithmetic verification',
        ]
        if 'open_items' in state:
            state['open_items'] = list(state['step2_open_items'])
        completed = state.setdefault('step2_completed_diagnostics', [])
        for item in [
            'CDI E480 extraction, four E/U cells18methods144AUC156contrasts and independent saved arithmetic/fit/statistics verification',
            'CDI patient-cluster reference Welch8, original two-sided and paper one-sided formulas checked',
            'Posthoc scaler/coefficient hybrid64AUC80contrasts, two arithmetic paths in one module',
            'Prompt null/generic/weak-label actual4320F and saved raw/metric verification; no clear contrast in this fixed diagnostic',
            'Medical MoFit input, full source-schedule kernel and local FD convergence checks; explicit one-patient scope',
        ]:
            if item not in completed:
                completed.append(item)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'MUTABLE_STATES_SYNCHRONIZED', 'model2_verified': complete_m2,
                      'remaining_pair_verified': remaining_complete, 'pooling_complete': pooling_complete,
                      'stage': 2, 'stage2_complete': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
