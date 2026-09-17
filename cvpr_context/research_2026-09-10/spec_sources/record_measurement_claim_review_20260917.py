"""Record a source-based interpretation correction, preserving actual experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
ROOT = R.parent.parent
REPORT = 'MEASUREMENT_CLAIM_REVIEW_20260917.md'
DEST = R / 'spec_sources/measurement_claim_review_20260917'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def dump(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepend_after_title(p, text):
    old = p.read_text(encoding='utf-8')
    title, rest = old.split('\n', 1)
    p.write_text(title + '\n\n' + text + '\n' + rest, encoding='utf-8')

def main():
    receipt = DEST / 'review_record.json'
    if receipt.exists():
        raise FileExistsError('Review already recorded; do not duplicate history')
    now = datetime.now(timezone.utc).isoformat()
    actual = 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md'
    status = {
        'report': REPORT, 'completed_utc': now, 'current_step': 2, 'track': 1,
        'type': 'source_and_measurement_claim_review_not_model_experiment',
        'judgment': 'positive_for_existing_measurement_routes_not_private_efficacy',
        'universal_measurement_impossibility_established': False,
        'weak_label_error_as_main_cause_established': False,
        'current_joint_classifier_search_closed': True,
        'expert_collaboration_availability': 'unconfirmed_user_question_pending',
        'expert_label_files_acquired': False,
        'clinical_generation_correctness_established': False,
        'private_generation_utility_established': False,
        'NIH_all14_expert_images_documented': 810,
        'NIH_all14_expert_target_counts_and_patient_overlap': 'not_checked_without_label_manifest',
        'VinDr_majority_label_counts_paper_table2': {
            'Emphysema': {'train': 14, 'test': 3},
            'Pneumothorax': {'train': 58, 'test': 18}},
        'VinDr_patient_grouping': 'image_UID_is_not_verified_patient_ID',
        'new_model_inference': 0, 'new_generation': 0, 'new_training': 0,
        'new_DP': 0, 'reserved_confirmation_consumed': False,
        'external_messages_or_access_form_submissions': 0,
        'report_sha256': sha(R / REPORT),
        'roentgen_arxiv_pdf_sha256': sha(DEST / 'roentgen_2211.12737.pdf'),
        'last_actual_model_result_report': actual,
    }
    old_states = {name: read(R/name) for name in ['research_state.json', 'patient_baseline_spec.json']}
    preserved = {}
    for name, s in old_states.items():
        preserved[name] = {k: s.get(k) for k in ['current_result_report', 'current_planning_report', 'next_task', 'research_question_status']}
        if s['current_result_report'] != actual:
            raise ValueError('Unexpected latest actual result')
        s['pre_measurement_claim_review_20260917'] = preserved[name]
        s['completed_measurement_claim_review_20260917'] = status
        s['current_planning_report'] = REPORT
        s['next_task_plan_report'] = REPORT
        s['last_actual_model_result_report'] = actual
        s['step2_active_task'] = 'Source-based measurement correction complete; match the intended utility claim to an accessible independent measurement resource.'
        s['research_question_status'] = 'Private added generation utility remains unvalidated. Two fixed clinical classifiers failed; neither universal measurement impossibility nor lack of all measurement resources is established.'
        s['next_task'] = 'Prioritize access feasibility, target counts and patient-role overlap for existing NIH all14 expert labels, without model scores or locked-data reuse. Specify a downstream-utility alternative separately from generated disease correctness. No rescue of the failed joint-classifier experiment and no automatic generation or DP.'
        s['next_task_output'] = 'An accessible label/reference manifest and claim-matched comparison specification, or precise unresolved access requirements; do not infer scientific impossibility from resource uncertainty.'
        s['updated_utc'] = now
        a = s['active_execution']
        a['measurement_claim_review'] = status
        a['latest_design_review'] = REPORT
        a['latest_planning_status'] = 'measurement_routes_exist_access_and_claim_match_pending'
        a['next_substep_estimated_work_minutes'] = [20, 30]
        a['next_substep_timing_condition'] = 'Metadata count/overlap estimate starts after obtaining a label manifest; external access delay is unknown.'
        a['status'] = 'no_running_execution'
        dump(R/name, s)
    note = '**2026-09-17 최신 해석 정정 — 측정 경로는 존재, 방법 효용은 미판정:** [공식 주석·downstream 평가·현실적 다음 단계](' + REPORT + '). 두 classifier의 고정 NIH 검증 실패는 보존하지만 측정 불가능이나 방향1 전체 종료로 확대하지 않는다. NIH에는 14소견 전문의 재판독810장이 별도로 있다. 낮은 AUC의 주원인이 weak label이라고도 확정하지 않는다. 기존 전문 주석의 실제 접근·target 수·환자 중복과 downstream utility를 검토하며, 전문가 협력 여부는 미확인이다. 새GPU·생성·학습·DP0, reserved confirmation과 기존192장 실패를 보존한다. 현재 큰단계2·방향1이며 아래 최신/다음 문장은 이전 이력이다.'
    prepend_after_title(R/'RESEARCH_FRAMEWORK.md', note)
    prepend_after_title(ROOT/'CURRENT_STATUS.md', note.replace('('+REPORT+')', '(CVPR%20주제%20탐색/research_2026-09-10/measurement_claim_review.html)'))
    authority = '- Latest MEASUREMENT INTERPRETATION CORRECTION (2026-09-17): read `MEASUREMENT_CLAIM_REVIEW_20260917.md`. Two fixed evaluator gate failures do NOT establish measurement impossibility or a main weak-label-error cause. NIH has an all14 expert reread set of810PA images, separate from four-finding labels; actual access/target counts/patient overlap remain unverified. VinDr supports both labels but consensus positives are sparse (E14train/3test;P58/18), and image UID is not patient ID. Expert collaboration is unknown, not absent. Distribution, generated pathology correctness and downstream usefulness are distinct estimands. Existing DP-LoRA uses FID and train-synthetic/test-real utility, so lack of a clinical classifier is not a universal protection-efficiency stop rule. Preserve BOTH frozen evaluator failures, current joint-classifier search closure, prior192 failure and all locked cohorts. No automatic third classifier, generation, DP or track2 experiment. A separately justified prospective measurement design remains allowed; prior wording that only expert collaboration can keep all of track1 alive is superseded. Review only, no model execution. Current actual result remains CheXzero; latest planning is this correction.'
    prepend_after_title(ROOT/'AGENTS.md', authority)
    log = '## 137-MEASUREMENT-CLAIM-CORRECTION (2026-09-17 KST)\n\n사용자 지적에 따라 두 평가기 실패를 측정 불가능으로 확대한 해석을 철회했다. 원 실험 실패는 유지한다. NIH14소견810장·VinDr 두 target 주석 및 작은 합의양성 수·SIIM의 NIH 출처·DP-LoRA/RoentGen downstream 평가 근거를 확인했다. 실제 라벨 접근·중복·전문의 협력은 미확정이며 새 환자영상/추론/생성/DP0이다. 최종 actual-result 포인터를 CheXzero로 유지하고 planning/HTML/상태에 정정을 반영했다. [검토 문서](CVPR%20주제%20탐색/research_2026-09-10/measurement_claim_review.html).'
    prepend_after_title(ROOT/'WORKLOG.md', log)
    dump(receipt, {'review': status, 'pre_review_state_pointers': preserved})
    print(json.dumps({'review_report': REPORT, 'actual_result_preserved': actual, 'new_model_experiments': 0}, ensure_ascii=False))

if __name__ == '__main__':
    main()
