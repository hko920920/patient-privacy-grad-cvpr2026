"""Record access observations without changing prior experiment results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
ROOT = R.parent.parent
RAW = ROOT / 'code_working/_reports/nih_expert_label_inventory_20260917_v1'
DEST = R / 'spec_sources/nih_expert_label_access_20260917'
REPORT = 'NIH_EXPERT_LABEL_ACCESS_20260917.md'
ACTUAL = 'TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepend(p, note):
    title, rest = p.read_text(encoding='utf-8-sig').split('\n', 1)
    p.write_text(title + '\n\n' + note + '\n' + rest, encoding='utf-8')

def main():
    DEST.mkdir(exist_ok=True)
    receipt = DEST / 'access_record.json'
    if receipt.exists():
        raise FileExistsError('Access review already recorded')
    frozen = {
        ACTUAL: 'a2f1556f6dcfac42f40fbed445afba14f86db0891d851a47d40ad6165abe7b9b',
        'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md': '8837f86cbb422272ba13d08d60c7ea979539e907aaad4e0c442d125a569a1a4f',
        'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md': '08f6b02f71498bb454654aa94a432be4ffc24bf7c06501806d49d199b9f5cb47',
    }
    for name, expected in frozen.items():
        if sha(R/name) != expected:
            raise ValueError('Historical report hash changed: ' + name)
    inventory = read(RAW/'inventory_status.json')
    inventory['existing_local_inventory_counts']['backbone_manifest_roles'] = {
        'train': 640, 'selection': 128, 'confirmation': 128, 'reserve': 9}
    dump(RAW/'inventory_status.json', inventory)
    now = datetime.now(timezone.utc).isoformat()
    status = {
        'report': REPORT, 'completed_utc': now, 'type': 'access_and_document_inventory_not_model_experiment',
        'judgment': 'negative_for_immediate_access_and_two_target_sample_sufficiency_not_method_efficacy',
        'expert_csv_acquired': False, 'patient_overlap_computed': False,
        'published_positive_image_counts': {'Emphysema': 7, 'Pneumothorax': 136, 'Effusion': 226},
        'counts_source': 'Official paper Supplementary Table 4; image counts, not independent patients',
        'counts_recomputed_from_label_csv': False,
        'official_download_form_submissions': 1, 'identity_or_contact_submitted': False,
        'anonymous_object_get_http_status': 403,
        'user_reports_authenticated_browser_url_http403': True,
        'user_account_login_and_IAM_independently_inspected': False,
        'author_email_draft': 'NIH_EXPERT_LABEL_ACCESS_REQUEST_20260917.txt',
        'author_emails_sent': 0, 'specific_label_file_uniquely_required': False,
        'new_model_inference': 0, 'new_generation': 0, 'new_training': 0, 'new_DP': 0,
        'reserved_confirmation_model_or_pixel_consumption': False,
        'last_actual_model_result_report': ACTUAL,
        'report_sha256': sha(R/REPORT),
        'supplement_sha256': sha(RAW/'41598_2021_93967_MOESM1_ESM.docx'),
        'inventory_record_sha256': sha(RAW/'inventory_status.json'),
    }
    old = {}
    for name in ['research_state.json', 'patient_baseline_spec.json']:
        state = read(R/name)
        if state['current_result_report'] != ACTUAL:
            raise ValueError('Actual model result pointer changed')
        old[name] = {k: state.get(k) for k in ['current_result_report', 'current_planning_report', 'next_task']}
        state['pre_nih_expert_label_access_20260917'] = old[name]
        state['completed_nih_expert_label_access_20260917'] = status
        state['current_planning_report'] = REPORT
        state['next_task_plan_report'] = REPORT
        state['step2_active_task'] = 'Expert CSV access blocked; official supplement image counts verified E7/P136. Patient-overlap audit pending access; match a feasible real-test resource to the downstream claim.'
        state['next_task'] = 'Use the prepared author inquiry to obtain current official access instructions. No message sent. After CSV access, audit patient-role overlap; do not presume seven emphysema-positive images suffice for the old joint-target design. The exact file is not a unique prerequisite; separate prospective claim/resource design remains possible.'
        state['next_task_output'] = 'Current official access instructions or explicit pending access, followed by a feasible claim/reference specification; no automatic classifier search, generation or DP.'
        state['updated_utc'] = now
        active = state['active_execution']
        active['status'] = 'no_running_execution'
        active['nih_expert_label_access_review'] = status
        active['latest_design_review'] = REPORT
        active['latest_planning_status'] = 'expert_csv_access_denied_published_counts_E7_P136_overlap_pending'
        active['next_substep_timing_condition'] = 'Patient-overlap work estimated20-30min only after CSV access. External owner response time unknown.'
        dump(R/name, state)
    note = '**2026-09-17 최신 자료 접근 결과 — CSV403, 원문 부록 E7/P136:** [접근 기록과 문의 문안](' + REPORT + '). 공식 요청 양식·직접 객체 경로를 확인했지만 expert CSV는 미확보다. 부록의 폐기종7장·기흉136장은 영상 수이며 환자 중복은 미계산이다. 이 파일이 유일한 필수자료는 아니고 접근 성공도 두 질환 표본 충분성을 보장하지 않는다. 배포자 문의 문안만 준비했고 발송하지 않았다. 새 모델·생성·DP0, 마지막 실제 모델 결과는 CheXzero 실패 그대로다.'
    prepend(R/'RESEARCH_FRAMEWORK.md', note)
    local_link = 'CVPR%20주제%20탐색/research_2026-09-10/nih_expert_label_access.html'
    prepend(ROOT/'CURRENT_STATUS.md', note.replace('('+REPORT+')', '('+local_link+')'))
    prepend(ROOT/'AGENTS.md', '- Latest NIH expert-label ACCESS review (2026-09-17): read `NIH_EXPERT_LABEL_ACCESS_20260917.md`. Official anonymous category-only download form completed; bucket/object CSVs403, user also reports direct-browser URL403. CSV NOT acquired and expert patient overlap NOT computed. Official supplement Table4 verifies E7/P136/Eff226 positive IMAGES, not patients; E-positive patients at most7 before exclusions. This exact CSV is not the unique prerequisite for credible downstream evaluation. Author inquiry draft prepared but NOT sent; do not send externally without user authorization. No new model/generation/DP. Preserve old evaluator/192 failures and reserved cohorts. Current actual model result stays CheXzero; latest planning is access review. Older access/count-pending notes are historical.')
    prepend(ROOT/'WORKLOG.md', '## 138-NIH-EXPERT-LABEL-ACCESS (2026-09-17 KST)\n\n공식 익명 범주형 요청 양식 완료 후 CSV 접근403을 확인했고 사용자의 직접 링크403도 기록했다. 공식 부록을 내려받아 XML와 python-docx로 폐기종7장·기흉136장·흉수226장을 확인했다. expert 환자 중복 감사는 미완료다. 공식 교신저자 문의 초안만 준비했으며 발송하지 않았다. 새 모델/생성/DP0, 기존 실패와 reserved 자료 보존. [접근 보고서]('+local_link+').')
    dump(receipt, {'status': status, 'previous_state_pointers': old, 'preserved_historical_report_hashes': frozen})
    print(json.dumps({'recorded': REPORT, 'actual_result_preserved': ACTUAL, 'email_sent': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
