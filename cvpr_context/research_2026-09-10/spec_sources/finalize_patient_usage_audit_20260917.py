"""Record the completed audit without mutating frozen experiment artifacts."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent.parent;ROOT=R.parent.parent;P=ROOT/'code_working/_reports/patient_usage_audit_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=read(P/'audit.json');v=read(P/'verification.json')
 d=dict(status='POSITIVE_SEPARATE_CVPR_REFERENCE_FEASIBILITY_NOT_EFFICACY',completed_utc=datetime.now(timezone.utc).isoformat(),
  report='TRACK1_PATIENT_USAGE_AUDIT_RESULTS_20260917.md',protocol='TRACK1_PATIENT_USAGE_AUDIT_PROTOCOL_20260917.md',directory=str(P),
  current_step=2,track=1,candidate_patients=8426,target_candidate_patients={'Emphysema':273,'Pneumothorax':559},
  development_target_patients={'Emphysema':135,'Pneumothorax':268},confirmation_target_patients={'Emphysema':138,'Pneumothorax':291},
  actual_old_private_dryrun_patients=50,actual_old_private_dryrun_images=59,artifact_scan_files=3680,
  independent_checks=v['checks'],audit_seconds=a['seconds'],verification_seconds=v['seconds'],
  new_GPU=0,new_generation=0,new_model_inference=0,original_roles_changed=False,provisional_split_only=True,
  locked_original_sets_preserved=True,private_generation_utility_established=False,previous192_utility_gate_passed=False,
  evaluator_validated=False,DP_allowed_to_expand=False,
  source_sha256={n:sha(P/n) for n in ['contract.json','audit.json','verification.json','patient_usage_private.csv','provisional_patient_split_private.csv','provisional_condition_images_private.csv']})
 with (P/'research_decision.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
 for n in ['research_state.json','patient_baseline_spec.json']:
  s=read(R/n);s['pre_patient_usage_current_result_report_20260917']=s['current_result_report']
  s['current_result_report']=d['report'];s['current_planning_report']=d['report'];s['next_task_plan_report']=d['report']
  s['next_task']='One bounded20-35min fixed PadChest evaluator validation: predeclare real-image patient/label selection and preprocessing/raw-score semantics on nonlocked development-only patients, independently verify loading/output and measure Emphysema/Pneumothorax discrimination with patient-level uncertainty and normal/effusion controls. Do not use the provisional reserved confirmation, pick evaluator/thresholds from generated outcomes, launch targeted generation or DP automatically.'
  s['next_task_output']='Bound evaluator-validation manifest and actual performance/implementation verdict; then a separate targeted non-DP comparison contract only if justified. Preserve existing private80, public32, all old results and original thesis roles.'
  s['step2_open_items']=['Separate CVPR reference is feasible by recorded use:8426 original-private candidates, provisional development/confirmation counts sufficient for both targets; not absolute never-used proof.', 'Original thesis roles remain fixed. Future models trained on candidate reference patients cannot treat that reference as independent.', 'Fixed PadChest condition-sensitive evaluator needs actual validation; old nonlocked324patient pool is development-only and has just15 emphysema-positive patients in its available images.', 'Targeted private incremental generation utility, DP retention and fair quality-cost advantages remain unestablished.', 'Prior192-image private utility gate remains failed; no posthoc rewriting.']
  s['active_execution']['status']='no_running_execution';s['active_execution']['current_step']=2;s['active_execution']['patient_usage_audit']=d
  s['completed_patient_usage_audit_20260917']=d;s['latest_user_execution_scope_20260917']='Complete the patient-level actual-use ledger and separate-CVPR reference feasibility audit; original thesis roles/final tests preserved, no model execution.'
  (R/n).write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 w=ROOT/'WORKLOG.md';text=w.read_text(encoding='utf-8');lo=text.index('## 132-TARGET-EVALUATION-INVENTORY');hi=text.index('## 131-PRIVATE-SIGNAL',lo)
 with (P/'worklog132_original_mojibake.txt').open('x',encoding='utf-8') as f:f.write(text[lo:hi])
 repaired='''## 132-TARGET-EVALUATION-INVENTORY (2026-09-17 KST)

- 한글 저장 인코딩 손상을 133번 감사 마무리 중 복구했다. 손상 원문은 patient_usage_audit_20260917_v1/worklog132_original_mojibake.txt에 보존했다. 아래 사실은 당시 결과 보고서와 검산 파일에서 복원했으며 실험 결과나 역할을 변경하지 않았다.
- 큰 단계2·방향1. 예상15–25분의 환자분리 reference/evaluator inventory를 완료했다. 새GPU·모델추론·생성·학습·DP는0이다.
- 전체 NIH local42,423장14,755명을 대조했다. Public1,816명 중 backbone640 및 CVPR752명 제외 후424명511장, 폐기종3명·기흉6명. 과거reference/진단 제외 후264명264장, 폐기종3명·기흉1명이다. 기존 예약 역할까지 제외한 미배정 환자는0명이다.
- 후보511장 SHA·크기·PNG decode 일치 및 독립2,124항목 재집계PASS. Inventory11.470초, 검산19.926초. 기존192장과실패판정·frozen source 유지. 최초32건label순서 차이로 중단한v1도보존했고v2에서전체label집합을확인했다.
- 미다운로드 공식train_val PA 기존배정밖14,113명22,329장에 폐기종157명180장, 기흉/effusion0명. Local0장으로 단순추가다운로드는두target문제를해결하지못했다. Original private_train273명폐기종/563명기흉은 새reference로채택하지않았다.
- PadChest-only classifier cached file과 두target의학습출력 확인, 실제판별력미검증. MIMIC/CheXpert 폐기종slot은미학습이다.
- 판정: 당시public역할에서자료feasibility부정적, 평가기후보긍정적. 다음은실제사용이력20–30분감사. TRACK1_TARGET_EVALUATION_INVENTORY_RESULTS_20260917.md/HTML과실행v2에기록했다. 원격main확인/push는하지않았다.

'''
 text=text[:lo]+repaired+text[hi:]
 first,rest=text.split('\n',1)
 new='''

## 133-PATIENT-ACTUAL-USE-AUDIT (2026-09-17 KST)

- 큰 단계2·방향1. 예상20–30분으로 역할명과 실제 소비를 구분하는 감사를 진행했다. 중간 sandbox 폴더쓰기 거부로 중단됐고 사용자가 권한을 바꾼 후 동일코드로 재개했다. 중단된 프로세스/결과폴더가 없음을 먼저 확인했다.
- Original private_train8476명 전체가 실제훈련된 것은 아니었다. 원본 restricted runtime log/public hash가 일치했고 실제4step/arm 시험학습은50명59장이다. 그환자의전체175장을제외했다. 본학습gate미시작, 지정runtime/arms폴더부재, source의selected-unit준비경로를확인했다.
- M1/M2각456명실제coverage, 공개backbone640명committed trace, head32/80·개발40·reference40·cache/공격/기존preflight를구분했다. 3680개JSON/CSV/JSONL182,820,591bytes를검색해추가private사용hit0을확인했다. 조사기록상미사용이지외부미기록실행까지부정하는증명은아니다.
- 전체14755명환자별A/B/C/D/E/U·originalrole·사용flag·근거목록을생성했다. 별도CVPR후보는D중originalprivate8426명에한정했고폐기종273명/기흉559명이남았다. Officialtest/census,CVPRlocked140+140,originalprivacyholdout은가져오지않았다.
- 사전salt/hash로4213/4213명가상분할: 개발폐기종135/기흉268,별도확인138/291명. condition당1장hash선택명부8338행을만들었으나실험채택/원래분할변경은아니다. 환자multi-label중복을보존하고독립표본으로더하지않았다.
- 감사35.649초, 독립검산1.565초/100,948항목PASS. 이횟수는행별검사수로통계증거가아니다. 기존192장/실패adoption불변,새GPU/추론/생성/DP0이다.
- 판정: 별도CVPR자료구성에긍정적. Private효용은미확인. 미래졸논K5/K10이이환자로학습되면그모델에독립reference로쓸수없다. 다음은20–35분고정PadChest전처리/출력/질환구별력검증이다. 기존소비공개324명(폐기종15/기흉45)후보는최종확인과분리한다.
- TRACK1_PATIENT_USAGE_AUDIT_RESULTS_20260917.md/HTML, 환자이력CSV·가상명부·독립검산·상태문서를연결했다. 직전132번worklog한글인코딩만복원했다. 원격main확인/push는하지않았다.
'''
 w.write_text(first+new+'\n'+rest.lstrip('\n'),encoding='utf-8')
 print(json.dumps({'status':d['status'],'decision_sha256':sha(P/'research_decision.json')},indent=2))
if __name__=='__main__':main()
