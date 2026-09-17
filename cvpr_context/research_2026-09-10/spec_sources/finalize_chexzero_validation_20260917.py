"""Record the completed failed measurement gate and stop this classifier search."""
import csv,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];ROOT=R.parent.parent;C=ROOT/'code_working';O=C/'_reports/chexzero_validation_20260917_v1'
REPORT='TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md';PROTOCOL='TRACK1_CHEXZERO_VALIDATION_PROTOCOL_20260917.md'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def prepend(p,text):
 first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+text+'\n\n'+rest.lstrip('\n'),encoding='utf-8')

def main():
 res=read(O/'result.json');met=read(O/'metrics.json');v=read(O/'verification.json');par=read(O/'parity.json')
 if v['status']!='PASS_INDEPENDENT_CHEXZERO_VALIDATION' or met['joint_pass'] or any(x['pass'] for x in met['criterion'].values()):raise ValueError('This report is specific to the verified two-target failure')
 if v['result_sha256']!=sha(O/'result.json') or v['metrics_sha256']!=sha(O/'metrics.json'):raise ValueError('Verification binding')
 # Standalone figure: operational reference lines do not imply clinical thresholds.
 fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),sharex=True,sharey=True)
 comparisons=['rest','opposite','normal','effusion'];labels=['Other three groups','Opposite target','No Finding','Target-free effusion']
 for ax,target,color in zip(axes,['Emphysema','Pneumothorax'],['#2874a6','#b04454']):
  for i,name in enumerate(comparisons):
   x=met['metrics'][target+'__'+name];lo,hi=x['auc_ci95'];auc=x['auc']
   ax.errorbar(auc,3-i,xerr=[[auc-lo],[hi-auc]],fmt='o',color=color,capsize=4,markersize=6)
   ax.text(.99,3-i+.14,f'{auc:.3f} [{lo:.3f}, {hi:.3f}]',ha='right',va='bottom',fontsize=9)
  ax.axvline(.5,color='#888888',linestyle=':',linewidth=1);ax.set_xlim(.2,1);ax.set_ylim(-.5,3.6)
  ax.set_yticks([3,2,1,0],labels);ax.set_title(target+' — gate FAIL');ax.set_xlabel('AUC with patient bootstrap 95% interval');ax.grid(axis='x',alpha=.15)
 axes[1].tick_params(labelleft=True)
 fig.suptitle('Fixed CheXzero ensemble: fresh NIH 80 patients, 20 per group',fontsize=12)
 fig.tight_layout();(R/'assets').mkdir(exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(R/'assets'/f'chexzero_validation_auc_20260917.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
 table=['|출력 / 양성군|대조군|양성/음성|AUC (95% CI)|AP / prevalence|','|---|---|---:|---:|---:|']
 for target in ['Emphysema','Pneumothorax']:
  for name,label in zip(comparisons,['나머지 세 군','상대 target','No Finding','target-free Effusion']):
   x=met['metrics'][target+'__'+name]
   table.append(f"|{target}|{label}|{x['n_positive']}/{x['n_negative']}|{x['auc']:.4f} ({x['auc_ci95'][0]:.4f}–{x['auc_ci95'][1]:.4f})|{x['ap']:.4f} / {x['prevalence']:.4f}|")
 table='\n'.join(table)
 max_norm=max(max(x['normalized_image_maxabs'],x['normalized_text_maxabs']) for x in par['per_checkpoint'])
 max_score=max(x['score_maxabs'] for x in par['per_checkpoint'])
 public=R/'spec_sources/chexzero_validation_20260917_v1';public.mkdir(exist_ok=False)
 for name in ['result.json','metrics.json','verification.json','parity.json']:
  shutil.copyfile(O/name,public/name)
 public_record=dict(status='VERIFIED_FAILED_JOINT_MEASUREMENT_GATE',contract_sha256=sha(O/'contract.json'),selected_manifest_sha256=sha(O/'selected_images_private.csv'),
  original_plan_manifest_sha256=sha(C/'_reports/chexzero_evaluator_plan_20260917_v1/selected_images_private.csv'),
  evidence_files={n:sha(public/n) for n in ['result.json','metrics.json','verification.json','parity.json']},
  scope='Public aggregate record; individual images, patient IDs, tensors and checkpoint outputs remain in local execution packet.')
 save(public/'aggregate_record.json',public_record)
 report=f'''# 방향1: CheXzero 실제 NIH80명 검증 결과

**판정: 나쁜 결과다. 마지막 단일 대안 CheXzero도 두 질환 모두 사전 운영 관문을 통과하지 못했다. 이 폐기종·기흉 공동 가설에서 classifier 탐색을 종료하고, classifier에 의존한 targeted 생성 경로를 보류한다.**

이번에는 준비에 그치지 않았다. 예약한80명·80장을 공식10개 checkpoint ensemble로 실제 추론했다. 폐기종의 rest/opposite AUC는 **0.6742/0.5900**, 기흉은 **0.5967/0.4900**이다. 공식 CPU와 GPU 대응 검사 및 독립 pixel/통계 검산은 통과했다. 구현 검산 PASS를 평가기 성능 성공으로 바꾸지 않는다.

2026-09-17, 큰 단계2·방향1. [실행 전 명세]({PROTOCOL}) · [후보 검토](TRACK1_CHEXZERO_REVIEW_20260917.md) · [직전 PadChest 실패](TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md). 이 결과는 생성모델·private head의 효과를 직접 시험한 것이 아니므로 patient-private adaptation 전체의 반증은 아니다. **현재 두 조건의 private 생성 효용은 여전히 미확인이다.**

## 1. 실제로 실행한 범위

직전 예약 manifest를 byte 그대로 사용했다. 환자당 hash로 선택한 PA 영상1장, 폐기종+/기흉−20명, 기흉+/폐기종−20명, 두 target 없는 흉수20명, target/흉수 없는 환자의 No Finding 영상20명이다. ‘E-only/P-only’는 상대 target이 없다는 뜻이며 다른 동반질환이 전혀 없다는 뜻이 아니다. NIH weak label의 부재는 임상적으로 확정된 질환 부재와 다르다.

공식10개 checkpoint 모두를 실행했고 좋은 checkpoint만 고르지 않았다. 네 문자열·전처리·patient manifest·GPU precision·score·AUC gate·bootstrap·수치 허용오차·생산/독립검산 source를 추론 전에 한 contract에 결속했다. 새로운 환자·prompt·temperature·모델 조합을 결과에 맞춰 추가하지 않았다.

공식 방식대로 각 checkpoint의 positive/negative cosine에 pair softmax를 적용한 뒤 평균했다. CLIP logit_scale 미적용, template `{{}}`/`no {{}}`, GPU FP32. 개별모델 cosine와 pair-score는 검산용으로 저장했으며 개별 AUC를 이용한 선택은 하지 않았다.

## 2. 조건 구별력 — 두 target 모두 실패

{table}

![CheXzero 고정 NIH80명 AUC와 불확실성](assets/chexzero_validation_auc_20260917.png)

[그림 PDF](assets/chexzero_validation_auc_20260917.pdf). 각 AUC는 동일 출력에서 정해진 양성군과 대조군을 비교한다. 두 질환 출력의 score 절대값을 서로 교정된 확률로 비교하지 않는다.

사전 기준은 질환별 rest AUC≥0.70/95%하한>0.50, opposite AUC≥0.60/95%하한>0.50 모두다.

- **폐기종:** rest의 CI하한은0.5283으로0.5보다 높지만 점추정0.6742가0.70에 못 미친다. Opposite0.5900은0.60 미만이며 CI0.4100–0.7700이 우연 수준을 포함한다. 경계와0.01 차이라고 기준을 낮추거나 환자를 더 넣어 구제하지 않는다.
- **기흉:** rest0.5967/CI0.4425–0.7467, opposite0.4900/CI0.3050–0.6700. 이번 표본에서 다른 target과 구분한다는 근거를 확보하지 못했다.
- **부분적인 신호:** 정상 대비 폐기종0.7975, 기흉0.6875로 더 높다. 이 신호를 숨기지는 않지만, 정상/비정상 구분 일부만으로 질환 특이적 생성 평가기를 채택할 수는 없다.

이 결과는 ‘CheXzero가 모든 NIH 설정에서 쓸모없다’는 뜻이 아니다. 고정된 weak-label 환자군20명씩·현재 전처리/질의/운영 기준에서 후속 생성평가용 근거를 확보하지 못했다는 결과다. 표본과 동반질환·라벨 오류·기관 차이의 영향을 따로 인과적으로 분리하지 않았다.

앞선 PadChest0.5308과 숫자를 나란히 볼 수는 있지만 **서로 다른80명**이므로 CheXzero가 같은 환자에서 더 좋았다는 대응 비교로 쓰지 않는다. 외부 PadChest 원논문의0.8232/0.7659와 이번 NIH 결과도 데이터·라벨·대조군이 다르다.

## 3. 계산이 맞는지 실제로 확인했다

80장 전부 파일 SHA·PNG decode·공식 label/환자·PA view와 inventory의 cross-patient exact duplicate를 검사했다. 320 grayscale와224 float32 tensor·개별 tensor SHA를 저장했다. 독립 검산기는 OpenCV RGB decode와 고정 공식 `preprocess` 함수를 사용해320 입력을 다시 만들고, 별도의 tensor normalization/interpolation으로224 입력을 재계산했다. **80장 모두 exact equality**였다. Near-duplicate 임상 판정이나 weak label의 정답성을 검증한 것은 아니다.

각 질환군의 명부상 첫 환자1명, 총4명을 모든 checkpoint에서 CPU/GPU replay했다. 공식 `zero_shot.run_softmax_eval`을 CPU에서 직접 호출했고, encoder raw/normalized features, cosine, per-checkpoint score, 최종 ensemble을 GPU와 대조했다. 모델/환자를 결과 이후 고른 replay가 아니다.

|검산 항목|실측 최대 차이|
|---|---:|
|CPU–GPU normalized image/text feature|{max_norm:.3g}|
|CPU–GPU 각 checkpoint pair-score|{max_score:.3g}|
|CPU–GPU 최종 평균 score|{par['ensemble_maxabs']:.3g}|
|독립 embedding→cosine 재계산|{v['max_cosine_reconstruction_difference']:.3g}|
|독립 AUC/AP·bootstrap 재계산|{v['max_metric_abs_difference']:.3g}|

모두 사전 허용오차 안이었다. CPU 공식 경로는 positive/negative마다 이미지를 반복 계산했고, 저장된 raw image feature도 같았다. 학습된 logit_scale을 잘못 곱하거나 단일 positive cosine을 최종 score로 사용한 경로가 아니다.

독립 metric 코드는 생산 함수를 import하지 않고 pair-count AUC와 threshold-block AP를 사용했다.8개 대조×2,000회 = **16,000개 patient bootstrap AUC/AP 쌍**을 전부 다시 계산했다. Independent **{v['checks']:,}항목 PASS**는 산술/결속 검사 수이며 새로운 환자 수나 성능 증거 수가 아니다. 검산기 자체가 모든 모델을 또 forward한 것은 아니며, 실제 모델 대조는 producer 안의 별도 공식 CPU/GPU 경로다.

점수가0.5 근처라는 사실만으로 구현 오류나 질환 확률50%라고 해석하지 않는다. 고정된 unscaled cosine의 pair-score이며, 이번 판정은 그 순위를 사용하는 AUC/AP로 했다. Temperature·threshold를 바꾸는 사후 구제는 하지 않았다.

## 4. 실행 비용과 보존

|실측|값|
|---|---:|
|전체 validation 프로그램|{res['total_seconds']:.3f}초|
|GPU primary+replay inference|{res['gpu_inference_seconds']:.3f}초|
|CPU 공식 reference inference|{res['cpu_reference_seconds']:.3f}초|
|Metric/2,000 bootstrap 계산|{res['metric_seconds']:.3f}초|
|별도 독립 검산|{v['seconds']:.3f}초|
|Peak GPU allocated / reserved|{res['peak_allocated_bytes']/2**30:.3f} / {res['peak_reserved_bytes']/2**30:.3f}GiB|
|GPU image calls / image-examples|110 /840|
|CPU image calls / image-examples|80 /80|
|GPU / CPU text-examples|40 /40|
|새 generator / 학습 / DP|0 /0 /0|

예상20–35분은 구현·문서·실행·검산을 포함한 작업 시간이었다. 위52초는 프로그램 실행 부분만이다. 첫 GPU 실행의 초기화 비용도 측정 구간에 포함한다. Pandas optional accelerator warning이 있었으나 metric/CSV는 고정 NumPy/SciPy/sklearn/stdlib 경로로 완료했고 환경은 바꾸지 않았다.

이번80명은 이제 실제 evaluator 결과를 소비한 환자다. `evaluator_exclusions_private.csv`를 새 사용이력 overlay로 남겼다. 준비 당시 reserved manifest와 과거 감사 ledger는 그대로 보존했다. 후속 생성 reference에서 이80명과 앞선 PadChest evaluator80명을 제외한다. Remaining development4,053명/E-only44명, 탐색 reference32명/군이라는 자원 제한은 유지한다.

Reserved confirmation4,213명은 ID 교집합만 확인했고 pixel·추론·tuning에 쓰지 않았다. 원래 졸논 분할·final/calibration/test, 고정 backbone/head, 과거192장·private utility 미통과, E4/CFG1 adoption 실패를 보존했다.

## 5. 이번 결과로 내리는 결정

**이 공동 가설의 classifier 탐색은 끝낸다.** 두 번째 평가기까지 고정 조건에서 실패했으므로 세 번째 모델, prompt 검색, checkpoint subset, 성공한 질환만 사후에 남기는 실험을 하지 않는다. KID나 raw text cosine만 좋아지는 것을 target-specific private 효용이라고 선언하지 않는다.

현재 중단 범위는 **폐기종·기흉 classifier 기반 targeted 평가 경로**다. 작은 public head의 생성 영향, 사용 이력상 분리된 평가자료, 기존 public/private 방향 차이는 남아 있다. 그러나 이 사실들이 private 생성 효용이나 patient-DP 기여를 대신하지 않는다. 이번에는 생성모델을 실행하지 않았으므로 private 효과 유무를 결론내릴 수 없다.

다음 실험은 자동으로 이어지지 않는다. 다음 설계 작업은20–30분의 **논문 수준 측정·자원 재검토**다. 실제 확보 가능한 전문 판독/신뢰할 주석 등 독립 측정 자원이 있으면 그 자원에 맞는 새 명세를 만들고, 없으면 현재 targeted 설정을 내려놓고 두 보호 연구 방향의 문제–측정–기여 연결을 다시 선택해야 한다. 자료가 없는데 또 다른 classifier만 돌리는 방식으로 이어가지 않는다. 방향2나 새 solver로 자동 전환하지도 않는다.

**좋은 측정 도구를 찾았다는 결과는 아니다. 이번 실행은 부정적이며, 정한 중단선을 실제로 적용한 결과다. CVPR 핵심 privacy 기여는 여전히 미확보다.**

## 6. 기록과 재현

실행 폴더: `code_working/_reports/chexzero_validation_20260917_v1`. 실제 selected manifest/tensor/320배열/HDF5,10개 checkpoint score/feature, CPU/GPU replay10개,patient score, bootstrap,metric,검산,새 consumed overlay를 보존했다.

[공개 집계 기록](spec_sources/chexzero_validation_20260917_v1/aggregate_record.json) · [전체 AUC/AP 및 CI](spec_sources/chexzero_validation_20260917_v1/metrics.json) · [실행 통계](spec_sources/chexzero_validation_20260917_v1/result.json) · [독립 검산](spec_sources/chexzero_validation_20260917_v1/verification.json). 공개 JSON만으로 로컬 환자 pixel 전체를 재검증했다고 주장하지 않는다.

- Contract SHA: `{sha(O/'contract.json')}`
- Result SHA: `{sha(O/'result.json')}`
- Metrics SHA: `{sha(O/'metrics.json')}`
- Selected80 manifest SHA: `{sha(O/'selected_images_private.csv')}`

원격 main 확인/push는 이번 범위에 포함하지 않았다. 모든 판단은 로컬 실제 실행 기록에 근거한다.
'''
 with (R/REPORT).open('x',encoding='utf-8') as f:f.write(report)
 d=dict(status='NEGATIVE_CHEXZERO_JOINT_MEASUREMENT_GATE_CLASSIFIER_SEARCH_CLOSED',completed_utc=datetime.now(timezone.utc).isoformat(),report=REPORT,protocol=PROTOCOL,directory=str(O),current_step=2,track=1,
  actual_validation_patients=80,official_checkpoint_count=10,evaluator_validated=False,joint_gate_passed=False,both_target_gates_failed=True,
  primary_auc={t:met['metrics'][t+'__rest']['auc'] for t in ['Emphysema','Pneumothorax']},opposite_auc={t:met['metrics'][t+'__opposite']['auc'] for t in ['Emphysema','Pneumothorax']},
  classifier_search_closed=True,targeted_classifier_path_paused=True,private_generation_utility_established=False,previous192_utility_gate_passed=False,DP_allowed_to_expand=False,automatic_targeted_generation_allowed=False,
  reserved_confirmation_consumed=False,original_roles_changed=False,new_GPU_inference=True,new_generation=0,new_DP=0,remaining_target_development_patients=4053,remaining_exclusive_emphysema=44,
  evaluator_exclusions_manifest=str(O/'evaluator_exclusions_private.csv'),independent_checks=v['checks'],execution_seconds=res['total_seconds'],gpu_inference_seconds=res['gpu_inference_seconds'],verification_seconds=v['seconds'],
  source_sha256={n:sha(O/n) for n in ['contract.json','result.json','metrics.json','verification.json','selected_images_private.csv','evaluator_exclusions_private.csv','remaining_development_private.csv']})
 save(O/'research_decision.json',d)
 for name in ['research_state.json','patient_baseline_spec.json']:
  s=read(R/name);s['pre_chexzero_validation_result_report_20260917']=s['current_result_report'];s['current_result_report']=REPORT;s['current_planning_report']=REPORT;s['next_task_plan_report']=REPORT;s['last_actual_model_result_report']=REPORT
  s['next_task']='One20-30min paper-level measurement/resource decision: retain the failed fixed CheXzero result and close classifier search for the joint Emphysema/Pneumothorax hypothesis. Determine whether real expert/annotation resources justify a separately specified measurement route; if unavailable, revise the paper question within the two authorized protection directions. No third classifier, prompt/ensemble search, automatic targeted generation, solver or DP.'
  s['next_task_output']='A concrete measurement-resource-backed next question, or a documented hold on this targeted path. Do not represent this evaluator failure as measured private-generation failure or a completed CVPR contribution.'
  s['step2_open_items']=['CheXzero NIH80 actual inference complete: E/P rest AUC .6741667/.5966667 and opposite .5900/.4900; BOTH fixed gates failed. Official CPU/GPU and independent pixel/score/bootstrap verification passed.', 'Classifier search for the joint target hypothesis CLOSED; no third candidate, checkpoint subset, prompt search or target deletion.', 'New80 now evaluator-consumed; exclude both prior PadChest80 and new CheXzero80. Remaining4053/E-only44; original reserved4213/final sets unchanged.', 'Private targeted generation utility remains unmeasured. Public-head generation effect and original192-image private-utility failure remain historical evidence, not current privacy success.', 'Next is a paper-level measurement/resource decision, not automatic direction2, solver or DP execution.']
  active=s['active_execution'];keys=['last_completed_output','last_completed_report','latest_planning_status','next_substep_estimated_work_minutes','new_two_track_experiment_status']
  active['historical_pre_chexzero_validation_status_pointers']={k:active.get(k) for k in keys}
  active.update(status='no_running_execution',current_step=2,chexzero_validation=d,last_completed_output=str(O),last_completed_report=REPORT,latest_planning_status='joint_target_classifier_path_paused_measurement_resource_decision_next',next_substep_estimated_work_minutes=[20,30],new_two_track_experiment_status='chexzero_actual80_complete_both_measurement_gates_failed',additional_execution_contract_fixed=False,next_execution_not_started=True)
  s['completed_chexzero_validation_20260917']=d;s['latest_user_execution_scope_20260917']='Complete the already-fixed NIH80 CheXzero validation, official CPU/GPU parity, independent verification and consumption records; apply the agreed stop rule on any target failure.'
  (R/name).write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 authority=f'- Latest ACTUAL CheXzero validation (2026-09-17): read `{REPORT}`. NEGATIVE: fresh80patients, all10 official checkpoints, E/P rest AUC .6741667/.5966667, opposite .5900/.4900; BOTH gates failed. Official CPU/GPU four-patient replay on each checkpoint PASS, finalscore maxdiff5.96e-8; all80 pixel preprocessing exact and16000 bootstrap pairs independently recomputed,17222 integrity checks PASS. These are not efficacy samples. CLASSIFIER SEARCH CLOSED for joint Emphysema/Pneumothorax targeted path; no third model, prompt/checkpoint/target rescue, generation or DP. Private-generation efficacy remains unmeasured; prior192 failure preserved. Newly consumed80 must be excluded through chexzero_validation_20260917_v1/evaluator_exclusions_private.csv in addition to previous PadChest80. Remaining4053/E-only44; reserved4213 and original thesis final/roles preserved. Actual program52.374s/GPU4.962s/CPUref7.511s/verification10.489s; no training or generation. Next20-30min is a paper-level measurement/resource decision: only actual expert/annotation resources can justify a new measurement specification; otherwise revise the question, no automatic direction2 or solver experiments. Stage2/track1,no running execution. Earlier latest/next statements are history.'
 prepend(ROOT/'AGENTS.md',authority)
 short='**2026-09-17 최신 실제 검증 — CheXzero도 두 질환 미통과, classifier 탐색 종료:** [NIH80명 실제 추론·공식 CPU/GPU·독립 검산]('+REPORT+'). 폐기종/기흉 rest AUC0.6742/0.5967, 상대 target0.5900/0.4900으로 두 관문 모두 실패했다. 구현 검산은 통과했지만 신뢰할 조건 평가기 채택에는 실패다. 새80명은 소비자료로 제외하고 잔여4053명·reserved4213명·졸논 final·기존192장 실패를 보존한다. 이 두 질환의 classifier 기반 targeted 경로를 보류한다. 다음20–30분은 실제 전문판독/주석 자원과 논문 질문의 연결을 재검토하는 설계 판단이며, 제3평가기·prompt구제·새생성·DP·방향2 자동전환은 없다. Private 생성효용은 미판정, 큰단계2·방향1 유지. 아래 최신/다음은 이전 이력이다.'
 prepend(R/'RESEARCH_FRAMEWORK.md',short);prepend(R/'REALISTIC_RESEARCH_PLAN_20260916.md',short)
 prepend(ROOT/'CURRENT_STATUS.md',short.replace('('+REPORT+')','(CVPR%20주제%20탐색/research_2026-09-10/track1_chexzero_validation_results.html)'))
 prepend(ROOT/'WORKLOG.md',f'''## 136-CHEXZERO-REAL-NIH-VALIDATION (2026-09-17 KST)

- 큰단계2·방향1. 예상20–35분으로 시작해 예약80명·공식10checkpoint를 실제 실행했다. 명부/score/기준/runner/verifier를 outcome 전에 고정했다.
- 나쁜 결과: E/P rest AUC0.6741667/0.5966667, 상대 target0.5900/0.4900. 둘 다 gate 실패. 정상 대조0.7975/0.6875의 일부 신호와 질환 간 구별 실패를 함께 기록한다. 이전 PadChest와 환자가 달라 paired 우월 비교가 아니다.
- CPU 공식 run_softmax_eval을10모델×4환자에서 실제 실행. GPU full80+4replay, 모두사전오차내; ensemble차이5.96e-8. 독립pixel80개 exact,bootstrap16000쌍 재계산,max3.33e-16,17222항목PASS. 1회실행/검산 첫시도통과,결과후모델/허용오차/데이터변경없음.
- 실측전체52.374초/GPU4.962초/CPUreference7.511초/검산10.489초/peakallocated0.601GiB. 새학습/생성/DP0.
- classifier탐색종료·공동targeted경로보류를실제적용. 새로운classifier/prompt/checkpoint구제없음. Private head효용은여전히미평가이고프로젝트전체불가능이라고해석하지않는다.
- evaluator80소비overlay추가,계획manifest/과거ledger보존. 잔여4053/E-only44,원reserved4213/final/원역할/기존192실패보존.
- 보고서 `{REPORT}`, protocol,PNG/PDF,공개aggregate JSON,실행packet,상태문서기록. 다음은20–30분측정자원/논문문제의설계판단. 원격확인/push없음.
''')
 print(json.dumps({'status':d['status'],'report':REPORT,'decision_sha256':sha(O/'research_decision.json')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
