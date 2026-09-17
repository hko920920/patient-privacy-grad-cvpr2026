"""Publish the completed fixed evaluator failure; no model execution."""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

R=Path(__file__).resolve().parents[1];ROOT=R.parent.parent;C=ROOT/'code_working';O=C/'_reports/padchest_validation_20260917_v1'
REPORT='TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md'
PROTOCOL='TRACK1_PADCHEST_VALIDATION_PROTOCOL_20260917.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepend(p,txt):
 old=p.read_text(encoding='utf-8');first,rest=old.split('\n',1);p.write_text(first+'\n\n'+txt+'\n\n'+rest.lstrip('\n'),encoding='utf-8')
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 res=read(O/'result.json');v=read(O/'verification.json');met=read(O/'metrics.json');con=read(O/'contract.json')
 if res['joint_pass'] or v['status']!='PASS_INDEPENDENT_EVALUATOR_VERIFICATION':raise RuntimeError('Unexpected decision; review report wording')
 # Standalone scientific score-distribution figure, from saved logits only.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rr=rows(O/'selected_images_private.csv');log=np.load(O/'logits.npy');model=read(O/'model_metadata.json');groups=['Emphysema','Pneumothorax','No Finding','Effusion']
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');rng=np.random.default_rng(0)
 for ax,target in zip(axes,groups[:2]):
  j=model['pathologies'].index(target)
  data=[np.array([log[i,j] for i,x in enumerate(rr) if x['stratum']=='development_supplement' and x['group']==g]) for g in groups]
  ax.boxplot(data,labels=['Emphysema','Pneumothorax','No Finding','Effusion'],showfliers=False,whis=(0,100))
  for k,vals in enumerate(data):ax.scatter(np.full(len(vals),k+1)+rng.uniform(-.1,.1,len(vals)),vals,s=15,alpha=.65,color='#2563a6')
  ax.set_title(target+' output (raw logit)');ax.set_ylabel('Raw logit; not a calibrated probability');ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Fixed PadChest evaluator on fresh NIH development patients\n20 different patients per group; one image each')
 assets=R/'assets';assets.mkdir(exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(assets/f'padchest_validation_scores_20260917.{ext}',dpi=160)
 plt.close(fig)
 def fmt(q):return f"{q:.4f}"
 def interval(q):return f"{q[0]:.4f}–{q[1]:.4f}"
 def table(st):
  out=['|출력 / 양성군|대조군|양성/음성 환자|AUC (95% bootstrap)|AP / prevalence|','|---|---|---:|---:|---:|']
  for t in groups[:2]:
   for cmp,label in [('rest','나머지 세 군'),('opposite','상대 target'),('normal','No Finding'),('effusion','target-free Effusion')]:
    x=met['metrics'][st+'__'+t+'__'+cmp]
    out.append(f"|{t}|{label}|{x['n_positive']}/{x['n_negative']}|{fmt(x['auc'])} ({interval(x['auc_ci95'])})|{fmt(x['ap'])} / {fmt(x['prevalence'])}|")
  return '\n'.join(out)
 report=f'''# 방향1: PadChest 평가기 실제 NIH 검증 결과

**판정: 나쁜 결과다. 고정 PadChest 평가기는 이번 NIH 폐기종·기흉의 조건특이적 생성 효용을 판정할 개발 관문을 통과하지 못했다.** 코드/전처리/통계 검산은 통과했지만, 새 검증군에서 두 target의 one-versus-rest AUC가 모두 **0.5308**이다. 이 결과는 private adaptation 방법의 실패가 아니라 **현재 평가기 후보의 부적합**이다. 작은 head의 targeted 생성 효용과 patient-DP는 아직 판단하지 않았다.

2026-09-17 실제 실행. 큰 단계2·방향1. [추론 전 고정 명세]({PROTOCOL}) · [직전 사용 이력 감사](TRACK1_PATIENT_USAGE_AUDIT_RESULTS_20260917.md). 예상20–35분으로 시작했고 아래 실측은 프로그램 실행 시간이다. 새 생성·학습·DP는0이다.

## 1. 준비에서 끝내지 않고 무엇을 실행했는가

**309명·309장의 실제 NIH PA 영상**을 해시·디코딩하고, 고정 `densenet121-res224-pc`를 RTX3070에서 실행했다. 환자당1장이다. 모델을 비교해 가장 좋은 것을 고르거나 score를 보고 환자/영상/전처리를 변경하지 않았다.

|Stratum|폐기종-only|기흉-only|No Finding|target-free Effusion|Dual-secondary|합계|
|---|---:|---:|---:|---:|---:|---:|
|기존 비잠금 evaluator 자료|4|34|95|85|11|229|
|새 target-development evaluator 전용|20|20|20|20|0|80|

기존 폐기종15명 중11명은 기흉도 있어 단독군이4명뿐이었다. 이것은 **score 확인 전 metadata에서 확인**했고, 부족한 support 때문에 새 development의 군별20명을 hash로 골라 주 검증군으로 고정했다. 결과가 나빠서 환자를 추가한 것이 아니다. 새로운80명과 기존229명은 별도 stratum으로 보고하며 좋은 쪽만 선택하지 않는다.

환자군은 현재 보유 PA 영상의 label 합집합으로 상호배타적으로 정했다. E-only/P-only는 상대 target이 없다는 뜻이며, 다른 질환이 전혀 없는 환자라는 뜻은 아니다. No Finding/Effusion control도 같은 환자의 다른 보유 영상에 target이 있으면 제외했다. 이후 해당군 label을 가진 영상 중 hash 최솟값1장을 사용했다. Dual11명은 primary 판정에 넣지 않았다. NIH weak label의 부재를 임상적으로 확정된 질환 부재라고 부르지 않는다.

## 2. 주 검증군80명 — 두 조건 모두 미통과

{table('development_supplement')}

사전 운영 기준은 target 대 나머지군 AUC≥0.70/95%하한>0.50, 상대 target 대조 AUC≥0.60/95%하한>0.50을 **각 질환 모두** 만족하는 것이다. 두 질환 모두 두 항목을 만족하지 못했다. 평균값도 기준과 거리가 있어, 단순히 엄격한 신뢰구간 때문에 탈락한 결과가 아니다.

특히 폐기종과 기흉을 서로 구분하는 AUC는0.5025/0.5425이고, 흉수와 구분하는 AUC는0.4375/0.3700이다. 정상 대조에선0.6525/0.6800으로 더 높다. 즉 이 자료에서 점수가 일부 일반적 비정상 소견을 반영할 수는 있어도, 우리가 필요한 target-specific 판별력이 확인되지 않았다. **생성물의 이 점수만 올라갔다고 해당 target이 더 잘 표현됐다고 해석할 수 없다.**

![주 검증군 raw-logit 분포](assets/padchest_validation_scores_20260917.png)

[점수 분포 PDF](assets/padchest_validation_scores_20260917.pdf). 점은 독립 환자1명이다. 두 출력의 logit 절대값은 서로 교정된 scale이 아니므로 질환 간 값 자체를 비교하는 그림이 아니다.

## 3. 기존 공개자료 결과도 함께 남긴다

{table('old_nonlocked')}

기존군은 모델/metric 개발 등에 이미 소비된 공개자료다. 폐기종 양성4명은 특히 작아 구간 해석이 불안정하다. 일부 정상 대조 수치가 좋다고 새80명의 결과를 대신하지 않는다. 기존군에서도 기흉 대 흉수0.4066으로 조건특이성 문제와 같은 방향이다. 이는 원인을 확정하는 독립 외부기관 재현은 아니다.

AP는 stratum과 대조군의 양성 비율에 의존한다. 예를 들어 기존 기흉 대 폐기종의 AP0.8597은 prevalence0.8947보다 낮고, 높은 AP 숫자만으로 좋은 평가기라고 할 수 없다. 모든 표에 prevalence를 같이 적었다. Bootstrap은 군별 환자를 복원추출해 크기를 유지한2,000회, percentile95%구간이며 다중 비교 조정이나 임상 power 보장은 없다.

## 4. 실행 오류나 출력 해석 오류인가

현재 확인 범위에서는 그 근거를 찾지 못했다.

- 설치 소스와 checkpoint SHA, PyTorch/XRV/resize 등 버전을 추론 전에 결속했다. PC checkpoint의 두 출력은 학습된 slot이며 Emphysema5/Pneumothorax3 index를 확인했다. NIH/all 모델은 실행하지 않았다.
- 8bit grayscale → 공식 normalize[-1024,1024] → center crop → skimage224 resize를 실행했다. 독립 코드가309장 전체의 pixel에서 tensor를 재계산했고 exact equality였다.
- 주 점수는 classifier의 raw logit이다. 기본 forward의 sigmoid+op_norm을 별도로 재계산했고 일치했다. 미학습 slot의0.5 출력을 target 점수로 쓰지 않았다.
- 고정4개 입력의 batch/replay raw logit 최대 차이는0이었다. 모델의 전체 추론을 별도 구현으로 재현했다는 뜻은 아니다.
- 독립 코드는 생산 metric 함수를 import하지 않고 pair-count AUC와 threshold-block AP,32,000개 bootstrap AUC/AP 쌍을 전부 재계산했다. 최대 차이 **{v['max_bootstrap_auc_ap_abs_difference']:.3g}**이다.
- 선택 환자/영상 hash 규칙, 전체 local inventory의 cross-patient exact duplicate 부재, 공식 metadata·환자·label, 원본 파일 SHA를 확인했다. Near-duplicate 임상 판정은 하지 않았다.

독립 검산 **{v['checks']:,}항목 PASS**는 위 무결성 확인 수이며 성능 표본 수가 아니다. 평가기 판별력은 새80명에서 본 결과 그대로 나쁘다. 가능한 원인에는 PadChest→NIH 분포 차이, weak-label 정의/동반질환 차이, 비특이적 특징 의존 등이 있지만 이번 실험은 이를 인과적으로 분리하지 않았다.

## 5. 실제 비용과 보존 상태

|실측|값|
|---|---:|
|실제 validation 프로그램 전체|{res['total_seconds']:.3f}초|
|그 중 GPU 추론·4입력 replay|{res['inference_seconds']:.3f}초|
|주 forward + replay|{res['forward_calls']}회 / {res['forward_examples']} image-examples|
|측정 peak allocated GPU memory|{res['peak_allocated_bytes']/2**30:.3f}GiB|
|독립 검산|{v['seconds']:.3f}초|
|새 generator forward / backward / 생성 / DP|0 / 0 / 0 / 0|

메모리는 PyTorch allocated 값이며 전체 프로세스 reserved/시스템 GPU 사용량을 뜻하지 않는다. 추론1.19초는 package 전체 작업시간과 다르다. Pandas optional accelerator 버전 warning은 있었으나 이 실행의 CSV/통계는 stdlib/NumPy/Scikit-learn 경로로 완료됐고 환경을 바꾸지 않았다.

새 evaluator80명은 이제 **평가 결과를 소비한 환자**다. 직전 감사의 D등급은 당시 상태의 역사적 기록으로 남기고, `evaluator_exclusions_private.csv`를 현재 추가 사용 이력으로 결속했다. 이들을 이후 생성 reference/confirmation에 재사용하지 않는다. 나머지 development4,133명 중 상호배타적 폐기종64명·기흉197명·No Finding2,152명·Effusion774명, dual51명·기타895명이 남는다.

Reserved confirmation4,213명은 새 이미지 접근·추론·tuning에 쓰지 않았다. 명부상의 ID 교집합만 확인했다. 졸논 final/test와 원본 분할, 고정 backbone/head, 과거192장 생성물과 당시 private 효용 미통과 판정, 과거 E4/CFG1 실패 adoption 모두 보존했다. 미래 졸논 모델이 candidate 환자를 학습하면 독립 reference로 쓸 수 없다는 기존 조건도 유지한다.

## 6. 다음 판단 — 자료는 있지만 신뢰할 조건 평가기가 없다

이번 작업은 평가기 검증까지 **실제로 완료**했다. 새 데이터 부족 문제는 다시 생기지 않았다. 현재 병목은 두 조건의 생성 효용을 판정할 수 있는 측정 방법이다. Private 생성이 유용한지/무용한지는 여전히 미확인이다.

다음은 **20–30분의 평가방법 재설계 검토 한 번**이다. 필요한 것은 여러 classifier를 돌려 좋은 점수를 고르는 일이 아니라, 두 조건의 label 정의·학습 출처·NIH overlap과 검증 근거를 확인해 현재 문제에 맞는 대안 한 후보를 정당화할 수 있는지 판단하는 것이다. 후보가 정당화되면 새 명세 아래 개발용 검증을 하고, 기존80명은 평가기 선택에 이미 사용된 자료로 명시해야 한다. Reserved confirmation은 그 선택에 사용하지 않는다.

대안이 없으면 RAD-DINO KID나 raw text cosine만으로 target-specific private 효용이 성공했다고 판정하지 않는다. 임상 판독 자원/독립 자료/문제 설정을 재검토해야 한다. 조건 민감한 평가 근거를 확보하기 전에는 targeted 생성 확대·residual solver·patient-DP로 넘어가지 않는다. 현재 실패는 평가기 후보 하나의 실패이며 patient-private adaptation 전체의 반증은 아니다.

## 7. 재현 파일

로컬 실행 폴더: `code_working/_reports/padchest_validation_20260917_v1`.

- `contract.json` / `selected_images_private.csv`: score 전 코드·source·환자·영상 고정.
- `normalized_inputs.npy` / `image_audit.json` / `logits.npy` / `forward_replay.npz`: 실제 입력/출력.
- `metrics.json` / `bootstrap.npz` / `result.json` / `verification.json`: 전 결과와 독립 계산.
- `evaluator_exclusions_private.csv` / `remaining_development_private.csv`: 새 소비 이력 및 잔여 후보.
- Contract SHA: `{sha(O/'contract.json')}`.
- Result SHA: `{sha(O/'result.json')}`.
- Metrics SHA: `{sha(O/'metrics.json')}`.

원격 main 확인이나 push는 이 검증 범위에 포함하지 않았다. 보고서는 로컬 실제 실행과 기록에 근거한다.
'''
 (R/REPORT).write_text(report,encoding='utf-8')
 d=dict(status='NEGATIVE_PADCHEST_CONDITION_DISCRIMINATION',completed_utc=datetime.now(timezone.utc).isoformat(),report=REPORT,protocol=PROTOCOL,directory=str(O),current_step=2,track=1,
  actual_validation_patients=309,primary_new_development_patients=80,evaluator_validated=False,private_generation_utility_established=False,previous192_utility_gate_passed=False,
  DP_allowed_to_expand=False,automatic_targeted_generation_allowed=False,reserved_confirmation_consumed=False,original_roles_changed=False,
  new_GPU_inference=True,new_generation=0,new_DP=0,remaining_target_development_patients=4133,remaining_exclusive_emphysema=64,
  evaluator_exclusions_manifest=str(O/'evaluator_exclusions_private.csv'),primary_auc={t:met['metrics']['development_supplement__'+t+'__rest']['auc'] for t in groups[:2]},
  independent_checks=v['checks'],execution_seconds=res['total_seconds'],inference_seconds=res['inference_seconds'],verification_seconds=v['seconds'],
  source_sha256={n:sha(O/n) for n in ['contract.json','result.json','metrics.json','verification.json','selected_images_private.csv','evaluator_exclusions_private.csv','remaining_development_private.csv']})
 with (O/'research_decision.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
 for name in ['research_state.json','patient_baseline_spec.json']:
  s=read(R/name);s['pre_padchest_validation_result_report_20260917']=s['current_result_report'];s['current_result_report']=REPORT;s['current_planning_report']=REPORT;s['next_task_plan_report']=REPORT
  s['next_task']='One bounded20-30min measurement redesign review: explain fixed PadChest failure without outcome tuning; examine target label definitions, training provenance/NIH overlap and external validation evidence to justify at most one alternative evaluator or clinical assessment route. No model sweep, reserved-confirmation use, targeted generation, solver search or DP execution.'
  s['next_task_output']='A defensible condition-sensitive measurement proposal with explicit validation independence and cost, or a documented measurement-resource blocker; retain the actual failed PadChest scores and all prior generator results.'
  s['step2_open_items']=['Fixed PadChest validation actually completed: fresh80patients, both target-vs-rest AUC .5308, both predeclared criteria failed. Correct implementation is not metric efficacy.', 'New80 evaluator-only patients are now consumed: always exclude evaluator_exclusions_private.csv from subsequent generation reference/confirmation;4133 development candidates remain, including64 E-only/197 P-only.', 'Reserved4213 confirmation patients and original final/census/roles remain protected; metadata intersection checks only.', 'Targeted private generation utility is unmeasured because the required condition-sensitive evaluator is not validated; prior192-image private-utility failure remains.', 'No automatic additional classifier search or generated/DP experiments; next is source-grounded measurement design, at most one justified alternative.']
  s['active_execution']['status']='no_running_execution';s['active_execution']['current_step']=2;s['active_execution']['padchest_validation']=d
  s['completed_padchest_validation_20260917']=d;s['latest_user_execution_scope_20260917']='Complete fixed evaluator actual real-image validation, source/metric verification and records; no generated image/DP expansion.'
  (R/name).write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 authority='- Latest ACTUAL PadChest validation (2026-09-17): read `'+REPORT+'`. NEGATIVE evaluator result, not private-adaptation failure. Actual309unique patients/309images, one hash-selected PA each; old229 (218primary+11dual), prospective newdevelopment80 (20per exclusive target/control). Both fresh target-vs-rest AUC .5308, both95%CI include .5; target-vs-opposite .5025/.5425. Both fixed gates FAILED. Independent34,805 checks PASS; raw logits/default transform/pixel preprocessing checked. Do not call the evaluator validated or proceed to targeted generation/DP. New80 patients are now consumed for evaluator validation and MUST be excluded from future generation reference using padchest_validation_20260917_v1/evaluator_exclusions_private.csv; original usage ledger is frozen history. Remaining development4133 includes64 E-only/197 P-only/2152NF/774Eff; reserved4213 confirmation untouched by images/inference. Original thesis roles, old192images/utilityfailure and public-adoption-failure preserved. GPU classifier inference1.191s, no generator/training/DP. Next one20-30min source/measurement redesign review, at most one defensible alternative, no outcome-based classifier sweep. Stage2/track1, no running execution. Earlier latest/next text is history.'
 prepend(ROOT/'AGENTS.md',authority)
 short='**2026-09-17 最新 실제 평가기 검증 — 나쁜 결과, 두 질환 판정에 사용 불가:** [309명 실제 추론·전처리·통계 검산]('+REPORT+'). 새 개발80명에서 폐기종·기흉 AUC가 각각0.5308로 사전 기준을 모두 통과하지 못했다. 정상 대조 일부 신호와 target 간 판별력을 구분하며, correctness PASS를 성능 성공으로 바꾸지 않는다. 새 evaluator80명은 결과를 소비했으므로 후속 생성 reference에서 제외한다. 잔여 개발4,133명/폐기종-only64명과 reserved confirmation4,213명은 보존한다. 기존192장 실패도 유지하며 새생성·DP0이다. 다음은20–30분 조건 평가방법의 근거 검토이며, 자동classifier탐색/생성 확대는 하지 않는다. 큰 단계2·방향1, 실행중 없음. 아래 최신/다음은 이전 이력이다.'
 prepend(R/'RESEARCH_FRAMEWORK.md',short);prepend(R/'REALISTIC_RESEARCH_PLAN_20260916.md',short)
 prepend(ROOT/'CURRENT_STATUS.md',short.replace('('+REPORT+')','(CVPR%20주제%20탐색/research_2026-09-10/track1_padchest_validation_results.html)'))
 work='''## 134-PADCHEST-REAL-IMAGE-VALIDATION (2026-09-17 KST)

- 큰 단계2·방향1, 예상20–35분. 고정 PadChest-only 평가기를 실제 NIH309명309장에 실행했다. 준비뿐인 단계가 아니다.
- 기존 pool324명 중 E-only4/P-only34/NF95/target-freeEff85/dual11명=229명을 사용했다. E-only4명 부족을 score 전 확인하고 새development에서 군별20명80명을 고정hash로 evaluator-only 분리했다. Reserved confirmation 이미지는 사용하지 않았다.
- 주 검증80명의 E/P one-vs-rest AUC 둘다0.5308,95%CI 각각0.3925–0.6675 /0.3841–0.6808. 상대target AUC0.5025/0.5425. 두질환 모두 사전 관문 실패. 정상과의 구별 일부를 질환특이성 성공으로 해석하지 않았다.
- 모델/소스/전처리/출력 사전결속. 실제pixel309장 및rawlogit/default transform,32,000bootstrap통계쌍을독립계산했다. 34,805확인PASS, 최대metric차이3.33e-16. 이는성능증거표본수가아니다.
- 실측전체30.951초/GPU추론1.191초/22F(317image-examples)/0B/0생성/0DP/peakallocated0.159GiB. 별도검산20.765초. 모델/환자/threshold사후변경없음.
- 새evaluator80명은이제소비자료이며후속reference에서제외한다. 별도제외CSV및잔여4,133명명부를저장했고E-only64명/P-only197명등이남는다. 기존frozen감사ledger의D는과거상태로보존하고새사용이력을별도로결속했다.
- 판정: 현재평가기후보에는부정적,private효용에는미판정. 현재평가기점수만으로targeted생성효용을판정하지않는다. 다음20–30분은학습출처/label/NIHoverlap/외부검증을근거로조건평가방법대안한후보가성립하는지검토한다. 무작위classifier탐색/생성/DP자동확대는없다.
- TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md/HTML·rawscore분포그림·명세·원본출력·검산·상태문서에기록한다. 원격확인/push는하지않았다.
'''
 prepend(ROOT/'WORKLOG.md',work)
 print(json.dumps({'status':d['status'],'report':REPORT,'decision_sha256':sha(O/'research_decision.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
