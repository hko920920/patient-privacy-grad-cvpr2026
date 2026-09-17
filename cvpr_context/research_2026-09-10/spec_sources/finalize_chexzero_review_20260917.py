"""Record justified candidate readiness, not a completed NIH efficacy experiment."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];ROOT=R.parent.parent;C=ROOT/'code_working';P=R/'spec_sources/chexzero_review_20260917_v1';D=C/'_reports/chexzero_evaluator_plan_20260917_v1'
REPORT='TRACK1_CHEXZERO_REVIEW_20260917.md';PROTOCOL='TRACK1_CHEXZERO_REVIEW_PROTOCOL_20260917.md';REV='5c341db0fe0db2f663a2c136c7120a8b303f1d11'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def prepend(p,txt):
 first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+txt+'\n\n'+rest.lstrip('\n'),encoding='utf-8')
def main():
 a=js(P/'asset_inspection.json');v=js(P/'readiness_verification.json');q=js(D/'cohort_contract.json');download=js(P/'checkpoint_acquisition.json')
 if v['status']!='PASS_CANDIDATE_READINESS_NOT_PERFORMANCE' or len(a['files'])!=10:raise ValueError('Readiness incomplete')
 spec=dict(status='FIXED_CANDIDATE_AND_COHORT_RUNNER_PENDING',created_utc=datetime.now(timezone.utc).isoformat(),candidate='CheXzero_official_ten_checkpoint_equal_weight_ensemble',
  revision=REV,model_count=10,checkpoints=[{k:x[k] for k in ['filename','sha256','bytes']} for x in a['files']],
  checkpoint_directory=str(C/'_models/chexzero_official_20220916'),source_directory=str(P/'source'),source_sha256={str(p.relative_to(P)):sha(p) for p in (P/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts},
  prompts=a['prompts'],prompt_templates=['{}','no {}'],token_ids=a['token_ids'],context_length=77,
  score='mean_over_10_checkpoints(sigmoid(cos(image,text_positive)-cos(image,text_negative)))',logit_scale_applied=False,temperature=1.0,threshold_selection=False,
  preprocessing=dict(grayscale_8bit=True,first_resize=320,first_filter='PIL_LANCZOS',preserve_aspect_ratio=True,zero_padding=True,channels=3,input_range=[0,255],mean=101.48761,std=83.43944,second_resize=224,second_filter='tensor_BICUBIC',antialias=False),
  inference=dict(device='cuda',dtype='float32',eval_mode=True,TF32=False,batch_size=8,expected_image_encoder_examples=800,training=False,backward=False),
  required_next_checks=['Bind actual producer/verifier source hashes before inference','Selected PNG SHA/decode/labels and cross-patient exact duplicates','Official CPU/GPU FP32 feature and +/- score parity on fixed replay cases','All ten checkpoints and identical preprocessing; no subset selection','Independent per-patient ranking/bootstrap and final gate aggregation'],
  cohort_manifest=str(D/'selected_images_private.csv'),cohort_manifest_sha256=sha(D/'selected_images_private.csv'),cohort_contract_sha256=sha(D/'cohort_contract.json'),
  patients=80,patients_per_group=20,bootstrap_iterations=2000,bootstrap_seed=20260917,
  gate=dict(each_target_vs_rest_auc_min=.70,each_target_vs_opposite_auc_min=.60,both_lower95CI_strictly_above=.50,both_targets_must_pass=True),
  forbidden=['NIH-performance-based checkpoint selection','synonym/template search after outcome','new disease deletion after outcome','use of reserved confirmation','third/fourth classifier rescue after joint failure','automatic generation or DP'],
  old_padchest80_excluded=True,reserved_confirmation_used=False,NIH_inference_done=False,patient_private_utility_established=False,
  later_reference_plan_per_condition=32,remaining_development_exclusive_emphysema=44,later_reference_power_established=False,
  next_package_estimated_minutes=[20,35],full_execution_contract_prepared=False)
 save(D/'execution_spec.json',spec)
 files='\n'.join('|'+x['filename']+'|`'+x['sha256']+'`|' for x in a['files'])
 report=f'''# 방향1: CheXzero 단일 대안 검토 결과

**판정: 후보 선택 근거와 로컬 실행 준비에는 긍정적이다. CheXzero10-checkpoint ensemble을 마지막 대안 하나로 고정해 새 NIH80명에서 한 번 검증할 근거를 확보했다. 아직 NIH 성능은 실행하지 않았고, 평가기 채택이나 private 생성 효용 성공은 아니다.**

2026-09-17, 큰 단계2·방향1. [점수 확인 전 검토/후속 규칙]({PROTOCOL}) · [직전 PadChest 실영상 실패](TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md). 이번에는 문헌과 공식 source를 대조하고, weight를 실제 확보/로딩하고, 새 환자 명부를 예약했다. 새 NIH pixel 접근·이미지/text encoder forward·GPU·생성·학습·DP는 모두0이다.

## 1. 왜 이 후보를 한 번 검증할 만한가

Tiu et al., *Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning*, Nature Biomedical Engineering(2022)의 공식 방법은 MIMIC-CXR image/report로 의료 적응을 하고 CheXpert validation으로 모델을 선택한다. NIH를 의료 적응/선택 데이터로 썼다는 표기는 없다. 다만 CLIP 초기화의 원 웹 corpus 전체에 대한 환자 배제증명은 아니므로 **NIH를 절대로 본 적 없다고 주장하지 않는다.** [공식 원문](https://www.nature.com/articles/s41551-022-00936-9).

이번에는 ‘임의 pathology를 질의할 수 있다’만으로 후보를 고른 것이 아니다. Fig.3의 공개 source XLSX에서 실제 두 질환의 외부 PadChest 결과를 확인했다.

|원문의 PadChest 소견|AUC|95%구간|해당 소견 n|
|---|---:|---:|---:|
|Emphysema|**0.8232**|0.8018–0.8428|376|
|Pneumothorax|**0.7659**|0.7187–0.8071|98|
|Subcutaneous emphysema — 다른 소견|0.9092|0.8694–0.9460|41|

폐기종 근거로 피하폐기종0.9092를 가져오지 않았다. 두 target에 대한 외부 실영상 판별 근거는 있지만, 이 수치는 **우리 NIH 환자군, target 간 직접 대조, 생성 artifact에서의 성능이 아니다.** [Fig.3 공개 원자료](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-022-00936-9/MediaObjects/41551_2022_936_MOESM4_ESM.xlsx). 다운로드한 workbook SHA는 `370aa032b4a6c37c0302b42ceb2111dc260851e8c41da8243b4973829b57dce1`이다.

이 근거는 기존 PadChest-only DenseNet과 다른 의료 적응 데이터/학습 방식의 후보를 선택하는 이유가 된다. 어떤 후보든 잘 될 것이라는 보장은 없고, 두 질환의 실제 NIH 구별력이 다음 질문이다. 다른 classifier는 실행하지 않았다.

## 2. 이름만 정한 것이 아니라 실제 자산을 확보했다

공식 repository commit **`{REV}`**을 고정했다. README가 연결한 Google Drive 공개 release의10개를 전부 받았다. 좋은 이름/filename 점수의 checkpoint 하나를 골라 NIH 결과를 본 것이 아니다.10개 전체의 equal-weight ensemble을 하나의 고정 후보로 사용한다. [공식 README](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/README.md), [공식 release](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno).

- 총 **{a['total_bytes']:,}bytes = {a['total_bytes']/2**30:.3f}GiB**, 다운로드{download['seconds']:.1f}초.
- 각 checkpoint302개 state key/151,277,313 tensor elements. ViT-B/32,224px,context77,embedding512로 모두 일치.
- tensor-only CPU 로딩, 모든 tensor finite, 공식 architecture strict-load, 누락 key0, 저장값과 로딩값의 추가 rounding 차이0.
- 검사{a['seconds']:.3f}초. 가중치를 로딩한 것과 환자의 모델 결과를 계산한 것을 구분한다. **모델 forward0회**다.
- 환경은 기존 PyTorch2.6.0+cu124 등을 유지했다. old requirements 전체 설치/downgrade는 하지 않았다.

|공식 checkpoint|실제 SHA256|
|---|---|
{files}

Source의 MIT LICENSE/NOTICE를 함께 보존했다. 공식 README는 code 라이선스를 설명하고 weights를 공개 배포한다. 이번 연구용 로컬 확보를 넘어 weight의 별도 재배포/상업적 사용 조건까지 확인했다고 쓰지는 않는다. [LICENSE](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/LICENSE), [NOTICE](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/NOTICE.md).

## 3. 점수 계산을 실제 source 기준으로 고정했다

공식 notebook의 template은 `("{{}}", "no {{}}")`다. 이번 query는 다음 네 문자열로 고정했다.

`Emphysema` / `no Emphysema` / `Pneumothorax` / `no Pneumothorax`.

각 모델 m에서 정규화된 image/text embedding의 cosine을 z+와 z−라고 하면:

`p_m = exp(z+) / (exp(z+) + exp(z−)) = sigmoid(z+ − z−)`

최종 score는 **모델10개의 p_m 평균**이다. 공식 `predict`/`run_softmax_eval` 경로에는 학습된 CLIP logit_scale 곱이 없다. Temperature를 따로 탐색하지 않는다. Ensemble 전에 sigmoid를 적용하므로 평균 margin에 sigmoid를 한 번 적용하는 것과 일반적으로 같지 않다. [고정 zero_shot.py](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/zero_shot.py), [공식 notebook](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/notebooks/zero_shot.ipynb).

이는 raw positive cosine만 보는 것보다 질문에 맞는 공식 점수 정의지만, NIH에 교정된 확률이나 질환 간 동일 scale은 아니다. 단일 checkpoint/template/동의어 중 좋은 결과를 고르지 않고,10개의 개별 결과를 final metric 선택에 쓰지 않는다.

## 4. 실제 전처리와 최신 환경 차이를 확인했다

공식 파일 변환과 inference를 연결하면8bit 영상→종횡비 유지320 LANCZOS/padding→gray3채널→0–255 intensity에서 mean101.48761/std83.43944 정규화→tensor bicubic224다. 일반 CLIP RGB mean/std나 앞선 XRV[-1024,1024]를 재사용하면 다른 입력이 된다. [data_process.py](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/data_process.py), [zero_shot.make](https://github.com/rajpurkarlab/CheXzero/blob/{REV}/zero_shot.py).

Pillow에서 제거된 `Image.ANTIALIAS`는 LANCZOS enum으로 치환해야 한다. 또 원 requirements의 torchvision0.11.3 tensor Resize 기본 antialias는 False였고 현재 버전과 다르므로 **False를 명시**해야 한다. [해당 버전 Resize source](https://github.com/pytorch/vision/blob/v0.11.3/torchvision/transforms/transforms.py). 아직 실제 환자 pixel의 변환/CPU–GPU 추론 대조는 실행하지 않았고 다음 packet의 검사 항목이다.

공식 아키텍처를 checkpoint에서 완전히 구성할 수 있으므로 원 CLIP weight를 다시 받아 임시 초기화할 필요는 없다. 그 대신 strict full-state loading을 확인했고, missing key를 무시하는 경로는 금지했다. 실제 후속 평가에서는 공식CPU경로와 GPU FP32 경로의 고정입력 feature/score 대조 및 전처리 검산을 결속한다.

## 5. 새 환자80명 예약과 reference 규모 정정

이미 PadChest 평가에 소비한80명을 새 CheXzero 통과 판정에 재사용하지 않는다. 남은 development4,133명에서 별도 salt/hash로 네 군20명씩 총80명을 **metadata만으로 예약**했다. 점수를 보고 명부를 고른 것이 아니다. 같은 환자의 보유 PA label 합집합, target/control 상호배타성, 해당 label 영상1장의 hash선택은 직전 정의와 같다.

|군|새 평가기 예약 전|이번 예약|이후 method-development 후보|
|---|---:|---:|---:|
|Emphysema+/Pneumothorax−|64|20|**44**|
|Pneumothorax+/Emphysema−|197|20|177|
|두 target 없는 Effusion|774|20|754|
|세 질환 없는 환자의 No Finding 영상|2,152|20|2,132|
|Dual / 기타|51 /895|0|51 /895|
|전체|4,133|80|4,053|

신규80명과 기존 평가309명 교집합0, reserved confirmation4,213명과 교집합0, 기존 기록상 훈련/모델결과 소비 환자와 교집합0이다. 원래 졸논 역할·test/calibration·official census를 변경하지 않았다. 아직80명의 이미지 content나 모델 결과를 열지 않았으므로 **새 소비 환자라기보다 다음 평가용 예약 환자**다. 실제 실행 후 소비 상태를 갱신해야 한다.

이 배정에서는 **기존 reference64명/군 권고를 유지할 수 없다.** Development의 E-only가44명으로 줄기 때문이다. 다음 소규모 생성 개발평가의 계획값을32명/군으로 정정했다. Reserved confirmation의 실제 reference 명부는 아직 고르지 않았고 개발용으로 쓰지 않는다.32명은 자원상 가능한 탐색 규모일 뿐 검정력/논문 최종 표본 충분성을 보장하지 않는다. 생성분포 지표는 표본 불확실성을 보고해야 하고, 더 큰 집합이 필요한 주장을 하려면 외부 자료 등 별도 자원이 필요하다.

원본 파일의 존재/bytes와 inventory의 저장 SHA/공식 metadata를 대조했으나 이번에 새80장의 pixel이나 실제 SHA를 다시 읽지는 않았다. 실제 validation 시작 전 선택80장 SHA/decode를 확인하는 항목이 남아 있다.

## 6. 다음 한 번의 검증과 중단선

다음 **20–35분** 패키지는 새80명에 실제 CheXzero 추론을 하고 저장 scores/metric을 검산하는 일이다. 공식10개 모델은 순차 로딩해 GPU 메모리를 제한한다. 주 image-encoder 계산은80×10=800image-examples, replay/text는 별도이며 실제 시간은 측정해서 갱신한다. 이미 weights를 받았으므로3.29GiB 다운로드를 다시 포함하지 않는다. 현재는 runner/독립계산 코드와 최종 실행 contract가 아직 없다.

각 질환의 target-vs-rest AUC≥0.70와95%하한>0.50, target-vs-opposite AUC≥0.60와95%하한>0.50을 그대로 요구한다. Normal/effusion 대조, AP와prevalence, 군별 patient bootstrap2,000회도 유지한다. 양 target 모두 통과해야 하며 결과를 보고 기준/환자/prompt/checkpoint를 바꾸지 않는다.

- **둘 다 통과:** real-image evaluator 관문만 통과한 것이다. 생성 artifact에 대한 취약성 확인을 포함한 고정 비DP 생성 비교가 다음이다.
- **하나 또는 둘 다 실패:** 현재 폐기종·기흉 공동 가설의 classifier 탐색을 종료한다. 세 번째/네 번째 모델이나 synonym/ensemble subset으로 구제하지 않는다.
- **Real 통과, artifact 취약:** 해당 score 단독으로 질환 생성 개선을 판정하지 않는다. 임상 판독/annotation 또는 독립적인 추가 측정 근거가 필요하다.

어느 경우에도 이번 검토가 private-only/pooled 생성 효용이나 patient-DP 실행을 승인하는 결과는 아니다. 원래192장 실험의 private 효용 실패와 PadChest 평가기 실패를 모두 보존한다.

## 7. 검증과 기록 경계

독립 준비 검산 **{v['checks']:,}항목 PASS**, {v['seconds']:.3f}초. 공식 공개10개 전체 membership, checkpoint 실제bytes/hash, 저장된 strict-load 결과, 새 cohort의 독립 재구성, 이전 source/192장/PadChest outputs 불변, 원문의 E/P 및 피하폐기종 source행을 대조했다. 성능 표본 수나 별도 모델 forward 재현 횟수가 아니다.

Source/asset 기록은 `spec_sources/chexzero_review_20260917_v1`, 새 비공개 명부/실행 계획은 `code_working/_reports/chexzero_evaluator_plan_20260917_v1`에 있다. 고정 [execution spec](../../code_working/_reports/chexzero_evaluator_plan_20260917_v1/execution_spec.json)은 candidate/cohort 계약이며 실제 runner contract와 구분한다.

일부 Nature URL을 일반 HTTP로 저장한 `paper.html/table4.html/table5.html`은3038byte의 브라우저 확인 페이지다. 이 파일들을 논문 본문을 확보한 증거로 세지 않았다. 원문 Methods/Table은 공식 웹 reader에서 읽었고, 위 실제 수치는 정상 다운로드한 Fig.3 XLSX에서 확인했다. 원격 main 확인/push는 이번 작업 범위가 아니다.

**현재 결정: CheXzero 한 번을 검증할 근거와 자산은 확보됐다. 실제 NIH 성능 판단은 아직 시작 전이다.**
'''
 (R/REPORT).write_text(report,encoding='utf-8')
 d=dict(status='POSITIVE_SINGLE_ALTERNATIVE_READINESS_NOT_EFFICACY',completed_utc=datetime.now(timezone.utc).isoformat(),report=REPORT,protocol=PROTOCOL,current_step=2,track=1,
  candidate='CheXzero_official_ten_checkpoint_ensemble',checkpoints=10,checkpoint_bytes=a['total_bytes'],weights_strict_load_passed=True,evaluator_validated=False,
  new_NIH_pixel_access=0,new_model_inference=0,new_GPU=0,new_generation=0,new_DP=0,fresh_reserved_evaluator_patients=80,
  reserved_manifest=str(D/'selected_images_private.csv'),selected_model_results_consumed=False,remaining_method_development=4053,remaining_exclusive_emphysema=44,
  later_reference_plan_per_condition=32,reference_power_established=False,reserved_confirmation_used=False,original_roles_changed=False,
  previous_padchest_gate_passed=False,private_generation_utility_established=False,DP_allowed_to_expand=False,
  final_allowed_classifier_alternative=True,source_revision=REV,independent_preparation_checks=v['checks'],
  source_sha256={str(p):sha(p) for p in [D/'execution_spec.json',D/'cohort_contract.json',P/'readiness_verification.json',P/'asset_inspection.json',P/'checkpoint_acquisition.json',P/'official_release.json',P/'figure3_source.xlsx',P/'additional_sources.json']})
 save(D/'research_decision.json',d)
 for name in ['research_state.json','patient_baseline_spec.json']:
  s=js(R/name);s['pre_chexzero_review_result_report_20260917']=s['current_result_report'];s['current_result_report']=REPORT;s['current_planning_report']=REPORT;s['next_task_plan_report']=REPORT
  s['last_actual_model_result_report']='TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md'
  s['next_task']='One bounded20-35min fixed CheXzero real-NIH validation on the fresh reserved80patients: official ten-checkpoint equal-weight ensemble, literal positive/negative templates, official preprocessing with explicit legacy antialias semantics, independent source/output/metric checks. Bind actual runner and verifier before inference. No checkpoint/template selection, reserved-confirmation consumption, third classifier, generated images or DP.'
  s['next_task_output']='Actual new80-patient condition-specific AUC/AP/bootstrap and two-target joint gate; if either target fails, stop classifier shopping for this joint targeted path. If both pass, only propose separately verified generated-artifact/non-DP comparison, not private utility success.'
  s['step2_open_items']=['CheXzero candidate justified by official external data and code;10 downloaded checkpoints strictly loaded, but new NIH inference and actual runner remain unexecuted.', 'New80 evaluator-only patients are reserved from remaining development; do not use them as generation reference. Prior PadChest80 stay consumed/excluded.4053 method-development candidates remain, including44 E-only; earlier64/reference proposal is infeasible after this reservation, provisional32/condition has no power guarantee.', 'NIH absent from declared medical training/selection is not proof of no overlap with original CLIP web pretraining.', 'Both fixed targets must pass real discrimination; generated artifact sensitivity remains a separate validity issue.', 'CheXzero is the last classifier alternative for the current joint hypothesis; failure requires clinical/annotation/external-measurement resources or suspension, not a third classifier.', 'Old192-image private utility failure and PadChest failure remain; targeted private generation utility/DP retention/cost advantage unestablished.']
  s['active_execution']['status']='no_running_execution';s['active_execution']['current_step']=2;s['active_execution']['chexzero_review']=d
  s['completed_chexzero_review_20260917']=d;s['latest_user_execution_scope_20260917']='Review and concretely pin one defensible alternative evaluator, its assets, independent patient reservation and stopping rule; no new medical model inference in this review.'
  (R/name).write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 prepend(ROOT/'AGENTS.md','- Latest CheXzero REVIEW (2026-09-17): read `'+REPORT+'`. POSITIVE candidate readiness, NOT NIH performance. Official revision '+REV+', all10 released checkpoints downloaded(3,535,487,090bytes), strict302-key loads/value preservation PASS, no forward/GPU/NIH pixel access. Official external PadChest source: emphysemaAUC.8232(n376), pneumothorax.7659(n98); NOT NIH or target-vs-target evidence. Use fixed literal +/- templates, sigmoid(cos+ - cos-) per checkpoint then mean10, no CLIP logit_scale. Source-normalization101.48761/83.43944,320LANCZOS/pad then224tensorbicubic/antialias=False. Fresh80 evaluator patients reserved by hash from4133 remaining dev, prior evaluator309/reserved4213 disjoint. Not yet consumed; exclude reserved80 from subsequent method reference. Remaining4053 includes44E-only, so earlier64-reference suggestion no longer fits; tentative32/condition is exploratory, not power assurance. Actual runner/NIHvalidation pending20-35min; bind code and CPU/GPU score parity first. Both targets same gates; if either fails STOP further classifier search on this joint hypothesis. No third model, template rescue, confirmation or generation/DP. Original PadChest failure/192 utility failure preserved. Stage2/track1,no running execution; older latest/next text is history.')
 short='**2026-09-17 最新单一대안 검토 — 후보 준비에는 긍정적, 실제 성능은 미실행:** [CheXzero 공식 근거·10개weight·새80명예약·중단선]('+REPORT+'). 외부 PadChest 원자료의 폐기종AUC0.8232/기흉0.7659를 확인하고 공개ensemble10개3.29GiB를 실제 확보·strict-load했다. 새NIH pixel/모델추론/GPU0이다. 새80명을 예약하면 development E-only64→44명으로 줄어 이후reference64명/군 권고는 유지할 수 없고32명/군을 탐색 계획값으로 둔다. Reserved confirmation은 보존했다. 다음은20–35분 고정 NIH 실영상 검증 한 번이며 어느target이라도 실패하면classifier탐색을종료한다. 기존실패와private효용미확인유지. 아래 최신/다음은 이전 이력이다.'
 short=short.replace('单一','단일 ')
 prepend(R/'RESEARCH_FRAMEWORK.md',short);prepend(R/'REALISTIC_RESEARCH_PLAN_20260916.md',short)
 prepend(ROOT/'CURRENT_STATUS.md',short.replace('('+REPORT+')','(CVPR%20주제%20탐색/research_2026-09-10/track1_chexzero_review.html)'))
 prepend(ROOT/'WORKLOG.md','''## 135-CHEXZERO-SINGLE-ALTERNATIVE-REVIEW (2026-09-17 KST)

- 큰 단계2·방향1. 예상20–30분으로 공식논문/source/배포파일과 별도 검증환자 구성을 검토했다. 판정은후보준비에긍정적이며NIH성능은미실행이다.
- MIMIC-CXR 의료적응/CheXpert모델선택/CLIP초기화를구분했다. 선언된의료훈련에NIH표기는없지만원CLIP웹corpus까지배제한증명은아니다. Fig3XLSX의Emphysema0.8232(n376),Pneumothorax0.7659(n98)를직접확인하고피하폐기종과구분했다.
- Officialcommit5c341db...의source와MIT/NOTICE를보존하고release10개전부를고정했다. 실제3,535,487,090bytes다운로드260.913초,CPU strict-load검사20.762초. 각302keys누락0/추가rounding0/모델forward0, GPU0, 새NIHpixel0.
- 공식score는positive-negativecosine softmax후checkpoint10개평균이다. CLIP학습logitscale미사용/동의어탐색금지. 전처리mean101.48761/std83.43944와320→224두단계resize, legacyantialiasFalse를확인했다.
- 이전소비80명제외후남은development4133명에서새salt/hash로네군20명씩예약했다. 새80명은아직추론에소비되지않았으며기존309명/reserved4213명과교집합0이다. 남은4053명중E-only44명/P-only177명. 이전reference64명/군권고는유지불가라32명/군탐색계획으로정정하되power보장하지않는다.
- 독립준비검산8197항목PASS(3.816초). 이는무결성확인수로성능표본수가아니다. 원래졸논역할/lockedsets/192생성물/PadChest실패보존.
- 다음20–35분은고정CheXzero실영상검증한번. Runner/verifier/CPU–GPU/전처리결속후실행한다. 두target중하나라도미통과면현재공동가설에서세번째classifier/prompt구제탐색없이중단한다. 생성·DP는아직안한다.
- TRACK1_CHEXZERO_REVIEW_20260917.md/HTML과sourceassetpacket,새비공개예약명부,execution_spec,상태문서에기록한다. 일부Nature로컬HTML은브라우저확인페이지여서본문증거로세지않았고웹원문및실제XLSX로검토했다. 원격main확인/push는하지않았다.
''')
 print(json.dumps({'status':d['status'],'report':REPORT,'execution_spec_sha256':sha(D/'execution_spec.json'),'decision_sha256':sha(D/'research_decision.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
