# 방향1: 고정 CheXzero NIH 실영상 검증

2026-09-17, 큰 단계2·방향1. 예상20–35분. 직전 후보 검토에서 예약한80명과 공식10개 checkpoint를 실제 실행한다. 기존 검토 protocol의 ‘추론0’은 지난 준비 단계의 이력이며, 이 문서는 그 후속 실행 명세다. [후보 검토](TRACK1_CHEXZERO_REVIEW_20260917.md)의 환자·score·기준을 변경하지 않는다.

## 입력과 계산

- `chexzero_evaluator_plan_20260917_v1/selected_images_private.csv`를 byte 그대로 복사한다. E+/P−, P+/E−, target-free effusion, target/effusion-free no-finding, 각각20명·1장. 다른 동반질환은 허용하며 원 weak-label 정의를 유지한다.
- 고정 official release10개 전부, commit5c341db0fe0db2f663a2c136c7120a8b303f1d11. `Emphysema`, `no Emphysema`, `Pneumothorax`, `no Pneumothorax` 외의 prompt는 쓰지 않는다.
- 8bit grayscale를320 LANCZOS·검은 padding,3채널,0–255 기준 mean101.48761/std83.43944,224 tensor bicubic/antialias=False로 변환한다. 공식 HDF5 dataset/transform 경로도 별도로 검사한다.
- GPU FP32, batch8, eval, TF32 off. 이미지와 text feature를 정규화하고 공식 text의 두 번째 normalization도 유지한다. Positive/negative cosine에 exp-pair softmax 후10모델 산술평균. 학습된 CLIP logit_scale과 temperature 보정은 사용하지 않는다.
- 모든80장의 실제 SHA/decode/label·patient 결속, inventory 전체와 cross-patient exact duplicate를 검사한다. 한 장이라도 실패하면 환자 교체 없이 중단한다. Perceptual near-duplicate 배제나 NIH weak label의 임상적 정답성을 증명하는 검사는 아니다.

## 공식 CPU와 GPU 대응 검사

각 군의 명부상 첫 환자1명, 총4명을 실행 전에 replay 대상으로 고정한다. 모델마다 공식 `zero_shot.run_softmax_eval`을 CPU FP32에서 직접 호출한다. 공식 메서드의 image/text encoder 출력을 capture하고 GPU 경로와 비교한다. 공식 source를 수정하지 않는다. tqdm 표시만 비활성화하며 Pillow 옛 enum은 동일 LANCZOS로 해석한다.

원칙적으로 CPU 공식 경로는 positive/negative에 이미지를 각각 계산하므로 checkpoint당8 image-examples/4 text-examples이다. GPU는 primary80와 replay4, text4를 계산한다. 전체 image예제는 GPU840/CPU80, text예제는 각40이다. 저장된 계산 검산은 독립 모델 재훈련이나 별도 의료 평가가 아니다.

허용오차는 결과 전에 고정한다: raw embedding atol1e-3/rtol1e-4, normalized image/text embedding maxabs2e-5, cosine maxabs5e-6, per-model 및 ensemble pair-score maxabs3e-6. GPU batch8 대 replay4에도 동일 score 허용오차를 적용한다. 원 checkpoint의 full-key strict-load와 모든 tensor값 보존을 요구한다. 검산 실패 시 성능을 판정하지 않는다. 점수가 좁은 범위에 몰릴 수 있어 score를 질환 확률로 해석하지 않는다.

## 저장과 통계

320 grayscale 배열·224 input tensor와 개별 tensor SHA,80명 manifest,10모델의 normalized image/text feature·네 cosine·두 pair-score·평균 score,4명 CPU/GPU raw/normalized replay를 저장한다. 개별 checkpoint로 primary 모델을 선택하거나 ensemble 일부를 제거하지 않는다.

각 target의 rest/opposite/normal/effusion 대조에서 AUC/AP/prevalence/score 분포를 계산한다. 환자당1장, 각 질환군 내2000회 bootstrap, seed20260917,95% percentile CI. Rest의 세 음성군도 각각20명으로 층화 재표집한다. Producer는 rank-sum AUC/sklearn AP, 별도 verifier는 pair-count AUC/threshold-block AP로 모든 bootstrap을 재계산한다.

두 target 각각 rest AUC≥.70 및 CI하한>.50, opposite AUC≥.60 및 CI하한>.50이어야 joint PASS다. 표본20명/군에 대한 개발용 운영 기준이며 임상 인증/검정력 보장이 아니다. NIH80점수·bootstrap·source binding 검사를 통과해야 결과를 보고한다.

## 결과별 결정

- Joint PASS: 고정 실영상 표본에서 조건 구별력을 확보했다. 생성 artifact 검사와 고정 backbone/public/private-only/pooled 비DP 비교의 명세가 다음이다. 자동 생성·DP 실행은 하지 않는다.
- 하나라도 FAIL: 이 폐기종·기흉 공동 가설의 classifier 탐색을 종료한다. 세 번째 classifier·prompt변경·checkpoint 재선택·성공 질환만 남기기로 구제하지 않는다. 전문 판독·주석·독립 측정 자원 없이 현재 targeted 경로는 보류한다.
- 구현/대응검산 FAIL: 실패 packet을 보존하고 구현 원인을 조사한다. 의료 판별 실패로 해석하거나 tolerance를 결과에 맞춰 넓히지 않는다.

실제 추론을 마친80명은 새 evaluator-consumed overlay로 기록하고 후속 reference에서 제외한다. 과거 예약 manifest·사용감사 ledger는 변경하지 않는다. Remaining development4053명/E-only44명, tentative reference32명/군. Reserved confirmation4213명·졸논 final/calibration/test·이전 PadChest 실패·기존192장과 private-utility 실패는 보존한다.
