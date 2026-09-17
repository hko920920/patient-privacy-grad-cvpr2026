# 비DP downstream 본 개발 실행 계약 — 2026-09-17

판정 전 실행 명세다. 이전 corrected all-arm replay는 좋은 연결 결과이며, 사적 효용은 아직 미확인이다. Master protocol의 512장 생성, 공개 R1 calibration, 21개 개발 학습을 실제로 수행한다. Stage2·방향1을 유지한다.

## 동결 범위

- 별도 `run_development_v1.py`를 사용한다. Legacy data/train import를 차단한다. 검증된 `data_v2.py`, `train_v2.py`는 변경 없이 SHA로 결속한다.
- 실자료는 public813/private320/selection5097/method-development5047장뿐이다. 기존 cache와 원본 파일 SHA·독립 decode를 다시 확인한다. Locked final 및 reserved 환자를 manifest 단계에서 거부한다. Expert final·reserved pixel/prediction과 DP 실행은 없다.
- E4 SHA, FP32 CFG7.5 DDIM30 eta0 256px, conditional-only head를 고정한다. 기존 검증된 conditioning 및 public/private-only/pooled weights를 쓴다. Prompt, seed, CFG, head scale, image filtering을 결과 후 변경하지 않는다.
- 새 hash seed64개를 결과 전에 저장한다. 각 latent는 기흉 요청1장과 사전 할당 음성1장에 사용한다. 음성은 index순 normal22/effusion21/cardiomegaly21이다. 네 방법 모두 같은128cell을 생성하여 총512장이다. Profile28장은 제외한다.

## 학습과 선택

공개 R1 calibration seed는11이다. LR1e-4와3e-4의 독립800-step trajectory에서400/800 state를 저장한다. AdamW state를 중간에 재초기화하지 않는다. Classifier-selection에서 최고 AUROC와 .002 이내인 후보 중 적은 step, 작은 LR을 우선한다. 총1600update이며21run과 별도다.

선택된 LR/step을 모두 고정한 뒤 R0/R1/S0/S1/S2/S3/Dreal × seed11/23/37을 실행한다. Batch32·16positive/16negative, 환자균등→영상균등 sampling, ImageNet ResNet18, AdamW weight-decay1e-4, 기존 augmentation을 유지한다. 모든21run 완료 전 method-development metric을 보지 않는다.

R0/R1은 public real32장, S0–S3은 public real16+해당synthetic16장, Dreal은 public real16+private real16장이다. 이는 같은 계산량에서 batch 절반을 대체하는 실험이다. 합성 라벨은 요청 조건이며 전문가 정답이 아니다. Dreal은 비DP 진단 기준이지 상한 또는 보호 방법이 아니다.

학습 자체는 unchanged `train_v2.train_steps`에서 수행한다. 별도 관찰 hook은 다음 forward 직전에 이미 완료된400step state를 저장하고 진행시간을 기록한다. RNG·입력·optimizer·값은 바꾸지 않는다. 본 실행 전 동일2step plain/observed의 입력·logit·loss·최종state exact parity와 관찰된1step checkpoint의 이전 검증 결과 일치를 요구한다. 이4update는 본 학습과 별도 instrumentation 검사다.

모든 run의 초기 state, source/array/image/patient/cell/label/hash, optimizer step, finite loss/gradient, 최종state, per-image/per-patient exposure, VRAM·시간을 저장한다. 400/800 calibration 노출량을 따로 기록한다. 첫10/20step 실측으로 ETA를 갱신한다.

## 평가와 중단

Method-development2026명·5047장의 영상별 weak label로 seed별 AUROC/AP를 계산한 후 평균한다. S2 또는 S3가 S1과R1 각각에 대해 평균AUROC≥+.01, positive delta≥2/3seed, 평균AP 비감소를 모두 만족해야 DP 개발 투자 후보가 된다. 임상 효과·유의성·final 성공 기준이 아니다.

보조 불확실성은 사전 hash seed로 환자2026명을 복원추출하는2000회 paired cluster bootstrap이다. 같은 draw를 모든 arm/seed에 사용하고 seed별 metric을 계산한 뒤 평균한다. S2−S1/S2−R1/S3−S1/S3−R1/Dreal−R1의95% percentile 구간을 전부 보고한다. 단일 생성bank/세 classifier에 조건부인 탐색 구간이며 최종 검정이 아니다.

## 불변·재개

완료 marker에 파일 SHA를 기록하고 일치하는 cell/run만 재개 시 건너뛴다. 미완료 폴더는 자동 덮어쓰지 않고 기술적 중단을 명시적으로 기록한 뒤 처리한다. 결과를 보고 run·checkpoint·seed를 선택하거나 바꾸지 않는다. 이전98회 오류 학습은 계속 무효이며 과거192장·자동평가기 실패도 보존한다.

본 패키지가 성공해도 expert final을 열지 않는다. 이후 DP 구현과 강한 기준선을 개발자료에서 완료하고 최종 전체 경로를 동결한 뒤 별도의 final 절차를 따른다.
