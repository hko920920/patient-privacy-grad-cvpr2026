# 사적 자료 추가 신호: 저장 자료·통계의 제한된 검토

시작 2026-09-16 23:49 KST. 예상 20–30분, 큰단계 2·방향 1. 새 GPU, 생성, DP, backbone 학습은 하지 않는다. 직전 192장 결과는 고정하고 별도 CPU 분석 폴더를 사용한다. 보고서 날짜는 자정을 넘길 가능성을 반영한 20260917 식별자이며, 실행 시각은 UTC로 따로 기록한다.

## 질문과 해석

현재 private-role80은 public32에 없는 유용한 target 정보를 제공하는가? 자료의 차이, head 방향의 차이, 실제 생성 효용은 별개다. 현재 NIH 역할 분할과 기존 개발40/reference40만으로 실제 병원 target-domain 적응의 일반화를 입증할 수 없다. 안정된 Delta나 metadata 차이만으로 이전 생성 효용 관문을 통과 처리하지 않는다.

## 고정 분석

1. 기존 images/patients/manifest/statistics/weights, NIH 원본 metadata, reference 명부를 hash로 결속한다. public32/private80/development40/reference40의 환자·영상 수, image-level weak label과 환자별 영상비율 평균, patient-any label, 선택영상의 환자별 median age·sex·PA/AP·영상 수를 기술한다. any-label 비율은 2장/4장 관측 기회의 영향을 받는다. 영상의 L256 Lanczos 평균 밝기·표준편차·p99−p1, 실제 원본 크기·metadata 원 해상도를 acquisition proxy로 기술하고 기관 차이로 단정하지 않는다. 나이는 원값을 고치지 않으며 >100 값의 수를 별도 기록한다. p-value나 사후 유의성 gate는 없다.
2. 동일 lambda .001에서 저장 A/B/Q로 public/private-only/pooled ridge를 계산한다. 원 public/pooled와 독립 Cholesky 재계산을 대조한다. W cosine·delta 상대 norm·최적 scalar와 잔여 norm을 보고한다. 같은 항목을 고정 development A의 prediction inner product에서도 계산한다. 가중치 좌표의 안정성을 생성 효과로 해석하지 않는다.
3. 모든 저장 raw에서 8개 timestep stratum, 출력 latent 4channel, 세 weak condition(No Finding/Effusion/Cardiomegaly)과 나머지의 prediction correction·delta energy와 개발 MSE를 계산한다. 조건들은 multi-label이면 겹친다. 채널은 RGB나 질환 축이 아닌 VAE latent channel이다. Generation trace의 기존 public/pooled trajectory에서 출력 변화도 보조로 기술하며 새 생성은 하지 않는다.
4. private80의 환자를 hash salt `private-signal-halves-v1|partition|patient_id`로 정렬해 8번의 disjoint40/40 분할(총16개 CPU head)을 고정한다. 각 partition 안에서는 환자 무중복, partition 간에는 동일80명이 반복된다. Delta=W_half−W_public의 Euclidean 및 개발 prediction cosine, 개발 MSE를 보고한다. 공통 W_public을 빼기 때문에 안정된 Delta가 public estimator의 공통 오차를 반영할 수 있다. 안정성 자체가 private-specific 의미 정보라는 증거가 아니다.
5. 관측량 대조는 각 private 환자의 두 장을 `private-signal-matched-images-v1|image_id` hash순으로 고정해 stats를 재구성하고, private80×2 head와 기존80×4 head를 비교한다. public32와 환자 수도 다른 비교임을 유지한다. 이번에는 public-budget grid나 유리한 subset 선택을 하지 않는다.
6. pooled loss의 질량은 public32/112와 private80/112다. 같은 loss와 동일 total-W ridge에서 W=W_public+Delta로 바꾸면 정확히 같은 해다. private-only loss+total-W ridge이면 private-only 해와 같다. Delta에만 ridge를 적용하면 다른 prior/목적함수이므로 식을 구분한다. 새 residual solver나 생성 후보를 자동 채택하지 않는다.

독립 검산은 저장 통계의 직접/Cholesky 풀이, 실제 raw prediction loss, scalar projection 직교성, reparameterization identity, subset 무중복·hash순서와 원본 불변을 확인한다. 이번 CPU 분석에서 유리한 threshold·환자·질환·배율을 선택하지 않는다. 명확한 차이가 없다는 관측도 표본·표현·평가 한계와 함께 쓴다. 반대로 metadata 차이 또는 안정된 Delta 하나만으로 후속 DP 실행을 정당화하지 않는다.

