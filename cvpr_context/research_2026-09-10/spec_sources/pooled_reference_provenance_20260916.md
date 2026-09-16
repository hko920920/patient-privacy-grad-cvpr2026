# Pooled 비DP 참고점: 캐시 출처와 목적 감사
2026-09-16. 이번 범위는 공개32+사적 train80의 동일 환자 가중 pooled ridge를 **실행하기 전** 자료·목적 일치 확인이다. Static16/full64, lambda=.001, 기존 development40을 그대로 사용한다. 새 모델 solve·MSE 계산·원시 특징 재계산·GPU·생성·DP 실험은 수행하지 않았다. 기존 파일은 수정하지 않았다.

## 1. 판정
**지정된 두 캐시로 pooled 참고점을 계산하는 데 필요한 출처·특징·목적의 불일치를 발견하지 못했다.** 두 계약의 핵심23개 필드가 모두 같고, projection.npz는 파일 SHA까지 완전히 같다. 환자 분리도 확인했다. 기존 독립 raw/stat 검산의 hash에 현재 캐시가 일치하므로, 이번에는 원시 UNet 계산을 반복하지 않고 검증된 충분통계를 사용할 근거가 있다.

이 판단은 새 pooled 결과의 수치 검산이나 성능 판정이 아니다. Root의 새 producer와 별도 독립 verifier가 새 평균·solve·평가 산술을 확인해야 한다.

## 2. 실제 확인한 자료와 hash
경로의 공통 앞부분은 `code_working/_reports/`이다.

| 파일 | Capacity/private train80+eval40 | Public32 |
|---|---|---|
| 폴더 | frozen_residual_capacity_20260916_v1 | frozen_residual_public32_20260916_v1 |
| contract.json | ff398ca316c80b95cf6d0b0394bba99ff8d76e0e3dc7d9c08a33c7b71b712b95 | 785379be961ed4d1417a74f7764816cb4fe75468d57c982f6328bfde75f66ea2 |
| patient_stats.npz | 2a69e909ee86f6b6166566ff38d510648bdede8e001e6a78d6f5f2edc3e6376f | 396575b651bdca3b33aef1f94562154c5b8e3acfb15eb1dfc0bb294ace32e974 |
| projection.npz | 5096130345df5ffb376f099d985ddd9eb7aeafb31c1abe5d9f22a39a1da8801e | 동일 |
| independent_verification.json | b31daa1bf10750bfc5bfb8d23cd211b113a58992c3e52efcaf5c70bde2f6bb8b | c058ae8847286e5de6a25b6d8a93eb23896a507e6524a0978647a70b6cfd99bf |

위 contract/stats/projection과 각 images.json/manifest.json, 합계10개 파일의 **현재 실제 SHA**를 다시 계산하여 각 기존 독립 검산의 input_sha256와 모두 일치함을 확인했다. 독립 검산 상태는 capacity `PASS_FROZEN_RESIDUAL_CAPACITY_SAVED_ARITHMETIC_AND_PROVENANCE`(114,873항목), public `PASS_PUBLIC32_SAVED_FEATURE_STATISTICS_AND_DISJOINT_PROVENANCE`(15,202항목)이다.

두 계약의 공통 source13개는 기록된 경로·hash가 모두 같다. 공통 residual_math.py hash는 `59db490cb427c99df4d6f2eed96618bbffd769821cf8bd8e6b5eba7c83b0d412`이다. 이번에 큰 base weight/cache를 다시 읽거나 원시 영상 파일 전체를 재hash하지는 않았다. 이 부분은 기존 검산의 source binding을 이어 사용한다.

## 3. 특징과 모델이 같은가
서로 일치하는23개 계약 필드를 비교했다. 중요한 내용은 다음과 같다.

- Manojb/stable-diffusion-2-1-base, revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`; 같은 fp16 artifact를 FP32로 올린 fresh base, LoRA 없음.
- Scheduler 출력은 epsilon, latent [1,4,32,32], batch1; TF32 비활성.
- conv_out 직전, conv_norm_out/activation 뒤의 동일 feature tap.
- seed26091601의 Gaussian-QR 투영15개+상수1, static16; 동일 log-SNR Legendre4항과 결합한 full64. log-SNR 범위도 exact 일치.
- FP32 모델·입력·projection, FP64 residual/statistics/solve; 각 이미지8개 timestep strata, CPU FP32 noise, 동일 image/draw hash seed 정책.
- 같은 캐시의 이미지별 weak-label prompt 정책. 사진마다 prompt가 같다는 뜻은 아니며, prompt를 만드는 규칙·텍스트 특징 cache가 같다.
- Residual은 `epsilon−base_prediction`이고, 공간 평균 및 출력 channel 합을 사용한다.

코드 대조: `run_capacity.py:288–333`과 `run_public.py:141–170`은 같은 residual 정의, static sufficient statistics와 같은 Kronecker full64 결합을 사용한다. Public은2장, capacity는4장으로 이미지별 통계를 환자 내 평균하는 부분만 의도적으로 다르다. 원 계약에 남은 time4 정보와 무관하게 이번 소비 대상은 실제 두 파일에 모두 있는 static/full뿐이다.

## 4. 환자 분리와 최종 자료 비사용
patient_stats의 ID/split과 기존 `cvpr_u_pilot_v1_001/cohort/evaluation_images.csv`의 역할을 집합으로 대조했다. 새 예측·영상 열람은 하지 않았다.

| 확인 | 실제 결과 |
|---|---|
| Capacity train | 80명, 기존 fit80과 집합 exact 일치 |
| Capacity eval | 40명, 기존 selection40과 집합 exact 일치 |
| Public | 32명, split=public |
| Train∩eval / public∩train / public∩eval | 모두0 |
| 선택된 train/eval/public∩원 calibration140 | 0 |
| 선택된 train/eval/public∩원 test140 | 0 |
| Capacity images manifest | 480행, 120명 모두4장 |
| Public images manifest | 64행, 32명 모두2장 |

`run_capacity.py:67–77`은 fit/selection만 선택하고 calibration/test를 제외한다. `run_public.py:28–32`는 quality32를 고른 뒤 기존 evaluation CSV 전체400명과의 중복을 금지한다. 새 producer에서도 단순 “앞80행” 가정 대신 split=train을 명시적으로 사용하고 정확 ID 집합을 결속해야 한다.

최종 calibration/test의 통계나 영상은 이번 pooled 참고점에 들어가지 않는다. 역할 검증을 위한 원 cohort **메타데이터** 읽기는 이들 영상/평가 결과를 사용하는 것과 구분한다. Development40은 이미 본 자료이며 독립 최종 test가 아니다. 공개32도 공개 calibration에 사용된 자료이므로 untouched quality test로 다시 이름 붙이지 않는다.

## 5. 환자 가중치와 ridge의 정확한 목적
각 환자 A_u/B_u/Q_u는 공간→draw8→이미지 순서로 평균된 통계다. Public 환자는2장, private/eval 환자는4장이지만 **환자 한 명의 최종 가중치는 동일**하다.

지정한 pooled 목적:
```
A_pool = (Σ_public32 A_u + Σ_train80 A_u) / 112
B_pool = (Σ_public32 B_u + Σ_train80 B_u) / 112
W_pool = (A_pool + .001 I)^(-1) B_pool
```

Public/private 영역 평균을 쓸 때는 `(32*A_public + 80*A_private)/112`이며, 영역 비중은2/7 대5/7이다. 영역 평균을 단순 반반 섞으면 다른 목적이다. Public64장+private320장=384장을 한꺼번에 평균하면 public 비중이1/6이 되어 역시 다른 목적이다. 각 이미지에 동일한 무게를 주는 것과 각 환자에 동일한 무게를 주는 것을 혼동하지 않는다. 사진 수 차이가 만드는 추정 분산은 남지만 이 차이를 환자 가중치 변경으로 자동 보정하지 않는다.

학습 목적은 **channel-SUM** quadratic `Q−2<W,B>+<W,AW>+.001||W||²`다. 평가 MSE는 **ridge 항 없이** `(Q−2<W,B>+<W,AW>)/4`다. 평가에는 개발 환자의 A/B/Q를 사용하고 훈련의 ridge penalty를 더하지 않는다. Training loss만/4로 바꾸면서 lambda=.001을 그대로 더하면 effective ridge가4배가 되므로 같은 문제가 아니다. MSE 형식으로 학습 목적 전체를 쓰려면 ridge coefficient도 .001/4로 맞춰야 한다. 근거: `residual_math.py:145–161,221–250`.

Pooled 해는 이 finite regularized objective의 최적해이며 생성품질·held-out MSE·사적 추가 효용의 상한이라는 뜻은 아니다. 이번 참고점은 non-DP이며 clipping/noise/epsilon 비교를 수행하지 않는다.

## 6. 새 producer/verifier에 필요한 최소 요구사항
1. 위 두 계약·stats·독립 verification·동일 projection hash를 새 protocol에 결속하고, 실행 전후 source/input 변경을 확인한다.
2. Train80/public32/eval40 ID 집합·중복0을 저장한다. Fit에는112명만, 평가에는40명만 사용한다. 대상 head는 static16/full64, lambda=.001 고정이다.
3. 동일 환자 가중 pooled A/B 및 독립 solve의 stationarity/가중치 일치를 검사한다. 개별 환자 MSE는 독립 행렬식으로 대조한다. 기존 public/private 참고 가중치를 복원하면 원 점수와의 일치도 확인할 수 있다.
4. Public-only/private-only/pooled는 같은40명·같은 bank·같은 metric으로 비교한다. rho/lambda/표현/seed/환자 subset을 결과에 맞춰 탐색하지 않는다.
5. 결과는 기존 개발자료의 finite-bank 진단으로 표시한다. 통계적 유의성·작은 MSE 차이를 생성 효용·최종 일반화로 해석하지 않는다. 새 생성/DP 실험은 이번 단계 밖이다.

## 7. 감사의 한계와 완료 범위
이번에는 기존 검증 캐시의 hash·계약·역할·소스 수식을 확인했다. 원시 activation/noise부터 A/B/Q를 다시 누적하거나 UNet을 독립 재실행하지 않았다. 원 검산도 모든 신경망 계산을 재실행한 것은 아니며, 저장 산술과 각각4개 projection witness 및 source/input binding을 확인한 범위다. 따라서 현재 결론은 **검증된 동일 목적의 캐시를 pooled 산술에 재사용해도 되는 출처 근거**이며 새 결과의 정확성은 별도 검산 책임이다.

