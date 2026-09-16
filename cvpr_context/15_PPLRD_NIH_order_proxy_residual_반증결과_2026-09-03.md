# PPLRD NIH order-proxy residual premise 반증 결과

- 기록일: 2026-09-03
- 공식 상태: `FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE`
- 최신 판단: **NIH weak-label/order-proxy 기반 PPLRD 분기 종료**
- 보존되는 양성 결과: same-patient previous CXR은 강한 identity/anatomy carrier다.
- 실패한 핵심: weak finding-label transition으로 예측한 residual은 단순 prior copy보다 나쁘다.

## 1. 한 문장 결론

> 이전 환자 영상은 다음 영상의 안정 구조를 강하게 제공하지만, NIH의 `Follow-up #`와 weak label
> 변화만으로 학습한 residual은 그 구조에 유용한 변화를 더하지 못하고 오히려 예측을 망쳤다.

따라서 14번에서 조건부로 올린 PPLRD를 현 로컬 NIH 자료로 generator까지 진행하지 않는다. 결과가
좋았던 identity carrier만 골라 `longitudinal change도 된다`고 바꾸어 말하지 않는다.

## 2. 결과 전에 고정한 설계

- 앞선 여섯 selection의 union인 528 unique patients를 모두 제외했다.
- remaining public-development PA records에서 exact `Follow-up #` gap 1 pair만 사용했다.
- 한 patient에서 pair 하나만 쓰고, `change/no_change` 각각 train 96+validation 48로 균형화했다.
- 총 288 unique patients, 288 ordered pairs, 576 unique images다.
- 14 findings+`No Finding`의 addition/removal 30차원 transition을 입력으로 하고, training 192 pairs의
  `future feature - prior feature`를 multi-output ridge로 예측했다.
- alpha `{0.01,0.1,1,10,100}`은 patient-hash 5-fold training-only CV로 골랐다.
- true transition, transition-row derangement, zero-residual copy와 finding-matched shuffled prior를
  동일 held-out patients에서 비교했다.
- generic DINOv2와 NIH-pretrained RAD-DINO가 모두 사전 gate를 통과해야 했다.
- 5,000 patient bootstrap, 결과 전 고정 threshold를 사용했고 feature/regressor는 저장하지 않았다.

## 3. 핵심 결과

### Changed-label validation 48 pairs

| encoder | true residual / prior copy | 95% CI | wins | 판정 |
|---|---:|---:|---:|---|
| DINOv2 | `1.140914543` | `[1.073630, 1.236754]` | `8/48` | 명확한 FAIL |
| RAD-DINO | `1.159127120` | `[1.101906, 1.245540]` | `1/48` | 명확한 FAIL |

사전 기준은 ratio `<=0.995`, upper CI `<1`, wins `>=0.60`이었다. 두 encoder 모두 residual을
더했을 때 copy보다 평균 오차가 14--16% 커졌다.

### True transition 대 deranged-training control

| encoder | true / deranged | 95% CI | wins | 판정 |
|---|---:|---:|---:|---|
| DINOv2 | `0.952613701` | `[0.884638, 1.019941]` | `24/48` | CI·wins FAIL |
| RAD-DINO | `0.978814107` | `[0.933881, 1.024487]` | `31/48` | CI FAIL |

transition label에 약한 평균 신호는 있을 수 있지만 CI가 모두 1을 포함한다. 더 중요하게는 true
transition model 자체가 zero-residual copy보다 훨씬 나쁘다. 따라서 이 secondary ratio를 주 방법
성공으로 승격하지 않는다.

### Same-patient prior 대 finding-matched shuffled prior

| encoder | correct / shuffled | 95% CI | wins | 판정 |
|---|---:|---:|---:|---|
| DINOv2 | `0.516104650` | `[0.396098, 0.685204]` | `46/48` | PASS |
| RAD-DINO | `0.319953448` | `[0.266193, 0.378757]` | `48/48` | PASS |

shuffled prior는 sex가 100% 같았고 previous exact-label match 62.5%, future exact-label match 54.17%,
평균 label Jaccard는 previous `0.7031`, future `0.6962`였다. 이 hard control에서도 correct prior가
강하게 이겼으므로 stable patient-related visual information은 재확인된다. 다만 이는 clinical
anatomy만이 아니라 acquisition signature와 biometric identity를 포함할 수 있다.

### No-change negative control

- DINOv2 true/copy: `1.008680087`, CI `[1.005181,1.012722]`, wins `11/48`
- RAD-DINO true/copy: `1.007988247`, CI `[1.003928,1.012177]`, wins `18/48`

사전 허용 상한 `1.01` 안에는 간신히 들었지만 두 encoder 모두 평균적으로 copy보다 나쁘다. 이는
transition regressor의 intercept/평균 drift도 유익하지 않음을 뒷받침한다.

## 4. 원인 해석

두 ridge arm 모두 training-only CV에서 가장 강한 shrinkage인 alpha `100`을 선택했다. 낮은 alpha의
CV error가 더 컸고, 큰 alpha에서도 held-out 변화 residual은 copy를 이기지 못했다. 따라서 단순
overfit 하나보다 다음 조합이 더 타당하다.

1. NIH weak NLP label은 공간적 범위·중증도·새 장치·치료 반응을 표현하지 못한다.
2. `Follow-up #`에는 실제 경과시간과 encounter context가 없다.
3. feature delta의 대부분은 label transition으로 설명되지 않는 촬영·자세·미세 해부 변화다.
4. 같은 환자 prior는 stable identity를 전달하지만, 그 위에 얹을 cross-patient change direction이
   현 conditioning에는 부족하다.

이 결과는 EHRXDiff처럼 richer EHR event를 쓰면 실패한다는 증거가 아니다. 반대로, 로컬에 그러한
자료가 없는데도 EHR 효과를 가정하고 generation을 시작할 근거도 아니다.

## 5. 판정과 금지 claim

### 종료하는 것

- NIH `Follow-up #`를 elapsed time으로 쓰는 것
- NIH weak label additions/removals만으로 longitudinal residual diffusion을 추진하는 것
- strong correct/shuffled prior 결과를 change prediction 성공으로 바꾸어 말하는 것
- encoder, alpha, gap, 환자 subset을 결과에 맞춰 바꿔 같은 v1 가설을 구제하는 것
- 이 feature diagnostic을 image generation, privacy, clinical utility 근거로 사용하는 것

### 남는 것

- 반복 CXR에는 매우 강한 patient-related identity carrier가 있다.
- 이 신호는 synthetic generation에서 보존해야 할 효용일 수도, 제거·보호해야 할 biometric leakage
  위험일 수도 있다.
- MIMIC-CXR timestamp+report/EHR access가 실제 확보되면 별도 가설과 새 protocol로 longitudinal
  direction을 다시 정의할 수 있다. 현재 PPLRD v1을 그대로 재실행하는 것은 아니다.

## 6. 현재 연구 순위에 미치는 영향

현 시점에 실증적으로 살아 있는 CVPR 방법 후보는 다시 **없다**. 실행 가능성 순서는 다음과 같다.

1. **Biometric-aware patient-private CXR diffusion의 신규성 조사**: 이번에 강하게 확인된 patient
   identity signal을 생성 효용이 아니라 leakage/suppression 대상으로 전환할 수 있는지 먼저 본다.
2. **image-DP 대 patient-DP generation audit/benchmark**: 기존 구현이 가장 많이 준비됐지만
   CheXGenBench가 fidelity/privacy/utility benchmark를 이미 점유하므로 privacy-unit mismatch와
   formal patient-DP가 아니면 독립 기여가 약하다.
3. **richer-data longitudinal residual**: MIMIC access가 생길 때만 재개한다.
4. **PSPD attention 변형**: 두 독립 반증과 cross-view 선행 밀집 때문에 종료 상태를 유지한다.

1순위는 아직 추천 확정이 아니라 **다음 문헌 falsification 대상**이다. 직접 선행이 이미 같은
교집합을 점유하면 구현 전에 폐기한다.

## 7. 재현 정보

- protocol SHA-256:
  `92B0B0D031255E6C606074F76FCCD081688AFC429779C53456C94DDAE995B14D`
- implementation SHA-256:
  `47B7096736563D3752F27AFDC0085441005809DD2AC381B0412E77A83AF25CCB`
- selection SHA-256:
  `4086BF15E8C50547C12EEBF26BB755C923F1799AB9D1AA83CEBD3C669383E60C`
- report SHA-256:
  `EF0856415E435E0808FF81090E7F58687DFBAB94F635A55FBF82D2DDDC07CA3F`
- elapsed feature+analysis: `73.3661 s`
- peak allocated CUDA: `0.65145 GiB`
- pre-run unit tests: `6/6 PASS`
- retained: selection CSV와 aggregate report만; feature, ridge, checkpoint, latent, gradient, optimizer,
  generated image는 없음

상세 machine-readable 결과는
`code_working/_reports/nih_cxr14_order_proxy_residual_premise_v1_001/report.json`에 있다.
