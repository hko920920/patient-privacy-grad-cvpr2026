# A1/A2 첫 200회 개발 비교 결과 — 2026-09-22

기록 시각: 2026-09-22T10:24:05.543758+00:00

## 판단

**추가 독립 bank 반복을 검토할 개발 신호 충족.** A1(BioViL 단일 source)과 A2(BioViL+DINOv2)를 각각 128장·200회·seed101로 완료하고, 두 최종 PNG bank를 동결한 뒤 동일 개발 V에서 평가했다. DenseNet A2−A1 AUROC는 +0.135277, AP는 +0.030610다.

이는 고정된 첫 200회 recipe와 한 합성 seed의 개발 결과다. DenseNet은 이미 재사용된 개발 receiver이고 V 역시 적응적으로 사용된 개발자료다. 독립 최종 확인, DP 효용, 논문 신규성의 확정으로 부르지 않는다. 기준 충족 여부와 무관하게 추가 seed·500회 연장·DP·final을 자동 실행하지 않았다.

## 고정한 비교

- A1과 A2의 공개 template 128장, seed101, labels 64/64, BioViL target/head/조건, 환자·class 가중, optimizer, update 수를 동일하게 유지했다.
- A2만 별도 DINOv2 조건별 gradient 제약을 추가했다. K 내 평균 후 encoder 간 동일 비중 평균이며 smooth-max/CVaR/gradient balancing이 아니다.
- 두 조건은 같은 update 수다. A2는 encoder가 하나 더 있어 총 연산량은 더 많다.
- 해상도8/16/32/64/112/224의 활성 update는1/33/65/97/129/161이고 마지막 수준을40회 학습했다.
- AdamW lr.01, decay0, betas(.9,.999), eps1e-8, FP32 영상/encoder와 FP64 작은 CE gradient를 유지했다. Anchor.01, TV1e-4는 한 번만 적용했다.
- 모멘트/기능/환자 관계 항을 추가하지 않았다. 최종200회 PNG만 평가했고 중간 수신자 성능으로 checkpoint나 설정을 고르지 않았다.
- P672명/813장은 공개, Q2027명/5097장은 보호 대상 정보 출처다. 이번 실행 자체는 비DP 내부 개발이다.

## 목표 준비와 구현 검산

BioViL의 준비된 A1 목표를 그대로 재사용했다. DINOv2 ViT-B/14는 로컬 공개 checkpoint와 timm 구현을 고정했다. Grayscale을 RGB로 반복하고 tensor resize short256/center224/ImageNet 정규화를 사용했다. 사전학습518 grid의 positional embedding은 timm의 고정 dynamic interpolation을 적용했다. 이것은 이번에 명시한 DINO source 전처리이며 BioViL 경로를 바꾸지 않았다.

DINO는 P만으로 환자 균등 PCA16 및 q95 scale을 정했고 whitening이나 label 선택은 하지 않았다. 이후 P/Q 네 조건의 target을 각각 저장했다. 각 조건은 A1과 정확히 같은 상대 affine/brightness 변환이고 head는 encoder ID에 결속했다. 이전 영상별 aug-mean target은 사용하지 않았다.

- DINO forward: 공개 projection813 + 조건별23640 = 24,453장.
- P/Q 목표 독립 검산 최대 오차: 2.274e-13 (기준1e-12).
- Class별 환자 수: P669/6, Q2015/114, pooled2684/120. 환자 내 방문 평균→class별 환자 평균→두 class 각0.5.
- 새200 일정 CPU3검사 통과. 모든 해상도 전환 직전 저장→재개→다음 update의 parameter/optimizer/RNG 상태가 정확히 일치했다.
- 실제 공개P4, K4, 두 encoder, microbatch1/4 검산 통과. 기준 loss2e-6, gradient2e-7+5e-4×reference peak 유지. 최대 gradient차이 1.490e-08.
- 메모리 제한 때문에 기준 gradient는 각 encoder의 전역 retained graph를 차례로 역전파해0.5씩 더하고 prior를 한 번 더했다. 실제 실행의 두-pass 재계산과 같은 수학 목적이다.
- 기존 PRRD 생산33파일과 A1 BioViL 목표는 변경하지 않았다. 새 경로·기존 hash·checkpoint·PNG 검증만 사용했다.

## 고정 V 결과

| 조건 | BioViL AUROC | BioViL AP | DenseNet AUROC | DenseNet AP |
|---|---:|---:|---:|---:|
| A1, 200회 | 0.731632 | 0.079062 | 0.536456 | 0.047955 |
| A2, 200회 | 0.725230 | 0.076155 | 0.671734 | 0.078565 |
| 이전 B, 500회(참고) | 0.777578 | 0.127708 | 0.567975 | 0.060098 |
| 공개 실자료 readout(참고) | — | — | 0.618009 | 0.064941 |

V는 2,026명·5,047장·양성223장이다. 기존 BioViL/DenseNet V 특징을 재사용했고 V 원영상 forward는0회다. 각 최종 PNG bank에서 각 encoder의 특징128개를 추출하고 기존 point-only ridge0.1 학습 규칙을 적용했다.

### A2−A1의 paired 환자 bootstrap

| 항목 | 차이 | 환자 cluster 95% CI |
|---|---:|---:|
| DenseNet AUROC | +0.135277 | [+0.088795, +0.182515] |
| DenseNet AP | +0.030610 | [+0.018458, +0.046869] |
| BioViL AUROC | -0.006402 | [-0.010740, -0.001777] |
| BioViL AP | -0.002906 | [-0.005713, -0.000972] |

기존2,000개 환자 재표집 배열을 그대로 사용했다. 고정된 합성 bank를 조건으로 V 환자 불확실성을 나타내며 합성 seed 불확실성은 포함하지 않는다. 직접 sklearn 가중 지표 검산 최대오차는 1.110e-16다. 이전 B500과 public-real은 맥락을 위한 참고이며 matched-budget 인과 비교/상한이 아니다.

## 사전 판정

| 조건 | 결과 |
|---|---|
| DenseNet ΔAUROC ≥ +0.02 | True |
| DenseNet AP 비감소 | True |
| BioViL ΔAUROC ≥ −0.02 | True |

세 조건을 모두 만족했는지로만 첫 투자 신호를 판정했다: **True**. CI를 모순되는 두 번째 GO/STOP 규칙으로 사용하지 않았다. 이 비교만으로 pairing, encoder 선택, DP 신호 압축의 독립 기여를 판정하지 않는다. 한 seed200회의 결과는 전체 방향의 가능/불가능 정리가 아니다.

## 실제 비용과 실행 경계

| 작업 | 실제 시간 |
|---|---:|
| DINO 공개 projection+P/Q target+검산 | 5.88분 |
| 실제 두-source gradient 검산 | 15.06초 |
| A2 공개128 profile (warm-up1+측정3) | 1.64분 |
| A1 worker (시작·저장·PNG 포함) | 40.09분 |
| A2 worker (시작·저장·PNG 포함) | 73.92분 |
| 기존 V 기반 고정 평가 | 0.22분 |

사전 상한 A1 60분/A2 120분을 유지했다. A2 profile의 measured 평균은 21.938초/update였다. 최대 allocated는 5.921GiB이다. 종료시 free space는 10.47GiB다.

이번 실행의 image-forward/backward 합계:

```json
{
  "BioViL": {
    "forward": 414048,
    "backward": 206912
  },
  "DINOv2": {
    "forward": 233445,
    "backward": 104512
  },
  "DenseNet": {
    "forward": 256,
    "backward": 0
  }
}
```

본학습400회와 기술 profile4회만 optimizer update했다. 원본 사적 pixels는 목표 준비 후 합성에서 재접근하지 않았다. 합성은 public templates와 내부 고정 target만 사용했다. DP를 적용하지 않았으므로 공개 가능한 patient-DP 합성자료로 주장하지 않는다.

저장 정리: 새 bank의 중복 checkpoint payload만 1047.34MiB 정리했고 초기 및 최신2개/final, 모든 hash receipt·trace는 보존했다. 이번 campaign 산출물은 현재 773.59MiB다. 과거 결과·원자료·기존 bank를 삭제하지 않았다.

## 재현 자료

- 실행 계약: code_working/_reports/receiver_a1a2_s200_20260922_v1/campaign_contract.json
- 실행 code/입력 결속: execution_bindings.json 및 implementation_snapshot/
- 새 target: second_targets/private/ (NONDP_INTERNAL)
- bank: banks/dev_A1_condk4_101_s200 및 banks/dev_A2_condk4_101_s200
- 두 최종 bank seal: two_bank_seal.json
- 점수·paired bootstrap: evaluation/predictions_private.npz, evaluation/paired_bootstrap.npz
- 전체 결과: evaluation/result.json, campaign_result.json

주요 evidence SHA256:

```json
{
  "campaign_contract.json": "60515338c8ea080830c43d9a717cc728492bf9796113520916862fa075c264d1",
  "execution_bindings.json": "fcf881966960bbb6bac84fc3f558e1c03c2754f1254c40730d585822d08342d6",
  "second_targets/result.json": "1c1df787e1cacf836ecd4109553725fbe3483f5c5f5ebc5337b2a501c6e7db07",
  "second_targets/private/target_handoff.json": "8b2a8ec3e8665bb4c69f6c4e2f77b371240430eea1c559f51174b6788ff6678c",
  "schedule_verification.json": "1e79a0293c1409f0ca543cebe13967f45d1677a653d901b4186d25823fad506c",
  "changed_path_verification/result.json": "d921e3ed905bc0297ba8f84bc7e38c5fe95682320b6fbd62f83e4629c33b5557",
  "profile_A2_result.json": "f98623444b58882a04536b3462625083884d248fbfdd83ed61f80001aa0c216a",
  "A1_result.json": "30ed4d2e8763e4b7f07cc0ac037cac6ad76ac4b77f902219b76466b807015d15",
  "A2_result.json": "40a4c0f3239d95a3fdc6553b585dc58a5b252d20d87fe35c8699495139b363f3",
  "two_bank_seal.json": "e09176777057a8f539e4b2bfab36990463931e25fe3dc20323f5a26514868f4d",
  "evaluation/result.json": "3c83e2f25cf719e7dc821f327b2564c3ff9009c97431525d5d41506197f59216"
}
```

## 종료 상태

두 bank 및 고정 개발 평가 완료. 추가 합성·다른 seed·500회 연장·새 계수·관계C/D·augmean·DP·Expert·Reserved·최종 unseen receiver 실행 없음. 이 결과를 보존하고 별도 근거 없이 자동 후속 실험을 시작하지 않는다.
