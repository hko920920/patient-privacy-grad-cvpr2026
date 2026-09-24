# DINO 환자-DP와 Dosser의 연산 비교 — 2026-09-24

범위: 현재 DINO 단독 코드와 Dosser 원문·공식 구현의 대응을 확인했다. 새 모델 실행, private target 생성, DP release, 합성학습, V 평가는 모두 0이다. 아래는 비교 후보의 명세이며 실행 계약은 아니다.

## 연산 비교표

| 항목 | 현재 DINO 단독 | Dosser 원문·공식 구현 | 공정한 적용 후 남는 차이 / 허용되는 주장 |
|---|---|---|---|
| 신호 | 공개 DINO의 P-only PCA16/q95 특징에서 고정 선형 head의 CE gradient 34좌표 | 일반 틀은 feature와 gradient 모두 허용; 공개 실험 구현은 embedding 분포 정합 | gradient 사용 자체는 Dosser 틀 밖의 새 원리가 아니다. 선형 gradient·cosine은 LGM에도 있다. |
| 공개 압축 | P의 환자 균등 PCA, whitening 없음; 특징을 줄인 뒤 gradient 계산 | 보조자료 PCA로 신호의 부분공간을 구함 | 같은 공개 DINO·P·projection을 허용해야 공개 encoder 이득을 방법 차이로 오인하지 않는다. feature PCA와 일반 gradient-space PCA는 동일 연산이라고 하지 않는다. |
| 조건 | 고정된 4개 변환/head; 조건별 target 보존 | 변환 seed·모델 상태·projection을 보존해 대응 신호 사용 | 조건 보존·재사용은 공통 원리다. 현재 head 분포/조건 수는 recipe 선택이며 그 자체가 신규성은 아니다. |
| 보호 | 한 환자의 class별 방문 평균, 두 class의 신호·존재 지표를 한 274좌표 query로 보호 | 원문은 class별 sample/Poisson sampling과 여러 보호 신호의 회계 | 환자는 양·음성 class 모두에 기여할 수 있다. class를 서로 다른 환자집합처럼 병렬 회계하면 안 된다. baseline도 환자 단위 및 비공개 분모를 올바르게 보호해야 한다. |
| 목표와 손실 | 보호된 class 평균을 0.5씩 합친 gradient에 조건별 cosine 정합; synthetic 쪽에는 Q patient clipping을 재적용하지 않음 | 원문/구현의 분포 정합은 class별 신호를 유지한 제곱 L2; synthetic 신호에도 clipping | 실제 차이는 신호·class 결합·거리·synthetic clipping의 묶음이다. 하나를 바꾼 원인 실험으로 해석하지 않는다. |
| 영상·평가 | 공개 template+pyramid, 128장/200회, 고정 receiver readout | 공개 원형은 다른 초기화·학습량·평가 조건 | 기존 renderer·초기화·학습량·receiver를 공통화한 비교는 제한된 적용 비교다. 원형 전체 재현/최적 설정의 Dosser 우위 검증과 구분한다. |
| 이름과 기여 | 현재 작동하는 환자-DP gradient 증류 구성 | 보호 신호 재사용·부분공간이라는 가까운 선행 | 같은 gradient/집계/보호/거리까지 허용하면 거의 같은 구성이 될 수 있다. 이를 서로 다른 방법 두 개처럼 실행하지 않는다. |

근거: [Dosser 원문 §3, §4.1](https://arxiv.org/html/2508.01749v1), [공식 dosser.py](https://github.com/humansensinglab/Dosser/blob/dff0c57f6f22f66d500a66bad4a9db757e5f714a/dosser.py), [공식 PCA](https://github.com/humansensinglab/Dosser/blob/dff0c57f6f22f66d500a66bad4a9db757e5f714a/pca.py), [LGM §3.1](https://arxiv.org/html/2511.16674v1).

현재 코드 근거: ../../code_working/receiver_distillation/signals.py:57, :80, :109, :123; second_source.py:17, :55; patient_dp_dino.py:101, :133, :181; prrd_v3/encoders.py:137. 세부 line 번호는 source_bindings.json과 결속된 snapshot 기준이다.

## 공식 코드와 논문의 차이

확인한 공식 revision은 dff0c57f6f22f66d500a66bad4a9db757e5f714a다. dosser.py:209–258에서는 clipped 무잡음 합을 저장하고, :376–379에서는 train_step 호출 시 그 합에 잡음을 더한다. 논문 §3.2의 보호된 tuple을 저장해 반복 사용하는 설명과 다르다. 별도 noise 보정 script는 sampling iteration 수를 입력으로 사용한다.

단, “step마다 잡음 생성 호출”을 “step마다 독립 잡음”이라고 해석하지 않는다. utils.py:497–530의 DiffAugment는 저장된 변환 seed로 전역 torch RNG를 재설정하고, 이후 train_step은 기본 RNG의 randn_like를 사용한다. 따라서 replay와 noise 난수의 결합 가능성도 있다. 이 확인은 정적 코드 대조이며 런타임 재현이나 전체 privacy 증명이 아니다. 비교 구현에서는 공개 변환 RNG와 비밀 DP RNG를 분리해야 한다. [공식 RNG 코드](https://github.com/humansensinglab/Dosser/blob/dff0c57f6f22f66d500a66bad4a9db757e5f714a/utils.py#L497-L530)

따라서 이 코드를 그대로 실행하고 원문의 “고정된 보호 summary 이후의 후처리” 보장이 자동 적용된다고 가정하지 않는다. 코드 경로만으로 논문 전체의 privacy 실패나 출판 결과 무효를 선언하지도 않는다. 비교 adapter는 논문에 명시된 보호 신호 선계산·재사용 원칙을 구현하고 원형과 변경점을 공개해야 한다. 이 구현 차이를 우리 방법의 신규성이나 우월성으로 세지 않는다.

## 한 비교로 확인할 질문

**같은 공개 DINO 표현과 환자-DP 예산에서, 현재 gradient·cosine 제작 구성은 class별 feature-mean·L2 제작 구성보다 개발 receiver 효용을 높이는가?**

후자는 Dosser의 distribution-matching 원리를 현재 접근 조건에 적용한 대조 후보다. “최강 Dosser 전체”라는 이름을 붙이지 않는다. 현재 구성 대비 실용적인 신호 선택의 가치를 확인하는 비교이며, gradient 자체의 새로운 발명이나 Dosser가 허용하는 모든 variant에 대한 우위 검사가 아니다.

공통 조건은 기존 P/Q 역할·class별 환자 균등 가중치, 공개 DINO checkpoint/PCA16/q95, 네 변환, 128장·200회·seed101·microbatch16, renderer·공개 초기상태, 마지막 PNG 채택, 기존 DenseNet/ResNet18 V readout이다. 기존 DINO DP 두 bank를 모두 보존·비교하며 좋은 한 bank만 선택하지 않는다. 새로운 receiver나 final 자료를 열지 않는다.

feature 대조는 class별 16차원 특징 신호를 보존한다. 네 조건과 두 count를 묶으면 query는 2×4×16+2=130좌표다. 자기 신호에서 P-only clipping 경계를 계산하고, 전체 add/remove 환자 epsilon8/delta1e-5를 사용한다. 현재 gradient query의 clipping 숫자를 복사하거나 차원을 274로 억지로 늘려 불리하게 만들지 않는다. 환자의 class 내부 전체 조건 기여를 제한하고 두 class 및 두 count를 함께 보호한다. synthetic clipping과 공개 P 신호 처리도 baseline에서 맞추어 정의해야 한다.

다만 cosine에서 L2로 바뀌면 matching loss의 크기와 영상 규제의 상대 세기가 달라진다. 동일 숫자의 learning rate/규제 계수를 무조건 유지했다고 공정성이 확보되지는 않는다. 실제 실행 전에 공개 P/구성 배열에서 loss 정규화·단위와 clipping 미분 규칙을 명시해야 한다. V에서 유리한 계수를 탐색하지 않는다. 이 필요한 항목 때문에 현재 문서는 실행 준비 완료를 선언하지 않는다.

현재 DINO의 P/Q 조건 특징 cache가 존재하므로 encoder 재추출은 필요하지 않다. 그러나 그 Q cache는 비DP 내부 자료다. 새로운 feature target의 보호 요약을 만들면 새 private release 1회가 발생한다. 기존 DP gradient 요약의 단순 후처리라고 기록해서는 안 된다.

## 판정과 비용의 범위

- gradient 구성이 낫다면, 이 고정 조건에서 해당 제작 구성의 추가 효용을 지지한다. 전체 Dosser 대비 우위나 특정 구성요소의 단독 인과 효과로 확대하지 않는다.
- feature 구성이 비슷하거나 낫다면, 현재 복잡한 gradient 구성의 필요성을 뒷받침하지 못한 결과로 보존한다.
- 혼합 결과이면 차이를 미확정으로 남긴다. 계수·잡음·checkpoint를 자동으로 다시 고르지 않는다.
- 어느 결과도 단일 seed·신규 잡음 하나로 방법 전체의 분산이나 논문 신규성을 확정하지 못한다.

기존 DINO bank 약35분은 4조건 encoder 연산을 공유하는 제작 단계의 참고치다. 새 loss adapter·공개 검산·목표 생성·평가 시간을 측정하지 않았으므로 전체 소요시간이나 실행 상한을 확정하지 않았다. 비교를 실행한다면 변경 경로 검증과 새 release 범위만 추가로 결속하고 기존 재개·PNG·평가 검증은 재사용한다.

## 완료 및 보존

이 작업에서 닫은 것은 연산 차이, 공통 선행 범위, 비교 후보 및 그 해석 한계다. CVPR 기여나 strong-baseline 우위는 미확정이다. 환자-DP 효용 및 A2의 mixed 결과는 그대로 보존한다. 새 합성/release/평가 0, 연구 단계2 진행 중, 현재 실제 결과는 RECEIVER_RESNET18_REUSE_RESULTS_20260924.md다.

공식 공개 소스 6개와 revision/hash를 spec_sources/dosser_operation_comparison_20260924/dff0c57f6f22/에 보관했다. 공식 코드는 읽기만 했으며 실행하거나 프로젝트 학습 코드에 복사하지 않았다.
