# 첫 A2 환자-DP 성능 비교 — 실행 계약

2026-09-23 사용자 지시로 한 묶음 실행을 시작한다. 기존 목표 생성 명세·검산을 재사용하고, 정상 구간에서 단계별 재승인을 요구하지 않는다. 큰 연구 단계2의 개발 비교다.

## 허용 범위와 시간

- Q의 보호 신호 **한 번 생성**, 제한-only A2 한 bank와 DP A2 한 bank를 각각 새로 학습한다.
- 기존 공개전용/P+Q 비DP seed101 결과를 재사용하여 총 네 조건을 비교한다.
- 새 두 bank는128장·200회·seed101·microbatch16. 기존 BioViL+DINOv2, K4/head seed101, public projection, AdamW lr0.01, weight_decay0, 해상도 활성1/33/65/97/129/161을 유지한다.
- 기존 공개 seed101 template·renderer/optimizer/RNG 초기상태와 같아야 한다. 기존 학습 checkpoint에서 이어 시작하지 않는다.
- 예상 전체155~180분, 합성·저장 참고148분. **bank당 누적120분, 전체 작업5시간** 상한이다.
- 전체 시작2026-09-23 11:08:39 UTC, 절대 종료상한2026-09-23 16:08:39 UTC. 재개해도 이 상한을 재설정하지 않는다.

## 보호 계약

[기존 목표 명세](RECEIVER_PATIENT_DP_TARGET_SPEC_20260923.md)와 [검산 결과](RECEIVER_PATIENT_DP_TARGET_RESULTS_20260923.md)를 그대로 적용한다.

P 공개, Q 환자 한 명 전체 추가/제거 인접성. 환자/class 안 방문 평균 후 두 encoder×네 조건의 gradient272좌표를 만들고, class별 P-only norm q95로 제한한다. 두 class gradient와 class 존재 count를 묶어546좌표, 환자 norm≤1, 합계 sensitivity1이다.

공개 clipping 기준은 음성2.7412545531339387, 양성2.689519212597726이다. 한 합계 query 전체에 **ε=8, δ=1e−5**, analytic Gaussian σ=0.6002290722589745를 적용한다. 별도 class count release나 subsampling은 없다. 여러 조건·200회 합성은 같은 보호 신호의 후처리다.

분모는 보호된 Q class count의 비음수 부분과 공개 P count로 구성한다. noisy class 합계의 norm은 해당 noisy 질량×clipping 기준 이하로 투영한다. 영점 target 예외는 같은 공개 P target으로 대체한다. 참 Q 분모 재사용·잡음 재추출은 하지 않는다.

제한-only는 동일한 집계·제한·분모 복원·후처리를 사용하되 Gaussian noise만 제외한다. 이는 비DP 내부 대조다. 실제 DP noise는 운영체제 난수에서 한 번 뽑고 재현 seed·실현 noise를 저장하지 않는다. 이 한 요약의 내부 보호 target과 합성 결과를 여러 개 공개해도 그 자체는 같은 release의 후처리이나, 비DP 대조·원자료 진단·평가까지 보호되는 것은 아니다.

DP PNG 별도 패키지에는 PNG·합성 label·공개 학습 규칙과 그 파일 hash만 넣는다. 내부 checkpoint/계약 signature/Q cache hash/원분모·원합계·실현 noise·원 평가 예측은 넣지 않는다. 이번에는 로컬 산출물을 만들며 외부 게시·배포는 하지 않는다. 과거 Q/V 기반 방법 선택을 소급해서 DP로 주장하지 않는다.

## 실행·평가

기존 학습·저장·재개·PNG 코드는 변경하지 않는다. 별도 adapter에서 두 job의 target/hash, 절대 시간상한, DP/비DP metadata만 연결한다. 구성 특징의 loss/gradient 유한성·잘못된 job 거부·기존 초기상태 동일성을 확인한다. 추가 profile이나 이전 전체 검사를 반복하지 않는다.

순서는 제한-only → DP8이며, **두 최종200회 PNG가 모두 고정된 뒤** 평가한다. 기존 BioViL/DenseNet V 특징과2,000개 동일 환자 bootstrap draw를 사용한다. V 원영상의 새로운 특징 추출은 필요 없다. AP/AUROC는 기존 receiver의 point-only ridge0.1 학습 규칙을 유지한다.

주 비교는 DenseNet DP8−공개전용 A2이다. 이번에 사용할 개발 판단은 실행 전에 다음으로 정한다.

- 긍정적 개발 신호: AUROC 점차이>0, paired 환자95% 구간 하한>0, AP 점차이≥0을 모두 만족.
- 혼합/불확실: AUROC 점차이는 양수이지만 나머지 중 하나 이상 미충족.
- 추가 AUROC 신호 없음: AUROC 점차이≤0.

새 효과크기 문턱이나 사후의 성공 기준은 추가하지 않는다. 차이 크기와 모든 구간을 함께 보고한다. BioViL은 source 보존의 보조 결과다. 제한-only−비DP, DP8−제한-only, DP8−비DP도 보고하여 제한과 noise 포함 처리의 영향을 구분한다.

환자 bootstrap은 고정된 두 bank와 한 DP 잡음 실현에 조건부다. 합성 seed나 DP 잡음 반복의 분산, 독립 수신자·환자집단, 강한 선행 우위를 대신하지 않는다.

검증 실패·비유한값·안전한 재개 불가·시간 상한·범위 변경 필요 시에만 멈춘다. 중간 성능·이미지 모양으로 checkpoint/계수/잡음을 선택하지 않는다. 추가 release·seed·학습 연장·DP budget 탐색·Expert·Reserved·확인용 수신자 자동 실행은 없다.

실제 machine contract와 jobs: `code_working/_reports/receiver_dp8_first_20260923_v1/`. 작성된 target과 실행 코드 hash를 이곳에서 동결한다.
