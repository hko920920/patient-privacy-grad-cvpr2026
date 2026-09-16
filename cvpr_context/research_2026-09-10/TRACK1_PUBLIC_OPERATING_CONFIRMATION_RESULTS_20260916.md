# 공개 의료 LoRA: E4·CFG7.5 예약 입력 확인 결과

**판정: 이번 확인은 좋다. 두 판독자 모두16/16장 통과, prompt별4/4로 사전에 정한 형태 관문을 넘었다. E4·CFG7.5를 후속 비DP 작은 head 비교의 운영 backbone으로 조건부 채택한다.**

이제 이 공개 recipe의 step/seed/CFG 조정을 멈추고, 새 backbone에서 작은 head와 사적 자료가 실제 생성 효용을 더하는지 검증할 수 있다. 임상 품질·일반화 안정성·CFG7.5 우월성·private 추가 효용·patient-DP 성능을 확보했다는 뜻은 아니다. 이전 E4/CFG1 확인10/16 실패는 그대로 보존했다.

## 1. 무엇을 실제 실행했나

[생성 전에 고정한 확인 명세](TRACK1_PUBLIC_OPERATING_CONFIRMATION_PROTOCOL_20260916.md)에 따라 직전 개발 실험의 계약에 이미 예약된 C0–C3 네 seed를 사용했다. 이 네 실제 initial tensor를 generic/normal/effusion/cardiomegaly 네 prompt에 공통 적용해16장을 한 번만 생성했다.

E4 checkpoint, CFG7.5, FP32, DDIM30 eta0,256px, pinned SD2.1/VAE를 유지했다. conditional/null embedding은 직전 검증된 입력에서 exact 재사용했다. 새로운 text encoding·학습·역전파·작은 head·DP 실행은 모두0이다. 결과를 보고 checkpoint·seed·CFG를 바꾸거나 재생성하지 않았다.

이번 실행 계약 SHA256: `9cb1d51b6d11ad5e046d5d207283788459e5980306d82836c9729231746c31bb`.

## 2. 판독 결과

| 항목 | 통과 |
|---|---:|
| root 판독 | 16/16 |
| 별도 독립 판독 | 16/16 |
| 두 판독자의 PASS 교집합 | **16/16** |
| Generic | 4/4 |
| Normal | 4/4 |
| Effusion | 4/4 |
| Cardiomegaly | 4/4 |
| 판독 불일치 | 0 |

관문은 전체≥12/16 및 모든 prompt≥2/4였다. 네 latent block별로도4/4였다. 모든 영상을 보존했다.

![고정 확인16장 전체와 교집합 판정](../../code_working/_reports/public_operating_confirmation_20260916_v1/confirmation_reviewed.png)

두 판독자는 다른 원표를 읽기 전에 각각 판정을 파일에 확정했다. 독립 판독 담당은 opaque ID grid·ID 목록·경계 PNG4장만 읽었고, 이번 checkpoint/CFG/prompt/seed 매핑이나 다른 판독 결과를 보지 않았다. root는 실행자이므로 E4/CFG7.5 정체를 알고 있었으며, prompt/seed label이 없는 전체 grid를 먼저 판독했다. 완전한 이중 눈가림으로 부르지 않는다.

두 명 모두 AI 판독자이며 임상 전문가 검토가 아니다. 별도 판독은 기준 적용의 일관성을 확인하는 절차이지, 서로 독립인 두 임상 검사나 모집단 표본으로 해석하지 않는다. 같은 네 latent를 모든 prompt가 공유하므로16개 독립 입력의 안정성 추정도 아니다.

## 3. 좋아진 점과 남은 한계

**좋아진 점:** 공개자료만으로 만든 E4가 이전 개발 묶음과 별도로 예약한 새 입력에서도 기본 흉부 형태를 유지했다. 고정한 형태 관문을 통과했으므로 작은 head의 다음 적용성 검사를 시작할 운영 근거가 생겼다. 현재 공개 backbone을 더 조정하는 데 시간을 쓰지 않는다.

**남은 한계:** 영상이 매우 매끈하고 대비가 강하며 렌더링처럼 보인다. prompt가 다른 영상끼리도 구도가 비슷해 보인다. 이것은 두 비임상 판독자의 시각 관찰이고, 임상 사실성·질환 conditioning·다양성·memorization을 정량 평가한 결과가 아니다. 정면 흉부 형태가 식별된다는 좁은 기준을 넘었을 뿐이다.

이번 확인은 E4/CFG7.5 하나만 실행했다. CFG1 대조가 없으므로 CFG7.5의 상대 우월성이나 이전 실패를 고친 원인을 주장할 수 없다. 앞선 새 개발64장에서는 CFG1도 잘 나왔다는 사실과 이전 pilot 확인 미통과를 모두 유지한다.

## 4. 수치 검산과 정정

최종 독립 저장 산술 검산은 **5,734항목 PASS**다. 예약seed→실제 CPU latent, 조건 cache와 null/conditional branch 순서,16개 고정 cell,480개 CFG/DDIM 전이,PNG 후처리,checkpoint exact·추론 전후 모델 불변, 기존 source hash를 확인했다. 이는 저장 tensor를 별도 산술로 검산한 것이며 전체 UNet/VAE 재추론이나 생성 품질의 증거 수가 아니다.

최초 검산은 scheduler config 전체 dict 비교에서 실패했다. 달랐던 것은 `_use_default_values`의 항목 순서뿐이었다. 해당 diffusers 구현은 이 목록을 `list(set(...))`으로 생성하므로 프로세스에 따라 직렬화 순서가 달라질 수 있다. 실제 scheduler parameter 값과 목록의 항목은 모두 같았다.

원 코드·[최초 실패 기록](../../code_working/_reports/public_operating_confirmation_20260916_v1/verification_attempt1_failed.json)을 보존하고 별도v2에서 이 metadata 목록만 정렬해 항목과 중복 개수를 대조했다. 나머지 config 값은 exact 비교를 유지했다. 수치 알고리즘·허용오차·영상·입력·모델은 바꾸지 않았고 GPU 재실행도0이다. v2 및 실제 library source hash를 최종 검산 결과에 기록했다.

## 5. 실제 비용

| 항목 | 실측 |
|---|---:|
| 확인 실행 함수 전체 | **63.620초** |
| 16장 sampling/decode 합계 | 45.438초 |
| 장당 sampling/decode 평균 | 2.840초 |
| UNet API calls / batch examples | 480 / 960 |
| VAE decode | 16 |
| 새 text encoding / 학습 / backward | 모두0 |
| Peak allocated GPU memory | 4,218,274,304bytes, 약3.93GiB |
| 최종 저장 산술 검산 | 5.739초 |

실행 함수 전체와 구현·판독·기록을 포함한 사람 기준 작업 시간을 구분한다. 시작 전 전체15–25분, GPU1–2분을 예상했다. 생성 이미지는16장으로 고정했고 추가 seed/예시를 생성하지 않았다.

## 6. 채택 범위와 다음 한 단계

새 [채택 상태](../../code_working/_reports/public_operating_confirmation_20260916_v1/adoption_status.json)는 **후속 비DP 작은 head 적용성 평가만 조건부 승인**한다. 기존 pilot의 adoption=false 파일을 수정하지 않는다. 새 판단은 별도 계약·판정·상태로 기록한다. DP 실행과 임상 사용은 승인하지 않으며, 새 효용 근거가 필요하다.

고정 checkpoint SHA256은 `7bc5ce421cc02091e8c3812676e9c27dc58330afce5e9384abd8edc9f61a3b91`이다. base revision·VAE·scheduler·prompt encoding·CFG·precision·해상도는 이번 계약/실제 입력/실행 packet에 결속돼 있다.

다음은 **새 backbone 위의 비DP head 생성 비교 한 패키지**다. 예상45–75분이며, 첫 소량 추출의 실측으로 계산시간을 갱신한다.

1. 기존 분리된 public32/private80/development40 역할을 유지하고 새 backbone의 특징·base prediction·residual·A/B/Q·witness를 재추출한다. 같은 채널 tap의 데이터 독립 random P만 재사용 가능하다. 기존 generic-base W를 이식하지 않는다.
2. 새 public-only·public+private head를 계산하고, CFG7.5에서 conditional-only correction의 의미·배율과 offline/online 일치를 검산한다. `Delta=0`의 동일 경로 복원도 확인한다. head 학습목표와 guided sampling 출력의 연결을 실행 명세에 먼저 적는다.
3. 고정 backbone / public head / pooled head를 같은 prompt·latent로 실제 생성해 비교한다. 본 패키지의 이미지 수와 평가 범위는 시작 때 고정한다. denoising MSE만으로 생성 효용을 판정하지 않는다.

CFG는 conditional correction뿐 아니라 그 오차와 추후 DP noise의 출력 영향도 확대한다. 이번 형태 통과만으로 small-head 방식이 유리해졌다고 가정하지 않는다. 다음 결과에서도 추가 생성 효용이 없으면 새 DP solver 탐색을 자동으로 붙이지 않는다.

## 7. 재현·판정 파일

- [실행 계약](../../code_working/_reports/public_operating_confirmation_20260916_v1/contract.json), [실제 실행](../../code_working/_reports/public_operating_confirmation_20260916_v1/execution.json), [16장 manifest](../../code_working/_reports/public_operating_confirmation_20260916_v1/manifest.json)
- [root 원표](../../code_working/_reports/public_operating_confirmation_20260916_v1/review_root.json), [별도 독립 원표](../../code_working/_reports/public_operating_confirmation_20260916_v1/review_independent.json), [고정 관문 집계](../../code_working/_reports/public_operating_confirmation_20260916_v1/confirmation_decision.json)
- [수치 검산](../../code_working/_reports/public_operating_confirmation_20260916_v1/verification.json), [생산 코드](../../code_working/public_medical_backbone/confirm_operating.py), [원 검산기](../../code_working/public_medical_backbone/verify_confirmation.py), [정정 검산기v2](../../code_working/public_medical_backbone/verify_confirmation_v2.py)
- [직전64장 개발 결과](TRACK1_PUBLIC_OPERATING_RESULTS_20260916.md), [그대로 보존한 이전 확인 실패](TRACK1_PUBLIC_MEDICAL_BACKBONE_RESULTS_20260916.md)

이번은 로컬 실행·검산·판독 결과다. 사용자가 제공한 원격 main SHA를 별도로 확인하거나 push했다고 주장하지 않는다.
