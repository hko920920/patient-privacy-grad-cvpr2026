# 2번 — CDI 26개 특징의 실제 모델 계산·검산

완료: 2026-09-15 00:08 KST. 연구의 [고정 네 단계](RESEARCH_FRAMEWORK.md) 중 **2번의 구현·비용 확인**이다. 한 환자 자료로 기존 방법의 성능이나 실패 원인을 확정하지 않았다. 현재 다음 작업은 [U 전체 비교 준비](PATIENT_BASELINE_SPEC.md)다.

## 실제 완료한 범위

기존 fit14393의 E2/U2, 고정 model1 step1000에서 CDI 공개 코드의 26개 특징을 모두 계산했다. E는 각4회 학습 노출, U는0회임을 checkpoint exposure와 학습 목록으로 확인했다. P256 mode latent·generic hidden cache를 FP32로 올리고, 모델 가중치를 동결한 채 GM/NO 입력 미분만 사용했다.

소스와 현재 모델을 연결한 방식은 6개 extractor, General wrapper의 noise/use_grad 분기, SecMI·DL·PIA scalar의 원본 AST를 재사용한 것이다. class-conditioned CompVis 전체 재현, 원형 CDI의 집합별 reference 검정, 환자 scorer fitting과 구별한다.

| 영상 역할 | UNet F | 입력 B | 계산 시간 |
|---|---:|---:|---:|
| E 첫 영상 | 59 | 18 | 8.116초 |
| E 둘째 영상 | 57 | 16 | 5.588초 |
| U 첫 영상 | 58 | 17 | 5.715초 |
| U 둘째 영상 | 58 | 17 | 5.790초 |
| **합계** | **232** | **68** | **측정 구간25.238초** |

전체 실행기29.724초, peak CUDA allocated 4,096,370,688 bytes, reserved 4,596,957,184 bytes. 장비 RTX3070 8GB. 첫 영상에는 cold-start 비용이 포함돼 있다. 이후 영상 실측5.6–5.8초는 이번 한 환자의 범위이며 다른 환자의 NO line search 횟수·입출력 비용이 달라질 수 있다.

VAE/text encoder 호출은0이다. 이는 기존 입력 cache를 재사용했기 때문이며 독립 공격의 encoding 비용이 공짜라는 뜻이 아니다. 현재 설치된 diffusers의 checkpoint 재계산 분기는 training일 때 작동하고, 이번 eval 경로에서는 사용되지 않았다.

## 독립 검산 결과

- 4×26=104개 특징을 저장된 원시 예측·noise·mask·delta에서 CPU float64로 별도 재계산했다. 최대 절대차이2.97065e−6, 상대차이1.68013e−7.
- 기존 SecMI 실행의 같은 네 영상 z_det/z_recon이 **모두 정확히 일치**했다.
- CPU noise32개를 고정 image/module/stream seed에서 독립 재생성하여 값·hash를 확인했다.
- GM의 top20% gradient mask와 clean-latent 교체, NO의 objective 좌표와 최종 추가 noising을 각각 확인했다.
- 동결 파일40개 hash가 일치했고, producer의 모델 parameter version·weight gradient 부재와 LoRA tensor의 실행 전후 동일성 기록을 확인했다.
- 독립 검산 내부 CPU 시간1.169초. 저장된 UNet 입력 gradient의 전체 Jacobian을 새 GPU에서 다시 계산한 것은 아니다.

검산기의 NO 비영점 요구는 일반적인 공격 성능 판정 기준이 아니다. 이번 입력의 gradient는 실제 비영점이라 이 범위 차이가 결과에 영향을 주지 않았다.

## NO에서 구분해야 할 두 값

| 영상 역할 | 초기 objective | 최종 objective | 공개 코드가 내보낸 final feature |
|---|---:|---:|---:|
| E 첫 영상 | 37.03 | 19.24 | 36.05 |
| E 둘째 영상 | 35.66 | 20.20 | 35.34 |
| U 첫 영상 | 36.32 | 19.02 | 35.85 |
| U 둘째 영상 | 36.40 | 18.31 | 34.43 |

공개 코드의 최종 feature는 최적화 입력에 다시 noise를 적용해 평가하므로 최적화 objective와 다르다. 실제 네 장에서도 이 차이가 나타났다. 이것이 membership 성능을 악화시킨다는 증거는 아직 아니다. 이후 강한 대조에 objective endpoint 정보를 추가한 표준 결합 변형을 따로 두어 이 설명을 확인한다. 작은 구현 수정 효과를 새 공격 기여라고 하지 않는다.

NO는 모두 설정된5 iterations에 도달해 status1/success=False로 끝났다. 목적값은 감소했지만 수렴 성공을 뜻하지 않는다. 실제 objective 호출은8/6/7/7회였다. 앞선 합성 CPU 검사에서는 SciPy nfev43 대 실제42처럼 MemoizeJac 캐시 차이도 확인했다. 비용의 q는 실제 계산 횟수이며 nfev를 그대로 forward로 세지 않는다.

## 다음 비교와 시간

**U fit80+selection40, 두 target, 환자당2장 =480개 image-target records.** 기존 kernel의 model1 U2개는 같은 입력·코드·noise가 확인되므로 재사용하고478개를 새로 계산한다. 학습형 CDI 점수와 표준 환자 집계를 같은 fit 정보로 비교하고, 이미 관측된 selection의 개발 결과로 보고한다. calibration/test는 사용하지 않는다.

현재 실측을 단순 적용하면 새 추출 약45–55분이다. 새 실행기·고정된 분석 계약·검산·분석까지 추가60–75분으로 안내했다. 실제 NO 횟수와 시간에 따라 갱신한다. 이 비교에서도 약한 결과만으로 연구 불가능이나 2번 완료를 선언하지 않는다.

## 원시 기록과 코드

- [실행 계약](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_kernel_contract_v1.json)
- [실행 결과](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_kernel_v1/execution.json)
- [104개 특징과 모듈별 기록](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_kernel_v1/results.json)
- [독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_kernel_v1/verification.json)
- [원시 tensor packet](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_kernel_v1/raw.pt)
- [source adapter](../../code_working/u_patient_audit/cdi_adapter.py), [실행기](../../code_working/u_patient_audit/run_cdi_kernel.py), [독립 검산기](../../code_working/u_patient_audit/verify_cdi_kernel.py)

adapter SHA256 70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce. 독립 검산기 SHA256 57b7719c54fee4c32c2d92e29535ce44aceb5b784669b4be2fff43aa6dd7d342. 실행 당시 코드·입력은 이후 작업에서도 보존한다.

