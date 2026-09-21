# PRRD 공개 microbatch 속도 비교 — 2026-09-21

완료 범위는 공개 P 입력의 수치 대응·전체 update 속도 비교와 실행 설정 결속이다. 환자 효용, Q/V 특징 추출, 본 bank, 수신자, DP, Expert/Reserved 실행은 없다.

**같은 128장·FP32·목적·optimizer에서 microbatch16이 가장 빨랐다.** 이번 microbatch4 대비 평균 시간은 약 26% 줄었다. 다음 본실행에는 별도 runtime amendment로16을 결속하며, 기존 원계약·과거 결과는 보존한다.

| 설정 | 전체 update 평균 | 중앙값 | 측정 범위 | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| 기존 microbatch4 | 8.079초 | 8.016초 | 7.984–8.212초 | 1.730GiB | 1.861GiB |
| 기존 microbatch8 | 7.216초 | 7.223초 | 7.185–7.241초 | 3.045GiB | 3.246GiB |
| 기존 microbatch16 | **5.982초** | **5.974초** | 5.957–6.008초 | **5.633GiB** | **6.188GiB** |
| microbatch8＋로그용 CPU 동기화 감소 | 7.251초 | 7.230초 | 7.209–7.316초 | 3.045GiB | 3.246GiB |

RTX3070 8GB, PyTorch2.6.0+cu124. 설정당 warm-up3＋측정5회, 매번 같은 공개 초기 상태·증강 stream에서 시작했다. 128장 전체 clean/augmented 특징 추출, 전역 통계·guarded learner·기능 loss, two-pass 역전파와 AdamW 완료까지 CUDA synchronize로 잰 값이다. 모든6 pyramid level을 활성화한 비용 검사이며 500회 학습을 수행한 것이 아니다. 작은 통계/readout FP64, encoder/renderer FP32, AMP/TF32 off를 유지했다.

총32회의 공개 profile update가 유한한 loss/gradient와 실제 parameter 변경을 보였다. 마지막 parameter/PNG를 본실험에 재사용하지 않는다. 각 update는 encoder image presentation 기준512 forward/256 backward를 수행한다. 공개64개 고유 template을 사용했다.

## 수치 검증과 검사 정책 정정

A/C/R_joint × microbatch4/8/16의 **동일 분할 one-pass/two-pass 9조건이 모두 기존 기준을 통과**했다. 기준은 loss 절대차2e-6, gradient 최대 절대차2e-7＋5e-4×reference peak이며 바꾸지 않았다.

실제 최대 parameter-gradient 차이는1.1921e-7, 최대 상대 L2 차이는2.1056e-7, encoder 입력 pixel-gradient 차이는0, loss 차이는1.1369e-13이었다. 기준 계산은 encoder activation checkpointing으로 전체 목적의 graph를 유지했고, 시간 측정은 기존 production two-pass를 사용했다. Source 가중치·buffer·projection은 불변이다.

**최초 검사 스크립트의 추가 조건을 명시적으로 정정한다.** 이 스크립트는 동일 분할 대응 기준을 서로 다른 분할의 gradient 비교에도 적용하여16을 자동 제외하고8을 임시 선택했다. A의4대16 gradient 최대차4.2421e-4, 상대 L2 차이0.0011767은 그 추가 조건을 넘었다. 이 기록과 최초 선택은 삭제하지 않았다.

사용자가 요구한 것은 새 분할 안에서의 one-pass/two-pass 대응이며 서로 다른 분할의 bitwise 동일성이 아니다. 따라서 서로 다른 분할의 gradient 차이를 별도의 새 합격 관문으로 만들지 않고,16에서도 C/R_joint의 동일 분할 대응과 실제 C128 전체 update를 마저 확인했다. 최종16 선택은 **측정 후 이루어진 검사 적용 범위 정정**을 포함한다. 기존 수치 허용오차를 올린 것은 아니다.4와16의 gradient·학습 경로가 같다고 주장하지 않으며500회 후 산출물의 동일성도 검증하지 않았다.

처음 시도한 CPU-offload reference는 host free RAM이 약0.7GiB까지 줄어 시간 상한을 지키기 위해 중단했다. A4/C4의 완료 검사와 추가 진행 중 계산을 기록하고, 검증용 reference만 activation checkpointing으로 바꿨다. 이 중단은 새 공개 검사 범위 안의 기술적 시도이며, 이전 요청 밖에서 중단된12-update budget diagnostic과는 별개다. 이전 diagnostic을 재개하거나 이용하지 않았다.

## 동기화 후보

no-grad 수집 중 매 microbatch마다 regularization을 Python float로 가져오는 대신 GPU FP64로 합산 후 한 번 가져오는 후보를 별도 검사했다. 같은 microbatch8에서 gradient와 loss 차이는0이었다. 평균 시간7.251초로 기존7.216초보다 나아지지 않아 **채택하지 않았다**. 해당 동기화가 주된 병목이라고 결론내리지 않는다. guarded_objectives.py는 수정하지 않았다.

## 실행 반영과 비용

원계약 PRRD_FIRST_NONDP_EXECUTION_CONTRACT_20260918.md의 SHA는 그대로다. 추가 계약은 [runtime amendment](spec_sources/prrd_runtime_microbatch_amendment_20260921.json)이다. microbatch만16으로 바꾸고128장·500회·계수·비교군·seed·FP32는 유지한다.

fit_banks.py는 결속된 job의 microbatch를 run_spec에 기록해 실행하고, run.py는 SHA로 결속된 추가 계약·기존 재개 증거·새 속도 증거에서 그 값을 전달한다. 기존 checkpoint에 다른 microbatch를 덮어씌우지 않는다. 실행 권한 없는 main 요청의 차단과16/128/500 결속은 **메타데이터만으로 검사**했고, 승인 파일이나 본 bank를 만들지 않았다.

기존 checkpoint/export kernel은 그대로이고 CPU I/O·재개 검사6개도 통과했다. 기존 실제 프로세스 종료 후 재개·PNG 증거는 상속한다. 이번에는 microbatch16에서 별도의 프로세스 종료/재개 실험을 새로 했다고 표시하지 않는다. 새 receipt 상태도 PASS_PUBLIC_MICROBATCH_WITH_INHERITED_RESUME로 이를 구분한다.

새 속도의 단순 환산은 다음과 같다.

- C bank500회: 약49.85분.
- 30bank 합성15,000회: 약24.92시간.
- 이전7.836초 실측 환산32.65시간보다 약7.73시간 짧고, 이번 동일 세션 microbatch4 환산33.66시간보다 약8.74시간 짧다.

이는 공개 C 비용에서 얻은 **합성 계산 추정**이다. 다른 arm별 차이, Q 추출, 저장·PNG, 수신자·ResNet 평가 등을 합친 총시간이 아니다. 500회의 충분성이나 방법 효용을 새로 검증한 결과도 아니다.

근거: code_working/_reports/prrd_microbatch_speed_20260921_v1 아래 predeclared contract, 각 parity/profile, final_summary.json, runtime_binding_checks.json, runtime_verification_extension.json. 전체 실행량은 공개 profile32회, Q/V pixels0, main banks0, recipient0, DP0, Expert/Reserved0이다. 현재 GPU 실행은 끝났으며 본실험 허용 bank와 숫자 시간 상한은 별도로 결속해야 한다. 기존 Rwide/LoRA 효용 기록을 바꾸지 않았다.

