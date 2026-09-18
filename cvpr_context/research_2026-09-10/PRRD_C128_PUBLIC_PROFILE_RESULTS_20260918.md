# PRRD 공개 C bank 128장 전체 update 비용 측정

작성일: 2026-09-18. **PASS_PUBLIC_C128_FULL_UPDATE_COST_ONLY**. 요청한 비용 측정만 완료하고 중단했다. 본실험 계수·규모·예산을 확정한 결과가 아니다.

## 측정 결과

RTX 3070 8GiB에서 공개 P로 초기화한 **C bank 128장 하나**, microbatch4로 실행했다. Warm-up5회 후 측정10회, 총15회의 측정용 optimizer 갱신이다.

| 항목 | 실제 값 |
|---|---:|
| 128장 전체 update 평균 | **7.8364초** |
| 중앙값 | 7.8321초 |
| 최소–최대 | 7.7720–7.9343초 |
| 표준편차(측정10회 기술 통계) | 0.0505초 |
| GPU peak allocated | **1,732.42MiB (1.692GiB)** |
| GPU peak reserved | **1,852.00MiB (1.809GiB)** |
| 유한 손실·gradient·optimizer 상태 | 15/15회 통과 |
| 실제 합성 parameter 갱신 | 15/15회 확인 |
| 고정 encoder 가중치·buffer·projection | 검사 전후 hash 동일 |
| 고정 target 통계·행렬·분류기 | 검사 전후 hash 동일 |
| OOM·NaN·실행 실패 | 없음 |

시간은 CUDA 동기화 후 **128장 전체 특징 계산 → 전역 통계·관계 제한·기능 손실 → two-pass 역전파 → AdamW 갱신 완료**까지 잰 wall time이다. 일부 microbatch 시간의 환산값이 아니다. 매 update의 모델 접근량은512 forward/256 backward image presentations이며, microbatch4 기준128/64 model calls임을 실행 중 검사했다. 측정용 counter와 기존 목적 코드의 유한성 검사 overhead도 포함된다.

모델/목표 준비5.510초, template decode·GPU 전송1.272초, 별도 parameter/state 검증2.658초, progress IO0.017초는 update 시간에서 제외해 따로 기록했다. Warm-up 포함15회 update 합계119.231초, setup 이후 runner 전체129.367초다. Peak 수치는 CUDA allocator 기준이며 reserved는 캐시를 포함한다. 다른 arm이나500회 학습 비용을 이 결과에서 측정했다고 하지 않는다.

## 실행 전에 고정한 범위

[사전 계약](../../code_working/_reports/prrd_c128_profile_20260918_v1/run/profile_contract.json)에 microbatch4, warm-up5/측정10, seed101,128장, 전체 계산 경계와 source/input hash를 GPU 실행 전에 기록했다. 이전 문서의20/30회는 과거 전체 profile 제안이고, 이번 단일 C 비용 측정은 별도의5/10회 계약이다. Microbatch 탐색·실패 후 자동 재시도는 하지 않았다.

- 목적: 3단계에서 검증한 `guarded_objectives.two_pass`를 변경 없이 재사용. Clean 기능 정합1회, 증강 통계 정합, 기존 영상 규제 유지.
- 목표: 2단계의 P-only 환자 가중 점별 통계·mixed 관계 통계. Q 통계는 쓰지 않았다.
- 수치 검사용 계수: beta1, ridge.1, kappa.1의 상대 반경, eta1. 성능을 보고 선정하거나 본실험 값으로 동결하지 않았다.
- Optimizer: AdamW lr.01, wd0, betas(.9,.999), eps1e-8, foreach=false. FP32 영상/encoder, 기존 FP64 작은 통계·readout.
- Renderer: 8/16/32/64/112/224의 여섯 해상도를 모두 활성화해 비용을 측정했다. 활성화 설정500 및 증강 nonce500–514는 **학습500회**를 뜻하지 않는다. 실제 갱신은15회다.
- 초기 bank:64 marginal images+32 virtual pairs, 총128slot. 기존 공개 patient-hash 정책을 그대로 사용했고 고유 P영상64장을 읽어 반복 template을 구성했다.

추가한 코드는 [profile_guarded_c128.py](../../code_working/prrd_v3/profile_guarded_c128.py) 한 개다. 기존 objective/encoder/renderer/learner를 수정하지 않았으며3단계에 결속한 모든 기존 source SHA를 실행 전에 대조했다.

## 실행 이상 확인과 보존

매 갱신마다 모든 활성 parameter의 gradient가 존재하고 유한한지, parameter와 Adam 상태가 유한한지, 실제 parameter 값과 optimizer step counter가 변했는지 확인했다. 최소 parameter 변경 최대값은0.0097326으로,15회 모두 실제 갱신이 있었다. 이는 학습 효과의 지표가 아니라 실행 확인이다. Source parameter에는 gradient가 생기지 않았고 전체 weights/buffers/projections hash는 동일했다.

총1,920F/960B model calls,7,680F/3,840B image presentations. P-only allow-list에서 고유 공개64장 decode64회였다. Q/V pixel, 수신 모델, DP, expert/reserved, 다른 arm, 본500회 학습은0이다. 기존 Rwide/LoRA 효용 및 gradient 수리/2·3단계 기록을 유지했다.

최종 학습 parameter나 PNG를 export하지 않았다. 저장된 것은 사전 계약·template metadata·매회 비용/검증 로그·상태 hash·결과 JSON이며 `PROFILE_ONLY_DO_NOT_USE_AS_MAIN_BANK` 표식을 남겼다. 따라서 측정 결과물을 본실험 bank로 재사용하지 않는다.

[전체 결과 JSON](../../code_working/_reports/prrd_c128_profile_20260918_v1/run/full128_profile.json), [매회 로그](../../code_working/_reports/prrd_c128_profile_20260918_v1/run/progress.jsonl).

재현 명령(`code_working`, 기존 결과를 덮어쓰지 않는 새 출력 경로):

```powershell
& ./base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.profile_guarded_c128 --output <new_profile_directory>
```

**완료 범위는128장짜리 실제 계산이 돌아가며 비용을 측정했다는 것**이다. 계수·본실험 규모·예산 및 전체 runtime freeze는 여전히 미확정이다. 결과 기록 후 멈췄으며 추가 실행을 예약하지 않았다.
