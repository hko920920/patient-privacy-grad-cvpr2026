# PRRD 실행 계획 v3

이 묶음은 사용자의 저장소 결과, 기존 PRRD v2 설계, 최신 Codex 정정을 연결한 **실행 전 연구 계획**입니다. 원격 저장소에 업로드하거나 새 환자 모델을 실행한 산출물이 아닙니다.

## 파일

- `PRRD_MASTER_EXECUTION_PLAN_V3_20260918.md`: 자료, 수식, 비교, DP, 전이, 최종 평가, 논문 구성의 통합 계획.
- `experiment_contract_template.yaml`: 이번에 제안하는 구체 실행값. null 값은 로컬에서 채우기 전까지 실행 계약이 동결되지 않았음을 뜻합니다.
- `planned_run_matrix.json`: 비DP15bank, 조건부DP54bank의 계획. 실행 결과가 아닙니다.
- `CODEX_NEXT_PACKAGE.md`: 다음 구현 패키지의 작업 지시와 인수 기준.
- `reference_math.py`: 환자 데이터 없는 NumPy/SciPy/Torch 대수·미분 검산.
- `reference_checks.json`: 위 코드를 실제 실행한 결과. 의료 효용이나 production privacy를 인증하지 않습니다.
- `build_manifest.py`: YAML과 run matrix를 생성하는 코드.
- `SHA256SUMS.txt`: 배포 묶음의 파일 무결성 목록.

## 참고 코드 실행

```bash
python reference_math.py --output my_reference_checks.json
```

NumPy, SciPy, PyTorch가 필요합니다. 어떤 network download, private data read, 학습 모델 forward도 없습니다. 생성된 배열에서만 계산합니다.

## 변경 요약

주력은 PRRD로 유지합니다. A/B/C/D와 다른 encoder의 관계 활용, 직접 통계 분류기를 묶습니다. A/B/C/D 제거실험만으로 기존 방법 대비 신규성을 판정하지 않도록, 표준 endpoint 공동모멘트 기반 `R_joint`를 이번 계획의 추가 baseline으로 명시했습니다. 같은 차분 연산을 다른 알고리즘인 척 다시 세지 않습니다.

첫 핵심 수신자는 이번 권고값으로 DenseNet121 ImageNet1K V1입니다. 이전 기록에서 이미 승인되거나 질환 성능이 검증된 모델이라는 뜻은 아닙니다. 정확한 checkpoint와 전처리는 공개 입력으로 먼저 결속합니다. 모르는 로컬 GPU 속도를 가정한 시간 보장은 없습니다.

## 유지되는 경계

기존 head/LoRA의 실패는 그대로 보존합니다. Qwide는 공개자료로 바꾸지 않습니다. 이미 사용한 개발자료는 새 독립 자료가 아닙니다. DP, expert final, reserved는 별도 동결·승인 전까지 닫습니다. 이 계획은 채택이나 성능을 보장하지 않고, 무엇을 실험하면 어떤 논문 주장을 책임질 수 있는지 정합니다.
