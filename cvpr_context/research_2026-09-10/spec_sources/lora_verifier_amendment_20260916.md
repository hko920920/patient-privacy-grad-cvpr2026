# LoRA 양성 대조 검산기의 정수 저장 형식 정정

원 실행 계약과 생산 코드, 독립 검산기 및 최초 실패 결과를 그대로 보존한다. 과거 재현 생성은 다시 실행하지 않는다.

원 검산기는 replay packet의 timestep dtype을 int64로 한정했다. 실제 원형 경로 캡처는 Python 정수 목록을 NumPy로 묶으며 이 Windows 환경에서 int32로 저장됐다. 30개 timestep의 값은 계약과 정확히 일치하고, 두 생성 PNG도 과거 PNG와 SHA-256이 각각 정확히 같다. 최초 검산은 127번째 검사에서 이 저장 형식 조건으로 실패했다. 그 실패를 성공으로 덮어쓰지 않는다.

새 `verify_lora_positive_control_v2.py`는 timestep에 한해서 int32/int64를 허용하고 정수 값·범위·일정 전체를 검증한다. 부동소수 텐서의 dtype과 CFG/DDIM/VAE 변환 연산 및 오차 기준은 유지한다. 새 검산 결과와 검산 명세는 v2 파일로 분리한다.

새 `run_lora_positive_control_v2.py`는 이미 생성한 replay를 다시 만들지 않고 bridge만 실행할 수 있다. 수치 생성 코드는 그대로이며, v2 성공 결과를 읽는 경로와 amendment hash 검사만 추가한다. `verification_amendment_v2.json`이 원 계약·원 실패·원 검산기와 새 두 코드의 해시를 결속한다. 원 계약에 결속된 과거 자료도 다시 확인한다.

이 정정은 **최초 재현 결과를 본 뒤, bridge 실행 전**에 이루어졌다. 최초 검산기에서 모두 통과했다고 보고하지 않으며, 이미지를 다시 선택하거나 수치 허용오차를 결과에 맞춰 넓힌 것으로 처리하지 않는다. 실제 v2의 성공 여부는 [최종 결과](../TRACK1_LORA_POSITIVE_CONTROL_RESULTS_20260916.md)에 기록한다.
