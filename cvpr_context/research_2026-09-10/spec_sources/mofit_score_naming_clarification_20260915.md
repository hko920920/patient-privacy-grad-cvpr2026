# MoFit 보조 점수 명칭 정정 — 계산식 변경 없음

2026-09-15. 추가 E006/U004 실행 중, 이미 동결된 분석기와 계약의 수식 표기 혼동을 발견했다. 실행 코드·계약·원시 자료는 변경하지 않고 이 정정을 함께 읽는다. 실제 산술은 아래 코드 식대로이며 오류는 `L_VLM`이라는 기호를 논문과 다르게 사용한 명칭에 있다.

| 동결 machine score | 실제 계산과 정확한 의미 |
|---|---|
| `h` | `h=L(original, 최적화 embedding)−u`; MoFit 식7의 core score |
| `negative_u` | `−u=−L(original, null)`; 식9에서 허용하는 보조항 중 하나 |
| `negative_v` | `−v=−[L(original, 초기 VLM embedding)−u]`; 논문 식8의 **−L_VLM**, 식9에서 허용하는 다른 보조항 |
| `negative_u_plus_v` | `−(u+v)=−L(original, 초기 VLM embedding)`; **VLM 조건부 원시 denoising loss의 음수**. 식8의 −L_VLM과 다르며 별도 기본 진단 |

동결 분석 계약의 `negative_u_plus_v` 설명 중 `=−L_VLM`, 그리고 `negative_v` 설명의 `L_VLM−L_null`은 `L_VLM`을 원시 조건부 loss 이름으로 썼다. 논문 식8의 기호와 혼동되므로 위 표로 정정한다. 분석기의 실제 `−v`와 `−(u+v)` 연산, 부호 적용 후 mean/max, score ID, 사전 선언한 출력 개수는 그대로다. 숫자를 다시 고르거나 Eq9 fusion을 계산한 것으로 바꾸지 않는다.

- 동결 분석기 SHA256: `aa97d16f3e85c844cb6d411cbcb8a79d76a6ef247be3e6652afff7d85cf1080e`
- 동결 분석 계약 SHA256: `15001af7879637bf82f432c54084691f04244f4746f2135d362c180c821ffce0`
- 추가 GPU 계약 SHA256: `55d9526e08755c3c6dde788d641b6a304f56ab83b2ac326b07bd6e7367c38aab`
- [논문 식7–9](../pdfs/X12.pdf#page=6)

최종 표와 후속 설명에서는 위 정확한 한글 명칭·`u/h/v` 계산식을 사용한다. 원래 계약의 표기 오류를 지우거나, 점수 정의를 실험 후 바꾼 것으로 기록하지 않는다.
