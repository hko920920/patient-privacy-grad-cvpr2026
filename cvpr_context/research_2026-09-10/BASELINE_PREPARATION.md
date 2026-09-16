# 비교 구현 준비 상태

2026-09-10. 이 문서는 문헌 대조 후의 구현 연결표다. 코드 다운로드와 실제 실행을 구분한다. 정확한 성공/실패 및 commit은 [코드 장부](code_acquisition.json)에 있다.

| 방법 | 확인한 공식 진입점/자료 | 지금 된 것 | 의료 환자 비교까지 남은 작업 |
|---|---|---|---|
| CDI | `main.py`, `src/attacks/features_extraction`, `src/evaluation` | source 확보, feature/score/evaluation 단계 분리 확인 | SD2.1 wrapper와 bag split, reference null 반복 평가 |
| CLiD | NeurIPS supplement `CLID_MIA`, `cal_clid_th.py` | COCO 공개 점수 재계산, 양성 방향/첫 행 차이 확인 | 짧은 NIH class prompt 변형과 feature 추출 adapter |
| Tracing | NeurIPS supplement `train.py`, `eval.py`, `utils.py` | 공개 CIFAR feature linear probe 3설정 재계산 | LoRA gradient feature 추출 및 module/timestep 고정 |
| GSA | `DDPM/gen_l2_gradients_ddpm.py`, `Imagen/gen_l2_gradients_imagen.py`, `test_attack_accuracy.py` | 공식 source 및 1/2 variant 진입점 확인 | SD2.1 LoRA gradient 연결, shadow 예산 통제 |
| DP-LoRA | `main.py`, `ldm/models/diffusion/ddpm.py` | config 기반 LoRA 및 Opacus 연결 확인 | 현재 sampler와 patient 평균·clipping 회계 검증 |
| RAPID | `main.py`, `train_feature_extractor.py`, `conditional_sampling.py` | 공개 KB/feature extractor/private sampler 연결 확인 | domain-matched public KB와 private endpoint patient bound |
| DPImageBench | `run.py`, `eval.py`, `models` | 저자 repository source 확보 | 공통 backbone/validation 조건을 로컬 contract에 대조 |
| DP-FETA | [저자 코드](https://github.com/SunnierLee/DP-FETA) | source 확보 | prototype release와 patient gradient 양쪽 비용 |
| FETA-Pro | [저자 코드](https://github.com/2019ChenGong/Feta-Pro) | source 수집·무결성 대조 상태는 장부 참조 | random feature query·central query·fine-tuning 합성 |
| PFAMI | [저자 코드](https://github.com/wjfu99/MIA-Gen) | source 확보 | gray-box query wrapper, 동일 crop·MC 수·score 부호 |
| MoFit | [저자 코드](https://github.com/JoonsungJeon/MoFit), `COCO/MoFit_COCO.py`, `Eval` | 공식 repo와 pipeline 구조 확인; 수집 상태는 장부 참조 | 별도 embedding 최적화 환경, 시간 예산과 독립 calibration |

DP-LoRA `ddpm.py`의 LoRA 설정은 `r`, target modules, `bias='lora_only'`를 사용한다. 이름이 같은 LoRA라도 로컬 구현에서 bias 및 conditioning module이 다르면 parameter 예산이 달라진다. 또한 DP-LoRA/RAPID는 `make_private`에 넘긴 loader와 실제 `main.py`에서 재구성한 loader의 일치를 점검해야 한다고 source 주석이 설명한다. noise multiplier만 복사해서 patient DP가 됐다고 하지 않는다.

미확보 공식 구현은 재현 완료로 표시하지 않는다. Quantile, DIME, SD-MIA 및 NDSS/ReDiffuse의 이식은 문헌·공식 공개 상태에 따라 별도 작업이다. 코드가 없거나 접근 조건이 다르면 그 사실과 재구현 여부를 비교표에 명시한다. 모든 소스의 의존성을 한 환경에 설치하지 않았고 대형 checkpoint 학습도 실행하지 않았다.

공통 평가 도구는 다음처럼 호출한다. 아래 CSV는 사용자가 직접 제공해야 한다는 뜻이 아니라 공식/로컬 feature 추출기가 출력할 공통 규격이다.

```text
python evaluate_patient_scores.py scores.csv --score-direction high --aggregation mean --alpha 0.01 --out evaluation.json
```

CSV columns: `patient_id,image_id,split,member,score`. 하나의 고정 model/attack 파일에서 환자 전체가 한 split에만 속해야 한다. `split`은 `development`, `calibration`, `test`; `member=1`이 학습 포함 환자다. score 방향을 바꾸는 기준은 개발자료에서 고정하며 test AUC를 보고 뒤집지 않는다. 동일 환자의 알려진 training 사진 공격과 학습에 없던 다른 사진 공격은 별도 시나리오 파일로 만든다.

평가기는 표준 집계/보정을 구현한 **기준선 도구**다. 신규 공격 기여로 세지 않는다. 서로 다른 bag 크기의 marginal calibration과 각 크기의 conditional calibration도 같은 보장이 아니므로 subgroup 결과를 함께 출력한다. Bootstrap/기관별 strata/feature extraction adapter는 아직 구현하지 않았다. [전체 기여도 판정](REVIEW_REPORT.md)
