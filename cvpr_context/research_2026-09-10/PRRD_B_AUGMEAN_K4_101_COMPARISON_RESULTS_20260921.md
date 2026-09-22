# B 증강 평균 target 한 bank 비교 결과 — 2026-09-21

**요청한 단일 학습→최종PNG→비교 평가를 완료했다.** DenseNet AUROC는 기존B 대비 -0.080594(감소), AP는 -0.013875(감소)다. 사전 개발 투자 기준은 미충족이다. AUROC 차이의 환자 조건부95% 구간은0을 포함하지 않는다. 이는 고정seed101·재사용 개발V의 비교이며 독립 확인이나 seed 반복성, 사적 추가 효용 또는 논문 기여 확정이 아니다.

## 1. 결과와 실제 비용

점수는 마지막500회 PNG로 학습한 고정 readout의 개발V image-level AUROC/AP다. 차이 행의 대괄호는 기존2,000개 환자 cluster draw를 공유한 paired percentile95% 구간이다.

|조건|BioViL AUROC|BioViL AP|DenseNet AUROC|DenseNet AP|합성·최종 저장 실측|
|---|---:|---:|---:|---:|---:|
|기존 B|0.777578|0.127708|0.567975|0.060098|49.33분|
|수정 B_augmean_k4|0.777800|0.128531|0.487381|0.046223|49.39분|
|수정−기존|+0.000222 [-0.000126, +0.000566]|+0.000823 [-0.001571, +0.001898]|-0.080594 [-0.131770, -0.029003]|-0.013875 [-0.028148, -0.001806]|+0.06분|
|DenseNet 공개 실자료 기준|—|—|0.618009|0.064941|기존 결과 재사용|
|수정−공개 실자료 기준|—|—|-0.130628 [-0.190940, -0.069021]|-0.018718 [-0.034503, -0.004871]|—|

공개 실자료 readout은 비교 기준이지 수학적 상한이 아니다. Source 목표 통계의 AUROC/AP는 0.777791 / 0.127907로 기존과 동일하다. Source의 학습기능 재현 수치와 DenseNet의 실제 효용을 구분한다.

## 2. 실행한 한 가지 변경

- Run: `dev_B_augmean_k4_101`, 새 공개 초기화,128장·500 successful updates·seed101·FP32·microbatch16.
- 기존 B의 checkpoint에서 이어 학습하지 않았다. 완료 후 초기 renderer/optimizer/RNG 상태를 기존 B의 보존된 step0과 대조해 모두 exact 일치했다. Template와 source model/projection hash도 동일하다.
- 변경은 증강 통계항의 target을 기존 clean 통계에서 이미 준비한 P/Q K4 실자료 증강 평균 통계로 바꾼 것뿐이다. Clean target·목표 readout·기능 정합·PCA/scale·증강 범위·pixel/TV·optimizer·계수는 그대로다.
- B의 beta0/ridge0.1/eta1, 기존 guard 설정과 pyramid schedule을 유지했다. 새 계수 선택·profile·추가 seed/arm은 없다.
- 이것은 유한 증강 **평균** 정합이다. 변환별 real/synthetic 반응 일대일 정합이나 LGM/Dosser 재현을 의미하지 않는다.

## 3. 완료·평가 정합성

- 500회 성공 이후 최종224×224 grayscale uint8 PNG128장을 저장했다. Label·빈 pair 목록·파일 hash·float→uint8 변환 검증을 통과했다.
- 학습 중 V/수신자 성능을 계산하지 않았다. 최종PNG seal 이후 BioViL128회 및 DenseNet128회 forward만 추가했고, 기존 V 특징을 그대로 재사용했다. Q/V 원영상 신규 접근0.
- 기존 V5,047장/2,026명의 image ID·patient ID·label 순서를 source/recipient/기존예측 간 대조했다. 기존 bootstrap2000배열과 기존 baseline 예측/구간은 재사용했다.
- 새 두 분류기의 bootstrap 산술은 가중 sklearn 지표로 독립 대조했다. 최대 오차 1.11022302463e-16 (기준1e-12).
- Clean 목표 분류기 가중치의 기존 대비 최대 차이: 0. PNG 기능 정합 오차: 9.64056208967e-06. 이 오차만으로 전이 성공을 판정하지 않았다.
- 모델 parameters/buffers와 고정 target 불변 확인. Production 연구 코드 수정0. 실행 계약/입력 해시는 학습과 평가 후 재확인했다.

## 4. 비용과 저장

- 새 합성 image-forward 256,000, image-backward 128,000; optimizer500회.
- 새 PNG 평가 forward: BioViL128＋DenseNet128. 새V feature 추출0. 기존B 재학습0.
- 합성·checkpoint·최종PNG 저장: 2963.422초 (49.39분).
- 최종 평가·검산: 5.640초 (0.09분).
- 실행 campaign 전체: 2980.423초 (49.67분). 계약/평가 wrapper 연결과 보고서까지 이번 작업 경과 약54.7분.
- 앞 단계 K4 목표 준비 비용은 별도: 추출5.82분, 프로그램7.11분·23,640 source forward. 이번에 재추출하지 않았다. 전체방법 비용 계산에서는 이 일회 준비도 포함해야 한다.
- 이번 새 bank의 중복 중간 checkpoint payload 18개/1.666GiB를 정해둔 보관 규칙으로 정리했다. 초기·최근2개/최종, 모든receipt·업데이트trace·PNG는 보존했다. 기존 B/과거실험·원자료는 삭제하지 않았다.
- 현재 이번 폴더 저장량 0.376GiB, 디스크 여유 11.489GiB.

## 5. 해시와 재현물

- 실행 계약: `code_working/_reports/prrd_b_augmean_k4_101_20260921_v1/campaign_contract.json`와 `execution_contract.md`.
- Job: 같은 폴더의 `job_private.json`; clean target SHA `9fff657c5f2868b24f5097542dcf18bb0314ff3d70c5600735c8e38429baade7`; augmented target SHA `538f3fcd3474c6f3b91a3503ce174846636b82a3e9f71aac433423b6219c8054`; rule SHA `d852cb9e9f2f55a95f915b93d43cf98efa7cc9a18f025e7129525abcc02c92f4`.
- 최종 bank: `banks/dev_B_augmean_k4_101/`; `final_bank_seal.json`의 artifact SHA `cbc09cee1c70270017d90de1b95d03db085bbb0ecb839bf924f04a7e65fc7701`.
- 평가: `evaluation/result.json`, 내부예측 `predictions_private.npz`, `paired_bootstrap.npz`, source/recipient PNG 특징.
- 실행 실제 receipt: `training_receipt_*.json`, `campaign_result.json`, `storage_retention.jsonl`.
- 환자자료 기반 비DP 요약·매핑·예측은 내부 산출물이며 공개 privacy receipt로 취급하지 않는다. 원격 업로드 없음.

## 6. 현재 판단과 종료 범위

DenseNet AUROC는 기존B 대비 -0.080594(감소), AP는 -0.013875(감소)다. 사전 개발 투자 기준은 미충족이다. AUROC 차이의 환자 조건부95% 구간은0을 포함하지 않는다. 이는 고정seed101·재사용 개발V의 비교이며 독립 확인이나 seed 반복성, 사적 추가 효용 또는 논문 기여 확정이 아니다.

사전 투자 기준은 DenseNet ΔAUROC≥0.01과 AP 비감소다. Source 정합/이미지 모양/평균특징 차이로 대신 판정하지 않았다. 이번 결과로 평가한 것은 증강평균 target 변경 하나이며, 관계 중심 PRRD의 가치를 되살리는 결과로 해석하지 않는다.

수정 A 비교와 반복을 하지 않았으므로 사적 추가 효용·반복성은 미확인이다. DenseNet과V는 이미 개발 선택에 사용된 대상이다. 추가seed·A/C/D·RN18·ViT·DP·Expert·Reserved는 실행하지 않았고 다음 run도 자동 예약하지 않았다. 기존 음성결과는 보존하며 **이번 요청한 한 bank 비교에서 종료**한다.
