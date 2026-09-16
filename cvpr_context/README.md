# CVPR 주제 탐색

이 폴더는 CVPR 2026의 **Personalized Federated Training of Diffusion Models with
Privacy Guarantees**와 현재 박사학위논문 작업을 연결해 독립 논문 주제를 탐색한 기록을
학위논문 본문·동결 실험계약과 분리하여 보존한다.

전체 작업의 현재 상태와 B0 시간순서 정정은 `../CURRENT_STATUS.md`가 우선한다. 이 폴더의 최신
선행연구·기여도 판정은 아래 2026-09-10 직접 검토 기록을 우선하며 18–22번은 당시 판단의 이력이다. 19번 문서는 졸업논문과 공통 실험을 재사용하고 필요한
조건만 분기하는 병행 운영 계획이며, 새로운 기여도 판정이나 학습·평가 계약의 확정 문서가 아니다.
여기의 계획은 학위논문 M0 또는 동결 실험계약을 자동으로 변경·승인하지 않는다.

## 2026-09-10 원문 직접 검토와 비교 준비

[졸업논문의 중심과 CVPR의 관계](research_2026-09-10/thesis_alignment.html) · [26번 기록](26_CVPR추진시_졸업논문_중심과_공통실증_2026-09-10.md): DP 감사·모델 증거 결속·PP-Mark의 의료 통합 실증을 졸논 중심에 유지하고, 새 감사법은 노출 평가를 심화하는 권장안이다.

[실제 설계 v0.1](research_2026-09-10/design.html) · [25번 기록](25_환자감사와_DP보호비교_실제설계_2026-09-10.md): 새 감사법과 보호 비교를 연결하는 첫 구체적 후보·반증 실험·공통 작업을 설계했다. 알고리즘 구현·의료 성능·기여가 확인됐다는 기록은 아니다.

[연구 목적 확인: 보호·보장 환산·재사용 비용과 공격의 역할](research_2026-09-10/purpose.html). 기존의 환자 보호 목표에서 공격은 평가 도구로 필요할 수 있으며, 새 공격법을 주기여로 올리는 것은 별도 전환 제안이다. 아래 후보 우선순위를 전환 확정으로 읽지 않는다.

[환자 집합 감사법을 우선 검증할 상세 근거](research_2026-09-10/rationale.html): '가장 먼저 검증할 후보'의 근거와 한계, 관계 특징·비학습 자료에 대한 추가 선행 2편, 명확한 반증 조건을 설명한다. 상관·가변 사진 수·표준 보정 자체를 새 기여로 취급하지 않는다.

[79편 검토 HTML·드롭다운](research_2026-09-10/index.html) · [기여도 재판정](research_2026-09-10/report.html) · [구현 준비](research_2026-09-10/baseline_preparation.html) · [비교군 CSV](research_2026-09-10/comparison_matrix.csv).

기존 관련 67편과 추가 12편 모두에 개별 기록을 작성했다. 75편은 로컬 PDF의 선택 절을 대조했고,
3편은 접근 가능한 공식 본문 일부, 1편은 저자 초록만 확인했다. 전편 전문 정독·전체 재현 완료로
표시하지 않는다. CLiD 및 Tracing the Roots의 공개 score/feature packet은 실제 CPU 재계산했다.
최신 판단은 **환자 bag 감사 후보 유지, 독립 기여·CVPR 경쟁력 미입증**이다. 학습 완료 상태는
변경하지 않았고 신규 large private training은 실행하지 않았다.

## 2026-09-09 선행연구 전체 목록과 PDF — 당시 기록

**본학회 비교 정정:** 기존 6편은 기여 중복을 따지기 위한 선별이며, 입문 목록이나 CVPR SOTA
비교군 목록이 아니다. CVPR 2025 CDI는 10번에 이름만 있었고 35편·6편에서 빠졌으며, NeurIPS
2025 Tracing the Roots도 추가 대조가 필요하다. 따라서 최신 본학회 조사를 충분히 끝냈다는
해석은 철회한다. [22_본학회 비교군과 6편 선정 정정](<22_CVPR_본학회_비교군과_6편선정_정정_2026-09-09.md>)에
발표 형식·누락·비교 우선 축을 정리했다. 문헌 검토 범위는 이 정정이 18·21번보다 우선하며,
새 방법의 신규성·경쟁력은 미검증이다.

선행을 검토하고도 방향을 유지한 이유, 기존 기법 조합만으로 부족한 이유, 아직 필요한 증거는
[21_현재방향 유지 근거와 검증 조건](<21_CVPR_현재방향_유지근거와_검증조건_2026-09-09.md>)에
정리했다. 현재 판단은 작은 검증실험을 해볼 근거가 있다는 것이며, 신규성·CVPR 경쟁력이
확인됐다는 판단은 아니다. 18번의 조건부 판정을 유지하고 표준 보정과의 대조 필요성을 보충한다.

**근접성 정정:** 35편은 최신 판정의 관련 문헌 수이며 모두 직접 경쟁 논문은 아니다. 현 후보의
핵심 주장·방법 대조에 우선 둘 6편, 직접 방법·평가 비교군 13편, 기반·확장 문헌 16편으로 역할을
나눴다. [핵심 6편의 공통점·차이와 전체 역할 분류](reading_list_2026-09-09/cvpr_proximity_review.md)를
먼저 본다. 이는 현재 후보에 따른 검토 우선순위이며 정확히 같은 연구를 수행한 논문의 확정 개수가 아니다.

**사용자 범위 정정: 지금 논의할 대상은 현재 CVPR의 환자 DP 의료영상 diffusion·환자별 노출 감사
후보 하나다.** 최신 18번에 직접 인용된 35편만 보려면
[현재 CVPR 후보 PDF 목록](reading_list_2026-09-09/cvpr_patient_dp.html),
[35편과 기존 판정 기록](reading_list_2026-09-09/cvpr_patient_dp.md),
[35편 CSV](reading_list_2026-09-09/cvpr_patient_dp.csv)를 사용한다. 아래 71편은 다른 탐색 방향까지
포함한 역사 목록이며, 지금 읽을 대상 전체로 제시한 것은 범위가 지나치게 넓었다.

사용자가 다운로드해 읽을 수 있도록 기존 01~18번의 URL 91개를 전수 추출했다. 같은 논문의
공개본·proceedings·PDF를 통합하고 논문 외 자료 8개를 분리한 결과, 근접·관련 67편과 탈옥 트렌드
참고 4편, 총 71편이다. 이 중 35편이 최신 18번 기여도 판정에 직접 인용되어 있다.

- [20_근접선행연구_전체목록과_PDF_검토기록_2026-09-09.md](<20_근접선행연구_전체목록과_PDF_검토기록_2026-09-09.md>) — 전체 제목·PDF·연구 관계·기존 인용 행 번호, 먼저 읽을 15편, 과거 판단의 변화
- [검색·필터 가능한 PDF 목록](reading_list_2026-09-09/index.html) · [Excel용 CSV](reading_list_2026-09-09/reading_list.csv)

PDF 링크 67/71개에서 HTTP 200과 PDF 시작 바이트를 확인했고 나머지는 브라우저 확인 필요로
표시했다. 이번 작업은 전문 일괄 정독이나 재현 완료가 아니다. 기존 claim matrix와 조건부 가능성
판정은 확인되지만, 모든 논문의 전문·부록 정독 및 재현 완료를 증명하는 장부는 확인되지 않는다.
최신 과학적 판정은 계속 18번이며, 20번은 문헌 목록과 검토 증거의 범위를 정리한 문서다.

## 2026-09-09 공통 실험 재사용과 병행 운영

사용자는 CVPR 기여도 검토와 졸업논문 전체 실증을 병행하면서, 조건이 같은 산출물은 공유하고
추가 학습·평가가 필요한 부분은 따로 진행하는 방향을 밝혔다. 공통 자산, K10 재사용 조건,
GPU 작업 배치와 개발/확증 경계는 다음 문서에 정리했다. 후속 대화에서 정리한 두 원고의 초점,
반복 수의 근거와 한계, 단계별 시간 추정, 지속 기록 방식도 같은 문서 Sections 7--9에 반영했다.

- [19_졸논_CVPR_공통실험_재사용과_분기계획_2026-09-09.md](<19_졸논_CVPR_공통실험_재사용과_분기계획_2026-09-09.md>)

## 2026-09-08 latest authority — CVPR SOTA 전수 대조

CVPR·ICCV·WACV와 ICLR·ICML·NeurIPS·USENIX 및 2026년 핵심 의료영상 연구를 다시 대조한 결과,
주제의 **범위 적합성은 높지만 현재 3-arm 비교만으로는 CVPR main 기여가 부족할 가능성이 높다**고
판정했다. CheXGenBench가 CXR fidelity/privacy/utility와 sample-level max ReID를, Kaiser et al.이
CheXpert variable-record ULDP의 sampling/clipping adaptation을, Nature 2026이 patient-specific MIA
tail을, DP-LoRA가 private LDM fine-tuning을 각각 차지한다.

따라서 현재 1순위는 단순 비교가 아니라 **새 patient-calibrated set-level diffusion attack/audit +
동일 patient budget의 image/group/native/Kaiser DP 비교**로 강화한다. 정확한 신규성은 기록 수와
within-patient correlation에 의해 patient FPR이 왜곡되지 않게 생성모델 membership/exposure evidence를
결합·보정하고, 이를 formal patient-DP arms와 의료 utility로 검증하는 데 있다. M0·private full run·새
감사법·여러 데이터셋은 아직 없으므로 현재 상태를 `CVPR 제출 가능 결과`로 부르지는 않는다.

- `18_CVPR_SOTA_전수대조와_투고가능성_판정_2026-09-08.md`: 최상위·직접 선행 충돌표, 남은 신규성, 최신 공격·평가 baseline, CVPR용 1순위 강화안과 go/no-go

## 2026-09-08 prior authority — 팩트 검증

텍스트 탈옥은 문헌 축적도가 높지만 여전히 강한 신규 공격이 계속 보고되므로 `철옹성·해결된 성숙기`로
쓰지 않는다. 멀티모달 탈옥은 비교적 새롭고 활발하지만, 텍스트보다 수십 배 어렵거나 확실한 블루오션이라는
정량 근거는 없다. 또한 처음 검토한 CVPR 2026 PFDM은 visual jailbreak가 아니라 privacy-preserving
federated diffusion 논문이다.

Patient-DP 자체도 P3SGD(CVPR 2019), DP-FedEmb(CVPR 2023), medical user-level DP(TPDP 2025)
등의 직접 선행이 있어 `미개척지·최초`로 주장하지 않는다. Nature 2026은 aggregate MIA가 환자별 극단
위험을 가릴 수 있고 완전한 완화에는 patient-level DP가 필요하다는 근거를 제공하지만, 실험 대상은
분류모델이며 의료영상 diffusion은 후속 질문으로 남긴다.

따라서 현 1순위는 유지한다. 정확한 간극은 **동일 patient budget·근접 compute에서 image-DP,
group-converted record-DP, native patient-DP CXR diffusion을 비교하고, 병리·품질과 함께 환자별
biometric/membership/near-copy exposure tail을 평가하는 것**이다. 좋은 문제와 측정 가능성은 확보했지만,
native patient-DP의 우월성이나 신규 mechanism은 아직 입증되지 않았다. 이는 17번 문서가 정리한
당시의 판정이며, 현재 신규성·실행 우선순위는 18번 문서가 supersede한다.

- `17_탈옥트렌드_팩트검증과_현1순위_신규성경계_2026-09-08.md`: 외부 주장 팩트 검증, Nature 해석 범위, 현 1순위의 허용·금지 claim과 성공 조건

## 2026-09-03 prior authority

표준 PSPD SetAdapter와 NIH weak-label PPLRD는 각각 held-out 반증으로 종료했다. 이어 검토한
`identity suppression + pathology preservation`도 PrivDiff-Net(PMLR 2026)과 ICCV 2025
Semantic-versus-Identity가 직접 점유하고, MIDL 2026은 patient-level diffusion unlearning까지
다룬다. 따라서 biometric suppression network를 신규 방법으로 구현하지 않는다.

현재 가장 실행 가능한 질문은 **동일 patient privacy budget에서 image-DP/group-converted DP/native
patient-DP CXR diffusion을 비교하고, 생성물의 환자별 biometric exposure tail을 감사하는 것**이다.
공개 NIH 886명 전원에서 환자당 query/gallery를 하나씩만 둔 두-encoder 전제 검정은 PASS했다:
DINO/RAD-DINO AUC `0.8258/0.9884`, R@1 `0.2393/0.7111`, cross-encoder margin Spearman
`0.5054`였다. 이는 endpoint 측정 가능성만 입증하며 generator leakage, DP 효용 또는 CVPR 방법
성공은 아니다. 현재 살아 있는 것은 **audit 연구 질문**이지 검증된 신규 mechanism이 아니다.

- `10_CVPR_reset_관련근접연구_새방향_2026-09-03.md`: 현재 종합 판단과 1순위 방법·실험계획
- `11_UCAN_공개_Xray_반증결과_2026-09-03.md`: antithetic corruption-noise 가설 실패·종료
- `12_PatientSetDiff_데이터전제_진단결과_2026-09-03.md`: PSPD 데이터 전제의 첫 공개 진단
- `13_PSPD_독립확인_SetAdapter_구조smoke_2026-09-03.md`: 데이터 전제 확인과 구조 실행 가능성
- `14_SetAdapter_context_반증과_주제재판정_2026-09-03.md`: 두 context FAIL, 방법군 종료와 새 순위
- `15_PPLRD_NIH_order_proxy_residual_반증결과_2026-09-03.md`: identity carrier PASS와 change residual FAIL
- `16_biometric_선행충돌과_patient_exposure_tail_결과_2026-09-03.md`: suppression·unlearning·ULDP 선행 충돌, exposure-tail PASS와 현 1순위
- 문서 01--09는 탐색과 실패 경로를 보존하는 역사 기록이다. 방법 추천과 다음 행동은 10--13이
  supersede하며, 16번 문서가 10--15를 다시 supersede했고 17번이 이를 좁혔다. **현재 최종
  authority는 18번 문서다.**
- 16--17번 시점의 다음 단계였던 단순 generator exposure-tail addendum은 18번 SOTA 대조 뒤
  단독 중심기여에서 강등됐다. 현재 다음 단계는 PC-SMEA exact statistic·threat model·null
  calibration·baseline 계약을 결과 전에 동결하는 것이다.

## 현재 문서

1. `01_독립_논문_방향_추천_순위와_근거_2026-09-02.md`
   - 독립 논문 방향 1--4순위
   - 1순위의 한 문장 설명
   - 1순위의 가능성이 가장 높은 근거
   - PFDM/PF-LDM/SPIRE/FedDP-PALD, DPDM/DP-LoRA, P3SGD/user-wise DP,
     DP 실행증명과 의료 워터마킹 근접 연구 경계
   - 현재 실험계약에서 후속 승인 전에 보강할 사항
2. `02_AAAI_텍스트_Audit과_의료영상_CVPR_후속논문_관계_2026-09-02.md`
   - 기존 AAAI 텍스트/비영상 audit과 의료영상 독립 논문의 역할 분리
   - AAAI 채택·탈락·심사 중인 경우의 후속 연구 전략
   - CVPR 공식 범위에 대한 적합성 및 현재 설계의 경쟁력 판단
   - 단순 도메인 확장이 되지 않기 위한 독립 신규성 경계
   - patient-aware private diffusion method를 중심으로 한 권장 CVPR형 구성
3. `03_근접연구_재검증과_1순위_최종판단_2026-09-02.md`
   - Nature 2026, JAIR 2026 및 의료 DP diffusion까지 포함한 최신 근접연구 재검증
   - 이미 점유된 기여와 아직 남은 정확한 연구 교집합의 구분
   - audit-only가 아니라 method+audit이어야 한다는 최종 판단
   - 1순위 유지 사유, 대안 대비 우위, 신규성·실험 성공 조건과 중단 기준
4. `04_관련근접연구_claim_matrix_2026-09-02.md`
   - P3SGD, HiGradAvgDP, user-wise DP, DP-FedEmb, DPDM, DP-LoRA, 의료 DP-LDM,
     Nature 2026 patient audit까지 claim 단위로 대조
   - patient 평균·clipping·hierarchical averaging·noise multiplicity는 이미 선행이
     있음을 반영한 금지 claim
   - 남은 교집합을 visual coverage와 diffusion stochasticity의 fixed-compute 배분으로 축소
   - NIH X-ray 예비 반복기록 분포와 ISIC K5/K10 동결 분포의 검증
5. `05_patient_coverage_multiplicity_접근법과_실험계획_2026-09-02.md`
   - Patient-Coverage Multiplicity(PCM)의 문제 정의, 수식, 알고리즘과 privacy 경계
   - naive ULS, noise-only, uniform distinct-first, PCM의 compute-matched 비교
   - NIH chest X-ray core와 ISIC cross-domain extension의 단계별 데이터셋 전략
   - accounting, patient MIA, memorization, medical utility, 통계와 go/no-go 기준
   - 구현 순서와 동결 실험계약 변경 전 승인 경계
6. `06_ISIC_실제_gradient_진단과_현재_방법판정_2026-09-02.md`
   - 16-patient VAE diagnostic, oracle headroom, 새 환자·8-bank confirmatory 결과
   - generic perturbation 중복 방지와 실제 timestep-bin 효과의 exact ablation
   - VDM·DDIM·DPDM·CleanDIFT·HDiT까지 반영한 직접 선행 충돌
   - VAE coverage 폐기와 unit-local timestep-balanced 후보의 단계별 판정
   - 5-new-bank replication·proxy PASS 뒤 actual joint-batch FAIL까지의 증거 사슬
7. `07_patient_unit_local_clipping_proxy_결과_2026-09-03.md`
   - 동일 계산량의 B=2 batch-global 대 patient-unit-local 구성의 정확한 정의
   - 결과 전에 고정한 `C=0.20` post-clipping gate와 5-bank exact-enumeration 결과
   - ratio `0.9094`, hierarchical CI `[0.8446, 0.9506]`, 78/80 cell wins
   - 독립 raw-Gram 감사 PASS와 actual joint-batch cross-patient 검정의 다음 경계
8. `08_actual_B2_joint_batch_locality_결과_2026-09-03.md`
   - 새 다섯 banks, 3,200 gradients와 full cross-patient Gram의 실제 B=2 검정
   - primary joint local/global ratio `1.0065`, bank CI `[1.0021, 1.0117]`
   - marginal 개선과 cross-patient 오차 상쇄 손실을 분리한 실패 원인
   - 독립 감사 PASS와 단순 unit-local 방법 claim 중단 결정
9. `09_covariance_allocation_discovery_결과_2026-09-03.md`
   - 35개 orientation-symmetric rank templates의 post hoc 전수탐색
   - leave-one-bank-out 공통규칙 `rt_0123`의 ratio `0.9614`, CI `[0.9449, 0.9763]`
   - 5/5 bank 방향에도 사전 5%·pair-wins 기준 실패 및 oracle headroom 한계
   - 현재 timestep-allocation method branch 종료 결정

10. `10_CVPR_reset_관련근접연구_새방향_2026-09-03.md`
    - PFDM에서 출발하되 sample-level LDP와 patient add/remove DP의 경계를 정확히 분리
    - DP diffusion, patient/user DP, multi-image diffusion, longitudinal CXR, medical ReID와
      federated medical synthesis를 claim 단위로 대조
    - PSPD의 SetAdapter, patient-vector clipping, joint bundle generation과 평가·중단 기준
11. `11_UCAN_공개_Xray_반증결과_2026-09-03.md`
    - 320 actual full gradients의 complete/noise-only/shared-t 비교
    - complete UCAN/shared-t IID clipped B=2 ratio `1.0742`; antithetic noise branch 종료
12. `12_PatientSetDiff_데이터전제_진단결과_2026-09-03.md`
    - 비중복 80 patients·320 X-rays의 frozen DINO probe
    - 사전 절대-gap 기준 실패를 숨기지 않고 AUC `0.8779`, retrieval R@1 `0.6563`, 기록별
      label variation을 다음 비중복 confirmation의 근거로만 사용
13. `13_PSPD_독립확인_SetAdapter_구조smoke_2026-09-03.md`
    - 앞선 240명을 제외한 exact 2--3 record 160 patients의 scale-free confirmation 5/5 PASS
    - pooled AUC `0.8806`, 네 cell AUC `0.8633--0.9085`, patient ordering win `0.975`
    - 463,872-parameter SetAdapter의 실제 SD 2.1 Q=2 invariance·backward·memory smoke PASS
    - public matched non-DP falsification으로만 진행하도록 generation/DP claim 경계 고정
14. `14_SetAdapter_context_반증과_주제재판정_2026-09-03.md`
    - 표준 SetAdapter의 held-out correct/shuffled `1.000214`, wins `0.5000` FAIL
    - strict cross-record-only 독립 확인도 `1.001070`, wins `0.5078` FAIL
    - `standard IID denoising + pooled patient context` 방법군 종료
    - EHRXDiff·DDL-CXR·longitudinal diffusion·cross-view diffusion과의 새 claim 경계
    - PPLRD를 조건부 1순위로 두되 NIH order-proxy와 MIMIC 접근성 gate를 명시
15. `15_PPLRD_NIH_order_proxy_residual_반증결과_2026-09-03.md`
    - 이전 528명을 제외한 288 patients·576 images의 frozen two-encoder 진단
    - true residual/copy DINO `1.1409`, RAD-DINO `1.1591`로 명확한 FAIL
    - correct/shuffled prior는 DINO `0.5161`, RAD-DINO `0.3200`으로 identity carrier만 PASS
    - NIH weak-label/order-proxy PPLRD 종료와 다음 biometric-leakage 문헌 gate
16. `16_biometric_선행충돌과_patient_exposure_tail_결과_2026-09-03.md`
    - PrivDiff-Net·ICCV DeID·patient unlearning·medical ULDP와의 직접 충돌
    - 남은 exact intersection과 기존 K5/K10 계약의 충족·누락 항목
    - 886-patient equal-opportunity DINO/RAD-DINO tail premise 전체 PASS
    - 현재 1순위의 CVPR 경쟁력 한계와 다음 generator-attack addendum
17. `17_탈옥트렌드_팩트검증과_현1순위_신규성경계_2026-09-08.md`
    - 텍스트 탈옥 `철옹성`, 멀티모달 `확실한 블루오션`, Patient-DP `미개척지` 주장 검증
    - CVPR 2026 PFDM이 visual jailbreak가 아님을 명시
    - Nature 2026이 실제로 지지하는 범위와 생성모델로 확대할 때의 경계
    - 현 1순위의 정확한 한 문장, 허용·금지 claim, CVPR 강화 조건
18. `18_CVPR_SOTA_전수대조와_투고가능성_판정_2026-09-08.md`
    - CVPR/ICCV 및 DP diffusion·의료 생성 privacy SOTA의 직접 충돌표
    - 단순 3-arm 비교안의 강등과 patient-calibrated set-level audit 중심 강화안
    - Kaiser/DP-LoRA/PIA/DIME/MedReID/CheXGenBench/DeepSSIM++ baseline 요구
    - gold patient-specific risk, split-bias control, 두 데이터셋·복제 조건과 CVPR go/no-go

## 이전 1순위와 종료 판정

> 한 환자의 복수 영상 및 diffusion timestep 기여를 clipping/noising 전에 통합하는
> patient-aware DP latent diffusion 방법을 만들고, 기존 image-DP checkpoint를 환자 단위로
> 재사용할 수 있는 구간과 native patient-DP 재학습이 필요한 구간을 동일한 target patient
> `(epsilon, delta)` 아래에서 unit-aware audit으로 판정한다.

이를 관련연구 경계까지 반영해 처음 구체화했던 VAE 기반 PCM은 multi-patient 실험에서
uniform baseline을 이기지 못해 **폐기했다**. 현재의 조건부 방법 후보는 다음과 같다.

> 한 환자의 patient vector를 clipping하기 전에 서로 다른 records를 균형 선택하고,
> equal-probability diffusion timestep strata를 record slots에 무작위 배정해 고정 계산량의
> unbiased gradient를 만든 뒤 환자 단위로 한 번 clipping/noising한다.

환자별 평균·clipping 자체는 P3SGD, HiGradAvgDP와 user-wise DP 선행이 있으므로 신규
claim으로 사용하지 않는다. timestep stratification과 multiplicity도 직접 선행이 있다.
독립 5-bank gradient-replication과 patient-marginal clipping proxy는 통과했지만,
cross-patient Gram을 포함한 실제 B=2 joint-batch test는 local/global `1.0065`, bank CI
`[1.0021, 1.0117]`로 실패했다. 따라서 **현재의 단순 unit-local placement는 방법 후보에서
중단**했다. 뒤이어 35개 covariance-aware rank templates를 post hoc 전수 조사했으나,
가장 안정적인 공통규칙도 joint MSE `3.86%` 감소로 사전 최소 `5%`에 못 미쳤고
pair-specific oracle 한계도 `5.81%`뿐이었다. 따라서 **현재 B=2 timestep-allocation method
branch 전체를 종료**한다. 상위 patient-level DP 의료영상 연구 문제는 유지하되 다음 방법
가설은 다른 mechanism class에서 다시 찾아야 한다.

## 최신 데이터셋 hierarchy

- **Primary/core:** frontal chest X-ray, 첫 intake 후보 NIH ChestXray14
- **Cross-domain extension:** 별도 generator로 학습하는 동결 SIIM–ISIC 2020 dermoscopy
- **Optional third dataset:** 자원과 필요성이 확인된 뒤의 MURA
- X-ray와 dermoscopy를 한 unconditioned generator에 혼합하지 않는다.
- NIH official metadata와 PA-only patient manifest는 동결됐고, 12개 official archives의
  42,423-image selective acquisition과 독립 full-file verification이 PASS했다. exact
  raw-to-model preprocessing 및 SD 2.1 VAE/LoRA interface gate도 PASS했다.
- ISIC K5 images와 manifest는 확보 완료다.
- K5 4-step 연구용 dry-run은 완료됐지만 full K5 matrix와 M0는 미시작이다. B0 448장은 무결성
  PASS·gross off-domain이며, B0 시간순서는 full matrix 이전으로만 주장한다.

## CVPR 적합성 판단

- **주제 범위:** 높음. 의료영상, image synthesis, privacy·accountability의 교차점이다.
- **현재 3-arm 비교만으로 본 경쟁력:** CVPR main에는 부족할 가능성이 높다.
- **권장 강화:** 새 patient-calibrated set-level diffusion attack/audit를 중심 기여로 하고,
  동일 patient budget의 image/group/native/Kaiser DP 비교, split-bias control, gold-risk subset,
  강한 의료 ReID/near-copy baseline과 복수 데이터셋 또는 backbone replication을 결합한다.
- **AAAI와의 관계:** audit 인프라는 재사용하되, 의료영상 논문은 독립된 vision method와
  연구 질문으로 완결한다.

## 상태 경계

- 이 폴더의 문서는 주제 탐색과 추천 근거다.
- `code_working/INTEGRATED_EXPERIMENT_CONTRACT.md`의 동결 내용을 자동으로 변경하지 않는다.
- generator/DP training, attack experiment, model-bound receipt와 PP-Mark medical integration이
  완료되었다는 뜻이 아니다.
- 활성 학위논문 LaTeX·초록의 변경 승인도 아니다.

## 역사적 진행 순서와 현재 다음 기록

이전 claim matrix와 problem/method/RQ/실험계획은 04·05에 보존한다. 아래 1--9는 완료된
역사적 진행이며 현재 행동은 18번 authority와 10번 항목을 따른다.

1. 완료: UCAN 공개 X-ray 반증 실험; complete/noise-only antithetic branch FAIL·종료
2. 완료: 첫 PSPD patient-set data-premise probe; 사전 기준 4/5로 형식상 FAIL
3. 완료: 완전 비중복 2--3 record cohort의 scale-free confirmation 5/5 PASS
4. 완료: public Q=2 SetAdapter architecture smoke PASS
5. 완료: standard SetAdapter held-out context-signal FAIL
6. 완료: strict cross-record-only fresh-validation confirmation FAIL; 현 PSPD method class 종료
7. 완료: NIH exact consecutive `followup_no` order-proxy에서 longitudinal residual premise 반증
8. 완료: PPLRD premise FAIL; NIH weak-label/order-proxy route 종료
9. 완료: biometric suppression 직접 선행 충돌과 patient-exposure-tail endpoint premise 확인
10. 다음: PC-SMEA exact statistic·threat model·null calibration·baseline·gold-risk pilot 계약 동결

불리한 actual joint 결과를 건너뛰고 기존 proxy만 근거로 X-ray 대규모 방법 실험에
진입하지 않는다.

## 2026-09-02 사전진단 방법 상태

- 여러 record의 단순 평균과 patient/user-level clipping은 선행 baseline이다.
- record 간 variance와 diffusion perturbation variance의 분해 자체는 신규 방법이 아니라
  분석 도구다.
- 이 시점의 PCM은 확정된 신규 알고리즘이 아니었다. 목표는 동일 privacy·clipping·계산량에서
  `record coverage`와 `noise repetition`을 배분하여 post-clipping gradient 오차를 줄이는
  선행 비중복 규칙을 찾는 것이다.
- 첫 테스트는 full training이 아니라 reference-gradient diagnostic이다. `noise-only`,
  `uniform-distinct`, `coverage-stratified`를 같은 `Q`에서 비교하고, uniform 대비 개선이 없으면
  방법 claim을 중단한다.

## 2026-09-02 첫 진단 상태(이후 multi-patient 결과로 superseded)

- synthetic positive/negative control과 관련 unit test는 PASS했다.
- ISIC `public_development`의 hash-selected 한 환자에서 실제 SD 2.1 rank-8 LoRA gradient
  20개를 계산하는 smoke도 실행 gate는 PASS했다.
- 해당 한 사례에서는 uniform-distinct가 noise-only보다 좋았지만, frozen VAE
  coverage-stratified는 uniform-distinct보다 나빴다. `C=1.0`에서는 clip이 발생하지 않았다.
- 이는 한 환자 smoke이므로 방법 실패 판정도 성공 판정도 하지 않는다. 다만 현재 coverage
  규칙을 주 방법으로 승격하지 않으며, multi-patient 사전동결 diagnostic 전에는 feature나
  환자를 결과에 맞춰 교체하지 않는다.
- 코드·프로토콜·결과는 `code_working/pcm_diagnostic/`과
  `code_working/_reports/pcm_*`에 저장했다.

## 2026-09-03 최신 실제-gradient 판정

- 16-patient VAE coverage diagnostic은 VAE/uniform `1.0322`, CI
  `[0.9674, 1.1005]`, 6/16 wins로 실패했다. paired-strata oracle도 `0.9749`, 8/16
  wins여서 VAE coverage와 pair family를 폐기했다.
- overlap 0인 새 16 patients와 새 8-perturbation bank confirmatory에서
  true-bin/replacement는 `0.8592`, CI `[0.7931, 0.9220]`, 16/16 wins였다.
- generic without-replacement를 분리한 exact ablation에서도 true-bin/distinct는 `0.9211`,
  CI `[0.8814, 0.9572]`, 16/16 wins로 사전 gate를 통과했다.
- 결과 전에 고정한 새 5 perturbation banks에서도 bank별 ratio가 모두 1 미만이었고,
  equal-bank ratio `0.8965`, hierarchical CI `[0.8309, 0.9419]`로 multi-bank gate와 독립
  aggregate audit가 PASS했다.
- 동일한 batch-wide perturbation 사용량에서 global-only와 patient-unit-local marginal을
  비교한 사전 고정 `C=0.20` clipping proxy도 local/global `0.9094`, hierarchical CI
  `[0.8446, 0.9506]`, 5/5 banks와 78/80 cells 방향 일치로 PASS했다. 독립 raw-Gram audit와
  전체 진단 test suite 31개가 PASS했다.
- 후속 새 다섯 full cross-patient Gram banks의 actual B=2 test는 local/global `1.0065`,
  bank CI `[1.0021, 1.0117]`, 1/5 bank wins로 실패했다. marginal `0.9430` 개선보다 global의
  음의 cross-patient covariance 상쇄가 더 컸다.
- 따라서 VDM·DDIM·DPDM·CleanDIFT·HDiT 선행과 구별하려던 **단순 unit-local placement**는
  신규 방법 후보에서 중단한다. 전체 test suite는 37개 PASS이고 독립 joint audit도 같은
  실패 판정을 재현했다.
- 이 문단은 timestep-allocation 분기의 최종 기록이다. 전체 연구 방향의 최신 authority는
  `10_CVPR_reset_관련근접연구_새방향_2026-09-03.md`와 11--13 결과 문서다.
