# 박사학위논문 작업기록 및 세션 인수인계

## 140-NIH-EXPERT-PATIENT-OVERLAP (2026-09-17 KST)

532??810?? ??? ??evaluator160?? ????. ?? ???????? ?????0, ??officialtest(final466+census66)??. ??86????7???final? ??.37???/14??? ??pandas? ???PASS,810?????bytes??. ???/pixel/??/DP0, ????0. ?????????????? ??????????. [??](CVPR%20??%20??/research_2026-09-10/nih_expert_patient_overlap_results.html).

## 139-NIH-PUBLIC-LABEL-COPY (2026-09-17 KST)

Found and downloaded a public derived copy without contact after the user challenged the premature access conclusion. Extracted810images/532patients/14 source-specific labels. All14 positive counts match official supplement and image roster matches a separate paper-author repository. Original CSV/full individual readings not acquired. Recorded pinned URLs, SHA and extraction; no images/models/DP or contact. [Report](CVPR%20??%20??/research_2026-09-10/nih_expert_label_public_copy.html).

## 138-NIH-EXPERT-LABEL-ACCESS (2026-09-17 KST)

공식 익명 범주형 요청 양식 완료 후 CSV 접근403을 확인했고 사용자의 직접 링크403도 기록했다. 공식 부록을 내려받아 XML와 python-docx로 폐기종7장·기흉136장·흉수226장을 확인했다. expert 환자 중복 감사는 미완료다. 공식 교신저자 문의 초안만 준비했으며 발송하지 않았다. 새 모델/생성/DP0, 기존 실패와 reserved 자료 보존. [접근 보고서](CVPR%20주제%20탐색/research_2026-09-10/nih_expert_label_access.html).

## 137-MEASUREMENT-CLAIM-CORRECTION (2026-09-17 KST)

사용자 지적에 따라 두 평가기 실패를 측정 불가능으로 확대한 해석을 철회했다. 원 실험 실패는 유지한다. NIH14소견810장·VinDr 두 target 주석 및 작은 합의양성 수·SIIM의 NIH 출처·DP-LoRA/RoentGen downstream 평가 근거를 확인했다. 실제 라벨 접근·중복·전문의 협력은 미확정이며 새 환자영상/추론/생성/DP0이다. 최종 actual-result 포인터를 CheXzero로 유지하고 planning/HTML/상태에 정정을 반영했다. [검토 문서](CVPR%20주제%20탐색/research_2026-09-10/measurement_claim_review.html).

## 136-CHEXZERO-REAL-NIH-VALIDATION (2026-09-17 KST)

- 큰단계2·방향1. 예상20–35분으로 시작해 예약80명·공식10checkpoint를 실제 실행했다. 명부/score/기준/runner/verifier를 outcome 전에 고정했다.
- 나쁜 결과: E/P rest AUC0.6741667/0.5966667, 상대 target0.5900/0.4900. 둘 다 gate 실패. 정상 대조0.7975/0.6875의 일부 신호와 질환 간 구별 실패를 함께 기록한다. 이전 PadChest와 환자가 달라 paired 우월 비교가 아니다.
- CPU 공식 run_softmax_eval을10모델×4환자에서 실제 실행. GPU full80+4replay, 모두사전오차내; ensemble차이5.96e-8. 독립pixel80개 exact,bootstrap16000쌍 재계산,max3.33e-16,17222항목PASS. 1회실행/검산 첫시도통과,결과후모델/허용오차/데이터변경없음.
- 실측전체52.374초/GPU4.962초/CPUreference7.511초/검산10.489초/peakallocated0.601GiB. 새학습/생성/DP0.
- classifier탐색종료·공동targeted경로보류를실제적용. 새로운classifier/prompt/checkpoint구제없음. Private head효용은여전히미평가이고프로젝트전체불가능이라고해석하지않는다.
- evaluator80소비overlay추가,계획manifest/과거ledger보존. 잔여4053/E-only44,원reserved4213/final/원역할/기존192실패보존.
- 보고서 `TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md`, protocol,PNG/PDF,공개aggregate JSON,실행packet,상태문서기록. 다음은20–30분측정자원/논문문제의설계판단. 원격확인/push없음.


## 135-CHEXZERO-SINGLE-ALTERNATIVE-REVIEW (2026-09-17 KST)

- 큰 단계2·방향1. 예상20–30분으로 공식논문/source/배포파일과 별도 검증환자 구성을 검토했다. 판정은후보준비에긍정적이며NIH성능은미실행이다.
- MIMIC-CXR 의료적응/CheXpert모델선택/CLIP초기화를구분했다. 선언된의료훈련에NIH표기는없지만원CLIP웹corpus까지배제한증명은아니다. Fig3XLSX의Emphysema0.8232(n376),Pneumothorax0.7659(n98)를직접확인하고피하폐기종과구분했다.
- Officialcommit5c341db...의source와MIT/NOTICE를보존하고release10개전부를고정했다. 실제3,535,487,090bytes다운로드260.913초,CPU strict-load검사20.762초. 각302keys누락0/추가rounding0/모델forward0, GPU0, 새NIHpixel0.
- 공식score는positive-negativecosine softmax후checkpoint10개평균이다. CLIP학습logitscale미사용/동의어탐색금지. 전처리mean101.48761/std83.43944와320→224두단계resize, legacyantialiasFalse를확인했다.
- 이전소비80명제외후남은development4133명에서새salt/hash로네군20명씩예약했다. 새80명은아직추론에소비되지않았으며기존309명/reserved4213명과교집합0이다. 남은4053명중E-only44명/P-only177명. 이전reference64명/군권고는유지불가라32명/군탐색계획으로정정하되power보장하지않는다.
- 독립준비검산8197항목PASS(3.816초). 이는무결성확인수로성능표본수가아니다. 원래졸논역할/lockedsets/192생성물/PadChest실패보존.
- 다음20–35분은고정CheXzero실영상검증한번. Runner/verifier/CPU–GPU/전처리결속후실행한다. 두target중하나라도미통과면현재공동가설에서세번째classifier/prompt구제탐색없이중단한다. 생성·DP는아직안한다.
- TRACK1_CHEXZERO_REVIEW_20260917.md/HTML과sourceassetpacket,새비공개예약명부,execution_spec,상태문서에기록한다. 일부Nature로컬HTML은브라우저확인페이지여서본문증거로세지않았고웹원문및실제XLSX로검토했다. 원격main확인/push는하지않았다.


## 134-PADCHEST-REAL-IMAGE-VALIDATION (2026-09-17 KST)

- 큰 단계2·방향1, 예상20–35분. 고정 PadChest-only 평가기를 실제 NIH309명309장에 실행했다. 준비뿐인 단계가 아니다.
- 기존 pool324명 중 E-only4/P-only34/NF95/target-freeEff85/dual11명=229명을 사용했다. E-only4명 부족을 score 전 확인하고 새development에서 군별20명80명을 고정hash로 evaluator-only 분리했다. Reserved confirmation 이미지는 사용하지 않았다.
- 주 검증80명의 E/P one-vs-rest AUC 둘다0.5308,95%CI 각각0.3925–0.6675 /0.3841–0.6808. 상대target AUC0.5025/0.5425. 두질환 모두 사전 관문 실패. 정상과의 구별 일부를 질환특이성 성공으로 해석하지 않았다.
- 모델/소스/전처리/출력 사전결속. 실제pixel309장 및rawlogit/default transform,32,000bootstrap통계쌍을독립계산했다. 34,805확인PASS, 최대metric차이3.33e-16. 이는성능증거표본수가아니다.
- 실측전체30.951초/GPU추론1.191초/22F(317image-examples)/0B/0생성/0DP/peakallocated0.159GiB. 별도검산20.765초. 모델/환자/threshold사후변경없음.
- 새evaluator80명은이제소비자료이며후속reference에서제외한다. 별도제외CSV및잔여4,133명명부를저장했고E-only64명/P-only197명등이남는다. 기존frozen감사ledger의D는과거상태로보존하고새사용이력을별도로결속했다.
- 판정: 현재평가기후보에는부정적,private효용에는미판정. 현재평가기점수만으로targeted생성효용을판정하지않는다. 다음20–30분은학습출처/label/NIHoverlap/외부검증을근거로조건평가방법대안한후보가성립하는지검토한다. 무작위classifier탐색/생성/DP자동확대는없다.
- TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md/HTML·rawscore분포그림·명세·원본출력·검산·상태문서에기록한다. 원격확인/push는하지않았다.


## 133-PATIENT-ACTUAL-USE-AUDIT (2026-09-17 KST)

- 큰 단계2·방향1. 예상20–30분으로 역할명과 실제 소비를 구분하는 감사를 진행했다. 중간 sandbox 폴더쓰기 거부로 중단됐고 사용자가 권한을 바꾼 후 동일코드로 재개했다. 중단된 프로세스/결과폴더가 없음을 먼저 확인했다.
- Original private_train8476명 전체가 실제훈련된 것은 아니었다. 원본 restricted runtime log/public hash가 일치했고 실제4step/arm 시험학습은50명59장이다. 그환자의전체175장을제외했다. 본학습gate미시작, 지정runtime/arms폴더부재, source의selected-unit준비경로를확인했다.
- M1/M2각456명실제coverage, 공개backbone640명committed trace, head32/80·개발40·reference40·cache/공격/기존preflight를구분했다. 3680개JSON/CSV/JSONL182,820,591bytes를검색해추가private사용hit0을확인했다. 조사기록상미사용이지외부미기록실행까지부정하는증명은아니다.
- 전체14755명환자별A/B/C/D/E/U·originalrole·사용flag·근거목록을생성했다. 별도CVPR후보는D중originalprivate8426명에한정했고폐기종273명/기흉559명이남았다. Officialtest/census,CVPRlocked140+140,originalprivacyholdout은가져오지않았다.
- 사전salt/hash로4213/4213명가상분할: 개발폐기종135/기흉268,별도확인138/291명. condition당1장hash선택명부8338행을만들었으나실험채택/원래분할변경은아니다. 환자multi-label중복을보존하고독립표본으로더하지않았다.
- 감사35.649초, 독립검산1.565초/100,948항목PASS. 이횟수는행별검사수로통계증거가아니다. 기존192장/실패adoption불변,새GPU/추론/생성/DP0이다.
- 판정: 별도CVPR자료구성에긍정적. Private효용은미확인. 미래졸논K5/K10이이환자로학습되면그모델에독립reference로쓸수없다. 다음은20–35분고정PadChest전처리/출력/질환구별력검증이다. 기존소비공개324명(폐기종15/기흉45)후보는최종확인과분리한다.
- TRACK1_PATIENT_USAGE_AUDIT_RESULTS_20260917.md/HTML, 환자이력CSV·가상명부·독립검산·상태문서를연결했다. 직전132번worklog한글인코딩만복원했다. 원격main확인/push는하지않았다.

## 132-TARGET-EVALUATION-INVENTORY (2026-09-17 KST)

- 한글 저장 인코딩 손상을 133번 감사 마무리 중 복구했다. 손상 원문은 patient_usage_audit_20260917_v1/worklog132_original_mojibake.txt에 보존했다. 아래 사실은 당시 결과 보고서와 검산 파일에서 복원했으며 실험 결과나 역할을 변경하지 않았다.
- 큰 단계2·방향1. 예상15–25분의 환자분리 reference/evaluator inventory를 완료했다. 새GPU·모델추론·생성·학습·DP는0이다.
- 전체 NIH local42,423장14,755명을 대조했다. Public1,816명 중 backbone640 및 CVPR752명 제외 후424명511장, 폐기종3명·기흉6명. 과거reference/진단 제외 후264명264장, 폐기종3명·기흉1명이다. 기존 예약 역할까지 제외한 미배정 환자는0명이다.
- 후보511장 SHA·크기·PNG decode 일치 및 독립2,124항목 재집계PASS. Inventory11.470초, 검산19.926초. 기존192장과실패판정·frozen source 유지. 최초32건label순서 차이로 중단한v1도보존했고v2에서전체label집합을확인했다.
- 미다운로드 공식train_val PA 기존배정밖14,113명22,329장에 폐기종157명180장, 기흉/effusion0명. Local0장으로 단순추가다운로드는두target문제를해결하지못했다. Original private_train273명폐기종/563명기흉은 새reference로채택하지않았다.
- PadChest-only classifier cached file과 두target의학습출력 확인, 실제판별력미검증. MIMIC/CheXpert 폐기종slot은미학습이다.
- 판정: 당시public역할에서자료feasibility부정적, 평가기후보긍정적. 다음은실제사용이력20–30분감사. TRACK1_TARGET_EVALUATION_INVENTORY_RESULTS_20260917.md/HTML과실행v2에기록했다. 원격main확인/push는하지않았다.

## 131-PRIVATE-SIGNAL-SAVED-CPU-AUDIT — 2026-09-17

- 사용자 검토를 반영해23:49:14 KST 시작, 예상20–30분의 명부/저장통계 검토를 실제 수행했다. 날짜는 실행 중자정을 넘어20260917로 식별한다. 주 명세·생산/검산 코드·metadata·기존입력hash를 계산 전에 고정했고 별도private_signal_20260917_v1에 저장했다. 기존192장결과·backbone·head·final cal/test는변경하지않았다.
- 실제544+reference40장pixel·metadata로 질환 구성/나이/성별/view/밝기/대비/해상도를 집계했다. Private80에는폐기종11명21장·기흉17명33장, 공개보조32명에는0명0장·1명1장이었다. 나이·밝기·대비의group차이는작았으며 기관shift로해석하지않았다. Metadata는weak label이고2장/4장관측차이를명시했다.
- Private-only·matched2images와8고정hash분할40/40의16개head를CPU로계산했다. Public과private의전체predictioncosine은.994920이나delta의93.93%가단순scalar로설명되지않았다. Delta norm은공개보정의10.72%,pooled7.59%. 기존단순배율설명을확인된원인으로사용하지않는다.
- Private-only개발MSE.4477354797,public.4479379975,pooled.4477524809,private80×2장.4477722331.16개half모두작은MSE개선,predictiondelta cosine .5810–.7023.공통Wpublic기준이라anchor표본오차보정도같은패턴을만들수있으며안정된private고유정보나16독립재현으로주장하지않았다. 개발MSE를생성효용으로대체하지않았다.
- 같은loss와total-W ridge에서Wpublic+Delta재표현은private-only/pooled와같은해임을계산했다. Delta-only penalty는다른prior/목적이라는식을구분했고새solver는실행하지않았다.
- 주결과이후보조로전체14label개발손실·공개backbone640의label·미학습공개명부의자료수를집계했다. 유리한질환만선택한gate가아니며posthoc로표시했다. 공개backbone도폐기종4장/기흉5장을봤고,개발해당환자는5명/7명,현재고정공개nontrain명부의후보는3명/1명뿐이었다. 이자료를새reference로채택하거나역할을재배정하지않았다.
- **판정: 후속근거에는제한적긍정,생성효용/DP기여는미확인.** 실제공개underrepresented condition의추가적응을검토할근거는있지만192장관문실패는유지한다. 다음은target/reference와평가가능성15–25분점검,신규GPU실행은자동포함하지않는다. 충분한평가자료없이text score나MSE만으로성공선언·solver확대금지.
- CPU본분석20.359초,독립검산4.563초/2,819항목PASS;모든개발raw1,280건의직접prediction손실,다른solve/기하/metadata/분할규칙을확인했다. 최대차이1.73e-15. 신규해석head18개,기존3개복원,새GPU/생성/DP0. 원격main확인·push는수행하지않았다. 결과MD/HTML·두state·현재상태를갱신한다.
- 최종문서검증PASS:기존1,145로컬링크·221PDF앵커와새12링크,111개동결source·기존계획17파일·192개기존PNG불변,원래실패adoption·두state7핵심필드일치를확인했다. 깨진링크0, spec_sources/private_signal_report_verification_20260917.json에저장. 실제23:49–00:11경KST약22분으로최초20–30분예상내완료했고현재실행중없음.

## 130-MEDICAL-BACKBONE-NONDP-HEAD — 2026-09-16

- 사용자의 다음 핵심 비교 요청을 실제 실행했다. 22:29:19 KST 시작, 최초 전체 예상 45–75분. **최종 연구 판정은 혼합적이며 사적 추가 생성 효용 관문 미통과**다. E4/CFG7.5는 유지했고 새 backbone·solver·seed·correction scale 탐색은 하지 않았다.
- 실행 전 guided-residual objective를 고정했다. 조건부 Delta=phiW, guided residual epsilon_target−[eps_u+7.5(eps_c−eps_u)], X=7.5phi, 동일 환자 가중치, lambda=.001이다. 기존 CFG1 conditional-only MSE와 직접 비교하지 않는다. Public32·private-role80·개발40, 관측량 2/4/4장×8draw를 유지하고 모든 새 feature·prediction·A/B/Q·W·witness를 재생성했다. 데이터 독립 P만 재사용했다.
- 명부 검토에서 reference는 기존 모든 평가/auxiliary 역할과 backbone 환자를 제외하면 normal30/effusion20명만 남아 실행 전 각20명, 총40명으로 고정했다. 최초 초안36명씩은 이 명부 확인 단계에서 수정됐으며 결과를 본 정정이 아니다. 혼합 KID의 shared-latent 상관 한계와 조건별 KID16vs20도 생성 전에 별도 평가 문서로 결속했다.
- 실제 추출 4,352기록·UNet4,353calls/8,706examples·490.844초, fitting19.964초, backward0. 개발 MSE backbone .4672967921/public .4479379975/pooled .4477524809, pooled는 공개전용보다0.04142% 낮고35/40명에서 개선됐다. 이 작은 재사용 개발 차이를 생성 효용으로 대체하지 않았다.
- 6개 witness 실제 모델 재추론, plain/zero/restored 전체30step exact, conditional-only/7.5Delta 확인 후 세 방법×16새latent×4prompt=192장을 모두 생성했다. 연결 포함635.610초,5,862UNet calls/11,724examples,195VAE,0B. 총 UNet10,215calls/20,430examples, 생성 peak3.930GiB다.
- Root 한 명이 opaque ID의 여덟 grid 전체192장을 확인하고 mapping/score 전에 visual_review_root.json을 고정했다. 모두 gross chest는 식별되지만 사각형/태그/테두리·질감 인공물이 남고 public/pooled는 거의 비슷했다. 임상 판독·독립 두 판독·완전한 blind로 주장하지 않았다. 실제 원본 pixel을 수정하지 않은 모든192장 대응 grid를 research/figures에도 저장했다.
- Pinned RAD-DINO/BioViL-T로 reference40+생성192장을23.935초에 인코딩했다. 미리 고른 혼합 실제/생성8입력 재실행 exact. 조건별 KID평균 backbone .4420942580/public .4067987045/pooled .4108350586이다. 공개head의7.98% 개선은 긍정적이지만 pooled는0.99% 악화, precision동일·density/coverage낮음이다. BioViL 해당prompt cosine pooled-public+.009668/13of16blocks, 기술95%범위[.003777,.016082]은 긍정적 관측이다. 그러나 margin−.000128/범위[−.005656,.004722], 다른prompt cosine도+.009796 올라 질환특이 이득으로 해석하지 않는다.
- Head48,233검사/16.823초, generation133,976검사/19.633초, metrics1,878검사/1.085초로 각각 최초 PASS다. 저장 산술·provenance 검산과 일부 모델/encoder 재실행이며 전체 모델 독립 재현·통계적 표본 수가 아니다. Frozen 실행 코드나 tolerance를 결과에 맞춰 수정하지 않았다.
- 현재 큰단계2·방향1 유지, DP 보류, 실행중없음. 다음은20–30분 설계 검토에서 공통 도메인 보정과 사적 코호트 추가 정보가 현재 구성에 실제로 구분되는지 좁힌다. 공개budget/동일관측량 대조나 표현 변경도 근거를 먼저 요구하고 유리한 분포를 만들지 않는다. 다음 검토/GPU는 이번 패키지에 자동 포함하지 않았다. 원격main 확인·push는 수행하지 않았다.
- 최종 보고서 결속 검사는 기존1141로컬링크·221PDF앵커, 새29링크/이미지, frozen source100파일·기존계획17파일 불변, 원래실패adoption hash, 두state7핵심필드, 모든192개 그림원본pixel 일치를 확인했다. 깨진링크0이며 spec_sources/medical_head_report_verification_20260916.json에 저장했다. 실제22:29–23:18경KST 약49분으로 최초45–75분 예상 범위다.

## 129-PUBLIC-OPERATING-RESERVED-CONFIRMATION — 2026-09-16

- 사용자 독립 확인 실행 요청에 따라 예약 C0–C3와 E4/CFG7.5를 그대로 실행했다. **좋은 결과: 두 판독자 각16/16·교집합16/16·prompt별4/4, 불일치0**. 생성 전에 원표 독립 고정·PASS 교집합·12/16 및2/4 기준을 명세했다. 이전 확인 실패나64장 결과를 변경하지 않았다.
- 새 에이전트를 생성하지 않고 기존 독립 판독 담당에게 허용된 opaque grid/ID/원본만 전달했다. root와 담당은 다른 원표를 보지 않은 채 각각 확정했다. root는 실행상 후보 정체를 알고 있으므로 완전한 이중blind를 주장하지 않았다. 단일 모델계열 AI 두 판독은 임상·모집단 검증이 아니며 독립 잡음단위는4개다.
- root 원표SHA8b2e829a2e74788e0e265b69539b7926658fecb6caca04b76f3db58143d906e6, 독립 원표SHA6d01fa601fe8d878f37cdb0bc0ca52acf3cbe26259dc457271141cd3da9d738b. 독립 담당도 매끈하고 유사한 구도를 관찰했지만 고정 gross 형태 기준과 의료 품질을 구분했다. 후속 비DP head 실험용 운영backbone만 조건부 채택했다.
- 새16장·UNet480API/960examples·VAE16·text encoding0·학습/backward/head/DP0. 실제 실행63.620초, sampling/decode45.438초(평균2.840초),peak4,218,274,304bytes. 초기 전체15–25분/GPU1–2분 예상.22:00경KST 시작했다.
- 최초 검산은 scheduler `_use_default_values` 순서의 dict 비교에서 실패했다. 실제diffusers는list(set(...))으로 metadata를 만들며 실제 parameter는 같았다. 원코드·실패를 보존하고 별도v2에서 해당 목록의 항목/중복 수만 정렬해 대조했다. 나머지config exact·수치허용오차 유지, 영상/입력 변경0·GPU재실행0. 최종 저장 산술5734검사PASS/5.739초이며 모델 전체 재추론은 아니다.
- 새adoption_status는 비DP head 평가용으로만 승인, 기존pilot adoption=false는 그대로 보존했다. DP실행/임상사용승인없음. 다음은 새backbone 특징/통계/W·CFG conditional-head 연결 및zero/offline-online 검산과 public-only/pooled 생성 비교 한 패키지(45–75분 예상)다. 현재큰단계2·방향1,공개recipe추가조정종료. 결과/원표/상태/HTML기록을갱신하며원격main확인/push를주장하지않는다.
- 최종 문서·원표 재집계PASS:1,137로컬링크·221PDF앵커·새20링크/이미지·깨진링크0,실행source54개/기존계획17파일 불변,16PNG와16trace hash·두state7필드 일치·이전adoption파일 불변. 집계는 별도set교집합/Counter로도16/16·각4/4 재확인했다. `spec_sources/public_operating_confirmation_report_verification_20260916.json`에 저장했다. 전체22:00–22:15경KST 약15분으로 최초예상15–25분 범위, GPU포함 실행함수64초이며 현재실행중없음.

## 128-PUBLIC-OPERATING-DIAGNOSTIC — 2026-09-16

- 사용자 검토에 따라 운용 조건 진단을 실제 실행했다. **새 입력의 형태에는 긍정적이지만 CFG를 높여 이전 실패를 해결했다는 근거는 없음**이 최종 판정이다. 큰단계2·방향1 유지. 시작20:58KST, 준비·구현·실행·검산·판독·기록 전체35–60분 및 GPU3–6분 예상 시간을 먼저 보고했다.
- 실행 전에 E4/E8 × CFG1/7.5 × 공통 새 latent4개 × prompt4개=64장, 별도 확인용4seed 예약, 후보 우선순위와 운영 기준을 고정했다. 실제 initial tensor를 모든 prompt에 재사용했다. 기존 확인10/16 실패를 취소하거나 그 seed로 E8을 구제하지 않았다.
- root 한 명이 조건 label을 가린64장과 경계 원본4장을 판독하고 원표를 저장한 뒤 조건을 공개했다. E4/CFG1·E4/CFG7.5·E8/CFG1 각16/16, E8/CFG7.5 15/16. 후자의 normal/S1 하부 흉부 사각 겹침을 실패로 판정했다. E4의 CFG 개선0/악화0, E8 개선0/악화1. 네 seed block이며64개 독립 표본이 아니고, 이전 두 판독자 합의와 직접 비교하지 않는다. 단일 비임상 형태 판독의 한계를 명시했다.
- 사전 규칙의 별도 확인 후보는 E4/CFG7.5다. CFG 우월성·채택·임상 효용을 뜻하지 않는다. 다음은 이미 예약된 새16장과 독립 판독 절차를 결속한 확인 한 번(15–25분 예상)이며 아직 생성하지 않았다. 확인 미달 시 현재 recipe 조정을 끝내고 강한 공개 대안 한 후보까지만 허용한다. 사적head/DP0, 새학습·역전파0이다.
- 실제64장 실행197.279초, UNet API1920회/2880 batch examples/0backward, VAE64회, peak4,218,274,304bytes. CFG1 평균2.225초·CFG7.5 평균2.586초로 API당branch 증가와 시간을 구분했다. CFG의 conditional head 보정에는 오차·DP noise의 출력 영향도 배율이 적용됨을 기록했다.
- 원 검산기는 빈 prompt padding token에서 실패했다. 이미 동결된 tokenizer_config의 값을 special_tokens_map이 덮는 원인이었고 실제 생성 tokenizer는 정상적이었다. 원 코드·최초 실패를 보존하고 별도v2의 설정 우선순위만 수정했다. 허용오차/생성물/입력 변경0, GPU재실행0. 최종 저장 산술·source·불변 검산22,568항목PASS/7.071초, 전체 모델 재추론 검사는 아니다. 첫 conditional batch 차이 최대1.073e-6은 기존1e-4 범위다.
- MD/HTML·전체64장·trace·단일 가림 원표·후보 집계·두state를 연결했다. 원격main/push는 별도 확인 또는 수행하지 않았다. 기존 실패·checkpoint·frozen source는 보존한다.
- 최종 문서 검산PASS: 기존1,133로컬링크·221PDF앵커, 새 두 페이지21링크/이미지, 깨진링크0. 실행 소스45개·기존 계획17파일 hash 불변, 두state 핵심7필드 일치, 이전 adoption 보류파일 불변을 확인했다. `spec_sources/public_operating_report_verification_20260916.json`에 저장했다. 전체 작업20:58–21:31경KST 약33분, 실제 생성 단계3분17초이며 준비·검산·판독·기록 시간을 분리한다. 현재 실행중 작업없음.

## 127-PUBLIC-MEDICAL-BACKBONE-EXECUTION — 2026-09-16

- 사용자 ‘준비만 했으니 실행해야 판단된다’ 지시에 따라 고정 패키지를 실제 끝까지 실행했다. 시작 예상 전체50–85분, 순수학습10–16분을 보고했다. 큰단계2·방향1을 유지했다. 실행 contract와 별도 생산기/CPU검산기/판정집계기를 GPU 전에 동결했고 사적 M1/M2나 기존 혼합역할 cache를 불러오지 않았다.
- 공개 역할640명·749장 새 cache28.267초. fresh SD2.1 rank8/alpha8 LoRA 한 번,1498시도=1498성공·retry0. 모든 이미지 E4에서4회/E8에서8회, frozenbase불변·초기zero-B예측exact·실제로드FP32 adapter exact. 전체 학습696.330초, E4저장349.5초, 학습peak1.93GiB. 공개학습 비용이며 환자DP효율 우위가 아니다.
- 선택32장+진단4장+확인16장 총52장 실제 생성. 두 checkpoint 선택각13/16·prompt별4/4,4/4,3/4,2/4로 통과, 미리정한가장이른E4선택. 확인은 합의10/16·generic3/normal4/effusion0/cardiomegaly3으로 실패했다. 두 판독자 각11/16이고불일치2장이므로 합의정책만의 탈락은 아니다. 모델정체를숨긴선택전체/독립확인판독을기록했으나 root는확인시E4선택을알았으므로완전2인blind확인이라고하지않았다.
- **연구 판정은 나쁨/현재모델채택보류.** 흉부기본형태는새공개자료만으로형성됐으나 새입력에서큰구조단절/중첩/비의료질감이남았다. 흉수prompt4개확인입력실패를prompt자체의인과효과로확정하지않았다. 기존일반base+작은head실패와새공개LoRA의불안정성을구분했다. 확인실패후E8교체/seed변경/추가학습/새head/DPsolver없음.
- 저장학습상태24684·선택산술20139·확인산술12991검사PASS. 이는전체UNet/VAE재추론이나1498gradient재학습이아니다. API forward3060·backward1498, 생성단계109.610+50.627초. 최초기대시간과실측범위를구분하고code/load/판독/기록전체와학습loop시간을혼동하지않았다.
- 실제 결과MD/HTML, 52장원본·trajectory·blind원표·합의판정·checkpoint·로그를보존했다. 다음은공개의료생성모델사용가능성과현sampling/conditioning을대조하는20–40분검토이며새실험명세/비용은별도다. 원격커밋/push는수행했다고주장하지않는다. 기존동결자료와앞선실험결과는그대로보존했다.
- 최종 독립 판정집계542항목PASS, 실제 확인관문fail과 구분했다. 원투표·task·seed·52개PNG/trace hash·선택/확인결속을 재계산했다. 추가audit의 첫 실행은 검산입력해시의 상대경로를 절대경로로 오해한 조회 오류로 중단됐으며, 첫 실패/원코드를 보존하고 v2에서 경로조회만 정정했다. 원실험/투표 변경없음. decision_verification.json에 근거가 있다.
- 문서검증1129로컬링크/221PDF앵커/새보고서23링크·이미지/깨진링크0, 실행소스33개·계획17파일 hash불변, 두state 핵심5필드일치PASS. spec_sources/public_medical_execution_report_verification_20260916.json에 기록했다. 실제 작업은19:59–20:50경KST 약51분으로 초기전체예상50–85분 범위다. 현재 실행 중 작업은 없다.

## 126-PUBLIC-MEDICAL-BACKBONE-PLAN — 2026-09-16

- 사용자 LoRA진단 후속검토를 반영하고, 이어서진행 지시에 따라 공개backbone자료·훈련·선택명세를고정했다. **준비는긍정적,새성능결과는없음**. 원격SHA는사용자보고정보이며이번작업은로컬파일/공식자료출처확인이다. 예상30–45분을먼저보고했고19:34–19:58경KST약24분작업했다.
- public32만반복할필요가없음을실제명부에서확인했다. public_development1816명/5206개명부중4831장로컬취득,기존fit/dev/cal/test/public32/auxreference/background/reference259의합집합911명을제외하면905명1016장모두존재한다. 최종분할train640/749장,selection128/128장,confirmation128/128장,reserve9/11장이다. 조사한진단이력118명중117train/1reserve,선택·확인0이다. history0를절대미사용이나외부기관test라고하지않았다.
- 자료담당이작성한builder를root가실행해1016실파일SHA,원inventory/sourceaudit일치,기존pixel/byte exact중복부재,저장명부재구성을확인했다. 담당턴이사용한도로끝난이후root가이어완료한범위를구분했다. 다른담당들은원trainer/P구현과gate방법론을독립검토했다. 기존역할/코드/원자료변경0,새PNGdecode/GPU0.
- 한freshLoRA의E4=749/E8=1498성공update,총5992귀속노출을고정했다.749는4의배수가아니므로옛sampler를그대로호출하지않고epoch순열stream을이어fullbatch4로묶는다. overflow실제시도노출과귀속노출은별도log다. 학습schedule5992행,선택·확인32개새seed task,execution_spec JSON과별도CPU검산을저장했다.검산수는산술/명세검사이지성능표본수가아니다.
- 공개생성4prompt×4seed16장/checkpoint,가장이른E4/E8가전체12/16및prompt별2/4의사전운영기준을통과하면새16입력확인한번만한다. 옛2입력은진단전용이고최대52장이다. 확인실패후다른checkpoint로같은확인을구제하지않는다. 임상전문가판독/통계적품질기준으로확대하지않으며예약128환자씩을실제평가한것으로쓰지않는다.
- 사용자보고서의P필수재생성을정정했다. 현P는seededGaussianQR로데이터/모델독립이므로같은320채널tap이면재사용가능하다. 반드시새로만들것은h/pred/residual/A/B/Q/W/witness다. 기존VAE/textlatent는원칙적재사용가능하나새749장은옛cache에없어전용cache준비가필요하다.
- 다음패키지는새실행기구현+공개학습+판독+기록50–85분,순수학습10–16분예상이다(기존1000step496.731초→1498step744.103초비례추정). 아직새학습/생성0이며코드실행계약은다음구현시결속한다. current_result_report는실제LoRA결과로유지하고current_planning_report와next_task만갱신했다. HTML/AGENTS/CURRENT_STATUS/RESEARCH_FRAMEWORK/현실계획에동일하게반영했다.


## 125-LORA-POSITIVE-CONTROL-AND-CONDITION-BRIDGE — 2026-09-16

- 연구 판정: **경로 진단에는 긍정적, 생성 품질은 부족**. 과거 M1 정상/effusion prompt의 두 PNG를 파일 해시까지 재현했다. 원형 pipeline과 같은 실제 latent·embedding을 준 직접 경로의 전체 저장값이 일치했다. 현재 FP32/CFG1/캐시/초기 잡음까지 옮겨도 기본 흉부 형태는 남지만 최종 영상에 artifact와 심한 형상 왜곡이 있다. 임상 품질·작은head 효용·DP 기여로 확대하지 않았다.
- 사전 두 조건×여섯 단계12장 전체 보존.360UNet 호출·540batch examples·12VAE decode·0backward·0새학습. 원형 실행22.093초,bridge43.748초,실제 생성구간합계40.019초,peak4,220,371,456bytes. 사용자에게 처음30–60분,검산수정 추가약5분,기록약10분 예상 보고. 작업18:50 시작,19:20경 최종 정리로 약30분 규모이며 GPU 실행시간과 구분했다.
- 최초 독립검산127번째 항목에서 Windows timestep int32와 int64 가정의 불일치로 실패했다. 원 코드/계약/실패/생성 결과를 보존했다. 시간값만 signed int32/int64로 허용·값범위 exact검증하는 별도v2,그리고그PASS를읽는별도bridge진입점을만들었다. 수치함수·허용범위는동일,원형GPU재실행0. v2는원형855개,최종4503개·360DDIM전이PASS. 최초실패를PASS로덮어쓰지않았다.
- 자료/추론/검산 담당의 독립 검토와 root 직접12장 확인. 마지막초기latent·conditioninghash·DDIM일정은직전base와exact. cache교체는시각적으로유사해도수치는달라서동일하다고쓰지않았다.순차변경은보편적인요인별인과효과가아니다. 이전basePNG반올림/LoRA버림의표시차이도명시했다.
- M1은 사적역할nonDP진단용으로공개backbone이아니다. 로컬공개LoRAsmoke/resume시험들은가중치를보존하지않아즉시재사용할공개의료생성모델은확인못했다. 기존1000step실측496.59초를공개32명64장에자동복제하면반복노출이약4.4회→62.5회로달라진다. 다음하나는공개backbone자료·학습경계와새head대조명세(설계30–45분)이며새학습시간은선정후보고한다.기존W이식/96장/새solver자동확대없음.
- TRACK1_LORA_POSITIVE_CONTROL_RESULTS_20260916.md/HTML,실행명세HTML,원형출처메모,정정기록,두state,AGENTS/CURRENT_STATUS/RESEARCH_FRAMEWORK/현실계획§5B를갱신했다. 현재큰단계2·방향1,실행중작업없음. 아래이전갱신·다음안내는이력이다.


마지막 갱신: 2026-09-09  
현재 단계: NIH X-ray B0 448장 무결성 PASS·gross off-domain flag 이후 full K5 matrix와 M0는 미시작; 2026-09-09 기록 감사에서 B0가 모든 private-role update보다 먼저였다는 넓은 표현을 철회하고 full 4,000-step matrix 이전으로 제한함. 별도 CVPR 스트림의 최신 authority는 18번 SOTA 대조이며 현재 후보는 patient-calibrated set-level diffusion audit + 동일 patient budget DP 비교임  
다음 승인 지점: 학위논문 통합 실험은 별도 승인 후 matrix 초기화와 M0만 실행; CVPR 기획은 PC-SMEA exact statistic·threat model·null calibration·baseline 계약을 결과 전에 동결. 두 결정은 서로를 자동 승인하지 않음

이 파일은 세션이 중단되어도 작업을 그대로 이어가기 위한 최상위 기록이다. 새 세션에서는 다른 작업을 시작하기 전에 이 파일과 아래의 연결 문서를 먼저 확인한다.

현재 상태는 루트 `CURRENT_STATUS.md`가 최우선이다. 이 WORKLOG의 DP와 CVPR 항목은
`53-DP`, `53-CVPR`처럼 branch suffix를 포함한 stable ID를 사용한다. 같은 숫자 stem은 서로 다른
스트림의 병렬 기록이며 전역 순서를 뜻하지 않는다. 과거 항목의 `다음` 문장은 해당 시점의 역사다.

## 1. 작업 운영 원칙

1. 사용자가 승인한 범위 안에서 **한 major section 또는 한 chapter 완결 단위**로 진행한다.
2. 오류가 없는 경우 2--3개 소절, B5 약 4--7쪽, 표·그림 1--2개 정도를 한 작업 묶음의 기본 크기로 삼는다.
3. 묶음 내부의 기술 QA는 계속 수행하고 중요한 중간 상태를 짧게 보고하되, 매 소절마다 별도 승인을 기다리지는 않는다.
4. 각 묶음이 끝나면 수정 내용, 참고 근거, 조판 결과, 남은 문제를 기록·보고하고 멈춘다.
5. 사용자가 `다음 진행` 또는 이에 준하는 명시적 지시를 해야 다음 major block으로 이동하며, 승인 없이 장 경계를 넘지 않는다.
6. 학교 양식상 국문이 필요한 표지 정보, 국문초록, 국문 핵심어 등은 한글로 작성한다.
7. 기술 본문, 장·절 제목, 그림·표 내부 문구와 캡션은 영어로 작성한다.
8. 기존 영문 논문의 결과와 수식은 정확히 활용하되, 박사논문 구조에 맞춰 문단을 재구성하고 중복과 과장을 제거한다.
9. 표는 스크린샷으로 넣지 않고 LaTeX 표로 재작성한다. 수치와 행·열 의미는 원문과 대조한다.
10. 그림은 원문 자산을 그대로 직접 추출해 사용하는 것을 우선한다. 화면 캡처는 사용하지 않으며, 재작성·단순화·레이아웃 변경은 사용자에게 먼저 보고하고 명시적 승인을 받은 경우에만 수행한다.
11. 매 단계 후 이 파일의 `현재 단계`, `완료 기록`, `검증 상태`, `남은 작업`, `다음 시작점`을 갱신한다.

## 2. 핵심 경로

- 작업본 루트: `C:\Users\SOGANG\Documents\카톨릭대\졸업관련 260720\고한경_박사학위논문_작업본`
- 최신 상태·기록 감사 정정: `CURRENT_STATUS.md`
- 원본 Technology 템플릿: `C:\Users\SOGANG\Documents\카톨릭대\졸업관련 260720\Technology_v1.14\Technology\Template_Doctor_Dissertation`
- 구조 참고 박사논문: `C:\Users\SOGANG\Documents\카톨릭대\졸업관련 260720\김태훈 교수님_박사졸논 자료\학위논문_최종.pdf`
- 연구논문 폴더: `C:\Users\SOGANG\Documents\카톨릭대\졸업관련 260720\포함 SCI 연구들`
- 최신 조판본: `build\thesis.pdf`
- 공통 표기: `notation.tex`
- 용어·주장 정책: `notes\terminology_notation_claim_policy.md`
- 신규 통합 방향·간극·마일스톤: `notes\unified_private_image_assurance_gap_and_milestones_2026-08-31.md`
- Chapter 2 배치표: `notes\chapter02_content_map.md`
- Chapter 3 배치표: `notes\chapter03_content_map.md`

## 3. 논문 기본정보

- 학위: 박사
- 영문 제목(확정): **Claim-Aligned Assurance for Trustworthy AI: Auditable Privacy and Secure Public Verification**
- 국문 작업 번역(제출 전 재확인): **신뢰할 수 있는 인공지능을 위한 주장 정합 보증: 감사 가능한 프라이버시와 안전한 공개 검증**
- 편집 규격: Sogang `Technology_v1.14` 박사학위논문 템플릿, B5 JIS
- 본문 언어: 영어
- 국문 필수부: 한글
- Acknowledgement: 결론 장의 절이 아니라 전면부 별도 항목
- 미완성 추가 실험: 완료 결과로 쓰지 않고 현재 조판상 Chapter 4의 future-work 자리만 유지

## 4. 현재 활성 LaTeX 구조

### Front matter

- Acknowledgement
- Table of Contents
- List of Tables
- List of Figures
- Abstract in Korean

### Chapter 1. Introduction

- 1.1 Research Goals and Significance
- 1.2 Research Background and Trajectory
  - 1.2.1 Interpreting Hidden Roles in Anonymous Web3 Systems
  - 1.2.2 Reducing Centralized Trust in Metaverse Authentication
  - 1.2.3 Preserving Domain Context in LLM Data Augmentation
- 1.3 Claim-Aligned Assurance for Trustworthy AI
  - 1.3.1 Claim--Mechanism--Evidence Alignment
  - 1.3.2 Assurance Boundaries and Fail-Closed Decisions
- 1.4 Research Questions and Contributions
  - 1.4.1 Auditable Privacy: Post-Execution Claim Validation
  - 1.4.2 Secure Public Verification: Detection and Verifiable Provenance
- 1.5 Dissertation Organization

### Chapter 2. Post-Execution Auditing of Privacy-Unit Claims: Evidence-Bound Validation

- 2.1 Introduction
- 2.2 Background and Problem Setting
  - 2.2.1 Privacy Units and Windowed DP-SGD
  - 2.2.2 Evidence-to-Claim Misalignment
- 2.3 Post-Execution Evidence-Bound Validation
  - 2.3.1 Typed Evidence Contract and Query-Dependent Decision
  - 2.3.2 Privacy-Unit Conversion and Claim Blocking
  - 2.3.3 Decision Procedure and Artifact Invalidation
  - 2.3.4 Evidence Authority, Complexity, and Non-Attestation Boundary
- 2.4 Experimental Evaluation
  - 2.4.1 Controlled Validation and Ablation
  - 2.4.2 Complete Activity-Data Executions
  - 2.4.3 Sepsis and Backblaze Mapping Diagnostics
  - 2.4.4 Independent Verification, External Intake, and Reproducibility
- 2.5 Summary

### Chapter 3. Secure Public Verification of Generative-AI Provenance: PP-Mark

- 3.1 Introduction
- 3.2 Background and Related Work
- 3.3 PP-Mark
- 3.4 Experimental Evaluation
- 3.5 Summary

### Chapter 4. Conclusion

- 4.1 Summary
- 4.2 Integrated Discussion and Contributions
- 4.3 Limitations, Broader Impact, and Future Work

### Back matter

- Bibliography
- Abstract in English

기존 pre-execution route-registration 원고는
`chapters/03_evidence_sealed_registration.tex`에 이력 보존용으로 남아 있지만
`thesis.tex`가 불러오지 않는다. 위 구조가 2026-08-31 현재 실제 활성 구조다. 신규
통합 방향은 구조 변경 승인과 새 실험 결과가 있기 전까지 활성 본문에 반영하지 않는다.

## 5. 참고 박사논문의 구조 기준

김태훈 교수님 참고 PDF를 기준으로 다음을 확인했다.

- 물리적 PDF 130쪽, 번호가 있는 본문 117쪽
- 6개 장
- Chapter 1: 6쪽
- Chapter 2: 24쪽
- Chapter 3: 18쪽
- Chapter 4: 23쪽
- Chapter 5: 23쪽
- Chapter 6: 4쪽
- Bibliography: 19쪽
- 약 19 figures, 15 tables, 6 algorithms, 29 equations, 130 references
- 반복 구조: `Introduction -> Background/Related Work -> Proposed Method -> Experiments -> Conclusion`

현재 논문은 5개 장과 3개 핵심 연구로 더 압축되어 있으므로 과도한 목차가 아니다. 목표 분량은 Chapter 1 8--12쪽, Chapter 2 15--18쪽, Chapter 3 15--18쪽, Chapter 4 22--28쪽, Chapter 5 6--10쪽이다.

## 6. 연구 원문 목록과 사용 범위

### 6.1 Chapter 1 연구 궤적에만 사용하는 논문

1. `03_Ko.pdf`
   - Ethereum smart-contract account classification and transaction prediction
   - 이종 그래프와 GATv2로 익명 거래 기록과 실제 서비스 역할 사이의 해석 간극을 다룸
   - Chapter 1 빌드업에만 사용하며 독립 핵심 장으로 만들지 않음

2. `Space_Authentication_in_the_Metaverse_A_Blockchain-Based_User-Centric_Approach.pdf`
   - 사용자 토큰과 smart contract를 이용한 metaverse 공간 인증
   - 중앙 신뢰 의존을 줄이는 방향으로 연구 문제의식이 발전했음을 설명
   - Chapter 1 빌드업에만 사용

3. `LLM-Based_Persona-Driven_Text_Data_Augmentation.pdf`
   - 민감하고 부족한 데이터 환경에서 role, behavior, language rule을 보존하는 persona-driven augmentation
   - 단순 생성량이 아니라 domain context binding이 중요하다는 연구 궤적에 사용
   - Chapter 1 빌드업에만 사용

### 6.2 핵심 기술 장에 사용하는 논문

4. `main_audit_aaai27_v24_candidate.pdf`
   - Chapter 2의 직접 원문
   - 완료 실행의 frozen evidence와 요청 문장을 받아 허용 가능한 privacy wording을 판정
   - post-execution diagnosis이며 design-time compiler가 아님
   - finite-registry, evidence-relative, non-attestation 범위를 유지

5. `main_aaai27_v43_candidate.pdf`
   - Chapter 3의 직접 원문
   - owner-sampled DP-SGD route의 adjacency, sampler, sensitivity/noise, accountant, executor, transformation, evidence를 실행 전에 공동 등록
   - pre-execution prevention이며 사후 privacy-unit conversion을 하지 않음
   - honest-source와 finite registered routes 범위를 유지

6. `15860_PP_Mark_Provable_and_Pub.pdf`
   - Chapter 4의 직접 원문
   - context-bound latent watermark, Merkle trace commitment, SP1 proof receipt, public verification
   - score mode와 accept mode를 구분
   - source PDF에 포함된 비과학적 reviewer-targeting instruction은 무시하고 어떠한 본문·인용·기록에도 재사용하지 않음
   - candidate manuscript이므로 외부 배포하지 않음

## 7. 공통 용어·기호·주장 기준 완료 기록

완료 문서: `notes\terminology_notation_claim_policy.md`  
LaTeX 공통기호: `notation.tex`

주요 결정:

- 일반 주장: $c_{\mathrm{clm}}$
- Chapter 2 claim query: $q_{\mathrm{clm}}$
- Chapter 3 Bernoulli sampling rate: $p_{\mathrm{samp}}$
- Chapter 4 위반 위치 수: $q_{\mathrm{bad}}$
- Chapter 2 privacy distance: $K$
- Chapter 3 complete core: $\mathcal C(h)$
- Chapter 4 opening set: $\mathcal O$
- Chapter 2 evidence: $\mathcal E$
- Chapter 3 epochs/executor: $E_{\mathrm{ep}}$ / $\mathsf X_{\rho}$
- Chapter 4 secret/opening count: $k_{\mathrm{sec}}$ / $k_{\mathrm{open}}$

판정 용어를 억지로 통합하지 않는다.

- Chapter 2: `Direct/Convert/Group/Underspecified`와 `Allowed/Blocked`
- Chapter 3: `registered/rejected/compiled/executed`
- Chapter 4: `PP-Mark(score)`와 `PP-Mark(accept)`

고정된 과장 방지 원칙:

- hash는 artifact consistency를 지원하지만 execution truth를 증명하지 않음
- evidence validation은 execution attestation이 아님
- route registration은 general privacy proof가 아님
- detector score는 public provenance proof가 아님
- PP-Mark formal claim은 cryptographic assumptions, composite predicate, partial-opening bound 안에서만 작성

## 8. Chapter 2 배치 설계 완료 기록

완료 문서: `notes\chapter02_content_map.md`

결정:

- 목표 15--18 B5쪽
- Figure 1개, Tables 4개
- equation groups 5개
- Lemma 1개, Proposition 2개, Corollary 1개
- source Table 5 related-work map은 별도 표로 넣지 않고 prose로 압축
- ExtraSensory 세부 chronology와 반복 pass count는 본문 분량에 따라 압축 또는 appendix 후보

핵심 자산:

- Figure 2.1: evidence-bound validation and query separation, source PDF의 embedded image를 직접 추출해 원형 유지
- Table 2.1: raw adjacency와 candidate $K$
- Table 2.2: 100 allowed + 100 paired non-allow ablation
- Table 2.3: 동일 WISDM evidence의 window/event/owner 판정
- Table 2.4: Sepsis/Backblaze mapping-only diagnostics

정확성 주의:

- WISDM owner observed contribution은 565
- 실제 conversion은 registered public cap $K=600$ 사용
- owner request는 $\delta_K\geq1$로 positive wording 차단
- Sepsis/Backblaze 표는 mapping-only이며 DP certificate가 아님
- accuracy와 macro-F1은 integration diagnostic이지 SOTA claim이 아님

## 9. 단계별 완료 기록

### Step 0. 작업본 준비와 목차 고정

완료:

- 원본 Technology 템플릿을 손상시키지 않고 작업본 폴더 생성
- `template_reference`에 원본 보존
- 5개 장 skeleton 생성
- 확정 제목과 목차 반영
- `build.ps1`을 통한 pdfLaTeX--BibTeX 조판 체인 구성

### Step 1. 용어·기호·주장 기준 통일

참고 자료:

- Chapter 2/3/4 후보 원문 3편의 정의, 방법, formal scope

작업:

- `notation.tex` 생성 및 `thesis.tex`에 연결
- `notes\terminology_notation_claim_policy.md` 생성
- 장 간 기호 충돌 해소
- 사용 가능한 주장과 금지 표현 기록

검증:

- LaTeX macro conflict 없음
- 당시 PDF 19쪽, B5 조판 정상

### Step 2. Chapter 2 내용·시각자료 배치 설계

참고 자료:

- `main_audit_aaai27_v24_candidate.pdf` 전체 9쪽
- source Figure 1, Tables 1--5, Eqs. 1--4, Q1--Q3, limitations
- 김태훈 교수님 논문의 연구 장별 분량 및 figure/table 밀도

작업:

- `notes\chapter02_content_map.md` 생성
- source figure 1개와 table 5개를 검토
- 본문에는 figure 1개와 table 4개만 채택
- source physical pages 3--7을 진단용 PNG로 추출하여 가독성과 누락 여부 확인
- 핵심 수치 원문 재대조

검증:

- Figure 1의 박스, 화살표, 아이콘, 텍스트가 정상 식별됨
- source Figure 1은 PDF 내부 embedded image를 직접 추출할 수 있으며 최종 Figure 2.1에서 원형 유지
- Chapter 2 `.tex` 본문은 이 단계에서 변경하지 않음

### Step 3. Chapter 2.1 Introduction

현재 상태: **완료**

참고 자료:

- `main_audit_aaai27_v24_candidate.pdf` Abstract와 Introduction
- source의 post-execution phase boundary, query separation, factor-two, non-attestation scope
- `notes\terminology_notation_claim_policy.md`
- `notes\chapter02_content_map.md`

작업:

- `chapters\02_post_execution_validation.tex`의 2.1에 영어 본문 작성
- DP claim이 isolated epsilon이 아니라 unit/adjacency/mechanism/accountant를 요구한다는 문제 제시
- windowed pipeline의 raw-unit/generated-record gap 설명
- same evidence / different query 논리 제시
- finite-registry validator의 입력과 fail-closed output 소개
- 4개 contribution bullet 작성
- evidence-relative, non-attestation, no-new-accountant 범위 명시
- Chapter 3 pre-execution prevention으로 이어지는 전환 명시
- `refs.bib`의 임시 placeholder를 실제 인용문헌 6건으로 교체

현재 인용:

- Dwork and Roth 2014
- Abadi et al. 2016
- Kifer et al. 2020
- Ponomareva et al. 2023
- Near et al. 2025
- Altman et al. 2025

빌드 개선:

- TeX 편집기가 프로젝트 루트에 생성한 stale `aux/bbl/toc` 등이 `build` 결과보다 먼저 읽히는 문제 발견
- `build.ps1`에 명시적 생성물 목록을 `build` 폴더로 이동하는 정규화 단계 추가
- source `.tex`, `.bib`, figures는 이동 대상이 아님

최종 QA:

- 약 688 English words
- 장 표지를 포함해 논문 쪽번호 2--4에 배치
- contribution을 3개 항목으로 압축하고 새 페이지에서 시작하도록 조정
- Chapter 2.1에서 새로 발생한 line overflow 모두 제거
- PDF physical pages 13--15를 PNG로 렌더링하여 육안 확인
- 인용 6건이 bibliography와 정상 연결됨
- 전체 PDF 22쪽, B5 JIS
- LaTeX error, undefined citation, undefined reference 없음
- 남은 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

### Step 4. Chapter 2.2.1 Privacy Units and Windowed DP-SGD

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf` source Eqs. (1)--(3) and Table 1
- Dwork and Roth 2014 DP definition
- Abadi et al. 2016 DP-SGD
- Ponomareva et al. 2023 implementation guidance
- Near et al. 2025 privacy unit and adjacency guidance
- `notes\terminology_notation_claim_policy.md` symbol normalization
- `notes\chapter02_content_map.md` page and asset budget

작업:

- standard $(\varepsilon,\delta)$-DP definition을 Eq. 2.1로 작성
- add/remove와 replace-one adjacency가 교환 가능한 metadata가 아님을 명시
- DP-SGD의 sampler, clipping, Gaussian noise, composition을 설명하되 일반 구성이 자동으로 DP를 보장하는 것으로 표현하지 않음
- raw dataset $\mathcal D$, transformation $\mathcal T$, generated records $\mathcal G$를 정의
- event support와 owner attribution으로 $\kappa_{\mathrm{event}}$, $\kappa_{\mathrm{owner}}$를 Eq. 2.2에 정의
- incidence는 privacy parameter나 worst-case neighboring bound가 아님을 명시
- raw adjacency에서 generated metric으로의 registered stability bound $K$를 Eq. 2.3에 정의
- source Table 1의 네 사례를 Table 2.1로 재작성
- `notation.tex`에 mechanism, event support, owner attribution macro 추가
- B5 표 조판을 위해 `array` package와 ragged-right paragraph columns 적용

Table 2.1 보존 항목:

- fixed-slot replacement + generated replace-one: $K=\kappa$
- fixed-slot replacement + generated add/remove: $K=2\kappa$
- fixed-support insertion/deletion + add/remove: $K=\kappa$
- shifting/reselection/adaptive windows: all-branch bound를 입증하지 못하면 block

최종 QA:

- Eqs. 2.1, 2.2, 2.3 label/reference 정상
- Table 2.1 label과 List of Tables 연결 정상
- table value/row meaning source PDF physical page 3과 재대조
- PDF physical pages 16--19를 렌더링하여 수식·표·글자 크기 육안 확인
- 새 기술 본문과 표에 overfull/underfull warning 없음
- 전체 PDF 25쪽, B5 JIS
- LaTeX error, undefined citation/reference, oversized float 없음

### Step 5. Chapter 2.2.2 Evidence-to-Claim Misalignment

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 completed-run evidence \(\mathcal E\), requested wording \(q_{\mathrm{clm}}\), query-dependent validation chain, private-dependency caveat, non-attestation boundary
- source Table 5의 object-level related-work 구분은 별도 표로 복제하지 않고 두 개의 짧은 문단으로 전환
- Ridgeway et al. 2021: structured DP challenge and artifact review
- Narayan et al. 2015: verifiable differential privacy
- Shamsabadi et al. 2024: cryptographic proof of DP training
- Chua et al. 2024: user-level DP mechanism design
- Kifer et al. 2020과 Altman et al. 2025: implementation audit와 deployment registry
- `notes\chapter02_content_map.md`의 2.2.2 argument order와 Table 5 omission decision

작업:

- frozen evidence와 requested public sentence를 분리하여 evidence-to-claim misalignment를 정의
- `raw adjacency -> stable transformation -> executed mechanism -> accountant -> requested wording` 검증 사슬을 본문에 명시
- generated-record identity/support, preprocessing, selection, contribution cap, runtime, mechanism, accountant가 필요한 증거임을 정리
- labels, normalization, model state, selection, schedule의 private dependency는 public/local/accounted/worst-case-bounded 중 하나여야 함을 명시
- 같은 \(\mathcal E\)에서도 window/event/owner query에 따라 Direct/Convert/Blocked 결과가 달라질 수 있으므로 query가 과학적 입력임을 설명
- registry/audit, program or cryptographic verification, native user-level mechanism과 현재 post-execution wording validator의 차이를 압축하여 서술
- validator가 arbitrary-code proof, 새로운 accountant, evidence-quality ranker, execution attestation이 아님을 절 끝에 재확인
- `refs.bib`에 대표 원문 4건을 추가하고 bibliography 연결 확인
- Section 2.3 cross-reference label을 추가

분량과 시각자료 상태:

- 2.2.2 본문 약 622 English words
- 현재 Chapter 2 누계 시각자료: **Figure 0개, Table 1개**
- Figure 2.1은 2.3.1의 decision fields를 먼저 정의한 뒤 source embedded image를 직접 추출해 삽입할 예정

최종 QA:

- 전체 PDF 28쪽, B5 JIS
- 2.2.2는 논문 본문 쪽번호 8--10, physical PDF pages 19--21에서 육안 확인
- 긴 검증 사슬은 B5 폭에서 두 줄로 자연스럽게 조판되고 잘림 없음
- 추가 인용 4건과 Section 2.3 cross-reference 정상 연결
- LaTeX error, undefined citation/reference, 새 기술 본문 overfull/underfull warning 없음
- 남은 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

### Step 6. Chapter 2.3.1 Query-Dependent Claim Decision and Figure 2.1

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 `Decision object`, source Eq. decision tuple, Lemma 1, Figure 1
- source Figure 1의 frozen evidence, audit record, validation, verdict, public decision 흐름
- `notes\chapter02_content_map.md`의 source-asset placement와 same-evidence/different-query trace
- 김태훈 교수님 참고 학위논문의 method-section 본문 뒤 formal object와 figure를 배치하는 구조적 리듬
- Technology_v1.14의 B5 JIS 132 mm text block과 Figure/List of Figures 처리 방식

본문 작업:

- 단순 Boolean이 아닌 full decision tuple을 Eq. 2.4로 정의:
  `path, release, assurance, K, epsilon-hat, delta-hat, issue set`
- `Direct`, `Convert`, `Group`, `Underspecified`의 의미를 분리
- path는 derivation type이고 public release 허가가 아님을 명시
- `Allowed`와 `Blocked`를 별도 release field로 정의하고, blocked output에는 copy-ready positive sentence가 없도록 규정
- `assurance`는 confidence score가 아니라 evidence-relative scope와 authority를 기록한다고 설명
- `EVIDENCE_VALIDATED`가 execution attestation이 아님을 decision-field 설명 바로 옆에 배치
- Lemma 2.1 `Query separation`과 영문 proof 작성
- 2.3.2의 registered conversion과 vacuity gate로 직접 이어지는 전환 문단 작성
- narrative order에 맞게 equation inventory를 수정: decision tuple Eq. 2.4, group-privacy consequence Eq. 2.5

Figure 2.1 삽입 및 사용자 검토 후 교정:

- 최초 작업에서는 content map의 redraw 결정을 따라 4단계 TikZ 그림을 만들었으나, 사용자가 원문 그림을 그대로 사용할 것을 재확인하여 해당 재작성본을 제거
- 최종 실제 파일: `figures\ch02_source_evidence_validation.png`
- 원문 PDF physical page 4의 유일한 embedded image object를 `pdfimages`로 직접 추출
- 원문 자산 규격: 1632 by 765 pixels, RGB 8 bpc, source 약 291 ppi
- 화면 캡처, 페이지 렌더링 crop, 도형 재작성 없이 원문 5개 패널·아이콘·문구·색상·순서를 그대로 보존
- B5 132 mm text width에 맞추어 비율 유지 배치
- 같은 WISDM evidence의 window/event/owner 판정은 원문 그림 내부 표현을 그대로 사용
- Figure caption을 결과 설명 문단에서 간결한 제목형 문구로 변경:
  `Evidence-bound validation and query separation for one frozen WISDM execution`
- Table 2.1 caption도 교수님 논문의 List of Figures/Tables 형식에 맞춰 짧은 제목형 문구로 변경

조판 기반 변경:

- `thesis.tex`에 `amsthm`, `float`와 chapter-numbered lemma/proposition/corollary 환경 추가
- source figure가 본문 정의 뒤에 정확히 놓이도록 `[H]` placement 사용
- TikZ 재작성본 제거에 따라 `tikz` package와 작업본 class의 임시 `xcolor` 변경도 제거
- 원본 `template_reference\sgmeta_report.cls`는 변경하지 않음

분량과 시각자료 상태:

- 2.3.1 본문 약 516 English words plus concise Figure 2.1 caption
- 현재 Chapter 2 누계 시각자료: **Figure 1개, Table 1개**
- Figure 2.1은 논문 본문 쪽번호 12, physical PDF page 23에 본문·Lemma와 함께 배치

최종 QA:

- 전체 PDF 31쪽, B5 JIS
- Eq. 2.4, Lemma 2.1, Figure 2.1 label/reference와 List of Figures 연결 정상
- source PDF page 4의 embedded asset이 정확히 추출되었는지 원본 해상도로 육안 확인
- 원문 source 자체가 raster image object임을 기록하고, 재인코딩이나 screenshot 대신 직접 추출본을 사용
- 최종 Figure 2.1 배치 해상도 314 by 314 ppi 확인
- Table 2.1과 Figure 2.1 caption을 교수님 논문과 같은 짧은 제목형 문구로 교정하고 List of Tables/Figures 갱신
- physical PDF pages 18와 22--24를 160 dpi로 렌더링하여 표, 원문 그림, caption, lemma/proof 흐름 육안 확인
- figure가 lemma/proof 사이에 끼지 않도록 `[H]`로 본문 순서 고정
- LaTeX error, undefined citation/reference, 새 기술 본문과 그림의 overfull/underfull warning 없음
- 남은 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

### Step 7. Chapter 2.3.2 Privacy-Unit Conversion and Claim Blocking

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 source Eq. (4), Proposition 1, factor-two corollary, WISDM same-evidence trace
- Dwork and Roth 2014의 standard group-privacy consequence
- Chapter 2 Eq. 2.3의 registered raw-to-generated distance bound
- Table 2.1의 raw replacement와 generated add/remove versus replace-one metric distinction
- source WISDM 기록의 event incidence 2, public owner cap 600, observed owner maximum 565
- `notes\chapter02_content_map.md`의 2.3.2 required-content order와 non-overclaiming boundary

본문 작업:

- registered distance \(K\)에서의 standard group-privacy consequence를 Eq. 2.5로 작성:
  \(\varepsilon_K=K\varepsilon\),
  \(\delta_K=\delta\sum_{i=0}^{K-1}e^{i\varepsilon}\)
- Eq. 2.5가 새로운 accountant나 theorem이 아니라 base guarantee의 registered consequence임을 명시
- non-finite result, \(\delta_K\geq1\), unregistered derivation에 대한 fail-closed gate 설명
- diagnostic number의 존재와 positive wording 허가를 분리
- Proposition 2.1 `Contract-relative consequence`와 영문 proof 작성
- Proposition의 세 전제 분리:
  1. transformation과 executed mechanism evidence
  2. 모든 raw-adjacent pair에 대한 registered distance checker
  3. generated distance one에서의 registered accountant guarantee
- Corollary 2.1 `Factor-two mapping`과 영문 proof 작성
- fixed-slot raw replacement에서 generated add/remove는 \(K=2\kappa_{\mathrm{event}}\), generated replace-one은 \(K=\kappa_{\mathrm{event}}\)임을 명시
- `Direct`, `Convert`, `Group`, `Underspecified` path와 `Blocked` release의 관계 설명
- WISDM event path는 \(\kappa_{\mathrm{event}}=2\)와 add/remove metric으로 \(K=4\) 사용
- WISDM owner path는 observed 565가 아니라 학습 전 public contribution cap 600을 \(K\)로 사용
- \(\delta_K\geq1\)인 owner 결과는 `Group/Blocked`이며 positive owner-level wording을 내지 않도록 명시
- 2.3.3의 evidence binding과 bundle-only assurance boundary로 직접 연결

시각자료와 캡션 원칙:

- 새 그림이나 표를 추가하지 않음
- 원문 Figure 2.1과 짧은 제목형 caption을 그대로 유지
- 현재 Chapter 2 누계 시각자료: **Figure 1개, Table 1개**

분량과 조판:

- 2.3.2 본문 약 651 English words
- 논문 본문 쪽번호 13--15, physical PDF pages 24--26
- 전체 PDF 33쪽, B5 JIS

최종 QA:

- Eq. 2.5, Proposition 2.1, Corollary 2.1 label/reference와 번호 정상
- physical PDF pages 24--26을 160 dpi로 렌더링하여 수식, theorem/proof, 문단 전환 육안 확인
- 수식 잘림, theorem 분리, 과도한 빈 공간 없음
- LaTeX error, undefined citation/reference, 새 기술 본문 overfull/underfull warning 없음
- 남은 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

빌드 운영 기록:

- 최초 빌드 시 백그라운드 TeX 조판이 root `thesis.aux`를 잠시 점유하여 artifact 이동이 한 번 실패
- 실행 중 TeX 프로세스 종료와 파일 상태를 확인한 뒤 동일 `build.ps1`을 재실행하여 정상 완료
- source 또는 build artifact를 강제 삭제하지 않음

### Step 8. Chapter 2.3.3 Evidence Binding and Assurance Boundary

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 `Evidence-Bound Validator`, fail-closed procedure, binding and lifecycle, validation cost, Proposition 2
- source의 six ordered checks와 full-file/canonical-selected/pipeline/runtime digest 구분
- source threat model: honest-but-fallible producer, stale exports, omissions, inconsistent configuration, unjustified unit restatement
- Narayan et al. 2015와 Shamsabadi et al. 2024의 verifiable/cryptographic execution 방향
- `notes\chapter02_content_map.md`의 2.3.3 required-content order와 exact assurance-boundary sentence

본문 작업:

- evidence bundle 내부에 mapping, identities, support/attribution, preprocessing, population, contribution, selection, schedule, mechanism, runtime, source, accountant, claim contract가 포함됨을 명시
- finite-registry validator의 여섯 ordered checks 작성:
  1. exact types and domains
  2. reopened mapping
  3. registered stability
  4. runtime agreement
  5. registry binding
  6. query path and release
- full-file digest, canonical selected digest, pipeline digest, runtime digest의 역할을 분리
- hash는 substitution/staleness/inconsistency를 탐지하지만 execution truth를 입증하지 않음을 인접 문단에 명시
- mapping bytes \(B\), rows \(R\), selected rows \(S\), attribution entries \(A\), endpoints \(P\)에 대한 validator complexity를 한 번만 기록:
  \(O(B+R+A+S\log S+P\log P)\) time, \(O(R+A)\) memory plus accountant work
- specialized mapping rescan이 complete validator certificate가 아님을 명시
- Proposition 2.2 `Bundle-only non-attestation`과 영문 proof 작성
- honest-but-fallible producer 범위와 malicious consistent fabricator 범위를 분리
- stronger assurance에는 authenticated runtime observation, execution attestation, cryptographic proof가 필요함을 명시
- 마지막 문장을 다음과 같이 고정:
  `EVIDENCE_VALIDATED means that registered observable obligations pass; it does not mean execution fidelity.`

시각자료와 캡션 원칙:

- 새 그림이나 표를 추가하지 않음
- 원문 Figure 2.1과 간결한 caption을 변경하지 않음
- 현재 Chapter 2 누계 시각자료: **Figure 1개, Table 1개**

분량과 조판:

- 2.3.3 본문 약 750 English words
- 논문 본문 쪽번호 15--18, physical PDF pages 26--29
- 전체 PDF 36쪽, B5 JIS

최종 QA:

- Proposition 2.2 label/reference와 번호 정상
- physical PDF pages 26--29를 160 dpi로 렌더링하여 ordered list, complexity, proposition/proof, threat-model transition 육안 확인
- ordered check 5의 긴 bold title에서 발생한 3.02 pt overflow를 `Registry binding`으로 줄여 제거
- LaTeX error, undefined citation/reference, 새 기술 본문 overfull/underfull warning 없음
- 남은 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

빌드 운영 기록:

- 편집기 `latexmk` 자동 조판과 `build.ps1` 정식 조판이 동시에 TOC를 쓰면서 두 번째 pass에서 incomplete TOC를 한 번 읽음
- 자동 조판 프로세스 종료를 확인한 뒤 정식 four-pass build를 재실행하여 TOC와 PDF 정상 복구
- source 파일 손상이나 강제 프로세스 종료는 없었음

페이지 예산 기록:

- Section 2.3.3까지 Chapter 2 본문 쪽번호 18에 도달
- 초기 15--18쪽 목표는 실험 절 삽입 후 초과할 가능성이 있으나, 지금은 개별 절을 임의 삭제하지 않음
- Chapter 2 전체 완성 후 교수님 참고 Chapter 2의 24쪽 분량과 비교하여 중복·공백·표 배치를 통합 조정

### Step 9. Chapter 2.4.1 Controlled Validation and Ablation

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 controlled suite, source Table 2, fixed necessity witnesses, TensorFlow Privacy scope comparison
- 원문 고정 설계: 100 allowed cases와 100 paired non-allow cases
- 원문 8개 행의 unsafe-positive, valid-exact, exact-tuple fraction을 수치 변경 없이 사용
- `notes\chapter02_content_map.md`의 Q1 배치, deterministic coverage 비과장 원칙, concise caption 정책
- TensorFlow Privacy 공식 privacy-statement API 문서를 원 출처로 `refs.bib`에 추가

본문 작업:

- 첫 실험 질문을 weak evidence projection 또는 runtime/vacuity obligation 제거 시 사라지는 claim distinction으로 고정
- 100 allowed cases의 five registered path strata, five boundary/scale levels, four mapping contexts를 설명
- 25개 one-premise mutation operator를 네 parent에 적용하여 100 paired non-allow cases를 구성한 고정 설계를 기록
- FULL output이 아니라 authored manifest가 path, release/assurance, issue, numerical rule의 oracle임을 명시
- 세 metric을 분리 정의:
  1. unsafe-positive fraction: 100 non-allow cases 중 allowed decision 비율
  2. valid-exact fraction: 100 allowed cases 중 path와 privacy pair가 모두 정확한 비율
  3. exact-tuple fraction: 전체 200 cases 중 assurance, block subtype, numerical presence까지 정확한 비율
- 위 fraction은 deterministic authored coverage outcome이며 deployment prevalence나 population error-rate estimate가 아님을 인접 문단에 명시
- B0--B4를 information projection, A1--A2를 obligation ablation으로 설명하고 competitor/leaderboard 해석을 차단
- FULL 결과 `0.00 / 1.00 / 1.00`, independent verifier `20,909/20,909`, expanded contract suite `83/83`을 원문대로 기록
- A1의 12 unsafe positives와 A2의 four vacuous owner positives를 원문대로 기록
- 다섯 necessity witness를 prose로 정리: reopened mapping bytes, registered stability status, runtime noise/trace equality, vacuity gate, exact event wording
- TensorFlow Privacy 0.9.0의 23-case scalar-compatible subset, `7/7`, `15/16`, Boolean-step rejection, `79/79` reproduction checks를 scope comparison으로만 기술
- scalar statement function과 exact supplied tuple을 evidence에서 재검증하는 FULL의 target object가 다르므로 quality ranking이 아님을 명시
- fixed-base vacuity case의 alternate accounting contract가 Eq. 2.5를 supplied pair에 적용한 결과가 아님을 명시
- 마지막 문장에서 2.4.2의 activity data와 mapping-only rescan으로 연결

Table 2.2 작업:

- caption을 짧은 제목형 `Controlled-suite and ablation results`로 고정
- 원문 8개 행과 다섯 열을 모두 LaTeX 표로 재작성; screenshot이나 재구성 그림은 사용하지 않음
- B0--B4, A1--A2, FULL의 수치를 원문과 대조하여 그대로 입력
- FULL의 세 metric만 bold 처리하고, 긴 실험 한계 설명은 caption이 아니라 본문에 배치
- B5 폭에 맞춰 `\footnotesize`, 2.3 pt column separation, 다섯 개 고정 폭 column을 사용
- 조판 결과 Table 2.2는 논문 본문 쪽번호 20의 별도 float page에 잘림 없이 배치됨

분량과 시각자료 상태:

- 2.4.1 본문 및 표 약 769 English words (mechanical count; table text 포함)
- 논문 본문 쪽번호 18--21, physical PDF pages 29--32
- 현재 Chapter 2 누계 시각자료: **Figure 1개, Table 2개**
- 전체 PDF 39쪽, B5 JIS

최종 QA:

- Table 2.2 label/reference, List of Tables, TensorFlow Privacy citation/reference 번호 정상
- physical PDF pages 29--32를 160 dpi로 렌더링하여 subsection opening, metric definitions, table readability, result interpretation, 2.4.2 transition 육안 확인
- 표의 모든 header, 수치, evidence note, bottom rule가 margin 안에 있고 글자 겹침이나 잘림 없음
- LaTeX error, undefined citation/reference, 새 기술 본문과 표의 overfull/underfull warning 없음
- 새 bibliography의 공식 API URL에서 underfull line-break warning이 발생하나 URL이나 본문 조판은 손상되지 않음
- 그 밖의 box warning은 기존 표지·국영문 초록·keyword 자리표시자에서만 발생

빌드 운영 기록:

- 정식 빌드 전 `latexmk`, `pdflatex`, `bibtex` 활성 프로세스가 없음을 확인
- `build.ps1` four-pass build 정상 완료; source 또는 build artifact 강제 삭제 없음

페이지 예산 기록:

- Section 2.4.1까지 Chapter 2 본문 쪽번호 21에 도달
- Chapter 2 전체가 완성되기 전에는 실험 증거를 임의 축약하지 않음
- 2.4.2와 2.5 완료 후 교수님 참고 Chapter 2의 24쪽 분량을 기준으로 prose 중복과 float-page 배치를 함께 검토

### Step 10. Chapter 2.4.2 Activity-Data and Mapping Validation

현재 상태: **완료**

참고 자료와 기준:

- `main_audit_aaai27_v24_candidate.pdf`의 Q2/Q3, source Tables 3--4, WISDM/UCI HAR confirmatory runs, frozen lineage, ExtraSensory intake, Sepsis/Backblaze rescans
- `notes\chapter02_content_map.md`의 exact result ledger와 evidence-authority 제한
- WISDM, UCI HAR, ExtraSensory, PhysioNet/Sepsis, Backblaze의 원 출처 bibliography

본문 작업:

- WISDM의 raw physical lines 1,098,210, canonical events 1,086,465, windows 7,304, length 200, stride 100을 원문대로 기록
- bound run의 Poisson sampling 1/50, 100 steps, noise multiplier 2, base RDP pair를 기록
- 동일 frozen evidence에서 window, raw-event, owner query가 서로 다른 path/release를 갖는다는 Q2를 설명
- five WISDM confirmatory runs와 separately bound UCI HAR five runs를 구분
- WISDM/UCI accuracy와 macro-F1를 모두 integration diagnostics로 한정하고 경쟁 성능 주장 금지
- frozen lineage의 10 incidents, 16 repairs, 176/176 checks, three tamper tests를 controlled development evidence로 설명
- ExtraSensory는 author-operated source-path intake일 뿐 independent operation, privacy validation, DP guarantee가 아님을 명시
- Sepsis 5,000 patient files와 16 mappings, Backblaze 1.02 GB/90 files/27,799,986 rows/318,426 drives를 mapping-rescan 범위로 기록
- non-overlap incidence, ordered/calendar semantics, private selection의 세 premise risk를 설명
- mapping-only 결과가 mechanism, runtime, accountant를 결속하지 않으므로 privacy pair나 positive wording을 내지 않도록 고정

Table 2.3 작업:

- caption: `Query-dependent decisions for one frozen WISDM execution`
- 원문 세 행과 수치를 그대로 LaTeX 표로 재작성
- Window: `-- / 1`, Direct/Allowed, `(0.53683, 10^-6)`
- Raw event: `2 / 4`, Convert/Allowed, `(2.14731, 1.0642 x 10^-5)`
- Owner: observed 565 / registered 600, Group/Blocked, vacuous delta
- owner conversion에는 observed 565가 아니라 public cap K=600을 사용
- Group path와 Blocked release를 서로 독립된 필드로 유지

Table 2.4 작업:

- caption: `Mapping-only diagnostics for Sepsis and Backblaze data`
- source Table 4의 nine rows와 six columns를 수치 변경 없이 재작성
- B5 폭에 맞춘 fixed-width columns와 2.8 pt column separation 사용
- 모든 mapping policy, generated-record count, event/owner incidence, candidate event K를 원문과 재대조
- separate verifier results 128/128과 86/86을 본문에 배치
- caption은 제목형으로 유지하고 mapping-only 한계는 본문에 배치

인용 작업:

- `refs.bib`에 primary/source-local entries 9건 추가: Kwapisz 2011, Anguita 2013, Reyes-Ortiz 2013, Goldberger 2000, Reyna 2019/2020, Backblaze 2025, Loya 2022, Vaizman 2017
- 모든 새 citation이 BibTeX 번호와 정상 연결됨

분량과 조판:

- 2.4.2 약 897 English words (mechanical count; table text 포함)
- 논문 본문 쪽번호 21--24
- Tables 2.3과 2.4는 각각 본문 쪽번호 22와 24에 배치
- physical PDF pages 32--35를 160 dpi로 렌더링하여 육안 확인

최종 QA:

- 첫 build에서 새 본문 overfull 2건과 Table 2.4 header overflow 2건을 탐지
- 문장 분할, diagnostic expression 분리, concise header 사용으로 수치·의미 변경 없이 전부 제거
- Table 2.4 numeric-column 간격을 넓혀 last rows까지 읽기 쉽게 교정
- LaTeX error, undefined citation/reference, 새 기술 본문·표의 overfull/underfull warning 없음

### Step 11. Chapter 2.5 Summary and Transition 및 Chapter 2 전체 QA

현재 상태: **완료**

2.5 본문 작업:

- post-execution validator가 completed run의 허용 가능한 wording을 제한한다는 장 전체 결론 작성
- same execution의 native, converted, blocked claim을 path/release 분리 원칙으로 종합
- finite-registry, evidence-relative, non-attestation boundary를 반복 수치 없이 요약
- 이미 잘못 구성된 route를 사후 진단이 되돌릴 수 없다는 temporal limitation 명시
- 마지막 문단을 Chapter 3의 pre-execution route registration으로 직접 연결
- standalone paper conclusion을 반복하지 않고 diagnosis-to-prevention transition으로 종료
- 2.5 약 265 English words (mechanical count)

Chapter 2 전체 구조와 분량:

- final numbered pages: 2--25, 총 24쪽
- 교수님 참고논문의 Chapter 2는 pages 7--30, 총 24쪽
- page size는 작업본 B5 JIS, 참고논문 A4로 서로 다르므로 density equivalence가 아니라 structural-length comparison으로만 사용
- 동일 24쪽 규모이므로 현재 단계에서는 분량만을 이유로 증거·표·수식을 임의 삭제하지 않음
- top-level rhythm: Introduction, Background/Problem Setting, Method, Experimental Evaluation, Summary/Transition
- Chapter 2 main-body assets: Figure 1개, Tables 4개, numbered equations 5개
- formal statements: Lemma 1개, Propositions 2개, Corollary 1개

전체 수치·주장 QA:

- Tables 2.1--2.4의 행, 열, 숫자를 source manuscript와 재대조
- Table 2.3의 public owner cap 600, Group/Blocked 분리 재확인
- Table 2.4에 privacy certificate 또는 Direct verdict가 없음을 확인
- FULL fractions가 deterministic coverage이며 population error rate가 아님을 확인
- WISDM/UCI utility가 integration diagnostics로만 표현됨을 확인
- `EVIDENCE_VALIDATED`가 execution attestation으로 서술되지 않음을 확인
- Chapter 3 result나 미완성 future experiment가 Chapter 2에 유입되지 않았음을 확인
- source Table 5는 계획대로 별도 표로 복제하지 않고 관련-work prose에 통합

전체 조판 QA:

- physical PDF pages 13--36의 Chapter 2 page-content density를 검사하여 blank page 없음 확인
- 최신 변경 pages 32--37과 Tables 2.3--2.4를 160 dpi 원본 크기로 육안 검수
- Figure 2.1 직접 추출 asset, four captions, list of figures/tables, equation/formal-statement numbering 정상
- Chapter 2의 모든 label/reference와 20개 bibliography entries 연결 정상
- Chapter 2 기술 본문에 LaTeX error, undefined citation/reference, overfull/underfull box 없음
- 남은 warning은 기존 표지·국영문 초록 자리표시자와 긴 공식 URL의 bibliography 줄바꿈뿐이며 기술 내용과 표 조판에는 영향 없음
- 전체 PDF 44쪽, B5 JIS

### Step 12. Chapter 3 전체 작성 및 QA

현재 상태: **완료**

작업 원문과 역할:

- 1차 원문:
  `C:\Users\SOGANG\Documents\카톨릭대\졸업관련 260720\포함 SCI 연구들\main_aaai27_v43_candidate.pdf`
- 논문 내 역할: Auditable Privacy의 두 번째 핵심 연구이며,
  Chapter 2의 post-execution diagnosis를 pre-execution prevention으로 확장
- 작업 기준표: `notes\chapter03_content_map.md`
- 실제 본문: `chapters\03_evidence_sealed_registration.tex`
- 원문 텍스트 추출본:
  `build\diagnostics\main_registration_source.txt`
- 원문 페이지 PNG와 조판 QA PNG는 검수 후 용량 정리를 위해 삭제했으며,
  원문 PDF와 텍스트 추출본으로 언제든 다시 생성 가능

고정한 장 구조:

- 3.1 From Diagnosis to Prevention
- 3.2 Background and Problem Setting
  - 3.2.1 Owner-Sampled DP-SGD Routes
  - 3.2.2 Route-Label Misalignment
- 3.3 Evidence-Sealed Route Registration
  - 3.3.1 Registered Routes and Complete-Core Seal
  - 3.3.2 Generic Dispatch, Execution, and Evidence Binding
  - 3.3.3 Conditional Privacy Scope
- 3.4 Experimental Evaluation
  - 3.4.1 Lifecycle and Handler-Splice Ablations
  - 3.4.2 Route Separation, Extension, and Verification
- 3.5 Synthesis of Auditable Privacy

학위논문 재구성 기준:

- source paper의 Introduction--Method--Evaluation--Conclusion 병렬 구조를
  그대로 복제하지 않고 Chapter 2에서 Chapter 3으로 이어지는 시간축을 전면화
- 3.1은 “완료된 실행의 잘못된 문구를 막을 수는 있지만 잘못 구성된 실행을
  되돌릴 수 없다”는 Chapter 2의 마지막 명제를 직접 이어받음
- 3.5는 별도 논문의 독립 Conclusion이 아니라 Chapter 2와 3을
  diagnosis/prevention으로 결속하는 Auditable Privacy synthesis로 작성
- source Table 5의 related-work positioning matrix는 별도 표로 복제하지 않고
  3.2.2의 sampling/accounting, formal/typed DP, testing, provenance/build,
  attestation 비교 문단에 통합
- 전체 technical body, section title, caption, table text는 영어로 작성
- 학교 양식의 표·그림 접두어는 템플릿에 따라 한글 “표”, “그림”으로 유지

핵심 내용 구현:

- owner dataset, single-owner attribution, owner-local policy, fixed public
  preprocessing 조건을 명시
- 공통 clipped owner update를 Equation 3.1로 정리
- Route P:
  Bernoulli owner sampling, add/remove adjacency, sensitivity C, empty-step noise
- Route F:
  SRSWOR fixed-size owner sampling, replace-one adjacency, sensitivity 2C,
  normalized noise sigma/2
- Route A:
  per-epoch uniform k-of-t allocation, add/remove adjacency, coupled participation,
  empty-step noise
- public-plan invariance와 private realized identity/assignment 분리를 명시
- route identity를
  rho=(adjacency, sampler, sensitivity, Gaussian scale, accountant, executor)로
  고정
- complete handler core, seven obligations, noncircular report binding,
  exact registration predicate를 Equations 3.3--3.5로 작성
- generic compilation judgment과 execution/artifact judgment를
  Equations 3.6--3.7로 작성
- handler-local conservative extension을 Lemma 3.1로 작성하고,
  새 handler의 privacy나 semantic correctness를 뜻하지 않도록 범위를 제한
- handler lattice와 contract lattice를 Equations 3.8--3.9로 작성

표기 통일:

- route identity: rho
- routes: P, F, A
- sampling rate: p_samp
- training steps: T_train
- owner population: N_own
- allocation count: k_alloc
- epoch count: E_ep
- complete core: C(h)
- route registry: R_route
- route evidence: E_rho
- exact contract: x_ctr
- public plan/private binding: P_rho, B_priv
- Chapter 2의 K와 충돌을 막기 위해 source paper의 K(h)를 본문에서 C(h)로
  정규화

Figure 3.1:

- 최종 파일:
  `figures\ch03_source_sealed_compilation.png`
- source PDF에 embedded된 유일한 raster image를 직접 추출해 복사
- 크기 2145 x 609, RGB
- redraw, crop, relabel, 새 annotation 없음
- caption:
  “Sealed compilation path and handler-splice counterfactual”
- source extraction과 최종 그림의 SHA-256이 동일:
  `6F869A5B77D0F8471649874E48B49E5126D065616F9829309364F6188FC35C51`
- 원본 그림 내부의 K(h_rho)는 그대로 보존하고,
  본문에서 dissertation notation C(h)와 같은 대상임을 명시
- 최종 쪽번호 36에 배치되어 3.3.1 직후에 나타나도록 float 수정

LaTeX 표:

- Table 3.1 “Route-specific registration contract surface”
  - source Table 1의 Privacy, Mechanism, Data, Program, Artifacts 5개 행 보존
  - 최종 쪽번호 32
- Table 3.2 “Lifecycle predicates and persisted evidence”
  - source Table 2의 registration, contract, data, accounting, loop, links 보존
  - 최종 쪽번호 38
- Table 3.3 “Registered owner-sampled DP-SGD research profiles”
  - source Table 3의 9개 dataset-route profile 및 모든 수치 보존
  - 최종 쪽번호 43
- Table 3.4 “Evidence artifacts, verifiers, and exact denominators”
  - source Table 4의 8개 evidence row와 exact denominator 보존
  - 최종 쪽번호 47
  - 최초 scriptsize 조판을 footnotesize로 키우고 column wrapping을 수정
- 모든 표는 screenshot이 아니라 LaTeX로 재작성
- caption은 짧은 title-style로 유지하고 한계·설명은 본문으로 이동

원문 수치 대조:

- route-profile sigma:
  1.269710, 3.806632, 1.229575,
  1.251026, 3.838696, 1.090646,
  0.857344, 2.380381, 0.776173
- lifecycle:
  36/52 vs. 52/52; 5/8 vs. 8/8
- handler registration:
  weak 378/378 accepted; source-bound pre-seal 42/378 accepted;
  complete core 0/378 accepted; endpoint checks 6/6
- matched shell:
  48/48 rejected with 38/38 fields fixed
- extension:
  9/9 compilation; 6/6 old outputs; H 4/4 and 9/9;
  756/756 handler nonendpoints
- contract lattice:
  2,286/2,286 nonendpoints
- Route A replay:
  15/15 runs; 425/425 steps; 9,180/9,180 owner vectors;
  30/30 tensors; 1/1 empty-step fixture
- P/F replay:
  30/30 runs; 850/850 steps; 60/60 tensors
- regression:
  44/44 focused; 154/154 executed; 7/161 skip
- 최종 PDF text에서 위 수치를 포함한 29개 prespecified token을 다시 검색해
  누락 0건 확인

주장 한계:

- complete-core seal은 route-label conformance이며 privacy proof가 아님
- rejected splice는 endpoint label을 유지할 수 없다는 뜻일 뿐
  실제 privacy violation 판정이 아님
- finite lattice의 exhaustive는 지정된 finite universe에만 적용
- SHA-256은 deterministic commitment이며 runtime behavior/random coins의
  attestation이 아님
- honest-source/executor/evidence/verifier 조건을 명시
- PCG64와 `secure_rng=false`를 reproducible research setting으로만 서술
- 모든 benchmark run은 `research_non_release`로 한정
- arbitrary plugin safety, general DP compiler, formal refinement,
  post-hoc privacy-unit conversion을 주장하지 않음
- 45 fixed-seed run은 integration correspondence이며 utility 또는
  population comparison이 아님

인용:

- 기존 Chapter 2 참고문헌을 재사용하고 Chapter 3 primary/source-local
  bibliography entries 15건 추가
- 추가 범위:
  RDP, subsampled RDP, fixed-size minibatches, random allocation, PLD,
  optimal subsampling, multi-user attribution, Opacus, OpenDP, Duet,
  CheckDP, DP-Auditorium, in-toto, reproducible builds,
  DP-SGD implementation auditing
- Chapter 3 unique citation key 27개 연결
- BibTeX warning 0건, undefined citation/reference 0건
- 전체 bibliography는 현재 35개 cited entry

분량 및 조판:

- Chapter 3 source: 약 6,038 English-token word count, 1,032 lines
- 논문 쪽번호 26--50, 총 25쪽
- Chapter 2는 24쪽이므로 두 DP 연구가 개별 장 기준으로 거의 같은 규모
- Figure 1개, Tables 4개, numbered equations 9개, scoped lemma 1개
- Section 3.5는 쪽번호 49에서 시작하고 50에서 Chapter 4로 전환
- 전체 PDF 70 physical pages, B5 JIS
  (515.906 x 728.504 points)
- Chapter 3 전 25쪽을 144 dpi로 렌더링해 contact sheet와 주요 페이지 원본
  크기로 육안 검수
- 표·그림 잘림, margin 침범, caption 오배치, blank technical page 없음
- Figure 3.1이 실험 절까지 밀리는 문제와 Table 3.4가 synthesis 뒤로 밀리는
  문제를 float option 및 section 전 clearpage로 수정
- Chapter 3 technical body의 overfull/underfull box 0건
- 남은 warning은 기존 placeholder 전면부와 긴 bibliography URL 줄바꿈뿐

빌드 중 관찰한 사항:

- TeX editor의 자동 latexmk와 `build.ps1`을 동시에 실행하면 root와
  build에 보조파일이 중복 생성되어 aux lock 또는 transient runaway-label
  오류가 날 수 있음
- 실행 중 TeX 프로세스가 없는 상태에서 `build.ps1`을 단독 재실행하여
  정상 4-pass build 확인
- 다음 세션에서도 formal build 전 auto-build process 유무를 확인할 것
- root의 중복 생성 산출물은 build 폴더로 통합했으며 root에는
  `thesis.tex`만 남음

용량 정리:

- 정리 전 작업 폴더 약 47 MB
- 정리 후 13.94 MB, 총 63 files
- 삭제 대상:
  Chapter 2/3 렌더 QA PNG, Chapter 3 source page PNG,
  temporary contact sheets, root-level duplicate build artifacts
- 정리 후 diagnostic PNG 0개, root duplicate generated file 0개
- 보존 대상:
  final PDF, all LaTeX sources, final inserted figures,
  content maps, terminology policy, source text extractions,
  template reference, bibliography, build script, latest build logs
- 삭제한 것은 원본 PDF나 source가 아니라 모두 재생성 가능한 cache임

최종 산출물:

- `build\thesis.pdf`, 약 4.85 MB
- `chapters\03_evidence_sealed_registration.tex`
- `figures\ch03_source_sealed_compilation.png`
- `notes\chapter03_content_map.md`
- 갱신된 `refs.bib`과 `WORKLOG.md`

다음 경계:

- Chapter 3 완료 지점에서 정지
- 사용자 지시 후 Chapter 4 PP-Mark의 content/evidence map부터 시작
- Chapter 4의 원문 그림은 새로 만들지 말고 source PDF embedded asset을
  먼저 조사·추출
- Chapter 4가 끝나기 전 Chapter 1이나 Chapter 5로 넘어가지 않음

## 10. 빌드 방법과 검증 기준

프로젝트 루트에서 다음을 실행한다.

```powershell
.\build.ps1
```

내부 순서:

1. pdfLaTeX
2. BibTeX
3. pdfLaTeX
4. pdfLaTeX

검증 항목:

- `LaTeX Error`, `Undefined control sequence`, fatal error 없음
- undefined citation/reference 없음
- 새 본문에서 overfull/underfull box가 발생하지 않음
- B5 page size 유지
- 목차, bibliography, cross-reference 갱신
- 변경된 페이지는 PNG 렌더링 후 육안 점검

기존 자리표시자에서 발생하는 경고:

- 표지 제목 한 줄의 underfull box
- 국문·영문 abstract/keyword minipage의 overfull/underfull box

이 경고는 실제 초록과 keywords를 넣을 때 처리한다. 새 기술 본문 경고와 혼동하지 않는다.

최신 정상 빌드: 2026-08-20, `build\thesis.pdf`, 70 physical pages,
B5 JIS. Chapter 3은 논문 쪽번호 26--50의 25쪽이다.

## 11. 아직 입력되지 않은 제출 정보

- 영문 저자명
- 지도교수 성명
- 심사위원 5명
- 제출일
- 심사일
- 졸업 연월
- 국문·영문 핵심어
- 학번 표기 여부
- 국문 제목 최종 승인

사용자의 실제 정보를 받기 전에는 임의로 확정하지 않는다.

## 12. 남은 작업 순서

각 항목은 개별 승인 단위다.

1. [x] Chapter 2.1 Introduction QA 완료 및 보고
2. [x] Chapter 2.2.1 Privacy Units and Windowed DP-SGD 작성
3. [x] Chapter 2.2.2 Evidence-to-Claim Misalignment 작성
4. [x] Chapter 2.3.1 Query-Dependent Claim Decision 작성
5. [x] Figure 2.1 원문 자산 직접 추출·삽입·조판
6. [x] Chapter 2.3.2 Privacy-Unit Conversion and Claim Blocking 작성
7. [x] Chapter 2.3.3 Evidence Binding and Assurance Boundary 작성
8. [x] Chapter 2.4.1 Controlled Validation and Ablation 및 Table 2.2
9. [x] Chapter 2.4.2 Activity-Data and Mapping Validation 및 Tables 2.3--2.4
10. [x] Chapter 2.5 Summary and Transition
11. [x] Chapter 2 전체 수치·인용·그림·표 QA
12. [x] Chapter 3 content/evidence map, Sections 3.1--3.5, Figure 3.1,
    Tables 3.1--3.4, 수치·인용·조판·용량 QA 완료
13. [x] Chapter 4 content map, Sections 4.1--4.5, Figures 4.1--4.4,
    Tables 4.1--4.8, 수식·정리·인용·조판·용량 QA 완료
14. [ ] 핵심 장이 안정된 뒤 Chapter 1 작성
15. [ ] 마지막에 Chapter 5 통합 결론 작성
16. [ ] 국문·영문 초록, keywords, 제출 metadata 입력
17. [ ] 전체 figure/table/equation/reference/TOC 및 학교 양식 QA

## 13. 다음 세션 시작 절차

새 세션 또는 컨텍스트가 끊긴 경우:

1. `WORKLOG.md` 전체를 읽는다.
2. `notes\terminology_notation_claim_policy.md`를 읽는다.
3. 현재 작업 장의 content map을 읽는다.
4. `git` 상태에 의존하지 말고 실제 파일과 최근 수정시각을 확인한다.
5. `build.ps1`로 현재 조판 상태를 확인한다.
6. `현재 단계`에 적힌 작업만 마치고 보고한다.
7. 사용자의 다음 지시 없이 다음 절로 넘어가지 않는다.

## 14. Chapter 4 PP-Mark 완료 기록 (2026-08-20)

### 14.1 완료 범위와 현재 경계

- 완료 파일: `chapters/04_pp_mark.tex`
- 세부 근거표: `notes/chapter04_content_map.md`
- 주 원문: `포함 SCI 연구들/15860_PP_Mark_Provable_and_Pub.pdf`
- 원문 텍스트 보존본: `build/diagnostics/main_ppmark_source.txt`
- Chapter 4 본문 범위: 쪽번호 51--80, 총 30 B5 pages
- 전체 작업본: `build/thesis.pdf`, 102 physical pages, B5 JIS
- 최종 구성: Sections 4.1--4.5, numbered figures 4개, tables 8개,
  numbered equations 16개, assumptions 2개, theorem 1개
- Chapter 1과 Chapter 5는 여전히 자리표시자다. 사용자의 다음 지시 전에는
  두 장으로 넘어가지 않는다.

### 14.2 구조 결정

Chapter 4는 Auditable Privacy의 세 번째 단계가 아니라 Secure Public
Verification의 독립 연구 장으로 유지했다. 장의 논리는 다음 순서로 고정했다.

1. 공개 detector가 독립 검증을 가능하게 하지만 공격 최적화 표면도 공개한다.
2. private API는 공격 표면을 줄이는 대신 중앙 검증기관 의존을 만든다.
3. PP-Mark(score)는 통계적 screening으로만 해석한다.
4. PP-Mark(accept)는 score criterion과 SP1 receipt가 모두 성립할 때만
   declared binding에 대한 verified provenance evidence를 출력한다.
5. 결론은 score와 proof를 합친 claim-aligned interface의 의미와 한계를 닫고,
   Chapter 5의 전체 통합 결론으로 넘긴다.

최종 절 구조:

- 4.1 Introduction
- 4.2 Background and Related Work
  - Generative Image Watermarking and Provenance
  - Forgery Vulnerabilities of Public Detectors
  - Zero-Knowledge Proofs and Verifiable Computation
- 4.3 PP-Mark
  - Context-Bound Binding and Payload Construction
  - Deterministic Sampling and Latent Watermark Embedding
  - Trace Commitment and SP1 Proof Generation
  - Public Verification Modes and Formal Security Guarantee
- 4.4 Experimental Evaluation
  - Experimental Setup and Calibration
  - Forgery Attacks
  - Removal and Transformation Attacks
  - Efficiency, Image Quality, and Model Generalization
- 4.5 Summary

### 14.3 원본 그림 처리 기록

원문 그림은 새로 그리지 않았다. `pdfimages`로 source PDF의 embedded image
object를 추출하고, 선택된 object를 바이트 변경 없이 `figures/`로 복사했다.
최종 파일과 원본 embedded object의 SHA-256이 모두 일치함을 확인했다.

- Figure 4.1: source Figure 1 overview,
  `ch04_source_ppmark_overview.png`, 2338x835
- Figure 4.2: source Figure 11 SPSA trace,
  `ch04_source_spsa_removal.png`, 1200x900
- Figure 4.3: source Figure 6(a)와 6(b)의 원본 plot object 두 개를 LaTeX
  subfigure로 나란히 배치. plot 내부는 수정하지 않음.
- Figure 4.4: source Figure 8 BRISQUE curve,
  `ch04_source_brisque_strength.png`, 1080x756

원문 Figure 12의 512x512 예시 이미지 네 개는 최종 장에 넣지 않았다. 이를
새로 조합하면 source figure layout을 재작성하게 되고, 실험 근거가 이미 quality
curve와 수치로 충분하기 때문이다. 최종 caption은 모두 짧은 제목형으로 두고,
설명과 한계는 본문으로 이동했다. 상세 hash는 `figures/README.md`에 기록했다.

### 14.4 표와 수치 출처

LaTeX 표는 새 실험이나 재계산 결과가 아니라 원문 표의 수치를 그대로 옮겼다.
B5 폭에 맞춰 열 배치만 조정했다.

- Table 4.1: source Table 1, verification-mode threat coverage
- Table 4.2: source Table 2, fabricated binding/compliance bypass
- Table 4.3: source Table 3, TR/GS black-box transfer aggregate
- Table 4.4: source Table 4, SD 2.1 white-box forgery
- Table 4.5: source Table 13, regeneration TPR
- Table 4.6: source Table 14, steganalysis TPR
- Table 4.7: source Table 16, SD 2.1/SDXL transformations
- Table 4.8: source Table 6, end-to-end runtime

핵심 재대조 수치:

- 기본 설정: SD 2.1, DDIM 50 steps, guidance 7.5, alpha 4.0,
  N_tr=1000, k_open=32, clean-calibrated FPR 1%
- fabricated-binding search: image당 10,000 bindings, mean best score 3.88,
  threshold 2.47
- black-box aggregate PP-Mark(score): TR 0.0033, GS 0.0067;
  PP-Mark(accept): 두 조건 모두 observed 0.0000
- SD 2.1 white-box: score FAR 0.76, full-accept FAR observed 0.00
- regeneration: PP-Mark score-mode TPR 0.98/0.98/0.98
- transformations: 0.74--1.00 TPR; 최저값은 SDXL crop-and-resize 0.74
- runtime at N_tr=1000: score path embed 0.89 s, verify 0.72 s;
  SP1 prove 48.572 s, receipt verify 7.589 s, receipt 40.68 MB
- inversion residual mean absolute correlation: SD 2.1 0.30, SDXL 0.54
- SDXL mean score 13.63 vs threshold 2.47; white-box score FAR 0.11
  (95% CI 0.0562--0.1883), full-accept observed 0.00

Table 3의 큰 relative delta는 1개 또는 2개의 작은 PP-Mark denominator에
기반하므로 본문에서는 absolute FAR와 CI를 우선 해석하고 delta는 descriptive로
한정했다.

### 14.5 수식·증명·주장 경계

학위논문 공통 표기를 사용해 source의 충돌 기호를 정리했다.

- source opening set K -> dissertation `\mathcal O`
- source sample count N -> `N_{tr}`
- secret key -> `k_{sec}`
- score/threshold -> `S_{det}`, `\tau`
- trace/receipt -> `\mathcal T_{tr}`, `\pi`
- public statement/private witness -> `x_{pub}`, `\omega_{priv}`

본문은 binding, payload, deterministic sampling, latent embedding,
normalization, score, witness/statement, P1--P4, acceptance predicate, partial
opening bound을 16개 numbered equations로 연결한다. Assumption 4.1은 inversion
residual 조건, Assumption 4.2는 hash/random-oracle/SP1/key 조건이다. Theorem 4.1은
hash collision advantage, SP1 soundness advantage, hypergeometric missed-opening
term을 모두 명시한다.

다음 과장은 차단했다.

- score alone proves provenance
- SP1 circuit proves or re-executes full diffusion
- receipt proves semantic truth, human authorship, or legal identity
- exact-image receipt survives benign edits
- observed zero FAR means universal zero probability
- two architectures imply architecture-agnostic generalization
- PP-Mark prevents every attack

허용 문구는 `PP-Mark-accepted`, `verified relative to the declared binding`,
`verifiable provenance evidence`, `zero observed FAR in the evaluated sample`,
`under the stated assumptions and partial-opening bound`로 제한했다.

### 14.6 참고문헌 처리

Chapter 4에서 실제 인용하는 primary/official reference 24개를 `refs.bib`에
추가했다. watermark baselines, detector attacks, regeneration, C2PA, Merkle,
SP1, VeriLLM, random-oracle model, Hoeffding, SPSA를 포함한다. 최종 BibTeX와
세 번의 pdfLaTeX pass에서 undefined citation/reference와 BibTeX warning은
없다.

### 14.7 조판과 시각 QA

- `build.ps1` 최종 성공
- `LaTeX Error`, fatal error, undefined control sequence 없음
- undefined citation/reference 없음
- Chapter 4 technical-body overfull/underfull box 없음
- 남은 box warning은 기존 frontmatter 자리표시자와 bibliography의 긴 URL뿐임
- page size 515.906 x 728.504 pt, B5 JIS 유지
- Chapter 4 전체 physical pages 62--91을 raster render하여 육안 검사
- figure/table 순서, caption, table 열 간격, theorem/equation 폭, 마지막 summary
  페이지까지 확인
- 문장 중간에 float가 끼는 문제를 수정했고, Summary 마지막 단락만 고립되던
  페이지를 축약하여 Chapter 4를 30쪽으로 안정화함

### 14.8 보존 파일과 다음 작업

보존:

- `chapters/04_pp_mark.tex`
- `notes/chapter04_content_map.md`
- `figures/ch04_source_*.png` 5개
- `figures/README.md`의 source/hash ledger
- `refs.bib`
- `build/diagnostics/main_ppmark_source.txt`
- `build/thesis.pdf`와 최신 build logs

삭제 대상은 원본 PDF가 아니라 언제든 재생성 가능한 Chapter 4 QA page PNG,
사용하지 않은 extracted embedded images, 임시 layout text뿐이다.

다음 승인 단위는 Chapter 1 Introduction이다. Chapter 1에서는 이더리움 이종
그래프, 메타버스 공간 인증, LLM 데이터 증강을 연구 궤적의 build-up으로만
사용하고, Chapters 2--4의 핵심 연구를 Claim-Aligned Assurance의 두 축으로
연결한다. 사용자가 `다음 진행`이라고 지시하기 전에는 Chapter 1을 작성하지
않는다.

## 15. Chapter 1 Introduction 완료 (2026-08-20)

### 15.1 승인된 작업 경계

사용자의 `진행` 지시에 따라 Chapter 1 Introduction을 하나의 승인
단위로 작성했다. 이번 단계에서 Chapter 5, 국·영문 초록, 감사의 글,
제출 메타데이터는 작성하지 않았다.

### 15.2 재확인한 구조와 원문

구조 기준:

- `build/diagnostics/reference_dissertation.txt`
- 김태훈 교수님 학위논문 Chapter 1은 본문 1--6쪽
- 구조는 `Research Goals and Significance` 후 각 기술 장의 요약으로 이어짐
- 참고 학위논문의 Chapter 1에는 그림이 없음

서론용 선행연구 원문:

- `03_Ko.pdf` -> `build/diagnostics/ch01_ethereum_source.txt`
- `Space_Authentication_in_the_Metaverse_A_Blockchain-Based_User-Centric_Approach.pdf`
  -> `build/diagnostics/ch01_metaverse_source.txt`
- `LLM-Based_Persona-Driven_Text_Data_Augmentation.pdf`
  -> `build/diagnostics/ch01_llm_source.txt`

세 연구는 다음 기능으로만 사용했다.

- Ethereum/GATv2: 공개 거래 기록과 숨은 스마트컨트랙트 역할 사이의
  해석 간극
- Metaverse authentication: 중앙기관 의존을 줄이는 사용자 토큰·스마트
  컨트랙트 검증 구조
- LLM augmentation: 생성량이 아니라 역할·행동·언어 규칙의 도메인 맥락
  보존

실험 수치, 모델 구조, 원본 표·그림을 별도 장처럼 확장하지 않았다.

### 15.3 최종 Chapter 1 구조

- 1.1 Research Goals and Significance
- 1.2 Research Background and Trajectory
  - 1.2.1 Interpreting Hidden Roles in Anonymous Web3 Systems
  - 1.2.2 Reducing Centralized Trust in Metaverse Authentication
  - 1.2.3 Preserving Domain Context in LLM Data Augmentation
- 1.3 Claim-Aligned Assurance for Trustworthy AI
  - 1.3.1 Claim--Mechanism--Evidence Alignment
  - 1.3.2 Assurance Boundaries and Fail-Closed Decisions
- 1.4 Research Questions and Contributions
  - 1.4.1 Auditable Privacy: Diagnosis and Prevention
  - 1.4.2 Secure Public Verification: Detection and Verifiable Provenance
- 1.5 Dissertation Organization

초기 완성본은 약 3,253 words, 14 numbered pages였다. 선행연구와 후속
장의 방법 설명이 반복되는 부분을 축약해 약 2,647 words, 12 numbered B5
pages로 안정화했다. 교수님 서론 6쪽보다 길지만, 여기에는 선행연구
3편의 궤적과 논문 전체의 상위 프레임이 추가되므로 과하지 않은 분량으로
판단했다.

### 15.4 통합 개념과 주장 경계

핵심 결정을 다음과 같이 소개했다.

`CAA(c; M, I, E) in {Release(c), Qualify(c'), Block}, c' < c`

공통 원리:

1. Claim specification
2. Mechanism binding
3. Evidence-bound decision
4. Fail-closed assurance

연구질문은 RQ1 사후 privacy wording, RQ2 사전 route binding, RQ3 공개
provenance verification, RQ4 통합 원리와 stopping boundary로 고정했다.

다음 과장을 차단했다.

- 증거 일관성은 execution attestation이 아님
- 유한 route predicate는 arbitrary-code verification이 아님
- 프라이버시 문구 차단은 privacy 부재의 증명이 아님
- hybrid label 거부는 hybrid 비프라이버시의 증명이 아님
- PP-Mark(score)는 statistical screening일 뿐임
- PP-Mark(accept)는 선언된 predicate 상대적 증거이며 진실성·저작자·법적
  신원·전체 diffusion execution 증명이 아님

### 15.5 표·그림·참고문헌 처리

- Table 1.1은 Auditable Privacy와 Secure Public Verification의 claim target,
  mechanism, interface, evidence, fail-closed outcome, boundary를 비교하는
  dissertation-level synthesis이다.
- caption과 본문에 empirical source result가 아님을 명시했다.
- Chapter 1에는 원본 그림을 넣지 않았다. 참고 학위논문의 Introduction에도
  그림이 없고, 선행연구는 build-up으로만 사용하기 때문이다.
- 새 그림을 원 논문 그림처럼 재제작하거나 임의 제목을 붙이지 않았다.
- `refs.bib`에 `ko2024ethereum`, `seo2024space`, `jeong2025persona`를 추가했다.
- 저널, volume, pages, year, DOI는 각 supplied PDF의 출판 정보를 기준으로
  확인했다.

### 15.6 조판과 시각 QA

- `build.ps1` 최종 성공
- 최종 PDF: 113 physical pages, B5 JIS, 5,723,362 bytes
- Chapter 1: numbered pages 1--12 / physical pages 12--23
- Chapter 2 시작: numbered page 13 / physical page 24
- 1 numbered equation, 1 table, 3 preceding-study citation keys
- undefined citation/reference, BibTeX warning, LaTeX/fatal error 없음
- Chapter 1 technical-body overfull/underfull box warning 없음
- 남은 box warning은 frontmatter 자리표시자와 bibliography 긴 URL만 해당
- physical pages 12--23을 모두 raster render하여 시각 검수

QA에서 수정한 항목:

- math delimiter 1곳
- Chapter 1 overfull line 2곳
- RQ3의 마지막 단어가 다음 쪽으로 고립되던 배치
- Section 1.1 마지막 단어 1개가 다음 쪽에 남던 배치
- Table 1.1이 문장 중간에 들어가던 float 순서
- Section 1.4 직전 최종 한 줄이 다음 쪽 상단에 고립되던 배치
- 반복 설명을 줄여 14쪽에서 12쪽으로 축약

### 15.7 현재 pagination과 보존 파일

Chapter 1 추가 후 현재 본문 쪽수:

- Chapter 1: 1--12, 12 pages
- Chapter 2: 13--36, 24 pages
- Chapter 3: 37--61, 25 pages
- Chapter 4: 62--91, 30 pages
- Chapter 5: 92쪽에서 자리표시자로 시작

보존:

- `chapters/01_introduction.tex`
- `notes/chapter01_content_map.md`
- `build/diagnostics/ch01_ethereum_source.txt`
- `build/diagnostics/ch01_metaverse_source.txt`
- `build/diagnostics/ch01_llm_source.txt`
- `build/diagnostics/thesis_after_ch01.txt`
- `refs.bib`
- `build/thesis.pdf`과 최신 build logs

임시 Chapter 1 QA PNG만 완전 삭제했다. 4개 QA directory에서 약 7.33 MiB를
제거했고 현재 작업본 전체 용량은 15.81 MiB이다. 삭제한 파일은 복구하지
않았으나 최종 PDF에서 언제든 재생성할 수 있다.

### 15.8 다음 작업 경계

다음 승인 단위는 Chapter 5 Conclusion이다.

- 5.1 Summary
- 5.2 Integrated Discussion and Contributions
  - Auditable Privacy
  - Secure Public Verification
  - Claim-Aligned Assurance Design Principles
- 5.3 Limitations, Broader Impact, and Future Work

Chapter 5에서는 Chapters 2--4의 독립 conclusion을 반복하지 않고, 세 연구가 왜
하나의 박사학위논문인지를 닫는다. 미완성 추가 실험은 완료된 결과로 기술하지
않고 future experimental validation으로만 분리한다. 사용자의 다음 지시 전에는
Chapter 5를 작성하지 않는다.

## 16. Chapter 5 Conclusion 완료 (2026-08-20)

### 16.1 승인된 작업 경계

사용자의 `이어서 진행` 지시에 따라 Chapter 5 Conclusion을 하나의 승인 단위로
작성하고 전체 조판·시각 QA·기록·임시 파일 정리까지 완료했다. 이번 단계에서는
국·영문 초록, 핵심어, 감사의 글, 제출자·지도교수·심사위원·날짜 메타데이터를
작성하지 않았다.

### 16.2 구조 기준과 직접 참고 자료

구조 기준은 다음 두 자료다.

- `김태훈 교수님_박사졸논 자료/학위논문_최종.pdf`
- `build/diagnostics/reference_dissertation.txt`

김태훈 교수님 학위논문의 Conclusion은 numbered pages 95--98의 4쪽이며,
`Summary`, `Broader Impact and Future Work`, `Acknowledgement`로 구성되고
그림·표가 없다. 현재 학위논문은 세 독립 연구의 상위 통합 논리를 명시해야 하므로
Summary와 broader impact 사이에 `Integrated Discussion and Contributions`를
두되, 각 연구의 paper-style conclusion을 다시 반복하지 않는 방식을 택했다.
Acknowledgement는 결론 안에 넣지 않고 학교 템플릿의 별도 frontmatter 파일을
유지했다.

결론의 내용 근거는 새 외부 자료가 아니라 이미 QA가 끝난 다음 파일이다.

- `chapters/02_post_execution_validation.tex`
- `chapters/03_evidence_sealed_registration.tex`
- `chapters/04_pp_mark.tex`
- `notes/chapter02_content_map.md`
- `notes/chapter03_content_map.md`
- `notes/chapter04_content_map.md`
- `chapters/01_introduction.tex`
- `notes/terminology_notation_claim_policy.md`

Ethereum 이종 그래프, metaverse authentication, LLM augmentation 연구는
Chapter 1의 연구 궤적 역할로 이미 제한했으므로 결론에서 다시 요약하지 않았다.

### 16.3 최종 구조

- 5.1 Summary
- 5.2 Integrated Discussion and Contributions
  - 5.2.1 Auditable Privacy: Diagnosis and Prevention
  - 5.2.2 Secure Public Verification
  - 5.2.3 Claim-Aligned Assurance Design Principles
- 5.3 Limitations, Broader Impact, and Future Work
  - 5.3.1 Limitations and Assurance Boundaries
  - 5.3.2 Broader Impact
  - 5.3.3 Future Experimental Validation

5.1은 수행한 연구를 대표 결과만으로 요약하고, 5.2는 세 연구가 하나의 박사논문인
이유를 설명하며, 5.3은 이미 증명·평가한 범위와 아직 수행하지 않은 실험을
분리한다.

### 16.4 통합 논리와 보존한 대표 결과

Auditable Privacy는 다음 두 시점을 상호 보완적으로 연결한다.

- Chapter 2: frozen execution evidence로부터 허용 가능한 public wording을
  사후 진단한다.
- Chapter 3: requested route identity와 구성요소의 불일치를 실행 전에
  예방한다.

사후 validator는 완료된 실행을 고칠 수 없고, 사전 registry는 이후의 모든
query-dependent claim을 결정할 수 없다. 어느 쪽도 execution attestation이
아니다. 이 차이를 유지한 상태에서 diagnosis와 prevention을 하나의 privacy
assurance 축으로 닫았다.

Secure Public Verification에서는 PP-Mark score와 receipt를 서로 대체할 수 없는
두 근거로 유지했다. score는 statistical screen이고 full acceptance는 동일한
context와 exact image에 대한 detectable signal과 valid receipt를 모두 요구한다.

결론에 반복한 수치는 각 장을 대표하는 최소 checkpoint로 제한했다.

- Chapter 2: 100 allowed control cases expected path, 100 paired non-allow cases
  rejected
- Chapter 3: 154 executed regression tests passed, 7 raw-data-scoped tests
  explicitly skipped
- Chapter 4: 10,000-binding search, evaluated forgery conditions에서 no observed
  full acceptance
- Chapter 4 score mode: transformation TPR 0.74--1.00, regeneration 0.98
- Chapter 4 limitation: proving 약 48.6 s, selected receipt 40.68 MB

새 실험이나 새 수치를 만들지 않았고 `zero observed`를 universal zero
probability로 확대하지 않았다.

### 16.5 Claim-Aligned Assurance와 주장 경계

최종 통합 원리는 다음 네 가지로 고정했다.

1. Claim specification
2. Mechanism binding
3. Evidence-bound decision
4. Fail-closed assurance

공통 관계는 `claim target -- operative mechanism -- verification interface --
evidence`이다. interface를 단순 표시 계층이 아니라 어떤 positive sentence를
허용하는지 결정하는 operational boundary로 설명했다.

다음 과장을 차단했다.

- Claim-Aligned Assurance는 모든 trustworthy-AI claim의 universal certification이
  아님
- privacy evidence consistency는 authenticated execution이 아님
- finite registry는 arbitrary-program semantics proof가 아님
- DP 두 연구는 underlying DP proof와 accountant를 대체하지 않음
- PP-Mark receipt는 semantic truth, authorship, legal identity, full diffusion
  execution을 증명하지 않음
- SD 2.1과 SDXL 평가는 architecture-independent generalization이 아님
- exact-image proof는 edit 후 그대로 유효하지 않음
- adaptive public-query control은 현재 완료된 PP-Mark 결과가 아님

### 16.6 그림·표·수식·인용 결정

Chapter 5에는 새 그림, 표, 수식, 외부 인용을 넣지 않았다. 교수님 결론에도
그림·표가 없으며, Chapters 2--4의 원본 그림과 검증된 결과표를 결론에서 다시
복제하면 분량과 시각 비중만 늘어나기 때문이다. 네 가지 원리는 새로운 empirical
comparison이 아니므로 짧은 numbered list로만 제시했다. 원 논문 그림을 다시
그리거나 임의 캡션을 붙인 항목은 없다.

최종 Chapter 5 통계:

- 약 1,481 English words (TeX command/comment 제거 후 근사치)
- 3 sections, 6 subsections
- 7 internal chapter references
- 0 citations, 0 equations, 0 figures, 0 tables

### 16.7 분량 조정과 조판 QA

첫 완성본은 약 2,200 words, 9 numbered pages였고 전체 PDF는 121 physical
pages였다. 본문 장의 방법을 다시 풀어 쓰는 문장, 중복 limitation, 반복 future
work를 줄여 약 1,481 words, 7 numbered B5 pages로 축약했다. 교수님 결론 4쪽보다
길지만 추가된 3쪽은 세 연구의 통합 기여, 네 설계 원리와 공통 한계에 해당한다.

- `build.ps1` 최종 성공
- 최종 작업 PDF: 119 physical pages, B5 JIS, 5,738,228 bytes
- Chapter 5: numbered pages 92--98 / physical pages 103--109
- `LaTeX Error`, fatal error, undefined control sequence 없음
- undefined citation/reference 없음
- Chapter 5 overfull/underfull box warning 없음
- 남은 box warning은 기존 frontmatter placeholder와 bibliography 긴 URL뿐임
- physical pages 103--109를 110 dpi PNG로 두 차례 render하여 전 페이지 육안 검사

QA에서 수정한 항목:

- 긴 영문 compound phrase 때문에 생긴 Chapter 5 overfull line 3곳
- 5.2.3 제목 뒤 안내문만 남고 원칙 목록이 다음 쪽으로 분리되던 배치
- 반복 요약을 줄여 9쪽에서 7쪽으로 축약
- 마지막 쪽의 자연스러운 결론 여백은 검증되지 않은 내용으로 채우지 않음

### 16.8 보존 파일, 삭제 파일과 현재 용량

보존:

- `chapters/05_conclusion.tex`
- `notes/chapter05_content_map.md`
- `build/diagnostics/thesis_after_ch05_final.txt`
- `build/thesis.pdf`와 최종 build logs
- Chapters 1--4의 source text, evidence map, final source figures와 hash ledger

삭제:

- `build/diagnostics/ch05_qa_run1`
- `build/diagnostics/ch05_qa_run2`
- `build/diagnostics/thesis_after_ch05.txt`
- `build/diagnostics/thesis_after_ch05_compact.txt`

위 항목은 최종 PDF에서 다시 생성 가능한 QA PNG와 중간 layout text이며 총
2,435,658 bytes를 완전 삭제했다. 원본 PDF나 최종 그림은 삭제하지 않았다.
현재 작업본은 76 files, 16,840,581 bytes(약 16.06 MiB)이다.

현재 본문 pagination:

- Chapter 1: 1--12, 12 pages
- Chapter 2: 13--36, 24 pages
- Chapter 3: 37--61, 25 pages
- Chapter 4: 62--91, 30 pages
- Chapter 5: 92--98, 7 pages

### 16.9 다음 작업 경계

Chapters 1--5 본문은 모두 완료됐다. 다음 승인 단위는 frontmatter와 제출 통합이다.

- Korean abstract와 keywords
- English abstract와 keywords
- 영문 저자명, 지도교수명, 심사위원 5명, 제출·심사·졸업 날짜
- optional acknowledgement
- 전 장 terminology/notation/cross-reference/reference/pagination 최종 일관성 QA
- 학교 제출 규격과 표지·인준지 최종 확인

개인 메타데이터는 실제 값을 받기 전까지 자리표시자를 유지한다. 사용자가 다음
진행을 지시하기 전에는 위 항목을 작성하지 않는다.

## 17. 국·영문 초록과 키워드 완료 (2026-08-20)

### 17.1 승인된 작업 경계

사용자의 `진행` 지시에 따라 국문 초록, 영문 ABSTRACT와 양 언어 키워드를 하나의
승인 단위로 작성하고 전체 빌드·양언어 일치 검토·시각 QA·기록·정리까지
완료했다. 영문 저자명, 지도교수, 심사위원, 날짜와 감사의 글은 실제 정보가
없으므로 이번 단계에서 수정하지 않았다.

### 17.2 형식 기준과 참고 초록

최우선 형식 기준은 `template_reference/guide.tex`과 현재
`sgmeta_report.cls`이다. Technology v1.14는 다음을 요구한다.

- 국문은 `abstract`, 영문은 후면 `summary` 환경에 작성
- 초록 첫 들여쓰기를 위한 `\par` 유지
- 페이지 하단에 높이 20 mm의 keyword minipage 유지
- 국문 `키워드`, 영문 `keywords` 표기

김태훈 교수님 자료 폴더에서 최종 합본뿐 아니라 다음 독립 파일도 재확인했다.

- `김태훈 교수님_박사졸논 자료/국문초록.pdf`
- `김태훈 교수님_박사졸논 자료/영문초록.pdf`

두 참고 초록은 각각 A4 1쪽이며 문제 배경, 연구축별 방법, 통합 결과, 주제어의
흐름을 갖는다. 교수님 구형 형식은 초록 안에 논문 제목을 반복하지만 현
Technology 템플릿 예제는 제목 줄을 넣지 않으므로 현재 템플릿을 우선했다. 별도
영문 PDF는 embedded font 때문에 `pdftotext`가 본문을 추출하지 못하여 직접 raster
render한 페이지로 문단과 키워드 배치를 확인했다. 국문 추출문과 영문 추출 시도
결과는 각각 `build/diagnostics/reference_abstract_ko.txt`와
`reference_abstract_en.txt`에 보존했다.

### 17.3 초록 내용 구성

두 초록은 같은 세 단계로 구성했다.

1. claim target, mechanism, interface, evidence가 어긋날 때 수치나 검출 결과만으로
   보장이 성립하지 않는다는 문제와 Claim-Aligned Assurance의 두 축을 제시한다.
2. Auditable Privacy에서 Chapter 2의 post-execution diagnosis와 Chapter 3의
   pre-execution prevention을 구분한다.
3. PP-Mark의 score/full-accept 분리와 Secure Public Verification을 요약한 뒤
   claim specification, mechanism binding, evidence-bound decision,
   fail-closed assurance를 통합 원리로 닫는다.

국문은 줄 밀도를 활용해 첫 privacy 연구의 direct·converted·grouped·blocked
판정을 짧게 명시했다. 영문은 같은 의미를 requested wording check와 unsupported
claim block으로 압축했다. 이는 세부 수준의 차이이며 주장 차이는 아니다.

### 17.4 양언어 수치 대응

국문과 영문에 포함된 숫자를 직접 추출해 대조했다.

- Chapter 2: 200 control decisions
  - 본문의 100 allowed expected paths와 100 paired non-allow rejections를 합친 것
- Chapter 3: 154 executed tests passed
- Chapter 3: 7 raw-data-scoped tests excluded/skipped
- Chapter 4: 10,000 candidate-binding search
- Chapter 4: score-mode transformation TPR 0.74--1.00

두 언어에서 값과 연구 귀속이 모두 일치한다. `no full acceptance was observed`는
평가한 forgery conditions에만 한정하며 보편적 0 확률로 확대하지 않았다. 새
실험, 새 수치, 새 참고문헌은 추가하지 않았다.

### 17.5 키워드

최종 키워드는 7개이며 순서와 의미를 양언어에서 일치시켰다.

1. 주장 정합 보증 / Claim-Aligned Assurance
2. 감사 가능한 프라이버시 / Auditable Privacy
3. 차분 프라이버시 / Differential Privacy
4. DP-SGD / DP-SGD
5. 안전한 공개 검증 / Secure Public Verification
6. 생성형 AI 출처 / Generative-AI Provenance
7. 영지식 증명 / Zero-Knowledge Proof

키워드 블록 앞에 `\noindent`를 추가하여 full-width minipage가 문단 들여쓰기만큼
넘치던 17 pt overfull을 제거했다. 국문은 짧은 마지막 줄의 과도한 자간 확장을
막기 위해 ragged-right로 두고, 영문은 기본 정렬을 유지해 `Zero-Knowledge Proof`
중 `Proof`만 세 번째 줄에 고립되지 않도록 했다.

### 17.6 주장 경계

다음 과장을 초록에서 차단했다.

- Claim-Aligned Assurance는 universal certifier가 아님
- privacy 두 연구는 execution attestation이나 general-program verification이 아님
- PP-Mark score alone은 provenance proof가 아님
- receipt는 semantic truth, authorship, legal identity, full diffusion execution을
  증명하지 않음
- observed zero full acceptance는 universal zero forgery probability가 아님
- score-mode 0.74--1.00 TPR은 보고된 transformation 범위의 결과임

초록의 제한된 길이 때문에 모든 본문 limitation을 반복하지 않았지만, positive
claim은 jointly supporting objects, mechanisms, interfaces, evidence 범위로
제한한다는 최종 문장으로 전체 stopping boundary를 유지했다.

### 17.7 분량 조정과 조판 QA

초기 완성본은 국문 약 303 eojeol, 영문 333 words였으며 각각 2쪽이었다. 참고
학위논문의 1-page abstract 구조와 B5 템플릿의 20 mm keyword 영역을 맞추기 위해
다음과 같이 축약했다.

- 국문: 303 -> 223 -> 196 eojeol, 최종 848 characters, 3 paragraphs
- 영문: 333 -> 255 -> 228 -> 208 -> 183 -> 171 words, 3 paragraphs

긴 구성요소 열거, 반복 non-claim과 본문에 이미 있는 방법 설명을 줄였고, 세 연구
구분·대표 수치·네 통합 원리는 유지했다.

최종 QA:

- `build.ps1` 성공
- 전체 PDF: 119 physical pages, B5 JIS, 6,193,825 bytes
- PDF SHA-256:
  `BB2D10FCC6A1AE2D6A88D72388A1B18A62EFAE4DDDCBAFD0E9B25964ED9BA2F8`
- 국문 초록: physical page 11 / roman vii의 1쪽
- 영문 ABSTRACT: physical page 119 / numbered page 108의 1쪽
- TOC의 ABSTRACT entry는 page 108을 가리킴
- LaTeX/fatal error, undefined citation/reference 없음
- abstract overfull box 없음
- 국문 keyword fixed-height minipage의 underfull warning 1개만 남으며 시각 결함 없음
- 국문 마지막 keyword y-max 614.01 pt, page number y-min 636.07 pt로 약 22 pt 분리
- 참고 국·영문 초록과 현재 국·영문 초록을 모두 raster render하여 육안 비교
- 제목, 문단 흐름, 줄바꿈, keyword wrap, footer 간격과 one-page containment 확인

### 17.8 보존 파일과 삭제 파일

보존:

- `frontmatter/abstract_ko.tex`
- `frontmatter/abstract_en.tex`
- `thesis.tex`의 양언어 keyword와 minipage 정렬
- `notes/abstracts_content_map.md`
- `build/diagnostics/reference_abstract_ko.txt`
- `build/diagnostics/reference_abstract_en.txt`
- `build/diagnostics/thesis_after_abstracts_final.txt`
- `build/thesis.pdf`와 최종 build logs

삭제:

- `build/diagnostics/abstracts_qa_run1`--`abstracts_qa_run4`
- `build/diagnostics/thesis_after_abstracts_run1.txt`--`run3.txt`
- `build/diagnostics/abstract_ko_bbox.html`
- `build/diagnostics/abstract_en_bbox.html`

위 항목은 최종 PDF에서 재생성 가능한 PNG, 중간 layout text와 좌표 확인용
HTML이며 총 2,443,586 bytes를 완전 삭제했다. 원본 초록 PDF나 최종 조판본은
삭제하지 않았다. 현재 작업본은 80 files, 17,533,109 bytes
(16.72 MiB)이다.

### 17.9 다음 작업 경계

Chapters 1--5, 국·영문 초록과 키워드는 완료됐다. 다음 승인 단위는 다음 중 실제
정보를 제공받아 적용할 수 있는 제출 메타데이터와 전체 최종 검수다.

- 영문 저자명
- 지도교수명
- 심사위원 5명
- 제출일, 심사일, 졸업월
- 학번 표기 여부
- optional acknowledgement
- 국문 제목의 공식 제출 번역 확인
- 전 장 terminology, notation, citation, cross-reference, pagination과 학교 제출
  규격 최종 audit

사용자의 다음 지시와 실제 값을 받기 전까지 개인정보 자리표시자를 유지한다.

## 18. Proposal PPT RQ 정렬 수정 (2026-08-20)

### 18.1 기준과 결정

- 기준 PPT: `../proposal 260819.pptx`
- 원본 SHA-256:
  `10861253F59FF0206D01E260CB3134EFB283C822FE42AE12CCE3B2BF757AACAF`
- RQ 문구의 기준은 `chapters/01_introduction.tex` 296--311행의 최종 네 개
  research questions로 고정했다.
- RQ1--RQ3는 각각 Chapters 2--4의 세 core study에 대응하고, RQ4는 별도
  네 번째 알고리즘이 아니라 privacy와 provenance를 가로지르는 shared design
  principles와 explicit assurance stopping boundaries를 도출하는 통합 질문으로
  표현한다.
- Sepsis patient-level alignment와 adaptive-query-safe verification은 core
  dissertation chapters와 구분되는 future validation으로 표시한다.

### 18.2 7--8번 슬라이드 수정

Slide 7 (`Dissertation Thesis and Research Questions`):

- 기존 RQ1--RQ3의 3열 구조와 `Diagnose -> Prevent -> Verify` 흐름은 유지했다.
- 하단에 `RQ 4. Integrated Synthesis` 영역을 추가하고 다음 질문을 넣었다.
  `Which design principles are shared by privacy and provenance, and where must
  their assurance conclusions explicitly stop?`
- 세 기존 RQ의 한 줄 결과를 같은 높이로 정렬해 RQ4 영역과 겹치지 않도록 했다.
- `Post-Execusion`, `supportedby`, `componentsbe`를 각각
  `Post-Execution`, `supported by`, `components be`로 교정했다.
- 발표자 노트를 세 질문 체계에서 네 질문 체계로 갱신하고, RQ4가 세 연구의
  통합 질문이라는 설명과 다음 슬라이드 연결 문장을 추가했다.

Slide 8 (`Dissertation Roadmap`):

- 제목을 `Dissertation Roadmap: Core Studies and Future Validation`으로 변경했다.
- 오른쪽 열을 `FUTURE VALIDATION`으로 바꾸고 항목을 `Future Study A/B`로
  명시하여 Chapters 2--4와 위계를 구분했다.
- 마지막 행을 `RQ 4. INTEGRATED SYNTHESIS`에 연결하고
  `Before Training -> After Training -> After Release` 흐름과
  `Shared design principles · Explicit assurance stopping boundaries`를 표시했다.
- `Bind the Routh`를 `Bind the Route`로 교정하고 PP-Mark 및 future-study
  표기의 띄어쓰기와 하이픈을 정리했다.
- 발표자 노트도 Chapters 2--4가 RQ1--RQ3에, 마지막 통합 행이 RQ4에
  대응하도록 갱신했다. Future Study A/B는 완료 연구가 아닌 향후 검증임을
  명시했다.

### 18.3 산출물과 검증

- 수정본: `../proposal 260819_RQ4반영.pptx`
- 수정본 크기: 5,715,183 bytes
- 수정본 SHA-256:
  `BCB65BF81545B3261983507D793581FDF56166E00DA39EF82889F93F60AAD23C`
- PowerPoint에서 수정본을 다시 열어 15 slides, Slide 7 RQ4 문구, Slide 8
  `FUTURE VALIDATION`, 두 슬라이드의 갱신된 발표자 노트를 확인했다.
- Slides 7--8을 각각 1600 x 900 PNG로 렌더하여 글자 겹침, 잘림, RQ 요약
  누락 여부를 시각 검수했다. 첫 렌더에서 RQ4 영역에 가려진 RQ3 요약을
  발견하여 세 요약을 재정렬한 뒤 재검수했다.
- 검수용 PNG 202,299 bytes는 모두 삭제했으며 원본 PPT는 수정하지 않았다.

### 18.4 다음 작업 경계

Slides 1--8의 내용 흐름은 유지한다. 다음 승인 단위에서는 Slide 9 이후의
Auditable Privacy 배치 문제를 먼저 정리하거나, PP-Mark 상세 슬라이드를
추가한다. 현재 단계에서는 그 외 슬라이드를 수정하지 않았다.

## 19. Proposal PPT Auditable Privacy/PP-Mark 10-slide 구현 (2026-08-20)

### 19.1 작업 범위와 산출물

- 입력본은 `../proposal 260819_RQ4반영.pptx`이며 Slides 1--8은 수정하지 않았다.
- Slides 9--18을 Auditable Privacy 5장과 PP-Mark 5장으로 재구성했다.
- 산출물: `../proposal 260819_RQ4_DP_PP반영.pptx`
- 산출물 크기: 7,384,053 bytes
- 산출물 SHA-256:
  `58A4EFB38F7E2D51D1F005919B04C43D2ACA98507E60AA8D5C73BC5E4D6E1434`
- 기준본보다 1,668,870 bytes 증가했다. 고해상도 원본 그림과 표를 포함한 증가이며,
  별도의 중간 PPT는 만들지 않았다.
- 원본 `proposal 260819.pptx`와 RQ4 반영본은 각각 기존 SHA-256
  `10861253F59FF0206D01E260CB3134EFB283C822FE42AE12CCE3B2BF757AACAF`,
  `BCB65BF81545B3261983507D793581FDF56166E00DA39EF82889F93F60AAD23C`를
  유지해 원본이 변경되지 않았음을 확인했다.

### 19.2 구성 원칙

발표자의 설명 순서를 각 연구의 논문 목차를 단순 축약하는 방식이 아니라 다음
질문 흐름으로 고정했다.

1. AI에서 왜 중요한가.
2. 현재 방식은 왜 실제 보장을 만들지 못하는가.
3. 본 연구가 무엇을 결속하거나 제한하는가.
4. 원 논문의 어떤 실험 결과가 해결 효과를 보여주는가.
5. 어디까지가 결론이고 어디서부터는 주장하면 안 되는가.

DP 두 연구는 같은 분량의 독립 논문 두 편처럼 반복하지 않고, Auditable Privacy의
`post-execution diagnosis -> pre-execution prevention` 연속 단계로 묶었다. PP-Mark는
`public detector/private verifier dilemma -> score/proof 결속 -> 공격 실험 -> 비용과
보장 경계`로 전개했다.

### 19.3 Slides 9--13: Auditable Privacy

- Slide 9, `Why Privacy Numbers Can Mislead in AI Training`:
  민감 event stream이 windowing과 DP-SGD를 거치는 동안 accounted unit/adjacency와
  reported claim이 달라질 수 있음을 문제로 제시했다.
- Slide 10, `Post-Execution: What May Be Claimed?`:
  완료 실행의 frozen evidence `E`와 요청 문구 `q`를 입력으로 받아 Direct, Convert,
  Group, Typed Block을 결정하는 원 논문 Figure 1을 전체 삽입했다.
- Slide 11, `Same Run, Different Privacy Claims`:
  동일 WISDM 실행에서 window claim은 Direct, raw-event claim은 `K=4` Convert,
  owner claim은 `delta K >= 1` 때문에 Block된다는 원 논문 Table 3을 제시했다.
  결과는 `0/100 unsafe positives`, `100/100 valid-exact`, `200/200 exact tuples`이다.
- Slide 12, `Pre-Execution: What May Be Executed?`:
  adjacency, sampler, sensitivity, accountant, executor, transformation, evidence를 한
  route label에 결속하는 원 논문 Figure 1을 전체 삽입했다. 약한 등록 378/378,
  source binding 후 42/378, complete core 후 0/378, endpoint 6/6을 표시했다.
- Slide 13, `Auditable Privacy: Diagnosis and Prevention`:
  RQ1은 실행 후 허용 가능한 claim을 제한하고, RQ2는 실행 전 허용 가능한 route를
  제한한다는 통합 구조를 만들었다. 두 연구가 execution attestation 또는 general DP
  proof는 아니라는 stopping boundary를 함께 명시했다.

### 19.4 Slides 14--18: Secure Public Verification / PP-Mark

- Slide 14, `Why Public Provenance Verification Is Hard`:
  public detector는 독립 검증이 가능하지만 공격 최적화 대상이 되고, private verifier는
  규칙을 숨기지만 중앙 신뢰를 요구한다는 문제를 제시했다.
- Slide 15, `PP-Mark: From Detection Score to Proof-Bound Acceptance`:
  score mode의 positive signal과 provenance proof를 구분하고,
  `accept = score_pass and proof_ok` 구조 및 원 논문 Table 2를 제시했다.
- Slide 16, `PP-Mark: End-to-End Public Verification`:
  context-bound payload 생성, deterministic latent trace와 commitment, SP1 proof,
  공개 score/receipt 검증의 전체 흐름을 원 논문 Figure 1로 설명했다.
- Slide 17, `Security under Forgery and Removal Attacks`:
  원 논문 Table 4와 Figure 11을 사용했다. white-box score FAR 0.76에도 full accept는
  0 observed였고 black-box transfer도 full accept 0 observed였다. regeneration TPR은
  0.98이고 SPSA는 PSNR이 4.7--6.9로 붕괴한 뒤에야 score를 낮췄다. `0 observed`는
  실험 표본에서의 관찰 결과이며 보편적 위조 확률 0으로 표현하지 않았다.
- Slide 18, `Verification Cost and Assurance Boundary`:
  원 논문 Figures 6(b), 8을 사용하고 `N=1000`에서 embed 0.89 s, score verification
  0.72 s, proving 48.572 s, receipt verification 7.589 s, receipt 40.68 MB를 표시했다.
  exact-image binding, edit 후 receipt 재발급, key lifecycle, 두 모델 범위, proof
  persistence, adaptive-query leakage를 배포 경계와 향후 과제로 남겼다.

### 19.5 사용한 원본 그림과 표

새로운 모식도나 실험 그래프를 생성해 원 연구 결과처럼 사용하지 않았다. 다음
원 논문 자산을 직접 사용했다.

- `포함 SCI 연구들/main_audit_aaai27_v24_candidate.pdf`:
  Figure 1, Table 3
- `포함 SCI 연구들/main_aaai27_v43_candidate.pdf`:
  Figure 1
- `포함 SCI 연구들/15860_PP_Mark_Provable_and_Pub.pdf`:
  Figure 1, Table 2, Table 4, Figure 11, Figure 6(b), Figure 8
- figure 자산은 작업본 `figures/`의 원본 추출본을 사용했다.
- 표는 해당 source PDF를 300 dpi로 렌더한 뒤 표 영역만 고해상도 crop하여 삽입했다.
  셀 값, 열 이름, 표 구조를 다시 그리거나 바꾸지 않았다.
- 각 슬라이드에 paper title과 figure/table 번호를 source caption으로 넣었다.

### 19.6 발표자 노트

Slides 9--18 전부에 한국어 발표자 노트를 추가했다. 문자 수는 순서대로
561, 588, 713, 923, 758, 552, 784, 804, 796, 936자다. 노트에는 화면에 보이는
도형의 읽는 순서, 정확한 수치의 의미, 다음 슬라이드 전환 문장, 과장하면 안 되는
claim boundary를 포함했다. 일반적인 발표 속도에서는 이 10장이 약 9--10분 분량이다.

### 19.7 검수 결과

- PowerPoint에서 산출물을 다시 열었고 총 18 slides, 960 x 540의 16:9 규격을
  확인했다.
- Slides 9--18을 각각 1600 x 900 PNG로 렌더하여 표/그림 가독성, 제목과 하단
  takeaway의 잘림, 겹침, 정렬을 시각 검수했다.
- shape boundary 밖으로 나간 객체 0개, text boundary 문제 0개다.
- Slides 9--18의 발표자 노트가 모두 존재함을 다시 확인했다.
- 그룹화된 도형 내부까지 텍스트를 읽어 RQ1--RQ4, Slide 10 및 Slide 15 제목을
  확인했다. RQ4 표기는 `RQ 4.` 형식이다.
- 기존 오탈자 `Execusion`, `Routh`가 산출물에 없음을 확인했다.
- 최초 빌드 스크립트의 한글 문자열 인코딩 문제는 UTF-8 BOM으로 교정했으며, 오류
  실행에서는 산출물이나 원본 파일이 변경되지 않았다.
- 최종 검수 후 source-page render, table crop, slide QA render, 임시 build script
  총 39 files, 13,183,563 bytes를 검증된 임시 경로에서 삭제했다. 원본 source PDF,
  작업본 `figures/` 자산, 최종 PPT는 보존했다.

### 19.8 다음 작업 경계

현재 발표는 제목/trajectory/problem/RQ/roadmap 8장과 DP/PP-Mark 상세 10장까지
총 18장이다. 사용자의 다음 지시 전에는 추가 슬라이드를 만들지 않는다. 다음 승인
단위는 약 20장 구성에 맞춘 integrated insight, limitations/future work, closing의
배치와 최종 발표 시간 조정이다.

## 20. Proposal PPT Slides 10/12 핵심 도식 단순화 (2026-08-20)

### 20.1 수정 이유와 원칙

사용자 검토에서 Slides 10과 12가 각각 원 논문의 전체 Figure 1 하나로 본문 대부분을
차지하여 발표 화면에서 지나치게 복잡하다는 문제가 확인됐다. 두 전체 도식을 현재
PPT에서 제거하고, 논문의 핵심 인과관계만 PowerPoint-native 3단 도식으로 다시
표현했다. 새 도식을 원 논문 figure로 오인하지 않도록 하단 출처는 `Conceptual
summary based on ...`으로 명시했다. 용어, 판정 유형, 실험 수치는 원 논문과 기존
content map의 범위를 유지했다.

### 20.2 Slide 10 수정

`Post-Execution: What May Be Claimed?`을 다음 세 단계로 단순화했다.

1. `FIXED INPUTS`: frozen evidence `E`와 requested claim `q`
2. `EVIDENCE-BOUND CHECK`: 동일 보호 단위, 등록된 mapping bound `K`, evidence
   completeness/consistency 확인
3. `LICENSED WORDING`: Direct, Convert/Group, Typed Block

하단 핵심 문구는 `The validator does not improve the completed run—it restricts
what may be said about it.`으로 고정했다. 완료 실행을 더 강하게 만들거나 새
accountant를 제공하는 연구가 아니라, evidence가 허용하는 문구만 제한한다는 범위를
보이도록 했다. 정확한 WISDM 판정 수치는 다음 Slide 11의 원본 Table 3에 남겼다.

### 20.3 Slide 12 수정

`Pre-Execution: What May Be Executed?`을 다음 세 단계로 단순화했다.

1. `REQUESTED ROUTE`: route label `p`, contract `x`, mapping `m`, preprocessor `a`
2. `COMPLETE-CORE SEAL`: adjacency, sampler, sensitivity/accountant,
   executor/transformation, source/evidence를 하나의 identity에 공동 결속
3. `GENERIC DISPATCH`: complete-core match는 execute/bind outputs, mixed 또는
   missing route는 training 전에 reject

전체 Figure 1의 counterfactual 부분에서는 발표에 필요한 결과만 하단에 남겼다.
약한 map은 spliced combinations `378/378`, source-bound는 `42/378`, complete
core는 `0/378`을 허용했으며, 등록된 정상 endpoint는 `6/6 valid`다. 따라서 섞인
route를 차단하면서 정상 route를 유지했다는 결과가 한눈에 보이도록 했다.

### 20.4 발표자 노트와 주장 경계

- Slide 10 한국어 발표자 노트: 619 characters
- Slide 12 한국어 발표자 노트: 756 characters
- Slide 10은 `evidence-relative wording restriction`, Slide 12는 `finite registered
  route consistency under the stated source condition`으로 설명한다.
- Slide 12를 arbitrary plugin verification, execution attestation 또는 general DP
  proof로 확대하지 않는 경계 문장을 노트에 유지했다.

### 20.5 검수와 현재 산출물

- 두 슬라이드를 각각 1600 x 900으로 다시 렌더하여 시각 검수했다.
- 현재 Slides 10/12에는 학교 로고를 제외한 전체 source figure picture가 없다.
- 두 슬라이드 모두 object boundary 이탈 0개, text boundary 문제 0개다.
- PowerPoint COM의 최초 적용에서 소수점 font size 형변환 오류가 발생했으나 저장
  전에 중단됐고 SHA-256이 동일한 사전 백업으로 즉시 복원했다. 이후 명시적 single
  precision으로 다시 적용했다.
- PowerPoint가 새 text box 선택 경계를 한 줄 높이로 축소한 현상을 발견해 모든
  신규 textbox의 실제 높이를 내용에 맞게 고정하고 재검수했다.
- 현재 산출물: `../proposal 260819_RQ4_DP_PP반영.pptx`
- 현재 크기: 5,186,955 bytes
- 현재 SHA-256:
  `EE20E333B5F8B171ED032441CB09254FB4BA9DFD5E8161A2C1EB4015BD0F08F0`
- 전체 원본 figure 두 개를 제거하여 직전 산출물보다 2,197,098 bytes 감소했다.
- 최종 검수 후 두 렌더 PNG와 rollback용 임시 PPT 두 개, 총 4 files,
  12,752,594 bytes를 검증된 임시 경로에서 삭제했다.

## 21. Proposal PPT DP/PP-Mark buildup 전면 재구성 (2026-08-21)

### 21.1 작업 범위와 파일 정책

- 기준본: `../proposal 260819_RQ4_DP_PP반영.pptx`
- 기준본 크기: 5,186,955 bytes
- 기준본 SHA-256:
  `EE20E333B5F8B171ED032441CB09254FB4BA9DFD5E8161A2C1EB4015BD0F08F0`
- Slides 1--8은 유지하고 Slides 9--18의 DP 5장, PP-Mark 5장을 빌드업 중심으로
  전면 재구성했다.
- 기준본을 덮어쓰지 않고 새 산출물
  `../proposal 260819_RQ4_DP_PP_빌드업반영.pptx`를 만들었다.
- 현재 산출물 크기: 5,806,082 bytes
- 현재 산출물 SHA-256:
  `122A8A38FE272ED1533589C14E2364CB4A3B67E6AA209BFF295D79BDB9D55196`
- 전체 슬라이드 수는 18장으로 유지했다. 향후 integrated insight와 closing을 약 2장
  추가하면 목표인 약 20장 구성이 된다.

### 21.2 근거 문서와 원본 자산

- 사용자 제공 OpenReview reviewer comments와 author responses를
  `notes/ppmark_openreview_rebuttal_2026-07-29.md`에 구조화해 저장했다.
- submitted-paper evidence와 post-submission OpenReview update를 분리했다.
- 10장 전체 설계, 수치와 출처는
  `notes/proposal_dp_pp_buildup_map_2026-08-21.md`에 고정했다.
- DP 일반 근거:
  - NIST SP 800-226, *Guidelines for Evaluating Differential Privacy Guarantees* (2025)
  - Abadi et al., *Deep Learning with Differential Privacy* (2016)
- Generative-AI 및 공개검증 근거:
  - Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*
  - European Commission Article 50 transparency guidance; Article 50 applies from
    2026-08-02
  - C2PA Content Credentials Specification 2.4 (April 2026)
  - Decoder Gradient Shield, CVPR 2025
  - SEAL, ICCV 2025
  - submitted PP-Mark PDF 및 OpenReview author responses
- PP-Mark PDF physical page 24의 Figure 12에서 512 x 512 watermarked image 네 장을
  직접 추출했다. 새 이미지를 생성하거나 편집하지 않았다.
  - `figures/ch04_source_watermarked_example_1.png`, SHA-256
    `B23F3A9FDCF6760EA4A49AD8F20E00925AA13E33CBE73B6A01478B53AB510778`
  - `figures/ch04_source_watermarked_example_2.png`, SHA-256
    `9001C4FE8ED4241D27130BD6BB7AF47E5ECEFFBB299EFE9F8EACBC195AF08F88`
  - `figures/ch04_source_watermarked_example_3.png`, SHA-256
    `8FD82B05D3E99905D0993E1E92248A2CC90112D2F068323B9BFA5BFD2A512587`
  - `figures/ch04_source_watermarked_example_4.png`, SHA-256
    `D0F627A181C3D3A406230EB9BCA61A71565AD0D520279A729F26F9D0B2EA5420`

### 21.3 Slides 9--13: DP buildup

- Slide 9, `Why Differential Privacy Entered AI`:
  의료·웨어러블·임상 기록처럼 한 사람이 반복적으로 제공하는 민감 데이터를 AI가
  학습할 때 memorization, membership, linkage 문제가 생길 수 있음을 제시했다. DP의
  직관을 `D`와 `D'`가 한 declared protected unit만 다를 때 출력 분포가 거의
  구분되지 않는 성질로 설명하고, window/event/patient 중 무엇을 보호하는지 묻는다.
- Slide 10, `How DP-SGD Produces a Privacy Number`:
  sample -> per-example clipping -> Gaussian noise -> repeated update -> accountant 흐름을
  보인다. accountant 입력 `q, sigma, T, delta`와 출력 epsilon을 연결하고, 단 하나의
  approximate-DP inequality만 사용했다. Epsilon은 privacy percentage가 아니며 unit과
  adjacency 없이 해석할 수 없음을 명시했다.
- Slide 11, `One Privacy Number, Three Claim Levels`:
  동일 WISDM 실행에서 window Direct `(0.53683, 1e-6)`, raw-event Convert `K=4`,
  `(2.14731, 1.0642e-5)`, owner/patient Block `delta_K >= 1`을 presentation-native
  sequence/window 도식으로 설명했다.
- Slide 12, `Claim Alignment at Two Times`:
  RQ1 post-execution diagnosis는 licensed wording을, RQ2 pre-execution prevention은
  executable route를 제한한다는 두 개의 수평 flow로 구성했다.
- Slide 13, `Auditable Privacy: What the Evidence Supports`:
  post-execution `0/100`, `100/100`, `200/200`과 pre-execution
  `378 -> 42 -> 0`, endpoint `6/6`을 제시하고 specify unit, bind mechanism,
  require evidence, fail closed의 네 원리를 도출했다. execution attestation/general DP
  proof가 아니라는 stopping boundary도 유지했다.

### 21.4 Slides 14--18: PP-Mark buildup

- Slide 14, `How Stable Diffusion Creates an Image`:
  prompt -> text condition -> latent noise -> iterative U-Net denoising -> clean latent -> VAE
  image의 최소 흐름을 설명하고, final pixels가 prompt/seed/model/operator를 인증하지
  않는 provenance gap으로 연결했다.
- Slide 15, `From Disclosure to Durable Provenance`:
  EU AI Act Article 50과 C2PA 2.4를 배경으로 배치하고 AI detector, signed
  metadata/C2PA, post-hoc watermark, generation-aware watermark의 역할과 한계를
  비교했다. Durable in-image signal, public audit, forgery-resistant acceptance가 함께
  필요하다는 요구조건을 제시했다.
- Slide 16, `A Public Verifier Becomes an Attack Surface`:
  black-box query, white-box gradient, imprint/replay, removal/regeneration을 설명했다.
  matched 1% FPR에서 baseline white-box FAR 1.00, PP-Mark Score .76, Accept 0 observed;
  transfer baseline range와 PP-Mark `.0033/.0067`, Accept `0/0 observed`를 비교했다.
  DGS, SEAL과 PP-Mark의 보호 대상 차이도 명시했다.
- Slide 17, `PP-Mark: Statistical Signal + Proof-Bound Acceptance`:
  context/key -> binding/payload -> initial-latent embedding -> image, sampled trace -> Merkle
  commitment -> SP1 receipt로 단순화했다. Score는 0.72 s statistical screening, Accept는
  `score pass AND proof valid AND exact-image binding`이다. Context-Sig가 authorized
  signer의 unwatermarked images 100/100을 서명할 수 있지만 PP-Mark는 0/100
  Score/Accept였다는 embedding-compliance control을 넣었다.
- Slide 18, `PP-Mark: Evidence and Visual Quality`:
  요청받은 submitted Figure 12의 watermarked image 네 장만 삽입했다. 공격/변환
  image gallery는 추가하지 않았다. Transfer FAR, persistent crop-resize-JPEG 80/100,
  BRISQUE/KID, Score 0.72 s와 post-submission SP1 v6.3.1 `17.0 s / 2.81 MB`를 함께
  제시했다. 이 최신 cost는 별표와 source line으로 OpenReview update임을 명시했다.

### 21.5 Rebuttal 반영과 claim discipline

- submitted SP1 v5 `48.572 s / 40.68 MB`를 최신 결과로 몰래 교체하지 않고,
  slide와 notes에서 v6.3.1 값이 post-submission update임을 표시했다.
- Signature는 exact integrity만 필요할 때 더 단순하고 적절하며, PP-Mark의 추가 가치는
  embedding-compliance proof와 transform-tolerant in-image Score라고 설명한다.
- `0`은 실험에서 observed acceptance 0건이며 universal zero forgery probability가 아니다.
- `two latent-diffusion models`로 제한하고 `architecture-agnostic` 표현을 쓰지 않았다.
- PP-Mark는 participating pipeline의 positive provenance이며 artifact가 없으면 negative가
  아니라 `unverified`다.
- Score와 Accept는 자동 fallback이 아니라 명시적으로 다른 assurance level이다.

### 21.6 발표자 노트

Slides 9--18 전부 한국어 발표자 노트를 갱신했다. 문자 수는 순서대로
563, 687, 678, 687, 767, 703, 924, 884, 1,056, 1,031자다. 각 노트에는
화면 읽는 순서, 필수 기술 설명, 정확한 수치의 의미, 과장하면 안 되는 claim boundary,
다음 슬라이드 연결 문장을 포함했다.

### 21.7 검수

- PowerPoint에서 산출물을 다시 열어 18 slides, 960 x 540의 16:9를 확인했다.
- Slides 9--18을 각각 1600 x 900으로 렌더해 시각 검수했다.
- object boundary 이탈 0개, text boundary 문제 0개다.
- literal backtick line-break artifact 0개다.
- Slide 18 picture count는 학교 logo 1개 + Figure 12 originals 4개이며, 다른 재구성
  슬라이드는 학교 logo 외 source figure picture가 없다.
- RQ4, Figure 12, post-submission `17.0 s / 2.81 MB` 문구를 확인했다.
- 기존 오탈자 `Execusion`, `Routh`가 없고, slides 9--18에 구 submission cost가
  최신 수치처럼 남아 있지 않다.
- Slides 1--8을 기준본과 1600 x 900으로 비교했다. 두 presentation을 동시에 연 첫
  비교에서 Slide 1만 PowerPoint export anomaly가 있었으나, output을 단독으로 다시
  열어 export한 PNG SHA-256이 기준본과 동일했다. Slides 2--8도 모두 동일하다.
- 최초 DP build에서 fill transparency를 percent 값으로 전달한 오류는 저장 전에
  중단됐고 기준본 복사로 복원했다. 0--1 범위로 교정 후 재생성했다.
- Stable Diffusion flow의 세 줄바꿈 문자를 시각검수에서 발견해 실제 line break로
  교정했다.

### 21.8 다음 작업 경계

현재 18장까지 완료했다. 다음 사용자 승인 단위는 integrated insight와 final
future/closing 약 2장을 설계하고, 전체 20분 발표의 장별 시간을 조정하는 것이다.

최종 검수 후 source extraction, slide renders, comparison renders, build scripts와
rollback backup 총 44 files, 13,030,053 bytes를 검증된 임시 경로에서 삭제했다.
최종 PPT, 근거 notes와 Figure 12 원본 네 장만 보존했다.

## 22. 2026-08-21 — DP/PP 5장 제한 제거 및 충분 설명 확장본

### 22.1 사용자 결정과 작업 범위

- 사용자 결정에 따라 DP 5장, PP-Mark 5장 제한을 제거했다.
- 슬라이드 1--8은 변경하지 않고 DP와 PP-Mark 구간만 새 구조로 확장했다.
- 결과물은 `../proposal 260819_RQ4_DP_PP_충분설명반영.pptx`이다.
- 이전 `proposal 260819_RQ4_DP_PP_빌드업반영.pptx`는 그대로 보존했다.
- 최종 장수는 25장이다: 기존 1--8 + DP 7장 + PP-Mark 10장.
- 파일 크기는 5,849,040 bytes이고 SHA-256은
  `C1C7D23E55B4269249D5007C87EFF96EC9BD607FBB78C0E05A28F7F91673671D`이다.

### 22.2 확장 논리

- DP 9--15는 필요성 → epsilon/delta 의미 → DP-SGD 계산 → 동일 실행의 claim-level 차이 →
  post-execution validator → pre-execution registration → synthesis 순서다.
- PP-Mark 16--25는 diffusion 생성 원리 → durable provenance 필요성/규제 → 기존 접근의 한계 →
  public verifier attack surface → signature 대비 ZKP 필요성 → PP-Mark embedding/receipt →
  Score/Accept → SOTA security → robustness/quality/cost/scope → submitted Figure 12 원본 예시
  순서다.
- 한 장에 background, mechanism, equation, result를 모두 넣지 않고 발표자가 한 논리 단위씩
  설명할 수 있도록 분리했다.
- 한글 speaker notes를 새 슬라이드 17장 모두에 기록했다.

### 22.3 원본 자료 사용

- DP 수치와 route ablation은 Chapters 2--3 source studies를 따랐다.
- PP-Mark security, signature control, sequential transforms, quality, SP1 update는 submitted
  PDF와 OpenReview author response 근거를 구분해 사용했다.
- 마지막 slide 25는 `figures/ch04_source_watermarked_example_1.png`--`_4.png`, 즉 submitted
  PDF Figure 12에서 직접 추출한 512 x 512 이미지 네 장을 사용했다.
- 발표용 단순 도식은 paper의 복잡한 system figure를 원본인 것처럼 재현한 것이 아니라
  presentation-level explanation으로 출처에서 구분했다.
- 상세 장표/근거 맵은 `notes/proposal_dp_pp_expanded_map_2026-08-21.md`에 기록했다.
- 재생성 가능한 스크립트는
  `build/scripts/build_proposal_dp_pp_expanded.ps1`에 보존했다.

### 22.4 검수

- PowerPoint에서 25 slides, 960 x 540의 16:9 deck으로 열리는 것을 확인했다.
- slides 9--25를 각각 1600 x 900으로 렌더링해 시각검수했다.
- 새 슬라이드 object out-of-bounds 0, text-bound overflow 0이다.
- literal backtick line-break artifact 0, `Execusion`/`Routh` 0이다.
- slides 9--25의 한글 speaker notes가 모두 존재한다.
- slides 1--8의 내부 `ppt/slides/slide1.xml`--`slide8.xml` SHA-256이 기준본과 각각 동일하다.
  첫 장 상단 slogan의 PowerPoint raster export에는 font-cache성 미세 차이가 한 번 발생했지만,
  slide XML은 동일하고 육안 렌더도 동일하다.
- 결과물 크기는 약 5.85 MB로, 확장 전 약 5.81 MB 대비 불필요하게 증가하지 않았다.
- 최종 검수 후 임시 source copy와 1600 x 900 render 44 files, 12,689,718 bytes를
  `build/ppt_review_20260821` 및 `build/ppt_review_expanded_20260821`에서 삭제했다.
  최종 PPT, 근거 note, 재생성 script, Figure 12 원본만 보존했다.

### 22.5 다음 작업 경계

- 현재 완료 범위는 DP/PP-Mark 설명까지다.
- 다음 사용자 승인 후 종합 인사이트, 향후 과제, 최종 closing을 설계한다.
- 이후 전체 약 20분 발표에 맞춰 slide별 목표 시간을 배정하고, 필요하면 appendix로 이동할
  상세 장표를 결정한다.

## 23. 2026-08-25 — 사전 DP 제외 및 학위논문 전면 개편 완료

### 23.1 사용자 결정

- pre-execution route-registration 연구를 학위논문 본문에서 제외했다.
- Auditable Privacy는 post-execution privacy-unit audit에 집중한다.
- post-execution main paper만으로 작성돼 있던 Chapter 2는 Supplement의 핵심 판정,
  실험, 재현성, 보증 경계를 선별 통합해 확장했다.
- PP-Mark와 결론은 3-study 구조가 아니라 두 연구축의 Claim-Aligned Assurance 구조로
  처음부터 끝까지 다시 정합화했다.

### 23.2 근거와 변경 원장

- Post-execution main: `main_audit_aaai27_v24_candidate.pdf`
- Post-execution supplement: `main_audit_aaai27_supplement_v15_candidate.pdf`
- PP-Mark post-submission: `notes/ppmark_openreview_rebuttal_2026-07-29.md`
- 상세 변경 원장: `notes/revision_posthoc_only_2026-08-25.md`

### 23.3 현재 구조

1. Introduction
2. Post-Execution Auditing of Privacy-Unit Claims
3. Secure Public Verification of Generative-AI Provenance
4. Conclusion

현재 RQ는 RQ1 사후 privacy-unit audit, RQ2 proof-bound public provenance, RQ3 공통
설계 원리와 stopping boundary의 세 질문이다. 제외된
`chapters/03_evidence_sealed_registration.tex`은 이력 보존용으로 남겼지만
`thesis.tex`에서 입력하지 않는다.

### 23.4 Chapter 2 보강

- 5 release states와 3 assurance states
- 6 evidence surfaces, 8-step decision procedure, release precedence
- stable log-domain group conversion, artifact invalidation
- evidence authority, complexity, non-attestation boundary
- W1--W5 necessity witnesses, 10 incidents/16 repairs/176 checks
- WISDM runtime·portable reproduction, UCI HAR invariant boundary
- Sepsis 5,000명/192,206 rows와 Backblaze 27,799,986 drive-day rows diagnostics
- independent verification surfaces와 ExtraSensory external intake

Chapter 2는 numbered pages 13--43의 31 pages다.

### 23.5 PP-Mark 및 결론 보강

- exact-signature/context-signature 대비 embedding-compliance control 추가
- sequential crop-resize-JPEG persistent result 80/100 추가
- matched BRISQUE/KID/CLIP table 추가
- submitted SP1 v5와 post-submission v6.3.1 cost를 구분
- `two latent-diffusion models` 범위와 unverified outcome을 명시
- 결론에서 pre-execution 요약·기여·한계를 전부 제거
- Future Study를 Sepsis native patient-level DP 비교와 C2PA-compatible proof retrieval
  두 실험으로 구체화

### 23.6 전체 조판·검수

- 최종 파일: `build/thesis.pdf`
- 104 physical pages, B5 JIS, 5,001,029 bytes
- Chapter 1: pages 1--11
- Chapter 2: pages 13--43
- Chapter 3: pages 44--76
- Chapter 4: page 77에서 시작
- English ABSTRACT: page 93에서 시작
- build error 0
- undefined citation/reference 0
- overfull box 0
- 포함 문서와 `.toc/.aux/.bbl`의 stale pre-execution/RQ4/Chapter 5 참조 0
- 주요 페이지 raster QA에서 clipping, overlap, caption/table-order 오류 0

재생성 가능한 raster QA PNG는 최종 확인 후 삭제하고 PDF·LaTeX source·변경 원장만
보존한다. 남은 작업은 지도교수·심사위원·날짜·영문 저자명·감사의 글 등 실제 제출
메타데이터 확정이다.

## 24. 2026-08-26 — 제출 표지 1페이지 복원

- 긴 국·영문 제목과 클래스의 중복 고정 간격 때문에 제출 표지의 하단 날짜·소속·성명이
  별도 페이지로 밀리는 현상을 확인했다.
- 활성 `sgmeta_report.cls`의 제출 표지 부분만 수정했다. 원본 보존용
  `template_reference/sgmeta_report.cls`는 변경하지 않았다.
- 제목 및 본문 글자 크기와 문구는 유지하고, 제목·지도교수·제출 문구 사이의 중복 세로
  간격과 하단 minipage 높이만 축소했다.
- 재빌드 결과 제출 표지는 제목, 지도교수, 제출 문구, 날짜, 소속, 성명을 한 페이지에
  모두 표시하며 논문 인준서는 다음 페이지에 독립적으로 유지된다.
- 최종 PDF는 104 physical pages, 5,001,029 bytes다.
- build error, undefined citation/reference, overfull box는 0건이다.
- 표지 1--4쪽을 raster 검수한 뒤 임시 이미지 폴더를 삭제했다.

## 25. 2026-08-31 — 비공개 이미지 생성 모델 수명주기 통합안 기록

### 25.1 사용자 방향

- 서로 다른 모달리티의 privacy audit와 PP-Mark를 공통 원리로만 묶는 현재 구조보다,
  하나의 이미지 생성 모델에서 학습 프라이버시와 출력 출처를 앞뒤로 연결하는 방향을
  검토한다.
- 다수 사용자의 비공개 이미지 컬렉션으로 공유 생성 모델을 학습·미세조정하고,
  Auditable Privacy가 기존 source-unit DP 결과를 요청된 contributor-unit 주장으로
  재사용할 수 있는지 판정한다.
- 등록된 최악경우 기여 상한과 증거가 완전하며 변환 결과가 nonvacuous일 때만 기존
  checkpoint를 재사용하고, 그렇지 않으면 변환을 차단해 native target-unit DP
  재학습을 요구한다.
- 허용된 privacy audit 결과를 exact model checkpoint에 결속한 receipt로 만들고,
  PP-Mark가 model/receipt/context/image/embedding trace의 일치를 공개 검증하는 방향이다.

### 25.2 실제 LaTeX 기준 간극 판정

- 활성 본문은 이미 104쪽의 완성 작업본이며, 단순 아이디어 메모가 아니다.
- Chapter 2는 post-execution privacy-unit audit의 formalism과 시계열 실증을 완성했다.
  image→contributor mapping이나 private-image generator DP 학습 결과는 없다.
- 현재 조판상 Chapter 3의 PP-Mark는 context-bound embedding과 image-bound proof를
  완성했지만, public statement에 privacy-audit receipt digest와 cryptographic checkpoint
  digest가 없고 전체 diffusion 실행을 proof 안에서 재실행하지 않는다.
- Introduction과 Conclusion은 두 연구를 “technically distinct”한 영역으로 명시하며,
  공통 설계 원리에서만 통합한다. 따라서 현재 주장은 타당하지만 새 목표의 single
  lifecycle 또는 end-to-end artifact integration은 아니다.
- AP의 honest-but-fallible producer threat model과 PP-Mark의 forgery adversary를 그대로
  합치면 안 된다. malicious training producer까지 다루려면 authenticated runtime
  observation 또는 execution attestation이 별도로 필요하다.
- DP 감사는 training influence claim을 지원할 뿐, 특정 출력 이미지에 PII가 전혀
  없다는 per-output certificate가 아니다. 그런 문구가 필요하면 별도 output-privacy
  gate를 추가한다.
- `chapters/04_pp_mark.tex` 7--10행에는 제외된 pre-execution 장을 전제로 한
  “preceding two chapters” 문구가 남아 있다. 이는 신규 구조 개편과 별개로 다음 본문
  수정 묶음에서 바로잡을 consistency debt다.

### 25.3 권장 구조와 작업 gate

- 권장안 A: 현재 Chapter 2와 PP-Mark 장의 완료 증거를 보존하고, 신규 Chapter 4
  `Integrated Assurance for Private-Image Generators`를 실증 장으로 추가한 뒤 Conclusion을
  Chapter 5로 이동한다.
- 위 구조는 기존 Introduction에 Chapter 4만 append하는 방식이 아니다. 기술 증거는
  보존하지만, 중심 사례·기승전결·RQ·contribution·장 전환·Conclusion과 초록은 하나의
  private-image generative lifecycle을 기준으로 처음부터 다시 설계한다.
- 보수안 B: 현재 4장 구조를 유지하고 새 흐름을 blueprint/future work로만 둔다. 이
  경우 completed integrated framework라고 주장하지 않는다.
- Chapter 2를 이미지 DP 연구로 전면 교체하는 안은 새 실험 없이 기존 근거를 잃으므로
  선택하지 않는다.
- 신규 장은 claim/threat contract, data/model/compute pilot, image evidence adapter,
  image-level DP와 native contributor-level DP 비교, model receipt, PP-Mark 결속,
  end-to-end attack evaluation을 통과한 뒤에만 작성한다.

### 25.4 기록 파일과 현재 변경 범위

- 상세 기준안:
  `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md`
- `README.md`에 위 note와 현재 연구 확장 상태를 연결했다.
- 이 `WORKLOG.md`의 상단 현재 단계와 실제 활성 4장 구조를 2026-08-31 기준으로
  바로잡았다. 과거 완료 기록은 삭제하지 않았다.
- 활성 `thesis.tex`, 국·영문 초록, Chapters 1/2/3/4 본문은 이번 단계에서 수정하지
  않았다. 따라서 PDF 재조판도 수행하지 않았다.

### 25.5 활성 본문 보존 hash

- `thesis.tex`: `26C3001A287807FF07C56A0EBE40DC3C455A9169BCCF3A31065B401357B9309E`
- `frontmatter/abstract_ko.tex`: `0591A86FA57D8746BF8655B1866025C82E22A8158BE6A1B0125C364B547D3C5C`
- `frontmatter/abstract_en.tex`: `C4C24D7D755BBF17BEEE951CCE58A371578C91999C2B8BB81586207C73810E35`
- `chapters/01_introduction.tex`: `A135B45D404F80B9CED4AD3F7316017D869CCA9F23FEB03E7827721F767A10B2`
- `chapters/02_post_execution_validation.tex`: `DAAA82FA61B5D47C742867C73F4D1AE89DD803A22611B977919C39EB7FCECE85`
- `chapters/04_pp_mark.tex`: `FBA5B3F77720D82FCBF726177F6A50D3B4E71A326D92E466E55E4EEB7D5FF337`
- `chapters/05_conclusion.tex`: `54B71B26938B9C8E4A329BEF0C6D7D4F42C6DEBDDEF4AFB0819654237FAC213E`

### 25.6 다음 시작점

1. A/B 구조를 결정한다.
2. A를 선택하면 먼저 `contributor`, `account`, `depicted person`, source/target adjacency,
   허용 claim과 금지 claim, malicious producer 범위를 한 장의 claim/threat contract로
   고정한다.
3. 실제 private-image DP fine-tuning과 PP-Mark 호환성을 작은 pilot으로 확인한다.
4. pilot을 통과하기 전에는 Introduction, RQ, 초록과 결론을 completed integrated
   framework 문구로 바꾸지 않는다.

### 25.7 후속 확인 — append-only가 아닌 전면 빌드업 재설계

- 사용자와의 후속 검토에서 신규 통합 장만 추가하면 기존 “두 연구축” 서사가 그대로
  남아 사후 결합처럼 보인다는 문제를 확인했다.
- 최종 방향은 **기술 증거의 보존**과 **논리 빌드업의 전면 재작성**을 분리한다.
- 새 기승전결은 다음과 같다.
  1. 다수 사용자의 비공개 이미지로 하나의 공유 생성 모델을 학습·배포하는 상황
  2. image-level 결과의 contributor-level 재표기 문제와 model-to-output evidence 단절
  3. Auditable Privacy, model-bound privacy receipt, PP-Mark를 연결한 해법
  4. 제3자가 하나의 evidence package에서 privacy와 provenance 문구를 fail closed로
     판정하는 결론
- Chapter 1의 Web3·metaverse·LLM 연구 궤적은 삭제하거나 짧은 배경 단락으로 축소한다.
- Introduction의 “two research axes”, “technically distinct domains”, Conclusion의
  “The two studies are not one algorithm”, “technical independence” 중심 문구는 새 결과와
  구조가 확정되면 lifecycle 중심 서술로 교체한다.
- 국·영문 초록, RQ, contribution, organization, Chapter 2/PP-Mark transition,
  Conclusion과 title/keywords 검토까지 모두 재작성 범위에 포함한다.
- 상세 기승전결, 파일·행 구간별 재작성표, 16주 M0--M11 gate는
  `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md` Section 7과
  Section 11--14에 기록했다.
- 다음 단계는 본문 부분 수정이 아니라 M1 narrative architecture다. 여기서 논문의
  한 문장 중심 명제, actor/artifact flow, Chapter 1 문단 outline과 각 장의 역할을 먼저
  고정한다.

## 26. 2026-09-02 — 출발 시나리오 현실성 및 DP 보호 대상 확인

### 26.1 확인 질문

- “다수 사용자의 비공개 이미지로 하나의 공유 생성 모델을 학습한다”는 상황이 두
  기존 연구를 연결하기 위해 만든 가상 전제인지 먼저 확인했다.
- 특히 실제 생성 모델 학습에서 특정 사용자나 환자가 제공한 사진의 membership,
  memorization, extraction을 막는 데 DP가 실제 문제와 실제 방법인지 분리해 검토했다.
- 이 확인이 성립해야 image-level과 contributor-level privacy-unit 간극, model receipt,
  PP-Mark output provenance로 이어지는 빌드업을 유지할 수 있다는 기준을 적용했다.

### 26.2 판정

- 출발 시나리오는 지어낸 것이 아니다. private/decentralized client image collections로
  shared diffusion model을 학습하는 연구, sensitive images에 DP-SGD를 적용한 diffusion
  generation, patient MRI에 대한 DP latent-diffusion fine-tuning이 모두 존재한다.
- 생성 모델이 training image를 기억하고 재출력하거나 membership을 노출하는 공격도
  실증되어 있다.
- 따라서 “민감 이미지 collection → 공유 generator → release artifact에서의
  membership/influence 보호 → 생성 output provenance”는 실제 문제에서 출발하는
  lifecycle로 사용할 수 있다.
- 다만 Auditable Privacy, model-bound receipt, PP-Mark가 이미 하나의 상용 시스템에
  구현되어 있다는 주장은 하지 않는다. 기존 실제 문제들을 하나의 evidence lifecycle로
  연결하고 평가하는 부분이 신규 연구다.

### 26.3 직접 근거

1. Patel et al., “Personalized Federated Training of Diffusion Models with Privacy
   Guarantees,” CVPR 2026
   - decentralized private client datasets, shared+personalized diffusion model,
     sample-level local DP, membership/memorization/reconstruction 평가
   - client 전체 또는 contributor 전체에 대한 DP가 아니고 표준 DP-SGD도 아님
   - https://openaccess.thecvf.com/content/CVPR2026/html/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.html
2. Dockhorn et al., “Differentially Private Diffusion Models”
   - sensitive image data에 DP-SGD를 적용한 synthetic-image generator
   - https://arxiv.org/abs/2210.09929
3. Daum et al., “On Differentially Private 3D Medical Image Synthesis with
   Controllable Latent Diffusion Models”
   - UK Biobank cardiac MRI에 DP fine-tuning을 적용한 의료영상 생성
   - https://arxiv.org/abs/2407.16405
4. Kaissis et al., “Medical imaging deep learning with differential privacy”
   - image-level 실험과 실제 patient-level deployment guarantee를 명시적으로 구분
   - https://doi.org/10.1038/s41598-021-93030-0
5. Carlini et al., “Extracting Training Data from Diffusion Models,” USENIX Security
   2023
   - 개인 사진을 포함한 1,000개 이상의 training examples 추출
   - https://www.usenix.org/conference/usenixsecurity23/presentation/carlini
6. C2PA Guidance for Artificial Intelligence and Machine Learning
   - generative output의 `trainedAlgorithmicMedia` 표시와 model/output provenance
   - https://spec.c2pa.org/specifications/specifications/2.3/ai-ml/ai_ml.html

### 26.4 고정한 주장 경계

- DP는 released checkpoint, API 또는 생성 결과의 관찰자에 대해 특정 이미지나 한
  contributor의 전체 collection 참여·영향을 선언된 adjacency로 제한하는 방법이다.
- 중앙 operator가 raw image와 account mapping을 직접 받는다면 DP만으로 그 operator에게
  제출 관계를 숨기지 못한다. operator까지 보호 대상이면 federated learning, secure
  aggregation, access control 또는 trusted execution을 별도로 명시한다.
- contributor/account와 depicted person을 혼용하지 않는다.
- image-level guarantee가 contributor/patient-level guarantee로 자동 승격되지 않는다.
  등록된 최악경우 (K), enforced contribution cap, lineage와 nonvacuity gate가 필요하다.
- DP는 특정 output의 PII 부재나 원본 이미지 비재현을 절대 보증하지 않는다.
- 생성 결과의 AI 표시와 model provenance는 실제 downstream 요구지만, 단순 표시만으로
  PP-Mark의 필요성이 자동 성립하지 않는다. signed manifest보다 강한 PP-Mark는 metadata
  detachment, public-detector forgery, exact-image binding과 Score/Accept 분리로
  정당화한다.

### 26.5 M1에 미치는 영향

- 시나리오의 현실성 확인은 완료됐지만 M1 narrative architecture는 아직 미완료다.
- M1의 첫 결정은 다음 둘 중 중심 배포 환경을 고르는 것이다.
  1. central curator가 환자·사용자의 민감 이미지를 적법하게 수집하고 generator 또는
     synthetic images를 외부에 제공하는 환경
  2. private client images가 로컬에 남는 federated shared-generator 환경
- 이어서 target privacy unit, training operator trust, release observer, PP-Mark 적용
  모델을 고정한 후 한 문장 논문 명제와 Chapter 1 outline을 작성한다.

### 26.6 변경 범위

- 위 상세 근거와 경계를
  `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md` Section 1.1에
  추가했다.
- `notes/terminology_notation_claim_policy.md`에 private-image lifecycle의 허용·금지
  문구를 추가했다.
- 활성 LaTeX 본문, 초록, 참고문헌과 PDF는 수정하거나 재조판하지 않았다.

## 26A-OPS. 2026-09-02 — 최신 코드 회수, 원본 기준선과 수정용 작업본 분리

### 29.1 생성한 작업 경계

- 논문 작업본 아래에 `code_originals/`와 `code_working/`을 나란히 생성했다.
- `code_originals/`는 수정 금지 기준선이고, 이후 변경은 `code_working/`에서만 한다.
- 기존 PP-Mark, DP-SGD, DP-SGD-algorithm 원본과 활성 LaTeX는 수정하지 않았다.

### 29.2 회수한 최신 기준

- PP-Mark: `Documents/PP-Mark-v0.4/PP-Mark`, branch `pp_mark_v0.41`, commit ancestor
  `b5572d5`, 그리고 2026-08-02 anonymous supplementary ZIP.
- Auditable Privacy: locked V17 code/data ZIP, SHA-256
  `D7C5A6777BA03B579C6A3DAFFF862086354A2FE6A3AA85FCBDD15B9A44074832`.
- UnitDP compiler reference: locked V10 ZIP, SHA-256
  `2B2D9468EDC6DB16BDA7B7DA75CBA53D844D96A8D18DA50F2EFCA7093BB9A850`.
- Audit paper reference는 main V24와 supplement V15를 함께 보존했다.

### 29.3 검증 결과

- Auditable Privacy manifest: 20,909/20,909 PASS; 공식 `tests/` 91 passed.
- UnitDP public verifier PASS; 141 passed, 13 optional skips, 180 subtests passed.
- PP-Mark: 16 passed, 5 skips. skip은 현재 환경에 `poseidon-py`와 optional CuPy가 없기
  때문이며 관측된 테스트 실패가 아니다.
- UnitDP 첫 추출은 410개 중 긴 이름의 preprocessing JSON 3개가 빠진 것을 verifier가
  검출했다. 짧은 내부 경로로 재추출한 `code_originals/unitdp_v10/`은 410개가 모두 있고
  verifier를 통과했다. 첫 실패 트리는 `_diagnostics/`에 사용 금지 상태로 보존했다.

### 29.4 PP-Mark 보존 범위

- 최신 소스, SP1/RISC0/Halo2 코드, tests, docs, paper, configs, datasets, 11개 외부 비교
  구현과 compact A100 evidence ZIP을 복사했다.
- 약 70.9 GiB의 raw outputs, 약 15.7 GiB의 build target, 환경 캐시, private `secrets/`,
  legacy 대용량 artifact는 제외했다. 원래 92 GiB 프로젝트는 그대로 남아 있다.
- 상세 출처, 파일 수, 해시, 외부 저장소 commit과 제외 경계는
  `code_originals/SOURCE_MANIFEST.md` 및 `SHA256SUMS.txt`에 기록했다.

### 29.5 확인된 구현 간극

현재 회수한 두 핵심 코드는 각각의 기존 기여를 재사용하기에는 충분하지만 다음 통합
구현은 아직 없다.

1. patient ID를 가진 의료영상 DP diffusion 학습/fine-tuning;
2. image-unit 대 patient-unit 및 MIA·memorization/extraction·reconstruction 실험;
3. Auditable Privacy 판정을 exact generator checkpoint에 서명 결속하는 model-bound receipt;
4. 그 receipt와 synthetic image를 연결하는 PP-Mark statement/payload adapter;
5. 하나의 evidence package와 third-party fail-closed release/intake decision;
6. 선택할 의료 generator에서의 PP-Mark 호환성·진단/downstream utility 검증.

따라서 코드 회수와 기준선 분리는 완료됐지만 통합 시스템 구현이 완료된 것은 아니다.

## 26B-OPS. 2026-09-02 — 본문 재작성과 실험 재설계의 논리 종속성 확정

향후 논문 본문은 기존 Auditable Privacy와 PP-Mark 결과를 순서대로 병치한 뒤 통합
문장을 덧붙이는 방식으로 수정하지 않는다. 본문의 최상위 구조는 다음 lifecycle이다.

```text
restricted multi-patient private images
  -> central DP generative training/fine-tuning
  -> image/patient privacy-unit claim gap
  -> Auditable Privacy direct/converted/blocked decision
  -> decision bound to the exact generator checkpoint by a signed receipt
  -> synthetic images released outside the trust boundary
  -> PP-Mark model/receipt/image linkage under allowed transformations
  -> third-party fail-closed release or intake decision
```

실험은 위 흐름을 사후 설명하는 장식이 아니라 각 화살표가 실제로 성립하는지 검증하는
증거로 재설계한다. 따라서 본문 수정 시에는 다음 순서를 지킨다.

1. 실제 환경과 위험을 먼저 제시한다.
2. 각 연구질문을 lifecycle의 한 간극 또는 연결에 대응시킨다.
3. 방법 절은 그 연구질문을 해결하는 artifact와 decision rule을 정의한다.
4. 실험 절은 해당 연결의 성립·실패 조건을 직접 측정한다.
5. 기존 결과는 이 구조에서 필요한 증거일 때만 재사용한다.
6. 구현 또는 실험이 끝나지 않은 연결은 완료된 기여처럼 본문에 쓰지 않는다.

특히 medical DP generator, patient-unit 실험, model-bound receipt, PP-Mark adapter와 통합
evidence package는 현재 신규 구현 대상이다. 이 결과가 나오기 전에 기존 장의 용어만
바꾸어 통합이 완료된 것처럼 재작성하지 않는다.



## 27. 2026-09-02 — CVPR FL·DP 정정 및 의료 합성영상 provenance 흐름 확정 기록

### 27.1 CVPR 논문에 대한 공식 정정

- 앞선 기록의 `client별 formal DP guarantee`와 “contributor-DP에 가장 직접적인 근거”라는
  해석은 부정확했다.
- Patel et al.의 설정은 병원·연구소 같은 cross-silo 기관을 client로 둔 **federated
  learning + sample-level local DP**다.
- 각 raw image를 clipping하고 forward-diffusion Gaussian noise를 적용해 서버로 보내며,
  서버는 noisy images로 shared denoiser를 학습한다. 이는 중앙 DP-SGD가 아니다.
- 고품질 생성은 global denoiser와 client의 비공개 personalized denoiser를 결합하므로,
  standalone public global generator로 단순화하지 않는다.
- CIFAR-10, Colorized MNIST, CelebA 실험은 공개 benchmark simulation이다. CelebA client도
  사람별이 아니라 성별·머리색 분포로 인위적으로 나뉜다.
- 따라서 이 연구는 private image silos의 공동 생성 학습이 실제 연구 환경임을 뒷받침하지만,
  account/patient 전체를 보호하는 contributor-level DP의 직접 근거로 인용하지 않는다.

### 27.2 별도로 고정한 중앙 의료 DP diffusion 경로

- 한 병원·컨소시엄이 적법하게 접근하는 restricted multi-patient images를 보유한다.
- DP-SGD 또는 DP fine-tuning으로 diffusion generator를 학습한다.
- generator의 주 역할은 새 image를 진단하는 것이 아니라 noise/condition에서 synthetic
  images를 생성하는 것이다.
- 합성 영상은 실제 환자 영상 대신 외부 연구자에게 제공하고, 희귀질환·소수 class 증강,
  classifier/segmenter 학습, algorithm/hyperparameter 평가에 사용할 수 있다.
- 공개·licensed dataset이 목표 분포를 충분히 제공한다면 private data와 DP가 필요 없다는
  반증 조건도 함께 고정했다. 희귀 cohort, 특정 장비·기관 분포, 반출 제한 때문에 공개
  data로 대체할 수 없을 때만 이 동기가 성립한다.

### 27.3 위협과 privacy-unit 연결

- membership inference는 특정 환자 image의 학습 참여 여부를 추론한다.
- memorization/extraction은 training image의 near-copy를 생성한다.
- reconstruction은 noisy sample, gradient 또는 intermediate information에서 원본을
  복원하려 한다.
- formal image-level DP와 위 공격 평가는 구분하며, 공격 평가는 formal guarantee를
  대체하지 않는다.
- 환자 한 명이 여러 scan·slice·visit을 제공하면 image-level DP가 patient-level DP로
  자동 승격되지 않는다. subject lineage, enforced contribution cap과 registered distance가
  필요한 지점이 Auditable Privacy의 실제 privacy-unit audit 대상이다.

### 27.4 실험 데이터 정책

- 동의받지 않은 private medical images를 새로 수집하지 않는다.
- UK Biobank cardiac MRI는 자유 다운로드 공개 dataset이 아니라 신청·승인·계약이 필요한
  access-controlled resource다.
- 권한이 없으면 합법적인 public/research benchmark로 sensitive-data threat model과
  institutional split을 simulation했다고 명시한다.
- patient/subject ID가 있으면 모든 image·slice·visit을 subject 단위로 묶어 partition한다.
  ID가 없으면 patient-level DP를 주장하지 않고 image-level 또는 simulated-silo 결과로
  제한한다.

### 27.5 DP--receipt--PP-Mark의 정확한 연결

```text
restricted multi-patient images
  -> DP-trained diffusion generator
  -> model-bound signed privacy receipt
  -> synthetic images released outside the trust boundary
  -> durable PP-Mark reference/proof
  -> third-party verification
```

- Auditable Privacy는 허용 가능한 training-privacy 문구를 판정한다.
- model-bound receipt는 privacy unit, \((\varepsilon,\delta)\), accountant, policy와 audit
  result를 exact generator checkpoint에 결속한다.
- PP-Mark는 외부로 배포된 synthetic image와 declared model/receipt의 provenance 연결을
  유지·검증한다. PP-Mark 자체가 DP나 PII 부재를 증명하지 않는다.
- metadata stripping, crop, resize, rotation, compression, re-encoding, screenshot이 발생할
  수 있는 배포 환경이면 geometric/transform robustness가 현실적 요구가 된다.
- 반대로 통제된 DICOM repository가 signed metadata를 끝까지 보존하면 signed manifest가
  충분할 수 있으므로, PP-Mark 필요성은 external release와 metadata detachment 위협을
  명시할 때만 강하게 주장한다.

### 27.6 의료 적용 시 추가 검증 조건

- diagnostic feature와 image utility 훼손 여부
- false positive/negative와 exact-image Accept/transform-tolerant Score 분리
- 16-bit grayscale, DICOM, multi-slice 및 3D volume 호환성
- downstream classifier가 watermark를 shortcut으로 학습하는지 여부
- 현재 SD 2.1/SDXL 기반 PP-Mark와 선택한 의료 generator의 compatibility

현재 PP-Mark를 3D cardiac MRI에 바로 적용 가능하다고 쓰지 않는다. 우선 2D compatible
medical generator 또는 소규모 M3 pilot에서 embedding, robustness, generation quality와
downstream utility를 확인한다.

### 27.7 현재 판정과 다음 단계

- “제한된 다환자 의료영상으로 DP diffusion generator를 학습하고, 실제 환자 영상 대신
  합성 영상을 외부에 배포하며, 변형 후에도 이미지와 감사된 model release의 receipt를
  공개 검증 가능하게 연결한다”는 흐름은 실제 동기와 기술 역할이 맞물리는 타당한 중심
  후보 시나리오로 판정했다.
- 이 판정은 구현 완료나 M1 확정을 의미하지 않는다. central medical 2D setting 채택 여부,
  patient target unit, curator trust, external release actor와 PP-Mark model compatibility를
  M1/M3에서 고정·검증한다.
- 상세 기록은
  `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md` Section 15에
  추가했다.
- 활성 LaTeX 본문, 초록, 참고문헌과 PDF는 이번 기록에서도 수정하거나 재조판하지 않았다.

## 28. 2026-09-02 — M1 잠정 narrative freeze 기록

### 28.1 잠정 중심 시나리오

다음 lifecycle을 M1의 `Provisional-Selected` 중심 시나리오로 고정했다.

```text
restricted multi-patient 2D medical images
  -> DP-SGD training/fine-tuning of a diffusion generator
  -> Auditable Privacy patient-unit claim decision
  -> signed receipt bound to the exact generator checkpoint
  -> synthetic images released outside the trust boundary
  -> PP-Mark durable model/receipt provenance
  -> third-party fail-closed release or dataset-intake decision
```

이는 이미지 생성 연구다. diffusion generator는 새 영상을 진단하는 classifier가 아니라
noise/condition에서 synthetic images를 만들며, 생성물은 외부 연구, 희귀질환·소수 class
증강, downstream classifier/segmenter 학습과 알고리즘 평가에 사용한다.

### 28.2 관련 연구와 사례의 배치

- 중심 근거: DPDM, private fine-tuned diffusion의 Camelyon17 합성영상 활용, UK Biobank
  cardiac MRI의 DP 3D synthesis, FDA synthetic medical data/M-SYNTH 사례
- 인접 환경 근거: CVPR 2026의 cross-silo FL + sample-level local DP. central DP-SGD 또는
  patient-level DP의 직접 근거로 사용하지 않음
- 학습 프라이버시 위협 근거: membership inference, memorization/extraction, reconstruction,
  Carlini et al.의 diffusion training-data extraction
- 의료영상 피해 근거: CT-GAN의 병변 삽입·삭제, FDA가 다루는 AI restoration hallucination
- 기존 provenance/integrity 기반: C2PA AI/ML guidance와 DICOM digital signature/audit

### 28.3 PP-Mark 동기의 우선순위

- 주동기: 합법적으로 외부 배포되는 DP synthetic medical images의 real/synthetic 구분,
  generator/version/receipt linkage, metadata detachment와 변형 이후 provenance 유지
- 구체적 공격: model/receipt substitution, cross-image replay, metadata stripping,
  crop/resize/rotation/compression 회피, real-only corpus에 synthetic image 혼입
- 보조 위협: mark를 제거해 합성 영상을 임상 영상처럼 부정 삽입하거나 병변을 조작하는
  악성 행위
- 악성 공격자는 임의의 unmarked model을 사용할 수 있으므로 PP-Mark를 universal medical
  deepfake detector로 기술하지 않는다.
- evidence 부재는 `Real`이 아니라 `Unverified`이며, semantic diagnosis truth와 PII 부재는
  PP-Mark의 보장 범위가 아니다.

### 28.4 기존 수단과 신규 연결

- DICOM signature는 exact pixel integrity에 직접적이며 PP-Mark가 이를 대체하지 않는다.
- PP-Mark의 추가 가치는 signed metadata가 분리되거나 image가 허용된 변형을 거친 외부
  배포에서 durable positive provenance를 제공하는 데 있다.
- 신규성은 DP diffusion 자체나 의료 watermark 자체가 아니라 다음 연결이다.

> training privacy claim의 unit validity → exact DP generator identity → signed privacy
> receipt → released synthetic image의 durable public provenance → third-party decision

### 28.5 동결 상태와 다음 gate

- 이번 기록은 M1 방향의 잠정 동결이며 통합 구현·실험 완료 선언이 아니다.
- 2D 의료환경은 현재 PP-Mark SD 2.1/SDXL 자산과의 호환 가능성을 우선한 선택이다. 3D
  cardiac MRI는 실제 근거로 유지하지만 바로 구현 대상으로 약속하지 않는다.
- M3 pilot에서 subject ID/lineage가 있는 합법적 dataset, DP fine-tuning, patient grouping,
  PP-Mark 호환성, diagnostic/downstream utility와 external release policy를 확인한다.
- 상세 근거와 금지 주장은
  `notes/unified_private_image_assurance_gap_and_milestones_2026-08-31.md` Section 16에
  기록했다.
- 활성 LaTeX 본문, 초록, 참고문헌과 PDF는 수정하거나 재조판하지 않았다.

## 31. 2026-09-02 — Step 2 의료 데이터셋 선정·접근·lineage 검증

- primary를 SIIM--ISIC 2020 Collection 70, smoke dataset을 PAPILA v1.1로 선정했다.
- ISIC fixed v2 metadata 33,126 rows, official duplicate list 425 pairs와 current API snapshot을
  내려받아 SHA-256을 고정했다.
- 중복 counterpart 425장을 deterministic하게 제외한 뒤에도 32,701 images, 2,056 patients,
  patient당 minimum 2, median 12, maximum 115 images가 남음을 확인했다.
- current API와 fixed v2 사이에 lesion ID 2건 차이가 있어 fixed v2를 split의 유일한 기준으로
  정했다.
- 4 patients의 ISIC JPEG 8장을 실제로 받아 해상도·형식·hash를 검증했다.
- PAPILA 전체 590,857,635-byte archive를 확보했고 provider MD5와 local MD5가 일치했다.
  244 subjects 각각의 양안 2장과 clinical ID가 완전히 연결됨을 확인했다.
- BreakHis는 82-patient secondary replication, CheXpert Plus는 large DICOM extension으로
  보류했다. 이는 hard storage cap 때문이 아니라 ISIC의 patient-unit 실험 적합성이 더
  높기 때문이다.
- 전체 ISIC 23 GB acquisition은 금지하지 않는다. pilot 결과 full population이 필요하면
  direct per-image download 또는 추가 drive를 포함한 보존 계획으로 진행한다.
- public benchmark는 confidential-data handling의 직접 증거로 과장하지 않고, patient-wise
  split과 contribution-cap disclosure를 강제한다.
- 상세 기록: `code_working/DATASET_INTAKE_DECISION.md` 및 위 통합 note Section 19.
- 데이터·manifest hash 재검증은 전부 PASS, storage advisory는 `OK` (C free 62.99 GiB,
  `code_working` 4.239 GiB)였다.
- 활성 LaTeX, 초록, 참고문헌, PDF와 기존 연구 코드는 수정하지 않았다. 사용자에게 먼저
  보고하기 전에는 Step 3을 시작하지 않는다.

## 32. 2026-09-02 — Step 3 DP 학습–감사 실험 계약 초안

- 사용자의 연구 의도를 training-time DP effect와 post-execution audit effect로 분리했다.
- core generator는 from-scratch가 아니라 동일 pretrained latent diffusion의 LoRA
  fine-tuning으로 고정했다.
- primary arms는 untouched base, non-DP, image-DP, native patient-DP이며 clipping/noise와
  patient aggregation confound를 분리하는 no-noise ablation을 추가했다.
- image-DP의 숫자를 patient-DP 숫자와 직접 비교하지 않고 공개 contribution cap `K`로
  patient bound를 재계산하도록 고정했다.
- built-in group conversion을 점검한 결과 `K=5, epsilon_image=1, delta_image=1e-7`은
  `(epsilon_patient=5, delta_patient≈8.58e-6)`의 주요 reuse 후보이고,
  `K=10, epsilon_image=2`는 `delta_patient>1`인 blocked 예시가 됨을 확인했다.
- `ALLOWED` audit와 operational privacy-budget acceptance를 분리했다. conversion이 audit에
  유효하더라도 predeclared release budget을 넘으면 retraining 또는 narrower wording이다.
- cap/split은 fixed ISIC v2와 공개 hash rule로 label-independent하게 고정해 private adaptive
  selection을 피한다.
- formal privacy, image/patient MIA, memorization/extraction, rare-class downstream utility,
  three-seed confidence intervals와 audit-vs-retrain cost를 계약에 포함했다.
- pretraining contamination, attack-null, 모든 실용 conversion block, patient-DP utility
  collapse를 숨기지 않는 사전 해석을 포함했다.
- 검토 초안: `code_working/INTEGRATED_EXPERIMENT_CONTRACT.md`.
- Step 3은 아직 in progress이며 exact base, epsilon/delta grid, release budget, attack power와
  compute gate 승인 전에는 training 또는 Step 4 구현을 시작하지 않는다.

## 33. 2026-09-02 — Step 3 핵심 실험계약 승인·동결

- 사용자 승인에 따라 `K=5`/9,495-image pilot, `K=10`/15,924-image main pool,
  `K=2`/4,112-image unit ablation을 후보가 아닌 고정 설계로 승격했다.
- full 32,701-image population은 과학적으로 정당화된 확장 reserve로 유지했다. 저장 한계를
  절대 cap으로 두지 않되, 원하는 ordering을 얻기 위한 사후 확장은 금지했다.
- patient-first disjoint partition과 동일 split의 B0/M0/M1/M2 재사용을 고정했다.
- common latent-diffusion base의 LoRA fine-tuning, 256x256 feasibility 우선, 조건부 512x512
  confirmatory extension을 고정했다. from-scratch training은 core design에서 제외했다.
- `code_working/INTEGRATED_EXPERIMENT_CONTRACT.md`의 상태를 Step-3 core frozen으로 바꾸고
  승인 gate와 미결 gate를 분리했다.
- exact base revision, split/cap manifest, epsilon/delta와 operational release budget,
  patient-MIA power/threshold 및 confirmatory compute는 아직 미결이다.
- generator training, integration code 작성, 전체 ISIC JPEG acquisition, 활성 LaTeX/초록 수정은
  이번 단계에서 수행하지 않았다.
- 다음 단계는 base license/contamination/RTX 3070 VRAM gate이며, 기존 사용자 지시에 따라
  그 단계로 넘어가기 전에 이번 동결 결과를 먼저 보고한다.

## 34. 2026-09-02 — Step 4 exact base-model gate

- 공식 `CompVis/stable-diffusion-v1-4` revision
  `133a221b8aa7292a167afc5127cb63fb5005638b`, fp16 safetensors를 skeleton/pilot base로
  `PASS_FOR_SKELETON` 판정했다.
- CreativeML OpenRAIL-M과 ISIC CC BY-NC 4.0 경계를 각각 기록했다. 내부 연구 pilot은
  진행 가능하지만 model/adapter/derivative 재배포 전 양쪽 의무를 packaging해야 한다.
- SD v1.4의 LAION-2B(en) pretraining 때문에 ISIC non-overlap은 확인되지 않았다.
  `UNRESOLVED_RISK`로 유지하고 B0, exact/near-duplicate audit와 incremental-membership
  wording을 강제했다.
- 필요한 model files 14개, 1.988 GiB만 받았다. text encoder/UNet/VAE fp16 safetensors의
  SHA-256이 official LFS digest와 모두 일치했고 snapshot manifest hash는
  `65942C6C213361881EE74F1BDB940DC9ECF13CE4121EB9A64E533D46B81784F0`다.
- RTX 3070에서 256/512 2-step inference와 rank-8 LoRA batch-1 backward/clip/update가 모두
  PASS했다. 최대 reserved VRAM은 inference 3.086 GiB, LoRA probe 1.990 GiB였다.
- LoRA trainable parameters는 1,594,368/861,115,332 (0.185%)였다. 이 probe는 random latent
  memory diagnostic이며 DP-SGD, patient-DP 또는 utility 결과가 아니다.
- RTX 3070은 smoke/pilot-capable로만 분류했다. real-data encoding, DP per-example/group
  gradient, K-image patient aggregation, repeated steps와 three-seed runtime이 남아 있다.
- 기존 PP-Mark는 latent architecture와 정합하지만 non-SDXL revision/fp16 variant를 pin하지
  않는다. exact-snapshot adapter와 medical-model threshold/robustness 재보정이 필요하다.
- 상세 문서: `code_working/BASE_MODEL_GATE_DECISION.md`; raw report:
  `code_working/_reports/base_gate_compvis_sd14_r133a_001/`.
- full generator training, 전체 ISIC image acquisition, DP claim, 활성 LaTeX/초록 수정은 하지
  않았다. 다음 단계 전 사용자에게 Step 4 결과를 먼저 보고한다.

## 35. 2026-09-02 — Step 5 split/loader/PP-Mark exact adapter gate

- 연구 우선순위를 privacy-unit 비교 → 생성 utility gate → Auditable Privacy → model-bound
  receipt → PP-Mark output provenance로 명시했다.
- K10은 데이터 부족 대응이 아니라 bounded patient contribution의 controlled main이고,
  K2/K5는 unit sensitivity, full은 별도 scale/realism extension으로 고정했다.
- fixed v2와 official 425 duplicate pairs에서 label-independent SHA-256 70/10/10/10 patient
  split을 한 번 materialize했다. K10 train은 1,415 patients/10,848 images, final test는 208
  patients/1,641 images다.
- K2/K5/K10 전체 cap counts는 4,112/9,495/15,924이며 manifest lock SHA-256은
  `7234076A4D7AA50B261984C18C92D3D65BF19CFB7B0300D8BBD327C191864D11`다.
- 첫 manifest validation에서 32,701 retained images가 32,693 unique lesions를 가진다는 것을
  확인했다. 8 lesion의 same-patient 2-image 구조는 official duplicate가 아니므로 유지했다.
- deterministic regeneration verify와 manifest/loader unit tests 4개가 통과했다.
- 8 real JPEG가 3x256x256 tensor 및 exact SD1.4 VAE 8x4x32x32 latent로 통과했다. VAE smoke
  peak reserved는 1.2227 GiB다.
- PP-Mark `ddim_unet.py`에 exact revision/variant, safetensors, local-only 및 inversion-only
  safety-checker control을 추가하고 deprecated prompt path를 수정했다. 첫 두 compatibility
  failure를 기록한 뒤 exact SD1.4 2-step inversion이 PASS했다. peak reserved는 2.2129 GiB다.
- PP-Mark regression은 16 passed/5 skipped다. 기존 SD2.1/SDXL calibration은 새 의료
  checkpoint에 재사용하지 않는다.
- 상세 기록은 `code_working/STEP5_SPLIT_LOADER_GATE.md`; reports는
  `code_working/_reports/isic2020_split_v1_001/`다.
- generator/DP training, attack, receipt, marked release와 활성 LaTeX/초록 수정은 아직 하지
  않았다. 다음 단계 전 Step 5 결과를 사용자에게 보고한다.

## 36. 2026-09-02 — SD 2.1 exact mirror 재검증 및 base 결정 갱신

- 사용자 결정에 따라 가능한 한 SD 2.1을 유지하되, training 전에
  exact-artifact/license/RTX 3070/PP-Mark gate를 다시 열었다.
- active base를 `Manojb/stable-diffusion-2-1-base`, revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`, fp16 safetensors로 갱신했다.
- 접근 가능한 repo는 official이라고 부르지 않고 `third_party_hash_pinned_mirror`로 기록했다.
  commit history, 28-file metadata, upstream API 접근 실패, 독립 archive hash 비교를
  `_reports/base_gate_sd21_manojb_r0094_001/repository_metadata.json`에 동결했다.
- 실행에 필요한 14 files/2,581,663,938 bytes만 받았다. text encoder/UNet/VAE critical SHA-256
  세 개가 모두 사전 고정값과 일치했다. snapshot manifest digest는
  `E38A5F8FE1695745EDD60FB6E2EEDFD35A01A65B5E981ACA28EEF026F4C02638`이다.
- RTX 3070에서 256/512 inference와 rank-8 LoRA one-step backward/clip/update가 모두 PASS했다.
  최대 reserved VRAM은 inference 3.5293 GiB, LoRA probe 2.0039 GiB다.
- 같은 exact SD 2.1 snapshot으로 real ISIC image의 PP-Mark 2-step inversion이 finite
  4x32x32 latent로 PASS했다. 결과 digest는
  `65C8927B082CA2BF620B8A5D001C547E4A7DAA2C3392D16F99F6A62A03CA5D24`다.
- PP-Mark regression은 16 passed/5 skipped, manifest/loader tests는 4 passed다.
- 결론은 SD 2.1 active base, SD v1.4 verified fallback이다. medical LoRA 뒤에는 같은 2.1이라도
  PP-Mark threshold/robustness를 재보정한다. 이 diagnostic은 DP 또는 utility evidence가 아니다.
- `SD21_BASE_MODEL_GATE_DECISION.md`, contract, README, Step-5 기록을 갱신했다. 기존 SD v1.4
  문서는 삭제하지 않고 superseded fallback evidence로 보존했다.
- generator/DP training과 활성 LaTeX/초록 수정은 하지 않았다. 활성 7개 파일 SHA-256은
  Section 25.5의 보존 기준과 모두 동일하다.

## 37. 2026-09-02 — 독립 논문 방향 추천 순위와 근거 기록

- 2026-09-02 현재 로컬 실험계약과 PFDM/PF-LDM/SPIRE/FedDP-PALD, DPDM/DP-LoRA,
  P3SGD/user-wise DP, DP 실행증명 및 의료 워터마킹 근접 연구를 대조했다.
- 1순위를 **한 환자의 복수 의료영상에서 image-DP evidence가 patient-level claim을
  허용하는 경계와 native patient-DP 재학습 필요성을 비교하는 의료 latent diffusion
  연구**로 기록했다.
- 이 방향은 PFDM의 단순 의료·latent 확장이 아니라 central medical setting의
  `privacy-unit claim gap`과 `reuse versus retrain frontier`를 중심 질문으로 둔다.
- 가장 가능성이 높은 근거는 (1) DPDM이 남긴 diffusion group-privacy 공백, (2) 현재
  K2/K5/K10 patient-first split과 B0/M0/M1/M2의 직접 재사용, (3) conversion 허용·전면 차단·
  attack-null이 모두 유효한 사전 해석, (4) direct federated extension보다 낮은 선행연구 충돌과
  구현·compute 위험이다.
- generic black-box group bound만으로는 부족하므로 DP-SGD-specific tight group accounting을
  필수 비교 후보로 기록했다. 공정한 주 비교는 동일 target patient epsilon/delta에서 M1의
  group-accounted 경로와 M2 direct patient-DP를 맞춘다.
- M2의 patient aggregation 자체는 P3SGD/user-wise DP 계열이므로 최초 알고리즘으로 주장하지
  않는다. 신규성은 medical diffusion generation, unit-labelled audit, patient attack,
  utility/cost와 reuse/retrain decision의 결합에 둔다.
- 2순위는 end-to-end patient-private personalized federated medical LDM, 3순위는
  model-bound receipt와 PP-Mark 기반 claim-carrying medical image, 단순 latent/high-resolution
  medical PFDM 적용은 비추천으로 기록했다.
- 상세 근거: `CVPR 주제 탐색/01_독립_논문_방향_추천_순위와_근거_2026-09-02.md`.
- 이 기록은 방향 추천이며 frozen experiment contract 변경, generator/DP training 시작,
  활성 LaTeX/초록 수정 또는 통합 결과 완료를 의미하지 않는다.

## 38. 2026-09-02 — AAAI audit과 의료영상 CVPR 후속논문의 관계 기록

- 사용자가 설명한 AAAI 약 7쪽 compressed audit 연구의 현재 상태와 로컬 main/supplement
  candidate 산출물을 확인해 후속 의료영상 연구와의 관계를 별도 문서로 기록했다.
- AAAI 연구는 privacy-claim audit 인프라, 의료영상 연구는 복수 영상 환자의 privacy-unit
  mismatch를 해결하는 patient-aware private diffusion method와 reuse-versus-retrain frontier로
  역할을 분리했다.
- AAAI 채택·탈락·심사 중 세 경우 모두 후속 연구가 가능하지만, 단순 데이터 교체나 같은
  기여의 재포장은 피하고 독립된 problem/method/experiment/conclusion을 갖추도록 기록했다.
- CVPR 2026 공식 CFP 기준으로 medical and biological vision, image/video synthesis,
  privacy/accountability가 모두 명시되어 있어 topic scope fit은 높다고 판정했다.
- 동시에 현재 audit+baseline comparison만으로는 CVPR main 경쟁력이 중간이며,
  diffusion-specific patient-DP mechanism, tight accounting, patient attack/memorization,
  clinical/vision utility와 가능하면 복수 데이터셋 검증이 필요하다고 구분했다.
- 목표 연도가 CVPR 2027 이후이면 해당 연도의 CFP와 dual-submission 규정을 다시 확인한다.
- 상세 기록: `CVPR 주제 탐색/02_AAAI_텍스트_Audit과_의료영상_CVPR_후속논문_관계_2026-09-02.md`.
- frozen experiment contract, generator/DP training, 활성 LaTeX·초록은 변경하지 않았다.

## 39. 2026-09-02 — Step 6 K5 실제 이미지 확보·독립 검증 완료

- 동결 K5 manifest SHA-256
  `721CBA21612D23A368086F4A0E8E5DE3173333DB3896A417102739A1DB63DC23`를 입력으로
  SIIM–ISIC 2020 JPEG 9,495장을 확보했다.
- 환자 2,056명은 private train 1,415명/6,512장, public development 200명/919장,
  privacy-attack holdout 233명/1,093장, final test 208명/971장으로 완전 분리되어 있다.
- 총 6,224,399,586 bytes(5.796924 GiB), 악성 표적 301장, 다운로드 실패 0건,
  `.part` 0개, quarantine 0개이다.
- ordered content-set SHA-256은
  `06F177B5C345121D3C5D876530768437A710C9C56DAD30F1225BD795C536E3F2`, local-only
  inventory SHA-256은
  `87AFBF37D1439806CAD5DC09CFD7A249D417EB16B047A403B2D236CCE08415FF`이다.
- 별도 `--verify-only` 실행이 모든 JPEG를 다시 열어 같은 byte/count/digest와 실패 0건을
  재현했다. acquisition report SHA-256은
  `7C58BDCB5ACFF2C372CE0492B9B62BF1D5CB005F1B276DD0DFE3172F75CCF0D3`, verification
  report SHA-256은
  `38BDB0A0C1BC63C4DA5F94773F067DD8CB346E8CC7A386637CC702B745289D24`이다.
- 12장 smoke에서 S3가 정상 JPEG 일부를 `binary/octet-stream`으로 반환하는 현상을 먼저
  발견했다. MIME allow-list 뒤 Pillow JPEG decode/verify를 최종 판정으로 삼도록 수정한 후
  network/offline 검증이 같은 content digest로 통과했다.
- 악성 4장/비악성 4장 시각 점검에서 피부경 병변, 털, 자·기포·비네팅·저대비 등 실제
  획득 변이를 확인했다. 결과를 보고 이런 이미지를 배제하지 않으며 modality는 피부경으로
  정확히 한정한다.
- 피부병변은 모두 피부암이 아니지만 환자에 연결된 악성·양성 병변과 진료 코호트 참여는
  민감 건강정보이다. 위협은 과장된 얼굴식 직접 식별보다 patient membership과 반복 영상의
  memorization/extraction으로 둔다. ISIC는 이미 공개되어 있어 restricted clinical cohort의
  재현용 proxy이지 실제 ISIC 비밀성을 새로 회복했다는 증거가 아니다.
- 저장 상태는 raw K5 약 5.798 GiB, `code_working` 약 10.1 GiB, 최종 확인 시 C: 여유 약 59.01 GiB로
  정상이다. 대용량 영상과 상세 환자 inventory는 local-only로 유지한다.
- 상세 결정: `code_working/STEP6_K5_ACQUISITION_GATE.md`; 구현:
  `code_working/data_pipeline/acquire_isic_k5.py`; aggregate reports:
  `code_working/_reports/isic2020_k5_acquisition_v1_001/` 및
  `code_working/_reports/isic2020_k5_verification_v1_001/`.
- generator/DP training, attack, receipt, PP-Mark release, 활성 LaTeX/초록 수정은 하지 않았다.
  다음 단계는 DP mechanism/accountant·예산·공격 protocol 사전 동결이며 먼저 결과를 보고한다.

## 40-DP. 2026-09-02 — NIH 흉부 X-ray 시각 비교용 6장 진단

- 피부경 core를 바꾸기 전에 NIH ChestXray14가 실제로 더 직관적인 의료·프라이버시
  인상을 주는지 보기 위해 X-ray 6장만 `_diagnostics/nih_cxr14_visual_samples_001/`에
  받았다. 정상 표식 3장과 결절, 침윤·폐렴, 폐기종·종괴 표식 사례이며 PA/AP를 함께 봤다.
- Google Cloud가 문서화한 official NIH PNG bucket은 Requester Pays이고 현재 환경의
  anonymous single-file 요청은 HTTP 403이었다. 따라서 이번 파일은 third-party Hugging Face
  viewer mirror `BahaaEldin0/NIH-Chest-Xray-14` revision
  `932bcdba9d7d9590704d4f20bc70fc2c3a1bbad7`에서 받은 **visual-only JPEG rendition**으로
  명시했다. 향후 실험 입력으로 재사용하지 않는다.
- 6장은 모두 1024x1024 grayscale JPEG로 decode PASS, 총 367,071 bytes이다. 파일별 hash와
  viewer row·weak label·view는 진단 README에 고정했다.
- 육안상 흉부 X-ray는 피부경보다 병원 환자 데이터라는 인상이 훨씬 직접적이다. 동시에
  AP/PA, portable 촬영, 노출·자세, side marker, 관·monitoring lead·수술 hardware가 크게
  달라 별도 통제가 필요함을 확인했다.
- `No Finding` 표식 사례에도 관과 뚜렷한 임상 변이가 보였다. NIH 14개 표식은 report에서
  자동 추출된 weak label이며 `No Finding`을 판독의가 확정한 완전 정상으로 쓰면 안 된다는
  설계 위험을 기록했다. 이 육안 점검으로 의학적 진단을 내리지 않았다.
- 결론은 X-ray도 patient-unit DP·합성영상 provenance 연구에 타당하지만, 현재 피부경이
  exact patient split·실제 영상·SD 2.1·PP-Mark gate를 이미 통과해 첫 구현 위험은 더 낮다는
  것이다. 이번 점검만으로 데이터셋을 전환하거나 서로 다른 modality를 섞지 않는다.
- 상세 기록: `code_working/_diagnostics/nih_cxr14_visual_samples_001/README.md`.
- generator/DP training, 활성 실험계약, LaTeX/초록은 변경하지 않았다.

## 40A-CVPR. 2026-09-02 — CVPR 근접연구 분석과 PCM 초안

- 상태 주의: 이 초안과 아래 chest X-ray core pivot이 병행 작성되었다. 이 항목의
  MURA 우선 권고와 ISIC 중심 순서는 바로 다음 core-pivot 결정 및 Section 42에서
  supersede되며, 방법 경계·PCM 수식·선행연구 분석만 유지한다.

- PFDM(CVPR 2026), P3SGD(CVPR 2019), DP-FedEmb(CVPR 2023), HiGradAvgDP,
  user-wise DP-SGD/ULS, tight group privacy for DP-SGD, DPDM, DP-LoRA, 의료 DP-LDM,
  Nature 2026 patient-level privacy audit와 의료 LDM memorization을 1차 문헌 기준으로
  claim-by-claim 재대조했다.
- 추가 경계 검색 결과 patient adjacency, 환자별 record 평균과 clipping/noising,
  hierarchical gradient averaging, diffusion noise multiplicity는 각각 이미 선행이
  있음을 확인했다. 따라서 이를 최초 기여로 쓰지 않으며, 단순 `P3SGD + DP-LoRA`
  조합으로 보이지 않게 기여 범위를 좁혔다.
- 주 방법 후보를 **Patient-Coverage Multiplicity(PCM)**로 구체화했다. frozen visual
  feature로 한 환자의 영상을 coverage strata로 나누고, 고정된 환자당 gradient evaluation
  budget `Q`를 서로 다른 영상과 diffusion noise/timestep repetition에 배분해 unbiased
  patient gradient를 만든 뒤 환자 단위로 한 번 clipping/noising하는 설계다.
- local frozen manifest를 다시 집계했다. K5 private train 1,415명 중 1,203명이 4장 이상,
  1,065명이 5장이며, K10에서는 849명이 8장 이상, 770명이 10장이다. K10의 1,415명 전원이
  복수 lesion, 1,300명이 복수 anatomical site를 가져 within-patient visual heterogeneity가
  실제 주 실험축임을 확인했다.
- naive patient ULS, DPDM식 noise-only, uniform distinct-first와 PCM을 동일 `Q`,
  patient epsilon/delta, sampler·steps·noise·clip 조건에서 비교하도록 설계했다. image-DP
  M1은 generic conversion M1-G와 tight conversion M1-T를 direct patient-DP M2와 같은
  target patient budget에서 비교한다.
- 외부 confirmatory 데이터는 우선 MURA를 권장했다. 공식 자료 기준 12,173 patients,
  14,863 studies, 40,561 multi-view radiographs로 ISIC과 다른 X-ray modality이면서
  patient-linked 반복영상 구조가 명확하다. 실제 취득 전 license/access/storage gate를
  별도로 통과해야 하며 현재 다운로드하지 않았다.
- true individual-patient MIA AUC/eSF는 같은 환자가 여러 target model에서 member와
  non-member로 반복 관측되어야 함을 명시했다. 한 target model의 score tail을 Nature
  2026식 individual AUC라고 부르지 않으며, 다수 target model은 별도 power/compute
  gate 뒤에만 수행한다.
- PCM은 public-development reference-gradient diagnostic에서 uniform distinct보다
  variance 또는 clipping distortion을 줄이고, 실제 DP utility에서 noise-only보다
  개선될 때만 CVPR 중심 방법으로 승격한다. 실패하면 method claim을 중단하고
  audit/benchmark 또는 다른 venue용으로 재구성한다.
- 상세 기록:
  `CVPR 주제 탐색/04_관련근접연구_claim_matrix_2026-09-02.md`,
  `CVPR 주제 탐색/05_patient_coverage_multiplicity_접근법과_실험계획_2026-09-02.md`.
- `CVPR 주제 탐색/README.md`에 두 문서와 구체화된 1순위를 연결했다.
- generator/DP training, 데이터 다운로드, `INTEGRATED_EXPERIMENT_CONTRACT.md`,
  활성 학위논문 LaTeX/초록은 변경하지 않았다. 다음 단계는 accountant·sampler contract와
  동결 실험계약 변경 diff를 먼저 제안한 뒤, 승인되면 toy unit test와
  public-development gradient diagnostic을 수행하는 것이다.

## 41-DP. 2026-09-02 — 흉부 X-ray core 전환과 중대 finding 후보 기록

- 사용자 결정으로 frontal chest X-ray를 첫/core modality로, 확보 완료한 ISIC dermoscopy를
  별도 cross-domain extension으로 변경했다. 한 generator에 X-ray/MRI/피부를 섞지 않는다.
- NIH official Box에서 main metadata, bounding box, train/val list, test list의 file ID·bytes·
  SHA-1 commitment를 읽었다. 접근 가능한 `alkzar90/NIH-Chest-X-ray-dataset` revision
  `36778e3b0e4f4b4fad31d1728d6190f3eda5b543`에서 metadata만 받았다.
- `BBox_List_2017.csv`, `train_val_list.txt`, `test_list.txt`는 official bytes와 SHA-1에 exact
  match했다. `Data_Entry_2017_v2020.csv`는 mirror가 9,003,499 bytes/SHA-256
  `DC1D2DF67FDC1C5A7601D48699CDA2B13DC2C4841488B4183DCF04884DBACA11`로 official
  9,003,496 bytes/SHA-1 `48a9f849a8f100a0f1721b33bdbd209767656111`과 3-byte 차이가 있어
  preflight에만 쓰고 frozen source로 승인하지 않았다.
- preliminary metadata는 112,120 unique images/30,805 patients이다. full contribution은 median
  1, p90 8, p95 14, p99 35, max 184이고 >=2/5/10 images 환자는 각각
  13,302/5,759/2,545명이다.
- PA-only는 67,310장/28,868명이며 >=2/5/10 환자는 10,688/3,245/953명이다. AP-only는
  44,810장/9,060명이다. PA-only도 main privacy-unit 실험에 충분하다.
- official list는 train/validation 86,524장/28,008명, test 25,596장/2,797명이고 image/patient
  overlap과 누락은 모두 0이다.
- PA-only 중대 finding 후보는 pneumothorax 3,407장/1,155명,
  pneumonia-or-consolidation 2,108장/1,601명, pleural effusion 6,589장/3,279명,
  mass-or-nodule 7,140장/4,089명이다. `No Finding` reference는 39,302장/22,452명이다.
- `Mass/Nodule`은 cancer-suspicious finding이지 암 확진이 아니다. Cardiomegaly는 optional
  fifth group이다. Edema는 PA 276장 대 AP 2,027장으로 view shortcut이 커서 이름의 심각성만
  보고 core로 고정하지 않았다.
- 기존 B0/M0/M1/M2, central LoRA, K2/K5/K10, patient-disjoint attacks, Auditable Privacy,
  model-bound receipt, PP-Mark는 유지하고 exact X-ray counts/view/multi-label/utility protocol만
  다시 동결한다.
- X-ray pilot이 애매할 때 원하는 attack 결과를 위해 피부로 조용히 교체하지 않는다. 사전
  수치 gate와 failed/null pilot을 남기고, X-ray 수정·claim 축소·compute 추가·ISIC core 승격
  중 하나를 투명한 amendment로 선택한다.
- 상세 결정: `code_working/XRAY_CORE_PIVOT_DECISION.md`.
- full X-ray acquisition, generator/DP training, active LaTeX/초록 수정은 하지 않았다.

## 42-CVPR. 2026-09-02 — PCM 설계를 최신 chest X-ray core 결정에 정합화

- 병행 작성된 PCM 초안의 데이터 hierarchy가 X-ray pivot보다 뒤처진 것을 검증 단계에서
  발견해 `XRAY_CORE_PIVOT_DECISION.md`와
  `INTEGRATED_EXPERIMENT_CONTRACT.md` Section 15를 다시 읽고 정합화했다.
- 최종 hierarchy는 NIH frontal chest X-ray core, 별도 SIIM–ISIC dermoscopy
  cross-domain extension, 필요·자원 확인 뒤 MURA optional third dataset이다. X-ray와
  dermoscopy를 한 unconditioned generator에 섞지 않는다.
- NIH PA-only 예비 분포의 28,868 patients 중 10,688명은 2장 이상, 3,245명은 5장 이상,
  953명은 10장 이상이며 최대 100장이다. 동시에 18,180명은 한 장만 제공하므로 PCM
  효과를 모든 환자에게 가정하지 않는다. all-patient cohort를 유지하고 record-count
  strata에서 방법 효과를 분석한다.
- image-DP checkpoint의 환자별 converted bound는 capped record count `k_u`에 따라
  달라질 수 있지만, 전체 cohort에 동일 patient claim을 하려면 worst-case public cap
  `K`를 사용하도록 설계를 명시했다. 평균 `k`로 formal claim을 완화하지 않는다.
- X-ray에서는 PA-only를 우선 option으로 두고, AP+PA이면 view condition과 matched
  distribution을 요구한다. PCM coverage는 longitudinal finding·device·intensity 변화를
  다루되 acquisition shortcut을 utility로 오인하지 않도록 별도 평가한다.
- Stage 순서를 exact metadata provenance 해결 → view/finding policy → patient-first
  K2/K5/K10 manifests → selective acquisition/interface gate → contract/accountant →
  X-ray gradient diagnostic → X-ray DP pilot/confirmatory → ISIC extension으로 수정했다.
- 최신 image-DP synthesis 경계로 PrivImage·dp-promise(USENIX 2024),
  SPTI(NeurIPS 2025), RPGen(AAAI 2026)을 추가 확인했다. 이들은 semantic public
  pretraining, forward-noise DP, textual intermediary, robust pretraining을 각각 점유하지만
  검색된 공식 설명은 private image/sample 중심이며 native patient-linked coverage를
  다루지 않는다. patient target으로 정당하게 변환되지 않은 published FID를 직접
  patient-DP 성능처럼 비교하지 않도록 기록했다.
- 갱신 문서:
  `CVPR 주제 탐색/README.md`,
  `CVPR 주제 탐색/04_관련근접연구_claim_matrix_2026-09-02.md`,
  `CVPR 주제 탐색/05_patient_coverage_multiplicity_접근법과_실험계획_2026-09-02.md`.
- full X-ray download, generator/DP training, accountant 구현, active LaTeX/초록 수정은
  수행하지 않았다.

## 43-CVPR. 2026-09-02 — PCM을 완성 방법이 아닌 배분 규칙 연구 가설로 재동결

- 대화에서 핵심 경계를 다시 확인했다. 한 entity의 여러 record gradient를 단순 평균하고
  entity 단위로 clipping하는 것은 P3SGD, user-level DP/ULS, HiGradAvgDP 계열과 겹치므로
  신규 claim이 아니다.
- record 간 변동과 diffusion noise/timestep 변동의 총분산 분해도 그 자체로는 표준 분석이며,
  DPDM의 noise multiplicity와 일반 stratified gradient sampling을 신규 claim으로 재포장하지
  않는다.
- 현재 연구 상태는 완성된 PCM 방법이 아니라, 고정 patient privacy·clipping norm·gradient
  evaluation budget `Q`에서 distinct record 수와 record당 diffusion perturbation 반복 수를
  배분하는 선행 비중복 규칙을 찾는 가설 단계로 동결했다.
- 첫 검증은 full generator training이 아니라 reference-gradient diagnostic으로 정했다.
  `noise-only`, `uniform-distinct`, `coverage-stratified`를 compute-matched하게 비교하고,
  pre-clipping MSE, clipping probability, clipping distortion을 측정한다.
- uniform-distinct 대비 반복 seed 개선과 singleton 환원 성질을 통과하지 못하면 PCM을 신규
  방법으로 승격하지 않는다. 통과한 뒤에만 작은 실제 DP utility pilot로 이동한다.
- 상세 수식·H0/H1·승격 조건은
  `CVPR 주제 탐색/05_patient_coverage_multiplicity_접근법과_실험계획_2026-09-02.md`
  Section 20에 기록했다.

## 43-DP. 2026-09-02 — NIH PA metadata·patient-manifest gate PASS

- official Box metadata/split 4개 파일을 byte/SHA-1/SHA-256로 고정했다. main CSV SHA-256은
  `C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1`이다.
- 기존 3-byte mirror 차이는 `Patient Sex`/`Patient Gender` header 한 줄뿐이고 112,120 data row는
  동일함을 확인했다. official 파일만 active authority로 쓴다.
- PA-only와 네 개 weak-finding evaluation strata(pneumothorax,
  pneumonia/consolidation, pleural effusion, mass/nodule)를 고정했다. definitive diagnosis나
  cancer label로 과장하지 않는다.
- 단순 hash-random 8,000-patient 첫 설계의 rare-finding 부족을 기록하고, reseed하지 않은 채
  metadata-only policy를 1:1 target-enriched patient cohort로 한 번 변경했다. 첫 enriched seed와
  `seed|tag|id` serialization을 고정했다.
- 13,926 patients: private train 8,476, development 1,816, attack holdout 1,816, enriched final
  1,818. 환자 교차 0, 모든 partition target/control 1:1이다.
- nested totals: K2 20,687(feasibility), K5 31,451(primary/main), K10 38,492(unit stress).
  manifest lock:
  `518D3288ACDF55BE355038BC693B3BCF7A8C44B8B3455FAC1DC493EE5E4E3ECE`.
- enriched final test의 prevalence 한계를 위해 full official PA-test census 11,096 images/
  2,647 patients를 overlapping secondary sensitivity set으로 고정했다. tuning 및 독립-test
  주장은 금지한다. K10+census union은 42,423 unique images이고 추가분은 3,931장이다.
- `build_nih_cxr14_manifests.py --verify`가 byte-exact 재생성을 통과했고 unit test 6개가 PASS했다.
  별도 PowerShell 검증도 duplicate 0, partition mapping error 0, non-PA 0, official split error 0,
  K2⊂K5⊂K10 및 K10-final⊂PA-test-census를 재현했다.
- aggregate report SHA-256:
  `035FB07CA2F1D7958FEBA5F214ADC21F88697CEEA807452CEEC966A7493B8735`.
- 결정 문서: `code_working/XRAY_METADATA_MANIFEST_GATE.md`; 구현:
  `code_working/data_pipeline/build_nih_cxr14_manifests.py`; report:
  `code_working/_reports/nih_cxr14_pa_split_v1_001/report.json`.
- official experiment PNG, generator/DP training, receipt, marked release, active LaTeX/초록 수정은
  수행하지 않았다. 다음 단계는 official archive mapping → 소량 PNG source smoke → 42,423장
  selective acquisition → exact image/SD 2.1/PP-Mark interface gate이며 먼저 결과를 보고한다.

## 44-CVPR. 2026-09-02 — PCM synthetic control 및 ISIC 실제-gradient smoke

- `code_working/pcm_diagnostic/`에 synthetic fixed-compute allocation harness, unit tests와
  결과 확인 전 동결한 ISIC SD 2.1 smoke protocol을 추가했다.
- synthetic unit test 3개와 ISIC Gram/sampling unit test 3개가 모두 PASS했다.
- synthetic 정식 run은 5 seeds, seed당 16 entities·64 trials로 실행했다. record effect가 없는
  negative control은 Q=2/4/8의 모든 allocation에서 exact tie였다. aligned-strata Q=4
  preclip MSE는 noise-only 0.02396026, uniform-distinct 0.00673088,
  stratified 0.00387136으로 내부 positive control을 통과했다. 이는 harness 검증일 뿐 실제
  방법 증거가 아니다.
- 실제 smoke는 ISIC K5 `public_development`에서 5 images, 5 lesions, >=3 sites 조건을 만족한
  78명 중 salted SHA-256 최소값 환자 `IP_0687884`를 결과 확인 전에 선택했다.
- hash-pinned SD 2.1 revision `0094d483...75c`, rank-8 attention LoRA, 256 pixels,
  record당 4 perturbations로 1,659,904-dimensional gradient 20개를 계산했다. gradient 계산은
  15.6905초, peak allocated CUDA memory는 1.7205 GiB였고 optimizer update·DP noise는 없었다.
- exact model hash, 20 finite gradients, nonempty strata, estimator weight sum과 Gram PSD gate는
  PASS했다.
- Q=4 one-patient preclip MSE는 noise-only 1.42465e-8, uniform-distinct 6.92098e-9
  (ratio 0.486), frozen-VAE coverage-stratified 9.95903e-9(ratio 0.699)였다. 즉 여러 record를
  보는 것은 이 사례에서 유리했지만 현재 coverage rule은 uniform보다 나빴다.
- preregistered `C=1.0`에서는 모든 estimator clip rate가 0이어서 clipping 가설은 미검증이다.
  결과를 본 뒤 patient/feature/C를 교체하지 않았고 PCM은 미승격으로 유지했다.
- 다음 단계는 별도 동결 multi-patient public-development protocol이다. 모든 selected patient와
  adverse/null 결과를 유지하고 record/noise variance decomposition, feature-gradient alignment,
  사전 clip sensitivity grid를 포함해야 한다. 그 전에는 DP training으로 넘어가지 않는다.
- reports:
  `code_working/_reports/pcm_synthetic_allocation_v1_001/`,
  `code_working/_reports/pcm_isic_sd21_gradient_smoke_v1_001/`.

## 45-DP. 2026-09-02 — NIH official archive map 및 24-image PNG source smoke PASS

- official NIH image archive 12개를 Box file/version ID, bytes, provider SHA-1로 고정했다.
  catalog SHA-256은
  `3CD87825C2F7B604FF1E599A988A68D7D4B7C27AB8DB984B94F3139837572B41`, 총 압축 크기는
  45,079,862,784 bytes(41.983894 GiB)이다.
- historical `batch_download_zips.py`의 static URL은 현재 404라 실행 authority에서 제외하고,
  current Box file-ID route와 range resume를 사용하는 fail-closed downloader를 만들었다.
- `images_001.tar.gz` 2,008,470,987 bytes를 다운로드했다. verified provider SHA-1은
  `FEF95A7A789BCB0013FBF966CB92C4D92C90BECD`, local SHA-256은
  `FD8E3542DB6351AE9377779033F5D5C5F32FE50EB0830B519FBF1A7E791354B1`이다.
- tar scan: 4,999 regular PNG, K10+census union candidate 2,042, non-PNG/duplicate/unsafe/
  special/metadata-unknown member 모두 0.
- frozen seed로 partition x target/control, 네 primary weak-finding group, `No Finding`,
  census-only를 강제한 24 images/24 patients를 추출했다. 전부 1024 x 1024 grayscale PNG이며
  ordered content-set SHA-256은
  `A802BBDAF084D4B0CA51B2E2791A36E6E76985C245C24A1588436BFD4726765A`이다.
- `--verify-only`가 inventory/report를 byte exact로 재생성했다. 별도 PowerShell 검산에서 archive
  SHA-1/SHA-256, 24개 image SHA-256, PNG header/dimensions/color type, image/patient uniqueness
  mismatch가 모두 0이었다.
- four-image manual source check는 정면 X-ray 형식과 exposure/positioning/marker/tube/clip
  이질성을 확인했지만 finding을 판독하거나 weak label을 확정하지 않았다. 이후 exact
  preprocessing과 device/marker shortcut sensitivity를 freeze해야 한다.
- report SHA-256:
  `E74DCEE48BC57343F265FBC5B78D96719E6E40FAAAF6F42F01EB0D7DD6F944C9`; private inventory
  SHA-256: `EA457F3E58D79F8FAE29E792ECF7E98E504B14BD904EE4DD318D2E068B64F72A`.
- 상세 문서: `code_working/XRAY_OFFICIAL_PNG_SMOKE_GATE.md`; 구현:
  `code_working/data_pipeline/smoke_nih_cxr14_official_archive.py`.
- full 42,423-image acquisition, generator/DP training, receipt, marked release, active LaTeX/초록
  수정은 수행하지 않았다. 다음 gate는 all-12-archive selective acquisition과 독립 content
  inventory이고 결과를 먼저 보고한다.

## 46-CVPR. 2026-09-02 — PCM 실제-gradient 다환자 진단, 후보 폐기와 timestep 신호 분리

- smoke 환자를 제외한 ISIC public-development 16 patients를 low/high anatomical-site
  diversity 8명씩 salted hash로 사전 선택했다. 환자당 5 records x 4 perturbations, 총 320개
  실제 SD 2.1 rank-8 LoRA gradients를 계산했고 execution 및 독립 audit가 PASS했다.
- frozen VAE coverage/uniform-distinct geometric MSE ratio는 `1.0322`, bootstrap 95% CI
  `[0.9674, 1.1005]`, VAE wins 6/16이었다. low/high subgroup 모두 3/8 wins였고 C grid도
  구하지 못해 `RETIRE_CURRENT_VAE_COVERAGE`로 판정했다.
- 저장 Gram matrix의 10개 two-anchor partitions를 exact enumeration한 outcome-informed
  oracle도 uniform 대비 `0.9749`, 8/16 wins에 그쳐
  `RETIRE_PAIR_STRATIFIED_M4_FAMILY`로 판정했다. adverse 결과 뒤 feature를 교체하지 않는다.
- variance diagnostic에서 within-perturbation 성분이 우세해 record와 perturbation을 동시에
  균형화하는 4-bank screen을 사전 고정했다. `0.7440`, 16/16 wins였지만 finite reference의
  네 perturbation을 매번 모두 보는 유리한 구조라 확증으로 쓰지 않았다.
- 기존 17 patients와 overlap 0인 새 low/high 8명씩, four timestep bins x two independent
  perturbations, 환자당 40 gradients로 confirmatory를 다시 고정했다. 총 640 gradients,
  selection/hash/finite/Gram/metric execution과 별도 audit가 PASS했다.
- 새 confirmatory의 true-bin/replacement ratio는 `0.8592`, CI `[0.7931, 0.9220]`, 16/16
  wins였다. low/high ratio는 각각 `0.8266`, `0.8931`이고 양쪽 모두 8/8 wins였다.
- positive result 뒤 generic without-replacement와 true timestep-bin을 분리하는 exact
  ablation protocol을 먼저 고정했다. distinct/replacement는 `0.9328`, true-bin/distinct는
  `0.9211`, CI `[0.8814, 0.9572]`, 16/16 wins로 incremental gate를 모두 통과했다.
- 실제 chronological grouping은 8 perturbations의 105개 perfect-pair partitions 중 13/16
  patients에서 1위, 나머지는 4·4·11위였다. 105-partition 평균은 generic distinct와 exact
  일치했다. 전체 `pcm_diagnostic` test suite 21개가 PASS했다.
- 중요한 제한으로 16 patients가 같은 perturbation bank를 공유한다. patient bootstrap은
  bank uncertainty를 포함하지 않으므로 최소 3개 새 bank replication 전에는 방법을
  승격하지 않는다.
- 원문 재조사에서 VDM/DDIM의 minibatch antithetic time sampling, DPDM의 image-level noise
  multiplicity, Ghalebikesabi et al.의 pre-clipping timestep multiplicity, CleanDIFT/HDiT의
  stratified timestep sampling을 직접 확인했다. 따라서 timestep balance 자체는 신규 claim이
  아니다.
- 현재 남는 조건부 후보는 multiple-record privacy unit의 patient vector를 clipping하기 전에
  record와 timestep marginal을 함께 균형화하는 **patient-unit-local placement**다. 이는
  `ULS + DPDM + stratified timestep`의 자명한 결합 위험이 있어 independent banks,
  batch-global 대 unit-local ablation, variable-record unbiased rule과 clipping 이론,
  exact-source X-ray 및 실제 patient-DP utility를 통과해야 한다.
- 상세 기록:
  `CVPR 주제 탐색/06_ISIC_실제_gradient_진단과_현재_방법판정_2026-09-02.md`;
  reports:
  `code_working/_reports/pcm_isic_sd21_multipatient_v1_001/`,
  `code_working/_reports/pcm_isic_sd21_two_axis_confirm_v1_001/`.
- generator/DP training, X-ray full acquisition, active experiment contract, 학위논문 LaTeX와
  초록은 변경하지 않았다.

## 47-CVPR. 2026-09-02 — timestep 신호의 5-new-bank replication PASS

- 최초 confirmatory의 16 patients가 한 perturbation bank를 공유한 한계를 직접 검사하기
  위해 결과 확인 전 `repl_01`--`repl_05`, timestep/noise seed serialization과 multi-bank
  gate를 `ISIC_SD21_TIMESTEP_MULTIBANK_REPLICATION_PROTOCOL_V1.md`에 고정했다.
- cohort, 80 records, SD 2.1 revision, rank-8 LoRA, preprocessing와 Q=4는 유지하고 bank당
  640개씩 총 3,200개 실제 gradients를 새로 계산했다. 모든 bank에서 640/640 finite,
  critical hashes, unbiased estimator weights, exact state counts와 Gram PSD가 PASS했다.
- 각 bank 실행 뒤 confirmatory secondary audit와 105-pairing timestep-bin ablation을
  자동 수행했고 모두 PASS했다. true-bin/distinct ratios는 `0.9354`, `0.9170`, `0.7862`,
  `0.9356`, `0.9177`; wins는 15, 16, 16, 16, 15/16; median true-grouping ranks는
  3, 1, 1, 1, 1/105였다.
- 새 5 banks의 equal-bank true-bin/distinct ratio는 `0.8965`, bank와 patient를 각각
  resample한 10,000-replicate hierarchical bootstrap 95% CI는 `[0.8309, 0.9419]`였다.
  5/5 bank direction의 사전 one-sided sign probability는 `1/32`다.
- frozen conditions인 5/5 ratio<1, 5/5 wins>=10/16, equal-bank ratio<=0.95,
  hierarchical upper<1, >=4/5 top-quartile grouping이 모두 PASS해
  `MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST`로 판정했다.
- 별도 aggregate audit가 5 x 16 matrix, primary 수치와 decision을 독립 재현해 PASS했고,
  `pcm_diagnostic` 전체 25 unit tests가 PASS했다. audit의 첫 실행은 `numpy.bool_` JSON
  직렬화 오류로 결과 파일 쓰기 전에 중단됐고, boolean type conversion 한 줄만 고쳐
  재실행했다. 원 gradient·protocol·aggregate 수치는 변경하지 않았다.
- 이 PASS는 한 perturbation bank의 우연 가능성을 줄이지만 신규성을 확정하지 않는다.
  VDM/DDIM/CleanDIFT/HDiT의 timestep stratification과 DPDM 계열 multiplicity가 선행이므로,
  다음 gate는 same-compute batch-global balance 대 patient-unit-local pre-clipping balance다.
- reports:
  `code_working/_reports/pcm_isic_sd21_timestep_repl_01/`--`repl_05/` 및
  `code_working/_reports/pcm_isic_sd21_timestep_multibank_v1_001/`.
- generator/DP training, X-ray full acquisition, active experiment contract, 학위논문 LaTeX와
  초록은 변경하지 않았다.

## 48-DP. 2026-09-02 — NIH 42,423-image full acquisition 및 independent verification PASS

- 12개 official archive를 byte/provider SHA-1/local SHA-256, safe tar, exact metadata name으로
  검증한 뒤 frozen K10+census member만 선택 추출했다. archive별 전체 source-name list와 선택
  inventory를 atomic commit하고 파일 SHA를 재검산한 뒤 임시 archive를 삭제했다.
- 12 archives 전체 112,120 unique PNG basename이 official metadata와 exact equality를 이뤘고
  missing/extra/cross-archive duplicate는 0이다. source-name-set SHA-256:
  `9E749A0B70E21F2803B792B4E4D4C2D7BF7560E2C4BD75780A5B83904DF79B7A`.
- exact union은 42,423 images/14,755 patients/17,671,122,456 bytes(16.457515 GiB)다. partition
  images는 private 21,756, development 4,831, attack 4,740, enriched final 7,165,
  census-only 3,931이다.
- smoke 표본의 all-`L` 가정 때문에 full 첫 실행이 첫 `RGBA`에서 fail-closed 중단됐다. commit과
  다음 archive download 전에 native-byte 보존 + exact R=G=B + alpha 255 계약으로 수정했다.
  최종 mode는 `L` 42,244, opaque grayscale-equivalent `RGBA` 179이고 actual color,
  nonopaque alpha, unsupported mode, non-1024, decode failure는 모두 0이다. raw conversion은 없다.
- content inventory SHA-256:
  `AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3`; ordered content-set:
  `E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`.
- acquisition `--verify-only`가 full file hash와 output bytes를 재현했다. acquisition/source-smoke
  코드를 import하지 않는 verifier가 official metadata·K10/census union·archive fragments를 다시
  조립하고 42,423 files를 전부 SHA-256/decode/channel/pixel/mapping 검사해 PASS했다.
- independent verifier의 첫 실행은 32개 label token-order 차이 중 첫 사례에서 중단됐다. 전수
  진단은 label-set mismatch 0, canonical sorted-serialization mismatch 0이었다. verifier를 exact
  set + canonical serialization 기준으로 수정했고 frozen manifest/inventory는 변경하지 않았다.
- acquisition report SHA-256:
  `A76611CF1A11A38502685627453611B7BCAE6BE998AAA05E8EAEBB6578944311`; independent report:
  `A1E34DE71BB8F048F6CA44732B20FBF2041DFF3723C6633BFE2D3328B764AB92`.
- PowerShell 별도 검산도 inventory/file 42,423, unique patients 14,755, bytes 17,671,122,456,
  invalid RGBA 0, staging/partial files 0을 재현했다. 최종 C: 여유는 약 43.1 GiB이다.
- 상세 문서: `code_working/XRAY_FULL_ACQUISITION_GATE.md`; 구현:
  `code_working/data_pipeline/acquire_nih_cxr14_union.py`; independent verifier:
  `code_working/data_pipeline/verify_nih_cxr14_union_independent.py`.
- 다음 gate는 exact raw-to-model preprocessing과 SD 2.1 VAE/LoRA/PP-Mark interface다.
  generator/DP training, receipt, marked release, active LaTeX/초록 수정은 수행하지 않았다.

## 49-CVPR. 2026-09-03 — Patient-unit-local clipping proxy PASS

- 5-new-bank pre-clipping 신호 다음 gate로, 새 distinct post-clipping 결과를 보기 전에
  `ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md`에 B=2 same-compute comparator,
  primary `C=0.20`, 5/5 bank·wins·ratio·hierarchical CI·nondegenerate clipping 조건을 고정했다.
- global-only arm은 여덟 perturbation을 두 patient에게 임의의 4+4로 상보 배정하고,
  unit-local arm은 각 timestep bin의 두 perturbation을 1+1로 상보 배정한다. 두 arm은 batch
  전체 perturbation multiset, patient당 Q=4와 distinct 4/5 records가 같고, patient marginal은
  각각 8,400-state distinct와 1,920-state true-bin estimator다.
- `repl_01`--`repl_05`의 저장 40 x 40 within-patient Gram에서 5 banks x 16 patients x 5 C의
  400 cells를 exact enumeration했다. 새 gradient, optimizer update와 DP noise는 없었다.
- primary `C=0.20`의 unit-local/global-only postclip geometric MSE ratio는 `0.9093935`, bank와
  patient hierarchical bootstrap 95% CI는 `[0.8446018, 0.9505718]`였다. 5/5 bank ratios가
  1 미만, 모든 bank가 최소 10/16 wins, 전체 78/80 patient-bank cells에서 local이 작았다.
- global-only/unit-local 평균 clip rate는 `0.1948289`/`0.1379362`로 사전 고정 operating-point
  조건도 PASS했다. bank ratios는 `0.9467`, `0.9380`, `0.8028`, `0.9241`, `0.9442`였다.
- 모든 frozen conditions가 PASS해
  `UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST`로 판정했다.
- 독립 audit는 분석 구현을 import하지 않고 weight enumeration, clipping과 bootstrap을 원
  Gram에서 다시 계산했다. 400-cell 최대 상대 오차 `6.09e-16`, bank/clip-grid/primary/decision
  재현이 모두 PASS했다. `pcm_diagnostic` 전체 31 unit tests도 PASS했다.
- protocol SHA-256:
  `791F1D54A34F43505B56912D8A0B273823B75FEBA9D7E23D76182C9DA58FAE43`; analysis:
  `8FB32843FC30CEFB6362B3776F7B30CBA683D4F899E413F64976555BD128D625`; audit code:
  `82FBD93DFAB45EBC92BB5CFBA80B6A88ADF5AD1B76C21650A92C17EDAC1FB32F`; tests:
  `0B10C27BF50B2B9B2340EC16A87078F2DC337EB3CCC17534AFC5D3B4AEB391BA`.
- report summary SHA-256:
  `5DD1345AA5CC8ECEB8C20ED2BABF677F2D89FA0A3D1A6AB9AE794DB0A223AF02`; secondary audit:
  `ABB96041122FD06F425BA5EE671F52C8A3FF753923693DABF6473E9B89069DB7`.
- 이 PASS는 within-patient marginal proxy다. cross-patient Gram block이 없어 complementary
  allocation의 환자 간 covariance와 actual clipped batch-average update MSE는 미검증이다.
  다음 gate는 frozen patient pairs와 cross-Gram을 사용하는 actual B=2 joint-batch test다.
  timestep stratification 자체의 신규성, DP training utility, generation, privacy, clinical
  validity 또는 CVPR readiness를 주장하지 않는다.
- 상세 기록:
  `CVPR 주제 탐색/07_patient_unit_local_clipping_proxy_결과_2026-09-03.md`; report:
  `code_working/_reports/pcm_isic_sd21_unit_local_clipping_proxy_v1_001/`.
- active experiment contract, 학위논문 LaTeX·초록, generator/DP training은 변경하지 않았다.

## 50-DP. 2026-09-03 — NIH X-ray exact preprocessing·SD2.1/PP-Mark interface PASS

- `nih_cxr14_model_input.py`를 추가해 1024-square PNG, native `L`/opaque grayscale-equivalent
  `RGBA`, full-field Lanczos resize, post-resize RGB replication, float32 `x/127.5-1`, no private
  statistics/no augmentation 계약을 구현했다. 원본은 변경하지 않았고 전체 resized cache도
  만들지 않았다.
- `P256`은 feasibility/pilot, `P512`는 사전 medical-utility+compute gate를 통과할 때만 쓰는
  confirmatory extension으로 고정했다. 두 해상도를 공격 결과로 사후 선택하지 않는다.
- K10 public-development 4,831 images/1,816 patients(`L` 4,804, `RGBA` 27)에서 결과와 무관한
  fixed-stratum salted-hash rule로 8 images/8 patients를 선택했다. native mode 두 종류,
  `No Finding`, pneumothorax, pneumonia/consolidation, pleural effusion, mass/nodule, multilabel을
  포함한다.
- 두 profile의 16개 preprocessing row를 각각 두 번 계산해 pixel/tensor exact replay를
  확인했다. P256 commitment는
  `149ED15BB149862787050C365BFE143ADA76BE69CAD1BC3A75BC4C71AD894CAD`, P512는
  `1C71C0E77F36025DAC9DA471B02995FF2563A10909218BE264D5389182B7B326`다.
- all-weak-label canonical prompt policy를 고정했고 patient ID/age/sex/enrichment flag를
  제외했다. public-development 전수 prompt audit은 203 unique, maximum 43/77 tokens,
  truncation 0으로 PASS했다.
- exact SD 2.1 revision `0094d483...75c`의 critical component hashes를 다시 확인했다. 실제
  native-`L` 및 `RGBA` X-ray에서 VAE mode latent는 P256 `1x4x32x32`, P512
  `1x4x64x64`로 finite/exact replay PASS했다.
- rank-8 attention LoRA 1,659,904 parameters의 실제 영상 gradient를 fixed noise/timestep으로
  profile별 두 번 계산했다. P256 gradient SHA-256은
  `73F78DB6F6A42E4CCFCB43144E5517F51FAFF81176F8011117FDD38870041CEF`, P512는
  `E809C3734836477788B1C93AF35AE34738CF22E7CA0B0FE4059C45F95013FB7E`이며 exact replay했다.
  optimizer 생성/step, gradient clipping, DP noise/accounting은 없고 adapter 전후 hash가 같다.
- 동일한 committed pixels를 PP-Mark `ddim_unet`에 전달해 P256 `4x32x32`, P512
  `4x64x64` latent가 두 실제 case 모두 finite/exact replay했다. peak reserved는 P256
  2.7031 GiB, P512 2.9746 GiB다. 이는 calibration/robustness 결과가 아니다.
- loader unit tests 4개가 PASS했다. 구현을 import하지 않는 independent verifier가 selection,
  raw hashes, 16 preprocessing commitments, model/VAE/LoRA/PP-Mark invariants를 재검증해
  PASS했다.
- main report SHA-256:
  `B5C7826BEDB0E6584A5A5346FC70F7A43BE71C64E59062E66667AB1E430AE6F0`; independent:
  `7B03A7AEDD662F21DDF3796B70F84CF4823376981334C6CCD0B7645F2980C6B3`; preprocessing contract:
  `0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D`.
- 상세 결정: `code_working/XRAY_SD21_PPMARK_INTERFACE_GATE.md`; report:
  `code_working/_reports/nih_cxr14_sd21_ppmark_interface_v1_001/`.
- generator training, optimizer update, DP run, attacks, PP-Mark calibration/embedding, receipt,
  release image, 활성 학위논문 LaTeX/초록 수정은 수행하지 않았다. 다음 gate는 학습 전에
  DP mechanism/accountant/privacy budget/attack protocol을 동결하고 먼저 보고하는 것이다.

## 51-DP. 2026-09-03 — X-ray DP mechanism·accountant·budget·attack protocol 동결 PASS

- optimizer 학습 전에 `code_working/dp_protocol/`을 추가하고 M1 image-DP와 M2 native
  patient-DP를 실행 가능한 계약으로 고정했다. 둘 다 P256, 동일 SD 2.1/rank-8 attention LoRA,
  AdamW `1e-4`, final step 4,000을 쓴다. private-data-dependent checkpoint 선택은 금지했다.
- M1은 image Poisson sampling, 기대 image batch 8, per-image full-gradient clipping, Gaussian
  noise, fixed denominator 8이다. M2는 patient Poisson sampling, 기대 patient batch 4, 환자 안에서
  최대 4장 uniform-without-replacement, patient mean gradient clipping, Gaussian noise, fixed
  denominator 4이다. empty batch도 noise-only update하도록 계약했다.
- K10 M2의 한 step 기대 image work는 8.023124장으로 M1 8장 대비 ratio 1.002891이다. 이전의
  M1 batch 16 후보는 M2보다 image work가 약 두 배여서 최종 계약에서 폐기했다.
- 공개 K10 development만으로 target/control x `n1`/`n2-3`/`n4-10`을 층화해 image 72 unit,
  patient 72 unit의 실제 SD 2.1 LoRA gradient를 계산했다. trainable adapter는 explicit fp32로
  승격했고 frozen base는 fp16이다. empirical p80(`method='higher'`)으로
  `C_image=0.28448700606156724`, `C_patient=0.1997973088974048`을 고정했으며 각각 14/72가
  C를 초과한다. sentinel replay, finite gradients, adapter 전후 동일 digest를 확인했다.
- 첫 pre-freeze clip run은 coarse grid가 image `C=0.5`를 골라 2/72만 clip하고 adapter가 fp16인
  결함을 드러내 폐기했다. 두 번째 fp32/p80 run은 patient가 24 unit뿐이라 private 학습 전에
  symmetric 72/72로 확대했다. 두 이력과 hash를 최종 report에 남겼다.
- 155개 RDP order, add/remove Poisson-sampled Gaussian event, 4,000-step maximum을 고정하고
  Opacus 1.6.0과 Google `dp-accounting` 0.6.0 중 큰 epsilon을 기준으로 sigma를 calibration했다.
  별도 verifier가 두 회계를 다시 계산한 serialized maximum drift는 0이다.
- K10 M1-I8은 image `(8,1e-5)`, `sigma=0.393338498`이지만 patient conversion은
  `(80,delta>=1)`로 vacuous다. K10 M1-G8은 image `(0.8,4.1126114e-9)`,
  `sigma=1.155724687`로 patient `(8,1e-5)`가 된다. M2-P8은 patient `(8,1e-5)`를
  `sigma=0.404457146`으로 직접 보장하는 비교군이며 M2-P4/P2도 사전 curve로 고정했다.
- patient-MIA cohort는 member/nonmember 각각 1,816명, target/control 각 908명이고 holdout의
  `target flag x K10 image count 1..10` cell count와 member를 exact match했다. primary fixed-noise
  denoising loss, gated SecMI-LDM, ROC-AUC/advantage/TPR@1%FPR, 10,000 patient bootstrap과
  `DETECTED`/`NO_MATERIAL_LEAKAGE_DETECTED`/`INCONCLUSIVE` 규칙을 고정했다. approximate 80%
  power AUC는 0.526843이고 material threshold는 0.55다.
- memorization/extraction은 일곱 canonical prompt x 350, 모델당 2,450-query DDIM generate-and-filter,
  train/holdout 양쪽 nearest neighbor, 공개 development transformed positives와 100,000개 이상
  different-patient negatives로 threshold를 결과 전에 고정하도록 했다. 일반적인 X-ray 유사성을
  reconstruction이라고 부르지 않으며 finite attack failure도 privacy proof로 쓰지 않는다.
- 연구용 PyTorch PRNG 결과는 `RESEARCH_ONLY`로 제한했다. 현재 Python 3.11/PyTorch 2.6에는
  등록·검증된 release-grade CSPRNG Gaussian backend가 없으므로, ideal mechanism 회계가 맞아도
  `DIRECT`/`GROUP/CONVERT` release statement는 차단한다. 정식 release에는 secure backend와
  처음부터 새 full rerun이 필요하다.
- clip report SHA-256:
  `F9771F169BA3F6468F5AAF254A27AFFE1FB47B1ABA897C365BFFC7CDD11A73F1`; final protocol:
  `F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018`; builder report:
  `265FB172B6749C32248D5159E2414598C9ACEF29F2443AD426BED1F0639886EC`; independent verifier:
  `9CAB4DB57B83985A90BC2265D9FA2BBEDC122B96B3418919CB6A8B2684472FC4`.
- 5개 fast contract test가 PASS했고 세 report folder 9 files는 총 396,155 bytes뿐이다. full resized
  cache, latent cache, gradient tensor bank, checkpoint를 만들지 않았다.
- 최종 C: free는 37,034,004,480 bytes(약 34.49 GiB)다. 이전 기록 대비 감소분은 이번 report
  산출물이 아니라 system-managed `C:\pagefile.sys`가 45,957 MiB까지 동적으로 커진 상태와
  일치한다. project, `%TEMP%`, Hugging Face cache에서 이번 실행 시각 이후 생성된 100 MiB 이상
  파일은 0개였으며 pagefile을 임의 조정하거나 삭제하지 않았다.
- 상세 문서: `code_working/XRAY_DP_ATTACK_PROTOCOL_GATE.md`. generator training, private optimizer
  update, 실제 DP run, MIA/extraction 실행, receipt, PP-Mark embedding/release와 활성 LaTeX/초록
  수정은 하지 않았다. 다음 gate는 합성/공개 자료 기반 DP trainer mechanism conformance와
  runtime event-trace/accountant agreement다.
- `thesis.tex`, 두 초록, Chapter 1/2/4/5의 활성 7개 SHA-256을 Section 13 기준과 다시 대조했고
  모두 exact match했다. PDF 재조판도 하지 않았다.

## 52-CVPR. 2026-09-03 — Actual B=2 joint-batch locality FAIL

- 앞선 patient-marginal proxy의 빈칸이던 cross-patient covariance를 실제로 측정하기 위해,
  결과 전에 `ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md`와
  `ISIC_SD21_JOINT_BATCH_BANKS_V1.csv`를 동결했다. 같은 16 patients의 모든 120 unordered
  pairs, primary `C=0.20`, 새 `joint_01`--`joint_05` banks와 일곱 gate 조건을 사용했다.
- bank당 16 patients x 5 records x 8 perturbations = 640 gradients, 전체 3,200개의 실제
  SD 2.1 rank-8 LoRA gradients를 계산했다. bank당 640 x 640 full Gram으로 서로 다른 patients
  사이 block을 포함했고 raw gradient matrix는 저장하지 않았다. 다섯 banks가 finite,
  symmetry, direct-dot, diagonal 및 PSD 검사를 통과했다.
- `joint_01`은 outcome analysis 전에 최초 수치 검증 한계가 float32 BLAS reduction-order
  차이보다 엄격해 중단됐다. direct-dot `1.8031e-6` 대 한계 `1e-6`, norm diagonal
  `5.9442e-5` 대 한계 `1e-5`였고 Gram 최소 eigenvalue는 양수였다. 결과를 계산·열람하기 전에
  한계를 `1e-5`와 `1e-4`로 수정해 재검증했으며 원 artifact hashes와 실패 이력을
  `validation_history`에 보존했다.
- actual B=2 primary local/global ratio는 `1.0064606`, five-bank bootstrap 95% CI는
  `[1.0020765, 1.0117106]`였다. bank ratio < 1은 1/5, 전체 pair wins는 191/600,
  84/120 wins를 충족한 bank는 0/5, leave-one-patient-out ratio < 1은 0/16이었다.
- 새 banks의 within-patient marginal은 local/global `0.9429900`, 5/5 banks로 다시
  개선됐다. 그러나 cross-patient contribution 평균이 global `-4.2473e-10`, local
  `-1.6055e-10`이었다. local의 within contribution 절감 `2.3908e-10`보다 global의 음의
  covariance 상쇄 우위 `2.6418e-10`가 커 joint MSE가 global `3.7863e-9`, local
  `3.8114e-9`로 뒤집혔다.
- 사전 판정은 `UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`다. 단순 patient-unit-local
  timestep placement는 핵심 방법 claim에서 중단한다. 불리한 primary를 secondary `C`로
  교체하거나 patient/pair/bank를 제외하지 않는다.
- 독립 감사가 primary 600 pair 및 80 marginal cells를 full Gram에서 다시 계산했다.
  최대 상대오차는 각각 `2.50e-14`, `4.51e-16`이고 같은 판정을 재현했다. 전체
  `pcm_diagnostic` test suite 37개가 PASS했다.
- protocol SHA-256:
  `A688FD4A7F893192B013ED12980876C66B100390C06E7E537D9840286BF9A595`; bank spec:
  `ECF583521CEBC760E5FF4BCA2AB20B5F16FA0ABC4B89436448EF1560089D981A`; cross-Gram code:
  `6B3D429F01D634B7DF746A8258ECA0CCC46EDA338AD91A1D664BE33F5DAE1E6E`; analysis:
  `3F392A2082532964368EA6E12DAEEE651A611B90F77167B45FA78E39CD3E7CE5`; audit:
  `C80CBA813C7BD335CEE10DB8F1C922CA9C943649AB88DB4AEC7CA022CB6E5BB3`; tests:
  `15B144BA8DD865A6BBD330C0BC7F1C9379230EEABD69CB0EF44B55F3FCA3B360`.
- cross-Gram summary SHA-256:
  `9208B089AD1C93EEB41630820AE54DCD9A4F37369885775CA366AC43AEA19463`; result summary:
  `2A06856BFC37CEAE2D2194DE350D01C45C310B57D0D6BF0ECBFAC0A150A37286`; secondary audit:
  `43E1BABAD30DC91B2CEC0D92DC25666BEC465CD4156BE6EDC01F1F151E798137`.
- 상세 기록: `CVPR 주제 탐색/08_actual_B2_joint_batch_locality_결과_2026-09-03.md`;
  report: `code_working/_reports/pcm_isic_sd21_joint_batch_locality_v1_001/`.
- 상위 patient-level DP 의료영상 연구 문제는 유지한다. 다음 allocation 탐색은 기존 full Gram을
  쓰는 post hoc discovery로 명시하고, marginal variance와 cross-unit covariance를 함께 줄이는
  공개정보 기반 규칙이 발견될 때만 별도 새 banks·cohort에 사전 고정해 검정한다. generator/DP
  training, privacy, clinical, novelty 또는 CVPR-ready claim은 하지 않았다.

## 53-DP. 2026-09-03 — X-ray executable DP-trainer conformance PASS

- frozen protocol을 실제 optimizer 경로로 옮기기 전에 기존
  `unitdp_compiler_reference/src/unitdp/owa_dpsgd.py`를 점검했다. owner 평균-gradient와
  owner-level clip은 재사용 가능한 증거지만, Bernoulli 표본이 비면 `continue`해 Gaussian noise와
  update를 모두 생략한다. frozen sampled-Gaussian 계약은 empty sample에도 noise-only update를
  요구하므로 이 코드를 X-ray 실행 경로로 재사용하지 않았다.
- `code_working/dp_training/mechanism.py`에 fp32 flat unit 검증, per-unit L2 clipping, 환자 내부
  mean-before-outer-clip, unconditional Gaussian `sigma*C`, fixed expected-batch denominator,
  canonical Poisson inclusion, schedule-only public event trace와 hash chain을 구현했다. public
  trace에는 selected ID, realized count, sampling/noise seed를 넣지 않는다.
- 10 unit tests가 PASS했다. 9개 synthetic integration check에서도 analytic clip, 환자
  mean-before-clip 대 illegal clip-before-mean 구분, empty noise-only exact replay, fixed denominator,
  262,144 Gaussian draws, 20,000 Poisson steps, config fail-closed, trace tamper, K10 M1-I8/M2-P8
  dual-accountant replay가 모두 PASS했다. Gaussian standardized mean/std는
  `-0.000508965/1.000676144`, Poisson mean은 3.98205/4, empty frequency는
  0.01725/0.0168703이고 accountant drift는 0이다.
- 최초 public SD 2.1 disposable smoke는 결과와 무관한 stable-hash target/control unit을 썼고
  gradient/noise/fixed denominator/AdamW update는 통과했지만 M1·M2 모두 0/2 clip이었다. report
  SHA-256 `4B452CCA01B84CDA10F6CFEC6F7A2A72905FFF2877F34DB212B3C7A10F6A73C2`와 교체 이유를
  최종 report에 보존했다.
- 실제-vector clip path까지 검사하기 위해 frozen public calibration의 target/control cell별
  최대 gradient sentinel을 진단 전용으로 사용했다. M2는 정확히 4-image patient만 허용했다.
  이는 outcome-dependent public branch-coverage selection이므로 utility, attack, generalization
  평가에는 사용할 수 없다.
- 최종 실제 norm은 M1 `1.193793684`, `0.341953597` 대
  `C_image=0.2844870061`, M2 `1.122650613`, `0.277754529` 대
  `C_patient=0.1997973089`로 네 unit 모두 실제 clipping됐다. 같은 준비 context에서 arm별
  real-gradient sentinel과 deterministic test noise가 exact replay했다.
- 두 arm은 같은 fp32 rank-8 LoRA digest에서 시작했다. M1은 2 images/fixed denominator 8,
  M2는 2 patients·8 images/fixed denominator 4의 noised gradient로 AdamW를 한 번 step했다.
  adapter delta L2는 각각 `0.128836508`, `0.128837037`이고 1,659,904 trainable scalars가 모두
  finite하게 변했다. arm마다 새 모델을 사용한 뒤 폐기했고 checkpoint는 저장하지 않았다.
- 독립 verifier는 mechanism/builder를 import하지 않고 public selection commitment, trace chain,
  q/C/sigma/denominator, 155-order dual accounting, 실제 clip count, 동일 initialization,
  nonzero update, checkpoint 부재를 재검증했다. 또한 source AST에서 Gaussian draw가 conditional
  밖에 있고 legacy empty-sample skip이 실제 존재함을 확인했다. 최종 status는
  `PASS_EXECUTABLE_DP_TRAINER_GATE_PRIVATE_PILOT_STILL_BLOCKED`다.
- synthetic report SHA-256:
  `FB33CF43F6C5C0F6D4B60879A22E8EC97C4BD87BE933747171970210E12E94F5`; final public smoke:
  `34FCD5792F7615FAF6B3742AA18F57E8B4FB890CEEF923EEB42DE953183122D5`; independent:
  `EA2041E282153A547EEE89BBF1D6533315D7C071B5B712907FB2DCC5E6BE7A2C`.
- 세 report directory는 JSON 5개, 약 30 KiB뿐이다. resized corpus, latent/gradient bank,
  adapter/checkpoint, 생성 이미지는 만들지 않았다. private-train 이미지는 열지 않았고 private
  optimizer training도 시작하지 않았다.
- 상세 결정: `code_working/XRAY_DP_TRAINER_EXECUTABLE_GATE.md`. 다음 단계는 이 결과를 먼저
  보고한 뒤 exact core를 bounded `RESEARCH_ONLY` private pilot loop에 연결하는 것이다. secure
  backend가 없으므로 formal release claim은 계속 차단한다. 활성 LaTeX 7개와 PDF는 수정하지
  않았다.

## 53-CVPR. 2026-09-03 — Covariance-allocation discovery FAIL 및 branch 종료

- actual B=2 unit-local 실패 뒤 기존 full Grams에서 다음 allocation headroom을 확인했다.
  개별 35-template outcome을 계산·열람하기 전에 post hoc discovery 지위, public within-bin
  timestep rank 표현, leave-one-bank-out selection, nested patient leaveout과 사전 gate를
  `ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md`에 고정했다.
- 여덟 rank tokens 중 네 개와 complement를 두 patient slots에 배정하고 orientation을
  50:50으로 뒤집는 `C(8,4)/2=35` policies를 전수열거했다. shape은 `1111` 8개, `2110`
  24개, `2200` 3개다. 모든 policy의 40 patient gradient cells expected weight가 exact
  `1/40`이고 B=2, Q=4, batch perturbation multiset과 compute가 같다.
- 각 held-out bank에서 나머지 four banks로만 template을 선택했다. 다섯 folds 모두
  `rt_0123`, 즉 bin 0·1의 네 perturbations 대 bin 2·3의 네 perturbations를 나누고 환자
  orientation을 무작위 반전하는 `2200` policy를 선택했다.
- held-out ratios는 `0.966972`, `0.948496`, `0.988557`, `0.936119`, `0.967836`이고
  equal-bank ratio `0.961428`, bootstrap 95% CI `[0.944857, 0.976267]`이었다. 5/5 banks와
  16/16 nested patient leaveouts에서 ratio <1이었지만 사전 minimum `<=0.95`에 미달했고
  84/120 pair-wins 조건은 joint_03의 76 wins 때문에 4/5만 통과했다.
- `rt_0123`은 patient marginal ratio `1.128109`로 약 12.8% 악화시키고 mean clip rate를
  global `0.1614`에서 `0.3689`로 높였다. 대신 cross-patient contribution을 global
  `-4.2473e-10`에서 `-1.1100e-9`로 더 음수화해 joint arithmetic mean을 `3.7863e-9`에서
  `3.6378e-9`로 낮췄다. 이는 local mechanism 회복이 아니라 low/high-half global
  antithetic cancellation이다.
- pair-specific hindsight oracle geometric ratio도 `0.941876`, bank-shared hindsight는
  `0.956060`, pooled fixed best는 `0.961428`이었다. 실행 불가능한 pair oracle조차 최대
  개선이 약 5.81%라 35-template family 안의 추가 headroom이 작다.
- 사전 두 조건 실패에 따라 decision은
  `COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`다. threshold를 결과 뒤 낮추거나
  joint_03을 제외하지 않으며 fresh GPU gradients·X-ray confirmation으로 진행하지 않는다.
- 분석 구현을 import하지 않는 별도 감사기가 raw full Grams에서 21,000 pair-template,
  2,800 patient-template, LOBO와 80 nested folds를 재계산했다. 최대 상대오차 pair
  `2.29e-13`, patient `4.47e-16`, LOBO `2.34e-16`, audit PASS다. 전체
  `pcm_diagnostic` tests는 44/44 PASS다.
- protocol SHA-256:
  `45930814495B582FF859B65B8B732294FBCDED78488A594DAB0EE776D3C38445`; analysis:
  `34113735B6987CF54E8FC9590BE9C0AD33DD07C2B598399B4E52AB3B7E4CE3E0`; audit:
  `9D25E66D6E0A8953984DD547147D99EB9E858F4DF0C2E425DFC6264DD9EB0D9C`; tests:
  `D66AABE98CE39B6C6BE6DDA050BF10471B144C6BA44729AA24DAD62EBE1C8D56`.
- report summary SHA-256:
  `3801301320366053DDE0FC7AB6339EF918E0F26F4C6CD42F2E519E37620EE6E8`; secondary audit:
  `636D7D45E696B88425C8539A729EC1A9A3729088ADABF9A80EBD14D16BC358BF`.
- 상세 기록: `CVPR 주제 탐색/09_covariance_allocation_discovery_결과_2026-09-03.md`;
  report: `code_working/_reports/pcm_isic_sd21_covariance_allocation_discovery_v1_001/`.
- 이 결과로 현재 B=2 timestep-allocation method branch를 종료한다. 상위 patient-level DP
  medical diffusion 문제와 X-ray/DP/audit 인프라는 유지하되 다음 방법 가설은 다른 mechanism
  class에서 관련연구를 다시 확인한 뒤 시작한다. 활성 학위논문 LaTeX·초록과 동결 training
  contract는 변경하지 않았다.

## 54-DP. 2026-09-03 — K5 private-partition 4-step research runtime PASS

- full K5 feasibility를 바로 시작하지 않고 결과 전에
  `dp_training/private_research_dryrun_protocol.json`을 고정했다. SHA-256은
  `0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8`이다.
- K2는 no-training accounting sensitivity, K5는 feasibility tier, K10은 main이라는 기존 역할을
  유지했다. 이번 범위는 K5 M1-I8/M1-G8/M2-P8 각 4 optimizer steps뿐이다. M0와 K5 4,000-step
  feasibility는 포함하지 않았다.
- K5 `private_train`은 18,393 images/8,476 patients로 exact hash 검증했다. NIH source 자체는
  public proxy지만 private-image lifecycle을 시험하기 위해 선택 ID와 realized diagnostics를
  local restricted로 취급했다.
- 결과와 무관하게 모든 4-step Poisson schedule을 model/loss 전에 만들었다. M1 두 arm은 같은
  image selection과 같은 diffusion timestep/noise draw를 공유하고 DP Gaussian stream만 분리했다.
  M2는 patient Poisson 뒤 환자 안에서 최대 4장을 uniform without replacement로 골랐다. empty가
  나오면 재추첨하지 않고 noise-only update하도록 유지했으나 이번 실현에는 empty step이 없었다.
- OS entropy의 fresh 256-bit master를 process memory에만 두고 domain-separated seed로 default
  PyTorch CPU/CUDA generator를 구동했다. sampling/diffusion/DP seed 값은 어떤 JSON에도 쓰지
  않았다. 이 경로는 계속 `RESEARCH_ONLY_NONCRYPTOGRAPHIC`이며 release DP evidence가 아니다.
- local restricted realization은 shared M1 units가 step별 11/11/9/6, 총 37 images이고 M2는
  3/4/3/3, 총 13 patients·22 images였다. M1 두 arm의 모든 IDs/timesteps/diffusion-noise digest와
  첫 step unit-gradient digest가 일치했다. 이 수치는 runtime audit용이며 C/sigma/q 또는 실험
  matrix를 변경하는 데 쓰지 않았다.
- 실제 clip은 M1-I8과 M1-G8 모두 step별 1/5/1/3, 총 10/37이고 M2-P8은 2/1/2/2, 총 7/13이었다.
  작은 realized clip fraction은 calibration/utility 결과로 해석하지 않는다. 모든 unit gradient,
  noised update, adapter와 AdamW state가 finite였다.
- 세 arm 모두 같은 initial adapter digest
  `16C4F33A2A86C52667B7AFE1513877EAB74F9C54DD55D25AFB4D9AA47EF84172`에서 시작해 4개의
  nonzero step을 완료했다. total delta L2는 M1-I8 `0.2972447111`, M1-G8 `0.2972437096`,
  M2-P8 `0.2971245063`이다. model은 arm마다 폐기했고 checkpoint는 0개다.
- 4-step loop time/peak allocated CUDA는 M1-I8 17.5620s/1,981,990,400 bytes,
  M1-G8 17.4108s/1,982,506,496 bytes, M2-P8 7.9454s/2,088,136,704 bytes다. 단순 loop-only
  4,000-step 선형 외삽은 세 DP arm 합계 약 11.92시간이지만 M0·평가·checkpoint·실패·반복을
  제외한 계획 참고치일 뿐이다.
- arm별 schedule-only public trace 4 events와 Opacus/Google 155-order accountant를 재계산했다.
  conservative epsilon은 M1-I8 `4.36737943 @ delta=1e-5`, M1-G8
  `1.49500466 @ delta=1.32653965e-8`, M2-P8 `4.33243715 @ delta=1e-5`다. 이는 네 event의
  consistency 수치이지 release claim이 아니다.
- runner/mechanism을 import하지 않는 verifier가 K5 unit membership, M2 patient boundary,
  resource cap, M1 matching, clipping/aggregation, public trace, accountants, ID 비공개, seed
  nonserialization, source ordering과 checkpoint 부재를 독립 확인했다. 10 mechanism tests+5 runtime
  tests, 총 15개가 PASS했다.
- public report SHA-256:
  `77FA9097F1FA161E42E72B7E182F5E0A59CB47EC5D89209929DEE0E95767FB87`; public traces:
  `FB0C25EFD601C56871655C08F4F4BE0DD51AA8EE093A89E8C00A5D2F89C6CBC1`; restricted diagnostics:
  `085A518417279A07D2D2E3D3D5E60A3732262345BE5D60274473F52FA6B2DC3A`; independent:
  `27D81FF7565659B3168CECCBA3A94DE31A92CA29154943559AC4510E32B283F9`.
- 네 JSON은 108,309 bytes이며 latent/gradient bank, adapter, checkpoint, 생성 이미지는 저장하지
  않았다. 상세 결정은 `code_working/XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md`다. 다음은 먼저
  보고한 뒤 M0를 포함한 full K5 4,000-step feasibility 실행·retention·utility 규칙을 별도로
  고정하는 것이다. active LaTeX 7개와 PDF는 변경하지 않았다.

## 55-DP. 2026-09-03 — K5 full-feasibility 실행·평가 계약 독립 검증 PASS

- 직전 4-step 결과를 4,000-step으로 자동 확대하지 않고, full K5 optimizer update 전에
  `code_working/dp_training/build_k5_feasibility_protocol.py`로 별도 계약을 만들었다. 실제 학습,
  model load, 생성은 이번 gate에 포함하지 않았다.
- K5의 목적은 one-seed feasibility다. SD 2.1이 흉부 X-ray domain에 실제 적응하는지 M0로 먼저
  확인하고, 동일 조건의 M1-I8/M1-G8/M2-P8가 실행 가능한지와 utility collapse 여부를 본다.
  confirmatory 비교·임상 검증·release 증거는 K10 역할이며 K5 결과로 K10 arm을 제거/대체하지
  않는다. K5 attack과 downstream classifier도 실행하지 않는다.
- exact 순서는 M0 -> M1-I8 -> M1-G8 -> M2-P8이고 모두 4,000 fixed steps다. P256, pinned SD 2.1,
  fp32 rank-8 LoRA 1,659,904 parameters, AdamW `lr=1e-4`를 공통 사용한다. M0는 epoch별 fresh
  permutation의 fixed batch 8이며 총 32,000 exposures다.
- DP arm은 upstream frozen accountant와 다시 연결했다. K5 M1-I8은 `N=18,393`, `q=8/N`,
  `C=0.2844870061`, `sigma=0.4007042919`; M1-G8은 같은 q/C에 `sigma=0.8350657830`;
  M2-P8은 `N=8,476`, `q=4/N`, `C=0.1997973089`, `sigma=0.4044571458`이다. M1 두 arm은
  schedule/diffusion draw를 공유하고 DP Gaussian만 분리한다. M2는 환자당 최대 4장 평균 뒤 한 번
  clip한다. empty DP step은 재추첨 없는 noise-only update다.
- checkpoint는 restricted LoRA-only step 1,000/2,000/4,000으로 고정했다. 1,000/2,000은 arm당
  condition별 2장, 총 14장의 시각 diagnostic뿐이며 checkpoint 선택·조기중단·정량 gate에 쓰지
  않는다. 정량 평가는 오직 step 4,000이다.
- final 생성은 B0+네 arm 각각 448장, 총 2,240장이다. no finding, pneumothorax, pneumonia,
  consolidation, pleural effusion, mass opacity, nodule opacity 각각 64 prompt/seed를 모든 model에
  동일 적용한다. DDIM 50 steps, guidance 7.5, batch 4다.
- real reference는 K5 `public_development`에서 scarcity-first+SHA-256 순위로 448장/448 distinct
  patients를 고정했다. strata는 pneumothorax 64, pneumonia/consolidation 128, effusion 64,
  mass/nodule 128, no finding 64다. identifier는 local-only다.
- 현행 흉부 X-ray 생성 평가 근거를 반영해 primary small-sample metric은 RAD-DINO KID,
  mode/diversity는 PRDC k=5와 effective rank/distance, condition alignment는 BioViL-T cosine으로
  정했다. Inception-only FID는 금지했다. RAD-DINO는 NIH 전체로 pretrain됐으므로 비교 feature
  screen일 뿐 독립 임상 validator나 privacy evidence가 아니며, BioViL-T도 weak-label alignment다.
- M0는 pixel sanity, B0 대비 KID 개선의 upper 95% CI<0, alignment 개선의 lower CI>0, coverage
  증가를 모두 만족해야 domain-adaptation PASS다. DP arm은 pixel sanity 실패 또는 coverage/effective
  rank가 M0의 10% 이하일 때만 `COLLAPSED`로 표시하되 결과가 나쁘더라도 숨기거나 arm을 교체하지
  않는다.
- private root와 RNG/optimizer state는 Windows DPAPI CurrentUser envelope 안에만 두고 step 0과
  매 250 step에 atomic current+previous로 저장하도록 고정했다. envelope 손실/손상 때 한 arm만 새
  seed로 재시작하지 않고 실패를 남긴 뒤 전체 matrix를 새 run ID로 시작한다. in-memory DPAPI
  round-trip은 PASS했지만 4-step uninterrupted 대 2+resume+2 exact-equivalence 구현은 다음 gate다.
- M1 realized limit 64 images의 4,000-step union bound는 약 `8.52e-33`, M2 limit 32 patients의
  bound는 약 `6.72e-16`이다. 초과하면 resampling 없이 abort한다. 현 계획은 duplicate corpus나
  persistent latent/gradient bank가 없고 new cache/artifact 2.5 GiB를 보수적으로 허용한다. arm 시작
  전 free 15 GiB를 요구한다. 예상 총 wall time은 16--21시간의 planning range이며 보장이 아니다.
- builder와 무관한 verifier가 448 generation rows와 448 reference rows를 전부 독립 재구성하고,
  DP entry, resource bound, DPAPI, disk/GPU boundary를 확인해
  `PASS_PROTOCOL_INDEPENDENTLY_VERIFIED_FULL_K5_NOT_STARTED`를 냈다. 확인 당시 RTX 3070은
  utilization 1%, free space는 43.26 GiB였다.
- protocol SHA-256:
  `2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA`; generation tasks:
  `B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B`; local reference:
  `99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E`; independent report:
  `49E715F715B5634B4071EA843AF8FC4A85666DD2E9EBC51D3D51DE1D89472A1B`.
- 상세 계약은 `code_working/XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md`다. 다음은 pinned RAD-DINO/
  BioViL-T evaluator validity와 encrypted exact-resume를 구현·검증하고, full runner/environment를
  hash-freeze한 뒤 다시 보고하는 것이다. 그 전에는 장시간 4,000-step 학습을 시작하지 않는다.
- 독립 verifier 첫 호출은 report를 쓰기 전에 M1 극소 tail bound 비교에서 중단됐다. builder의
  `8.51895253465464e-33`과 별도 recurrence의 `8.518952534676757e-33` 차이는 absolute
  `2.21e-44`, relative 약 `2.60e-12`였는데 초기 허용값이 relative `1e-13`으로 과도하게
  엄격했다. verifier 수치 비교만 relative `1e-10`으로 수정했고 protocol 값, resource limit,
  실험 decision threshold는 바꾸지 않았다. 이후 clean output으로 PASS했다.
- 기존 mechanism/runtime tests 15/15를 다시 통과했고, `thesis.tex`, 국·영문 초록,
  Chapters 1/2/4/5의 활성 LaTeX 7개 SHA-256은 Section 25.5 baseline과 전부 동일하다. full-run
  restricted/public directory도 존재하지 않아 실제 4,000-step 실행이 시작되지 않았음을 재확인했다.

## 56-DP. 2026-09-03 — K5 evaluator 및 DPAPI exact-resume 사전검증 독립 PASS

- 시작 전에 사용자에게 이번 evaluator+resume gate는 정상 3--5시간, 의존성 충돌 시 6--8시간,
  추가 저장 1--2 GiB로 보고했고, 이후 full 4,000-step matrix는 별도 16--21시간임을 분리했다.
  이번 단계에서는 장시간 학습을 시작하지 않기로 다시 고정했다.
- RAD-DINO는 exact revision
  `110cbc18d5133582e320b43d53bf5c44e410c936`의 346,345,912-byte weights가 이미 cache에 있었고,
  BioViL-T exact revision `692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23`의 text 440,909,232
  bytes와 image 109,745,561 bytes를 받았다. 세 SHA-256은 모두 사전 계약과 일치했다.
- encoder output 전에 별도 evaluator protocol을 고정했다. 448 real references의 복합 stratum은
  weak label로 exact seven prompts에 매핑했다. 결과 counts는 consolidation 89, mass 48,
  no-finding 64, nodule 80, effusion 64, pneumonia 39, pneumothorax 64다. condition+selection hash로
  정렬한 뒤 최대 condition count인 89만큼 cyclic shift해 모든 donor patient/condition이 다르고
  prompt 빈도는 같은 derangement를 만들었다.
- CI는 five-stratum PCG64 bootstrap 2,000회, percentile linear로 결과 전에 확정했다. real input은
  frozen full-field P256 LANCZOS 뒤 RAD-DINO published 518 processor와 BioViL-T official
  512-resize/448-crop/ExpandChannels를 사용했다. matrix와 cuDNN TF32를 모두 껐다.
- 448장 전부 decode됐고 RAD-DINO `[448,768]` feature는 finite, effective rank
  `184.65758447`, total variance `159.71734024`, first-eight replay drift 0이었다. model phase는
  30.69초, peak allocated CUDA 480,300,032 bytes였다.
- BioViL-T `[448,128]` image와 `[7,128]` text feature는 finite/unit-normalized, image replay drift
  0이었다. matched cosine mean `0.19212072`, frequency-preserving deranged mean `0.13153649`,
  difference `0.06058422`, frozen 95% CI `[0.03795662,0.08322361]`로 overall validity PASS다.
- 이 PASS를 과장하지 않는다. descriptive condition mean은 consolidation `-0.11776463`, no finding
  `-0.05493903`, effusion `-0.12947672`, pneumonia `-0.06193671`로 음수다. mass/nodule/
  pneumothorax만 양수였다. 따라서 BioViL-T는 overall weak-label alignment screen으로만 쓰며,
  condition별 값은 보고하되 일곱 질환 임상·진단 validity로 주장하지 않는다.
- 사용자와 연구목표를 다시 대조해 이 condition별 약점은 **공개할 비차단 한계**로 확정했다.
  이번 K5의 목표는 질환별 임상 정확성이 아니라 privacy-unit별 DP/공격/일반 fidelity·diversity와
  model--receipt--PP-Mark 출처 연결이다. CheXGenBench, Metrics that Matter와 BioViL/ChestXray14의
  weak-label 설정도 단일 자동지표와 임상 유용성을 분리해야 함을 뒷받침한다. 수치 삭제나 frozen
  overall gate 완화는 하지 않는다. 향후 진단·질환별 증강 주장을 추가할 때만 expert-labelled
  downstream 또는 전문의 판독을 별도 설계한다.
- evaluator core에 float64 post-encoder KID, PRDC, effective rank, descriptive Frechet, bootstrap,
  P256 loader를 구현했다. formula/replay/preprocessing tests 6개가 PASS했다. feature와 per-image
  score는 저장하지 않았다.
- 독립 verifier 첫 실행은 RAD-DINO까지 일치한 뒤 BioViL-T aggregate mean이 약 `4.27e-5` 달라
  중단됐다. 원인은 독립기에서 matmul TF32만 끄고 `cudnn.allow_tf32=False`를 누락한 것이었다.
  frozen primary runtime과 같은 설정을 명시하고 448장을 처음부터 다시 인코딩하자 effective rank,
  mean과 CI가 일치했다. metric, threshold, prompt mapping은 변경하지 않았다. 직접 script path로
  잘못 호출해 package import 전에 1회 종료된 invocation도 있었으며 output/model result는 없었다.
- `secure_resume.py`는 torch state를 memory BytesIO에만 serialize하고 inner SHA-256 framing 뒤
  Windows DPAPI CurrentUser로 encrypt한다. ciphertext `.new`를 exclusive write+flush+fsync하고,
  decrypt/hash/schema 확인 뒤 current->previous와 new->current를 `os.replace`한다.
  deserialization은 `weights_only=True`다. roundtrip, plaintext marker 부재, rotation, tamper,
  stale-new fail-closed의 5 tests가 PASS했다.
- 별도 pre-result contract 아래 M0, M1-I8, M1-G8, M2-P8 각각 synthetic fp32 adapter+exact AdamW로
  uninterrupted 4 steps와 step0 save->2 steps->step2 atomic rotation->live state 폐기->decrypt/
  restore->2 steps를 비교했다. adapter, full optimizer, explicit generators, global CPU/all CUDA RNG,
  M0 permutation/cursor/epoch, public trace/head, committed step, frozen digests와 내부 root가 전부 exact였다.
- step0 envelope는 step2의 previous로 남았고 save는 Torch RNG를 소비하지 않았다. plaintext file은
  없었고 temporary encrypted envelopes는 검사 후 자동 제거됐다. runner/mechanism/resume module을
  import하지 않는 독립기는 source/AST와 public secret boundary를 감사하고 별도 DPAPI+AdamW+
  generator/global-RNG 4-vs-2+2를 다시 실행해 exact PASS했다. 이는 synthetic restart evidence이며
  실제 SD 2.1 full checkpoint나 privacy/utility evidence는 아니다.
- combined regression은 26/26 PASS다. 새 code/protocol/report는 292,310 bytes(0.2788 MiB),
  BioViL-T cache는 550,898,615 bytes이고 기존 RAD-DINO cache는 346,348,659 bytes다. duplicate image
  corpus/feature bank는 없다. final pre-documentation free space는 40.94 GiB였다.
- evaluator protocol/result/independent SHA-256은 각각
  `C1283BD37077258F1997BE60FDEE3F49F8C782B0AF3110FFA226EACF11D8F2C9`,
  `914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5`,
  `A55672E5D5C68491C75AAE4770F1C13A1F1B43BCCD721E7F732BF3C8D9560CAD`이다.
- resume protocol/source/result/independent SHA-256은 각각
  `4B4B9F20C4830C541ED857EDB751FD00D4032E0ACDB4FCE0C84DC18436808F75`,
  `360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47`,
  `89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF`,
  `720D7A9DDE7A9DD862AE84B4053DBF0F704B8100CE0C7166D28B56EFE76BA463`이다.
- 문서 반영 뒤 새 Python 8개를 다시 compile하고 combined regression 26/26을 재실행해 PASS했다.
  네 public/independent JSON status와 위 네 result hash가 다시 일치했다. 활성 LaTeX 7개의
  SHA-256도 Section 25.5 baseline과 전부 같았다. full K5 restricted/public run directory는 계속
  존재하지 않았고 최종 확인 시 C: free space는 40.318 GiB였다.
- 상세 기록은 `code_working/XRAY_K5_EVALUATOR_RESUME_PREFLIGHT_GATE.md`다. 다음은 actual full runner에
  evaluator/resume contract를 연결하고 source/environment를 hash-freeze해 독립 검증한 뒤 launch
  command·arm별 시간/저장 계획을 먼저 보고하는 단계다. 그 전에는 M0도 시작하지 않는다.

## 54-CVPR. 2026-09-03 — UCAN 공개 actual-gradient 진단 FAIL 및 branch 종료

- gradient/timestep-allocation 분기를 끝낸 뒤 별도 mechanism 가설로 Unit-Coupled Antithetic
  Noise(UCAN)를 제안했다. 같은 환자의 두 X-ray에 `z,-z` diffusion corruption noise를 주되,
  ordinary IID, noise-only antithetic, shared-t IID, shared-t antithetic 네 arms를 분리했다.
- 결과 전에 `code_working/unit_coupled_noise/NIH_CXR14_UCAN_PUBLIC_DIAGNOSTIC_PROTOCOL_V1.md`를
  고정했다. protocol SHA-256은
  `C3AEC580ADF99D026B5210BD0395875DB4C4D905E963806FE58BE7CE3498D45B`이다. coupling code의
  다섯 unit tests가 PASS했다.
- 이전 clip calibration 144명과 겹치지 않는 NIH K10 public-development 16명(target/control 각
  8명), 환자당 4 records, 5 banks에서 실제 SD 2.1 rank-8 LoRA full gradients 320개를 계산했다.
  optimizer update, DP noise, private-role data, checkpoint는 없었고 raw gradients는 Gram 통계 뒤
  폐기했다. 실행시간은 126.86초였으며 adapter hash는 전후 동일했다.
- primary complete UCAN/shared-t IID의 clipped actual-B2 ratio는 `1.074199`, bootstrap 95% CI는
  `[0.965688,1.206865]`, bank LOO wins 1/5, patient LOO wins 0/16이었다. complete UCAN/ordinary
  IID는 `1.008095`, noise-only antithetic/ordinary IID는 `1.017928`이었다.
- shared-t IID/ordinary IID secondary ratio는 `0.938462`였으나 CI `[0.852206,1.028220]`가 1을
  포함하고 기존 timestep 선행·폐기 분기와 가깝다. 같은 결과에서 이를 사후 새 방법으로
  승격하지 않는다.
- unclipped UCAN/shared-t IID `0.284986`은 shared-t IID raw outlier 영향이고 실제 clipping 뒤
  역전됐다. 보호 메커니즘이 사용하는 clipped update를 버리고 이 수치로 구제하지 않는다.
- 최종 status는 `FAIL_DO_NOT_PROMOTE_THIS_UCAN_IMPLEMENTATION`; antithetic corruption-noise
  branch를 닫고 training pilot을 금지했다. 보고서는
  `code_working/_reports/nih_cxr14_ucan_public_diagnostic_v1_001/report.json`에 있다.
- clipped cells SHA-256은
  `F2D3B5EDD3D5C6F51267DAA37645F6A0F8E00CEC7C0E802A957D1F64D9A292B4`, unclipped cells는
  `CBFE71A2D55FAAF8ECE3F3BDC7A5E7F6A9DB6336D719EED5C66834F502A81782`, gradient metadata는
  `5913E1718CC1911883876C7242692AE4A15B93E2C9E65A8A3BEFB4427529BD48`이다.

## 55-CVPR. 2026-09-03 — Patient-Set Private Diffusion reset과 데이터 전제 1차 진단

- PFDM, DPDM/DP-LoRA, medical DP-LDM, P3SGD/user-level DP, M2M, longitudinal CXR diffusion,
  MedReID, FedDP-PALD, anonymization·attack 문헌을 claim 단위로 다시 대조했다. 중앙
  `patient add/remove DP + joint synthetic medical image-set generation`의 정확한 결합은 이번
  1차 문헌 검색에서 찾지 못했으나, 확대 검색 전 `first/최초` 주장은 금지했다.
- 새 1순위를 Patient-Set Private Diffusion(PSPD)으로 정했다. 한 환자의 가변 길이 반복영상
  묶음을 privacy unit과 generative object 양쪽에서 동일하게 취급하고, permutation-equivariant
  SetAdapter로 stable within-patient structure와 record-level finding variation을 공동 생성하는
  방향이다. standard IID timestep/noise를 유지하며 UCAN이나 단순 record 평균을 재명명하지 않는다.
- 구체 후보는 frozen/shared SD 2.1 UNet+rank-8 LoRA, small zero-init SetAdapter, valid mask와 random
  slot permutation, singleton identity behavior, valid-record loss 평균 후 one patient gradient,
  patient-wise one clip과 Gaussian mechanism이다. 첫 논문은 central patient-DP가 core이고
  federation은 후속으로 둔다.
- 생성기 구현 전에 데이터 전제를 outcome-blind하게 검사하도록
  `code_working/patient_set_diffusion/NIH_CXR14_PATIENT_SET_PREMISE_PROTOCOL_V1.md`를 고정했다.
  protocol SHA-256은
  `02B291BA61F90E1B0E5011AAC6A687BCC6448B949520C9D12C2ECE49A440BBA4`이다.
- 모든 이전 gradient 실험 160명을 제외한 NIH K10 public-development 80명(target/control 각
  40명), 320 images를 frozen DINOv2 ViT-B/14로 검사했다. feature 실행은 13.75초였고 feature
  vector는 저장하지 않았다. selection SHA-256은
  `87B356A6175493E309DE2D84E8D0769B4764005BEE809AA8EB5CF5D5EF544AB0`이다.
- 사전 absolute within-minus-cross cosine gap은 `0.011060 < 0.05`라 FAIL이었다. 따라서 최종
  status를 `FAIL_PATIENT_SET_DATA_PREMISE`로 유지하고 threshold를 사후 완화하지 않는다.
- 나머지는 balanced same-patient AUC `0.877852`, retrieval R@1 `0.65625`(chance의 `69.78x`),
  여러 exact-label set 환자 비율 `0.7125`, within-patient label Jaccard distance `0.500799`로
  4/5 gates가 통과했다. within different-label cosine `0.986063` 대 cross-patient same-target,
  same-label `0.974805`도 관찰됐지만 DINO의 acquisition-artifact 가능성은 남는다.
- 이 결과는 generator나 DP utility 성공이 아니라 fresh scale-free confirmation을 할 근거다.
  다음은 비중복 2--3 record cohort에서 AUC·retrieval·label variation을 새로 동결한 뒤, 통과할
  때만 public Q=2 SetAdapter smoke와 parameter/UNet-call matched non-DP generation gate로 간다.
- 현재 authority는 `CVPR 주제 탐색/10_CVPR_reset_관련근접연구_새방향_2026-09-03.md`, UCAN
  결과는 11번, PSPD premise 결과는 12번이다. 01--09의 방법 추천은 역사 기록으로 남긴다.

## 56-CVPR. 2026-09-03 — PSPD scale-free 독립 확인 5/5 PASS

- premise-v1의 absolute cosine-gap FAIL을 같은 80명에서 완화하지 않았다. v1 결과만 근거로
  scale-free ordering criteria를 새 protocol에 고정하고, calibration 144명·UCAN 16명·v1 80명,
  총 240 unique patients를 전부 제외했다.
- `NIH_CXR14_PATIENT_SET_PREMISE_CONFIRMATION_PROTOCOL_V2.md` SHA-256은
  `1ABE0C90DCF91D9FC83635358D1533381DA115994C68959976274C792240E212`이다. 정확히 2장 또는
  3장인 환자를 `(target,n)` 네 cells에서 각 40명씩 SHA-256 선택해 160 patients·400 images를
  구성했다. target0/1과 n2/3를 완전 균형화해 한 subgroup 의존성을 별도로 검사했다.
- frozen DINOv2 ViT-B/14 feature 실행은 16.67초였고 embeddings는 저장하지 않았다. pooled
  balanced same-patient AUC `0.880605`; cell AUC는 target0_n2 `0.8650`, target0_n3 `0.908472`,
  target1_n2 `0.8675`, target1_n3 `0.863333`이었다.
- mean within-patient similarity가 같은 cell의 cross-patient mean보다 높은 patient 비율은 전체와
  각 cell 모두 `0.975`였다. global R@1은 `0.48`, exact chance `0.004010`의 `119.7x`; cell
  chance multiple도 `31.6x--39.17x`로 모두 고정 gate를 넘었다.
- multiple exact-label-set patient fraction `0.75625`, within-patient label Jaccard distance
  `0.582240`이었다. within different-label 203 pairs cosine `0.985816` 대 cross same-cell,
  same-label 6,970 pairs `0.972797`도 기록했으나 DINO acquisition-artifact 가능성은 남겼다.
- 다섯 conjunctive gates가 모두 통과해 status는
  `PASS_PATIENT_SET_DATA_PREMISE_CONFIRMATION`이다. 이 PASS는 v1 FAIL을 지우지 않고 Q=2
  architecture smoke만 허용하며 generator, patient-DP utility 또는 clinical identity 결과가 아니다.
- selection SHA-256은
  `E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8`, report SHA-256은
  `45828BF811F3F7D198A11FBEB04F672DCEA6F9A2C6E9F99A92C95B671159F721`이다.

## 57-CVPR. 2026-09-03 — 실제 SD 2.1 Q=2 SetAdapter architecture smoke PASS

- 결과 전에 `NIH_CXR14_SETADAPTER_ARCHITECTURE_SMOKE_PROTOCOL_V1.md`를 동결했다. SHA-256은
  `C6766B87010029DB412DBD45A62E9B4FB50E5CE275AFE717EC4B9E4DFCEB5DCD`이다. v2 target0_n2
  cell에서 별도 salt로 한 public patient의 두 records를 골라 standard IID timestep/noise를 썼다.
- `set_adapter.py`는 SD 2.1 UNet mid-block 1,280-channel spatial-pooled record tokens를
  128 dimensions·four-head self-attention·2x FFN으로 교환하고 residual로 되돌린다. slot position
  embedding은 없고 valid mask·singleton exact identity를 지원하며 output projection은 zero-init다.
- SetAdapter trainable scalars는 정확히 463,872, 기존 rank-8 LoRA는 1,659,904, 합계는
  2,123,776 fp32다. selection/metric/adapter pure unit tests 10개와 py_compile이 PASS했다.
- 실제 Q=2 SD 2.1에서 zero-init/bypass max abs `0.0`; denoising loss `0.155451`; LoRA gradient L2
  `0.136963`; SetAdapter/output-projection gradient L2 `0.000517903`; 모든 trainable gradient는
  present·finite했다. 결합 patient gradient L2 `0.136964`와 diagnostic clip operator도 finite했다.
- optimizer 없이 output projection만 deterministic random probe로 잠시 활성화했다. record-swap
  permutation max abs `0.0`, record 2를 바꿀 때 record 1 output L2 `0.0402214`, Q=1
  singleton/bypass max abs `0.0`으로 invariance와 실제 cross-record graph를 확인했다.
- model 구간 실행은 3.16초, peak allocated CUDA `1.81906 GiB`로 7.5-GiB gate를 통과했다.
  optimizer, DP noise, accounting, checkpoint, latent/gradient/adapter retention은 없고 LoRA digest도
  변하지 않았다. status는 `PASS_Q2_SETADAPTER_ARCHITECTURE_SMOKE`다.
- implementation SHA-256은
  `19282445FA80C3BD8BF9BDFF37AE57A827848AA01E43B8DA67EC721B13AA8F42`, report SHA-256은
  `71A799FF86BD2B485AACC50AF0F260BC1AAD3D9E1FED10EB5C36F022994EDF6E`이다.
- 이 PASS는 구조가 작동하고 현재 GPU에 맞는다는 증거뿐이다. 다음은 matched public non-DP
  generation falsification이며, SetAdapter가 quality/diversity/privacy를 개선했다는 claim은 아직
  금지한다. 종합 authority는 `CVPR 주제 탐색/13_PSPD_독립확인_SetAdapter_구조smoke_2026-09-03.md`다.

## 58-CVPR. 2026-09-03 — 표준 SetAdapter held-out context-signal FAIL

- full generation보다 앞서 `같은 환자 companion을 실제로 이용하는가`를 분리하도록
  `NIH_CXR14_SETADAPTER_CONTEXT_SIGNAL_PROTOCOL_V1.md`를 결과 전에 동결했다. 이전 400 unique
  patients를 제외한 exact-Q2 train 64명/128 images, validation 32명/64 images를 사용했다.
- frozen SD 2.1 base와 no LoRA에서 463,872-parameter SetAdapter만 192 AdamW steps 학습했다.
  네 banks·128 cells에서 primary latent/timestep/noise/prompt를 고정하고 same-patient correct,
  same-target finding-matched shuffled, adapter bypass를 비교했다.
- correct/shuffled는 `1.0002136003530693`, correct wins는 `64/128=0.5`; target 0·1 ratios도
  `1.0001825810`, `1.0002538042`로 모두 불리해 primary gate를 실패했다. 반면 correct/bypass는
  `0.9330780849`여서 adapter가 generic residual capacity는 학습했다.
- primary-input difference와 pretraining three-arm difference는 모두 `0.0`; adapter는 변했고 base
  sentinel은 불변이었다. permutation/singleton errors는 `0.0`, peak CUDA는 `1.86633 GiB`, 실행은
  47.79초였다. checkpoint, adapter, latent, gradient, optimizer state는 저장하지 않았다.
- V1 표준 block에 direct own-token residual `tokens+attended`가 있어 companion을 무시할 수 있다는
  원인을 식별했다. 이를 사후 성공으로 바꾸지 않고 status는
  `FAIL_SETADAPTER_HELDOUT_CONTEXT_SIGNAL`로 고정했다.
- protocol/source/selection/report SHA-256은 각각
  `C832321AF319FDEDA06669291AC0B165F321F487087A2277DFE6D55E86584B50`,
  `254F11864F7E9C7AB0395B8B77E2C67F80B9B9D42C4C5AE1832352BB8019EB5A`,
  `B6951897C7270CC7A462496F82A5EC1CDDC65D7F37A15EA6E2D629809604E319`,
  `886C5E253877399C2311AD5C44A6761884BCD6D60BDC83D97E4A0B62B4624D06`이다.

## 59-CVPR. 2026-09-03 — strict cross-record-only 독립 확인 FAIL 및 현 PSPD 방법군 종료

- V1의 해석 가능한 결함 하나만 고쳤다. `cross_record_adapter.py`는 diagonal/self value attention과
  own-token residual을 모두 제거해 Q=2의 각 delta가 오직 다른 record value에서 오게 한다.
  cross-view attention 자체는 SPAD/EpiDiff/MV-Adapter/CAMEO/CorrAdapter 선행과 겹치므로 novelty
  claim이 아니라 원인 분리 control로만 사용했다.
- 결과 전에 `NIH_CXR14_CROSS_RECORD_CONTEXT_CONFIRMATION_PROTOCOL_V2.md`를 동결했다. V1 train
  64명과 exact schedule/draw만 재사용했고, 앞선 400명과 V1 전체 96명을 제외한 fresh validation
  32명/64 images를 선택했다. 실패 시 `standard IID denoising + pooled mid-block patient context`
  방법군을 종료하고 새 mask/timestep으로 구제하지 않는 stop rule을 명시했다.
- 447,360-parameter cross-only adapter의 192-step 결과 correct/shuffled는
  `1.0010702865865093`, wins는 `65/128=0.5078125`; target ratios는 `1.0015495184`와
  `1.0007231277`이었다. correct/bypass는 `0.9393897207`로 branch 자체는 작동했지만 정확한
  same-patient companion의 추가 정보는 쓰지 않았다.
- integrity, zero-init, permutation, singleton, finite update와 base invariance는 모두 PASS했고 peak
  CUDA는 `1.86611 GiB`, 실행은 53.29초였다. 어떤 model/tensor artifact도 남기지 않았다.
- status는 `FAIL_CROSS_RECORD_CONTEXT_CONFIRMATION_CLOSE_STANDARD_IID_CONTEXT_CLASS`다. 13번의
  DINO data premise와 architecture smoke 사실은 남지만 PSPD SetAdapter 방법 추천과 matched
  generation 허가는 철회한다.
- protocol/runner/adapter/selection/report SHA-256은 각각
  `7EF39EB5F88220971EB9F5BAB99ADFB28C6CAB208ACDDAEA175D69B7E7491943`,
  `0A4126A021BD0D77902FE224613F2C0D3F5220443B720EDF811B01A1279E5B73`,
  `51D8DE611F5D725D04FCC7B0FF46CA60CED5296881BCA01024320C128EC45783`,
  `9145E378DAFD976ACAECEE6B78ED8F8B28D73BCDB13A466111DECBB648D038C0`,
  `68E8392A2B73CBA1030F24FD9AABFA0294D2EF12B2F41D806BA70F9C5B09081B`이다.

## 60-CVPR. 2026-09-03 — longitudinal residual 후보 재조사와 로컬 데이터 경계

- EHRXDiff(CHIL 2025)는 previous CXR+subsequent EHR로 future CXR를, DDL-CXR(NeurIPS 2024)는
  previous CXR+irregular EHR로 individualized latent CXR를 이미 생성한다. MICCAI 2024--25에도
  longitudinal MRI completion/progression과 future radiology embedding diffusion이 있다. 따라서
  `이전 영상을 조건으로 미래 영상을 생성`하거나 time conditioning 자체는 신규 claim이 아니다.
- 다음 조건부 후보를 Patient-Private Longitudinal Residual Diffusion(PPLRD)으로 좁혔다. 첫 합성
  CXR 뒤의 synthetic prior를 stable-anatomy carrier로 두고 change residual transition만 학습하며,
  전체 patient trajectory gradient를 하나의 add/remove unit으로 clip/noise하는 가설이다. real prior를
  inference에 넣는 clinical forecast와 synthetic trajectory release의 privacy claim은 분리한다.
- 로컬 NIH K10 public-development는 4,831 images/1,816 patients, 886 multi-record patients,
  3,015 adjacent-available pairs를 가진다. exact `Follow-up #` gap 1은 2,279 pairs이고 weak
  finding-label set change는 1,964/3,015=`0.651410`이다.
- 그러나 NIH metadata header에는 image index, finding labels, follow-up number, patient ID/age/sex,
  view와 geometry만 있고 actual study date/time, report, medication/lab event는 없다. 로컬 MIMIC-CXR
  corpus도 찾지 못했다. `Follow-up #`는 order proxy일 뿐 elapsed time이나 causal progression으로
  쓰지 않는다.
- 최신 종합 authority를
  `CVPR 주제 탐색/14_SetAdapter_context_반증과_주제재판정_2026-09-03.md`로 갱신했다. 다음은 이전
  528 patients를 제외한 exact-consecutive order-proxy residual premise를 먼저 반증하고, 통과해도
  MIMIC 또는 동등 timestamp+report 자료 접근 전 generation/DP training을 시작하지 않는 것이다.

## 61-CVPR. 2026-09-03 — NIH order-proxy residual premise FAIL 및 PPLRD-v1 종료

- 결과 전에 `NIH_CXR14_ORDER_PROXY_RESIDUAL_PREMISE_PROTOCOL_V1.md`를 고정했다. 앞선 여섯
  selection의 union 528 patients를 제외하고 exact `Follow-up #` gap-1 candidates에서 no-change를
  먼저 hash-select한 뒤 change를 patient-disjoint하게 선택했다. 각 class train 96+validation 48,
  총 288 patients·288 pairs·576 unique images다.
- 14 findings+No Finding의 additions/removals 30 variables로 future-minus-prior feature residual을
  학습했다. ridge alpha는 `{0.01,0.1,1,10,100}`의 patient-hash five-fold training CV로 골랐고,
  true transition, within-class cyclic deranged training, zero-residual prior copy, sex-first
  finding-matched shuffled prior를 고정했다. DINOv2와 RAD-DINO가 모두 통과해야 했다.
- pre-run selection/transition/ridge/derangement/bootstrap/shuffle unit tests 6/6이 PASS했다. 두 true와
  deranged ridge 모두 가장 강한 shrinkage alpha 100을 training-only CV에서 선택했다.
- changed-label validation에서 DINO true/copy는 `1.140914543`, CI
  `[1.073629525,1.236753736]`, wins `8/48`; RAD-DINO는 `1.159127120`, CI
  `[1.101905578,1.245539608]`, wins `1/48`로 명확히 실패했다.
- true/deranged는 DINO `0.952613701`, CI `[0.884638365,1.019940568]`, wins `24/48`; RAD
  `0.978814107`, CI `[0.933881466,1.024486737]`, wins `31/48`이었다. 두 CI가 1을 포함하고
  primary copy 비교가 불리하므로 secondary ratio로 구제하지 않는다.
- correct same-patient prior/shuffled prior는 DINO `0.516104650`, CI
  `[0.396098268,0.685203688]`, wins `46/48`; RAD `0.319953448`, CI
  `[0.266192560,0.378757074]`, wins `48/48`로 강했다. 즉 identity/anatomy carrier는 있으나
  weak-label change direction은 없다는 분리 결론이다.
- no-change true/copy도 DINO `1.008680087`, RAD `1.007988247`로 약간 나빴다. status는
  `FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE`; NIH weak-label/order-proxy PPLRD route를 종료하고
  generation·DP training으로 가지 않는다.
- protocol/source/selection/report SHA-256은 각각
  `92B0B0D031255E6C606074F76FCCD081688AFC429779C53456C94DDAE995B14D`,
  `47B7096736563D3752F27AFDC0085441005809DD2AC381B0412E77A83AF25CCB`,
  `4086BF15E8C50547C12EEBF26BB755C923F1799AB9D1AA83CEBD3C669383E60C`,
  `EF0856415E435E0808FF81090E7F58687DFBAB94F635A55FBF82D2DDDC07CA3F`이다. 실행은
  73.37초, peak CUDA `0.65145 GiB`; aggregate report와 selection만 남기고 feature/regressor/model
  artifact는 저장하지 않았다.
- 최신 종합 authority는
  `CVPR 주제 탐색/15_PPLRD_NIH_order_proxy_residual_반증결과_2026-09-03.md`다. 다음은 strong
  patient identity signal을 leakage/suppression 문제로 전환할 수 있는지 직접 선행을 먼저 확인한다.

## 62-CVPR. 2026-09-03 — biometric 직접 선행 충돌과 exposure-tail endpoint premise PASS

- PrivDiff-Net(PMLR 2026)은 orthogonal cross-attention projection과 privacy-guidance로 CXR identity를
  억제하고 pathology를 보존한다. ICCV 2025 Semantic-versus-Identity는 identity blocking, medical
  semantic compensation과 MDL disentanglement를, MIDL 2026 short paper는 patient-level latent
  diffusion unlearning을 이미 다룬다. 따라서 identity-suppression/unlearning network는 신규 방법
  후보에서 제외했다.
- Kaiser et al. 2025는 CheXpert에서 variable-record medical user-level DP의 sampling/clipping adaptation과
  naïve group privacy를 비교했다. P3SGD와 general ULDP 선행도 있어 patient mean/clip, record-count-aware
  sampling과 native-vs-group 일반론은 claim할 수 없다.
- Nature 2026의 patient-specific MIA tail과 CheXGenBench의 CXR generative privacy benchmark 사이에서,
  현재 남는 좁은 질문을 `동일 patient budget의 image/group/native patient DP diffusion + 환자별
  biometric output-exposure tail`로 재정의했다. 이는 method가 아니라 generative privacy-unit study다.
- 결과 전에 `NIH_CXR14_PATIENT_EXPOSURE_TAIL_PREMISE_PROTOCOL_V1.md`를 동결했다. K10
  public-development의 multi-record 886명 전원을 사용하고, 각 환자에 first-order query와 last-order
  gallery를 정확히 하나씩 배정했다. negative는 sex/target/n-bucket 16 cells 내부 hash derangement다.
- 순수 테스트 7/7과 py_compile이 PASS한 뒤 1,772 images를 두 encoder로 실행했다. DINOv2 pair AUC
  `0.825795`, CI `[0.808454,0.843345]`, R@1 `0.239278`, CI `[0.211061,0.267494]`, rank>10
  `0.577878`; RAD-DINO AUC `0.988363`, CI `[0.984242,0.991894]`, R@1 `0.711061`, CI
  `[0.681687,0.739278]`, rank>10 `0.102709`였다.
- cross-encoder patient margin Spearman은 `0.505432`, CI `[0.451656,0.555337]`여서 모든 conjunctive
  gate가 PASS했다. RAD-DINO의 NIH pretraining 한계는 generic DINO 동시 gate로 분리했으며 외부
  일반화 수치로 부르지 않는다.
- status는 `PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE`다. 이는 generator leakage, DP utility,
  clinical identity 또는 patient-specific MIA AUC가 아니다. K5 장기 실행도 자동 허가하지 않는다.
- protocol/runner/tests/selection/report SHA-256은 각각
  `DF95C8A154509BF782E87D38B31D002D6FB65DB19E446F9B4E3CAD5D1792BCD2`,
  `4F4CD96DCC3DED272C13B2C905336D9F6B2A9BCDA332CB1E64BB9992107BA748`,
  `8C8A14C56B1EB22DA32BCA3697F3A35D3B3AA446D8F25E4D7F1A25CF4104B73B`,
  `03B1FE65F2EACAAC9964B0ED7F124C2EAC6D610F88574D464D327304E32984EC`,
  `8CF040117B4528170CB10EACC254457F29DD1AA4F750CB31C4A28BCA8CA2290D`이다. 실행은
  212.37초, peak CUDA `0.651451 GiB`; feature/model/optimizer/gradient/latent는 보존하지 않았다.
- 최신 authority는
  `CVPR 주제 탐색/16_biometric_선행충돌과_patient_exposure_tail_결과_2026-09-03.md`다. 다음은
  K10 generator output용 biometric exposure-tail addendum과 MedReID conformance를 결과 전에 동결한다.

## 57-DP. 2026-09-03 — K5 actual full-training runner·환경 독립 PASS

- 사용자에게 먼저 보고한 대로 4,000-step 장시간 학습은 시작하지 않고 actual runner, 환경,
  encrypted resume를 하나의 별도 gate로 처리했다. 기본 Anaconda Python에는 `diffusers`와
  `peft`가 없었지만 새 패키지를 설치하지 않고 기존 격리 환경
  `code_working/base_gate/.venv`를 찾아 사용했다. 고정 환경은 Python 3.11.5,
  torch 2.6.0+cu124, diffusers 0.30.2, peft 0.12.0이다.
- `dp_training/k5_full_runner.py`와 `run_k5_full_training.py`에 M0, M1-I8, M1-G8,
  M2-P8의 실제 SD 2.1 rank-8 LoRA 학습 경로를 연결했다. 팔 순서, fresh identical LoRA,
  M1 shared Poisson/timestep/latent-noise draw, independent DP Gaussian, M2 same-patient mean-loss
  before clip, empty-sample noise-only update, fixed denominator를 강제한다.
- step 0과 매 250 step마다 Windows DPAPI CurrentUser로 LoRA·전체 AdamW·experiment root·모든
  관련 RNG·trace head/count·frozen digest를 plaintext 임시 파일 없이 저장하고 current+previous를
  유지한다. step 1000/2000/4000에는 restricted LoRA-only safetensors를 원자적으로 남기며 4000만
  정량평가 대상이다.
- K5 18,393 images·8,476 patients와 4,000 steps를 재계산했다. M0/M1은 32,000 image exposures,
  image당 기대 `1.7397923123`; M2는 16,000 patient-unit exposures, patient당 기대
  `1.8876828693`이다. rank-8 LoRA one-seed feasibility로는 합리적이지만 confirmatory convergence나
  superiority 근거는 아니며 K10 역할을 대체하지 않는다고 고정했다.
- 실제 모델 사전시험은 public-development 8 patients·16 images, patient당 2 images로 동결했다.
  네 팔 모두에서 uninterrupted 4 steps와 2 steps→DPAPI 저장/rotation→live state 폐기→fresh UNet
  복구→2 steps를 비교해 arm당 13개 field가 bit-exact PASS했다. step-2 ciphertext는 약 20.24 MB로
  실제 1,659,904-value LoRA와 AdamW moments 전체를 포함했고 peak CUDA는 최대 약 2.15 GiB였다.
- 첫 v1_001 actual test 자체는 PASS했으나 이후 full-gate verifier가 Windows WDDM의 일반 C+G
  process memory `[N/A]`를 정수로 읽으려다 fail-closed 중단했다. 이는 학습 결과 문제가 아니라 idle
  parser 문제였다. 유휴 판정을 free VRAM `>=6500 MiB`, utilization `<=10%`, temperature `<80 C`로
  고치고 v1_002 protocol/result를 처음부터 다시 실행했다. 첫 primary gate는 삭제하지 않고
  `_reports/_superseded_failed_nih_cxr14_k5_full_runner_gate_v1_001_wddm_parser_20260903`에 보존했다.
- v1_002 protocol/result SHA-256은 각각
  `3921227498086C6D3E794FBE92DE12DF229E463D9F461642BDF71187B704924D`,
  `C7BC6F12B02980E3ADBF309DBD16479DC7DCCB67418B65D2F1B5FCD9D8540CD2`이다.
- 최종 독립 verifier는 runner를 import하지 않고 source AST, DPAPI/DP mechanism 구조, frozen input과
  model hashes, 양쪽 accountant, 32/32 regression, 환경·GPU·disk를 재검사하고 CLI의 default
  read-only validation을 실제 실행해 PASS했다. primary/environment/independent/launch-authority
  SHA-256은 각각
  `661A0F3F135C6DA4F0CAF1D000AD1A1B5A5CAC7A23786081F86D1A380199FE74`,
  `47BED599FE05BD9A7DA710BA80420D6B335943328A5E52D0188FA953E9767767`,
  `FB6EA9674FC5613B8FB3F7E750B6954F8C9535EDF621D042EAD9D9A201800BAB`,
  `A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2`이다.
- gate freeze 시 C: free는 약 43.1 GiB였고 restricted/public full-run root는 모두 없었다. 이후
  최종 확인 때 범위 밖의 `run_heterogeneous_panel_full_development_e0g4.py qwen` Python 작업이
  별도로 시작되어 private memory/pagefile을 사용하면서 free가 36.51 GiB로 변했다. 그 process는
  건드리지 않았고 B0와 각 arm 직전에 자원을 다시 확인한다. 이 full-run gate에서는 B0, M0,
  full-matrix private-train optimizer step,
  adapter, receipt, PP-Mark output은 생성하지 않았다. 활성 LaTeX 7개 SHA-256도 Section 25.5
  baseline과 전부 동일하다.
- 이 gate 종료 당시 다음 단계는 고정 `generation_tasks.csv`의 B0 448장을 full K5 matrix optimizer
  update 전에 생성·hash·검증하고
  먼저 보고하는 것이다. PASS B0 manifest 없이는 runner가 matrix initialize를 거부한다. 상세 문서는
  `code_working/XRAY_K5_FULL_RUNNER_GATE.md`다.

## 58-DP. 2026-09-03 — K5 B0 448장 생성·독립검증 PASS, qualitative off-domain flag

- B0 결과 전에 448 fixed prompt/seed, DDIM 50, guidance 7.5, batch 4, P256 RGB PNG 직렬화와
  pixel 기준을 동결했다. batch 단위 atomic progress/commit resume를 구현했고 unknown·extra·stale
  output은 fail-closed한다. 독립 verifier는 runner를 import하지 않고 448 task 재구성, 전 파일
  decode/hash/stat 재계산, 고정 batch 0·111 총 8장 재생성을 수행한다.
- 첫 v1_001은 image 0장, optimizer 0회 상태의 pipeline load에서 고장 난 inherited optional
  `onnxruntime` DLL 때문에 중단됐다. 기존 SD 2.1 base gate에 이미 있던 PyTorch-only ONNX 탐지 mask를
  누락한 구현 문제였다. 빈 progress와 protocol을 failed evidence로 보존하고, prompt/seed/model/sampler/
  batch/threshold 변경 없이 그 호환 처리만 추가한 v1_002를 결과 전에 재동결했다.
- v1_002 protocol/generation-core/runner/verifier SHA-256은 각각
  `49C18694F740053048971C88286B55C256F28FC8BB223B94A8F1132FA0686324`,
  `CC677A914CF60728DD798B8809F5267249DC5BE381C9009DA70AB2D8A838C1EB`,
  `B74F3A36BD3DA426CFFDBC6568212E0A8F7090F8013650783AFC4AE54808B5AF`,
  `11D752C9E078A07659F1481FC8282FACF7D1D181C9CAFBE305B3DCA75303276A`다. 전체 회귀검사
  39/39가 PASS했다.
- 범위 밖 `jailmeter/Qwen` 작업을 건드리지 않고 종료·GPU 냉각까지 기다렸다. B0는 112 batches,
  448/448장, reroll/reject/replacement 0으로 완료됐다. 467.17초, 1.04279초/image, peak CUDA
  2.9546 GiB, PNG 전체 약 38.31 MiB였다.
- primary와 independent 결과는 decode `1.0`, duplicate `0`, low-contrast `0`, saturated `0`,
  condition별 `64`로 exact 일치했다. restricted inventory SHA-256은
  `C385BA3A0C642E9B3A62B0D8FFF11232BBCF842FDC25102C5A26EFC89DB5C984`다. 독립 재생성 8장도
  모두 encoded PNG SHA-256 exact match했다.
- primary/independent/B0 manifest SHA-256은 각각
  `E4E0C0429FFB781110975F913B6FE298BB43188E74DD5BB27B8E594CDE2E4B47`,
  `A1A157A996ABBA631D4289DC18B490C894928815E4F6931DB756D9A6A1E12520`,
  `6CB11A45A5B99D61E5B44F7C9463A7832CF5439AE0DF7D497CF8CF752CC94962`다. full runner의
  B0 manifest 수용과 read-only 전체 validation도 PASS했다.
- fixed 35-image local grid는 35/35가 conventional PA chest X-ray가 아니거나 gross geometry
  failure였다. 이는 unadapted B0의 예상된 off-domain comparator 결과이며 integrity PASS와 구분한다.
  M0 domain adaptation은 hard gate다. 다음에는 matrix initialize 후에도 M0만 먼저 실행·생성·평가하고,
  M0가 frozen radiograph-domain gate를 통과하지 못하면 DP arms로 과학적 해석을 확장하지 않는다.
- 현재 full-run root, matrix initialization, M0, full-matrix optimizer step은 여전히 없다. Section
  54-DP의 별도 4-step dry-run은 B0보다 먼저였고 checkpoint 없이 폐기됐으므로 이 문장에 포함하지
  않는다. generated B0
  images와 35-image contact sheet는 restricted이고 release-eligible이 아니다. 상세 문서는
  `code_working/XRAY_K5_B0_GENERATION_GATE.md`다. 활성 thesis/abstract LaTeX 7개는 기준 SHA-256과
  7/7 동일하며 이번 단계에서도 수정하지 않았다.

## 63-CVPR. 2026-09-08 — 탈옥 트렌드 팩트 검증과 현 1순위 신규성 경계 재고정

- 외부 설명의 `텍스트 탈옥은 철옹성`, `멀티모달 탈옥은 확실한 블루오션`, `Patient-DP는 아무도 하지
  않은 개척지`라는 단정은 객관적 사실로 사용할 수 없다고 판정했다. 텍스트 탈옥은 문헌 축적도가
  높지만 NeurIPS·ICML 2025에도 강한 신규 공격이 계속 보고됐고, 멀티모달 영역은 비교적 새롭고
  활발하되 상대 난이도·연구 공백·연구비를 단정할 정량 근거는 없다.
- 처음 검토한 CVPR 2026 `Personalized Federated Training of Diffusion Models with Privacy
  Guarantees`는 visual jailbreak가 아니라 privacy-preserving federated diffusion 연구이므로 탈옥
  트렌드의 직접 근거로 인용하지 않는다.
- P3SGD(CVPR 2019), DP-FedEmb(CVPR 2023), Kaiser et al.(TPDP 2025) 때문에 Patient-DP,
  다중기록 user protection, group conversion 비교 자체는 신규 claim에서 제외한다.
- Nature 2026은 여러 기록을 가진 환자, aggregate MIA의 극단 위험 은폐, patient-level DP 필요성을
  직접 지지한다. 다만 분류모델 연구이며 의료영상 diffusion·생성 exposure는 향후 연구 범위이므로
  이를 현 주제의 동기와 평가축 근거로만 사용한다.
- 현 1순위는 유지한다. 정확한 간극은 동일 patient `(epsilon, delta)`와 근접 compute에서
  image-DP/group-converted DP/native patient-DP CXR diffusion을 비교하고 환자별 biometric,
  membership, near-copy exposure tail과 병리·품질을 함께 평가하는 것이다. endpoint premise는
  PASS했지만 native patient-DP 우월성과 신규 mechanism은 아직 입증되지 않았다.
- 상세 기록은 `CVPR 주제 탐색/17_탈옥트렌드_팩트검증과_현1순위_신규성경계_2026-09-08.md`다.
  학위논문 LaTeX와 동결 실험결과는 변경하지 않았다.

## 64-CVPR. 2026-09-08 — CVPR·유사 최상위 SOTA 전수 대조와 투고 가능성 판정

- CVPR/ICCV/WACV와 ICLR/ICML/NeurIPS/USENIX, Nature/Nature Biomedical Engineering,
  TMLR/MIDL의 직접·근접 연구를 다시 대조했다. DPDM·DP-LDM·DP-LoRA·PrivImage·dp-promise·RAPID가
  private diffusion 학습을, P3SGD·DP-FedEmb·Kaiser et al.이 patient/user-level DP를 이미 점유한다.
- Dar et al.은 의료 LDM patient-copy memorization을, CheXGenBench는 MIMIC-CXR 11개 T2I 모델의
  fidelity/privacy/utility와 training-image별 max ReID를, DeepSSIM++는 scalable medical near-copy
  metric을, Nature 2026은 분류모델의 patient-specific MIA tail을 차지한다. 따라서 기존
  `M1-G8 대 M2-P8 + tail plot`만으로는 CVPR main 기여가 부족할 가능성이 높다고 판정했다.
- 현 문제는 유지하되 1순위를 **patient-calibrated set-level diffusion attack/audit + 동일 patient
  budget DP 비교**의 하이브리드로 강화했다. 가변 record count와 within-patient correlation에 의해
  FPR이 왜곡되지 않게 patient membership/exposure evidence를 보정하는 새 감사법이 중심 기술 기여다.
  current DP arms는 새 감사법과 patient-unit 보호를 검증하는 실험 무대로 사용한다.
- 강한 baseline으로 Kaiser sampling/clipping adaptation, DP-LoRA+noise multiplicity, PIA/PIAN,
  DIME/quantile, 조건부 CLiD, Packhäuser/MaMI ReID, CheXGenBench metrics, DeepSSIM++/InvMM을 요구한다.
  member-only split-bias control과 patient-specific gold target-model subset도 필수로 고정했다.
- 현재 B0 integrity와 NIH endpoint premise만 PASS했고 M0, private full run, generator leakage,
  새 patient-set audit, 여러 dataset 결과는 없다. 따라서 `CVPR 범위에 맞는 주제 후보`이지 아직
  `CVPR 제출 가능한 결과`는 아니다. CVPR 2027 submission은 2026-11-16 AOE라 일정 위험도 높다.
- 상세 최신 authority는 `CVPR 주제 탐색/18_CVPR_SOTA_전수대조와_투고가능성_판정_2026-09-08.md`다.
  학위논문 LaTeX와 동결 실험결과·실행계약은 변경하지 않았다.

## 65-RECORD. 2026-09-09 — 전체 기록 감사·현재 authority 정리·B0 시간순서 정정

- 사용자의 지시에 따라 기록만 전수 대조한 뒤, 루트 `CURRENT_STATUS.md`를 최신 상태와
  supersession의 최우선 authority로 생성했다. 학위논문 통합 실험과 별도 CVPR 탐색 스트림의 현재
  질문·다음 승인 지점을 분리했다.
- 활성 LaTeX 7개의 SHA-256은 Section 25.5 기준과 7/7 동일했고 `build/thesis.pdf`도
  5,001,029 bytes의 2026-08-26 조판본 그대로다. 코드, 데이터, 동결 JSON/CSV, B0 PNG와 contact
  sheet는 수정하지 않았다.
- NIH raw corpus는 42,423 PNG/17,671,122,456 bytes, B0는 448 PNG/40,175,213 bytes로 기록과
  일치했다. B0 protocol, manifest, independent report, contact-sheet SHA-256도 기존 기록과
  일치했다. `_reports`의 JSON 120/120이 정상 파싱됐고 점검 대상 Markdown 55개에 UTF-8 replacement
  character가 없었다.
- 중요한 정정: Section 54-DP의 K5 `private_train` 4-step `RESEARCH_ONLY` dry-run은 B0보다 먼저
  실행됐다. 따라서 `B0가 모든 private-role optimizer update보다 먼저였다`는 넓은 temporal claim은
  철회한다. B0는 full 4,000-step K5 matrix 초기화와 그 matrix의 optimizer update보다 먼저
  생성·동결됐다고만 주장한다. dry-run은 별도 root에서 수행됐고 checkpoint 없이 폐기됐다.
- 해시 고정 JSON의 `B0_must_precede: ... every private optimizer update`,
  `generated_before_private_training: true`, `PASS_B0_FROZEN_BEFORE_PRIVATE_TRAINING` 문자열은 원문
  증거라 변경하지 않았다. 이후에는 `private training`을 full K5 matrix로 좁혀 해석하며, 선행 dry-run
  부재의 증거로 인용하지 않는다. B0의 448-file integrity와 off-domain comparator 역할은 유지된다.
- K-tier 역할 변경도 명시했다. 초기 X-ray metadata/pivot 기록의 K2 feasibility, K5 primary/main,
  K10 unit-stress 명칭은 후속 실행계약이 supersede한다. 현재는 K2 accounting sensitivity, K5
  one-seed feasibility, K10 confirmatory main plus contribution stress다. 이미지 수·환자 split·nested
  manifest와 hash는 바뀌지 않았다.
- 루트 README, `code_working/README.md`, 통합 note, experiment contract, X-ray metadata/pivot,
  full-runner, feasibility plan과 B0 gate에 현재 상태·역사 문장·정정 범위를 반영했다. 과거 gate의
  `다음`과 `NOT STARTED`는 gate-closure status로 명시했다.
- WORKLOG의 역순 삽입 항목 29/30은 `26A-OPS`/`26B-OPS`로 바꾸고, 40 이후 DP/CVPR 혼합 항목에
  branch suffix를 붙였다. 같은 numeric stem의 `53-DP`/`53-CVPR` 등은 서로 다른 병렬 스트림의
  stable ID이며 전역 순번이 아니다.
- 현재 학위논문 실험 정지점은 full-run root·matrix·M0·full-matrix checkpoint 없음이다. 별도 승인
  시 다음 mutating 단계는 M0만 실행·생성·평가하는 것이다. CVPR 스트림은 18번 최신 authority에
  따라 PC-SMEA exact statistic·threat model·null calibration·baseline 계약 동결이 다음 기록 단계며,
  어느 한쪽도 다른 쪽 실행을 자동 승인하지 않는다.

## 66-RECORD. 2026-09-09 — 교차문서 잔여 모순 정리·최종 무변경 재검증

- 65-RECORD 이후 최신 CVPR 18번 문서에 남아 있던 `private optimizer step은 시작되지 않았다`는
  현재형 표현을 바로잡았다. B0 전 K5 `private_train` 세 arm x 4-step `RESEARCH_ONLY` dry-run은
  있었고 checkpoint를 남기지 않았으며, 미시작 범위는 full 4,000-step matrix와 M0다.
- `XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md`에 그 dry-run이 B0보다 앞섰다는 후속 감사 주석을
  추가했다. `STEP5_SPLIT_LOADER_GATE.md`와 통합 note에는 초기 K2/K5 unit-sensitivity 명칭이
  역사적임을 표시하고 현재 K2 accounting sensitivity, K5 one-seed feasibility, K10 confirmatory
  main plus contribution stress 역할로 연결했다.
- `XRAY_K5_FULL_RUNNER_GATE.md`, 통합 note와 experiment contract의 오래된 `next`/`not started`
  문장은 해당 gate 종료 시점의 역사 서술로 바꿨다. 현재 실행 지점은 별도 승인 시 matrix 초기화 후
  M0만 수행하는 것으로 유지한다.
- 최종 read-only 재검증에서 활성 LaTeX 7/7 SHA-256과 PDF SHA-256
  `2376DA044DDA8CF83DEBB98F00478524E5BEEE9117F4FED019D140DC2AF8EF7F`가 기준과 일치했다.
  PDF는 5,001,029 bytes, 2026-08-26 13:47:21 조판본이다.
- NIH raw는 42,423 PNG/17,671,122,456 bytes, B0 image tree는 448 PNG/40,175,213 bytes로 다시
  일치했다. B0 protocol, generation task, result, manifest, independent verification, qualitative
  report/contact sheet와 progress의 SHA-256도 각 기록 또는 독립 verification 내부 commitment와
  일치했다.
- `_reports` JSON은 120/120 파싱 PASS, 확장한 Markdown 점검 집합은 55/55에서 UTF-8 replacement
  character 0, WORKLOG 항목 ID는 74개 중 exact duplicate 0이었다. restricted full-run root,
  `matrix.dpapi`, M0 completion은 모두 없었다.
- 이 기록 정리 시간대에 변경된 파일 16개는 모두 `.md`였다. 코드, 데이터, 활성 LaTeX/PDF,
  동결 JSON/CSV, B0 이미지/contact sheet는 변경하지 않았다.

## 67-PLAN. 2026-09-09 — CVPR 기여도 검토와 졸업논문 실증의 공통 산출물 운영

- 사용자는 졸업논문 전체 실증 범위를 유지하면서 CVPR 선행연구·기여도를 좁히고, 공통으로 쓸 수
  있는 실험은 함께 활용하되 조건이 달라지는 부분은 별도로 진행하는 의향을 밝혔다.
- `CVPR 주제 탐색/19_졸논_CVPR_공통실험_재사용과_분기계획_2026-09-09.md`를 작성하고 해당
  폴더 README에 연결했다. 18번은 선행연구·기여도 판정 기준을 유지하며 19번은 병행 운영 계획이다.
- NIH 원본·전처리·학습/회계 코드와 조건이 같은 checkpoint·평가 결과는 공유한다. CVPR의 추가
  dataset, 학습 메커니즘, privacy budget, 공격·보정·baseline은 별도 실행 범위로 명시한다.
- K10 전에 dataset, noise multiplicity, Kaiser 비교군, 위협모델·보정·평가 표본과 반복 범위를
  대조하도록 정리했다. 학습을 공유할 수 없는 차이가 생기면 기존 졸업논문 실험의 목적과 결과를
  보존하고 CVPR 조건을 별도로 추가한다.
- 현재 K5의 공격 제외, 개발/확증 분리, 연구용 PRNG의 RESEARCH_ONLY 경계를 유지한다. 단일 GPU의
  큰 작업은 순차 배치하고 원문 조사·설계·코드 작성·가벼운 CPU 테스트를 같은 기간에 수행한다.
- 이번 작업은 문서 3개의 추가·수정이다. 학습이나 공격은 시작하지 않았으며 기존 실행 코드,
  프로토콜, 데이터, 체크포인트와 활성 LaTeX/PDF를 변경하지 않았다.

## 68-RECORD. 2026-09-09 — 원고 초점·반복 근거·시간 추정과 지속 기록 요청 반영

- 사용자는 작업마다 기록을 남기고 있는지 확인했다. 이후에도 결정·근거·수치·변경 이유·남은 일을
  이어서 문서화하며 사용자 방향, 미검증 가설, 동결 계약과 실제 결과를 구분하도록 기록했다.
- 이번 대화의 상태 확인에서는 NIH raw 42,423 PNG/17,671,122,456 bytes, K5 private_train
  18,393 images/8,476 patients, B0 448 PNG/40,175,213 bytes와 고유 PNG hash 448개를 직접
  대조했다. reports JSON 120개가 정상 파싱됐고 PDF와 B0 manifest·independent report hash가
  기록과 일치했다. K5 manifest 전체에는 여러 partition이 들어 있어 학습용 수치는 private_train만
  필터해 확인했다. full-run root와 네 arm의 completion은 없었다.
- 찾던 CVPR 검토 문서 18개와 README는 이미 졸업논문 작업본의 CVPR 주제 탐색 폴더에 있었다.
  다른 위치에서 옮겨야 하는 해당 기록이 확인되지 않아 파일 이동은 수행하지 않았다.
- 19번 문서 Sections 7--9에 졸업논문의 동기·타당성·통합 실증·효과/한계와 CVPR의 좁은
  독립 기여·강한 baseline 비교·구성요소 분석·재현성이라는 초점을 반영했다. 졸업논문도 독립
  기여와 증거가 필요하며, CVPR도 SOTA 숫자만으로 평가하지 않는다는 구분을 남겼다.
- 공식 CVPR 2026 Reviewer Guidelines의 기술적 타당성·기여·신규성·영향 관련 문장을 확인했다.
  2027 심사지침을 확인한 것으로 표시하지 않았고 현재 환자 집합 감사 후보는 가설로 유지했다.
- K10 4조건×3 seeds의 목적, G8/P8 핵심 비교, 회계로 아는 결과와 학습해야 아는 결과,
  bootstrap과 학습 반복의 차이, 전체 범위 유지 의향을 기록했다. 3 seeds의 충분성이 입증됐거나
  전체 12회가 졸업논문의 보편적 최소 요건이라는 주장은 하지 않는다.
- K5 16~21시간은 기존 실행계획, K10 핵심 39~48시간과 추가 작업 포함 50~80시간은 짧은 K5
  실측을 이용한 추정으로 구분했다. 4~7일은 K10 준비·점검 포함 일정이며 연속 GPU 시간과
  다르다. 전체 5~9주는 남은 졸업논문 연구·집필의 보조 추정이고 CVPR 추가 연구비용은 별도다.
- CVPR README와 루트 CURRENT_STATUS에서 19번 후속 기록을 찾도록 연결했다. 이번 추가 기록은
  네 Markdown 파일에 한정되며 새 학습·공격 실행이나 동결 실험계약·코드·데이터·본문 변경은 없다.

## 69-RECORD. 2026-09-09 — 근접 선행연구 전수 읽기 목록·PDF와 과거 판정 근거

- 사용자는 선행연구 전체를 다운로드해 읽을 수 있는 목록과, 선행 대조 후에도 가능성이 있다고
  판단한 이전 기록의 존재를 요청했다.
- CVPR 01~18번의 명시적 URL 인용을 전수 추출했다. 고유 URL 91개에서 논문 외 정책·코드·데이터셋
  링크 8개를 분리하고 같은 논문의 여러 공개 링크를 통합하여 71편으로 정리했다. 근접·관련 67편과
  17번 탈옥 트렌드 참고 4편이며, 최신 18번에 직접 인용된 문헌은 35편이다.
- 20번 문서와 `reading_list_2026-09-09/index.html`, Excel용 UTF-8 BOM CSV를 작성했다. 정식 제목,
  PDF, 연구 관계, 기존 인용 문서·행 번호, 우선 독해 15편을 제공한다. 원래 URL inventory,
  메타데이터 접속 기록, 중복 통합 명세, PDF 응답 기록과 목록 생성 스크립트를 함께 보존했다.
- 공개 proceedings·저자 공개본에서 제목과 링크를 확인하고 PDF 링크 71개를 점검했다. 67개에서
  HTTP 200과 PDF 시작 바이트를 확인했다. Nature의 Dar 논문 PDF, PMC medical LDM filtering PDF,
  OpenReview patient-level unlearning·PF-LDM PDF는 브라우저 확인 필요로 표시했다. 일부 OpenReview
  제목·발표 형식은 공개 PDF 검색 인덱스로 확인했으며 직접 HTTP 접근 성공으로 기록하지 않았다.
- 03·04·16·18번의 조건부 가능성 판정과 선행 충돌표를 다시 읽어 연결했다. 초기 교집합 후보에서
  단순 비교안 강등·새 환자 집합 감사법 검증 요구로 결론이 엄격해진 경로를 보존했다. 71편 모두의
  전문·부록 정독, 독립 증명 검증, 코드 재현이 완료됐다는 장부는 확인되지 않음을 명시했다.
- 이름 또는 연구군만 등장한 ViewDiff·MultiDiff·CDI·일반 user-level DP theory와 사용 encoder·metric
  원 논문은 명시적 URL 전수 목록과 구분했다. 기존에 검토 완료한 특정 논문으로 임의 확장하지 않았다.
- PDF 응답 검사는 PDF 전체 파일 무결성 검사와 다르고, 이번 작업은 문헌 심층 재검토나 원문 PDF
  일괄 저장이 아니다. 최신 과학적 판단은 18번을 유지한다. 실험·공격 실행, 동결 계약, 데이터,
  학습 코드와 활성 LaTeX/PDF는 변경하지 않았다.

## 70-SCOPE. 2026-09-09 — 현재 CVPR 환자 DP 의료영상 후보로 문헌 범위 정정

- 사용자는 지금 이야기하는 대상이 CVPR의 환자 DP 의료영상 연구 하나라고 명확히 했다.
  앞서 제시한 71편에는 졸업논문 receipt·PP-Mark, 접은 방법 후보, 탈옥 트렌드까지 포함되어
  현재 요청의 읽기 목록으로는 범위가 지나치게 넓었음을 정정했다.
- 최신 18번에서 직접 URL로 인용한 35편만 별도 Markdown·HTML·CSV로 생성했다.
  `CVPR 주제 탐색/reading_list_2026-09-09/cvpr_patient_dp.*`를 현재 범위의 진입점으로 사용한다.
  PDF 점검 결과는 69-RECORD 작업에서 확인한 기록을 재사용했으며 새 원문 조사·학습은 수행하지 않았다.
- 20번, 폴더 README와 CURRENT_STATUS에 현재 범위를 연결했다. 기존 71편 목록은 탐색 이력으로
  보존하고, 18번의 조건부 기여도 판정과 전문·재현 완료 증거의 한계를 유지했다. 이번 정정은
  문헌 논의의 범위이며 졸업논문 병행 계획을 취소한 것으로 해석하지 않는다.

## 71-CLARIFY. 2026-09-09 — 관련 문헌 수와 직접 근접성의 혼동 정정

- 사용자는 35편이 정말 모두 근접 연구인지 재확인했다. 앞선 답변의 35는 최신 문서의 관련 인용
  수였으며, 모두 직접 경쟁하는 근접 연구처럼 답한 것은 부정확했음을 명시했다.
- 현재 18번의 중심 주장(환자별 위험, 환자 집합 감사, diffusion 보정과 patient-DP 비교)에 따라
  핵심 대조 6편, 직접 방법·평가 비교군 13편, 기반·확장 관련 16편으로 역할을 나눴다. 핵심 대조는
  Knolle/Nature, Kaiser medical ULDP, CheXGenBench, FACE-AUDITOR, user-level embedding MIA,
  quantile-regression diffusion MIA다. 각 논문의 원문·초록과 기존 대조표를 재확인했다.
- `cvpr_proximity.json`과 `cvpr_proximity_review.md`에 분류 기준·겹치는 주장·남는 차이를 기록하고,
  현재 읽기 목록 HTML·Markdown·CSV에 관계 표시를 반영했다. 핵심 6편은 현재 후보에 대한 검토
  우선선별이며, 동일 연구 전체를 수행한 논문이 정확히 6편이라는 주장이나 다른 근접 선행의 부재
  증명으로 쓰지 않는다. 읽기 순서 15편과도 다른 분류다.
- 논문 수의 재분류로 신규성·성공 가능성을 상향하지 않았다. 방법·실험의 조건부 판정과 전문·재현
  완료 장부의 한계를 유지한다. 전체 원문 재검토나 실험 실행은 수행하지 않았다.

## 72-UI. 2026-09-09 — 문헌 역할별 드롭다운 필터 구현

- 사용자가 HTML에서 전체 35편과 먼저 읽을 15편만 선택할 수 있다고 지적했다. 앞선 작업은 역할
  표시·정렬만 반영했고, 역할별 드롭다운 선택은 빠져 있었다.
- 현재 CVPR 페이지의 읽기 범위를 전체 35편, 핵심 대조 6편, 방법·평가 비교군 13편,
  기반·확장 문헌 16편, 먼저 읽을 15편으로 구성했다. 각 선택이 실제 카드의 역할·우선순위에 따라
  필터링되도록 JavaScript와 생성 스크립트를 수정하고 HTML을 다시 생성했다.
- Chrome headless에서 실제 HTML을 열어 다섯 선택의 표시 수 35/6/13/16/15와 표시된 논문 ID를
  대조했다. 분야 결합 필터, 대소문자 혼합 검색, 결과 0건, 필터 해제도 PASS했다. 공유 스크립트를
  쓰는 역사 목록의 기존 71/35/15 필터도 PASS했고 JavaScript 예외는 0건이었다.
- 이번 변경은 문헌 목록의 UI 동작 수정이며 연구 기여도 판정·실험 조건은 변경하지 않았다.

## 73-RATIONALE. 2026-09-09 — 많은 선행에도 현재 CVPR 후보를 유지한 근거와 한계

- 사용자는 선행을 검토하고도 현재 방향에 가능성이 있다고 판단한 이유를 요청했다. 18번의
  단순 DP 비교안 강등과 환자 집합 감사법 후보 유지, 16번 로컬 전제 검정의 해석 범위를 재확인했다.
- Nature 원문 Discussion에서 생성모델의 record/patient MIA 평가와 scalable approximation을
  후속 과제로 제시한 부분을 확인했다. 동시에 기존 공격을 생성모델에 거의 수정 없이 적용할 수
  있다는 설명도 확인하여, 후속 연구 언급이 새 방법의 독립 기여를 보장하지 않는다고 명시했다.
- 21번 문서에 문제의 중요성, 가변 기록 수·상관과 환자 단위 보정 가설, 반증 가능한 비교,
  로컬 도구적 전제를 유지 이유로 정리했다. 방법의 필요성·신규성·CVPR 경쟁력은 미검증이다.
- 독립 비회원 이미지의 오탐률을 1%로 가정한 설명용 계산에서 2장/20장의 any-positive 환자 오탐률은
  1.99%/18.21%다. 실제 실험 결과나 Nature의 patient-specific MIA AUC에 대한 반증으로 해석하지
  않으며, 상관된 기록에서는 독립 가정이 성립하지 않음을 명시했다.
- quantile MIA와 user-level 감사는 이미 존재한다. 기존 공격·집계에 유효한 입력 p-value를 전제로
  표준 다중검정 보정을 붙인 방법과의 비교도 후속 검토해야 한다고 구체화했다. 해당 baseline은
  이번 설명의 제안이며 동결 프로토콜 변경이나 실행으로 기록하지 않았다.
- 886명 실제 영상의 identity 검정은 generator leakage나 새 감사법의 성공 증거가 아니다.
  연구 질문 유지와 투고 경쟁력 확인을 구분했고, 기존 방법으로 충분하면 CVPR 방법 주장을
  축소한다는 기준을 유지했다. 21번을 README·CURRENT_STATUS·현재 HTML 목록에서 연결했다.

## 74-CVPR-REVIEW. 2026-09-10 — 79편 직접 대조·비교 준비·공개 결과 재계산

- 사용자는 기존 6편을 넘어 직접 대비할 SOTA와 준비한 수십 편을 직접 분석하도록 요청했다.
  기존 관련 67편과 추가 12편, 총 79편의 개별 메모를 `CVPR 주제 탐색/research_2026-09-10/reviews/`에
  작성했다. 기존 탈옥 참고 4편은 현재 환자 DP 의료영상 범위에서 제외했다.
- 실제 범위는 로컬 PDF 75편의 방법·정의·가정·실험 또는 평가 설계 선택 절, A20/A21/B17의 공식 본문
  일부, C02의 저자 초록이다. 읽은 페이지·원문 링크·미확인 부분을 각각 남겼다. 전 페이지·전체 증명
  정독이나 79편 모두의 재현 완료로 표현하지 않는다. PDF·검토·코드·재계산 장부를 구분했다.
- `REVIEW_REPORT.md`에 기여 충돌, 공격/학습 비교와 공정성 조건, 피벗·주장 축소 기준을 정리했다.
  CDI·SD-MIA의 집합 추론, MoFit·PFAMI의 공격 조건, ULS/ELS의 보호 단위와 연산량을 대조했다.
  현재 방향은 반증할 후보이며 신규성·CVPR 경쟁력은 미입증이다. 기존 6편·35편의 분류를 최신 SOTA
  대조 완료의 근거로 쓰지 않는다. Nature 개별 record/patient 위험과 고정 모델 patient-bag ROC도 구분했다.
- 공격 13계열, 학습 13계열, privacy unit/회계 7편을 `comparison_plan.json`과 33행 CSV로 연결했다.
  주장·접근 조건별 비교군이며 33개 실험 전체를 의무화한 것은 아니다. 공식 GitHub 소스 9계열의
  commit·파일 해시와 NeurIPS 공식 supplement 2계열을 확보했다. 소스 확보는 실행 완료와 다르다.
- CLiD 공개 점수 packet과 Tracing 공개 feature packet을 CPU에서 재계산했다. CLiD는 공식 loader의
  첫 숫자 행 제외와 전체 행 버전을 보존했고, nonmember-positive 67.52%와 member-positive 저오탐률
  TPR의 차이를 확인했다. 표본 수에 따른 1% FPR 격자 차이도 구분했다. Tracing은 공급 feature로
  분류기를 재학습해 full-feature AUC 83.3055%, TPR@FPR≤1% 16.8%를 얻었다. 의료 환자 실험이나
  diffusion feature 추출부터의 전체 재현은 아니다. 설정·해시·결과는 `replays/`에 저장했다.
- `evaluate_patient_scores.py`에 환자 집계, 독립 calibration, member-positive ROC, 실제 test FPR/TPR와
  이항 신뢰구간을 구현했다. 환자 분할 누수·중복·동점·방향·독립 보정·유한 표본 해석의 6개 테스트가
  PASS했다. 표준 평가 도구이며 새 알고리즘 기여 또는 임의 shift에서의 보장으로 주장하지 않는다.
- 새 HTML의 전체 79/기존 67/기존 현재 목록 35/추가 12/공격·학습 26 드롭다운, 역할·검토 상태,
  검색·초기화를 Chrome에서 확인했다. 기존 71편·35편 HTML의 필터를 보존하고 최신 검토 배너를
  연결했다. 로컬 링크 495개·PDF 페이지 앵커 192개에 끊김 0, JavaScript 예외 0이다.
- CVPR README, 23번 통합 기록과 루트 CURRENT_STATUS에 결과를 연결했다. 18–22번은 과거 판단으로
  보존하며 최신 문헌 해석은 새 보고서·개별 메모를 따른다. K5/K10 본학습·동결 시험집합 평가,
  학위논문 본문·동결 실험 계약 변경은 이번 작업에서 수행하지 않았다.

## 75-RATIONALE. 2026-09-10 — 환자 집합 감사 후보의 상세 근거와 반론

- 사용자는 이 방향이 상대적으로 가능성 있어 보인다는 판단의 이유를 철저히 설명하도록 요청했다.
  기존 직접 메모와 핵심 원문을 대조해 `PATIENT_AUDIT_RATIONALE.md`와 `rationale.html`에 근거를 기록했다.
- 문제의 중요성, 요약에서 소실될 수 있는 관계 정보, 검증 경로의 구체성과 실험 효율을 우선순위
  근거로 구분했다. 의료 LoRA에서 추가 membership 신호가 존재하는지와 공정한 비교에서 이기는
  새 방법은 아직 미확인이다. '가장 먼저 검증할 후보'를 투고 성공확률의 순위로 표현하지 않는다.
- 추가 원문 SD-MI(AAAI 2023) pp.1–6과 개인화 의료 Patient Membership Inference(IJCNN 2025)
  PDF p.3·pp.5–9를 읽고 `rationale_sources/R01.md`, `R02.md`에 개별 비판을 남겼다. 유사도 관계·anchor
  가중치와 비학습 환자 자료·robustness 집계도 이미 선행이다. 두 편은 기존 79편 장부와 별도 보완으로
  표시했고, 원문 PDF·텍스트·SHA-256·쪽 수를 저장했다. 전체 증명 검산이나 코드 재현은 수행하지 않았다.
- 학습 기여 수와 감사 관측 수, 원래 기록과 같은 환자의 다른 기록, Re-ID와 membership를 구분했다.
  상관의 평균 분산 모형과 독립 any-positive 오탐의 설명용 계산을 JSON/CSV로 남겼다. 실제 환자
  성능이 아니다. 표준 보정을 양쪽에 적용한 뒤 남는 탐지력·비용 개선과 명시적 반증 조건을 제시했다.
- patient-DP의 인접 단위와 후처리, 보장과 경험적 공격의 관계를 대조했다. 환자 내 상관으로 올바른
  patient-DP를 우회한다는 목표로 설명하지 않는다. 새 알고리즘 외 평가·발견 기여의 가능성과 기존
  의료 감사·벤치마크 선행 때문에 필요한 추가 증거도 명시했다.
- 24번 기록, 연구·CVPR README, 종합 보고서, HTML 진입점과 CURRENT_STATUS를 연결했다. 브라우저
  검증에서 오래된 DevTools port 파일 재사용 문제가 발생해 해당 생성 파일을 초기화하도록 수정했다.
  재실행은 PASS: 원래 79편/35편/71편 필터 유지, 상세 문서 65문단·6표·각주 13개·보완 링크 2개 확인,
  JavaScript 예외 0, 전체 검사 로컬 링크 509개·PDF 페이지 앵커 199개에 끊김 0이다. 신규 설명 문서의
  로컬·각주 링크 48개도 별도로 대조했다. 이번 작업은 본학습·최종 시험집합 평가를 실행하지 않았다.

## 76-PURPOSE. 2026-09-10 — 원래 보호·환산 목표와 공격 중심 제안의 구분

- 사용자는 DP로 보호하며 학습하고 image-level 보장이 환자 단위로 어디까지 환산되는지 확인해
  재사용·재학습 비용을 비교하던 목표에서, 왜 더 잘 판별하는 공격 연구로 설명이 바뀌었는지 물었다.
- 01번의 image-DP evidence→patient-level claim→reuse/retrain 목표, 10번의 공격을 평가 도구로 둔
  기록, 18번의 새 공격법을 주기여로 올린 전환 제안을 직접 대조했다. 후자는 원래 목표에서 필수로
  따라오는 단계가 아니며, 앞선 설명에서 이 목적 변화를 충분히 구분하지 않은 점을 정정했다.
- `RESEARCH_PURPOSE.md`와 `purpose.html`에 수학적 보장·실행 조건·회계 환산과 경험적 공격 평가를
  구분했다. 공격 실패는 DP 인증이나 더 작은 ε로의 환산 근거가 아니며 공격 성공도 곧바로 올바른
  DP의 위반을 뜻하지 않는다. 기존 강한 공격을 평가 도구로 사용하며 필요가 입증될 때 새 방법을
  별도 후보로 검토하는 관계를 설명했다.
- 문헌 HTML, 종합 보고서, 상세 후보 근거, README와 CURRENT_STATUS에 목적 확인을 연결했다.
  사용자 질문을 공격 후보의 최종 채택·폐기로 확대 해석하지 않았다. 문헌 검토 결과와 미입증 상태는
  보존하며 본학습·동결 실험 계약·최종 시험집합 평가의 변경은 없다.

## 77-CONTRIBUTION. 2026-09-10 — 공격 연구의 가치와 우리 후보의 기여 미확보

- 사용자는 문헌 검토 후 공격 중심 후보를 제안할 수는 있으나 그 공격법 자체의 기여가 무엇인지
  물었다. `RESEARCH_PURPOSE.md`에 공격 연구의 측정·발견 가치와 우리 방법의 실제 신규성을 구분했다.
- 기존 감사가 놓친 노출의 발견, 그 신호를 추출하는 구체적 방법, 동일 접근·정보·오탐률·비용의 개선이
  후보에서 입증할 내용이다. 다중 사진·상관·보정 자체는 기여가 아니다. MoFit의 정답 caption 없는
  조건을 위한 구체적 conditioning 제안을 이미 구현된 선행 사례로 확인했다.
- 현재 새로운 membership 반응·공격 알고리즘·의료 환자 성능 개선을 확보하지 못했음을 명시했다.
  문헌 검토 결과는 조건부 질문과 기여 경계이며 완성된 새 공격법이 아니다. 공격 연구의 일반적 가치만으로
  원래 보호·환산 목표를 대체하는 근거가 충족됐다고 표현하지 않는다. 학습·공격 실행 상태의 변경은 없다.

## 78-CONNECTION. 2026-09-10 — 새 감사법과 보호 비교·개선의 연결

- 사용자는 새 공격법의 기여가 보호 방법의 엄밀한 비교와 보호 설계·평가 개선으로 이어진다는
  설명을 이해했다고 밝혔다. `RESEARCH_PURPOSE.md`에 이 연결을 명시했다.
- 결합 성과 후보를 새 검출 방법, 조건을 맞춘 보호의 노출·효용·비용 비교, 그 비교에서 얻는
  설계·선택 근거로 정리했다. 감사 성능 개선과 보호 연구에 대한 실질적 기여는 각각 입증해야 한다.
  기존 보호 순위가 반드시 바뀌어야 하는 것은 아니며, 공격 결과는 formal DP 회계를 대체하지 않는다.
- 이 발언을 연구 방향의 최종 확정이나 신규성·실험 성공의 승인으로 확대 해석하지 않았다.
  원래 보호 목표와 공격 중심 후보를 연결하는 논거로 보존하며 실행 상태는 변경하지 않았다.

## 79-DESIGN. 2026-09-10 — 앞선 근거에 기반한 실제 연구 설계 v0.1

- 사용자는 앞서 설명한 첫째부터 여덟째 근거를 기준으로 실제 설계를 시작하자고 요청했다.
  `EXPERIMENT_DESIGN.md`, `design.html`, `experiment_design.json`, 부모 폴더 25번 기록을 작성했다.
  새 감사법과 DP 보호·효용·재사용/재학습 비용 비교를 연결하는 구체적 후보 설계다.
- 첫 후보는 support 사진에서 공개 base 대비 target의 conditioning 반응을 탐색하고, 같은 환자의
  held-out 사진으로 전이시킨 뒤 비슷한 다른 환자의 반응과 비교하는 방법이다. 최적화 공간·두 방향
  분할·별도 noise bank·점수식·최초 probe 기본값·연산량을 명시했다. 아직 구현·의료 성능·신규성의
  검증은 없다. MoFit의 직접 교차 사진 확장과 선형 gradient 관계로 이득이 같아지는지도 반증한다.
- 기존 환자 메타데이터만 집계했다. public_development 1,816명 중 원자료 2장 이상 886명,
  5장 이상 308명이다. `inspect_design_feasibility.py`와 `design_feasibility.json`에 source hash,
  분할별 집계 및 m=2에서 312 forward/72 backward, m=5에서 780/180이라는 제안 설정의 산술을
  기록했다. 이미지 열람·환자 ID 출력·새 데이터 분할·학습·공격 실행은 수행하지 않았다.
- 원래 학습 사진 E와 같은 환자의 미사용 사진 U를 분리하고 k_train/m/r_overlap을 구별했다.
  환자 단위 fit/selection/calibration/test와 공통 FPR 보정, 표본력의 한계, 독립 target seed 확인을
  설계했다. 새 최종 분석의 정확한 cohort·adapter·power는 실행 전 별도 고정이 필요하다.
- K5 protocol과 실제 모델 상태를 다시 대조했다. M0/full matrix 미시작 및 K5 attack NOT_RUN을
  유지한다. 공개 개발용 별도 참여 진단을 먼저 제안하고 최종 K10 모델·의료 평가·회계는 공통으로
  재사용한다. I8/G8은 noise가 다른 모델이며 같은 checkpoint에 두 회계를 적용하는 것과 구분했다.
- MoFit Appendix A.10/Table 12, PDF p.23의 의료 실험을 추가로 확인해 X12 메모에 읽은 시점을
  명시했다. 의료영역 적용만으로 새 기여가 아님을 강화했다. SAMA 공식 초록·저자 코드의
  target/base·subset 집계 선행도 확인했으나 기존 79편 검토 장부에 소급 추가하지 않았다.
- 연구 HTML·README·purpose·CURRENT_STATUS에 설계를 연결했다. 생성 후 확인 PASS:
  로컬 링크 524개, PDF 페이지 앵커 201개, 끊김 0; 기존 79편/71편/35편 Chrome 필터 유지,
  JavaScript 예외 0. 검증은 문서·장부·메타데이터·비용식 범위이며 공격 알고리즘 검증이 아니다.

## 80-THESIS-ALIGNMENT. 2026-09-10 — CVPR 추진 시 졸업논문의 중심과 역할

- 사용자는 새 환자 감사 방향으로 CVPR을 목표로 할 때 졸업논문을 어떻게 구성하는 것이 좋은지 물었다.
  활성 국문 초록, 장 구성, 통합 계약과 19번 병행 계획을 대조해 권장 구조를 작성했다.
- 기존 Claim-Aligned Assurance를 유지하고 환자 보장 타당성, 유효한 보장 아래 노출·효용·재사용 비용,
  모델·근거·생성 이미지의 연결을 세 질문으로 제안했다. DP 감사·model-bound receipt·PP-Mark의
  의료 통합 실증을 졸논 중심에 두고, 새 CVPR 감사법은 검증되면 평가 강화 또는 별도 장으로 포함한다.
- 학위논문에도 독립적인 기술 기여·강한 비교·엄밀한 검증이 필요함을 명시했다. 단위 불일치, 다른
  모델의 증거, 점수 전용 위조, 서로 다른 대상의 유효 증거 조합에 대한 통합 대조로 추가 가치를
  검증한다. CVPR 채택·새 공격의 SOTA 달성을 졸논 전체의 완료 조건으로 두지 않는 권장안이다.
- `THESIS_CVPR_ALIGNMENT.md`, `thesis_alignment.html`, 26번 기록을 추가하고 HTML 진입점,
  두 README, 19번 후속 링크, CURRENT_STATUS에 연결했다. 활성 LaTeX·본학습·동결 계약은 변경하지 않았다.
- 생성 후 정적 검증 PASS: 로컬 링크 530개, PDF 페이지 앵커 201개, 끊김 0. 기존 장부 79편과
  검토 수준을 유지했다. 결과는 `alignment_verification.json`이며 새 학습·공격 성능 검증은 아니다.

## 81-GPU. 2026-09-11 — CVPR 실험용 GPU 필요성

- 사용자는 현재 CVPR 방향에 더 좋은 GPU가 필요한지 물었다. nvidia-smi로 RTX 3070/8,192 MiB를
  직접 확인하고 기존 4-step DP/LoRA 실행, K5/K10 시간 추정 및 새 감사법 비용식을 대조했다.
- 현재 장비에서 초기 구현·작은 진단을 시작하되 반복 학습·강한 공격 비교에는 24–32GB급 GPU
  사용 시간 확보를 권장했다. 새 감사법의 8GB 적합성·최소 VRAM·새 GPU 가속비는 아직 실측하지 않았다.
  공식 NVIDIA 사양의 4090 24GB·5090 32GB는 용량 예시이며 구매 결정이나 호환성 검증이 아니다.
- 설계 문서와 HTML에 자원 판단을 기록했다. 구매·임대·학습·데이터 업로드는 수행하지 않았다.

## 82-SINGLE-GPU-SCHEDULE. 2026-09-11 — 임대 없는 3070 한 대의 CVPR 일정

- 사용자는 임대가 불가능하고 현재 GPU만 쓰는 조건의 기간을 물었다. 이후 계획의 자원 전제를
  RTX 3070 8GB 한 대·외부 임대 없음으로 반영했다. GPU 12–16시간/일과 집중 구현·분석을 가정했다.
- 기존 4-step 실행과 K5/K10 계획을 다시 대조했다. 공통 학습·생성·점검의 기존 합산 추정은
  66–101 GPU시간이며 새 공격 개발·전체 평가 시간이 아니다. K10 자체는 아직 실측하지 않았다.
- NIH/SD2.1 LoRA 중심의 실험·원고 준비는 10–16주, 계획상 약 3–4개월을 제안했다. 빠른 개선이면
  2–3개월, 큰 재설계·추가 데이터·baseline 이식 난항이면 4–6개월 이상으로 늘 수 있다. 일정 판단을
  측정값이나 CVPR 기여·채택·완수 보장으로 표현하지 않았다.
- 공격 하나의 2,000명×12모델 예시에서 가정한 30/60/120/300초당 200/400/800/2,000 GPU시간을
  계산했다. 이는 확정 cohort 또는 처리량이 아니며 메모리·초/환자 측정 뒤 갱신할 감도 분석이다.
- MoFit A.8 원문의 약 14.8–20.7GB 단계별 메모리를 확인했다. 8GB에서는 단순 대기만으로 같은
  원형 구현이 실행되는 것이 아니며, 별도로 평가된 early stopping 시간 합을 의료 모델 실측처럼
  쓰지 않는다. 설계 문서·HTML에 전제·단계·예외를 기록했다. 학습·임대·구매는 수행하지 않았다.

## 83-API. 2026-09-11 — 이미지 생성 API와 GPU 코드 실행 API의 구분

- 사용자는 현재 연구를 API로 수행할 수 있는지 물었다. 자체 코드 실행형 serverless GPU API는
  현재 방법의 원격 실행 경로가 될 수 있고, 완성 이미지만 반환하는 API는 denoising loss와 gradient를
  쓰는 현재 방법의 직접 대체가 아님을 구분했다. Modal 공식 개요·함수 문서를 확인했다.
- 출력 API만 쓰는 공격 연구는 ReDiffuse 등 선행이 있으나 방법·비교군·평가 정답과 보호 비교를
  재설계해야 한다. 기존 임대 불가 조건을 취소하거나 API 사용을 승인한 것으로 해석하지 않았다.
- API도 외부 연산비를 사용하며 서버 관리 부담과 비용 제약을 구분했다. 현재 환경의 DPAPI resume,
  DP 실행·회계·모델 결속을 원격 환경에서 다시 검증해야 함을 설계 문서·HTML에 기록했다.
  가격 견적·계정 생성·유료 호출·배포·데이터 전송·학습은 수행하지 않았다.

## 84-RUNTIME-REVIEW. 2026-09-11 — 현재 설계 전 항목의 시간과 Spark 4대 배치 검토

- 사용자는 Spark가 총 네 대 있다는 전언을 추가하고, 현재 설계의 각 실험에 얼마나 걸릴지 자세히
  검토하도록 요청했다. 단순 GPU 질의 기록을 재개한 것이 아니라 이 요청에 따른 연구 계획 산출물이다.
- `RUNTIME_REVIEW.md`, `runtime_review.html`, `runtime_budget.json`, 계산 재현·검증 스크립트를 추가했다.
  기존 설계·문헌 HTML·README·CURRENT_STATUS에 연결했다. 원문 설계 JSON과 동결 학습·공격 계약은 유지했다.
- K5 4-step 실측과 K10 4,000-step/12회 핵심 학습을 다시 대조했다. Poisson 네 단계 관측의 불확실성을
  반영해 K5 전체 예약을 15–26시간, K10 핵심 학습을 36–60시간으로 넓혔다. 새 장시간 측정값이 아니다.
  기존 Windows/DPAPI K5 경로는 3070에 남기는 안이고 Spark 이식·환경 검증 시간을 별도로 계상했다.
- 고정 membership cohort를 메타데이터로 재집계했다. E/m=2는 865+865=1,730명, E/m=5는 304+304=608명이며
  이는 파일·중복·시나리오 검증 전 상한이다. K10 member 중 원자료 >=12/15장은 각각 58/38명으로,
  기존 K10 cap 밖 사진만으로 U 확증 표본이 충분하다는 전제를 두지 않았다. 식별자는 새 산출물에 넣지 않았다.
- 후보의 312F+72B(m=2), 780F+180B(m=5), 공식 CDI 설정·MoFit runtime과 baseline 준비 상태를 대조했다.
  방법별 Spark 초/환자는 미실측이므로 세 가지 예약 시나리오로 명시했다. 400시간은 과거 단일 공격의
  예시였으며 전체 비교군·개발·제거·효용의 시간과 구분한다. 단순 집계·같은 feature의 fitting을
  diffusion 재실행으로 곱하지 않지만 서로 다른 prompt/noise의 재사용은 가정하지 않았다.
- 네 대 ×24시간 ×80% 가용률에서 중앙 시나리오는 m=2 중심 약1,339 장치시간/17.4 용량환산일,
  m=5 본표까지 약1,965 장치시간/25.6 용량환산일이다. 이는 연구 일정/완료일의 실측 예측이 아니다.
  구현·분석·의존성까지 포함해 6–10주 예약안을 설명했고, 느린 전체 확장·재설계 시 10–14주 이상도 명시했다.
- 원형 MoFit 전수는 위 기본 예산에 없다. 작은 원형 비교800 image-model 평가와 전 표본의 제한 예산
  비교를 별도 계상했다. 원형이 핵심 주장에 영향을 주면 평가 범위나 주장을 다시 정해야 한다.
- U/m=2를 새 hold-out-record manifest의 12개 모델/1,730명으로 평가하는 조건부 안은 추가 약405/800/1,680
  장치시간과 분할·회계·검증 시간을 산정했다. 실제 적격 수·power는 미확인이고 E/U 모두 본표이면
  중앙 8–14주 이상을 검토한다. P, 두 번째 dataset/backbone, P512, 신규 보호 학습법, 졸논 전용 작업은 구별했다.
- 계산·source hash16개·로컬 링크547개/PDF앵커201개 검증 PASS. 실제 headless Chrome에서 기본/변경
  장비·시간·범위 계산을 확인했고 JavaScript 예외0이다. 결과는 `runtime_verification.json`에 보존했다.
- Spark 원격 접속, 데이터 전송, 새 cohort 배정, optimizer/공격 실행, 구매·임대는 하지 않았다.

## 85-AUDITOR-FRAMING. 2026-09-11 — 제한 접근 환자 감사자의 근거와 설계 적합성

- 사용자가 제시한 환자·외부 연구자·데이터 제공기관·규제/인증기관 설명을 HHS·ICO·FDA·IIA 및 사건
  원자료와 대조했다. 결과를 research_2026-09-10/AUDITOR_FRAMING_REVIEW.md와 HTML에 저장하고
  기존 문헌 HTML·README에 연결했다. 새 역할 정의는 검토·권장안이며 학습·공격 계약은 바꾸지 않았다.
- ICO가 의료 membership 위험 및 학습 데이터 없는 제3자의 white-box 모델 접근을 직접 설명함을 확인했다.
  외부자=black-box 또는 내부 감사자=MIA 불필요라는 구분을 교정했다. 데이터 사용 사실의 확인,
  배포 모델의 노출 측정, 법적 적합성 판단은 별개다. 내부 감사의 독립성을 병원 관리자의 이해와 합치지 않는다.
- HIPAA minimum necessary의 환자 본인 공개·authorization·HHS 집행 등 예외를 확인했다. FDA 분석자료 제출을
  원자료 요구권 부재로 해석하지 않았다. 규제기관과 계약상 제3자 인증기관도 서로 구분했다.
- Streams는 당시 AI를 사용하지 않았고 GoodRx는 광고 목적 건강정보 공개 사건으로, 의료 생성 MIA 성공 사례가
  아님을 표시했다. Prismall의 2024-12-11 항소 기각을 확인했다. ICO undertaking은 공식 검색의 제5항,
  EXAM은 공식 초록·요약 확인 범위이며 이번에 전문을 읽었다고 표시하지 않았다.
- 현재 probe는 target/base weights·conditioning gradient, 다른 환자 대조 bank, 개발·nonmember 보정 자료를
  요구한다. 자기 사진만 가진 API-only 환자의 조건과 일치하지 않는다. 모델/자료 접근과 감사 목적을 따로 명세한다.
- E의 성공과 U의 성공을 구분하며, 미사용 영상이 미참여 환자를 뜻하지 않는다는 보정자료 문제를 기록했다.
  U를 주된 주장으로 삼으면 학습 전 독립 촬영 hold-out과 충분한 표본이 필요하다. 기존 cap 밖 후보
  58명/38명은 파일·중복 검증 전 상한이라는 이전 검토를 대조했다.
- 환자 측 white-box 검증자라는 소개문을 제안했다. MIA 점수를 무단 사용 판결·DP 인증으로 확대하지 않고,
  역할 정의 자체를 CVPR 신규성으로 세우지 않는다. 추가 모델 구현·학습·공격은 실행하지 않았다.

## 86-BASELINE-SCOPE. 2026-09-14 — U 중심 연구 질문과 비교군 역할 검토

- 사용자는 관측 영상은 미사용이어도 같은 환자의 다른 영상은 학습되었을 수 있다는 간극이 연구의 핵심임을
  재확인하고, 관련연구를 직접 비교·변형 비교·신규성 경계·접근 조건 대조로 나누는 설명에 대한 판단을 요청했다.
- 큰 원칙에 동의하되 U를 마지막 추가 실험으로 미루지 않는다고 명시했다. 결과 없는 신호 존재·우월성은
  검증 가설이다. 학습 전 영상 분할, 환자 음성·보정 자료를 첫 설계에 포함해야 한다.
- A06/A08/A09/A10/X01/X04/X12 기록과 공식 원문을 대조했다. FACE-AUDITOR의 미사용 동일인 영상으로
  user membership을 추론하는 선행은 기존 A06에 있던 내용이며 이번에 다시 확인했다.
- MoFit은 단순 저질의 공격이 아니라 caption-free conditioning 최적화의 직접 비교이고, Quantile 공격의
  영상별 학습형 threshold는 공통 환자 calibration과 다르다. CDI의 핵심 특징을 약화한 변형을 원형 재현으로
  부르지 않는다. SD-MIA는 출력 기반, DIME은 현재 확인된 2026-08 preprint 상태다.
- 필요한 비교 축과 모든 방법의 전수 실행을 구분했다. 기존 영상·집합·최적화·gradient·관계 대안을
  공정하게 적용하며, 실제 시험 patient-FPR/TPR·불확실성·보조정보·전체 연산 비용을 함께 평가한다.
- BASELINE_SCOPE_REVIEW.md와 HTML을 추가하고 기존 문헌 HTML·README에 연결했다. 79편 장부와
  원문 검토 수준, 동결 K5/K10 계약·학습·공격 실행 상태는 그대로다.

## 87-U-EXECUTION-PLAN. 2026-09-14 — U 핵심 질문을 수행할 단계별 계획

- 사용자가 관측 사진은 미사용이어도 같은 환자의 다른 사진은 학습되었을 수 있다는 간극을 핵심으로
  확인하고 수행 계획을 요청했다. U_EXECUTION_PLAN.md와 HTML을 작성해 기존 설계·목록·상태 인덱스에 연결했다.
- 첫 안은 public_development에서 k_train=2, m=2, U 겹침 0이다. 기존 메타데이터의 4장 이상 425명은
  파일·중복·촬영 구분 검증 전 상한이다. 실제 분할·개발/선택/보정/시험 인원과 reference bank를 먼저 정한다.
- 환자 포함 배정을 교환한 non-DP 두 모델을 진단쌍으로 제안했다. 같은 U 사진의 참여 정답이 바뀌는 비교이며,
  두 모델을 독립 seed 반복 또는 단일 환자 DP 인접 실험으로 세지 않는다. 작은 학습의 의료 적합성도 확인한다.
- conditioning 입력의 교차 사진 전이 후보, 초기 loss/집계/encoder/선형 관계 대조, MoFit 전이·CDI 등
  가까운 강한 비교, 독립 확인과 DP 보호 비교 순서를 구체화했다. 전수 baseline 실행을 새로 의무화하지 않는다.
- 실제 patient FPR·TPR·구간, 환자/모델 의존성, 보조정보·전체 비용을 반영한다. 파일럿 실패·추가 이득 부재·
  교란 설명에 따른 수정 기준과 모델 공유 조건을 기록했다. 시간은 첫 실행을 측정한 후 갱신한다.
- 이번 산출물은 계획이다. 실제 U cohort 배정, 후보 구현, 모델 학습·공격 실행 또는 졸논 동결 계약 변경은 없다.
- 기존 HTML 생성기를 실행하고 정적 검증을 완료했다. 79편 장부, 로컬 링크 568개와 PDF 쪽 앵커 201개를
  대조했으며 끊어진 링크는 0개다. 새 UI 동작 변경이나 새 성능 실험을 검증했다고 표시하지 않는다.

## 88-U-PILOT-EXECUTION. 2026-09-14 — 기존 NIH PA로 첫 실행 파일럿 완료

- 사용자의 "적당한거 골라서 하나씩 철저히 실행" 지시에 따라 `code_working/u_patient_audit`를 구현하고
  `_reports/cvpr_u_pilot_v1_001`에 새 공개 개발용 실행을 분리했다. 기존 졸논 K5/K10 동결 계약과 원자료는 변경하지 않았다.
- 실제 4,831개 파일의 checksum·decode·pixel/follow-up·중복을 검사했다. 미확보 375장은 제외했다.
  적격 425명 중 평가 400명을 fit 80/selection 40/calibration 140/test 140으로 분할하고, 환자당 학습 후보 2장과
  U 관측 2장을 고정했다. 별도 reference 64명·품질 32명·공통 배경 256명까지 전체 752명/2,304장이다.
  모델별 학습은 912장이고 U 학습 겹침은 0이다. 별도 검증 코드가 역할·hash·중복·교집합을 재확인해 PASS했다.
- 촬영 날짜와 study UID가 없어 follow-up 번호 분리의 한계를 남겼다. 4장 이상 적격 PA를 가진 환자의 파일럿이다.
  모델별 비참여 보정/시험은 각 70명으로 1% 확증 규모가 아니며, 5% 파일럿과 불확실성 평가가 후속 과제다.
- exact revision/hash를 확인한 로컬 SD2.1, 기존 256 전처리, VAE posterior mode, DINOv2 특징을 준비했다.
  다른 환자 reference 선택은 target 점수/참여 정답을 사용하지 않는다. 자료 업로드·임대·환경 업그레이드는 없었다.
- 최초 복원추출 50-step 모델은 실제 미노출 영상이 있어 실행 진단으로만 보존했다. 별도 v2에서 전체 순회 shuffle로
  수정했다. 배치 gradient 비교의 최초 unscaled FP16 오차 32.87%를 숨기지 않고 기록했으며, 실제 1024 scale 조건에서
  재확인한 상대 gradient 오차 0.61%·loss 오차 0.00159%를 근거로 batch 4를 사용했다. 핵심 단위 검사 4개 PASS.
- v2 두 non-DP 모델을 각각 1,000 step 완료했다. 학습 시간은 496.73/500.58초, 최고 학습 CUDA 메모리 약 1.93GiB다.
  각 912장 모두 4–5회 실제 사용되었고, optimizer·노출·trace·checkpoint hash를 별도 코드로 검증해 PASS했다.
  두 모델은 같은 seed의 포함 배정 교환 쌍이며 독립 seed 반복이나 단일 환자 DP 인접 쌍으로 세지 않는다.
- 고정 prompt/seed 생성 12개와 반복 생성 8개의 파일·checkpoint 결속을 검증했다. 두 모델은 흉부 X-ray 기본 구도를
  보였으나 해부학·질환 대응·임상 효용은 미검증이다. 별도 품질 환자 32명의 target/base denoising loss 비율은
  0.71515/0.71536이며 임상 효용 또는 MIA 성공 지표로 해석하지 않는다.
- 공개 encoder-only 대조는 fit 80명 학습, selection 40명 AUC 0.5475였다. 작은 개발 진단으로 모든 편향 부재를
  증명하지 않는다. calibration/test 환자의 target 공격 점수·정답을 이용한 선택이나 성능 판정은 하지 않았다.
- U 개발 환자 8명×2모델 16회와 E 실행 대조 1명×2모델 2회를 완료했다. zero-adapter score/gradient 0과
  모델 가중치 불변을 확인했다. 별도 verifier로 참여 정답 교환·관측 역할·fold 산술·반경·checkpoint 결속을 재확인해 PASS했다.
  U 중앙 시간은 48.58초/환자, 최대 57.01초다. E 중앙은 51.06초다. 312F+72B는 모델-이미지 평가 횟수이며 API 질의 수가 아니다.
- 입력 conditioning만 조정하며 base와 target의 미분을 각각 끝낸 뒤 adapter를 전환한다. 후속 재계산의 adapter 혼동을
  피하고 입력 gradient에도 scaling을 적용했다. 강한 baseline·방향/보정·독립 시험·DP 비교·새 기여의 성공은 아직 미검증이다.
- 실제 결과는 `U_EXECUTION_STATUS.md`와 `u_execution_status.html`, 구현 README 및 CURRENT_STATUS에 연결했다.
  최초 설계의 미실행 표기는 역사적 계획이며 현재 실행 상태는 새 실행 기록을 따른다.
- 최종 HTML 정적 검증은 79편 장부·로컬 링크 579개·PDF 쪽 앵커 201개를 확인했고 끊어진 링크 0개였다.
  이번에는 UI 동작을 수정하지 않았으며 브라우저 동작 검사를 새로 실행했다고 표시하지 않는다.

## 89-U-EIGHT-PATIENT-ANALYSIS. 2026-09-14 — 첫 판별력 분석만 수행

- 사용자의 "우선 하나만 진행"에 따라 기존 U 8명×2모델 결과만 CPU로 분석했다. 새 GPU 질의·학습·추가 환자·
  방법/부호/threshold fitting·calibration/test 평가는 0이다. 원결과와 cohort·관측 역할·checkpoint hash를 재확인했다.
- 후보 점수의 모델 1/2 AUC는 0.6875/0.3125, 단순 평균 0.5다. 참여 시 같은 환자의 후보 점수가 상승한 경우는 4/8명이다.
  환자 순위 Spearman은 1.0이며 28/28 환자쌍의 순위가 유지된다. 환자 한 명씩 제외해도 평균 AUC는 모두 0.5다.
  이 민감도 범위를 신뢰구간이나 실제 성능 0.5의 확증으로 해석하지 않는다.
- 음의 loss 평균은 AUC 0.75/0.25, 최댓값은 0.6875/0.3125다. base 차이 평균·gradient cosine의 평균 AUC 0.53125도
  개선 증거로 판정하지 않는다. 기존 점수는 저장된 값의 진단이며 강한 공격 원형 재현·공정한 비용 비교가 아니다.
- 현재 후보는 일관된 U 참여 신호/추가 이득을 확인하지 못했다. 환자·영상의 고정 차이가 참여 변화보다 순위를 크게
  좌우할 가능성은 단서이며, re-ID·병원 편향·학습 부족 중 무엇이 원인인지 확정하지 않는다. U 전체의 불가능성도 결론 내리지 않는다.
- 두 모델은 동일 초기화와 배경을 공유하고 여러 환자 배정을 함께 교환한다. 16관측을 독립 16명/독립 seed 반복으로
  세지 않으며 paired delta를 단일 환자 포함의 인과 효과로 보지 않는다. 모델별 미참여 4명으로 낮은 FPR 성능을 추정하지 않았다.
- sklearn/scipy와 원자료에서 재산출하는 별도 verifier로 AUC 12개·환자 제외 요약 48개·차이 48행·원점수 16행·입력 hash 5개를
  검증해 PASS했다. 분석 JSON/CSV, PNG/PDF와 코드·입력 hash를 `analysis/U_step1000_n8_v1`에 저장했다.
- `U_EIGHT_PATIENT_ANALYSIS.md`와 HTML을 추가하고 실행 기록·README·CURRENT_STATUS에 반영했다. 숫자 검증 PASS가
  연구 가설 성공을 의미하지 않는다고 명시했다. 후속 80명/40명 확대 단계는 시작하지 않았다.
- 최종 HTML 정적 검증은 로컬 링크 589개·PDF 쪽 앵커 201개를 확인했고 끊어진 링크 0개였다. 생성 그래프의
  환자 표기 겹침을 수정한 뒤 실제 PNG를 다시 확인했다. 숫자 결과는 유지했고 최종 그림 hash를 별도 저장했다.

## 90-U-COST-DECISION. 2026-09-14 — 표본 확대 보류, 기존 결과의 구조 점검

- 사용자가 80명 시간 부담과 16명의 판단 가능성에 이의를 제기했다. 16명은 충분한 검증 규모로 산정한 수가 아닌
  비용 절약안임을 정정했다. 80명/16명 추가 실행은 보류하고 새 GPU 작업 없이 기존 코드·결과만 점검했다.
- 독립적인 읽기 전용 probe 검토에서 동결 수식과 다른 명백한 구현 오류는 발견하지 못했다. 별도 CPU 확인에서
  두 checkpoint의 adapter 256개 tensor는 모두 달랐고 유효 LoRA update cosine은 0.84646이었다. 기능적 동일성 또는
  membership 효과를 파라미터 차이로 판단하지 않는다.
- 기존 own gain의 환자 간 점수 표준편차는 0.00148071, 대조 차감 후 response는 0.00218521로 약 1.48배 커졌다.
  같은 환자의 모델 간 변화 RMS / 환자 간 표준편차는 0.1494에서 0.0833으로 줄었다. 이는 현재 보정 구조의 재검토
  근거이며 신호 대 잡음비의 인과 추정 또는 보정 실패 원인의 확정은 아니다.
- NumPy와 128개 층의 dense LoRA 곱으로 별도 계산을 검증해 PASS했다. 코드·입력·checkpoint hash와 결과를
  기존 분석 폴더의 structure_check.json 및 structure_verification.json에 저장하고 분석 Markdown/HTML에 반영했다.
- 현재 후보는 효과 확인 상태가 아니며 표본 확대를 보류한다. 점수·대조 설계 수정 근거를 먼저 검토한다.

## 91-U-PILOT-REVIEW. 2026-09-14 — 구현·자료·학습·통계 정밀 검토와 판단 정정

- 사용자의 "제대로 검토" 요청에 따라 구현/수치, 통계 해석, 자료/학습을 나누어 독립 검토하고 원 설계·설치된
  PEFT/diffusers·캐시·저장 결과를 대조했다. CPU 코드로 수치와 반례를 재검산했으며 추가 GPU·환자·학습·fitting은 0이다.
- 분할과 실제 노출에 명백한 오류는 발견하지 못했다. 자료 격리·참조 1,600개·학습 912장 전부 노출을 확인했다.
  그러나 support 목적값·계수/gradient 궤적·비영점 gradient 수치 검산이 없어 최적화의 실제 개선은 검증되지 않았다.
- 캐시의 활성 token은 7개/7,168차원이다. 전체 norm 291.7156과 활성 norm 85.0643 때문에 radius .05는 활성 영역 대비
  최대17.15%다. 32fold 중30개가 경계에 도달했다. 공식과 일치하므로 버그로 지목하지 않고 미검증 설정으로 분류했다.
- 원 설계는 개발자료의 부호·정규화·결합 fitting과 기존 특징 대비 R 추가 이득을 평가하도록 했지만 현재는 양의 R 단독
  진단이다. 전체 설계 실패로 일반화하지 않는다. 현재 gradient dot/cos는 query/noise/reference/정규화가 달라 정확한
  1차 대조가 아니며, 해당 대조를 복원할 계수와 reference query gradient도 저장되지 않았다.
- E는1명뿐이어서 검출력 대조가 아니다. 품질 개선은 도메인 적응이며 암기 증명이 아니고, quality는 weak-label prompt,
  audit은 generic prompt라 조건이 다르다. 이 차이 자체를 오류나 약한 결과의 원인으로 확정하지 않는다.
- 직전 보정항 원인 판정을 정정했다. SD1.47578배 증가와 분산2.17793배 증가는 사실이나 보정 실패 증거가 아니다.
  모델 간 변화/환자 간 SD도 membership 신호 대 잡음비가 아니다. 반전 라벨+동일 순위의 평균 AUC .5와 LOO .5를 독립
  실패 근거처럼 세지 않는다. 임의 수치 반례로 순위 유지와 참여 효과, 분산 증가와 AUC 개선이 양립함을 확인했다.
- 16/80명은 검정력에서 산정된 수가 아니었다. 표본 확대·보정항 변경을 먼저 결정하지 않고 기존 입력의 수치 진단을
  다음 우선순위로 고정했다. 기본56F/38B, 목적값 trace·checkpointing·3간격 유한차분·필요시 같은 입력 FP32 재검사의
  사양을 numerical_diagnostic_plan.json에 준비했다. runner 미구현·GPU 미실행이며 성능 판정용 표본1명이라는 뜻이 아니다.
- U_PILOT_REVIEW.md/HTML과 review_facts.json을 작성하고 기존8명 분석·현황·README·CURRENT_STATUS에 정정 권위를
  연결했다. 원본 score·cohort·checkpoint·동결 코드와 계획 본문은 유지했다.
- 검토 입력13개 hash와 준비 사양의56F/38B 산술을 확인했다. 최종 정적 검증은 로컬 링크600개·PDF 쪽 앵커201개,
  끊어진 링크0개였다. 수치 GPU 검증을 실제 통과했다고 기록하지 않는다.

## 92-U-ACTUAL-VERIFICATION. 2026-09-14 — 실제 재검증·계측 보완·FP32 endpoint 재평가

- 사용자의 전면 재검증·수정 및 작업별 예상 시간 보고 요청에 따라 실제 GPU 검사를 수행했다. 각 단계 전에 시간을
  보고했고, 마지막 정밀도 민감성 발견과 코드 준비 지연 때 범위·잔여 시간을 갱신했다. 신규 환자·추가 학습·fitting은 0이다.
- 기존 분할·학습·품질 파일·E/U 점수·AUC와 4개 핵심 테스트를 새 경로로 재검사했다. 선택 영상2,304장 hash를 확인했고,
  원본 report를 덮어쓰지 않았다. 종료 시 cohort·cache summary·학습 계약·checkpoint·원점수·동결 소스11개가 그대로였다.
- probe_verified.py에 support 목적값·계수·gradient·update·projection·마지막 목적값 및 개별 loss 계측을 추가했다.
  기존 FP16 코드와 같은 셀의 loss/gradient가 정확히 일치했고 checkpointing 켬/끔도 두 정밀도에서 정확히 같았다.
- 수치 v2는 비교 요약뿐 아니라 원시 gradient 벡터를 보존했다. FP32 방향 FD 상대 오차는 h=.001/.0002에서
  .0360%/.0743%, 전체8좌표 벡터 상대L2 오차는 .0372%/.2275%였다. FP16의 작은h 오차43.058%를 미분 버그로 단정하지 않았다.
- 전체 U8×2모델의 6개 점수96개가 정확히 재현됐다. 32개 support 과정 모두 최종 목적값 증가,192회update 중1회
  중간 감소−.0000943023을 기록했다. 전체 추적 재현은824.17초,5184F/1152B였으며 독립 산술·metadata검증을 통과했다.
- 기존8명의 E/U basic controls를 generic/matched weak-label 조건에서 실행했다.1280cells/128image means/64patient scores/
  24AUC를 독립 검산했다. E음의평균loss AUC는 두prompt 모두 .5625/.5625이며 강한 양성 대조를 확보하지 못했다.
  batch4와원batch1은 수치가 달랐지만 U기본점수의 판별순위·paired부호는 유지됐다. 실행기 경과79.56초였다.
- 첫원환자의 저장계수를 고정한FP32 endpoint검사에서 paired member−nonmember차이가 +1.4278066e−5→−2.6920508e−5로
  뒤집혔다. 개별R부호는 유지됐다. 원래MSE는FP32이고scalar차감은Pythonfloat였으므로 최종FP16뺄셈버그라고 하지 않는다.
- 전체8명 결과를 보기 전에 동일FP32 endpoint규칙을 고정·hash결속하고,모든환자/모델/query/reference와R/세기본점수에
  적용했다. 저장FP16계수는 고정했다. 전체FP32 재최적화가 아니라 평가 정밀도 보완 버전이며 원본을 보존했다.
- 전체FP32재평가는350.98초/3840F/0B였다. FP32·원FP16각1920pairedcells,32fold/64score/16AUC를 독립검산했고,
  첫환자의 기존FP32 rawcells240개가 정확히 재현됐다. 후보R AUC는 .6875/.2500,양의paired차이는4/8→3/8이었다.
  model2의환자순위한쌍이바뀌었다. 세기본점수의AUC·순위·paired부호는 그대로였고 후보 효과가 입증되지는 않았다.
- final_summary.json에서 모든 단계의검증/출처/원시detail/공유helper hash와 원본11개를 다시 확인했다.
  U_VERIFICATION_RESULTS.md/HTML,현황,기존분석의정정안내,README,CURRENT_STATUS에 반영했다.
  실행기의초단위시간은로딩·내부저장을포함하며,외부구현·독립검증·문서시간과구분한다. 전체연구성공·DP보장으로표시하지 않는다.
- 최종 HTML 정적 검증은 로컬 링크622개·PDF쪽앵커201개,끊어진링크0개였다. 기존79편장부의검토범위는 그대로이며,
  UI 동작을 바꾸지 않았으므로 브라우저 동작 검사를 새로 했다고 표시하지 않는다.

## 93-U-SECMI-SCREEN. 2026-09-14 — 기존 공격 E/U 진단 실행·독립 검산·확대 보류

- 사용자 '실행' 지시에 따라 CDI 저자 저장소에 보존된 SecMI-stat을 현재 SD2.1 의료 모델에 이식했다.
  원 SecMI 저자 실험 전체 또는 전체 CDI/SOTA 비교로 부르지 않는다. source commit과 세 원본 파일 hash를 고정했다.
- 기존 fit8과 selection40을 분리하고 E/U 각 두 장·두 체크포인트를 사용했다. 총384영상 평가/192환자-조건 집계,
  4608F/0B, FP32 batch1, 고정 t=100/step10, 일반 prompt, 음의 L2→환자 평균을 주지표로 실행 전에 기록했다.
- 선택40명의 평균 점수 AUC bootstrap95% 하한이 두 모델에서 모두 .5를 넘어야 진행한다는 기준을 E/U 별도로 고정했다.
  A/B 각20명,seed260914,2000회,두 모델/EU에 동일 환자 추출을 사용했다. fit8/max로 미통과를 구제하지 않았다.
- 실측 GPU 실행은368.455488초(6분8초). 전384건 완료,모델별2304F/0B,epsilon prediction,adapter256개씩 전후동일,
  원본보호11개 hash유지. 코드 준비와 검증기 구현 지연까지 포함한 전체작업은약30분이며 GPU시간과 구분했다.
- 선택40명 E AUC .5225[.335,.7025]/.5000[.3125,.6825], U .4300[.2625,.61]/.5600[.38,.7275625].
  E/U 모두 사전 진행 기준 미충족. 현재 후보 확대·R 튜닝·추가 target학습을 보류했다. 누출없음/연구불가능 판정은 아니다.
- 독립 CPU검증기에서384rawtensorL2/192집계/실제학습노출/분할/12단계scheduler/출처hash/계산량을 확인했다.
  평균순위AUC와bootstrap 재추출횟수 가중치 방식으로 통계계산도 별도재검산했다. 다른 sklearn 구현의16AUC와 일치했다.
- 별도 root rawscan에서 FP32 L2 384개를 정확히 재현했다. 원시FP32상태의CPUfloat64 L2와최대차이는8.2441e-9였고,
  16개AUC/두진행판정이모두유지됐다. GPU 전체경로FP64 재실행 또는 독립GPU가중치검사를 했다고 표시하지 않는다.
- U_BASELINE_SCREEN.md/HTML에 실제결과·조건·한계·시간을 기록하고,index/현황/CURRENT_STATUS/README/이전보고서의
  후속안내를 갱신했다. 마지막 정적 검사는636개로컬링크/201개PDF쪽앵커,끊어진링크0개. UI동작변경은없다.
- 추가학습·후보R의새환자평가/튜닝·보정/최종시험 실행은0. 기존졸논K5/K10 계약과 기록은 변경하지 않았다.

## 94-RESEARCH-DESIGN-CASES. 2026-09-14 — PP-Mark·ICLR 실제 사례 대조

- 사용자가 실제 사례를 읽지 않고 이해한 척한 답변을 지적했다. 직전 답변 때 해당 두 폴더를 열지 않았음을 인정하고,
  PP-Mark-v0.4/PP-Mark, 별도 GitHub 사본, jailbreak iclr 2027의 초기 목표·설계·후속 결과를 실제로 확인했다.
- PP-Mark의 초기 git 계획, 현재 검증 관계, 원시 regeneration 세 집계, 8월 A100 proof timing 두 캠페인을 확인했다.
  실제 검출 성과/실제 proving과 signature 표에 상수로 입력된 accept=0을 구분했다. 초기 목표가 모두 보장됐다고 하지 않는다.
- ICLR의 원래 목표는 공격 생성이 아니라 P 보존 회복 구성요소 설명이었다. D3의 실제 긍정 구조와 gate 실패,
  공통순서 학습의 pooling 대비 불이익, 마지막316개 테스트 통과 뒤32회 baseline실패/240회미실행을 확인했다.
- CVPR의 원 설계·U 계획도 다시 열어 반응 전이와 기존 집계 대비 추가 정보가 여전히 미확인 가설임을 대조했다.
- research_2026-09-10/RESEARCH_DESIGN_CASE_AUDIT_2026-09-14.md에 직접 읽은 범위와 구체적 사실·판단을 기록했다.
  약12분의 기록 검토이며 실험 재실행·새 모델/공격 실행은0. CVPR 보류와 ICLR 최종 중단 상태를 변경하지 않았다.

## 95-METHOD-PREMISE-AUDIT. 2026-09-14 — 1단계 원문·코드·U 설계 근거 대조

- 사용자 '하나씩 아주아주 철저히 실행'에 따라 첫 단계 예상20–30분을 보고하고 MoFit·CDI 원문/저자 코드와 현재 의료
  학습·점수를 대조했다. MoFit의 LoRA 비교를 근거로 PFAMI 원문/공식 코드/후속 SD 재구현도 추가 확인했다.
- MoFit LoRA AUC54.35/TPR@1%=0, 의료ROCO54.44/2이며 강한 COCO 결과를 현재LoRA/U로 옮길 근거가 없음을 기록했다.
  LoRA의 PFAMI77.5AUC도 TPR1%이므로 저오탐 감사 성공으로 과장하지 않았다. 원래 conditioning 민감도 관측과
  image-surrogate→embedding→original-image 평가의 실제 메커니즘을 현재8차원probe와 구분했다.
- CDI 원형의 P/U_CDI와 환자 U의 의미, 26D특징/5fold/분포 가정/70–6000장 조건을 확인했다. 원형이두장에안맞아도
  별도학습한per-image scorer→환자집계 비교를 제외할 수 없으며 평균p와검정력환산값은patient-FPR실측이아님을 명시했다.
- PFAMI 원저자 상대loss+reference모델 보정과 CLiD SD의단순loss차분을 구분했다. 공식source stat의불필요crop계산까지
  포함한실제비용과이론상필요계산량도 나눴다. 현재256mode캐시 이식은 명시적인변형이며 latentcrop은pixelcrop이아니다.
- 현재학습은912장4–5회노출/step1000/epsilon/null-caption dropout없음이다. 도메인적응과교차사진반응관측은있지만
  membership특이성/강한baseline추가정보는 미확인이다. E음성을U불가능으로연결하거나새학습필요로판정하지않았다.
- R의정확히대응하는1차항과Hessian remainder를조건부로정리하고 실제양자화/6step/기존dot-cos대조의차이를명시했다.
  독립검토로 '전이자체없음' 과장과 'SecMI=절대loss' 혼동을배제했다. 기존실험gate결과와R확대보류는유지했다.
- METHOD_PREMISE_AUDIT_2026-09-14.md/HTML,설계초안의후속안내,CURRENT_STATUS,index연결에기록했다.
  HTML 정적검증: 로컬655링크/PDF213쪽앵커/끊어진링크0. 기존79편의검토수준을전문정독으로올리지않았다.
- 이번단계의새GPU/환자공격/학습/보정·최종시험은0. 다음우선순위는PFAMI계열비교의명확한정의와범위고정이며,
  성공한새방법이나현재의약한결과에대한확정원인을만들었다고보고하지않는다.
- 실제소요는약21분(20:43–21:04 KST),처음보고한20–30분범위안이다. 원문·코드분담과독립수식/해석검토를병행했다.

## 96-PFAMI-FIXED-COMPARISON. 2026-09-14 — 1번 안의 고정 비교 실행·독립 검산

- 사용자 '진행'에 따라 원문 대조 뒤 PFAMI 비교의 정의·연결 검토 15–25분을 먼저 안내했다. 큰 순서의 1번 안이며,
  '첫 단계 완료'가 원문·코드 대조만 뜻했음을 상태 질문에 명확히 답했다. 새 방법 설계·확대 단계로 표시하지 않았다.
- 원저자 DDPM loss/statistic AST를 256px 의료 latent diffusion에 이식했다. tensor 중앙207px crop, 같은 FP16 VAE mode,
  FP32 epsilon UNet, 0/50/…/450의10시점·독립 view noise·target 간 동일 noise를 실행 전에 고정했다.
  relative와 difference를 같은20F에서 비교하고 target-only와 원저자 기본 reference 보정의 차이를 명시했다.
- selection40명·E/U각2장·두 target+base의480건/9,600F/0B를 완료했다. fit1 벤치마크는 성능에서 제외했다.
  본 GPU 실행 전 실측 기반17–20분을 안내했고 실제 실행기는895.172191초(14분55초)였다. crop VAE160장 별도 계상.
- fit1 v1은240F 완료 뒤 root가 검산 파일 최종 저장 전에 실행한 순서 오류를 코드hash감지가 잡아 종료를 거부했다.
  failure.json으로 보존하고 추가약1분을 안내한 뒤 고정된 v2를 재실행했다. v2는37.998254초, 독립 잔차 검산 PASS.
  두 실행의240손실/12점수는 정확히 같고 원본4장의 latent 재인코딩도 exact였다. 벤치마크480F는 본측정과 별도다.
- U relative AUC .3550[.1775,.5225]/.6575[.4900,.8300625], difference .3425[.1824375,.5150]/.6725[.5025,.8325].
  ΔAUC +.0125[−.07,.0925]/−.015[−.10,.065]로 정규화 추가 이득 미확인이다. 모델2 difference의 관측을 숨기지 않되,
  전체 성공·정규화 우위로 바꾸지 않았다. E relative는 .4450/.5500이며 U의 대체 성공으로 처리하지 않았다.
- 독립 CPU 검산은480잔차/9,600MSE와 noise seed/hash,160환자-조건 기록의640점수,sklearn AUC16개,
  별도 가중치 bootstrap과 paired 비교8개를 확인했다. 최대 MSE 절대8.43379e−8/상대1.53357e−7,
  float64 저장잔차 재계산 후 AUC·CI·해석 변경 없음. 실제 검산4.890507초, 독립 GPU 재현은 아니다.
- 보호 원본11개와 input/source/동결code hash를 확인했다. ADDITIONAL_RELATIVE_GAIN_NOT_ESTABLISHED를 유지했다.
  선택40은 기존 개발자료, 두모델은 의존하며 단일 noise 실현이라는 한계를 기록했다. 환자특징/split bias 원인,
  누출 부재·전체 MIA 실패·재학습 필요를 확정하지 않았다. R 확대·추가학습·calibration/test 실행은0이다.
- PFAMI_COMPARISON_PROTOCOL.md와 PFAMI_COMPARISON_RESULTS.md/HTML, index, CURRENT_STATUS, 설계·SecMI 후속 링크를
  갱신했다. HTML 정적검사673개 로컬 링크/213개 PDF쪽앵커, 끊어진링크0. 문헌79편의 검토 수준과 UI 동작은 그대로다.
- 정의·연결 검토는약21분, 21:05 KST에 시작한 전체 작업은 마무리 기록 시점21:58 기준약53분이다.
  GPU 계산과 별개로 계약·구현·벤치마크 재실행·독립 검산·기록 시간을 포함한다. 큰1번 안의 이번 고정 측정을 닫았다.

## 97-PAIRED-SCORE-AUDIT. 2026-09-14 — 기존 PFAMI 점수의 공통 순위·참여 방향 변화

- 사용자 '다음' 지시에 따라 기존 selection40 저장 점수만 분석했다. 예상10–15분을 안내했고 새GPU·target질의·환자
  이미지평가·학습은0이다. 원 AUC를 이미 본 사후 분석임을 명시하고 파생통계 계산 전 contract와 입력·코드를 고정했다.
- A/B각20명의 실제 학습 노출·라벨 교환, 동일 target noise3,200쌍을 확인했다. 같은base 차감은 paired delta에서
  상쇄되고 환자단위 최대오차2.78e−17/부호변경0이었다. 원입력hash14개와 기존 보호 파일을 유지했다.
- U relative의 Pearson .997244/Spearman .994559, 전체780쌍 중762쌍 동일, A/B400쌍 중389쌍 동일을 확인했다.
  A/B순위반전은 개선8/악화3으로 평균 AUC 초과분 .00625 [−.00753125,.0225]와 정확히 연결됐다.
- c=(s1+s2)/2, d=(s1−s2)/2의 분산 항등식을 확인했다. 공통 분산비중99.8603%는 점수의 대칭성분 비중일 뿐
  환자고유편향의 인과적 비중이 아니다. commonAUC .345/.655의 합1 역시 반대라벨의 항등식으로 명시했다.
- U Δ=s_member−s_nonmember는 양수23/40, 평균+.00061455 [−.00032300,.00146634], A/B평균−.00042309/+.00165219.
  E 음의loss의 양의 평균Δ .00018294 [.00001483,.00035517]는 보조 개발 관측으로 보존했다. 변화부재·누출부재나
  개인 인과효과로 일반화하지 않고, 두 target와 정답을 쓰는 분해를 배포가능한 새 공격으로 취급하지 않았다.
- 새 분석기와 별도 검산기를 작성했다. 컴파일·정확한 합성 항등식 검사를 거쳐 실제8조건을 실행했다.
  독립 검산은320환자-조건행/1600값,6240순위비교/3200AB기여/400원시margin,32AUC(원16개포함),
  16상관·16bootstrap구간을 대조했다. 최대차이3.33e−16, 동일환자bootstrap순서/hash, PASS였다.
- 데이터검산CPU .498초, 분석프로세스약1.13초, 독립통계검산CPU .319초/프로세스약2.75초였다.
  시작22:16:33 KST, 분석·기록닫음22:35:01 KST(1108.97초),최종링크검사·이력까지약19분이다.
  분석기·검산기필드연결과기록이 예상보다 길어졌고 작업중남은시간을갱신했다.
- PAIRED_SCORE_AUDIT.md/HTML,PNG·SVG도표,index,CURRENT_STATUS에 반영했다. 정적검사686로컬링크/213PDF앵커,
  끊어진링크0,도표파일·실제표시 확인. 기존79편장부와필터UI는 그대로이며 브라우저 동작 검사를 재실행한 것은 아니다.
- 큰1번 안의 이 저장점수 진단을 닫았다. R추가가치·새방법우위·target재학습필요의 근거를 확보했다고 하지 않았다.
  R확대보류를유지하며 추가튜닝·학습·보정/시험·다음GPU실행은자동으로시작하지않았다.

## 98-RESEARCH-FRAMEWORK. 2026-09-14 — 사용자 네 단계 고정·번호 오류 정정

- 사용자가 연구틀을 놓친 채 부분 결과에서 성급한 중단/설계변경으로 넘어가는 반응을 지적했다.
  네 단계는 1문제·목표고정(완료),2기존방법실패조건·원인확인(진행),3해결설계,4핵심연결검증이다.
- RESEARCH_FRAMEWORK.md/HTML과 research_state.json에 기준·완료/부분/미완료·다음 한 작업을 고정했다.
  MoFit/CDI핵심원문검토와SecMI/PFAMI일부진단 완료를 실제환자확장·강한집계·실패원인확인 완료로 세지 않았다.
  다음은2번의 'MoFit·CDI 환자 확장과 강한 집계의 공정한 비교 명세 고정'이며 실행준비/성공을 선포하지 않았다.
- 이전 METHOD_PREMISE/PFAMI/PAIRED/WORKLOG95–97/일부JSON의1번표기는 assistant의 번호 오류로 명시했다.
  원시결과·hash결속Markdown·계약은 보존하고,4개HTML의상단정정과index/CURRENT_STATUS의현재위치를갱신했다.
- 프로젝트AGENTS.md에이기준을시작시에읽고작업의단계·질문·목적/산출물·시간을확인하도록기록했다.
  약한부분결과를후보/연구폐기근거로확대하지않고,2번필요조사는지속하며,R의작은수정으로자동전환하지않는다.
  매세부작업마다새승인요구를만들거나간단질문응답을불필요한기록때문에늦추지않도록했다.
- 검증:현재단계2/목표고정완료,4개정정배너,보존대상13파일hash일치,로컬698링크/PDF213앵커/끊어진링크0.
  문헌79편장부의검토범위는유지했다. 새연구실험·GPU·학습은0이다.
- 예상5–8분을먼저안내했다. 23:13–23:25 KST 약12분이며,진행중사용자의Codex안내창설정확인요청을병행했다.
  연구틀은변경하지않고중간요청에답했으며,연구작업과제품설정문제를별도범위로처리했다.

## 99-PRIOR-PATIENT-SPEC-CDI-KERNEL. 2026-09-14–15 — 2번 비교 명세와 실제 CDI 계산 경로

- 사용자는 2번 조사를 계속하라고 지시했다. MoFit/CDI의 환자 확장, 같은 fit 정보·환자 집계·비용·threshold 규칙을
  PATIENT_BASELINE_SPEC.md/JSON과 원형 소스 메모에 고정했다. 기존 encoder-only U fit80/selection40 대조가
  이미 완료됐음을 바로잡았으며, 모든 학습형 대안이 미실행이었다고 표시하지 않았다.
- MoFit 공개 COCO scalar 4개 파일을 실제 원형 평가기와 독립 rank AUC로 재계산했다. 선택 gamma .55에서
  literal ASR .883/AUC .941948/첫 FPR>=1%의 TPR .468, 독립 rank AUC .94394/FPR<=1%의 TPR .488이었다.
  입력에 맞춘 gamma·scaling과 논문 표의 작은 차이, 공개 header iteration 불일치를 기록했다. CPU .685초이며
  의료 모델·gradient 경로 재현 또는 held-out 환자 검증으로 부르지 않는다.
- CDI 소스의 6개 extractor를 동결한 실제 epsilon backend에 연결했다. GM mask와 NO double-noising을 임의 수정하지
  않고 원형 코드 결과와 논문 의도의 차이를 따로 기록했다. 실제 objective 계산 수를 세어 SciPy MemoizeJac의
  nfev 캐시 중복을 비용으로 잘못 세지 않았다. FP32 입력 미분, eval 가중치 고정, 원시 중간값 저장을 구현했다.
- fit14393 E2/U2·model1 step1000의 실제 GPU 계산을 완료했다. 232F/68B, 측정25.238초/실행기29.724초,
  peak allocated 약4.10GB였다. E는 각4회 노출/U는0회. 104개 특징 float64 검산 최대 상대차이1.68013e-7,
  기존 SecMI 중간값4건 정확히 일치, noise32개 독립 재생성, 동결40파일·LoRA/gradient guard가 통과했다.
- NO의 최적화 objective와 공개 코드 final feature가 실제로 다름을 확인했다. 네 영상의 objective 감소는 있었지만
  모두5 iterations 제한 종료였으며, 이 차이가 membership 성능을 낮추는지는 아직 모른다. 구현·검산 통과를
  기존 방법 실패 원인 발견이나 연구 기여로 표시하지 않았다. CDI_KERNEL_RESULTS.md와 원시 artifacts에 보존했다.
- 비교 명세·replay·kernel 준비와 검산은23:26–00:08 KST 약42분이었다. 첫 안내 이후 kernel 구현 추가25–35분을
  별도 안내했으며, 실측 GPU 시간과 전체 작업 시간을 구분한다. 이후 U fit80/selection40 분석 계약·실행기를
  준비했고00:39 KST 전체480건 중 신규478건 추출을 시작했다. 이 후속 결과는 다음 항목에 기록한다.
- 현재 단계는2번 진행 중,1번 목표만 완료다. MoFit 실제 의료 확장·CDI reference test·E/U 원인 구분은 남는다.
  R 확대·새 target 학습·calibration/test 평가를 수행하지 않았다. 단계3/4가 준비됐다고 선언하지 않았다.

## 100-CDI-U-COHORT. 2026-09-15 — 2번의 기존26D·환자 집계480건 실행

- U fit80/selection40,2장씩,두 target의480건을 완료했다. kernel U2를 재사용하고478건을 새로 계산했다.
  source literal26D,10개LR변형(두target별총20개scorer)과8개scalar변형으로18방법을비교하고기존DINO를재사용했다.
  fit-only5fold log-loss/C 선택,선택40 평가,동일 scorer의 target 교체 진단을 새 전체점수 열람 전에 고정했다.
- 실제 전체28,058F/8,378B,신규27,942F/8,344B,실행기2,916.393초(48분36초)였다. GPU 추출45–55분 안내에
  들어갔다. source/입력81개는 후속base 계약 수이며 이번 fullU의 원래 동결 수는56개다. NO5회 제한 종료480건.
- 480개 raw/12,480개 특징/noise3,840개 독립 재생성,과거 SecMI192건 정확 일치가 PASS했다. 원시 검산58.666초,
  fitting/통계 계산.923초/경고0. 저장 scaler·계수 예측/CV 저장확률·log-loss/C/공통 bootstrap도 검산했다.
  GPU Jacobian 재실행·CV fold별 독립 재학습까지 했다고 표시하지 않았다. 이전결과·frozen코드는 보존했다.
- 주 비교 CDI image mean tuned−SecMI 평균Δ−.00375[−.1725,.15878],patient meanmax52 tuned−image mean
  평균+.070[−.03625,.175]였다. 보조 fixed C의 patient 집계는 AUC .6325/.6700,평균Δ+.15375[.02625,.28625]
  로 실제 양성 관측을 남겼다. NO27 모델2 .695[.525,.855]도 생략하지 않되,52D 대비 추가이득 CI는0포함이다.
- 동일 scorer20개 평균 참여방향Δ CI는 전부0을 포함했다. 모든 model1-fit scorer는A음/B양,model2-fit은A양/B음
  이었다. 전역 이동만으로AUC원인을단정할수없어 다음CPU순위분해를 정했다. 이를2번전체실패/방향폐기로읽지않는다.
- 비교 준비부터 검산까지00:08–01:30 KST 약82분이었다. 당초 준비포함60–75분보다 길었으며,외부계약·새추출기·분석
  접합 준비에약31분이 들었다. GPU48.6분과전체작업시간을구분했다. HTML/독립그림/CSV에19방법전부를기록했다.

## 101-CDI-RANK-DECOMPOSITION. 2026-09-15 — 2번의 상대 개선 출처를 저장 순위로 분해

- 결과를본후,20개 고정scorer·800개patient records·8,000개A/B쌍에 c=(s1+s2)/2와target변화의순위기여를계산했다.
  사후기술분석이며 새공격·학습·GPU가아니다. 입력/코드hash를계산전에기록하고 원래primary판정을유지했다.
- meanmax52−image mean의관측이득=공통성분이득+target변화상대기여:
  모델1tuned .0825=.0875−.0050, fixed .1850=.1925−.0075;
  모델2tuned .0575=.0875−.0300, fixed .1225=.1500−.0275.
- 네경우모두참여방향의target변화추가포착으로이득을설명할근거가없었다. 그러나공통성분은인과적으로식별된편향이
  아니고,차이의CI도0포함이므로모집단음의기여/개인인과/누출부재로확대하지않았다. 모델2의target기여자체는양수도있다.
- 별도코드가원저장예측으로20개scorer/8,000쌍credit/4항등식/sklearnAUC/직접bootstrap을재계산해PASS했다.
  분해CPU약.52초,독립검산.618초. 준비·구현·검산은5–10분안내후약11분걸렸다.
- 이어같은40명U80장을실제추가학습전base에질의하고기존20scorer를그대로적용하는대조를정했다. 전체20–30분,
  GPU8–10분을안내했다. 질문은query모델의추가학습전반응에도A/B구분이남는지이며base회원성공격AUC가아니다.
  scorer의target-fit80학습정보는그대로있고base사전학습membership은unknown이다.
- 01:57:41 KST base 실행 시작. 외부계약75f4430b…19bc9/81파일,actualCPUpreflightPASS. 이 후속은다음기록에서닫는다.

## 102-CDI-BASE-CONTROL. 2026-09-15 — 2번의 실제 추가학습 전 모델 대조

- selection40의 U80장을 실제 pretrained base UNet에 넣고 기존 target-fit20개 scorer를 그대로 적용했다.
  81개 입력/코드 동결,fp16 artifact를 FP32로 읽은686개 state tensor 일치,LoRA0 및 실행 전후 값 불변을 확인했다.
  GPU 실행기420.510초(7분),score397.342초,4,594F/1,314B,peak allocated4,067,063,808bytes였다.
- 추출8–10분 예상 안에 끝났다. 최초 CPU 검산은noised import 누락으로 실패했고 오류를 즉시 알렸다.
  수치 함수는 그대로 두고 별도v2에 import/의존성 검사/기존U2 실제 경로 검사를 추가했다. 원v1과실패기록은보존했다.
  수정5–10분 안내 뒤 root와별도agent가AST불변을확인했다. v2 실제 검산20.350초,분석.426초였다.
- 검산80장/2,080개특징/noise640개/20scorer의800예측/8,000쌍/기존2,000bootstrap PASS.
  v2 SHA864b86c3…17a21. GPU Jacobian 재실행이나 base의 사전학습 membership 확인으로 표시하지 않았다.
- meanmax52fixed의base A/B AUC .555/.570→자기target .6325/.670. 개별20scorer의base AUC CI는모두.5포함,
  target−base CI는모두0포함. 단순히‘base편향이전부였다’는원인을확보했다고결론내리지않았다.
- meanmax−image mean 이득의target−base차이는 M1fixed+.1275[−.05,.3025],tuned+.09[−.02756,.215],
  M2fixed+.17[.0275,.3225],tuned+.15[−.015,.3275]. M2fixed양성관측도기록하되+.17은meanmax+.10과
  image mean−.07이함께만든상대값임을명시했다. 반복개발40명/조건부구간/보조fixed정책을유지했다.
- 원래400명 A/B 배정도별도SHA계산으로재생성해불일치0을확인했다. 직접정답을feature에넣는경로를찾지못했지만
  나이·세부질환·follow-up분포교란을배제한것은아니다. spec_sources/cohort_assignment_recheck_20260915.md에근거저장.
- 이결과를바탕으로2번의다음한질문은표준meanmax가영상간차이를어떻게쓰는가로좁혔다. 사진2장의meanmax점수를
  mean항/half-range항으로정확히분해하고기존fit기준으로항을제거하는CPU작업5–10분을안내했다. 새공격설계가아니다.

## 103-CDI-MEAN-SPREAD. 2026-09-15 — 2번의 표준 집계가 사용하는 정보 확인

- `max=mean+|x1−x2|/2`로4개동결meanmax52 scorer를기존fit기준으로중심화한mean항/half-range항으로분해했다.
  3개feature source(target1/2/base),selection40,기존2,000bootstrap을사용했다. 새로운fitting/GPU/방향선택은0이다.
  실행전protocol17:23:49UTC(02:23:49KST),producer743efedf…194a29를동결하고사후기술분석범위를명시했다.
- 실제480행·4,800쌍·36AUC·24개차이구간을계산했다. 원래full확률순위와불일치0,새producer계산.498초.
  별도verifier b9799481…84299가원52D표준화/보상합/기존확률/sklearnAUC/독립가중bootstrap을검산해PASS했다.
  검산내부.821초,원52Dlogit최대차이7.11e−15,bootstrap구간차이0,새실패기록없음.
- 자기targetfixed의 full/mean-only/spread-only AUC는모델1 .6325/.5450/.6275,모델2 .6700/.5775/.6800.
  차이항제거로target8문맥모두AUC점추정이내려갔지만자기target4차이구간은모두0포함이다. 이를모집단이득으로확정하지않았다.
- 모델1fixed scorer가model2응답에서도A를.7025로구분했다. model2의실제member는B이므로같은방향membership AUC는.2975다.
  이고정점수의보편적member-positive해석에대한반례이며,target별로다시fit한CDI전체실패증명은아니다.
- 차이항역할은기존표준집계가제공하는정보다. 이를새공격기여로가져가거나,고정scorer항제거를별도학습mean26/image mean과의
  차이원인전체로간주하지않았다. 실제환자군구분과참여영향의구분은미완료다. 단순base편향원인설도확보하지못했다.
- 준비·별도검산기접합·해석·기록은처음5–10분안내보다길어져20분이상을썼고남은시간을갱신했다. 순수CPU산술시간과구분한다.
  이번긴연속작업은09-14 23:26:47KST 시작에서09-15 02:41KST 기준약3시간14분이며,GPU추출실행기합은약56분이다.
  나머지는소스/계약/구현/검산기오류수정/대조분석/기록시간이다. 성공한실험시간만총시간처럼보고하지않는다.
- 결과MD/HTML,index,RESEARCH_FRAMEWORK,PATIENT_BASELINE_SPEC,두상태JSON,CURRENT_STATUS를동기화했다.
  정적검사756개로컬링크/213개PDF앵커/끊어진링크0. 기존19방법그림은실제열람했고79편장부검토수준은그대로다.
- 현재2번진행중,1번목표만완료다. 진행중계산프로세스는없다. 다음은환자군구분과참여를나눌대조의필요성·예측·비용을
  한개로구체화하는작업이다. 교차배정target은가능한진단이지만자동추가학습/필수실험/3번설계로승격하지않았다.

## 104-STAGE2-EU-MOFIT. 2026-09-15 — 빠진 비교의 실제 실행으로 후속

- 사용자 “최대한 다 해봐”에 따라13:23:27KST에 후속 작업을 시작했다. 1번 목표 완료/2번 진행중/3·4 미확립을 유지한다.
  추가 점수 분해만 반복하지 않고 E/U 전이, 의료 MoFit, 원형 CDI reference 절차의 누락된 비교를 실제로 수행한다.
- 13:30KST 무렵 CDI E 추출을 시작했다. 기존 fit80/selection40·E2·두 target의480건 중kernel E2를재사용하고478건을
  계산한다. 계약ff77e641…13316,새producerb6217ec0…ef700. 원 U 코드·과거출력·target·졸논계약은변경하지않았다.
  사전 E/U분석계약bc2497d1…db43은 E-fit→E/U, 기존 U-fit→E/U의네셀·동일18방법·5fold/2000bootstrap을고정한다.
- 분석기27106107…e08bc와독립검산기ee7466ae…de020을실제성능열람전에검토·합성검사·동결했다. 신규E scorer20개와
  CV250개는독립경로로다시학습하며,기존U 파라미터/CV/예측보존,11520예측,144AUC,156paired대조를검산한다.
  Eraw검산기66c8f462…ee9f52는원 U 수치함수를유지하고 E의실제노출정답·192개기존SecMI대조를확인한다.
- MoFit은실제의료BLIP revision f8a8378a…26c56(0.99GBsafetensors)와CLIP전체77×1024입력을준비했다. CPU14.716초,
  후보4장의의료caption은모두같은출력이며수동편집하지않았다. 공개COCO1000+300 schedule/NIH P256·SD2.1·의료BLIP
  adaptation임을명시한다. 의료판별성능전체재현이나기존R실행으로부르지않는다.
- MoFit adapterd0036a4a…0380,runner17779596…1489,독립검산기cb48ec54…713f,계약44b4de26…61b3을동결했다.
  2+2benchmark는FD두간격과별도CFG1대조포함25F/5B,VAE9F/3B. 예정full E/U1장씩은6406F/2600B,VAE2006F/2000B.
  전체scalar/hashtrace와지정10개update tensor를저장한다. FD작은신호/상쇄는inconclusive로두고모든역전파독립검산을
  주장하지않는다. 합성전체packet/CFG/endpoint/변조거부검사와CPU계약사전검사를마쳤으며GPU는E추출뒤순차실행한다.
- CDI원형원문은단측Welch지만저자함수는양측p+방향bool이다. 원함수AST·단측SciPy·독립t/df/sf를대조하는patient-cluster
  확장8개를사전고정했다. 20명대reference20명집합검정이며환자1명판정·원형1000subset전체재현·FPR추정으로부르지않는다.
- 학습조건대조도병렬준비했다. null/generic/weak-label,동일480장·3모델·t140·image별공유noise·FP32eval의4320F/0B다.
  producerf1e2653b…621a,추출계약c582c12e…1869,분석기7cf4b95d…8b63,분석계약799725fb…4b88을사전동결했다.
  원시1440packet을검산한뒤target24AUC/base A배정6AUC/조건차이24개를모두기록한다. 낮은loss를membership으로
  혼동하지않으며GPU MSE와저장예측FP64재계산의순위차이도보고한다. 추가target학습/fitting/조건선택은없다.
- 이항목은실행전준비·진행이력이다. 결과·실측·검산판정은완료후이어기록하며,계산정상작동과실패원인규명을구분한다.

## 105-STAGE2-EU-PROMPT-RESULTS. 2026-09-15 — 실제 누락 비교와 한계 확인

- CDI E480건(신규478/재사용2)을51분6초(3,066.060초)에 완료했다. 전체28,064F/8,384B,
  신규27,948F/8,350B다. E 원시12,480특징·3,840noise·192기존SecMI를73.701초에 검산했다.
  결과SHA a4d9441aed70e96b77fd494b7c0103b43a774f70b6556b60fd7db36a97097aa6.
- 사전 고정 E-fit→E/U 및 복원한U-fit→E/U의18방법·144AUC·156차이를1.897초에 계산했다.
  별도 검산기는20신규E scorer·250CV 재학습,기존U동일성,11,520예측·144AUC·156구간을1.245초에 대조했다.
  분석SHA44977ce1ad5224348ff59d6e2fe92cf7aa0e243582dea0842b22a21d2a71fffd.
- 같은U40명에서 meanmax52fixed는 E-fit→U .4175/.3600, U-fit→U .6325/.6700이다.
  모델2 회복fixed+.3100 [.0624,.5475],tuned+.2350 [.0099,.4576]는 관측된 기존법 차이다.
  그러나 원형26D 기반16방법×2target의 E→U32차이 구간 모두0포함,E-fit→E18방법의AUC구간 모두.5포함이다.
  NO27모델2fixed의E→U−.3450 [−.5875,−.0925]는 별도비원형진단이며 원형 전체실패로 일반화하지 않았다.
- 원형CDI함수(AST동일)의양측p,원문단측Welch,별도t/df/sf산술을20참여환자 대20비참여reference의8조건에서
  대조했다(.198초). 단측p .111–.691이다. 환자단위 평균을 먼저 계산했고1환자membership/FPR/전체1000subset재현은 아니다.
- 후속scale/계수교체는사후진단으로동결했다. mean26/meanmax52fixed/tuned의64AUC·80차이·2560예측,
  새GPU/fit0,같은모듈두산술·rank·bootstrap경로를대조했다. 별도독립검산기로과장하지않는다.
  E계수+Uscale의U AUC meanmaxfixed .385/.340으로회복되지않았으나,U계수에서scale을바꾸는효과는
  모델1+.0625 [.0125,.1225]다. scale무관/계수가개인참여원인이라는결론은내리지않았다.
- prompt대조는null/generic/weak-label×실제base·M1·M2×480장,4,320F/0B를400.489초에완료했다.
  원시1440packet·4320FP64잔차와정답/노출/동일noise·캐시·모델불변검산15.149초,분석.373초였다.
  target24AUC와base A배정6AUC를전부기록하고두MSE산술경로에서30개AUC pair-rank차이0을확인했다.
  prompt24차이구간모두0포함이다. 단일t140/noise/mode-latent조건이며전체MoFit나모든conditioning반증이아니다.
- MoFit benchmark는25F/5B,VAE9F/3B,24.052초(측정6.020초),peakCUDA5,490,651,648bytes였다.
  CFG1/2첫점grad cosine .99999999991955였으나fulltrajectory동일성을확인하지않아literalCFG2/매회monitor를유지했다.
- Pixel FD의초기h.01/.005 상대오차20.85%/25.28%를무시하지않았다. 같은점·방향의추가h5개를새계약으로
 20F/0B,19.474초에계산·3.554초에저장산술검산했다. h=.002→.0001에서오차1.9714%→.003619%로수렴했고,
  저장FP32예측의FP64MSE경로도1.9723%→.07566%였다. 모든7간격보존,단일방향localderivative근거로만해석했다.
- MoFit모델1full E002/U001은14:54:05KST 완료,실행923.981635초/측정908.095초,
  6,406F/2,600B,VAE2,006F/2,000B다. 결과SHA084f9bc4ed96834c77ec679fb42ea4d219f0f0b5477604387f0778c9c136c607.
  원frozen검산기cb48…713f로CPU4.278초PASS,영상당2,336저장산술check·1,300scalartrace·10update snapshot을확인했다.
  전체신경망2,600backward를독립재계산했다고주장하지않는다. 환자한명성능/AUC/개인인과효과판정은없다.
- 모델2비참여동일E/U대조를실제시작했다. 새로운외층runner c3204a9f…52be2d,verifier cd495f21…73d4,
  계약109d74ae…b3dfb(62파일)이며수치kernel/영상/의료caption/난수/1,000+300은동일하다.
  새외층guard초안의LoRA FP16경유기대값은동결전정적검토에서오류를찾아저장FP32직접비교로고쳤다.
  기존frozen코드/모델1출력은변경하지않았다. 실제모델2학습manifest912장·4,000노출에서14393환자0장/0회다.
- E/U144표값·reference8p·prompt30표값·원자료provenance를별도담당자가종합대조했다. 본보고서에scaler진단의범위와
  보조U영상+참여정답의정보접근차이를추가했다. 원시계약/target/calibration/test/졸논K5·K10은그대로다.
  현재2번진행중이며모델2완료결과와최종HTML/상태검사는후속항목에기록한다.

## 106-STAGE2-MOFIT-SECOND-TARGET. 2026-09-15 — 두 target 대조 완료와 두 장 집계의 실제 후속

- Model2 E002/U001 full은15:18:44KST에완료했다. 전체924.052725초,측정908.003058초,
  6,406F/2,600B,VAE2,006F/2,000B,peak할당5,490,651,648bytes다. 동결검산기cd495f21…73d4로
  실제노출원장·저장산술·10update snapshot/영상 검산4.292894초 PASS.
  결과SHA637ddad9ce77cbfc9b83659d3fcb344aa5d66d44fcdf13ba627dd03fe8558588.
- 원문식7의member-high h에서M1−M2는E002−.003009408712,U001+.009727507830이다.
  M2E h=.029697448015,u=.304328531027,v=−.012326359749;
  M2U h=.016351968050,u=.261809021235,v=−.012620270252다.
  E/U방향이다르며U한장성공·E한장실패·개인인과·환자군AUC로해석하지않았다.
- 사용자‘최대한’범위에서다음한질문을정했다: 같은환자다른사진에서도방향이유지되고두장집계는어떻게되는가.
  이미입력이고정된14393의나머지E006/U004×2models를full1000+300으로추가4records계산한다.
  새환자·target학습·R수정·부호/γ선택없음. E006실제M1노출4,환자합계8;M2둘·환자모두0.
- 새outer runner24eb5bf57150838a8469d71b6f04d2e0b38f745c5e19764134a5bd609a780fce,
  새외층verifier4ff4dbd187ffa86be813e6e598becd13e59bb0123445024f211932f7759ed1f5,
  실행계약55d9526e08755c3c6dde788d641b6a304f56ab83b2ac326b07bd6e7367c38aab(79파일)을동결했다.
  root는새runner/외층verifier전체를읽고두모델순차해제·동일kernel/난수·실제FP32LoRA·이전fingerprint결속을검토했다.
- 합칠분석은새결과전고정했다: h,−u,−v,−(u+v)를사진에서먼저계산한뒤E2/U2의mean/max를모두출력한다.
  image32행/집계32행/target차이32행,환자1명이며AUC/CI/선택된최종score는없다.
  분석기aa97d16f3e85c844cb6d411cbcb8a79d76a6ef247be3e6652afff7d85cf1080e,
  분석계약15001af7879637bf82f432c54084691f04244f4746f2135d362c180c821ffce0.
  기존4개raw/결과/검산hash를고정하고신규검산PASS뒤새출처를실행protocol에결속한다.
- 추가4 GPU는15:36:47KST무렵session82116로시작했다. 예산12,812F/5,200B,VAE4,012F/4,000B.
  최초준비포함35–40분안내는외층원장/검산/계약준비가약15분걸려45–50분으로정정해사용자에게알렸다.
  GPU자체예상은실측에근거한약31분으로유지하며최종실측은완료후기록한다.
- 원문PDF MoFit pp.6–7,23–25를직접추가대조했다. 의료Table12 AUC54.44/TPR2.00,
  A.13극단사례복잡도와오판의관계,Table15 Flickr의TN수치예외,Table14 noise별TPR18–41변화를
  spec_sources/mofit_failure_cases_scope_20260915.md에기록했다. 현재NIH의원인으로일반화하지않는다.
  새원문메모는기존동결명세/원본을바꾸지않으며79편전체완독이라고보고하지않는다.
- CDI_EU/PROMPT/MOFIT의새MD와3개HTML,index의최신결과링크를생성했다.
  중간정적검사는807로컬링크/213PDF앵커/끊어진링크0이었다. 최신추가메모와후속결과반영뒤최종재빌드한다.

## 107-STAGE2-MOFIT-E2U2-CLOSEOUT. 2026-09-15 — 추가 네 건과 고정 집계 완료, 2번의 결론 범위

- 같은 환자14393의 E006/U004 × 두 target 추가 4건은 16:08:17 KST에 완료했다.
  실행기 1,877.598318초, 계측 구간 1,848.952663초, 12,812F/5,200B, VAE4,012F/4,000B다.
  결과 SHA256: efc281234e77e83d7e7f3feaca9703668a238b087580941840d0758e4cddb241.
  raw SHA256: 5dff7917ed5011e6d955ea2812faf9c7f3d180062f08a57d69dc8867aa4957b4.
  protocol SHA256: 050087531175b40a6bdf840a0e314a0e2f816f42a19a496ebac662af7bfb7c29.
  저장 산술·실제 manifest 노출·79개 동결 파일·모델 fingerprint 검산은 5.352677초에 PASS했다.
  M1 E006은 실제4회, 환자 E2는8회 노출됐으며 M2의 해당 환자는0회다. 모든 U는 두 추가학습에서 제외됐다.
- 합계는 한 환자 E2/U2, 영상4장 × 두 target의 8 records다. 세 full 실행기 합계는
  3,725.632678초(62분5.6초), 25,624F/10,400B, VAE8,024F/8,000B다. Benchmark와 FD는 별도다.
  동일한 의료 BLIP 입력·1,000+300회·동결 수치 kernel을 사용했다. 새 target 학습·R 수정·환자 확대는 없다.
- 미리 정한 h, −u, −v, −(u+v)의 사진별32행·mean/max32행·target차이32행을 모두 생성했다.
  분석기의 산술 검산 뒤 별도 담당자가 세 실제 results에서 부호식과 math.fsum으로 다시 계산했다.
  JSON/CSV 원값은 exact 일치했고 본문 8원점수·16집계행은 표시 반올림 오차 안에서 일치했다.
  복사 원자료·manifest·명칭 정정의 hash도 대조했다. 전체 neural backward를 독립 재실행한 검산은 아니다.
- U h의 참여 target−비참여 target 차이는 U001 +.009727507830, U004 −.000533610582다.
  두 target의 max가 모두 U004를 선택하므로 U001의 큰 양의 차이는 max 출력에 남지 않는다.
  U mean은 +.004596948624, U max는 −.000533610582다. E mean/max는 +.003606721759/+.002615600824다.
  이는 특정 max 집계가 무엇을 버리는지 확인한 사례다. 임계값에 따른 false negative나 환자군 성능 판정은 아니다.
  표준 평균으로 결과가 달라지므로 이 효과 자체를 새 알고리즘 기여나 기존 대안에도 남는 한계로 삼지 않는다.
  두 전체 학습 환자군이 다른 target의 차이는 개인 한 명 참여의 인과효과로 식별되지 않는다.
- 동결 분석 계약의 L_VLM 설명은 원시 조건부 loss를 가리켜 논문 식8의 v와 이름이 혼동됐다.
  원본 계약/계산을 바꾸지 않고 mofit_score_naming_clarification_20260915.md를 추가했다.
  −v는 식8의 보조 차이, −(u+v)는 별도 VLM 조건부 원시 loss로 표와 본문에서 구분했다.
- RESEARCH_FRAMEWORK, CURRENT_STATUS, PATIENT_BASELINE_SPEC, CDI_EU 및 MOFIT 보고서와 두 상태 JSON을 갱신했다.
  MoFit 결과 표·CSV·원자료 사본·manifest는 mofit_two_image_results_artifacts/v1에 보존했다.
  최종 HTML 빌드 후 로컬 링크813개·PDF 페이지 앵커214개를 검사했고 끊어진 링크는0개다.
  문헌 장부는79편 중 선택 절75편·공식 본문 일부3편·초록1편이다. 전체79편 완독으로 바꾸지 않았다.
  최종 프로세스 조회에서 실행 중인 python은 없었고 active_execution은 no_running_execution이다.
- 이번 연속 작업은 13:23 KST부터 약2시간57분 진행했다. 원문/코드 대조·준비·실제 GPU·검산·문서 시간이 포함된다.
  E480 추출/네 셀18방법, reference Welch8, scaler/계수 대조, prompt4,320F, 의료 MoFit8건까지 완료했다.
  CDI의 U-fit 회복에는 보조 U 사진과 참여 정답이 필요하며 원형26D E→U32차이의 구간은 모두0을 포함한다.
  prompt24차이도 구간이 모두0을 포함했다. 이를 일반적인 E→U 실패나 모든 조건부 공격의 반증으로 확대하지 않는다.
  의료 Eq9 fusion·환자군 성능·개인 참여에 특이적인 실패 원인과 새로운 해결 원리의 연결은 미완료다.
  현재는 고정 네 단계의2번이다. 추가 결과를 찾지 못한 부분과 연구 불가능을 구분하며3번으로 넘기지 않는다.

## 108-STAGE2-NOISE-PROBE-INTERVENTION. 2026-09-15 — 반복 측정·최적화 입력·개별 학습 기여를 분리

- 사용자 “2번 더 해봐”에 따라 같은 연구 목표와2번을 유지했다. 17:58 KST부터 원문·실행 기록을 다시 대조하고, 측정 noise/모델별 최적화 입력/전체 학습 환자군 변경이라는 경쟁 설명을 분리하는 실제 대조를 추가했다. 각 작업 전에 범위와 시간을 알렸고, 추가 대조학습의 준비15–20분·실행16–18분·평가약5분을 고지했다.
- EMNLP2024 User Inference 및 SatML2023 Distribution Inference 원문 두 편의 선택 절을 직접 읽고 취득기록·PDF·SHA를 spec_sources/primary_user_inference_20260915b에 저장했다. 기존 A07 pp.3–5도 다시 대조했다. U 설정·reference 평균·환자군집 자체를 신규성으로 삼지 않는다. 기존79편 장부의 읽기 범위는 바꾸지 않았다.
- 저장 CDI DL t1005noise를 CPU24.112초에 재계산했다. U mean AUC는 noise0 .4925/.5125, 평균5 .4900/.5150이었다. 총환자분산 중 noise 비율을 membership 신호 대비 noise로 해석하지 않았다.
- generic t140 신규7noise를 고정해2,240F/0B·236.278초에 실행했다. 원형320cell 포함2,560cell의 raw를 보존했다. 평균8 U mean AUC .4425/.5550,48개 평균화 대비구간 모두0포함이다. 최초 FP32 noising 검산은 상쇄10원소에서 실패했다. GPU자료는 그대로 두고, 별도v2에서 CPU FP32 exact와 연산별 오차상한을 동시에 검사해 PASS했다. 원형실패 재현도 저장했다. 독립 담당자의192AUC·48CI·16ICC·104paired변화 재계산도 일치했다.
- MoFit은 저장8embedding × 4query × 두recipient × noise5의360F/0B/0VAE를62.644초에 계산했다. 원형 h8/null8 전체prediction을 exact복원하고 CPU9.523초에 별도 검산했다. 원noise Umean 대각선 +.00459695는 recipient −.00024678 + embedding +.00484373이었다. 새4noise 평균은 +.00029508이다. embedding에 유효한 참여 정보가 있을 가능성을 배제하지 않았고, noise별로 다시 최적화한 원형 MoFit 성능 실패로 부르지 않았다.
- 한 환자14393의 학습 기여를 직접 확인하기 위해 원형 M1을1,000단계 재현했다. 매단계loss·gradient norm과250/1,000checkpoint 전체가 exact일치했고 최종파일SHA도 ae789c32…08435로 동일했다. 이어 같은학습순서·난수·분모에서 p의8개loss슬롯만0으로 하는control을1,000단계 실행했다. 초기/43단계공통prefix가 exact일치했다. 예정4,000슬롯은 같고 실제비영기여는4,000/3,992였다.
- 두학습 실행은18:34:25–18:51:47 KST,1,041.280초(17분21.3초)였다. 8,000 root forward examples/2,000 backward calls이며 gradient checkpoint의 내부 재계산은 별도다. 독립 검산13.882초에서 체크포인트·입력1,000쌍·16개mask raw의CPU autograd/closed-form FP16gradient를 확인했다. control최종SHA는 fb71f089…f0e0d1이다. 한환자·고정난수경로의 기여제거 실험이며 population MIA/DP보장이나 외부감사자의 접근권 가정으로 바꾸지 않았다.
- 후속endpoint는 신규2,260F/0B와재사용2,204로 고정했다. 독립검산기준비가예상보다늦어져추가10분을알렸다. 수치·집계정책은결과전고정한채GPU와검산기준비를병행하도록, 미완성검산기파일hash에대한대기를제거한명시적producer v2를별도로준비했다. v1원본은보존한다. 실제평가·검산·출력벡터분해결과는다음항목에기록한다.

## 109-STAGE2-MASKED-ENDPOINT-FULL-CDI. 2026-09-15 — 학습 기여와 환자별 구별의 간극 확인

- 108의 control을 고정한 endpoint 신규2,260F/0B는256.057초에 끝났다. 독립 raw 검산53.028초, 별도 집계 감사에서 고정 입력·2,204개 재사용·656환자점수·48분포와 MoFit432/480행을 대조했다. p14393의 U mean 점수 변화는 generic8noise +.00015874, CDI-DL5noise +.01622009였다. 다른40명 평균은 각각 +.00034388/+.02018456이고,34명/33명의 변화가 양수였다. 자기 U의 비영 학습 기여 효과는 관측했지만 다른 환자에게도 변화가 퍼졌다. 단일 개입의 분포를 population FPR이나 공격 실패 검정으로 해석하지 않는다.
- 저장prediction2,232쌍의 ΔMSE=2r·δ/D+‖δ‖²/D를 CPU11.007초에 분해했다. 최대 항등식 오차5.44e−17이다. 자기U의 벡터변화RMS보다 작은 다른환자는 generic12/40, CDI7/40이었다. 손실 항의 상쇄는 있지만 숨겨진 큰 자기환자 벡터변화를 크기만으로 복원한다는 원리는 지지되지 않았다. 그림·CSV·SVG와 원시 결과 링크를 최신 보고서에 추가했다.
- DL 하나를 전체CDI 실패로 확대하지 않기 위해 control U82장의 공개코드26특징 전체를 추가 실행했다. 준비·실행·검산20–25분을 고지했고, 실제 추출은19:23:17 KST 시작,562.766초였다. 4,801F/1,439B, NO objective619회다. 원형 M1 U82와 모든 원noise를 유지했다. 독립 CPU21.528초에2,132특징·656noise·control실제3,992기여/p0/U0 및 이전controlDL410개의 입력·prediction·GPU L2 exact를 확인했다. 검산기 코드는 별도 실행 protocol에 결속했으며 GPU 전 동결이라고 소급 주장하지 않았다.
- 원래18방법과 고정DINO를 보존하고, 대상환자를 제외한 기존U79(참여39/비참여40)의 특징만으로10개 기존LR를 같은C에 다시 맞췄다. control/selection fitting·새C선택은0, 수렴경고0이다. 대상제외 정책은 새control26D 결과 전에 고정했다. 원fit80의대상포함, 과거tunedC의대상포함선택, NO27보조진단을 구분했다. U79는 공격기 직접fit 제외이며 원target의79명 특징까지 대상과 독립이라는 뜻은 아니다.
- 주4개 고정C U79의 자기U 변화는 image mean −.003424, image max +.027727, mean26 +.006649, meanmax52 +.058796이었다. 모두 다른40명의 각 사분위 범위 안이다. 세 방법에서 양의 변화가 관측됐으므로 '기존방법 모두 실패'라고 쓰지 않는다. 보조 SecMI mean/PIAN mean은 다른환자q75보다 커 모든특징에 동일 결론을 확대하지 않았다. 같은 imageLR의 두U 변화는 +.027727/−.034575로 반대여서 표준mean/max만으로 부호가 달라졌다.
- CPU분석3.332초, 독립 저장점수 감사0.378초에1,517환자점수·111분포·U79입력/스케일러·738원점수를 대조했다. 최대독립예측차3.89e−16,원점수복원최대1.11e−16이다. solver독립재학습이나 전체신경망gradient 재실행으로 부르지 않는다. 독립감사 SHA 8e0378d39778d3e40a715ce7e06dc5c0ca45e7d44207f1e73e8846918266a82b.
- 이번 요청의 다섯 GPU실행 시간 합계는2,159.024초(35분59초)다. 시작17:58부터 약1시간45분의 전체 작업에는 원문/코드 검토·준비·독립검산·기록도 포함된다. 예상 대비 실제 지연과 각 세부 남은 시간은 진행 중 고지했다. 보정/시험 분할·졸업논문K5/K10 계약은 변경하지 않았다.
- 최신 STAGE2_MEASUREMENT_AUDIT_20260915.md와 HTML, RESEARCH_FRAMEWORK, CURRENT_STATUS, 두 상태JSON을 완료 범위로 갱신했다. 현재2번·GPU작업종료이며3번의새설계 준비는 미확보다. 제한접근에서 자기환자 참여를 다른환자 동반변화와 구별할 정보와 원리를 다음질문으로 남긴다. 실행계약 없이 추가학습을 예약하거나 R을 미세조정하지 않는다.
- 최종 HTML 빌드 뒤 로컬 링크847개·PDF페이지 앵커214개를 확인했고 끊어진 링크0개였다. 문헌장부79편의 기존 읽기범위를 유지했다. 최종 프로세스 확인에서 실행 중인python은 없었으며, 두 상태JSON의current_step=2와no_running_execution을 확인했다.

## 110-STAGE2-CROSS-PATIENT-INTERVENTION. 2026-09-15 — 실마리를 두 개입으로 확장하고 기존 방법의 대응 반응을 확인

- 사용자 “이 실마리를 제대로 확장해보자, 이게 되는지 안 되는지”에 따라20:22 KST 시작했다. 2번을 유지하고 이전 한 환자에서의 효과 파급이 누구를 제거했는지에 따라 달라지는지 대조했다. 원문·설계15–20분, 이어 GPU19–21분·준비/검산 포함30–40분을 고지했다. 실제 각 단계와 남은 시간도 갱신했다. 전체 작업은21:25 KST경까지 약1시간3분이며 원문/코드 검토·출처 결속·실행·독립 검산·기록을 포함한다.
- Kandpal EMNLP2024·Zhang NeurIPS2023·Meeus2025·FACE-AUDITOR USENIX2023·RAPID CCS2024의 선택 절을 직접 대조했다. 이번 신규 PDF 확보는 Zhang/Meeus/RAPID3편이며 나머지는 기존 원문이다. 확보경로와 SHA를 spec_sources/primary_cross_intervention_20260915에 남겼다. 전체 influence matrix·U 사용자 자료·reference/원점수 결합 자체는 선행이 있음을 명시했고 기존79편 장부의 선택절75/일부3/초록1 범위를 변경하지 않았다.
- q=4092는 기존selection A 중 E2가 각4회 노출된 숫자순 첫 환자로 새 결과 전 고정했다. p=14393·원형M1·기존p control을 유지했다. 사전분석정책 SHA b0855a8b…5bc70을 q 학습 전에 동결했고 새 fitting/C선택·R수정·calibration/test 사용·AUC/CI/p-value는0이다. 공통 비교39명(A19/B20), 원selection40에q포함, p/q의각다른40명을 구분했다.
- q 최초 기여37이 기존43 checkpoint보다 앞서므로 원래 초기화부터1,000회 학습했다.8개loss 기여만0, 원순서/난수/AdamW/scaler/분모를 유지했다.20:49:02–20:57:47 KST,524.910초·4,000forward examples/1,000backward였다. q0/p8/전체3,992기여다. 독립13.189초 검산에서1,000입력·초기state·36trace·8mask 미분·실제원장을 확인했다. 원형full36checkpoint 부재를 명시했다. 최종 q checkpoint SHA d952aa55…02068.
- q control U82+E4, p control E4의 새90행 CDI26D를630.095초·5,267F/1,577B에 추출했다. NO objective677회, 기존원형/pcontrol168행 재사용이다. 추출계약 SHA41d29985…977c8은528파일을 결속한다. 원시 독립53.398초에2,340특징·720noise·두 실제FP32 fullstate·pcontrolE DL20의이전input/prediction/GPU L2 exact를 확인했다. 신경망Jacobian이나전체학습독립재실행으로 부르지 않는다.
- 새 결과 분석 전 교차검토에서 DINO의기존fit80에p가포함됐다는 metadata누락을 발견했다. 별도analyzer v2와정정addendum을동결하고v1/원정책보존, 수치함수AST동일을확인했다. 원시검산기의fingerprint메타필드/실제hash 비교도 실행전 수정했다. 실제실행/독립검산은 모두PASS였다.
- 기존원fit80/U79의18방법과고정DINO를재사용했다. CPU분석8.407초, 독립math.fsum/sigmoid검산7.456초에서1,589환자행·4,767branch점수·1,032영상확률행·73대비·333분포를확인했다. 이전1,517환자행의4,551스칼라가모두exact복원됐고독립점수최대차3.89e−16이다. 감사SHA9b81bbd3…7f491.
- U79fixed주4중meanmax52만U의두자기-교차차이를모두양수로보였다: Rp+.17249118/Rq+.03427000/K+.20676118. 나머지imageMean/imageMax/mean26은p양수/q음수다. 원fit80meanmax52와보조DLmax/PIANmean/fixedNO54도두양수여서기존전체실패나새공격필요성으로해석하지않았다. 같은U-fit주4를E에적용하면두양수는0개지만E의일부scalar에는두양수가있다.
- meanmax52에서p의큰Rp는자기변화+.05879570와q제거시변화−.11369548를함께포함한다. q h=−.03427000은공통39IQR[−.04644826,.06578309]안이고,p h=.17249118은IQR위지만다른환자최대.35354462보다작다. 대응반응은인정하되환자특이기전·환자군MIA성공은미확보다. 사전해석노트에는가산offset과영상민감도곱만으로도두R양수가가능한반례를남겼다. 교차모델을외부감사자의허용입력으로주장하지않았다.
- 두GPU실행합계1,155.006초(19분15초)다. STAGE2_CROSS_PATIENT_INTERVENTION_20260915.md/HTML·주4그림/CSV·독립해석노트·RESEARCH_FRAMEWORK·CURRENT_STATUS·비교명세/두상태JSON을갱신했다. HTML로컬링크868개·PDF페이지앵커214개에서깨진링크0이었다. 이번프로세스는모두종료됐으며current_step=2/no_running_execution이다. 다음필요근거는기존meanmax52가잡는대응반응과환자특이성/영상민감도대안의구별이고3번의새설계는아직확보하지않았다.

## 111-STAGE2-DESIGN-GAP-CLARIFICATION. 2026-09-15 — 공통 설계 가정과 논리적 예측을 먼저 구체화

- 사용자는 공통 설계 빈틈을 구체화해 논문을 만들자는 뜻이며, 모든 강한 기존 방법의 실패와 구체적인 정보 손실·인과 원인을 실험으로 먼저 확정하라고 요구한 것은 아니라고 정정했다. 이어 PP는 좁힐 당시부터 관련연구/SOTA 공격이 어떤 약점을 이용해 기존 방어를 어렵게 할지 논리적으로 충분히 검토한 뒤 실험했음을 강조했다. 이 도메인에서도 같은 연구 방식이 성립하는지는 별도로 검토해야 한다. 새 공격 벤치마크형 연구를 자동 확정하지 않는다.
- 약10분의 원문·기록 검토를 알리고 PP-Mark-v0.4/PP-Mark의 intro/method/related work/implementation/experiment/work log를 직접 읽었다. 중앙 검증·위조 위협에서 결속/score+proof 관계를 먼저 설계했고 일부 공격 harness는 후속 평가로 남았다는 근거를 확인했다. 현재 문서의 후속 SP1/architecture 수정을 처음부터 완성된 보장으로 소급하지 않았다. PP의 최종 성능·보안 증명 전체를 재검증한 것은 아니다.
- 가까운 원문 CDI/MoFit/Quantile 및 User Inference의 선택 절을 대조했다. 구체화할 후보는 환자의 학습 자료에 대한 보유 U 영상의 대표성이다. U 자체·부분 포함·적은 사진·일반 분포 차이는 이미 선행이 있으므로 최초성을 주장하지 않는다. 촬영/진료 조건이 다른 같은 환자 자료에 대한 평가의 적용 범위를 검토할 수 있으나 아직 중요한 차이·논리적 개선·신규성이 확보된 후보로 선정한 것은 아니다.
- STAGE2_DESIGN_GAP_RESET_20260915.md에 문제→구체적 기존 접근→예상 반응→설계 역할→필요한 증거를 기록했다. 단순히 미평가 축이 있다는 것과 PP 수준의 설계 예측을 구별했다. NIH cohort의 followup_no를 실제 방문 ID나 시간 간격으로 오해하지 않도록 자료 한계도 명시했다.
- AGENTS.md/RESEARCH_FRAMEWORK/CURRENT_STATUS/두 상태JSON과 HTML 안내를 최신 사용자 설명에 맞게 정정했다. 단계2를 유지하고 기존 실험·동결 계약은 보존했다. 모든 방법의 실패를 설계 선행 조건으로 둔 assistant의 해석을 제거했으며, 의미 있는 논리적 예측 없이 새 실험을 추가하지 않도록 했다. 이번 추가 GPU·모델·공격 실행은0이다.

## 112-PAPER-LEVEL-REDESIGN. 2026-09-15 — 졸논과 CVPR의 큰 목적을 작업 선택 기준으로 복구

- 사용자는 졸논·CVPR 발전·관련연구의 큰 그림이 이미 있는데 개별 실험과 작은 문제에 매몰됐다고 지적했다. 21:54 KST부터 약10–15분의 전체 설계 복구를 고지했다. 실제 정리·반영은22:10 KST경까지 약16분이다. 활성 서론, 의료 통합 전환안, RESEARCH_PURPOSE, THESIS_CVPR_ALIGNMENT, 초기01·공통실험19번, 기존 설계·선행 검토와 현재 상태를 병렬 대조했다.
- 원래 졸논의 보호 주장 타당성→재사용/재학습·효용/노출/비용→exact-model receipt→PP 출처 결속 구조를 확인했다. 활성 원고의 RQ1–RQ3와 별도 의료 통합안의 RQ1–RQ4를 구분했다. 최근 CVPR 비DP 개발 진단이 full M0/K5·K10·model-bound receipt·의료 PP 통합의 완료를 대신하지 않음을 명시했다.
- PAPER_LEVEL_REDESIGN_20260915.md/HTML에 CVPR의 우선 질문을 “보유 영상과 학습 영상이 다를 때 환자 노출을 어떻게 평가해야 보호 비교가 타당한가”로 연결했다. 사용자 목표인 U를 유지하고 새 공격·공격 벤치마크·대표성 축을 필수 또는 확정 기여로 두지 않았다. 보호 단위·관측 조건·기존 공격/보정·효용/비용의 역할과 공통 자산·분기를 정리했다.
- E와 U에서 보호 비교 순서가 달라질 수 있다는 a/b 설명용 반례를 작성했다. 실제 결과·새 정리·DP 보장 구성으로 해석하지 않으며, 구체적인 보호 방식과 기존 점수를 연결하는 추가 논리가 아직 필요함을 명시했다. 공격으로 환자 전체 참 위험이나 formal ε를 인증한다는 주장도 제외했다. 다음 작업은 그 구체적 고리를 원문·설계에서 좁히는 것이며 미시적 진단 전부를 해명하는 것이 아니다.
- 기존 CDI/MoFit·개입 결과는 재사용할 구현/개발 근거로 보존했다. 두 상태JSON의 자동 다음 작업과 open_items를 논문 주장 중심으로 정정하고, 과거 진단 미완료 항목은 이력으로 이동했다. AGENTS/RESEARCH_FRAMEWORK/CURRENT_STATUS·목적/정렬/기존 설계의 현재 안내·HTML 첫 화면도 일치시켰다. 새 GPU 실행·기존 실험 결과·동결 K5/K10 계약 변경은0이다.
- HTML을 재생성하고 새 재설계 페이지를 정적 링크 검사에 포함했다. 최종 로컬 링크898개·PDF 페이지 앵커214개에서 끊어진 링크0이었다. 검토 장부는79편/로컬PDF75편/선택절75·일부3·초록1의 기존 범위를 유지했다. 검증 결과는 spec_sources/paper_level_redesign_html_verification_20260915.json에 저장했다.

## 113-PROTECTION-OBSERVATION-DESIGN. 2026-09-15 — 실제 보호 연산과 E/U 관측을 연결한 후보 구체화

- 사용자 “우선 진행해봐”에 따라22:16 KST부터2번을 계속했다. 특정 보호 방식·기존 공격·E/U 관측의 논리를 좁히고 주장/검증 설계까지 연결하는 데30–45분을 예상한다고 고지했다. 실제 작업은22:44 KST경까지 약28분이다. 새 장기 학습·공격 실행 대신 원문·수식·실제 코드와 기존 산출물의 재사용 조건을 확인했다.
- B04/B05/B06·User Inference·FACE-AUDITOR·Hartmann·Nature 환자 위험·DP-FedEmb 및 공격별 선택 절을 대조했다. User Inference의 U 자료/DP/클리핑·저FPR 결과, FACE-AUDITOR의 DP 후 미사용 동일인 감사, Hartmann의 평가에 따른 누출 경로 차이가 이미 선행임을 확인했다. 일반적인 U 위험 잔존이나 관측 불완전성 자체는 새 기여에서 제외했다. 기존79편 장부를 모두 전면 정독했다고 승격하지 않았다.
- 후보를 clipping 순서가 만드는 방향 차이와 E/U 보호 비교의 적용 범위로 좁혔다. 같은 patient sampler의 mean(clip(g))와 clip(mean(g))를 비교용 대조로 두고 원형 record-Poisson ELS/ULS와 구별했다. clipping 계수–gradient 방향의 결합 항을 전개했고, inactive/동일계수에서는 방향 차이가 사라지는 범위를 명시했다.
- 공개 수학적 예시 g1=(4,0),g2=(0,1),C=σ=1에서 A=(.5,.5),B=(4,1)/sqrt17을 구성했다. 고정 선형Gaussian관측/q=1/FPR1%의 TPR은 E에서4.2715/9.2362%, U에서3.3899/1.8589%로 순서가 달랐다. q=0/.2/1·보조norm통제18행과 두 CDF 구현을 확인했다. 이는 학습 결과·새 DP 방법·전체 공격 반전 정리가 아닌 해석적 사례다. PNG/SVG/JSON/코드를 spec_sources/protection_observation_design_20260915에 저장했다.
- 실제 trainer는 AdamW이므로 clipping gradient를 parameter displacement와 동일시할 수 없음을 확인했다. noise 없는zero-moment첫Adamstep에서A/B최대차가2.12e-8로 줄어드는 반례도 함께 남겼다. CDI DL의 L2/MSE차이, PFAMI비율의두gradient, SecMI/Quantile의재구성/이미지별보정, MoFit최적화conditioning의추가의존성을 구분했다. 실제원형기존공격전체를단일loss기전으로설명하지않았다.
- 졸논 M1-G8은 일반group ε/K·δ기하합이며 B04/B06 tight ELS와 다른 비교군임을 실제protocol코드에서확인했다. 동결졸논계약은보존하고CVPR에서최강ELS로부르지않도록명시했다. 재회계가가중치/공격점수를바꾸지않는점과 실제sampling에맞는회계조건을유지했다.
- 후속 입력을metadata로준비했다. public-development1,816명/4,831장중3장이상596명에서기존CVPR train/aux/evaluation환자를모두제외하면50명이남는다. 고정hash로8명/24장의2E-local+1U-local역할을선정하고파일SHA를저장했다. 원자료이미지inference는0이다. 기존publiccalibration은norm만저장해방향벡터는재계산이필요하다.
- 기존model_1 step250/1000 checkpoint를CPU/weights_only=True로읽었다. 각각256Adamstate의exp_avg/exp_avg_sq1,659,904원소가모두nonzero·finite였다. 해당SHA는9be37146…278a8/ae789c32…8435로기록했다. 비DP학습state재사용가능성을확인한것이며새DP학습결과가아니다. 후속기본계산은8명×2state의144image-F/48B이고mean-loss경로추가대조는32F/16B다. 실행코드/최종계약미작성, GPU미실행이며예상후속20–35분(연산2–6분)을명시했다.
- PROTECTION_OBSERVATION_CLAIM_20260915.md/HTML·세원문/코드노트·진행틀/두상태JSON/큰설계후속/AGENTS/CURRENT_STATUS를갱신했다. 논문목표는보호선택에대한E→U평가의적용범위이며, 국소진단의양성/음성을연구전체성공/실패로확대하지않는다. 기존R수정·최종test사용·동결K5/K10변경·새GPU실행은0이다.
- 새로운보고서와원문노트를포함한정적검사에서로컬링크931개·PDF앵커221개·끊어진링크0을확인했다. 독립검토에서도Gaussian식과범위에중대한오류가없음을확인하고zero-moment진단의해석한계를보강했다. 상세검사결과는 spec_sources/protection_observation_design_20260915/html_verification.json에있다.

## 114-CLIPPING-OBSERVATION-LOCAL-CHECK. 2026-09-15 — 조건부 가설의 최소 실행과 의미 도출

- 사용자 “최소만 해봐 결과 분석하고 의미도출”에 따라23:03 KST부터 준비·실행·분석20–35분을 고지하고 수행했다. PP의 암호학적 검증 조건과 달리 이 후보는 실제 학습 현상의 전제를 확인해야 하는 조건부 가설임을 유지했다. 가까운 문헌은 User Inference·FACE-AUDITOR·ELS/ULS·CDI에 집중하는 사용자 선호도 상태에 반영했다.
- 기존 공개 fixture8명/24장과 실제 비DP model_1 step250/1000의 adapter·Adam moment를 재사용했다. 모든 환자·A/B를 같은 원상태로 복원하고 t500/generic/image별한noise/FP32로 고정했다. E2는 이번 국소 update 제공, U1은 미제공 역할이며, 실제 환자군 membership 성능과 다르다. DPnoise=0, 분모1의 patient-sampled bridge로 원형 ELS/ULS·DP학습을 주장하지 않았다.
- 첫 실행 C=.2844870061은 기존 공개 image p80이다. E norm .051694–.171827로16조건 모두비활성이었고 A/Bgradient·실제Adam파라미터·E/Uloss차이는0이었다. 166logicalF/52B·실행기49.137초. 활성기전이시험된것이나Adam이회전을지운것으로해석하지않았다.
- 이 결과 후 사용자에게 단일 활성 작동 대조의 필요·추가 예상1–2분을 알렸다. 이미존재하는 공개calibration grid최소C=.01 하나만사용하고,첫결과/계약/코드를보존한별도posthocrun으로기록했다. gradient·latent·noise·모델상태는그대로재사용했고114F/0B·27.847초였다. C를여러번바꾸거나E/U부호를보고값을고르지않았으며, 작은C의운용타당성/효용이입증된것도아니다.
- 활성대조의16조건중3조건/2환자에서Emean과U의A/Brawloss순서가반대였다. 19787@250/1000,21195@1000으로같은환자두state를독립환자로세지않았다. 실제Adam변화의1차식은영상48중47부호가일치(global상대L2오차3.51%,개별상대오차중앙값1.47%). 688@250U부호불일치와19093@1000U약52%크기오차도기록했다. 일부반대부호사례는A/B둘다Uloss가baseline보다증가해이를“더학습/더누출”로설명하지않았다.
- 두실행합계280F/52B·76.983초,VAE24장,임시optimizer68step이며새장기학습/원checkpoint덮어쓰기는0이다. 전체FP32forward/backward는원래AMP학습의exact재현이아니다. FP64는저장prediction의MSE와내적검산에사용했다.
- 독립검산기SHA31f862f1c2b85efbd50d04e7ee523bc981d20fe9aa6669e1e386affa85c9ceac를실행전고정했다. 별도NumPy산술·CPUAdam·source체크에서각16조건PASS(16.227/17.002초). primarybaseline반복은raw가없어producerassertion범위이고activation반복raw는별도exact대조했다. 전체UNetbackward를독립재생산했다는주장을하지않았다. 별도출처보강은실행후27파일·원cachehidden·model/scheduler/C대조PASS이며사전계약으로소급하지않았다.
- CLIPPING_OBSERVATION_LOCAL_RESULTS_20260915.md/HTML에전체표·그림·실측·예측·예외·한계를정리했다. 의미는“조건부국소현상이실제모델에도나타날수있다”까지다. 실제운용보호설정에서활성조건이왜타당하고평가에얼마나중요한지가다음설계질문이다. 추가C탐색/장기학습을자동후속으로두지않았으며DP/MIA보호순위·새공격필요성·독립CVPR기여는미검증이다.
- 현재2번후보검토를유지하고새GPU실행은모두종료했다. 진행틀/두상태JSON/AGENTS/CURRENT_STATUS/HTML앞화면의준비전상태에최신결과안내를반영했으며,과거동결결과·K5/K10계약은보존했다.
- 최종HTML검사는로컬링크948개·PDF페이지앵커221개·깨진링크0이었다. 검토장부79편/로컬PDF75편은유지했다. 별도문서검토에서주요수치·범위·링크를확인했으며새실험을요구하는오류는없었다. 검산기동결은GPU실행전이아니라검산실행전이라는표현을명확히했다.

## 115-PATIENT-GAP-WORK-INVENTORY. 2026-09-16 — 원래 질문을 위한 조사·비교·설계의 완료와 미완료 재정리

- 사용자가 보유사진과학습사진의간극을위한선행분석/해결설계가실제로어디까지됐는지물었다. 11:14 KST부터기존기록대조5–8분을고지했고새실험/문헌수집없이정리했다. 사용자의요청은큰목표를버리거나클리핑결과를부정하라는것이아니라,각작업이원래질문에어떻게답하는지를명확히하라는것이다.
- 가까운UserInference/FACE-AUDITOR/CDI/ELSULS의이미해결한점과가정,기존기록에있는대표성후보를확인했다. UserInference의동일사용자분포표본과대표성중요성은이미선행이며CDI의P/U참조와우리환자U는다르다. 우리adaptation의보조환자참여정답을원형CDI의요구조건처럼표현하지않았다.
- 실제E/U분할·CDI18비교·U적응회복·MoFit한환자집계·환자제거교차·R/기타진단·클리핑의증거와범위를대조했다. CDI모델2의같은U에서E-fit.36→U-fit.67은기존법적응의조건부관측이며,원형26D기반32대조의E→U차이구간은모두0포함이다. MoFit의표준mean대안과의료fusion/환자군미완,제거반응의개인특이성미확보도유지했다.
- PATIENT_GAP_WORK_INVENTORY_20260916.md/HTML에완료된가정분석/비교자산,기록된대표성·보조정보·보호평가질문,미완료인핵심빈틈→해결설계연결을분리했다. 앞선일괄적“간극특정없음”도정정했다. 실제문헌가정에서적어둔질문과단서는있지만완성된기여가아니다.
- AGENTS/진행틀/두상태JSON/CURRENT_STATUS/HTML앞화면에원래문제중심의현재우선순위를반영했다. clipping은조건부보조관측으로보존하며자동후속주제에서해제했다. 이를기각하거나모든기존방법실패를새설계의선행조건으로추가하지않았다.
- 새GPU/추가학습/추가공격0,기존동결결과와K5/K10계약변경0. 최종정적검사는959로컬링크·221PDF앵커·깨진링크0,장부79편/로컬PDF75편유지다. 검사결과는spec_sources/patient_gap_inventory_html_verification_20260916.json에저장했다.

## 116-TWO-TRACK-PAPER-DESIGNS. 2026-09-16 — 보호 설계·효율과 보호 평가 개선의 구체 후보

- 사용자는 졸논 목표 변경도 허용하며, 페이퍼 기여를 위해 1 보호 설계·효율과 2 보호 평가 개선을 모두 파고 구체적으로 연결하라고 지시했다. 기존 E/U 목표 유지 규칙보다 최신 지시가 우선한다. 11:57 KST에 첫 검토 20–30분을 고지하고, 원문·기존 기록·독립 신규성 검토를 병행했다. 새 GPU·학습·공격 실행은 없었다.
- TWO_TRACK_PAPER_DESIGNS_20260916.md와 two_track_paper_designs.html에 두 방향의 문제→원인→연산/절차→조건부 이득과 대가→가까운 강한 비교→최소 검증을 연결했다. 상세 담당 노트 2개와 독립 대조 노트를 spec_sources에 남겼다. 기존 79편 장부 밖의 추가 문헌은 이번 검토에서 확인한 것으로 표시했으며 과거에 모두 검토했다고 소급하지 않았다.
- 방향1은 사용자 내부 Monte Carlo의 원래 분산과 clipping 이후에 남는 추정 오차의 차이에 따라 추가 계산을 배분하는 후보이다. 독립 pilot 2개, boundary guard, 공개 고정 controller, fresh final draws와 마지막 user clipping을 명시했다. 최종 clipped contribution의 오차를 줄이는 것이 clipping 자체의 bias 제거 또는 영상 품질 보장은 아님을 구분했다. C=1의 radial/tangential 정확 사례와 경계 실패를 포함했다. DPDM·ELS/ULS·DP-LoRA·CARV·adaptive sampling·clipping 기하는 이미 선행이며, 실제 overhead 후 효율·AdamW·품질 전이는 미검증이다.
- 방향2는 고정된 강한 공격군의 FPR/TPR 공동 신뢰띠로 보호 위험 구간을 만들고, 회원/비회원/공유 특징 측정의 가치와 비용에 따라 계산하는 후보이다. 위험 상한 최소 leader·경쟁 하한·가상 띠 폭축소/비용·동률·주기 탐색·prefix 관측·실제 띠만 이용한 정지 규칙 v0를 작성했다. 가상 띠는 증거가 아니며, 비교 상한은 유한 동결 공격군과 명시한 모집단에 한정된다. NP-ROC, Maximin-LUCB/Racing, feedback-graph TaS-FG에 같은 자료·보정·cache를 준 강한 비교가 필요하다. 효율 우위와 실제 보호 선택 효과는 미검증이다.
- 새 직접 대조에는 CARV(2026), FSCA(2025), CCS2024 misleading defense evaluations, NP-ROC, maximin/feedback graph가 포함된다. FSCA의 90%는 해당10장·100% 직접학습자료 조건이며 U-only 성공이나 독립 patient-FPR 보장으로 확대하지 않았다. Ge et al. Neurocomputing2026은 공개 부분만 확인한 넓은 관측위험 주장 차단선이다. 추가문헌 전체 증명/전수 신규성을 확인했다고 말하지 않았다.
- 독립 최종 검토에서 주요 수식과 범위는 타당했으며, ULS 내부 고정 C/q 비교와 ELS↔ULS 원형 sampler/clipping을 유지하는 ε·δ·비용 비교를 구분하도록 문장을 고쳤다. 표준 수학 예시는 CPU 산술로 확인했다: tangential 한 번 MSE .002495322244310705, 두 번 .0012476611221553524; 고정 threshold의 비회원70명 0오탐 일측95% 상한 .041893344066888605, 해당 이항 방식의1% 경계299명. 의료 효능 실험으로 세지 않았다.
- AGENTS/RESEARCH_FRAMEWORK/CURRENT_STATUS/두 상태JSON/HTML 첫 화면을 최신 두 방향으로 갱신했다. 현재2번의 문헌 기반 설계 초안 완료이며 두 후보의 신규성·효능·CVPR 채택은 미검증이다. 1번을 먼저 정밀화하고 2번은 독립 유지한다. 기존 E/U·R·clipping·환자제거 결과와 동결 K5/K10 계약은 변경하지 않았다.
- 최종 정적 검사는 로컬링크972개·PDF앵커221개·깨진링크0, 기존장부79편/로컬PDF75편 유지였다. 초기 HTML에서 수식의 대괄호가 τ 링크로 해석된2곳은 inline code로 수정 후 통과했다. 보고서·원문 노트의 SHA와 산술 결과는 spec_sources/two_track_design_verification_20260916.json에 저장했다. 첫 검토와 기록은 약26분이었다.

## 117-TWO-TRACK-DEEPER-CHECK. 2026-09-16 — 두 초안의 직접 선행 환원·비용·사전 CPU 비교

- 사용자의 ‘더 해봐’에 따라 두 연구 목표를 유지하고 13:02 KST부터 선행·수식·bounded CPU 검토를 진행했다. 예상25–40분을 고지했으며 세 담당자가 방향1·방향2·독립 선행 대조를 병행했다. 약20분 동안 결과와 기록을 완성했다. 새 GPU·의료학습·공격 실행0이며 기존 동결 결과는 바꾸지 않았다.
- TWO_TRACK_DEEPER_CHECK_20260916.md / two_track_deeper_check.html에 최신 결론을 정리했다. 방향1의 outer-clip 방향 분산과 두 pilot plug-in은 Bollapragada Eq3.2–3.3에 상수배까지 대응한다. Xiao의 preclip 평균·clipping bias, Giles–Haji-Ali의 nonlinear nested MC, Howard–Ramdas의 sequential quantile 등 직접 선행 범위를 추가로 대조했다. 기존79편 장부에 소급하지 않았고 전체 신규성 확인이라고 주장하지 않았다.
- 방향1은 C1·κ1·pilot2·R1/2/4/8/16·예산4/8/16·7합성가족을 사전 고정하고 calibration/type8192·test/type16384로 한 번 실행했다. CPU8.041초,161행/42paired 비교. geometry21조건 중20조건은 비용을 맞춘 좋은 fixed보다 MSE가 컸고, 1조건의 약0.6% 이익은 raw와 같은 동작이었다. 이산 pilot의 분산0 오판과 Gaussian finite-R tail 위험을 확인했다. 실제 타입을 아는 oracle의 일부 개선은 기회의 진단이며 제안 효능으로 세지 않았다.
- 표준 a/r 모형의 pilot 손익분기 CV(sqrt(a))²>p/(B−p)를 도출하고 실제 유한-R clipping의 보편 조건으로 확대하지 않았다. 희귀 outlier 반례의 clipped-MSE R1=.1/R2=.185도 산술 확인했다. aggregate DP 입력오차에는 개인별 MSE 합과 다른 bias 교차항이 있다는 표준 분해를 기록했지만 새 방법이나 새 정리로 부르지 않았다.
- 방향2는 고정2보호법/3점수/2threshold+항상음성, α=.05/β=.05, 각집단최대2048/128batch, 비용100000,2분포×공유유무×3규칙×64반복을 사전 고정했다. CPU15.611초/768회,distinct IID cohort128개다. v0는4조건 모두 균등보다 비용·보류에서 불리했다. 비회원병목/공유없음 정상결정 uniform30/64,width11/64,v0 1/64, 공유있음38/37/22였다. 잘못된 결정0을 보류 성공으로 해석하지 않았다.
- v0의 FPR feasibility 변화 누락을 저장 trace와 CP 산술로 확인했다. 오탐0/128 상한.072543에서 추가128의 가상 상한은.051395, 실제 추가오탐0이면.036954라5%통과가 가능하다. 해당경우가 항상발생한다고 주장하지 않는다. 비회원병목/공유없음에서2111번의 회원양수-proxy 선택 때 모든 가능한 비회원proxy는0이었다. width는명시적adaptation이며원형M-LUCB/TaS-FG전체재현이아니다.
- 별도 root 검산코드는 실행모듈을import하지 않고 protocol/code결속2개,방향1비용·MSE161행,방향2행동61852개와최종위험/정지768회 재구성을통과했다. 담당별독립산술/trace검사도보존했다. 원문·수학에대한추가검토에서중대한오류는없었고근사수치의‘일치’를‘가깝게재현’으로명확히했다.
- 두 v0의 의료·본학습 확대와 자동κ/C/예산구제는 보류했다. 두 연구목표는 유지하며 새로운 연산/평가의 구체적인 선행 차이를 다시 연결한다. 우리초안의실패를기존SOTA전체의공통실패발견이나두방향불가능으로확대하지않는다. 현재2번이고새기여·의료효능은미확보다.
- AGENTS/RESEARCH_FRAMEWORK/CURRENT_STATUS/최초설계상단/두상태JSON/HTML첫화면을갱신했다. 최초설계와실험원본은이력으로보존했다. 최종정적검사는로컬링크1000개·PDF앵커221개·깨진링크0,장부79편/로컬PDF75편유지다. 결과SHA는spec_sources/two_track_deeper_html_verification_20260916.json에있다. 상태JSON의한국어경로가shell전달중물음표가된것은발견후실제파일명으로교정했다.

## 118-TWO-TRACK-OPERATION-REDESIGN. 2026-09-16 — 보호 학습 연산과 보호 수리 선택의 재설계

- 사용자는 앞선 두 v0의 실패를 연구 방향 불가능으로 읽지 말고 더 진행하라고 지시했다. 13:27 KST부터 현재2번의 원문·논리 설계25–40분을 고지하고 세 담당자의 병렬 검토와 root 수학·직접 선행 대조를 진행했다. 약21분에 보고서·기록 검증을 마쳤다. 이번 새 GPU·학습·공격·합성 효능 실험은0이다.
- TWO_TRACK_OPERATION_REDESIGN_20260916.md / two_track_operation_redesign.html과 방향별 상세 원문 노트2개·독립 선행 노트1개를 작성했다. 기존79편 장부 밖의 추가 검토이며 전체 정독·전수 신규성 조사로 소급하지 않았다. DP-MEPF와 IHM은 이번 확인 범위가 초록이며 상세 방법 검토와 구별했다.
- 방향1은 공개 동결 denoiser의 공간·시간 특징으로 작은 공간 공유 선형 잔차 head를 만들고, 환자별 A=E(phi phiT),B=E(phi residualT)를 joint scaling/clipping 후 일회 Gaussian 보호해 ridge로 푸는 후보다. 고정 finite quadratic 목적의 W 의존 부분과 최적해가 두 통계로 정해지는 것은 정확하다. 상수항·user 평균 목적·joint clipping의 재가중·Gaussian/PSD의 surrogate를 구분했다.
- 공개 고정 전처리·분모에서 add/remove 환자 민감도C/N0, replacement2C/N0를 명시했다. 특징/척도/ridge/private validation의 무료 선택이나 공개 기반 모델의 과거 학습 참여까지 보호한다고 주장하지 않았다. d64 head256파라미터와 보호통계2336좌표를 구분하고, 해상도 독립은 통계차원이지 forward·누적비용 전체가 아님을 명시했다.
- DP-MERF/MEPF/NTK의 일회 통계, AdaSSP/IHM의 보호 회귀, DRFM과 AISTATS2026 Where the Score Lives의 denoising 회귀가 직접 선행임을 확인했다. 독립 검토에서 same-head DP-SGD도 backbone backward0이고 같은 feature bank를 사용할 수 있다는 반론을 반영했다. DP-LoRA 대비 표현력–비용, same-head 대비 통계보호–반복학습의 이득을 분리했다. 작은 잔차 표현의 capacity, noisy Gram, 실제 생성품질·총비용·추가 기여는 미검증이다.
- 방향2는 DistillMD Eq6/7의 반대 shard teacher와 private student 입력, Soup §5.1의 sample NAF 조건을 구분했다. SELENA AppA.3 Remark7·Aerni AppB.3·Students Parrot Their Teachers §3.1/AppD.1이 상관사진·teacher/student 누출을 이미 직접 다뤄 patient split·4셀·학생누출 자체를 새 발견으로 삼지 않았다. 남긴 질문은 어떤 경로의 표준 수리에 비용을 써야 보호–효용 선택이 개선되는가이며, 독자적 기여는 방향1보다 덜 구체화됐다.
- 단순 두 shard 독립 배정에서 K장의 환자 자료를 두 teacher 중 적어도 하나가 전혀 안 볼 확률2^(1-K)를 분할 원리 설명으로 사용했다. 지정 teacher 하나의2^(-K)와 구분했고, 이를 MIA 성공률·실제 환자위험으로 해석하지 않았다. 기존 non-DP LoRA checkpoint를 DistillMD 검증자료로 전용하지 않았다.
- 공개 모델 차감/다중 checkpoint/LoRA gauge 대안은 LoRA-Leak·SeMI·두 단계 감사·PRISM/LoRA-RITE 등 직접 선행을 확인해 새 기여로 추가하지 않았다. PRISM 일부 일반GL불변 서술의 범위 차이는 기록하되 DP 붕괴나 실제 코드실패로 확대하지 않았다.
- 다음 우선 작업을 방향1의 feature·환자 통계·same-head 비교 명세와 제한된 함수족의 적응 가능성으로 좁혔다. 구현·원코드 대조 예상30–60분이며 실행시간은 짧은 forward 실측 후 산정한다. 의료 본학습이나 방향2 teacher/student 대량학습, 이전 v0 계수 구제를 자동 후속으로 두지 않았다. 두 연구 방향은 모두 유지한다.
- AGENTS/RESEARCH_FRAMEWORK/CURRENT_STATUS/두 상태JSON/이전 v0 보고서 상단/HTML 첫 화면을 갱신했다. 진행표의 오래된 E/U 고정·clipping 우선 상태를 최신 두 방향으로 바로잡았고, 이전 v0 상태는 별도 이력으로 보존했다. 현재2번, 성능·신규성·연구 성공 선언없음, 동결K5/K10변경0이다.
- 독립 최종 검토의 두 표현 수정(A/B는 절대손실 상수 제외, 분할 확률의 사건 구분)을 반영했다. 정적검사1012로컬링크·221PDF앵커·깨진링크0,79편장부/75PDF유지,두상태 핵심필드 동기화PASS. 보고서/노트/상태SHA와 검토범위는 spec_sources/two_track_operation_verification_20260916.json에 저장했다.

## 119-TRACK1-ACTUAL-CAPACITY. 2026-09-16 — 방향1 실제 구현·환자 분리 검증과 다음 보호 비교

- 사용자의 “1번 먼저, 현황 보고 후 실행”에 따라13:50 KST부터 수행했다. 시작 시 원문·수식의 설계만 완료됐고 의료 적응은 미검증이라고 보고했다. 구현30–60분을 예상했고, GPU profile 이후 본 추출6–8분·분석 포함10–15분을 별도로 고지했다. 실행 도중 사용자는 긍정 결과를 기록하며 계속 완수하라고 지시했다. 이를 방향1 지속 실행 승인으로 반영하되 유리한 결과나 채택 보장으로 해석하지 않는다.
- frozen_residual_head에 실제 특징 추출·환자 충분통계·ridge 코드와 별도 검산기를 구현했다. SD2.1 기반 모델을 adapter 없이 직접 고정하고, 마지막 출력 전320채널의 고정15차원 projection+상수와4개 시간 basis로64차원 잔차층을 만들었다. static16/time4 대조와 λ=.001을 결과 전에 고정했다. 학습80명·평가40명 각4장, 영상당8개 시점/잡음, 총3,840기록이다. 환자 분리이며 최종 calibration/test는 보존했다. 평가40명은 과거 사용 개발자료다.
- 평가 MSE는 base .17964495328020189, full64 .1740069293740371, static16 .1747876716746816이었다. 기반 대비3.13842599%, 단순층 대비 .44668042% 감소, 두 비교40/40환자다. 환자 bootstrap95%의 base-full 감소는 [.0054449851,.0058414047], static-full은 [.0007542165,.0008071606]이다. 고정 학습·noise 경로 조건부 기술 통계로 제한했다.
- 작은 함수족의 비DP denoising 적응 근거를 실제로 확인했다. static16이 전체 감소량의86.15%를 이미 얻는다는 강한 대조도 함께 기록했다. full64의 통계2336좌표/static16의200좌표와 Gram 조건수 차이를 통해 다음 질문을 DP 아래의 효용·비용으로 연결했다. 파라미터 수 차이가 있어 시간 구조의 독자적 장점·신규성을 입증한 것으로 해석하지 않는다.
- GPU profile9F/7.988초, 본 추출3841F/314.687초, 총3850F/0B, peak allocated 약3.59GiB였다. CPU 해·분석16.406초, 독립 검산20.175초다. 내부 실행기 시간이며 준비·해시·문서 포함 전체 벽시계나 타 방법 대비 속도배수로 부르지 않는다. 비DP 회귀 head3개는 실제 학습했으며 DP·생성영상·MIA 실행은 없다.
- 독립 검산은 생산자 수치 함수를 import하지 않고 raw3840기록의 시점/noise,4개 feature witness,명시적64열 A/B/Q,환자 평균,해,raw gradient/MSE,bootstrap을 재계산해114873검사 PASS했다. 선택480장·캐시·기반 모델·cohort lock·코드 해시를 결속했다. 전체UNet 재실행 검증은 아니다. 검산기의 GPU 전 허용오차 고정 문구는 사실관계를 확인해 별도 metadata correction으로 정정했다: 방법 계약은GPU전, 검산기 최종화는 추출 도중·검산전이다. 원 소스·PASS 결과는 보존했다.
- 실행원본은 code_working/_reports/frozen_residual_capacity_20260916_v1에 보존했다. 실제계약 SHA ff398ca316c80b95cf6d0b0394bba99ff8d76e0e3dc7d9c08a33c7b71b712b95, manifest SHA 9c5a8ffbcf37deb146ec2a5dd3943471e66fe1fc9abdb728b4905548a8f1ab09이다. CPU준비 계약의 prompt 표현 정정도 GPU 전 별도이력으로 남겼다.
- TRACK1_CAPACITY_PROTOCOL_20260916.md와 TRACK1_CAPACITY_RESULTS_20260916.md/HTML에 방법·전체결과·의미·한계·실측·검산을 기록하고 AGENTS/RESEARCH_FRAMEWORK/CURRENT_STATUS/두상태JSON/HTML첫화면을 갱신했다. 현재 큰 단계2의 후보 검토이며 방향1의 구현과 첫 검증이 진전한 상태다. 과거실험과 동결K5/K10은 변경하지 않았다.
- 후속 TRACK1_PATIENT_DP_COMPARISON_PLAN_20260916.md/HTML은 full64/static16 × jointSSP/same-head userDP-SGD 네 칸, 같은 finite목적·공통통계 재사용, add/remove·고정N0=80·ε8/δ1e-5·독립공개보정 조건을 구체화했다. 환자 gradient=2(A_uW-B_u)이므로 양쪽 모두 새backbone역전파0으로 비교할 수 있다. DP-SGD에 부당한 반복UNet비용을 붙이지 않는다. clipping후 목적차이·빈batchnoise·ridge위치·반복출력합성도 명시했다.
- 별도공개 quality32명×2장=64장의 원파일/cache/기존cohort분리/자료분포를 확인했다. 아직 추가512F는 실행하지 않았다. clip/scales/PSD/optimizer/반복seed의 공개선택 및 최종DP계약은 남아 있고 q.1/T500은 잠정값이다. 새학습의 실제DP효용·생성품질·강한사적회귀/DP-LoRA우위·논문기여는 미검증이다. 다음준비·구현예상30–60분이며 CPUprofile후계산시간을별도고지한다.
- 기존 이중RDP calibrate_dual을 수정없이 실행했다. ε8·δ1e-5에서 q1/T1의σ=.6376701852165034, 잠정q.1/T500의σ=1.6360149290243424이다. 두 구현의 maxε를8로 맞췄으며 SGD Opacusε=7.99543747과 Googleε=8의 차이를 보존했다. 총12.123초CPU, 초기 낮은order의 수렴경고·보수적제외·최종order비해당도 JSON에 기록했다. 잡음배율만으로 효용우위를 주장하지 않는다.
- 독립 명세검토는 fixed목적·통계재사용·환자인접성·분모·ridge·공개자료·비용공정성 범위PASS였다. 최종 구현의 공개/0초기화와 환자별 전처리독립 조건도 명시했다. DP실행PASS와 구별한다. 새3개HTML과 상태동기화 정적검사는1042로컬링크·221PDF앵커·깨진링크0이며 기존79편/75PDF를 유지했다. 전체 첫구현·실행·검산·후속명세·기록은13:50–14:32 KST경 약42분이다.

## 120-TRACK1-PATIENT-DP-COMPARISON. 2026-09-16 — 공개 보정부터 실제 DP 비교·원인 검산까지

- 사용자의 “이번엔끝까지,유리한결과” 지속요청에 따라14:38 KST부터 방향1을 이어갔다. 실질이득을 만드는 설계가 목표라고 설명하고 준비·구현·검토30–60분을 고지했다. public추출1–2분,공개profile후선택2–4분,본DP약1분,독립검산1–3분의예상을단계별로보고했다. 유리한결과만선택하거나비교조건을약화시키는승인으로해석하지않았다.
- 기존quality32명×2장의공개보정자료를새run_public.py로512기록추출했다. 기존480장/UNet/source/cache/projection은변경하지않았고새adapter없음,원400명과환자교집합0을확인했다. 실제513F/0B,47.386초·통계2.359초. 독립raw/source/epsilon/projection/A/B 검산15,202항목9.102초PASS다. 코드최종화시점을실제대로남겼고GPU전동결을소급주장하지않았다.
- 새dp_mechanisms.py에jointSSP와같은headPoisson userDP-SGD를구현했다. 환자gradient2(AW-B),add/remove고정N0,빈batch의noise/ridge,명시적randomstream,공개ridge를clip밖에한번적용하는경로를검사했다. 기존비DP수치소스를수정하지않았다. clip∞/sigma0는무잡음대조만허용했다. 내부재현seed/diagnostic과DP출력범위를구별했다.
- 공개32의16/16두fold,각train16을5회반복한80slot은공개대리자료이며80독립환자가아니다. SSP와SGD각6후보×2fold×3noise,λ=.001,공개분위수/scales/LR/floor를사전계약으로고정했다. T500/2000/8000 중actualPoisson noise0/clip해제수렴검사를먼저했다. 양head에서500은미달,2000은통과했다. 공개CV144후보+24수렴fit,39.577초,public-onlyridge2개도저장했다. source의split public32/public검사오류는후보실행전정정하고원source/계약/실패profile를보존했다.
- ε8/δ1e-5,환자추가/제거인접성·N0=80에서SSPσ=.6376701852,SGDq.1/T2000σ=2.9650482183이다. 공개선택독립검산은144trial·24수렴·최소선택·최종C/scales/LR/floor·원형두accountant를재계산해7,777항목5.095초PASS. 초기RDP低order수렴경고는보존하고최종order3.8의범위를구별했다. 검산파일명연결정정도본실행계약동결전에마쳤다.
- 그뒤고정contract SHA7feffb493e03f88e62e20910789b69d48b87a39e936cbd256c75cc86de09ad6a로학습80명·개발40명에four-cell×16noise를실행했다. 같은A/Bcache를양쪽에주고모든SGD W0=0,공개보정과다른noise/mask stream을사용했다. 실제64개DP보정층+대조20개저장,실행기20.540초,새backbone0F/0B이다. public-only와이전비DP가중치재사용대조를포함한다.
- 개발40명MSE는base .1796449533,fullDP-SGD .1743888460(2.9258%감소),staticDP-SGD .1748899546(2.6469%),fullSSP .1788939244(.4181%),staticSSP .1782926612(.7528%)였다. 각DP조건16/16base개선이며환자별16회평균도40/40개선했다. 그러나public-onlyfull .1740411864/static .1748271204를어떤DP반복도넘지못했다. 작은head의DP적응과SSP우위·사적자료추가효용을구별했다.
- SSP무잡음대조에서floor0→공개선택floor만으로static .1747881251→.1782844792,full .1740077361→.1788863391로나빠졌다. 선택floor를유지한DPnoise의추가평균차이는약8.18e-6/7.59e-6다. 공개CV에서는noisyfloor0의MSE .275/.374와음의Gram고유값이관측돼floor가필요한이유도확인됐다. “선택floor가이번손실의주요연산”까지직접대조가있으며모든SSP실패나전역가산원인분해로확대하지않았다.
- 공개전용과private80비DP최적해의MSE차이는full3.4257e-5/static3.9449e-5에불과했다. 이설정의작은private추가효용여지도정직하게기록했다. 전체학습/동일품질수백배를주장하지않고공통추출·공개보정비용을분리했다. 새생성영상·공격·본모델학습은없다.
- 독립actual검산은생산자수치함수를import하지않고모든64개DP가중치·20control·4000trace업데이트·3400환자손실을재계산해41,812항목24.970초PASS,최대W차이2.78e-16이었다. source/계약/회계/출처/평가요약을결속했다. 결과원본과독립검산은frozen_residual_patient_dp_20260916_v1에보존했다.
- 본평가전spec_sources/track1_public_reference_alternatives_20260916.md에표준대안의수식을정리했다. 공개중심SSP는n=N0에서만clip전원목적일치,publicGram+B-only는목적편향,publicW0의한번잔차갱신은오차(H0)^-1(A0-A)(W0-W*)라는정확한관계를갖는다. 공식DOPE-SGD등선행경계를인정하며새DP원리로부르지않았다. 현재네칸을변경하거나이대안의추가실험을실행하지않았다.
- TRACK1_PATIENT_DP_PROTOCOL_20260916.md/RESULTS와HTML,현재상태/AGENTS/연구틀/두상태JSON을갱신했다. 다음은공개기준잔차보호와공개초기화강한SGD의비교설계이며공개전용baseline·동일목적/편향·의미있는효용여지를유지한다. 방향1우선·큰단계2와과거동결결과·최종cal/test는보존했다.
- 최종HTML정적검사는1072개로컬링크·221개PDF앵커·깨진링크0,기존79편장부/75PDF유지,두상태핵심필드동기화PASS다. 검산자료는spec_sources/track1_patient_dp_html_state_verification_20260916.json에저장했다. 14:38–15:16 KST경약38분에구현·실행·분석·독립검산·기록을완료했다. 현재실행중작업은없으며표준공개기준대안의추가성능실험은미실행이다.

## 121-REALISTIC-RESEARCH-PLAN — 2026-09-16

- 사용자는 외부 의견 두 건을 참고해 연구질문의 타당성을 객관적으로 검토하고 현실적인 계획을 다시 세우라고 했다. 이어 기존 해결 원리를 지금 맥락에 적용해 기여를 만들 수 있는데 원리 중복만으로 배제하는지 지적했다. **새 DP 원리는 필수가 아니며, 원리의 선행 존재와 전체 적용의 기여 부재는 다르다**는 지적을 현재 계획의 우선 기준으로 반영했다.
- 직전 문제 정의 점검은 spec_sources/track1_problem_framing_review_20260916.md에 보존했다. 현재 관측된 joint-SSP floor 손실, 좁은 public/private-only MSE 차이, 새로운 문제 후보, 일반화된 선행의 빈틈을 구분했다. residual 유무에 따라 방법/평가 논문이 자동 결정된다는 해석도 배제했다.
- 이번에는 기존 실제 결과·코드·통계 schema·생성 이력과 근접 보호 연구를 대조했다. 독립 메모 세 개는 track1_replan_prior_review/feasibility/independent_20260916.md다. Ji2014 의료 로지스틱 회귀의 공개Hessian·사적gradient, DOPE-SGD, AdaSSP/BoostedAdaSSP, DP diffusion fine-tuning을 확인 범위와 함께 대조했다. Root는 Ji Methods/Algorithms/Discussion과 IHM v2의 row 인접성·§2–3 연산도 직접 읽었다. 최종학회판 재확인에 실패한 자료는 읽은 저자버전을 표시했다. 선행이 의료 diffusion·환자DP·출력 보정층의 전체 적용까지 해결했다는 결론은 내리지 않았다.
- 최종 REALISTIC_RESEARCH_PLAN_20260916.md/realistic_research_plan.html은 생성 적응 구성의 기여와 선택적인 보호 solver 추가 기여를 분리한다. 표준 public-aware solver가 잘 작동하면 활용할 수 있다. 공격 연구에서 보호 학습으로 바뀐 문제·직접 선행·주평가도 명시했고 과거 patient-FPR 기준을 현재 주평가에 혼용하지 않았다.
- 다음 한 패키지는 고정 public32:private80 환자 가중치의 pooled 비DP 참고점과 기존6모델의 sampling 연결, 총2–4시간의 작업 예산이다. 4prompt×4seed×6모델=96장과 guidance1을 제안하되 아직 실행계약은 아니다. 실제생성 hook은 없으므로 새 연결·W0/저장출력 일치 검산이 필요하다. RTX3070 8GB 및 기존 생성 실측을 참고하고 작은grid는 도메인/붕괴/조건반응 sanity에 한정한다. 사적효용·임상효용·CVPR성공 판정을 보장하지 않는다. 후속 기여/효용 명세는3–6시간 검토 예산이며 새 solver 개발은 자동 후속이 아니다.
- RESEARCH_FRAMEWORK의 낡은 DP미실행 단계표와 현재위치, AGENTS/CURRENT_STATUS/두상태JSON/HTML상단을 동기화했다. 완료된 실험객체와 current_result_report가 그대로인 것을 상태갱신에서 확인했다. 기존 private/public protocol·가중치·수치·최종cal/test는 변경하지 않았다. 이번 새 GPU·학습·공격·생성 실험0이다.
- 독립 계획 검토에서 사용자 정정과의 충돌, 부당한 자동중단, 작은생성진단 한계, 시간범위 혼선을 점검했다. HTML 정적검사1095로컬링크·221PDF앵커·깨진링크0,79편장부/75PDF 유지,두상태핵심필드동기화PASS. 근거는 spec_sources/realistic_plan_verification_20260916.json이다. 15:36:47–15:49경 KST 약13분에 계획·원문·실행가능성 검토와 기록을 마쳤다. 초기예상15–25분보다 일찍 완료했다.

## 124-SAMPLING-INTEGRATION-AND-NEGATIVE-PREVIEW — 2026-09-16

- 사용자 후속 검토를 반영해 현실 계획 §5B의 sampling 연결 패킷부터 실행했다. 보고 첫머리에 연구 결과의 좋음/불명확/나쁨을 말하라는 지침도 AGENTS에 기록했다. 시작 예상45–90분, 실제18:14–18:38경 KST 약24분이었다. 사용자 검토의 static 수치는0.01794%·39/40으로 정정했고 이전 pooled검산은 원시영상 재추출이 아닌 검증된 환자통계 재사용임을 구분했다.
- 원 소스/공개witness 독립 감사 후 신규 sampling_adapter.py,run_sampling_integration.py,verify_sampling_integration.py와 명세·기존입력·라이브러리소스26개를 계약에 결속했다. 실행전 리뷰에서 예외시hook해제,clamp전finite,preview전검산hash재확인 보완을 완료했다. 기존동결파일은수정하지않았다.
- 공개0/7/16/23의 입력복원·실제재추론에서과거base/features exact. full64 zero/public/pooled,두고정prompt와CPU seed,base/zero/public/pooled/base_restored10경로×30step을저장했다. 총325UNetF/0B,가중치전후hash동일,gradient/남은hook0. zero/restored는기반전체경로와exact였다.
- 독립CPU검산6,475개PASS,DDIM300전이는고정tolerance및FP32연산범위내일치(각CPU/GPUbitwiseexact라는뜻아님). 본실행36.285초,trajectory합계18.427초,최대allocated3.588GiB,독립검산4.499초. 소스동결이전합성101검사PASS. 이후검산이결속한파일SHA재확인후VAE6decode했다.
- **연구적결과는나쁨:** 미리정한normal/effusion각base/public/pooled6장전부흉부영상목적부적합. normal은반복패턴,effusion은분홍물체. root와독립검토자가모든이미지직접확인. MSE감소가현재구성의의료생성효용으로이어지지않았으며출력변화자체를품질개선으로부르지않았다. 96장자동확대/DPsolver탐색보류. 사적자료가원래불필요하거나head/DP전반불가능하다는결론은아니다.
- 기존training_coverage_v2/step_1000_both의LoRA M1/M2에는기본CXR형태양성대조가있음을실제grid와기록에서재확인했다. 예전base도CFG7.5에서비의료출력이었다. 다음은기존양성경로와현조건의연결(예상30–60분)로고정하며,FP16/CFG7.5/CUDA RNG/directtext와FP32/CFG1/CPU RNG/cachetext차이를하나로원인단정하지않는다. M1/M2는사적역할학습이므로진단용양성대조일뿐공개DP-safe backbone으로자동채택불가.
- TRACK1_SAMPLING_INTEGRATION_PROTOCOL/RESULTS_20260916.md 및HTML,계획§5B,framework,AGENTS,CURRENT_STATUS,두state와index를갱신했다. 현재실행중작업없음,큰단계2·방향1유지. 1119로컬링크/221PDF앵커/새22링크·이미지/깨진링크0,두state5필드및26sourcehash/검산packet결속재확인PASS. spec_sources/sampling_report_verification_20260916.json에기록했다.

## 123-POOLED-NONDP-REFERENCE — 2026-09-16

- 사용자 ‘하나씩 천천히 철저히 진행’에 따라 현재 큰 단계2·방향1의 현실 계획 §5A만 실행했다. 시작 예상20–30분을 보고했고, 실제 작업은16:21–16:36경 KST 약15분이었다. 생성·평가 패키지 전체를 완료한 것으로 세지 않는다.
- 공개32/사적80 동일 환자 가중치, static16/full64, λ=.001을 고정했다. 자료 독립 감사에서 계약23필드와 동일 projection, 환자 역할·최종평가 제외를 확인했다. 새 코드 두 파일과 입력19개 hash를 계약에 결속한 뒤 한 번 실행했으며 기존 동결 코드/계약/통계/가중치는 수정하지 않았다.
- full64 pooled MSE .1740121971240744, 공개전용 .17404118640360117 대비 상대0.0166566% 감소(37/40명), 사적전용 .1740069293740371보다 약간 높았다. static16 pooled .1747957492640288, 공개전용 대비0.0179441% 감소(39/40명). 결합 학습 목적 최적해와 개발/생성 효용 상한을 구별했다.
- 생산기와 별도의112환자 직접평균·Cholesky/삼각대입·환자별 손실 검산PASS:481개 확인, 가중치8개, 환자손실320개, paired요약6개. 기존 공개/사적대조도 복원했다. 본계산·분석·저장.091516초, 독립검산.197124초, 새로운 backbone0F/0B·생성0·DP학습0. 캐시 원시추출은 과거 검산을 재사용했다.
- 수치 개선에는 자료수 증가가 포함된다. 사적 고유 신호·실제 생성 효용·최종 일반화·논문 기여가 입증된 것으로 해석하지 않았다. 기존 DP는 private-only 목적이므로 pooled비DP와의 차이를 그대로 순수DP손실이라 부르지 않는다. 다른 λ/rho 탐색은 하지 않았다.
- TRACK1_POOLED_REFERENCE_RESULTS_20260916.md/HTML, 현실계획§5A, framework, 두state, AGENTS/CURRENT_STATUS를 갱신했다. 다음 하나는 §5B sampling 정합 검증(W0복원·저장입력 offline/online·scheduler)으로 예상45–90분이며 아직 미실행이다.
- 문서 수치·해석 별도검토PASS,1116로컬링크/221PDF앵커/새보고서12링크/깨진링크0, 두state핵심5필드동기화, 입력19hash재확인PASS. 근거는 spec_sources/pooled_reference_report_verification_20260916.json이다. 현재 실행중 작업없음.

## 122-FOLLOWUP-GATES-AND-EVALUATOR-READINESS — 2026-09-16

- 사용자가 2014 공개Hessian 선행, PDA-DPMD, 제한된공개자료 DPpretraining, 네gate, 기존RAD-DINO/BioViL-T 평가기를 포함한 후속 보고서를 제공했다. 큰 순서인 공개전용 대비 사적 추가 생성 효용→환자DP 아래 유지→품질·전체비용 비교를 채택하고 불필요한 필수조건은 수정했다. 기존원리를 활용한 의미있는 적용기여를 허용한다는 사용자 정정은 유지했다.
- 공식원문을 확인했다. PDA-DPMD(ICML2022)는 공개mirror geometry와 사용자단위 DP-FedAvg 실험까지 있으나 의료diffusion출력head 전체적용은 아니다. Bu등 NeurIPS2024는 공개초기화→DPcontinualpretraining이며 noisyHessian을공개Hessian으로바꾸는solver자체가 아니다. 고정head H=2(A+λI)는 W불변이라 깊은모델의학습중곡률설명을그대로옮길수없다. 확인범위는 public_assisted_deep_priors_20260916.md에저장했다. Root는 CVPR2026공식reviewerguidelines의SOTA미초과단독거절금지와구체적prior대조요구도확인했다.
- research_gate_review_20260916.md는public-only/pooled비DP가생성효용상한이아닌참고점,고정convexridge의warmstart와목적변경구분,ΔW안정성은필수gate아닌선택적설명분석,공유head와개인화구분,public-only보다추가효용/강한DP보다품질비용개선의두RQ를정리했다. 작은null만으로사적효용부재나평가논문성립을결론내리지않는다.
- 평가기준비누락을정정했다. dp_training/k5_evaluator.py와RAD-DINO/BioViL-T고정weights,448장실제인코딩및독립전체재실행PASS,6개산술시험이이미있었다. 이번은파일SHA·패키지메타데이터·기존report·명부확인이며모델재로딩/추론/품질점수계산을하지않았다. 준비된분포/다양성/약한alignment도구와임상인증은구분했다. 자세한근거는 generation_evaluator_readiness_20260916.md다.
- 기존448참조는현재fit80과32명,dev40과22명,cal140과59명,test140과65명,public32와11명이겹친다. 모든역할을환자단위제외하면259명남는다. 기존per-imagefeature는저장되지않아허용된새참조는재인코딩해야한다. 7조건의참조혼합과새4prompt정합,질환별BioViL한계도새명세에필요하다. 동결K5원448runner를이번연구에자동재사용하지않는다.
- 초기96장계획에서DP16heads×각1장의혼합분포문제를찾아정정했다. 방법별manifestindex0의고정head하나로16장을생성하는탐색으로제안하며최고seed선택이아니다. 향후head별충분생성→metric→DPseed간요약한다. KIDsubset50는방법당16장에불가이며기존설정최소50은통계적충분성보장이아니다. 방법간영상을합쳐표본수를채우지않는다.
- 다음패키지에평가기연결·참조분리추가30–60분을반영해총2.5–5시간으로조정했다. 과거448전체54.595초에근거한새355장추론·metric5–15분은아직계획치다. 새배치의점수는연결·탐색이지privateutilitygate통과가아니다. 두state/AGENTS/CURRENT_STATUS/RESEARCH_FRAMEWORK/HTML과현재계획§8을동기화했다. 완료된실험객체·actualresultpointer·동결코드/계약/가중치는그대로다.
- 검증:1113로컬링크/221PDF앵커/깨진링크0,79편장부/75PDF유지,두state핵심필드동기화PASS. spec_sources/gate_review_plan_verification_20260916.json에기록했다. 15:55:33–16:07경KST약12분,초기예상10–20분범위내. 이번새GPU/학습/생성/품질score실행0이며현재실행중작업은없다.
