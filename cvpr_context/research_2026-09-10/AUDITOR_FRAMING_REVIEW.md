# 제한 접근 환자 감사자 설정 — 근거와 현재 설계의 적합성 검토

2026-09-11. 사용자가 제시한 환자·외부 연구자·데이터 제공기관·규제/인증기관 설명을 공식 자료 및 현재 설계와 대조했다. 아래 권장안은 연구 동기와 가정에 대한 검토이며, 실험 계약 변경이나 방법의 성능·신규성 확인은 아니다.

[기존 설계](EXPERIMENT_DESIGN.md) · [환자 수와 U 조건의 제약](RUNTIME_REVIEW.md) · [연구 목적](RESEARCH_PURPOSE.md) · [전체 문헌 장부](index.html).

**제한된 환자 자료로 학습 참여를 추론하는 설정은 타당한 연구 동기다. 그러나 외부자=black-box, 내부 감사자=MIA 불필요, 규제기관=원자료 접근 불가라는 등식은 성립하지 않는다. 현재 방법에 맞는 설정은 학습 데이터 접근은 제한되지만 모델 가중치·gradient에는 접근할 수 있는 환자 측 또는 위임받은 검증자다.**

**1. 가장 직접적인 근거**

ICO의 AI 보안 지침은 병원 데이터로 학습한 모델에서 특정 개인의 학습 포함 여부가 추론되어 병원 방문 사실이 드러날 수 있다는 예를 든다. 또한 학습 데이터는 제공하지 않으면서 제3자에게 모델을 배포하는 white-box 환경과, 입출력 질의만 허용하는 black-box 환경을 구분한다. 모델 제공자에게도 공개 모델의 개인정보 노출 위험을 평가하도록 설명한다. 이는 의료 membership 위험과 모델/데이터 접근권의 분리를 직접 뒷받침한다. 다만 여러 의료영상의 결합, unseen same-patient 추론, 우리 방법의 성능까지 입증하는 문서는 아니다. [ICO: AI 보안과 데이터 최소화](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/how-should-we-assess-security-and-data-minimisation-in-ai/).

ICO의 AI 감사 도구도 membership inference를 방어 대상에 포함하고, 내부 점검과 외부 기술 검토를 설명한다. 따라서 공격을 사용한 보호 평가를 병원 외부인의 전유물로 둘 근거는 없다. [ICO: Information security and integrity](https://ico.org.uk/for-organisations/advice-and-services/audits/data-protection-audit-framework/toolkits/artificial-intelligence/information-security-and-integrity/).

두 문서는 확인 시점에 법률 변경에 따른 검토·갱신 안내가 있는 지침이다. 여기서는 공개된 기술적 위험 설명을 연구 동기의 근거로 사용하며, 특정 감사법이 법정 인증 수단으로 승인되었다고 해석하지 않는다.

**2. 서로 다른 세 질문**

| 질문 | 필요한 증거·평가 | MIA의 위치 |
|---|---|---|
| 특정 환자의 기록이 해당 학습에 포함되었는가? | 신뢰할 수 있고 해당 checkpoint에 연결된 manifest, provenance, 로그 | 정답 자료를 볼 수 없는 주체의 통계적 추론 수단 |
| 모델 접근자가 그 포함 사실을 얼마나 알아낼 수 있는가? | 접근권·보조자료를 고정한 공격 실험, 환자별 오탐·검출률 | 포함 정답을 아는 내부 평가자도 MIA를 실행할 이유가 있음 |
| 그 데이터 사용이 적법하고 허용 범위에 맞는가? | 적용 법률, 동의 또는 다른 처리 근거, 계약, 목적·범위 등 | MIA 점수 하나로 적법성·위법성을 판정할 수 없음 |

첫째는 데이터 사용 사실의 확인이고, 둘째는 그 사실의 외부 노출 측정이다. manifest를 안다는 이유로 둘째 질문이 사라지지 않는다. 내부 연구자가 정답을 보유하되 공격 코드에 제공하지 않는 실험은 일관된 설정이다. 연구자의 정답 접근과 모의 공격자의 정보 접근을 분리하면 된다.

반대로 전체 raw dataset을 못 보는 상황이라도 후보 기록에 대한 조회, 제한된 로그 검토, 모델과 연결된 출처 증빙 같은 경로가 있을 수 있다. 원자료 비공개만으로 MIA가 유일한 검증법이라고 결론 내리지 않는다. 우리 연구의 강점은 제한 정보에서 남는 추론 위험과 보호 차이를 측정하는 데서 입증해야 한다.

**3. 사용자가 제시한 네 경우**

아래 적합성 평가는 자료를 바탕으로 한 연구 설계상의 판단이다. 제도나 사건이 이 연구의 구체적인 감사자를 이미 구현했다는 뜻은 아니다.

| 주체 | 타당한 부분 | 반드시 추가하거나 고칠 조건 | 현재 방법과의 적합성 |
|---|---|---|---|
| A. 환자 본인·환자에게 위임받은 검증자 | 자기 의료영상 일부를 확보할 수 있고 전체 학습 목록은 갖지 않는 상황 | 대상 모델 접근 경로, 같은 환자라는 연결, 보조 대조·보정 자료의 출처 | 공개 target/adapter 또는 허용된 weight 접근이 있으면 주된 동기로 적합 |
| B. 외부 연구자·시민단체·법률 대리인 | 후보 환자가 제공한 자료나 제한된 증거로 사용 여부를 질문할 수 있음 | 직함이 모델 접근권·자료 적법성·정답 membership 정보를 자동 부여하지 않음. black-box 여부도 별도 조건 | 환자 측 검증 역할로 묶을 수 있으나 기술 접근 조건을 먼저 고정 |
| C. 데이터 제공기관 | 다기관 협업에서 타 기관 원자료를 갖지 않을 수 있음 | 자기 기관의 학습 참여는 로컬 로그로 이미 알 수도 있음. 타 기관 자료를 모른다는 사실만으로 자기 환자 MIA가 필요한 것은 아님 | 자기 환자의 노출 평가, 불명확한 후속 모델/재사용 등 추가 질문이 있어야 설득력 있음 |
| D1. 제3자 인증·평가기관 | 계약에 따라 정해진 제출물·실행 환경으로 평가 가능 | 어떤 인증·평가 업무인지, 무엇을 제공받고 무엇을 제공받지 않는지 명시 | 제한된 데이터+모델 접근권을 합의한 평가라는 조건에서 가능 |
| D2. 규제기관 | 문서·분석자료·외부 감사보고서도 감독에 사용 가능 | 규제기관의 법적 조사권한과 실제 확보 자료를 구분. 원자료를 요구할 수 없는 기관으로 일반화 금지 | 가능한 응용 맥락이며 현재 공격법의 유일한 주된 사용자로 두기에는 근거 부족 |

HHS는 지정 기록 집합에 포함된 본인의 X-ray 등 진단영상에 대한 접근권을 명시한다. 이는 환자가 후보 영상을 갖는 동기를 뒷받침한다. 해당 권리 자체가 전체 병원 학습셋, 학습 manifest, 모델 weights/gradient에 대한 접근권을 부여하는 근거는 아니다. [HHS: 진단영상 접근권 FAQ](https://www.hhs.gov/hipaa/for-professionals/faq/2059/do-individuals-have-a-right-under-hipaa-to-get-copies/index.html).

EXAM은 20개 기관이 참여한 의료 federated learning 연구다. 다기관 협력의 배경 근거로 쓸 수 있지만 기관별 membership 지식이나 모델 접근권을 모두 규정해 주지는 않는다. 이 검토에서는 공식 검색에 노출된 초록과 공개 요약을 확인했으며, Nature/PMC 본문 접근 제한 때문에 이번에 EXAM 전문을 새로 정독했다고 표시하지 않는다. FL 자체를 익명성이나 DP의 보장으로 표현하지 않는다. [EXAM, Nature Medicine 2021](https://www.nature.com/articles/s41591-021-01506-3).

**4. HIPAA와 병원 내부자의 역할**

HIPAA minimum necessary는 적용되는 PHI 사용·공개·요청을 목적상 필요한 범위로 제한하는 원칙이다. 전체 기록이 필요한 경우의 정책과 정당화도 설명한다. 그러나 환자 본인에 대한 공개, 개인의 authorization에 따른 사용·공개, 치료 목적의 일정 공개·요청, HHS 집행에 필요한 공개, 다른 법률이 요구하는 사용·공개 등에는 예외가 있다. 따라서 “HIPAA 때문에 감독기관은 전체 원자료를 볼 수 없어서 MIA를 한다”는 추론은 지지되지 않는다. [HHS: Minimum Necessary Requirement](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html).

데이터 보유 조직이 접근을 통제해야 한다는 점과 내부 감사자가 은폐·방어를 목적으로 한다는 주장은 다르다. IIA의 현행 기준 설명은 내부 감사의 독립성과 객관적인 평가를 강조한다. 조직의 데이터 관리 책임, 모델 개발자, 내부 감사 기능을 동일 인물·동일 목적의 역할로 합치지 않는다. [IIA: Global Internal Audit Standards](https://www.theiia.org/en/standards/).

FDA의 clinical data 제출 안내는 관측치/대상자 단위 분석 데이터, 메타데이터, 통계 분석 코드를 통해 분석을 재현하는 구조를 설명한다. 집계 통계만 제출한다는 뜻도, raw training set을 감독기관이 요청할 수 없다는 뜻도 아니다. 의료기기 임상 심사 자료와 생성모델의 학습 membership 감사 자료를 동일시하지 않는다. [FDA: Clinical Data for Premarket Submissions](https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/clinical-data-premarket-submissions).

ICO는 공급업체와의 관계·문서화, 개발 정보 요청, 독립 평가도 설명한다. 이는 협조적 외부 검증의 동기를 보완하지만, 모든 외부기관이 제한 자료만으로 환자 MIA를 수행한다는 증거는 아니다. [ICO: Contracts and third parties](https://ico.org.uk/for-organisations/advice-and-services/audits/data-protection-audit-framework/toolkits/artificial-intelligence/contracts-and-third-parties/).

**5. 실제 사건은 어디까지 인용할 수 있는가**

| 자료 | 직접 확인되는 내용 | 이 연구에서의 사용 한계 |
|---|---|---|
| Royal Free–DeepMind / ICO | 약 160만 명 자료의 처리와 투명성 문제가 조사되었고 제3자 감사가 요구됨 | 외부 감독의 배경 사례. 생성모델 학습이나 MIA 성공 사례가 아님 |
| Streams에 대한 개발사 설명 | 당시 Streams는 AI를 사용하지 않았고 해당 Royal Free 협력에서 AI 연구를 진행하지 않았다고 설명 | 의료 생성 AI 학습 사건으로 쓰면 사실관계가 틀림 |
| Prismall v Google/DeepMind | 환자 대표 청구와 2024-12-11 항소 기각 판결 확인 | 소송 존재가 MIA의 유효성·증거력·성공을 입증하지 않음. 대표소송 요건 판단을 모든 개인 피해 부정으로 확대하지 않음 |
| GoodRx / FTC | 건강정보의 광고 목적 제공 및 통지 문제에 대한 집행 조치 | 건강정보 사용의 외부 감독 배경. 학습 membership을 공격으로 발견한 사례가 아님 |

Royal Free 사건은 [ICO의 사건 설명](https://ico.org.uk/for-the-public/ico-40/google-deepmind-and-class-action-lawsuit/)과 [2017 undertaking의 제3자 감사 조항](https://ico.org.uk/media/action-weve-taken/undertakings/2014352/royal-free-undertaking-03072017.pdf)을 대조했다. undertaking PDF 직접 열기는 실패했으며 공식 검색에 노출된 8쪽 제5항을 확인한 범위다. 전체 원문을 이번에 모두 읽었다고 표시하지 않는다.

Streams의 당시 기술적 성격은 [DeepMind: Why doesn't Streams use AI?](https://deepmind.google/blog/why-doesnt-streams-use-ai/)에서 확인했다. Prismall은 [2024-12-11 항소심 판결, 문단 86](https://www.judiciary.uk/wp-content/uploads/2024/12/Prismall-v-Google-UK-Ltd-Approved-judgment-11.12.24.pdf)을 확인했다. 이 날짜의 판단을 명시하며 이후 모든 절차를 전수 추적했다는 주장은 아니다. GoodRx는 [FTC의 2023-02-01 집행 발표](https://www.ftc.gov/news-events/news/press-releases/2023/02/ftc-enforcement-action-bar-goodrx-sharing-consumers-sensitive-health-info-advertising)를 확인했다.

미국 HIPAA/FDA와 영국 ICO 사례는 서로 다른 제도적 맥락이다. 이들을 합쳐 한국 의료기관에 적용되는 하나의 권한 규칙으로 제시하지 않는다.

**6. 현재 설계와 맞추면 반드시 드러나는 네 조건**

기존 설계의 제2절·제3절·제5절을 다시 읽었다. 다음은 새로 수행한 공격 결과가 아니라 문서에 있는 요구사항의 대조다.

| 항목 | 현재 설계의 실제 조건 | 소개문에 반영할 사항 |
|---|---|---|
| 모델 접근 | SD2.1 target/LoRA와 공개 base, denoising loss, conditioning에 대한 gradient | 외부 검증자라도 white-box임을 명시. 완성 이미지 출력만 주는 API로 동일 방법을 수행한다고 쓰지 않음 |
| 후보 자료 | 여러 영상 및 동일 환자에 속한다는 연결 | 이름 재식별 공격이 아님. 익명 환자 연결이 주어진 membership 추론 |
| 보조 정보 | 다른 환자의 공개 대조 bank, encoder, 학습형 조합의 개발 label, nonmember calibration | “자기 사진만 있으면 된다”는 표현은 현재 방법과 불일치 |
| 추론 대상 | 특정 추가 미세조정 manifest의 환자 포함 여부 | base 사전학습까지 통틀어 “그 사람이 어디서도 사용되지 않았다/되었다”로 확대하지 않음 |

대조 bank가 public이라는 사실만으로 target nonmember가 되는 것은 아니다. 특히 “새로 촬영한 영상”은 image nonmember를 보장할 수 있어도 환자의 과거 영상이 학습되지 않았다는 보장은 아니다. 보정 환자가 실제로 nonmember임을 누가 어떤 근거로 확인하는지 명세해야 한다.

연구자가 target 정답 manifest로 평가 label을 만드는 것과 실제 감사자에게 정답이 알려진 보정 자료를 제공하는 것은 별개의 정보 제공이다. 현재 설계에서 알려진 nonmember 보정을 사용한다면 그 정보 조건을 솔직하게 유지하고, 그러한 자료를 전혀 확보하지 못한 외부 사용자의 운영 오탐률까지 보장하지 않는다. shadow/공개 개발 자료의 보정값도 target과 모집단이 달라지면 같은 보장으로 자동 이전되지 않는다.

따라서 가장 일관된 초기 설정은 후보 영상만 있는 일반 환자가 계산을 모두 직접 수행하는 서비스보다, 정해진 모델·보조자료 접근권을 가진 환자 측 기술 검증자다. “환자 측”은 목적과 대표 관계, “white-box”는 모델 접근, “제한 자료”는 데이터 접근을 설명한다. 이 세 축을 하나의 직함으로 대신하지 않는다.

**7. E/U/P, 이미지 MIA, 복제 검출**

특정 환자의 기록 집합을 X_p, 대상 미세조정의 학습 목록을 D, 감사자가 가진 실제 영상 집합을 C_p라고 하자. 환자 membership은 X_p와 D가 하나 이상 겹치는 경우다. 각 후보 영상의 image membership과 후보 중 학습 영상 수 r_overlap을 별도로 기록한다.

| 평가 상황 | 후보 영상과 실제 학습의 관계 | 환자 label | 허용되는 결론 |
|---|---|---|---|
| E | 후보가 모두 학습 기록 | member | 알려진 학습 영상 여러 장을 이용한 환자 판정 |
| U | 후보는 모두 미사용이며 같은 환자의 다른 기록만 학습 | member | 독립 촬영의 다른 영상으로도 참여를 추론할 수 있는지 검증 |
| P | 후보의 일부만 학습 | member | 겹침 비율이 불완전한 조건의 판정 |
| N | 해당 환자의 기록이 대상 학습에 없음 | nonmember | 실제 오탐을 계산하는 음성 환자 |

E의 성능도 유효한 patient-level 평가다. 다만 영상별 공격을 집계한 결과와 구별되는 기여를 보여야 하고, E만으로 U를 해결했다고 할 수 없다. 외부자의 미사용 영상에서도 참여 흔적을 찾는다고 주장하려면 U가 핵심 검증이 된다. U만이 현실적이라고 단정할 필요도 없다. 환자가 학습에 쓰인 원본을 보유하는 E/P도 가능하다.

U에서 남는 신호는 질환·병원·장비·촬영 수·중복·환자 identity 특성만으로 설명되지 않는지 대조한다. 사진이 같은 환자의 것임을 알아본 사실과 그 환자의 데이터가 해당 미세조정에 쓰였음을 알아낸 사실은 다르다. 동일 조건의 비참여 환자, 강한 영상 점수 집계, 다른 환자 대조, base 대비 변화와 참여를 바꾼 통제 모델의 역할을 유지한다.

synthetic-real copy 탐지는 생성 결과와 실제 영상의 복제·근접 중복 관계를 묻는 별도 평가다. 환자 MIA 성공이 복제 이미지의 존재를 뜻하지 않으며 유사한 합성 이미지 하나로 환자 참여를 확정하지 않는다. 별도 복제 검사와 출처·중복 통제가 필요하다.

기존 시간 검토에서 확인한 고정 K10 member cohort 중 cap 밖에 U 두 장/다섯 장을 남길 수 있는 메타데이터 후보는 각각 58명/38명이다. 파일 확보·중복 검증 전 상한이다. U를 주된 결론으로 선택하면 학습 전에 독립 촬영을 남기는 분할과 충분한 환자 수를 먼저 확정해야 한다. 이번 글 검토만으로 그 분할이나 새로운 학습을 확정하지 않는다.

**8. 논문 소개와 기여를 어떻게 좁힐지**

권장하는 소개문 초안:

> 본 연구는 전체 미세조정 학습 목록에 접근하지 못하는 환자 측 검증자가, 동일 환자의 복수 후보 영상과 공개되거나 검증 목적으로 제공된 생성모델을 이용해 환자 단위 학습 참여를 통계적으로 추론하는 문제를 다룬다. 주 설정은 모델 가중치와 입력 gradient에 접근 가능한 white-box 환경이며, 필요한 대조·개발·보정 자료와 계산 예산을 명시한다. 후보 영상의 학습 겹침을 구분하고, 동일한 환자 오탐률에서 공격 성능과 이미지 단위·환자 단위 DP의 보호·효용·비용을 비교한다.

영문 초안:

> We study patient-level membership inference for a specified fine-tuning stage under limited access to training records. A patient-side verifier observes multiple linked candidate images and has white-box access to the released or audit-accessible generator and its public base model. We explicitly specify auxiliary reference, development, and calibration data, distinguish overlap with fine-tuning records from unseen same-patient observations, and compare privacy protections under matched patient-level false-positive rates and computational budgets.

의심 사례 탐색·추가 조사에 사용할 통계적 증거와 특정 개인에 대한 법적 입증을 구분한다. 공격 성공은 무단 사용의 자동 판결이 아니고, 실패는 미사용·무유출·DP 보장의 증명이 아니다. 연구의 DP 보장은 보호 단위·메커니즘·회계에서 다루며 공격 결과는 명시된 조건에서 관측한 위험으로 다룬다.

이 역할 정의는 연구 동기의 타당성을 높인다. CVPR 기여는 별도로 입증해야 한다. 현재 후보에서는 동일한 정보와 계산량을 가진 강한 image-level 집계·집합/최적화 공격보다 여러 영상 사이의 반응이 추가 정보를 주는지, 그 차이가 보호 방법의 평가를 실제로 바꾸는지가 기여 후보다. 외부 감사자라는 명칭이나 HIPAA 인용 자체를 방법의 신규성으로 삼지 않는다.

우선순위 제안은 (1) 현재 white-box·보조정보 조건을 정확히 명세, (2) E에서 기본 신호를 확인하되 U를 주장할지 결정, (3) 주장할 U가 학습 전 분할에 반영되었는지 확인, (4) 동일 조건의 강한 비교와 환자 DP 보호 평가다. 규제기관 전용 인증 도구, API-only 공격, 전체 학습 역사 추론은 현재 설정에서 자동으로 따라오는 결과가 아니다.

