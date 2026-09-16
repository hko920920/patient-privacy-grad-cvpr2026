"""Review existing design/runtime evidence; compute planning scenarios, never launch ML jobs."""
from pathlib import Path
import csv, json, hashlib, math
from html import escape
import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent.parent
INPUT = json.loads(r'''{"methods":[{"id":"loss","name":"고정 denoising loss","seconds":[1,2,4],"calls":"20 F/장 (K10 기본); m=2에서 40 F","role":"기본"},{"id":"secmi","name":"SecMI","seconds":[0.5,1.5,3],"calls":"논문 stat 설정 약 12 NFE/장; SD2.1 adapter에 따라 확정","role":"기본"},{"id":"clid","name":"CLiD","seconds":[1,2,5],"calls":"경제형 설정 약 15 queries/장; prompt 정보 일치","role":"직접"},{"id":"pfami","name":"PFAMI","seconds":[2,6,15],"calls":"20/110 queries/장 설정; 시간표는 높은 예산도 포함하는 가정","role":"직접"},{"id":"gradient","name":"GSA1/2·Tracing 묶음","seconds":[6,10,30],"calls":"선택 timestep의 F/B와 gradient feature; 세 방법의 가족 전체 예산","role":"직접"},{"id":"cdi","name":"CDI 환자 확장","seconds":[5,12,30],"calls":"multiple loss 50 F/장 + gradient masking 20 F/10 B + 기타 score + L-BFGS 호출","role":"직접"},{"id":"ours","name":"교차 사진 probe","seconds":[15,30,60],"calls":"m=2: 312 F + 72 B / 환자","role":"후보"},{"id":"mofit_budget","name":"MoFit 제한 예산 + 교차 사진 적용","seconds":[30,60,120],"calls":"원형과 별도인 제한 예산 비교; 원형 소규모 비교는 별도 계상","role":"직접"}],"stages":[{"id":"setup","name":"Spark 이식·환경/재시작 확인","hours":[4,8,16]},{"id":"D1","name":"새 feature/probe 구현 점검","hours":[2,6,12]},{"id":"D2_train","name":"공개 비DP 모델 2개 × 1000 steps","hours":[0.5,1,2]},{"id":"D2_pilot","name":"256명 × 2모델 첫 진단","hours":[3.631111111111111,7.6177777777777775,15.662222222222223]},{"id":"D3_baselines","name":"200명 × 2모델 baseline 개발 평가 1회","hours":[6.722222222222222,13.722222222222221,29.666666666666668]},{"id":"D3_ablations","name":"개발 제거 실험·공정성 진단","hours":[8,16,40]},{"id":"D3_mofit_full","name":"MoFit 원형 800 이미지 평가","hours":[100,200,400]},{"id":"K10_core","name":"K10 핵심 12회 학습","hours":[36,48,60]},{"id":"K10_extra","name":"A0/A1/A2 + P4 각 1회 예산","hours":[11,15,19]},{"id":"K10_quality","name":"소규모 품질 생성·특징·검증","hours":[3,6,12]},{"id":"K10_mia","name":"기존 기본 MIA·B0 및 추가모델","hours":[16,40,80]},{"id":"K10_extraction_gen","name":"17모델 ×2450 추출용 생성","hours":[9.255555555555556,12.555623888888888,23.13888888888889]},{"id":"K10_extraction_filter","name":"추출 대조·검색·후보 확인","hours":[6,12,30]},{"id":"D4_E2","name":"E m=2 1730명 ×12모델 전체 비교","hours":[348.8833333333333,712.1833333333333,1539.7]},{"id":"D4_ablations","name":"최종 제거 실험 6모델·후보 2회 상당 예산","hours":[86.5,173,346]},{"id":"D4_fit","name":"Quantile·Deep Sets·moments fitting","hours":[2,6,12]},{"id":"D5_utility","name":"5천장/모델 ×12 생성, classifier39회","hours":[34.833333333333336,61.08733333333333,117.33333333333334]},{"id":"D5_stats","name":"통계·보호 비교 보조 GPU","hours":[0,1,3]}],"extension":{"id":"D4_E5","name":"E m=5 608명 ×12모델 전체 비교","hours":[306.53333333333336,625.7333333333333,1352.8]}}''')
SCENARIOS = ["작업이 빠른 경우", "중앙 계획값", "작업이 느린 경우"]
DEVICES, HOURS_PER_DAY, UTILIZATION = 4, 24, 0.8

def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|"] +
        ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])

def fmt(x):
    return f"{x:,.1f}"

def main():
    protocol_path = THESIS / "code_working/_reports/nih_cxr14_dp_attack_protocol_v1_001/protocol.json"
    protocol = read_json(protocol_path)
    assert protocol["optimizer"]["max_steps"] == 4000
    cohort_dir = protocol_path.parent
    patient_path = THESIS / "code_working/_data/derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv"
    metadata = {r["patient_id"]: r for r in read_csv(patient_path)}
    cohorts = {}
    for role in ("members", "nonmembers"):
        rows = read_csv(cohort_dir / f"mia_{role}_private.csv")
        cohorts[role] = {
            "patients": len(rows),
            "capped_images": sum(int(r["image_count_k10"]) for r in rows),
            "E_m2": sum(int(r["image_count_k10"]) >= 2 for r in rows),
            "E_m5": sum(int(r["image_count_k10"]) >= 5 for r in rows),
            "metadata_raw_ge12": sum(int(metadata[r["patient_id"]]["pa_image_count"]) >= 12 for r in rows),
            "metadata_raw_ge15": sum(int(metadata[r["patient_id"]]["pa_image_count"]) >= 15 for r in rows),
        }
    n2 = sum(c["E_m2"] for c in cohorts.values())
    n5 = sum(c["E_m5"] for c in cohorts.values())
    assert (n2,n5) == (1730,608), "Update planning matrix before using changed cohorts"
    dry_path = THESIS / "code_working/_reports/nih_cxr14_k5_private_research_dryrun_v1_001/public_report.json"
    dry = read_json(dry_path)
    measured = {k:{"steps":v["completed_steps"],"seconds":v["elapsed_seconds"],
                  "naive_4000_step_hours":v["elapsed_seconds"]/v["completed_steps"]*4000/3600}
                for k,v in dry["arms"].items()}
    spec = read_json(ROOT / "experiment_design.json")
    assert spec["cost_before_caching"]["forward_per_observed_image"] == 156
    assert spec["cost_before_caching"]["backward_per_observed_image"] == 36
    methods = INPUT["methods"]
    stages = INPUT["stages"]
    extension = INPUT["extension"]
    for method in methods:
        method["status"] = "PLANNING_RATE_ASSUMPTION_NOT_SPARK_BENCHMARK"
        method["hours_E2"] = [n2*12*x/3600 for x in method["seconds"]]
        method["hours_E5"] = [n5*12*2.5*x/3600 for x in method["seconds"]]
    totals = [sum(r["hours"][i] for r in stages) for i in range(3)]
    full = [totals[i]+extension["hours"][i] for i in range(3)]
    calc = dict(
        schema="patient-audit-runtime-review/v1",
        date="2026-09-11",
        scope="LOCAL_DOCUMENT_CODE_METADATA_REVIEW_AND_BUDGET_ARITHMETIC",
        experiment_executed=False, remote_access_used=False,
        training_contract_modified=False, patient_identifiers_emitted=False,
        spark_access="User reports four units exist; simultaneous reservation is unconfirmed",
        assumptions=dict(devices=DEVICES,hours_per_day=HOURS_PER_DAY,utilization=UTILIZATION,
          spark_vs_3070_training_speed_assumed=1.0,
          m2_models=12,primary_model_runs=12,extra_model_runs_for_budget=4,
          P4_seed_count_for_budget_only=1,P2_training_count=0,
          B0_scored_once=True,m5_rate_scaling=2.5,
          pilot_patients=256,pilot_models=2,strong_pilot_patients=200,strong_pilot_models=2,
          mofit_original_pilot_images=800,mofit_minutes_per_image_assumed=[7.5,15,30],
          synthetic_images_per_generator=5000,classifier_fit_count=39,
          fitting_and_calibration_patients_are_additional_to_final_test=True,
          rates_include_processed_calibration_patients_in_cohort=True,
          unimplemented_method_ranges_are_not_confidence_intervals=True),
        measured_3070_dryrun=measured,
        measured_3070_generation_seconds_per_image=1.08524,
        cohort_metadata=cohorts,methods=methods,stages=stages,extension=extension,
        spark_device_hours_core=totals,spark_device_hours_with_m5=full,
        pool_equivalent_days_core=[h/(DEVICES*HOURS_PER_DAY*UTILIZATION) for h in totals],
        pool_equivalent_days_with_m5=[h/(DEVICES*HOURS_PER_DAY*UTILIZATION) for h in full],
        separate_3070_K5_hours=[15,20,26],
        exclusions=[
          "Full original MoFit over every final patient/model; only 800-image original-method study budgeted",
          "Scenario U/P confirmatory training and data acquisition",
          "Second dataset/backbone, P512, new protection algorithm, broad HPO and repeated redesign",
          "Dissertation-only receipt, PP-Mark and formal release reruns",
          "Unspecified additional labeled shadow models needed by a comparator's original protocol"],
        limitations=[
          "Every Spark rate is an assumption; no Spark or new attack runtime measured.",
          "Pool-equivalent days are workload/capacity, not a dependency-aware completion forecast.",
          "K5 stays on existing 3070 runner; Spark model training requires a separately reviewed port.",
          "Cached rows may be reused only with identical model/input/prompt/noise/score contracts.",
          "Final method comparison and early stopping must be fixed before final results."
        ])
    sources = [
        ROOT/"EXPERIMENT_DESIGN.md", ROOT/"experiment_design.json",ROOT/"design_feasibility.json",
        ROOT/"comparison_plan.json",ROOT/"BASELINE_PREPARATION.md",
        protocol_path,dry_path,patient_path,
        cohort_dir/"mia_members_private.csv",cohort_dir/"mia_nonmembers_private.csv",
        THESIS/"code_working/XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md",
        THESIS/"code_working/dp_training/k5_full_runner.py",
        ROOT/"code_sources/X01/conf/attack/cdi.yaml",
        ROOT/"code_sources/X01/conf/attack/noise_optim.yaml",
        ROOT/"code_sources/X01/conf/attack/multiple_loss.yaml",
        ROOT/"code_sources/X12/COCO/MoFit_COCO.py",
    ]
    calc["conditional_U_m2_matrix"] = {
        "status": "NEW_SPLIT_AND_TRAINING_PROPOSAL_NOT_FROZEN",
        "patients_assumed": 1730, "models_assumed": 12,
        "new_training_hours": [36,48,60], "setup_and_evaluation_overhead_hours": [20,40,80],
        "all_method_evaluation_hours": next(s["hours"] for s in stages if s["id"] == "D4_E2"),
        "additional_total_hours": [next(s["hours"][i] for s in stages if s["id"] == "D4_E2")+[36,48,60][i]+[20,40,80][i] for i in range(3)],
        "eligibility_and_power_verified": False
    }
    calc["source_sha256"] = {str(p.relative_to(THESIS)).replace("\\","/"):digest(p) for p in sources}
    (ROOT/"runtime_budget.json").write_text(json.dumps(calc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    training_rows = [
      ["M0",1,"1–3","추정. 비DP batch 8; 장시간 완료 기록 없음"],
      ["M1-I8",1,"4–6",f"4-step {measured['M1-I8']['seconds']:.3f}초; 단순 환산 4.88시간"],
      ["M1-G8",1,"4–6",f"4-step {measured['M1-G8']['seconds']:.3f}초; 단순 환산 4.84시간"],
      ["M2-P8",1,"3–5",f"4-step {measured['M2-P8']['seconds']:.3f}초; 단순 환산 2.21시간을 넓혀 예약"],
      ["K5 학습 합계",4,"12–20","기존 3070 실행 경로의 계획 범위"],
      ["K5 준비·생성·검증",1,"3–6","B0 448장 기완료; 새 arm 생성·평가 및 여유"],
      ["K5 전체",4,"15–26","현재 실행 순서 유지; Spark 4대로 나누지 않음"],
      ["K10 핵심",12,"36–60","위 네 조건 ×3 독립 seed ×4000 steps"],
      ["K10 A0/A1/A2",3,"8–14","각 1 seed; 기전 대조는 유지"],
      ["K10 P4",1,"3–5","시간 산정용 1 seed 가정; 최종 반복 수 미고정"],
      ["K10 학습 합계",16,"47–79","P2는 회계 곡선으로 계상, 학습 추가하지 않음"],
    ]
    method_rows = []
    for m in methods:
        method_rows.append([m["name"],m["calls"],
          " / ".join(fmt(x) for x in m["seconds"]),
          " / ".join(fmt(x) for x in m["hours_E2"]),
          fmt(m["hours_E2"][1]/3.2)])
    stage_rows = [[s["id"],s["name"]," / ".join(fmt(x) for x in s["hours"]),fmt(s["hours"][1]/3.2)] for s in stages]
    stage_rows.append([extension["id"],extension["name"]," / ".join(fmt(x) for x in extension["hours"]),fmt(extension["hours"][1]/3.2)])
    scenario_rows = []
    for i,label in enumerate(SCENARIOS):
        scenario_rows.append([label,fmt(totals[i]),fmt(totals[i]/76.8),fmt(full[i]),fmt(full[i]/76.8)])
    device_rows = [[n,h,fmt(n*h*.8),fmt(totals[1]/(n*h*.8)),fmt(full[1]/(n*h*.8))]
                   for n,h in [(4,24),(4,16),(4,8),(2,24),(1,24)]]
    candidate_rows = [[sec,fmt(n2*12*sec/3600),fmt(n2*12*sec/3600/3.2),fmt(n2*12*sec/3600/76.8)]
                      for sec in (15,30,60,120)]
    md = """# 현재 설계의 실험별 시간 검토 — Spark 4대 조건

2026-09-11. 기존 설계·실행 코드·저장된 시간·환자 메타데이터를 대조한 계획 검토다. Spark 접속이나 새 학습·공격을 실행한 결과가 아니다. [설계](design.html) · [근거와 계산 JSON](runtime_budget.json) · [계산 재현 스크립트](review_runtime_budget.py).

**먼저 작은 진단으로 갈 방향을 결정하고, 그 뒤 본실험을 확정하는 순서다.** 4대를 전용으로 하루 24시간 쓰고, 한 명이 구현·분석을 집중 수행하며 큰 방향 변경이 없는 중앙 시나리오에서 NIH/SD2.1 중심의 CVPR 실험·원고는 약 **6–10주를 예약하는 안**이다. m=5를 모든 방법에 확장하고 처리량이 느린 시나리오면 **10–14주 이상**도 가능하다. 이는 투고 기여의 성립이나 완수 시점 보장이 아니다.

첫 **1–2주** 목표는 공개 모델에서 기본 신호와 실행 가능성을 확인하는 것이다. 강한 비교군까지 갖춘 기여 판정은 **3–4주** 정도를 먼저 배정한다. 1,000-step 약식 학습에서 신호가 없다는 사실만으로 방법을 반증하지 않고, 모델의 도메인 적응 여부를 함께 확인한다.

**1. 시간표의 전제**

- Spark 네 대의 존재는 사용자 전언이며 전용·동시 사용 시간은 미확인이다. 기본 계산은 4대 ×24시간 ×0.8 = **하루 76.8 장치시간** 처리다. 80%는 실측 효율이 아닌 작업 불균형·재시작·빈 슬롯 여유다.
- 한 대의 학습 속도는 우선 기존 3070과 같다고 둔다. 대화의 이미지 생성 1–2배 추정치를 DP 학습·gradient 공격에 적용하지 않는다.
- **장치시간**은 각 장비의 실행시간 합이다. **병렬 환산시간**은 이를 3.2로 나눈 용량 계산이다. 구현, 단계 의존성, 사람의 판정 대기는 포함되지 않아서 실제 완료일과 같지 않다.
- 아래 빠름/중앙/느림 값은 가정한 처리량과 예약 예산이다. 신뢰구간·실측 최소/최대가 아니다. 특히 새 공격, CDI, GSA/Tracing, MoFit 이식의 초/환자는 아직 측정하지 않았다.
- K5는 기존 3070의 순차 실행기를 유지하는 기본안이다. Spark는 CVPR 개발과 이식된 K10을 독립 작업으로 나눈다. [공식 Spark 이식 문서](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/index.html)는 ARM/Blackwell 환경을 설명한다. 현재 Windows DPAPI 저장·재시작과 고정 라이브러리 환경은 이식 검증이 필요하다.
- 비용 비교용 학습 batch, 환자 sampling, clipping, noise, 최종 step은 유지한다. 메모리가 늘었다는 이유로 DP의 논리적 batch를 임의로 키우지 않는다.

**2. 실제 환자 수와 아직 성립하지 않은 U 조건**

"""+table(["기존 K10 cohort","전체 환자","보유 기록 2장 이상","보유 기록 5장 이상","K10 이미지 수"],
    [[k,v["patients"],v["E_m2"],v["E_m5"],v["capped_images"]] for k,v in cohorts.items()])+"""

따라서 E/m=2에는 **865 member +865 nonmember =1,730명**, E/m=5에는 **304+304 =608명**이라는 메타데이터 상한이 있다. 이미지 접근·중복·시나리오 확인 전 숫자다. calibration으로 보류한 nonmember도 점수는 계산하므로 비용에는 포함한다. 1/3을 calibration에 쓰면 대략 m=2에서 288명, m=5에서 101명이며 정확한 hash 분할·검정력은 별도 고정한다. m=5의 1% FPR 결과는 특히 불확실성이 크다.

U는 실제 학습에 포함되지 않은 같은 환자의 기록이다. 현재 K10은 cap 안의 기록을 모두 training manifest에 둔다. 따라서 cap 밖에서 U 두 장을 얻으려면 원자료가 12장 이상, U 다섯 장이면 15장 이상 필요하다. **고정 member cohort에서 해당 환자는 각각 58명·38명**이다. 파일이 실제로 확보됐거나 중복을 제거해도 적격이라는 뜻은 아니다. 이 숫자는 U의 큰 확증 실험이 현재 checkpoint만으로 바로 가능하다는 전제를 깨뜨린다.

U를 중심 주장으로 삼으려면 공개 개발 단계에서부터 기록을 사전에 남기는 분할, 적절한 표본력, 별도 최종 training manifest 또는 추가 데이터가 필요하다. 그러면 기존 K10과 조건이 달라진 모델은 재학습 예산을 추가한다. U/P, 두 번째 dataset, P512, 새 보호 알고리즘은 아래 기본 합계에 자동 포함하지 않았다. **U가 핵심 기여라면 아래 추가 행까지 포함한 일정으로 판단해야 한다.**

| U/P를 수행하는 방식 | 추가 계산의 가정 | 추가 일정·제약 |
|---|---|---|
| 기존 K10의 cap 밖 사진만 사용 | 잠재 member58명 이하의 U/m=2 탐색 | 파일 확보·중복 검사가 선행. 큰 표본의 low-FPR 확증을 대체하지 못함 |
| 사진을 미리 남겨 둔 별도 U/m=2 최종 모델 | 4조건×3seeds 새 학습36–60시간 +1,730명×12모델의 전체 baseline 약349–1,540시간 +준비·평가20–80시간 | 합계 약405/800/1,680 장치시간(빠름/중앙/느림), 네 대의 용량상 약5.3/10.4/21.9일. 새 분할·회계·검증에5–10작업일 추가, 대략2–5주 이상 별도 예약 |
| 부분 겹침 P 한 조건 | U를 위해 만든 모델의 training/held-out 기록을 섞어 평가할 수 있으면 추가 학습 없음 | 환자×모델×선택한 방법의 초/환자로 새 평가량 계산. E/U와 동일 score 캐시가 아닌 probe는 재계산 |

U/m=2의 1,730명도 **시간 산정용 규모**다. 원자료4장 이상 private_train은1,893명으로 기록 두 장을 남기는 대안의 후보 풀이지만, exact matching·파일·독립 분할·검정력이 확인된 것은 아니다. 이 대안은 cap 밖의 추가 파일만 찾는 방식과 달리, 이미 cap 안에 확보된 기록을 미리 학습에서 제외하는 경로도 가능하다. 최종 train manifest와 noise 회계를 다시 고정해야 한다. **E와 U 모두 큰 본표로 유지하면 중앙 시나리오에서 약8–14주 이상으로 확대해 예약하는 편이 맞다.**

**3. 공통 학습을 모델별로 검토**

"""+table(["항목","학습 횟수","3070 기준 장치시간","근거"],training_rows)+"""

K5의 4-step 시간을 1,000배 한 수치가 과거 16–21시간, K10 핵심 39–48시간 추정의 출발점이었다. 그러나 Poisson sampling의 네 번 관측은 실제 이미지·환자 작업량의 장기 평균이 아니다. 이번 검토에서는 M2와 전체 범위를 넓혀 예약한다. 이는 새 장시간 실측 결과가 아니다.

K10은 cap이 커져도 고정 4,000 steps이고, M1은 기대 8장/step, M2는 약 8.02장/step이므로 dataset 장수 비율로 학습시간을 다시 곱하지 않는다. K10 핵심 36–60시간은 네 Spark에 작업을 잘 배분하면 용량상 **11.3–18.8시간**, 추가 모델까지 47–79시간은 **14.7–24.7시간**이다. 환경 준비, 입력 구성, 각 모델의 생성·검증 및 도메인 판정 때문에 실제 예약은 핵심 **1–2일**, 추가 조건 포함 **2–3일** 정도를 먼저 잡는다. 이식 완료 이후의 예상이다.

M1-I8/M1-G8의 sampling·diffusion draw는 조건을 맞추되 Gaussian stream은 독립이다. 여러 장비에서 동일한 분할·초기화 계약·stream 분리를 보존해야 한다. 한 arm을 특정 장비에만 고정하지 말고 장비 성능 차이도 기록한다. B0는 추가 학습이 없으며 동일 base 검사를 seed 수만큼 독립 모델로 반복 계산할 필요가 없다.

**4. D0–D3: 주제를 좁히는 데 필요한 실제 작업**

| 작업 | 구현·검토에 배정할 시간 | GPU 실행과 산출 |
|---|---|---|
| D0: 핵심 주장·비교 조건·표본 역할 정리 | 1–2 작업일 | E/U 중 주장을 분명히 하고 실패 기준을 적는다. GPU 거의 없음 |
| Spark 환경·코드 이식 | 2–5 작업일, D1과 일부 병행 | 한 대에서 검증 후 네 대에 같은 환경. 4–16 장치시간 예약 |
| D1: loss/base 차이·probe·비용 기록 | 3–5 작업일 | 기존 공통 점수 평가기는 재사용, 새 feature/gradient 경로는 구현. 2–12 장치시간 |
| D2: 공개 LoRA 두 개, 각 1,000 steps | 준비·해석 1–2 작업일 | 학습 0.5–2 장치시간 가정. 두 모델의 포함 집단 교환은 단일 환자 인과 실험이 아님 |
| D2: 256명 ×2모델, m=2 | 구현 완료 후 0.5–1일 | 후보+loss+SecMI+PFAMI 1회, 대조 준비 포함 약 3.6–15.7 장치시간 |
| D3: 강한 비교군 이식 | 5–10 작업일 | CDI·MoFit·CLiD·GSA/Tracing adapter와 원형/변형을 구별 |
| D3: 200명 ×2모델의 공통 baseline 비교 | 구현 후 0.5–2일 | 한 설정 6.7–29.7 장치시간. 반복 탐색 횟수만큼 추가 |
| D3: 제거·교란·선형 관계 대조 | 2–4 작업일 | 8–40 장치시간 예약. 최초에 모든 반증 대조를 수행 |
| D3: MoFit 원형의 작은 직접 비교 | 해석 1–2 작업일, 계산과 병행 | 200명 ×2장 ×2모델 =800 image-model 평가. 다음 절의 원형 예산 |

구현·개발·calibration·최종 test 환자는 분리해야 한다. 여기의 256명/200명은 시간 산정용 규모이고 이번에 실제 환자를 배정한 것이 아니다. 원형 baseline이 더 많은 별도 shadow 모델·보조 label을 요구하면 그 재현 비용을 추가하고 정보 조건을 맞춘다.

**5. D4: 주요 공격을 하나씩 계산**

E/m=2, 1,730명 ×12개 최종 모델에 한 설정씩 적용하는 예산이다. 초/환자는 해당 환자의 두 사진을 처리하는 전체 시간 가정이며 base·대조 호출을 후보 시간에 포함한다. 전처리·모델 load·보조 fitting의 공통 준비는 별도 행이다. 여러 hyperparameter나 시간 예산점을 돌리면 그 횟수를 추가한다.

"""+table(["방법","연산량 검토","초/환자 가정 (빠름/중앙/느림)","전체 장치시간 (빠름/중앙/느림)","4대 환산시간, 중앙"],method_rows)+"""

GSA/Tracing 행은 가족 전체의 예산이며 구현이 다른 세 독립 완전 재현에 그대로 적용하는 고정 수치가 아니다. 모든 module/timestep feature가 실제로 필요한지는 method contract에서 확정한다. CDI 공식 코드의 noise optimization은 max_iter=5이나 L-BFGS line search는 iteration당 forward/backward를 여러 번 부를 수 있다. 이를 단순히 5 query로 세지 않는다.

단일영상 점수의 mean/max/median/top-25%, 동일 feature의 moments/Deep Sets, threshold·bootstrap은 feature를 이미 얻었다면 주로 CPU 또는 작은 fitting 작업이다. 집계 방법 수만큼 diffusion forward 전체를 중복 실행하지 않는다. 반대로 prompt/noise/모델이 다른 졸논 score는 CVPR score와 같은 것으로 간주해 비용에서 빼지 않는다.

**새 probe만의 감도**

m=2의 312 forward +72 backward는 batching 전 model-example 연산 수다. 위 중앙 30초는 이 연산이 실제 30초에 끝났다는 뜻이 아니다.

"""+table(["가정한 초/환자","1,730명×12모델 장치시간","4대·80% 환산시간","하루24시간 환산일"],candidate_rows)+"""

m=5는 같은 구현에서 사진 수에 비례한다고 가정하면 환자당 2.5배다. 하지만 해당 cohort는 608명이므로 전체 비용은 m=2의 **약 87.9%**다. 후보의 중앙 예산은 m=2 **173.0시간**, m=5 **152.0시간**으로 합계 **325.0 장치시간**이다. 네 대·80%로 약 101.6시간, 24시간 가동 시 4.2일의 용량이다. 이것은 새 probe 한 방법의 E 두 조건만 계산한 값이다.

**6. 가장 큰 위험 항목: MoFit 원형**

저자는 특정 RTX 4090 설정에서 원형 영상당 **7–9분**, 단계별 VRAM 약 **14.8–20.7GB**를 보고한다. [논문 A.8](https://arxiv.org/html/2602.22689v1#A1.SS8). 이 값은 Spark 또는 우리 P256/LoRA의 실측이 아니다. 두 단계 early stopping을 각각 평가한 시간을 더해서 검증된 전체 runtime으로 쓰지 않는다.

현재 E/m=2 전수를 원형으로 돌리면 1,730×2×12 = **41,520 image-model 평가**다. 저자의 7–9분 속도와 같다고만 가정해도 **4,844–6,228 장치시간**, 네 대·80%·24시간 기준 **63–81일**이다. Spark에서 더 느리거나 P256에서 빨라질 가능성은 이 별도 속도 가정에 들어 있지 않다.

그래서 시간 검토용 계획은 **원형 소규모 직접 대조 + 전 표본의 제한 예산 비교**를 분리한다. 원형 소규모 800평가에 Spark에서 7.5/15/30분씩 든다고 가정해 100/200/400 장치시간을 배정했다. 이 세 속도는 **새로 둔 예약 시나리오**이며 NVIDIA/저자가 보고한 Spark 숫자가 아니다. 대조군들도 같은 작은 표본으로 평가하고 이 표에서는 정밀한 1% FPR 우월성을 주장하지 않는다.

전 표본의 제한 예산 MoFit은 원형 재현이라고 이름 붙이지 않는다. 우리 방법과 같은 wall-time·접근권한·label 예산을 비교하고, MoFit의 conditioning을 다른 사진에 적용하는 직접 확장도 포함한다. 원형이 더 강하고 그 차이가 핵심 주장에 영향을 주면 표본과 계산을 확대하거나 주장을 제한해야 한다. **원형 전수 SOTA 비교까지 완료한다는 약속은 6–10주 계획에 없다.**

**7. 제거 실험은 어떤 것을 다시 돌려야 하는가**

| 제거/대조 | 추가 계산 | 배치 |
|---|---|---|
| 자기 사진에서 탐색·평가 | support probe 재사용 가능, 자기 사진 평가 추가 | D3 전부, D4 고정 핵심 조건 |
| 다른 사진으로 전이 | 주 방법의 본 계산 | D4 비용에 포함 |
| base 차이 제거 | 최적화 목적이 달라져 probe 재탐색 필요 | 단순 score 후처리로 대체 불가 |
| 다른 환자 대조 제거 | T와 C를 별도 보관했다면 T 재집계 | 새 diffusion 계산 거의 없음 |
| 선형 gradient Gram | 고정점의 gradient 추출이 없으면 새 F/B 필요 | nonlinear probe와 비교 |
| 조건을 맞춘 다른 환자 묶음 | 바뀐 support/query에 필요한 추가 계산 | cohort·교란 조건을 유지 |
| MoFit conditioning의 교차 사진 적용 | MoFit embedding 재사용 + query 평가 | 제한 예산/원형 각각 구별 |
| Quantile/Deep Sets/moments·동일 보정 | 보조 fitting/CPU 집계 | 동일 feature이면 재추출 불필요 |

최종 제거 실험 시간은 **M0/P8 ×3 seeds =6모델**, 1,730명에 후보를 두 번 더 돌리는 정도를 예약한 계산이다. 실제 호출 수를 증명한 등식은 아니다. 약 **86.5/173/346 장치시간**의 빠름/중앙/느림 예산이다. 표본과 모델 범위는 결과를 보기 전에 고정하며, I8/G8에서도 같은 분석을 주장하려면 해당 추가 비용을 더한다. m=5 제거 실험 전수는 포함하지 않았다.

**8. 기존 졸논과 공통인 평가·D5**

| 항목 | 현재 정해진 작업량 / 시간 가정 | 중복 방지·미정 부분 |
|---|---|---|
| 기존 K10 기본 membership | 모델당 3,632명, 총 9,480장 ×20 loss cells; SecMI 추가 | 기본 표와 CVPR의 prompt/noise가 다르면 각각 평가. B0·추가모델 포함 16–80 장치시간 예약 |
| bounded extraction 생성 | B0+16학습모델 =17모델 ×2,450장 =41,650장 | 3070의 1.08524초/장과 같은 속도 가정이면 생성만 12.6시간. 빠름/느림 가정 9.3–23.1시간 |
| extraction 비교·필터 | reference 21,756 train+4,740 holdout, 공개 음성 pair 10만 이상 | 생성 외 검색/후보 검증 6–30 장치시간 예약. 모든 이미지 쌍의 SSIM을 단순 전수 계산하면 이 예산을 크게 넘을 수 있어, 필터 구현 검토가 선행 |
| 소규모 생성 품질·RAD-DINO/BioViL-T | 기존 evaluator 재사용 | 3–12 장치시간 예약, 의료적 독립 검증으로 확대 해석하지 않음 |
| downstream 효용 | 시간 산정용 5,000장/모델 ×12 =60,000장 | 생성 18.1시간(기존 속도). generator별 classifier 3 seeds + real baseline3 =39회, 회당0.5/1/2시간 가정. 전체 약34.8/61.1/117.3 장치시간 |
| 동일 환자 예산 보호·회계·재사용 비용 | G8/P8 비교, I8 단위 불일치, exact checkpoint 재분석 | 구현·분석 3–5 작업일; 같은 checkpoint 회계 대조는 새 학습이 아님 |
| 통계·CI·표·원고 | 기존 K10 10,000 patient bootstrap와 별도 CVPR calibration 분석 | 분석3–5일, 집필5–10일. GPU4대여도 사람이 검토하는 시간은 그대로 필요 |

downstream의 생성량·classifier 설정은 아직 확정 계약이 아니다. 생성량이 모델당 5천/1만/5만장이면 12모델의 생성만 약 **18.1/36.2/180.9시간**으로 변한다. 확정 전에 의료적 효용 질문과 필요한 표본량을 먼저 정한다.

**9. 전체 작업량의 합산과 네 대의 효과**

다음 표는 각 가정을 빠짐없이 합친 것이다. 중앙값은 확률적 기대값이 아니라 비교 가능한 예약안이다. 원형 MoFit 전수·U 재학습·두 번째 dataset과 방법 재설계 반복은 제외돼 있다.

"""+table(["단계","작업","장치시간 빠름/중앙/느림","4대 환산시간 중앙"],stage_rows)+"""

"""+table(["처리량 시나리오","m=2 중심 전체 장치시간","4대 환산일","m=5 본표까지 장치시간","4대 환산일"],scenario_rows)+"""

네 대에서 독립 작업이 충분할 때의 **용량 환산**이다. K5는 별도로 기존 3070에 15–26시간을 예약하며 상당 부분 병행할 수 있다. 표의 환산일에 실제 구현·분석·선행 단계의 대기를 더해야 한다. 각 단계의 일정을 4로 나누거나, 총 장치시간이 4분의 1이 됐다고 쓰지 않는다.

"""+table(["Spark 대수","하루 사용시간","하루 유효 장치시간","m=2 중심 중앙 환산일","m=5 포함 중앙 환산일"],device_rows)+"""

중앙 작업량은 m=2 중심 **약 1,339 장치시간**, m=5 본표까지 **약 1,965 장치시간**이다. 예전 400시간은 한 공격 ×2,000명 ×12모델 ×1분이라는 예시였다. 이번 수치는 여러 방법, 개발, 원형 MoFit의 작은 대조, 제거 실험, 생성·효용까지 합쳤으므로 대상 범위가 다르다. 실제 짧은 처리량을 넣으면 이 표의 각 행을 즉시 다시 계산할 수 있다.

**10. 순서를 반영한 실행 예약안**

| 기간 | 우선 산출 | 장비 사용 |
|---|---|---|
| 1주차 | 질문·E/U 범위와 data 역할, 한 Spark 환경, D1, D2 시작 | 3070은 K5 M0 도메인 검증; Spark는 개발·이식 |
| 2주차 | 작은 신호 진단, 유사 환자·선형 반응 대조 | 개발용 모델/평가를 분리 배치; K5는 기존 순서 유지 |
| 3–4주차 | 강한 baseline·MoFit 원형 소규모 대조·독립 pilot로 방법 판정 | 네 Spark에 서로 독립인 방법/환자 작업, 완료 후 최종 설정 고정 |
| 이후 2–4주 이상 | K10 공유 모델·최종 E/m=2·제거 실험·필요한 m=5 | 완료 checkpoint별로 평가 시작; 의존성 없는 작업은 빈 장비에 배치 |
| 결과가 모이는 기간~마지막 1–2주 | 보호/효용 비교·오탐/불확실성·그림·원고 | CPU 분석/집필과 GPU 작업 일부 병행 |

중앙 가정이면 전체 **6–10주**를 예약할 수 있으나, m=5까지 모든 방법을 느린 처리량으로 수행하면 계산 용량만 약 **53.5일**이다. 그 경우 구현·검토까지 **10–14주 이상**을 잡아야 한다. 하루16시간만 쓸 수 있으면 계산 기간은 24시간 사용의1.5배다. U를 주요 기여로 유지할 추가 모델·표본 설계가 필요하면 이 일정은 별도 상향해야 한다.

처음부터 네 Spark를 하나의 분산 학습으로 묶는 시간 이득을 전제하지 않는다. 현재 크기의 실험은 조건·seed·환자 batch를 독립 job으로 나누는 방식이 먼저다. 큰 모델 하나가 한 대 메모리에 들어가지 않을 때에만 별도 분산 실행 검토가 필요하다.

**11. 시간을 줄일 수 있는 곳과 첫 측정**

- D2/D3에서 가능성을 먼저 검토해 성립하지 않는 후보의 대규모 최종 실행을 피한다. 확정된 졸논 대조군은 원하는 결과가 아니라고 삭제하지 않는다.
- 같은 K10 checkpoint의 학습·생성·동일 평가를 두 원고에서 재사용한다. 통계 집계법 수, 두 회계 방식 수를 새 diffusion 실행 수로 곱하지 않는다.
- exact model/prompt/noise/전처리가 같은 base·feature 계산만 캐시한다. 환자 집합, probe, m, noise bank가 달라져 필요한 계산은 남긴다. 기존 restricted 저장 경로에 무단 영구 feature bank를 추가하지 않는다.
- 첫 Spark 검증에서 공개 개발용 각 학습 경로의 warm-up 뒤 고정 구간 step time, 후보/각 baseline의 m=2·m=5 초/환자, 최대 메모리, cold/warm load와 batch scaling을 기록한다. 예컨대 공개 환자32명과 warm-up20/measured100-step 진단은 **향후 제안**이며 이번에 실행하지 않았다.
- 시간 비교는 mean/p50/p95와 총 wall time, forward/backward 수를 함께 확인한다. 두 단계 MoFit은 전체 경로를 시간 재야 한다.
- 장비별 runtime 차이 때문에 logical DP batch나 연구 조건을 바꾸지 않는다. 계산 batch만 바꾸는 경우에도 같은 점수·gradient·회계 의미를 유지하는지 검증한다.

**졸업논문 전용으로 별도 남는 작업:** model-bound receipt 3–5 작업일, PP-Mark 의료 재보정·통합 3–7 작업일, 통합 evidence 검증 3–5 작업일, 본문 개편5–10 작업일이라는 기존 추정이 있다. 이들은 CVPR 방법의 신규성 검증과 별도이며 위 CVPR 완료 예산에 모두 더한 것이 아니다. 실제 보호 release를 목표로 한 secure RNG 이식·fresh retraining도 별도다.

근거: [실제 설계](EXPERIMENT_DESIGN.md), [공통 운영 기록](../19_졸논_CVPR_공통실험_재사용과_분기계획_2026-09-09.md), [K10 프로토콜](../../code_working/XRAY_DP_ATTACK_PROTOCOL_GATE.md), [K5 계획](../../code_working/XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md), [비교군 구현 상태](BASELINE_PREPARATION.md), [CDI 설정](code_sources/X01/conf/attack/cdi.yaml), [MoFit 코드](code_sources/X12/COCO/MoFit_COCO.py). source hash와 계산의 전제는 [JSON](runtime_budget.json)에 함께 보관한다.
"""
    (ROOT/"RUNTIME_REVIEW.md").write_text(md,encoding="utf-8")
    from build_review import html_doc, convert
    calculator = """<section class="panel" aria-labelledby="budget-title"><h2 id="budget-title">사용 가능한 장비·시간으로 다시 계산</h2>
<p>아래는 가정한 작업량을 장비 용량으로 나눈 값입니다. 구현·단계 의존성·사람의 검토 시간과 K5의 별도 3070 실행은 제외합니다.</p>
<div class="filters">
<label>Spark 대수<select id="nodes"><option>1</option><option>2</option><option>3</option><option selected>4</option></select></label>
<label>하루 사용시간<select id="daily"><option>8</option><option>12</option><option>16</option><option selected>24</option></select></label>
<label>유효 가동률<select id="util"><option value="0.6">60%</option><option value="0.8" selected>80%</option><option value="1">100%</option></select></label>
<label>처리량 가정<select id="scenario"><option value="0">빠름</option><option value="1" selected>중앙 계획값</option><option value="2">느림</option></select></label>
<label>최종 비교 범위<select id="extension"><option value="0">m=2 중심</option><option value="1" selected>m=5 본표 포함</option></select></label>
</div><p id="budget-output" role="status" aria-live="polite"></p>
<p class="meta">실측 전 일정 판단용. 모든 원형 MoFit의 전수 평가, U 재학습, 추가 dataset/backbone, 재설계 반복은 포함되지 않습니다.</p></section>"""
    payload=json.dumps({"core":totals,"full":full},ensure_ascii=False)
    js = """<script>const budgetData=PAYLOAD;
function updateBudget(){const n=Number(document.getElementById('nodes').value);
const h=Number(document.getElementById('daily').value),u=Number(document.getElementById('util').value);
const s=Number(document.getElementById('scenario').value),extra=document.getElementById('extension').value==='1';
const work=(extra?budgetData.full:budgetData.core)[s],capacity=n*h*u;
document.getElementById('budget-output').textContent='총 '+work.toLocaleString('ko-KR',{maximumFractionDigits:1})+' 장치시간 ÷ 하루 '+capacity.toLocaleString('ko-KR',{maximumFractionDigits:1})+' 유효 장치시간 = 계산 용량 약 '+(work/capacity).toFixed(1)+'일';}
for(const el of document.querySelectorAll('select'))el.addEventListener('change',updateBudget);
updateBudget();</script>""".replace("PAYLOAD",payload)
    body='<p><a href="index.html">문헌 검토</a> · <a href="design.html">설계</a> · <a href="RUNTIME_REVIEW.md">Markdown</a></p>'+calculator+convert(md)+js
    (ROOT/"runtime_review.html").write_text(html_doc("실험별 시간 검토 · Spark 4대",body),encoding="utf-8")
    soup=BeautifulSoup(body,"html.parser")
    assert len(soup.select("select"))==5
    assert all(math.isfinite(x) and x>=0 for x in totals+full)
    print(json.dumps({"cohorts":cohorts,"core_hours":totals,"with_m5_hours":full,
        "pool_days_core":calc["pool_equivalent_days_core"],
        "pool_days_with_m5":calc["pool_equivalent_days_with_m5"],
        "experiment_executed":False},ensure_ascii=False))

if __name__=="__main__":
    main()
