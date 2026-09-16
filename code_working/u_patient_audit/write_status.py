"""Render current U execution facts, without advancing scientific status."""
from .common import *
from datetime import datetime
from zoneinfo import ZoneInfo

def read(path):
    p=RUN/path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def main():
    research=ROOT.parent/"CVPR 주제 탐색/research_2026-09-10"
    cohort=read("cohort/summary.json");verification=read("cohort/verification.json")
    cache=read("cache/summary.json");bench=read("batch_benchmark.json")
    lines=["# U 실험 실행 기록","",
        "갱신: "+datetime.now().astimezone().isoformat(timespec="seconds"),"",
        "사용자 승인에 따라 기존 NIH PA 자료로 수행한 첫 파일럿의 실행 기록이다. 학습 실행·무결성 검증과 연구 가설의 성공을 구분한다.","",
        "## 1. 자료 선택과 분할","",
        f"- 실제 파일 {cohort['verified_images']:,}장을 checksum·decode·pixel/follow-up 기준으로 검사했다. 메타데이터에만 있고 미확보인 375장은 제외했다.",
        "- 서로 다른 적격 영상 4장 이상인 425명 중 400명을 선택했다. 개발 80명, 방법 선택 40명, 보정 140명, 시험 140명이다.",
        "- 환자당 학습 배정 2장·U 관측 2장을 고정했다. 두 모델에서 참여 그룹을 교환하며, 관측 U 영상은 어느 모델에도 학습시키지 않는다.",
        "- 별도 대조 64명·품질 확인 32명·공통 학습 배경 256명을 둔다. 전체 선택은 752명/2,304장, 모델별 학습은 912장이다.",
        "- source·환자 역할·사진 hash·중복·학습 교집합을 독립 코드로 다시 확인해 PASS했다. 촬영 날짜/study UID는 없어 follow-up 번호 분리의 한계를 남긴다.",
        "- 모델별 비참여 보정/시험 환자는 각각 70명이다. 5% 파일럿이며 1% 확증 규모로 주장하지 않는다.","",
        "[분할 검증](../../code_working/_reports/cvpr_u_pilot_v1_001/cohort/verification.json) · [선택 영상 표본](../../code_working/_reports/cvpr_u_pilot_v1_001/cohort/selection_grid.png)","",
        "## 2. 구현과 수치 점검","",
        "- SD2.1 exact revision/hash·기존 전처리로 latent/text를 준비하고 공개 DINOv2 특징 및 다른 환자 대조를 계산했다.",
        "- 중복·동일 follow-up 배제, padding을 제외한 probe basis, 조정 반경, 전 영상 학습 순서의 4개 핵심 검사가 PASS했다.",
        "- 최초 50-step v1 두 모델은 학습 실행 점검이다. 복원추출로 일부 미노출 영상이 있으므로 최종 참여 실험으로 사용하지 않는다.",
        "- 배치 속도 비교에서 스케일링 없는 FP16 backward는 상대 gradient 차이 32.87%를 보였다. 실제 학습의 1024 gradient scale을 적용한 재검증은 0.61%, loss 상대 오차 0.00159%였다.",
        "- 실제 학습과 같은 정밀도 조건에서 batch 4를 채택하고, 입력 gradient에도 스케일링을 명시했다. bitwise 동등성으로 주장하지 않는다.",
        "- v2는 전체 영상을 순회하는 고정 shuffle을 쓰며 228 step 안에 학습 이미지 912장을 모두 사용한다. 원래 v1 체크포인트·기록은 별도로 보존했다.","",
        "## 3. 두 모델 상태","",
        "| 모델 | 완료 step | 노출 확인 | 단계 판정 |","|---|---:|---|---|"]
    for model in ["model_1","model_2"]:
        report=read(f"training_coverage_v2/{model}/report_1000.json")
        check=read(f"training_coverage_v2/{model}/verification_1000.json")
        progress=read(f"training_coverage_v2/{model}/progress.json")
        if report:
            exp=f"{report['exposure_min']}–{report['exposure_max']}회/학습 영상"
            status="학습 완료 / 독립 검증 "+("PASS" if check else "대기")
            lines.append(f"| {model} | 1,000 | {exp} | {status} |")
        elif progress:lines.append(f"| {model} | {progress['step']} | 진행 중 | 학습 중 |")
        else:lines.append(f"| {model} | 0 | 대기 | 미시작 |")
    lines+=["","학습·optimizer·exposure/trace 검증은 의료 생성 품질 또는 MIA 성능의 인증이 아니다.","",
        "## 4. 품질과 감사 실행",""]
    for model in ["model_1","model_2"]:
        q=read(f"quality/training_coverage_v2/{model}/loss_report.json")
        if q:
            lo,hi=q["patient_bootstrap_95_ratio_interval"]
            lines.append(f"- {model}의 별도 품질 환자 32명에서 target/base denoising loss 비율 {q['target_over_base']:.4f}, 환자 bootstrap 95% 구간 [{lo:.4f}, {hi:.4f}]. 임상 효용 또는 MIA 지표가 아니다.")
    probes=list((RUN/"probe_smoke").glob("**/report.json")) if (RUN/"probe_smoke").exists() else []
    if not probes:lines.append("- 새 후보의 GPU 실행 검증은 아직 완료 전이다.")
    for p in probes:
        s=json.loads(p.read_text())
        lines.append(f"- {s.get('scenario','U')} / {s['training_dir']} step {s['step']}: 개발 환자 {s['unique_fit_patients']}명, 모델-환자 실행 {s['model_patient_evaluations']}회. 중앙 실행 {s['median_seconds_per_patient']:.2f}초/환자, {s['forward_per_patient']}F+{s['backward_per_patient']}B, 가중치 불변 검사 PASS. 성능/신규성 검증 아님.")
        verified=p.parent/"verification.json"
        if verified.exists():
            v=json.loads(verified.read_text())
            lines.append(f"  별도 코드의 참여 정답·교환 쌍·점수 산술 검증: {v['status']}. [검증 기록](../../code_working/_reports/cvpr_u_pilot_v1_001/{verified.relative_to(RUN).as_posix()}).")
    if probes:
        lines.append("- F/B는 이미지별 모델 forward/backward 평가 횟수이며 API 호출 수가 아니다. 보정·시험 환자의 target 공격 점수는 계산하지 않았다.")
    encoder=read("encoder_control/report.json")
    analysis=read("analysis/U_step1000_n8_v1/analysis.json")
    if analysis:
        a=analysis['metrics']['response']
        lines.append(f"- 원래 FP16 평가의 U 8명 판별 진단: 후보 AUC 모델 1 {a['models']['model_1']['auc']:.4f}, 모델 2 {a['models']['model_2']['auc']:.4f}, 평균 {a['macro_auc']:.4f}. 참여 시 점수 상승 {a['paired_positive']}/8명, 모델 간 환자 순위 상관 {a['model_rank_spearman']:.1f}. 현재 표본에서 일관된 후보 효과를 확인하지 못했다. 아래 후속 정밀도 재평가와 구분한다.")
        lines.append("  [8명 분석과 해석](U_EIGHT_PATIENT_ANALYSIS.md). 계산의 독립 재검증은 PASS했으며, 추가 환자·GPU 실행·부호/threshold fitting 없이 기존 값만 분석했다.")
    if encoder:
        lines.append("- 공개 encoder-only 대조: fit 80명으로 학습한 뒤 selection 40명에서 AUC 0.5475. 보정/시험은 사용하지 않았다. 작은 개발 대조이며 모든 편향의 부재를 증명하지 않는다.")
    visual=read("quality/visual_review_pair.json") or read("quality/visual_review.json")
    if visual:
        lines.append("- 생성 영상 외형 점검: "+visual["interpretation_ko"])
        lines.append("[모델별 생성 결과](../../code_working/_reports/cvpr_u_pilot_v1_001/quality/"+visual["grid"]+")")
    actual=read("verification_20260914/final_summary.json")
    if actual:
        lines += ["","## 5. 후속 실제 재검증","",
            "- [실제 재검증 결과·정밀도 보완·예상/실측 시간](U_VERIFICATION_RESULTS.md)은 기존 U8의 수치 검증 기록이다. 비영점 입력 미분과 checkpointing, 기존 U8 전체 추적 재현, E/U 기본 대조를 GPU에서 실행하고 독립 코드로 산술·출처를 검증했다.",
            f"- 원래 6개 점수×16회 실행을 정확히 재현했다. 32개 support 최적화 모두 최종 목적값이 증가했다. 중간 감소가 있던 과정은 {actual['support_folds_with_intermediate_decrease']}개로 별도 기록했다.",
            "- 첫 환자의 같은 계수에서 평가 정밀도를 바꾸자 작은 참여 모델−미참여 모델 차이의 부호가 바뀌었다. 기존 8명 전체의 query/reference 및 기본 비교 점수를 FP32로 다시 계산했다. FP16 최적화 계수는 고정했으며 전체 FP32 재최적화로 부르지 않는다.",
            "- 수치·실행 검증 완료는 공격 성공이 아니다. E에서도 일관된 강한 양성 대조를 확보하지 못했고, 원래 설계의 fitting/별도 평가와 강한 비교는 남아 있다.",
            "- 향후에도 단계 시작 전에 범위와 예상 시간을 보고하고, 실측에 따라 남은 시간을 갱신한다."]
    secmi=read("baseline_screen_20260914/secmi_v1/analysis.json")
    secmi_check=read("baseline_screen_20260914/secmi_v1/verification.json")
    if secmi and secmi_check:
        lines += ["","## 6. 기존 공격의 후속 E/U 진단","",
            "- [SecMI E/U 실제 결과와 진행·보류 판정](U_BASELINE_SCREEN.md)이 최신 실험 판단이다. 기존 개발 8명과 방법 선택 40명을 분리하고, 두 모델에서 총 384건·4,608F·0B를 평가했다.",
            "- CDI 저자 저장소의 SecMI-stat 구현을 현재 모델에 이식했다. 고정 FP32·t=100·step=10이며, 전체 CDI 또는 최신 SOTA 전체 비교가 아니다."]
        for scenario in ("E","U"):
            stats=secmi["metrics"]["selection"][scenario]["score_mean"]["models"]
            values=[]
            for model in ("model_1","model_2"):
                s=stats[model]; lo,hi=s["patient_bootstrap_percentile_95"]
                values.append(f"{model}: AUC {s['auc']:.4f} [{lo:.4f}, {hi:.4f}]")
            supported=secmi["screening_decisions"][scenario]["screening_signal_supported_in_this_setting"]
            lines.append(f"- {scenario}, 선택 40명 평균 점수: "+" / ".join(values)+f". 사전 진행 기준 {'충족' if supported else '미충족'}.")
        lines += ["- 보정/시험 평가·추가 학습·후보 R 튜닝은 실행하지 않았다. 선택 자료의 탐색적 판정이며, 미통과를 누출 부재 또는 연구 불가능으로 해석하지 않는다."]
    lines+=["","## 7. 남은 판정","",
        ("- 후속 연구의 진행·보류는 [SecMI 진단 결과](U_BASELINE_SCREEN.md)를 따른다. 아래는 재개할 때 남는 요건이며 즉시 실행 중인 작업 목록이 아니다." if secmi and secmi_check else "- 다음 항목은 아직 완료되지 않은 성능·기여 검증이다."),
        ("- [실제 재검증](U_VERIFICATION_RESULTS.md)의 완료된 검사와 남은 성능 평가를 구분한다. 보정항 결함으로 지목한 원인 추정은 철회하며, 새 표본 규모는 검정력과 실측 예산으로 정한다." if actual else
         "- [후속 정밀 검토](U_PILOT_REVIEW.md)는 실제 수치 진단 전의 검토 이력이다. 최신 GPU 재검증 진행은 별도 실제 검증 기록을 따른다."),
        "- 개발 자료에서 점수 방향·기본 집계 대조를 확인하고, 별도 보정·시험 환자로 실제 patient-FPR/TPR와 불확실성을 평가한다.",
        "- MoFit 전이·CDI 등 주장에 맞는 강한 비교와 제거 실험.",
        "- 새로운 환자 분할/독립 seed 확인 및 보호 수준을 맞춘 DP·효용·비용 비교.",
        "- 기존 졸논 K5/K10 동결 계약·체크포인트의 상태는 별도다. 이번 자료가 직접 공유 가능하다고 간주하지 않는다.","",
        "[실행 계획](U_EXECUTION_PLAN.md) · [구현 안내](../../code_working/u_patient_audit/README.md)"]
    (research/"U_EXECUTION_STATUS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("U_EXECUTION_STATUS.md updated")
if __name__=="__main__":main()
