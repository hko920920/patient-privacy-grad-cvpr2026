import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readj(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def writej(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


tr, rr, out = map(Path, sys.argv[1:4])
report_name = "PUBLIC_TRANSFER_2_2R2_WEIGHT_CROSSOVER_RESULTS_20260929.md"
report = rr / report_name
assert not (out / "finalization.json").exists(), "Already finalized"
contract = readj(out / "contract.json")
result = readj(out / "results.json")
verification = readj(out / "verification.json")
completion = readj(out / "completion.json")
assert verification["status"] == "PASS"
assert result["new_labels"] == 4 and result["new_readouts"] == 8
assert not result["decisions"]["shared_weight_separation_hypothesis_passes"]
assert result["decisions"]["HR_at_matched_U_passes_hurdle"]
for name, expected in completion["files"].items():
    assert sha(out / name) == expected, name

now = datetime.now(timezone.utc)
start = datetime.fromisoformat(contract["started_utc"])
minutes = (now - start).total_seconds() / 60
assert minutes <= 45
finished = now.isoformat()
timing = (
    "\n## 완료 실측\n\n"
    f"시작 {contract['started_utc']}, 완료 기록 {finished}. "
    f"코드 연결·실행·독립 검산·보고를 포함한 실측 **{minutes:.2f}분**이다. "
    "사전 예상20~30분, 상한45분 이내에 완료했다. "
    "최종 상태 파일 저장과 해시 확인은 이 완료 기록 직후 수행한다.\n"
)
report.write_text(report.read_text(encoding="utf-8-sig") + timing, encoding="utf-8")
report_hash = sha(report)

banner = (
    "**2-2R2 COMPLETE / COMMON WEIGHT HYPOTHESIS FAILED — 2026-09-29**\n\n"
    "공개 대응 가중치 교차 완료. DenseNet 두 고정 release 평균 AUROC에서 "
    "HR의 환자균등−class균형 +0.028012, CI[+0.014857,+0.042020]; "
    "ACKD의 같은 대비 −0.025069, CI[−0.047561,−0.001852]. "
    "두 방식 모두 환자균등에서 개선된다는 사전 가설은 실패했다. "
    "같은 환자균등에서 HR−ACKD +0.060416, CI[+0.033064,+0.088087]의 이득은 남았다. "
    "AP의 평균 구간은0을 포함한다. 더 강했던 기존 ACKD-B와 과거 혼합 결과를 보존한다.\n\n"
    f"새4라벨·8평가, 기존12평가 재사용, 독립 검산 통과. 실측{minutes:.2f}분. "
    "새 Q/noise/release/pixel/forward/Expert/Reserved0. 현재 실행 없음. "
    "원래2-3 보류·2-3R1 혼합·상위3 미착수·연구 단계2 유지. "
    "기여 완료나 일반 가중치 원리로 승격하지 않는다.\n\n"
    "**다음 제안2-1R3(미실행):** 같은 가중치에서 공개 기준값 항과 source 차이 전달 연산을 "
    "수식으로 분리하고, 기존 CME/POST로 설명되는 부분과 남는 차이를 정리한다. "
    "필요할 경우에만 대조 하나의 설계를 제시한다. 설계만15~25분; 새 라벨/V/Q/release/수신자0.\n\n"
)
status_paths = [
    tr / "AGENTS.md",
    tr / "CURRENT_STATUS.md",
    tr / "WORKLOG.md",
    rr / "RESEARCH_FRAMEWORK.md",
    rr / "PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md",
]
for path in status_paths:
    old = path.read_text(encoding="utf-8-sig")
    first, rest = old.split("\n", 1)
    relative = report_name if path.parent == rr else "CVPR 주제 탐색/research_2026-09-10/" + report_name
    block = banner + f"[2-2R2 결과](<{relative}>)\n\n"
    path.write_text(first + "\n\n" + block + rest.lstrip("\n"), encoding="utf-8")

state_path = rr / "research_state.json"
state = readj(state_path)
state["current_result_report"] = report_name
state["last_efficacy_result_report"] = report_name
state["step2_active_task"] = "2-2R2 complete: shared weight hypothesis failed; HR advantage at matched U remains. Next proposed 2-1R3 design only."
state["updated_utc"] = finished
state["active_execution"] = {
    "status": "no_running_execution",
    "last_completed_item": "2-2R2",
    "last_completed_report": report_name,
    "out_dir": str(out),
    "started_utc": contract["started_utc"],
    "completed_utc": finished,
    "elapsed_minutes": minutes,
    "next_item": "2-1R3",
    "next_status": "proposed_design_not_executed",
}
state["next_task"] = (
    "2-1R3 design only: decompose matched-weight HR and ACKD into source-contrast "
    "transport and public reference term; identify known CME/POST components and "
    "whether one remaining contrast is justified. Do not automatically run labels or V."
)
state["next_task_output"] = (
    "Operation decomposition, existing-prior overlap, at most one necessary bounded "
    "control design; no new V/labels/Q/release/receiver. ETA15-25min."
)
plan = state["numbered_execution_plan_20260928"]
plan["updated_utc"] = finished
plan["report_sha256"] = sha(rr / plan["report"])
plan["current_item"] = "2-2R2 complete_shared_hypothesis_failed; next2-1R3 design proposed, not executed"
plan["new_execution"] = False
plan["upper_2_supplements"]["2-2R2"].update({
    "status": "complete_shared_hypothesis_failed",
    "report": report_name,
    "report_sha256": report_hash,
    "elapsed_minutes": minutes,
    "matched_U_HR_advantage_remains": True,
})
plan["upper_2_supplements"]["2-1R3"] = {
    "status": "proposed_design_not_executed",
    "question": "Separate the public reference and source-contrast operator at matched weights before specifying any further contrast.",
    "scope": "Existing public matrices, protected heads, prior operation review; no new labels/V/Q/release/receiver",
    "eta_minutes": [15, 25],
    "automatic_next_experiment": False,
}
state["weight_crossover_2_2R2_20260929"] = {
    "status": "complete_shared_hypothesis_failed",
    "report": report_name,
    "report_sha256": report_hash,
    "out_dir": str(out),
    "started_utc": contract["started_utc"],
    "completed_utc": finished,
    "elapsed_minutes": minutes,
    "decisions": result["decisions"],
    "fixed_release_mean_contrasts": result["fixed_two_release_mean_contrasts"],
    "verification": verification,
    "new_labels": 4,
    "new_readouts": 8,
    "reused_readouts": 12,
    "new_Q_access": 0,
    "new_releases": 0,
    "Expert_Reserved_access": False,
    "next_item": "2-1R3",
    "next_status": "proposed_design_not_executed",
}
writej(state_path, state)

final = {
    "item": "2-2R2",
    "status": "complete_shared_hypothesis_failed",
    "started_utc": contract["started_utc"],
    "completed_utc": finished,
    "elapsed_minutes": minutes,
    "time_basis": contract["time_basis"],
    "report": str(report),
    "report_sha256": report_hash,
    "verification_sha256": sha(out / "verification.json"),
    "completion_sha256": sha(out / "completion.json"),
    "sealed_runtime_outputs_unchanged": True,
    "new_labels": 4,
    "new_readouts": 8,
    "reused_readouts": 12,
    "new_private_queries": 0,
    "new_private_releases": 0,
    "next_item": "2-1R3",
    "next_status": "proposed_design_not_executed",
    "next_eta_minutes": [15, 25],
    "status_files": {str(p): sha(p) for p in status_paths + [state_path]},
}
writej(out / "finalization.json", final)
print(json.dumps({
    "status": final["status"],
    "elapsed_minutes": minutes,
    "report_sha256": report_hash,
    "status_files_updated": len(final["status_files"]),
    "runtime_outputs_unchanged": True,
    "next_item": "2-1R3",
    "next_executed": False,
}, ensure_ascii=True))
