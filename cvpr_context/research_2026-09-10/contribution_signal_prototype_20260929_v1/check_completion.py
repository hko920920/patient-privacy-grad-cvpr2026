from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent
r=p.parent
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
c=json.loads((p/"completion.json").read_text(encoding="utf-8"))
for f,h in c["files"].items():assert sha(p/f)==h,f
assert sha(r/c["report"])==c["report_sha256"]
for f,h in c["preserved_evidence"].items():assert sha(r/f)==h,f
before=json.loads((p/"state_before.json").read_text(encoding="utf-8"))["public_transfer_2_4A_design_20260929"]
assert sha(r/"contribution_signal_design_20260929_v1/public_geometry.py")==before["public_geometry_code_sha256"]
assert sha(r/"contribution_signal_design_20260929_v1/public_geometry.json")==before["public_geometry_sha256"]
s=json.loads((r/"research_state.json").read_text(encoding="utf-8"))
assert s["active_execution"]["status"]=="no_running_execution"
assert s["active_execution"]["last_completed_item"]=="2-4B"
assert s["current_result_report"]=="PUBLIC_TRANSFER_2_2R2_WEIGHT_CROSSOVER_RESULTS_20260929.md"
assert s["numbered_execution_plan_20260928"]["upper_3"]["status"]=="not_started"
assert s["numbered_execution_plan_20260928"]["upper_2_supplements"]["2-1R3"]["status"]=="not_executed_followup_recommendation_withdrawn"
assert sha(r/"PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md")==s["numbered_execution_plan_20260928"]["report_sha256"]
print(json.dumps({"artifact_audit":"PASS","frozen_files":len(c["files"]),"preserved_reports":len(c["preserved_evidence"]),"design_code_and_geometry_unchanged":True,"active_execution":"none","actual_minutes":c["elapsed_minutes"],"report":str(r/c["report"])},ensure_ascii=False,indent=2))

