"""Record design-only status without replacing the latest efficacy result."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parent
thesis = root.parent.parent
report = 'PUBLIC_TRANSFER_2_4A_SIGNAL_DESIGN_20260929.md'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
state_path = root/'research_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
expected_evidence = state['continuation_decision_20260929']['evidence_hashes']
assert all(sha(root/name)==value for name,value in expected_evidence.items())
assert state['current_result_report']=='PUBLIC_TRANSFER_2_2R2_WEIGHT_CROSSOVER_RESULTS_20260929.md'
assert not (out/'completion.json').exists()
previous_state_sha = sha(state_path)
started = datetime(2026,9,29,9,27,33,tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
minutes = (now-started).total_seconds()/60
record = {
    'item':'2-4A','status':'DESIGN_CANDIDATE_AND_PUBLIC_GEOMETRY_COMPLETE_NOT_EFFICACY',
    'started_utc':started.isoformat(),'recorded_utc':now.isoformat(),
    'eta_minutes':[45,60],'elapsed_minutes_through_record_write':minutes,
    'report':report,'report_sha256':sha(root/report),
    'public_geometry_sha256':sha(out/'public_geometry.json'),
    'public_geometry_code_sha256':sha(out/'public_geometry.py'),
    'previous_state_sha256':previous_state_sha,
    'preserved_evidence_sha256':expected_evidence,
    'candidate':'Choose patient-DP signal using the receiver space attainable by source-preserving labels on a fixed public carrier; project before clipping.',
    'new_Q_queries':0,'new_DP_releases':0,'new_labels':0,'new_V_evaluations':0,
    'Expert_Reserved_access':False,'production_code_changed':False,
    'public_numeric_calculation':True,'novelty_established':False,
    'efficacy_established':False,'paper_complete':False,
    'previous_HR_followup_queue_withdrawn':True,
    'next_item':'2-4B','next_status':'PROPOSED_NOT_EXECUTED',
    'next_eta_minutes':[60,90],
    'next_scope':'One public-input implementation and strong subspace-control package. No Q, V efficacy, final evaluation, or new private release.',
    'new_private_release_requires_a_separate_concrete_contract':True,
    'scope_change':'Candidate may require a new receiver-specific protected query; not a zero-additional-budget consequence of the old DINO summary.',
}
notice = '''**2-4A 완료 — 새 보호 신호 설계 후보, 성능 검증 전 (2026-09-29)**

기존 HR 가중치·기준값 대조의 자동 후속은 중단 유지. 새 후보는 ‘기존 source의 순위를 보존하는 라벨 변경 공간에서 receiver로 전달 가능한 성분을 구하고, 그 성분을 환자 clipping 전에 선택해 보호’하는 설계다. 공개128/P만으로 source 보존 차원64·receiver 변경 rank64를 확인했다. 설명용 동일 cap에서 공개 환자/class 제한 행34→1은 기하 검산이며 Q/V 효용 결과가 아니다. 차원 감소만의 noise 이득은 주장하지 않는다. KIP/GEP/Dosser/Common Mechanism/workload DP의 알려진 원리를 인정하며 독자성·최종 효용은 미확정이다.

2-4A는 설계·공개 계산 완료, 새 labels/V/Q/release/Expert/Reserved0. 기존 결과와2-1R3 미실행·권고 철회 보존. 다음2-4B는 공개 입력 구현/표준 부분공간 대조 한 묶음(예상60~90분), **계획만·미실행**이다. 새 receiver private query가 필요한 설계이므로 기존ε8 요약의 무료 후처리라고 부르지 않으며, 실제 Q 실행은 별도 보호 계약이 필요하다. 상위2 기여 검증 미완료·상위3 미착수.

'''
files = [thesis/'AGENTS.md',thesis/'CURRENT_STATUS.md',thesis/'WORKLOG.md',root/'RESEARCH_FRAMEWORK.md',root/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
before = {}
after = {}
for p in files:
    before[str(p)] = sha(p)
    rel = ('CVPR 주제 탐색/research_2026-09-10/' if p.parent==thesis else '')+report
    block = notice+'[2-4A 설계와 근거](<'+rel+'>)\n\n'
    raw = p.read_bytes()
    assert b'2-4A ' not in raw[:120]
    position = raw.find(b'\n')+1
    assert position > 0
    p.write_bytes(raw[:position]+b'\n'+block.encode('utf-8')+raw[position:])
    after[str(p)] = sha(p)
state['public_transfer_2_4A_design_20260929'] = record
state['latest_design_report']=report
state['latest_review_report']=report
state['current_planning_report']=report
state['current_decision_report']=report
state['next_task_plan_report']=report
state['step2_active_task']='2-4A design/public geometry complete; one candidate selected, contribution unestablished. 2-4B public prototype proposed, not executed.'
state['next_task']='Proposed2-4B: one public-input prototype and full/projected/PCA control package; 60-90min. No further HR efficacy ablations or new private query scheduled.'
state['next_task_output']='Verify source-preserving compilation, equal-noise null control, patient sensitivity and clipping/subspace difference; determine whether the candidate merits a separate private-query contract.'
state['current_paper_problem_fixed']=False
state['stage2_status']='in_progress'
state['active_execution']['status']='no_running_execution'
state['active_execution']['next_item']=None
state['active_execution']['next_status']='2-4B_proposed_not_started_no_private_execution_scheduled'
plan = state['numbered_execution_plan_20260928']
plan['current_item']='2-4A design/public geometry complete; 2-4B public prototype proposed only. 2-1R3 remains withdrawn and unexecuted.'
plan['updated_utc']=now.isoformat()
plan['report_sha256']=sha(root/plan['report'])
plan['upper_2_supplements']['2-4A']={'status':record['status'],'report':report,'elapsed_minutes':minutes}
plan['upper_2_supplements']['2-4B']={'status':'proposed_not_executed','eta_minutes':[60,90],'public_only':True,'Q_release_Reserved':False}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all(sha(root/name)==value for name,value in expected_evidence.items())
readback = json.loads(state_path.read_text(encoding='utf-8'))
assert readback['current_result_report']=='PUBLIC_TRANSFER_2_2R2_WEIGHT_CROSSOVER_RESULTS_20260929.md'
assert readback['active_execution']['last_completed_item']=='2-2R2'
record['status_files_before_sha256']=before
record['status_files_after_sha256']=after
record['state_after_sha256']=sha(state_path)
record['historical_efficacy_reports_unchanged']=True
(out/'completion.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'item':record['item'],'status':record['status'],'elapsed_minutes':minutes,'historical_evidence_unchanged':True,'next':'2-4B proposed, not executed'},ensure_ascii=False))
