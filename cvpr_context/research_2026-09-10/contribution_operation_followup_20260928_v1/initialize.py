import datetime,hashlib,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pre(p,x):
 s=p.read_text(encoding='utf-8');a,_,b=s.partition('\n');p.write_text(a+'\n\n'+x+'\n\n'+b,encoding='utf-8')
assert not (out/'contract.json').exists()
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json',rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
(out/'records_before').mkdir(exist_ok=False)
for p in records:shutil.copy2(p,out/'records_before'/p.name)
start=datetime.datetime.fromisoformat('2026-09-28T17:37:02+09:00').astimezone(datetime.timezone.utc)
c={'item':'2-1R1','display_name':'2-1 보완','parent_item':'2-1','started_utc':start.isoformat(),'eta_minutes':[40,60],
 'authorization':'User requested continuing after clarification that2-3 is deferred but research/design continues. ETA40-60min announced before tools.',
 'goal':'Separate public anchoring and transfer geometry from extra protected information; closest strong alternatives and one bounded discriminating design.',
 'allowed':['Read existing code/reports and primary papers','Public-only and existing-DP-output linear algebra','Derive a fair next contrast and gate before evaluating it'],
 'forbidden':['New V efficacy/readout during this design item','New labels during this design item','Raw Q','New DP noise/release','Pixel/model forward','Expert/Reserved'],
 'prior_state':'2-2 complete;2-3 original consistency gate unmet, not an entire-research stop',
 'numbering_policy':'R1 is a visible supplement, not replacement/renumbering of completed2-1 or2-2. A subsequent new contrast, if justified, is2-2R1; original2-3 remains deferred.',
 'records_before':{str(p):sha(p) for p in records},
 'previous_results':{str(rr/n):sha(rr/n) for n in ['PUBLIC_TRANSFER_1_5_SOURCE_CONTROL_RESULTS_20260928.md','PUBLIC_TRANSFER_2_1_OPERATION_COMPARISON_20260928.md','PUBLIC_TRANSFER_2_2_HEAD_RECONSTRUCTION_RESULTS_20260928.md']}}
dump(out/'contract.json',c)
block='**2-1 보완(2-1R1) STARTED — 2026-09-28**\n\n사용자 계속 진행 지시에 따라 설계를 이어간다. 예상40~60분을 먼저 보고했다. 2-2에서 같은DINO head의 공개처리로KD개선이 남았으므로, 공개anchor/전달연산 차이를 선행과 대조하고 구별할 예측 하나를 설계한다. 공개·기존DP 대수만 허용,새label/V효용/Q/release/Expert/Reserved0. 기존2-1/2-2결과와2-3보류는보존. 별도보완 식별자R1이며 과거번호를 바꾸지 않는다. Stage2유지.'
for p in records[:4]:pre(p,block)
pre(records[5],'**진행 갱신 — 2026-09-28:** 2-3보류는유지. 사용자연속진행지시에 따라 **2-1보완(2-1R1)**으로 공개처리와 강한선행의 차이를 설계한다(예상40~60분). 기존2-1/2-2를 미완료로 되돌리거나 번호를 바꾸지 않는다.')
s=json.loads(records[4].read_text(encoding='utf-8'))
s['step2_active_task']='2-1R1 operation/design supplement after2-2; keep original2-3 deferred.'
s['active_execution']={'status':'analysis_in_progress','item':'2-1R1','started_utc':c['started_utc'],'eta_minutes':[40,60],'out_dir':str(out)}
s['numbered_execution_plan_20260928']['current_item']='2-1 supplement (2-1R1) in progress;2-3 deferred'
s['numbered_execution_plan_20260928']['report_sha256']=sha(records[5]);s['numbered_execution_plan_20260928']['plan_only']=False
s['next_task']='Finish2-1R1 concrete design and closest-prior comparison; announce result/time then any justified2-2R1 scale/design/ETA before execution. Preserve2-3 gate and independent evaluation.'
dump(records[4],s)
print(json.dumps({'item':'2-1R1','status':'STARTED','started_utc':c['started_utc']},ensure_ascii=True))
