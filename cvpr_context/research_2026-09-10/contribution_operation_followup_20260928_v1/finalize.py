import datetime,hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pre(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
c=json.loads((out/'contract.json').read_text(encoding='utf-8'));a=json.loads((out/'public_algebra.json').read_text(encoding='utf-8'));i=json.loads((out/'algebra_inputs.json').read_text(encoding='utf-8'))
assert not (out/'completion.json').exists()
for v in i['inputs'].values():assert sha(v['path'])==v['sha256']
assert sha(out/'audit_public_transfer.py')==i['script_sha256']
assert sha(out/'contract.json')==i['contract_sha256']
for p,h in c['previous_results'].items():assert sha(p)==h
assert max(a['checks'].values())<1e-10
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
report=rr/'PUBLIC_TRANSFER_2_1R1_ANCHOR_GEOMETRY_DESIGN_20260928.md'
with report.open('a',encoding='utf-8') as f:f.write('\n실행 전 예상40~60분, 실제 문헌·대수·설계·기록 **'+format(mins,'.2f')+'분**. 대수 계산 '+format(a['seconds'],'.3f')+'초.\n')
block='**2-1 보완(2-1R1) COMPLETE / NEXT 2-2R1 — 2026-09-28**\n\nHR=공개기준값+source head 변화량의 공개전달, 일치3.61e-16. POST/NeurIPS2020 regularization/CME/ICLR2024 transfer와 겹침확인. 기존KD에 공개RN기준값보존/공분산보정/둘다를 허용하는2×2대조 설계완료. 새3종×기존2release=6라벨/12readout,기존24재사용,다음예상20~35분. 이번보완에서새라벨/V0,새Q/release/Expert/Reserved0. 기존2-3 gate와보류상태유지,기여완료아님.'
for p in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if p.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 pre(p,block+'\n\n[Design](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
pre(plan,'**2-1R1완료 / 다음2-2R1 — 2026-09-28:** 공개기준값 보존과 전달기하 보정을 표준KD에도 허용하는3새구성×2release 대조. 6labels/12readouts,기존24재사용,예상20~35분. 기존2-3은계속보류. [설계]('+report.name+')')
s=json.loads((rr/'research_state.json').read_text(encoding='utf-8'))
s['current_planning_report']=report.name;s['step2_active_task']='2-1R1 complete; next2-2R1 bounded anchor/covariance correction contrast.'
s['active_execution']={'status':'no_running_execution','last_completed':'2-1R1','completed_utc':tm.isoformat(),'report':report.name,'next_item':'2-2R1'}
s['next_task']='Announce2-2R1 design/6labels/12readouts and ETA20-35min, execute frozen correction contrast; original2-3 gate remains unchanged. No new DP release or Expert/Reserved.'
s['numbered_execution_plan_20260928']['current_item']='2-1R1 complete;next2-2R1;original2-3 deferred'
s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['numbered_execution_plan_20260928'].setdefault('upper_2_supplements',{})['2-1R1']={'status':'complete','report':report.name,'elapsed_minutes':mins}
dump(rr/'research_state.json',s)
dump(out/'completion.json',{'item':'2-1R1','status':'COMPLETE','utc':tm.isoformat(),'elapsed_minutes':mins,
 'report':str(report),'report_sha256':sha(report),'new_labels':0,'new_V_evaluations':0,'new_Q_access':0,'new_DP_releases':0,
 'files':{p.name:sha(p) for p in [out/'contract.json',out/'audit_public_transfer.py',out/'algebra_inputs.json',out/'public_algebra.json']}})
print(json.dumps({'item':'2-1R1','status':'COMPLETE','elapsed_minutes':mins,'checks':a['checks']},ensure_ascii=True))

