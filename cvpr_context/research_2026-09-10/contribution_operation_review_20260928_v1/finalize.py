import datetime,hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepend(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
assert not (out/'finalization.json').exists()
c=read(out/'contract.json');a=read(out/'operation_algebra.json');i=read(out/'algebra_inputs.json')
assert a['status']=='PASS' and max(a['checks'].values())<1e-10
for p,h in c['code_hashes'].items():assert sha(p)==h
for p,h in c['input_report_hashes'].items():assert sha(p)==h
for v in i['inputs'].values():assert sha(v['path'])==v['sha256']
assert sha(out/'audit_operations.py')==i['script_sha256']
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
report=rr/'PUBLIC_TRANSFER_2_1_OPERATION_COMPARISON_20260928.md'
assert report.exists()
with report.open('a',encoding='utf-8') as f:f.write(f'\n실측: 예상60~120분에 대해 분석 시작부터 기록까지 **{mins:.2f}분**. 정식 선행4편의 관련 절, 현재 핵심 연산, 공개/기존DP 대수 검산으로 범위를 닫았다. 관련 논문 전체 재현이나 전수 문헌조사로 부르지 않는다.\n')
block='**2-1 COMPLETE / NEXT2-2 HEAD RECONSTRUCTION CONTROL — 2026-09-28**\n\n2-1 완료. MT의 signed-public-label ridge 동치1.03e-14, Label Solve/KME/POST/soft-label 선행4편과 연산 대조. KD와 MT는 정보뿐 아니라 공개 가중/제한/중심화/regularizer도 다름. source head에서 class 차이 정확 복구 가능. 별도 class 공통 평균을 P로 바꿀 때 공개 목표 예측 차이2.53%/3.51%; downstream 필요성 미검증. 다음2-2는 같은 MT 지도·solver에서 공통 평균만 P로 대체한 HR RN-only 라벨2개/평가4개(기존20재사용),예상20~40분. MT/KME 동치를 별도 baseline으로 재실행하지 않음. 새Q/release/Expert/Reserved0. Stage2 유지.'
for p in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if p.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 prepend(p,block+'\n\n[Comparison](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
prepend(plan,'**진행 상태 갱신 — 2026-09-28:** 2-1 완료. 다음2-2는 teacher-head 재구성 대조: class 공통 평균만 공개값으로 대체한 RN-only2라벨. 예상20~40분. 2-3은 MT/HR의 DINO-only 대비 사전 투자 기준이 충족될 때만 진행하며, 그렇지 않으면 반복/독립평가를 보류한다. [연산 비교와 상세 설계]('+report.name+')')
s=read(rr/'research_state.json');s['last_efficacy_result_report']=s['current_result_report'];s['current_analysis_report']=report.name
s['step2_active_task']='2-1 complete; next2-2 teacher-head reconstruction ablation. Standard KME/Label Solve overlap explicit.'
s['active_execution']={'status':'no_running_execution','last_completed':'2-1','completed_utc':tm.isoformat(),'next_item':'2-2'}
s['next_task']='Announce2-1 result/actual time and2-2 scope/ETA20-40min, then execute two HR RN-only label packets. No new private release. Repetition2-3 conditional on predeclared source-only hurdle.'
s['numbered_execution_plan_20260928']['current_item']='2-2 pending'
s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['operation_review_2_1_20260928']={'status':'complete','report':report.name,'report_sha256':sha(report),'elapsed_minutes':mins,'algebra':a,
 'novelty_status':'Core mapping / label solving overlap known principles; no paper novelty established',
 'next_contrast':'Two RN-only HR labels: teacher-head contrast retained, class midpoint fixed to public P; same maps/solver.',
 'repetition_gate':'MT or HR must meet existing AUROC/AP hurdle over DINO-only before2-3; no equivalence claims from overlapping CIs.'}
dump(rr/'research_state.json',s)
receipt={'item':'2-1','status':'COMPLETE','elapsed_minutes':mins,'report_sha256':sha(report),'new_labels':0,'new_V_evaluations':0,'new_private_releases':0,
 'code_files_unchanged':len(c['code_hashes']),'prior_reports_unchanged_before_state_update':len(c['input_report_hashes']),'algebra_inputs_unchanged':len(i['inputs']),
 'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'next_item':'2-2'}
dump(out/'finalization.json',receipt)
print(json.dumps({'item':'2-1','status':'COMPLETE','elapsed_minutes':mins,'next_item':'2-2','algebra_checks':a['checks']},ensure_ascii=True,indent=2))
