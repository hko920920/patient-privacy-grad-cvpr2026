import datetime, hashlib, json, shutil, sys
from pathlib import Path

out=Path(__file__).resolve().parent
rr=out.parent
tr=rr.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepend(p,s):
    t=p.read_text(encoding='utf-8'); head,sep,tail=t.partition('\n')
    p.write_text(head+'\n\n'+s+'\n\n'+tail,encoding='utf-8')
assert not (out/'contract.json').exists()
now=datetime.datetime.now(datetime.timezone.utc)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json',rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
(out/'records_before').mkdir(exist_ok=True)
before={}
for p in records:
    shutil.copy2(p,out/'records_before'/p.name);before[str(p)]=sha(p)
hist=rr/'contribution_search_20260924'
sources=[hist/n for n in ['public_transport_probe.py','validate_public_transport.py','joint_compiler.py','source_label_baselines.py','fixed_image_compilation.py','geometry_diagnostic.py']]
sources += [rr/'contribution_public128_control_20260928_v1'/n for n in ['construct_labels.py','carrier_common.py']]
sources += [rr/'contribution_public128_kd_20260928_v1'/'run_kd.py']
sources=[p for p in sources if p.exists()]
reports=[rr/n for n in ['PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md','PUBLIC_TRANSFER_LABEL_STRATEGY_20260928.md','PUBLIC_TRANSFER_1_5_SOURCE_CONTROL_RESULTS_20260928.md','PUBLIC_TRANSFER_PUBLIC128_KD_RESULTS_20260928.md']]
c={'item':'2-1','started_utc':'2026-09-28T08:12:31+00:00','contract_recorded_utc':now.isoformat(),'started_before_contract':'Read-only code and primary-literature inspection began after the user-facing ETA; no numerical execution preceded this contract.',
   'eta_minutes':[60,120],'scope':'Operation and closest-prior comparison; no efficacy experiment.',
   'questions':['Exact current MT, teacher-score KD, source-only label solve and signed public-weight equivalence','Closest primary papers and their overlap, differences and access conditions','Whether one distinct and meaningful 2-2 contrast remains'],
   'allowed':['Read existing local code/contracts/aggregate reports','Read primary publication sources','Reuse already computed algebraic validation','Optional small public/toy algebra only, with no V'],
   'prohibited':['Raw Q access','New DP summary or noise','New image/label bank or V efficacy/readout','Encoder forward','Expert/Reserved access','Tuning against V','Renaming equivalent operations as independent baselines'],
   'planned_sources':['Balog et al. ICML2018 kernel mean embedding database release','Nguyen et al. ICLR2021 KIP / Label Solve','Wang et al. ICML2025 POST','Optional CVPR2026 Hard Truths about Soft Labels'],
   'code_hashes':{str(p):sha(p) for p in sources},'input_report_hashes':{str(p):sha(p) for p in reports},'records_before':before,
   'output':'One operation/novelty comparison report and a justified 2-2 specification, or explicit absence of a distinct comparison.'}
dump(out/'contract.json',c)
block='**2-1 OPERATION / PRIOR REVIEW STARTED — 2026-09-28**\n\n2-1 연산·근접연구 비교 진행. 예상1~2시간을 먼저 안내했다. MT/KD/DINO-only/public signed-weight 연산과 직접 선행3~4편을 같은 정보 접근 조건에서 대응한다. 기존 대수 검산을 재사용하며 새 bank·V 효용평가·Q·release·Expert/Reserved는 없다. 산출물은 비교표와 구별되는2-2의 필요성/설계 판단. 1-5 혼합 결과와 상위1 현재범위 완료를 유지한다.'
for p in records[:4]: prepend(p,block)
s=json.loads((rr/'research_state.json').read_text(encoding='utf-8'))
s['step2_active_task']='2-1 operation / closest-prior review in progress; no new efficacy run.'
s['active_execution']={'status':'analysis_in_progress','item':'2-1','started_utc':c['started_utc'],'out_dir':str(out),'eta_minutes':[60,120],'new_private_releases':0}
s['numbered_execution_plan_20260928']['current_item']='2-1 in progress'
s['next_task']='Finish2-1 report, announce result/actual time and justified2-2 scope/ETA before executing a distinct comparison. Do not fabricate an equivalent baseline.'
dump(rr/'research_state.json',s)
print(json.dumps({'item':'2-1','status':'CONTRACT_RECORDED','code_files':len(sources),'started_utc':c['started_utc']},ensure_ascii=True))
