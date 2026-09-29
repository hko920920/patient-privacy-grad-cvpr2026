import datetime,hashlib,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pre(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
assert not (out/'contract.json').exists()
prior=rr/'contribution_head_reconstruction_20260928_v1';pc=json.loads((prior/'contract.json').read_text(encoding='utf-8'))
keys=['P_DINO','P_RN','P_DN','prior_target_weights','prior_label_seal','patient_bootstrap','V_DenseNet','V_ResNet18','old_selection','old_manifest','public_target']
inputs={k:pc['inputs'][k] for k in keys}
for v in inputs.values():assert sha(v['path'])==v['sha256']
for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz'),('comparison_completion','completion.json')]:
 p=prior/n;inputs[k]={'path':str(p),'sha256':sha(p),'kind':'frozen_2-2_output','phases':['evaluate']}
implementation=dict(pc['implementation'])
for p,h in implementation.items():assert sha(p)==h
for p in out.glob('*.py'):implementation[str(p)]=sha(p)
start=datetime.datetime.fromtimestamp((out/'EXECUTION_CONTRACT.md').stat().st_ctime,datetime.timezone.utc)
c={**{k:pc[k] for k in ['code_root','historical_code_dir','step3_dir','criteria','solver']},
 'item':'2-2R1','schema':'anchor-covariance-factorial/v1','whole_project_stage':2,
 'authorization':'Continuing numbered research; design and ETA20-35min announced before execution',
 'started_utc':start.isoformat(),'time_basis':'Human execution contract creation, including adapter preparation',
 'deadline_utc':(start+datetime.timedelta(minutes=60)).isoformat(),'estimated_minutes':[20,35],
 'inputs':inputs,'implementation':implementation,'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
 'design_report':'PUBLIC_TRANSFER_2_1R1_ANCHOR_GEOMETRY_DESIGN_20260928.md','design_report_sha256':sha(rr/'PUBLIC_TRANSFER_2_1R1_ANCHOR_GEOMETRY_DESIGN_20260928.md'),
 'new_labels':6,'new_readouts':12,'reused_readouts':24,'label_models':['ResNet18'],'ridge':.1,'bounds':[-1,1],'anchor':'zero',
 'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'primary_contrast':'HR_RN_ONLY-ACKD_RN_ONLY','additional_controls':['AKD_RN_ONLY','CKD_RN_ONLY'],
 'correction_ridge_rule':'0.001*trace(DINO public image weighted covariance)/64',
 'public_anchor':'Actual P-label RN supervised ridge; source reference uses schema-matched P clipped means',
 'HR_extra_value_rule':'All three HR-minus-control contrasts meet the unchanged development hurdle',
 'original_2_3_gate':'Unchanged and deferred; this is a distinct supplement, no retrospective gate relaxation',
 'historical_result_hashes':{**pc['historical_result_hashes'],str(rr/'PUBLIC_TRANSFER_2_2_HEAD_RECONSTRUCTION_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_2_2_HEAD_RECONSTRUCTION_RESULTS_20260928.md')}}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json',rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
(out/'records_before').mkdir(exist_ok=False)
for p in records:shutil.copy2(p,out/'records_before'/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
block='**2-2 보완(2-2R1) STARTED — 2026-09-28**\n\n예상20~35분을 먼저 보고했다. 공개기준값보존AKD/공분산보정CKD/둘다ACKD×기존DP1/DP2의6labels,12readouts;기존24재사용. 모두봉인후V. 공개RN실제labelhead를기준값으로허용. λ는고정P trace규칙. HR추가차별효용과요소대조를함께보고. Q/noise/release/pixel/forward/Expert/Reserved0. 원래2-3gate와보류유지.'
for p in records[:4]:pre(p,block)
pre(records[5],'**현재진행2-2R1 — 2026-09-28:** 2-1R1에서고정한기준값/공분산보정대조6labels·12readouts 실행. 예상20~35분. 기존2-3보류유지.')
s=json.loads(records[4].read_text(encoding='utf-8'));s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md'
s['step2_active_task']='2-2R1: public anchor x covariance correction factorial, six new labels.'
s['active_execution']={'status':'in_progress','item':'2-2R1','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['numbered_execution_plan_20260928']['current_item']='2-2R1 running; original2-3 deferred';s['numbered_execution_plan_20260928']['report_sha256']=sha(records[5])
s['next_task']='Complete frozen2-2R1 once, report exact result and elapsed time; retain all controls and original2-3 gate. Define any next research question/scale/ETA explicitly, no automatic new release or final evaluation.'
dump(records[4],s)
print(json.dumps({'item':'2-2R1','status':'FROZEN','inputs':len(inputs),'implementation':len(implementation),'started_utc':c['started_utc']},ensure_ascii=True))

