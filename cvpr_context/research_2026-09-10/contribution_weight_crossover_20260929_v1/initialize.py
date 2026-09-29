import ast, datetime, hashlib, json, shutil
from pathlib import Path

out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pre(p,s):
 raw=p.read_text(encoding='utf-8');h,_,r=raw.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
assert not (out/'contract.json').exists()
prior=rr/'contribution_kd_corrections_20260928_v1'
design=rr/'contribution_patient_map_design_20260929_v1'
pc=read(prior/'contract.json')
keys=['P_DINO','P_RN','P_DN','prior_target_weights','prior_label_seal','patient_bootstrap','V_DenseNet','V_ResNet18','old_selection','old_manifest','public_target']
inputs={k:pc['inputs'][k] for k in keys}
for v in inputs.values():assert sha(v['path'])==v['sha256']
extras={
 'comparison_results':(prior/'results.json',['evaluate'],'prior_frozen_evaluation'),
 'comparison_predictions':(prior/'predictions_private.npz',['evaluate'],'prior_frozen_evaluation'),
 'comparison_bootstrap':(prior/'paired_bootstrap.npz',['evaluate'],'prior_frozen_evaluation'),
 'comparison_completion':(prior/'completion.json',['evaluate'],'prior_seal'),
 'bridge_targets':(design/'weight_bridge_targets_NOT_LABELS.npz',['construct'],'public_existing_DP_head_postprocessing'),
 'design_completion':(design/'completion.json',['construct'],'prior_design_seal'),
 'old_public_operators':(prior/'public_operators.npz',['construct'],'public'),
 'old_HR_targets':(rr/'contribution_head_reconstruction_20260928_v1/HR_target_weights.npz',['construct'],'frozen_existing_DP_postprocessing'),
 'old_ACKD_targets':(prior/'corrected_target_weights.npz',['construct'],'frozen_existing_DP_postprocessing'),
}
for k,(p,phases,kind) in extras.items():inputs[k]={'path':str(p),'sha256':sha(p),'kind':kind,'phases':phases}
implementation=dict(pc['implementation'])
for p,h in implementation.items():assert sha(p)==h,p
implementation[str(design/'analyze_public_maps.py')]=sha(design/'analyze_public_maps.py')
for p in out.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));implementation[str(p)]=sha(p)
start=datetime.datetime.fromisoformat('2026-09-29T06:27:27+00:00')
dre=read(design/'NEXT_2_2R2_DESIGN.json')
assert dre['new_labels']==4 and dre['new_readouts']==8 and dre['status']=='designed_not_executed'
c={**{k:pc[k] for k in ['code_root','historical_code_dir','step3_dir','criteria','solver']},
 'item':'2-2R2','schema':'public-correspondence-weight-crossover/v1','whole_project_stage':2,
 'started_utc':start.isoformat(),'deadline_utc':(start+datetime.timedelta(minutes=45)).isoformat(),
 'time_basis':'First tool timestamp after user-facing estimate, including adapter preparation',
 'estimated_minutes':[20,30],'authorization':'User explicitly said proceed after design2-1R2; ETA20-30min announced before tools.',
 'inputs':inputs,'implementation':implementation,'design_dir':str(design),'design':dre,
 'design_json_sha256':sha(design/'NEXT_2_2R2_DESIGN.json'),
 'design_report':'PUBLIC_TRANSFER_2_1R2_WEIGHT_SEPARATION_DESIGN_20260929.md',
 'design_report_sha256':sha(rr/'PUBLIC_TRANSFER_2_1R2_WEIGHT_SEPARATION_DESIGN_20260929.md'),
 'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
 'new_labels':4,'new_readouts':8,'reused_readouts':12,'label_models':['ResNet18'],'ridge':.1,'bounds':[-1,1],'anchor':'zero',
 'new_Q_access':0,'new_DP_releases':0,'new_encoder_forwards':0,'new_pixel_training':0,'Expert_Reserved':False,
 'historical_result_hashes':{**pc['historical_result_hashes'],
  str(rr/'PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md'),
  str(rr/'PUBLIC_TRANSFER_2_3R1_SELECTION_REPEAT_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_2_3R1_SELECTION_REPEAT_RESULTS_20260928.md')}}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json',rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
(out/'records_before').mkdir(exist_ok=False)
for p in records:shutil.copy2(p,out/'records_before'/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
block='**2-2R2 STARTED — 2026-09-29**\n\n예상20~30분 안내 후 공개 대응 가중치 교차 진행. HR-B/ACKD-U×기존DP1·DP2=새4labels/8readouts,기존HR-U/ACKD-B/DINO의12readouts재사용. 최초public128,질환class균형·ridge·기준값·solver고정. 네labels봉인후V. 두주대비모두보고. Q/noise/release/pixel/forward/Expert/Reserved0. 기존혼합gate유지.'
for p in [records[0],records[1],records[2],records[3],records[5]]:pre(p,block)
s=read(records[4]);s['active_execution']={'status':'in_progress','item':'2-2R2','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['step2_active_task']='2-2R2 correspondence-weight crossover, four labels/eight readouts.'
s['next_task']='Complete four sealed labels and one evaluation bundle; report primary weight effects and matched-weight residual difference. No automatic further experiments.'
s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md';s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
s['numbered_execution_plan_20260928']['current_item']='2-2R2 running; previous mixed gates unchanged'
s['numbered_execution_plan_20260928']['report_sha256']=sha(records[5])
s['numbered_execution_plan_20260928']['upper_2_supplements']['2-2R2']['status']='running'
dump(records[4],s)
print(json.dumps({'item':'2-2R2','status':'FROZEN','inputs':len(inputs),'implementation':len(implementation),'deadline_utc':c['deadline_utc']}))
