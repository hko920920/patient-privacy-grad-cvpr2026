import datetime,hashlib,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepend(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
assert not (out/'contract.json').exists()
prior=rr/'contribution_public128_source_20260928_v1';pc=json.loads((prior/'contract.json').read_text(encoding='utf-8'))
phases={'P_DINO':['construct'],'P_RN':['construct','evaluate'],'P_DN':['evaluate'],'prior_target_weights':['construct'],'prior_label_seal':['construct'],
 'patient_bootstrap':['evaluate'],'V_DenseNet':['evaluate'],'V_ResNet18':['evaluate'],'old_selection':['construct','evaluate'],'old_manifest':['construct']}
inputs={}
for k,ph in phases.items():
 v=pc['inputs'][k];assert sha(v['path'])==v['sha256'];inputs[k]={**v,'phases':ph}
for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz'),('comparison_completion','completion.json')]:
 p=prior/n;inputs[k]={'path':str(p),'sha256':sha(p),'kind':'frozen_1-5_output','phases':['evaluate']}
for k,p,kind in [
 ('public_target',tr/'code_working/_reports/receiver_feature_followup_20260924_v1/public_target/target.npz','public_P_only_target'),
 ('public_maps',rr/'contribution_search_20260924/transport_public_maps_and_dp_weights.npz','public_maps_RN_keys_only')]:
 inputs[k]={'path':str(p),'sha256':sha(p),'kind':kind,'phases':['construct']}
implementation=dict(pc['implementation'])
for p,h in implementation.items():assert sha(p)==h
for p in out.glob('*.py'):implementation[str(p)]=sha(p)
start=datetime.datetime.fromtimestamp((out/'EXECUTION_CONTRACT.md').stat().st_ctime,datetime.timezone.utc)
c={**{k:pc[k] for k in ['code_root','historical_code_dir','step3_dir','criteria','solver']},
 'item':'2-2','schema':'head-reconstruction-2-2/v1','whole_project_stage':2,
 'authorization':'Sequential numbered user authorization after pre-run design and ETA20-40min',
 'started_utc':start.isoformat(),'time_basis':'Creation of execution contract, including subsequent adapter preparation','contract_recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'deadline_utc':(start+datetime.timedelta(minutes=60)).isoformat(),'estimated_minutes':[20,40],
 'inputs':inputs,'implementation':implementation,'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
 'new_labels':2,'new_readouts':4,'reused_readouts':20,'label_models':['ResNet18'],'private_information':'Existing DINO head contrast only; public class midpoint',
 'ridge':.1,'bounds':[-1,1],'anchor':'zero','new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'primary_contrast':'MT_RN_ONLY-HR_RN_ONLY','secondary_contrasts':['HR_RN_ONLY-KD_RN_ONLY','HR_RN_ONLY-DINO_ONLY'],
 'next_2_3_gate':'MT or HR meets fixed source-only hurdle; otherwise defer repeated selection and independent test',
 'historical_result_hashes':{**pc['historical_result_hashes'],
 str(rr/'PUBLIC_TRANSFER_1_5_SOURCE_CONTROL_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_1_5_SOURCE_CONTROL_RESULTS_20260928.md'),
 str(rr/'PUBLIC_TRANSFER_2_1_OPERATION_COMPARISON_20260928.md'):sha(rr/'PUBLIC_TRANSFER_2_1_OPERATION_COMPARISON_20260928.md')}}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json']
(out/'records_before').mkdir(exist_ok=False)
for p in records:shutil.copy2(p,out/'records_before'/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
block='**2-2 HEAD RECONSTRUCTION CONTROL STARTED — 2026-09-28**\n\n예상20~40분을 먼저 안내하고2-2 진행. 기존DINO head에서 class차이 복구, class공통평균만 P로 대체. 같은 MT지도/solver의 RN-only2라벨, DenseNet/RN4새readout,기존20재사용. 라벨 봉인 후V. MT−HR/HR−KD/HR−DINO 판정. 새Q/noise/release/pixel/forward/Expert/Reserved0. 2-3은source-only 대비기존 투자hurdle 충족 시만. Stage2 유지.'
for p in records[:-1]:prepend(p,block)
s=json.loads(records[-1].read_text(encoding='utf-8'))
s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md';s['step2_active_task']='2-2 head reconstruction RN-only two-label comparison.'
s['active_execution']={'status':'in_progress','item':'2-2','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['numbered_execution_plan_20260928']['current_item']='2-2 running'
s['next_task']='Finish2-2, report actual results/time, then execute2-3 only if predeclared source-only hurdle met. No automatic independent test.'
dump(records[-1],s)
print(json.dumps({'item':'2-2','status':'FROZEN','inputs':len(inputs),'implementation':len(implementation),'started_utc':c['started_utc']}))
