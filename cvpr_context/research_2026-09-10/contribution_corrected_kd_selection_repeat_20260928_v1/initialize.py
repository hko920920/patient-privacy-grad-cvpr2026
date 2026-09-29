import datetime,hashlib,json,shutil,ast
from pathlib import Path
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pre(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
assert not (out/'contract.json').exists()
prior=rr/'contribution_kd_corrections_20260928_v1';pc=json.loads((prior/'contract.json').read_text(encoding='utf-8'))
sc=json.loads((rr/'contribution_public128_control_20260928_v1/contract.json').read_text(encoding='utf-8'))
phases={'P_DINO':['select','construct'],'P_RN':['select','construct','evaluate'],'P_DN':['evaluate'],
 'prior_target_weights':['construct'],'prior_label_seal':['construct'],'patient_bootstrap':['evaluate'],'V_DenseNet':['evaluate'],'V_ResNet18':['evaluate'],'old_selection':['construct']}
inputs={k:{**pc['inputs'][k],'phases':ph} for k,ph in phases.items()}
inputs['P_roster']={**sc['inputs']['P_roster'],'phases':['select']}
for v in inputs.values():assert sha(v['path'])==v['sha256']
for k,p,kind,phase in [
 ('HR_targets',rr/'contribution_head_reconstruction_20260928_v1/HR_target_weights.npz','frozen_head_only_targets','construct'),
 ('HR_label_seal',rr/'contribution_head_reconstruction_20260928_v1/label_seal.json','frozen_head_only_seal','construct'),
 ('correction_targets',prior/'corrected_target_weights.npz','frozen_correction_targets','construct'),
 ('correction_label_seal',prior/'label_seal.json','frozen_correction_seal','construct')]:
 inputs[k]={'path':str(p),'sha256':sha(p),'kind':kind,'phases':[phase]}
for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz'),('comparison_completion','completion.json')]:
 p=prior/n;inputs[k]={'path':str(p),'sha256':sha(p),'kind':'frozen_2-2R1_output','phases':['evaluate']}
implementation=dict(pc['implementation'])
for p,h in implementation.items():assert sha(p)==h
for p in out.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));implementation[str(p)]=sha(p)
start=datetime.datetime.fromtimestamp((out/'EXECUTION_CONTRACT.md').stat().st_ctime,datetime.timezone.utc)
selection={**sc['selection'],'salt':'public-carrier-20260928-v1-repeat202'}
c={**{k:pc[k] for k in ['code_root','historical_code_dir','step3_dir','criteria','solver']},
 'item':'2-3R1','schema':'corrected-KD-carrier-repeat/v1','whole_project_stage':2,
 'authorization':'Sequential user instruction; design10labels/20readouts and ETA20-35min announced before work',
 'started_utc':start.isoformat(),'time_basis':'Contract creation including adapter preparation','deadline_utc':(start+datetime.timedelta(minutes=60)).isoformat(),
 'estimated_minutes':[20,35],'inputs':inputs,'implementation':implementation,'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
 'selection':selection,'public_image_root':sc['public_image_root'],'new_labels':10,'new_readouts':20,'reused_readouts':36,
 'methods':['HR_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','ACKD_RN_ONLY','DINO_ONLY'],'ridge':.1,'bounds':[-1,1],'anchor':'zero',
 'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'primary_contrast':'HR_RN_ONLY-ACKD_RN_ONLY','replication_rule':'HR passes original hurdle against each of AKD/CKD/ACKD on new selection',
 'original_2_3_source_only_gate':'Preserved; repeat does not erase original DP2 source-only result',
 'historical_result_hashes':{**pc['historical_result_hashes'],str(rr/'PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md')}}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json',rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md']
(out/'records_before').mkdir(exist_ok=False)
for p in records:shutil.copy2(p,out/'records_before'/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
block='**2-3 보완(2-3R1) STARTED — 2026-09-28**\n\n예상20~35분을보고하고강화KD차이의공개carrier반복1회진행. 기존선정규칙+고정salt202,HR/AKD/CKD/ACKD/DINO-only×기존DP1/DP2=10labels·20readouts. 기존36재사용. 목표재추출/수정없음. 새선정·모든labels봉인후V. 원래2-3source-onlygate와과거혼합결과보존. Q/noise/release/Expert/Reserved0.'
for p in records[:4]:pre(p,block)
pre(records[5],'**2-3R1진행 — 2026-09-28:** 10labels/20readouts의고정public128선정반복1회. 대상은강화KD대조재현,원래source-onlygate보류는보존.')
s=json.loads(records[4].read_text(encoding='utf-8'));s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md'
s['step2_active_task']='2-3R1 fixed public carrier repeat of strengthened KD contrasts.'
s['active_execution']={'status':'in_progress','item':'2-3R1','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['numbered_execution_plan_20260928']['current_item']='2-3R1 running; original source-only gate unchanged';s['numbered_execution_plan_20260928']['report_sha256']=sha(records[5])
s['next_task']='Finish2-3R1 once and record repeat/source-only results and actual time; no further carrier salt or final-evaluation selection. Remaining mechanism/prior contribution design must be specific.'
dump(records[4],s)
print(json.dumps({'item':'2-3R1','status':'FROZEN','started_utc':c['started_utc'],'inputs':len(inputs),'implementation':len(implementation)},ensure_ascii=True))

