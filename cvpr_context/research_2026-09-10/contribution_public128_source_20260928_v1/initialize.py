import datetime,hashlib,json,shutil
from pathlib import Path

out=Path(__file__).resolve().parent;r=out.parent;tr=r.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepend(p,t):
 b=p.read_bytes();i=b.index(b'\n')+1;p.write_bytes(b[:i]+('\n'+t+'\n\n').encode('utf-8')+b[i:])
assert not (out/'contract.json').exists()
prior=r/'contribution_public128_kd_20260928_v1';pc=json.loads((prior/'contract.json').read_text(encoding='utf-8'))
inputs={}
phases={'P_DINO':['construct'],'P_RN':['evaluate'],'P_DN':['evaluate'],'prior_target_weights':['construct'],
        'prior_label_seal':['construct'],'patient_bootstrap':['evaluate'],'V_DenseNet':['evaluate'],'V_ResNet18':['evaluate'],
        'old_selection':['construct','evaluate'],'old_manifest':['construct']}
for k,ph in phases.items():
 v=pc['inputs'][k];assert sha(v['path'])==v['sha256'];inputs[k]={**v,'phases':ph}
for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),
            ('comparison_bootstrap','paired_bootstrap.npz'),('comparison_completion','completion.json')]:
 p=prior/n;inputs[k]={'path':str(p),'sha256':sha(p),'kind':'frozen_1-4_output','phases':['evaluate']}
implementation=dict(pc['implementation'])
for p,h in implementation.items():assert sha(p)==h
for p in out.glob('*.py'):implementation[str(p)]=sha(p)
start=datetime.datetime.fromisoformat('2026-09-28T17:04:48.706254+09:00').astimezone(datetime.timezone.utc)
c={**{k:pc[k] for k in ('code_root','historical_code_dir','step3_dir','criteria')},
   'schema':'public128-source-1-5/v1','item':'1-5','whole_project_stage':2,
   'authorization':'User requested sequential numbered execution with mandatory pre-run ETA and post-run reports, then continue with next scoped item',
   'started_utc':start.isoformat(),'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'deadline_utc':(start+datetime.timedelta(minutes=45)).isoformat(),'estimated_minutes':[15,30],
   'new_labels':2,'releases':['DP1','DP2'],'carrier':'frozen public128','label_anchor':'zero','ridge':.1,
   'label_bounds':[-1,1],'eta_rule':'0.001*||A||_F^2/128','target_scaling':'P-weighted RMS, single DINO block weight1',
   'solver':pc['solver'],'primary_receiver':'DenseNet','construction_reference':'ResNet18','metric_unit':'image',
   'bootstrap':'2000 paired patient-cluster draws, conditional fixed labels/releases',
   'main_contrasts':['MT_RN_ONLY-DINO_ONLY','MT_JOINT-DINO_ONLY'],'reference_contrasts':['KD_RN_ONLY-DINO_ONLY','KD_JOINT-DINO_ONLY'],
   'inputs':inputs,'implementation':implementation,'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
   'historical_result_hashes':{**pc['historical_result_hashes'],str(r/'PUBLIC_TRANSFER_PUBLIC128_KD_RESULTS_20260928.md'):sha(r/'PUBLIC_TRANSFER_PUBLIC128_KD_RESULTS_20260928.md')},
   'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',r/'RESEARCH_FRAMEWORK.md',r/'research_state.json']
b=out/'records_before';b.mkdir(exist_ok=False)
for p in records:shutil.copy2(p,b/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
protocol='**NUMBERED EXECUTION REPORTING RULE — 2026-09-28**\n\nUser requires stable upper1/2/3 and subtask1-1 numbering. Before every execution, report item ID, question/design, bounded scale and estimated time. After completion, report item ID, actual work, results/limitations and measured time; then specify the next item, justified design/scale and ETA before proceeding. Continue sequentially within authorized scope without repeated routine permission requests. Preserve explicit privacy/final-evaluation boundaries. Do not rename completed experiments or add an unnumbered diagnostic queue.'
prepend(tr/'AGENTS.md',protocol)
text='**1-5 PUBLIC128 SOURCE CONTROL STARTED — 2026-09-28**\n\n1-5: two new DINO-only zero-anchor label packets on the fixed public128 and existing DP1/DP2. Frozen MT/KD labels and scores reused. ETA15-30minutes; cap45minutes. No Q/noise/release/pixel/encoder forward/Expert/Reserved. Both labels sealed before V evaluation. Stage2 retained. Next2-1 only after reporting this result and announcing its design/scale/ETA.'
for p in records[:-1]:prepend(p,text)
s=json.loads(records[-1].read_text(encoding='utf-8-sig'))
s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md'
s['step2_active_task']='Executing numbered item1-5 public128 DINO-only control.'
s['active_execution']={'status':'in_progress','item':'1-5','task':'public128_source_control_20260928','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['numbered_execution_reporting_rule']={'stable_ids':True,'before':['item','design','scale','ETA'],'after':['item','actual_work','results','limitations','actual_time'],'then':['next_item','necessary_design','scale','ETA','continue_in_scope'],'user_authorized_continuation':True}
s['numbered_execution_plan_20260928']['current_item']='1-5 running'
s['next_task']='Finish1-5 and report; then announce and execute2-1 operation/closest-prior comparison.'
dump(records[-1],s)
print(json.dumps({'status':'1-5_FROZEN','inputs':len(inputs),'implementation':len(implementation),'started_utc':c['started_utc']}))
