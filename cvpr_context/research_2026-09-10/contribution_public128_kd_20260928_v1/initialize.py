"""Bind the bounded follow-up to frozen Step1/Step3 artifacts."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j): Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prepend(p,s):
    b=p.read_bytes();i=b.index(b'\n')+1
    p.write_bytes(b[:i]+('\n'+s+'\n\n').encode('utf-8')+b[i:])

out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parent.parent
assert not (out/'contract.json').exists()
s3=rr/'contribution_public128_control_20260928_v1'
c3=json.loads((s3/'contract.json').read_text(encoding='utf-8'))
inputs={}
for k,ph in {'P_DINO':['construct'],'P_RN':['construct','evaluate'],'P_DN':['evaluate'],
    'prior_target_weights':['construct'],'prior_label_seal':['construct'],
    'patient_bootstrap':['evaluate'],'V_DenseNet':['evaluate'],'V_ResNet18':['evaluate']}.items():
    v=c3['inputs'][k];assert sha(v['path'])==v['sha256'];inputs[k]={**v,'phases':ph}
for k,name,ph in [
    ('old_selection','public128_selection.npz',['construct','evaluate']),
    ('old_manifest','public128_manifest.json',['construct']),
    ('old_labels','soft_labels.npz',['construct']),
    ('old_label_seal','label_seal.json',['construct']),
    ('old_fits','construction_results.json',['construct']),
    ('old_results','results.json',['evaluate']),
    ('old_predictions','predictions_private.npz',['evaluate']),
    ('old_bootstrap','paired_bootstrap.npz',['evaluate']),
    ('old_completion','completion.json',['evaluate'])]:
    p=s3/name;inputs[k]={'path':str(p),'sha256':sha(p),'kind':'frozen_Step3_'+name,'phases':ph}
implementation=dict(c3['implementation'])
for p,h in implementation.items(): assert sha(p)==h
for p in out.glob('*.py'):implementation[str(p)]=sha(p)
tm=datetime.datetime.now(datetime.timezone.utc)
c={'schema':'public128-kd/v1','status':'FROZEN_BEFORE_NEW_LABELS_AND_SCORES',
   'authorization':'User asked to continue; bounded public128 KD vs MT follow-up',
   'whole_project_stage':2,'started_utc':tm.isoformat(),'deadline_utc':(tm+datetime.timedelta(minutes=60)).isoformat(),
   'estimated_minutes':[20,40],'question':'Does the defined MT vs ordinary KD utility difference remain on the fixed public128 carrier?',
   'methods':['MT','KD'],'recipes':['RN_ONLY','JOINT'],'releases':['DP1','DP2'],'new_label_solves':4,
   'reused_label_packets':4,'label_anchor':'zero','label_bounds':[-1,1],'ridge':.1,
   'regularizer':'eta=0.001*||A||_F^2/128','target_scaling':'P-weighted target RMS, equal model weights',
   'solver':{'method':'bvls','tol':1e-12,'max_iter':500,'KKT_max':1e-7},
   'primary_receiver':'DenseNet','construction_reference':'ResNet18','primary_metric':'AUROC','secondary_metric':'AP',
   'metric_unit':'image','uncertainty':'paired patient-cluster, conditional on fixed releases and labels',
   'criteria':c3['criteria'],'code_root':c3['code_root'],'historical_code_dir':c3['historical_code_dir'],
   'step3_dir':str(s3),'inputs':inputs,'implementation':implementation,
   'human_contract_sha256':sha(out/'EXECUTION_CONTRACT.md'),
   'historical_result_hashes':{**c3['historical_result_hashes'],str(rr/'PUBLIC_TRANSFER_PUBLIC128_CONTROL_RESULTS_20260928.md'):sha(rr/'PUBLIC_TRANSFER_PUBLIC128_CONTROL_RESULTS_20260928.md')},
   'new_Q_access':0,'new_DP_releases':0,'new_model_forwards':0,'new_pixel_training':0,'Expert_Reserved':False,
   'storage_cap_MiB':50,'automatic_followup':False,'new_KD_results_observed':False}
dump(out/'contract.json',c)
records=[tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md',rr/'research_state.json']
back=out/'records_before';back.mkdir(exist_ok=False)
for p in records: shutil.copy2(p,back/p.name)
dump(out/'records_before_manifest.json',{str(p):sha(p) for p in records})
block='**PUBLIC128 KD CONTROL STARTED — 2026-09-28**\n\n'+(
    '\uacf5\uac1c128\uc5d0\uc11c KD\uc640 MT\ub97c \uac19\uc740 zero-anchor \uaddc\uce59\uc73c\ub85c \ube44\uad50\ud55c\ub2e4. '
    'RN-only/Joint \uac01\uac01 DP1/DP2, \uc0c8 KD \ub77c\ubca84\uac1c. \uae30\uc874 MT \ub77c\ubca8/\ud3c9\uac00 \uc7ac\uc0ac\uc6a9. '
    '\uc608\uc0c120~40\ubd84. Q/release/pixel/forward/Expert/Reserved0. Stage2 \uc720\uc9c0. '
    '\uc0c8 KD \uacb0\uacfc \uc804 \uacc4\uc57d\uacfc \ubaa8\ub4e0 \ub77c\ubca8\uc744 \uace0\uc815\ud55c\ub2e4.')
for p in records[:-1]:prepend(p,block)
s=json.loads(records[-1].read_text(encoding='utf-8-sig'))
s['current_step']=2;s['current_planning_report']=out.name+'/EXECUTION_CONTRACT.md'
s['step2_active_task']='Public128 KD vs MT follow-up: four new KD labels, existing MT controls, no new private release.'
s['active_execution']={'status':'in_progress','task':'public128_kd_control_20260928','out_dir':str(out),'started_utc':c['started_utc'],'deadline_utc':c['deadline_utc']}
s['next_task']='Complete the authorized public128 KD/MT comparison and record bounded research judgment.'
dump(records[-1],s)
print(json.dumps({'status':'FROZEN','inputs':len(inputs),'implementation':len(implementation),'started_utc':c['started_utc']}))
