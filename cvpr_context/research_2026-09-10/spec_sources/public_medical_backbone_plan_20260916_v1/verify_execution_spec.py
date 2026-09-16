"""Independent plan arithmetic/role verifier; imports no producer code or ML library."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv, json, hashlib
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text(encoding='utf-8'))
def rows(n):
    with (P/n).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
count=0
def check(ok,label):
    global count
    count+=1
    if not ok:raise RuntimeError(label)
s=read('execution_spec.json');people=rows('patients.csv');images=rows('images.csv');schedule=rows('training_schedule.csv');tasks=rows('generation_tasks.csv')
for name,h in s['input_sha256'].items():check(sha(P/name)==h,'input hash '+name)
for name,h in s['output_sha256'].items():check(sha(P/name)==h,'output hash '+name)
check(sha(P/'build_execution_spec.py')==s['builder_sha256'],'producer source')
by_patient={x['patient_id']:x for x in people}
check(len(by_patient)==len(people)==905,'unique patients')
check(Counter(x['backbone_role'] for x in people)==dict(train=640,selection=128,confirmation=128,reserve=9),'patient counts')
check(Counter(x['backbone_role'] for x in images)==dict(train=749,selection=128,confirmation=128,reserve=11),'image counts')
check(len({x['image_id'] for x in images})==1016,'unique images')
for x in images:check(x['backbone_role']==by_patient[x['patient_id']]['backbone_role'],'patient role consistency')
check(not any(x['known_diagnostic_history']=='1' for x in people if x['backbone_role'] in ('selection','confirmation')),'known history disjoint reserved cohorts')
train={x['image_id']:x for x in images if x['backbone_role']=='train'}
check(len(schedule)==5992,'total committed image presentations')
salt='public-medical-backbone-execution-20260916-v1'
def hash_parts(*values):return hashlib.sha256('|'.join([salt]+[str(v) for v in values]).encode('utf8')).hexdigest()
for e in range(8):
    block=schedule[e*749:(e+1)*749]
    expected=sorted(train,key=lambda iid:hash_parts('epoch',e,iid))
    check([x['image_id'] for x in block]==expected,'exact without-replacement epoch '+str(e))
for i,x in enumerate(schedule):
    step=i//4
    check(int(x['step'])==step+1 and int(x['batch_position'])==i%4 and int(x['epoch'])==i//749+1,'step epoch mapping')
    check(x['patient_id']==train[x['image_id']]['patient_id'],'scheduled training patient only')
    check(int(x['noise_seed'])==int(hash_parts('noise',step)[:15],16),'noise seed')
    check(int(x['timestep'])==int(hash_parts('timestep',step,i%4)[:15],16)%1000,'time seed')
for e,step in [(4,749),(8,1498)]:
    seen=Counter(x['image_id'] for x in schedule[:step*4])
    check(set(seen)==set(train) and set(seen.values())=={e},'exact exposure '+str(e))
    c=next(x for x in s['training']['checkpoints'] if x['name']=='E'+str(e))
    check(c['successful_optimizer_steps']==step and c['planned_image_presentations']==step*4,'checkpoint counts')
check(len(tasks)==32 and len({x['seed'] for x in tasks})==32,'unique new input seeds')
for phase in ('selection','confirmation'):
    group=[x for x in tasks if x['phase']==phase]
    check(Counter(x['prompt_id'] for x in group)==dict(generic=4,normal=4,effusion=4,cardiomegaly=4),'prompt counts')
check(not {x['seed'] for x in tasks} & {'26091631','26091632'},'new versus old bridge seed IDs')
check(s['generation']['max_images']==2*16+16+2*2==52,'maximum generated images')
check(s['generation']['max_UNet_forward_calls']==52*30,'maximum sampling forwards')
check(abs(s['cost']['linear_training_estimate_seconds']-1498*496.731/1000)<1e-10,'time arithmetic')
check(s['status']=='PLANNED_NOT_EXECUTED' and s['new_GPU_runs']==0,'no execution result claim')
result=dict(status='PASS_INDEPENDENT_PROSPECTIVE_ROLE_SCHEDULE_AND_INPUT_CHECKS',utc=datetime.now(timezone.utc).isoformat(),checks=count,
    manifest_reconstruction='Separate build_verify_inventory.py --verify result retained; this verifier additionally checks saved roles and independently reconstructs schedules.',
    GPU_runs=0,train_images=749,checkpoints=[749,1498],exposure_counts=[4,8],generated_images_max=52,
    sources={name:sha(P/name) for name in ('patients.csv','images.csv','training_schedule.csv','generation_tasks.csv','execution_spec.json','verify_execution_spec.py')})
with (P/'execution_spec_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
