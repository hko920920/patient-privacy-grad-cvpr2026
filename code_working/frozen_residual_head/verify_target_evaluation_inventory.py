"""Independent metadata-first recount; never imports the production inventory code."""
import csv,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
CODE=Path(__file__).resolve().parents[1]
P=CODE/'_reports/target_evaluation_inventory_20260917_v2'
checks=0
def ck(ok,m):
 global checks
 checks+=1
 if not ok:raise ValueError(m)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rr(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
 start=time.perf_counter();c=read(P/'contract.json');a=read(P/'inventory.json');s={k:Path(v) for k,v in c['sources'].items()}
 for p,h in c['source_sha256'].items():ck(sha(p)==h,'Current source mutation')
 for p,h in c['frozen_sha256'].items():ck(sha(p)==h,'Frozen source mutation')
 for n,h in a['output_sha256'].items():ck(sha(P/n)==h,'Output binding')
 meta={x['Image Index']:x for x in rr(s['metadata'])};inv=rr(s['inventory']);pids=lambda rs:{int(x['patient_id']) for x in rs}
 cv=pids(rr(s['evaluation'])+rr(s['auxiliary']));bb=rr(s['backbone']);train=pids([x for x in bb if x['backbone_role']=='train']);reserved=pids([x for x in bb if x['backbone_role']!='train'])
 previous=pids(rr(s['preflight']))|pids(read(s['reference']));history=pids(rr(s['history']));public=pids([x for x in inv if x['partition']=='public_development'])
 subsets=dict(public_all=public,after_current_training_and_all_cvpr_roles=public-cv-train,
 after_all_prior_reference_and_recorded_diagnostics=public-cv-train-previous-history,
 strict_unassigned_public=public-cv-train-previous-history-reserved,
 previously_evaluated_remaining_public=(public-cv-train)&previous,reserved_nontraining_public=public&reserved)
 patient_lists=read(P/'patient_sets_private.json')
 def check_summary(rrr,claimed):
  ck(len(rrr)==claimed['images'],'Image count')
  ck(len({int(x['Patient ID']) for x in rrr})==claimed['patients'],'Patient count')
  labelsets={l:{int(x['Patient ID']) for x in rrr if l in x['Finding Labels'].split('|')} for l in c['labels']}
  for l in c['labels']:
   ck(len(labelsets[l])==claimed['labels'][l]['patients'],'Label patient count '+l)
   ck(sum(l in x['Finding Labels'].split('|') for x in rrr)==claimed['labels'][l]['images'],'Label image count '+l)
  ck(len(labelsets['Emphysema']&labelsets['Pneumothorax'])==claimed['target_patient_overlap'],'Multi-label overlap')
 for name,pp in subsets.items():
  ck(pp==set(map(int,patient_lists[name])),'Patient exclusion reconstruction '+name)
  check_summary([meta[x['image_id']] for x in inv if int(x['patient_id']) in pp and x['partition']=='public_development'],a['stages'][name])
 check_summary([meta[x['image_id']] for x in inv],a['inventory'])
 for part,claimed in a['partitions'].items():check_summary([meta[x['image_id']] for x in inv if x['partition']==part],claimed)
 for role,claimed in a['reserved_roles'].items():
  pp=pids([x for x in bb if x['backbone_role']==role]);check_summary([meta[x['image_id']] for x in inv if int(x['patient_id']) in pp and x['partition']=='public_development'],claimed)
 for n in ['model1','model2']:ck(pids(rr(s[n]))<=cv,'Historical model train overlap excluded')
 trainfiles=set(s['train_list'].read_text().splitlines());testfiles=set(s['test_list'].read_text().splitlines())
 testpatients={int(meta[i]['Patient ID']) for i in testfiles};allocated=pids(rr(s['patients']))
 outside=[x for i,x in meta.items() if i in trainfiles and x['View Position']=='PA' and int(x['Patient ID']) not in allocated|testpatients]
 check_summary(outside,a['outside_existing_allocation_trainval_PA'])
 ck({x['Image Index'] for x in outside}=={x['image_id'] for x in read(P/'outside_metadata_private.json')},'Outside allocation IDs')
 root=s['inventory'].parent/'images';ck(not any((root/x['Image Index']).exists() for x in outside),'Outside metadata not locally acquired')
 audit={x['image_id']:x for x in rr(s['audit'])};index={x['image_id']:x for x in inv}
 hps=defaultdict(set);pps=defaultdict(set)
 for x in inv:hps[x['sha256'].lower()].add(int(x['patient_id']))
 for x in audit.values():pps[x['pixel_sha256']].add(int(x['patient_id']))
 packet=read(P/'file_checks.json')
 expected={x['image_id'] for x in inv if x['partition']=='public_development' and int(x['patient_id']) in subsets['after_current_training_and_all_cvpr_roles']}
 ck({x['image_id'] for x in packet}==expected and len(packet)==len(expected),'Every eligible file included exactly once')
 for x in packet:
  f=root/x['image_id'];original=index[x['image_id']]
  ck(f.is_file() and f.stat().st_size==int(original['bytes']),'Actual file size')
  ck(sha(f)==x['sha256']==original['sha256'].lower(),'Independent file SHA')
  ck(len(hps[x['sha256']])==1 and len(pps[audit[x['image_id']]['pixel_sha256']])==1,'No exact patient duplicate')
 ck(sha(a['evaluator']['installed_source'])==a['evaluator']['source_sha256'],'Evaluator source')
 for k,m in a['evaluator']['candidates'].items():
  ck(Path(m['weights_path']).stat().st_size==m['bytes'] and sha(m['weights_path'])==m['sha256'],'Evaluator weights '+k)
  for l in c['labels']:ck((l in m['valid_labels'])==m['target_support'][l],'Trained output '+k+l)
 ck(a['evaluator']['candidates']['pc']['target_support']['Emphysema'] and a['evaluator']['candidates']['pc']['target_support']['Pneumothorax'],'PC both target outputs')
 ck(not a['evaluator']['candidates']['mimic_ch']['target_support']['Emphysema'],'MIMIC invalid emphysema output')
 # Preserve all prior192 generated PNG bytes, not merely their manifest.
 old=CODE/'_reports/medical_head_20260916_v1'
 for x in read(old/'generation_manifest.json'):ck(sha(old/x['image_path'])==x['image_sha256'],'Old generated image unchanged')
 fail=P.with_name('target_evaluation_inventory_20260917_v1');fc=read(fail/'contract.json')
 ck(sha(fail/'failed_source.py')==fc['source_sha256'][str(s['script'])],'First failed source preserved')
 orderdiff=0
 for x in inv:
  original=meta[x['image_id']]['Finding Labels']
  if original!=x['finding_labels']:
   orderdiff+=1;ck(sorted(original.split('|'))==sorted(x['finding_labels'].split('|')),'Only label order corrected')
 ck(orderdiff==32,'Label-order exception count')
 result=dict(status='PASS_TARGET_INVENTORY_INDEPENDENT_RECOUNT',checks=checks,
  inventory_sha256=sha(P/'inventory.json'),contract_sha256=sha(P/'contract.json'),verifier_sha256=sha(__file__),
  seconds=time.perf_counter()-start,created_utc=datetime.now(timezone.utc).isoformat(),
  verified_public_files=len(packet),prior_generated_images_unchanged=192,label_order_corrections=orderdiff,
  first_failed_source_preserved=True,new_GPU=0,new_model_inference=0,clinical_or_statistical_validation=False)
 with (P/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
