"""Read-only NIH reference feasibility. No model imports/inference or role mutation."""
import csv, hashlib, json, time, ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

CODE=Path(__file__).resolve().parents[1]
R=next(CODE.parent.glob('CVPR */research_2026-09-10'))
OUT=CODE/'_reports/target_evaluation_inventory_20260917_v2'
DATA=CODE/'_data'
PLAN=R/'spec_sources/public_medical_backbone_plan_20260916_v1'
COHORT=CODE/'_reports/cvpr_u_pilot_v1_001/cohort'
LABELS=['Emphysema','Pneumothorax','No Finding','Effusion']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(x,m):
 if not x:raise RuntimeError(m)
def ids(rr):return {str(int(x['patient_id'])) for x in rr}
def counts(rr):
 p=ids(rr);labels={k:{'patients':len(ids([x for x in rr if k in x['finding_labels'].split('|')])),
 'images':sum(k in x['finding_labels'].split('|') for x in rr)} for k in LABELS}
 a=ids([x for x in rr if LABELS[0] in x['finding_labels'].split('|')]);b=ids([x for x in rr if LABELS[1] in x['finding_labels'].split('|')])
 return dict(patients=len(p),images=len(rr),labels=labels,target_patient_overlap=len(a&b))
def main():
 start=time.perf_counter();require(not OUT.exists(),'Preserve previous run');OUT.mkdir()
 files=dict(inventory=DATA/'raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv',
 metadata=DATA/'intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv',
 train_list=DATA/'intake/nih_chestxray14_metadata/train_val_list.txt',test_list=DATA/'intake/nih_chestxray14_metadata/test_list.txt',
 patients=DATA/'derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv',
 source=DATA/'derived/nih_cxr14_pa_target_enriched_v1/all_selected_pa_images_private.csv',
 census=DATA/'derived/nih_cxr14_pa_target_enriched_v1/official_pa_test_census_private.csv',
 evaluation=COHORT/'evaluation_images.csv',auxiliary=COHORT/'auxiliary_images.csv',
 model1=COHORT/'model_1_train.csv',model2=COHORT/'model_2_train.csv',audit=COHORT/'source_audit.csv',
 backbone=PLAN/'images.csv',history=PLAN/'diagnostic_history.csv',plan=PLAN/'plan_lock.json',
 exclusion=PLAN/'exclusion_calculation.json',
 preflight=CODE/'_reports/nih_cxr14_k5_evaluator_preflight_protocol_v1_001/real_evaluator_tasks_local.csv',
 reference=CODE/'_reports/medical_head_20260916_v1/reference.json',
 olddecision=CODE/'_reports/medical_head_20260916_v1/research_decision.json',
 prioranalysis=CODE/'_reports/private_signal_20260917_v1/analysis.json',
 oldadoption=CODE/'_reports/public_medical_backbone_20260916_v1/adoption_status.json',
 protocol=R/'TRACK1_TARGET_EVALUATION_INVENTORY_PROTOCOL_20260917.md',script=Path(__file__))
 # Preserve both recent and older frozen source contracts, not just the roster.
 frozen={}
 for p in [CODE/'_reports/private_signal_20260917_v1/contract.json']+[CODE/f'_reports/medical_head_20260916_v1/{n}.json' for n in ['contract','generation_contract','evaluation_contract']]:
  for q,h in read(p)['source_sha256'].items():require(q not in frozen or frozen[q]==h,'Source conflict');frozen[q]=h
 for rel,h in read(files['plan'])['files'].items():frozen[str(R/rel)]=h
 for p,h in frozen.items():require(sha(p)==h,'Old source changed '+p)
 sources={str(p):sha(p) for p in files.values()}
 save(OUT/'contract.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources={k:str(v) for k,v in files.items()},source_sha256=sources,
 frozen_sha256=frozen,labels=LABELS,new_model_inference=False,new_GPU=False,new_generation=False,roles_changed=False))
 inv=rows(files['inventory']);meta={x['Image Index']:x for x in rows(files['metadata'])};require(len(inv)==42423,'Inventory size')
 train=set(files['train_list'].read_text().splitlines());test=set(files['test_list'].read_text().splitlines())
 pats={x['patient_id']:x for x in rows(files['patients'])};pub=[x for x in inv if x['partition']=='public_development']
 for x in inv:
  m=meta[x['image_id']];require(str(int(m['Patient ID']))==x['patient_id'] and set(m['Finding Labels'].split('|'))==set(x['finding_labels'].split('|')) and m['View Position']=='PA','Inventory metadata mismatch')
 ev=rows(files['evaluation']);aux=rows(files['auxiliary']);bb=rows(files['backbone']);pf=rows(files['preflight']);ref=read(files['reference'])
 cv=ids(ev+aux);bp=ids([x for x in bb if x['backbone_role']=='train']);reserved=ids([x for x in bb if x['backbone_role']!='train'])
 history=ids(rows(files['history']));previous_reference=ids(pf)|ids(ref)
 modelsets={n:ids(rows(files[n])) for n in ['model1','model2']}
 require(all(x<=cv for x in modelsets.values()),'Prior M1/M2 patients outside excluded CVPR union')
 current_excluded=cv|bp
 stages={
 'public_all':pub,
 'after_current_training_and_all_cvpr_roles':[x for x in pub if x['patient_id'] not in current_excluded],
 'after_all_prior_reference_and_recorded_diagnostics':[x for x in pub if x['patient_id'] not in current_excluded|previous_reference|history],
 'strict_unassigned_public':[x for x in pub if x['patient_id'] not in current_excluded|previous_reference|history|reserved],
 'previously_evaluated_remaining_public':[x for x in pub if x['patient_id'] not in current_excluded and x['patient_id'] in previous_reference],
 'reserved_nontraining_public':[x for x in pub if x['patient_id'] in reserved],
 }
 summary={k:counts(v) for k,v in stages.items()}
 role_counts={}
 for role in ['selection','confirmation','reserve']:
  pp=ids([x for x in bb if x['backbone_role']==role]);role_counts[role]=counts([x for x in pub if x['patient_id'] in pp])
 patient_sets={k:sorted(ids(v),key=int) for k,v in stages.items()}
 # No selection of images by score or visual quality. Verify every current-disjoint public file.
 byhash=defaultdict(set);audit={x['image_id']:x for x in rows(files['audit'])};bypixel=defaultdict(set)
 for x in inv:byhash[x['sha256'].lower()].add(x['patient_id'])
 for x in audit.values():bypixel[x['pixel_sha256']].add(x['patient_id'])
 file_checks=[];root=DATA/'raw/nih_cxr14_pa_k10_plus_census_v1/images'
 for x in stages['after_current_training_and_all_cvpr_roles']:
  p=root/x['image_id'];require(p.is_file() and p.stat().st_size==int(x['bytes']),'Missing/size mismatch')
  h=sha(p);require(h==x['sha256'].lower()==audit[x['image_id']]['sha256'].lower(),'Image SHA mismatch')
  with Image.open(p) as im:im.load();size=im.size;mode=im.mode;require(size==(int(x['width']),int(x['height'])) and im.format=='PNG','Decode mismatch')
  file_checks.append(dict(image_id=x['image_id'],patient_id=x['patient_id'],sha256=h,size=list(size),mode=mode,
   cross_patient_exact_bytes=len(byhash[h])>1,cross_patient_prior_pixels=len(bypixel[audit[x['image_id']]['pixel_sha256']])>1))
 # Undownloaded metadata must be outside all existing patient roles AND official test patients.
 testp={str(int(m['Patient ID'])) for i,m in meta.items() if i in test}
 outside=[]
 for iid,m in meta.items():
  p=str(int(m['Patient ID']))
  if iid in train and m['View Position']=='PA' and p not in pats and p not in testp:
   outside.append(dict(image_id=iid,patient_id=p,finding_labels=m['Finding Labels'],local_file=(root/iid).is_file()))
 # A path may be locally present only if no ownership source conflicts: report, never adopt.
 allocated_partition={k:counts([x for x in inv if x['partition']==k]) for k in sorted({x['partition'] for x in inv})}
 locked={role:len(ids([x for x in ev if x['eval_role']==role])) for role in ['calibration','test']}
 # Parse valid labels from installed source without importing/loading any classifier.
 source=Path.home()/'anaconda3/Lib/site-packages/torchxrayvision/models.py';tree=ast.parse(source.read_text(encoding='utf-8'))
 models={}
 for node in tree.body:
  if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Dict):continue
  target=node.targets[0]
  if not isinstance(target,ast.Subscript) or not isinstance(target.value,ast.Name) or target.value.id!='model_urls':continue
  key=ast.literal_eval(target.slice);vals={ast.literal_eval(k):v for k,v in zip(node.value.keys,node.value.values)}
  if key not in ['pc','nih','all','mimic_ch','chex']:continue
  labels=ast.literal_eval(vals['labels']);url=ast.literal_eval(vals['weights_url']);p=Path.home()/'.torchxrayvision/models_data'/url.rsplit('/',1)[-1]
  models[key]=dict(valid_labels=labels,target_support={k:k in labels for k in LABELS},weights_path=str(p),exists=p.is_file(),
   bytes=p.stat().st_size if p.is_file() else None,sha256=sha(p) if p.is_file() else None,source_url=url)
 save(OUT/'file_checks.json',file_checks);save(OUT/'patient_sets_private.json',patient_sets);save(OUT/'outside_metadata_private.json',outside)
 result=dict(status='INVENTORY_COMPLETED_NO_REFERENCE_ADOPTED',created_utc=datetime.now(timezone.utc).isoformat(),
  sources_sha256=sources,inventory=counts(inv),partitions=allocated_partition,stages=summary,reserved_roles=role_counts,
  excluded=dict(all_cvpr_patients=len(cv),public_backbone_train=len(bp),old_preflight_patients=len(ids(pf)),generation_reference=len(ids(ref)),
   model1_train_patients=len(modelsets['model1']),model2_train_patients=len(modelsets['model2']),both_model_training_subsets_of_cvpr=True,
   locked_cvpr=locked),
  outside_existing_allocation_trainval_PA=dict(**counts(outside),local_images=sum(x['local_file'] for x in outside)),
  file_verification=dict(images=len(file_checks),all_decoded=True,all_SHA256_match=True,
   cross_patient_byte_duplicates=sum(x['cross_patient_exact_bytes'] for x in file_checks),
   cross_patient_prior_pixel_duplicates=sum(x['cross_patient_prior_pixels'] for x in file_checks),near_duplicate_search=False),
  evaluator=dict(installed_source=str(source),source_sha256=sha(source),candidates=models,
   nominated_candidate='pc',reason='PadChest-only training source, both target outputs trained; no NIH training by declared source.',
   actual_classifier_load=False,condition_sensitivity_tested=False,preprocessing='grayscale, 8bit to [-1024,1024], center crop, resize224; freeze output semantics before execution',
   raw_cosine_alone_insufficient=True),
  roles_changed=False,new_GPU=0,new_generation=0,new_model_inference=0,reference_adopted=False,
  seconds=time.perf_counter()-start,output_sha256={n:sha(OUT/n) for n in ['file_checks.json','patient_sets_private.json','outside_metadata_private.json']})
 for p,h in sources.items():require(sha(p)==h,'Input mutated '+p)
 save(OUT/'inventory.json',result)
 print(json.dumps({k:result[k] for k in ['status','stages','reserved_roles','outside_existing_allocation_trainval_PA','file_verification','seconds']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
