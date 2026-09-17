"""Reserve a fresh evaluator-only cohort by metadata, never decode patient images."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];C=R.parent.parent/'code_working';P=R/'spec_sources/chexzero_review_20260917_v1';D=C/'_reports/chexzero_evaluator_plan_20260917_v1';OLD=C/'_reports/padchest_validation_20260917_v1';AUD=C/'_reports/patient_usage_audit_20260917_v1';RAW=C/'_data/raw/nih_cxr14_pa_k10_plus_census_v1';SALT='cvpr-chexzero-evaluator-20260917-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def table(p,rr):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def need(x,m):
 if not x:raise ValueError(m)
def main():
 D.mkdir(exist_ok=False);sources={}
 def bind(p):p=Path(p).resolve();sources[str(p)]=sha(p);return p
 def rr(p):return rows(bind(p))
 bind(__file__);bind(R/'TRACK1_CHEXZERO_REVIEW_PROTOCOL_20260917.md')
 source=rr(OLD/'remaining_development_private.csv');old={x['patient_id'] for x in rr(OLD/'selected_images_private.csv')}
 split=rr(AUD/'provisional_patient_split_private.csv');reserved={x['patient_id'] for x in split if x['provisional_role']=='reserved_confirmation'}
 ledger={x['patient_id']:x for x in rr(AUD/'patient_usage_private.csv')}
 inv=rr(RAW/'content_inventory_private.csv');official={x['Image Index']:x for x in rr(C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 by=defaultdict(list)
 for x in inv:by[x['patient_id']].append(x)
 def group(p):
  u=set().union(*(set(x['finding_labels'].split('|')) for x in by[p]))
  if 'Emphysema' in u:return 'dual' if 'Pneumothorax' in u else 'Emphysema'
  if 'Pneumothorax' in u:return 'Pneumothorax'
  if 'Effusion' in u:return 'Effusion'
  return 'No Finding' if any(x['finding_labels']=='No Finding' for x in by[p]) else 'other'
 pp={x['patient_id'] for x in source};need(len(pp)==4133 and not (pp&old or pp&reserved),'Pool isolation')
 for x in source:need(group(x['patient_id'])==x['group'],'Prior exclusive group')
 def order(k,v):return hashlib.sha256(f'{SALT}|{k}|{v}'.encode()).hexdigest()
 selected=[]
 for g in ['Emphysema','Pneumothorax','No Finding','Effusion']:
  patients=sorted((p for p in pp if group(p)==g),key=lambda p:order('patient',p))[:20];need(len(patients)==20,'Support')
  for p in patients:
   l=ledger[p];need(l['audit_grade']=='D' and l['original_thesis_role']=='private_train','Recorded use')
   need(not any(int(l[k]) for k in l if k.startswith('used_in_') or k.startswith('locked_')),'Recorded prior exposure')
   eligible=[x for x in by[p] if (x['finding_labels']=='No Finding' if g=='No Finding' else g in x['finding_labels'].split('|'))]
   x=min(eligible,key=lambda x:order('image',x['image_id']));o=official[x['image_id']]
   need(str(int(o['Patient ID']))==p and set(o['Finding Labels'].split('|'))==set(x['finding_labels'].split('|')),'Official metadata')
   path=RAW/'images'/x['image_id'];need(path.exists() and path.stat().st_size==int(x['bytes']),'File available; no pixel access')
   selected.append(dict(patient_id=p,condition=g,image_id=x['image_id'],sha256=x['sha256'].lower(),finding_labels=x['finding_labels'],patient_hash=order('patient',p),image_hash=order('image',x['image_id']),role='reserved_chexzero_evaluator_only',model_result_consumed=False))
 used={x['patient_id'] for x in selected};need(len(used)==80 and not(used&old or used&reserved),'New-only selected patients')
 table(D/'selected_images_private.csv',selected)
 table(D/'reserved_evaluator_patients_private.csv',[dict(patient_id=p,role='reserved_chexzero_evaluator_only',model_result_consumed=False) for p in sorted(used,key=int)])
 rem=[dict(patient_id=p,group=group(p),role='remaining_target_method_development_candidate') for p in sorted(pp-used,key=int)]
 table(D/'remaining_method_development_private.csv',rem);counts=dict(Counter(x['group'] for x in rem));need(counts['Emphysema']==44,'Remaining E-only')
 contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),salt=SALT,source_sha256=sources,prior_development=4133,selected=80,selected_per_group=20,
  remaining_method_development=4053,remaining_counts=counts,old_evaluator_patient_overlap=0,reserved_confirmation_overlap=0,new_model_inference=0,new_NIH_pixel_access=0,
  proposed_later_generation_reference_per_condition=32,earlier_64_per_condition_no_longer_feasible_in_remaining_development=True,
  selection_outcome_independent=True,original_roles_changed=False,selected_model_result_consumed=False,
  output_sha256={n:sha(D/n) for n in ['selected_images_private.csv','reserved_evaluator_patients_private.csv','remaining_method_development_private.csv']})
 save(D/'cohort_contract.json',contract);print(json.dumps(contract,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
