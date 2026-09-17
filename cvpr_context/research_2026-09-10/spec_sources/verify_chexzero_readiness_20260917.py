"""Independently verify the reserved cohort and pinned official candidate assets."""
import json,csv,hashlib,re,time
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from bs4 import BeautifulSoup
import openpyxl
R=Path(__file__).resolve().parents[1];C=R.parent.parent/'code_working';P=R/'spec_sources/chexzero_review_20260917_v1';D=C/'_reports/chexzero_evaluator_plan_20260917_v1';OLD=C/'_reports/padchest_validation_20260917_v1';A=C/'_reports/patient_usage_audit_20260917_v1';checks=0
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rr(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ck(ok,msg):
 global checks;checks+=1
 if not ok:raise ValueError(msg)
def main():
 t=time.perf_counter();con=js(D/'cohort_contract.json');inspection=js(P/'asset_inspection.json');download=js(P/'checkpoint_acquisition.json');release=js(P/'official_release.json')
 for p,h in con['source_sha256'].items():ck(sha(p)==h,'Cohort source '+p)
 for n,h in con['output_sha256'].items():ck(sha(D/n)==h,'Prepared manifest')
 for r in js(P/'acquisition.json')['files']:
  if 'sha256' in r:ck(sha(P/r['path'])==r['sha256'],'Official source acquisition')
 parsed={}
 for e in BeautifulSoup((P/'official_drive.html').read_text(encoding='utf-8'),'html.parser').select('[data-id]'):
  m=re.search(r'best_[A-Za-z0-9_.-]+\.pt',e.get_text(' ',strip=True))
  if m:parsed[e['data-id']]=m.group(0)
 ck(len(parsed)==10 and parsed=={x['id']:x['filename'] for x in release['files']},'Entire official release; no subset selection')
 ck({x['filename'] for x in download['files']}==set(parsed.values()),'Downloaded release members')
 for a,b in zip(download['files'],inspection['files']):
  ck(a['filename']==b['filename'] and a['sha256']==b['sha256']==sha(a['path']),'Actual checkpoint bytes')
  ck(b['strict_load'] and b['all_parameter_values_preserved'] and b['all_tensors_finite'],'Strict load record')
  ck(b['image_resolution']==224 and b['patch_size']==[32,32] and b['context_length']==77 and b['embedding_dimension']==512,'Released ViT-B32 architecture')
 ck(inspection['model_forward_calls']==inspection['GPU_calls']==inspection['NIH_pixels_read']==0,'Review is not inference')
 ck(inspection['acquisition_sha256']==sha(P/'checkpoint_acquisition.json'),'Inspection acquisition binding')
 # Original local labels, independently regrouped (do not import producer).
 by=defaultdict(list)
 for x in rr(C/'_data/raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv'):by[x['patient_id']].append(x)
 def category(p):
  ss=[set(x['finding_labels'].split('|')) for x in by[p]];u=set.union(*ss)
  if {'Emphysema','Pneumothorax'}<=u:return 'dual'
  for label in ['Emphysema','Pneumothorax','Effusion']:
   if label in u:return label
  return 'No Finding' if {'No Finding'} in ss else 'other'
 def order(k,x):return hashlib.sha256((con['salt']+'|'+k+'|'+x).encode()).hexdigest()
 previous={x['patient_id'] for x in rr(OLD/'remaining_development_private.csv')};consumed={x['patient_id'] for x in rr(OLD/'selected_images_private.csv')}
 reserved={x['patient_id'] for x in rr(A/'provisional_patient_split_private.csv') if x['provisional_role']=='reserved_confirmation'}
 selection=rr(D/'selected_images_private.csv');expected=set()
 for g in ['Emphysema','Pneumothorax','No Finding','Effusion']:
  pp=sorted((p for p in previous if category(p)==g),key=lambda p:order('patient',p))[:20]
  observed={x['patient_id'] for x in selection if x['condition']==g};ck(set(pp)==observed and len(pp)==20,'Prospective group members');expected.update(pp)
  for x in selection:
   if x['condition']!=g:continue
   eligible=[v for v in by[x['patient_id']] if (set(v['finding_labels'].split('|'))=={'No Finding'} if g=='No Finding' else g in v['finding_labels'].split('|'))]
   one=min(eligible,key=lambda v:order('image',v['image_id']))
   ck(one['image_id']==x['image_id'] and one['sha256'].lower()==x['sha256'],'Image selection/acquisition SHA from metadata')
   ck(x['patient_hash']==order('patient',x['patient_id']) and x['image_hash']==order('image',x['image_id']),'Selection hash')
 ck(len(selection)==len(expected)==80 and not (expected&consumed or expected&reserved),'Disjoint evaluator-only cohort')
 rem=rr(D/'remaining_method_development_private.csv');ck({x['patient_id'] for x in rem}==previous-expected,'Remaining cohort')
 for x in rem:ck(x['group']==category(x['patient_id']),'Remaining patient category')
 ck(Counter(x['group'] for x in rem)==con['remaining_counts'],'Remaining totals')
 ck(con['remaining_counts']['Emphysema']==44 and con['remaining_method_development']==4053,'Resource cost')
 ck(not con['selected_model_result_consumed'] and not con['original_roles_changed'],'No actual validation or original-role mutation')
 # Verify original experiment sources/192images and failed evaluator unchanged.
 for key in ['source_sha256','frozen_sha256']:
  for p,h in js(OLD/'contract.json')[key].items():ck(sha(p)==h,'Previous experiment frozen evidence')
 for n,h in js(OLD/'result.json')['artifact_sha256'].items():ck(sha(OLD/n)==h,'Previous validation artifact')
 ck(not js(OLD/'metrics.json')['joint_pass'],'Previous evaluator remains failed')
 for p,h in js(A/'contract.json')['source_sha256'].items():ck(sha(p)==h,'Old audit evidence')
 # Source-data rows, no OCR or confusion with subcutaneous emphysema.
 workbook=openpyxl.load_workbook(P/'figure3_source.xlsx',read_only=True,data_only=True)
 values={row[0]:list(row[1:]) for row in list(workbook.active.values)[1:]}
 ck(values['emphysema_auc']==[.8232,.8018,.8428,376],'External E source evidence')
 ck(values['pneumothorax_auc']==[.7659,.7187,.8071,98],'External P source evidence')
 ck(values['subcutaneous emphysema_auc']!=values['emphysema_auc'],'Different emphysema labels')
 result=dict(status='PASS_CANDIDATE_READINESS_NOT_PERFORMANCE',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,seconds=time.perf_counter()-t,
  checkpoint_count=10,total_checkpoint_bytes=inspection['total_bytes'],fresh_reserved_evaluator_patients=80,remaining_method_development=4053,
  remaining_exclusive_emphysema=44,all_previous_outputs_preserved=True,new_NIH_pixel_access=0,new_model_inference=0,new_DP=0,
  external_padchest_source_values={k:values[k] for k in ['emphysema_auc','pneumothorax_auc','subcutaneous emphysema_auc']},
  bound_sha256={str(p):sha(p) for p in [D/'cohort_contract.json',P/'asset_inspection.json',P/'official_release.json',P/'checkpoint_acquisition.json',P/'figure3_source.xlsx']},
  verifier_sha256=sha(__file__))
 with (P/'readiness_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
