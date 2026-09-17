"""Independent population, actual-private-use and deterministic split verification."""
import csv,hashlib,json,time,re
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
CODE=Path(__file__).resolve().parents[1];P=CODE/'_reports/patient_usage_audit_20260917_v1'
COUNT=0
def check(ok,msg):
 global COUNT
 COUNT+=1
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rr(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 start=time.perf_counter();a=read(P/'audit.json');contract=read(P/'contract.json');ledger=rr(P/'patient_usage_private.csv')
 for p,h in contract['source_sha256'].items():check(sha(p)==h,'Source changed '+p)
 for n,h in a['outputs_sha256'].items():check(sha(P/n)==h,'Output changed '+n)
 inv=rr(CODE/'_data/raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv');index={x['image_id']:x for x in inv}
 meta={x['Image Index']:x for x in rr(CODE/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 original={int(x['patient_id']):x['partition'] for x in rr(CODE/'_data/derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv')}
 counts=Counter();by=defaultdict(list)
 for x in inv:
  m=meta[x['image_id']];check(int(x['patient_id'])==int(m['Patient ID']),'Image patient identity');by[int(x['patient_id'])].append(m)
 rt=read(CODE/'_reports/nih_cxr14_k5_private_research_dryrun_v1_001/restricted_runtime_diagnostics.json')
 dry=set();di=set()
 for arm in rt['arms'].values():
  for step in arm:
   for u in step['units']:
    dry.add(int(u['patient_id']));di.update(u['image_ids'])
    check(all(int(index[i]['patient_id'])==int(u['patient_id']) for i in u['image_ids']),'Runtime patient/image identity')
 check(len(dry)==50 and len(di)==59,'Runtime coverage, independent flattening')
 pool={int(x['patient_id']) for x in ledger if x['eligible_for_cvpr_confirmation']=='1'}
 expected={p for p,r in original.items() if r=='private_train'}-dry
 check(pool==expected,'All and only original-private patients outside actual dryrun')
 evidence=read(P/'patient_evidence_private.json');check(all(str(p) not in evidence for p in pool),'No recorded candidate result/training evidence')
 check(len(ledger)==len(by) and len({x['patient_id'] for x in ledger})==len(ledger),'One row per local patient')
 cv=CODE/'_reports/cvpr_u_pilot_v1_001/cohort'
 forbidden={int(x['patient_id']) for x in rr(cv/'evaluation_images.csv')+rr(cv/'auxiliary_images.csv')}
 R=Path(contract['source_protocol']).parent
 bb=rr(R/'spec_sources/public_medical_backbone_plan_20260916_v1/images.csv')
 forbidden|={int(x['patient_id']) for x in bb if x['backbone_role']=='train'}
 forbidden|={int(x['patient_id']) for x in rr(CODE/'_data/derived/nih_cxr14_pa_target_enriched_v1/official_pa_test_census_private.csv')}
 for n in ['model_1_train.csv','model_2_train.csv']:forbidden|={int(x['patient_id']) for x in rr(cv/n)}
 check(not pool&forbidden,'Candidate disjoint from actual models/current/locked roles')
 cfields=['emphysema_images','pneumothorax_images','normal_control_images','effusion_control_images']
 def ccount(ims):
  out=[0]*4
  for m in ims:
   labs=set(m['Finding Labels'].split('|'));out[0]+='Emphysema' in labs;out[1]+='Pneumothorax' in labs;out[2]+=labs=={'No Finding'};out[3]+='Effusion' in labs and not labs&{'Emphysema','Pneumothorax'}
  return out
 for x in ledger:
  p=int(x['patient_id']);check([int(x[f]) for f in cfields]==ccount(by[p]),'Per-patient metadata label counts');counts[x['audit_grade']]+=1
  check((x['used_in_thesis_dryrun_training']=='1')==(p in dry),'Dryrun flag on every patient')
 for grade,n in counts.items():check(n==a['grades'][grade]['patients'],'Grade count')
 # No production sorting helper imported. Same predeclared byte-level key, numeric ID text.
 salt=contract['salt'];order=sorted(pool,key=lambda p:hashlib.sha256((salt+'|patient|'+str(p)).encode('utf-8')).digest())
 cut=(len(order)+1)//2;assign={p:('target_development' if i<cut else 'reserved_confirmation') for i,p in enumerate(order)}
 split=rr(P/'provisional_patient_split_private.csv');check({int(x['patient_id']):x['provisional_role'] for x in split}==assign,'Complete deterministic split')
 selected=rr(P/'provisional_condition_images_private.csv');seen=set()
 for x in selected:
  p=int(x['patient_id']);condition=x['condition'];check(assign[p]==x['provisional_role'],'Patient cannot cross roles')
  options=[]
  for m in by[p]:
   labs=set(m['Finding Labels'].split('|'))
   ok=condition in labs
   if condition=='No Finding':ok=labs=={'No Finding'}
   if condition=='Effusion':ok='Effusion' in labs and not labs&{'Emphysema','Pneumothorax'}
   if ok:options.append(m['Image Index'])
  expected_image=min(options,key=lambda i:hashlib.sha256((salt+'|image|'+i).encode()).digest())
  check(expected_image==x['image_id'],'Hash-selected image, no visual selection')
  check((p,condition) not in seen,'One image per condition/patient');seen.add((p,condition))
 for role,v in a['provisional_splits'].items():
  patients=[p for p in pool if assign[p]==role];check(len(patients)==v['patients'],'Split patients')
  vals=[ccount(by[p]) for p in patients]
  for i,f in enumerate(cfields):check(sum(c[i]>0 for c in vals)==v['label_patients'][f] and sum(c[i] for c in vals)==v['label_images'][f],'Split labels')
 # Audit text scan is an evidence search, not an unexplained claim that the whole disk was read.
 scan=read(P/'artifact_scan.json');check(len(scan)==a['artifact_scan_files'],'Scan registry')
 check(not read(P/'unresolved_private_hits.json'),'No new unidentified private-use hit')
 check(not (CODE/'_restricted_runs').exists(),'No declared full-training runtime root')
 old=CODE/'_reports/medical_head_20260916_v1'
 for x in read(old/'generation_manifest.json'):check(sha(old/x['image_path'])==x['image_sha256'],'Old192 image unchanged')
 check(sha(CODE/'_reports/public_medical_backbone_20260916_v1/adoption_status.json')=='d08ecc95f0a1150c6eaaaacdd53b8406518e64d1027b3495da32b88d447cc4bd','Old failed adoption unchanged')
 v=dict(status='PASS_PATIENT_USAGE_AND_PROVISIONAL_SPLIT',checks=COUNT,created_utc=datetime.now(timezone.utc).isoformat(),
  audit_sha256=sha(P/'audit.json'),contract_sha256=sha(P/'contract.json'),verifier_sha256=sha(__file__),
  candidate_patients=len(pool),actual_dryrun_patients=len(dry),condition_rows_verified=len(selected),old192_images_unchanged=True,
  seconds=time.perf_counter()-start,new_GPU=0,new_model_inference=0,
  limits='Reconstructs recorded use/exclusions; does not prove absence of unknown external executions, clinical label validity, power or generator efficacy.')
 with (P/'verification.json').open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
 print(json.dumps(v,indent=2))
if __name__=='__main__':main()
