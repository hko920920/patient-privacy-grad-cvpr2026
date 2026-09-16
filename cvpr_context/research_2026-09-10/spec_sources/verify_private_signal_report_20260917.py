"""Final document/source binding for the private-signal CPU audit."""
import csv,hashlib,json,sys
from pathlib import Path
from urllib.parse import urlsplit,unquote
from datetime import datetime,timezone
import numpy as np
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent.parent
CODE=R.parent.parent/'code_working'
P=CODE/'_reports/private_signal_20260917_v1'
OLD=CODE/'_reports/medical_head_20260916_v1'
sys.path.insert(0,str(R))
from verify_review import static_check
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 oldstatic=static_check();sources={}
 for directory,names in [(P,['contract.json']),(OLD,['contract.json','generation_contract.json','evaluation_contract.json'])]:
  for name in names:
   for p,h in read(directory/name)['source_sha256'].items():
    require(p not in sources or sources[p]==h,'Source expectation conflict');sources[p]=h
 for p,h in sources.items():require(sha(p)==h,'Frozen source '+p)
 for supp in ['all_label_development_exploratory.json','public_backbone_label_context.json']:
  for p,h in read(P/supp)['source_sha256'].items():require(sha(p)==h,'Supplemental source '+p)
 a=read(P/'analysis.json');v=read(P/'verification.json');d=read(P/'research_decision.json')
 require(v['complete'] and v['analysis_sha256']==sha(P/'analysis.json'),'Complete bound verification')
 for f,h in d['source_sha256'].items():require(sha(P/f)==h,'Decision artifact '+f)
 states=[read(R/f) for f in ['research_state.json','patient_baseline_spec.json']]
 keys=['current_step','next_task','next_task_output','step2_open_items','current_result_report','current_planning_report','active_execution']
 for key in keys:require(states[0][key]==states[1][key],'State equality '+key)
 require(states[0]['active_execution']['status']=='no_running_execution','Execution status')
 require(states[0]['active_execution']['private_signal_audit']==d,'State/decision')
 require(not d['private_generation_utility_established'] and not d['previous192_utility_gate_passed'] and d['new_GPU']==0,'Evidence boundaries')
 text=(R/d['report']).read_text(encoding='utf-8')
 for name in ['public','private','pooled','private_two_images']:require(format(a['development'][name]['mean_MSE'],'.10f') in text,'Report MSE '+name)
 require(format(100*a['weight_comparisons']['private']['development_prediction']['delta_nonscalar_energy_fraction'],'.2f') in text,'Nonscalar fraction')
 # Independently re-count public backbone and unused-role label claims from the existing roster.
 roster=R/'spec_sources/public_medical_backbone_plan_20260916_v1/images.csv'
 with roster.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 back=read(P/'public_backbone_label_context.json');available=read(P/'public_unused_role_availability.json')
 for label in ['Emphysema','Pneumothorax']:
  tr=[x for x in rows if x['backbone_role']=='train' and label in x['finding_labels'].split('|')]
  require(len(tr)==back['labels'][label]['images'],'Public backbone label count')
  require(len({x['patient_id'] for x in tr})==back['labels'][label]['patients'],'Public backbone patient count')
 for role,counts in available['roles'].items():
  rr=[x for x in rows if x['backbone_role']==role]
  require(len(rr)==counts['images'] and len({x['patient_id'] for x in rr})==counts['patients'],'Role inventory')
  for label,vals in counts['labels'].items():
   picked=[x for x in rr if label in x['finding_labels'].split('|')]
   require(len(picked)==vals['images'] and len({x['patient_id'] for x in picked})==vals['patients'],'Role label inventory')
 # All primary patient metadata aggregates are consistent with the saved source-bound rows.
 pm=read(P/'patient_metadata.json')
 for sp,c in a['cohorts'].items():
  rr=[x for x in pm if x['split']==sp]
  require(abs(np.mean([x['age'] for x in rr])-c['age']['mean'])<1e-12,'Age summary')
  for label,pct in c['weak_label_patient_mean_image_fraction'].items():
   require(abs(np.mean([x['label_fractions'][label] for x in rr])-pct)<1e-12,'Label fraction')
 old_adopt=OLD.parent/'public_medical_backbone_20260916_v1/adoption_status.json'
 require(sha(old_adopt)=='d08ecc95f0a1150c6eaaaacdd53b8406518e64d1027b3495da32b88d447cc4bd','Old failed adoption changed')
 plan=read(R/'spec_sources/public_medical_backbone_plan_20260916_v1/plan_lock.json')
 for p,h in plan['files'].items():require(sha(R/p)==h,'Old plan changed')
 for row in read(OLD/'generation_manifest.json'):require(sha(OLD/row['image_path'])==row['image_sha256'],'Prior192 image changed')
 links=0
 for page in ['track1_private_signal_protocol.html','track1_private_signal_results.html']:
  body=(R/page).read_text(encoding='utf-8');require('\ufffd' not in body,'Encoding')
  soup=BeautifulSoup(body,'html.parser')
  for tag in soup.find_all(['a','img']):
   u=urlsplit(tag.get('href',tag.get('src','')))
   if not u.path or u.scheme or u.netloc:continue
   require((R/unquote(u.path)).resolve().exists(),'Broken link '+u.path);links+=1
 result=dict(status='PASS_PRIVATE_SIGNAL_REPORT_BINDINGS',created_utc=datetime.now(timezone.utc).isoformat(),
  existing_HTML_check=oldstatic,new_local_links=links,new_broken_links=0,frozen_source_files=len(sources),
  prior_plan_files=len(plan['files']),prior_generation_images_unchanged=192,old_failed_adoption_unchanged=True,
  state_fields=keys,independent_analysis_checks=v['checks'],new_GPU=0,new_generation=0,
  report_sha256=sha(R/d['report']),decision_sha256=sha(P/'research_decision.json'),verifier_sha256=sha(__file__))
 (R/'spec_sources/private_signal_report_verification_20260917.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

