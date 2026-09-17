"""Bind the completed inventory, local report/pages and current research state."""
import hashlib,json,sys
from pathlib import Path
from urllib.parse import urlsplit,unquote
from datetime import datetime,timezone
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent.parent
CODE=R.parent.parent/'code_working'
P=CODE/'_reports/target_evaluation_inventory_20260917_v2'
sys.path.insert(0,str(R))
from verify_review import static_check
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def main():
 d=read(P/'research_decision.json');a=read(P/'inventory.json');v=read(P/'verification.json');c=read(P/'contract.json')
 for n,h in d['source_sha256'].items():require(sha(P/n)==h,'Decision binding '+n)
 require(v['inventory_sha256']==sha(P/'inventory.json') and v['contract_sha256']==sha(P/'contract.json'),'Independent verification binding')
 for p,h in {**c['source_sha256'],**c['frozen_sha256']}.items():require(sha(p)==h,'Old/frozen source '+p)
 require(d['targets_after_current_exclusions']=={k:a['stages']['after_current_training_and_all_cvpr_roles']['labels'][k]['patients'] for k in ['Emphysema','Pneumothorax']},'Current target counts')
 require(d['targets_after_prior_reference_and_known_history_exclusions']=={k:a['stages']['after_all_prior_reference_and_recorded_diagnostics']['labels'][k]['patients'] for k in ['Emphysema','Pneumothorax']},'Strict target counts')
 require(not d['private_generation_utility_established'] and not d['previous192_utility_gate_passed'] and not d['roles_changed'],'Evidence boundary')
 states=[read(R/n) for n in ['research_state.json','patient_baseline_spec.json']]
 keys=['current_step','next_task','next_task_output','step2_open_items','current_result_report','current_planning_report','active_execution']
 for k in keys:require(states[0][k]==states[1][k],'State mismatch '+k)
 require(states[0]['current_result_report']==d['report'] and states[0]['active_execution']['target_reference_inventory']==d,'Result authority')
 require(states[0]['active_execution']['status']=='no_running_execution','Running state')
 text=(R/d['report']).read_text(encoding='utf-8')
 for token in ['424 / 511','264 / 264','157|180','2,124','32개','후속 가설을 검증할 자료가 부족','기흉|**0**|**0**']:
  require(token in text,'Report number/boundary '+token)
 old=CODE/'_reports/medical_head_20260916_v1'
 for x in read(old/'generation_manifest.json'):require(sha(old/x['image_path'])==x['image_sha256'],'Old192 image changed')
 require(sha(CODE/'_reports/public_medical_backbone_20260916_v1/adoption_status.json')=='d08ecc95f0a1150c6eaaaacdd53b8406518e64d1027b3495da32b88d447cc4bd','Old adoption changed')
 oldstatic=static_check();links=0
 for n in ['track1_target_evaluation_inventory_protocol.html','track1_target_evaluation_inventory_results.html']:
  body=(R/n).read_text(encoding='utf-8');require('\ufffd' not in body,'Encoding')
  for t in BeautifulSoup(body,'html.parser').find_all(['a','img']):
   u=urlsplit(t.get('href',t.get('src','')))
   if not u.path or u.scheme or u.netloc:continue
   require((R/unquote(u.path)).resolve().exists(),'Broken link '+u.path);links+=1
 require('track1_target_evaluation_inventory_results.html' in (R/'index.html').read_text(encoding='utf-8'),'Index pointer')
 result=dict(status='PASS_TARGET_INVENTORY_REPORT_BINDINGS',created_utc=datetime.now(timezone.utc).isoformat(),
  old_HTML_check=oldstatic,new_local_links=links,broken_links=0,state_fields=keys,
  independent_checks=v['checks'],actual_public_files_verified=511,prior_generated_images_unchanged=192,
  old_failed_adoption_preserved=True,new_GPU=0,new_generation=0,
  report_sha256=sha(R/d['report']),decision_sha256=sha(P/'research_decision.json'),verifier_sha256=sha(__file__))
 (R/'spec_sources/target_inventory_report_verification_20260917.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
