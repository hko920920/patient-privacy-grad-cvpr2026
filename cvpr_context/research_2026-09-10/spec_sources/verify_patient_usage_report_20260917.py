"""Bind completed actual-use audit, old evidence, reports and shared state."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent.parent;CODE=R.parent.parent/'code_working';P=CODE/'_reports/patient_usage_audit_20260917_v1'
sys.path.insert(0,str(R))
from verify_review import static_check
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 a=read(P/'audit.json');v=read(P/'verification.json');d=read(P/'research_decision.json');c=read(P/'contract.json')
 for n,h in d['source_sha256'].items():need(sha(P/n)==h,'Decision binding')
 for p,h in c['source_sha256'].items():need(sha(p)==h,'Audit source changed '+p)
 need(v['audit_sha256']==sha(P/'audit.json') and v['contract_sha256']==sha(P/'contract.json'),'Independent result binding')
 need(d['candidate_patients']==a['candidate_private_train']['patients']==8426,'Candidate count')
 need(not d['private_generation_utility_established'] and not d['previous192_utility_gate_passed'] and not d['original_roles_changed'],'Evidence limits')
 states=[read(R/n) for n in ['research_state.json','patient_baseline_spec.json']]
 keys=['current_step','next_task','next_task_output','step2_open_items','current_result_report','current_planning_report','active_execution']
 for k in keys:need(states[0][k]==states[1][k],'State mismatch '+k)
 need(states[0]['current_result_report']==d['report'] and states[0]['active_execution']['patient_usage_audit']==d,'Latest result pointers')
 need(states[0]['active_execution']['status']=='no_running_execution','Active run status')
 for p,h in read(R/'spec_sources/public_medical_backbone_plan_20260916_v1/plan_lock.json')['files'].items():need(sha(R/p)==h,'Frozen public plan')
 prior=CODE/'_reports/target_evaluation_inventory_20260917_v2'
 for p,h in read(prior/'contract.json')['frozen_sha256'].items():need(sha(p)==h,'Earlier frozen source')
 old=CODE/'_reports/medical_head_20260916_v1'
 for row in read(old/'generation_manifest.json'):need(sha(old/row['image_path'])==row['image_sha256'],'Old generation unchanged')
 page_text=(R/d['report']).read_text(encoding='utf-8')
 for t in ['8,426','273','559','**135**|**268**','**138**|**291**','100,948','15명·기흉45명','조사한 로컬 실행 기록상 미사용']:
  need(t in page_text,'Report fact '+t)
 work=(CODE.parent/'WORKLOG.md').read_text(encoding='utf-8');block=work[work.index('## 132-'):work.index('## 131-')]
 need('한글 저장 인코딩 손상' in block and '- ?' not in block,'Previous worklog repair')
 links=0
 for n in ['track1_patient_usage_audit_protocol.html','track1_patient_usage_audit_results.html']:
  text=(R/n).read_text(encoding='utf-8');need('\ufffd' not in text,'UTF-8 HTML')
  for tag in BeautifulSoup(text,'html.parser').find_all(['a','img']):
   u=urlsplit(tag.get('href',tag.get('src','')))
   if u.scheme or u.netloc or not u.path:continue
   need((R/unquote(u.path)).resolve().exists(),'Broken link '+u.path);links+=1
 result=dict(status='PASS_PATIENT_USAGE_REPORT_AND_STATE',created_utc=datetime.now(timezone.utc).isoformat(),
  existing_HTML_check=static_check(),new_local_links=links,broken_links=0,source_files=len(c['source_sha256']),
  independent_checks=v['checks'],old192_generated_images_unchanged=True,frozen_original_roles_preserved=True,
  state_fields=keys,new_GPU=0,new_generation=0,worklog132_encoding_restored=True,
  report_sha256=sha(R/d['report']),decision_sha256=sha(P/'research_decision.json'),verifier_sha256=sha(__file__))
 (R/'spec_sources/patient_usage_report_verification_20260917.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
