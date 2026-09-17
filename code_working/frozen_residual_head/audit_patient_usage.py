"""Evidence-linked patient usage ledger and provisional separate-study split. No ML."""
import csv,hashlib,json,re,time
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
CODE=Path(__file__).resolve().parents[1]
R=next(CODE.parent.glob('CVPR */research_2026-09-10'))
REPORTS=CODE/'_reports'
OUT=REPORTS/'patient_usage_audit_20260917_v1'
DATA=CODE/'_data'
COHORT=REPORTS/'cvpr_u_pilot_v1_001/cohort'
SALT='cvpr-target-usage-audit-20260917-v1'
FIELDS=['used_in_backbone_training','used_in_public_head','used_in_private_head','used_in_development_loss',
 'used_in_metric_reference','used_in_evaluator_preflight','used_in_visual_diagnostic',
 'used_in_checkpoint_or_hypothesis_selection','used_in_prior_M1_M2_training','used_in_thesis_dryrun_training',
 'used_in_other_disposable_training','used_in_attack_or_model_diagnostic','used_in_feature_cache',
 'unresolved_possible_use','cross_patient_exact_duplicate']
IMAGE=re.compile(r'(?<!\d)(\d{8}_\d{3}\.png)(?!\w)',re.I)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(p,rr):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def require(x,m):
 if not x:raise RuntimeError(m)
def key(kind,value):return hashlib.sha256(f'{SALT}|{kind}|{value}'.encode()).hexdigest()
def main():
 started=time.perf_counter();require(not OUT.exists(),'Existing run');OUT.mkdir()
 sources={};evidence=defaultdict(lambda:defaultdict(set))
 def bind(p):p=Path(p).resolve();sources[str(p)]=sha(p);return p
 def rr(p):return rows(bind(p))
 def js(p):return read(bind(p))
 bind(__file__);bind(R/'TRACK1_PATIENT_USAGE_AUDIT_PROTOCOL_20260917.md')
 inv=rr(DATA/'raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv');index={x['image_id']:x for x in inv}
 original={x['patient_id']:x['partition'] for x in rr(DATA/'derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv')}
 census={x['patient_id'] for x in rr(DATA/'derived/nih_cxr14_pa_target_enriched_v1/official_pa_test_census_private.csv')}
 meta={x['Image Index']:x for x in rr(DATA/'intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 bypatient=defaultdict(list)
 for x in inv:bypatient[x['patient_id']].append(x)
 pats=set(bypatient)
 def mark(pp,field,path):
  path=str(Path(path).resolve())
  for p in pp:
   p=str(int(p))
   if p in pats:evidence[p][field].add(path)
 def markrows(rrr,field,path):mark([x['patient_id'] for x in rrr],field,path)
 # Actual private runtime, not the declared full population.
 dp=REPORTS/'nih_cxr14_k5_private_research_dryrun_v1_001';dr=js(dp/'public_report.json');runtime=js(dp/'restricted_runtime_diagnostics.json')
 require(sha(dp/'restricted_runtime_diagnostics.json').upper()==dr['artifacts']['restricted_runtime_diagnostics_sha256'],'Dryrun runtime binding')
 units=[u for arm in runtime['arms'].values() for st in arm for u in st['units']]
 dryp={str(int(u['patient_id'])) for u in units};dryimages={iid for u in units for iid in u['image_ids']}
 require(len(dryp)==50 and len(dryimages)==runtime['input_summary']['unique_images']==59,'Actual private dryrun coverage')
 require(all(original[p]=='private_train' for p in dryp),'Dryrun partition')
 mark(dryp,'used_in_thesis_dryrun_training',dp/'restricted_runtime_diagnostics.json')
 gate=js(REPORTS/'nih_cxr14_k5_full_runner_gate_v1_001/independent_verification.json')
 bind(CODE/'dp_training/k5_full_runner.py');bind(CODE/'dp_training/run_k5_full_training.py');bind(CODE/'dp_training/run_xray_k5_private_research_dryrun.py')
 fullpaths=[CODE/'_restricted_runs',REPORTS/'nih_cxr14_k5_feasibility_v1_001/arms']
 require(not gate['full_K5_optimizer_execution_started'] and all(not p.exists() for p in fullpaths),'Full training exists: resolve coverage first')
 # M1/M2 completed coverage and public LoRA committed trace.
 modelcoverage={}
 for m in [1,2]:
  path=COHORT/f'model_{m}_train.csv';rrr=rr(path);d=REPORTS/f'cvpr_u_pilot_v1_001/training_coverage_v2/model_{m}'
  report=js(d/'report_1000.json');ver=js(d/'verification_1000.json');bind(d/'training_trace.jsonl')
  require(report['all_declared_member_patients_actually_used'] and ver['all_member_patients_used'] and report['unique_exposed_images']==len(rrr)==912,'M1/M2 runtime coverage')
  markrows(rrr,'used_in_prior_M1_M2_training',path);modelcoverage[str(m)]=len({x['patient_id'] for x in rrr})
 bp=REPORTS/'public_medical_backbone_20260916_v1';br=js(bp/'training_report.json');trace=bind(bp/'trace.jsonl')
 require(sha(trace)==br['trace_sha256'],'Backbone trace binding')
 seen=set()
 for line in trace.read_text(encoding='utf-8').splitlines():
  x=json.loads(line)
  if x['committed']:seen.update(x['image_ids'])
 require(len(seen)==749,'Backbone trace images');bpatients={index[i]['patient_id'] for i in seen};require(len(bpatients)==640,'Backbone trace patients')
 mark(bpatients,'used_in_backbone_training',trace);mark(bpatients,'used_in_checkpoint_or_hypothesis_selection',REPORTS/'private_signal_20260917_v1/public_backbone_label_context.json')
 bind(REPORTS/'private_signal_20260917_v1/public_backbone_label_context.json')
 headpath=REPORTS/'medical_head_20260916_v1/images.json';heads=js(headpath);extraction=js(REPORTS/'medical_head_20260916_v1/extraction.json')
 require(extraction['complete'] and extraction['records']==4352,'Head extraction incomplete')
 for sp,field in [('public','used_in_public_head'),('train','used_in_private_head'),('eval','used_in_development_loss')]:
  these=[x for x in heads if x['split']==sp];markrows(these,field,headpath);markrows(these,'used_in_checkpoint_or_hypothesis_selection',headpath)
 refpath=REPORTS/'medical_head_20260916_v1/reference.json';markrows(js(refpath),'used_in_metric_reference',refpath)
 efpath=REPORTS/'nih_cxr14_k5_evaluator_preflight_protocol_v1_001/real_evaluator_tasks_local.csv';oldrefs=rr(efpath)
 ef=js(REPORTS/'nih_cxr14_k5_evaluator_preflight_v1_001/public_report.json');require(ef['status']=='PASS_K5_EVALUATOR_PREFLIGHT','Evaluator not completed')
 markrows(oldrefs,'used_in_evaluator_preflight',efpath)
 ev=rr(COHORT/'evaluation_images.csv');aux=rr(COHORT/'auxiliary_images.csv');cache=js(REPORTS/'cvpr_u_pilot_v1_001/cache/summary.json')
 require(cache['status']=='PASS' and cache['images']==len(ev+aux),'Cache execution bound to full cohort')
 markrows(ev+aux,'used_in_feature_cache',COHORT/'evaluation_images.csv')
 locked_cvpr={x['patient_id'] for x in ev if x['eval_role'] in ['calibration','test']}
 # Public resume fixture is source-deterministic and has completed runtime reports.
 k5=rr(DATA/'derived/nih_cxr14_pa_target_enriched_v1/k5_private.csv');pubcounts=Counter(x['patient_id'] for x in k5 if x['partition']=='public_development')
 fixture=sorted(p for p,n in pubcounts.items() if n>=2)[:8]
 rp=REPORTS/'nih_cxr14_k5_actual_sd21_resume_preflight_v1_002/public_report.json';res=js(rp)
 bind(CODE/'dp_training/run_k5_actual_sd21_resume_preflight.py')
 require(res['fixture']['patients']==len(fixture) and not res['fixture']['private_train_records_loaded'],'Resume source fixture')
 mark(fixture,'used_in_other_disposable_training',rp)
 diagnostic_sources=[]
 for d in sorted(REPORTS.glob('nih_cxr14*')):
  for p in sorted(d.glob('selected_*.csv')):
   rrr=rr(p);diagnostic_sources.append(str(p.resolve()))
   for x in rrr:
    pp=[v for k,v in x.items() if (k=='patient_id' or k.endswith('_patient_id')) and v and v.isdigit()]
    if x.get('split')=='train' and 'setadapter_context' in d.name:mark(pp,'used_in_other_disposable_training',p)
    elif x.get('split')=='train' and 'cross_record_context' in d.name:mark(pp,'used_in_other_disposable_training',p)
    mark(pp,'used_in_attack_or_model_diagnostic',p)
 # Gradient-threshold selection is B even without any optimizer update.
 grad=REPORTS/'nih_cxr14_public_clip_calibration_v1_001/gradient_norms_private.csv';markrows(rr(grad),'used_in_checkpoint_or_hypothesis_selection',grad)
 # Broad machine-readable artifact census: conservative additional consumption/uncertainty.
 scan=[];new_private_hits=defaultdict(set)
 metadata_prefixes=('nih_cxr14_pa_split','nih_cxr14_union_','nih_cxr14_official_png','target_evaluation_inventory_')
 skip_prefixes=('pcm_','isic','base_gate','nih_cxr14_dp_trainer_synthetic','patient_usage_audit_')
 def extract(obj,found,parent=''):
  if isinstance(obj,dict):
   for k,v in obj.items():
    if k=='patient_id' or k.endswith('_patient_id'):
     if str(v).isdigit() and str(int(v)) in pats:found.add(str(int(v)))
    if k in ['patient_ids','patients_used','train_patients'] and isinstance(v,list):
     found.update(str(int(p)) for p in v if str(p).isdigit() and str(int(p)) in pats)
    for iid in IMAGE.findall(str(k)):
     if iid in index:found.add(index[iid]['patient_id'])
    extract(v,found,k)
  elif isinstance(obj,list):
   for v in obj:extract(v,found,parent)
  elif isinstance(obj,str):
   for iid in IMAGE.findall(obj):
    if iid in index:found.add(index[iid]['patient_id'])
 files=sorted(p for d in REPORTS.iterdir() if d.is_dir() and not d.name.startswith(skip_prefixes) for p in d.rglob('*') if p.suffix in ['.json','.csv','.jsonl'])
 for p in files:
  rel=p.relative_to(REPORTS);folder=rel.parts[0];found=set();mode='potential_model_or_result_use'
  if folder.startswith(metadata_prefixes) or p==COHORT/'source_audit.csv':mode='metadata_acquisition_only'
  elif p.name in ['protocol.json','contract.json','lock.json'] or 'protocol_v' in folder or p.name.endswith('_contract_v1.json'):mode='declaration_not_execution'
  # Data-copy metadata generated by the last inventory must not fabricate prior model use.
  text=p.read_text(encoding='utf-8-sig');sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
  try:
   if p.suffix=='.csv':
    for x in csv.DictReader(text.splitlines()):extract(x,found)
   elif p.suffix=='.jsonl':
    for line in text.splitlines():
     if line.strip():extract(json.loads(line),found)
   else:extract(json.loads(text),found)
  except Exception as e:raise RuntimeError('Unparsed artifact '+str(p)) from e
  if mode=='potential_model_or_result_use':
   # Known clinical model artifacts are C unless detailed training/head evidence above says A/B.
   for pp in found:
    if original.get(pp)=='private_train' and pp not in dryp:new_private_hits[pp].add(str(p))
   if folder=='cvpr_u_pilot_v1_001':mark(found,'used_in_attack_or_model_diagnostic',p)
   else:
    unknown={pp for pp in found if not evidence[pp]}
    mark(unknown,'unresolved_possible_use',p)
  scan.append(dict(path=str(p.resolve()),sha256=sources[str(p.resolve())],bytes=p.stat().st_size,mode=mode,patient_mentions=len(found)))
 for pp,paths in new_private_hits.items():
  for p in paths:mark([pp],'unresolved_possible_use',p)
 # Exact content duplicates are a separate reason to withhold a candidate.
 duplicates=defaultdict(set)
 for x in inv:duplicates[x['sha256'].lower()].add(x['patient_id'])
 for pp,ims in bypatient.items():
  if any(len(duplicates[x['sha256'].lower()])>1 for x in ims):mark([pp],'cross_patient_exact_duplicate',DATA/'raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv')
 ledger=[]
 for pp in sorted(pats,key=int):
  f={k:int(bool(evidence[pp].get(k))) for k in FIELDS};role=original.get(pp,'official_test_census_only');locked=int(pp in census or pp in locked_cvpr or role=='final_test')
  A=any(f[k] for k in ['used_in_backbone_training','used_in_public_head','used_in_private_head','used_in_prior_M1_M2_training','used_in_thesis_dryrun_training','used_in_other_disposable_training'])
  B=f['used_in_development_loss'] or f['used_in_checkpoint_or_hypothesis_selection']
  C=any(f[k] for k in ['used_in_metric_reference','used_in_evaluator_preflight','used_in_visual_diagnostic','used_in_attack_or_model_diagnostic','used_in_feature_cache'])
  U=f['unresolved_possible_use'] or f['cross_patient_exact_duplicate']
  grade='E' if locked else 'A' if A else 'B' if B else 'C' if C else 'U' if U else 'D'
  ims=bypatient[pp];ec=sum('Emphysema' in x['finding_labels'].split('|') for x in ims);pc=sum('Pneumothorax' in x['finding_labels'].split('|') for x in ims)
  nf=sum(x['finding_labels']=='No Finding' for x in ims);eff=sum('Effusion' in x['finding_labels'].split('|') and not {'Emphysema','Pneumothorax'}&set(x['finding_labels'].split('|')) for x in ims)
  eligible=int(grade=='D' and role=='private_train' and not U)
  ledger.append(dict(patient_id=pp,original_thesis_role=role,**f,locked_thesis_final=int(pp in census or role=='final_test'),locked_cvpr_calibration_test=int(pp in locked_cvpr),
   audit_grade=grade,local_images=len(ims),emphysema_images=ec,pneumothorax_images=pc,normal_control_images=nf,effusion_control_images=eff,
   eligible_for_cvpr_development=int(not locked and not U and (eligible or (role=='public_development' and grade in ['B','C']))),
   eligible_for_cvpr_confirmation=eligible,eligibility_scope='provisional_separate_study_only_original_roles_unchanged',
   evidence_count=len(set().union(*evidence[pp].values())) if evidence[pp] else 0))
 pool=sorted([x['patient_id'] for x in ledger if x['eligible_for_cvpr_confirmation']],key=lambda p:key('patient',p))
 divide=(len(pool)+1)//2;assignment={p:('target_development' if i<divide else 'reserved_confirmation') for i,p in enumerate(pool)}
 splitrows=[];selected=[]
 for p in pool:
  splitrows.append(dict(patient_id=p,provisional_role=assignment[p],patient_order_sha256=key('patient',p),original_thesis_role=original[p]))
  for label in ['Emphysema','Pneumothorax','No Finding','Effusion']:
   ims=[x for x in bypatient[p] if label in x['finding_labels'].split('|')]
   if label=='No Finding':ims=[x for x in ims if x['finding_labels']=='No Finding']
   if label=='Effusion':ims=[x for x in ims if not {'Emphysema','Pneumothorax'}&set(x['finding_labels'].split('|'))]
   if ims:
    x=min(ims,key=lambda x:key('image',x['image_id']));selected.append(dict(patient_id=p,provisional_role=assignment[p],condition=label,image_id=x['image_id'],sha256=x['sha256'].lower(),selection_sha256=key('image',x['image_id'])))
 # Only file existence/bytes checked: no new patient image decoding or model-dependent filtering.
 image_root=DATA/'raw/nih_cxr14_pa_k10_plus_census_v1/images'
 for x in selected:
  p=image_root/x['image_id'];require(p.is_file() and p.stat().st_size==int(index[x['image_id']]['bytes']),'Candidate file unavailable')
 def summarize(ll):
  return dict(patients=len(ll),images=sum(x['local_images'] for x in ll),
   label_patients={k:sum(x[k]>0 for x in ll) for k in ['emphysema_images','pneumothorax_images','normal_control_images','effusion_control_images']},
   label_images={k:sum(x[k] for x in ll) for k in ['emphysema_images','pneumothorax_images','normal_control_images','effusion_control_images']},
   target_overlap=sum(x['emphysema_images']>0 and x['pneumothorax_images']>0 for x in ll))
 strata={g:summarize([x for x in ledger if x['audit_grade']==g]) for g in ['A','B','C','D','E','U']}
 candidate=summarize([x for x in ledger if x['eligible_for_cvpr_confirmation']])
 splits={sp:summarize([x for x in ledger if assignment.get(x['patient_id'])==sp]) for sp in ['target_development','reserved_confirmation']}
 evaluator_public=[x for x in ledger if x['original_thesis_role']=='public_development' and not x['locked_cvpr_calibration_test'] and not x['locked_thesis_final'] and x['used_in_evaluator_preflight']]
 save(OUT/'artifact_scan.json',scan);table(OUT/'patient_usage_private.csv',ledger);table(OUT/'provisional_patient_split_private.csv',splitrows);table(OUT/'provisional_condition_images_private.csv',selected)
 save(OUT/'patient_evidence_private.json',{p:{k:sorted(v) for k,v in ee.items()} for p,ee in evidence.items() if ee})
 save(OUT/'unresolved_private_hits.json',{p:sorted(v) for p,v in new_private_hits.items()})
 save(OUT/'contract.json',dict(schema='actual-patient-use-audit/v1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,salt=SALT,
  source_protocol=str(R/'TRACK1_PATIENT_USAGE_AUDIT_PROTOCOL_20260917.md'),existing_roles_mutated=False,new_inference=False,
  scope='Recorded local NIH model artifacts; absence of a hit is no absolute proof of never used outside this workspace. Metadata/intake and unexecuted declarations distinguished.'))
 result=dict(status='ACTUAL_USE_AUDIT_COMPLETED_PROVISIONAL_SPLIT_ONLY',created_utc=datetime.now(timezone.utc).isoformat(),
  patients=len(ledger),grades=strata,original_private_train=summarize([x for x in ledger if x['original_thesis_role']=='private_train']),
  dryrun_actual_patients=len(dryp),dryrun_actual_images=len(dryimages),m1_m2_patients=modelcoverage,public_backbone_patients=len(bpatients),
  candidate_private_train=candidate,provisional_splits=splits,prior_nonlocked_evaluator_validation_pool=summarize(evaluator_public),
  artifact_scan_files=len(scan),artifact_scan_bytes=sum(x['bytes'] for x in scan),additional_private_use_hits=len(new_private_hits),
  full_training_gate_started=False,current_full_run_paths={str(p):p.exists() for p in fullpaths},
  provisional_condition_rows=len(selected),candidate_pixel_decodes=0,selected_files_exist_and_bytes_match=True,
  original_roles_changed=False,new_GPU=0,new_model_inference=0,new_generation=0,private_utility_established=False,
  seconds=time.perf_counter()-started,outputs_sha256={n:sha(OUT/n) for n in ['artifact_scan.json','patient_usage_private.csv','provisional_patient_split_private.csv','provisional_condition_images_private.csv','patient_evidence_private.json','unresolved_private_hits.json']})
 for p,h in sources.items():require(sha(p)==h,'Source mutated '+p)
 save(OUT/'audit.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
