"""Independent cohort, pixel, captured feature, ensemble and bootstrap verification."""
import ast,csv,hashlib,json,time,types
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

C=Path(__file__).resolve().parents[1];R=next(C.parent.glob('CVPR */research_2026-09-10'))
O=C/'_reports/chexzero_validation_20260917_v1';P=R/'spec_sources/chexzero_review_20260917_v1'
PLAN=C/'_reports/chexzero_evaluator_plan_20260917_v1';OLD=C/'_reports/padchest_validation_20260917_v1';A=C/'_reports/patient_usage_audit_20260917_v1'
RAW=C/'_data/raw/nih_cxr14_pa_k10_plus_census_v1';checks=0

def check(ok,msg):
 global checks;checks+=1
 if not ok:raise ValueError(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,atol,rtol=0):return np.allclose(a,b,atol=atol,rtol=rtol)

def main():
 started=time.perf_counter();torch.set_num_threads(4);con=read(O/'contract.json');res=read(O/'result.json');rr=rows(O/'selected_images_private.csv')
 check(res['contract_sha256']==sha(O/'contract.json'),'Contract hash')
 for key in ['source_sha256','frozen_sha256']:
  for p,h in con[key].items():check(sha(p)==h,'Source/history changed '+p)
 for n,h in con['outputs'].items():check(sha(O/n)==h,'Prepared manifest binding')
 for n,h in res['artifact_sha256'].items():check(sha(O/n)==h,'Actual output binding')
 check(sha(O/'selected_images_private.csv')==sha(PLAN/'selected_images_private.csv'),'Unchanged prospective manifest')
 check(sha(O/'remaining_development_private.csv')==sha(PLAN/'remaining_method_development_private.csv'),'Remaining cohort preserved')
 inv=rows(RAW/'content_inventory_private.csv');by=defaultdict(list);duplicates=defaultdict(set)
 for x in inv:by[x['patient_id']].append(x);duplicates[x['sha256'].lower()].add(x['patient_id'])
 def category(p):
  labels={t for x in by[p] for t in x['finding_labels'].split('|')}
  if {'Emphysema','Pneumothorax'}<=labels:return 'dual'
  for name in ['Emphysema','Pneumothorax','Effusion']:
   if name in labels:return name
  return 'No Finding' if any(x['finding_labels']=='No Finding' for x in by[p]) else 'other'
 available={x['patient_id'] for x in rows(OLD/'remaining_development_private.csv')}
 old_used={x['patient_id'] for x in rows(OLD/'selected_images_private.csv')}
 reserved={x['patient_id'] for x in rows(A/'provisional_patient_split_private.csv') if x['provisional_role']=='reserved_confirmation'}
 salt=read(PLAN/'cohort_contract.json')['salt']
 def order(k,x):return hashlib.sha256((salt+'|'+k+'|'+x).encode()).hexdigest()
 selected={x['patient_id'] for x in rr};expected=set()
 for g in con['groups']:
  pp=sorted((p for p in available if category(p)==g),key=lambda p:order('patient',p))[:20];expected.update(pp)
  check(set(pp)=={x['patient_id'] for x in rr if x['condition']==g},'Outcome-independent selected group')
 check(selected==expected and len(rr)==len(selected)==80,'Exactly 80 distinct patients')
 check(not selected&(old_used|reserved),'Previously evaluated or reserved overlap')
 check({x['patient_id'] for x in rows(O/'evaluator_exclusions_private.csv')}==selected,'New consumed overlay')
 check(all(x['model_result_consumed']=='True' for x in rows(O/'evaluator_exclusions_private.csv')),'Consumption status')
 remaining=rows(O/'remaining_development_private.csv')
 check({x['patient_id'] for x in remaining}==available-selected and len(remaining)==4053,'Remaining 4053')
 check(sum(category(x['patient_id'])=='Emphysema' for x in remaining)==44,'Remaining E-only44')
 # Execute the pinned official preprocessing function, with only the removed enum aliased.
 tree=ast.parse((P/'source/data_process.py').read_text(encoding='utf-8'))
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='preprocess')
 namespace={'Image':types.SimpleNamespace(ANTIALIAS=Image.Resampling.LANCZOS,new=Image.new)}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'official_preprocess','exec'),namespace)
 inputs=np.load(O/'normalized_inputs.npy',mmap_mode='r');smalls=np.load(O/'grayscale320.npy',mmap_mode='r');audit=read(O/'image_audit.json')
 meta={x['Image Index']:x for x in rows(C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 check(inputs.shape==(80,3,224,224) and inputs.dtype==np.float32 and np.isfinite(inputs).all(),'Tensor dtype/shape')
 for i,x in enumerate(rr):
  pid=x['patient_id'];g=x['condition'];check(category(pid)==g,'Patient-union label')
  eligible=[v for v in by[pid] if (v['finding_labels']=='No Finding' if g=='No Finding' else g in v['finding_labels'].split('|'))]
  expected_image=min(eligible,key=lambda v:order('image',v['image_id']))
  check(x['image_id']==expected_image['image_id'] and x['patient_hash']==order('patient',pid) and x['image_hash']==order('image',x['image_id']),'Image selection')
  file=RAW/'images'/x['image_id'];check(sha(file)==x['sha256']==expected_image['sha256'].lower(),'Actual image SHA')
  check(duplicates[x['sha256']]=={pid},'No cross-patient exact duplicate')
  m=meta[x['image_id']];check(str(int(m['Patient ID']))==pid and m['View Position']=='PA' and set(m['Finding Labels'].split('|'))==set(x['finding_labels'].split('|')),'Official label provenance')
  bgr=cv2.imdecode(np.frombuffer(file.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
  check(bgr is not None,'OpenCV PNG decode');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  image320=np.asarray(namespace['preprocess'](Image.fromarray(rgb)))
  check(np.array_equal(image320,smalls[i]),'Official 320 preprocessing exact')
  x320=torch.tensor(np.repeat(image320[None],3,axis=0),dtype=torch.float32)
  x320=x320.sub(torch.tensor(101.48761,dtype=torch.float32)).div(torch.tensor(83.43944,dtype=torch.float32))
  resized=torch.nn.functional.interpolate(x320[None],size=(224,224),mode='bicubic',align_corners=False,antialias=False)[0].numpy()
  check(np.array_equal(resized,inputs[i]),'Independent tensor normalization/resize exact')
  check(hashlib.sha256(inputs[i].tobytes()).hexdigest()==audit[i]['tensor_sha256'],'Tensor hash')
 check(np.array_equal(np.load(O/'tokens.npy'),con['plan']['token_ids']),'Literal prompt tokens')
 # Reconstruct score from captured embeddings without calling production arithmetic.
 data=np.load(O/'model_outputs.npz');tol=con['tolerances'];ix=np.asarray(con['replay_indices'])
 check(np.array_equal(data['replay_indices'],ix),'Fixed replay patients')
 check(data['cosines'].shape==(10,80,4) and data['probabilities'].shape==(10,80,2),'All released checkpoints retained')
 def normalized(v):return v.astype(np.float64)/np.linalg.norm(v.astype(np.float64),axis=-1,keepdims=True)
 max_cos=0.0;max_prob=0.0;cpu_scores=[]
 for k in range(10):
  f=data['image_features'][k].astype(np.float64);t=data['text_features'][k].astype(np.float64)
  check(close(np.linalg.norm(f,axis=1),1,2e-6) and close(np.linalg.norm(t,axis=1),1,2e-6),'Unit embeddings')
  cos=f@t.T;delta=float(np.max(np.abs(cos-data['cosines'][k])));max_cos=max(max_cos,delta);check(delta<tol['cosine_maxabs'],'Cosine from saved embeddings')
  z=data['cosines'][k].astype(np.float64);prob=1/(1+np.exp(z[:,[1,3]]-z[:,[0,2]]));delta=float(np.max(np.abs(prob-data['probabilities'][k])));max_prob=max(max_prob,delta);check(delta<2e-7,'No logit scale; pair sigmoid')
  rep=np.load(O/f'm{k:02d}_replay.npz')
  check(np.array_equal(rep['cpu_image_raw'][:4],rep['cpu_image_raw'][4:]),'Official CPU pair repeated input')
  for key in ['cpu_image','cpu_text']:
   check(close(normalized(rep[key+'_raw']),rep[key+'_normalized'],2e-6),'CPU normalized features')
  for raw,norm in [('gpu_replay_raw','gpu_replay_normalized'),('gpu_text_raw','gpu_text_normalized')]:check(close(normalized(rep[raw]),rep[norm],2e-6),'GPU normalized features')
  check(close(rep['cpu_image_raw'][:4],rep['gpu_replay_raw'],tol['raw_atol'],tol['raw_rtol']),'CPU/GPU raw image')
  check(close(rep['cpu_text_raw'],rep['gpu_text_raw'],tol['raw_atol'],tol['raw_rtol']),'CPU/GPU raw text')
  check(close(rep['cpu_image_normalized'][:4],rep['gpu_replay_normalized'],tol['normalized_maxabs']),'CPU/GPU normalized image')
  check(close(rep['cpu_text_normalized'],rep['gpu_text_normalized'],tol['normalized_maxabs']),'CPU/GPU normalized text')
  check(close(rep['cpu_cosine'],rep['gpu_replay_cosine'],tol['cosine_maxabs']),'CPU/GPU cosine')
  cpu_z=rep['cpu_image_normalized'][:4].astype(np.float64)@rep['cpu_text_normalized'].astype(np.float64).T
  cpu_p=1/(1+np.exp(cpu_z[:,[1,3]]-cpu_z[:,[0,2]]));check(close(cpu_p,rep['cpu_score'],tol['score_maxabs']),'Official CPU scores independently reconstructed')
  check(close(rep['cpu_score'],rep['gpu_replay_score'],tol['score_maxabs']),'Official CPU/GPU scores')
  check(close(rep['gpu_replay_score'],data['probabilities'][k,ix],tol['score_maxabs']),'GPU batch-vs-replay')
  cpu_scores.append(rep['cpu_score'])
 check(close(data['probabilities'].astype(np.float64).mean(axis=0),data['ensemble'],2e-7),'Sigmoid then equal mean10')
 check(close(np.stack(cpu_scores).mean(axis=0),data['ensemble'][ix],tol['score_maxabs']),'Final CPU ensemble parity')
 # Pairwise AUC and threshold-block AP; do not import producer or sklearn.
 def metric(pos,neg):
  au=((pos[:,None]>neg[None]).sum()+.5*(pos[:,None]==neg[None]).sum())/(len(pos)*len(neg))
  values=np.r_[pos,neg];y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];order_ix=np.argsort(-values,kind='stable');z=values[order_ix];labels=y[order_ix]
  end=np.r_[np.flatnonzero(z[1:]!=z[:-1]),len(z)-1];tp=np.cumsum(labels)[end];ap=np.sum(np.diff(np.r_[0,tp])/len(pos)*tp/(end+1))
  return np.array([au,ap])
 rng=np.random.default_rng(con['bootstrap_seed']);met=read(O/'metrics.json');boot=np.load(O/'bootstrap.npz');scores=data['ensemble'];max_metric=0.0
 for j,target in enumerate(con['targets']):
  pos=scores[[i for i,x in enumerate(rr) if x['condition']==target],j];other=con['targets'][1-j]
  for name,gg in [('rest',[g for g in con['groups'] if g!=target]),('opposite',[other]),('normal',['No Finding']),('effusion',['Effusion'])]:
   ns=[scores[[i for i,x in enumerate(rr) if x['condition']==g],j] for g in gg];neg=np.concatenate(ns);key=target+'__'+name;v=met['metrics'][key]
   check(v['n_positive']==len(pos) and v['n_negative']==len(neg) and v['prevalence']==len(pos)/(len(pos)+len(neg)),'Metric sample counts')
   check(close(metric(pos,neg),[v['auc'],v['ap']],1e-12),'Independent AUC/AP')
   for b in range(con['bootstrap_iterations']):
    pp=pos[rng.integers(len(pos),size=len(pos))];nn=np.concatenate([x[rng.integers(len(x),size=len(x))] for x in ns]);value=metric(pp,nn);d=float(np.max(np.abs(value-boot[key][b])));max_metric=max(max_metric,d);check(d<1e-12,'Independent patient bootstrap')
   check(close(np.quantile(boot[key][:,0],[.025,.975]),v['auc_ci95'],1e-12),'AUC interval')
   check(close(np.quantile(boot[key][:,1],[.025,.975]),v['ap_ci95'],1e-12),'AP interval')
   check(close(np.quantile(pos,[0,.25,.5,.75,1]),v['positive_quantiles'],1e-12),'Positive distribution')
   check(close(np.quantile(neg,[0,.25,.5,.75,1]),v['negative_quantiles'],1e-12),'Negative distribution')
 dec={}
 for t in con['targets']:
  a=met['metrics'][t+'__rest'];b=met['metrics'][t+'__opposite'];dec[t]=dict(rest_pass=a['auc']>=.7 and a['auc_ci95'][0]>.5,opposite_pass=b['auc']>=.6 and b['auc_ci95'][0]>.5);dec[t]['pass']=all(dec[t].values())
 check(dec==met['criterion']==res['criterion'],'Frozen decision reconstruction')
 check(all(d['pass'] for d in dec.values())==met['joint_pass']==res['joint_pass'],'Joint gate')
 check(res['backward_calls']==res['new_training']==res['new_generation']==res['new_DP']==0 and not res['reserved_confirmation_consumed'],'Scope preserved')
 result=dict(status='PASS_INDEPENDENT_CHEXZERO_VALIDATION',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,seconds=time.perf_counter()-started,
  patients=80,checkpoints=10,preprocessing_recomputed=80,bootstrap_pairs_recomputed=16000,max_metric_abs_difference=max_metric,max_cosine_reconstruction_difference=max_cos,max_pair_score_difference=max_prob,
  actual_CPU_GPU_reference=True,independent_verifier_repeats_model_forward=False,old192_preserved=True,reserved_confirmation_consumed=False,
  contract_sha256=sha(O/'contract.json'),result_sha256=sha(O/'result.json'),metrics_sha256=sha(O/'metrics.json'),verifier_sha256=sha(__file__))
 with (O/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
