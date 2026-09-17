"""Independent manifest, preprocessing, raw-output and pairwise metric checks."""
import csv,hashlib,json,time
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from skimage.transform import resize

C=Path(__file__).resolve().parents[1];O=C/'_reports/padchest_validation_20260917_v1';A=C/'_reports/patient_usage_audit_20260917_v1'
RAW=C/'_data/raw/nih_cxr14_pa_k10_plus_census_v1';checks=0
def check(x,msg):
 global checks;checks+=1
 if not x:raise ValueError(msg)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 start=time.perf_counter();con=read(O/'contract.json');res=read(O/'result.json');rr=rows(O/'selected_images_private.csv')
 check(res['contract_sha256']==sha(O/'contract.json'),'Contract binding')
 for key in ['source_sha256','frozen_sha256']:
  for p,h in con[key].items():check(sha(p)==h,'Source changed '+p)
 for n,h in con['outputs'].items():check(sha(O/n)==h,'Selection binding')
 for n,h in res['artifact_sha256'].items():check(sha(O/n)==h,'Execution binding')
 # Independently reconstruct mutually exclusive patients and SHA selections.
 inv=rows(RAW/'content_inventory_private.csv');by=defaultdict(list);sha_pat=defaultdict(set)
 for x in inv:by[x['patient_id']].append(x);sha_pat[x['sha256'].lower()].add(x['patient_id'])
 cats={}
 for p,xs in by.items():
  labels={q for x in xs for q in x['finding_labels'].split('|')}
  if 'Emphysema' in labels:cats[p]='dual' if 'Pneumothorax' in labels else 'Emphysema'
  elif 'Pneumothorax' in labels:cats[p]='Pneumothorax'
  elif 'Effusion' in labels:cats[p]='Effusion'
  elif any(x['finding_labels']=='No Finding' for x in xs):cats[p]='No Finding'
  else:cats[p]='other'
 ledger=rows(A/'patient_usage_private.csv');split=rows(A/'provisional_patient_split_private.csv')
 old={x['patient_id'] for x in ledger if int(x['used_in_evaluator_preflight']) and not int(x['locked_cvpr_calibration_test']) and not int(x['locked_thesis_final'])}
 dev={x['patient_id'] for x in split if x['provisional_role']=='target_development'}
 reserved={x['patient_id'] for x in split if x['provisional_role']=='reserved_confirmation'}
 def order(k,v):return hashlib.sha256((con['salt']+'|'+k+'|'+v).encode('utf-8')).hexdigest()
 selected=set()
 for g in ['Emphysema','Pneumothorax','No Finding','Effusion']:
  pp=sorted((p for p in dev if cats[p]==g),key=lambda p:order('patient',p))[:20];selected.update(pp)
 observed={x['patient_id'] for x in rr if x['stratum']=='development_supplement'}
 check(observed==selected and len(selected)==80,'Supplement determined before score')
 check({x['patient_id'] for x in rr if x['stratum']=='old_nonlocked'}=={p for p in old if cats[p]!='other'},'Old pool')
 check(len(rr)==len({x['patient_id'] for x in rr})==309,'Patient uniqueness')
 check(not {x['patient_id'] for x in rr}&reserved,'Reserved confirmation exclusion')
 check({x['patient_id'] for x in rows(O/'evaluator_exclusions_private.csv')}==selected,'Future exclusion manifest')
 remain=rows(O/'remaining_development_private.csv');check({x['patient_id'] for x in remain}==dev-selected,'Remaining candidates')
 check(sum(x['group']=='Emphysema' for x in remain)==64,'Remaining E-only support')
 check(dict(Counter(x['group'] for x in remain))==con['remaining_development_counts'],'Remaining counts')
 meta={x['Image Index']:x for x in rows(C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 xx=np.load(O/'normalized_inputs.npy',mmap_mode='r');audit=read(O/'image_audit.json')
 for i,x in enumerate(rr):
  p=x['patient_id'];g=x['group'];check(cats[p]==g,'Patient label union')
  if g=='dual':
   cand=[v for v in by[p] if {'Emphysema','Pneumothorax'}<=set(v['finding_labels'].split('|'))]
   if not cand:cand=[v for v in by[p] if {'Emphysema','Pneumothorax'}&set(v['finding_labels'].split('|'))]
  else:cand=[v for v in by[p] if (v['finding_labels']=='No Finding' if g=='No Finding' else g in v['finding_labels'].split('|'))]
  chosen=sorted(cand,key=lambda v:order('image',v['image_id']))[0]
  check(chosen['image_id']==x['image_id'] and x['image_hash']==order('image',x['image_id']) and x['patient_hash']==order('patient',p),'Hash-selected image')
  m=meta[x['image_id']];check(str(int(m['Patient ID']))==p and set(m['Finding Labels'].split('|'))==set(x['finding_labels'].split('|')),'Official metadata')
  path=RAW/'images'/x['image_id'];check(sha(path)==chosen['sha256'].lower()==x['sha256'],'PNG hash')
  check(sha_pat[x['sha256']]=={p},'No cross-patient exact duplicate')
  with Image.open(path) as im:
   aa=np.asarray(im)
   if aa.ndim==3:
    check(np.array_equal(aa[:,:,0],aa[:,:,1]) and np.array_equal(aa[:,:,0],aa[:,:,2]),'Gray channels')
    if aa.shape[2]==4:check(np.all(aa[:,:,3]==255),'Alpha opaque')
    aa=aa[:,:,0]
   check(aa.dtype==np.uint8,'8bit input')
   y=(aa.astype(np.float32)/np.float32(255)*np.float32(2)-np.float32(1))*np.float32(1024)
   h,w=y.shape;k=min(h,w);y=y[(h-k)//2:(h-k)//2+k,(w-k)//2:(w-k)//2+k]
   expected=resize(y[None],(1,224,224),mode='constant',preserve_range=True).astype(np.float32)
   check(np.array_equal(expected,xx[i]),'Independent preprocessing equality')
  check(hashlib.sha256(np.asarray(xx[i]).tobytes()).hexdigest()==audit[i]['tensor_sha256'],'Tensor binding')
 # Official forward uses sigmoid then piecewise threshold normalization.
 log=np.load(O/'logits.npy');model=read(O/'model_metadata.json');rep=np.load(O/'forward_replay.npz')
 check(log.shape==(309,18) and np.isfinite(log).all(),'Raw logit shape')
 check(np.allclose(log[rep['indices']],rep['raw_replay'],atol=2e-4,rtol=2e-5),'Batch-vs-replay FP32 tolerance')
 raw=rep['raw_replay'].astype(np.float64);prob=1/(1+np.exp(-raw));th=rep['op_threshs'];normal=np.full(prob.shape,.5)
 for j in range(18):
  if np.isfinite(th[j]):normal[:,j]=np.where(prob[:,j]<th[j],prob[:,j]/(2*th[j]),1-(1-prob[:,j])/(2*(1-th[j])))
 check(np.allclose(normal,rep['default_output'],atol=2e-6,rtol=0),'Default forward independent sigmoid/op_norm')
 # No production metric imports: pair-count AUC and threshold-block AP.
 def score(pos,neg):
  auc=float(((pos[:,None]>neg[None,:]).sum()+.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))
  vals=np.concatenate([pos,neg]);y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]);idx=np.argsort(-vals,kind='stable');ss=vals[idx];ys=y[idx]
  end=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1];tp=np.cumsum(ys)[end];ap=float(np.sum(np.diff(np.r_[0,tp])/len(pos)*tp/(end+1)))
  return np.array([auc,ap])
 met=read(O/'metrics.json');boot=np.load(O/'bootstrap.npz');rng=np.random.default_rng(20260917)
 groups=['Emphysema','Pneumothorax','No Finding','Effusion'];maxdiff=0.0
 for st in ['old_nonlocked','development_supplement']:
  for target in ['Emphysema','Pneumothorax']:
   ix=model['pathologies'].index(target);other='Pneumothorax' if target=='Emphysema' else 'Emphysema'
   pos=np.array([log[i,ix] for i,x in enumerate(rr) if x['stratum']==st and x['group']==target])
   for name,ng in [('rest',[g for g in groups if g!=target]),('opposite',[other]),('normal',['No Finding']),('effusion',['Effusion'])]:
    negs=[np.array([log[i,ix] for i,x in enumerate(rr) if x['stratum']==st and x['group']==g]) for g in ng];neg=np.concatenate(negs);k=st+'__'+target+'__'+name;v=met['metrics'][k]
    check(v['n_positive']==len(pos) and v['n_negative']==len(neg) and v['prevalence']==len(pos)/(len(pos)+len(neg)),'Counts/prevalence')
    check(np.allclose(score(pos,neg),[v['auc'],v['ap']],atol=1e-12,rtol=0),'Point AUC/AP independent calculation')
    for b in range(2000):
     pp=pos[rng.integers(len(pos),size=len(pos))];nn=np.concatenate([q[rng.integers(len(q),size=len(q))] for q in negs]);s=score(pp,nn)
     dif=float(np.max(np.abs(s-boot[k][b])));maxdiff=max(maxdiff,dif);check(dif<1e-12,'Patient-bootstrap independent replay')
    check(np.allclose(np.quantile(boot[k][:,0],[.025,.975]),v['auc_ci95'],atol=1e-12,rtol=0),'AUC interval')
    check(np.allclose(np.quantile(boot[k][:,1],[.025,.975]),v['ap_ci95'],atol=1e-12,rtol=0),'AP interval')
    check(np.allclose(np.quantile(pos,[0,.25,.5,.75,1]),v['positive_quantiles'],atol=1e-12,rtol=0),'Positive distribution')
    check(np.allclose(np.quantile(neg,[0,.25,.5,.75,1]),v['negative_quantiles'],atol=1e-12,rtol=0),'Negative distribution')
 decisions={}
 for t in ['Emphysema','Pneumothorax']:
  a=met['metrics']['development_supplement__'+t+'__rest'];b=met['metrics']['development_supplement__'+t+'__opposite']
  decisions[t]={'rest_pass':a['auc']>=.7 and a['auc_ci95'][0]>.5,'opposite_pass':b['auc']>=.6 and b['auc_ci95'][0]>.5}
  decisions[t]['pass']=all(decisions[t].values())
 check(decisions==met['development_criterion']==res['development_criterion'],'Predeclared decisions')
 check(all(v['pass'] for v in decisions.values())==met['joint_pass']==res['joint_pass'],'Joint gate')
 check(not res['reserved_confirmation_consumed'] and res['new_generation']==res['new_DP']==res['backward_calls']==0,'Execution scope')
 result=dict(status='PASS_INDEPENDENT_EVALUATOR_VERIFICATION',utc=datetime.now(timezone.utc).isoformat(),checks=checks,seconds=time.perf_counter()-start,
  contract_sha256=sha(O/'contract.json'),result_sha256=sha(O/'result.json'),metrics_sha256=sha(O/'metrics.json'),verifier_sha256=sha(__file__),
  max_bootstrap_auc_ap_abs_difference=maxdiff,patients=309,selected_pixel_preprocessing_recomputed=309,
  all_32000_bootstrap_pairs_recomputed=True,full_model_separate_replay=False,old192_unchanged=True,reserved_confirmation_unconsumed=True)
 with (O/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
