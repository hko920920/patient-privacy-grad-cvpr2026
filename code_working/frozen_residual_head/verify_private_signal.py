"""Independent CPU saved-output checks; never imports the producer's math."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,csv,hashlib,time
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
from PIL import Image
CODE=Path(__file__).resolve().parent.parent
P=CODE/'_reports/private_signal_20260917_v1'
OLD=CODE/'_reports/medical_head_20260916_v1'
checks=0;errors={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def eq(ok,msg):
 global checks
 checks+=1
 if not ok:raise ValueError(msg)
def close(x,y,msg,atol=1e-10):
 x=np.asarray(x);y=np.asarray(y);err=float(np.max(np.abs(x-y)))
 errors[msg]=max(errors.get(msg,0),err);eq(np.isfinite(err) and err<=atol,msg+': '+str(err))
def digest(t):return hashlib.sha256(t.encode()).hexdigest()
def main():
 start=time.perf_counter();c=read(P/'contract.json');a=read(P/'analysis.json');pts=read(OLD/'patients.json');ims=read(OLD/'images.json');man=read(OLD/'manifest.json')
 for p,h in c['source_sha256'].items():eq(sha(p)==h,'Source '+p)
 for f,key in [('analysis_weights.npz','weights_sha256'),('image_statistics.npz','image_statistics_sha256'),('eval_time_statistics.npz','time_statistics_sha256')]:eq(sha(P/f)==a[key],'Bound output '+f)
 with np.load(OLD/'patient_statistics.npz') as d:A=d['A'];B=d['B'];Q=d['Q']
 with np.load(P/'analysis_weights.npz') as d:W={k:d[k] for k in d.files}
 with np.load(P/'image_statistics.npz') as d:IA=d['A'];IB=d['B']
 ix={s:[i for i,p in enumerate(pts) if p['split']==s] for s in ['public','train','eval']};imindex={p['image_id']:i for i,p in enumerate(ims)};eps=.001*np.eye(64)
 def fit(ids):return np.linalg.solve(np.mean(A[ids],0)+eps,np.mean(B[ids],0))
 for name,ids in [('public',ix['public']),('private',ix['train']),('pooled',ix['public']+ix['train'])]:close(W[name],fit(ids),'Independent solve '+name)
 matched=[]
 for pid,names in c['matched_private_images'].items():
  eligible=[x['image_id'] for x in ims if x['split']=='train' and x['patient_id']==pid]
  expected=sorted(eligible,key=lambda x:digest('private-signal-matched-images-v1|'+x))[:2]
  eq(names==expected,'Matched-image rule');matched.extend(imindex[n] for n in names)
 close(W['private_two_images'],np.linalg.solve(IA[matched].mean(0)+eps,IB[matched].mean(0)),'Matched-image solve')
 pri={pts[i]['patient_id'] for i in ix['train']}
 for part in c['partitions']:
  order=sorted(pri,key=lambda x:digest(f"private-signal-halves-v1|{part['partition']}|{x}"))
  eq(part['left']==order[:40] and part['right']==order[40:],'Partition deterministic')
  eq(len(set(part['left'])&set(part['right']))==0 and set(part['left']+part['right'])==pri,'Half disjointness')
  for side in ['left','right']:
   ids=[i for i in ix['train'] if pts[i]['patient_id'] in part[side]]
   close(W[f"half{part['partition']}_{side}"],fit(ids),'Independent half solve')
 ae=A[ix['eval']].mean(0)
 for name,v in a['weight_comparisons'].items():
  for geom,G in [('coefficient',np.eye(64)),('development_prediction',ae)]:
   wp=W['public'];w=W[name];inner=lambda u,z:float(np.einsum('ic,ij,jc->',u,G,z,optimize=True))
   n=inner(wp,wp);m=inner(w,w);cross=inner(wp,w);delta=w-wp;alpha=cross/n;r=w-alpha*wp
   expected=dict(cosine=cross/np.sqrt(n*m),delta_relative_norm=np.sqrt(inner(delta,delta)/n),best_scalar=alpha,
     remainder_relative_to_v=np.sqrt(inner(r,r)/m),delta_nonscalar_energy_fraction=inner(r,r)/inner(delta,delta),
     scalar_projection_inner_product=inner(wp,r))
   for k,num in expected.items():close(v[geom][k],num,'Geometry '+geom+' '+k)
 for name,ids in [('private',ix['train']),('pooled',ix['public']+ix['train'])]:
  H=np.mean(A[ids],0)+eps;b=np.mean(B[ids],0);delta=np.linalg.solve(H,b-H@W['public'])
  close(W['public']+delta,W[name],'Same-objective reparameterization')
 # Direct predictions on every previously stored development observation.
 names=['backbone','public','private','pooled','private_two_images'];wm=np.stack([W[k].reshape(4,16,4) for k in names])
 bypatient=defaultdict(list);bytime=defaultdict(list);bycondition=defaultdict(lambda:defaultdict(list));allmeta=read(P/'image_metadata_and_proxies.json')
 info={x['image_id']:x for x in allmeta if x['split']=='eval'}
 raw_checked=0
 for row in man:
  if row['split']!='eval':continue
  q=OLD/row['raw_path'];eq(sha(q)==row['raw_sha256'],'Development raw unchanged')
  with np.load(q) as d:
   coeff=np.einsum('b,mbkc->mkc',d['basis'],wm);pred=np.einsum('nk,mkc->mnc',d['features'].astype(float)*7.5,coeff)
   residual=d['target'].astype(float)-d['base'].astype(float);mse=np.mean((residual[None]-pred)**2,axis=(1,2))
  bypatient[row['patient_id']].append(mse);bytime[row['draw_id']].append(mse)
  ls=set(info[row['image_id']]['labels'])
  for lab in ['No Finding','Effusion','Cardiomegaly','other']:
   if (lab in ls) if lab!='other' else not ls&{'No Finding','Effusion','Cardiomegaly'}:
    bycondition[lab][row['patient_id']].append(mse)
  raw_checked+=1
 eq(raw_checked==1280,'Every development record')
 for col,name in enumerate(names):
  vec=[np.mean(bypatient[p],axis=0)[col] for p in a['eval_patient_ids']]
  close(vec,a['development_losses'][name],'Direct1280 prediction MSE '+name)
  close(np.mean(vec),a['development'][name]['mean_MSE'],'Mean development MSE '+name)
 for d,rows in bytime.items():
  vec=np.mean(rows,axis=0)
  for col,name in enumerate(names):close(vec[col],a['time_strata'][d]['MSE'][name],'Time MSE '+name)
 for lab,g in bycondition.items():
  vec=np.mean([np.mean(z,axis=0) for z in g.values()],axis=0);eq(len(g)==a['development_conditions'][lab]['patients'],'Condition patient count')
  for col,name in enumerate(names):close(vec[col],a['development_conditions'][lab]['MSE'][name],'Condition MSE '+name)
 # Metadata independently cross-linked to the source, aggregates re-counted.
 meta_path=CODE/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv'
 with meta_path.open(encoding='utf-8-sig',newline='') as f:meta={x['Image Index']:x for x in csv.DictReader(f)}
 for row in allmeta:
  m=meta[row['image_id']];eq(set(row['labels'])==set(m['Finding Labels'].split('|')),'Weak label binding')
  eq(row['age']==int(m['Patient Age']) and row['sex']==m['Patient Sex'] and row['view']==m['View Position'],'Metadata fields')
 for sp,out in a['cohorts'].items():
  rows=[x for x in allmeta if x['split']==sp];eq(len(rows)==out['images'] and len({x['patient_id'] for x in rows})==out['patients'],'Cohort sample size')
  for label,count in out['weak_label_image_counts'].items():eq(sum(label in x['labels'] for x in rows)==count,'Cohort label count')
  eq(dict(Counter(x['view'] for x in rows))==out['view'],'View counts')
 # Fixed scalar-expression checks cannot establish meaningful private generation utility.
 for name,d in a['development'].items():
  w=W[name];vals=[(Q[i]-2*np.einsum('ic,ic->',B[i],w)+np.einsum('ic,ij,jc->',w,A[i],w,optimize=True))/4 for i in ix['eval']]
  close(vals,a['development_losses'][name],'Independent quadratic all heads')
  eq(int(np.sum(np.asarray(vals)<a['development_losses']['public']))==d['better_than_public'] if name!='public' else d['better_than_public']==0,'Patient improvement count')
 for item in a['private_disjoint_half_stability']:
  d0=W[f"half{item['partition']}_left"]-W['public'];d1=W[f"half{item['partition']}_right"]-W['public']
  close(item['delta_coefficient_cosine'],np.vdot(d0,d1)/np.linalg.norm(d0)/np.linalg.norm(d1),'Half coefficient cosine')
  inner=lambda u,v:float(np.einsum('ic,ij,jc->',u,ae,v,optimize=True))
  close(item['delta_prediction_cosine'],inner(d0,d1)/np.sqrt(inner(d0,d0)*inner(d1,d1)),'Half prediction cosine')
 result=dict(status='PASS_PRIVATE_SIGNAL_SAVED_CPU_ANALYSIS',complete=True,checks=checks,seconds=time.perf_counter()-start,
  development_raw_predictions_checked=raw_checked,analysis_sha256=sha(P/'analysis.json'),contract_sha256=sha(P/'contract.json'),
  verifier_sha256=sha(__file__),max_numerical_error=max(errors.values()),new_GPU=0,new_generation=0,
  errors=errors,scope='Saved statistics, direct old development predictions, cohort metadata, hashes and algebra; not generation utility or causal private signal.')
 (P/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2))

if __name__=='__main__':main()

