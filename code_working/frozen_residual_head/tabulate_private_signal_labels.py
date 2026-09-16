"""Post-hoc descriptive ALL-label table, no head fitting or selection."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parent.parent/'_reports/private_signal_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=read(P/'image_metadata_and_proxies.json')
old=P.parent/'medical_head_20260916_v1'
images=read(old/'images.json');idx={x['image_id']:i for i,x in enumerate(images)}
with np.load(P/'image_statistics.npz') as f:A=f['A'];B=f['B'];Q=f['Q_channels'].sum(1)
with np.load(P/'analysis_weights.npz') as f:W={k:f[k] for k in ['public','private','pooled','private_two_images']}
labels=sorted(set(l for x in meta for l in x['labels']))
rows=[]
for lab in labels:
 subset=defaultdict(list)
 for x in meta:
  if x['split']=='eval' and lab in x['labels']:subset[x['patient_id']].append(idx[x['image_id']])
 counts={sp:dict(images=sum(x['split']==sp and lab in x['labels'] for x in meta),patients=len({x['patient_id'] for x in meta if x['split']==sp and lab in x['labels']})) for sp in ['public','train','eval','reference']}
 scores={}
 if subset:
  aa=np.mean([A[js].mean(0) for js in subset.values()],0);bb=np.mean([B[js].mean(0) for js in subset.values()],0);qq=np.mean([Q[js].mean() for js in subset.values()])
  for name,w in W.items():
   one=(qq-2*np.sum(w*bb)+np.sum(w*(aa@w)))/4
   two=np.mean([np.mean([(Q[j]-2*np.einsum('ic,ic',w,B[j])+np.einsum('ic,ij,jc',w,A[j],w))/4 for j in js]) for js in subset.values()])
   if abs(one-two)>1e-10:raise ValueError('Aggregate/image loss inconsistency')
   scores[name]=float(one)
 rows.append(dict(label=lab,counts=counts,development_MSE=scores))
out=dict(posthoc=True,scope='All14 weak labels after seeing cohort counts; descriptive explanation only, no favorable subgroup selection or generation gate change.',rows=rows,new_GPU=0,new_head_fits=0,
 source_sha256={str(p):sha(p) for p in [Path(__file__),P/'image_metadata_and_proxies.json',P/'image_statistics.npz',P/'analysis_weights.npz']})
(P/'all_label_development_exploratory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in rows:
 print(r['label'],r['counts'],{k:round(v-r['development_MSE']['public'],9) for k,v in r['development_MSE'].items()} if r['development_MSE'] else {})

