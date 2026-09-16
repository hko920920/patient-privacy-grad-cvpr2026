"""CPU-only fixed analysis of saved medical-head cohorts and private directions."""
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[_key]='1'
import csv,hashlib,json,time
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
from PIL import Image

CODE=Path(__file__).resolve().parent.parent
ROOT=CODE.parent
R=ROOT/('CVPR '+chr(0xc8fc)+chr(0xc81c)+' '+chr(0xd0d0)+chr(0xc0c9))/'research_2026-09-10'
OLD=CODE/'_reports/medical_head_20260916_v1'
OUT=CODE/'_reports/private_signal_20260917_v1'
META=CODE/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv'
S=7.5;LAM=.001

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def solve(a,b):
 l=np.linalg.cholesky(a+LAM*np.eye(64));return np.linalg.solve(l.T,np.linalg.solve(l,b))
def loss(a,b,q,w):return (q-2*np.sum(b*w)+np.sum(w*(a@w)))/4
def ip(a,u,v):return float(np.sum(u*(a@v)))
def relation(w,v,a):
 # v is compared to w; coefficient geometry and real-noise prediction geometry.
 result={}
 for name,g in [('coefficient',np.eye(64)),('development_prediction',a)]:
  n=ip(g,w,w);m=ip(g,v,v);cross=ip(g,w,v);alpha=cross/n;delta=v-w;rem=v-alpha*w
  result[name]=dict(cosine=cross/np.sqrt(n*m),delta_relative_norm=np.sqrt(max(0,ip(g,delta,delta))/n),best_scalar=alpha,
   remainder_relative_to_v=np.sqrt(max(0,ip(g,rem,rem))/m),
   delta_nonscalar_energy_fraction=ip(g,rem,rem)/ip(g,delta,delta),
   scalar_projection_inner_product=ip(g,w,rem))
 return result
def describe(values):
 x=np.asarray(values,dtype=float)
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)) if len(x)>1 else 0.,min=float(x.min()),max=float(x.max()))
def standard_diff(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
 return float((b.mean()-a.mean())/den) if den else None
def prepare():
 require(not OUT.exists(),'Analysis output already exists');OUT.mkdir(parents=True)
 files=[OLD/f for f in ['contract.json','patients.json','patient_statistics.npz','weights.npz','images.json','manifest.json','reference.json','fit.json','generation_manifest.json','evaluation.json','head_verification.json','generation_verification.json','evaluation_verification.json']]
 files += [META,Path(__file__),Path(__file__).with_name('verify_private_signal.py'),R/'TRACK1_PRIVATE_SIGNAL_PROTOCOL_20260917.md']
 private=sorted([x['patient_id'] for x in read(OLD/'patients.json') if x['split']=='train'])
 partitions=[]
 for k in range(8):
  order=sorted(private,key=lambda pid:digest(f'private-signal-halves-v1|{k}|{pid}'))
  partitions.append(dict(partition=k,left=order[:40],right=order[40:]))
 image_rows=read(OLD/'images.json');group=defaultdict(list)
 for row in image_rows:
  if row['split']=='train':group[row['patient_id']].append(row['image_id'])
 matched={pid:sorted(ids,key=lambda iid:digest('private-signal-matched-images-v1|'+iid))[:2] for pid,ids in group.items()}
 save(OUT/'contract.json',dict(schema='private-signal-audit/v1',created_utc=datetime.now(timezone.utc).isoformat(),
  source_sha256={str(p):sha(p) for p in files},ridge_lambda=LAM,partitions=partitions,matched_private_images=matched,
  counts={'public':32,'train':80,'eval':40,'reference':40},new_GPU=0,new_generation=0,DP=False,
  role_interpretation='Same NIH source roles; no new hospital-target cohort or final test',report='TRACK1_PRIVATE_SIGNAL_RESULTS_20260917.md'))
 print('PRIVATE_SIGNAL_CONTRACT_FIXED',flush=True)
def check():
 c=read(OUT/'contract.json')
 for p,h in c['source_sha256'].items():require(sha(p)==h,'Frozen source '+p)
 return c

def run():
 start=time.perf_counter();c=check()
 images=read(OLD/'images.json');reference=read(OLD/'reference.json');patients=read(OLD/'patients.json');manifest=read(OLD/'manifest.json')
 with META.open(encoding='utf-8-sig',newline='') as f:metadata={x['Image Index']:x for x in csv.DictReader(f)}
 refs=[dict(**x,split='reference') for x in reference]
 image_info=[]
 for row in images+refs:
  m=metadata[row['image_id']];require(int(row['patient_id'])==int(m['Patient ID']),'Metadata patient')
  require(sha(row['path'])==row['sha256'],'Image hash')
  with Image.open(row['path']) as im:
   size=list(im.size);mode=im.mode;a=np.asarray(im.convert('L').resize((256,256),Image.Resampling.LANCZOS),float)/255
  image_info.append(dict(image_id=row['image_id'],patient_id=row['patient_id'],split=row['split'],labels=m['Finding Labels'].split('|'),
    age=int(m['Patient Age']),sex=m['Patient Sex'],view=m['View Position'],actual_size=size,mode=mode,
    original_size=[int(m['OriginalImage[Width']),int(m['Height]'])],brightness=float(a.mean()),contrast=float(a.std()),
    p99_minus_p1=float(np.quantile(a,.99)-np.quantile(a,.01)),prompt=row.get('prompt'),original_record_role=row.get('original_record_role')))
 save(OUT/'image_metadata_and_proxies.json',image_info)
 labels=sorted(set(l for x in image_info for l in x['labels']));groups=defaultdict(list)
 for x in image_info:groups[(x['split'],x['patient_id'])].append(x)
 patient_info=[]
 for (split,pid),rs in groups.items():
  sexes=set(x['sex'] for x in rs)
  patient_info.append(dict(split=split,patient_id=pid,images=len(rs),age=float(np.median([x['age'] for x in rs])),
   sex=next(iter(sexes)) if len(sexes)==1 else 'mixed',label_fractions={l:float(np.mean([l in x['labels'] for x in rs])) for l in labels},
   any_labels=sorted(set(l for x in rs for l in x['labels'])),**{k:float(np.mean([x[k] for x in rs])) for k in ['brightness','contrast','p99_minus_p1']}))
 save(OUT/'patient_metadata.json',patient_info)
 cohorts={}
 for sp in c['counts']:
  ps=[x for x in patient_info if x['split']==sp];rs=[x for x in image_info if x['split']==sp]
  require(len(ps)==c['counts'][sp],'Cohort counts')
  cohorts[sp]=dict(patients=len(ps),images=len(rs),images_per_patient=dict(Counter(str(x['images']) for x in ps)),
   age=describe([x['age'] for x in ps]),selected_image_ages_above100=sum(x['age']>100 for x in rs),
   sex=dict(Counter(x['sex'] for x in ps)),view=dict(Counter(x['view'] for x in rs)),
   weak_label_image_counts={l:sum(l in x['labels'] for x in rs) for l in labels},
   weak_label_patient_mean_image_fraction={l:float(np.mean([x['label_fractions'][l] for x in ps])) for l in labels},
   weak_label_patient_any_counts={l:sum(l in x['any_labels'] for x in ps) for l in labels},
   proxies={k:describe([x[k] for x in ps]) for k in ['brightness','contrast','p99_minus_p1']},
   actual_sizes=dict(Counter(str(x['actual_size']) for x in rs)),
   original_width=describe([x['original_size'][0] for x in rs]),original_height=describe([x['original_size'][1] for x in rs]),
   original_roles=dict(Counter(str(x['original_record_role']) for x in rs)))
 pub=[x for x in patient_info if x['split']=='public'];pri=[x for x in patient_info if x['split']=='train']
 cohort_differences=dict(private_minus_public_smd={k:standard_diff([x[k] for x in pub],[x[k] for x in pri]) for k in ['age','brightness','contrast','p99_minus_p1']},
   label_fraction_difference={l:cohorts['train']['weak_label_patient_mean_image_fraction'][l]-cohorts['public']['weak_label_patient_mean_image_fraction'][l] for l in labels})
 print('COHORT_PROXIES_COMPLETE',flush=True)
 with np.load(OLD/'patient_statistics.npz') as f:A=f['A'];B=f['B'];Q=f['Q']
 ii={sp:[j for j,x in enumerate(patients) if x['split']==sp] for sp in ['public','train','eval']}
 ae=A[ii['eval']].mean(0);be=B[ii['eval']].mean(0);qe=float(Q[ii['eval']].mean())
 weights={name:solve(A[ids].mean(0),B[ids].mean(0)) for name,ids in [('public',ii['public']),('private',ii['train']),('pooled',ii['public']+ii['train'])]}
 weights['backbone']=np.zeros((64,4))
 with np.load(OLD/'weights.npz') as f:
  prior_errors={name:float(np.max(np.abs(weights[name]-f[name]))) for name in f.files}
 require(max(prior_errors.values())<1e-10,'Prior weights changed')
 # Rebuild all image statistics, and every eval patient x time stratum. No model loading.
 image_index={x['image_id']:j for j,x in enumerate(images)};n=len(images)
 IA=np.zeros((n,64,64));IB=np.zeros((n,64,4));IQ=np.zeros((n,4));IC=np.zeros(n,dtype=np.int64)
 epids=[patients[j]['patient_id'] for j in ii['eval']];eindex={pid:j for j,pid in enumerate(epids)}
 TA=np.zeros((40,8,64,64));TB=np.zeros((40,8,64,4));TQ=np.zeros((40,8,4));TC=np.zeros((40,8),dtype=np.int64)
 for z,row in enumerate(manifest):
  path=OLD/row['raw_path'];require(sha(path)==row['raw_sha256'],'Raw hash')
  with np.load(path) as f:
   h=S*f['features'].astype(float);r=f['target'].astype(float)-f['base'].astype(float);b=f['basis']
  small=h.T@h/len(h);cross=h.T@r/len(h);aa=np.kron(np.outer(b,b),small);bb=np.kron(b[:,None],cross);qq=np.mean(r*r,axis=0)
  j=image_index[row['image_id']];IA[j]+=aa;IB[j]+=bb;IQ[j]+=qq;IC[j]+=1
  if row['split']=='eval':
   k=eindex[row['patient_id']];d=row['draw_id'];TA[k,d]+=aa;TB[k,d]+=bb;TQ[k,d]+=qq;TC[k,d]+=1
  if (z+1)%1024==0:print('RAW_STATS',z+1,len(manifest),flush=True)
 require(np.all(IC==8) and np.all(TC==4),'Expected record count')
 IA/=IC[:,None,None];IB/=IC[:,None,None];IQ/=IC[:,None];TA/=TC[:,:,None,None];TB/=TC[:,:,None,None];TQ/=TC[:,:,None]
 re_errors=[]
 for j,pt in enumerate(patients):
  ids=[k for k,x in enumerate(images) if x['split']==pt['split'] and x['patient_id']==pt['patient_id']]
  re_errors.extend([float(np.max(np.abs(IA[ids].mean(0)-A[j]))),float(np.max(np.abs(IB[ids].mean(0)-B[j]))),abs(float(IQ[ids].mean(0).sum()-Q[j]))])
 require(max(re_errors)<1e-10,'Reconstructed statistics')
 np.savez_compressed(OUT/'image_statistics.npz',A=IA,B=IB,Q_channels=IQ)
 np.savez_compressed(OUT/'eval_time_statistics.npz',A=TA,B=TB,Q_channels=TQ)
 matched_ids=[image_index[iid] for ids in c['matched_private_images'].values() for iid in ids]
 weights['private_two_images']=solve(IA[matched_ids].mean(0),IB[matched_ids].mean(0))
 subsets=[]
 for part in c['partitions']:
  ws=[]
  for side in ['left','right']:
   ids=[j for j,x in enumerate(patients) if x['split']=='train' and x['patient_id'] in part[side]]
   name=f"half{part['partition']}_{side}";w=solve(A[ids].mean(0),B[ids].mean(0));weights[name]=w;ws.append(w)
  deltas=[w-weights['public'] for w in ws]
  subsets.append(dict(partition=part['partition'],delta_coefficient_cosine=float(np.sum(deltas[0]*deltas[1])/np.linalg.norm(deltas[0])/np.linalg.norm(deltas[1])),
    delta_prediction_cosine=ip(ae,deltas[0],deltas[1])/np.sqrt(ip(ae,deltas[0],deltas[0])*ip(ae,deltas[1],deltas[1])),
    left_development_MSE=loss(ae,be,qe,ws[0]),right_development_MSE=loss(ae,be,qe,ws[1])))
 np.savez_compressed(OUT/'analysis_weights.npz',**weights)
 comparisons={name:relation(weights['public'],weights[name],ae) for name in ['private','pooled','private_two_images']}
 dev_losses={name:[loss(A[j],B[j],Q[j],w) for j in ii['eval']] for name,w in weights.items()}
 dev={name:dict(mean_MSE=float(np.mean(vals)),better_than_public=int(np.sum(np.array(vals)<dev_losses['public'])),mean_difference_from_public=float(np.mean(np.array(vals)-dev_losses['public']))) for name,vals in dev_losses.items()}
 time_data=[]
 for d in range(8):
  a,b,q=TA[:,d].mean(0),TB[:,d].mean(0),float(TQ[:,d].mean(0).sum());wp=weights['public']
  time_data.append(dict(stratum=d,timestep_range=[min(x['timestep'] for x in manifest if x['draw_id']==d),max(x['timestep'] for x in manifest if x['draw_id']==d)],
    MSE={name:loss(a,b,q,weights[name]) for name in ['backbone','public','private','pooled','private_two_images']},
    delta_prediction_energy={name:ip(a,weights[name]-wp,weights[name]-wp)/4 for name in ['private','pooled']},
    public_prediction_energy=ip(a,wp,wp)/4))
 channel={}
 for name in ['public','private','pooled','private_two_images']:
  dw=weights[name]-weights['public'];channel[name]=[float(dw[:,k]@ae@dw[:,k]) for k in range(4)]
 conditions={}
 for label in ['No Finding','Effusion','Cardiomegaly','other']:
  selected=[x for x in image_info if x['split']=='eval' and ((label in x['labels']) if label!='other' else not set(x['labels'])&{'No Finding','Effusion','Cardiomegaly'})]
  cg=defaultdict(list)
  for x in selected:cg[x['patient_id']].append(image_index[x['image_id']])
  if not cg:conditions[label]=dict(patients=0,images=0);continue
  ca=np.mean([IA[js].mean(0) for js in cg.values()],0);cb=np.mean([IB[js].mean(0) for js in cg.values()],0);cq=float(np.mean([IQ[js].mean(0).sum() for js in cg.values()]))
  conditions[label]=dict(patients=len(cg),images=len(selected),MSE={name:loss(ca,cb,cq,weights[name]) for name in ['backbone','public','private','pooled','private_two_images']})
 # Same-loss total-W-penalty residual parameterization must recover the original solve.
 reparam={}
 wp=weights['public']
 for name,js in [('private',ii['train']),('pooled',ii['public']+ii['train'])]:
  a,b=A[js].mean(0),B[js].mean(0);delta=solve(a,b-(a+LAM*np.eye(64))@wp)
  reparam[name]=float(np.max(np.abs(wp+delta-weights[name])))
 # Existing generated images were already fixed. Pixel deltas are response size, not utility.
 gm=read(OLD/'generation_manifest.json');gidx={(x['seed_id'],x['prompt_id'],x['method']):x for x in gm};pixel=[]
 for sid in read(OLD/'contract.json')['generation_seeds']:
  for pid in read(OLD/'contract.json')['prompts']:
   arr={}
   for name in ['backbone','public','pooled']:
    row=gidx[(sid,pid,name)];p=OLD/row['image_path'];require(sha(p)==row['image_sha256'],'Generated PNG unchanged')
    with Image.open(p) as im:arr[name]=np.array(im,dtype=float)/255
   pixel.append(dict(seed_id=sid,prompt_id=pid,public_base_RMS=float(np.sqrt(np.mean((arr['public']-arr['backbone'])**2))),
     pooled_public_RMS=float(np.sqrt(np.mean((arr['pooled']-arr['public'])**2)))))
 result=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
  cohorts=cohorts,cohort_differences=cohort_differences,weight_comparisons=comparisons,development=dev,development_losses=dev_losses,
  eval_patient_ids=epids,private_disjoint_half_stability=subsets,time_strata=time_data,latent_channel_delta_energy=channel,development_conditions=conditions,
  original_weight_max_errors=prior_errors,reconstructed_statistics_max_error=max(re_errors),reparameterization_max_errors=reparam,
  pixel_response=pixel,new_GPU=0,new_generation=0,new_DP=False,private_generation_utility_established=False,
  contract_sha256=sha(OUT/'contract.json'),weights_sha256=sha(OUT/'analysis_weights.npz'),image_statistics_sha256=sha(OUT/'image_statistics.npz'),
  time_statistics_sha256=sha(OUT/'eval_time_statistics.npz'),metadata_sha256=sha(OUT/'image_metadata_and_proxies.json'))
 check();save(OUT/'analysis.json',result)
 print(json.dumps({'complete':True,'seconds':result['seconds'],'weight_comparisons':comparisons,'development':{k:dev[k] for k in ['public','private','pooled','private_two_images']}},indent=2),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run']);args=ap.parse_args();globals()[args.phase]()

