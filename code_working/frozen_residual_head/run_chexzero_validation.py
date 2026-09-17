"""Fixed ten-checkpoint CheXzero validation; no generator or optimizer."""
import argparse, csv, gc, hashlib, importlib.metadata, json, platform, shutil, sys, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

C=Path(__file__).resolve().parents[1]
R=next(C.parent.glob('CVPR */research_2026-09-10'))
P=R/'spec_sources/chexzero_review_20260917_v1'
PLAN=C/'_reports/chexzero_evaluator_plan_20260917_v1'
O=C/'_reports/chexzero_validation_20260917_v1'
OLD=C/'_reports/padchest_validation_20260917_v1'
A=C/'_reports/patient_usage_audit_20260917_v1'
RAW=C/'_data/raw/nih_cxr14_pa_k10_plus_census_v1'
GROUPS=['Emphysema','Pneumothorax','No Finding','Effusion']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(p,rr):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def utc():return datetime.now(timezone.utc).isoformat()

def prepare():
 import torch,torchvision,PIL
 need(not O.exists(),'Output already exists');O.mkdir()
 plan=read(PLAN/'execution_spec.json');sources={}
 def bind(p):p=Path(p).resolve();sources[str(p)]=sha(p)
 for p in [__file__,Path(__file__).with_name('verify_chexzero_validation.py'),R/'TRACK1_CHEXZERO_VALIDATION_PROTOCOL_20260917.md',PLAN/'execution_spec.json',PLAN/'cohort_contract.json',PLAN/'selected_images_private.csv',PLAN/'remaining_method_development_private.csv',P/'asset_inspection.json',P/'official_release.json',RAW/'content_inventory_private.csv',C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv',A/'patient_usage_private.csv',A/'provisional_patient_split_private.csv',OLD/'contract.json',OLD/'result.json',OLD/'metrics.json',OLD/'selected_images_private.csv']:
  bind(p)
 for name,h in plan['source_sha256'].items():need(sha(P/name)==h,'Official source changed');bind(P/name)
 for cp in plan['checkpoints']:
  p=Path(plan['checkpoint_directory'])/cp['filename'];need(sha(p)==cp['sha256'],'Weight changed');bind(p)
 for mod in [torchvision.transforms,torchvision.transforms.functional,PIL.Image]:bind(mod.__file__)
 rr=rows(PLAN/'selected_images_private.csv')
 need(len(rr)==len({x['patient_id'] for x in rr})==80 and Counter(x['condition'] for x in rr)==Counter({g:20 for g in GROUPS}),'Fixed selection')
 previous={x['patient_id'] for x in rows(OLD/'selected_images_private.csv')}
 reserved={x['patient_id'] for x in rows(A/'provisional_patient_split_private.csv') if x['provisional_role']=='reserved_confirmation'}
 need(not {x['patient_id'] for x in rr}&(previous|reserved),'Cohort overlap')
 shutil.copyfile(PLAN/'selected_images_private.csv',O/'selected_images_private.csv')
 shutil.copyfile(PLAN/'remaining_method_development_private.csv',O/'remaining_development_private.csv')
 frozen=dict(read(OLD/'contract.json')['frozen_sha256'])
 for n,h in read(OLD/'result.json')['artifact_sha256'].items():frozen[str(OLD/n)]=h
 for p,h in frozen.items():need(sha(p)==h,'Historical evidence changed')
 contract=dict(created_utc=utc(),plan=plan,plan_sha256=sha(PLAN/'execution_spec.json'),source_sha256=sources,frozen_sha256=frozen,
  outputs={n:sha(O/n) for n in ['selected_images_private.csv','remaining_development_private.csv']},
  versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','scipy','scikit-learn','Pillow','h5py']},python=sys.version,platform=platform.platform(),
  groups=GROUPS,targets=GROUPS[:2],replay_indices=[next(i for i,x in enumerate(rr) if x['condition']==g) for g in GROUPS],
  tolerances=dict(raw_atol=1e-3,raw_rtol=1e-4,normalized_maxabs=2e-5,cosine_maxabs=5e-6,score_maxabs=3e-6),
  bootstrap_iterations=2000,bootstrap_seed=20260917,batch_size=8,seed=20260917,
  criterion=dict(rest_auc_min=.70,opposite_auc_min=.60,ci95_lower_strictly_above=.50),
  reserved_confirmation_consumed=False,new_training=0,new_generation=0,new_DP=0)
 save(O/'contract.json',contract)
 print(json.dumps({'prepared':True,'patients':80,'replay_indices':contract['replay_indices'],'contract_sha256':sha(O/'contract.json')}),flush=True)

def metrics(rr,scores,c):
 import numpy as np
 from scipy.stats import rankdata
 from sklearn.metrics import average_precision_score
 def calc(p,n):
  z=np.r_[p,n];y=np.r_[np.ones(len(p)),np.zeros(len(n))]
  au=(rankdata(z)[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
  return float(au),float(average_precision_score(y,z))
 rng=np.random.default_rng(c['bootstrap_seed']);out={};draws={}
 for j,t in enumerate(c['targets']):
  p=scores[[i for i,x in enumerate(rr) if x['condition']==t],j];other=c['targets'][1-j]
  for name,groups in [('rest',[g for g in GROUPS if g!=t]),('opposite',[other]),('normal',['No Finding']),('effusion',['Effusion'])]:
   ns=[scores[[i for i,x in enumerate(rr) if x['condition']==g],j] for g in groups];n=np.concatenate(ns);au,ap=calc(p,n);bb=np.empty((2000,2))
   for b in range(2000):bb[b]=calc(p[rng.integers(len(p),size=len(p))],np.concatenate([v[rng.integers(len(v),size=len(v))] for v in ns]))
   k=t+'__'+name;draws[k]=bb
   out[k]=dict(n_positive=len(p),n_negative=len(n),prevalence=len(p)/(len(p)+len(n)),auc=au,ap=ap,auc_ci95=np.quantile(bb[:,0],[.025,.975]).tolist(),ap_ci95=np.quantile(bb[:,1],[.025,.975]).tolist(),positive_quantiles=np.quantile(p,[0,.25,.5,.75,1]).tolist(),negative_quantiles=np.quantile(n,[0,.25,.5,.75,1]).tolist())
 decisions={}
 for t in c['targets']:
  a=out[t+'__rest'];b=out[t+'__opposite'];decisions[t]=dict(rest_pass=a['auc']>=.7 and a['auc_ci95'][0]>.5,opposite_pass=b['auc']>=.6 and b['auc_ci95'][0]>.5);decisions[t]['pass']=all(decisions[t].values())
 np.savez_compressed(O/'bootstrap.npz',**draws)
 return dict(metrics=out,criterion=decisions,joint_pass=all(v['pass'] for v in decisions.values()))

def run():
 import numpy as np,torch,h5py
 from PIL import Image
 from torchvision.transforms import Compose,Normalize,Resize,InterpolationMode
 started=time.perf_counter();con=read(O/'contract.json');plan=con['plan'];rr=rows(O/'selected_images_private.csv')
 need(not (O/'run_started.json').exists(),'Frozen run cannot be repeated')
 for p,h in con['source_sha256'].items():need(sha(p)==h,'Source mutation '+p)
 for n,h in con['outputs'].items():need(sha(O/n)==h,'Prepared output mutation')
 for k,v in con['versions'].items():need(importlib.metadata.version(k)==v,'Package version changed')
 save(O/'run_started.json',dict(utc=utc(),contract_sha256=sha(O/'contract.json')))
 torch.manual_seed(con['seed']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 sys.path.insert(0,str(P/'source'))
 import clip,model as official_model,zero_shot
 zero_shot.tqdm=lambda iterable,*args,**kwargs:iterable
 need(Path(zero_shot.__file__).resolve()==(P/'source/zero_shot.py').resolve(),'Official module binding')
 inv=rows(RAW/'content_inventory_private.csv');byid={x['image_id']:x for x in inv};sha_pat=defaultdict(set)
 for x in inv:sha_pat[x['sha256'].lower()].add(x['patient_id'])
 meta={x['Image Index']:x for x in rows(C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv')}
 transform=Compose([Normalize((101.48761,)*3,(83.43944,)*3),Resize(224,interpolation=InterpolationMode.BICUBIC,antialias=False)])
 prep=[];gray320=[];audit=[]
 for row in rr:
  path=RAW/'images'/row['image_id'];entry=byid[row['image_id']];m=meta[row['image_id']]
  need(sha(path)==row['sha256']==entry['sha256'].lower() and path.stat().st_size==int(entry['bytes']),'PNG binding')
  need(sha_pat[row['sha256']]=={row['patient_id']},'Cross-patient exact duplicate')
  need(str(int(m['Patient ID']))==row['patient_id'] and m['View Position']=='PA' and set(m['Finding Labels'].split('|'))==set(row['finding_labels'].split('|')),'Official metadata binding')
  with Image.open(path) as im:
   im.load();arr=np.asarray(im);mode=im.mode
   need(im.format=='PNG' and im.size==(1024,1024) and arr.dtype==np.uint8,'PNG format')
   if arr.ndim==3:
    need(np.array_equal(arr[:,:,0],arr[:,:,1]) and np.array_equal(arr[:,:,0],arr[:,:,2]),'Gray channels')
    if arr.shape[2]==4:need(np.all(arr[:,:,3]==255),'Opaque alpha')
    arr=arr[:,:,0]
   need(arr.ndim==2,'Grayscale shape');image=Image.fromarray(arr)
   ratio=320/max(image.size);size=tuple(int(v*ratio) for v in image.size)
   resized=image.resize(size,Image.Resampling.LANCZOS);canvas=Image.new('L',(320,320));canvas.paste(resized,((320-size[0])//2,(320-size[1])//2))
   small=np.array(canvas);x=transform(torch.from_numpy(np.repeat(small[None],3,axis=0).astype(np.float32)))
   need(x.shape==(3,224,224) and torch.isfinite(x).all().item(),'Transformed input')
   gray320.append(small);prep.append(x.numpy());audit.append(dict(patient_id=row['patient_id'],image_id=row['image_id'],file_sha256=row['sha256'],mode=mode,gray320_sha256=hashlib.sha256(small.tobytes()).hexdigest(),tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()))
 xx=np.stack(prep);small=np.stack(gray320);np.save(O/'normalized_inputs.npy',xx);np.save(O/'grayscale320.npy',small);save(O/'image_audit.json',audit)
 with h5py.File(O/'official_input.h5','w') as f:f.create_dataset('cxr',data=small.astype(np.float32))
 ds=zero_shot.CXRTestDataset(str(O/'official_input.h5'),transform=transform)
 for i in range(80):need(torch.equal(ds[i]['img'],torch.from_numpy(xx[i])),'Official HDF5 dataset transform equality')
 ix=con['replay_indices'];loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ix),batch_size=1,shuffle=False)
 tokens=clip.tokenize(plan['prompts'],context_length=77);need(tokens.tolist()==plan['token_ids'],'Frozen tokens');np.save(O/'tokens.npy',tokens.numpy())
 print(json.dumps({'preprocessing_verified':80,'seconds':time.perf_counter()-started}),flush=True)
 torch.cuda.reset_peak_memory_stats();pack={};stats=[];cpu_seconds=0.0;gpu_seconds=0.0
 class Capture:
  def __init__(self,m):self.m=m;self.images=[];self.texts=[]
  def encode_image(self,x):
   y=self.m.encode_image(x);self.images.append(y.detach().clone());return y
  def encode_text(self,x):
   y=self.m.encode_text(x);self.texts.append(y.detach().clone());return y
 def norm(x):return x/x.norm(dim=-1,keepdim=True)
 def soft(z):return np.exp(z[:,[0,2]])/(np.exp(z[:,[0,2]])+np.exp(z[:,[1,3]]))
 tol=con['tolerances'];allcos=[];allprob=[];allfeatures=[];alltext=[]
 with torch.inference_mode():
  for k,cp in enumerate(plan['checkpoints']):
   prefix=f'm{k:02d}_';t=time.perf_counter();state=torch.load(Path(plan['checkpoint_directory'])/cp['filename'],map_location='cpu',weights_only=True,mmap=True)
   net=official_model.build_model(dict(state)).float().eval()
   need(set(net.state_dict())==set(state) and all(torch.equal(net.state_dict()[n],v.float()) for n,v in state.items()),'Full strict checkpoint values')
   cpu_model=net;capture=Capture(cpu_model);ct=time.perf_counter()
   cpu_prob=zero_shot.run_softmax_eval(capture,loader,con['targets'],('{}','no {}'),context_length=77)
   cpu_raw=torch.cat(capture.images);cpu_text_raw=torch.cat(capture.texts)[[0,2,1,3]]
   cpu_feat=norm(cpu_raw);cpu_tf=norm(norm(cpu_text_raw));cpu_cos=(cpu_feat[:4]@cpu_tf.T).numpy()
   need(torch.equal(cpu_raw[:4],cpu_raw[4:]),'CPU positive/negative image repeat')
   need(np.max(np.abs(soft(cpu_cos)-cpu_prob))<=tol['score_maxabs'],'Official CPU pair score equivalence')
   elapsed_cpu=time.perf_counter()-ct;cpu_seconds+=elapsed_cpu
   net=cpu_model.cuda();torch.cuda.synchronize();gt=time.perf_counter()
   gpu_text_raw=net.encode_text(tokens.cuda());tf=norm(norm(gpu_text_raw));features=[]
   for b in range(0,80,con['batch_size']):features.append(norm(net.encode_image(torch.from_numpy(xx[b:b+con['batch_size']]).cuda())).cpu())
   feat=torch.cat(features);cos=(feat.cuda()@tf.T).cpu().numpy();prob=soft(cos)
   gpu_raw=net.encode_image(torch.from_numpy(xx[ix]).cuda());gpu_feat=norm(gpu_raw);gpu_cos=(gpu_feat@tf.T).cpu().numpy();gpu_prob=soft(gpu_cos)
   torch.cuda.synchronize();elapsed_gpu=time.perf_counter()-gt;gpu_seconds+=elapsed_gpu
   cr=cpu_raw[:4].numpy();gr=gpu_raw.cpu().numpy();ctr=cpu_text_raw.numpy();gtr=gpu_text_raw.cpu().numpy();cf=cpu_feat[:4].numpy();gf=gpu_feat.cpu().numpy();ctf=cpu_tf.numpy();gtf=tf.cpu().numpy()
   record=dict(filename=cp['filename'],cpu_seconds=elapsed_cpu,gpu_seconds=elapsed_gpu,raw_image_maxabs=float(np.max(np.abs(cr-gr))),raw_text_maxabs=float(np.max(np.abs(ctr-gtr))),normalized_image_maxabs=float(np.max(np.abs(cf-gf))),normalized_text_maxabs=float(np.max(np.abs(ctf-gtf))),cosine_maxabs=float(np.max(np.abs(cpu_cos-gpu_cos))),score_maxabs=float(np.max(np.abs(cpu_prob-gpu_prob))),batch_score_maxabs=float(np.max(np.abs(prob[ix]-gpu_prob))))
   # Preserve the packet even when a fixed numeric check fails.
   local=dict(cpu_image_raw=cpu_raw.numpy(),cpu_image_normalized=cpu_feat.numpy(),cpu_text_raw=ctr,cpu_text_normalized=ctf,cpu_cosine=cpu_cos,cpu_score=cpu_prob,gpu_replay_raw=gr,gpu_replay_normalized=gf,gpu_text_raw=gtr,gpu_text_normalized=gtf,gpu_replay_cosine=gpu_cos,gpu_replay_score=gpu_prob)
   np.savez(O/(prefix+'replay.npz'),**local)
   need(np.allclose(cr,gr,atol=tol['raw_atol'],rtol=tol['raw_rtol']) and np.allclose(ctr,gtr,atol=tol['raw_atol'],rtol=tol['raw_rtol']),'CPU/GPU raw feature tolerance')
   need(max(record['normalized_image_maxabs'],record['normalized_text_maxabs'])<=tol['normalized_maxabs'],'CPU/GPU normalized tolerance')
   need(record['cosine_maxabs']<=tol['cosine_maxabs'] and max(record['score_maxabs'],record['batch_score_maxabs'])<=tol['score_maxabs'],'CPU/GPU score tolerance')
   allcos.append(cos);allprob.append(prob);allfeatures.append(feat.numpy());alltext.append(gtf);stats.append(record)
   print(json.dumps({'checkpoint':k+1,'of':10,'parity_pass':True,'seconds':time.perf_counter()-t,'max_score_difference':record['score_maxabs']}),flush=True)
   del net,cpu_model,capture,state;gc.collect();torch.cuda.empty_cache()
 probabilities=np.stack(allprob);scores=probabilities.mean(axis=0)
 cpu_ensemble=np.stack([np.load(O/f'm{k:02d}_replay.npz')['cpu_score'] for k in range(10)]).mean(axis=0)
 need(np.max(np.abs(cpu_ensemble-scores[ix]))<=tol['score_maxabs'],'Final CPU/GPU ensemble parity')
 need(np.isfinite(scores).all(),'Finite ensemble')
 np.savez(O/'model_outputs.npz',image_features=np.stack(allfeatures),text_features=np.stack(alltext),cosines=np.stack(allcos),probabilities=probabilities,ensemble=scores,replay_indices=ix,cpu_ensemble=cpu_ensemble)
 save(O/'parity.json',dict(per_checkpoint=stats,ensemble_maxabs=float(np.max(np.abs(cpu_ensemble-scores[ix]))),passed=True))
 table(O/'patient_scores_private.csv',[dict(patient_id=x['patient_id'],image_id=x['image_id'],condition=x['condition'],Emphysema=float(scores[i,0]),Pneumothorax=float(scores[i,1])) for i,x in enumerate(rr)])
 table(O/'evaluator_exclusions_private.csv',[dict(patient_id=x['patient_id'],image_id=x['image_id'],role='chexzero_evaluator_consumed_exclude_generation_reference',model_result_consumed=True) for x in rr])
 mt=time.perf_counter();met=metrics(rr,scores,con);save(O/'metrics.json',met);metric_seconds=time.perf_counter()-mt
 for p,h in con['source_sha256'].items():need(sha(p)==h,'Source changed after execution')
 for p,h in con['frozen_sha256'].items():need(sha(p)==h,'Historical result mutation')
 ds.img_dset.file.close()
 artifacts=[p for p in O.iterdir() if p.is_file() and p.name not in ['contract.json','run_started.json']]
 result=dict(status='COMPLETED_REAL_NIH_CHEXZERO_VALIDATION',completed_utc=utc(),contract_sha256=sha(O/'contract.json'),patients=80,checkpoints=10,
  total_seconds=time.perf_counter()-started,cpu_reference_seconds=cpu_seconds,gpu_inference_seconds=gpu_seconds,metric_seconds=metric_seconds,
  gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  GPU_image_calls=110,GPU_image_examples=840,GPU_text_calls=10,GPU_text_examples=40,CPU_image_calls=80,CPU_image_examples=80,CPU_text_calls=40,CPU_text_examples=40,
  backward_calls=0,new_training=0,new_generation=0,new_DP=0,reserved_confirmation_consumed=False,parity_passed=True,
  joint_pass=met['joint_pass'],criterion=met['criterion'],remaining_development_patients=4053,remaining_exclusive_emphysema=44,
  artifact_sha256={p.name:sha(p) for p in artifacts})
 save(O/'result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
 try:prepare() if args.mode=='prepare' else run()
 except Exception:
  if O.exists() and not (O/'failure.json').exists():save(O/'failure.json',dict(utc=utc(),mode=args.mode,traceback=traceback.format_exc()))
  raise
