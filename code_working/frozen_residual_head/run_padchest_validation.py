"""Prospectively fixed PadChest evaluator validation. No generator or optimizer."""
import argparse,csv,hashlib,json,time,sys,platform
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone

CODE=Path(__file__).resolve().parents[1]
R=next(CODE.parent.glob('CVPR */research_2026-09-10'))
OUT=CODE/'_reports/padchest_validation_20260917_v1'
AUD=CODE/'_reports/patient_usage_audit_20260917_v1'
RAW=CODE/'_data/raw/nih_cxr14_pa_k10_plus_census_v1'
SALT='cvpr-padchest-validation-20260917-v1'
GROUPS=['Emphysema','Pneumothorax','No Finding','Effusion']
WEIGHT=Path.home()/'.torchxrayvision/models_data/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
WEIGHT_SHA='a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521'

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(p,x):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def key(k,v):return hashlib.sha256(f'{SALT}|{k}|{v}'.encode()).hexdigest()
def category(rr):
 u=set().union(*(set(x['finding_labels'].split('|')) for x in rr))
 e='Emphysema' in u;p='Pneumothorax' in u
 return 'dual' if e and p else 'Emphysema' if e else 'Pneumothorax' if p else 'Effusion' if 'Effusion' in u else 'No Finding' if any(x['finding_labels']=='No Finding' for x in rr) else 'other'

def prepare():
 import torchxrayvision as xrv,importlib.metadata as im
 need(not OUT.exists(),'Existing output directory');OUT.mkdir()
 sources={}
 def bind(p):p=Path(p).resolve();sources[str(p)]=sha(p);return p
 inv=rows(bind(RAW/'content_inventory_private.csv'))
 ledger=rows(bind(AUD/'patient_usage_private.csv'));split=rows(bind(AUD/'provisional_patient_split_private.csv'))
 bind(AUD/'audit.json');bind(AUD/'contract.json');bind(AUD/'verification.json')
 bind(__file__);bind(Path(__file__).with_name('verify_padchest_validation.py'))
 bind(R/'TRACK1_PADCHEST_VALIDATION_PROTOCOL_20260917.md')
 meta=rows(bind(CODE/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv'))
 mi={x['Image Index']:x for x in meta}
 bind(WEIGHT);need(sources[str(WEIGHT.resolve())]==WEIGHT_SHA,'Checkpoint SHA')
 for mod in [xrv.models,xrv.datasets,xrv.utils]:bind(mod.__file__)
 # Historical artifacts are bound but no model is executed on them.
 frozen={}
 for p,h in read(AUD/'contract.json')['source_sha256'].items():
  need(sha(p)==h,'Prior audit source changed '+p)
 bind(CODE/'_reports/medical_head_20260916_v1/generation_manifest.json')
 for rr in read(CODE/'_reports/medical_head_20260916_v1/generation_manifest.json'):
  p=CODE/'_reports/medical_head_20260916_v1'/rr['image_path'];frozen[str(p.resolve())]=rr['image_sha256']
 adopt=CODE/'_reports/public_medical_backbone_20260916_v1/adoption_status.json';frozen[str(adopt.resolve())]=sha(adopt)
 by=defaultdict(list);dups=defaultdict(set)
 for x in inv:by[x['patient_id']].append(x);dups[x['sha256'].lower()].add(x['patient_id'])
 old={x['patient_id'] for x in ledger if x['used_in_evaluator_preflight']=='1' and x['locked_cvpr_calibration_test']=='0' and x['locked_thesis_final']=='0'}
 dev={x['patient_id'] for x in split if x['provisional_role']=='target_development'}
 reserved={x['patient_id'] for x in split if x['provisional_role']=='reserved_confirmation'}
 need(len(old)==324 and len(dev)==len(reserved)==4213 and not(old&dev or old&reserved or dev&reserved),'Pool binding')
 pools={'old_nonlocked':old,'development_supplement':dev};counts={};chosen=[];supp=set()
 for name,pp in pools.items():
  cat={p:category(by[p]) for p in pp};counts[name]=dict(Counter(cat.values()))
  for g in GROUPS+(['dual'] if name=='old_nonlocked' else []):
   pats=sorted([p for p in pp if cat[p]==g],key=lambda p:key('patient',p))
   if name=='development_supplement':need(len(pats)>=20,'Insufficient support');pats=pats[:20];supp.update(pats)
   for p in pats:
    eligible=[x for x in by[p] if (x['finding_labels']=='No Finding' if g=='No Finding' else g in x['finding_labels'].split('|'))] if g!='dual' else [x for x in by[p] if {'Emphysema','Pneumothorax'}.issubset(x['finding_labels'].split('|'))]
    if g=='dual' and not eligible:eligible=[x for x in by[p] if {'Emphysema','Pneumothorax'}&set(x['finding_labels'].split('|'))]
    x=min(eligible,key=lambda x:key('image',x['image_id']));iid=x['image_id'];m=mi[iid]
    need(str(int(m['Patient ID']))==p and set(m['Finding Labels'].split('|'))==set(x['finding_labels'].split('|')),'Metadata mismatch')
    need(m['View Position']=='PA' and len(dups[x['sha256'].lower()])==1,'View or cross-patient exact duplicate')
    chosen.append(dict(stratum=name,group=g,patient_id=p,image_id=iid,sha256=x['sha256'].lower(),bytes=int(x['bytes']),finding_labels=x['finding_labels'],patient_hash=key('patient',p),image_hash=key('image',iid)))
 need(len(chosen)==309 and len({x['patient_id'] for x in chosen})==309 and len(supp)==80,'Fixed counts')
 remain=sorted(dev-supp,key=lambda p:key('patient',p));remain_counts=dict(Counter(category(by[p]) for p in remain))
 need(remain_counts['Emphysema']==64,'Preserve proposed 64 E-only reference candidates')
 table(OUT/'selected_images_private.csv',chosen)
 table(OUT/'evaluator_exclusions_private.csv',[{'patient_id':p,'cvpr_use':'evaluator_validation_only_exclude_future_generation_reference'} for p in sorted(supp,key=int)])
 table(OUT/'remaining_development_private.csv',[{'patient_id':p,'group':category(by[p]),'role':'remaining_target_development_candidate'} for p in remain])
 versions={k:im.version(k) for k in ['torch','torchvision','torchxrayvision','numpy','scikit-image','scikit-learn','scipy','Pillow']}
 contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),salt=SALT,source_sha256=sources,frozen_sha256=frozen,
  versions=versions,python=sys.version,platform=platform.platform(),weight_sha256=WEIGHT_SHA,model='densenet121-res224-pc',
  candidate_counts=counts,selected_counts=dict(Counter(x['stratum']+'/'+x['group'] for x in chosen)),remaining_development_counts=remain_counts,
  outputs={n:sha(OUT/n) for n in ['selected_images_private.csv','evaluator_exclusions_private.csv','remaining_development_private.csv']},
  bootstrap_iterations=2000,bootstrap_seed=20260917,batch_size=16,targets=['Emphysema','Pneumothorax'],
  criterion={'vs_rest_auc_min':.70,'vs_opposite_auc_min':.60,'both_ci95_lower_strictly_above':.50},
  reserved_confirmation_consumed=False,new_generation=0,new_training=0,new_DP=0)
 save(OUT/'contract.json',contract)
 print(json.dumps({k:contract[k] for k in ['created_utc','selected_counts','remaining_development_counts']},indent=2),flush=True)

def metrics(rr,logits,pathologies,c):
 import numpy as np
 from scipy.stats import rankdata
 from sklearn.metrics import average_precision_score
 def calc(pos,neg):
  zz=np.r_[pos,neg];yy=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
  auc=(rankdata(zz)[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
  return float(auc),float(average_precision_score(yy,zz))
 rng=np.random.default_rng(c['bootstrap_seed']);result={};draws={}
 for st in ['old_nonlocked','development_supplement']:
  for target in c['targets']:
   other='Pneumothorax' if target=='Emphysema' else 'Emphysema';ix=pathologies.index(target)
   pp=np.array([logits[i,ix] for i,x in enumerate(rr) if x['stratum']==st and x['group']==target])
   for name,gs in [('rest',[g for g in GROUPS if g!=target]),('opposite',[other]),('normal',['No Finding']),('effusion',['Effusion'])]:
    nn=[np.array([logits[i,ix] for i,x in enumerate(rr) if x['stratum']==st and x['group']==g]) for g in gs];neg=np.concatenate(nn)
    au,ap=calc(pp,neg);bb=np.empty((c['bootstrap_iterations'],2))
    for b in range(len(bb)):
     pb=pp[rng.integers(len(pp),size=len(pp))];nb=np.concatenate([v[rng.integers(len(v),size=len(v))] for v in nn]);bb[b]=calc(pb,nb)
    keyname=st+'__'+target+'__'+name;draws[keyname]=bb
    result[keyname]=dict(n_positive=len(pp),n_negative=len(neg),prevalence=len(pp)/(len(pp)+len(neg)),auc=au,ap=ap,
     auc_ci95=np.quantile(bb[:,0],[.025,.975]).tolist(),ap_ci95=np.quantile(bb[:,1],[.025,.975]).tolist(),
     positive_quantiles=np.quantile(pp,[0,.25,.5,.75,1]).tolist(),negative_quantiles=np.quantile(neg,[0,.25,.5,.75,1]).tolist())
 np.savez_compressed(OUT/'bootstrap.npz',**draws)
 decisions={}
 for t in c['targets']:
  a=result['development_supplement__'+t+'__rest'];b=result['development_supplement__'+t+'__opposite']
  decisions[t]=dict(rest_pass=a['auc']>=.7 and a['auc_ci95'][0]>.5,opposite_pass=b['auc']>=.6 and b['auc_ci95'][0]>.5)
  decisions[t]['pass']=all(decisions[t].values())
 return dict(metrics=result,development_criterion=decisions,joint_pass=all(x['pass'] for x in decisions.values()))

def run():
 import numpy as np,torch,torchxrayvision as xrv,importlib.metadata as im
 from PIL import Image
 start=time.perf_counter();c=read(OUT/'contract.json')
 need(not (OUT/'run_started.json').exists(),'No repeated inference into frozen run')
 for p,h in c['source_sha256'].items():need(sha(p)==h,'Source mutation '+p)
 for n,h in c['outputs'].items():need(sha(OUT/n)==h,'Prepared output mutation')
 for n,v in c['versions'].items():need(im.version(n)==v,'Package version changed')
 rr=rows(OUT/'selected_images_private.csv')
 save(OUT/'run_started.json',dict(utc=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(OUT/'contract.json')))
 torch.manual_seed(20260917);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 crop=xrv.datasets.XRayCenterCrop();resize=xrv.datasets.XRayResizer(224)
 prepared=[];pixels=[];image_audit=[]
 for row in rr:
  p=RAW/'images'/row['image_id'];need(p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],'Acquired image binding')
  with Image.open(p) as im:
   im.load();need(im.format=='PNG' and im.size==(1024,1024),'Image format/size');mode=im.mode;a=np.asarray(im)
   if a.ndim==3:
    need(np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2]),'Non-grayscale RGB')
    if a.shape[2]==4:need(np.all(a[:,:,3]==255),'Nonopaque alpha')
    a=a[:,:,0]
   need(a.dtype==np.uint8 and a.ndim==2,'8-bit grayscale required')
   b=resize(crop(xrv.datasets.normalize(a,255,reshape=True)))
   need(b.shape==(1,224,224) and np.isfinite(b).all(),'Normalized tensor')
   prepared.append(b);pixels.append(a)
   image_audit.append(dict(patient_id=row['patient_id'],image_id=row['image_id'],mode=mode,sha256=row['sha256'],pixel_min=int(a.min()),pixel_max=int(a.max()),tensor_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
 xx=np.stack(prepared);np.save(OUT/'normalized_inputs.npy',xx);save(OUT/'image_audit.json',image_audit)
 print(json.dumps({'images_verified':len(xx),'preprocessing_seconds':time.perf_counter()-start}),flush=True)
 model=xrv.models.DenseNet(weights=c['model'],cache_dir=str(WEIGHT.parent)).eval().cuda()
 need(model.weights_filename_local==str(WEIGHT) or Path(model.weights_filename_local).resolve()==WEIGHT.resolve(),'Loaded checkpoint path')
 need(not model.apply_sigmoid,'Unexpected sigmoid setting')
 pathologies=list(model.pathologies);need(pathologies.index('Pneumothorax')==3 and pathologies.index('Emphysema')==5,'Trained output mapping')
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();inferstart=time.perf_counter();outs=[];calls=0
 with torch.inference_mode():
  for i in range(0,len(xx),c['batch_size']):
   x=torch.from_numpy(xx[i:i+c['batch_size']]).cuda();outs.append(model.classifier(model.features2(x)).cpu().numpy());calls+=1
  # One already-selected supplement patient per group; deterministic replay, not extra subjects.
  replay_ix=[next(i for i,row in enumerate(rr) if row['stratum']=='development_supplement' and row['group']==g) for g in GROUPS]
  replay=torch.from_numpy(xx[replay_ix]).cuda();default=model(replay);calls+=1
  again=model.classifier(model.features2(replay));calls+=1
  thresholds=model.op_threshs.detach().cpu().numpy()
  np.savez(OUT/'forward_replay.npz',indices=replay_ix,default_output=default.cpu().numpy(),raw_replay=again.cpu().numpy(),op_threshs=thresholds)
 torch.cuda.synchronize();inference_seconds=time.perf_counter()-inferstart;peak=torch.cuda.max_memory_allocated();logits=np.concatenate(outs)
 need(np.isfinite(logits).all(),'Nonfinite logit');np.save(OUT/'logits.npy',logits)
 model_metadata=dict(pathologies=pathologies,op_threshs=[None if np.isnan(v) else float(v) for v in thresholds],apply_sigmoid=model.apply_sigmoid,input_resolution=model.input_resolution,source_checkpoint_sha256=sha(WEIGHT))
 save(OUT/'model_metadata.json',model_metadata)
 print(json.dumps({'inference_seconds':inference_seconds,'forward_calls':calls,'peak_GiB':peak/2**30}),flush=True)
 del model;torch.cuda.empty_cache();mt=time.perf_counter();m=metrics(rr,logits,pathologies,c);save(OUT/'metrics.json',m)
 for p,h in c['source_sha256'].items():need(sha(p)==h,'Source mutation after inference '+p)
 for p,h in c['frozen_sha256'].items():need(sha(p)==h,'Historical artifact changed')
 result=dict(status='COMPLETED_REAL_IMAGE_VALIDATION',completed_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(OUT/'contract.json'),
  patients=len(rr),images=len(rr),gpu=torch.cuda.get_device_name(0),inference_seconds=inference_seconds,metric_seconds=time.perf_counter()-mt,
  total_seconds=time.perf_counter()-start,peak_allocated_bytes=peak,forward_calls=calls,forward_examples=len(rr)+8,backward_calls=0,new_generation=0,new_DP=0,
  joint_pass=m['joint_pass'],development_criterion=m['development_criterion'],reserved_confirmation_consumed=False,
  artifact_sha256={n:sha(OUT/n) for n in ['normalized_inputs.npy','image_audit.json','logits.npy','forward_replay.npz','model_metadata.json','bootstrap.npz','metrics.json']})
 save(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
 if args.mode=='prepare':prepare()
 else:run()
