"""2-3R1: one prespecified public-carrier repeat; all targets are frozen."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet
from construct_labels import solve
METHODS=['HR_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','ACKD_RN_ONLY','DINO_ONLY']
CONTROLS=['ACKD_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','DINO_ONLY']
PREFIX='S202_'

def select(out):
 from select_public128 import run
 run(out)

def construct(out):
 tick=time.monotonic();io=Inputs(out,'construct');check(not (out/'label_seal.json').exists(),'Single label solve')
 check_seal(out,'selection_seal.json','PUBLIC128_SEALED_BEFORE_TARGET_ACCESS')
 sel=packet(out/'public128_selection.npz');rows=json.loads((out/'public128_manifest.json').read_text(encoding='utf-8'))['rows']
 pd=io.nz('P_DINO');pr=io.nz('P_RN');same_order(pd,pr);check(set(pd['roles'])==set(pr['roles'])=={'P'},'P only')
 D=pd['z'].transpose(1,0,2).reshape(813,64).astype(float);R=pr['z'].astype(float)
 for k in ['image_ids','patient_ids','labels']:check(np.array_equal(pd[k][sel['indices']].astype(str),sel[k].astype(str)),'Selection row order')
 check(np.array_equal(D[sel['indices']],sel['DINO']) and np.array_equal(R[sel['indices']],sel['ResNet18']),'Selection feature provenance')
 sys.path.insert(0,C['historical_code_dir']);from fixed_image_compilation import pw
 omega=pw(pd);check(np.array_equal(omega,pw(pr)),'Public weights')
 target_sources={}
 for key,sealkey,filename in [('prior_target_weights','prior_label_seal','target_weights.npz'),('HR_targets','HR_label_seal','HR_target_weights.npz'),('correction_targets','correction_label_seal','corrected_target_weights.npz')]:
  s=io.js(sealkey);check(C['inputs'][key]['sha256']==s['files'][filename],'Target provenance')
  target_sources[key]=io.nz(key)
 oldsel=io.nz('old_selection')
 overlap={'patients':len(set(sel['patient_ids'].astype(str))&set(oldsel['patient_ids'].astype(str))),
  'images':len(set(sel['image_ids'].astype(str))&set(oldsel['image_ids'].astype(str))),'total':128}
 (out/'label_packets').mkdir(exist_ok=False);labels={};fits={};targets={};files=[]
 for draw in ['DP1','DP2']:
  for method in METHODS:
   io.check_time();name=draw+'_'+method
   if method=='DINO_ONLY':model='DINO';w=target_sources['prior_target_weights'][draw+'_DINO']
   elif method=='HR_RN_ONLY':model='ResNet18';w=target_sources['HR_targets'][name]
   else:model='ResNet18';w=target_sources['correction_targets'][name]
   xp={'DINO':D,'ResNet18':R};targets[name]=w.copy()
   l,detail=solve(xp,sel,omega,{model:w},[model]);labels[name]=l;fits[name]=detail
   file='label_packets/'+name+'.csv';export_packet(rows,l,out/file);files.append(file)
 np.savez_compressed(out/'soft_labels.npz',**labels);np.savez_compressed(out/'used_target_weights.npz',**targets)
 io.verify();dump(out/'construction_inputs.json',io.receipts)
 result={'item':'2-3R1','status':'LABELS_READY','fits':fits,'new_labels':10,'selection_overlap_with_first':overlap,
  'targets_refitted':False,'V_access':False,'new_Q_access':0,'new_DP_releases':0,'seconds':time.monotonic()-tick}
 dump(out/'construction_results.json',result)
 write_seal(out,'label_seal.json','TEN_REPEAT_LABELS_SEALED_BEFORE_EVALUATION',
  ['soft_labels.npz','used_target_weights.npz','construction_results.json','construction_inputs.json','selection_seal.json','public128_selection.npz','public128_manifest.json']+files)
 print(json.dumps({'item':'2-3R1','phase':'CONSTRUCTED','seconds':result['seconds'],'overlap':overlap,
  'max_KKT':max(x['independent_projected_gradient_max_abs'] for x in fits.values())}),flush=True)

def evaluate(out):
 tick=time.monotonic();io=Inputs(out,'evaluate');check(not (out/'results.json').exists(),'No repeat evaluation')
 check_seal(out,'label_seal.json','TEN_REPEAT_LABELS_SEALED_BEFORE_EVALUATION')
 sel=packet(out/'public128_selection.npz');labels=packet(out/'soft_labels.npz')
 old=io.js('comparison_results');op=io.nz('comparison_predictions');ob=io.nz('comparison_bootstrap');seal=io.js('comparison_completion')
 for key,file in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz')]:
  check(C['inputs'][key]['sha256']==seal['files'][file],'Old result seal')
 check(len(op['names'])==36,'Old36 results')
 sys.path.insert(0,C['code_root']);from prrd_pilot_20260921.common import bootstrap_metrics
 from sklearn.metrics import roc_auc_score,average_precision_score
 sched=io.nz('patient_bootstrap');counts=sched['patient_counts'];ids0=sched['patient_ids'];h=C['inputs']['patient_bootstrap']['sha256']
 check(str(ob['source_draws_sha256'].item())==h and counts.shape==(2000,2026),'Same bootstrap')
 metrics=dict(old['metrics']);scores={str(n):op['scores'][:,i] for i,n in enumerate(op['names'])};boots={str(n):ob['metrics'][:,i,:] for i,n in enumerate(ob['names'])}
 weights={};checks={'ridge_primal_dual_max_abs':0.,'bootstrap_sklearn_max_abs':0.};ref=None
 for rec,key in [('DenseNet','P_DN'),('ResNet18','P_RN')]:
  p=io.nz(key);v=io.nz('V_'+rec);same_order(v,op);check(set(p['roles'])=={'P'},'P role')
  for k in ['image_ids','patient_ids','labels']:check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier ordering')
  if ref is None:ref=v
  else:same_order(ref,v)
  z=p['z'][sel['indices']].astype(float);zv=v['z'].astype(float);pid=v['patient_ids'].astype(np.int64);y=v['labels'];unique,inv=np.unique(pid,return_inverse=True)
  check(np.array_equal(unique,ids0),'Patients')
  for draw in ['DP1','DP2']:
   for method in METHODS:
    io.check_time();name=PREFIX+draw+'_'+method+'_'+rec;l=labels[draw+'_'+method]
    w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128);dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@w
    er=float(abs(s-zv@dual).max());check(er<1e-10,'Ridge primal dual');checks['ridge_primal_dual_max_abs']=max(checks['ridge_primal_dual_max_abs'],er)
    bb=bootstrap_metrics(y,s,pid,counts,ids0);points=[roc_auc_score(y,s),average_precision_score(y,s)]
    for j in [0,17,1999]:
     sw=counts[j,inv];reference=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
     er=float(abs(bb[j]-reference).max());check(er<1e-12,'Bootstrap witness');checks['bootstrap_sklearn_max_abs']=max(checks['bootstrap_sklearn_max_abs'],er)
    scores[name]=s;boots[name]=bb;weights[name]=w
    metrics[name]={m:{'point':float(points[k]),'patient_cluster_95':np.quantile(bb[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
  print(json.dumps({'item':'2-3R1','phase':'EVALUATED','receiver':rec}),flush=True)
 contrasts={};means={};differences={}
 for rec in ['DenseNet','ResNet18']:
  for control in CONTROLS:
   pts=[];ds=[];harm=False
   for draw in ['DP1','DP2']:
    aa=PREFIX+draw+'_HR_RN_ONLY_'+rec;bb=PREFIX+draw+'_'+control+'_'+rec
    key=PREFIX+draw+'_HR_RN_ONLY_minus_'+control+'_'+rec
    d=boots[aa]-boots[bb];pt=np.array([metrics[aa][m]['point']-metrics[bb][m]['point'] for m in ['AUROC','AP']])
    pts.append(pt);ds.append(d);differences[key]=d
    contrasts[key]={m:{'delta':float(pt[k]),'patient_cluster_95':np.quantile(d[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
    harm|=contrasts[key]['AP']['patient_cluster_95'][1]<0
   mp=np.mean(pts,axis=0);md=np.mean(ds,axis=0);key='MEAN_'+PREFIX+'HR_RN_ONLY_minus_'+control+'_'+rec;differences[key]=md
   value={m:{'mean_delta':float(mp[k]),'fixed_two_release_patient_cluster_95':np.quantile(md[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
   cr={'both_AUROC_positive':bool(all(p[0]>0 for p in pts)),'mean_AUROC_at_least_0_01':bool(mp[0]>=.01),
    'mean_AUROC_CI_lower_positive':bool(value['AUROC']['fixed_two_release_patient_cluster_95'][0]>0),'mean_AP_nonnegative':bool(mp[1]>=0),'no_release_significant_AP_harm':not bool(harm)}
   value['development_criteria']=cr;value['passes_development_hurdle']=all(cr.values());means[key]=value
 replicate={control:means['MEAN_'+PREFIX+'HR_RN_ONLY_minus_'+control+'_DenseNet']['passes_development_hurdle'] for control in CONTROLS}
 decisions={'HR_over_each_control_new_selection':replicate,'HR_over_all_corrected_KD_repeated':all(replicate[x] for x in CONTROLS if x!='DINO_ONLY'),
  'HR_over_DINO_new_selection':replicate['DINO_ONLY'],'original_selection_source_only_gate_changed':False,
  'independent_cohort_or_new_noise':False,'novelty_established':False}
 np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
 np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(h))
 np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(differences)),differences=np.stack(list(differences.values()),axis=1))
 np.savez_compressed(out/'new_readout_weights.npz',**weights);io.verify();dump(out/'evaluation_inputs.json',io.receipts)
 result={'item':'2-3R1','status':'COMPLETE','created':now(),'metrics':metrics,'per_release_contrasts':contrasts,'fixed_two_release_mean_contrasts':means,'decisions':decisions,'verification':checks,
 'new_labels':10,'new_readouts':20,'reused_readouts':36,'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'uncertainty':'Same fixed two DP releases/P/V; selection repeat, not independent cohort or mechanism repetitions',
 'seconds':time.monotonic()-tick,'contract_sha256':sha(out/'contract.json')}
 dump(out/'results.json',result)
 write_seal(out,'completion.json','2-3R1_COMPLETE',['contract.json','selection_seal.json','label_seal.json','results.json','evaluation_inputs.json','predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','new_readout_weights.npz'])
 print(json.dumps({'item':'2-3R1','status':'COMPLETE','new_DenseNet':{k:v for k,v in metrics.items() if k.startswith(PREFIX) and k.endswith('DenseNet')},
 'DenseNet_means':{k:v for k,v in means.items() if k.endswith('DenseNet')},'decisions':decisions,'checks':checks,'seconds':result['seconds']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','construct','evaluate']);args=p.parse_args()
 {'select':select,'construct':construct,'evaluate':evaluate}[args.phase](OUT)

