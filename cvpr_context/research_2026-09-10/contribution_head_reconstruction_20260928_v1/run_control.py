"""2-2: keep the existing source head; replace class midpoint with public P."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet
from construct_labels import solve

def construct(out):
 tick=time.monotonic();io=Inputs(out,'construct');check(not (out/'label_seal.json').exists(),'No repeated label solve')
 sel=io.nz('old_selection');rows=io.js('old_manifest')['rows'];pd=io.nz('P_DINO');pr=io.nz('P_RN');same_order(pd,pr)
 check(set(pd['roles'])==set(pr['roles'])=={'P'},'Public P only')
 D=pd['z'].transpose(1,0,2).reshape(813,64).astype(float);R=pr['z'].astype(float)
 check(np.array_equal(R[sel['indices']],sel['ResNet18']),'Fixed public128')
 for k in ['image_ids','patient_ids','labels']:check(np.array_equal(pr[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier order')
 sys.path.insert(0,C['historical_code_dir']);from fixed_image_compilation import pw
 omega=pw(pd);check(np.array_equal(omega,pw(pr)) and abs(omega.sum()-1)<1e-12,'Public weighting')
 HD=D.T@(omega[:,None]*D)+.1*np.eye(64);HR=R.T@(omega[:,None]*R)+.1*np.eye(128)
 # Only the public RN maps are used from the historical container.
 with np.load(io.path('public_maps'),allow_pickle=False) as f:
  T=[f['ResNet18_T'+str(c)].copy() for c in (0,1)];b=[f['ResNet18_b'+str(c)].copy() for c in (0,1)]
 mut=io.nz('public_target')['target'].reshape(2,64).astype(float)
 # Independently verify this is exactly the P-only class/visit aggregation.
 Cs=[1.9061329951133719,1.5917066731966771];rebuilt=[]
 pid=pd['patient_ids'].astype(str);y=pd['labels']
 for cl in (0,1):
  vals=[]
  for person in sorted(set(pid[y==cl])):
   x=D[(pid==person)&(y==cl)].mean(0);vals.append(x*min(1,Cs[cl]/max(np.linalg.norm(x),1e-300)))
  rebuilt.append(np.mean(vals,axis=0))
 pe=float(abs(np.stack(rebuilt)-mut).max());check(pe<1e-12,'P target is not method-matched')
 sp=mut.mean(0);tw=io.nz('prior_target_weights');seal=io.js('prior_label_seal')
 check(io.contract['inputs']['prior_target_weights']['sha256']==seal['files']['target_weights.npz'],'Source target provenance')
 (out/'label_packets').mkdir(exist_ok=False);labels={};fits={};targets={};files=[];inverse_err=0.;moment_err=0.
 for draw in ('DP1','DP2'):
  io.check_time();wd=tw[draw+'_DINO'];d=HD@wd
  inverse_err=max(inverse_err,float(abs(np.linalg.solve(HD,d)-wd).max()))
  # No private class midpoint/DP means are read by this construction.
  mu0=sp-d;mu1=sp+d
  br=.5*(mu1@T[1]+b[1]-mu0@T[0]-b[0])
  independent=.5*((T[1]+T[0]).T@d+(T[1]-T[0]).T@sp+b[1]-b[0])
  moment_err=max(moment_err,float(abs(br-independent).max()))
  wr=np.linalg.solve(HR,br);name=draw+'_HR_RN_ONLY';targets[name]=wr
  l,detail=solve({'ResNet18':R},{'ResNet18':sel['ResNet18'].astype(float)},omega,{'ResNet18':wr},['ResNet18'])
  labels[name]=l;fits[name]=detail;n='label_packets/'+name+'.csv';export_packet(rows,l,out/n);files.append(n)
 check(inverse_err<1e-10 and moment_err<1e-12,'Head or map algebra mismatch')
 np.savez_compressed(out/'soft_labels.npz',**labels);np.savez_compressed(out/'HR_target_weights.npz',**targets)
 io.verify();dump(out/'construction_inputs.json',io.receipts)
 result={'item':'2-2','status':'LABELS_READY','fits':fits,'P_target_reconstruction_max_abs':pe,'source_head_recovery_max_abs':inverse_err,
 'independent_target_formula_max_abs':moment_err,'new_labels':2,'private_class_midpoint_used':False,'V_access':False,'new_private_releases':0,'seconds':time.monotonic()-tick}
 dump(out/'construction_results.json',result)
 write_seal(out,'label_seal.json','TWO_HR_LABELS_SEALED_BEFORE_EVALUATION',['soft_labels.npz','HR_target_weights.npz','construction_results.json','construction_inputs.json']+files)
 print(json.dumps({'item':'2-2','phase':'CONSTRUCTED','seconds':result['seconds'],'P_error':pe,'head_recovery':inverse_err,'map_algebra':moment_err,'max_KKT':max(v['independent_projected_gradient_max_abs'] for v in fits.values())}),flush=True)

def evaluate(out):
 tick=time.monotonic();io=Inputs(out,'evaluate');check(not (out/'results.json').exists(),'No repeated evaluation')
 check_seal(out,'label_seal.json','TWO_HR_LABELS_SEALED_BEFORE_EVALUATION')
 labels=packet(out/'soft_labels.npz');sel=io.nz('old_selection');old=io.js('comparison_results');op=io.nz('comparison_predictions');ob=io.nz('comparison_bootstrap');seal=io.js('comparison_completion')
 for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz')]:
  check(io.contract['inputs'][k]['sha256']==seal['files'][n],'Previous result seal')
 check(len(op['names'])==20,'Expected all prior twenty readouts')
 sys.path.insert(0,C['code_root']);from prrd_pilot_20260921.common import bootstrap_metrics
 from sklearn.metrics import roc_auc_score,average_precision_score
 sched=io.nz('patient_bootstrap');counts=sched['patient_counts'];ids0=sched['patient_ids'];h=io.contract['inputs']['patient_bootstrap']['sha256']
 check(str(ob['source_draws_sha256'].item())==h and counts.shape==(2000,2026),'Identical patient bootstrap')
 metrics=dict(old['metrics']);scores={str(n):op['scores'][:,i] for i,n in enumerate(op['names'])};boots={str(n):ob['metrics'][:,i,:] for i,n in enumerate(ob['names'])}
 weights={};checks={'ridge_primal_dual_max_abs':0.,'bootstrap_sklearn_max_abs':0.};ref=None
 for rec,key in [('DenseNet','P_DN'),('ResNet18','P_RN')]:
  p=io.nz(key);v=io.nz('V_'+rec);same_order(v,op);check(set(p['roles'])=={'P'},'P role')
  for k in ['image_ids','patient_ids','labels']:check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier order')
  if ref is None:ref=v
  else:same_order(ref,v)
  z=p['z'][sel['indices']].astype(float);zv=v['z'].astype(float);pid=v['patient_ids'].astype(np.int64);y=v['labels'];unique,inv=np.unique(pid,return_inverse=True)
  check(np.array_equal(unique,ids0),'Patient ids')
  for draw in ['DP1','DP2']:
   io.check_time();name=draw+'_HR_RN_ONLY_'+rec;l=labels[draw+'_HR_RN_ONLY']
   w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128);dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@w
   err=float(abs(s-zv@dual).max());checks['ridge_primal_dual_max_abs']=max(checks['ridge_primal_dual_max_abs'],err);check(err<1e-10,'Ridge check')
   bb=bootstrap_metrics(y,s,pid,counts,ids0);points=[roc_auc_score(y,s),average_precision_score(y,s)]
   for j in [0,17,1999]:
    sw=counts[j,inv];rr=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
    err=float(abs(bb[j]-rr).max());checks['bootstrap_sklearn_max_abs']=max(checks['bootstrap_sklearn_max_abs'],err);check(err<1e-12,'Bootstrap check')
   scores[name]=s;boots[name]=bb;weights[name]=w
   metrics[name]={m:{'point':float(points[k]),'patient_cluster_95':np.quantile(bb[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
  print(json.dumps({'item':'2-2','phase':'EVALUATED','receiver':rec}),flush=True)
 contrasts={};means={};differences={}
 for rec in ['DenseNet','ResNet18']:
  for a,b in [('MT_RN_ONLY','HR_RN_ONLY'),('HR_RN_ONLY','KD_RN_ONLY'),('HR_RN_ONLY','DINO_ONLY')]:
   pts=[];ds=[];harm=False
   for draw in ['DP1','DP2']:
    aa=draw+'_'+a+'_'+rec;bb=draw+'_'+b+'_'+rec;key=draw+'_'+a+'_minus_'+b+'_'+rec
    d=boots[aa]-boots[bb];pt=np.array([metrics[aa][m]['point']-metrics[bb][m]['point'] for m in ['AUROC','AP']]);pts.append(pt);ds.append(d);differences[key]=d
    contrasts[key]={m:{'delta':float(pt[k]),'patient_cluster_95':np.quantile(d[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
    harm|=contrasts[key]['AP']['patient_cluster_95'][1]<0
   mp=np.mean(pts,axis=0);md=np.mean(ds,axis=0);key='MEAN_'+a+'_minus_'+b+'_'+rec;differences[key]=md
   value={m:{'mean_delta':float(mp[k]),'fixed_two_release_patient_cluster_95':np.quantile(md[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
   criteria={'both_AUROC_positive':bool(all(p[0]>0 for p in pts)),'mean_AUROC_at_least_0_01':bool(mp[0]>=.01),
    'mean_AUROC_CI_lower_positive':bool(value['AUROC']['fixed_two_release_patient_cluster_95'][0]>0),'mean_AP_nonnegative':bool(mp[1]>=0),'no_release_significant_AP_harm':not bool(harm)}
   value['development_criteria']=criteria;value['passes_development_hurdle']=all(criteria.values());means[key]=value
 hr_pass=means['MEAN_HR_RN_ONLY_minus_DINO_ONLY_DenseNet']['passes_development_hurdle']
 mt_pass=old['fixed_two_release_mean_contrasts']['MEAN_MT_RN_ONLY_minus_DINO_ONLY_DenseNet']['passes_development_hurdle']
 decisions={'extra_private_midpoint_supported':means['MEAN_MT_RN_ONLY_minus_HR_RN_ONLY_DenseNet']['passes_development_hurdle'],
  'head_reconstruction_over_simple_KD':means['MEAN_HR_RN_ONLY_minus_KD_RN_ONLY_DenseNet']['passes_development_hurdle'],
  'HR_over_source_only':hr_pass,'existing_MT_over_source_only':mt_pass,'proceed_2_3':bool(hr_pass or mt_pass)}
 np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
 np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(h))
 np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(differences)),differences=np.stack(list(differences.values()),axis=1))
 np.savez_compressed(out/'new_readout_weights.npz',**weights)
 io.verify();dump(out/'evaluation_inputs.json',io.receipts)
 result={'item':'2-2','status':'COMPLETE','created':now(),'metrics':metrics,'per_release_contrasts':contrasts,'fixed_two_release_mean_contrasts':means,'decisions':decisions,'verification':checks,
 'new_labels':2,'new_readouts':4,'reused_readouts':20,'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'uncertainty':'Fixed-bank/fixed-release patient-cluster bootstrap for image metrics','seconds':time.monotonic()-tick,'contract_sha256':sha(out/'contract.json')}
 dump(out/'results.json',result)
 write_seal(out,'completion.json','2-2_COMPLETE',['contract.json','label_seal.json','results.json','evaluation_inputs.json','predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','new_readout_weights.npz'])
 print(json.dumps({'item':'2-2','status':'COMPLETE','new_DenseNet':{k:v for k,v in metrics.items() if 'HR_RN_ONLY_DenseNet' in k},'means':{k:v for k,v in means.items() if k.endswith('DenseNet')},'decisions':decisions,'checks':checks,'seconds':result['seconds']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['construct','evaluate']);args=p.parse_args()
 (construct if args.phase=='construct' else evaluate)(OUT)
