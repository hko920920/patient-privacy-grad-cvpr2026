"""2-2R1: frozen public anchoring x covariance correction control."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet
from construct_labels import solve
RECIPES=['AKD_RN_ONLY','CKD_RN_ONLY','ACKD_RN_ONLY']
PAIRS=[('HR_RN_ONLY','ACKD_RN_ONLY'),('HR_RN_ONLY','AKD_RN_ONLY'),('HR_RN_ONLY','CKD_RN_ONLY'),
 ('AKD_RN_ONLY','KD_RN_ONLY'),('CKD_RN_ONLY','KD_RN_ONLY'),
 ('ACKD_RN_ONLY','AKD_RN_ONLY'),('ACKD_RN_ONLY','CKD_RN_ONLY'),
 ('AKD_RN_ONLY','DINO_ONLY'),('CKD_RN_ONLY','DINO_ONLY'),('ACKD_RN_ONLY','DINO_ONLY')]

def construct(out):
 tick=time.monotonic();io=Inputs(out,'construct');check(not (out/'label_seal.json').exists(),'No repeated solve')
 sel=io.nz('old_selection');rows=io.js('old_manifest')['rows'];pd=io.nz('P_DINO');pr=io.nz('P_RN');same_order(pd,pr)
 check(set(pd['roles'])==set(pr['roles'])=={'P'},'P only')
 D=pd['z'].transpose(1,0,2).reshape(813,64).astype(float);R=pr['z'].astype(float)
 check(np.array_equal(R[sel['indices']],sel['ResNet18']),'Fixed public128')
 for k in ['image_ids','patient_ids','labels']:check(np.array_equal(pr[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier order')
 sys.path.insert(0,C['historical_code_dir'])
 from fixed_image_compilation import pw
 from geometry_diagnostic import moments
 omega=pw(pd);check(np.array_equal(omega,pw(pr)) and abs(omega.sum()-1)<1e-12,'Weighting')
 G=D.T@(omega[:,None]*D);HD=G+.1*np.eye(64);HR=R.T@(omega[:,None]*R)+.1*np.eye(128)
 cross=R.T@(omega[:,None]*D);lam=.001*np.trace(G)/64
 mut=io.nz('public_target')['target'].reshape(2,64).astype(float)
 Cs=[1.9061329951133719,1.5917066731966771];rebuilt=[];pid=pd['patient_ids'].astype(str);y=pd['labels'].astype(int)
 for cl in (0,1):
  vals=[]
  for person in sorted(set(pid[y==cl])):
   v=D[(pid==person)&(y==cl)].mean(0);vals.append(v*min(1,Cs[cl]/max(np.linalg.norm(v),1e-300)))
  rebuilt.append(np.mean(vals,axis=0))
 pe=float(abs(np.stack(rebuilt)-mut).max());check(pe<1e-12,'P target')
 wdP=np.linalg.solve(HD,.5*(mut[1]-mut[0]))
 signed_y=2*y-1;wrP=np.linalg.solve(HR,R.T@(omega*signed_y))
 gc,bp=moments(pr,True);check(np.max(abs(HR-gc-.1*np.eye(128)))<1e-12 and np.max(abs(HR@wrP-bp))<1e-12,'Public RN head')
 AK=np.linalg.solve(HR,cross);AC=np.linalg.solve(HR,cross@np.linalg.solve(G+lam*np.eye(64),HD))
 with np.load(io.path('prior_target_weights'),allow_pickle=False) as f:
  tw={k:f[k] for k in ['DP1_DINO','DP2_DINO','DP1_KD_RN','DP2_KD_RN']}
 seal=io.js('prior_label_seal')
 check(io.contract['inputs']['prior_target_weights']['sha256']==seal['files']['target_weights.npz'],'Source provenance')
 (out/'label_packets').mkdir(exist_ok=False);labels={};fits={};targets={};files=[];checks={'public_target_max_abs':pe,'target_pseudoresponse_ridge_max_abs':0.,'old_KD_max_abs':0.,'public_anchor_invariant_max_abs':0.}
 for op in [AK,AC]:
  value=wrP+op@(wdP-wdP);checks['public_anchor_invariant_max_abs']=max(checks['public_anchor_invariant_max_abs'],float(abs(value-wrP).max()))
 for draw in ['DP1','DP2']:
  wd=tw[draw+'_DINO'];delta=wd-wdP
  checks['old_KD_max_abs']=max(checks['old_KD_max_abs'],float(abs(AK@wd-tw[draw+'_KD_RN']).max()))
  fixed={
   'AKD_RN_ONLY':(wrP+AK@delta,signed_y+D@delta),
   'CKD_RN_ONLY':(AC@wd,D@np.linalg.solve(G+lam*np.eye(64),HD@wd)),
   'ACKD_RN_ONLY':(wrP+AC@delta,signed_y+D@np.linalg.solve(G+lam*np.eye(64),HD@delta))}
  for recipe in RECIPES:
   io.check_time();wr,q=fixed[recipe]
   # Independent augmented least-squares verification, not the same normal solve.
   design=np.vstack([np.sqrt(omega[:,None])*R,np.sqrt(.1)*np.eye(128)])
   response=np.concatenate([np.sqrt(omega)*q,np.zeros(128)])
   independent=np.linalg.lstsq(design,response,rcond=None)[0]
   er=float(abs(wr-independent).max());check(er<1e-10,'Independent target ridge')
   checks['target_pseudoresponse_ridge_max_abs']=max(checks['target_pseudoresponse_ridge_max_abs'],er)
   name=draw+'_'+recipe;targets[name]=wr
   l,detail=solve({'ResNet18':R},{'ResNet18':sel['ResNet18'].astype(float)},omega,{'ResNet18':wr},['ResNet18'])
   labels[name]=l;fits[name]=detail;n='label_packets/'+name+'.csv';export_packet(rows,l,out/n);files.append(n)
 check(max(checks.values())<1e-10,'Construction algebra')
 np.savez_compressed(out/'soft_labels.npz',**labels);np.savez_compressed(out/'corrected_target_weights.npz',**targets)
 np.savez_compressed(out/'public_operators.npz',AK=AK,AC=AC,wDP=wdP,wRP=wrP,lambda_C=np.array(lam))
 io.verify();dump(out/'construction_inputs.json',io.receipts)
 result={'item':'2-2R1','status':'LABELS_READY','fits':fits,'verification':checks,'lambda_C':float(lam),
  'new_labels':6,'V_access':False,'new_Q_access':0,'new_DP_releases':0,'seconds':time.monotonic()-tick}
 dump(out/'construction_results.json',result)
 write_seal(out,'label_seal.json','SIX_CORRECTION_LABELS_SEALED_BEFORE_EVALUATION',
  ['soft_labels.npz','corrected_target_weights.npz','public_operators.npz','construction_results.json','construction_inputs.json']+files)
 print(json.dumps({'item':'2-2R1','phase':'CONSTRUCTED','seconds':result['seconds'],'checks':checks,'max_KKT':max(x['independent_projected_gradient_max_abs'] for x in fits.values())}),flush=True)

def evaluate(out):
 tick=time.monotonic();io=Inputs(out,'evaluate');check(not (out/'results.json').exists(),'No repeated evaluation')
 check_seal(out,'label_seal.json','SIX_CORRECTION_LABELS_SEALED_BEFORE_EVALUATION')
 labels=packet(out/'soft_labels.npz');sel=io.nz('old_selection')
 old=io.js('comparison_results');op=io.nz('comparison_predictions');ob=io.nz('comparison_bootstrap');seal=io.js('comparison_completion')
 for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz')]:
  check(io.contract['inputs'][k]['sha256']==seal['files'][n],'Previous result seal')
 check(len(op['names'])==24,'All old24 readouts')
 sys.path.insert(0,C['code_root']);from prrd_pilot_20260921.common import bootstrap_metrics
 from sklearn.metrics import roc_auc_score,average_precision_score
 sched=io.nz('patient_bootstrap');counts=sched['patient_counts'];ids0=sched['patient_ids'];h=io.contract['inputs']['patient_bootstrap']['sha256']
 check(str(ob['source_draws_sha256'].item())==h and counts.shape==(2000,2026),'Same bootstrap')
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
   for recipe in RECIPES:
    io.check_time();name=draw+'_'+recipe+'_'+rec;l=labels[draw+'_'+recipe]
    w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128);dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@w
    er=float(abs(s-zv@dual).max());check(er<1e-10,'Ridge check');checks['ridge_primal_dual_max_abs']=max(checks['ridge_primal_dual_max_abs'],er)
    bb=bootstrap_metrics(y,s,pid,counts,ids0);points=[roc_auc_score(y,s),average_precision_score(y,s)]
    for j in [0,17,1999]:
     sw=counts[j,inv];rr=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
     er=float(abs(bb[j]-rr).max());check(er<1e-12,'Bootstrap check');checks['bootstrap_sklearn_max_abs']=max(checks['bootstrap_sklearn_max_abs'],er)
    scores[name]=s;boots[name]=bb;weights[name]=w
    metrics[name]={m:{'point':float(points[k]),'patient_cluster_95':np.quantile(bb[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
  print(json.dumps({'item':'2-2R1','phase':'EVALUATED','receiver':rec}),flush=True)
 contrasts={};means={};differences={}
 for rec in ['DenseNet','ResNet18']:
  comparisons=[(a+'_minus_'+b,{a:1.,b:-1.}) for a,b in PAIRS]
  comparisons.append(('ANCHOR_BY_CORRECTION_INTERACTION',{'ACKD_RN_ONLY':1.,'AKD_RN_ONLY':-1.,'CKD_RN_ONLY':-1.,'KD_RN_ONLY':1.}))
  for stem,terms in comparisons:
   pts=[];ds=[];harm=False
   for draw in ['DP1','DP2']:
    names={draw+'_'+recipe+'_'+rec:v for recipe,v in terms.items()}
    d=sum(v*boots[n] for n,v in names.items());pt=sum(v*np.array([metrics[n][m]['point'] for m in ['AUROC','AP']]) for n,v in names.items())
    key=draw+'_'+stem+'_'+rec;pts.append(pt);ds.append(d);differences[key]=d
    contrasts[key]={m:{'delta':float(pt[k]),'patient_cluster_95':np.quantile(d[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
    harm|=contrasts[key]['AP']['patient_cluster_95'][1]<0
   mp=np.mean(pts,axis=0);md=np.mean(ds,axis=0);key='MEAN_'+stem+'_'+rec;differences[key]=md
   value={m:{'mean_delta':float(mp[k]),'fixed_two_release_patient_cluster_95':np.quantile(md[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
   if len(terms)==2:
    cr={'both_AUROC_positive':bool(all(p[0]>0 for p in pts)),'mean_AUROC_at_least_0_01':bool(mp[0]>=.01),
     'mean_AUROC_CI_lower_positive':bool(value['AUROC']['fixed_two_release_patient_cluster_95'][0]>0),'mean_AP_nonnegative':bool(mp[1]>=0),'no_release_significant_AP_harm':not bool(harm)}
    value['development_criteria']=cr;value['passes_development_hurdle']=all(cr.values())
   means[key]=value
 extra={recipe:means['MEAN_HR_RN_ONLY_minus_'+recipe+'_DenseNet']['passes_development_hurdle'] for recipe in RECIPES}
 versus_source={recipe:means['MEAN_'+recipe+'_minus_DINO_ONLY_DenseNet']['passes_development_hurdle'] for recipe in RECIPES}
 decisions={'HR_extra_value_over_all_corrections':all(extra.values()),'HR_vs_each_correction':extra,
  'standard_corrections_over_source_only':versus_source,'original_2_3_gate_changed':False,'original_2_3_remains_deferred':True,
  'novelty_established':False}
 np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
 np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(h))
 np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(differences)),differences=np.stack(list(differences.values()),axis=1))
 np.savez_compressed(out/'new_readout_weights.npz',**weights)
 io.verify();dump(out/'evaluation_inputs.json',io.receipts)
 result={'item':'2-2R1','status':'COMPLETE','created':now(),'metrics':metrics,'per_release_contrasts':contrasts,'fixed_two_release_mean_contrasts':means,'decisions':decisions,'verification':checks,
 'new_labels':6,'new_readouts':12,'reused_readouts':24,'new_Q_access':0,'new_DP_releases':0,'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
 'uncertainty':'Fixed-bank/fixed-release patient-cluster bootstrap for image metrics; supplementary comparisons exploratory',
 'seconds':time.monotonic()-tick,'contract_sha256':sha(out/'contract.json')}
 dump(out/'results.json',result)
 write_seal(out,'completion.json','2-2R1_COMPLETE',['contract.json','label_seal.json','results.json','evaluation_inputs.json','predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','new_readout_weights.npz'])
 print(json.dumps({'item':'2-2R1','status':'COMPLETE','new_DenseNet':{k:v for k,v in metrics.items() if any(x in k for x in RECIPES) and k.endswith('DenseNet')},
 'DenseNet_means':{k:v for k,v in means.items() if k.endswith('DenseNet')},'decisions':decisions,'checks':checks,'seconds':result['seconds']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['construct','evaluate']);args=p.parse_args()
 (construct if args.phase=='construct' else evaluate)(OUT)

