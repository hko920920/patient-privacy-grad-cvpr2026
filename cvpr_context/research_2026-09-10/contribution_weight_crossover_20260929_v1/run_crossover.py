"""2-2R2: correspondence-weight crossover; four labels, then eight readouts."""
import argparse, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs, check, dump, sha, now, packet, check_seal, write_seal, same_order, export_packet
from construct_labels import solve
NEW=['HR_B_RN_ONLY','ACKD_U_RN_ONLY']
OLD=['HR_RN_ONLY','ACKD_RN_ONLY','DINO_ONLY']
TERMS={
 'HR_U_minus_HR_B':{'HR_RN_ONLY':1.,'HR_B_RN_ONLY':-1.},
 'ACKD_U_minus_ACKD_B':{'ACKD_U_RN_ONLY':1.,'ACKD_RN_ONLY':-1.},
 'HR_U_minus_ACKD_U':{'HR_RN_ONLY':1.,'ACKD_U_RN_ONLY':-1.},
 'HR_B_minus_ACKD_B':{'HR_B_RN_ONLY':1.,'ACKD_RN_ONLY':-1.},
 'WEIGHT_BY_RECIPE_INTERACTION':{'HR_RN_ONLY':1.,'HR_B_RN_ONLY':-1.,'ACKD_U_RN_ONLY':-1.,'ACKD_RN_ONLY':1.},
 'HR_B_minus_DINO':{'HR_B_RN_ONLY':1.,'DINO_ONLY':-1.},
 'ACKD_U_minus_DINO':{'ACKD_U_RN_ONLY':1.,'DINO_ONLY':-1.},
 'HR_U_minus_DINO_REFERENCE':{'HR_RN_ONLY':1.,'DINO_ONLY':-1.},
 'HR_U_minus_ACKD_B_ORIGINAL_REFERENCE':{'HR_RN_ONLY':1.,'ACKD_RN_ONLY':-1.}
}

def construct(out):
 tick=time.monotonic();io=Inputs(out,'construct')
 check(not (out/'label_seal.json').exists(),'No repeated solve')
 sel=io.nz('old_selection');rows=io.js('old_manifest')['rows']
 pd=io.nz('P_DINO');pr=io.nz('P_RN');same_order(pd,pr)
 check(set(pd['roles'])==set(pr['roles'])=={'P'},'P only')
 D=pd['z'].transpose(1,0,2).reshape(813,64).astype(float);R=pr['z'].astype(float)
 check(np.array_equal(R[sel['indices']],sel['ResNet18']),'Fixed carrier features')
 for k in ['image_ids','patient_ids','labels']:
  check(np.array_equal(pr[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier row order')
 sys.path.insert(0,C['historical_code_dir'])
 from fixed_image_compilation import pw
 sys.path.insert(0,C['design_dir'])
 from analyze_public_maps import affine,clipped,CAPS
 omega=pw(pd);check(np.array_equal(omega,pw(pr)),'Fixed task weights')
 HD=D.T@(omega[:,None]*D)+.1*np.eye(64);HR=R.T@(omega[:,None]*R)+.1*np.eye(128)
 pid=pd['patient_ids'].astype(str);lab=pd['labels'].astype(int);people=sorted(set(pid))
 groups=[];gp=[];gc=[]
 for p in people:
  for cl in (0,1):
   ix=np.flatnonzero((pid==p)&(lab==cl))
   if len(ix):groups.append(ix);gp.append(p);gc.append(cl)
 gp=np.array(gp);gc=np.array(gc);pc=Counter(gp)
 U=np.array([1/len(people)/pc[p] for p in gp]);B=np.array([.5/np.sum(gc==c) for c in gc])
 ui=np.zeros(len(D));bi=np.zeros(len(D))
 for j,ix in enumerate(groups):ui[ix]=U[j]/len(ix);bi[ix]=B[j]/len(ix)
 check(np.array_equal(omega,bi),'Class-balanced image weighting exactly reproduced')
 X=np.stack([D[ix].mean(0) for ix in groups]);Y=np.stack([R[ix].mean(0) for ix in groups])
 muP=io.nz('public_target')['target'].reshape(2,64).astype(float);mid=muP.mean(0)
 wDP=np.linalg.solve(HD,.5*(muP[1]-muP[0]));signed=2*lab-1
 wRP=np.linalg.solve(HR,R.T@(omega*signed))
 opold=io.nz('old_public_operators')
 check(np.array_equal(wDP,opold['wDP']) and np.array_equal(wRP,opold['wRP']),'Public task anchors unchanged')
 check(float(opold['lambda_C'])==C['design']['fixed']['ACKD_map_lambda'],'Fixed KD lambda')
 with np.load(io.path('prior_target_weights'),allow_pickle=False) as f:
  heads={k:f[k] for k in ['DP1_DINO','DP2_DINO']}
 oldseal=io.js('prior_label_seal')
 check(C['inputs']['prior_target_weights']['sha256']==oldseal['files']['target_weights.npz'],'Source provenance')
 expected=io.nz('bridge_targets');designseal=io.js('design_completion')
 check(designseal['artifact_hashes'][C['inputs']['bridge_targets']['path']]==C['inputs']['bridge_targets']['sha256'],'Design target provenance')
 oldHR=io.nz('old_HR_targets');oldACKD=io.nz('old_ACKD_targets')
 maps={};mapcov={};maxfit=0.
 for name,ww in [('U',U),('B',B)]:
  maps[name]=[];mapcov[name]=[]
  for c,cap in enumerate(CAPS):
   x=clipped(X,cap);lam=C['design']['fixed']['HR_map_lambda'][c]
   t,b,_,ee=affine(x,Y,ww,lam);maxfit=max(maxfit,ee)
   maps[name].append((t,b))
   mx=ww@x;xc=x-mx
   mapcov[name].append((x,mx,xc,xc.T@(ww[:,None]*xc)+lam*np.eye(64)))
 GU=D.T@(ui[:,None]*D);crossU=R.T@(ui[:,None]*D);lam=C['design']['fixed']['ACKD_map_lambda']
 ACU=np.linalg.solve(HR,crossU@np.linalg.solve(GU+lam*np.eye(64),HD))
 targets={};labels={};fits={};files=[]
 checks={'independent_affine_fit':maxfit,'existing_targets_max_abs':0.,'design_targets_max_abs':0.,'independent_target_ridge_max_abs':0.}
 (out/'label_packets').mkdir(exist_ok=False)
 for draw in ['DP1','DP2']:
  wd=heads[draw+'_DINO'];delta=HD@wd;mus=[mid-delta,mid+delta]
  wrhr={}
  for name,ww in [('U',U),('B',B)]:
   means=[mus[c]@maps[name][c][0]+maps[name][c][1] for c in (0,1)]
   wrhr[name]=np.linalg.solve(HR,.5*(means[1]-means[0]))
   check(np.max(abs(wrhr[name]-expected['HR_'+name+'_'+draw]))<1e-10,'Frozen HR design target')
  oldack=wRP+opold['AC']@(wd-wDP)
  checks['existing_targets_max_abs']=max(checks['existing_targets_max_abs'],
   float(abs(wrhr['U']-oldHR[draw+'_HR_RN_ONLY']).max()),
   float(abs(oldack-oldACKD[draw+'_ACKD_RN_ONLY']).max()))
  acku=wRP+ACU@(wd-wDP)
  check(np.max(abs(acku-expected['ACKD_U_'+draw]))<1e-10,'Frozen ACKD-U design target')
  # Independent signed-public-response derivation of HR-B's transported mean.
  alpha=[]
  for c in (0,1):
   _,mx,xc,H=mapcov['B'][c]
   alpha.append(B+B*(xc@np.linalg.solve(H,mus[c]-mx)))
  v=np.zeros(len(D))
  for j,ix in enumerate(groups):v[ix]=.5*(alpha[1][j]-alpha[0][j])/len(ix)
  qhr=v/omega
  qack=signed+(ui/omega)*(D@np.linalg.solve(GU+lam*np.eye(64),HD@(wd-wDP)))
  for recipe,wr,q in [('HR_B_RN_ONLY',wrhr['B'],qhr),('ACKD_U_RN_ONLY',acku,qack)]:
   io.check_time()
   a=np.vstack([np.sqrt(omega[:,None])*R,np.sqrt(.1)*np.eye(128)])
   response=np.r_[np.sqrt(omega)*q,np.zeros(128)]
   independent=np.linalg.lstsq(a,response,rcond=None)[0]
   err=float(abs(independent-wr).max());check(err<1e-10,'Independent target ridge')
   checks['independent_target_ridge_max_abs']=max(checks['independent_target_ridge_max_abs'],err)
   targetkey=('HR_B_' if recipe=='HR_B_RN_ONLY' else 'ACKD_U_')+draw
   checks['design_targets_max_abs']=max(checks['design_targets_max_abs'],float(abs(wr-expected[targetkey]).max()))
   key=draw+'_'+recipe;targets[key]=wr
   l,detail=solve({'ResNet18':R},{'ResNet18':sel['ResNet18'].astype(float)},omega,{'ResNet18':wr},['ResNet18'])
   labels[key]=l;fits[key]=detail
   filename='label_packets/'+key+'.csv';export_packet(rows,l,out/filename);files.append(filename)
 check(max(checks.values())<1e-10 and len(labels)==4,'Construction verified')
 np.savez_compressed(out/'soft_labels.npz',**labels)
 np.savez_compressed(out/'crossover_target_weights.npz',**targets)
 io.verify();dump(out/'construction_inputs.json',io.receipts)
 result={'item':'2-2R2','status':'LABELS_READY','fits':fits,'verification':checks,'new_labels':4,
  'V_access':False,'new_Q_access':0,'new_DP_releases':0,'seconds':time.monotonic()-tick}
 dump(out/'construction_results.json',result)
 write_seal(out,'label_seal.json','FOUR_WEIGHT_CROSSOVER_LABELS_SEALED_BEFORE_EVALUATION',
  ['soft_labels.npz','crossover_target_weights.npz','construction_results.json','construction_inputs.json']+files)
 print(json.dumps({'item':'2-2R2','phase':'CONSTRUCTED','new_labels':4,'seconds':result['seconds'],'checks':checks,
  'max_KKT':max(x['independent_projected_gradient_max_abs'] for x in fits.values())}),flush=True)

def evaluate(out):
 tick=time.monotonic();io=Inputs(out,'evaluate')
 check(not (out/'results.json').exists(),'No repeated evaluation')
 check_seal(out,'label_seal.json','FOUR_WEIGHT_CROSSOVER_LABELS_SEALED_BEFORE_EVALUATION')
 labels=packet(out/'soft_labels.npz');sel=io.nz('old_selection')
 old=io.js('comparison_results');op=io.nz('comparison_predictions');ob=io.nz('comparison_bootstrap');seal=io.js('comparison_completion')
 for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz')]:
  check(C['inputs'][k]['sha256']==seal['files'][n],'Previous result seal')
 wanted=[draw+'_'+recipe+'_'+rec for draw in ['DP1','DP2'] for recipe in OLD for rec in ['DenseNet','ResNet18']]
 check(all(k in old['metrics'] for k in wanted),'Existing12 available')
 pn={str(n):i for i,n in enumerate(op['names'])};bn={str(n):i for i,n in enumerate(ob['names'])}
 metrics={k:old['metrics'][k] for k in wanted}
 scores={k:op['scores'][:,pn[k]] for k in wanted};boots={k:ob['metrics'][:,bn[k],:] for k in wanted}
 sys.path.insert(0,C['code_root'])
 from prrd_pilot_20260921.common import bootstrap_metrics
 from sklearn.metrics import roc_auc_score,average_precision_score
 sched=io.nz('patient_bootstrap');counts=sched['patient_counts'];ids0=sched['patient_ids'];h=C['inputs']['patient_bootstrap']['sha256']
 check(str(ob['source_draws_sha256'].item())==h and counts.shape==(2000,2026),'Same patient bootstrap')
 weights={};checks={'ridge_primal_dual_max_abs':0.,'bootstrap_sklearn_max_abs':0.};ref=None
 for rec,key in [('DenseNet','P_DN'),('ResNet18','P_RN')]:
  p=io.nz(key);v=io.nz('V_'+rec);same_order(v,op)
  check(set(p['roles'])=={'P'},'P role')
  for k in ['image_ids','patient_ids','labels']:
   check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier row order')
  if ref is None:ref=v
  else:same_order(ref,v)
  z=p['z'][sel['indices']].astype(float);zv=v['z'].astype(float);pid=v['patient_ids'].astype(np.int64);y=v['labels']
  unique,inv=np.unique(pid,return_inverse=True);check(np.array_equal(unique,ids0),'Patient ids')
  for draw in ['DP1','DP2']:
   for recipe in NEW:
    io.check_time();name=draw+'_'+recipe+'_'+rec;l=labels[draw+'_'+recipe]
    w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128)
    dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@w
    er=float(abs(s-zv@dual).max());check(er<1e-10,'Independent ridge')
    checks['ridge_primal_dual_max_abs']=max(checks['ridge_primal_dual_max_abs'],er)
    bb=bootstrap_metrics(y,s,pid,counts,ids0);points=[roc_auc_score(y,s),average_precision_score(y,s)]
    for j in [0,17,1999]:
     sw=counts[j,inv];witness=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
     er=float(abs(bb[j]-witness).max());check(er<1e-12,'Independent bootstrap')
     checks['bootstrap_sklearn_max_abs']=max(checks['bootstrap_sklearn_max_abs'],er)
    scores[name]=s;boots[name]=bb;weights[name]=w
    metrics[name]={m:{'point':float(points[k]),'patient_cluster_95':np.quantile(bb[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
  print(json.dumps({'item':'2-2R2','phase':'EVALUATED','receiver':rec}),flush=True)
 contrasts={};means={};differences={}
 for rec in ['DenseNet','ResNet18']:
  for stem,terms in TERMS.items():
   pts=[];ds=[];harm=False
   for draw in ['DP1','DP2']:
    names={draw+'_'+recipe+'_'+rec:value for recipe,value in terms.items()}
    d=sum(value*boots[name] for name,value in names.items())
    pt=sum(value*np.array([metrics[name][m]['point'] for m in ['AUROC','AP']]) for name,value in names.items())
    key=draw+'_'+stem+'_'+rec;pts.append(pt);ds.append(d);differences[key]=d
    contrasts[key]={m:{'delta':float(pt[k]),'patient_cluster_95':np.quantile(d[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
    harm|=contrasts[key]['AP']['patient_cluster_95'][1]<0
   mp=np.mean(pts,axis=0);md=np.mean(ds,axis=0);key='MEAN_'+stem+'_'+rec;differences[key]=md
   value={m:{'mean_delta':float(mp[k]),'fixed_two_release_patient_cluster_95':np.quantile(md[:,k],[.025,.975]).tolist()} for k,m in enumerate(['AUROC','AP'])}
   if len(terms)==2:
    cr={'both_AUROC_positive':bool(all(x[0]>0 for x in pts)),'mean_AUROC_at_least_0_01':bool(mp[0]>=.01),
     'mean_AUROC_CI_lower_positive':bool(value['AUROC']['fixed_two_release_patient_cluster_95'][0]>0),
     'mean_AP_nonnegative':bool(mp[1]>=0),'no_release_significant_AP_harm':not bool(harm)}
    value['development_criteria']=cr;value['passes_development_hurdle']=all(cr.values())
   means[key]=value
 primary={stem:means['MEAN_'+stem+'_DenseNet']['passes_development_hurdle'] for stem in ['HR_U_minus_HR_B','ACKD_U_minus_ACKD_B']}
 decisions={'shared_weight_separation_hypothesis_passes':all(primary.values()),'primary_hurdles':primary,
  'HR_at_matched_U_passes_hurdle':means['MEAN_HR_U_minus_ACKD_U_DenseNet']['passes_development_hurdle'],
  'new_variants_vs_source':{stem:means['MEAN_'+stem+'_DenseNet']['passes_development_hurdle'] for stem in ['HR_B_minus_DINO','ACKD_U_minus_DINO']},
  'original_2_3_gate_changed':False,'original_2_3_remains_deferred':True,'novelty_established':False}
 check(len(scores)==20 and len(weights)==8,'Exact planned scale')
 np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),
  labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
 np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(h))
 np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(differences)),differences=np.stack(list(differences.values()),axis=1))
 np.savez_compressed(out/'new_readout_weights.npz',**weights)
 io.verify();dump(out/'evaluation_inputs.json',io.receipts)
 result={'item':'2-2R2','status':'COMPLETE','created':now(),'metrics':metrics,'per_release_contrasts':contrasts,
  'fixed_two_release_mean_contrasts':means,'contrast_definitions':TERMS,'decisions':decisions,'verification':checks,
  'new_labels':4,'new_readouts':8,'reused_readouts':12,'new_Q_access':0,'new_DP_releases':0,
  'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,
  'uncertainty':'Image metrics with fixed-bank/fixed-release patient-cluster bootstrap; adaptive development; ResNet18 is a construction-model reference.',
  'seconds':time.monotonic()-tick,'contract_sha256':sha(out/'contract.json')}
 dump(out/'results.json',result)
 write_seal(out,'completion.json','2-2R2_COMPLETE',['contract.json','label_seal.json','results.json','evaluation_inputs.json',
  'predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','new_readout_weights.npz'])
 print(json.dumps({'item':'2-2R2','status':'COMPLETE','DenseNet':{k:v for k,v in metrics.items() if k.endswith('DenseNet')},
  'DenseNet_means':{k:v for k,v in means.items() if k.endswith('DenseNet')},'decisions':decisions,'checks':checks,'seconds':result['seconds']},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['construct','evaluate']);a=p.parse_args()
 (construct if a.phase=='construct' else evaluate)(OUT)
