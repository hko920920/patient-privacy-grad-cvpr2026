"""Numbered item1-5: two DINO-only labels and matched existing controls."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet
from construct_labels import solve

def construct(out):
    tick=time.monotonic();io=Inputs(out,'construct');check(not (out/'label_seal.json').exists(),'No label rerun')
    sel=io.nz('old_selection');manifest=io.js('old_manifest');p=io.nz('P_DINO')
    check(set(p['roles'])=={'P'},'Public P construction only')
    x=p['z'].transpose(1,0,2).reshape(813,64).astype(float);xs=sel['DINO'].astype(float)
    check(np.array_equal(x[sel['indices']],xs),'Frozen public128 features')
    for k in ('image_ids','patient_ids','labels'):
        check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Public128 row order')
    sys.path.insert(0,C['historical_code_dir'])
    from fixed_image_compilation import pw
    w=pw(p);check(abs(w.sum()-1)<1e-12 and np.all(w>0),'Patient/class weighting')
    tw=io.nz('prior_target_weights');seal=io.js('prior_label_seal')
    check(io.contract['inputs']['prior_target_weights']['sha256']==seal['files']['target_weights.npz'],'Target provenance')
    rows=manifest['rows'];check(len(rows)==128 and len(np.unique(sel['patient_ids']))==128,'Carrier size/patients')
    for i,r in enumerate(rows):check(str(r['image_id'])==str(sel['image_ids'][i]),'Manifest order')
    (out/'label_packets').mkdir(exist_ok=False);labels={};fits={};files=[]
    for draw in ('DP1','DP2'):
        io.check_time();name=draw+'_DINO_ONLY'
        l,d=solve({'DINO':x},{'DINO':xs},w,{'DINO':tw[draw+'_DINO']},['DINO'])
        labels[name]=l;fits[name]=d;n='label_packets/'+name+'.csv';export_packet(rows,l,out/n);files.append(n)
    np.savez_compressed(out/'soft_labels.npz',**labels)
    io.verify();dump(out/'construction_inputs.json',io.receipts)
    result={'status':'1-5_LABELS_READY','item':'1-5','fits':fits,'new_label_solves':2,'V_access':False,
            'new_Q_access':0,'new_DP_releases':0,'seconds':time.monotonic()-tick}
    dump(out/'construction_results.json',result)
    write_seal(out,'label_seal.json','TWO_DINO_LABELS_SEALED_BEFORE_EVALUATION',
               ['soft_labels.npz','construction_results.json','construction_inputs.json']+files)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],'max_KKT':max(x['independent_projected_gradient_max_abs'] for x in fits.values())}),flush=True)

def evaluate(out):
    tick=time.monotonic();io=Inputs(out,'evaluate');check(not (out/'results.json').exists(),'No silent reevaluation')
    check_seal(out,'label_seal.json','TWO_DINO_LABELS_SEALED_BEFORE_EVALUATION')
    lab=packet(out/'soft_labels.npz');sel=io.nz('old_selection')
    old=io.js('comparison_results');op=io.nz('comparison_predictions');ob=io.nz('comparison_bootstrap');seal=io.js('comparison_completion')
    for k,n in [('comparison_results','results.json'),('comparison_predictions','predictions_private.npz'),('comparison_bootstrap','paired_bootstrap.npz')]:
        check(io.contract['inputs'][k]['sha256']==seal['files'][n],'Comparison provenance')
    sys.path.insert(0,C['code_root'])
    from prrd_pilot_20260921.common import bootstrap_metrics
    from sklearn.metrics import roc_auc_score,average_precision_score
    schedule=io.nz('patient_bootstrap');counts=schedule['patient_counts'];bootids=schedule['patient_ids']
    h=io.contract['inputs']['patient_bootstrap']['sha256']
    check(str(ob['source_draws_sha256'].item())==h and counts.shape==(2000,2026),'Paired resampling source')
    metrics=dict(old['metrics']);scores={str(n):op['scores'][:,i] for i,n in enumerate(op['names'])}
    boots={str(n):ob['metrics'][:,i,:] for i,n in enumerate(ob['names'])};weights={};ref=None
    verification={'ridge_primal_dual_max_abs':0.,'bootstrap_sklearn_max_abs':0.}
    for receiver,key in [('DenseNet','P_DN'),('ResNet18','P_RN')]:
        p=io.nz(key);v=io.nz('V_'+receiver);same_order(v,op)
        check(set(p['roles'])=={'P'},'Carrier cache public only')
        for k in ('image_ids','patient_ids','labels'):
            check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Carrier evaluation rows')
        if ref is None:ref=v
        else:same_order(ref,v)
        z=p['z'][sel['indices']].astype(float);zv=v['z'].astype(float);ids=v['patient_ids'].astype(np.int64);y=v['labels']
        check(np.array_equal(np.unique(ids),bootids),'Patient alignment');_,inv=np.unique(ids,return_inverse=True)
        for draw in ('DP1','DP2'):
            io.check_time();name=draw+'_DINO_ONLY_'+receiver;l=lab[draw+'_DINO_ONLY']
            w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128)
            dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@w
            e=float(abs(s-zv@dual).max());verification['ridge_primal_dual_max_abs']=max(verification['ridge_primal_dual_max_abs'],e)
            check(e<1e-10,'Independent ridge')
            b=bootstrap_metrics(y,s,ids,counts,bootids);point=[roc_auc_score(y,s),average_precision_score(y,s)]
            for ix in (0,17,1999):
                sw=counts[ix,inv];ind=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
                e=float(abs(ind-b[ix]).max());verification['bootstrap_sklearn_max_abs']=max(verification['bootstrap_sklearn_max_abs'],e);check(e<1e-12,'Independent bootstrap')
            scores[name]=s;boots[name]=b;weights[name]=w
            metrics[name]={m:{'point':float(point[j]),'patient_cluster_95':np.quantile(b[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
        print(json.dumps({'item':'1-5','phase':'EVALUATED','receiver':receiver}),flush=True)
    contrasts={};means={};diffs={};decisions={}
    for receiver in ('DenseNet','ResNet18'):
        for method in ('MT_RN_ONLY','MT_JOINT','KD_RN_ONLY','KD_JOINT'):
            pts=[];dis=[];harm=False
            for draw in ('DP1','DP2'):
                a=draw+'_'+method+'_'+receiver;b=draw+'_DINO_ONLY_'+receiver
                pt=np.array([metrics[a][m]['point']-metrics[b][m]['point'] for m in ('AUROC','AP')]);dd=boots[a]-boots[b]
                key=draw+'_'+method+'_minus_DINO_ONLY_'+receiver;diffs[key]=dd;pts.append(pt);dis.append(dd)
                contrasts[key]={m:{'delta':float(pt[j]),'patient_cluster_95':np.quantile(dd[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
                harm |= contrasts[key]['AP']['patient_cluster_95'][1]<0
            pd=np.mean(pts,axis=0);bd=np.mean(dis,axis=0);key='MEAN_'+method+'_minus_DINO_ONLY_'+receiver;diffs[key]=bd
            rec={m:{'mean_delta':float(pd[j]),'fixed_two_release_patient_cluster_95':np.quantile(bd[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
            criteria={'both_AUROC_positive':bool(all(p[0]>0 for p in pts)),'mean_AUROC_at_least_0_01':bool(pd[0]>=.01),
                      'mean_AP_nonnegative':bool(pd[1]>=0),'mean_AUROC_CI_lower_positive':bool(rec['AUROC']['fixed_two_release_patient_cluster_95'][0]>0)}
            passed=all(criteria.values()) and not harm
            rec.update(development_criteria=criteria,per_release_significant_AP_harm=bool(harm),passes_development_hurdle=passed,
                       role='main' if method.startswith('MT') else 'reference')
            means[key]=rec
            if receiver=='DenseNet' and method.startswith('MT'):
                decisions[method]='POSITIVE_DEVELOPMENT_SUPPORT' if passed else 'ADDED_VALUE_OVER_SOURCE_ONLY_NOT_ESTABLISHED'
    np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),
                        labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
    np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(h))
    np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(diffs)),differences=np.stack(list(diffs.values()),axis=1))
    np.savez_compressed(out/'new_readout_weights.npz',**weights)
    io.verify();dump(out/'evaluation_inputs.json',io.receipts)
    result={'status':'1-5_COMPLETE','item':'1-5','created':now(),'metrics':metrics,'per_release_contrasts':contrasts,
            'fixed_two_release_mean_contrasts':means,'development_decisions':decisions,'verification':verification,
            'new_label_solves':2,'new_readouts':4,'reused_readouts':16,'new_Q_access':0,'new_DP_releases':0,
            'new_pixel_training':0,'new_encoder_forwards':0,'Expert_Reserved':False,'seconds':time.monotonic()-tick,
            'metric_unit':'image','uncertainty':'patient-cluster fixed-release conditional','contract_sha256':sha(out/'contract.json')}
    dump(out/'results.json',result)
    write_seal(out,'completion.json','1-5_COMPLETE',['contract.json','label_seal.json','results.json','evaluation_inputs.json',
        'predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','new_readout_weights.npz'])
    print(json.dumps({'status':'1-5_COMPLETE','decisions':decisions,'means':{k:v for k,v in means.items() if k.endswith('DenseNet')},
                     'source_points':{k:v for k,v in metrics.items() if 'DINO_ONLY_DenseNet' in k},'verification':verification,'seconds':result['seconds']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['construct','evaluate']);a=p.parse_args()
    (construct if a.phase=='construct' else evaluate)(OUT)
