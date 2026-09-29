"""Four new public-carrier KD labels; immutable MT labels and scores reused."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
sys.path.insert(0,C['step3_dir'])
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet
from construct_labels import solve

def construct(out):
    tick=time.monotonic();io=Inputs(out,'construct')
    check(not (out/'label_seal.json').exists(),'No silent label rerun')
    sel=io.nz('old_selection');manifest=io.js('old_manifest')
    oldseal=io.js('old_label_seal');oldlabels=io.nz('old_labels');oldfits=io.js('old_fits')
    for key,name in [('old_selection','public128_selection.npz'),('old_manifest','public128_manifest.json'),
                     ('old_labels','soft_labels.npz'),('old_fits','construction_results.json')]:
        check(io.contract['inputs'][key]['sha256']==oldseal['files'][name],'Step3 label provenance '+name)
    ps=io.nz('P_DINO');pr=io.nz('P_RN');same_order(ps,pr)
    check(set(ps['roles'])==set(pr['roles'])=={'P'},'Construction public P only')
    xp={'DINO':ps['z'].transpose(1,0,2).reshape(813,64).astype(float),'ResNet18':pr['z'].astype(float)}
    ix=sel['indices'];check(len(ix)==128 and len(np.unique(sel['patient_ids']))==128,'Fixed128 distinct patients')
    for k in ('image_ids','patient_ids','labels'):
        check(np.array_equal(ps[k][ix].astype(str),sel[k].astype(str)),'Selected row metadata '+k)
    xs={m:sel[m].astype(float) for m in xp}
    for m in xp:check(np.array_equal(xp[m][ix],xs[m]),'Cached public selection changed')
    sys.path.insert(0,C['historical_code_dir'])
    from fixed_image_compilation import pw
    w=pw(ps);check(np.array_equal(w,pw(pr)) and abs(w.sum()-1)<1e-12,'Class/patient weights')
    tw=io.nz('prior_target_weights');ts=io.js('prior_label_seal')
    check(io.contract['inputs']['prior_target_weights']['sha256']==ts['files']['target_weights.npz'],'Step1 target seal')
    rows=manifest['rows'];check(len(rows)==128,'Image packet rows')
    for i,r in enumerate(rows):check(str(r['image_id'])==str(sel['image_ids'][i]),'Manifest row order')
    (out/'label_packets').mkdir(exist_ok=False)
    all_labels={};fits={};aliases={};newfiles=[];kd_error=0.
    for draw in ('DP1','DP2'):
        wd=tw[draw+'_DINO'];kd=tw[draw+'_KD_RN'];xr=xp['ResNet18'];q=xp['DINO']@wd
        # Different factorization from the original normal-equation target builder.
        design=np.vstack([np.sqrt(w[:,None])*xr,np.sqrt(.1)*np.eye(xr.shape[1])])
        response=np.concatenate([np.sqrt(w)*q,np.zeros(xr.shape[1])])
        independent=np.linalg.lstsq(design,response,rcond=None)[0]
        kd_error=max(kd_error,float(abs(independent-kd).max()));check(kd_error<1e-10,'KD target mismatch')
        targets={'DINO':wd,'ResNet18':kd}
        for recipe,models in [('RN_ONLY',['ResNet18']),('JOINT',['DINO','ResNet18'])]:
            io.check_time();oldname=draw+'_PUBLIC_'+recipe;mtname=draw+'_MT_'+recipe;kdname=draw+'_KD_'+recipe
            mt=oldlabels[oldname].copy();detail=oldfits['fits'][oldname]
            check(detail['anchor']=='zero' and detail['models']==models,'MT label objective mismatch')
            check(mt.shape==(128,) and np.isfinite(mt).all() and abs(mt).max()<=1,'MT label validity')
            all_labels[mtname]=mt;fits[mtname]=dict(detail,reused=True)
            aliases[mtname]={'source':str(Path(C['step3_dir'])/'soft_labels.npz'),'key':oldname,
                            'source_sha256':io.contract['inputs']['old_labels']['sha256']}
            l,d=solve(xp,xs,w,targets,models);all_labels[kdname]=l;fits[kdname]=dict(d,reused=False)
            name='label_packets/'+kdname+'.csv';export_packet(rows,l,out/name);newfiles.append(name)
    check(len(all_labels)==8 and len(newfiles)==4,'Expected four new / four reused labels')
    np.savez_compressed(out/'soft_labels.npz',**all_labels)
    dump(out/'label_aliases.json',aliases)
    io.verify();dump(out/'construction_inputs.json',io.receipts)
    result={'status':'FOUR_KD_LABELS_READY','fits':fits,'new_label_solves':4,'reused_MT_labels':4,
            'KD_independent_lstsq_max_abs':kd_error,'V_access':False,'DenseNet_access':False,
            'new_Q_access':0,'new_DP_releases':0,'seconds':time.monotonic()-tick}
    dump(out/'construction_results.json',result)
    write_seal(out,'label_seal.json','ALL_LABELS_SEALED_BEFORE_NEW_KD_EVALUATION',
        ['soft_labels.npz','label_aliases.json','construction_results.json','construction_inputs.json']+newfiles,
        reused_selection_sha256=io.contract['inputs']['old_selection']['sha256'])
    print(json.dumps({'status':result['status'],'KD_target_error':kd_error,'seconds':result['seconds'],
                     'max_new_KKT':max(d['independent_projected_gradient_max_abs'] for d in fits.values() if not d['reused'])}),flush=True)

def evaluate(out):
    tick=time.monotonic();io=Inputs(out,'evaluate')
    check(not (out/'results.json').exists(),'No silent reevaluation')
    check_seal(out,'label_seal.json','ALL_LABELS_SEALED_BEFORE_NEW_KD_EVALUATION')
    labs=packet(out/'soft_labels.npz');sel=io.nz('old_selection')
    old=io.js('old_results');pred=io.nz('old_predictions');boot=io.nz('old_bootstrap');seal=io.js('old_completion')
    for k,n in [('old_results','results.json'),('old_predictions','predictions_private.npz'),('old_bootstrap','paired_bootstrap.npz')]:
        check(io.contract['inputs'][k]['sha256']==seal['files'][n],'Old evaluation seal '+n)
    sys.path.insert(0,C['code_root'])
    from prrd_pilot_20260921.common import bootstrap_metrics
    from sklearn.metrics import roc_auc_score,average_precision_score
    schedule=io.nz('patient_bootstrap');counts=schedule['patient_counts'];bootids=schedule['patient_ids']
    drawsha=io.contract['inputs']['patient_bootstrap']['sha256']
    check(str(boot['source_draws_sha256'].item())==drawsha and counts.shape==(2000,2026),'Same patient resampling')
    oldp={str(k):i for i,k in enumerate(pred['names'])};oldb={str(k):i for i,k in enumerate(boot['names'])}
    metrics={};scores={};boots={};weights={};ref=None
    verification={'new_ridge_primal_dual_max_abs':0.,'bootstrap_sklearn_max_abs':0.,'reused_MT_prediction_max_abs':0.,'reused_MT_metric_max_abs':0.}
    for receiver,pk in [('DenseNet','P_DN'),('ResNet18','P_RN')]:
        p=io.nz(pk);v=io.nz('V_'+receiver);same_order(v,pred)
        check(set(p['roles'])=={'P'},'Public cached feature role')
        if ref is None:ref=v
        else:same_order(ref,v)
        ids=v['patient_ids'].astype(np.int64);y=v['labels'];zv=v['z'].astype(float)
        check(np.array_equal(np.unique(ids),bootids),'Evaluation patients differ')
        for k in ('image_ids','patient_ids','labels'):
            check(np.array_equal(p[k][sel['indices']].astype(str),sel[k].astype(str)),'Public evaluation row order')
        z=p['z'][sel['indices']].astype(float)
        if receiver=='ResNet18':check(np.array_equal(z,sel['ResNet18']),'RN source/cache mismatch')
        _,inverse=np.unique(ids,return_inverse=True)
        for draw in ('DP1','DP2'):
            for recipe in ('RN_ONLY','JOINT'):
                for method in ('MT','KD'):
                    io.check_time();cell=draw+'_'+method+'_'+recipe;name=cell+'_'+receiver;l=labs[cell]
                    beta=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128)
                    if method=='MT':
                        oldname=draw+'_PUBLIC_'+recipe+'_'+receiver
                        s=pred['scores'][:,oldp[oldname]];b=boot['metrics'][:,oldb[oldname],:]
                        err=float(abs(zv@beta-s).max());verification['reused_MT_prediction_max_abs']=max(verification['reused_MT_prediction_max_abs'],err)
                        check(err<1e-12,'Old MT label/feature path not reproduced')
                        metrics[name]=old['metrics'][oldname]
                        point=np.array([roc_auc_score(y,s),average_precision_score(y,s)])
                        oldpoint=np.array([metrics[name][m]['point'] for m in ('AUROC','AP')])
                        me=float(abs(point-oldpoint).max());verification['reused_MT_metric_max_abs']=max(verification['reused_MT_metric_max_abs'],me)
                        check(me<1e-12,'Old MT point metric not reproduced')
                    else:
                        dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l);s=zv@beta
                        err=float(abs(s-zv@dual).max());verification['new_ridge_primal_dual_max_abs']=max(verification['new_ridge_primal_dual_max_abs'],err)
                        check(err<1e-10,'Independent ridge mismatch')
                        b=bootstrap_metrics(y,s,ids,counts,bootids)
                        point=[roc_auc_score(y,s),average_precision_score(y,s)]
                        metrics[name]={m:{'point':float(point[j]),'patient_cluster_95':np.quantile(b[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
                        for ix in (0,17,1999):
                            sw=counts[ix,inverse]
                            independent=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
                            e=float(abs(independent-b[ix]).max());verification['bootstrap_sklearn_max_abs']=max(verification['bootstrap_sklearn_max_abs'],e)
                            check(e<1e-12,'Patient bootstrap mismatch')
                    scores[name]=s;boots[name]=b;weights[name]=beta
        print(json.dumps({'phase':'EVALUATED','receiver':receiver}),flush=True)
    contrast={};means={};diffs={};decisions={}
    for receiver in ('DenseNet','ResNet18'):
        for recipe in ('RN_ONLY','JOINT'):
            points=[];bds=[];ap_harm=False
            for draw in ('DP1','DP2'):
                a=draw+'_MT_'+recipe+'_'+receiver;b=draw+'_KD_'+recipe+'_'+receiver
                pd=np.array([metrics[a][m]['point']-metrics[b][m]['point'] for m in ('AUROC','AP')]);bd=boots[a]-boots[b]
                name=draw+'_'+recipe+'_MT_minus_KD_'+receiver;diffs[name]=bd;points.append(pd);bds.append(bd)
                contrast[name]={m:{'delta':float(pd[j]),'patient_cluster_95':np.quantile(bd[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
                ap_harm |= contrast[name]['AP']['patient_cluster_95'][1]<0
            pd=np.mean(points,axis=0);bd=np.mean(bds,axis=0);name='MEAN_'+recipe+'_MT_minus_KD_'+receiver;diffs[name]=bd
            rec={m:{'mean_delta':float(pd[j]),'fixed_two_release_patient_cluster_95':np.quantile(bd[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
            criteria={'both_AUROC_positive':bool(all(p[0]>0 for p in points)), 'mean_AUROC_at_least_0_01':bool(pd[0]>=.01),
                      'mean_AP_nonnegative':bool(pd[1]>=0),'mean_AUROC_CI_lower_positive':bool(rec['AUROC']['fixed_two_release_patient_cluster_95'][0]>0)}
            passed=all(criteria.values()) and not ap_harm
            rec.update(development_criteria=criteria,per_release_significant_AP_harm=bool(ap_harm),passes_development_hurdle=passed)
            means[name]=rec
            if receiver=='DenseNet':decisions[recipe]='POSITIVE_DEVELOPMENT_SUPPORT' if passed else 'MT_ADDED_VALUE_NOT_ESTABLISHED_BY_PRESET_HURDLE'
    np.savez_compressed(out/'predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),
                        labels=ref['labels'],patient_ids=ref['patient_ids'],image_ids=ref['image_ids'])
    np.savez_compressed(out/'paired_bootstrap.npz',names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),source_draws_sha256=np.array(drawsha))
    np.savez_compressed(out/'contrast_bootstrap.npz',names=np.array(list(diffs)),differences=np.stack(list(diffs.values()),axis=1))
    np.savez_compressed(out/'readout_weights.npz',**weights)
    io.verify();dump(out/'evaluation_inputs.json',io.receipts)
    result={'status':'PUBLIC128_KD_COMPARISON_COMPLETE','created':now(),'whole_project_stage':2,'metrics':metrics,
            'per_release_contrasts':contrast,'fixed_two_release_mean_contrasts':means,'development_decisions':decisions,
            'verification':verification,'new_label_solves':4,'reused_MT_labels':4,'new_receiver_readouts':8,'reused_MT_readouts':8,
            'new_Q_access':0,'new_DP_releases':0,'new_model_forwards':0,'new_pixel_training':0,'Expert_Reserved':False,
            'metric_unit':'image','uncertainty_unit':'patient-cluster, fixed two releases','bootstrap_draws':2000,
            'primary_receiver':'DenseNet','construction_reference':'ResNet18','seconds':time.monotonic()-tick,
            'contract_sha256':sha(out/'contract.json'),'interpretation':'Defined KD comparator; adaptive development, not novelty or independent confirmation'}
    dump(out/'results.json',result)
    write_seal(out,'completion.json',result['status'],['contract.json','label_seal.json','results.json','evaluation_inputs.json',
        'predictions_private.npz','paired_bootstrap.npz','contrast_bootstrap.npz','readout_weights.npz'])
    print(json.dumps({'status':result['status'],'decisions':decisions,'means':means,'verification':verification,'seconds':result['seconds']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['construct','evaluate']);a=p.parse_args()
    (construct if a.phase=='construct' else evaluate)(OUT)
