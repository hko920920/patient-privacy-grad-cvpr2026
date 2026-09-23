"""One new P202 PNG evaluation and frozen two-seed P+Q minus P comparison."""
import argparse,csv,time,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from prrd_v3.contracts import CODE,OUT,read,save,sha,require,setup,now
from prrd_v3.datasets import manifests,PixelAccess
from prrd_v3.encoders import Encoder,state_hash,pil_tensor
from prrd_v3.bank_runtime import verify_artifact
from prrd_v3.guarded_objectives import bank_moments
from prrd_pilot_20260921.common import ROOT as PILOT,BOOT,packet,check_saved_order,bootstrap_metrics
from prrd_pilot_20260921.evaluate import learn
from .evaluate_feasibility import DIAG,dependencies as previous_dependencies

PUBLIC=CODE/'_reports/receiver_a2_public_s200_20260923_v1'
REPEAT=CODE/'_reports/receiver_repeat202_s200_20260923_v1'


def dependencies():
    result=previous_dependencies()
    paths=[Path(__file__),PUBLIC/'A2_P_result.json',REPEAT/'A2_result.json']
    for folder in (PUBLIC,REPEAT):
        paths += [folder/'evaluation'/n for n in ('result.json','predictions_private.npz','paired_bootstrap.npz')]
    for p in paths:result[str(p)]=sha(p)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;tick=time.monotonic();binding=read(root/'execution_bindings.json')
    require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation dependency changed')
    contract=read(root/'campaign_contract.json');seal=read(root/'bank_seal.json')
    bank=root/'banks'/contract['bank_id'];done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
    require(done['completed_updates']==200 and spec['population']=='P' and spec['seed']==202,'Wrong final bank')
    require(sha(bank/'COMPLETED.json')==seal['completion_sha256'],'Completion changed')
    require(sha(bank/'artifact/artifact_seal.json')==seal['artifact_sha256'],'PNG changed')
    verify_artifact(bank/'artifact',spec,done['signature'])
    output=root/'evaluation';output.mkdir(exist_ok=False)
    setup();groups=manifests();access=PixelAccess(groups,()).install()
    weights,details,counts,times={},{},{},{}
    with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==128 and [int(r['label']) for r in rows]==[0]*64+[1]*64,'PNG labels/order')
    for kind,title in (('source','BioViL'),('recipient','DenseNet')):
        started=time.monotonic();encoder=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(encoder);pieces=[]
        for i in range(0,128,16):
            images=[]
            for row in rows[i:i+16]:
                with Image.open(bank/'artifact'/row['file']) as im:images.append(pil_tensor(im.convert('L')))
            with torch.no_grad():z=encoder(torch.stack(images).cuda())
            pieces.append(z.cpu())
        z=torch.cat(pieces);name=title+'_A2_P_202'
        weights[name],details[name]=learn(bank_moments({'z':z,'a':z},'B',{}),0.)
        np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
        counts[name]=128;require(state_hash(encoder)==before,'Evaluation encoder changed')
        times[title]=time.monotonic()-started;del encoder;torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz');vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr):check_saved_order(v,groups['V'])
    for key in ('labels','image_ids','patient_ids'):require(np.array_equal(vr[key],vs[key]),'V order changed')
    scores,boots,expected_metrics={},{},{}
    for folder,seed in ((PUBLIC,101),(REPEAT,202)):
        previous=packet(folder/'evaluation/predictions_private.npz')
        old_boot=packet(folder/'evaluation/paired_bootstrap.npz');old_result=read(folder/'evaluation/result.json')
        for key in ('labels','image_ids','patient_ids'):require(np.array_equal(previous[key],vs[key]),'Historical V order')
        require(np.array_equal(previous['names'],old_boot['names']),'Cached bootstrap names')
        require(str(old_boot['original_patient_draws_sha256'])==sha(BOOT),'Draw hash changed')
        for title in ('BioViL','DenseNet'):
            for population in (('P','PQ') if seed==101 else ('PQ',)):
                old=title+'_A2_'+population if seed==101 else title+'_A2_202'
                name=title+'_A2_'+population+'_'+str(seed)
                index=list(previous['names']).index(old)
                scores[name]=previous['scores'][:,index];boots[name]=old_boot['metrics'][:,index]
                expected_metrics[name]=old_result['metrics'][old]
    for n,w in weights.items():scores[n]=np.asarray(vs['z'] if n.startswith('BioViL') else vr['z'],float)@w
    y=vs['labels'];p=packet(BOOT);pids=p['patient_ids'];draws=p['patient_counts'];integer=vs['patient_ids'].astype(np.int64)
    require(draws.shape==(2000,2026) and np.array_equal(np.unique(integer),pids),'Draw ordering')
    for n in weights:boots[n]=bootstrap_metrics(y,scores[n],integer,draws,pids)
    _,inverse=np.unique(integer,return_inverse=True);error=0.
    for k in (0,17,1999):
        sw=draws[k,inverse]
        for n in weights:
            expected=np.array([roc_auc_score(y,scores[n],sample_weight=sw),average_precision_score(y,scores[n],sample_weight=sw)])
            error=max(error,float(np.max(abs(expected-boots[n][k]))))
    require(error<=1e-12,'Bootstrap verification')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    for n in scores:
        if n in expected_metrics:
            require(all(metrics[n][m]==expected_metrics[n][m] for m in ('AUROC','AP')),'Historical metric changed')
        for k,m in enumerate(('AUROC','AP')):metrics[n][m+'_patient_cluster_95']=np.quantile(boots[n][:,k],[.025,.975]).tolist()
    contrasts={}
    for title in ('BioViL','DenseNet'):
        for seed in (101,202):
            a=f'{title}_A2_PQ_{seed}';b=f'{title}_A2_P_{seed}'
            contrasts[f'{title}_PQ_minus_P_{seed}']={m:{'delta':metrics[a][m]-metrics[b][m],
                'patient_cluster_95':np.quantile((boots[a]-boots[b])[:,k],[.025,.975]).tolist()}
                for k,m in enumerate(('AUROC','AP'))}
    summary={title:{m:float(np.mean([contrasts[f'{title}_PQ_minus_P_{seed}'][m]['delta'] for seed in (101,202)]))
                    for m in ('AUROC','AP')} for title in ('BioViL','DenseNet')}
    names=list(scores);matrix=np.column_stack(list(scores.values()));require(np.isfinite(matrix).all(),'Invalid scores')
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=matrix,labels=y,
        image_ids=vs['image_ids'],patient_ids=vs['patient_ids'],**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
        metrics=np.stack([boots[n] for n in names],axis=1),original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_MATCHED_A2_PQ_VS_P_TWO_SEEDS','finished':now(),'metrics':metrics,'comparisons':contrasts,
        'mean_Q_addition_effect':summary,'mean_scope':'arithmetic mean of two bank-specific metric differences; no synthesis-population CI',
        'seeds':[101,202],'new_banks':1,'new_bank_seed':202,'updates':200,'images':128,
        'interval_scope':'2000 paired V-patient draws per seed conditional on fixed banks; no synthesis-seed uncertainty',
        'receiver':'DenseNet adaptive development; BioViL is an A2 source','bootstrap_sklearn_error':error,
        'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
        'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
        'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
        'synthesis_seconds':{'A2_P_101':read(PUBLIC/'A2_P_result.json')['cumulative_worker_seconds'],
                             'A2_PQ_101':read(PUBLIC/'evaluation/result.json')['synthesis_seconds']['A2_PQ'],
                             'A2_P_202':read(root/'A2_P_result.json')['cumulative_worker_seconds'],
                             'A2_PQ_202':read(REPEAT/'A2_result.json')['cumulative_worker_seconds']},
        'evaluation_seconds':time.monotonic()-tick,'automatic_followup':False,
        'DP_Expert_Reserved_final_receiver':False,'predictions_sha256':sha(output/'predictions_private.npz')}
    save(output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

