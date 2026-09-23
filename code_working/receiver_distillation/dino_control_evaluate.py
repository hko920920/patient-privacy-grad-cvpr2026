"""Evaluate one sealed DINO-only control against cached A1/A2 predictions."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from prrd_v3.contracts import CODE, OUT, read, save, sha, require, setup, now
from prrd_v3.datasets import manifests, PixelAccess
from prrd_v3.encoders import Encoder, state_hash, pil_tensor
from prrd_v3.bank_runtime import verify_artifact
from prrd_v3.guarded_objectives import bank_moments
from prrd_pilot_20260921.common import ROOT as PILOT, BOOT, packet, check_saved_order, bootstrap_metrics
from prrd_pilot_20260921.evaluate import learn
from .evaluate_feasibility import DIAG, dependencies as old_dependencies

PREVIOUS=CODE/'_reports/receiver_a1a2_s200_20260922_v1'


def dependencies():
    result=old_dependencies()
    for p in (Path(__file__),PREVIOUS/'evaluation/result.json',PREVIOUS/'evaluation/predictions_private.npz',
              PREVIOUS/'evaluation/paired_bootstrap.npz',PREVIOUS/'two_bank_seal.json',
              PREVIOUS/'A1_result.json',PREVIOUS/'A2_result.json'):
        result[str(p)]=sha(p)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    root=args.root; started=time.monotonic()
    binding=read(root/'execution_bindings.json')
    require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation dependencies changed')
    seal=read(root/'bank_seal.json'); contract=read(root/'campaign_contract.json')
    bank=root/'banks'/contract['bank_id']; done=read(bank/'COMPLETED.json'); spec=read(bank/'run_spec.json')
    require(done['completed_updates']==200 and spec['updates']==200,'Final200 only')
    require(sha(bank/'COMPLETED.json')==seal['completion_sha256'],'Completion changed')
    require(sha(bank/'artifact/artifact_seal.json')==seal['artifact_sha256'],'Artifact changed')
    verify_artifact(bank/'artifact',spec,done['signature'])
    output=root/'evaluation'; output.mkdir(exist_ok=False)
    setup(); groups=manifests(); access=PixelAccess(groups,()).install()
    weights,details,counts,times={},{},{},{}
    with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==128 and [int(r['label']) for r in rows]==[0]*64+[1]*64,'PNG labels/order')
    for kind,title in (('source','BioViL'),('recipient','DenseNet')):
        tick=time.monotonic()
        encoder=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(encoder); pieces=[]
        for i in range(0,128,16):
            images=[]
            for row in rows[i:i+16]:
                with Image.open(bank/'artifact'/row['file']) as im: images.append(pil_tensor(im.convert('L')))
            with torch.no_grad(): z=encoder(torch.stack(images).cuda())
            pieces.append(z.cpu())
        z=torch.cat(pieces); moments=bank_moments({'z':z,'a':z},'B',{})
        name=title+'_DINO'
        weights[name],details[name]=learn(moments,0.)
        np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
        counts[name]=128
        require(state_hash(encoder)==before,'Evaluation encoder changed')
        times[title]=time.monotonic()-tick
        del encoder; torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz'); vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr):check_saved_order(v,groups['V'])
    previous=packet(PREVIOUS/'evaluation/predictions_private.npz')
    old_boot=packet(PREVIOUS/'evaluation/paired_bootstrap.npz')
    old_result=read(PREVIOUS/'evaluation/result.json')
    for v in (vr,previous):
        for key in ('labels','image_ids','patient_ids'):
            require(np.array_equal(v[key],vs[key]),'V order changed')
    require(np.array_equal(previous['names'],old_boot['names']),'Cached bootstrap order')
    require(str(old_boot['original_patient_draws_sha256'])==sha(BOOT),'Cached draws changed')
    scores={str(n):previous['scores'][:,j] for j,n in enumerate(previous['names'])}
    boots={str(n):old_boot['metrics'][:,j] for j,n in enumerate(old_boot['names'])}
    for n,w in weights.items():
        scores[n]=np.asarray(vs['z'] if n.startswith('BioViL') else vr['z'],float)@w
    y=vs['labels']; integer=vs['patient_ids'].astype(np.int64)
    b=packet(BOOT); pids=b['patient_ids']; draws=b['patient_counts']
    require(draws.shape==(2000,2026) and np.array_equal(np.unique(integer),pids)
            and np.array_equal(integer.astype(str),vs['patient_ids']),'Bootstrap ordering')
    for n in weights:boots[n]=bootstrap_metrics(y,scores[n],integer,draws,pids)
    _,inverse=np.unique(integer,return_inverse=True); error=0.
    for k in (0,17,1999):
        sw=draws[k,inverse]
        for n in weights:
            expected=np.array([roc_auc_score(y,scores[n],sample_weight=sw),
                               average_precision_score(y,scores[n],sample_weight=sw)])
            error=max(error,float(np.max(abs(expected-boots[n][k]))))
    require(error<=1e-12,'Independent bootstrap mismatch')
    names=list(scores); matrix=np.column_stack(list(scores.values()))
    require(np.isfinite(matrix).all(),'Nonfinite predictions')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))}
             for n,s in scores.items()}
    for n in previous['names']:
        require(all(metrics[n][k]==old_result['metrics'][n][k] for k in ('AUROC','AP')),
                'Historical metrics changed')
    for n in names:
        for k,metric in enumerate(('AUROC','AP')):
            metrics[n][metric+'_patient_cluster_95']=np.quantile(boots[n][:,k],[.025,.975]).tolist()
    pairs=[]
    for title in ('DenseNet','BioViL'):
        pairs.extend([(title+'_DINO',title+'_A1'),(title+'_A2',title+'_DINO')])
    pairs.extend([('DenseNet_DINO','DenseNet_B500'),('DenseNet_DINO','DenseNet_P_real')])
    comparisons={}
    for a,base in pairs:
        comparisons[a+' minus '+base]={metric:{
            'delta':metrics[a][metric]-metrics[base][metric],
            'patient_cluster_95':np.quantile((boots[a]-boots[base])[:,k],[.025,.975]).tolist()}
            for k,metric in enumerate(('AUROC','AP'))}
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=matrix,
                        labels=y,image_ids=vs['image_ids'],patient_ids=vs['patient_ids'],
                        **{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
                        metrics=np.stack([boots[n] for n in names],axis=1),
                        original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_DINO_ONLY_S200_DEVELOPMENT_CONTROL','finished':now(),
            'metrics':metrics,'comparisons':comparisons,'new_bank_count':1,'reused_banks':['A1','A2'],
            'seed':101,'images':128,'updates':200,'equal_updates_not_equal_FLOPs':True,
            'interval_scope':'paired percentile95 over V patients conditional on fixed seed101 banks; not synthesis-seed uncertainty',
            'receiver':'DenseNet development receiver, not unseen confirmation',
            'interpretation':'Compare source choice, combination tradeoff and cost; nonsignificance is not equivalence',
            'bootstrap_draws':2000,'bootstrap_sklearn_error':error,
            'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
            'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
            'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
            'synthesis_worker_seconds':{
                'A1':read(PREVIOUS/'A1_result.json')['worker_seconds'],
                'A2':read(PREVIOUS/'A2_result.json')['worker_seconds'],
                'DINO':read(root/'DINO_result.json')['cumulative_worker_seconds']},
            'evaluation_seconds':time.monotonic()-started,'automatic_followup':False,
            'DP_Expert_Reserved_final_receiver':False,
            'predictions_sha256':sha(output/'predictions_private.npz')}
    save(output/'result.json',result)
    print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':main()

