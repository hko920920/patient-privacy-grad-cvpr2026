"""Compare seed202 DINO/A2 only after both final banks are sealed."""
import argparse,csv,time
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

OLD=CODE/'_reports/receiver_dino_only_s200_20260923_v1'
FIRST=CODE/'_reports/receiver_a1a2_s200_20260922_v1'


def dependencies():
    result=previous_dependencies()
    for p in (Path(__file__),OLD/'evaluation/result.json',OLD/'evaluation/predictions_private.npz',
              OLD/'evaluation/paired_bootstrap.npz',OLD/'DINO_result.json',FIRST/'A2_result.json'):
        result[str(p)]=sha(p)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;started=time.monotonic();binding=read(root/'execution_bindings.json')
    require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation changed')
    contract=read(root/'campaign_contract.json');seal=read(root/'two_bank_seal.json')
    require(list(seal['banks'])==contract['bank_ids'],'Both seed202 banks must be sealed')
    banks={}
    for arm,rid in zip(('DINO','A2'),contract['bank_ids']):
        bank=root/'banks'/rid;done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        require(done['completed_updates']==200 and spec['seed']==202 and spec['updates']==200,'Wrong final bank')
        require(sha(bank/'COMPLETED.json')==seal['banks'][rid]['completion_sha256'],'Completion changed')
        require(sha(bank/'artifact/artifact_seal.json')==seal['banks'][rid]['artifact_sha256'],'PNG changed')
        verify_artifact(bank/'artifact',spec,done['signature'])
        require(read(bank/'artifact/provenance_public.json')['seed']==202,'Export seed mismatch')
        banks[arm]=bank
    output=root/'evaluation';output.mkdir(exist_ok=False)
    setup();groups=manifests();access=PixelAccess(groups,()).install()
    weights,details,counts,times={},{},{},{}
    for kind,title in (('source','BioViL'),('recipient','DenseNet')):
        tick=time.monotonic();encoder=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(encoder)
        for arm,bank in banks.items():
            with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            require(len(rows)==128 and [int(r['label']) for r in rows]==[0]*64+[1]*64,'PNG label/order')
            pieces=[]
            for i in range(0,128,16):
                images=[]
                for row in rows[i:i+16]:
                    with Image.open(bank/'artifact'/row['file']) as im:images.append(pil_tensor(im.convert('L')))
                with torch.no_grad():z=encoder(torch.stack(images).cuda())
                pieces.append(z.cpu())
            z=torch.cat(pieces);name=title+'_'+arm+'_202'
            weights[name],details[name]=learn(bank_moments({'z':z,'a':z},'B',{}),0.)
            np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
            counts[name]=128
        require(state_hash(encoder)==before,'Evaluation encoder mutated')
        times[title]=time.monotonic()-tick;del encoder;torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz');vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr):check_saved_order(v,groups['V'])
    previous=packet(OLD/'evaluation/predictions_private.npz');old_boot=packet(OLD/'evaluation/paired_bootstrap.npz')
    old_result=read(OLD/'evaluation/result.json')
    for v in (vr,previous):
        for k in ('labels','image_ids','patient_ids'):require(np.array_equal(v[k],vs[k]),'V order changed')
    require(np.array_equal(previous['names'],old_boot['names']),'Cached bootstrap names')
    require(str(old_boot['original_patient_draws_sha256'])==sha(BOOT),'Cached draw hash')
    scores,boots={},{}
    for title in ('BioViL','DenseNet'):
        for arm in ('DINO','A2'):
            old=title+'_'+arm;name=old+'_101';index=list(previous['names']).index(old)
            scores[name]=previous['scores'][:,index];boots[name]=old_boot['metrics'][:,index]
    for n,w in weights.items():scores[n]=np.asarray(vs['z'] if n.startswith('BioViL') else vr['z'],float)@w
    y=vs['labels'];p=packet(BOOT);pids=p['patient_ids'];draws=p['patient_counts'];integer=vs['patient_ids'].astype(np.int64)
    require(draws.shape==(2000,2026) and np.array_equal(np.unique(integer),pids),'Bootstrap ordering')
    for n in weights:boots[n]=bootstrap_metrics(y,scores[n],integer,draws,pids)
    _,inverse=np.unique(integer,return_inverse=True);error=0.
    for k in (0,17,1999):
        sw=draws[k,inverse]
        for n in weights:
            expected=np.array([roc_auc_score(y,scores[n],sample_weight=sw),average_precision_score(y,scores[n],sample_weight=sw)])
            error=max(error,float(np.max(abs(expected-boots[n][k]))))
    require(error<=1e-12,'Bootstrap independent calculation')
    names=list(scores);matrix=np.column_stack(list(scores.values()));require(np.isfinite(matrix).all(),'Invalid predictions')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    for n in names:
        if n.endswith('_101'):
            require(all(metrics[n][k]==old_result['metrics'][n[:-4]][k] for k in ('AUROC','AP')),'Old metric changed')
        for k,metric in enumerate(('AUROC','AP')):
            metrics[n][metric+'_patient_cluster_95']=np.quantile(boots[n][:,k],[.025,.975]).tolist()
    contrasts={}
    for seed in (101,202):
        for title in ('BioViL','DenseNet'):
            a=title+'_A2_'+str(seed);b=title+'_DINO_'+str(seed)
            contrasts[title+'_A2_minus_DINO_'+str(seed)]={metric:{
                'delta':metrics[a][metric]-metrics[b][metric],
                'patient_cluster_95':np.quantile((boots[a]-boots[b])[:,k],[.025,.975]).tolist()}
                for k,metric in enumerate(('AUROC','AP'))}
    means={title:{metric:float(np.mean([contrasts[title+'_A2_minus_DINO_'+str(seed)][metric]['delta']
                                     for seed in (101,202)])) for metric in ('AUROC','AP')}
           for title in ('BioViL','DenseNet')}
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=matrix,
                        labels=y,image_ids=vs['image_ids'],patient_ids=vs['patient_ids'],
                        **{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
                        metrics=np.stack([boots[n] for n in names],axis=1),
                        original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_MATCHED_DINO_A2_SEED202','finished':now(),'metrics':metrics,
            'comparisons':contrasts,'mean_two_seed_delta':means,
            'mean_interpretation':'arithmetic mean of bank metric differences, not prediction ensemble; no population-of-seeds CI from two seeds',
            'source_set':'DINO versus BioViL+DINO, equal encoder mean','new_seed':202,'reused_seed':101,
            'condition_head_seed':101,'template_and_residual_seed':202,'new_bank_count':2,
            'images_per_bank':128,'updates_per_bank':200,'equal_updates_not_equal_FLOPs':True,
            'interval_scope':'2000 paired V-patient percentile95 draws, conditional on each pair of fixed synthesis banks',
            'receiver':'DenseNet adaptive development; BioViL is A2 source, not an independent unseen receiver',
            'bootstrap_sklearn_error':error,'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
            'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
            'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
            'synthesis_seconds':{
                'DINO_101':read(OLD/'DINO_result.json')['cumulative_worker_seconds'],
                'A2_101':read(FIRST/'A2_result.json')['cumulative_worker_seconds'],
                **{arm+'_202':read(root/(arm+'_result.json'))['cumulative_worker_seconds'] for arm in ('DINO','A2')}},
            'evaluation_seconds':time.monotonic()-started,'automatic_followup':False,
            'DP_Expert_Reserved_final_receiver':False,'predictions_sha256':sha(output/'predictions_private.npz')}
    save(output/'result.json',result);print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':main()

