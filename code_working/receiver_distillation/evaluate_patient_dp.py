"""Four-cell seed101 comparison, only after both new final PNG banks seal."""
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime
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
from .evaluate_feasibility import DIAG,dependencies as existing_dependencies

PUBLIC=CODE/'_reports/receiver_a2_public_s200_20260923_v1'
FIRST=CODE/'_reports/receiver_a1a2_s200_20260922_v1'


def dependencies():
    result=existing_dependencies()
    for p in (Path(__file__),PUBLIC/'A2_P_result.json',FIRST/'A2_result.json',
              PUBLIC/'evaluation/result.json',PUBLIC/'evaluation/predictions_private.npz',
              PUBLIC/'evaluation/paired_bootstrap.npz'):
        result[str(p)]=sha(p)
    return result


def run(root):
    tick=time.monotonic();binding=read(root/'execution_bindings.json');c=read(root/'campaign_contract.json')
    deadline=datetime.fromisoformat(c['absolute_deadline_utc']).timestamp()
    require(time.time()<deadline,'Campaign time cap exhausted')
    require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation dependency changed')
    seal=read(root/'two_bank_seal.json');banks={}
    require(list(seal['banks'])==list(c['bank_ids'].values()),'Both banks must seal first')
    for arm,rid in c['bank_ids'].items():
        bank=root/'banks'/rid;done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        require(done['completed_updates']==200 and spec['variant']==arm and spec['seed']==101,'Wrong final bank')
        require(sha(bank/'COMPLETED.json')==seal['banks'][rid]['completion_sha256'],'Completion changed')
        require(sha(bank/'artifact/artifact_seal.json')==seal['banks'][rid]['artifact_sha256'],'PNG changed')
        verify_artifact(bank/'artifact',spec,done['signature']);banks[arm]=bank
    output=root/'evaluation';output.mkdir(exist_ok=False)
    setup();groups=manifests();access=PixelAccess(groups,()).install()
    weights,details,counts,times={},{},{},{}
    for kind,title in (('source','BioViL'),('recipient','DenseNet')):
        started=time.monotonic();encoder=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(encoder)
        for arm,bank in banks.items():
            require(time.time()<deadline,'Campaign time cap exhausted')
            with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
            require(len(manifest)==128 and [int(r['label']) for r in manifest]==[0]*64+[1]*64,'PNG labels/order')
            pieces=[]
            for i in range(0,128,16):
                images=[]
                for row in manifest[i:i+16]:
                    with Image.open(bank/'artifact'/row['file']) as im:images.append(pil_tensor(im.convert('L')))
                with torch.no_grad():z=encoder(torch.stack(images).cuda())
                pieces.append(z.cpu())
            z=torch.cat(pieces);name=title+'_'+arm
            weights[name],details[name]=learn(bank_moments({'z':z,'a':z},'B',{}),0.)
            np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
            counts[name]=128
        require(state_hash(encoder)==before,'Evaluation encoder changed')
        times[title]=time.monotonic()-started;del encoder;torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz');vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr):check_saved_order(v,groups['V'])
    prior=packet(PUBLIC/'evaluation/predictions_private.npz');pb=packet(PUBLIC/'evaluation/paired_bootstrap.npz')
    previous=read(PUBLIC/'evaluation/result.json')
    for v in (vr,prior):
        for key in ('labels','image_ids','patient_ids'):require(np.array_equal(v[key],vs[key]),'V order changed')
    require(np.array_equal(prior['names'],pb['names']) and str(pb['original_patient_draws_sha256'])==sha(BOOT),
            'Prior bootstrap labels/draws differ')
    scores,boots={},{}
    for title in ('BioViL','DenseNet'):
        for arm in ('P','PQ'):
            name=title+'_'+arm;old=title+'_A2_'+arm;idx=list(prior['names']).index(old)
            scores[name]=prior['scores'][:,idx];boots[name]=pb['metrics'][:,idx]
    for name,w in weights.items():scores[name]=np.asarray(vs['z'] if name.startswith('BioViL') else vr['z'],float)@w
    y=vs['labels'];draws=packet(BOOT);pids=draws['patient_ids'];multiplicity=draws['patient_counts']
    integer=vs['patient_ids'].astype(np.int64)
    require(multiplicity.shape==(2000,2026) and np.array_equal(np.unique(integer),pids),'Patient bootstrap changed')
    for name in weights:boots[name]=bootstrap_metrics(y,scores[name],integer,multiplicity,pids)
    _,inverse=np.unique(integer,return_inverse=True);error=0.
    for k in (0,17,1999):
        sw=multiplicity[k,inverse]
        for name in scores:
            expected=np.array([roc_auc_score(y,scores[name],sample_weight=sw),
                               average_precision_score(y,scores[name],sample_weight=sw)])
            error=max(error,float(abs(expected-boots[name][k]).max()))
    require(error<=1e-12,'Independent bootstrap mismatch')
    metrics={name:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for name,s in scores.items()}
    for title in ('BioViL','DenseNet'):
        for arm in ('P','PQ'):
            require(all(metrics[title+'_'+arm][m]==previous['metrics'][title+'_A2_'+arm][m] for m in ('AUROC','AP')),
                    'Historical baseline metrics changed')
    for name in metrics:
        for k,m in enumerate(('AUROC','AP')):metrics[name][m+'_patient_cluster_95']=np.quantile(boots[name][:,k],[.025,.975]).tolist()
    contrasts={}
    for title in ('BioViL','DenseNet'):
        for a,b in (('DP8','P'),('CLIP','PQ'),('DP8','CLIP'),('DP8','PQ'),('CLIP','P')):
            left,right=title+'_'+a,title+'_'+b
            contrasts[title+'_'+a+'_minus_'+b]={m:{'delta':metrics[left][m]-metrics[right][m],
                'patient_cluster_95':np.quantile((boots[left]-boots[right])[:,k],[.025,.975]).tolist()}
                for k,m in enumerate(('AUROC','AP'))}
    primary=contrasts['DenseNet_DP8_minus_P']
    gates={'AUROC_delta_positive':primary['AUROC']['delta']>0,
           'AUROC_patient_CI_lower_positive':primary['AUROC']['patient_cluster_95'][0]>0,
           'AP_point_nondecrease':primary['AP']['delta']>=0}
    judgment='POSITIVE_DEVELOPMENT_SIGNAL' if all(gates.values()) else (
        'MIXED_OR_UNCERTAIN' if gates['AUROC_delta_positive'] else 'NO_POSITIVE_AUROC_ADDITIONAL_VALUE')
    names=list(scores);matrix=np.column_stack(list(scores.values()));require(np.isfinite(matrix).all(),'Invalid scores')
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=matrix,labels=y,
        image_ids=vs['image_ids'],patient_ids=vs['patient_ids'],**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
        metrics=np.stack([boots[n] for n in names],axis=1),original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_FIRST_PATIENT_DP_FOUR_CELL_COMPARISON','finished':now(),
        'metrics':metrics,'comparisons':contrasts,'primary_gates':gates,'development_judgment':judgment,
        'interval_scope':'2000 paired V patient draws, conditional on fixed banks and single private-noise realization',
        'seed':101,'new_banks':2,'updates_per_bank':200,'images_per_bank':128,'private_releases':1,
        'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient','bootstrap_sklearn_error':error,
        'receiver':'DenseNet adaptive development; BioViL included in source objective',
        'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
        'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
        'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
        'synthesis_seconds':{arm:read(root/(arm+'_result.json'))['cumulative_worker_seconds'] for arm in banks},
        'evaluation_seconds':time.monotonic()-tick,'automatic_followup':False,
        'Expert_Reserved_final_receiver':False,'predictions_sha256':sha(output/'predictions_private.npz')}
    require(time.time()<deadline,'Campaign cap reached during evaluation')
    save(output/'result.json',result);print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();run(args.root)


if __name__=='__main__':main()
