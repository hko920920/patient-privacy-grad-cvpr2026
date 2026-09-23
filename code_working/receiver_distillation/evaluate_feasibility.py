"""Fixed A1/A2 development readout; both final PNG banks must already be sealed."""
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
from prrd_v3.encoders import Encoder, state_hash, pil_tensor, BIO_PATH, DENSE_PATH
from prrd_v3.bank_runtime import verify_artifact
from prrd_v3.guarded_objectives import bank_moments
from prrd_pilot_20260921.common import ROOT as PILOT, BOOT, packet, check_saved_order, bootstrap_metrics
from prrd_pilot_20260921.evaluate import learn

DIAG=CODE/'_reports/prrd_source_readout_20260921_v1'


def dependencies():
    paths=[BIO_PATH,DENSE_PATH,OUT/'source_P_projection.npz',OUT/'recipient_P_projection.npz',BOOT,
           DIAG/'source_V_point_features.npz',DIAG/'predictions_private.npz',DIAG/'paired_bootstrap.npz',
           DIAG/'result.json',PILOT/'evaluation/recipient_V_features.npz',
           PILOT/'evaluation/predictions_and_readouts_private.npz',
           PILOT/'evaluation/paired_patient_bootstrap.npz',PILOT/'evaluation/result.json',
           Path(__file__),CODE/'prrd_pilot_20260921/common.py',CODE/'prrd_pilot_20260921/evaluate.py']
    return {str(p):sha(p) for p in paths}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    root=args.root; start=time.monotonic()
    bindings=read(root/'execution_bindings.json')
    require(all(sha(p)==h for p,h in bindings['evaluation_dependencies'].items()),'Evaluation contract changed')
    seal=read(root/'two_bank_seal.json')
    contract=read(root/'campaign_contract.json')
    ids=contract['budget']['bank_ids']
    require(list(seal['banks'])==ids,'Both banks not sealed')
    banks={}
    for arm,rid in zip(('A1','A2'),ids):
        bank=root/'banks'/rid; done=read(bank/'COMPLETED.json'); spec=read(bank/'run_spec.json')
        require(done['completed_updates']==200 and spec['updates']==200,'Not final200')
        require(sha(bank/'COMPLETED.json')==seal['banks'][rid]['completion_sha256'],'Completion changed')
        require(sha(bank/'artifact/artifact_seal.json')==seal['banks'][rid]['artifact_sha256'],'PNG seal changed')
        verify_artifact(bank/'artifact',spec,done['signature'])
        banks[arm]=bank
    output=root/'evaluation'; output.mkdir(exist_ok=False)
    setup(); groups=manifests(); access=PixelAccess(groups,()).install()
    weights,details,counts,times={},{},{},{}
    for kind,title in (('source','BioViL'),('recipient','DenseNet')):
        tick=time.monotonic()
        encoder=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(encoder)
        for arm,bank in banks.items():
            with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            require(len(rows)==128 and [int(r['label']) for r in rows]==[0]*64+[1]*64,'PNG labels/order')
            pieces=[]
            for i in range(0,128,16):
                images=[]
                for row in rows[i:i+16]:
                    with Image.open(bank/'artifact'/row['file']) as im:images.append(pil_tensor(im.convert('L')))
                with torch.no_grad(): z=encoder(torch.stack(images).cuda())
                pieces.append(z.cpu())
            z=torch.cat(pieces)
            # B point-only branch never consumes the a placeholder or a relation scale.
            moments=bank_moments({'z':z,'a':z},'B',{})
            name=title+'_'+arm
            weights[name],details[name]=learn(moments,0.)
            np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
            counts[name]=128
        require(state_hash(encoder)==before,'Evaluation model mutated')
        times[title]=time.monotonic()-tick
        del encoder; torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz')
    vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr): check_saved_order(v,groups['V'])
    old_s=packet(DIAG/'predictions_private.npz')
    old_r=packet(PILOT/'evaluation/predictions_and_readouts_private.npz')
    for saved in (vr,old_s,old_r):
        for key in ('image_ids','patient_ids','labels'):
            require(np.array_equal(vs[key],saved[key]),'V order mismatch')
    scores={name:np.asarray(vs['z'] if name.startswith('BioViL') else vr['z'],float)@w
            for name,w in weights.items()}
    scores.update({
        'BioViL_B500':old_s['scores'][:,list(old_s['names']).index('B_PNG')],
        'DenseNet_B500':old_r['scores'][:,list(old_r['names']).index('B')],
        'DenseNet_P_real':old_r['scores'][:,list(old_r['names']).index('P_real_point')],
    })
    names=list(scores); y=vs['labels']
    matrix=np.column_stack(list(scores.values())); require(np.isfinite(matrix).all(),'Nonfinite evaluation')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    for new,old,path in [('BioViL_B500','B_PNG',DIAG/'result.json'),
                         ('DenseNet_B500','B',PILOT/'evaluation/result.json'),
                         ('DenseNet_P_real','P_real_point',PILOT/'evaluation/result.json')]:
        previous=read(path)['metrics'][old]
        require(all(metrics[new][key]==previous[key] for key in ('AUROC','AP')),'Historical reference changed')
    b=packet(BOOT); pids=b['patient_ids']; draws=b['patient_counts']
    integer=vs['patient_ids'].astype(np.int64)
    require(draws.shape==(2000,2026) and np.array_equal(integer.astype(str),vs['patient_ids'])
            and np.array_equal(np.unique(integer),pids),'Original bootstrap ordering changed')
    boots={n:bootstrap_metrics(y,scores[n],integer,draws,pids) for n in weights}
    bs=packet(DIAG/'paired_bootstrap.npz'); br=packet(PILOT/'evaluation/paired_patient_bootstrap.npz')
    require(str(bs['original_bootstrap_sha256'])==sha(BOOT) and str(br['patient_counts_sha256'])==sha(BOOT),
            'Cached bootstrap draws changed')
    boots['BioViL_B500']=bs['metrics'][:,list(bs['names']).index('B_PNG')]
    for new,old in [('DenseNet_B500','B'),('DenseNet_P_real','P_real_point')]:
        boots[new]=br['metrics'][:,list(br['names']).index(old)]
    _,inverse=np.unique(integer,return_inverse=True); error=0.
    for k in (0,17,1999):
        sw=draws[k,inverse]
        for n in names:
            expected=np.array([roc_auc_score(y,scores[n],sample_weight=sw),
                               average_precision_score(y,scores[n],sample_weight=sw)])
            error=max(error,float(np.max(abs(expected-boots[n][k]))))
    require(error<=1e-12,'Bootstrap independent check failed')
    for n in names:
        for k,metric in enumerate(('AUROC','AP')):
            metrics[n][metric+'_patient_cluster_95']=np.quantile(boots[n][:,k],[.025,.975]).tolist()
    comparisons={}
    for a,base in [('DenseNet_A2','DenseNet_A1'),('BioViL_A2','BioViL_A1'),
                   ('DenseNet_A1','DenseNet_B500'),('DenseNet_A2','DenseNet_B500'),
                   ('DenseNet_A1','DenseNet_P_real'),('DenseNet_A2','DenseNet_P_real')]:
        comparisons[a+' minus '+base]={metric:{
            'delta':metrics[a][metric]-metrics[base][metric],
            'patient_cluster_95':np.quantile((boots[a]-boots[base])[:,k],[.025,.975]).tolist()}
            for k,metric in enumerate(('AUROC','AP'))}
    primary=comparisons['DenseNet_A2 minus DenseNet_A1']
    source=comparisons['BioViL_A2 minus BioViL_A1']
    gates={'DenseNet_AUROC_delta_ge_0_02':primary['AUROC']['delta']>=.02,
           'DenseNet_AP_nondecrease':primary['AP']['delta']>=0,
           'BioViL_AUROC_delta_ge_minus0_02':source['AUROC']['delta']>=-.02}
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=matrix,
                        labels=y,image_ids=vs['image_ids'],patient_ids=vs['patient_ids'],
                        **{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
                        metrics=np.stack([boots[n] for n in names],axis=1),
                        original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_MATCHED_A1_A2_S200_DEVELOPMENT_COMPARISON','finished':now(),
            'metrics':metrics,'comparisons':comparisons,'gates':gates,'investment_signal':all(gates.values()),
            'seed':101,'bank_count':2,'images_per_bank':128,'updates_per_bank':200,
            'equal_updates_not_equal_FLOPs':True,'historical_B500_context_only':True,
            'receiver':'DenseNet reused development receiver; not unseen final',
            'interval_scope':'paired percentile95 over V patients conditional on fixed seed101 banks; no synthesis-seed uncertainty',
            'bootstrap_draws':2000,'bootstrap_sklearn_error':error,
            'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
            'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
            'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
            'evaluation_seconds':time.monotonic()-start,'automatic_followup':False,
            'DP_Expert_Reserved_final_receiver':False,
            'predictions_sha256':sha(output/'predictions_private.npz')}
    save(output/'result.json',result)
    print(__import__('json').dumps(result),flush=True)


if __name__=='__main__': main()
