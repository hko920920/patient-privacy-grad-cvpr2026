"""One new final DP PNG bank; reuse all prior scores and fixed patient draws."""
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
from .evaluate_feasibility import DIAG
from .evaluate_patient_dp_dino import dependencies as previous_dependencies

PREVIOUS=CODE/'_reports/receiver_dino_dp8_20260924_v1'

def dependencies():
    result=previous_dependencies()
    for p in (Path(__file__),PREVIOUS/'campaign_result.json',PREVIOUS/'evaluation/result.json',
              PREVIOUS/'evaluation/predictions_private.npz',PREVIOUS/'evaluation/paired_bootstrap.npz'):
        result[str(p)]=sha(p)
    return result

def classify(contrast):
    gates={'AUROC_delta_positive':contrast['AUROC']['delta']>0,
           'AUROC_patient_CI_lower_positive':contrast['AUROC']['patient_cluster_95'][0]>0,
           'AP_point_nondecrease':contrast['AP']['delta']>=0}
    judgment='POSITIVE_DEVELOPMENT_SIGNAL' if all(gates.values()) else (
        'MIXED_OR_UNCERTAIN' if gates['AUROC_delta_positive'] else 'NO_POSITIVE_AUROC_ADDITIONAL_VALUE')
    return gates,judgment

def run(root):
    tick=time.monotonic();binding=read(root/'execution_bindings.json');c=read(root/'campaign_contract.json')
    deadline=datetime.fromisoformat(c['absolute_deadline_utc']).timestamp()
    require(time.time()<deadline,'Whole package cap reached')
    require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation dependency changed')
    seal=read(root/'bank_seal.json');bank=root/'banks'/c['bank_ids']['DP8']
    done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
    require(done['completed_updates']==200 and spec['variant']=='DP8' and spec['seed']==101,'Wrong final bank')
    require(sha(bank/'COMPLETED.json')==seal['completion_sha256'],'Completion changed')
    require(sha(bank/'artifact/artifact_seal.json')==seal['artifact_sha256'],'PNG seal changed')
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
            require(time.time()<deadline,'Evaluation time cap reached')
            images=[]
            for row in rows[i:i+16]:
                with Image.open(bank/'artifact'/row['file']) as im:images.append(pil_tensor(im.convert('L')))
            with torch.no_grad():z=encoder(torch.stack(images).cuda())
            pieces.append(z.cpu())
        z=torch.cat(pieces);name=title+'_DINO_DP2'
        weights[name],details[name]=learn(bank_moments({'z':z,'a':z},'B',{}),0.)
        np.savez(output/(name+'_PNG_features.npz'),z=z.numpy(),labels=np.array([0]*64+[1]*64))
        counts[name]=128
        require(state_hash(encoder)==before,'Evaluation encoder changed')
        times[title]=time.monotonic()-started;del encoder;torch.cuda.empty_cache()
    vs=packet(DIAG/'source_V_point_features.npz');vr=packet(PILOT/'evaluation/recipient_V_features.npz')
    for v in (vs,vr):check_saved_order(v,groups['V'])
    prior=packet(PREVIOUS/'evaluation/predictions_private.npz')
    pb=packet(PREVIOUS/'evaluation/paired_bootstrap.npz');previous=read(PREVIOUS/'evaluation/result.json')
    for v in (vr,prior):
        for key in ('labels','image_ids','patient_ids'):require(np.array_equal(v[key],vs[key]),'V order changed')
    require(np.array_equal(prior['names'],pb['names']) and str(pb['original_patient_draws_sha256'])==sha(BOOT),'Bootstrap changed')
    scores,boots={},{}
    rename={'P':'P','PQ':'PQ','CLIP':'CLIP','DP1':'DP1','DP2':'DP2','DINO_DP1':'DINO_DP'}
    for title in ('BioViL','DenseNet'):
        for arm,oldarm in rename.items():
            name=title+'_'+arm;idx=list(prior['names']).index(title+'_'+oldarm)
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
        for arm,oldarm in rename.items():
            require(all(metrics[title+'_'+arm][m]==previous['metrics'][title+'_'+oldarm][m] for m in ('AUROC','AP')),'Historical metric changed')
    for name in metrics:
        for k,m in enumerate(('AUROC','AP')):
            metrics[name][m+'_patient_cluster_95']=np.quantile(boots[name][:,k],[.025,.975]).tolist()
    contrasts={}
    for title in ('BioViL','DenseNet'):
        for a,b in (('DP1','DINO_DP1'),('DP1','DINO_DP2'),('DP2','DINO_DP1'),('DP2','DINO_DP2'),('DINO_DP2','DINO_DP1'),('DINO_DP2','P')):
            left,right=title+'_'+a,title+'_'+b
            contrasts[title+'_'+a+'_minus_'+b]={m:{'delta':metrics[left][m]-metrics[right][m],
                'patient_cluster_95':np.quantile((boots[left]-boots[right])[:,k],[.025,.975]).tolist()}
                for k,m in enumerate(('AUROC','AP'))}
    means={title:{family:{metric:float(np.mean([metrics[title+'_'+arm][metric] for arm in arms]))
        for metric in ('AUROC','AP')} for family,arms in
        (('A2',('DP1','DP2')),('DINO',('DINO_DP1','DINO_DP2')))} for title in ('BioViL','DenseNet')}
    mean_differences={title:{metric:means[title]['DINO'][metric]-means[title]['A2'][metric]
        for metric in ('AUROC','AP')} for title in means}
    judgment='DESCRIPTIVE_TWO_NOISE_COMPARISON'
    names=list(scores)
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),
        scores=np.column_stack(list(scores.values())),labels=y,image_ids=vs['image_ids'],
        patient_ids=vs['patient_ids'],**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),
        metrics=np.stack([boots[n] for n in names],axis=1),original_patient_draws_sha256=np.array(sha(BOOT)))
    result={'status':'COMPLETE_DINO_DP_NOISE_REPEAT_COMPARISON','finished':now(),
        'metrics':metrics,'comparisons':contrasts,'two_noise_point_means':means,'DINO_minus_A2_point_mean':mean_differences,'development_judgment':judgment,
        'mean_scope':'Arithmetic mean of two fixed bank metrics per method; not ensemble/noise-population estimate or equivalence proof',
        'comparison_scope':'All four A2-minus-DINO fixed-bank comparisons; second-minus-first DINO; independent noise labels are not matched pairs; patient-paired CIs only',
        'interval_scope':'same2000 V patient draws; conditional on fixed PNGs and realized DP noise',
        'seed':101,'new_banks':1,'updates_per_bank':200,'images_per_bank':128,'new_private_releases':1,
        'A2_family_private_releases':2,'DINO_family_private_releases':2,'per_release_epsilon':8.,'per_release_delta':1e-5,
        'joint_basic_composition':{'epsilon':32.,'delta':4e-5,'optimal_accounting_claimed':False},
        'adjacency':'add_remove_one_patient','bootstrap_sklearn_error':error,
        'receiver':'DenseNet adaptive development; BioViL is source only for A2, excluded from DINO-only synthesis',
        'V_images':len(y),'V_patients':len(pids),'V_positive_images':int(y.sum()),
        'PNG_forwards':counts,'PNG_seconds':times,'raw_pixel_access':access.report(),
        'new_V_forwards':0,'new_Q_forwards':0,'readout_details':details,
        'synthesis_seconds':read(root/'DP8_result.json')['cumulative_worker_seconds'],
        'evaluation_seconds':time.monotonic()-tick,'automatic_followup':False,
        'Expert_Reserved_final_receiver':False,'predictions_sha256':sha(output/'predictions_private.npz')}
    require(time.time()<deadline,'Campaign cap reached during evaluation')
    save(output/'result.json',result);print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();run(args.root)

if __name__=='__main__':main()

