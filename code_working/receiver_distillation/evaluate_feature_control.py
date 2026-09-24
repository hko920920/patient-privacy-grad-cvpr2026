"""One sealed feature-control PNG; frozen three development readouts."""
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score,average_precision_score
from prrd_v3.contracts import CODE,OUT,read,save,sha,require,setup,now
from prrd_v3.datasets import manifests,PixelAccess
from prrd_v3.encoders import Encoder,state_hash,pil_tensor,preprocess
from prrd_v3.bank_runtime import verify_artifact
from prrd_v3.guarded_objectives import bank_moments
from prrd_pilot_20260921.common import ROOT as PILOT,BOOT,packet,check_saved_order,bootstrap_metrics
from prrd_pilot_20260921.evaluate import learn
from .evaluate_feasibility import DIAG
from receiver_resnet_eval_20260924.run import project,direct_weighted_solve,CHECKPOINT,CHECKPOINT_SHA

PRIOR=CODE/'_reports/receiver_dino_dp8_noise_repeat_20260924_v1'
RESNET=CODE/'_reports/receiver_resnet18_reuse_20260924_v1'


def dependencies():
    paths=[Path(__file__),CODE/'receiver_resnet_eval_20260924/run.py',CHECKPOINT,BOOT,
        DIAG/'source_V_point_features.npz',PILOT/'evaluation/recipient_V_features.npz',
        OUT/'source_P_projection.npz',OUT/'recipient_P_projection.npz']
    paths += [PRIOR/'evaluation'/n for n in ('result.json','predictions_private.npz','paired_bootstrap.npz')]
    paths += [RESNET/n for n in ('result.json','contract.json','V_features.npz','P_projection.npz','predictions_private.npz','paired_bootstrap.npz')]
    return {str(p):sha(p) for p in paths}


def run(root):
    tick=time.monotonic();c=read(root/'campaign_contract.json');binding=read(root/'execution_bindings.json')
    deadline=datetime.fromisoformat(c['absolute_deadline_utc']).timestamp()
    def on_time():require(time.time()<deadline,'Evaluation package cap reached')
    on_time();require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation input changed')
    bank=root/'banks'/c['bank_ids']['DP8'];done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
    seal=read(root/'bank_seal.json');require(done['completed_updates']==200,'Final200 required')
    require(sha(bank/'COMPLETED.json')==seal['completion_sha256'] and sha(bank/'artifact/artifact_seal.json')==seal['artifact_sha256'],'Bank seal changed')
    verify_artifact(bank/'artifact',spec,done['signature'])
    output=root/'evaluation';output.mkdir(exist_ok=False);setup();groups=manifests();access=PixelAccess(groups,()).install()
    with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    ysyn=np.array([int(r['label']) for r in rows]);require(len(rows)==128 and np.array_equal(ysyn,[0]*64+[1]*64),'PNG labels')
    v={'BioViL':packet(DIAG/'source_V_point_features.npz'),'DenseNet':packet(PILOT/'evaluation/recipient_V_features.npz'),
       'ResNet18':packet(RESNET/'V_features.npz')}
    for d in v.values():check_saved_order(d,groups['V'])
    cached={title:packet(PRIOR/'evaluation/predictions_private.npz') for title in ('BioViL','DenseNet')}
    cached['ResNet18']=packet(RESNET/'predictions_private.npz')
    oldboot={title:packet(PRIOR/'evaluation/paired_bootstrap.npz') for title in ('BioViL','DenseNet')}
    oldboot['ResNet18']=packet(RESNET/'paired_bootstrap.npz')
    draws=packet(BOOT);pids=draws['patient_ids'];mult=draws['patient_counts'];y=v['DenseNet']['labels'];integer=v['DenseNet']['patient_ids'].astype(np.int64)
    require(mult.shape==(2000,2026) and np.array_equal(np.unique(integer),pids),'Patient draws changed')
    scores={};boots={};png_times={};weights={};solve_details={};projection_errors={}
    for title,kind in (('BioViL','source'),('DenseNet','recipient'),('ResNet18','resnet')):
        on_time();start=time.monotonic();p=cached[title];pb=oldboot[title]
        for key in ('labels','image_ids','patient_ids'):require(np.array_equal(p[key],v[title][key]),'Cached prediction/feature order')
        require(np.array_equal(p['names'],pb['names']) and str(pb['original_patient_draws_sha256'])==sha(BOOT),'Historical bootstrap changed')
        for arm in ('DINO_DP1','DINO_DP2'):
            historical=arm if kind=='resnet' else title+'_'+arm
            ix=list(p['names']).index(historical);name=title+'_'+arm
            scores[name]=p['scores'][:,ix];boots[name]=pb['metrics'][:,ix]
        if kind=='resnet':
            require(sha(CHECKPOINT)==CHECKPOINT_SHA,'ResNet checkpoint changed')
            enc=resnet18();enc.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True));enc.fc=torch.nn.Identity()
            enc=enc.cuda().eval().requires_grad_(False)
        else:enc=Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=state_hash(enc);pieces=[]
        for i in range(0,128,16):
            on_time();images=[]
            for row in rows[i:i+16]:
                with Image.open(bank/'artifact'/row['file']) as im:im=pil_tensor(im.convert('L'))
                images.append(preprocess(im,'recipient') if kind=='resnet' else im)
            with torch.inference_mode():z=enc(torch.stack(images).cuda())
            pieces.append(z.cpu().numpy())
        require(state_hash(enc)==before,'Frozen evaluation encoder changed')
        h=np.concatenate(pieces);del enc;torch.cuda.empty_cache()
        if kind=='resnet':z,projection_errors[title]=project(h,RESNET/'P_projection.npz')
        else:z=h
        ztensor=torch.from_numpy(z);w,detail=learn(bank_moments({'z':ztensor,'a':ztensor},'B',{}),0.)
        independent,_=direct_weighted_solve(z,ysyn);require(np.max(abs(w-independent))<1e-9,'Readout solve mismatch')
        name=title+'_FEATURE_DP1';weights[name]=w;solve_details[title]=detail
        scores[name]=np.asarray(v[title]['z'],np.float64)@w
        boots[name]=bootstrap_metrics(y,scores[name],integer,mult,pids)
        np.savez(output/(title+'_PNG_features.npz'),z=z,labels=ysyn)
        png_times[title]=time.monotonic()-start
        print(json.dumps({'phase':'FEATURE_CONTROL_EVALUATED','receiver':title,'seconds':png_times[title]}),flush=True)
    _,inverse=np.unique(integer,return_inverse=True);err=0.
    for k in (0,17,1999):
        sw=mult[k,inverse]
        for name,s in scores.items():
            ref=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
            err=max(err,float(abs(ref-boots[name][k]).max()))
    require(err<1e-12,'Independent patient bootstrap mismatch')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    for n in metrics:
        for j,m in enumerate(('AUROC','AP')):metrics[n][m+'_patient_cluster_95']=np.quantile(boots[n][:,j],[.025,.975]).tolist()
    contrasts={}
    for title in v:
        for arm in ('DINO_DP1','DINO_DP2'):
            a=title+'_FEATURE_DP1';b=title+'_'+arm
            contrasts[a+'_minus_'+arm]={m:{'delta':metrics[a][m]-metrics[b][m],
                'patient_cluster_95':np.quantile((boots[a]-boots[b])[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
    names=list(scores)
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=np.column_stack(list(scores.values())),
                        labels=y,image_ids=v['DenseNet']['image_ids'],patient_ids=v['DenseNet']['patient_ids'],**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),metrics=np.stack([boots[n] for n in names],1),original_patient_draws_sha256=np.array(sha(BOOT)))
    on_time();result={'status':'COMPLETE_FEATURE_RECIPE_COMPARISON','finished':now(),'metrics':metrics,'comparisons':contrasts,
        'primary':'DenseNet feature-mean/L2 versus each existing DINO gradient/cosine bank; whole recipe effect',
        'secondary':'Existing ResNet18 development readout and BioViL diagnostic; no new receiver search',
        'inference_limit':'One feature noise versus two gradient noises; fixed-bank patient intervals; not noise-population method superiority',
        'new_private_releases':1,'new_training_banks':1,'new_updates':200,'new_Q_V_pixel_forwards':0,
        'PNG_forwards':{'BioViL':128,'DenseNet':128,'ResNet18':128},'PNG_seconds':png_times,
        'bootstrap_sklearn_error':err,'raw_pixel_access':access.report(),'readout_details':solve_details,
        'projection_errors':projection_errors,'evaluation_seconds':time.monotonic()-tick,
        'Dosser_full_reproduction':False,'automatic_followup':False,'Expert_Reserved_final_receiver':False}
    save(output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
