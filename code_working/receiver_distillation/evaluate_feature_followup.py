"""Two final banks, two fixed development receivers, existing patient draws."""
import argparse,csv,json,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from . import evaluate_feature_control as e
from prrd_v3.contracts import CODE,OUT,read,save,sha,require,setup,now
from prrd_v3.bank_runtime import verify_artifact
PREVIOUS=CODE/'_reports/receiver_feature_dp8_20260924_v1'

def dependencies():
    paths=[Path(__file__),Path(e.__file__),e.BOOT,e.CHECKPOINT,e.PILOT/'evaluation/recipient_V_features.npz',
           OUT/'recipient_P_projection.npz',e.RESNET/'V_features.npz',e.RESNET/'P_projection.npz',
           CODE/'receiver_resnet_eval_20260924/run.py']
    paths += [PREVIOUS/'evaluation'/n for n in ('result.json','predictions_private.npz','paired_bootstrap.npz')]
    return {str(p):sha(p) for p in paths}

def run(root):
    tick=time.monotonic();c=read(root/'campaign_contract.json');binding=read(root/'execution_bindings.json')
    def on_time():require(time.time()<datetime.fromisoformat(c['absolute_deadline_utc']).timestamp(),'Package cap')
    on_time();require(all(sha(p)==h for p,h in binding['evaluation_dependencies'].items()),'Evaluation input changed')
    seals=read(root/'both_banks_sealed.json');banks={}
    for variant in ('PUBLIC','DP8'):
        bank=root/'banks'/c['bank_ids'][variant];done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        require(done['completed_updates']==200 and sha(bank/'COMPLETED.json')==seals[variant]['completion_sha256'],'Final200 seal')
        require(sha(bank/'artifact/artifact_seal.json')==seals[variant]['artifact_sha256'],'PNG seal')
        verify_artifact(bank/'artifact',spec,done['signature']);banks[variant]=bank
    output=root/'evaluation';output.mkdir(exist_ok=False);setup()
    groups=e.manifests();access=e.PixelAccess(groups,()).install()
    v={'DenseNet':e.packet(e.PILOT/'evaluation/recipient_V_features.npz'),'ResNet18':e.packet(e.RESNET/'V_features.npz')}
    for d in v.values():e.check_saved_order(d,groups['V'])
    old=e.packet(PREVIOUS/'evaluation/predictions_private.npz');oldboot=e.packet(PREVIOUS/'evaluation/paired_bootstrap.npz')
    draws=e.packet(e.BOOT);pids=draws['patient_ids'];mult=draws['patient_counts']
    y=v['DenseNet']['labels'];integer=v['DenseNet']['patient_ids'].astype(np.int64)
    require(mult.shape==(2000,2026) and np.array_equal(np.unique(integer),pids),'Patient draws changed')
    require(np.array_equal(old['names'],oldboot['names']) and str(oldboot['original_patient_draws_sha256'])==sha(e.BOOT),'Historical bootstrap')
    scores={};boots={};weights={};times={};solve_details={}
    for title,kind in (('DenseNet','recipient'),('ResNet18','resnet')):
        on_time();start=time.monotonic()
        for key in ('labels','image_ids','patient_ids'):require(np.array_equal(old[key],v[title][key]),'Prediction/feature order')
        for arm in ('DINO_DP1','DINO_DP2','FEATURE_DP1'):
            name=title+'_'+arm;ix=list(old['names']).index(name)
            scores[name]=old['scores'][:,ix];boots[name]=oldboot['metrics'][:,ix]
        if kind=='resnet':
            require(sha(e.CHECKPOINT)==e.CHECKPOINT_SHA,'ResNet checkpoint changed')
            enc=e.resnet18();enc.load_state_dict(torch.load(e.CHECKPOINT,map_location='cpu',weights_only=True))
            enc.fc=torch.nn.Identity();enc=enc.cuda().eval().requires_grad_(False)
        else:enc=e.Encoder(kind).use_projection(OUT/(kind+'_P_projection.npz'))
        before=e.state_hash(enc)
        for variant,arm in (('PUBLIC','FEATURE_PUBLIC'),('DP8','FEATURE_DP2')):
            bank=banks[variant]
            with (bank/'artifact/images.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            ysyn=np.array([int(r['label']) for r in rows])
            require(len(rows)==128 and np.array_equal(ysyn,[0]*64+[1]*64),'PNG labels')
            pieces=[]
            for i in range(0,128,16):
                on_time();images=[]
                for row in rows[i:i+16]:
                    with Image.open(bank/'artifact'/row['file']) as im:x=e.pil_tensor(im.convert('L'))
                    images.append(e.preprocess(x,'recipient') if kind=='resnet' else x)
                with torch.inference_mode():z=enc(torch.stack(images).cuda())
                pieces.append(z.cpu().numpy())
            h=np.concatenate(pieces);z=e.project(h,e.RESNET/'P_projection.npz')[0] if kind=='resnet' else h
            zt=torch.from_numpy(z);w,detail=e.learn(e.bank_moments({'z':zt,'a':zt},'B',{}),0.)
            independent,_=e.direct_weighted_solve(z,ysyn)
            require(np.max(abs(w-independent))<1e-9,'Readout solve mismatch')
            name=title+'_'+arm;weights[name]=w;solve_details[name]=detail
            scores[name]=np.asarray(v[title]['z'],np.float64)@w
            boots[name]=e.bootstrap_metrics(y,scores[name],integer,mult,pids)
            np.savez(output/(name+'_PNG_features.npz'),z=z,labels=ysyn)
        require(e.state_hash(enc)==before,'Frozen receiver changed')
        del enc;torch.cuda.empty_cache();times[title]=time.monotonic()-start
        print(json.dumps({'phase':'FEATURE_FOLLOWUP_EVALUATED','receiver':title,'seconds':times[title]}),flush=True)
    _,inverse=np.unique(integer,return_inverse=True);err=0.
    for k in (0,17,1999):
        sw=mult[k,inverse]
        for name,s in scores.items():
            ref=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
            err=max(err,float(abs(ref-boots[name][k]).max()))
    require(err<1e-12,'Independent bootstrap mismatch')
    metrics={n:{'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))} for n,s in scores.items()}
    for n in metrics:
        for j,m in enumerate(('AUROC','AP')):metrics[n][m+'_patient_cluster_95']=np.quantile(boots[n][:,j],[.025,.975]).tolist()
    contrasts={}
    for title in v:
        pairs=[('FEATURE_DP2','FEATURE_DP1')]
        pairs += [(a,b) for a in ('FEATURE_DP1','FEATURE_DP2') for b in ('FEATURE_PUBLIC','DINO_DP1','DINO_DP2')]
        for a,b in pairs:
            an=title+'_'+a;bn=title+'_'+b
            contrasts[an+'_minus_'+b]={m:{'delta':metrics[an][m]-metrics[bn][m],
                'patient_cluster_95':np.quantile((boots[an]-boots[bn])[:,j],[.025,.975]).tolist()} for j,m in enumerate(('AUROC','AP'))}
    descriptive={}
    for title in v:
        for family in ('FEATURE','DINO'):
            a=np.array([[metrics[title+'_'+family+'_DP'+str(i)][m] for m in ('AUROC','AP')] for i in (1,2)])
            descriptive[title+'_'+family]={'metric_order':['AUROC','AP'],'mean':a.mean(0).tolist(),
                'sample_sd':a.std(0,ddof=1).tolist(),'min':a.min(0).tolist(),'max':a.max(0).tolist(),
                'population_variance_claimed':False}
    names=list(scores)
    np.savez_compressed(output/'predictions_private.npz',names=np.array(names),scores=np.column_stack([scores[n] for n in names]),
        labels=y,image_ids=v['DenseNet']['image_ids'],patient_ids=v['DenseNet']['patient_ids'],**{n+'_weight':w for n,w in weights.items()})
    np.savez_compressed(output/'paired_bootstrap.npz',names=np.array(names),metrics=np.stack([boots[n] for n in names],1),
        original_patient_draws_sha256=np.array(sha(e.BOOT)))
    result={'status':'COMPLETE_FEATURE_PUBLIC_AND_NOISE_REPEAT','finished':now(),'metrics':metrics,'comparisons':contrasts,
        'descriptive_two_noise':descriptive,'primary':'DenseNet fixed-bank Q increment and independent feature-noise repetition',
        'secondary':'Previously declared ResNet18 development receiver',
        'inference_limit':'Conditional patient intervals for fixed banks/noise; not general noninferiority, DP interaction or CVPR contribution',
        'new_private_releases':1,'new_training_banks':2,'new_updates':400,'new_Q_V_pixel_forwards':0,
        'PNG_forwards':{'DenseNet':256,'ResNet18':256},'PNG_seconds':times,'readout_details':solve_details,
        'bootstrap_sklearn_error':err,'raw_pixel_access':access.report(),'evaluation_seconds':time.monotonic()-tick,
        'automatic_followup':False,'Expert_Reserved_final_receiver':False,'existing_feature_DP1_reused':True}
    on_time();save(output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

