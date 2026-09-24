"""Two fixed source-only label baselines; only existing synthetic PNG forwards."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet,moments
from fixed_image_compilation import pw

def run(root,out):
    start=time.monotonic();sys.path.insert(0,str(root));r=root/'_reports'
    import torch
    from PIL import Image
    from prrd_v3.encoders import pil_tensor
    from receiver_distillation.second_source import DinoEncoder,WEIGHTS
    from receiver_distillation.signals import conditions
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ps=packet(r/'receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz')
    assert set(ps['roles'].tolist())=={'P'}
    source_data={**ps,'z':ps['z'].transpose(1,0,2).reshape(-1,64).astype(np.float64)}
    xp=source_data['z'];C,bp=moments(source_data,True);wpub=pw(source_data)
    models=DinoEncoder().use_projection(r/'receiver_a1a2_s200_20260922_v1/second_targets/public/dinov2_P_projection.npz')
    paths={
        'DP1':('receiver_feature_dp8_20260924_v1/protected_png','receiver_feature_dp8_20260924_v1/one_release/protected_target'),
        'DP2':('receiver_feature_followup_20260924_v1/protected_png','receiver_feature_followup_20260924_v1/one_release/protected_target')}
    extracted={};targets={}
    for draw,(folder,targ) in paths.items():
        bank=r/folder
        rule=json.loads((r/targ/'rule.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==rule['checkpoint_sha256']
        with (bank/'images.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==128 and [int(row['label']) for row in rows]==[0]*64+[1]*64
        parts=[]
        with torch.inference_mode():
            for i in range(0,128,16):
                ims=[]
                for row in rows[i:i+16]:
                    path=(bank/row['file']).resolve()
                    assert path.is_relative_to(bank.resolve())
                    with Image.open(path) as im:ims.append(pil_tensor(im.convert('L')))
                x=torch.stack(ims).cuda()
                z=torch.stack([models(c.apply(x)) for c in conditions()])
                parts.append(z.permute(1,0,2).flatten(1).cpu().numpy())
        extracted[draw]=np.concatenate(parts).astype(np.float64)
        targets[draw]=packet(r/targ/'target.npz')['target'].reshape(2,64)
        print(json.dumps({'phase':'PNG_DINO_ONLY','draw':draw,'seconds':time.monotonic()-start}),flush=True)
    del models;torch.cuda.empty_cache()
    np.savez_compressed(out/'existing_png_dino_K4_features.npz',**extracted)
    val={'DenseNet':packet(r/'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz'),
         'ResNet18':packet(r/'receiver_resnet18_reuse_20260924_v1/V_features.npz')}
    original=np.array([-1.]*64+[1.]*64)
    results={};scores={};labels={}
    for draw in ('DP1','DP2'):
        xs=extracted[draw];target=targets[draw]
        wt=np.linalg.solve(C+.1*np.eye(64),.5*(target[1]-target[0]))
        teacher_public=xp@wt
        op=np.linalg.solve(xs.T@xs/128+.1*np.eye(64),xs.T/128);A=xp@op
        lam=.001*np.sum(wpub[:,None]*A*A)/128
        design=np.vstack([np.sqrt(wpub[:,None])*A,np.sqrt(lam)*np.eye(128)])
        response=np.concatenate([np.sqrt(wpub)*teacher_public,np.sqrt(lam)*original])
        fit=lsq_linear(design,response,bounds=(-1,1),tol=1e-12,max_iter=500)
        assert fit.success
        labelsets={'DIRECT_SOURCE_PSEUDOLABEL':np.clip(xs@wt,-1,1),'SOURCE_LABEL_SOLVE':fit.x}
        for name,l in labelsets.items():
            key=draw+'_'+name;labels[key]=l
            results[key]={'source_public_relative_RMSE':float(np.sqrt(np.sum(wpub*(A@l-teacher_public)**2)/np.sum(wpub*teacher_public**2))),
                          'label_min':float(l.min()),'label_max':float(l.max()),'receivers':{}}
            for rec in val:
                f=r/'receiver_feature_dp8_20260924_v1/evaluation'/(rec+'_PNG_features.npz') if draw=='DP1' else r/'receiver_feature_followup_20260924_v1/evaluation'/(rec+'_FEATURE_DP2_PNG_features.npz')
                zr=packet(f)['z'].astype(np.float64)
                wr=np.linalg.solve(zr.T@zr/128+.1*np.eye(128),zr.T@l/128)
                v=val[rec];s=v['z'].astype(np.float64)@wr;scores[key+'_eval_'+rec]=s
                results[key]['receivers'][rec]={'AUROC':float(roc_auc_score(v['labels'],s)),'AP':float(average_precision_score(v['labels'],s))}
    np.savez_compressed(out/'source_label_baseline_soft_labels.npz',**labels)
    np.savez_compressed(out/'source_label_baseline_predictions_private.npz',names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),
        labels=val['DenseNet']['labels'],patient_ids=val['DenseNet']['patient_ids'],image_ids=val['DenseNet']['image_ids'])
    result={'scope':'Two source-only label baselines; exploratory, no raw Q, no new release or pixel updates.',
      'DINO_synthetic_condition_forwards':1024,'real_image_forwards':0,'new_Q_access':0,'new_releases':0,'Expert_Reserved':False,
      'results':results,'seconds':time.monotonic()-start,
      'plan_sha256':hashlib.sha256((out/'SOURCE_LABEL_BASELINE_PLAN.md').read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'source_label_baseline_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

