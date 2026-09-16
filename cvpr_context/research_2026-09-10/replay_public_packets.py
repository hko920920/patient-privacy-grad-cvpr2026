"""CPU replay of official supplementary feature packets; no target-model training.

CLiD uses exact sklearn ROC instead of the author's coarse threshold grid.
Tracing uses the published linear classifier and optimizer settings on supplied tensors.
Neither experiment reproduces image-level feature extraction or a medical patient attack.
"""
from pathlib import Path
import io, json, hashlib, time, zipfile
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'replays'
OUT.mkdir(exist_ok=True)

def metrics(member,external):
    labels=np.r_[np.ones(len(member)),np.zeros(len(external))]
    scores=np.r_[member,external]
    fpr,tpr,_=roc_curve(labels,scores,drop_intermediate=False)
    return {'member_positive_auc':float(roc_auc_score(labels,scores)),
            'member_tpr_at_fpr_le_0.01':float(tpr[fpr<=0.01].max()),
            'member_tpr_at_fpr_le_0.001':float(tpr[fpr<=0.001].max()),
            'members':len(member),'nonmembers':len(external),
            'threshold_type':'descriptive_test_ROC_not_frozen_deployment_threshold'}

def clid(skip_first=True):
    base=ROOT/'code_sources'/'A12_supp'/'CLID_MIA'/'inter_output'/'CLID'
    def read(split,kind):
        match=[p for p in base.glob('*.txt') if ('split1' in p.name)==(split=='shadow') and ('TRTE_'+kind+'_') in p.name]
        assert len(match)==1,(split,kind,match)
        data=np.loadtxt(match[0],delimiter='\t',skiprows=int(skip_first))
        assert data.ndim==2 and np.isfinite(data).all()
        features=np.c_[data[:,0],-data[:,1:].mean(axis=1)]
        return features,{'name':match[0].name,'sha256':hashlib.sha256(match[0].read_bytes()).hexdigest(),'shape':list(data.shape)}
    sm,smmeta=read('shadow','train');sn,snmeta=read('shadow','test')
    tm,tmmeta=read('target','train');tn,tnmeta=read('target','test')
    scaler=RobustScaler().fit(np.r_[sm,sn])
    sm,sn,tm,tn=[scaler.transform(x) for x in (sm,sn,tm,tn)]
    choices=[]
    for alpha in np.arange(11)/10:
        weights=np.array([1-alpha,alpha])
        auc=metrics(-sm@weights,-sn@weights)['member_positive_auc']
        choices.append((auc,float(alpha)))
    # Stable tie break: retain first alpha, matching the strict > update in the source.
    best=max(range(len(choices)),key=lambda i:choices[i][0])
    auc,alpha=choices[best]
    weights=np.array([1-alpha,alpha])
    member,external=-tm@weights,-tn@weights
    result=metrics(member,external)
    # Original get_1_fpr treats non-members as positive. Preserve both orientations explicitly.
    labels=np.r_[np.zeros(len(member)),np.ones(len(external))]
    fpr,tpr,_=roc_curve(labels,np.r_[-member,-external])
    index=np.flatnonzero(fpr>=.01)[0]
    result.update(id='A12',mode='official_scores_independent_exact_ROC_replay',alpha_from_shadow=alpha,
                  shadow_auc=auc,source_files=[smmeta,snmeta,tmmeta,tnmeta],
                  author_helper_nonmember_positive_tpr_first_fpr_ge_0_01=float(tpr[index]),
                  skip_first_numeric_row_to_match_author_loader=skip_first,
                  image_feature_extraction_reproduced=False,medical_patient_attack=False)
    stem='A12' if skip_first else 'A12_all_rows'
    np.savez_compressed(OUT/(stem+'_scores.npz'),member=member,nonmember=external)
    (OUT/(stem+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

def tracing():
    import torch
    torch.set_num_threads(2)
    path=ROOT/'code_sources'/'X02_supplement.zip'
    data={}
    with zipfile.ZipFile(path) as archive:
        for split in ['train','eval']:
            for label in ['member','external']:
                name=f'data/cifar10/{split}/{label}.pt'
                tensor=torch.load(io.BytesIO(archive.read(name)),map_location='cpu',weights_only=True)
                assert isinstance(tensor,torch.Tensor) and tensor.ndim==2
                assert tensor.shape[1]==3000 and torch.isfinite(tensor).all()
                data[(split,label)]=tensor
    results=[]
    variants=[('loss_full',[0],1),('all_step3',[0,1,2],3),('all_full',[0,1,2],1)]
    for name,features,step in variants:
        start=time.monotonic()
        torch.manual_seed(0)
        columns=torch.cat([torch.arange(f*1000,(f+1)*1000,step) for f in features])
        xm,xn=[data[('train',label)][:,columns] for label in ['member','external']]
        x=torch.cat([xm,xn]);y=torch.cat([torch.zeros(len(xm)),torch.ones(len(xn))]).long()
        dataset=torch.utils.data.TensorDataset(x,y)
        loader=torch.utils.data.DataLoader(dataset,batch_size=50,shuffle=True,num_workers=0)
        layer=torch.nn.Linear(x.shape[1],1)
        var,mean=torch.var_mean(x,dim=0,keepdim=True)
        assert (var>0).all()
        std=var.sqrt()
        opt=torch.optim.AdamW(layer.parameters(),lr=1e-3,weight_decay=10)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=.8)
        for epoch in range(100):
            for bx,by in loader:
                opt.zero_grad()
                logits=layer((bx-mean)/std)
                loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros_like(logits)],dim=1),by)
                loss.backward();opt.step()
            scheduler.step()
        with torch.inference_mode():
            scores={label:layer((data[('eval',label)][:,columns]-mean)/std).flatten().numpy() for label in ['member','external']}
        result=metrics(scores['member'],scores['external'])
        result.update(variant=name,features=len(columns),epochs=100,seed=0,seconds=time.monotonic()-start,
                      member_positive_asr=float(((scores['member']>0).mean()+(scores['external']<=0).mean())/2))
        np.savez_compressed(OUT/('X02_'+name+'_scores.npz'),member=scores['member'],nonmember=scores['external'])
        results.append(result)
        print(json.dumps({'id':'X02',**result}),flush=True)
    report={'id':'X02','mode':'official_feature_packet_linear_probe_replay',
            'zip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'torch_version':torch.__version__,
            'device':'cpu','source_shapes':{str(k):list(v.shape) for k,v in data.items()},'variants':results,
            'image_feature_extraction_reproduced':False,'medical_patient_attack':False}
    (OUT/'X02.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    clid()
    tracing()
