"""Exploratory fixed-bank, existing-V algebraic intervention. No new encoder or private query."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet,moments

def run(root,out):
    r=root/'_reports'
    config={
      'DenseNet':{
       'public':'prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz',
       'V':'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz',
       'FEATURE_PUBLIC':'receiver_feature_followup_20260924_v1/evaluation/DenseNet_FEATURE_PUBLIC_PNG_features.npz',
       'FEATURE1':'receiver_feature_dp8_20260924_v1/evaluation/DenseNet_PNG_features.npz',
       'FEATURE2':'receiver_feature_followup_20260924_v1/evaluation/DenseNet_FEATURE_DP2_PNG_features.npz',
       'GRAD1':'receiver_dino_dp8_20260924_v1/evaluation/DenseNet_DINO_DP_PNG_features.npz',
       'GRAD2':'receiver_dino_dp8_noise_repeat_20260924_v1/evaluation/DenseNet_DINO_DP2_PNG_features.npz',
       'A21':'receiver_dp8_first_20260923_v1/evaluation/DenseNet_DP8_PNG_features.npz',
       'A22':'receiver_dp8_noise_repeat_20260923_v1/evaluation/DenseNet_DP2_PNG_features.npz'},
      'ResNet18':{
       'public':'receiver_resnet18_reuse_20260924_v1/P_projected_features.npz',
       'V':'receiver_resnet18_reuse_20260924_v1/V_features.npz',
       'FEATURE_PUBLIC':'receiver_feature_followup_20260924_v1/evaluation/ResNet18_FEATURE_PUBLIC_PNG_features.npz',
       'FEATURE1':'receiver_feature_dp8_20260924_v1/evaluation/ResNet18_PNG_features.npz',
       'FEATURE2':'receiver_feature_followup_20260924_v1/evaluation/ResNet18_FEATURE_DP2_PNG_features.npz',
       'GRAD1':'receiver_resnet18_reuse_20260924_v1/DINO_DP1_features.npz',
       'GRAD2':'receiver_resnet18_reuse_20260924_v1/DINO_DP2_features.npz',
       'A21':'receiver_resnet18_reuse_20260924_v1/A2_DP1_features.npz',
       'A22':'receiver_resnet18_reuse_20260924_v1/A2_DP2_features.npz'}
    }
    result={'scope':'EXPLORATORY_FIXED_BANK_EXISTING_V_DIAGNOSTIC',
            'private_queries':0,'new_synthesis':0,'new_encoder_forwards':0,
            'new_receivers':0,'Reserved_Expert':False,'ridge':.1,'receivers':{},'input_hashes':{}}
    for receiver,c in config.items():
        data={}
        for name,path in c.items():
            f=r/path;data[name]=packet(f);result['input_hashes'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        assert set(data['public']['roles'].tolist())=={'P'}
        z=data['V']['z'].astype(np.float64);y=data['V']['labels']
        def evaluate(cov,mean):
            w=np.linalg.solve(cov+.1*np.eye(len(mean)),mean)
            score=z@w
            return {'AUROC':float(roc_auc_score(y,score)),'AP':float(average_precision_score(y,score))}
        stats={name:moments(d,patient=name=='public') for name,d in data.items() if name!='V'}
        cp,_=stats['public']
        rows={}
        for name,(cov,b) in stats.items():
            rows[name]={'original':evaluate(cov,b),'public_covariance':evaluate(cp,b)}
            rows[name]['change']={m:rows[name]['public_covariance'][m]-rows[name]['original'][m] for m in ('AUROC','AP')}
        crossed={}
        for method in ('FEATURE','GRAD','A2'):
            c1,b1=stats[method+'1'];c2,b2=stats[method+'2']
            cells={'mean1_cov1':evaluate(c1,b1),'mean1_cov2':evaluate(c2,b1),
                   'mean2_cov1':evaluate(c1,b2),'mean2_cov2':evaluate(c2,b2)}
            parts={}
            for m in ('AUROC','AP'):
                a=cells['mean1_cov1'][m];b=cells['mean1_cov2'][m];c_=cells['mean2_cov1'][m];d=cells['mean2_cov2'][m]
                mean=.5*((c_-a)+(d-b));cov=.5*((b-a)+(d-c_))
                assert abs((mean+cov)-(d-a))<1e-12
                parts[m]={'total_change':d-a,'mean_contribution':mean,'second_moment_contribution':cov}
            crossed[method]={'cells':cells,'decomposition':parts}
        result['receivers'][receiver]={'replacement':rows,'crossed':crossed}
    (out/'covariance_intervention_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['receivers'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

