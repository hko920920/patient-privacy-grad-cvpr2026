"""Independent saved-score/rank validation; no producer imports or new fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import RUN

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def same(a,b):assert np.allclose(a,b,rtol=0,atol=2e-14),(a,b)
def credit(a,b):return 2*int(a>b)+int(a==b)

def main():
    begun=time.perf_counter()
    base=RUN/'baseline_screen_20260914/cdi_u_cohort_v1'
    root=base/'rank_decomposition_v1'
    output=root/'independent_verification.json'
    assert not output.exists()
    protocol=read(root/'protocol.json')
    inputs=[root/n for n in ('protocol.json','summary.json','patients.json','pairs.json')]
    inputs += [base/'analysis_v1'/n for n in ('same_fitted_scorer_predictions.json','bootstrap_resamples.json','analysis.json')]
    before={str(p):sha(p) for p in inputs}
    source_sha=sha(__file__)
    with (root/'independent_verification_protocol.json').open('x',encoding='utf-8') as f:
        json.dump(dict(source_sha256=source_sha,input_sha256=before,no_GPU=True,no_fitting=True),f,indent=2)
    assert sha(protocol['source_path'])==protocol['source_sha256']
    for path,h in protocol['input_sha256'].items():assert sha(path)==h,path
    summary=read(root/'summary.json')
    for name in ('protocol','patients','pairs'):assert sha(root/(name+'.json'))==summary[name+'_sha256']
    original=read(base/'analysis_v1/same_fitted_scorer_predictions.json')
    assert len(original)==800
    groups={}
    for r in original:groups.setdefault((r['fit_target_scorer'],r['method']),{})[r['patient_id']]=r
    saved={ (r['fit_target_scorer'],r['method'],r['patient_id']):r for r in read(root/'patients.json') }
    pairs={(r['fit_target_scorer'],r['method']):r['pairs'] for r in read(root/'pairs.json')}
    summaries={(r['fit_target_scorer'],r['method']):r for r in summary['summaries']}
    assert len(groups)==len(pairs)==len(summaries)==20
    boot=read(base/'analysis_v1/bootstrap_resamples.json')
    order=boot['selection_patient_order'];ix=np.asarray(boot['indices'],dtype=int)
    assert ix.shape==(2000,40)
    derived={};seen_pairs=0
    for key,patients in groups.items():
        assert set(patients)==set(order)
        y=np.array([patients[p]['group']=='A' for p in order])
        assert y.sum()==20 and y[ix[:,:20]].all() and not y[ix[:,20:]].any()
        s1=np.array([patients[p]['score_model_1_features'] for p in order])
        s2=np.array([patients[p]['score_model_2_features'] for p in order])
        common=(s1+s2)/2;change=(s1-s2)/2
        for i,p in enumerate(order):
            row=saved[(*key,p)]
            assert row['group']==patients[p]['group']
            same([row['s1'],row['s2'],row['common'],row['change']],[s1[i],s2[i],common[i],change[i]])
        position={p:i for i,p in enumerate(order)}
        assert len(pairs[key])==400 and len({(r['patient_A'],r['patient_B']) for r in pairs[key]})==400
        sums=np.zeros(5,dtype=int);own_delta=[];target_delta=[]
        for r in pairs[key]:
            a,b=position[r['patient_A']],position[r['patient_B']]
            assert y[a] and not y[b]
            c1,c2,cc=credit(s1[a],s1[b]),credit(s2[a],s2[b]),credit(common[a],common[b])
            own,co=(c1,cc) if key[0]=='model_1' else (2-c2,2-cc)
            expected=[c1,c2,cc,own,co]
            assert [r[n] for n in ('s1_credit_twice','s2_credit_twice','common_yA_credit_twice','own_credit_twice','common_own_credit_twice')]==expected
            assert r['own_vs_common_delta_twice']==own-co
            same([r[n] for n in ('s1_margin','s2_margin','common_margin','change_margin')],[s1[a]-s1[b],s2[a]-s2[b],common[a]-common[b],change[a]-change[b]])
            sums+=expected;own_delta.append(own-co);target_delta.append(c1-c2);seen_pairs+=1
        ref=summaries[key]
        same([ref['AUC_s1_yA'],ref['AUC_s2_yA'],ref['AUC_common_yA']],[roc_auc_score(y,s1),roc_auc_score(y,s2),roc_auc_score(y,common)])
        same([ref['own_target_AUC'],ref['common_own_label_AUC'],ref['own_vs_common_AUC_contribution']],[sums[3]/800,sums[4]/800,(sums[3]-sums[4])/800])
        for name,delta in [('own_vs_common',own_delta),('s1_vs_s2_under_yA',target_delta)]:
            z=np.asarray(delta);record=ref[name]
            assert [record[n] for n in ('improved_pairs','worsened_pairs','same_credit_pairs','delta_twice_credit_sum')]==[int((z>0).sum()),int((z<0).sum()),int((z==0).sum()),int(z.sum())]
        def bootstrap(score):
            sampled=score[ix];a=sampled[:,:20,None];b=sampled[:,None,20:]
            auc=((a>b)+.5*(a==b)).mean(axis=(1,2))
            return auc if key[0]=='model_1' else 1-auc
        ownboot=bootstrap(s1 if key[0]=='model_1' else s2);commonboot=bootstrap(common)
        for field,values in [('own_target_AUC_CI95',ownboot),('common_own_label_AUC_CI95',commonboot),('own_vs_common_contribution_CI95',ownboot-commonboot)]:same(ref[field],np.quantile(values,[.025,.975]))
        derived[key]=(sums[3]/800,sums[4]/800,ownboot,commonboot)
    for r in summary['comparisons']:
        a=derived[(r['fit_target_scorer'],r['method_a'])];b=derived[(r['fit_target_scorer'],r['method_b'])]
        own=a[0]-b[0];common=a[1]-b[1]
        same([r['observed_own_AUC_gain'],r['common_component_AUC_gain'],r['difference_in_target_change_contribution']],[own,common,own-common])
        for field,arr in [('observed_gain_CI95',a[2]-b[2]),('common_gain_CI95',a[3]-b[3]),('response_contribution_CI95',(a[2]-a[3])-(b[2]-b[3]))]:same(r[field],np.quantile(arr,[.025,.975]))
    assert seen_pairs==8000 and sha(__file__)==source_sha
    for p,h in before.items():assert sha(p)==h,p
    result=dict(status='PASS_INDEPENDENT_SAVED_SCORE_AND_RANK_RECOMPUTATION',scorers=20,patients=800,pairs=8000,contrasts=4,bootstrap_resamples=2000,source_sha256=source_sha,input_sha256=before,seconds_cpu=time.perf_counter()-begun,new_GPU=0,new_fitting=False,identified_nuisance_or_causal_effect=False)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))

if __name__=='__main__':main()
