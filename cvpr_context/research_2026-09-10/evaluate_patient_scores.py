"""Common patient-score evaluation, not a new attack or a DP accountant.

CSV: patient_id,image_id,split,member,score. One fixed method/model per file.
Splits: development, calibration, test. No patient may cross splits.
Calibration uses nonmembers only. Fix feature/aggregation before calibration.
Conformal ranks control marginal error under exchangeable patient bags; they
do not certify conditional population FPR, DP, or performance under shift.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

def aggregate_rows(rows, direction, aggregation, top_k=3):
    if direction not in {'high','low'} or aggregation not in {'mean','max','median','topk'}:
        raise ValueError('Invalid direction or aggregation')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    bags=defaultdict(list)
    identities={}
    seen=set()
    for row in rows:
        pid=row['patient_id'].strip(); iid=row['image_id'].strip()
        split=row['split']; member=int(row['member']); score=float(row['score'])
        if not pid or not iid or split not in {'development','calibration','test'}:
            raise ValueError('Missing identity or invalid split')
        if member not in (0,1) or not np.isfinite(score):
            raise ValueError('Invalid label or nonfinite score')
        if pid in identities and identities[pid] != (split,member):
            raise ValueError('Patient crosses split or membership boundary: '+pid)
        identities[pid]=(split,member)
        if iid in seen:
            raise ValueError('Repeated image_id: '+iid)
        seen.add(iid)
        bags[pid].append(score if direction=='high' else -score)
    result=[]
    for pid,values in bags.items():
        x=np.array(values)
        score={'mean':np.mean,'max':np.max,'median':np.median}.get(aggregation)
        value=score(x) if score else np.sort(x)[-top_k:].mean()
        split,member=identities[pid]
        result.append(dict(patient_id=pid,split=split,member=member,images=len(x),score=float(value)))
    return result

def ranks(calibration, scores):
    cal=np.sort(np.asarray(calibration,dtype=float))
    if not len(cal):
        raise ValueError('No nonmember calibration patients')
    # >= handles ties conservatively; high scores mean membership.
    return (1+len(cal)-np.searchsorted(cal,np.asarray(scores),side='left'))/(len(cal)+1)

def binomial_interval(k,n,confidence=.95):
    if n==0:return None
    tail=(1-confidence)/2
    lo=0. if k==0 else beta.ppf(tail,k,n-k+1)
    hi=1. if k==n else beta.ppf(1-tail,k+1,n-k)
    return [float(lo),float(hi)]

def evaluate(bags, alpha=.01):
    if not 0 < alpha < 1:raise ValueError('alpha must lie in (0,1)')
    cal=[b['score'] for b in bags if b['split']=='calibration' and b['member']==0]
    test=[b for b in bags if b['split']=='test']
    y=np.array([b['member'] for b in test]); s=np.array([b['score'] for b in test])
    if set(y)!={0,1}:raise ValueError('Test requires member and nonmember patients')
    p=ranks(cal,s); decisions=p<=alpha
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
    n0=int((y==0).sum());n1=int((y==1).sum())
    fp=int(decisions[y==0].sum());tp=int(decisions[y==1].sum())
    rows=[{**b,'calibrated_p':float(v),'member_decision':bool(z)} for b,v,z in zip(test,p,decisions)]
    strata=[]
    for name,low,high in [('1',1,1),('2-5',2,5),('6-10',6,10),('11+',11,float('inf'))]:
        sub=[r for r in rows if low<=r['images']<=high]
        neg=[r for r in sub if not r['member']];pos=[r for r in sub if r['member']]
        k0=sum(r['member_decision'] for r in neg);k1=sum(r['member_decision'] for r in pos)
        strata.append(dict(images=name,nonmembers=len(neg),members=len(pos),
                           fpr=k0/len(neg) if neg else None,tpr=k1/len(pos) if pos else None,
                           fpr_ci95=binomial_interval(k0,len(neg)),tpr_ci95=binomial_interval(k1,len(pos))))
    return dict(member_positive_auc=float(roc_auc_score(y,s)),alpha=alpha,
                calibration_nonmembers=len(cal),calibration_rank_resolution=1/(len(cal)+1),
                test_nonmembers=n0,test_members=n1,frozen_rule_fp=fp,frozen_rule_tp=tp,
                frozen_rule_fpr=fp/n0,frozen_rule_tpr=tp/n1,
                fpr_ci95=binomial_interval(fp,n0),tpr_ci95=binomial_interval(tp,n1),
                descriptive_test_roc_tpr_at_fpr_le_1pct=float(tpr[fpr<=.01].max()),
                descriptive_test_roc_tpr_at_fpr_le_0_1pct=float(tpr[fpr<=.001].max()),
                calibration_claim='marginal_under_exchangeable_patient_bags_only; score_fixed_independently',
                intervals='two-sided Clopper-Pearson; assumes independent test patients; not simultaneous',
                strata=strata,patients=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--score-direction',choices=['high','low'],required=True)
    p.add_argument('--aggregation',choices=['mean','max','median','topk'],default='mean')
    p.add_argument('--top-k',type=int,default=3);p.add_argument('--alpha',type=float,default=.01)
    a=p.parse_args()
    with a.csv.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result=evaluate(aggregate_rows(rows,a.score_direction,a.aggregation,a.top_k),a.alpha)
    result.update(input_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  aggregation=a.aggregation,top_k=a.top_k,score_direction=a.score_direction)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'patients','strata'}}))

if __name__=='__main__':main()
