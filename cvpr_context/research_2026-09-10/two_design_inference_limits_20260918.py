"""Counterexamples to two design-ranking inferences; no medical data/models."""
import json
from pathlib import Path
import numpy as np

# Cached condition objective can improve while correct-condition prediction stays fixed.
alpha, margin, ridge = .7, 1., .01
cc=[]
for w in (0.,2.):
    dy, dn=0., w*w
    cc.append({"w":w,"correct_condition_error":dy,"wrong_condition_error":dn,
               "objective_with_anchor":float(dy+alpha*np.logaddexp(0,margin+dy-dn)+ridge*w*w)})
assert cc[1]["objective_with_anchor"] < cc[0]["objective_with_anchor"]
assert cc[1]["correct_condition_error"] == cc[0]["correct_condition_error"]

# E1 sees x[0]; E2 sees x[1]. Images are illustrative 2D inputs.
real=np.array([[-.5,-.5],[.5,.5],[-.25,-.25],[.25,.25]])
synthetic=real*np.array([1.,-1.])
y=np.array([0,1,0,1])
def stats(z):
    m0,m1=z[y==0].mean(),z[y==1].mean()
    a0,a1=(z[y==0]**2).mean(),(z[y==1]**2).mean()
    delta=(z[1::2]-z[::2])/2
    return np.array([m0,m1,a0,a1,delta.mean(),(delta**2).mean()])
def solve(s,beta=1.,lam=.1):
    m0,m1,a0,a1,dm,C=s
    return ((m1-m0)/2+beta*dm)/((a0+a1)/2+beta*C+lam)
def auc(scores):
    dif=scores[y==1,None]-scores[y==0][None,:]
    return float(np.mean((dif>0)+.5*(dif==0)))
r1,s1=stats(real[:,0]),stats(synthetic[:,0])
assert np.array_equal(r1,s1)
r2,s2=stats(real[:,1]),stats(synthetic[:,1])
real_auc=auc(solve(r2)*real[:,1])
synth_auc=auc(solve(s2)*real[:,1])
assert real_auc==1. and synth_auc==0.
output={
 "scope":"Two mathematical counterexamples, not patient efficacy estimates",
 "cached_condition":{"examples":cc,"conclusion":"Objective decrease does not imply correct-condition output improvement"},
 "relational_distillation":{"E1_moment_max_difference":float(np.max(np.abs(r1-s1))),
    "E2_real_readout_auc_on_toy_real":real_auc,
    "E2_synthetic_readout_auc_on_toy_real":synth_auc,
    "conclusion":"Exact E1 point and relation moments do not imply E2 transfer"},
 "interpretation":"Neither exact caching nor exact surrogate preservation establishes medical utility or ranks success probabilities."
}
p=Path(__file__).resolve().parent/"spec_sources/two_design_inference_limits_20260918.json"
assert not p.exists(),p
p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(output,ensure_ascii=False,indent=2))

