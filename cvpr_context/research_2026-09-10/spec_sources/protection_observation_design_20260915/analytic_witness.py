"""Analytic clipping-order witness; not a trained-model experiment."""
from pathlib import Path
import json, math, hashlib
from statistics import NormalDist
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
normal = NormalDist()
C, sigma, alpha = 1.0, 1.0, 0.01
gradients = np.array([[4.0, 0.0], [0.0, 1.0]])
mean_gradient = gradients.mean(axis=0)
def clip(v):
    return v * min(1.0, C / max(float(np.linalg.norm(v)), 1e-300))
A = np.mean([clip(g) for g in gradients], axis=0)
B = clip(mean_gradient)
A_equal_norm = C*A/np.linalg.norm(A)
queries = {"E_mean": mean_gradient, "U": np.array([0.0, 1.0])}
vectors = {"clip_then_mean": A, "mean_then_clip": B,
           "normalized_A_auxiliary_control": A_equal_norm}
rows = []
for q in [1.0, 0.2, 0.0]:
    for name, v in vectors.items():
        for query_name, h in queries.items():
            shift = float(h@v/(sigma*C*np.linalg.norm(h)))
            tpr = (1-q)*alpha+q*normal.cdf(shift-normal.inv_cdf(1-alpha))
            independent = (1-q)*alpha+q*norm.sf(norm.isf(alpha)-shift)
            assert abs(tpr-independent)<1e-13
            rows.append(dict(q=q,method=name,query=query_name,
                             standardized_mean_shift=shift,
                             tpr_at_1_percent_fpr=tpr))
assert np.allclose(A,[0.5,0.5])
assert np.allclose(B,np.array([4,1])/math.sqrt(17))
for q in [1.0, 0.2]:
    table={(r["method"],r["query"]):r["tpr_at_1_percent_fpr"]
           for r in rows if r["q"]==q}
    assert table["mean_then_clip","E_mean"]>table["clip_then_mean","E_mean"]
    assert table["clip_then_mean","U"]>table["mean_then_clip","U"]
assert all(abs(r["tpr_at_1_percent_fpr"]-alpha)<1e-15 for r in rows if r["q"]==0)
assert all(np.linalg.norm(v)<=C+1e-14 for v in vectors.values())
# Controls where the direction difference vanishes.
equal_norm_gradients = np.array([[1.,0.],[0.,1.]])
no_clip_gradients = gradients/10.
for gs in [equal_norm_gradients, no_clip_gradients]:
    assert np.allclose(np.mean([clip(g) for g in gs],axis=0),clip(gs.mean(axis=0)))
# Zero-moment/no-noise first Adam update: signs almost remove the magnitude distinction.
adam_eps=1e-8
adam_A=A/(np.abs(A)+adam_eps)
adam_B=B/(np.abs(B)+adam_eps)
report=dict(
  scope="Exact one-step bounded patient contribution plus isotropic Gaussian release and fixed linear score; no medical model inference or training",
  root_gradients=gradients.tolist(),clip_norm=C,sigma=sigma,fpr=alpha,
  vectors={k:v.tolist() for k,v in vectors.items()},rows=rows,
  noise_free_zero_moment_first_Adam={"A":adam_A.tolist(),"B":adam_B.tolist(),
    "max_difference":float(np.max(np.abs(adam_A-adam_B))),
    "interpretation":"SGD gradient-direction witness is not an automatic AdamW prediction"},
  controls_passed=["equal_norm_no_direction_change","clipping_inactive_equality","q_zero_no_signal"],
  independent_cdf_check="statistics.NormalDist vs scipy.stats.norm, tolerance 1e-13",
  dp_scope="Both bounded patient contributions admit the same add/remove Gaussian upper bound under a fixed public normalizer. This does not identify equal actual leakage or the ELS accountant.",
  not_claimed=["new DP algorithm","new theorem over prior literature","all-attack ranking reversal","diffusion efficacy","patient-DP violation"],
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/"analytic_witness.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
ax=axes[0]
for name,v,color in [("clip then mean",A,"#1f77b4"),("mean then clip",B,"#d95f02")]:
    ax.arrow(0,0,*v,color=color,width=.011,length_includes_head=True,head_width=.065,label=name)
for label,h in queries.items():
    h=.92*h/np.linalg.norm(h)
    ax.plot([0,h[0]],[0,h[1]],"--",color="#777777",linewidth=1)
    ax.text(h[0]+.02,h[1]+.03,label)
ax.set(xlim=(-.06,1.15),ylim=(-.06,1.1),xlabel="Coordinate 1",ylabel="Coordinate 2",
       title="Clipping order changes direction")
ax.set_aspect("equal")
ax.legend(loc="upper right",fontsize=8)
ax=axes[1]
x=np.arange(2);w=.34
for i,(method,label,color) in enumerate([
    ("clip_then_mean","clip then mean","#1f77b4"),
    ("mean_then_clip","mean then clip","#d95f02")]):
    ys=[100*next(r["tpr_at_1_percent_fpr"] for r in rows
        if r["q"]==1 and r["method"]==method and r["query"]==k) for k in queries]
    ax.bar(x+(i-.5)*w,ys,w,label=label,color=color)
ax.axhline(1,color="#777777",linestyle=":",linewidth=1)
ax.set_xticks(x,["E mean query","U query"])
ax.set(ylabel="Analytic TPR at 1% FPR (%)",title="Opposite order under two observations")
ax.legend(fontsize=8)
fig.suptitle("Analytic illustration only: same bound C and Gaussian noise, q = 1",fontsize=11)
fig.tight_layout()
fig.savefig(ROOT/"analytic_witness.png",dpi=160)
fig.savefig(ROOT/"analytic_witness.svg")
plt.close(fig)
print(json.dumps({"status":"PASS_ANALYTIC_EXAMPLE","rows":len(rows),"data_or_GPU_experiments":0,
 "q1_main":[r for r in rows if r["q"]==1 and r["method"]!="normalized_A_auxiliary_control"],
 "adam_max_difference":report["noise_free_zero_moment_first_Adam"]["max_difference"]},indent=2))

