"""Descriptive analysis of the fixed primary run and one post-hoc activation control."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent.parent
ROOT=RESEARCH.parent.parent/"code_working"
RUN=ROOT/"_reports/cvpr_u_pilot_v1_001/protection_observation_20260915"
names=["clipping_observation_local_v1","clipping_activation_control_v1"]
rows=[json.loads((RUN/n/"results.json").read_text()) for n in names]
reports=[json.loads((RUN/n/"report.json").read_text()) for n in names]
active=rows[1]
a=np.array([r["score_A_minus_B"] for r in active])
b=np.array([r["linear_prediction_A_minus_B"] for r in active])
relative=np.abs(a-b)/np.maximum(np.maximum(abs(a),abs(b)),1e-30)
cases=[dict(patient_id=r["patient_id"],step=r["step"],E=r["Emean_score_A_minus_B"],
            U=r["U_score_A_minus_B"],E_linear=r["Emean_linear_prediction"],U_linear=r["U_linear_prediction"])
       for r in active if r["E_U_raw_sign_opposition"]]
summary=dict(primary_inactive_conditions=sum(r["both_gradients_unclipped"] for r in rows[0]),
    primary_E_gradient_norm_min=min(v for r in rows[0] for v in r["gradient_norms"][:2]),
    primary_E_gradient_norm_max=max(v for r in rows[0] for v in r["gradient_norms"][:2]),
    activation_opposite_conditions=len(cases),activation_opposite_unique_patients=len({r["patient_id"] for r in cases}),
    cases=cases,linear_sign_agreement=int(((a>0)==(b>0)).sum()),linear_image_conditions=int(a.size),
    linear_global_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
    linear_median_relative_error=float(np.median(relative)),linear_max_abs_error=float(abs(a-b).max()),
    activation_Emean_sign_agreement=sum((r["Emean_score_A_minus_B"]>0)==(r["Emean_linear_prediction"]>0) for r in active),
    activation_U_sign_agreement=sum((r["U_score_A_minus_B"]>0)==(r["U_linear_prediction"]>0) for r in active),
    total_forward=sum(r["forward"] for r in reports),total_backward=sum(r["backward"] for r in reports),
    gpu_runner_seconds=sum(r["elapsed_seconds"] for r in reports),
    scope="descriptive local responses, no independent-patient sample of16, no MIA success threshold",
    primary_C=.28448700606156724,activation_C=.01,activation_posthoc=True)
(HERE/"analysis.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
with (HERE/"all_conditions.csv").open("w",newline="",encoding="utf-8") as f:
    fields=["patient_id","step","Emean_score_A_minus_B","U_score_A_minus_B","Emean_linear_prediction",
        "U_linear_prediction","E_U_raw_sign_opposition","clipped_A_B_cosine","delta_A_B_norm"]
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(active)
plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
ax=axes[0]
for step,color in [(250,"#1565c0"),(1000,"#b45309")]:
    x=[r for r in rows[0] if r["step"]==step]
    ns=np.array([r["gradient_norms"][:2] for r in x])
    ax.scatter(np.repeat(np.arange(1,9),2)+(0 if step==250 else .12),ns.ravel(),s=28,color=color,label=f"state {step}")
ax.axhline(summary["primary_C"],color="#333",label="primary C = 0.2845")
ax.axhline(.01,color="#a11754",ls="--",label="activation control C = 0.01")
ax.set(xlabel="Fixed public patient order",ylabel="Per-image gradient norm",title="Primary: clipping inactive in all 16 conditions",ylim=(0,.32))
ax.legend(fontsize=8)
ax=axes[1]
for step,color,marker in [(250,"#1565c0","o"),(1000,"#b45309","s")]:
    rs=[r for r in active if r["step"]==step]
    ax.scatter([r["Emean_score_A_minus_B"]*1e6 for r in rs],
               [r["U_score_A_minus_B"]*1e6 for r in rs],color=color,marker=marker,label=f"state {step}",s=42)
ax.axhline(0,color="gray",lw=1);ax.axvline(0,color="gray",lw=1)
ax.set(xlabel="E mean: score A minus B (x 1e-6)",ylabel="U: score A minus B (x 1e-6)",
       title="Single post-hoc activation control C=0.01\n3 opposite-order conditions / 2 patients")
ax.legend(loc="lower right",fontsize=8)
fig.suptitle("Local clipping + learned Adam response; no DP noise or membership-performance test",fontsize=11)
fig.savefig(HERE/"local_results.png",dpi=180)
fig.savefig(HERE/"local_results.svg")
print(json.dumps(summary,ensure_ascii=False))
