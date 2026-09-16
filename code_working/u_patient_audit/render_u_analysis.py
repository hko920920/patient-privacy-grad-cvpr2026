"""Render already saved U diagnostics with non-overlapping patient labels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import RUN, read_csv, write_json, digest
from .analyze_u_smoke import METRICS

def main():
    out=RUN/"analysis/U_step1000_n8_v1"
    data=json.loads((out/"analysis.json").read_text())
    rows=[r for r in read_csv(out/"paired_scores.csv") if r["metric"]=="response"]
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),constrained_layout=True)
    for i,(key,label) in enumerate(METRICS):
        for m,color,offset in [("model_1","#2466a0",-.12),("model_2","#cc6831",.12)]:
            axes[0].scatter(data["metrics"][key]["models"][m]["auc"],i+offset,color=color,
                            label=m.replace("_"," ") if i==0 else None,s=55)
    axes[0].set(yticks=range(len(METRICS)),yticklabels=[v for _,v in METRICS],xlim=(-.03,1.03),
                xlabel="AUROC (higher score = member)",title="4 members + 4 nonmembers in each model")
    axes[0].axvline(.5,color="#666666",linestyle="--",linewidth=1)
    axes[0].invert_yaxis();axes[0].legend(loc="lower right")
    last_label=None
    for r in sorted(rows,key=lambda r:float(r["member_score"])):
        y0,y1=float(r["nonmember_score"]),float(r["member_score"])
        color="#2466a0" if r["member_model"]=="model_1" else "#cc6831"
        axes[1].plot([0,1],[y0,y1],"o-",color=color,alpha=.8)
        label_y=y1 if last_label is None else max(y1,last_label+.00022)
        axes[1].annotate(r["patient"],xy=(1,y1),xytext=(1.09,label_y),fontsize=8,
                         va="center",arrowprops=dict(arrowstyle="-",color=color,lw=.7))
        last_label=label_y
    axes[1].set(xticks=[0,1],xticklabels=["Not included","Included"],xlim=(-.2,1.3),
                ylabel="Candidate response score",title="Same U photos; membership switched")
    axes[1].ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
    axes[1].text(.02,.02,"Line color: model that includes the patient",transform=axes[1].transAxes,fontsize=8)
    fig.suptitle("Existing U diagnostic: 8 unique patients, 2 paired models\n"
                 "Descriptive development results; no threshold fitting or new GPU execution",fontsize=13)
    fig.savefig(out/"u_eight_patient_diagnostic.png",dpi=180)
    fig.savefig(out/"u_eight_patient_diagnostic.pdf")
    plt.close(fig)
    write_json(out/"plot_manifest.json",dict(analysis_sha256=digest(out/"analysis.json"),
               render_code_sha256=digest(Path(__file__)),
               files={p.name:digest(p) for p in [out/"u_eight_patient_diagnostic.png",out/"u_eight_patient_diagnostic.pdf"]}))
    print("Diagnostic plot rendered from saved analysis")
if __name__=="__main__":main()

