"""Render the fixed U-relative descriptive audit from saved scores; no model calls."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parents[1] / "code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914"

def main():
    source = RUN / "pfami_v1/analysis.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = [r for r in data["patient_scores"] if r["scenario"] == "U"]
    lookup = {(r["model"], r["patient_id"]): r for r in rows}
    order = data["bootstrap"]["patient_order"]
    x = np.array([lookup["model_1", p]["scores"]["relative"] for p in order])
    z = np.array([lookup["model_2", p]["scores"]["relative"] for p in order])
    group_a = np.array([lookup["model_1", p]["assignment_group"] == "A" for p in order])
    delta = np.where(group_a, x-z, z-x)
    assert len(x) == 40 and group_a.sum() == 20
    colors = {"A": "#187c86", "B": "#bf7214"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
        limits = [min(x.min(), z.min())-.004, max(x.max(), z.max())+.004]
        axes[0].plot(limits, limits, "--", color="#777777", linewidth=1, label="Identical scores")
        for group, mask in [("A", group_a), ("B", ~group_a)]:
            label = group + (" (member in model 1)" if group == "A" else " (member in model 2)")
            axes[0].scatter(x[mask], z[mask], color=colors[group], s=34, alpha=.85, label=label)
            axes[1].scatter(np.arange(40)[mask]+1, delta[mask], color=colors[group], s=34, alpha=.85)
        axes[0].set(xlim=limits, ylim=limits, xlabel="Model 1: relative score",
                    ylabel="Model 2: relative score", title="A. Scores for the same U patients")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].axhline(0, color="#555555", linewidth=1)
        axes[1].axvline(20.5, color="#bbbbbb", linestyle=":", linewidth=1)
        axes[1].set(xlabel="Patients: group A (1-20), group B (21-40)",
                    ylabel="Score in member model - score in nonmember model",
                    title="B. Participation-aligned change (diagnostic)")
        fig.suptitle("PFAMI-style U audit: 40 patients, two fixed jointly swapped models", fontsize=12)
        out = RUN / "paired_score_audit_20260914_v1"
        outputs = [out/"u_relative_pair_plot.png", out/"u_relative_pair_plot.svg"]
        assert all(not p.exists() for p in outputs)
        for p in outputs:
            fig.savefig(p, dpi=180, metadata={"Title": "PFAMI U paired-score descriptive audit"})
        plt.close(fig)
        manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        plot_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                        scope="Descriptive analyst-only comparison; no individual causal or attack-utility claim.")
        (out/"plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"outputs": [p.name for p in outputs]}))

if __name__ == "__main__":
    main()
