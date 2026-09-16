"""Plot verified, fixed endpoint contrasts; no fitting or score selection."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/masked_slot_endpoints_v2'
OUT = ROOT / 'measurement_audit_artifacts'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    verification = json.loads((SOURCE / 'verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'PASS_MASKED_ENDPOINT_SAVED_ARITHMETIC_SOURCE_REUSE_AND_TRAINING_GATES'
    source = SOURCE / 'analysis_v1/analysis.json'
    assert digest(source) == verification['analysis_sha256']
    rows = json.loads(source.read_text(encoding='utf-8'))['patient_scores']
    configs = [('generic_t140', 'negative_MSE_FP32', 'mean_8', 1000., 'Generic denoising loss: 8 fixed noise draws', 'Change in negative MSE (x 0.001)'),
               ('cdi_dl_t100', 'negative_L2_FP32', 'mean_5', 1., 'CDI denoising loss: 5 fixed noise draws', 'Change in negative L2 loss')]
    OUT.mkdir(exist_ok=True)
    targets = [OUT / ('masked_contribution_effects' + ext) for ext in ('.png', '.svg', '.csv', '.json')]
    # Presentation artifacts may be regenerated; verified source data stay immutable.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)
    used = []
    for ax, (family, metric, setting, factor, title, xlabel) in zip(axes, configs):
        for y, scenario in enumerate(['U', 'E']):
            selected = [r for r in rows if (r['family'], r['metric'], r['setting'], r['scenario'], r['pool']) == (family, metric, setting, scenario, 'mean')]
            assert len(selected) == 41
            refs = sorted([r for r in selected if r['eval_role'] == 'selection'], key=lambda r: int(r['patient_id']))
            case = next(r for r in selected if r['patient_id'] == '14393')
            assert len(refs) == 40
            ax.scatter([r['delta'] * factor for r in refs], y + np.linspace(-.14, .14, 40), s=23, color='#55758a', alpha=.75, label='Other 40 patients' if y == 0 else None)
            ax.scatter([case['delta'] * factor], [y], s=180, marker='*', color='#c75b18', edgecolor='white', linewidth=.7, zorder=4, label='Patient whose contributions changed' if y == 0 else None)
            used.extend(selected)
        ax.axvline(0, color='#888888', linewidth=.8, linestyle='--')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_yticks([0, 1], ['U: held-out photos', 'E: candidate photos'])
        ax.set_ylim(-.4, 1.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .09), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Adding one patient\'s eight training contributions also changes other patients\' scores', fontsize=12, y=.96)
    fig.text(.5, .035, 'Treatment minus masked control; fixed patient-mean scores. One patient and one training path, not population attack performance.', ha='center', fontsize=8)
    fig.subplots_adjust(top=.80, bottom=.30, left=.13, right=.98, wspace=.28)
    fig.savefig(targets[0], dpi=180)
    fig.savefig(targets[1])
    plt.close(fig)
    fields = ['family', 'metric', 'setting', 'scenario', 'pool', 'patient_id', 'eval_role', 'assignment_group', 'treatment', 'control', 'delta']
    with targets[2].open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(used)
    targets[3].write_text(json.dumps(dict(source=str(source), source_sha256=digest(source), verification_sha256=digest(SOURCE/'verification.json'), code_sha256=digest(Path(__file__)), rows=len(used), outputs={p.name:digest(p) for p in targets[:3]}, visual_y_jitter='Deterministic patient-id order for visibility only; no analysis operation'), indent=2), encoding='utf-8')
    print(json.dumps(dict(rows=len(used), output=str(targets[0]))))


if __name__ == '__main__':
    main()
