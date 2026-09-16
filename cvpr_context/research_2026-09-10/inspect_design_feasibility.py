"""Read existing metadata and write aggregate design feasibility; no image/model access."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent.parent
SOURCE = THESIS / 'code_working/_data/derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv'


def main():
    grouped = defaultdict(list)
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            grouped[row['partition']].append(int(row['pa_image_count']))
    counts = {
        key: {
            'patients': len(values),
            'manifest_pa_images_before_cap': sum(values),
            'at_least_images': {str(k): sum(v >= k for v in values) for k in [1, 2, 4, 5, 7, 10, 12]},
        }
        for key, values in sorted(grouped.items())
    }
    spec = json.loads((ROOT / 'experiment_design.json').read_text(encoding='utf-8'))
    c = spec['candidate']
    optimization = 2 * c['steps'] * len(c['support_timesteps']) * c['support_noise_per_timestep']
    query = 2 * 2 * len(c['query_timesteps']) * c['query_noise_per_timestep']
    total_forward = optimization + query * (1 + c['matched_nonmember_controls_per_query_image'])
    assert total_forward == spec['cost_before_caching']['forward_per_observed_image']
    assert optimization == spec['cost_before_caching']['backward_per_observed_image']
    assert counts['public_development']['at_least_images']['2'] == 886
    assert counts['public_development']['at_least_images']['5'] == 308
    result = {
        'schema': 'patient-audit-design-feasibility/v0.1',
        'date': '2026-09-10',
        'scope': 'AGGREGATE_METADATA_AND_PROPOSED_COST_ARITHMETIC_ONLY',
        'source_relative_to_thesis': SOURCE.relative_to(THESIS).as_posix(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'patient_identifiers_emitted': False,
        'image_files_opened': False,
        'training_or_membership_experiment_run': False,
        'partitions': counts,
        'uncached_cost_per_observed_image': {'forward': total_forward, 'backward': optimization},
        'uncached_cost_by_m': {
            str(m): {'forward': total_forward * m, 'backward': optimization * m}
            for m in [2, 5]
        },
        'limitations': [
            'Metadata counts precede file-access, duplicate and scenario-eligibility checks.',
            'No patient split was selected or changed.',
            'Cost is model-example evaluations, not batched API calls or elapsed time.',
            'No medical attack performance or novelty validated.',
        ],
    }
    (ROOT / 'design_feasibility.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
