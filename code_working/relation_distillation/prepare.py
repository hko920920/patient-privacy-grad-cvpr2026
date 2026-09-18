"""Bind a prospective design using existing metadata and public-weight bytes only.

This command has no image decoder, encoder construction, training or DP release.
It never reads final/reserved rosters or pixels. Original role files are immutable.
"""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

CODE = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = CODE / '_reports/relation_distillation_plan_20260918_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def cohort(rows):
    patients = defaultdict(set)
    for r in rows:
        patients[r['patient_id']].add(int(r['label']))
    return {'patients':len(patients), 'images':len(rows),
            'positive_patients':sum(1 in labels for labels in patients.values()),
            'mixed':sum(labels == {0,1} for labels in patients.values())}


def ranked(items, *salt):
    prefix = '|'.join(map(str, salt)) + '|'
    return sorted(items, key=lambda x: hashlib.sha256((prefix+str(x)).encode()).hexdigest())


def save_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    settings_path = PACKAGE/'settings_v1.json'
    cfg = json.loads(settings_path.read_text(encoding='utf-8'))
    if OUT.exists():
        raise RuntimeError('Preparation output already exists; no overwrite')
    prior_path = CODE/cfg['data']['previous_plan']
    prior = json.loads(prior_path.read_text(encoding='utf-8-sig'))
    paths = {name:CODE/cfg['data'][name] for name in ('training_manifest','development_manifest','bootstrap')}
    bindings = {}
    for name, path in paths.items():
        digest = sha(path)
        expected = prior['source_sha256'].get(str(path))
        if expected != digest:
            raise RuntimeError('Changed original data binding: '+name)
        bindings[str(path)] = digest
    training = load_rows(paths['training_manifest'])
    dev = load_rows(paths['development_manifest'])
    public = [r for r in training if r['planned_namespace']==cfg['data']['public_namespace']]
    private = [r for r in training if r['planned_namespace']==cfg['data']['private_namespace']]
    if len(public)+len(private)!=len(training):
        raise RuntimeError('Unexpected training source')
    counts = {'public':cohort(public),'private':cohort(private),'development':cohort(dev)}
    for name in counts:
        for key, expected in cfg['data'][name+'_expected'].items():
            if counts[name][key] != expected:
                raise RuntimeError('Cohort count mismatch: '+name+'/'+key)
    if {r['original_thesis_role'] for r in private} != {'private_train'}:
        raise RuntimeError('Private origin changed')
    for key in ('patient_id','image_id','sha256'):
        sets = [set(r[key] for r in rows) for rows in (public,private,dev)]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
            raise RuntimeError('Cohort overlap in '+key)
    for rows in (training,dev):
        if len({r['image_id'] for r in rows})!=len(rows):
            raise RuntimeError('Duplicate physical image in eligible pool')

    weight_root=Path.home()/'.cache/torch/hub/checkpoints'
    for encoder in cfg['encoders'].values():
        path=weight_root/encoder['checkpoint']
        digest=sha(path)
        if digest != encoder['sha256']:
            raise RuntimeError('Public weight hash mismatch')
        bindings[str(path)]=digest
    for path in sorted(PACKAGE.glob('*.py'))+[settings_path,prior_path]:
        bindings[str(path)]=sha(path)

    by_patient=defaultdict(list)
    private_labels=defaultdict(set)
    for r in public:
        by_patient[r['patient_id']].append(r)
    for r in private:
        private_labels[r['patient_id']].add(int(r['label']))
    mixed=sorted(p for p,y in private_labels.items() if y=={0,1})
    rng=np.random.default_rng(cfg['bank']['permutation_seed'])
    for _ in range(10000):
        permutation=rng.permutation(len(mixed))
        if np.all(permutation!=np.arange(len(mixed))):
            break
    else:
        raise RuntimeError('Could not construct fixed derangement')
    pair_mapping=[{'positive_patient':p,'negative_patient':mixed[int(j)]}
                  for p,j in zip(mixed,permutation)]
    init=[]
    for seed in cfg['bank']['seeds']:
        patient_order=ranked(by_patient,'relation-init-v1',seed)[:64]
        for cell,pid in enumerate(patient_order):
            candidates={r['image_id']:r for r in by_patient[pid]}
            row=candidates[ranked(candidates,'relation-init-image-v1',seed,pid)[0]]
            for label in (0,1):
                init.append({'bank_seed':seed,'cell':cell+64*label,'requested_label':label,
                             'public_patient_id':pid,'image_id':row['image_id'],
                             'source_path':row['path'],'source_sha256':row['sha256']})
    OUT.mkdir(parents=True)
    save_json(OUT/'initialization_private.json',init)
    save_json(OUT/'shuffle_private.json',pair_mapping)
    bindings[str(OUT/'initialization_private.json')]=sha(OUT/'initialization_private.json')
    bindings[str(OUT/'shuffle_private.json')]=sha(OUT/'shuffle_private.json')
    plan={'schema':'relation-distillation-preparation-v1','status':'settings_and_core_prepared_runtime_pending',
          'settings':cfg,'counts':counts,'bindings':bindings,
          'environment':{n:importlib.metadata.version(n) for n in ['torch','torchvision','timm','numpy','scipy']},
          'prepared_bank_runs':12,'prepared_unique_private_pair_mapping':len(pair_mapping),
          'prepared_initialization_rows':len(init),'new_image_decodes':0,'new_encoder_forward':0,
          'new_training_updates':0,'new_synthetic_images':0,'new_dp_release':0,
          'original_roles_changed':False,'new_final_or_reserved_metadata_read':False,
          'runtime_ready':False,'final_ready':False,
          'next_runtime_work':['public source-to-tensor binding','DINO local-load/dynamic-resolution/input-gradient parity',
             'public-only PCA and immutable feature cache','pyramid renderer and two-pass optimizer',
             'all-bank seal before development readouts','legacy-kernel BCE adapter','resume and artifact verification'],
          'cost_workload':{'training_images':5910,'development_images':5047,'main_banks':12,
             'updates_per_bank':500,'bank_images':128,'planned_BCE_runs':36,
             'main_wall_time':'not measured; public-only profile required, include both encoder passes and IO'},
          'limitations':['Prospective fixed recipe; no efficacy evidence.',
             'DINOv2 is a public LVD-142M release, not certified patient-level NIH-free.',
             'E2 is excluded from distillation and selection in this branch, not historically unused.',
             'Standard delta moment matching is the relation operation, not a distinct novel primitive.',
             'Prior nonDP development and metadata disclosures are not retroactively private.']}
    save_json(OUT/'plan.json',plan)
    print(json.dumps({'output':str(OUT),'counts':counts,'bank_runs':12,'runtime_ready':False,
                      'new_model_work':0,'plan_sha256':sha(OUT/'plan.json')},ensure_ascii=True))


if __name__=='__main__':
    main()
