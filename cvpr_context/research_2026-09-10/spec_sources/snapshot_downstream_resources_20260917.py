"""Metadata-only prospective downstream inventory; no images or model scores read."""
import csv, hashlib, json
from collections import Counter
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
CODE = RESEARCH.parents[1] / 'code_working'
OUT = CODE / '_reports/downstream_master_20260917_v1'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pid(value):
    return str(int(value))

def main():
    sources = {
        'remaining': CODE/'_reports/chexzero_validation_20260917_v1/remaining_development_private.csv',
        'split': CODE/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv',
        'head_images': CODE/'_reports/medical_head_20260916_v1/images.json',
        'backbone_trace': CODE/'_reports/public_medical_backbone_20260916_v1/trace.jsonl',
        'inventory': CODE/'_data/raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv',
        'expert': CODE/'_reports/nih_expert_label_public_routes_20260917_v1/nih_all14_from_public_mirror_810.csv',
    }
    before = {k: digest(p) for k,p in sources.items()}
    inv = {x['image_id']: x for x in rows(sources['inventory'])}
    dev = {pid(x['patient_id']) for x in rows(sources['remaining'])}
    reserved = {pid(x['patient_id']) for x in rows(sources['split']) if x['provisional_role']=='reserved_confirmation'}
    expert = {pid(x['Patient ID']) for x in rows(sources['expert'])}
    head = json.loads(sources['head_images'].read_text(encoding='utf-8'))
    public = {x['image_id'] for x in head if x['split']=='public'}
    private = {x['image_id'] for x in head if x['split']=='train'}
    olddev = {pid(x['patient_id']) for x in head if x['split']=='eval'}
    traces = [json.loads(x) for x in sources['backbone_trace'].read_text(encoding='utf-8').splitlines()]
    public |= {i for x in traces if x['committed'] for i in x['image_ids']}
    def summarize(ids):
        rr = [inv[i] for i in sorted(ids)]
        positive = [r for r in rr if 'Pneumothorax' in r['finding_labels'].split('|')]
        return dict(patients=len({pid(r['patient_id']) for r in rr}),images=len(rr),
                    positive_images=len(positive),positive_patients=len({pid(r['patient_id']) for r in positive}))
    bypatient = {}
    for i,r in inv.items():
        p = pid(r['patient_id'])
        if p in dev:
            bypatient.setdefault(p,[]).append(i)
    if len(dev)!=4053 or len(reserved)!=4213 or len(expert)!=532 or set(bypatient)!=dev:
        raise ValueError('Unexpected input population')
    buckets = {False:[],True:[]}
    for p,ii in bypatient.items():
        positive = any('Pneumothorax' in inv[i]['finding_labels'].split('|') for i in ii)
        buckets[positive].append(p)
    salt = 'pneumothorax-downstream-master-20260917-v1'
    allocation = []
    for positive,pp in buckets.items():
        pp.sort(key=lambda p: hashlib.sha256((salt+'|'+p).encode()).hexdigest())
        for j,p in enumerate(pp):
            allocation.append(dict(patient_id=p,role='classifier_selection' if j%2==0 else 'method_development',
                stratification_any_weak_P=int(positive),patient_order_sha256=hashlib.sha256((salt+'|'+p).encode()).hexdigest()))
    groups = {'public_real':{pid(inv[i]['patient_id']) for i in public},'private80':{pid(inv[i]['patient_id']) for i in private},
              'old_development40':olddev,'expert_final':expert,'reserved4213':reserved}
    for role in ('classifier_selection','method_development'):
        groups[role] = {r['patient_id'] for r in allocation if r['role']==role}
    overlaps = {a+'__'+b:len(groups[a]&groups[b]) for a in groups for b in groups if a<b}
    if any(overlaps.values()):
        raise ValueError(overlaps)
    result = dict(schema='downstream-resource-snapshot/v1',scope='metadata_only_not_model_performance',
        source_sha256=before,source_paths={k:str(p) for k,p in sources.items()},
        public_real=summarize(public),private_real_diagnostic=summarize(private),
        development={role:summarize({i for p in groups[role] for i in bypatient[p]}) for role in ('classifier_selection','method_development')},
        remaining_development_patients=len(dev),reserved_confirmation_patients=len(reserved),
        expert_final_patients=len(expert),overlaps=overlaps,allocation_salt=salt,
        allocation_provisional=True,original_roles_changed=False,pixels_read=0,model_forwards=0,
        public_real_definition='All actual public-backbone training images plus public32 head images; no new private cohort relabelled public',
        dev_definition='Existing separate CVPR development pool, excluding both evaluator80 cohorts; weak image labels preserved',
        script_sha256=digest(Path(__file__)))
    if {k:digest(p) for k,p in sources.items()} != before:
        raise ValueError('Input changed')
    OUT.mkdir(exist_ok=False)
    with (OUT/'provisional_development_private.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allocation[0]));w.writeheader();w.writerows(allocation)
    with (OUT/'public_real_images_private.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','patient_id','weak_P']);w.writeheader()
        w.writerows(dict(image_id=i,patient_id=pid(inv[i]['patient_id']),weak_P=int('Pneumothorax' in inv[i]['finding_labels'].split('|'))) for i in sorted(public))
    result['allocation_sha256']=digest(OUT/'provisional_development_private.csv')
    result['public_real_manifest_sha256']=digest(OUT/'public_real_images_private.csv')
    (OUT/'resources.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_paths','source_sha256','overlaps')},indent=2))

if __name__=='__main__':
    main()
