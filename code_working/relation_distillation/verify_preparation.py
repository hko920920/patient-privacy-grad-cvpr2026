"""Check prepared metadata bindings and run CPU-only independent array tests."""
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest

CODE=Path(__file__).resolve().parents[1]
OUT=CODE/'_reports/relation_distillation_plan_20260918_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    result_path=OUT/'verification.json'
    if result_path.exists():
        raise RuntimeError('No overwrite of verification')
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    for path,digest in plan['bindings'].items():
        if sha(path)!=digest:
            raise RuntimeError('Binding changed: '+path)
    cfg=plan['settings']
    with (CODE/cfg['data']['training_manifest']).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    public={r['image_id']:r for r in rows if r['planned_namespace']=='real/public'}
    private_labels={}
    for row in rows:
        if row['planned_namespace']=='diagnostic_real/former_classifier_selection':
            private_labels.setdefault(row['patient_id'],set()).add(int(row['label']))
    mixed={p for p,y in private_labels.items() if y=={0,1}}
    init=json.loads((OUT/'initialization_private.json').read_text(encoding='utf-8'))
    assert len(init)==384
    for seed in cfg['bank']['seeds']:
        bank={row['cell']:row for row in init if row['bank_seed']==seed}
        assert set(bank)==set(range(128))
        assert len({row['public_patient_id'] for row in bank.values()})==64
        for k,row in bank.items():
            source=public[row['image_id']]
            assert row['requested_label']==int(k>=64)
            assert row['source_sha256']==source['sha256']
            assert row['source_path']==source['path']
            assert row['public_patient_id']==source['patient_id']
            assert row['image_id']==bank[(k+64)%128]['image_id']
    mapping=json.loads((OUT/'shuffle_private.json').read_text(encoding='utf-8'))
    assert len(mapping)==102
    assert {x['positive_patient'] for x in mapping}==mixed
    assert {x['negative_patient'] for x in mapping}==mixed
    assert all(x['positive_patient']!=x['negative_patient'] for x in mapping)
    assert not cfg['scope']['expert_final_allowed'] and not cfg['scope']['reserved_confirmation_allowed']
    assert not cfg['dp_future']['enabled'] and not plan['runtime_ready']
    capture=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName('relation_distillation.test_core')
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(capture.getvalue())
    report={'status':'PASS_PREPARATION_BINDINGS_AND_CPU_CORE_ONLY',
            'plan_sha256':sha(OUT/'plan.json'),'verifier_sha256':sha(__file__),
            'array_tests':result.testsRun,'bound_files':len(plan['bindings']),
            'initialization_rows_verified':len(init),'private_permutation_pairs':len(mapping),
            'test_log':capture.getvalue(),'new_patient_image_decodes':0,
            'new_encoder_forward':0,'new_medical_training':0,'new_synthesis':0,'new_dp_release':0,
            'not_verified':['pretrained-model loading/forward/backward','pixel/cache parity',
                            'end-to-end image optimization','clinical or downstream utility'],
            'runtime_ready':False,'final_ready':False}
    with result_path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('test_log','not_verified')},ensure_ascii=True))


if __name__=='__main__':
    main()
