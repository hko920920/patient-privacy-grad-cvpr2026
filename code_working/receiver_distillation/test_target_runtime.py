"""Changed-path target binding and checkpoint tests, with no model execution."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
import torch
from prrd_v3.contracts import save, sha
from prrd_v3.render import Renderer
from prrd_v3.bank_runtime import digest, save_checkpoint, load_checkpoint
from .signals import objective_digest
from .target_io import load_target, descriptor
from .runtime import OPTIMIZER, code_bindings, validate_job

HANDOFF = None


class TargetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        if HANDOFF is None:
            self.skipTest('A1 target handoff is an internal non-DP artifact; pass --handoff to run binding tests')
        self.handoff = json.loads(HANDOFF.read_text(encoding='utf-8'))

    def test_actual_condition_target_and_public_target_load_exact(self):
        with np.load(self.handoff['target_file'],allow_pickle=False) as z:
            for pop in ('P','Q','pooled'):
                objective, rule = load_target(self.handoff,(0,0,1,1),population=pop)
                self.assertEqual(rule['conditions'],4)
                self.assertEqual(objective.targets['biovil'].shape,(4,34))
                self.assertTrue(np.array_equal(objective.targets['biovil'].numpy(),z[pop+'_gradient']))
                self.assertFalse(objective.targets['biovil'].requires_grad)

    def test_changed_target_bytes_rejected(self):
        path = self.root/'target.npz'
        path.write_bytes(Path(self.handoff['target_file']).read_bytes()+b'changed')
        binding = dict(self.handoff,target_file=str(path))
        with self.assertRaisesRegex(RuntimeError,'target bytes'):
            load_target(binding,(0,1))

    def test_reordered_or_changed_conditions_rejected(self):
        rule = json.loads(Path(self.handoff['rule_file']).read_text(encoding='utf-8'))
        rule['tuples'][0],rule['tuples'][1] = rule['tuples'][1],rule['tuples'][0]
        path = self.root/'changed_rule.json'
        save(path,rule)
        binding = dict(self.handoff,rule_file=str(path),rule_file_sha256=sha(path),rule_sha256=digest(rule))
        with self.assertRaisesRegex(RuntimeError,'head/transform/preprocess'):
            load_target(binding,(0,1))

    def test_mean_target_or_historical_schema_cannot_replace_conditions(self):
        with np.load(self.handoff['target_file'],allow_pickle=False) as data:
            payload = {k:data[k].copy() for k in data.files}
        payload['pooled_gradient'] = payload['pooled_gradient'].mean(0)
        path = self.root/'mean.npz'
        np.savez(path,**payload)
        binding = dict(self.handoff,target_file=str(path),target_sha256=sha(path))
        with self.assertRaisesRegex(RuntimeError,'shape/values'):
            load_target(binding,(0,1))

    def test_resume_rejects_different_target_signature(self):
        patch = mock.patch('torch.cuda.is_available',return_value=False)
        with patch:
            renderer = Renderer(torch.full((4,1,224,224),.5),'B',101)
            optimizer = torch.optim.AdamW(renderer.parameters(),lr=.01,weight_decay=0.,foreach=False)
            renderer.set_step(1)
            one = digest({'target':self.handoff['target_sha256'],'rule':self.handoff['rule_sha256']})
            two = digest({'target':'different','rule':self.handoff['rule_sha256']})
            save_checkpoint(self.root,renderer,optimizer,0,one)
            with self.assertRaisesRegex(RuntimeError,'another run'):
                load_checkpoint(self.root,renderer,optimizer,two)

    def test_main_authorization_is_not_target_preparation(self):
        job = {'schema':'receiver.A1-bank-job/v1','id':'dev_A1_condition_k4_101','encoder_id':'biovil',
               'seed':101,'images':128,'updates':500,'microbatch':16,
               'artifact_kind':'MAIN_BANK','population':'pooled','optimizer':OPTIMIZER,
               'handoff':str(HANDOFF),'handoff_sha256':sha(HANDOFF),
               'rule_sha256':self.handoff['rule_sha256'],'code_bindings':code_bindings()}
        with self.assertRaisesRegex(RuntimeError,'outside this preparation'):
            validate_job(job)


def main():
    global HANDOFF
    ap = argparse.ArgumentParser()
    ap.add_argument('--handoff',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    HANDOFF = args.handoff
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TargetRuntimeTests))
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),
              'seconds':time.perf_counter()-started,'GPU_model_forward':0,'patient_pixels':0}
    save(args.output,report)
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

