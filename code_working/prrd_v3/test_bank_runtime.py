"""Small CPU I/O checks. These are constructed arrays, not patient results."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

from .render import Renderer
from .bank_runtime import (atomic_json, capture, checkpoint_record, digest, export_artifact,
                           load_checkpoint, rng_state, run_bank, save_checkpoint,
                           tree_digest, verify_artifact)
from .contracts import sha


class RuntimeFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patch = mock.patch('torch.cuda.is_available', return_value=False)
        patch.start(); self.addCleanup(patch.stop)

    def renderer(self, arm='C'):
        # A fixed non-patient pixel array exercises uint8 clipping and rounding.
        template = torch.linspace(0, 1, 224*224).reshape(1, 1, 224, 224).repeat(4, 1, 1, 1)
        m = Renderer(template, arm, 101)
        opt = torch.optim.AdamW(m.parameters(), lr=.01, weight_decay=0.,
                               betas=(.9, .999), eps=1e-8, foreach=False)
        return m, opt

    def spec(self, arm='C'):
        return {'id': 'technical_'+arm, 'arm': arm, 'seed': 101, 'images': 4,
                'updates': 2, 'microbatch': 4, 'checkpoint_every': 25,
                'artifact_kind': 'TECHNICAL_TEST_ONLY', 'contract_sha256': 'fixture',
                'implementation_sha256': 'fixture', 'learner_policy':
                {'beta': 0 if arm == 'A' else 1, 'ridge': .1, 'rho': None, 'kappa': .1, 'guard': True},
                'eta': 1}

    def test_exact_restore_includes_rng_optimizer_and_active_pyramid(self):
        m, opt = self.renderer(); m.set_step(82)
        sum(p.square().sum() for p in m.parameters() if p.requires_grad).backward()
        opt.step()
        random.seed(27); np.random.seed(28); torch.manual_seed(29)
        # Constructed serialized state, not a claim of82 actual optimizer updates.
        expected = capture(m, opt, 82, 'fixture')
        save_checkpoint(self.root, m, opt, 82, 'fixture')
        m2, o2 = self.renderer()
        random.seed(3); np.random.seed(4); torch.manual_seed(5)
        done, receipt = load_checkpoint(self.root, m2, o2, 'fixture')
        self.assertEqual(done, 82); self.assertEqual(m2.active, 2)
        self.assertTrue(receipt['restored_exact'])
        self.assertEqual(tree_digest(capture(m2, o2, 82, 'fixture')), tree_digest(expected))

    def test_changed_contract_and_corrupt_checkpoint_rejected(self):
        m, opt = self.renderer(); m.set_step(1)
        save_checkpoint(self.root, m, opt, 0, 'one')
        with self.assertRaisesRegex(RuntimeError, 'another run'):
            load_checkpoint(self.root, m, opt, 'two')
        record, p = checkpoint_record(self.root)
        with p.open('ab') as f:
            f.write(b'corruption')
        with self.assertRaisesRegex(RuntimeError, 'bytes changed'):
            checkpoint_record(self.root)

    def test_intermediate_checkpoint_cannot_export(self):
        m, _ = self.renderer()
        with self.assertRaisesRegex(RuntimeError, 'intermediate'):
            export_artifact(self.root, m, self.spec(), 's',
                            {'completed_updates': 1, 'sha256': 'intermediate'})
        self.assertFalse((self.root/'artifact').exists())

    def test_png_labels_pairs_hashes_and_crash_after_publish_recovery(self):
        for arm in ('A', 'C'):
            with self.subTest(arm=arm):
                d = self.root/arm; d.mkdir()
                m, opt = self.renderer(arm); m.set_step(2)
                spec = self.spec(arm); sig = digest(spec)
                cp = save_checkpoint(d, m, opt, 2, sig)
                report = export_artifact(d, m, spec, sig, cp)
                self.assertEqual(report['images'], 4)
                self.assertEqual(report['pairs'], 0 if arm == 'A' else 1)
                before = {str(p.relative_to(d/'artifact')): sha(p)
                          for p in (d/'artifact').rglob('*') if p.is_file()}
                # A crash before COMPLETED publication can finalize the same bytes.
                export_artifact(d, m, spec, sig, cp)
                after = {str(p.relative_to(d/'artifact')): sha(p)
                         for p in (d/'artifact').rglob('*') if p.is_file()}
                self.assertEqual(before, after)

    def test_complete_bank_and_existing_file_cannot_be_overwritten(self):
        m, opt = self.renderer()
        spec = self.spec()
        atomic_json(self.root/'COMPLETED.json', {'sealed': True})
        objective = type('Arm', (), {'arm': 'C'})()
        with self.assertRaisesRegex(RuntimeError, 'sealed'):
            run_bank(self.root, m, None, objective, opt, spec, resume=True)
        p = self.root/'fixed.json'; atomic_json(p, {'first': 1})
        old = p.read_bytes()
        with self.assertRaises(FileExistsError):
            atomic_json(p, {'second': 2})
        self.assertEqual(old, p.read_bytes())

    def test_modified_manifest_or_extra_png_rejected(self):
        m, opt = self.renderer(); m.set_step(2)
        spec = self.spec(); sig = digest(spec)
        cp = save_checkpoint(self.root, m, opt, 2, sig)
        export_artifact(self.root, m, spec, sig, cp)
        p = self.root/'artifact/images.csv'
        p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            verify_artifact(self.root/'artifact', spec, sig)


if __name__ == '__main__':
    unittest.main()

