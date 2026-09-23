"""Only changed schedule/checkpoint semantics; no image encoder execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from prrd_v3.bank_runtime import save_checkpoint, load_checkpoint, capture, tree_digest
from .schedule import ScheduledRenderer, ACTIVATIONS
from .runtime import OPTIMIZER, seed_all


def optimizer(renderer):
    options={k:v for k,v in OPTIMIZER.items() if k!='name'}
    options['betas']=tuple(options['betas'])
    return torch.optim.AdamW(renderer.parameters(),**options)


def step(renderer,opt,number):
    renderer.set_step(number); opt.zero_grad(set_to_none=True)
    # A random differentiable parameter objective exercises new optimizer slots.
    loss=sum((p*torch.randn_like(p)).sum() for p in renderer.parameters() if p.requires_grad)
    loss.backward(); opt.step()


class ScheduleTests(unittest.TestCase):
    def test_all_boundaries_and_final_exposure(self):
        r=ScheduledRenderer(torch.full((4,1,224,224),.5),'B',101)
        counts=[0]*6
        for s in range(1,201):
            r.set_step(s); expected=sum(s>=a for a in ACTIVATIONS)
            self.assertEqual(r.active,expected)
            self.assertEqual([p.requires_grad for p in r.parameters()],[j<expected for j in range(6)])
            counts[expected-1]+=1
        self.assertEqual(counts,[32,32,32,32,32,40])

    def test_resume_across_each_new_activation_is_exact(self):
        with patch('torch.cuda.is_available',return_value=False):
            for activation in ACTIVATIONS[1:]:
                with self.subTest(activation=activation), tempfile.TemporaryDirectory() as tmp:
                    a=ScheduledRenderer(torch.full((4,1,224,224),.5),'B',101)
                    opt=optimizer(a); seed_all(101)
                    step(a,opt,activation-1)
                    save_checkpoint(Path(tmp),a,opt,activation-1,'schedule200')
                    step(a,opt,activation)
                    continuous=capture(a,opt,activation,'schedule200')
                    b=ScheduledRenderer(torch.full((4,1,224,224),.5),'B',101)
                    other=optimizer(b); seed_all(98765)
                    done,info=load_checkpoint(Path(tmp),b,other,'schedule200')
                    self.assertEqual(done,activation-1); self.assertTrue(info['restored_exact'])
                    step(b,other,activation)
                    self.assertEqual(tree_digest(continuous),tree_digest(capture(b,other,activation,'schedule200')))

    def test_invalid_logical_steps_rejected(self):
        r=ScheduledRenderer(torch.full((4,1,224,224),.5),'B',101)
        for s in (0,201,500):
            with self.assertRaises(RuntimeError): r.set_step(s)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
