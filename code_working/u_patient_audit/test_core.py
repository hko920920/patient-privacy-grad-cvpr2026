import unittest
import numpy as np
import torch
from PIL import Image
from .prepare_cohort import fingerprint,near_duplicate,choose_distinct
from .probe import make_basis,projected_step
from .train_coverage import epoch_indices

class CoreTests(unittest.TestCase):
    def test_epoch_schedule_covers_every_training_image(self):
        first=[i for step in range(228) for i in epoch_indices(912,step)]
        second=[i for step in range(228,456) for i in epoch_indices(912,step)]
        self.assertEqual(sorted(first),list(range(912)))
        self.assertEqual(sorted(second),list(range(912)))
        self.assertNotEqual(first,second)
        self.assertEqual(epoch_indices(912,333),epoch_indices(912,333))
    def row(self,name,followup,pixels):
        pixel,phash,tiny=fingerprint(Image.fromarray(pixels))
        return {"image_id":name,"patient_id":"synthetic-patient","followup_no":followup,
            "pixel_sha256":pixel,"_phash":phash,"_tiny":tiny}
    def test_duplicate_and_followup_exclusion(self):
        rng=np.random.default_rng(4)
        arrays=[rng.integers(0,256,(128,128),dtype=np.uint8) for _ in range(5)]
        a=self.row("a","1",arrays[0]);duplicate=self.row("b","2",arrays[0])
        self.assertTrue(near_duplicate(a,duplicate))
        rows=[a,duplicate]+[self.row(str(i),str(i+3),v) for i,v in enumerate(arrays[1:])]
        chosen=choose_distinct(rows,4)
        self.assertEqual(len(chosen),4)
        self.assertEqual(len({r["pixel_sha256"] for r in chosen}),4)
        self.assertEqual([r["image_id"] for r in chosen],[r["image_id"] for r in choose_distinct(rows[::-1],4)])
        for r in rows:r["followup_no"]="same"
        self.assertEqual(choose_distinct(rows,4),[])
    def test_basis_padding_orthonormal_reproducible(self):
        hidden=torch.zeros(1,6,8);mask=torch.tensor([[1,1,1,0,0,0]])
        a=make_basis(hidden,mask)
        self.assertTrue(torch.equal(a,make_basis(hidden,mask)))
        self.assertTrue(torch.all(a[24:]==0))
        torch.testing.assert_close(a.T@a,torch.eye(8),atol=2e-6,rtol=2e-6)
    def test_projected_gradient_constraint_and_null(self):
        a=torch.zeros(8)
        self.assertTrue(torch.equal(projected_step(a,torch.zeros(8)),a))
        for _ in range(100):a=projected_step(a,torch.ones(8))
        self.assertLessEqual(float(a.norm()),.05000001)
        self.assertGreater(float(a.norm()),.0499)

if __name__=="__main__":unittest.main()
