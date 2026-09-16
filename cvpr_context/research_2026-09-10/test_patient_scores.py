"""Tests for leakage, ties, score orientation, and finite calibration ranks."""
import unittest
import numpy as np
from evaluate_patient_scores import aggregate_rows,evaluate,ranks,binomial_interval

class EvaluationTests(unittest.TestCase):
    def row(self,p,i,s,y,v):return dict(patient_id=p,image_id=i,split=s,member=y,score=v)
    def test_patient_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_rows([self.row('p','a','calibration',0,1),self.row('p','b','test',0,2)],'high','mean')
    def test_duplicate_image_and_nonfinite_rejected(self):
        for rows in [[self.row('a','i','test',0,1),self.row('b','i','test',0,1)],
                     [self.row('a','i','test',0,float('nan'))]]:
            with self.assertRaises(ValueError):aggregate_rows(rows,'high','mean')
    def test_ties_and_finite_resolution(self):
        self.assertTrue(np.allclose(ranks([1,1,2],[1,2,3]),[1,.5,.25]))
        self.assertGreater(ranks(np.arange(50),[100])[0],.01)
    def test_direction_before_topk(self):
        rows=[self.row('p',str(i),'test',1,v) for i,v in enumerate([1,2,10])]
        self.assertEqual(aggregate_rows(rows,'low','topk',2)[0]['score'],-1.5)
    def test_frozen_calibration_separate_from_test_roc(self):
        bags=[dict(patient_id=str(i),split='calibration',member=0,images=1,score=100+i) for i in range(99)]
        bags += [dict(patient_id='m',split='test',member=1,images=2,score=2),
                 dict(patient_id='n',split='test',member=0,images=1,score=1)]
        out=evaluate(bags)
        self.assertEqual(out['member_positive_auc'],1)
        self.assertEqual(out['frozen_rule_tpr'],0)
        self.assertEqual(out['descriptive_test_roc_tpr_at_fpr_le_1pct'],1)
    def test_zero_errors_does_not_mean_zero_population_fpr(self):
        self.assertGreater(binomial_interval(0,100)[1],.03)

if __name__=='__main__':unittest.main(verbosity=2)
