"""Meaningful algebra/accounting/edge-case tests; constructed data only."""
import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mpmath as mp
from scipy.special import ndtr
from dp_accounting.pld import privacy_loss_mechanism as google,common
from . import patient_dp as m
from .patient_dp_io import write_target_handoffs,release_from_contract
from prrd_v3.contracts import save

METRICS={}


class PatientDPTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(271)
        self.x=self.rng.normal(0,.1,(7,2,4,34))
        self.y=np.array([0,0,1,0,1,0,1])
        self.ids=np.array(['a','a','a','b','c','d','d'])
        self.p=m.aggregate_image_signals(self.x,self.y,self.ids)
        self.limits=m.Limits((1.,1.3))
        self.ps,self.pc=self.p.sums_counts()

    def test_patient_class_weighting_independent(self):
        expected=np.zeros((2,4,34))
        for c in (0,1):
            people=set(self.ids[self.y==c])
            for p in people:
                ii=np.flatnonzero((self.ids==p)&(self.y==c))
                for j in ii:expected+=.5/len(people)/len(ii)*self.x[j]
        actual=m.raw_balanced_target(self.ps,self.pc,np.zeros_like(self.ps),np.zeros(2))
        error=float(np.max(abs(actual-expected)));self.assertLess(error,1e-14)
        METRICS['independent_patient_target_error']=error

    def test_patient_sum_preserves_small_cancelling_terms(self):
        values=np.array([[1.,1.],[2**-53,2**-54],[-1.,-1.]])
        np.testing.assert_array_equal(m.sum_patient_axis(values),[2**-53,2**-54])
        np.testing.assert_array_equal(m.sum_patient_axis(np.zeros((0,3))),np.zeros(3))

    def test_visit_repetition_no_extra_patient_weight(self):
        ii=np.array([0,1,0,1,2,2,3,4,5,6])
        repeated=m.aggregate_image_signals(self.x[ii],self.y[ii],self.ids[ii])
        np.testing.assert_allclose(repeated.means,self.p.means,atol=1e-15)
        np.testing.assert_array_equal(repeated.present,self.p.present)

    def test_add_remove_and_replace_all_blocks(self):
        rows,_=m.clipped_contributions(self.p,self.limits);total=rows.sum(0)
        maximum=0.
        for j in range(len(rows)):
            reduced=m.Patients(tuple(p for k,p in enumerate(self.p.ids) if k!=j),
                               np.delete(self.p.means,j,0),np.delete(self.p.present,j,0))
            other,_=m.clipped_contributions(reduced,self.limits)
            error=np.linalg.norm(total-other.sum(0));maximum=max(maximum,error)
            self.assertLessEqual(error,1+1e-13)
        for _ in range(25):
            means=self.rng.normal(size=(1,2,2,4,34))*1e3
            candidate=m.Patients(('new',),means,np.ones((1,2),bool))
            replaced,_=m.clipped_contributions(candidate,self.limits)
            self.assertLessEqual(np.linalg.norm(replaced[0]-rows[0]),2+1e-13)
        METRICS['enumerated_max_add_remove_distance']=maximum

    def test_both_class_saturation_witness_and_single_presence(self):
        means=np.zeros((2,2,2,4,34));means[0,:,0,0,0]=10000;means[1,0,0,0,0]=-10000
        patient=m.Patients(('mixed','negative'),means,np.array([[True,True],[True,False]]))
        rows,_=m.clipped_contributions(patient,self.limits)
        self.assertLessEqual(np.linalg.norm(rows[0]),1.)
        self.assertGreater(np.linalg.norm(rows[0]),1-1e-12)
        self.assertAlmostEqual(np.linalg.norm(rows[1]),1/math.sqrt(2),places=14)
        self.assertEqual(rows.shape,(2,546))

    def test_absent_class_and_empty_dataset(self):
        empty=m.Patients((),np.zeros((0,2,2,4,34)),np.zeros((0,2),bool))
        rows,_=m.clipped_contributions(empty,self.limits)
        self.assertEqual(rows.shape,(0,546))
        targets,details=m.decode_target(rows.sum(0),self.limits,self.ps,self.pc)
        ref=m.raw_balanced_target(self.ps,self.pc,np.zeros_like(self.ps),np.zeros(2))
        np.testing.assert_allclose(np.stack(list(targets.values())),ref,atol=1e-15)
        self.assertEqual(details['nonnegative_Q_mass'],[0.,0.])

    def test_unclipped_no_noise_reconstructs_target(self):
        limits=m.Limits((100.,100.));rows,_=m.clipped_contributions(self.p,limits)
        targets,_=m.decode_target(rows.sum(0),limits,self.ps,self.pc)
        expected=m.raw_balanced_target(self.ps,self.pc,self.ps,self.pc)
        np.testing.assert_allclose(np.stack(list(targets.values())),expected,atol=1e-15)

    def test_bounded_no_noise_preserves_class_mass(self):
        rows,diag=m.clipped_contributions(self.p,self.limits)
        np.testing.assert_allclose(2*rows.sum(0)[-2:],self.pc,atol=1e-13)
        targets,info=m.decode_target(rows.sum(0),self.limits,self.ps,self.pc)
        self.assertTrue(all(np.isfinite(t).all() for t in targets.values()))
        np.testing.assert_allclose(info['consistency_projection_factor'],1.,atol=1e-13)
        self.assertTrue(np.any(diag['class_factors']<1))

    def test_negative_zero_and_tiny_noisy_denominators(self):
        for masses in ([-100.,-100.],[0.,0.],[-1.,1e-16]):
            q=np.ones(546);q[-2:]=np.asarray(masses)/2
            targets,info=m.decode_target(q,self.limits,self.ps,self.pc)
            self.assertTrue(all(np.isfinite(t).all() for t in targets.values()))
            self.assertTrue(np.all(np.asarray(info['pooled_denominator'])>=self.pc))
            if max(masses)<=0:
                expected=m.raw_balanced_target(self.ps,self.pc,np.zeros_like(self.ps),np.zeros(2))
                np.testing.assert_allclose(np.stack(list(targets.values())),expected,atol=1e-15)

    def test_public_fallback_uses_no_private_redraw(self):
        # Construct a protected value that exactly cancels P; large bound leaves it intact.
        limits=m.Limits((100.,100.))
        q=np.concatenate([(-self.ps/200).reshape(-1),self.pc/2])
        targets,info=m.decode_target(q,limits,self.ps,self.pc)
        self.assertTrue(np.asarray(info['public_fallback_conditions']).all())
        expected=m.raw_balanced_target(self.ps,self.pc,np.zeros_like(self.ps),np.zeros(2))
        np.testing.assert_array_equal(np.stack(list(targets.values())),expected)

    def test_invalid_input_and_public_limits(self):
        for bounds in ((0.,1.),(float('inf'),1.),(True,1.),(1.,)):
            with self.assertRaises((ValueError,TypeError)):m.Limits(bounds)
        with self.assertRaises(ValueError):m.aggregate_image_signals(self.x,self.y+.5,self.ids)
        with self.assertRaises(ValueError):m.raw_balanced_target(self.ps,-self.pc,self.ps,self.pc)
        with self.assertRaises(ValueError):m.decode_target(np.full(546,np.nan),self.limits,self.ps,self.pc)
        with self.assertRaises(ValueError):m.decode_target(np.zeros(545),self.limits,self.ps,self.pc)
        with self.assertRaises(ValueError):m.mechanism_metadata(self.limits,8,0)
        with self.assertRaises(ValueError):m.mechanism_metadata(self.limits,8,1e-5,'image')
        bad=self.p.means.copy();bad[~self.p.present]=1
        with self.assertRaises(ValueError):m.Patients(self.p.ids,bad,self.p.present).validate()
        chosen,details=m.public_limits(self.p)
        self.assertEqual([d['public_class_patients'] for d in details],[3,3])
        self.assertTrue(all(c>0 for c in chosen.class_bounds))

    def test_analytic_gaussian_against_google_and_high_precision(self):
        rows=[];mp.mp.dps=70
        for eps in (1.,4.,8.):
            meta=m.mechanism_metadata(self.limits,eps,1e-5);sigma=meta['sigma']
            reference=google.GaussianPrivacyLoss.from_privacy_guarantee(common.DifferentialPrivacyParameters(eps,1e-5))
            error=abs(sigma-reference.standard_deviation)
            self.assertLess(error,1e-7)
            sd=mp.mpf(str(sigma));ep=mp.mpf(str(eps))
            phi=lambda t:mp.erfc(-t/mp.sqrt(2))/2
            delta=phi(1/(2*sd)-ep*sd)-mp.exp(ep)*phi(-1/(2*sd)-ep*sd)
            self.assertLessEqual(delta,mp.mpf('0.00001'))
            for mode in (google.AdjacencyType.ADD,google.AdjacencyType.REMOVE):
                g=google.GaussianPrivacyLoss(sigma,adjacency_type=mode)
                self.assertLessEqual(g.get_delta_for_epsilon(eps),1e-5*(1+1e-10))
            replacement=m.mechanism_metadata(self.limits,eps,1e-5,'replace_one_patient')
            self.assertEqual(replacement['sigma'],2*sigma)
            rows.append({'epsilon':eps,'sigma':sigma,'google_sigma':reference.standard_deviation,
                         'sigma_abs_difference':error,'mp_delta':float(delta)})
        METRICS['accounting']=rows

    def test_full_vector_noise_once_even_with_empty_Q(self):
        empty=m.Patients((),np.zeros((0,2,2,4,34)),np.zeros((0,2),bool))
        class Fake:
            calls=0
            def normalvariate(self,a,b):self.calls+=1;return 1.
        fake=Fake()
        with patch.object(m.random,'SystemRandom',return_value=fake):
            value,meta=m.privatize_patients(empty,self.limits)
        self.assertEqual(fake.calls,2*546)
        np.testing.assert_allclose(value,math.sqrt(2)*meta['sigma'],rtol=1e-15)
        self.assertEqual(meta['private_queries'],1)
        self.assertFalse(any(k in meta for k in ('patient_ids','seed','noise','counts','clipping_factors')))

    def test_writer_has_no_private_payload_and_no_overwrite(self):
        paths={n:{'rule':{'public_rule':n}} for n in m.ENCODERS}
        targets={n:np.ones((4,34)) for n in m.ENCODERS}
        meta=m.mechanism_metadata(self.limits)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'target'
            h=write_target_handoffs(output,targets,paths,meta,privacy='PUBLIC_SIMULATION_ONLY')
            for f in output.rglob('*.npz'):
                with np.load(f) as z:self.assertEqual(set(z.files),{'schema','rule_sha256','condition_ids','pooled_gradient'})
            with self.assertRaises(FileExistsError):
                write_target_handoffs(output,targets,paths,meta,privacy='PUBLIC_SIMULATION_ONLY')
            with self.assertRaises(RuntimeError):
                write_target_handoffs(Path(tmp)/'bad',targets,paths,{**meta,'raw_count':99},privacy='PUBLIC_SIMULATION_ONLY')

    def test_draft_release_contract_cannot_touch_Q(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';save(p,{'schema':'receiver.patient-dp-release-plan/v1','status':'DRAFT_NOT_AUTHORIZED'})
            with self.assertRaisesRegex(RuntimeError,'frozen one-release'):
                release_from_contract(p)


def main():
    import argparse,time
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PatientDPTests))
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'metrics':METRICS,
            'seconds':time.monotonic()-start,'constructed_arrays_only':True,'Q_noise_draws':0,
            'encoder_forwards':0,'optimizer_updates':0,'private_release_executed':False}
    save(a.output,report);print(json.dumps(report),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
