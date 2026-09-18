"""Constructed-array contracts for the step2 feature/statistic paths only."""
import unittest
import numpy as np
import torch
from .relation_paths import (patient_paths,patient_uniform_center,public_scales,
    relation_target,paired_relation,bound)
from .patient_moments import patient_contributions,aggregate


class RelationPathTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(918)
        self.a=rng.normal(size=(12,3))*4
        self.z=bound(self.a,8)
        self.y=np.tile([0,1],6)
        self.ids=np.repeat(['a','b','c','d'],3)
        self.roles=np.array(['P']*12)
        self.ps=patient_paths(self.z,self.a,self.y,self.ids,self.roles)
        self.scales,_=public_scales(self.ps)

    def test_point_and_patient_mass_unchanged(self):
        expected=aggregate(patient_contributions(self.z,self.y,self.ids),'point')
        for arm in ('E','E_R','C','D','R_joint'):
            target,_=relation_target(self.ps,arm,self.scales)
            for key in ('m','A'):np.testing.assert_array_equal(target[key],expected[key])
            np.testing.assert_array_equal(target['counts'][:2],expected['counts'])
        idx=np.r_[np.arange(12),np.arange(3)]
        repeated=patient_paths(self.z[idx],self.a[idx],self.y[idx],self.ids[idx],self.roles[idx])
        original,_=relation_target(self.ps,'C',self.scales)
        doubled,_=relation_target(repeated,'C',self.scales)
        for key in ('m','A','delta','C'):np.testing.assert_allclose(doubled[key],original[key],atol=1e-14,rtol=0)

    def test_public_calibration_and_empty_populations(self):
        values,info=public_scales(self.ps)
        for key,norms in info['norms'].items():self.assertEqual(values[key],max(max(norms),1e-12))
        q=patient_paths(self.z,self.a,self.y,self.ids,np.array(['Q']*12))
        with self.assertRaises(RuntimeError):public_scales(q)
        empty=patient_paths(self.z,self.a,np.zeros(12,int),self.ids,self.roles)
        scales,info=public_scales(empty);self.assertEqual(info['mixed_patients'],0)
        for arm in ('E','E_R','C','D','R_joint'):
            t,rr=relation_target(empty,arm,scales)
            np.testing.assert_array_equal(t['counts'],[4,0,0])
            self.assertEqual(t['C'].sum(),0);self.assertEqual(len(rr['r']),0)

    def test_center_uses_equal_patient_mass(self):
        center,weights=patient_uniform_center(self.a,self.ids)
        ref=np.stack([self.a[self.ids==i].mean(0) for i in sorted(set(self.ids))]).mean(0)
        np.testing.assert_allclose(center,ref,atol=1e-14,rtol=0)
        for i in set(self.ids):self.assertAlmostEqual(weights[self.ids==i].sum(),.25)

    def test_raw_shift_invariance_and_joint_map(self):
        shifts=np.repeat(np.array([[100,1,2],[-90,5,3],[20,-30,9],[5,4,3]]),3,axis=0)
        moved=patient_paths(self.z,self.a+shifts,self.y,self.ids,self.roles)
        tc,rc=relation_target(self.ps,'C',self.scales);tm,rm=relation_target(moved,'C',self.scales)
        np.testing.assert_allclose(rc['r'],rm['r'],atol=1e-14,rtol=0)
        tj,rj=relation_target(self.ps,'R_joint',self.scales)
        L=np.concatenate([np.eye(3),-np.eye(3)],axis=1)/np.sqrt(2)
        np.testing.assert_allclose(L@tj['u'],tj['delta'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(L@tj['U']@L.T,tj['C'],atol=1e-14,rtol=0)
        self.assertGreater(np.linalg.norm(rc['r']-rj['r']),1e-4)

    def test_private_shuffle_only_and_post_bound_mean_can_change(self):
        a=np.array([[0.],[2.],[0.],[4.],[2.],[2.]])
        z=bound(a,4);y=np.tile([0,1],3);ids=np.repeat(['p','q1','q2'],2)
        ps=patient_paths(z,a,y,ids,np.repeat(['P','Q','Q'],2))
        scales={'C':1.,'E_R':1.,'R_joint':1.}
        tc,rc=relation_target(ps,'C',scales);td,rd=relation_target(ps,'D',scales,[1,0])
        for key in ('m','A'):np.testing.assert_array_equal(tc[key],td[key])
        np.testing.assert_allclose(rc['raw_delta'].mean(0),rd['raw_delta'].mean(0),atol=1e-14,rtol=0)
        np.testing.assert_array_equal(rc['r'][0],rd['r'][0])
        self.assertGreater(np.linalg.norm(tc['delta']-td['delta']),.1)
        with self.assertRaises(RuntimeError):relation_target(ps,'D',scales,[0,1])

    def test_torch_endpoint_gradients_and_numpy_agreement(self):
        p=self.ps[0];zp,zn,ap,an=[torch.tensor(x,dtype=torch.float64,requires_grad=True)
                                  for x in (p.point.means[1],p.point.means[0],p.raw_means[1],p.raw_means[0])]
        for arm in ('E','E_R','C','R_joint'):
            result=paired_relation(zp,zn,ap,an,arm,self.scales)
            _,rows=relation_target([p],arm,self.scales)
            np.testing.assert_allclose(result['r'].detach().numpy(),rows['r'][0],atol=1e-14,rtol=0)
            torch.autograd.gradcheck(lambda *xs:paired_relation(*xs,arm,self.scales)['r'],
                                     (zp,zn,ap,an),eps=1e-6,atol=1e-5,rtol=1e-3)


if __name__=='__main__':unittest.main()
