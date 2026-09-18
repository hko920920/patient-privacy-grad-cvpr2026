"""Constructed-array checks of the actual dual-path objective, not efficacy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from .contracts import save
from .guarded_objectives import (ARMS, LearnerPolicy, bank_moments, bind_objective,
                                feature_objective, statistical_loss)
from .guarded_readout import functional_loss
from .objectives import matching
from .render import Renderer, augment_parameters

METRICS = {}
SCALES = {'C': .73, 'E_R': .41, 'R_joint': 1.13}


def numpy_moments(z, a, arm):
    q = len(z)//4
    neg, pos = (z[:2*q], z[2*q:]) if arm in ('A','B') else (z[:q], z[2*q:3*q])
    result = {'m': np.array([neg.mean(0), pos.mean(0)]),
              'A': np.array([np.einsum('ni,nj->ij',v,v)/len(v) for v in (neg,pos)])}
    if arm in ('A','B'): return result
    if arm in ('E','E_R'):
        r = (z[3*q:]-z[q:2*q])/2
        if arm == 'E_R': r = r/np.maximum(SCALES['E_R'], np.linalg.norm(r,axis=1,keepdims=True))
    elif arm in ('C','D'):
        r = (a[3*q:]-a[q:2*q])/2
        r = r/np.maximum(SCALES['C'], np.linalg.norm(r,axis=1,keepdims=True))
    else:
        u = np.concatenate([a[3*q:],a[q:2*q]],axis=1)/np.sqrt(2)
        u = u/np.maximum(SCALES['R_joint'],np.linalg.norm(u,axis=1,keepdims=True))
        result.update(u=u.mean(0),U=np.einsum('ni,nj->ij',u,u)/len(u))
        r = (u[:,:z.shape[1]]-u[:,z.shape[1]:])/np.sqrt(2)
    result.update(delta=r.mean(0),C=np.einsum('ni,nj->ij',r,r)/len(r))
    return result


def numpy_readout(m, policy):
    M = (m['A'][0]+m['A'][1])/2+policy.ridge*np.eye(m['m'].shape[1])
    b = (m['m'][1]-m['m'][0])/2
    w0 = np.linalg.solve(M,b)
    w1 = np.linalg.solve(M+policy.beta*m['C'],b+policy.beta*m['delta']) if 'C' in m else w0
    d = w1-w0;energy=d@M@d
    radius2 = policy.rho**2 if policy.rho is not None else policy.kappa*(w0@M@w0)
    alpha = min(1.,np.sqrt(radius2/energy)) if policy.guard and energy>0 else 1.
    return w0+alpha*d,M,alpha,energy,radius2


class IntegratedObjectiveTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(351)
        self.z=rng.normal(size=(12,3))*.2;self.a=rng.normal(size=(12,3))*.8
        self.zt=rng.normal(size=(12,3))*.3;self.at=rng.normal(size=(12,3))*.7
        self.policy=LearnerPolicy(1.,.1,kappa=.1)

    def paths(self,z=None,a=None,grad=False):
        return {k:torch.tensor(v,dtype=torch.float64,requires_grad=grad)
                for k,v in [('z',self.z if z is None else z),('a',self.a if a is None else a)]}

    def target(self,arm='C',policy=None,eta=1.):
        return bind_objective(numpy_moments(self.zt,self.at,arm),arm,SCALES,
                              policy or self.policy,eta=eta)

    def test_global_moments_and_independent_guard_function(self):
        errors=[];func_errors=[]
        for arm in ARMS:
            expected=numpy_moments(self.z,self.a,arm)
            actual=bank_moments(self.paths(),arm,SCALES)
            for k,v in expected.items():
                error=float(np.max(abs(actual[k].numpy()-v)));errors.append(error)
                self.assertLess(error,1e-12)
            target=self.target(arm)
            got=feature_objective(self.paths(),self.paths(z=self.z*.9),target)
            ws,ms,alpha,energy,r2=numpy_readout(expected,self.policy)
            wt,mt,_,_,_=numpy_readout(numpy_moments(self.zt,self.at,arm),self.policy)
            np.testing.assert_allclose(got['readout'].weight.detach(),ws,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(got['readout'].point_matrix,ms,atol=1e-12,rtol=1e-12)
            self.assertAlmostEqual(float(got['readout'].alpha),alpha,places=12)
            np.testing.assert_allclose(target.functional_target.matrix,mt,atol=1e-12,rtol=1e-12)
            self.assertFalse(np.allclose(ms,mt))
            f=float((ws-wt)@mt@(ws-wt))
            func_errors.append(abs(float(got['functional'])-f))
            self.assertLess(func_errors[-1],1e-12)
            if arm in ('A','B'):
                self.assertTrue(torch.equal(statistical_loss(actual,target.moments,arm),
                                            matching(self.paths()['z'],target.moments,arm)))
        METRICS['independent_moment_max_abs']=max(errors)
        METRICS['independent_function_max_abs']=max(func_errors)

    def test_function_only_clean_and_frozen_target(self):
        clean=self.paths(grad=True);aug=self.paths(z=self.z*.9,a=self.a*.8,grad=True)
        objective=self.target();no_func=replace(objective,eta=0.)
        yes=feature_objective(clean,aug,objective);no=feature_objective(clean,aug,no_func)
        leaves=[*clean.values(),*aug.values()]
        difference=torch.autograd.grad(yes['loss']-no['loss'],leaves,retain_graph=True)
        direct=torch.autograd.grad(yes['functional'],list(clean.values()))
        for x,y in zip(difference[:2],direct):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
        self.assertTrue(all(torch.count_nonzero(x)==0 for x in difference[2:]))
        target_input={k:torch.tensor(v,dtype=torch.float64,requires_grad=True)
                      for k,v in numpy_moments(self.zt,self.at,'C').items()}
        fixed=bind_objective(target_input,'C',SCALES,self.policy,eta=1.)
        weight=fixed.functional_target.weight.clone();matrix=fixed.functional_target.matrix.clone()
        with torch.no_grad():
            for t in target_input.values():t.add_(3.)
        self.assertTrue(torch.equal(weight,fixed.functional_target.weight))
        self.assertTrue(torch.equal(matrix,fixed.functional_target.matrix))
        self.assertFalse(any(t.requires_grad for t in fixed.moments.values()))
        METRICS['augmentation_function_gradient_exact_zero']=True
        METRICS['fixed_target_copy_isolated']=True

    def test_caps_zero_and_bound(self):
        m=bank_moments(self.paths(),'C',SCALES)
        rows=[]
        for rho in (0.,.01,100.):
            p=LearnerPolicy(1.,.1,rho=rho)
            result=p.solve(m);d=result.weight-result.point_weight
            cost=.5*d@result.point_matrix@d
            self.assertLessEqual(float(cost),rho*rho/2+1e-12)
            if rho==0.:self.assertTrue(torch.equal(result.weight,result.point_weight))
            elif rho==.01:self.assertLess(float(result.alpha),1.)
            else:self.assertEqual(float(result.alpha),1.)
            rows.append({'rho':rho,'alpha':float(result.alpha),'point_excess':float(cost)})
        point=self.policy.solve({'m':m['m'],'A':m['A']})
        zero=dict(m,delta=m['C']@point.weight)
        same=self.policy.solve(zero)
        torch.testing.assert_close(same.weight,point.weight,atol=1e-12,rtol=1e-12)
        # Combined bound in the FIXED target metric, even with synthetic M_S.
        objective=self.target();s=self.policy.solve(m);t=objective.target_readout
        error=functional_loss(s.weight,objective.functional_target).sqrt()
        excess=.5*(s.weight-t.point_weight)@t.point_matrix@(s.weight-t.point_weight)
        self.assertLessEqual(float(excess),.5*float((t.radius_squared.sqrt()+error)**2)+1e-12)
        METRICS['cap_cases']=rows
        METRICS['zero_correction_max_abs']=float((same.weight-point.weight).abs().max())

    def test_feature_gradient_finite_difference(self):
        checks=[]
        for arm in ('C','R_joint'):
            for rho in (.01,100.):
                policy=LearnerPolicy(1.,.1,rho=rho)
                objective=self.target(arm,policy)
                clean=self.paths(self.z[:4],self.a[:4],grad=True)
                aug=self.paths(self.z[:4]*.8,self.a[:4]*.9)
                fn=lambda z,a:feature_objective({'z':z,'a':a},aug,objective)['loss']
                self.assertTrue(torch.autograd.gradcheck(fn,tuple(clean.values()),eps=1e-6,
                                                        atol=2e-6,rtol=2e-4))
                checks.append({'arm':arm,'rho':rho,'passed':True})
        METRICS['smooth_gradchecks']=checks

    def test_replay_global_denominators_and_unused_path(self):
        errors=[]
        for arm in ARMS:
            for micro in (1,5,12):
                objective=self.target(arm)
                x=torch.tensor(self.a,dtype=torch.float64,requires_grad=True)
                def encode(v):return {'z':v.tanh()*.25,'a':v+v.sin()*.1}
                direct=feature_objective(encode(x),encode(x*.9),objective)['loss']
                gd,=torch.autograd.grad(direct,x)
                with torch.no_grad():cs=encode(x);au=encode(x*.9)
                cs={k:v.detach().requires_grad_() for k,v in cs.items()}
                au={k:v.detach().requires_grad_() for k,v in au.items()}
                value=feature_objective(cs,au,objective)['loss']
                leaves=[*cs.values(),*au.values()]
                response=torch.autograd.grad(value,leaves,allow_unused=True)
                response=[torch.zeros_like(v) if g is None else g for g,v in zip(response,leaves)]
                for start in range(0,len(x),micro):
                    end=min(start+micro,len(x));c=encode(x[start:end]);a=encode(x[start:end]*.9)
                    sum((v*g[start:end]).sum() for v,g in zip([*c.values(),*a.values()],response)).backward()
                error=float((x.grad-gd).abs().max());errors.append(error)
                self.assertLess(error,1e-12)
        METRICS['array_replay_max_abs']=max(errors)

    def test_pair_rendering_controls_match(self):
        g=torch.Generator().manual_seed(41)
        templates=torch.rand(4,1,224,224,generator=g)*.8+.1
        c=Renderer(templates,'C',17);c.set_step(500)
        with torch.no_grad():expected=c([0,1,2,3])
        for arm in ('E','E_R','D','R_joint'):
            r=Renderer(templates,arm,17);r.set_step(500)
            with torch.no_grad():self.assertTrue(torch.equal(r([0,1,2,3]),expected))
            p=augment_parameters(4,17,500,r.relation,'cpu')
            self.assertTrue(torch.equal(p[0][1],p[0][3]) and torch.equal(p[1][1],p[1][3]))
        METRICS['all_relation_renderers_and_pair_augmentation_equal']=True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    torch.set_num_threads(4);start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegratedObjectiveTests))
    report={'status':'PASS_CONSTRUCTED_INTEGRATED_OBJECTIVE_ONLY' if result.wasSuccessful() else 'FAIL',
            'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'metrics':METRICS,'seconds':time.perf_counter()-start,'model_or_patient_input':False,
            'coefficients_numerical_test_only':True}
    save(args.report,report);print(json.dumps(report),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
