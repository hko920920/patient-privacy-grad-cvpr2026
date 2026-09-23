"""Changed one-source contribution, denominator and objective checks only."""
import json,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from prrd_v3.contracts import read,save,sha,require,rows
from prrd_v3.datasets import TRAIN
from . import patient_dp_dino as dp
from . import patient_dp as old
from .patient_dp_dino_io import read_cached_population,write_target_handoffs
from .dino_control_runtime import load_dino
from .signals import bind_objective


def constructed_checks():
    rng=np.random.default_rng(271);x=rng.normal(0,.1,(7,1,4,34))
    y=np.array([0,0,1,0,1,0,1]);ids=np.array(['a','a','a','b','c','d','d'])
    p=dp.aggregate_image_signals(x,y,ids);ps,pc=p.sums_counts();lim=dp.Limits((1.,1.3))
    expected=np.zeros((1,4,34))
    for c in (0,1):
        people=set(ids[y==c])
        for who in people:
            ix=np.flatnonzero((ids==who)&(y==c))
            for j in ix:expected+=.5/len(people)/len(ix)*x[j]
    err=float(abs(dp.raw_balanced_target(ps,pc,ps,pc)-expected).max())
    require(err<1e-14,'Independent patient weighting mismatch')
    repeat=np.array([0,1,0,1,2,2,3,4,5,6])
    repeated=dp.aggregate_image_signals(x[repeat],y[repeat],ids[repeat])
    require(np.max(abs(repeated.means-p.means))<1e-15,'Repeated visits increase patient weight')
    means=np.zeros((2,2,1,4,34));means[0,:,0,0,0]=1e6;means[1,0,0,0,0]=-1e6
    witness=dp.Patients(('mixed','negative'),means,np.array([[True,True],[True,False]]))
    wrows,_=dp.clipped_contributions(witness,lim)
    require(wrows.shape==(2,274) and 1-1e-12<np.linalg.norm(wrows[0])<=1,'Sensitivity witness')
    require(abs(np.linalg.norm(wrows[1])-1/math.sqrt(2))<1e-14,'Single-class bound')
    bounded,_=dp.clipped_contributions(p,lim);maximum=0.
    for j in range(len(p.ids)):
        reduced=dp.Patients(tuple(k for k in p.ids if k!=p.ids[j]),np.delete(p.means,j,0),np.delete(p.present,j,0))
        rr,_=dp.clipped_contributions(reduced,lim)
        dist=float(np.linalg.norm(dp.sum_patient_axis(bounded)-dp.sum_patient_axis(rr)))
        maximum=max(maximum,dist);require(dist<=1+1e-13,'Add/remove sensitivity')
    empty=dp.Patients((),np.zeros((0,2,1,4,34)),np.zeros((0,2),bool))
    require(dp.clipped_contributions(empty,lim)[0].shape==(0,274),'Empty Q')
    for mass in ([-100,-100],[0,0],[-1,1e-16]):
        q=np.ones(274);q[-2:]=np.array(mass)/2
        t,info=dp.decode_target(q,lim,ps,pc)
        require(np.isfinite(t['dinov2_vitb14']).all() and min(info['pooled_denominator'])>=1,'Bad denominator')
        if max(mass)<=0:require(np.max(abs(t['dinov2_vitb14']-expected[0]))<1e-14,'Zero mass not P-only')
    invalid=0
    for fn in (lambda:dp.Limits((0,1)),lambda:dp.decode_target(np.zeros(546),lim,ps,pc),
               lambda:dp.decode_target(np.full(274,np.nan),lim,ps,pc),
               lambda:dp.raw_balanced_target(ps,-pc,ps,pc),
               lambda:dp.aggregate_image_signals(np.zeros((7,2,4,34)),y,ids)):
        try:fn()
        except ValueError:invalid+=1
    require(invalid==5,'Invalid input accepted')
    meta=dp.mechanism_metadata(lim);ref=old.mechanism_metadata(old.Limits((1.,1.3)))
    require(meta['sigma']==ref['sigma'] and meta['dimension']==274,'Calibration changed')
    class Fake:
        calls=0
        def normalvariate(self,a,b):self.calls+=1;return 1.
    fake=Fake()
    with patch.object(dp.random,'SystemRandom',return_value=fake):value,meta=dp.privatize_patients(empty,lim)
    require(fake.calls==548 and np.max(abs(value-math.sqrt(2)*meta['sigma']))<1e-14,'Wrong noise vector')
    return {'status':'PASS','patient_weight_error':err,'max_add_remove_distance':maximum,
        'query_dimension':274,'saturated_patient_norm':float(np.linalg.norm(wrows[0])),
        'invalid_inputs_rejected':invalid,'fake_sampler_calls':fake.calls,'Q_noise_draws':0,
        'old_verified_analytic_calibration_exact':True}


def prepare_check(root,paths,p,pref,limits):
    q,qref=read_cached_population(paths,'Q');ps,pc=p.sums_counts();qs,qc=q.sums_counts()
    require(not(set(p.ids)&set(q.ids)),'P/Q overlap')
    manifest=rows(TRAIN);population={}
    for role,data,reference in (('P',p,pref),('Q',q,qref)):
        group=[r for r in manifest if r['planned_namespace']=={'P':'real/public','Q':'diagnostic_real/former_classifier_selection'}[role]]
        for key,exp in (('image_ids',[r['image_id'] for r in group]),('patient_ids',[r['patient_id'] for r in group]),('labels',[int(r['label']) for r in group])):
            require(np.array_equal(reference['metadata'][key],exp),'Manifest/cache order mismatch')
        sums,counts=data.sums_counts();require(np.array_equal(counts,reference['counts']),'Class counts differ')
        err=float(abs(sums-reference['sums']).max());require(err<=1e-12,'Historical sums differ')
        population[role]={'patients':len(data.ids),'images':len(group),'class_counts':counts.tolist(),'sum_error':err}
    raw=dp.raw_balanced_target(ps,pc,qs,qc)
    with np.load(paths['dinov2_vitb14']['target'],allow_pickle=False) as f:original=f['pooled_gradient'].copy()
    error=float(abs(raw[0]-original).max());require(error<=1e-12,'Original DINO target mismatch')
    rows_q,diag=dp.clipped_contributions(q,limits);total=dp.sum_patient_axis(rows_q)
    clipped,post=dp.decode_target(total,limits,ps,pc)
    independent=q.means*diag['class_factors'][:,:,None,None,None]*diag['joint_factors'][:,None,None,None,None]
    counts=(q.present*diag['joint_factors'][:,None]).sum(0)
    independent_target=dp.raw_balanced_target(ps,pc,dp.sum_patient_axis(independent),counts)
    cliperr=float(abs(clipped['dinov2_vitb14']-independent_target[0]).max())
    require(cliperr<=1e-12,'Clipped decode mismatch')
    huge=dp.Limits((1e6,1e6));unc,_=dp.clipped_contributions(q,huge)
    decoded,_=dp.decode_target(dp.sum_patient_axis(unc),huge,ps,pc)
    unerr=float(abs(decoded['dinov2_vitb14']-original).max());require(unerr<=1e-12,'Unbounded noiseless reproduction')
    # Public P simulates Q for objective serialization/feature-gradient checks.
    pubrows,_=dp.clipped_contributions(p,limits)
    mock,_=dp.simulate_public(dp.sum_patient_axis(pubrows),limits,np.random.default_rng(823).normal(size=274))
    target,_=dp.decode_target(mock,limits,ps,pc)
    hf=write_target_handoffs(root/'public_simulation',target,paths,dp.mechanism_metadata(limits),privacy='PUBLIC_SIMULATION_ONLY')
    objective,_=load_dino(hf['second_handoff'],[0,0,1,1])
    with np.load(paths['dinov2_vitb14']['P'],allow_pickle=False) as f:
        ix=np.r_[np.flatnonzero(f['labels']==0)[:2],np.flatnonzero(f['labels']==1)[:2]]
        z=torch.tensor(f['z'][:,ix].astype(np.float64),requires_grad=True)
    name='dinov2_vitb14';loss,_=objective.matching({name:z});grad=torch.autograd.grad(loss,z)[0]
    direct=bind_objective(objective.conditions,objective.heads,{name:torch.tensor(target[name])},[0,0,1,1])
    zz=z.detach().clone().requires_grad_(True);ll,_=direct.matching({name:zz});gg=torch.autograd.grad(ll,zz)[0]
    require(torch.equal(grad,gg) and float(loss)==float(ll) and torch.isfinite(grad).all() and grad.norm()>0,'Target/gradient roundtrip')
    result={'status':'PASS_DINO_PROTECTION_PATH','constructed':constructed_checks(),'population':population,
        'original_target_error':error,'unbounded_noiseless_error':unerr,'clipped_decode_error':cliperr,
        'max_patient_norm':float(np.linalg.norm(rows_q,axis=1).max()),
        'clipped_patients_by_class':diag['class_clipped_patients'].tolist(),
        'public_simulation_feature_gradient_error':float((grad-gg).abs().max()),
        'Q_noise_draws':0,'new_model_forwards':0,'raw_Q_V_pixel_reads':0,'optimizer_updates':0,
        'scope':'INTERNAL_NONDP_DIAGNOSTICS; not included in protected release'}
    save(root/'protection_verification.json',result)
    return result,qref
