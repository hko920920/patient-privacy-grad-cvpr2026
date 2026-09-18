"""Constructed-array checks of the proposed contrast-before-bounding argument.

No patients, encoder weights, GPU, DP noise release, or synthetic image fitting.
"""
import json,math
from pathlib import Path
import numpy as np

def unit(x):
    x=np.asarray(x,dtype=float)
    return x/max(float(np.linalg.norm(x)),1e-15)

def bounded(x,R=1.):
    x=np.asarray(x,dtype=float)
    return x/max(R,float(np.linalg.norm(x)))

def pack(r):
    d=len(r);i,j=np.triu_indices(d);a=np.outer(r,r)
    return np.concatenate([[1.],r,a[i,j]*np.where(i==j,1.,math.sqrt(2))])

def run():
    shrink=[]
    for M in (1,10,100,1000):
        p=np.array([M,1.]);n=np.array([M,-1.])
        first=(unit(p)-unit(n))/2
        after=bounded((p-n)/2)
        assert np.allclose(unit(first),after)
        shrink.append({'M':M,'endpoint_first_norm':float(np.linalg.norm(first)),
                       'after_unit_recalibration':unit(first).tolist(),'contrast_first':after.tolist()})
    # Exact information collision, stronger than a removable scalar attenuation.
    p=np.array([11.,0.]);n=np.array([9.,0.])
    collapsed=(unit(p)-unit(n))/2
    direct=bounded((p-n)/2)
    assert np.array_equal(collapsed,np.zeros(2)) and np.array_equal(direct,np.array([1.,0.]))
    # Reversing which endpoint is positive leaves both normalized datasets and
    # ANY summary of those normalized endpoints identical, but changes raw delta.
    sign_flip=bounded((n-p)/2)
    assert np.array_equal(unit(p),unit(n)) and np.array_equal(direct,-sign_flip)
    # Direction distortion can remain even after unit-length recalibration.
    common=np.array([100.,0.]);signal=np.array([1.,1.])
    p=common+signal;n=common-signal
    old_direction=unit((unit(p)-unit(n))/2);new_direction=unit((p-n)/2)
    angle=float(np.degrees(np.arccos(np.clip(old_direction@new_direction,-1,1))))
    assert angle>44
    # Bounded clipping is not unconditional normalization: inside the endpoint
    # clipping ball, the transform is linear and common shifts cancel exactly.
    R=200.
    assert np.allclose((bounded(p,R)-bounded(n,R))/2,(p-n)/(2*R))
    # Patient-local additive shifts cancel BEFORE nonlinearity, with fixed R/Pi.
    rng=np.random.default_rng(20260918);errors=[];norms=[];grad_errors=[]
    for _ in range(100):
        p=rng.normal(size=(3,16));n=rng.normal(size=(5,16));a=rng.normal(size=16)*10
        d=(p.mean(0)-n.mean(0))/2
        r=bounded(d)
        shifted=bounded(((p+a).mean(0)-(n+a).mean(0))/2)
        errors.append(float(np.max(abs(r-shifted))))
        norms.append(float(np.linalg.norm(pack(r))))
        # Class-balanced squared-loss gradient at w=0 is -d exactly.
        gradient=(-2*p.mean(0)+2*n.mean(0))/4
        grad_errors.append(float(np.max(abs(gradient+d))))
    assert max(errors)<1e-12 and max(norms)<=math.sqrt(3)+1e-12 and max(grad_errors)==0
    # Exact contrast recovery does NOT imply high cross-patient single-image AUC.
    # U,V iid Uniform[0,L], X+=U+1, X-=V-1: P(X+>X-)=.5+2/L-2/L^2.
    auc=[{'background_range':L,'exact_delta':1.,'single_image_AUC':.5+2/L-2/L**2} for L in (10.,100.,1000.)]
    result={'status':'PASS_CONSTRUCTED_ARRAY_LOGIC_ONLY','scope':'No patient/model utility evidence',
            'symmetric_example_recalibration':shrink,
            'irreversible_radial_collision':{'positive':[11.,0.],'negative':[9.,0.],
                'normalized_delta':collapsed.tolist(),'raw_delta':direct.tolist(),
                'flipped_label_raw_delta':sign_flip.tolist(),
                'normalized_release_distributions':'identical under either label-direction world',
                'restriction':'not all feature maps, clipping thresholds, or patient-level methods'},
            'direction_error_after_recalibration_degrees':angle,
            'below_endpoint_cap_difference_is_linear':True,
            'fixed_public_transform_shift_invariance_max_abs_error':max(errors),
            'relation_only_query_dimension_d16':1+16+16*17//2,
            'relation_only_sensitivity_bound':math.sqrt(3),'observed_max_query_norm':max(norms),
            'patient_balanced_squared_gradient_at_zero_equals_minus_delta_max_error':max(grad_errors),
            'perfect_relation_summary_not_high_single_image_AUC':auc,
            'new_patient_pixels':0,'model_forward':0,'DP_noise_releases':0}
    path=Path(__file__).parent/'spec_sources/contrast_before_bound_review_checks_20260918.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
