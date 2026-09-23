"""Patient-level bounded A2 signal query and Gaussian target post-processing.

No file access, encoder execution, private count chosen as public denominator,
or privacy spend occurs on import. Raw contributions/diagnostics are INTERNAL.
The real-arithmetic proof is for one full-vector Gaussian query. Numerical
checks do not constitute a finite-precision security proof.
"""
from dataclasses import dataclass
import math
import random
import numpy as np
from relation_distillation.moments import analytic_gaussian_std

ENCODERS=('biovil','dinov2_vitb14')
K=4
G=34


def finite(value,name):
    a=np.asarray(value,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError(name+' must be finite')
    return a


def scalar(value,name,*,positive=True):
    if isinstance(value,(bool,np.bool_)):raise ValueError(name+' cannot be boolean')
    v=float(value)
    if not math.isfinite(v) or (v<=0 if positive else v<0):raise ValueError('Invalid '+name)
    return v


def sum_patient_axis(values):
    """Accurately sum patient rows, without changing their weights/order."""
    a=finite(values,'patient aggregate')
    if a.ndim<2:raise ValueError('Patient row tensor required')
    width=math.prod(a.shape[1:])
    return np.array([math.fsum(column) for column in a.reshape(len(a),width).T]).reshape(a.shape[1:])


@dataclass(frozen=True)
class Limits:
    class_bounds: tuple
    def __post_init__(self):
        if len(self.class_bounds)!=2:raise ValueError('Two class bounds required')
        for v in self.class_bounds:scalar(v,'class bound')
    @property
    def dimension(self):return 2*len(ENCODERS)*K*G+2
    @property
    def sensitivity_add_remove(self):return 1.0


@dataclass(frozen=True)
class Patients:
    ids: tuple
    means: np.ndarray  # [patient, class, encoder, condition, coordinate]
    present: np.ndarray  # [patient, class]
    def validate(self):
        n=len(self.ids);m=finite(self.means,'patient means');b=np.asarray(self.present)
        if len(set(self.ids))!=n:raise ValueError('Duplicate patient rows')
        if m.shape!=(n,2,len(ENCODERS),K,G) or b.shape!=(n,2) or b.dtype!=np.bool_:
            raise ValueError('Wrong patient tensor shape/dtype')
        if n and (not b.any(axis=1).all() or np.any(m[~b]!=0)):
            raise ValueError('Absent class must have zero signal; each patient must have a visit')
        return self
    def sums_counts(self):
        self.validate()
        return sum_patient_axis(self.means),self.present.sum(0).astype(np.float64)


def aggregate_image_signals(signals,labels,patient_ids):
    """Each patient/class visits average once; no class-count normalization yet."""
    a=finite(signals,'image signals');y=np.asarray(labels);ids=np.asarray(patient_ids)
    n=len(y)
    if a.shape!=(n,len(ENCODERS),K,G) or y.shape!=(n,) or ids.shape!=(n,):
        raise ValueError('Image signal/metadata shape mismatch')
    if y.dtype.kind not in 'iu' or not np.isin(y,[0,1]).all():raise ValueError('Binary integer labels required')
    if ids.dtype.kind not in 'iuUS' or any(not str(p) for p in ids):raise ValueError('Stable patient IDs required')
    people=tuple(sorted(set(ids.tolist())))
    means=np.zeros((len(people),2,len(ENCODERS),K,G),dtype=np.float64)
    present=np.zeros((len(people),2),dtype=bool)
    for j,p in enumerate(people):
        for c in (0,1):
            select=(ids==p)&(y==c)
            if select.any():means[j,c]=a[select].mean(0);present[j,c]=True
    return Patients(people,means,present).validate()


def public_limits(public_patients):
    """Fixed P-only class norm q95, NumPy linear quantile, floor1e-6."""
    public_patients.validate();bounds=[];details=[]
    for c in (0,1):
        values=public_patients.means[public_patients.present[:,c],c].reshape(-1,len(ENCODERS)*K*G)
        if not len(values):raise ValueError('Public calibration requires each class')
        norms=np.linalg.norm(values,axis=1)
        if not np.isfinite(norms).all():raise ValueError('Public norm overflow')
        q=float(np.quantile(norms,.95,method='linear'));bounds.append(max(1e-6,q))
        details.append({'class':c,'public_class_patients':len(values),'q95':q,'bound':bounds[-1]})
    return Limits(tuple(bounds)),details


def clipped_contributions(patients,limits):
    patients.validate();n=len(patients.ids)
    vectors=patients.means.reshape(n,2,len(ENCODERS)*K*G)
    norms=np.linalg.norm(vectors,axis=-1)
    if not np.isfinite(norms).all():raise ValueError('Signal norm overflow')
    bounds=np.asarray(limits.class_bounds)
    factors=np.ones_like(norms)
    np.divide(bounds[None,:],norms,out=factors,where=norms>bounds[None,:])
    normalized=vectors*factors[:,:,None]/bounds[None,:,None]
    rows=.5*np.concatenate([normalized.reshape(n,2*len(ENCODERS)*K*G),patients.present.astype(float)],axis=1)
    sizes=np.linalg.norm(rows,axis=1)
    # Public FP margin; does not change the sensitivity bound1.
    radius=1-16*np.finfo(np.float64).eps
    last=np.minimum(1.,radius/np.maximum(sizes,np.finfo(float).tiny))
    rows*=last[:,None]
    if np.any(np.linalg.norm(rows,axis=1)>1.):raise FloatingPointError('Contribution bound violated')
    return rows,{'scope':'INTERNAL_NONPRIVATE_DO_NOT_RELEASE','class_norms':norms,
                 'class_factors':factors,'joint_factors':last,
                 'class_clipped_patients':(factors<1).sum(0),
                 'joint_roundoff_limited_patients':int((last<1).sum())}


def raw_balanced_target(public_sums,public_counts,private_sums,private_counts):
    ps=finite(public_sums,'public sums');qs=finite(private_sums,'private sums')
    pc=finite(public_counts,'public counts');qc=finite(private_counts,'private counts')
    if ps.shape!=(2,len(ENCODERS),K,G) or qs.shape!=ps.shape or pc.shape!=(2,) or qc.shape!=(2,):
        raise ValueError('Wrong sum/count shape')
    if np.any(pc<0) or np.any(qc<0):raise ValueError('Raw count cannot be negative')
    if np.any(ps[pc==0]!=0) or np.any(qs[qc==0]!=0):raise ValueError('Zero mass has nonzero raw sum')
    return .5*((ps+qs)/np.maximum(1.,pc+qc)[:,None,None,None]).sum(0)


def mechanism_metadata(limits,epsilon=8.,delta=1e-5,adjacency='add_remove_one_patient'):
    eps=scalar(epsilon,'epsilon');de=scalar(delta,'delta')
    if de>=1:raise ValueError('delta must be below1')
    # Supported numerical interval; no unsupported extreme-parameter promises.
    if not(.01<=eps<=64 and 1e-12<=de<=.1):raise ValueError('Outside validated calibration interval')
    if adjacency not in ('add_remove_one_patient','replace_one_patient'):raise ValueError('Unknown adjacency')
    sensitivity=1. if adjacency=='add_remove_one_patient' else 2.
    std=analytic_gaussian_std(eps,de,sensitivity)*(1+1e-10)
    return {'schema':'receiver.patient-gaussian/v1','epsilon':eps,'delta':de,
            'adjacency':adjacency,'sensitivity':sensitivity,'dimension':limits.dimension,
            'class_bounds':list(limits.class_bounds),'sigma':std,
            'calibration':'Balle-Wang analytic; conservative1e-10 multiplicative margin',
            'query':'full class-gradient sums plus two class masses',
            'private_queries':1,'subsampling':False,'count_noise_std':2*std,
            'class_gradient_sum_noise_std':[2*c*std for c in limits.class_bounds]}


def _validate_query(query,limits):
    q=finite(query,'query')
    if q.shape!=(limits.dimension,):raise ValueError('Wrong joint query dimension')
    return q


def privatize_patients(patients,limits,*,epsilon=8.,delta=1e-5,adjacency='add_remove_one_patient'):
    """One fresh vector; no seed, raw sum, counts or realized noise returned.

    OS randomness; two independent normalvariate draws per coordinate as in
    the public diffprivlib Gaussian sampler. Fixed-seed testing uses the
    separate simulate_public function. Do not serialize intermediate locals.
    """
    metadata=mechanism_metadata(limits,epsilon,delta,adjacency)
    rows,_=clipped_contributions(patients,limits)
    total=sum_patient_axis(rows)
    rng=random.SystemRandom()
    noise=np.array([(rng.normalvariate(0.,1.)+rng.normalvariate(0.,1.))/math.sqrt(2.)
                    for _ in range(limits.dimension)])
    value=finite(total+metadata['sigma']*noise,'noisy query')
    return value,metadata


def simulate_public(query,limits,standard_normal,*,epsilon=8.,delta=1e-5):
    """PUBLIC/CONSTRUCTED ARRAY TEST ONLY; never a Q release API."""
    q=_validate_query(query,limits);z=finite(standard_normal,'test noise')
    if z.shape!=q.shape:raise ValueError('Wrong supplied noise shape')
    metadata=mechanism_metadata(limits,epsilon,delta)
    return q+metadata['sigma']*z,metadata


def decode_target(noisy_query,limits,public_sums,public_counts):
    """Only DP value + P. No raw Q counts, sums, features, or noise-redraw."""
    q=_validate_query(noisy_query,limits);ps=finite(public_sums,'public sums');pc=finite(public_counts,'public counts')
    if ps.shape!=(2,len(ENCODERS),K,G) or pc.shape!=(2,) or np.any(pc<0):
        raise ValueError('Public shape/count invalid')
    if np.any(ps[pc==0]!=0):raise ValueError('Invalid public zero class')
    bound=np.asarray(limits.class_bounds)
    mass=np.maximum(0.,2*q[-2:])
    vectors=2*q[:-2].reshape(2,-1)*bound[:,None]
    norm=np.linalg.norm(vectors,axis=1);capacity=bound*mass
    if not np.isfinite(norm).all() or not np.isfinite(capacity).all():raise ValueError('Decoded overflow')
    factor=np.minimum(1.,capacity/np.maximum(norm,np.finfo(float).tiny))
    vectors*=factor[:,None]
    sums=vectors.reshape(ps.shape)
    denominator=np.maximum(1.,pc+mass)
    target=.5*((ps+sums)/denominator[:,None,None,None]).sum(0)
    public=.5*(ps/np.maximum(1.,pc)[:,None,None,None]).sum(0)
    if np.any(np.linalg.norm(public,axis=-1)<=1e-12):raise ValueError('Public fallback signal is zero')
    fallback=np.linalg.norm(target,axis=-1)<=1e-12
    target[fallback]=public[fallback]
    if not np.isfinite(target).all():raise ValueError('Nonfinite target')
    return {name:target[i].copy() for i,name in enumerate(ENCODERS)},{
        'scope':'POSTPROCESSING_OF_PROTECTED_QUERY_AND_P','nonnegative_Q_mass':mass.tolist(),
        'pooled_denominator':denominator.tolist(),'consistency_projection_factor':factor.tolist(),
        'public_fallback_conditions':fallback.tolist()}
