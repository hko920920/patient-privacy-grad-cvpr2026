"""Point-preserving patient statistics and explicit pre-normalization relations.

No image access, readout, optimizer, privacy release or calibration on Q/V.
Legacy patient_moments remains available for reproducing the old objective.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from .contracts import require
from .patient_moments import patient_contributions, aggregate

RELATION_ARMS=('E','E_R','C','D','R_joint')


def patient_uniform_center(h,patient_ids):
    h=np.asarray(h,dtype=np.float64);ids=np.asarray(patient_ids,dtype=str)
    require(h.ndim==2 and len(h)>0 and ids.shape==(len(h),) and np.isfinite(h).all(),
            'Invalid raw feature matrix')
    require(np.all(ids!=''),'Empty patient identifier')
    unique,counts=np.unique(ids,return_counts=True)
    sizes=dict(zip(unique,counts))
    weights=np.array([1/(len(unique)*sizes[p]) for p in ids],dtype=np.float64)
    return (h*weights[:,None]).sum(0),weights


@dataclass(frozen=True)
class PatientPaths:
    point: object
    raw_means: np.ndarray  # means of a=(h-mu_raw)@SAME_POINT_PCA_AXES
    role: str


def patient_paths(z,a,labels,patient_ids,roles):
    z=np.asarray(z,dtype=np.float64);a=np.asarray(a,dtype=np.float64)
    ids=np.asarray(patient_ids,dtype=str);y=np.asarray(labels);role=np.asarray(roles,dtype=str)
    require(a.shape==z.shape and np.isfinite(a).all(),'Point/raw affine feature mismatch')
    require(role.shape==(len(z),) and set(role)<= {'P','Q','S'},'Invalid training role')
    points=patient_contributions(z,y,ids)
    result=[]
    for p in points:
        selected=ids==p.patient_id;patient_roles=set(role[selected])
        require(len(patient_roles)==1,'A patient crosses role boundaries')
        means=np.zeros_like(p.means)
        for c in (0,1):
            rows=a[selected & (y==c)]
            if len(rows):means[c]=rows.mean(0)
        result.append(PatientPaths(p,means,next(iter(patient_roles))))
    return result


def bound(value,radius):
    x=np.asarray(value,dtype=np.float64)
    require(x.ndim>=1 and np.isfinite(x).all() and np.isfinite(radius) and radius>0,
            'Nonfinite vector or nonpositive bound scale')
    return x/np.maximum(float(radius),np.linalg.norm(x,axis=-1,keepdims=True))


def torch_bound(value,radius):
    require(math.isfinite(float(radius)) and radius>0,'Invalid fixed public scale')
    return value/torch.linalg.vector_norm(value,dim=-1,keepdim=True).clamp_min(float(radius))


def paired_relation(z_positive,z_negative,a_positive,a_negative,arm,scales):
    """Differentiable endpoint kernel, ready for later image-loss integration.

    Inputs can be patient centroids or synthetic paired features. A D permutation
    is applied to endpoints upstream, never to the public scale or point branch.
    """
    require(arm in RELATION_ARMS,'Unknown relation arm')
    if arm=='E':return {'r':(z_positive-z_negative)/2}
    if arm=='E_R':return {'r':torch_bound((z_positive-z_negative)/2,scales['E_R'])}
    if arm in ('C','D'):return {'r':torch_bound((a_positive-a_negative)/2,scales['C'])}
    u=torch_bound(torch.cat([a_positive,a_negative],dim=-1)/math.sqrt(2),scales['R_joint'])
    d=a_positive.shape[-1]
    return {'u':u,'r':(u[...,:d]-u[...,d:])/math.sqrt(2)}


def public_scales(patients):
    require(len(patients)>0 and all(p.role=='P' for p in patients),'Scale calibration is P-only')
    mixed=[p for p in patients if p.point.mixed]
    rows={'E_R':[],'C':[],'R_joint':[]}
    for p in mixed:
        rows['E_R'].append(np.linalg.norm((p.point.means[1]-p.point.means[0])/2))
        rows['C'].append(np.linalg.norm((p.raw_means[1]-p.raw_means[0])/2))
        rows['R_joint'].append(np.linalg.norm(np.concatenate([p.raw_means[1],p.raw_means[0]])/math.sqrt(2)))
    # Equal mixed-patient masses, inverse empirical CDF, NO interpolation.
    # For the current public M=3 this is the maximum norm.
    index=max(0,math.ceil(.95*len(mixed))-1)
    values={k:max(float(np.sort(v)[index]) if len(v) else 0.,1e-12) for k,v in rows.items()}
    return values,{'method':'equal-mixed-patient inverse empirical CDF at .95; no interpolation',
                   'floor':1e-12,'mixed_patients':len(mixed),'norms':rows,'labels_used_for_mixed_membership':True,
                   'point_PCA_refitted':False,'optimality_claim':False}


def relation_rows(patients,arm,scales,permutation=None):
    require(arm in RELATION_ARMS and len(patients)>0,'Invalid relation request')
    require(len({p.point.patient_id for p in patients})==len(patients),'Repeated patient')
    mixed=[p for p in patients if p.point.mixed];d=len(patients[0].point.delta)
    private=[j for j,p in enumerate(mixed) if p.role=='Q']
    destinations=list(range(len(mixed)))
    if arm=='D':
        if private:
            perm=np.asarray(permutation)
            require(perm.shape==(len(private),) and np.issubdtype(perm.dtype,np.integer)
                    and sorted(perm.tolist())==list(range(len(private)))
                    and np.all(perm!=np.arange(len(private))),'D requires a private-mixed derangement')
            for k,j in enumerate(private):destinations[j]=private[int(perm[k])]
        else:require(permutation is None or len(permutation)==0,'Cannot shuffle public patients')
    else:require(permutation is None,'Permutation is only defined for D')
    pre=[];rs=[];us=[]
    for j,p in enumerate(mixed):
        ap=p.raw_means[1];an=mixed[destinations[j]].raw_means[0]
        delta=(ap-an)/2;pre.append(delta)
        if arm=='E':r=(p.point.means[1]-p.point.means[0])/2
        elif arm=='E_R':r=bound((p.point.means[1]-p.point.means[0])/2,scales['E_R'])
        elif arm in ('C','D'):r=bound(delta,scales['C'])
        else:
            u=bound(np.concatenate([ap,an])/math.sqrt(2),scales['R_joint']);us.append(u)
            r=(u[:d]-u[d:])/math.sqrt(2)
        require(np.linalg.norm(r)<=1+1e-7,'Relation norm exceeds one')
        rs.append(r)
    return {'patient_ids':np.array([p.point.patient_id for p in mixed],dtype=str),
            'raw_delta':np.asarray(pre,dtype=np.float64).reshape(-1,d),
            'r':np.asarray(rs,dtype=np.float64).reshape(-1,d),
            'u':np.asarray(us,dtype=np.float64).reshape(-1,2*d)}


def relation_target(patients,arm,scales,permutation=None):
    target=aggregate([p.point for p in patients],mode='point')
    rr=relation_rows(patients,arm,scales,permutation);m=len(rr['r']);d=len(patients[0].point.delta)
    target['counts']=np.append(target['counts'],m)
    target['delta']=rr['r'].mean(0) if m else np.zeros(d)
    target['C']=rr['r'].T@rr['r']/max(m,1)
    if arm=='R_joint':
        target['u']=rr['u'].mean(0) if m else np.zeros(2*d)
        target['U']=rr['u'].T@rr['u']/max(m,1)
    return target,rr
