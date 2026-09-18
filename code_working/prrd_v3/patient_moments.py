"""Bounded patient-local summaries, separate all-patient and mixed populations."""
from __future__ import annotations
import math
import numpy as np
from relation_distillation.moments import patient_contributions, svec, smat
from .contracts import require

def dimension(d,mode):
    b=d*(d+1)//2+d
    return 2+2*b if mode=='point' else 3+2*b+(b if mode=='relation' else 2*d*(2*d+1)//2+2*d)

def vector(p,mode):
    require(mode in ('point','relation','joint_pair'),'Unknown moment mode')
    blocks=[np.array(list(p.present)+([] if mode=='point' else [p.mixed]),float)]
    for c in (0,1): blocks += [svec(p.seconds[c]),p.means[c]]
    if mode!='point':
        t=p.delta if mode=='relation' else p.mixed*np.concatenate([p.means[1],p.means[0]])/math.sqrt(2)
        blocks += [svec(np.outer(t,t)),t]
    q=np.concatenate(blocks)
    require(len(q)==dimension(len(p.delta),mode),'Query size')
    require(np.linalg.norm(q)<= (math.sqrt(6) if mode=='point' else 3)+1e-6,'Query norm bound')
    return q

def query(patients,mode):
    require(len({p.patient_id for p in patients})==len(patients),'Repeated patients')
    return np.stack([vector(p,mode) for p in patients]).sum(0)

def decode(q,d,mode,noisy=False):
    q=np.asarray(q,dtype=float)
    require(q.shape==(dimension(d,mode),) and np.isfinite(q).all(),'Invalid query')
    nc=2 if mode=='point' else 3; counts=q[:nc].copy()
    require(noisy or np.all(counts>=0),'Negative exact count')
    denominator=np.maximum(counts,1)  # Public floor, never consult exact Q counts.
    dims=[d,d]+([] if mode=='point' else [d if mode=='relation' else 2*d])
    k=nc; means=[]; seconds=[]
    for dim,n in zip(dims,denominator):
        size=dim*(dim+1)//2
        seconds.append(smat(q[k:k+size],dim)/n); k+=size
        means.append(q[k:k+dim]/n); k+=dim
    t={'counts':counts,'m':np.stack(means[:2]),'A':np.stack(seconds[:2]),
       'delta':np.zeros(d),'C':np.zeros((d,d))}
    if mode=='relation': t.update(delta=means[2],C=seconds[2])
    if mode=='joint_pair':
        L=np.concatenate([np.eye(d),-np.eye(d)],axis=1)/math.sqrt(2)
        t.update(u=means[2],U=seconds[2],delta=L@means[2],C=L@seconds[2]@L.T)
    return t

def aggregate(patients,mode='relation'): return decode(query(patients,mode),len(patients[0].delta),mode)

def shuffle_target(public,private,permutation):
    mixed=[p for p in private if p.mixed]; fixed=[p.delta for p in public if p.mixed]
    a=np.asarray(permutation)
    require(sorted(a.tolist())==list(range(len(mixed))) and np.all(a!=np.arange(len(mixed))),'Invalid private derangement')
    delta=np.stack(fixed+[(p.means[1]-mixed[int(j)].means[0])/2 for p,j in zip(mixed,a)])
    t=aggregate(public+private)
    before=t['delta'].copy()
    t.update(delta=delta.mean(0),C=delta.T@delta/len(delta))
    require(np.allclose(t['delta'],before,atol=1e-12,rtol=0),'Shuffle changed mean delta')
    return t

def system(t,beta=1.,ridge=.1,repair=False):
    h=(t['A'][0]+t['A'][1])/2+beta*t['C']
    v=(t['m'][1]-t['m'][0])/2+beta*t['delta']
    before=h.copy(); h=(h+h.T)/2
    values,vectors=np.linalg.eigh(h)
    if repair: h=(vectors*np.maximum(values,0))@vectors.T
    k=h+ridge*np.eye(len(v))
    require(ridge>0 and np.linalg.eigvalsh(k).min()>0,'Unstable readout')
    counts=t['counts']
    constant=.25*sum(counts[:2]>0)+.5*beta*(len(counts)>2 and counts[2]>0)
    return k,v,float(constant),{'minimum_H_eigenvalue':float(values.min()),'repair_fro':float(np.linalg.norm(h-before)),
                                'condition_number':float(np.linalg.cond(k))}

def solve(t,beta=1.,ridge=.1,repair=False):
    k,v,_,info=system(t,beta,ridge,repair)
    return np.linalg.solve(k,v),info

