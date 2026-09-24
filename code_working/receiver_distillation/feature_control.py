"""Patient-DP feature-mean control; immutable gradient-DP sources are reused.

The signal is patient/class mean of K4 DINO PCA16 features. A class vector
is clipped across its four conditions. Public and synthetic signals use the
same clipping map. The loss averages classes of half squared normalized L2.
"""
from dataclasses import dataclass
from pathlib import Path
import types
import numpy as np
import torch
from prrd_v3.contracts import read, save, sha, require
from prrd_v3.bank_runtime import digest
from . import patient_dp_dino as old
from .signals import conditions
from .second_source import ENCODER_ID

K, G = 4, 16
SCHEMA = 'receiver.patient-dp-feature-mean/v1'
LOSS_RULE = {
    'class_reduction': 'mean of two half-squared L2 values',
    'condition_reduction': 'sum across four separate conditions',
    'coordinate_reduction': 'sum across sixteen coordinates',
    'normalization': 'each class K4 vector divided by its P-only clip bound',
    'formula': '0.25 * sum_c,k,j ((synthetic_mean[c,k,j]-target[c,k,j])/C[c])**2',
    'public_private_synthetic_clipping': 'visit mean per patient/class, then clip full K4 vector',
    'clip_derivative': 'differentiate actual radial projection; no detached scale',
    'image_prior': {'anchor': .01, 'TV': .0001, 'multiplicity': 1},
    'nominal_range': [0., 2.],
    'same_gradient_norm_as_cosine_claimed': False,
}


def clone(f, **changes):
    ns = dict(f.__globals__); ns.update(changes)
    result=types.FunctionType(f.__code__, ns, f.__name__, f.__defaults__, f.__closure__)
    result.__kwdefaults__=None if f.__kwdefaults__ is None else dict(f.__kwdefaults__)
    return result


class Limits(old.Limits):
    @property
    def dimension(self): return 130


class Patients(old.Patients):
    def validate(self):
        n=len(self.ids); m=old.finite(self.means,'patient features'); b=np.asarray(self.present)
        if len(set(self.ids))!=n: raise ValueError('Duplicate patient')
        if m.shape!=(n,2,1,K,G) or b.shape!=(n,2) or b.dtype!=np.bool_:
            raise ValueError('Feature patient shape/dtype')
        if n and (not b.any(axis=1).all() or np.any(m[~b]!=0)):
            raise ValueError('Missing-class feature must be zero')
        return self


aggregate_image_signals = clone(old.aggregate_image_signals, G=G, Patients=Patients)
public_limits = clone(old.public_limits, G=G, Limits=Limits)
clipped_contributions = clone(old.clipped_contributions, G=G)


def mechanism_metadata(limits,epsilon=8.,delta=1e-5,adjacency='add_remove_one_patient'):
    m=old.mechanism_metadata(limits,epsilon,delta,adjacency)
    m['schema']=SCHEMA
    m['query']='full class-conditional K4 feature sums and two class masses'
    m['class_feature_sum_noise_std']=m.pop('class_gradient_sum_noise_std')
    return m


privatize_patients = clone(old.privatize_patients,
    clipped_contributions=clipped_contributions, mechanism_metadata=mechanism_metadata)


def bounded_sums(patients,limits):
    rows,info=clipped_contributions(patients,limits)
    total=old.sum_patient_axis(rows)
    counts=2*total[-2:]
    sums=(2*total[:-2].reshape(2,K,G)*np.asarray(limits.class_bounds)[:,None,None])
    return sums,counts,info


def decode_target(query,limits,public_sums,public_counts):
    q=old.finite(query,'protected feature query')
    ps=old.finite(public_sums,'public feature sums')
    pc=old.finite(public_counts,'public counts')
    if q.shape!=(130,) or ps.shape!=(2,K,G) or pc.shape!=(2,) or np.any(pc<0):
        raise ValueError('Feature decode shape/count')
    if np.any(ps[pc==0]!=0): raise ValueError('Nonzero public sum with absent class')
    bounds=np.asarray(limits.class_bounds)
    mass=np.maximum(0.,2*q[-2:])
    sums=2*q[:-2].reshape(2,K,G)*bounds[:,None,None]
    norms=np.linalg.norm(sums.reshape(2,-1),axis=1)
    if not np.isfinite(norms).all(): raise ValueError('Feature decode overflow')
    factors=np.minimum(1.,bounds*mass/np.maximum(norms,np.finfo(float).tiny))
    sums*=factors[:,None,None]
    denominator=np.maximum(1.,pc+mass)
    target=(ps+sums)/denominator[:,None,None]
    if not np.isfinite(target).all(): raise ValueError('Nonfinite feature target')
    return target,{'scope':'DP_POSTPROCESSING_PLUS_PUBLIC_P',
                  'nonnegative_Q_mass':mass.tolist(),'denominator':denominator.tolist(),
                  'consistency_projection_factor':factors.tolist()}


def read_population(item,role):
    require(role in ('P','Q'),'Invalid cache role')
    path=Path(item[role])
    with np.load(path,allow_pickle=False) as d:
        require(np.array_equal(d['condition_ids'],np.arange(K)),'Conditions changed')
        require(str(d['rule_sha256'])==digest(item['rule']),'Cache rule changed')
        require(np.all(d['roles']==role),'Cache role mismatch')
        z=d['z'].astype(np.float64)
        meta={k:d[k].copy() for k in ('image_ids','patient_ids','labels')}
    require(z.shape==(K,len(meta['labels']),G) and np.isfinite(z).all(),'Invalid feature cache')
    p=aggregate_image_signals(z.transpose(1,0,2)[:,None],meta['labels'],meta['patient_ids'])
    return p,meta,z


@dataclass(frozen=True)
class FeatureObjective:
    conditions: tuple
    targets: dict
    labels: tuple
    bounds: tuple
    @property
    def heads(self): return {ENCODER_ID: ()}

    def signal(self,z):
        # z[K,N,D]. Each synthetic image is one virtual single-class patient.
        x=z.double().permute(1,0,2)
        y=torch.as_tensor(self.labels,device=x.device)
        pieces=[]
        for c in (0,1):
            part=x[y==c]
            if len(part)==0:
                pieces.append(x.sum(0)*0)
                continue
            norm=torch.linalg.vector_norm(part.flatten(1),dim=1)
            # max(norm,C)>0 even at norm=0; no unused 0/0 branch.
            factor=self.bounds[c]/norm.clamp_min(self.bounds[c])
            pieces.append((part*factor[:,None,None]).mean(0))
        return torch.stack(pieces)

    def matching(self,features):
        require(set(features)=={ENCODER_ID},'Only fixed DINO source')
        z=features[ENCODER_ID]
        require(z.shape==(K,len(self.labels),G),'Feature objective shape')
        t=self.targets[ENCODER_ID]
        require(t.shape==(2,K,G) and not t.requires_grad,'Feature target shape/frozen')
        scale=z.new_tensor(self.bounds,dtype=torch.float64)[:,None,None]
        loss=.25*((self.signal(z)-t)/scale).square().sum()
        return loss,{ENCODER_ID:loss}


def objective(target,bounds,labels,device='cpu'):
    lim=Limits(tuple(float(x) for x in bounds))
    t=torch.as_tensor(target,dtype=torch.float64,device=device).detach().clone()
    require(t.shape==(2,K,G) and bool(torch.isfinite(t).all()),'Invalid feature target')
    require(set(labels)<={0,1},'Invalid labels')
    return FeatureObjective(conditions(),{ENCODER_ID:t},tuple(labels),lim.class_bounds)


def write_target(directory,target,bounds,rule,metadata,privacy):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    rule=dict(rule,signal='class-conditional feature means',gradient_layout=None,
              feature_objective=LOSS_RULE,feature_class_bounds=list(bounds))
    save(directory/'rule.json',rule);save(directory/'mechanism.json',metadata)
    np.savez(directory/'target.npz',target=target,schema=np.array(SCHEMA))
    handoff={'schema':SCHEMA,'population':'pooled','privacy':privacy,
        'target_file':str(directory/'target.npz'),'target_sha256':sha(directory/'target.npz'),
        'rule_file':str(directory/'rule.json'),'rule_sha256':sha(directory/'rule.json'),
        'mechanism_file':str(directory/'mechanism.json'),'mechanism_sha256':sha(directory/'mechanism.json'),
        'class_bounds':list(bounds)}
    save(directory/'handoff.json',handoff)
    return str(directory/'handoff.json')


def load_feature(handoff,labels,population='pooled',device='cpu'):
    h=read(handoff);require(h['schema']==SCHEMA and population=='pooled','Wrong feature handoff')
    for key in ('target','rule','mechanism'):
        require(sha(h[key+'_file'])==h[key+'_sha256'],'Feature handoff changed: '+key)
    rule=read(h['rule_file'])
    require(rule['feature_objective']==LOSS_RULE and rule['feature_class_bounds']==h['class_bounds'],
            'Feature objective contract changed')
    require(sha(rule['checkpoint'])==rule['checkpoint_sha256']
            and sha(rule['projection'])==rule['projection_sha256'],'Encoder/projection changed')
    with np.load(h['target_file'],allow_pickle=False) as d: t=d['target'].copy()
    return objective(t,h['class_bounds'],labels,device),{ENCODER_ID:rule}
