"""Disjoint real/synthetic pool namespaces; v1 collision retained in frozen source."""
from .common import seed,require,METHODS
from .data import pools,draw,letterbox,preprocess_batch

def source_pools(real,generated):
    result={'real/'+role:pools([r for r in real if r['role']==role]) for role in ('public','private')}
    result.update({'synthetic/'+method:pools([r for r in generated if r['method']==method]) for method in METHODS})
    require(len(result)==6,'Six distinct real/synthetic namespaces')
    return result

def batch_records(source_pools,arm,run_seed,step):
    shared=draw(source_pools['real/public'],16,seed(run_seed,step,'public-shared'))
    if arm in ('R0','R1'):
        other=draw(source_pools['real/public'],16,seed(run_seed,step,'public-extra'))
    elif arm=='Dreal':
        other=draw(source_pools['real/private'],16,seed(run_seed,step,'private-extra'))
    else:
        method=dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]
        other=draw(source_pools['synthetic/'+method],16,seed(run_seed,step,'synthetic-extra'))
    return shared+other
