"""Public-only / independent-noise feature follow-up.
Only population selection, contract validation and export metadata vary.
The feature objective and DINO optimizer/checkpoint kernel remain frozen.
"""
import argparse,json,time
from pathlib import Path
from datetime import datetime
from prrd_v3.contracts import read,sha,require
from prrd_v3.bank_runtime import atomic_json
from . import feature_control as f
from . import feature_runtime as previous
from . import dino_control_runtime as base

def code_bindings():
    r=previous.code_bindings()
    r[str(Path(__file__))]=sha(__file__)
    return r

def load_feature(handoff,labels,population='pooled',device='cpu'):
    h=read(handoff)
    require(h['population']==population,'Population mismatch in feature target')
    expected='PUBLIC_ONLY_NO_Q' if population=='P' else 'PATIENT_DP_SINGLE_QUERY_INTERNAL'
    require(h['privacy']==expected,'Wrong target privacy/population')
    # The old numerical loader has a historical pooled-only argument.
    # Population is explicitly checked above; its numerical operation is unchanged.
    return f.load_feature(handoff,labels,'pooled',device)

def validate_job(job):
    require(job['schema']=='receiver.feature-followup-s200-bank/v1','Wrong follow-up schema')
    require(job['variant'] in ('PUBLIC','DP8') and job['source_set']=='DINO_FEATURE','Wrong arm')
    population='P' if job['variant']=='PUBLIC' else 'pooled'
    require(job['population']==population and job['seed']==101 and job['artifact_kind']=='MAIN_BANK','Wrong recipe')
    require(all(sha(p)==v for p,v in job['code_bindings'].items()),'Runtime changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Contract changed')
    c=read(job['campaign_contract']);ref=read(c['reference_job'])
    require(c['status']=='AUTHORIZED_FROZEN' and c['max_private_releases']==1 and c['max_banks']==2,'Scope')
    require(job['id']==c['bank_ids'][job['variant']],'Wrong bank')
    for key in ('images','updates','microbatch','activation_updates','optimizer','templates',
                'A2_job','A2_bank','A2_initial_receipt','A2_template_tensor_digest'):
        require(job[key]==ref[key],'Matched initialization/schedule changed: '+key)
    require(job['maximum_seconds']==3600 and job['absolute_deadline_utc']==c['absolute_deadline_utc'],'Budget')
    require(time.time()<datetime.fromisoformat(c['absolute_deadline_utc']).timestamp(),'Campaign cap')
    require(sha(job['second_handoff'])==job['second_handoff_sha256'],'Target changed')
    h=read(job['second_handoff']);m=read(h['mechanism_file'])
    require(h['schema']==f.SCHEMA and h['population']==population,'Handoff population')
    require(h['class_bounds']==c['mechanism']['class_bounds'],'Class clipping changed')
    if job['variant']=='DP8':
        require(h['privacy']=='PATIENT_DP_SINGLE_QUERY_INTERNAL' and m==c['mechanism'],'Protection changed')
    else:
        require(h['privacy']=='PUBLIC_ONLY_NO_Q' and m['private_queries']==0 and m['Q_used'] is False,'Public target includes Q')
    for k in ('rule','target','mechanism'):
        require(sha(h[k+'_file'])==h[k+'_sha256'],'Target dependency changed')
    require(c['loss_definition']==f.LOSS_RULE,'Loss changed')
    require(all(sha(p)==v for p,v in c['historical_dependencies'].items()),'Historical evidence changed')
    return (0,)*64+(1,)*64

def export(directory,renderer,job,spec,signature,checkpoint):
    dp=job['variant']=='DP8'
    def write(path,value,**kwargs):
        if Path(path).name in ('learning_contract.json','provenance_public.json'):
            value=dict(value,DP_applied=dp,source_objective='class-conditional K4 feature-mean half-squared normalized L2',
                privacy_scope='INTERNAL_AUDIT_PACKAGE; shareable PNG package separate',
                raw_Q_access_during_synthesis=False,feature_loss_definition=f.LOSS_RULE,
                data_population=job['population'],protected_queries=int(dp),
                patient_dp=({'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient',
                    'query_dimension':130,'protected_queries':1,'development_selection_protected':False} if dp else None))
        return atomic_json(path,value,**kwargs)
    return f.clone(base.export,atomic_json=write)(directory,renderer,job,spec,signature,checkpoint)

_kernel=f.clone(base._run,load_dino=load_feature,export=export)
def _run(job,directory,labels,profile,resume,started,deadline):
    remaining=datetime.fromisoformat(job['absolute_deadline_utc']).timestamp()-time.time()
    require(remaining>60,'Insufficient remaining time for safe update')
    return _kernel(job,directory,labels,profile,resume,started,min(deadline,time.monotonic()+remaining))
execute=f.clone(base.execute,validate_job=validate_job,_run=_run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    r=execute(read(a.job),a.output,resume=a.resume)
    atomic_json(a.result,r);print(json.dumps(r),flush=True)

