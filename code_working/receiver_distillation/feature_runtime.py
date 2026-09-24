"""Feature-objective adapter around the verified DINO main loop."""
import argparse,time,json
from pathlib import Path
from datetime import datetime
from prrd_v3.contracts import read,sha,require
from prrd_v3.bank_runtime import atomic_json
from . import dino_control_runtime as base
from .patient_dp_runtime import clone
from .feature_control import load_feature,LOSS_RULE,SCHEMA

def code_bindings():
    r=base.code_bindings()
    for name in ('feature_control.py','feature_runtime.py','patient_dp_dino.py','patient_dp_runtime.py'):
        p=Path(__file__).with_name(name);r[str(p)]=sha(p)
    return r

def validate_job(job):
    require(job['schema']=='receiver.feature-patient-dp-s200-bank/v1','Wrong feature job')
    require(job['variant']=='DP8' and job['source_set']=='DINO_FEATURE','Wrong signal')
    require(job['seed']==101 and job['population']=='pooled' and job['artifact_kind']=='MAIN_BANK','Wrong recipe')
    require(all(sha(p)==v for p,v in job['code_bindings'].items()),'Runtime changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Contract changed')
    c=read(job['campaign_contract']);ref=read(c['reference_job'])
    require(c['status']=='AUTHORIZED_FROZEN' and c['max_private_releases']==1,'Scope')
    require(job['id']==c['bank_ids']['DP8'],'Wrong bank')
    for key in ('images','updates','microbatch','activation_updates','optimizer','templates',
                'A2_job','A2_bank','A2_initial_receipt','A2_template_tensor_digest'):
        require(job[key]==ref[key],'Matched initialization/schedule changed: '+key)
    require(job['maximum_seconds']==3600 and job['absolute_deadline_utc']==c['absolute_deadline_utc'],'Budget')
    require(time.time()<datetime.fromisoformat(c['absolute_deadline_utc']).timestamp(),'Campaign cap')
    require(sha(job['second_handoff'])==job['second_handoff_sha256'],'Target changed')
    h=read(job['second_handoff']);m=read(h['mechanism_file'])
    require(h['schema']==SCHEMA and h['privacy']=='PATIENT_DP_SINGLE_QUERY_INTERNAL','Unprotected target')
    require(m==c['mechanism'] and m['dimension']==130 and m['epsilon']==8
            and m['delta']==1e-5 and m['private_queries']==1 and m['sensitivity']==1
            and m['adjacency']=='add_remove_one_patient','Mechanism changed')
    require(c['loss_definition']==LOSS_RULE,'Loss changed')
    require(all(sha(p)==v for p,v in c['historical_dependencies'].items()),'Historical evidence changed')
    return (0,)*64+(1,)*64

def export(directory,renderer,job,spec,signature,checkpoint):
    def write(path,value,**kwargs):
        if Path(path).name in ('learning_contract.json','provenance_public.json'):
            value=dict(value,DP_applied=True,source_objective='class-conditional K4 feature-mean half-squared normalized L2',
                       privacy_scope='INTERNAL_AUDIT_PACKAGE; only protected_png is release candidate',
                       raw_Q_access_during_synthesis=False,feature_loss_definition=LOSS_RULE,
                       patient_dp={'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient',
                                   'query_dimension':130,'protected_queries':1,'development_selection_protected':False})
        return atomic_json(path,value,**kwargs)
    return clone(base.export,atomic_json=write)(directory,renderer,job,spec,signature,checkpoint)

_kernel=clone(base._run,load_dino=load_feature,export=export)
def _run(job,directory,labels,profile,resume,started,deadline):
    remaining=datetime.fromisoformat(job['absolute_deadline_utc']).timestamp()-time.time()
    require(remaining>60,'Insufficient remaining time for safe update')
    return _kernel(job,directory,labels,profile,resume,started,min(deadline,time.monotonic()+remaining))
execute=clone(base.execute,validate_job=validate_job,_run=_run)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--result',type=Path,required=True)
    a.add_argument('--resume',action='store_true');v=a.parse_args()
    result=execute(read(v.job),v.output,resume=v.resume)
    atomic_json(v.result,result);print(json.dumps(result),flush=True)

