"""DP/clip job adapter. Reuses the frozen A2 training/replay/export bytecode.

Only job validation, the absolute campaign deadline and artifact metadata are
adapted. Internal provenance/checkpoint hashes are NOT a protected release.
"""
import argparse,json,time,types
from pathlib import Path
from datetime import datetime
from prrd_v3.contracts import read,require,sha
from prrd_v3.bank_runtime import atomic_json
from . import repeat_runtime as base


def clone(function,**changes):
    namespace=dict(function.__globals__);namespace.update(changes)
    return types.FunctionType(function.__code__,namespace,function.__name__,
                              function.__defaults__,function.__closure__)


def code_bindings():
    files=base.code_bindings()
    files[str(Path(__file__))]=sha(__file__)
    return files


def validate_job(job):
    require(job['schema']=='receiver.patient-dp-s200-bank/v1','Wrong patient-DP job')
    require(job['variant'] in ('CLIP','DP8') and job['source_set']=='A2','Only two declared A2 arms')
    require(job['seed']==job['template_seed']==job['condition_seed']==101,'Seed domains changed')
    require(job['population']=='pooled' and job['artifact_kind']=='MAIN_BANK','Wrong target population')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Frozen training code changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Campaign changed')
    c=read(job['campaign_contract']);reference=read(c['reference_job'])
    require(c['status']=='AUTHORIZED_FROZEN' and c['max_private_releases']==1,'Wrong authorized scope')
    require(job['id']==c['bank_ids'][job['variant']],'Wrong bank ID')
    for key in ('images','updates','microbatch','activation_updates','optimizer','templates'):
        require(job[key]==reference[key],'Matched A2 recipe changed: '+key)
    require(job['maximum_seconds']==7200==c['maximum_seconds_per_bank'],'Wrong worker cap')
    require(job['absolute_deadline_utc']==c['absolute_deadline_utc'],'Campaign cap changed')
    require(time.time()<datetime.fromisoformat(c['absolute_deadline_utc']).timestamp(),'Campaign time cap exhausted')
    require(sha(job['paired_initialization'])==c['initialization_sha256'],'Original initialization changed')
    target_binding=read(job['target_binding'])
    require(sha(job['target_binding'])==job['target_binding_sha256'],'Target binding changed')
    require(target_binding['variant']==job['variant'],'Wrong target arm')
    expected='PATIENT_DP_SINGLE_QUERY_INTERNAL' if job['variant']=='DP8' else 'NONDP_CLIPPING_CONTROL'
    for key in ('a1_handoff','second_handoff'):
        require(sha(job[key])==job[key+'_sha256']==target_binding[key+'_sha256'],'Target hash changed')
        h=read(job[key]);require(h['privacy']==expected and h['population']=='pooled','Wrong target scope')
        if job['variant']=='DP8':
            meta=read(h['contract_file'])
            require(sha(h['contract_file'])==h['contract_sha256'],'Protected mechanism metadata changed')
            require(meta['epsilon']==8 and meta['delta']==1e-5 and meta['sensitivity']==1
                    and meta['dimension']==546 and meta['private_queries']==1
                    and meta['adjacency']=='add_remove_one_patient','Protection contract changed')
    return (0,)*64+(1,)*64


def annotate_artifact_metadata(name,value,job):
    if name not in ('learning_contract.json','provenance_public.json'):return value
    result=dict(value,DP_applied=job['variant']=='DP8',
        privacy_scope='INTERNAL_AUDIT_PACKAGE; release only the separate protected_png directory',
        target_variant=job['variant'],private_raw_data_access_during_synthesis=False)
    if job['variant']=='DP8':
        result['patient_dp']={'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient',
            'protected_queries':1,'postprocessing':True,'development_selection_protected':False}
    else:result['patient_dp']=None
    return result


def export(directory,renderer,job,spec,signature,checkpoint):
    def write(path,value,**kwargs):
        return atomic_json(path,annotate_artifact_metadata(Path(path).name,value,job),**kwargs)
    return clone(base.export,atomic_json=write)(directory,renderer,job,spec,signature,checkpoint)


_unchanged_run=clone(base._run,export=export)


def _run(job,directory,labels,profile,resume,started,deadline):
    remaining=datetime.fromisoformat(job['absolute_deadline_utc']).timestamp()-time.time()
    require(remaining>60,'No campaign time remaining for a safe update')
    return _unchanged_run(job,directory,labels,profile,resume,started,
                          min(deadline,time.monotonic()+remaining))


# Same optimizer, initial state, forward/backward, checkpoints, resume, pruning.
execute=clone(base.execute,validate_job=validate_job,_run=_run)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    result=execute(read(args.job),args.output,resume=args.resume)
    atomic_json(args.result,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
