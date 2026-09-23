"""One DINO-only DP bank using the frozen DINO training/replay bytecode."""
import argparse,json,time
from pathlib import Path
from datetime import datetime
from prrd_v3.contracts import read,require,sha
from prrd_v3.bank_runtime import atomic_json
from . import dino_control_runtime as base
from .patient_dp_runtime import clone,annotate_artifact_metadata


def code_bindings():
    files=base.code_bindings()
    for name in ('patient_dp_runtime.py','patient_dp_dino_runtime.py'):
        p=Path(__file__).with_name(name);files[str(p)]=sha(p)
    return files


def validate_job(job):
    require(job['schema']=='receiver.dino-patient-dp-s200-bank/v1','Wrong DINO-DP job')
    require(job['variant']=='DP8' and job['source_set']=='DINO','Wrong source/variant')
    require(job['seed']==job['template_seed']==job['condition_seed']==101,'Seed domains changed')
    require(job['population']=='pooled' and job['artifact_kind']=='MAIN_BANK','Wrong population/kind')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Training code changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Campaign changed')
    c=read(job['campaign_contract']);reference=read(c['reference_job'])
    require(c['status']=='AUTHORIZED_FROZEN' and c['max_private_releases']==1,'Wrong scope')
    require(job['id']==c['bank_ids']['DP8'],'Wrong bank ID')
    for key in ('images','updates','microbatch','activation_updates','optimizer','templates',
                'A2_job','A2_bank','A2_initial_receipt','A2_template_tensor_digest'):
        require(job[key]==reference[key],'Matched DINO recipe changed: '+key)
    require(job['maximum_seconds']==3600==c['maximum_seconds_per_bank'],'Wrong worker cap')
    require(job['absolute_deadline_utc']==c['absolute_deadline_utc'],'Deadline changed')
    require(time.time()<datetime.fromisoformat(c['absolute_deadline_utc']).timestamp(),'Time cap reached')
    require(all(sha(p)==h for p,h in c['historical_dependencies'].items()),'Historical evidence changed')
    require('a1_handoff' not in job,'Unexpected BioViL target')
    t=read(job['target_binding'])
    require(sha(job['target_binding'])==job['target_binding_sha256'],'Binding changed')
    key='second_handoff'
    require(sha(job[key])==job[key+'_sha256']==t[key+'_sha256'],'DINO target changed')
    h=read(job[key]);meta=read(h['contract_file'])
    require(h['privacy']=='PATIENT_DP_SINGLE_QUERY_INTERNAL' and h['population']=='pooled','Wrong target scope')
    require(sha(h['contract_file'])==h['contract_sha256'],'Mechanism metadata changed')
    require(meta==c['mechanism'],'DINO mechanism differs from frozen contract')
    require(meta['dimension']==274 and meta['sensitivity']==1 and meta['epsilon']==8
            and meta['delta']==1e-5 and meta['adjacency']=='add_remove_one_patient'
            and meta['private_queries']==1,'Protection parameters changed')
    return (0,)*64+(1,)*64


def export(directory,renderer,job,spec,signature,checkpoint):
    def write(path,value,**kwargs):
        return atomic_json(path,annotate_artifact_metadata(Path(path).name,value,job),**kwargs)
    return clone(base.export,atomic_json=write)(directory,renderer,job,spec,signature,checkpoint)


_unchanged_run=clone(base._run,export=export)


def _run(job,directory,labels,profile,resume,started,deadline):
    remaining=datetime.fromisoformat(job['absolute_deadline_utc']).timestamp()-time.time()
    require(remaining>60,'No safe time left')
    return _unchanged_run(job,directory,labels,profile,resume,started,
                          min(deadline,time.monotonic()+remaining))


execute=clone(base.execute,validate_job=validate_job,_run=_run)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    result=execute(read(a.job),a.output,resume=a.resume)
    atomic_json(a.result,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
