"""Independent final ledger/ROC verification; separate labeled source-policy trace."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta

P=Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
    with (P/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    inputs={n:digest(P/n) for n in ['protocol.json','run_cpu_audit.py','execution_protocol.json','results.json','traces.json','results.csv','output_manifest.json']}
    save('verification_protocol.json',dict(source_sha256=digest(__file__),input_sha256=inputs,
        scope='Independent final cost/patient-prefix/Clopper-Pearson/decision arithmetic; allocation trace separately reuses frozen producer helpers. No new scenarios.'))
    protocol=json.loads((P/'protocol.json').read_text())
    result=json.loads((P/'results.json').read_text())
    traces=json.loads((P/'traces.json').read_text())
    manifest=json.loads((P/'output_manifest.json').read_text())
    for name,value in manifest.items():
        assert digest(P/name)==value
    q=protocol['beta_family_error']/(2*3*2*2*len(protocol['look_counts'])*2)
    index={(t['law'],t['graph'],t['algorithm'],t['replicate']):t['trace'] for t in traces}
    streams={}
    independent_decisions=0
    for row in result['records']:
        key=(row['law'],row['graph'],row['algorithm'],row['replicate'])
        trace=index[key]
        assert hashlib.sha256(json.dumps(trace).encode()).hexdigest()==row['trace_sha256']
        n=np.zeros((2,3,2),dtype=int)
        cache=np.zeros((2,2,2048),dtype=int)
        total=0
        stratum=[0,0]
        call_counts=np.zeros((2,3,2),dtype=int)
        c=[0,1,4,16]
        for d,b,z,end,claimed_cost,reason in trace:
            start=int(n[d,b,z])
            assert end==start+128 and end in protocol['look_counts']
            if row['graph']=='no_sharing':
                cost=128*c[b+1]
                n[d,b,z]=end
            else:
                cost=sum(c[max(int(cache[d,z,i]),b+1)]-c[int(cache[d,z,i])] for i in range(start,end))
                cache[d,z,start:end]=np.maximum(cache[d,z,start:end],b+1)
                n[d,:b+1,z]=np.maximum(n[d,:b+1,z],end)
            assert cost==claimed_cost
            total+=cost;stratum[z]+=cost;call_counts[d,b,z]+=1
        assert total==row['cost'] and total<=protocol['cost_budget']
        assert stratum==[row['null_cost'],row['member_cost']]
        assert n.tolist()==row['counts'] and call_counts.tolist()==row['action_counts']
        streamkey=(row['law'],row['replicate'])
        if streamkey not in streams:
            u=np.random.default_rng(row['seed']).random((2,2048))
            streams[streamkey]=u
        u=streams[streamkey]
        assert hashlib.sha256(u.tobytes()).hexdigest()==row['stream_sha256']
        law=protocol['laws'][row['law']]
        dl,du=[],[]
        for d in range(2):
            lows,highs=[],[]
            for a in range(3):
                intervals=[]
                for z,name in enumerate(['nonmember','member']):
                    # Construct integer scores independently of producer indicator broadcasting.
                    scores=np.zeros(2048,dtype=int)
                    scores[u[z]>1-law[name+'_tail_low'][d][a]]=1
                    scores[u[z]>1-law[name+'_tail_high'][d][a]]=2
                    size=int(n[d,a,z]); by_threshold=[]
                    for threshold in protocol['thresholds']:
                        k=int(np.count_nonzero(scores[:size]>threshold))
                        lower=0. if k==0 else float(beta.ppf(q,k,size-k+1))
                        upper=1. if k==size else float(beta.ppf(1-q,k+1,size-k))
                        by_threshold.append((lower,upper))
                    intervals.append(by_threshold)
                lows.append(max([0.]+[intervals[1][t][0] for t in range(2) if intervals[0][t][1]<=.05]))
                highs.append(max([0.]+[intervals[1][t][1] for t in range(2) if intervals[0][t][0]<=.05]))
            dl.append(max(lows));du.append(max(highs))
        assert np.allclose(dl,row['lower'],rtol=0,atol=1e-14)
        assert np.allclose(du,row['upper'],rtol=0,atol=1e-14)
        if row['decision'] is not None:
            selected=row['decision'];independent_decisions+=1
            assert du[selected]<dl[1-selected]
            assert selected==row['true_best'] and row['correct']
    for s in result['summary']:
        rows=[r for r in result['records'] if (r['law'],r['graph'],r['algorithm'])==(s['law'],s['graph'],s['algorithm'])]
        assert len(rows)==64 and sum(r['decision'] is not None for r in rows)==s['decisions']
        assert math.isclose(math.fsum(r['cost'] for r in rows)/64,s['mean_cost_including_abstention'],abs_tol=1e-12)

    # Read-only trace of the declared heuristic using its exact, hash-bound arithmetic.
    # This section is not independent policy reimplementation and makes no new choices.
    spec=importlib.util.spec_from_file_location('frozen_source',P/'run_cpu_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lut,_=module.cp_lookup(protocol)
    diagnostic={}
    for graph in protocol['feature_graphs']:
        stats={'allocated_member':0,'allocated_null':0,'positive_proxy_member':0,'positive_proxy_null':0,
               'zero_proxy_member':0,'zero_proxy_null':0,'zero_all_null_proxy_but_positive_member_selected':0,
               'forced_member':0,'forced_null':0}
        for row in result['records']:
            if row['law']!='null_bottleneck' or row['graph']!=graph or row['algorithm']!='batched_v0':
                continue
            bs,es,_,_=module.prepare_streams(protocol,protocol['laws'][row['law']],row['seed'],lut)
            audit=module.Auditor(protocol,graph,'batched_v0',bs,es)
            for d,b,z,end,cost,reason in index[(row['law'],graph,'batched_v0',row['replicate'])]:
                lo,up,dl,du=module.risks(audit.bounds,.05)
                leader=int(np.argmin(du));challenger=1-leader;active=up>=dl[:,None]
                candidates=[x for action in audit.actions if (x:=audit.candidate(action)) is not None]
                expected,expected_reason=audit.choose(candidates,leader,challenger,lo,up,dl,du)
                assert expected[0]==(d,b,z) and expected_reason==reason
                suffix='member' if z else 'null'
                if reason=='forced':
                    stats['forced_'+suffix]+=1
                elif reason=='allocated':
                    stats['allocated_'+suffix]+=1
                    keys=[(x,audit.proxy_score(x,leader,challenger,active,du[leader]-dl[challenger]))
                          for x in candidates if any(active[x[0][0],a] for a in x[3])]
                    chosenkey=audit.proxy_score(expected,leader,challenger,active,du[leader]-dl[challenger])
                    positive=chosenkey[0]>0
                    stats[('positive_proxy_' if positive else 'zero_proxy_')+suffix]+=1
                    nullkeys=[value[0] for x,value in keys if x[0][2]==0]
                    if z==1 and positive and nullkeys and max(nullkeys)==0:
                        stats['zero_all_null_proxy_but_positive_member_selected']+=1
                audit.observe(expected,reason)
        diagnostic[graph]=stats
    cp_example=dict(zero_128_upper=float(beta.ppf(1-q,1,128)),
                    zero_256_upper=float(beta.ppf(1-q,1,256)),
                    proxy_128_to_256_upper=float(beta.ppf(1-q,1,128)*math.sqrt(129/257)),
                    scope='Exact standard CP arithmetic, illustrative boundary of the frozen proxy; not an extra score-law trial.')
    for name,value in inputs.items():
        assert digest(P/name)==value
    save('verification.json',dict(status='PASS_INDEPENDENT_FINAL_LEDGER_ROC_AND_STOP_ARITHMETIC',
        records_verified=len(result['records']),distinct_iid_cohorts=len(streams),
        independently_checked_decisions=independent_decisions,
        allocation_trace_scope='Source-policy replay for null-bottleneck v0 only; no independent allocation-policy proof.',
        allocation_trace=diagnostic,cp_flatness_example=cp_example,input_sha256=inputs,
        verifier_sha256=digest(__file__)))
    print(json.dumps({'status':'PASS','decisions':independent_decisions,'trace':diagnostic,'CP':cp_example}))

if __name__=='__main__':
    main()
