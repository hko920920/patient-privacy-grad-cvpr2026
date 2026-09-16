"""Independent saved-output arithmetic/trace check; no policy rerun or ML calls."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

checks = {}
for track, folder, script in [(1, 'track1_cpu_allocation_20260916', 'run_cpu_allocation.py'),
                              (2, 'track2_cpu_audit_20260916', 'run_cpu_audit.py')]:
    base = HERE/folder
    binding = read(base/'execution_protocol.json')
    assert binding['protocol_sha256'] == sha(base/'protocol.json')
    expected = binding.get('script_sha256', binding.get('code_sha256'))
    assert expected == sha(base/script)
checks['frozen_protocol_and_code_bindings'] = 2

base = HERE/'track1_cpu_allocation_20260916'
one = read(base/'results.json')
levels = np.array([1,2,4,8,16])
for row in one['rows']:
    overhead = 2 if row['method'].startswith(('raw_', 'geometry_', 'oracle_charged')) else 0
    assert abs(np.mean(np.array(row['R_frequencies_by_type']) @ levels)+overhead-row['actual_mean_cost']) < 1e-12
    assert abs(np.mean(row['mse_by_type'])-row['clipped_mse']) < 1e-12
checks['track1_cost_and_mse_rows'] = len(one['rows'])

# Exact enumeration shows clipped finite-sample MSE need not decrease with R.
p, L = .1, 3.
support, mass = [0.,L,-L], [1-p,p/2,p/2]
risk1 = sum(w*np.clip(x,-1,1)**2 for x,w in zip(support,mass))
risk2 = sum(mass[i]*mass[j]*np.clip((support[i]+support[j])/2,-1,1)**2
            for i,j in itertools.product(range(3), repeat=2))
assert abs(risk1-p) < 1e-14 and abs(risk2-(2*p-1.5*p*p)) < 1e-14
checks['finite_clipping_counterexample'] = dict(p=p,L=L,R1=risk1,R2=risk2)

base = HERE/'track2_cpu_audit_20260916'
cfg, two, traces = read(base/'protocol.json'), read(base/'results.json'), read(base/'traces.json')
tail = cfg['beta_family_error']/(2*3*2*2*len(cfg['look_counts'])*2)
by_trace = {(r['law'],r['graph'],r['algorithm'],r['replicate']):r['trace'] for r in traces}
cp = {}
for n in [0]+cfg['look_counts']:
    if n == 0:
        cp[n] = (np.array([0.]), np.array([1.]))
    else:
        k = np.arange(n+1)
        low, high = np.zeros(n+1), np.ones(n+1)
        low[1:] = beta.ppf(tail,k[1:],n-k[1:]+1)
        high[:-1] = beta.ppf(1-tail,k[:-1]+1,n-k[:-1])
        cp[n] = (low,high)
streams = {}
steps = 0
for row in two['records']:
    trace = by_trace[(row['law'],row['graph'],row['algorithm'],row['replicate'])]
    assert hashlib.sha256(json.dumps(trace).encode()).hexdigest() == row['trace_sha256']
    counts = np.zeros((2,3,2),dtype=int)
    cache = np.zeros((2,2,cfg['max_patients_per_defense_stratum']),dtype=int)
    expenditure = [0,0]
    for d,b,z,end,saved_cost,reason in trace:
        start = int(counts[d,b,z])
        assert end-start == cfg['batch_patients']
        if row['graph'] == 'no_sharing':
            cost = [1,4,16][b]*(end-start)
            counts[d,b,z] = end
        else:
            requested = [1,4,16][b]
            cost = int(np.maximum(0,requested-cache[d,z,start:end]).sum())
            cache[d,z,start:end] = np.maximum(cache[d,z,start:end],requested)
            for a in range(b+1): counts[d,a,z] = max(counts[d,a,z],end)
        assert cost == saved_cost
        expenditure[z] += cost
        steps += 1
    assert np.array_equal(counts,row['counts'])
    assert expenditure == [row['null_cost'],row['member_cost']]
    assert sum(expenditure) == row['cost'] <= cfg['cost_budget']
    if row['seed'] not in streams:
        streams[row['seed']] = np.random.default_rng(row['seed']).random((2,cfg['max_patients_per_defense_stratum']))
    u = streams[row['seed']]
    assert hashlib.sha256(u.tobytes()).hexdigest() == row['stream_sha256']
    law = cfg['laws'][row['law']]
    dl,du = [],[]
    for d in range(2):
        al,au = [],[]
        for a in range(3):
            lows,ups = [0.],[0.]
            for label in ['low','high']:
                intervals = []
                for z,stratum in enumerate(['nonmember','member']):
                    n = int(counts[d,a,z])
                    prob = law[stratum+'_tail_'+label][d][a]
                    successes = int(np.count_nonzero(u[z,:n] > 1-prob))
                    intervals.append((float(cp[n][0][successes]),float(cp[n][1][successes])))
                if intervals[0][1] <= cfg['alpha_fpr']: lows.append(intervals[1][0])
                if intervals[0][0] <= cfg['alpha_fpr']: ups.append(intervals[1][1])
            al.append(max(lows)); au.append(max(ups))
        dl.append(max(al)); du.append(max(au))
    assert np.allclose(dl,row['lower'],atol=1e-14,rtol=0)
    assert np.allclose(du,row['upper'],atol=1e-14,rtol=0)
    if row['decision'] is not None:
        d = row['decision']
        assert du[d] < dl[1-d]
        assert row['correct'] == (d == row['true_best'])
checks['track2_independent_trace_cost_and_final_risk_records'] = len(two['records'])
checks['track2_independent_replayed_actions'] = steps
checks['track2_distinct_seed_streams'] = len(streams)
checks['limits'] = ['Saved-output arithmetic and final-decision check; not a new policy experiment.',
                   'No medical data, added DP noise, model training or whole-paper novelty proof.',
                   'Same seed streams reused across methods/conditions; 768 runs are not independent population trials.']
result = dict(status='PASS',checks=checks,script_sha256=sha(Path(__file__)))
(HERE/'two_track_deeper_independent_verification_20260916.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
