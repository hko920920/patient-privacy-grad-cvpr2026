"""Finite known-law comparison; no ML model imports or GPU access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def cp_lookup(protocol):
    """One table reused by all algorithms; Bonferroni across all fixed looks."""
    tail = protocol['beta_family_error'] / (2*3*2*2*len(protocol['look_counts'])*2)
    lut = {}
    for n in protocol['look_counts']:
        k = np.arange(n + 1)
        low = np.zeros(n + 1)
        high = np.ones(n + 1)
        low[1:] = beta_distribution.ppf(tail, k[1:], n-k[1:]+1)
        high[:-1] = beta_distribution.ppf(1-tail, k[:-1]+1, n-k[:-1])
        lut[n] = np.stack([low, high])
    return lut, tail

def parameters(law):
    # shape defense, attack, stratum, threshold
    return np.stack([
        np.stack([law['nonmember_tail_low'], law['nonmember_tail_high']], axis=-1),
        np.stack([law['member_tail_low'], law['member_tail_high']], axis=-1),
    ], axis=2)

def prepare_streams(protocol, law, seed, lut):
    p = parameters(law)
    assert p.shape == (2,3,2,2)
    assert np.all((p >= 0) & (p <= 1)) and np.all(p[...,0] >= p[...,1])
    rng = np.random.default_rng(seed)
    uniforms = rng.random((2, protocol['max_patients_per_defense_stratum']))
    counts, bands, empiricals = {}, {}, {}
    for n in protocol['look_counts']:
        k = (uniforms[None,None,:,:n,None] > 1-p[:,:,:,None,:]).sum(axis=3)
        counts[n] = k
        empiricals[n] = k / n
        bands[n] = np.stack([lut[n][0,k], lut[n][1,k]], axis=-1)
    covered = all(np.all((v[...,0] <= p) & (p <= v[...,1])) for v in bands.values())
    fingerprint = hashlib.sha256(uniforms.tobytes()).hexdigest()
    return bands, empiricals, bool(covered), fingerprint

def risks(bounds, alpha):
    # bounds[d,a,z,threshold,lower/upper]; always-negative adds exact zero.
    lower = np.max(np.where(bounds[:,:,0,:,1] <= alpha, bounds[:,:,1,:,0], 0), axis=-1)
    upper = np.max(np.where(bounds[:,:,0,:,0] <= alpha, bounds[:,:,1,:,1], 0), axis=-1)
    return lower, upper, lower.max(axis=1), upper.max(axis=1)

def true_risks(law, alpha):
    p = parameters(law)
    return np.max(np.where(p[:,:,0,:] <= alpha, p[:,:,1,:], 0), axis=(1,2))

class Auditor:
    def __init__(self, protocol, graph, algorithm, band_stream, empirical_stream):
        self.p = protocol
        self.graph = graph
        self.algorithm = algorithm
        self.bands_stream = band_stream
        self.empirical_stream = empirical_stream
        self.batch = protocol['batch_patients']
        self.nmax = protocol['max_patients_per_defense_stratum']
        self.budget = protocol['cost_budget']
        self.cover = protocol['feature_graphs'][graph]['bundle_attacks']
        self.n = np.zeros((2,3,2), dtype=np.int64)
        self.bounds = np.zeros((2,3,2,2,2), dtype=np.float64)
        self.bounds[...,1] = 1
        self.empirical = np.zeros((2,3,2,2), dtype=np.float64)
        self.cache = np.zeros((2,2,self.nmax), dtype=np.int8)
        self.costs = np.array([0,1,4,16], dtype=np.int64)
        self.total_cost = 0
        self.stratum_cost = np.zeros(2, dtype=np.int64)
        self.action_counts = np.zeros((2,3,2), dtype=np.int64)
        self.actions = [(d,b,z) for d in range(2) for b in range(3) for z in range(2)]
        self.uniform_cursor = 0
        self.round = 0
        self.trace = []

    def candidate(self, action):
        d,b,z = action
        start = int(self.n[d,b,z])
        if start == self.nmax:
            return None
        end = start + self.batch
        changed = [a for a in self.cover[b] if self.n[d,a,z] < end]
        if self.graph == 'no_sharing':
            cost = int(self.costs[b+1] * self.batch)
        else:
            level = self.cache[d,z,start:end]
            cost = int((self.costs[np.maximum(level,b+1)]-self.costs[level]).sum())
        assert cost > 0 and b in changed
        if self.total_cost + cost > self.budget:
            return None
        return action, start, end, changed, cost

    def width_score(self, candidate, lower, upper, active):
        (d,b,z), start, end, changed, cost = candidate
        score = 0.0
        cold = 0
        for a in changed:
            if not active[d,a]:
                continue
            feasible_u = self.bounds[d,a,0,:,0] <= self.p['alpha_fpr']
            feasible_l = self.bounds[d,a,0,:,1] <= self.p['alpha_fpr']
            witness_u = feasible_u & (self.bounds[d,a,1,:,1] >= upper[d,a]-1e-15)
            witness_l = feasible_l & (self.bounds[d,a,1,:,0] >= lower[d,a]-1e-15)
            witness = witness_u | witness_l
            if not witness.any():
                witness[:] = True
            width = (self.bounds[d,a,z,:,1]-self.bounds[d,a,z,:,0])[witness].max()
            score += (upper[d,a]-lower[d,a]) * width
            cold += int(self.n[d,a,z] == 0)
        return score/cost, cold/cost

    def proxy_score(self, candidate, leader, challenger, active, gap):
        (d,b,z), start, end, changed, cost = candidate
        fake = self.bounds.copy()
        shrink, cold = 0.0, 0
        for a in changed:
            if not active[d,a]:
                continue
            n = int(self.n[d,a,z])
            if not n:
                cold += 1
                continue
            old = self.bounds[d,a,z]
            center = np.clip(self.empirical[d,a,z], old[:,0], old[:,1])
            rho = math.sqrt((n+1)/(end+1))
            fake[d,a,z,:,0] = center-rho*(center-old[:,0])
            fake[d,a,z,:,1] = center+rho*(old[:,1]-center)
            shrink += (old[:,1]-old[:,0]).max()-(fake[d,a,z,:,1]-fake[d,a,z,:,0]).max()
        _,_,new_l,new_u = risks(fake, self.p['alpha_fpr'])
        reduction = max(0., gap-(new_u[leader]-new_l[challenger]))
        return reduction/cost, cold/cost, shrink/cost

    def round_robin(self, candidates, start):
        lookup = {x[0]:x for x in candidates}
        for offset in range(len(self.actions)):
            index = (start+offset) % len(self.actions)
            if self.actions[index] in lookup:
                return lookup[self.actions[index]], (index+1) % len(self.actions)
        raise AssertionError('No affordable action')

    def choose(self, candidates, leader, challenger, lower, upper, dl, du):
        t = self.round + 1
        if t & (t-1) == 0:
            return self.round_robin(candidates, int(math.log2(t)) % len(self.actions))[0], 'forced'
        if self.algorithm == 'uniform':
            selected, self.uniform_cursor = self.round_robin(candidates, self.uniform_cursor)
            return selected, 'uniform'
        active = upper >= dl[:,None]
        filtered = [c for c in candidates if c[0][0] in (leader,challenger)
                    and any(active[c[0][0],a] for a in c[3])]
        if not filtered:
            return self.round_robin(candidates, t % len(self.actions))[0], 'fallback'
        if self.algorithm == 'maximin_width_shared_cost_adaptation':
            key = lambda c: self.width_score(c,lower,upper,active)
        elif self.algorithm == 'batched_v0':
            key = lambda c: self.proxy_score(c,leader,challenger,active,du[leader]-dl[challenger])
        else:
            raise ValueError(self.algorithm)
        # max returns the first fixed-ID candidate on ties.
        return max(filtered,key=key), 'allocated'

    def observe(self, candidate, reason):
        (d,b,z), start, end, changed, cost = candidate
        if self.graph == 'nested_sharing':
            self.cache[d,z,start:end] = np.maximum(self.cache[d,z,start:end],b+1)
        for a in changed:
            assert end > self.n[d,a,z]
            self.n[d,a,z] = end
            self.bounds[d,a,z] = self.bands_stream[end][d,a,z]
            self.empirical[d,a,z] = self.empirical_stream[end][d,a,z]
        self.total_cost += cost
        self.stratum_cost[z] += cost
        self.action_counts[d,b,z] += 1
        self.round += 1
        self.trace.append([d,b,z,end,cost,reason])

    def run(self):
        decision = None
        while True:
            lower,upper,dl,du = risks(self.bounds,self.p['alpha_fpr'])
            leader = int(np.argmin(du))
            challenger = 1-leader
            if du[leader] < dl[challenger]:
                decision = leader
                break
            candidates = [c for a in self.actions if (c:=self.candidate(a)) is not None]
            if not candidates:
                break
            candidate,reason = self.choose(candidates,leader,challenger,lower,upper,dl,du)
            self.observe(candidate,reason)
        assert self.total_cost <= self.budget
        return dict(decision=decision,cost=self.total_cost,rounds=self.round,
                    null_cost=int(self.stratum_cost[0]),member_cost=int(self.stratum_cost[1]),
                    lower=dl.tolist(),upper=du.tolist(),counts=self.n.tolist(),
                    action_counts=self.action_counts.tolist(),
                    trace_sha256=hashlib.sha256(json.dumps(self.trace).encode()).hexdigest(),
                    trace=self.trace)

def self_test(protocol, lut, tail):
    # Exact finite-threshold computation vs an independent scalar loop.
    rng = np.random.default_rng(7)
    for _ in range(100):
        endpoints = np.sort(rng.random((2,3,2,2,2)),axis=-1)
        lo,hi,dl,du = risks(endpoints,protocol['alpha_fpr'])
        for d in range(2):
            for a in range(3):
                explicit_l = max([0.]+[float(endpoints[d,a,1,t,0]) for t in range(2)
                                      if endpoints[d,a,0,t,1] <= protocol['alpha_fpr']])
                explicit_u = max([0.]+[float(endpoints[d,a,1,t,1]) for t in range(2)
                                      if endpoints[d,a,0,t,0] <= protocol['alpha_fpr']])
                assert lo[d,a] == explicit_l and hi[d,a] == explicit_u
    # Analytic CP boundary cases and monotonicity.
    for n,v in lut.items():
        assert abs(v[1,0]-(1-tail**(1/n))) < 1e-13
        assert abs(v[0,n]-tail**(1/n)) < 1e-13
        assert np.all(v[0] <= v[1]) and np.all(np.diff(v,axis=1) >= 0)
    # Shared feature upgrade 1 -> 4 -> 16 is counted once for the same patients.
    bounds={n:np.zeros((2,3,2,2,2)) for n in protocol['look_counts']}
    empirical={n:np.zeros((2,3,2,2)) for n in protocol['look_counts']}
    for graph,expected in [('nested_sharing',16*128),('no_sharing',21*128)]:
        agent=Auditor(protocol,graph,'uniform',bounds,empirical)
        for b in range(3):
            agent.observe(agent.candidate((0,b,0)),'test')
        assert agent.total_cost == expected
        assert np.array_equal(agent.n[0,:,0],np.array([128,128,128]))
    assert np.allclose(true_risks(protocol['laws']['member_bottleneck'],.05),[.65,.35])
    assert np.allclose(true_risks(protocol['laws']['null_bottleneck'],.05),[.20,.60])
    return {'status':'PASS','checks':['scalar_vs_vector_ROC_100','CP_extreme_counts',
            'CP_monotonicity','nested_cache_upgrade_no_doublecount','known_risk_truth']}

def summarize(records, protocol):
    summaries=[]
    for law in protocol['laws']:
        for graph in protocol['feature_graphs']:
            for algorithm in protocol['algorithms']:
                subset=[r for r in records if (r['law'],r['graph'],r['algorithm'])==(law,graph,algorithm)]
                decided=[r for r in subset if r['decision'] is not None]
                correct=[r for r in decided if r['correct']]
                costs=[r['cost'] for r in subset]
                summaries.append(dict(law=law,graph=graph,algorithm=algorithm,replicates=len(subset),
                    decisions=len(decided),correct_decisions=len(correct),wrong_decisions=len(decided)-len(correct),
                    abstentions=len(subset)-len(decided),decision_rate=len(decided)/len(subset),
                    mean_cost_including_abstention=float(np.mean(costs)),
                    mean_cost_correct_only=float(np.mean([r['cost'] for r in correct])) if correct else None,
                    median_cost_correct_only=float(np.median([r['cost'] for r in correct])) if correct else None,
                    mean_null_cost=float(np.mean([r['null_cost'] for r in subset])),
                    mean_member_cost=float(np.mean([r['member_cost'] for r in subset])),
                    band_coverage_failures=sum(not r['family_covered'] for r in subset)))
    paired=[]
    for law in protocol['laws']:
        for graph in protocol['feature_graphs']:
            idx={(r['algorithm'],r['replicate']):r for r in records if r['law']==law and r['graph']==graph}
            for baseline in protocol['algorithms'][:2]:
                delta=[idx[('batched_v0',k)]['cost']-idx[(baseline,k)]['cost'] for k in range(protocol['replicates'])]
                both=[k for k in range(protocol['replicates']) if idx[('batched_v0',k)]['correct'] and idx[(baseline,k)]['correct']]
                paired.append(dict(law=law,graph=graph,baseline=baseline,
                    mean_cost_delta_all_runs=float(np.mean(delta)),
                    paired_decision_count_delta=sum(idx[('batched_v0',k)]['decision'] is not None for k in range(protocol['replicates']))-sum(idx[(baseline,k)]['decision'] is not None for k in range(protocol['replicates'])),
                    both_correct_count=len(both),
                    mean_cost_delta_both_correct=float(np.mean([delta[k] for k in both])) if both else None))
    return summaries,paired

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    assert args.self_test != args.run
    protocol_path=HERE/'protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    lut,tail=cp_lookup(protocol)
    tests=self_test(protocol,lut,tail)
    if args.self_test:
        print(json.dumps(tests))
        return
    source_hash=sha(__file__)
    protocol_hash=sha(protocol_path)
    # Immutable provenance is committed before the first actual score stream.
    write_new(HERE/'execution_protocol.json',dict(schema='track2-cpu-execution/v1',
        protocol_sha256=protocol_hash,code_sha256=source_hash,python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,tail_error=tail,self_test=tests,
        target_calls=0,gpu_calls=0,training_calls=0))
    start=time.perf_counter()
    records=[]
    trace_records=[]
    for law_index,(name,law) in enumerate(protocol['laws'].items()):
        truth=true_risks(law,protocol['alpha_fpr'])
        true_best=int(np.argmin(truth))
        for replicate in range(protocol['replicates']):
            seed=protocol['seed']+100000*law_index+replicate
            streams,empirical,covered,fingerprint=prepare_streams(protocol,law,seed,lut)
            for graph in protocol['feature_graphs']:
                for algorithm in protocol['algorithms']:
                    result=Auditor(protocol,graph,algorithm,streams,empirical).run()
                    trace=result.pop('trace')
                    result.update(law=name,graph=graph,algorithm=algorithm,replicate=replicate,seed=seed,
                        true_risks=truth.tolist(),true_best=true_best,
                        correct=result['decision']==true_best,family_covered=covered,stream_sha256=fingerprint)
                    # On simultaneous coverage, a wrong certified decision is impossible.
                    assert not (covered and result['decision'] is not None and not result['correct'])
                    records.append(result)
                    trace_records.append(dict(law=name,graph=graph,algorithm=algorithm,replicate=replicate,trace=trace))
        print(json.dumps({'finished_law':name,'runs':len(records),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    summaries,paired=summarize(records,protocol)
    elapsed=time.perf_counter()-start
    assert sha(__file__)==source_hash and sha(protocol_path)==protocol_hash
    write_new(HERE/'results.json',dict(schema='track2-cpu-results/v1',status='PASS_CPU_SAVED_SIMULATION',
        seconds=elapsed,protocol_sha256=protocol_hash,code_sha256=source_hash,
        self_test=tests,summary=summaries,paired_comparisons=paired,records=records,
        caveat='Known artificial score laws, fixed models/scores, finite threshold portfolio; no model/DP/privacy efficacy inference.'))
    write_new(HERE/'traces.json',trace_records)
    with (HERE/'results.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    write_new(HERE/'output_manifest.json',{name:sha(HERE/name) for name in ['protocol.json','run_cpu_audit.py','execution_protocol.json','results.json','results.csv','traces.json']})
    print(json.dumps({'status':'PASS','runs':len(records),'seconds':elapsed,'summary':summaries,'paired':paired},ensure_ascii=False))

if __name__=='__main__':
    main()
