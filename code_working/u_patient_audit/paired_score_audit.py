"""Post hoc descriptive paired-score decomposition; CPU-only, no new scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import RUN, digest, write_json

INPUT = RUN / "baseline_screen_20260914/pfami_v1"
SCENARIOS = ("U", "E")
SCORES = ("relative", "difference", "negative_loss", "base_relative")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def H(values):
    values = np.asarray(values)
    return (values > 0).astype(np.float64) + .5*(values == 0)


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left+1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        result[order[left:right]] = (left+right-1)/2+1
        left = right
    return result


def correlation(a,b):
    a,b = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
    x,y = a-a.mean(), b-b.mean()
    denominator = float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/denominator) if denominator else None


def counts(values):
    values=np.asarray(values,dtype=np.float64)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
        positive=int(np.sum(values>0)),zero=int(np.sum(values==0)),negative=int(np.sum(values<0)))


def draws():
    rng=np.random.default_rng(260914)
    return np.concatenate([rng.integers(0,20,size=(2000,20),dtype=np.int64),
                           rng.integers(20,40,size=(2000,20),dtype=np.int64)],axis=1)


def interval(values):
    return np.quantile(values,[.025,.975],method="linear").tolist()


def decompose(s1,s2):
    s1,s2=np.asarray(s1,dtype=np.float64),np.asarray(s2,dtype=np.float64)
    assert s1.shape == s2.shape and len(s1)%2==0 and np.isfinite(s1).all() and np.isfinite(s2).all()
    n=len(s1)//2
    c,d=(s1+s2)/2,(s1-s2)/2
    a=c[:n,None]-c[None,n:]
    b=d[:n,None]-d[None,n:]
    plus,minus=H(a+b),H(a-b)
    contribution=.5*(plus-minus)
    auc1=float(H(s1[:n,None]-s1[None,n:]).mean())
    auc2=float(H(s2[n:,None]-s2[None,:n]).mean())
    excess=(auc1+auc2)/2-.5
    assert math.isclose(float(contribution.mean()),excess,abs_tol=1e-12)
    return c,d,a,b,contribution,auc1,auc2,excess


def analyse(original,contract):
    assert contract["scope"]["patients"]==40 and contract["scope"]["scenarios"]==list(SCENARIOS)
    assert contract["scope"]["scores"]==list(SCORES)
    order=original["bootstrap"]["patient_order"]
    patients=original["patient_scores"]
    indexed={(p["scenario"],p["model"],p["patient_id"]):p for p in patients}
    assert len(indexed)==len(patients)==160 and len(set(order))==len(order)==40
    for i,p in enumerate(order):
        for scenario in SCENARIOS:
            for model in ("model_1","model_2"):
                row=indexed[scenario,model,p]
                group="A" if i<20 else "B"
                assert row["eval_role"]=="selection" and row["assignment_group"]==group
                assert row["member"]==int(group==("A" if model=="model_1" else "B"))
    assert order[:20]==sorted(order[:20],key=int) and order[20:]==sorted(order[20:],key=int)
    boot=draws();boot_hash=hashlib.sha256(boot.tobytes()).hexdigest()
    assert boot_hash==original["bootstrap"]["indices_sha256"]
    summaries,patient_values,primary_pairs=[],[],[]
    decompositions={}
    original_aucs={(r["scenario"],r["model"],r["score"]):r["auc"] for r in original["metrics"]}
    alignment=np.r_[np.ones(20),-np.ones(20)]
    for scenario in SCENARIOS:
        for score in SCORES:
            s1=np.array([indexed[scenario,"model_1",p]["scores"][score] for p in order],dtype=np.float64)
            s2=np.array([indexed[scenario,"model_2",p]["scores"][score] for p in order],dtype=np.float64)
            c,d,a,b,g,auc1,auc2,excess=decompose(s1,s2)
            aligned=alignment*(s1-s2)
            decompositions[scenario,score]=(d,aligned)
            assert math.isclose(auc1,original_aucs[scenario,"model_1",score],abs_tol=1e-12)
            assert math.isclose(auc2,original_aucs[scenario,"model_2",score],abs_tol=1e-12)
            upper=np.triu_indices(40,1)
            r1=np.sign((s1[:,None]-s1[None,:])[upper])
            r2=np.sign((s2[:,None]-s2[None,:])[upper])
            rank_counts=dict(total=780,agreement=int(np.sum((r1==r2)&(r1!=0))),
                reversal=int(np.sum(r1*r2<0)),one_model_tie=int(np.sum((r1==0)^(r2==0))),
                both_model_tie=int(np.sum((r1==0)&(r2==0))))
            assert sum(rank_counts[k] for k in ("agreement","reversal","one_model_tie","both_model_tie"))==780
            v1,v2,vc,vd=[float(np.var(x,ddof=1)) for x in (s1,s2,c,d)]
            common_auc1=float(H(a).mean());common_auc2=float(H(-a).mean())
            assert common_auc1+common_auc2==1
            sampled1,sampled2=s1[boot],s2[boot]
            boot_auc1=H(sampled1[:,:20,None]-sampled1[:,None,20:]).mean(axis=(1,2))
            boot_auc2=H(sampled2[:,20:,None]-sampled2[:,None,:20]).mean(axis=(1,2))
            delta_summary=counts(aligned)
            delta_summary.update(mean_bootstrap95=interval(aligned[boot].mean(axis=1)),
                                 by_group={"A":counts(aligned[:20]),"B":counts(aligned[20:])})
            summary=dict(scenario=scenario,score=score,patients=40,
                correlations=dict(pearson=correlation(s1,s2),spearman=correlation(ranks(s1),ranks(s2))),
                rank_pairs=rank_counts,
                variance=dict(ddof=1,s1_variance=v1,s2_variance=v2,common_variance=vc,half_change_variance=vd,
                    common_sd=math.sqrt(vc),half_change_sd=math.sqrt(vd),
                    identity_left=(v1+v2)/2,identity_right=vc+vd,identity_error=(v1+v2)/2-vc-vd,
                    common_share=vc/(vc+vd) if vc+vd else None),
                aligned_delta=delta_summary,
                aucs=dict(model_1=auc1,model_2=auc2,common_A_labels=common_auc1,common_B_labels=common_auc2,
                    common_average=(common_auc1+common_auc2)/2,model_1_gain_over_common=auc1-common_auc1,
                    model_2_gain_over_common=auc2-common_auc2,mean_auc_excess=excess,
                    mean_auc_excess_bootstrap95=interval((boot_auc1+boot_auc2)/2-.5)),
                pair_contributions=dict(total=400,changed=int(np.sum(g!=0)),net_positive=int(np.sum(g>0)),
                    net_negative=int(np.sum(g<0)),zero=int(np.sum(g==0)),sum=float(g.sum()),mean=float(g.mean()),
                    preserved_auc_excess_error=float(g.mean())-excess))
            summaries.append(summary)
            for i,p in enumerate(order):
                patient_values.append(dict(scenario=scenario,score=score,patient_id=p,
                    group="A" if i<20 else "B",s1=float(s1[i]),s2=float(s2[i]),common=float(c[i]),
                    half_change=float(d[i]),aligned_delta=float(aligned[i])))
            if (scenario,score)==("U","relative"):
                for i,p_a in enumerate(order[:20]):
                    for j,p_b in enumerate(order[20:]):
                        primary_pairs.append(dict(patient_A=p_a,patient_B=p_b,
                            common_margin=float(a[i,j]),change_margin=float(b[i,j]),
                            model1_margin=float(s1[i]-s1[20+j]),model2_A_minus_B_margin=float(s2[i]-s2[20+j]),
                            H_a_plus_b=float(H(a[i,j]+b[i,j])),H_a_minus_b=float(H(a[i,j]-b[i,j])),
                            contribution=float(g[i,j])))
    base_identity=[]
    for scenario in SCENARIOS:
        d,delta=decompositions[scenario,"relative"]
        bd,bdelta=decompositions[scenario,"base_relative"]
        base_identity.append(dict(scenario=scenario,half_change_max_abs=float(np.max(np.abs(d-bd))),
            aligned_delta_max_abs=float(np.max(np.abs(delta-bdelta))),
            interpretation="same raw base subtracted from both targets; differences may reflect float64 rounding"))
    return dict(schema="paired-score-descriptive-audit-result/v1",summaries=summaries,patient_values=patient_values,
        primary_patient_values=[r for r in patient_values if (r["scenario"],r["score"])==("U","relative")],
        primary_pair_margins=primary_pairs,base_relative_identity=base_identity,
        bootstrap=dict(seed=260914,replicates=2000,patient_order=order,indices_sha256=boot_hash,
            interval=contract["bootstrap"]["interval"]),
        independent_patients=40,rank_pairs_are_not_independent_observations=True,
        groundtruth_aligned_delta_auc_computed=False,new_gpu_forward=0,new_backward=0,new_training=0,
        causal_membership_effect_identified=False,attack_available_inputs_changed=False,
        automatic_followup=False,constraints=contract["constraints"])


def synthetic_check():
    cases=[([4,3,2,1],[4,3,2,1]),([4,3,2,1],[1,2,3,4]),([1,1,1,1],[1,1,1,1]),
           ([2,1,1,0],[1,2,0,1])]
    for s1,s2 in cases:
        c,d,a,b,g,a1,a2,excess=decompose(s1,s2)
        assert math.isclose(float(g.mean()),(a1+a2)/2-.5,abs_tol=1e-14)
        assert math.isclose((np.var(s1,ddof=1)+np.var(s2,ddof=1))/2,
                            np.var(c,ddof=1)+np.var(d,ddof=1),abs_tol=1e-14)
        assert float(H(a).mean()+H(-a).mean())==1
    assert ranks([3,1,1,2]).tolist()==[4,1.5,1.5,3]
    return dict(status="PASS_SYNTHETIC_PAIRED_IDENTITIES",cases=len(cases),actual_patient_data_used=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve();contract_path=output/"contract.json"
    assert contract_path.exists()
    for name in ("protocol.json","analysis.json"):
        if (output/name).exists():raise FileExistsError(output/name)
    input_paths=[INPUT/name for name in ("analysis.json","report.json","verification.json","protocol.json","image_scores.json")]
    hashes={str(p.resolve()):digest(p) for p in input_paths}
    code_hash=digest(Path(__file__));contract_hash=digest(contract_path)
    protocol=dict(schema="paired-score-audit-provenance/v1",input_sha256=hashes,
        analysis_code_path=str(Path(__file__).resolve()),analysis_code_sha256=code_hash,
        contract_sha256=contract_hash,mode="post_hoc_descriptive_cpu_only",new_gpu_forward=0,new_backward=0)
    write_json(output/"protocol.json",protocol)
    original=load(INPUT/"analysis.json");report=load(INPUT/"report.json")
    assert original["input_image_scores_sha256"]==digest(INPUT/"image_scores.json")==report["image_scores_sha256"]
    assert original["protocol_sha256"]==digest(INPUT/"protocol.json")==report["protocol_sha256"]
    assert load(INPUT/"verification.json")["status"].startswith("PASS_")
    result=analyse(original,load(contract_path))
    result["protocol_sha256"]=digest(output/"protocol.json")
    result["contract_sha256"]=contract_hash
    result["analysis_code_sha256"]=code_hash
    for path,value in hashes.items():assert digest(path)==value
    assert digest(Path(__file__))==code_hash and digest(contract_path)==contract_hash
    write_json(output/"analysis.json",result)
    print(json.dumps(dict(status="COMPLETED_DESCRIPTIVE_CPU_ANALYSIS",conditions=len(result["summaries"]),
        patients=40,primary_pairs=len(result["primary_pair_margins"]),output=str(output)),ensure_ascii=False))


if __name__=="__main__":main()
