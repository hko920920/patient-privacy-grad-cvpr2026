"""Evaluate eight frozen label packets with the existing receiver/cluster protocol."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order

def run(out):
    tick=time.monotonic();io=Inputs(out,"evaluate")
    check(not (out/"results.json").exists(),"No silent reevaluation")
    check_seal(out,"label_seal.json","ALL_EIGHT_LABELS_SEALED_BEFORE_EVALUATION")
    check_seal(out,"selection_seal.json","PUBLIC128_SEALED_BEFORE_TARGET_ACCESS")
    labels=packet(out/"soft_labels.npz");sel=packet(out/"public128_selection.npz")
    check(len(labels)==8,"Unexpected label cells")
    sys.path.insert(0,io.contract["code_root"])
    from prrd_pilot_20260921.common import bootstrap_metrics
    draws=io.nz("patient_bootstrap")
    transport=io.nz("historical_transport_bootstrap")
    check(str(transport["source_draws_sha256"].item())==sha(io.path("patient_bootstrap")),"Patient schedule changed")
    counts=draws["patient_counts"];boot_ids=draws["patient_ids"]
    check(counts.shape==(2000,2026),"Patient bootstrap dimensions")
    construction=json.loads((out/"construction_results.json").read_text(encoding="utf-8"))
    metrics={};scores={};boots={};readout_weights={};reference=None
    verification={"ridge_primal_dual_max_abs":0.,"independent_bootstrap_max_abs":0.}
    for receiver,short in (("DenseNet","DN"),("ResNet18","RN")):
        p=io.nz("P_"+short);v=io.nz("V_"+receiver)
        check(set(p["roles"])=={"P"},"Public carrier cache role")
        for k in ("image_ids","patient_ids","labels"):
            check(np.array_equal(p[k][sel["indices"]].astype(str),sel[k].astype(str)),"Public carrier evaluation order")
        if reference is None:reference=v
        else:same_order(reference,v)
        ids=v["patient_ids"].astype(np.int64);y=v["labels"]
        check(np.array_equal(np.unique(ids),boot_ids),"Evaluation patient order")
        zv=v["z"].astype(float);public=p["z"][sel["indices"]].astype(float)
        if receiver=="ResNet18":check(np.array_equal(public,sel["ResNet18"]),"Public RN cache mismatch")
        syn={d:io.nz(d+"_synthetic_"+short) for d in ("DP1","DP2")}
        for draw in ("DP1","DP2"):
            check(np.array_equal(syn[draw]["labels"],[0]*64+[1]*64),"Synthetic image order")
            for carrier,z in (("SYN",syn[draw]["z"].astype(float)),("PUBLIC",public)):
                for recipe in ("RN_ONLY","JOINT"):
                    io.check_time();cell=draw+"_"+carrier+"_"+recipe;l=labels[cell]
                    w=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128)
                    dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l)
                    s=zv@w;e=float(abs(s-zv@dual).max())
                    verification["ridge_primal_dual_max_abs"]=max(verification["ridge_primal_dual_max_abs"],e)
                    check(e<1e-10,"Independent receiver ridge check")
                    name=cell+"_"+receiver;scores[name]=s;readout_weights[name]=w
                    b=bootstrap_metrics(y,s,ids,counts,boot_ids);boots[name]=b
                    point=[roc_auc_score(y,s),average_precision_score(y,s)]
                    metrics[name]={m:{"point":float(point[j]),"patient_cluster_95":np.quantile(b[:,j],[.025,.975]).tolist()}
                                   for j,m in enumerate(("AUROC","AP"))}
                    _,inverse=np.unique(ids,return_inverse=True)
                    for index in (0,17,1999):
                        sw=counts[index,inverse]
                        independent=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
                        err=float(abs(independent-b[index]).max())
                        verification["independent_bootstrap_max_abs"]=max(verification["independent_bootstrap_max_abs"],err)
                        check(err<1e-12,"Paired bootstrap differs from independent sklearn witness")
        print('{"phase":"RECEIVER_EVALUATED","receiver":"'+receiver+'"}',flush=True)
    contrasts={};meancontrasts={};contrast_draws={};summary={}
    for receiver in ("DenseNet","ResNet18"):
        for recipe in ("RN_ONLY","JOINT"):
            differences=[];pointdiffs=[]
            for draw in ("DP1","DP2"):
                a=draw+"_SYN_"+recipe+"_"+receiver;b=draw+"_PUBLIC_"+recipe+"_"+receiver
                dist=boots[a]-boots[b];differences.append(dist)
                points=np.array([metrics[a][m]["point"]-metrics[b][m]["point"] for m in ("AUROC","AP")])
                pointdiffs.append(points);key=draw+"_"+recipe+"_SYN_minus_PUBLIC_"+receiver;contrast_draws[key]=dist
                contrasts[key]={m:{"delta":float(points[j]),"patient_cluster_95":np.quantile(dist[:,j],[.025,.975]).tolist()}
                                for j,m in enumerate(("AUROC","AP"))}
            md=(differences[0]+differences[1])/2;mp=np.mean(pointdiffs,axis=0)
            key="MEAN_"+recipe+"_SYN_minus_PUBLIC_"+receiver;contrast_draws[key]=md
            record={m:{"mean_delta":float(mp[j]),"fixed_two_release_patient_cluster_95":np.quantile(md[:,j],[.025,.975]).tolist()}
                    for j,m in enumerate(("AUROC","AP"))}
            criteria={"both_AUROC_differences_positive":bool(all(v[0]>0 for v in pointdiffs)),
                "mean_AUROC_at_least_0_01":bool(mp[0]>=.01),"mean_AP_nonnegative":bool(mp[1]>=0),
                "mean_conditional_AUROC_CI_positive":bool(record["AUROC"]["fixed_two_release_patient_cluster_95"][0]>0)}
            conflict=any(contrasts[d+"_"+recipe+"_SYN_minus_PUBLIC_"+receiver]["AP"]["patient_cluster_95"][1]<0 for d in ("DP1","DP2"))
            record.update(development_criteria=criteria,per_release_significant_AP_harm=conflict,
                          passes_development_hurdle=all(criteria.values()) and not conflict,
                          statistical_scope="Fixed labels/carriers/two DP releases; not noise-population expectation")
            meancontrasts[key]=record
            if receiver=="DenseNet":
                summary[recipe]=("SYNTHETIC_CARRIER_POSITIVE_DEVELOPMENT_SUPPORT" if record["passes_development_hurdle"]
                                 else "SYNTHETIC_CARRIER_ADDITIONAL_VALUE_NOT_ESTABLISHED")
    for name in labels:
        check(name in construction["fits"],"Frozen construction/evaluation cells differ")
    np.savez_compressed(out/"predictions_private.npz",names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),
        labels=reference["labels"],patient_ids=reference["patient_ids"],image_ids=reference["image_ids"])
    np.savez_compressed(out/"paired_bootstrap.npz",names=np.array(list(boots)),metrics=np.stack(list(boots.values()),axis=1),
        source_draws_sha256=np.array(sha(io.path("patient_bootstrap"))))
    np.savez_compressed(out/"contrast_bootstrap.npz",names=np.array(list(contrast_draws)),differences=np.stack(list(contrast_draws.values()),axis=1))
    np.savez_compressed(out/"readout_weights.npz",**readout_weights)
    io.verify();dump(out/"evaluation_inputs.json",io.receipts)
    result={"status":"STEP3_PUBLIC128_COMPARISON_COMPLETE","created":now(),"whole_project_stage":2,
        "primary_receiver":"DenseNet","construction_receiver":"ResNet18","metric_unit":"image",
        "uncertainty_unit":"paired patient cluster","bootstrap_draws":2000,
        "metrics":metrics,"per_release_carrier_contrasts":contrasts,"fixed_two_release_mean_contrasts":meancontrasts,
        "development_decisions":summary,"verification":verification,
        "label_anchor":"zero for all carriers/recipes; historical hard-anchor scores not part of primary contrasts",
        "recipe_scope":"RN-only and Joint evaluated separately; no best-recipe selection",
        "new_label_packets":8,"new_private_releases":0,"new_Q_access":0,"new_pixel_training":0,
        "new_model_forwards":0,"Expert_Reserved":False,"new_receivers":0,
        "evaluation_seconds":time.monotonic()-tick,"contract_sha256":sha(out/"contract.json")}
    dump(out/"results.json",result)
    write_seal(out,"completion.json","STEP3_PUBLIC128_COMPARISON_COMPLETE",
       ["contract.json","selection_seal.json","label_seal.json","results.json","evaluation_inputs.json",
        "predictions_private.npz","paired_bootstrap.npz","contrast_bootstrap.npz","readout_weights.npz"])
    print(json.dumps({"status":result["status"],"decisions":summary,"means":meancontrasts,
                      "verification":verification,"seconds":result["evaluation_seconds"]},indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);run(p.parse_args().out)

