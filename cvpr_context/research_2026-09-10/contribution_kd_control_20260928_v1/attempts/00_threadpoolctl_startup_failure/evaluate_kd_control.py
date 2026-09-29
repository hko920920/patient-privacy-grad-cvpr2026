"""Evaluate only after both new label packets have been frozen."""
import argparse, sys, time, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from common import Inputs, check, dump, sha, now

def run(out):
    started=time.monotonic()
    io=Inputs(out,"evaluate")
    seal=json.loads((out/"label_seal.json").read_text(encoding="utf-8"))
    check(seal["status"]=="BOTH_KD_LABELS_FROZEN_BEFORE_EVALUATION","Missing construction seal")
    check(sha(out/"contract.json")==seal["contract_sha256"],"Contract changed")
    for name,h in seal["files"].items():
        check(sha(out/name)==h,"Sealed construction artifact changed")
    for packet in seal["packets"].values():
        check(sha(packet["CSV"])==packet["CSV_sha256"],"Label CSV changed")
    check(not (out/"results.json").exists(),"Evaluation already complete")
    sys.path.insert(0,io.contract["code_root"])
    from prrd_pilot_20260921.common import bootstrap_metrics
    with np.load(out/"KD_soft_labels.npz",allow_pickle=False) as f:
        labels={k:f[k] for k in f.files}
    with np.load(out/"target_weights.npz",allow_pickle=False) as f:
        target_weights={k:f[k] for k in f.files}
    oldlabels=io.npz("historical_MT_labels")
    oldpred=io.npz("historical_MT_predictions")
    oldboot=io.npz("historical_MT_bootstrap")
    oldmetrics=io.js("historical_MT_results")
    oldtransport=io.npz("historical_transport_predictions")
    oldtransportboot=io.npz("historical_transport_bootstrap")
    boot=io.npz("patient_bootstrap")
    check(str(oldtransportboot["source_draws_sha256"].item())==sha(io.receipts["patient_bootstrap"]["path"]),
          "Bootstrap schedule differs from historical evaluation")
    check(boot["patient_counts"].shape[0]==2000,"Wrong number of patient draws")
    y=oldpred["labels"]; ids=oldpred["patient_ids"].astype(np.int64)
    patients,inverse=np.unique(ids,return_inverse=True)
    check(np.array_equal(patients,boot["patient_ids"]),"Bootstrap patient order")
    scores,boots,metrics={},{},{}
    validation={"primal_dual_score_max_abs":0.0,"historical_MT_score_reproduction_max_abs":0.0,
                "independent_bootstrap_max_abs":0.0,"historical_target_weight_max_abs":0.0}
    for receiver in ("DenseNet","ResNet18"):
        val=io.npz("V_"+receiver)
        for key in ("labels","patient_ids","image_ids"):
            check(np.array_equal(val[key].astype(str),oldpred[key].astype(str)),"Evaluation alignment: "+key)
        zval=val["z"].astype(np.float64)
        for draw in ("DP1","DP2"):
            check(time.monotonic()-started<600,"Evaluation worker time cap")
            syn=io.npz(draw+"_synthetic_"+("RN" if receiver=="ResNet18" else "DN"))
            z=syn["z"].astype(np.float64)
            check(np.array_equal(syn["labels"],[0]*64+[1]*64),"PNG feature label order")
            l=labels[draw]
            wk=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@l/128)
            dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),l)
            sk=zval@wk
            error=float(np.max(abs(sk-zval@dual)))
            validation["primal_dual_score_max_abs"]=max(validation["primal_dual_score_max_abs"],error)
            check(error<1e-10,"Primal / dual prediction mismatch")
            n=draw+"_KD_"+receiver
            scores[n]=sk
            boots[n]=bootstrap_metrics(y,sk,ids,boot["patient_counts"],boot["patient_ids"])
            oldname=draw+"_JOINT_eval_"+receiver
            oi=list(oldpred["names"]).index(oldname)
            bi=list(oldboot["names"]).index(oldname)
            m=draw+"_MT_"+receiver
            scores[m]=oldpred["scores"][:,oi]
            boots[m]=oldboot["metrics"][:,bi,:]
            wm=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T@oldlabels[draw]/128)
            error=float(np.max(abs(zval@wm-scores[m])))
            validation["historical_MT_score_reproduction_max_abs"]=max(validation["historical_MT_score_reproduction_max_abs"],error)
            check(error<1e-10,"Historical MT prediction no longer reproduced")
            if receiver=="ResNet18":
                error=float(np.max(abs(target_weights[draw+"_MT_RN"]-oldtransport["ResNet18_TRANSPORT_"+draw+"_weight"])))
                validation["historical_target_weight_max_abs"]=max(validation["historical_target_weight_max_abs"],error)
                check(error<1e-10,"MT target reconstruction differs")
            for name in (n,m):
                point=np.array([roc_auc_score(y,scores[name]),average_precision_score(y,scores[name])])
                metrics[name]={metric:{"point":float(point[j]),
                    "patient_cluster_95":np.quantile(boots[name][:,j],[.025,.975]).tolist()}
                    for j,metric in enumerate(("AUROC","AP"))}
                if name==m:
                    check(max(abs(point[j]-oldmetrics["metrics"][oldname][metric]) for j,metric in enumerate(("AUROC","AP")))<1e-12,
                          "Historical metric changed")
                for b in (0,17,1999):
                    weights=boot["patient_counts"][b,inverse]
                    independent=np.array([roc_auc_score(y,scores[name],sample_weight=weights),
                        average_precision_score(y,scores[name],sample_weight=weights)])
                    error=float(np.max(abs(independent-boots[name][b])))
                    validation["independent_bootstrap_max_abs"]=max(validation["independent_bootstrap_max_abs"],error)
                    check(error<1e-12,"Independent sklearn bootstrap mismatch")
    contrasts={}; averages={}
    for receiver in ("DenseNet","ResNet18"):
        distributions=[]
        pointdiffs=[]
        for draw in ("DP1","DP2"):
            m=draw+"_MT_"+receiver; k=draw+"_KD_"+receiver
            delta=np.array([metrics[m][v]["point"]-metrics[k][v]["point"] for v in ("AUROC","AP")])
            dist=boots[m]-boots[k]
            distributions.append(dist);pointdiffs.append(delta)
            contrasts[draw+"_MT_minus_KD_"+receiver]={v:{"delta":float(delta[j]),
                "patient_cluster_95":np.quantile(dist[:,j],[.025,.975]).tolist()}
                for j,v in enumerate(("AUROC","AP"))}
        points=np.mean(pointdiffs,axis=0); distribution=np.mean(distributions,axis=0)
        averages[receiver]={v:{"mean_delta":float(points[j]),
            "fixed_two_release_patient_cluster_95":np.quantile(distribution[:,j],[.025,.975]).tolist()}
            for j,v in enumerate(("AUROC","AP"))}
        averages[receiver]["not_prediction_ensemble"]=True
    dn=averages["DenseNet"]
    both_positive=all(contrasts[d+"_MT_minus_KD_DenseNet"]["AUROC"]["delta"]>0 for d in ("DP1","DP2"))
    practical=both_positive and dn["AUROC"]["mean_delta"]>=.01 and dn["AP"]["mean_delta"]>=0
    ci=dn["AUROC"]["fixed_two_release_patient_cluster_95"]
    if practical and ci[0]>0:
        decision="MT_ADDITIONAL_DEVELOPMENT_VALUE_OVER_DEFINED_KD"
    elif practical:
        decision="PROMISING_MT_POINT_GAINS_BUT_CONDITIONAL_UNCERTAINTY"
    elif ci[1]<0:
        decision="DEFINED_KD_FAVORED_IN_CURRENT_FIXED_BANK_COMPARISON"
    else:
        decision="MIXED_OR_NO_CLEAR_MT_ADDITIONAL_VALUE"
    names=list(scores)
    np.savez_compressed(out/"predictions_private.npz",names=np.array(names),
        scores=np.column_stack([scores[n] for n in names]),labels=y,patient_ids=ids,image_ids=oldpred["image_ids"])
    np.savez_compressed(out/"paired_bootstrap.npz",names=np.array(names),metrics=np.stack([boots[n] for n in names],axis=1),
        source_draws_sha256=np.array(sha(io.receipts["patient_bootstrap"]["path"])))
    io.verify_receipts()
    dump(out/"evaluation_inputs.json",io.receipts)
    result={"status":"COMPLETE_STEP1_KD_CONTROL","completed_utc":now(),
        "scope":"Same two existing DP summaries and PNGs, unchanged hard-anchor joint solver; only RN target rule differs.",
        "decision":decision,"construction_models":["DINOv2","ResNet18"],"primary_development_receiver":"DenseNet",
        "metrics":metrics,"MT_minus_KD":contrasts,"two_fixed_release_averages":averages,
        "validation":validation,"public128":False,"source_only_new_runs":0,"new_Q_access":0,
        "new_DP_releases":0,"new_pixel_updates":0,"new_model_forwards":0,"Expert_Reserved":False,
        "bootstrap_repeats":2000,"metrics_unit":"images","bootstrap_unit":"patient_cluster",
        "new_label_packets":2,"evaluation_seconds":time.monotonic()-started,
        "contract_sha256":sha(out/"contract.json"),"label_seal_sha256":sha(out/"label_seal.json")}
    dump(out/"results.json",result)
    dump(out/"completion.json",{"status":result["status"],"completed_utc":now(),
        "files":{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ("completion.json","access_log.jsonl")},
        "output_bytes":sum(p.stat().st_size for p in out.rglob("*") if p.is_file())})
    print(json.dumps({"decision":decision,"metrics":{n:v for n,v in metrics.items() if "DenseNet" in n},
        "contrasts":{n:v for n,v in contrasts.items() if "DenseNet" in n},
        "averages":averages,"validation":validation,"evaluation_seconds":result["evaluation_seconds"]},indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4):
        run(a.out)

