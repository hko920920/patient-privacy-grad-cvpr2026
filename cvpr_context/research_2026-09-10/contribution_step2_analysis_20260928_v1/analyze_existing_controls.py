"""Step 2: analysis of existing predictions/bootstrap only; no training or label solve."""
from pathlib import Path
import argparse, datetime, hashlib, json, time
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(value,message):
    if not value:
        raise RuntimeError(message)

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def manual_interval(x):
    # Independent sorted order-statistic interpolation for default linear quantiles.
    x=sorted(float(v) for v in x)
    ans=[]
    for p in (.025,.975):
        index=(len(x)-1)*p
        a=int(np.floor(index)); b=int(np.ceil(index))
        ans.append(x[a]+(index-a)*(x[b]-x[a]))
    return np.asarray(ans)

def run(out):
    tick=time.monotonic()
    contract=json.loads((out/"contract.json").read_text(encoding="utf-8"))
    require(sha(Path(__file__))==contract["analysis_script_sha256"],"Analysis code changed after freeze")
    require(not (out/"results.json").exists(),"Results already exist; no silent rerun")
    receipts={}
    def path(name):
        item=contract["inputs"][name];p=Path(item["path"])
        require(sha(p)==item["sha256"],"Frozen input changed: "+name)
        receipts[name]=item
        return p
    def js(name):
        return json.loads(path(name).read_text(encoding="utf-8"))
    def nz(name):
        with np.load(path(name),allow_pickle=False) as f:
            return {k:f[k] for k in f.files}
    old=js("joint_results")
    source_results=js("source_results")
    rn_results=js("compiler_results")
    hist_manifest=js("historical_manifest")
    step1_receipts=js("step1_evaluation_inputs")
    # Reuse prior validated script and input bindings, do not rerun historical workers.
    for key in ("joint_code","source_code","compiler_code"):
        p=path(key)
        require(sha(p)==hist_manifest["file_hashes"][p.name],"Historical operation changed: "+key)
    for oldkey,key in (("historical_MT_predictions","joint_predictions"),
                       ("historical_MT_bootstrap","joint_bootstrap"),
                       ("historical_MT_results","joint_results")):
        require(step1_receipts[oldkey]["sha256"]==contract["inputs"][key]["sha256"],
                "Input differs from completed step1: "+key)
    pred=nz("joint_predictions");boot=nz("joint_bootstrap")
    srcpred=nz("source_predictions");rnpred=nz("compiler_predictions")
    transport_boot=nz("transport_bootstrap")
    require(str(transport_boot["source_draws_sha256"].item())==sha(path("patient_schedule")),
            "Patient bootstrap schedule changed")
    for data in (srcpred,rnpred):
        for k in ("labels","patient_ids","image_ids"):
            require(np.array_equal(pred[k].astype(str),data[k].astype(str)),"Historical patient/image alignment: "+k)
    require(boot["metrics"].shape[0]==2000 and boot["metrics"].shape[2]==2,"Bootstrap schema")
    require(np.isfinite(boot["metrics"]).all(),"Nonfinite historical bootstrap metric")
    labs={"Joint":nz("joint_labels"),"DINO_only":nz("source_labels"),"RN_only":nz("compiler_labels")}
    table={};names=[];distributions=[];contrasts={};averages={};verification={
        "historical_score_column_max_abs":0.0,"historical_point_difference_max_abs":0.0,
        "historical_CI_max_abs":0.0,"manual_quantile_max_abs":0.0,
        "same_patient_schedule":True,"bootstrap_draws":2000,
        "new_patient_metric_evaluations":0,"new_bootstrap_resampling":0}
    def interval(x):
        q=np.quantile(x,[.025,.975])
        error=float(np.max(abs(q-manual_interval(x))))
        verification["manual_quantile_max_abs"]=max(verification["manual_quantile_max_abs"],error)
        require(error<1e-12,"Independent quantile interpolation mismatch")
        return q.tolist()
    for draw in ("DP1","DP2"):
        points={}
        keys={"Joint":draw+"_JOINT_eval_DenseNet",
              "DINO_only":draw+"_SOURCE_LABEL_SOLVE_eval_DenseNet",
              "RN_only":draw+"_RESNET_ONLY_eval_DenseNet"}
        points={short:old["metrics"][name] for short,name in keys.items()}
        table[draw]=points
        for short in keys:
            labelkey=draw if short=="Joint" else draw+"_SOURCE_LABEL_SOLVE" if short=="DINO_only" else draw+"_fit_ResNet18_DP"
            label=labs[short][labelkey]
            require(label.shape==(128,) and np.isfinite(label).all() and abs(label).max()<=1,
                    "Saved label packet invalid")
        for short,data,external_name in (
            ("DINO_only",srcpred,draw+"_SOURCE_LABEL_SOLVE_eval_DenseNet"),
            ("RN_only",rnpred,draw+"_fit_ResNet18_DP_eval_DenseNet")):
            internal=pred["scores"][:,list(pred["names"]).index(keys[short])]
            external=data["scores"][:,list(data["names"]).index(external_name)]
            error=float(np.max(abs(internal-external)))
            verification["historical_score_column_max_abs"]=max(verification["historical_score_column_max_abs"],error)
            require(error<1e-12,"Control prediction column not identical")
            independent=source_results["results"][draw+"_SOURCE_LABEL_SOLVE"]["receivers"]["DenseNet"] if short=="DINO_only" else rn_results["metrics"][external_name]
            require(max(abs(points[short][m]-independent[m]) for m in ("AUROC","AP"))<1e-12,
                    "Independent historical JSON metric differs")
        bj=boot["metrics"][:,list(boot["names"]).index(keys["Joint"]),:]
        for comparator,kind in (("DINO_only","SOURCE_LABEL_SOLVE"),("RN_only","RESNET_ONLY")):
            baseline=boot["metrics"][:,list(boot["names"]).index(keys[comparator]),:]
            diff=bj-baseline
            contrast=draw+"_Joint_minus_"+comparator
            names.append(contrast);distributions.append(diff)
            observed=old["comparisons"][draw+"_JOINT_eval_DenseNet_minus_"+kind]
            record={}
            for j,m in enumerate(("AUROC","AP")):
                point=points["Joint"][m]-points[comparator][m]
                ci=interval(diff[:,j])
                pe=abs(point-observed[m]["delta"])
                ce=float(np.max(abs(np.asarray(ci)-observed[m]["patient_cluster_95"])))
                verification["historical_point_difference_max_abs"]=max(verification["historical_point_difference_max_abs"],pe)
                verification["historical_CI_max_abs"]=max(verification["historical_CI_max_abs"],ce)
                require(pe<1e-12 and ce<1e-12,"Existing contrast not reproduced")
                record[m]={"delta":float(point),"patient_cluster_95":ci}
            contrasts[contrast]=record
    for comparator in ("DINO_only","RN_only"):
        keys=["DP1_Joint_minus_"+comparator,"DP2_Joint_minus_"+comparator]
        average=(distributions[names.index(keys[0])]+distributions[names.index(keys[1])])/2
        name="MEAN_Joint_minus_"+comparator
        names.append(name);distributions.append(average)
        record={}
        for j,m in enumerate(("AUROC","AP")):
            delta=(contrasts[keys[0]][m]["delta"]+contrasts[keys[1]][m]["delta"])/2
            alternative=(table["DP1"]["Joint"][m]+table["DP2"]["Joint"][m]
                        -table["DP1"][comparator][m]-table["DP2"][comparator][m])/2
            require(abs(delta-alternative)<1e-14,"Average difference arithmetic")
            record[m]={"mean_delta":float(delta),"fixed_two_release_patient_cluster_95":interval(average[:,j])}
        gate={"both_AUROC_point_differences_positive":all(contrasts[k]["AUROC"]["delta"]>0 for k in keys),
              "mean_AUROC_at_least_0_01":record["AUROC"]["mean_delta"]>=.01,
              "mean_AP_nonnegative":record["AP"]["mean_delta"]>=0,
              "conditional_mean_AUROC_CI_positive":record["AUROC"]["fixed_two_release_patient_cluster_95"][0]>0}
        record["development_investment_checks"]=gate
        record["passes_all_development_checks"]=all(gate.values())
        record["not_an_ensemble"]=True
        record["not_independent_confirmation"]=True
        averages[comparator]=record
    method_averages={method:{m:float((table["DP1"][method][m]+table["DP2"][method][m])/2)
                              for m in ("AUROC","AP")} for method in ("DINO_only","RN_only","Joint")}
    for item in contract["inputs"].values():
        require(sha(item["path"])==item["sha256"],"Input mutated during analysis")
    a=averages["DINO_only"]["passes_all_development_checks"]
    b=averages["RN_only"]["passes_all_development_checks"]
    result={
        "status":"COMPLETE_EXISTING_STEP2_ANALYSIS","completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question_2A":"Adding publicly transported RN target versus DINO-only",
        "question_2A_decision":"POSITIVE_CONDITIONAL_DEVELOPMENT_SUPPORT" if a else "MIXED_OR_UNCERTAIN",
        "question_2B":"Joint target versus publicly transported RN-only",
        "question_2B_decision":"POSITIVE_CONDITIONAL_DEVELOPMENT_SUPPORT" if b else "JOINT_NECESSITY_NOT_ESTABLISHED",
        "primary_receiver":"DenseNet","metric_unit":"image","uncertainty_unit":"paired_patient_cluster",
        "table":table,"method_metric_means":method_averages,"per_release_contrasts":contrasts,
        "fixed_two_release_mean_contrasts":averages,"verification":verification,
        "historical_results_reused":True,"new_model_forwards":0,"new_label_solves":0,
        "new_pixel_training":0,"new_Q_access":0,"new_DP_releases":0,"Expert_Reserved":False,
        "public128":False,"step3_started":False,"analysis_seconds":time.monotonic()-tick,
        "contract_sha256":sha(out/"contract.json"),"analysis_script_sha256":sha(__file__)}
    np.savez_compressed(out/"derived_contrast_bootstrap.npz",names=np.array(names),
        differences=np.stack(distributions,axis=1),source_schedule_sha256=np.array(sha(path("patient_schedule"))))
    dump(out/"input_receipts.json",receipts)
    dump(out/"results.json",result)
    dump(out/"completion.json",{"status":result["status"],"files":{
        n:sha(out/n) for n in ("contract.json","analyze_existing_controls.py","results.json","derived_contrast_bootstrap.npz","input_receipts.json")}})
    print(json.dumps({"2A":result["question_2A_decision"],"2B":result["question_2B_decision"],
                      "means":averages,"method_means":method_averages,
                      "verification":verification,"seconds":result["analysis_seconds"]},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)

