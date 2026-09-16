"""Four-cell E/U evaluation with hash-pinned original CDI patient analysis kernels.
E-fit scorers are newly fit on E-fit80 only; U-fit scorers/choices are restored.
Original U outputs are immutable and are used verbatim after restoration checks.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from . import analyze_cdi_u_cohort as original

ROOT=original.ROOT
RUN=original.RUN
CONTRACT=original.DEFAULT_CONTRACT.parent/"cdi_eu_analysis_contract_20260915.json"
CONTRACT_SHA="bc2497d17b3110ec8b994eaa541d4eda00891ff5d30215a9aff26b7aa5f1db43"
HELPER_SHA="752be6cdf1db57b2309bd897fee8f43f53f1807d23fa92cb07973b4720d14268"
digest=original.digest
read=original.read_json
write=original.save_json
require=original.require

def restore(entry,config):
    """Restore fitted sklearn attributes, without calling any fitting method."""
    p=entry["parameters"]
    scaler=StandardScaler(with_mean=True,with_std=True)
    scaler.mean_=np.asarray(p["scaler_mean"],dtype=np.float64)
    scaler.scale_=np.asarray(p["scaler_scale"],dtype=np.float64)
    scaler.var_=np.asarray(p["scaler_var"],dtype=np.float64)
    scaler.n_samples_seen_=np.int64(p["scaler_n_samples_seen"])
    scaler.n_features_in_=len(scaler.mean_)
    lr=LogisticRegression(C=entry["chosen_C"],solver=config["solver"],penalty="l2",
        max_iter=config["max_iter"],random_state=config["random_state"],class_weight=None)
    lr.coef_=np.asarray(p["coef"],dtype=np.float64)
    lr.intercept_=np.asarray(p["intercept"],dtype=np.float64)
    lr.classes_=np.asarray(p["classes"],dtype=np.int64)
    lr.n_iter_=np.asarray(p["n_iter"],dtype=np.int32)
    lr.n_features_in_=scaler.n_features_in_
    require(lr.classes_.tolist()==[0,1],"saved patient-member direction")
    return scaler,lr

def validate_contract(path):
    require(digest(path)==CONTRACT_SHA,"frozen pre-E analysis contract")
    require(digest(original.__file__)==HELPER_SHA,"frozen original numerical/fitting helper")
    c=read(path)
    require(c["schema"]=="cdi-eu-patient-analysis-contract/v1" and c["research_stage"]==2,"contract scope")
    require(c["models"]==["model_1","model_2"] and c["fit_scenarios"]==c["evaluation_scenarios"]==["E","U"],"four fixed cells")
    for p,h in c["frozen_sha256"].items():require(digest(p)==h,"unchanged predeclared input "+p)
    prior=read(original.DEFAULT_CONTRACT)
    require(c["methods"]==prior["methods"] and c["folds"]==prior["folds"]
            and c["bootstrap"]==prior["bootstrap"],"same18 methods/folds/bootstrap")
    require(len(c["methods"])==18 and sum(s["kind"]=="logistic_regression" for s in c["methods"])==10,"same method count")
    for key,value in prior["logistic_regression"].items():
        if key!="per_image_fit_label":require(c["logistic_regression"][key]==value,"unchanged LR policy "+key)
    require(c["original_within_cell_contrasts"]==prior["contrasts"],"same within-cell method comparisons")
    for p,old in zip(c["patient_rows"],prior["patient_rows"]):
        require(all(p[k]==old[k] for k in ("patient_id","role","group"))
                and p["images"]["U"]==old["images"] and len(p["images"]["E"])==2
                and not set(p["images"]["E"])&set(p["images"]["U"]),"same patients and disjoint E/U photographs")
    return c

def load_data(c):
    data,all_rows={},{}
    protocols={}
    for scenario in ("E","U"):
        directory=Path(c["inputs"][scenario]["directory"])
        verification=read(directory/c["inputs"][scenario]["verification_file"])
        require(verification["status"]==c["inputs"][scenario]["required_status"],scenario+" raw verification required")
        if scenario=="U":
            require(verification["analysis_verification"]["status"]=="PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS","original U analysis verified")
        protocol,execution=read(directory/"protocol.json"),read(directory/"execution.json")
        protocols[scenario]=protocol
        require(execution["complete"] is True and execution["records"]==480
                and execution["results_sha256"]==digest(directory/"results.json")
                and execution["protocol_sha256"]==digest(directory/"protocol.json"),scenario+" extraction binding")
        require(verification["results_sha256"]==digest(directory/"results.json"),scenario+" verified exact results")
        rows=read(directory/"results.json")
        index={(r["model"],r["image_id"]):r for r in rows}
        expected={(m,i) for m in c["models"] for p in c["patient_rows"] for i in p["images"][scenario]}
        require(len(rows)==len(index)==480 and set(index)==expected,scenario+" exact480 planned images")
        all_rows[scenario]=index;data[scenario]={}
        for model in c["models"]:
            data[scenario][model]={}
            for role in ("fit","selection"):
                patients=[dict(p,images=p["images"][scenario]) for p in c["patient_rows"] if p["role"]==role]
                features,labels=[],[]
                for p in patients:
                    rs=[index[(model,i)] for i in p["images"]]
                    expected_member=int(p["group"]==("A" if model=="model_1" else "B"))
                    for r in rs:
                        require(r["patient_id"]==p["patient_id"] and r["eval_role"]==role
                                and r["assignment_group"]==p["group"] and r["scenario"]==scenario,"image/patient metadata")
                        actual_member=int(r["patient_training_exposures"]>0)
                        require(r["member"]==actual_member==expected_member,"actual patient participation label")
                        expected_image_member=expected_member if scenario=="E" else 0
                        require(r["image_member"]==expected_image_member
                                and int(r["actual_training_exposures"]>0)==expected_image_member,"E/U actual image exposure semantics")
                        require(r["record_role"]==("train_candidate" if scenario=="E" else "U_observed"),"E/U image role")
                        require(r["feature_names"]==original.FEATURE_NAMES and len(r["features"])==26,"original26D order")
                    require(rs[0]["patient_training_exposures"]==rs[1]["patient_training_exposures"],"same patient exposure total")
                    labels.append(expected_member)
                    features.append([r["features"]+[r["modules"]["noise_optim"]["optimizer"]["fun"]] for r in rs])
                x=np.asarray(features,dtype=np.float64)
                require(x.shape==(80 if role=="fit" else 40,2,27) and np.isfinite(x).all(),"same finite feature dimensions")
                data[scenario][model][role]=dict(patients=patients,x=x,y=np.asarray(labels,dtype=np.int64))
    require(protocols["E"]["source_bindings"]==protocols["U"]["source_bindings"],"same literal source extractors")
    for key in ("master_seed","stream","batch_size","prediction_type","dtype","prompt","code_policy"):
        require(protocols["E"]["contract"][key]==protocols["U"]["contract"][key],"matched E/U extraction "+key)
    for model in c["models"]:
        for p in c["patient_rows"]:
            e=all_rows["E"][(model,p["images"]["E"][0])]
            u=all_rows["U"][(model,p["images"]["U"][0])]
            for k in ("member","patient_training_exposures","checkpoint_sha256","cache_sha256","cohort_lock_sha256"):
                require(e[k]==u[k],"matched actual E/U target/input "+k)
    return data

def scalar_score(x,spec):
    scores=spec["direction"]*x[:,:,spec["feature_index"]]
    return scores.mean(1) if spec["pool_after_direction"]=="mean" else scores.max(1)

def run(contract_path,expected_source):
    started=time.perf_counter()
    require(digest(__file__)==expected_source,"externally frozen analyzer SHA")
    c=validate_contract(contract_path)
    out=Path(c["output_directory"])
    require(not out.exists(),"immutable E/U analysis output")
    U=Path(c["inputs"]["U"]["directory"]);old_dir=U/"analysis_v1"
    inputs={Path(p) for p in c["frozen_sha256"]}
    inputs.add(Path(contract_path))
    for scenario in ("E","U"):
        directory=Path(c["inputs"][scenario]["directory"])
        inputs.update(directory/n for n in ("protocol.json","execution.json","results.json",c["inputs"][scenario]["verification_file"]))
    hashes={str(p.resolve()):digest(p) for p in sorted(inputs,key=str)}
    out.mkdir()
    write(out/"protocol.json",dict(schema="cdi-eu-analysis-execution/v1",created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_contract_path=str(contract_path.resolve()),analysis_contract_sha256=CONTRACT_SHA,
        source_path=str(Path(__file__).resolve()),source_sha256=expected_source,helper_sha256=HELPER_SHA,
        input_sha256=hashes,analysis_policy="four predeclared cells; new E-fit only; exact saved U reuse",
        numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version()))
    data=load_data(c)
    old_prov=read(old_dir/"provenance.json")
    require(old_prov["numpy"]==np.__version__ and old_prov["sklearn"]==sklearn.__version__,"same libraries for exact U restoration")
    for name,h in old_prov["output_sha256"].items():require(digest(old_dir/name)==h,"original U artifacts unchanged")
    old_params={(r["model"],r["method"]):r for r in read(old_dir/"fit_parameters.json")}
    old_cv={(r["model"],r["method"]):r for r in read(old_dir/"cv_results.json")}
    old_predictions={(r["model"],r["method"],r["role"],r["patient_id"]):r for r in read(old_dir/"predictions.json")}
    require(len(old_params)==len(old_cv)==20,"20 original frozen U scorers")
    scores,fit_scores,parameters,cv_results={},{},[],[]
    restored_max_error=0.
    for fit_scenario in ("E","U"):
        for model in c["models"]:
            fit=data[fit_scenario][model]["fit"]
            for spec in c["methods"]:
                method=spec["id"]
                if spec["kind"]=="logistic_regression":
                    if fit_scenario=="E":
                        pred,par,cv,scaler,lr=original.fit_family(model,fit,data["E"][model]["selection"],spec,c)
                        fit_pred=pred["fit"]
                        selection_predictions={"E":pred["selection"],
                            "U":original.predict_patients(data["U"][model]["selection"]["x"],spec,scaler,lr)}
                    else:
                        par,cv=old_params[(model,method)],old_cv[(model,method)]
                        scaler,lr=restore(par,c["logistic_regression"])
                        fit_pred=original.predict_patients(fit["x"],spec,scaler,lr)
                        selection_predictions={s:original.predict_patients(data[s][model]["selection"]["x"],spec,scaler,lr) for s in ("E","U")}
                    parameters.append(dict(fit_scenario=fit_scenario,reused_original_U=fit_scenario=="U",**par))
                    cv_results.append(dict(fit_scenario=fit_scenario,reused_original_U=fit_scenario=="U",**cv))
                else:
                    fit_pred=scalar_score(fit["x"],spec)
                    selection_predictions={s:scalar_score(data[s][model]["selection"]["x"],spec) for s in ("E","U")}
                if fit_scenario=="U":
                    for role,computed in (("fit",fit_pred),("selection",selection_predictions["U"])):
                        patients=data["U"][model][role]["patients"]
                        old=np.asarray([old_predictions[(model,method,role,p["patient_id"])]["score"] for p in patients])
                        error=float(np.max(np.abs(computed-old)))
                        restored_max_error=max(restored_max_error,error)
                        require(np.array_equal(computed,old),"exact unchanged U score restoration: "+model+"/"+method+"/"+role)
                        if role=="fit":fit_pred=old
                        else:selection_predictions["U"]=old
                fit_scores[(fit_scenario,model,method)]=fit_pred
                for evaluation_scenario,pred in selection_predictions.items():
                    require(pred.shape==(40,) and np.isfinite(pred).all(),"patient selection probabilities/scores")
                    scores[(fit_scenario,evaluation_scenario,model,method)]=pred
            print(json.dumps(dict(event="scorers_ready",fit_scenario=fit_scenario,model=model,new_fit=fit_scenario=="E",methods=18)),flush=True)
    for model in c["models"]:
        for spec in c["methods"]:
            if spec["kind"]=="scalar":
                for ev in ("E","U"):
                    require(np.array_equal(scores[("E",ev,model,spec["id"])],scores[("U",ev,model,spec["id"])]),
                            "scalar has no fit-scenario effect")
    patients=data["U"]["model_1"]["selection"]["patients"]
    indices=original.bootstrap_indices(patients,c["bootstrap"]["resamples"],c["bootstrap"]["seed"])
    old_boot=read(old_dir/"bootstrap_resamples.json")
    require(old_boot["selection_patient_order"]==[p["patient_id"] for p in patients]
            and np.array_equal(indices,np.asarray(old_boot["indices"])),"same historical paired patient resamples")
    prediction_rows,fit_rows,results=[],[],[]
    points,boots={},{}
    for key,pred in scores.items():
        fs,es,model,method=key;d=data[es][model]["selection"];y=d["y"]
        auc=float(roc_auc_score(y,pred));bs=original.bootstrap_auc(pred,y,indices)
        points[key]=auc;boots[key]=bs
        results.append(dict(fit_scenario=fs,evaluation_scenario=es,model=model,method=method,
            selection_patient_AUC=auc,selection_patient_AUC_CI95=original.ci(bs),
            selection_patients=40,selection_members=20,selection_nonmembers=20,
            fit_patient_AUC_descriptive_only=float(roc_auc_score(data[fs][model]["fit"]["y"],fit_scores[(fs,model,method)]))))
        for p,label,value in zip(d["patients"],y,pred):
            prediction_rows.append(dict(fit_scenario=fs,evaluation_scenario=es,model=model,method=method,
                patient_id=p["patient_id"],role="selection",group=p["group"],member=int(label),score=float(value)))
    for (fs,model,method),pred in fit_scores.items():
        d=data[fs][model]["fit"]
        for p,label,value in zip(d["patients"],d["y"],pred):
            fit_rows.append(dict(fit_scenario=fs,evaluation_scenario=fs,model=model,method=method,
                patient_id=p["patient_id"],role="fit",group=p["group"],member=int(label),score=float(value)))
    within,cross=[],[]
    for fs in ("E","U"):
        for es in ("E","U"):
            for model in c["models"]:
                for comp in c["original_within_cell_contrasts"]:
                    ka,kb=(fs,es,model,comp["a"]),(fs,es,model,comp["b"])
                    within.append(dict(fit_scenario=fs,evaluation_scenario=es,model=model,**comp,
                        selection_delta_AUC=points[ka]-points[kb],paired_CI95=original.ci(boots[ka]-boots[kb])))
    for model in c["models"]:
        for spec in c["methods"]:
            method=spec["id"]
            for comp in c["cross_cell_contrasts"]:
                a,b=comp["a"],comp["b"]
                ka=(a["fit_scenario"],a["evaluation_scenario"],model,method)
                kb=(b["fit_scenario"],b["evaluation_scenario"],model,method)
                cross.append(dict(model=model,method=method,**comp,
                    selection_delta_AUC=points[ka]-points[kb],paired_CI95=original.ci(boots[ka]-boots[kb])))
    old_result=read(old_dir/"analysis.json")
    for r in old_result["results"]:
        if r["method"]=="dino_existing":continue
        key=("U","U",r["model"],r["method"])
        require(points[key]==r["selection_patient_AUC"] and original.ci(boots[key])==r["selection_patient_AUC_CI95"],"original U AUC/CI exact")
    counts=c["expected_counts"]
    require(len(prediction_rows)==counts["selection_prediction_rows"]==5760
            and len(fit_rows)==counts["same_scenario_fit_prediction_rows"]==5760
            and len(results)==144 and len(within)==48 and len(cross)==108
            and len(parameters)==len(cv_results)==40,"complete predeclared output counts")
    for p,h in hashes.items():require(digest(p)==h,"unchanged input after fitting/analysis")
    require(digest(__file__)==expected_source and digest(original.__file__)==HELPER_SHA,"unchanged sources")
    result=dict(schema="cdi-eu-patient-analysis/v1",status="COMPUTED_PENDING_INDEPENDENT_ANALYSIS_VERIFICATION",
        research_stage=2,computed_methods=18,new_E_scorers=20,reused_U_scorers=20,
        results=results,within_cell_contrasts=within,cross_cell_contrasts=cross,
        original_U_scores_exact=True,original_U_max_abs_restoration_error=restored_max_error,
        original_U_AUC_CI_exact=True,new_GPU_calls=0,new_target_training=False,
        selection_used_for_fitting_tuning_or_direction=False,calibration_test_used=False,
        scope_limits=c["scope_limits"],bootstrap=c["bootstrap"],overall_stage2_gate=None,
        seconds_cpu_analysis=time.perf_counter()-started)
    write(out/"contract_copy.json",c);write(out/"analysis.json",result)
    write(out/"predictions.json",prediction_rows);write(out/"fit_predictions.json",fit_rows)
    write(out/"fit_parameters.json",parameters);write(out/"cv_results.json",cv_results)
    write(out/"bootstrap_resamples.json",old_boot)
    write(out/"provenance.json",dict(source_sha256=expected_source,helper_sha256=HELPER_SHA,
        analysis_contract_sha256=CONTRACT_SHA,input_sha256=hashes,protocol_sha256=digest(out/"protocol.json"),
        output_sha256={n:digest(out/n) for n in ("contract_copy.json","analysis.json","predictions.json",
            "fit_predictions.json","fit_parameters.json","cv_results.json","bootstrap_resamples.json")}))
    print(json.dumps(dict(status=result["status"],output=str(out),seconds=result["seconds_cpu_analysis"],
                          original_U_scores_exact=True,new_E_scorers=20,reused_U_scorers=20)),flush=True)

def self_test():
    c=validate_contract(CONTRACT)
    rng=np.random.default_rng(615)
    patients={r:[dict(p,images=p["images"]["E"]) for p in c["patient_rows"] if p["role"]==r] for r in ("fit","selection")}
    fit=dict(patients=patients["fit"],x=rng.normal(size=(80,2,27)),
             y=np.asarray([int(p["group"]=="A") for p in patients["fit"]]))
    selection=dict(patients=patients["selection"],x=rng.normal(size=(40,2,27)),
             y=np.asarray([int(p["group"]=="A") for p in patients["selection"]]))
    for name in ("cdi_image_mean_tuned","patient_meanmax52_tuned"):
        spec=next(s for s in c["methods"] if s["id"]==name)
        pred,par,cv,scaler,lr=original.fit_family("model_1",fit,selection,spec,c)
        changed=dict(selection,x=selection["x"]*123+456,y=1-selection["y"])
        _,par2,cv2,_,_=original.fit_family("model_1",fit,changed,spec,c)
        require(par==par2 and cv==cv2,"E fitting/C/scaler independent of selection features and labels")
        restored_scaler,restored_lr=restore(par,c["logistic_regression"])
        for d in (fit,selection):
            require(np.array_equal(original.predict_patients(d["x"],spec,scaler,lr),
                                   original.predict_patients(d["x"],spec,restored_scaler,restored_lr)),"exact no-fit restoration")
    from sklearn.metrics import roc_auc_score
    idx=original.bootstrap_indices(patients["selection"],17,260914)
    y=selection["y"];s=(np.arange(40)%7).astype(float)
    require(np.allclose(original.bootstrap_auc(s,y,idx),[roc_auc_score(y[i],s[i]) for i in idx],atol=1e-15),"bootstrap ties")
    import builtins,symtable
    table=symtable.symtable(Path(__file__).read_text(encoding="utf-8"),__file__,"exec")
    missing=[]
    def walk(scope):
        if scope.get_type()=="function":
            for sym in scope.get_symbols():
                if sym.is_referenced() and sym.is_global() and sym.get_name() not in globals() and not hasattr(builtins,sym.get_name()):
                    missing.append((scope.get_name(),sym.get_name()))
        for child in scope.get_children():walk(child)
    walk(table);require(not missing,"undefined globals "+repr(missing))
    print("PASS_SYNTHETIC_FIT_ONLY_C_SELECTION_EXACT_RESTORATION_BOOTSTRAP_GLOBALS; no E/U performance run")

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test",action="store_true");mode.add_argument("--run",action="store_true")
    p.add_argument("--contract",type=Path,default=CONTRACT);p.add_argument("--expected-code-sha256")
    a=p.parse_args()
    if a.self_test:self_test();return
    require(a.expected_code_sha256,"externally frozen analyzer hash required")
    run(a.contract.resolve(),a.expected_code_sha256.lower())

if __name__=="__main__":main()

