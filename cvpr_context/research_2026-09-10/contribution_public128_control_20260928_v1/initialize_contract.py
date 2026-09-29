"""Freeze the bounded Step3 contract; no feature/metric values are loaded."""
import argparse,ast,datetime,importlib.metadata,json,shutil
from pathlib import Path
from carrier_common import check,sha,dump,now

def run(tr,rr,out):
    check(not (out/"contract.json").exists(),"Contract already frozen")
    cw=tr/"code_working";reports=cw/"_reports";s1=rr/"contribution_kd_control_20260928_v1";hist=rr/"contribution_search_20260924"
    prior=json.loads((s1/"contract.json").read_text(encoding="utf-8"))
    receipts={}
    for f in ("construction_inputs.json","evaluation_inputs.json"):
        receipts.update(json.loads((s1/f).read_text(encoding="utf-8")))
    inputs={}
    def old(key,phases):
        value=dict(prior["inputs"][key]);p=Path(value["path"])
        check(p.is_file(),"Missing input: "+key)
        expected=value.get("sha256") or receipts[key]["sha256"]
        # Use completed evaluation receipts without loading V values during construction.
        if "evaluate" not in phases or len(phases)>1:check(sha(p)==expected,"Prior input changed: "+key)
        inputs[key]={**value,"phases":phases,"sha256":expected}
    for key,phases in {
        "P_DINO":["select","construct"],"P_RN":["select","construct","evaluate"],
        "synthetic_DINO":["construct"],"DP1_target":["construct"],"DP2_target":["construct"],
        "DP1_synthetic_RN":["construct","evaluate"],"DP2_synthetic_RN":["construct","evaluate"],
        "DP1_synthetic_DN":["evaluate"],"DP2_synthetic_DN":["evaluate"],
        "DP1_PNG_manifest":["construct"],"DP2_PNG_manifest":["construct"],
        "DP1_PNG_csv":["construct"],"DP2_PNG_csv":["construct"],
        "V_DenseNet":["evaluate"],"V_ResNet18":["evaluate"],"patient_bootstrap":["evaluate"],
        "historical_transport_bootstrap":["evaluate"]
    }.items():old(key,phases)
    def add(key,path,kind,phases):
        check(path.is_file(),"Missing input: "+key)
        inputs[key]={"path":str(path),"sha256":sha(path),"kind":kind,"phases":phases}
    add("P_roster",reports/"downstream_master_20260917_v1/public_real_images_private.csv","public_P_metadata",["select"])
    add("P_DN",reports/"prrd_pilot_abcd101_20260921_v1/public/recipient_P_dual_features.npz","public_P_evaluation_features",["evaluate"])
    add("prior_target_weights",s1/"target_weights.npz","public_and_existing_DP_postprocessing",["construct"])
    add("prior_label_seal",s1/"label_seal.json","prior_provenance",["construct"])
    paths=list(out.glob("*.py"))+[
        hist/"fixed_image_compilation.py",hist/"geometry_diagnostic.py",
        cw/"prrd_pilot_20260921/common.py",cw/"prrd_v3/contracts.py",
        cw/"prrd_v3/encoders.py",cw/"prrd_v3/resampling.py",
        cw/"receiver_distillation/second_source.py",cw/"receiver_distillation/signals.py",
        cw/"receiver_distillation/prepare_second.py",cw/"receiver_resnet_eval_20260924/run.py",
        cw/"prrd_pilot_20260921/declare.py",cw/"receiver_distillation/evaluate_feature_followup.py"]
    for p in out.glob("*.py"):ast.parse(p.read_text(encoding="utf-8"),filename=p.name)
    # Bind historical operation definitions to the completed Step1 receipt.
    for p in (hist/"fixed_image_compilation.py",hist/"geometry_diagnostic.py",cw/"prrd_pilot_20260921/common.py"):
        check(sha(p)==prior["implementation"][str(p)],"Verified reused implementation changed")
    start=datetime.datetime.fromisoformat("2026-09-28T16:14:01.4916855+09:00")
    c={"schema":"receiver.public128-step3/v1","status":"AUTHORIZED_FROZEN_BEFORE_SELECTION_AND_NEW_SCORES",
       "authorization":"User said proceed after Step3 scope and roughly one-hour estimate",
       "started_local":start.isoformat(),"frozen_utc":now(),
       "deadline_utc":(start+datetime.timedelta(minutes=90)).astimezone(datetime.timezone.utc).isoformat(),
       "whole_project_stage":2,"estimated_minutes":[45,75],"numerical_phase_cap_seconds":600,
       "question":"Synthetic128 versus public128 under the same transported RN-only and joint label objectives",
       "releases":["DP1","DP2"],"carriers":["SYN","PUBLIC"],"recipes":["RN_ONLY","JOINT"],
       "label_packets":8,"label_size":128,"label_bounds":[-1,1],"ridge":.1,"label_anchor":"zero",
       "regularizer":"eta=1e-3*||A||_F^2/128; same rule, not same numeric eta across operators",
       "target_scaling":"public patient/class weighted RMS; equal model weighting",
       "solver":{"library":"scipy.optimize.lsq_linear","tol":1e-12,"max_iter":500},
       "criteria":{"both_AUROC_positive":True,"mean_AUROC_min":.01,"mean_AP_min":0,"mean_AUROC_CI_lower_positive":True},
       "primary_receiver":"DenseNet","construction_reference":"ResNet18","primary_metric":"AUROC","secondary_metric":"AP",
       "metric_unit":"image","CI_unit":"patient-cluster; fixed bank/release conditional; 2000 paired draws",
       "selection":{"salt":"public-carrier-20260928-v1","patient_image_hash":"SHA256(salt|patient_id|image_id)",
         "candidate":"one per P patient, positive image for any patient with positive visit",
         "initial":6,"total":128,"method":"greedy farthest-first squared Euclidean, SHA256 order tie breaking",
         "features":"DINO K4 flattened64 and RN128, each candidate block divided by sqrt(mean squared norm), equal weight",
         "uses_DP_targets":False,"uses_DenseNet":False,"uses_V":False},
       "public_image_root":str(cw/"_data/raw/nih_cxr14_pa_k10_plus_census_v1/images"),
       "cache_policy":"Reference original native P images and existing synthetic PNGs; same bound tensor paths and projections; no new re-encoding",
       "banks":prior["banks"],"code_root":str(cw),"historical_code_dir":str(hist),
       "inputs":inputs,"implementation":{str(p):sha(p) for p in paths},
       "human_contract_sha256":sha(out/"EXECUTION_CONTRACT.md"),
       "prior_plan_sha256":sha(rr/"PUBLIC_TRANSFER_LABEL_EXPERIMENT_PLAN_20260928.md"),
       "historical_result_hashes":{str(rr/n):sha(rr/n) for n in [
           "CONTRIBUTION_SEARCH_RESULTS_20260924.md","PUBLIC_TRANSFER_KD_CONTROL_RESULTS_20260928.md",
           "PUBLIC_TRANSFER_STEP2_EXISTING_CONTROL_RESULTS_20260928.md"]},
       "new_Q_access":0,"new_DP_releases":0,"new_image_synthesis":0,"new_model_forwards":0,"new_HPO":False,
       "Expert_Reserved":False,"automatic_followup":False,"storage_target_MiB":250,"storage_cap_MiB":500,
       "versions":{x:importlib.metadata.version(x) for x in ("numpy","scipy","scikit-learn","Pillow")},
       "free_disk_GiB":shutil.disk_usage(out).free/1024**3}
    check(c["free_disk_GiB"]>1,"Insufficient disk")
    dump(out/"contract.json",c)
    print(json.dumps({"status":c["status"],"inputs":len(inputs),"label_packets":8,"new_forwards":0}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tr",type=Path,required=True);p.add_argument("--rr",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();run(a.tr,a.rr,a.out)

