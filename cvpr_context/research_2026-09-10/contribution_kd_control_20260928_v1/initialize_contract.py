"""Freeze the authorized step-1 scope and document its start. No V arrays loaded."""
from pathlib import Path
import argparse,datetime,json,shutil,hashlib,ast
from common import dump,sha,check,now

def run(tr,rr,out):
    check(not (out/"contract.json").exists(),"Contract already exists")
    hist=rr/"contribution_search_20260924"
    reports=tr/"code_working/_reports"
    public={
        "P_DINO":"receiver_a1a2_s200_20260922_v1/second_targets/private/P_condition_features.npz",
        "P_RN":"receiver_resnet18_reuse_20260924_v1/P_projected_features.npz",
        "public_clipping":"receiver_feature_dp8_20260924_v1/public_calibration.json"}
    inputs={}
    def add(key,path,kind,phases):
        check(path.is_file(),"Missing input: "+str(path))
        inputs[key]={"path":str(path.resolve()),"kind":kind,"phases":phases}
        if "construct" in phases:
            inputs[key]["sha256"]=sha(path)
    for key,rel in public.items():
        add(key,reports/rel,"public_P",["construct"])
    add("synthetic_DINO",hist/"existing_png_dino_K4_features.npz","existing_DP_PNG_features",["construct"])
    add("public_maps_and_DP_weights",hist/"transport_public_maps_and_dp_weights.npz","public_maps_and_existing_DP_postprocessing",["construct"])
    add("historical_MT_labels",hist/"joint_compiler_soft_labels.npz","existing_DP_label_postprocessing",["construct","evaluate"])
    add("historical_artifact_manifest",hist/"research_artifact_manifest.json","historical_provenance",["construct"])
    prefixes={"DP1":"receiver_feature_dp8_20260924_v1","DP2":"receiver_feature_followup_20260924_v1"}
    banks={}
    for draw,prefix in prefixes.items():
        base=reports/prefix
        add(draw+"_target",base/"one_release/protected_target/target.npz","existing_protected_summary",["construct"])
        add(draw+"_synthetic_RN",base/"evaluation"/("ResNet18_PNG_features.npz" if draw=="DP1" else "ResNet18_FEATURE_DP2_PNG_features.npz"),
            "existing_DP_PNG_features",["construct","evaluate"])
        add(draw+"_synthetic_DN",base/"evaluation"/("DenseNet_PNG_features.npz" if draw=="DP1" else "DenseNet_FEATURE_DP2_PNG_features.npz"),
            "existing_DP_PNG_features",["evaluate"])
        banks[draw]=str((base/"protected_png").resolve())
        add(draw+"_PNG_manifest",base/"protected_png/release_manifest.json","existing_PNG_manifest",["construct"])
        add(draw+"_PNG_csv",base/"protected_png/images.csv","existing_PNG_index",["construct"])
    for key,name in {
        "historical_MT_predictions":"joint_compiler_predictions_private.npz",
        "historical_MT_bootstrap":"joint_compiler_bootstrap.npz",
        "historical_MT_results":"joint_compiler_results.json",
        "historical_transport_predictions":"transport_predictions_private.npz",
        "historical_transport_bootstrap":"transport_paired_bootstrap.npz"}.items():
        add(key,hist/name,"historical_V_evaluation",["evaluate"])
    add("patient_bootstrap",reports/"downstream_development_20260917_v1/cluster_bootstrap.npz","V_patient_resampling",["evaluate"])
    add("V_DenseNet",reports/"prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz","V_cached_features",["evaluate"])
    add("V_ResNet18",reports/"receiver_resnet18_reuse_20260924_v1/V_features.npz","V_cached_features",["evaluate"])
    implementation={}
    for p in list(out.glob("*.py"))+[hist/"geometry_diagnostic.py",hist/"fixed_image_compilation.py",
                                    tr/"code_working/prrd_pilot_20260921/common.py"]:
        ast.parse(p.read_text(encoding="utf-8"))
        implementation[str(p.resolve())]=sha(p)
    started=datetime.datetime.now(datetime.timezone.utc)
    contract={
        "schema":"receiver.KD-step1-only/v1","status":"AUTHORIZED_FROZEN",
        "authorization":"User said proceed after step1-only 30-60 minute estimate.",
        "started_utc":started.isoformat(),"target_completion_utc":(started+datetime.timedelta(minutes=60)).isoformat(),
        "estimated_minutes":[30,60],"numerical_worker_seconds_cap":600,
        "question":"Current mean transport versus ordinary teacher-score KD with identical two-source label solve.",
        "whole_project_stage":2,"prior_full_plan":"PUBLIC_TRANSFER_LABEL_EXPERIMENT_PLAN_20260928.md",
        "scope_override":"Step1 only. Retain the historical hard-label anchor; no zero-centered regularizer or public128.",
        "new_label_packets":2,"historical_reproduction_solves":2,"max_new_pixel_banks":0,"max_new_private_releases":0,
        "construction_models":["DINOv2","ResNet18"],"evaluation_models":["DenseNet","ResNet18"],
        "primary_receiver":"DenseNet","primary_metric":"AUROC","secondary_metric":"AP",
        "ridge":.1,"public_class_weights":[.5,.5],"labels":128,
        "label_bounds":[-1,1],"label_anchor":"64 negatives, then 64 positives; unchanged",
        "solver":{"relative_regularizer":.001,"tol":1e-12,"max_iter":500},
        "target_normalization":"per-model public weighted RMS, then equal model weights",
        "class_clipping_bounds":[1.9061329951133719,1.5917066731966771],
        "KD_teacher_clipping":False,"new_HPO":False,"new_Q_access":0,"Expert_Reserved":False,
        "historical_code_dir":str(hist.resolve()),"code_root":str((tr/"code_working").resolve()),
        "banks":banks,"inputs":inputs,"implementation":implementation,
        "human_contract_sha256":sha(out/"EXECUTION_CONTRACT.md"),
        "statistical_scope":"Image metrics; fixed-bank, paired patient-cluster intervals; not independent confirmation."
    }
    dump(out/"contract.json",contract)
    roots=[tr/"AGENTS.md",tr/"CURRENT_STATUS.md",tr/"WORKLOG.md",rr/"RESEARCH_FRAMEWORK.md",rr/"research_state.json"]
    backup=out/"records_before"
    backup.mkdir()
    for p in roots:shutil.copy2(p,backup/p.name)
    s=json.loads((rr/"research_state.json").read_text(encoding="utf-8-sig"))
    s["active_execution"]["status"]="running_step1_KD_control"
    s["active_execution"]["current_scope"]="Only two ordinary KD label packets, no new images/private release; hard anchor retained."
    s["active_execution"]["current_output"]=str(out)
    s["active_execution"]["current_contract"]=str(out/"contract.json")
    s["active_execution"]["started_utc"]=contract["started_utc"]
    s["step2_active_task"]="Step1-only ordinary KD control authorized and being implemented/executed. Steps2/3 are not running."
    s["next_task"]="Finish sealed two-label KD comparison and report; no automatic follow-up."
    dump(rr/"research_state.json",s)
    note=("**STEP1 KD CONTROL STARTED - 2026-09-28**\n\n"
          "User authorized step1 only after the 30-60 minute estimate. Two existing DP summaries/PNG banks are reused; "
          "only the ResNet18 target changes to ordinary teacher-score KD. Original hard-label anchor remains. "
          "No public128/source-only expansion, no new Q access, DP release, pixel updates, or Expert/Reserved. "
          "Construction and two-label sealing precede V evaluation. Stage2 remains in progress.\n\n")
    for p in roots[:-1]:
        old=p.read_text(encoding="utf-8-sig")
        link=(out/"EXECUTION_CONTRACT.md").relative_to(p.parent).as_posix()
        block=note+"[Execution contract](<"+link+">)\n\n"
        if old.startswith("# "):
            first,_,rest=old.partition("\n")
            new=first+"\n\n"+block+rest.lstrip("\r\n")
        else:new=block+old
        p.write_text(new,encoding="utf-8")
    print(json.dumps({"status":"CONTRACT_FROZEN","construction_input_files":sum("construct" in v["phases"] for v in inputs.values()),
                      "V_arrays_read":0,"new_labels_planned":2,"created_utc":contract["started_utc"]}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tr",type=Path,required=True);p.add_argument("--rr",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();run(a.tr,a.rr,a.out)

