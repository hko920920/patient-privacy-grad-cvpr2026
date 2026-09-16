"""Fixed eight-record, one-patient MoFit diagnostic; no learned/composite score.

All four scalar orientations, both images, mean/max and target differences are
reported. This is not a patient cohort estimate, AUC, or causal membership test.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

CODE=Path(__file__).resolve().parent.parent
RESEARCH=CODE.parent/"CVPR 주제 탐색/research_2026-09-10"
SCREEN=CODE/"_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915"
CONTRACT=RESEARCH/"spec_sources/mofit_patient14393_two_image_analysis_contract_20260915.json"
OUTPUT=SCREEN/"mofit_patient14393_two_image_analysis_v1"
OLD=(SCREEN/"mofit_medical_full_v1",SCREEN/"mofit_medical_target2_full_v1")
NEW=SCREEN/"mofit_medical_remaining_pair_v1"
NEW_CONTRACT=SCREEN/"mofit_medical_remaining_pair_contract_v1.json"
MODELS=("model_1","model_2")
IMAGES={"E":("00014393_002.png","00014393_006.png"),"U":("00014393_001.png","00014393_004.png")}
SCORES=("h","negative_u","negative_v","negative_u_plus_v")
STATES=("PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE",
        "PASS_TARGET2_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_NONMEMBERSHIP",
        "PASS_REMAINING_PAIR_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_MEMBERSHIP")


def need(ok,message):
    if not ok:raise AssertionError(message)


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write(p,value):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def write_csv(p,rows):
    with Path(p).open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def close(a,b,label):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12),label)


def prepare():
    need(not CONTRACT.exists(),"Immutable pre-additional-result analysis contract")
    files=[]
    for d in OLD:files.extend(d/n for n in ("results.json","protocol.json","execution.json","verification.json","raw.pt"))
    write(CONTRACT,dict(schema="mofit-patient14393-two-image-analysis-contract/v1",created_utc=datetime.now(timezone.utc).isoformat(),
        current_step=2,patient_id="14393",eval_role="fit",assignment_group="A",models=list(MODELS),images=IMAGES,
        original_run_directories=[str(p) for p in OLD],new_run_directory=str(NEW),new_extraction_contract=str(NEW_CONTRACT),
        required_verification_statuses=list(STATES),original_frozen_sha256={str(p):sha(p) for p in files},
        analysis_source_sha256=sha(__file__),score_order=list(SCORES),
        definitions={"h":"L_optimized-L_null (source member-high)","negative_u":"-L_null",
            "negative_v":"-(L_VLM-L_null)","negative_u_plus_v":"-(u+v)=-L_VLM"},
        pool_order=["mean","max"],pool_policy="Apply fixed member-high sign first, then pool the two independently optimized image scores",
        target_difference="model_1(participant)-model_2(nonparticipant), separately for every image and every E/U mean/max",
        expected_records=8,expected_unique_images=4,independent_patients=1,expected_image_score_rows=32,
        expected_aggregate_rows=32,expected_target_difference_rows=32,
        exposure_policy="model1 E image_member=1 and positive actual exposure; every U and all model2 images image_member=0",
        fixed_scope=["No final-score winner, gamma, robust scaling, auxiliary selection, AUC, CI, threshold, or fitting",
            "No individual-patient causal conclusion: entire A/B training cohorts differ",
            "One patient, two E and two U images, one frozen draw configuration; not noise-repetition confirmation",
            "Both-target differences are analyst-only diagnostics, not inputs available to a single-target deployed attack",
            "Additional images were selected by the existing cohort plan, not by their unseen new scores",
            "All eight records and all four scalar orientations are retained; prior four results stay unchanged",
            "Raw arithmetic/optimizer validation is delegated to the frozen input verifiers; this module verifies scalar aggregation only"],
        output_directory=str(OUTPUT),new_fitting=False,new_GPU_calls=0,stage2_completion_claim=False))
    print(json.dumps(dict(status="POLICY_FROZEN_BEFORE_ADDITIONAL_RESULTS",contract_sha256=sha(CONTRACT),source_sha256=sha(__file__))))


def scalar_scores(r):
    u,h,v=(float(r[k]) for k in ("u","h","v"))
    need(all(math.isfinite(x) for x in (u,h,v)),"Finite verified raw scores")
    primary=dict(h=h,negative_u=-u,negative_v=-v,negative_u_plus_v=-(u+v))
    independent=dict(zip(SCORES,(h,0.-u,0.-v,math.fsum((-u,-v)))))
    for k in SCORES:close(primary[k],independent[k],"Independent score sign/sum")
    return primary


def summarize(rows):
    ix={(r["model"],r["image_id"]):r for r in rows}
    expected={(m,i) for m in MODELS for images in IMAGES.values() for i in images}
    need(len(rows)==len(ix)==8 and set(ix)==expected,"Exact eight fixed identities")
    image_scores=[];aggregates=[];differences=[];values={};pooled={}
    for model in MODELS:
        member=int(model=="model_1")
        for scenario,images in IMAGES.items():
            for image in images:
                r=ix[model,image]
                need(r["patient_id"]=="14393" and r["eval_role"]=="fit" and r["assignment_group"]=="A" and r["scenario"]==scenario,"Patient/role/scenario identity")
                need(r["member"]==member,"Fixed actual patient participation direction")
                image_member=member if scenario=="E" else 0
                need(r["image_member"]==int(r["actual_training_exposures"]>0)==image_member,"Actual image exposure semantics")
                need(r["record_role"]==("train_candidate" if scenario=="E" else "U_observed"),"Image role")
                if "patient_training_exposures" in r:need(int(r["patient_training_exposures"]>0)==member,"Recorded patient exposure")
                need(r["stage1_steps"]==1000 and r["stage2_steps"]==300 and r["mode"]=="literal_cfg2"
                     and r["monitor"]=="literal_every_step","Same full numerical schedule")
                for score,value in scalar_scores(r).items():
                    values[model,image,score]=value
                    image_scores.append(dict(patient_id="14393",model=model,member=member,scenario=scenario,
                        image_id=image,image_member=image_member,score=score,value=value))
            for score in SCORES:
                a,b=(values[model,i,score] for i in images)
                for pool,value,alternate in (("mean",math.fsum((a,b))/2,(a+b)/2),("max",max(a,b),(a+b+abs(a-b))/2)):
                    close(value,alternate,"Independent two-image pooling identity")
                    pooled[model,scenario,pool,score]=value
                    aggregates.append(dict(patient_id="14393",model=model,member=member,scenario=scenario,
                        score=score,pool=pool,images=list(images),value=value))
    for scenario,images in IMAGES.items():
        for score in SCORES:
            for image in images:
                a,b=(values[m,image,score] for m in MODELS)
                differences.append(dict(patient_id="14393",scenario=scenario,score=score,unit="image",image_id=image,pool=None,
                    participant_score=a,nonparticipant_score=b,participant_minus_nonparticipant=a-b))
            for pool in ("mean","max"):
                a,b=(pooled[m,scenario,pool,score] for m in MODELS)
                delta=a-b
                if pool=="mean":
                    alternate=math.fsum(values["model_1",i,score]-values["model_2",i,score] for i in images)/2
                    close(delta,alternate,"Mean target difference equals mean image differences")
                differences.append(dict(patient_id="14393",scenario=scenario,score=score,unit="patient_two_image_pool",image_id=None,pool=pool,
                    participant_score=a,nonparticipant_score=b,participant_minus_nonparticipant=delta))
    need(len(image_scores)==len(aggregates)==len(differences)==32,"All predeclared summaries")
    for image in [i for images in IMAGES.values() for i in images]:
        a,b=(ix[m,image] for m in MODELS)
        for k in ("caption","pixels_sha256","initial_hidden_sha256","policy"):
            need(a[k]==b[k],"Matched frozen same-image inputs across targets: "+k)
    return image_scores,aggregates,differences


def run(expected_source,expected_policy,expected_new_contract):
    need(sha(__file__)==expected_source and sha(CONTRACT)==expected_policy,"Frozen analyzer and pre-result policy")
    c=read(CONTRACT);need(c["analysis_source_sha256"]==expected_source and not OUTPUT.exists(),"Policy binding/immutable output")
    for p,h in c["original_frozen_sha256"].items():need(sha(p)==h,"Original result unchanged: "+p)
    need(expected_new_contract and sha(NEW_CONTRACT)==expected_new_contract,"Externally bound additional GPU contract")
    inputs=dict(c["original_frozen_sha256"])
    for n in ("results.json","protocol.json","execution.json","verification.json","raw.pt"):
        inputs[str(NEW/n)]=sha(NEW/n)
    inputs[str(NEW_CONTRACT)]=sha(NEW_CONTRACT)
    inputs[str(CONTRACT)]=sha(CONTRACT)
    directories=(*OLD,NEW)
    for d,status in zip(directories,STATES):
        v=read(d/"verification.json");execution=read(d/"execution.json")
        need(v["status"]==status,"Independent saved-raw verification required: "+d.name)
        need(v["results_sha256"]==execution["results_sha256"]==sha(d/"results.json"),"Verified results binding")
        need(v["protocol_sha256"]==execution["protocol_sha256"]==sha(d/"protocol.json"),"Verified protocol binding")
        need(v["raw_sha256"]==execution["raw_sha256"]==sha(d/"raw.pt"),"Verified raw binding")
        need(execution["complete"] is True,"Completed raw extraction")
    new_protocol=read(NEW/"protocol.json")
    need(new_protocol["contract_sha256"]==expected_new_contract,"Actual extraction contract binding")
    OUTPUT.mkdir()
    write(OUTPUT/"protocol.json",dict(schema="mofit-one-patient-two-image-analysis-execution/v1",created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_source_sha256=expected_source,analysis_contract_sha256=expected_policy,
        additional_extraction_contract_sha256=expected_new_contract,input_sha256=inputs,policy=c))
    rows=[]
    for d,count in zip(directories,(2,2,4)):
        rr=read(d/"results.json");need(len(rr)==count,"Expected old2+old2+new4 records");rows.extend(rr)
    image_scores,aggregates,differences=summarize(rows)
    for p,h in inputs.items():need(sha(p)==h,"Input changed during reporting: "+p)
    need(sha(__file__)==expected_source,"Source changed")
    result=dict(schema="mofit-one-patient-two-image-analysis/v1",status="PASS_FIXED_SCALAR_AND_POOLING_ARITHMETIC",
        current_step=2,patient_id="14393",independent_patients=1,records=8,unique_images=4,
        image_scores=image_scores,aggregates=aggregates,target_differences=differences,
        original_huv=[dict(model=r["model"],image_id=r["image_id"],scenario=r["scenario"],h=r["h"],u=r["u"],v=r["v"]) for r in rows],
        scope=c["fixed_scope"],AUC_computed=False,CI_computed=False,gamma_or_final_score_selected=False,
        new_fitting=False,new_GPU_calls=0,individual_causal_claim=False,stage2_completion_claim=False)
    write(OUTPUT/"analysis.json",result)
    write_csv(OUTPUT/"image_scores.csv",image_scores)
    write_csv(OUTPUT/"aggregates.csv",[dict(r,images="|".join(r["images"])) for r in aggregates])
    write_csv(OUTPUT/"target_differences.csv",differences)
    write(OUTPUT/"provenance.json",dict(source_sha256=expected_source,contract_sha256=expected_policy,input_sha256=inputs,
        output_sha256={n:sha(OUTPUT/n) for n in ("protocol.json","analysis.json","image_scores.csv","aggregates.csv","target_differences.csv")}))
    print(json.dumps(dict(status=result["status"],image_scores=32,aggregates=32,target_differences=32,output=str(OUTPUT))))


def self_test():
    rows=[]
    for mi,model in enumerate(MODELS):
        for scenario,images in IMAGES.items():
            for i,image in enumerate(images):
                rows.append(dict(model=model,image_id=image,patient_id="14393",eval_role="fit",assignment_group="A",scenario=scenario,
                    member=1-mi,image_member=int(mi==0 and scenario=="E"),actual_training_exposures=4*int(mi==0 and scenario=="E"),
                    record_role="train_candidate" if scenario=="E" else "U_observed",stage1_steps=1000,stage2_steps=300,
                    mode="literal_cfg2",monitor="literal_every_step",u=2.+3*i+mi,h=.3+i-mi*.2,v=-.1*(i+1),
                    caption="fixed",pixels_sha256=image,initial_hidden_sha256=image,policy={"fixed":True}))
    image,aggregate,difference=summarize(rows)
    # max(-u) must select the smaller loss, never negate max(u).
    r=next(r for r in aggregate if r["model"]=="model_1" and r["scenario"]=="E" and r["score"]=="negative_u" and r["pool"]=="max")
    need(r["value"]==-2.,"Sign before max")
    need(len(image)==len(aggregate)==len(difference)==32,"Synthetic complete output inventory")
    print("PASS_SYNTHETIC_SIGNS_MEAN_MAX_TARGET_DIFFERENCES; no additional results read")


if __name__=="__main__":
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract",action="store_true");mode.add_argument("--self-test",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--expected-code-sha256");parser.add_argument("--expected-contract-sha256");parser.add_argument("--expected-extraction-contract-sha256")
    a=parser.parse_args()
    if a.prepare_contract:prepare()
    elif a.self_test:self_test()
    else:run(a.expected_code_sha256,a.expected_contract_sha256,a.expected_extraction_contract_sha256)
