"""Check materialized checkpoints, optimizer steps and exposure coverage on CPU."""
import argparse, json
from collections import Counter
import torch
from .common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["model_1","model_2"],required=True)
    parser.add_argument("--step",type=int,choices=[250,1000],default=1000)
    args=parser.parse_args();verify_inputs()
    directory=RUN/"training_coverage_v2"/args.model
    state=torch.load(directory/f"step_{args.step:04d}.pt",map_location="cpu",weights_only=True)
    contract=json.loads((directory.parent/"protocol.json").read_text())
    assert state["contract"]==contract
    assert state["step"]==args.step and len(state["losses"])==args.step
    rows=read_csv(RUN/"cohort"/(args.model+"_train.csv"))
    rows.sort(key=lambda r:(r["eval_role"]!="background",stable("training-order",r["patient_id"]),r["image_id"]))
    expected=Counter()
    # Independent full-epoch reconstruction, not a call to the training schedule helper.
    remaining=args.step*4
    for epoch in range((remaining+len(rows)-1)//len(rows)):
        gen=torch.Generator().manual_seed(seed("coverage-epoch",epoch))
        permutation=torch.randperm(len(rows),generator=gen).tolist()
        take=min(remaining,len(rows));remaining-=take
        expected.update(rows[i]["image_id"] for i in permutation[:take])
    assert dict(expected)==state["exposures"]
    assert len(expected)==len(rows) and min(expected.values())>=1
    assert sum(expected.values())==args.step*4
    assert len(state["adapter"])>0
    assert sum(t.numel() for t in state["adapter"].values())==1659904
    assert all(torch.isfinite(t).all() for t in state["adapter"].values())
    assert any(float(t.abs().max())>0 for n,t in state["adapter"].items() if "lora_B" in n)
    for optstate in state["optimizer"]["state"].values():
        assert int(optstate["step"])==args.step
        assert all(torch.isfinite(t).all() for t in optstate.values() if torch.is_tensor(t))
    entries=[json.loads(line) for line in (directory/"training_trace.jsonl").read_text().splitlines()]
    entries=[e for e in entries if e["step"]<=args.step]
    assert [e["step"] for e in entries]==list(range(1,args.step+1))
    assert [e["loss"] for e in entries]==state["losses"]
    evaluation=read_csv(RUN/"cohort/evaluation_images.csv")
    by_patient={}
    for row in evaluation:
        by_patient.setdefault(row["patient_id"],[]).append(row["image_id"])
    group="A" if args.model=="model_1" else "B"
    for row in evaluation:
        participating=any(expected.get(i,0)>0 for i in by_patient[row["patient_id"]])
        assert participating==(row["assignment_group"]==group)
        if row["record_role"]=="U_observed":assert expected.get(row["image_id"],0)==0
    report={"status":"PASS","model":args.model,"steps":args.step,
        "optimizer_states_checked":len(state["optimizer"]["state"]),
        "trainable_parameters":1659904,"images":len(expected),
        "all_member_patients_used":True,"U_image_exposures":0,
        "exposure_min":min(expected.values()),"exposure_max":max(expected.values()),
        "scope":"checkpoint/exposure/optimizer/trace integrity; no utility or privacy certification"}
    write_json(directory/f"verification_{args.step:04d}.json",report)
    print(json.dumps(report))
if __name__=="__main__":main()
