"""Independent arithmetic, membership and role checks for the actual probe execution."""
import argparse,json,math
from collections import Counter,defaultdict
from .common import *

def verify(path):
    report=json.loads((path/"report.json").read_text())
    results=json.loads((path/"results.json").read_text())
    eval_rows=read_csv(RUN/"cohort/evaluation_images.csv")
    mapping={r["image_id"]:r for r in eval_rows}
    train={m:{r["image_id"] for r in read_csv(RUN/"cohort"/(m+"_train.csv"))} for m in ["model_1","model_2"]}
    conditions=Counter();by_patient=defaultdict(list)
    def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(a,b)
    for r in results:
        assert r["eval_role"]=="fit" and r["weights_unchanged"]
        assert r["forward"]==6*3*2*2+2*2*10*3*2
        assert r["backward"]==6*3*2*2
        assert r["seconds"]>0
        model=r["model"];folds=r["folds"];p=r["patient_id"]
        assert len(folds)==2
        assert folds[0]["support"]==folds[1]["query"] and folds[0]["query"]==folds[1]["support"]
        known_member=any(x["patient_id"]==p and x["image_id"] in train[model] for x in eval_rows)
        assert r["declared_member"]==r["realized_member"]==int(known_member)
        for fold in folds:
            for key in ["support","query"]:
                item=mapping[fold[key]]
                assert item["patient_id"]==p and item["eval_role"]=="fit"
                if r["scenario"]=="U":assert item["image_id"] not in train[model]
                else:assert (item["image_id"] in train[model])==known_member
            assert 0<=fold["coefficient_norm"]<=.050001
            close(fold["own_gain"]-fold["reference_gain"],fold["response"])
            close(fold["base_loss"]-fold["target_loss"],fold["base_minus_target"])
        close(sum(f["response"] for f in folds)/2,r["response"])
        close(-sum(f["target_loss"] for f in folds)/2,r["loss_mean"])
        close(max(-f["target_loss"] for f in folds),r["loss_max"])
        assert digest(RUN/report["training_dir"]/model/f"step_{report['step']:04d}.pt")==r["checkpoint_sha256"]
        conditions[(model,r["scenario"])]+=1;by_patient[p].append(r["realized_member"])
    assert len(results)==report["model_patient_evaluations"]
    assert len(by_patient)==report["unique_fit_patients"]
    assert all(sorted(v)==[0,1] for v in by_patient.values())
    assert all(n==report["unique_fit_patients"] for n in conditions.values())
    assert report["zero_adapter_delta_and_gradient"]==0.
    assert not report["calibration_test_patients_queried"]
    checked={"status":"PASS","scenario":report["scenario"],"unique_fit_patients":len(by_patient),
        "evaluations":len(results),"opposite_membership_pair_verified":True,
        "counts_arithmetic_and_fold_roles_verified":True,"test_patients_used":False,
        "results_sha256":digest(path/"results.json"),"report_sha256":digest(path/"report.json"),
        "scope":"execution verification; insufficient data for attack superiority/low-FPR performance"}
    write_json(path/"verification.json",checked);return checked

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--scenario",choices=["U","E"],default="U")
    parser.add_argument("--count",type=int,default=8);args=parser.parse_args()
    path=RUN/"probe_smoke/training_coverage_v2"/f"{args.scenario}_step_1000_n{args.count}"
    print(json.dumps(verify(path)))
if __name__=="__main__":main()

