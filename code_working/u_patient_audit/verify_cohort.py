"""Independent role and identity checks on the materialized U manifests."""
from collections import defaultdict, Counter
from scipy.stats import beta
from .common import *

def main():
    lock=verify_inputs()
    folder=RUN/"cohort"
    evaluation=read_csv(folder/"evaluation_images.csv")
    auxiliary=read_csv(folder/"auxiliary_images.csv")
    source={r["image_id"]:r for r in read_csv(SOURCE)}
    patients=defaultdict(list)
    for r in evaluation+auxiliary:
        assert source[r["image_id"]]["partition"]=="public_development"
        assert source[r["image_id"]]["patient_id"]==r["patient_id"]
        assert digest(IMAGES/r["image_filename"])==r["sha256"]
        patients[r["patient_id"]].append(r)
    all_ids=[r["image_id"] for r in evaluation+auxiliary]
    assert len(all_ids)==len(set(all_ids))
    for p,rows in patients.items():
        assert len({r["eval_role"] for r in rows})==1
        assert len({r["followup_no"] for r in rows})==len(rows)
        assert len({r["pixel_sha256"] for r in rows})==len(rows)
        if rows[0]["eval_role"] in {"fit","selection","calibration","test"}:
            assert len(rows)==4
            assert Counter(r["record_role"] for r in rows)=={"train_candidate":2,"U_observed":2}
            assert len({r["assignment_group"] for r in rows})==1
        else: assert len(rows)==2
    model_ids={}
    for model,group in [("model_1","A"),("model_2","B")]:
        train=read_csv(folder/(model+"_train.csv"))
        ids={r["image_id"] for r in train};model_ids[model]=ids
        train_patients={r["patient_id"] for r in train}
        assert len(ids)==len(train)
        for p,rows in patients.items():
            role=rows[0]["eval_role"]
            if role in {"reference","quality"}:assert p not in train_patients
            if role in {"fit","selection","calibration","test"}:
                member=rows[0]["assignment_group"]==group
                assert (p in train_patients)==member
                for row in rows:
                    assert (row["image_id"] in ids)==(member and row["record_role"]=="train_candidate")
        assert all(r["eval_role"]=="background" or r["assignment_group"]==group for r in train)
    assert model_ids["model_1"]&model_ids["model_2"]=={
        r["image_id"] for r in auxiliary if r["eval_role"]=="background"}
    role_counts=Counter(rows[0]["eval_role"] for rows in patients.values())
    n_cal=role_counts["calibration"]//2;n_test=role_counts["test"]//2
    report={"status":"PASS","patients_total":len(patients),"images_total":len(all_ids),
            "role_counts":dict(role_counts),"U_training_overlap":0,
            "nonmember_calibration_per_model":n_cal,"nonmember_test_per_model":n_test,
            "rank_pvalue_resolution":1/(n_cal+1),
            "zero_false_positives_one_sided_95_upper":float(beta.ppf(.95,1,n_test)),
            "independent_patient_repeats":False,
            "scope":"cohort/identity/hash/isolation verification; no medical performance claim"}
    write_json(folder/"verification.json",report)
    print(json.dumps(report))

if __name__=="__main__":main()

