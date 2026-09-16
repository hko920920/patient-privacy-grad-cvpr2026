"""One fixed development comparison; no admission gate or automated expansion."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import digest, write_json

MODELS = ("model_1", "model_2")
SCENARIOS = ("E", "U")
SCORES = ("relative", "difference", "negative_loss", "base_relative")

def image_features(row):
    cells = row["cells"]
    assert [c["timestep"] for c in cells] == list(range(0, 500, 50))
    originals = [float(c["original_loss"]) for c in cells]
    crops = [float(c["crop_loss"]) for c in cells]
    assert all(math.isfinite(x) and x > 0 for x in originals)
    assert all(math.isfinite(x) and x >= 0 for x in crops)
    return {
        "relative": math.fsum((c-o)/o for o,c in zip(originals,crops))/10,
        "difference": math.fsum(c-o for o,c in zip(originals,crops))/10,
        "negative_loss": -math.fsum(originals)/10,
    }

def auc(values, labels):
    s = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    assert s.shape == y.shape == (40,) and np.isfinite(s).all()
    assert int(y.sum()) == 20
    delta = s[y == 1, None]-s[None, y == 0]
    return float(np.mean((delta > 0)+0.5*(delta == 0)))

def bootstrap_draws():
    rng = np.random.default_rng(260914)
    return np.concatenate([
        rng.integers(0, 20, size=(2000,20), dtype=np.int64),
        rng.integers(20, 40, size=(2000,20), dtype=np.int64),
    ],axis=1)

def bootstrap_auc(values, labels, draws):
    s = np.asarray(values, dtype=np.float64)[draws]
    assert len(set(labels[:20])) == len(set(labels[20:])) == 1
    a,b = (s[:,:20],s[:,20:]) if labels[0] else (s[:,20:],s[:,:20])
    delta = a[:,:,None]-b[:,None,:]
    return np.mean((delta > 0)+0.5*(delta == 0),axis=(1,2))

def analyse(records):
    assert len(records) == 480
    indexed = {}
    assignments = {}
    patient_images = defaultdict(set)
    for row in records:
        assert row["eval_role"] == "selection"
        assert row["scenario"] in SCENARIOS
        assert row["model"] in MODELS+("base",)
        key = row["model"],row["image_id"]
        assert key not in indexed
        indexed[key] = row
        p = str(row["patient_id"])
        previous = assignments.setdefault(p,row["assignment_group"])
        assert previous == row["assignment_group"]
        assert previous in ("A","B")
        patient_images[p,row["scenario"]].add(row["image_id"])
        assert row["forward"] == 20 and row["backward"] == 0
        if row["model"] == "base":
            assert row["member"] is None and row["image_member"] is None
        else:
            expected = int(row["assignment_group"] == ("A" if row["model"] == "model_1" else "B"))
            assert row["member"] == expected
            assert row["image_member"] == int(row["scenario"] == "E" and expected)
    assert len(assignments) == 40
    assert len(patient_images) == 80 and all(len(v) == 2 for v in patient_images.values())
    groups = {g:sorted((p for p,a in assignments.items() if a == g),key=int) for g in ("A","B")}
    assert len(groups["A"]) == len(groups["B"]) == 20
    order = groups["A"]+groups["B"]
    patients = []
    for model in MODELS:
        for scenario in SCENARIOS:
            for patient in order:
                ids = sorted(patient_images[patient,scenario])
                features = []
                for iid in ids:
                    row = indexed[model,iid]
                    base = indexed["base",iid]
                    assert row["patient_id"] == base["patient_id"] and row["scenario"] == base["scenario"]
                    f = image_features(row)
                    f["base_relative"] = f["relative"]-image_features(base)["relative"]
                    features.append(f)
                patients.append(dict(model=model,scenario=scenario,eval_role="selection",patient_id=patient,
                    assignment_group=assignments[patient],member=int(assignments[patient] == ("A" if model == "model_1" else "B")),
                    image_ids=ids,scores={s:math.fsum(f[s] for f in features)/2 for s in SCORES}))
    draws = bootstrap_draws()
    metrics,contrasts = [],[]
    for model in MODELS:
        for scenario in SCENARIOS:
            selected = [p for p in patients if p["model"] == model and p["scenario"] == scenario]
            assert [p["patient_id"] for p in selected] == order
            labels = [p["member"] for p in selected]
            boots,aucs = {},{}
            for score in SCORES:
                values = [p["scores"][score] for p in selected]
                boots[score] = bootstrap_auc(values,labels,draws)
                aucs[score] = auc(values,labels)
                metrics.append(dict(model=model,scenario=scenario,score=score,auc=aucs[score],
                    patient_bootstrap95=np.quantile(boots[score],[.025,.975],method="linear").tolist(),
                    member_mean=math.fsum(v for v,y in zip(values,labels) if y)/20,
                    nonmember_mean=math.fsum(v for v,y in zip(values,labels) if not y)/20,
                    patients=40,members=20,nonmembers=20))
            for comparator in ("difference","negative_loss"):
                contrast = boots["relative"]-boots[comparator]
                contrasts.append(dict(model=model,scenario=scenario,contrast="relative_minus_"+comparator,
                    delta_auc=aucs["relative"]-aucs[comparator],
                    patient_bootstrap95=np.quantile(contrast,[.025,.975],method="linear").tolist(),
                    primary=scenario == "U" and comparator == "difference",
                    raw_forward_budget_matched=comparator == "difference"))
    primary = [c for c in contrasts if c["primary"]]
    assert len(primary) == 2
    if all(c["patient_bootstrap95"][0] > 0 for c in primary):
        interpretation = "RELATIVE_GAIN_OBSERVED_IN_THIS_DEVELOPMENT_DATA"
    elif all(c["delta_auc"] <= 0 for c in primary):
        interpretation = "NO_POSITIVE_POINT_ESTIMATE_GAIN_FROM_NORMALIZATION_IN_THIS_SETTING"
    else:
        interpretation = "ADDITIONAL_RELATIVE_GAIN_NOT_ESTABLISHED"
    return dict(schema="pfami-style-development-analysis/v1",metrics=metrics,contrasts=contrasts,
        patient_scores=patients,
        bootstrap=dict(seed=260914,replicates=2000,unit="patient",strata={"A":20,"B":20},patient_order=order,
            indices_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),dtype="int64",
            shared_across_models_scenarios_scores=True,interval="exploratory conditional percentile 95, not confirmatory"),
        interpretation=interpretation,primary_scenario="U",primary_score="relative",primary_contrast="relative_minus_difference",
        admission_gate_created=False,followup_authorized_by_analysis=False,automatic_followup=False,
        useful_identification_must_consider_absolute_auc_levels=True,
        low_fpr_utility_claimed=False,novel_method_success_claimed=False,
        limitations=[
            "Selection40 has been observed in earlier method development; this is not an untouched test.",
            "This fixed comparison does not correct confidence intervals for prior adaptive method selection.",
            "Targets share initialization and swap many patients jointly; they are not independent training seeds.",
            "E and secondary base-relative scores do not rescue the primary U comparison.",
            "The public base is not the paper's separately trained disjoint medical reference model.",
            "Single noise realization per view and timestep is a fixed-budget diagnostic, not noise-robust confirmation.",
            "Difference and relative use the same 20F; standalone negative loss uses 10F, so no equal-cost superiority claim for that contrast.",
            "No outcome automatically starts tuning, more patients, new target training or calibration/test evaluation.",
        ])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True)
    args=p.parse_args()
    run=Path(args.run_dir).resolve()
    report=json.loads((run/"report.json").read_text(encoding="utf-8"))
    assert report.get("mode","full") not in ("benchmark","dry-run")
    images=run/"image_scores.json"
    output=run/"analysis.json"
    if output.exists(): raise FileExistsError(output)
    result=analyse(json.loads(images.read_text(encoding="utf-8")))
    result["input_image_scores_sha256"]=digest(images)
    result["protocol_sha256"]=digest(run/"protocol.json")
    result["analysis_code_sha256"]=digest(Path(__file__))
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("patient_scores","limitations")},ensure_ascii=False))

if __name__ == "__main__": main()

