"""File integrity and independent repeat comparison for the fixed generation grid."""
from PIL import Image
from .common import *
def main():
    quality=RUN/"quality/training_coverage_v2"
    first=json.loads((quality/"step_1000_first/report.json").read_text())
    both=json.loads((quality/"step_1000_both/report.json").read_text())
    for model,expected in both["checkpoint_hashes"].items():
        assert digest(RUN/"training_coverage_v2"/model/"step_1000.pt")==expected
    first_map={(r["model"],r["prompt"],r["seed"]):r for r in first["images"]}
    repeated=0
    for row in both["images"]:
        path=quality/"step_1000_both"/row["path"]
        assert digest(path)==row["sha256"]
        with Image.open(path) as im:
            im.load();assert im.size==(256,256) and im.mode=="RGB"
        key=(row["model"],row["prompt"],row["seed"])
        if key in first_map:
            assert row["sha256"]==first_map[key]["sha256"]
            repeated+=1
    assert repeated==8 and len(both["images"])==12
    report={"status":"PASS","unique_model_prompt_seed_outputs":12,"repeated_identical_pngs":8,
        "source_reports_sha256":{"first":digest(quality/"step_1000_first/report.json"),
                                  "both":digest(quality/"step_1000_both/report.json")},
        "scope":"image/checkpoint binding and repeat integrity, not clinical validity"}
    write_json(RUN/"quality/generation_verification.json",report);print(json.dumps(report))
if __name__=="__main__":main()

