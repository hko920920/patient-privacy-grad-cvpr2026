from pathlib import Path
import csv, hashlib, json

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports" / "cvpr_u_pilot_v1_001"
DATA = ROOT / "_data"
SOURCE = DATA / "derived/nih_cxr14_pa_target_enriched_v1/all_selected_pa_images_private.csv"
IMAGES = DATA / "raw/nih_cxr14_pa_k10_plus_census_v1/images"
INVENTORY = IMAGES.parent / "content_inventory_private.csv"
SALT = "cvpr-u-pilot-20260914-v1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
PROMPT = "a frontal chest radiograph"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def stable(*values):
    return hashlib.sha256("|".join(map(str,(SALT,)+values)).encode()).hexdigest()

def seed(*values):
    return int(stable(*values)[:15],16)

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        raise ValueError("empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)

def verify_inputs():
    lock=json.loads((RUN/"cohort"/"lock.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if digest(RUN/"cohort"/name)!=expected:
            raise RuntimeError("cohort file changed: "+name)
    for name,expected in lock["source_files"].items():
        if digest(ROOT/name)!=expected:
            raise RuntimeError("source contract changed: "+name)
    return lock

