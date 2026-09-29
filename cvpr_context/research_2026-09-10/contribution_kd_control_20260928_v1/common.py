"""Shared I/O for the explicitly bounded KD control."""
from pathlib import Path
import csv, hashlib, json, datetime
import numpy as np

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

class Inputs:
    def __init__(self, out, phase):
        self.out = Path(out)
        self.contract = json.loads((self.out / "contract.json").read_text(encoding="utf-8"))
        self.phase = phase
        self.receipts = {}
        for p, expected in self.contract["implementation"].items():
            check(sha(p) == expected, "Frozen implementation changed: " + p)

    def path(self, key):
        spec = self.contract["inputs"][key]
        check(self.phase in spec["phases"], "Input forbidden in phase: " + key)
        p = Path(spec["path"])
        h = sha(p)
        if spec.get("sha256"):
            check(h == spec["sha256"], "Frozen input changed: " + key)
        self.receipts[key] = {"path": str(p), "sha256": h, "kind": spec["kind"]}
        with (self.out / "access_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": now(), "phase": self.phase, "key": key,
                                "kind": spec["kind"], "sha256": h}) + "\n")
        return p

    def npz(self, key):
        with np.load(self.path(key), allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    def js(self, key):
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def verify_receipts(self):
        for v in self.receipts.values():
            check(sha(v["path"]) == v["sha256"], "Input mutated during execution")

def export_labels(bank, labels, destination):
    bank = Path(bank).resolve()
    manifest = json.loads((bank / "release_manifest.json").read_text(encoding="utf-8"))
    with (bank / "images.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 128, "Expected 128 existing PNGs")
    check([int(r["label"]) for r in rows] == [0]*64 + [1]*64, "Original PNG order")
    fields = ["file", "original_hard_label", "regression_target", "soft_class1_target", "png_sha256"]
    with Path(destination).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, value in zip(rows, labels):
            p = (bank / row["file"]).resolve()
            check(p.is_relative_to(bank), "PNG path outside original bank")
            h = sha(p)
            check(h == manifest["files_sha256"][row["file"]], "Original PNG hash mismatch")
            writer.writerow({"file": row["file"], "original_hard_label": row["label"],
                             "regression_target": format(float(value), ".17g"),
                             "soft_class1_target": format(float((value + 1)/2), ".17g"),
                             "png_sha256": h})
    with Path(destination).open(encoding="utf-8", newline="") as f:
        recovered = np.array([float(row["regression_target"]) for row in csv.DictReader(f)])
    check(np.array_equal(labels, recovered), "CSV float roundtrip")
    return {"images_directory": str(bank), "CSV": str(destination),
            "CSV_sha256": sha(destination), "verified_original_PNGs": 128}

