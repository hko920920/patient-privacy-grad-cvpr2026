"""Bounded I/O for Step3; no raw private-data loading."""
import csv, datetime, hashlib, json, time
from pathlib import Path
import numpy as np

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""): h.update(part)
    return h.hexdigest()
def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def check(value,message):
    if not value: raise RuntimeError(message)
class Inputs:
    def __init__(self,out,phase):
        self.out=Path(out); self.phase=phase; self.receipts={}
        self.contract=json.loads((self.out/"contract.json").read_text(encoding="utf-8"))
        for p,h in self.contract["implementation"].items():
            check(sha(p)==h,"Frozen implementation changed: "+p)
        self.tick=time.monotonic()
    def check_time(self):
        check(time.monotonic()-self.tick<600,"Numerical phase exceeded 600 seconds")
        check(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(self.contract["deadline_utc"]),"Package deadline")
    def path(self,key):
        self.check_time();item=self.contract["inputs"][key]
        check(self.phase in item["phases"],"Input forbidden in this phase: "+key)
        p=Path(item["path"]);check(sha(p)==item["sha256"],"Frozen input changed: "+key)
        self.receipts[key]=item
        with (self.out/"access_log.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"utc":now(),"phase":self.phase,"key":key,"kind":item["kind"]})+"\n")
        return p
    def nz(self,key):
        with np.load(self.path(key),allow_pickle=False) as f:return {k:f[k] for k in f.files}
    def js(self,key): return json.loads(self.path(key).read_text(encoding="utf-8"))
    def verify(self):
        self.check_time()
        for item in self.receipts.values():
            check(sha(item["path"])==item["sha256"],"Input changed during phase")
def packet(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def check_seal(out,name,status):
    s=json.loads((out/name).read_text(encoding="utf-8"))
    check(s["status"]==status,"Wrong phase seal")
    check(sha(out/"contract.json")==s["contract_sha256"],"Contract changed after phase seal")
    for p,h in s["files"].items():check(sha(out/p)==h,"Phase artifact changed: "+p)
    return s
def write_seal(out,name,status,files,**kw):
    dump(out/name,{"status":status,"utc":now(),"contract_sha256":sha(out/"contract.json"),
         "files":{f:sha(out/f) for f in files},**kw})
def same_order(a,b):
    for key in ("image_ids","patient_ids","labels"):
        check(np.array_equal(a[key].astype(str),b[key].astype(str)),"Row order mismatch: "+key)
def export_packet(rows,labels,path):
    fields=["path","image_id","original_hard_label","regression_target","soft_class1_target","image_sha256"]
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r,l in zip(rows,labels):
            w.writerow({"path":r["path"],"image_id":r["image_id"],"original_hard_label":r["label"],
                "regression_target":format(float(l),".17g"),"soft_class1_target":format(float((l+1)/2),".17g"),
                "image_sha256":r["sha256"]})
    with Path(path).open(encoding="utf-8",newline="") as f:
        back=np.array([float(r["regression_target"]) for r in csv.DictReader(f)])
    check(np.array_equal(labels,back),"Label CSV float roundtrip")

