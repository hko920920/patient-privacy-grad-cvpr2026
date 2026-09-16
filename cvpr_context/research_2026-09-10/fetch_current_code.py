"""Add current official source snapshots without installing or running them."""
import concurrent.futures as cf
import json, sys
from fetch_code import ROOT, fetch

if __name__ == '__main__':
    items = [('X08','SunnierLee/DP-FETA'), ('X09','2019ChenGong/Feta-Pro'),
             ('X11','wjfu99/MIA-Gen'), ('X12','JoonsungJeon/MoFit')]
    if len(sys.argv)>1:items=[x for x in items if x[0] in sys.argv[1:]]
    records = json.loads((ROOT/'code_acquisition.json').read_text(encoding='utf-8'))
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(fetch, items):
            records = [r for r in records if r['id'] != record['id']] + [record]
            (ROOT/'code_acquisition.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps({k:record[k] for k in ['id','status','commit','error'] if k in record}),flush=True)
