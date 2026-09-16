from pathlib import Path
import concurrent.futures as cf
import json, requests
from acquire import fetch_pdf, save_json

r=Path(__file__).resolve().parent
jobs=[('A02','https://arxiv.org/pdf/2402.01054'),('B17','https://ojs.aaai.org/index.php/AAAI/article/download/38016/41978')]
records=[]
with cf.ThreadPoolExecutor(max_workers=3) as pool:
    for x in pool.map(lambda pair: fetch_pdf(dict(id=pair[0],pdf_url=pair[1])),jobs):
        records.append(x)
        print(json.dumps(x,ensure_ascii=False),flush=True)
save_json(r/'fallback_acquisition.json',records)
for ident,url in [('A20','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13034593/fullTextXML'),('A21','https://api2.openreview.net/notes?id=x83b7ybcB1'),('C02','https://api2.openreview.net/notes?id=SBHfiFsWRV')]:
    try:
        q=requests.get(url,timeout=40)
        (r/'texts'/f'{ident}_fallback.txt').write_text(q.text,encoding='utf-8')
        print(ident,q.status_code,len(q.content),q.text[:160],flush=True)
    except Exception as e: print(ident,str(e),flush=True)
