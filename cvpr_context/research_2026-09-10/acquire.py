"""Acquire public research sources; downloading never marks a paper as reviewed."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
import requests
import fitz

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PDFS = OUT / 'pdfs'
TEXT = OUT / 'texts'
for directory in (PDFS, TEXT, OUT / 'reviews', OUT / 'code_sources'):
    directory.mkdir(exist_ok=True)

EXTRA = [
 ('X01','CDI','CVPR 2025','CDI: Copyrighted Data Identification in Diffusion Models',
  'https://openaccess.thecvf.com/content/CVPR2025/html/Dubinski_CDI_Copyrighted_Data_Identification_in_Diffusion_Models_CVPR_2025_paper.html',
  'https://openaccess.thecvf.com/content/CVPR2025/papers/Dubinski_CDI_Copyrighted_Data_Identification_in_Diffusion_Models_CVPR_2025_paper.pdf'),
 ('X02','Tracing the Roots','NeurIPS 2025','Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution',
  'https://proceedings.neurips.cc/paper_files/paper/2025/hash/901b713c3d1ecfce0d4556752eed4e02-Abstract-Conference.html',
  'https://proceedings.neurips.cc/paper_files/paper/2025/file/901b713c3d1ecfce0d4556752eed4e02-Paper-Conference.pdf'),
 ('X03','GSA','PoPETs 2025','White-box Membership Inference Attacks against Diffusion Models',
  'https://www.petsymposium.org/popets/2025/popets-2025-0068.pdf',
  'https://www.petsymposium.org/popets/2025/popets-2025-0068.pdf'),
 ('X04','SD-MIA','CVPR 2026','Black-box Membership Inference Attacks on the Pre-training Data of Image-generation Models',
  'https://openaccess.thecvf.com/content/CVPR2026/html/Qi_Black-box_Membership_Inference_Attacks_on_the_Pre-training_Data_of_Image-generation_CVPR_2026_paper.html',
  'https://openaccess.thecvf.com/content/CVPR2026/papers/Qi_Black-box_Membership_Inference_Attacks_on_the_Pre-training_Data_of_Image-generation_CVPR_2026_paper.pdf'),
 ('X05','Black-box fine-tuned diffusion MIA','NDSS 2025','Black-box Membership Inference Attacks against Fine-tuned Diffusion Models',
  'https://www.ndss-symposium.org/wp-content/uploads/2025-324-paper.pdf',
  'https://www.ndss-symposium.org/wp-content/uploads/2025-324-paper.pdf'),
 ('X06','DPImageBench','CCS 2025','DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis',
  'https://arxiv.org/abs/2503.14681','https://arxiv.org/pdf/2503.14681'),
 ('X07','Realistic diffusion MIA','WACV 2024','Towards More Realistic Membership Inference Attacks on Large Diffusion Models',
  'https://openaccess.thecvf.com/content/WACV2024/html/Dubinski_Towards_More_Realistic_Membership_Inference_Attacks_on_Large_Diffusion_Models_WACV_2024_paper.html',
  'https://openaccess.thecvf.com/content/WACV2024/papers/Dubinski_Towards_More_Realistic_Membership_Inference_Attacks_on_Large_Diffusion_Models_WACV_2024_paper.pdf'),
]
CORE = ['X01','A09','A12','X02','B08','B13','X03','X04','X05','X06','X07']

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def fetch_pdf(row):
    pid=row['id']
    path=PDFS/(pid+'.pdf')
    record={'id':pid,'requested_url':row['pdf_url'],'checked_at':datetime.now(timezone.utc).isoformat()}
    try:
        if path.exists() and path.read_bytes()[:5]==b'%PDF-':
            content=path.read_bytes()
            record['download']='cached'
        else:
            response=requests.get(row['pdf_url'],timeout=(15,50))
            record.update(http_status=response.status_code,final_url=response.url)
            response.raise_for_status()
            content=response.content
            if not content.startswith(b'%PDF-'):
                raise ValueError('Response is not PDF')
            path.write_bytes(content)
            record['download']='ok'
        doc=fitz.open(stream=content,filetype='pdf')
        pages=[page.get_text(sort=True) for page in doc]
        save_json(TEXT/(pid+'.json'),pages)
        (TEXT/(pid+'.txt')).write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{text}' for i,text in enumerate(pages)),encoding='utf-8')
        record.update(pages=len(pages),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),characters=sum(map(len,pages)))
        links=[]
        for page in doc:
            links.extend(link['uri'] for link in page.get_links() if link.get('uri'))
        record['embedded_urls']=sorted(set(links))
        record['review_status']='downloaded_not_reviewed'
    except Exception as exc:
        record.update(download='failed',error=str(exc),review_status='source_unavailable')
    return record

def main():
    rows=json.loads((ROOT/'reading_list_2026-09-09'/'reading_list.json').read_text(encoding='utf-8-sig'))
    for pid,alias,venue,title,source,pdf in EXTRA:
        rows.append(dict(id=pid,alias=alias,venue=venue,title=title,source_url=source,pdf_url=pdf,latest18=False))
    for row in rows:
        row['scope']='historical_jailbreak_outside_current_topic' if row['id'].startswith('F') else 'current_or_related_cvpr_research'
    save_json(OUT/'registry.json',rows)
    requested=set(sys.argv[1:])
    selected=[r for r in rows if r['scope']=='current_or_related_cvpr_research' and (not requested or r['id'] in requested)]
    selected.sort(key=lambda r: (CORE.index(r['id']) if r['id'] in CORE else (20 if r.get('latest18') else 30),r['id']))
    old=json.loads((OUT/'acquisition.json').read_text(encoding='utf-8')) if (OUT/'acquisition.json').exists() else []
    results={r['id']:r for r in old}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch_pdf,row) for row in selected]
        for future in cf.as_completed(futures):
            record=future.result()
            results[record['id']]=record
            save_json(OUT/'acquisition.json',list(results.values()))
            print(json.dumps({k:record[k] for k in ['id','download','pages','error'] if k in record}),flush=True)
    print(json.dumps({'requested':len(selected),'downloaded':sum(r['download']!='failed' for r in results.values()),'failed':sum(r['download']=='failed' for r in results.values())}),flush=True)

if __name__=='__main__':
    main()
