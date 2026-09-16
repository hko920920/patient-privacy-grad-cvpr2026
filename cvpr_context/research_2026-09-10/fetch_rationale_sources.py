"""Acquire two primary-source additions for the rationale audit, without executing source code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import requests
import fitz

ROOT = Path(__file__).resolve().parent / 'rationale_sources'
SOURCES = [
    ('R01', 'Similarity Distribution Based Membership Inference Attack on Person Re-identification',
     'https://iip.tongji.edu.cn/pdf/2023AAAI-GJY.pdf'),
    ('R02', 'Robustness and Privacy Interplay in Patient Membership Inference',
     'https://lup.lub.lu.se/search/files/227035246/RobustnessPrivacy_IJCNN25.pdf'),
]

def acquire(item):
    source_id, title, url = item
    pdf = ROOT / f'{source_id}.pdf'
    out = {'id': source_id, 'title': title, 'url': url, 'retrieved_for': '2026-09-10 rationale audit'}
    try:
        if not pdf.exists():
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Response is not PDF')
            pdf.write_bytes(response.content)
        doc = fitz.open(pdf)
        (ROOT / f'{source_id}.txt').write_text('\n\n'.join(
            f'\nPAGE {i + 1}\n' + page.get_text() for i, page in enumerate(doc)), encoding='utf-8')
        out.update(status='downloaded_text_extracted', pages=len(doc),
                   sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), bytes=pdf.stat().st_size)
    except Exception as exc:
        out.update(status='unavailable', error=str(exc))
    return out

if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        ledger = list(pool.map(acquire, SOURCES))
    (ROOT / 'acquisition.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(ledger, indent=2, ensure_ascii=False))
