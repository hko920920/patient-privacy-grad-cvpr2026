"""Inventory citations in historical notes and check public source metadata.

Run from the thesis root. Does not change historical notes or download a PDF corpus.
"""
import concurrent.futures as cf
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = next(p for p in Path('.').glob('CVPR*') if p.is_dir())
OUT = ROOT / 'reading_list_2026-09-09'
OUT.mkdir(exist_ok=True)

def inventory():
    entries = {}
    for path in sorted(ROOT.glob('*.md')):
        match = re.match(r'^(\d+)_', path.name)
        if not match or not 1 <= int(match.group(1)) <= 18:
            continue
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        for number, line in enumerate(lines, 1):
            for url in re.findall(r'https?://[^\s<>\)\]`]+', line):
                url = url.rstrip('.,;')
                entries.setdefault(url, []).append({
                    'file': path.name, 'line': number,
                    'context': '\n'.join(lines[max(0, number-3):min(len(lines), number+1)])
                })
    rows = [{'source_id': i, 'url': u, 'occurrences': occurrences}
            for i, (u, occurrences) in enumerate(entries.items(), 1)]
    (OUT / 'source_inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows

def check(row):
    result = dict(row)
    result['checked_at_local'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        response = requests.get(row['url'], timeout=(12, 30), headers={'User-Agent': 'Mozilla/5.0 (research bibliography link check)'})
        result.update(http_status=response.status_code, resolved_url=response.url,
                      content_type=response.headers.get('Content-Type', ''), bytes_received=len(response.content))
        if response.status_code != 200:
            result['check_status'] = 'source_http_' + str(response.status_code)
            return result
        if response.content.startswith(b'%PDF'):
            reader = PdfReader(io.BytesIO(response.content))
            result.update(check_status='pdf_accessible', pdf_url=response.url,
                          title=str((reader.metadata or {}).get('/Title', '')).strip(),
                          first_page=reader.pages[0].extract_text()[:9000])
            return result
        soup = BeautifulSoup(response.content, 'html.parser')
        metadata = {}
        for tag in soup.find_all('meta'):
            key = tag.get('name') or tag.get('property')
            if key and tag.get('content'):
                metadata.setdefault(key.lower(), []).append(tag['content'])
        title = next((metadata[k][0] for k in ['citation_title','dc.title','og:title','twitter:title'] if metadata.get(k)), None)
        result['title'] = title or (soup.title.get_text(' ', strip=True) if soup.title else '')
        result['citation_metadata'] = {k:v for k,v in metadata.items() if k.startswith('citation_') or k in ('dc.title','dc.date','dc.creator','dc.identifier')}
        links = []
        for a in soup.find_all('a', href=True):
            href = urljoin(response.url, a['href'])
            label = a.get_text(' ', strip=True)
            if any(s in href.lower() for s in ('.pdf','arxiv.org/abs/','arxiv.org/pdf/','openreview.net/pdf?')) or label.lower() in ('pdf','download pdf','paper'):
                links.append({'label': label[:140], 'url': href})
        result['paper_links'] = links[:60]
        if metadata.get('citation_pdf_url'):
            result['pdf_url'] = urljoin(response.url, metadata['citation_pdf_url'][0])
        elif urlparse(response.url).hostname == 'arxiv.org':
            aid = re.search(r'/(?:abs|html|pdf)/(\d+\.\d+)(v\d+)?', response.url)
            if aid:
                result['pdf_url'] = 'https://arxiv.org/pdf/' + aid.group(1) + (aid.group(2) or '')
        else:
            pdfs = [x for x in links if '.pdf' in x['url'].lower() or 'openreview.net/pdf?' in x['url']]
            if pdfs:
                result['pdf_url'] = pdfs[0]['url']
        result['check_status'] = 'page_accessible'
        result['text_excerpt'] = soup.get_text(' ', strip=True)[:16000]
    except Exception as exc:
        result.update(check_status='error', error=str(exc)[:350])
    return result

def main():
    rows = inventory()
    done = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(check, row): row for row in rows}
        for future in cf.as_completed(futures):
            result = future.result()
            # Keep bibliographic evidence, not republished article excerpts.
            result.pop('first_page', None)
            result.pop('text_excerpt', None)
            result.get('citation_metadata', {}).pop('citation_abstract', None)
            done.append(result)
            (OUT / 'source_checks.json').write_text(json.dumps(sorted(done, key=lambda r:r['source_id']), ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({k:result.get(k) for k in ('source_id','check_status','title','pdf_url','error')}, ensure_ascii=False), flush=True)
    print('COMPLETED', len(done), flush=True)

if __name__ == '__main__':
    main()
