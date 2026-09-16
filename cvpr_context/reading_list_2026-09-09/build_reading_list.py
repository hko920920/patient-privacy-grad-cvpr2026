"""Build a deduplicated reading list, preserving each historical citation.

Run from the thesis root. PDF checks read only the response prefix; this is not
full-text review, a complete-file integrity check, or a bulk PDF download.
"""
import concurrent.futures as cf
import csv
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

ROOT = next(p for p in Path('.').glob('CVPR*') if p.is_dir())
OUT = ROOT / 'reading_list_2026-09-09'
CURATION = json.loads((OUT / 'curation.json').read_text(encoding='utf-8'))
PROXIMITY = json.loads((OUT / 'cvpr_proximity.json').read_text(encoding='utf-8'))
SOURCES = {r['source_id']: r for r in json.loads((OUT / 'source_checks.json').read_text(encoding='utf-8'))}
REPORT = ROOT / '20_근접선행연구_전체목록과_PDF_검토기록_2026-09-09.md'

def make_rows():
    rows = []
    seen = []
    counts = Counter()
    for entry in CURATION['papers']:
        row = dict(entry)
        source_rows = [SOURCES[i] for i in row['sources']]
        main = source_rows[0]
        seen.extend(row['sources'])
        counts[row['group']] += 1
        row['id'] = row['group'] + str(counts[row['group']]).zfill(2)
        row['title'] = row.get('title') or main['title'].replace('{','').replace('}','')
        row['source_url'] = main['url']
        row['pdf_url'] = row.get('pdf_url') or main.get('pdf_url')
        if row['pdf_url'] and row['pdf_url'].startswith('http://openaccess.thecvf.com/'):
            row['pdf_url'] = row['pdf_url'].replace('http://','https://',1)
        row['occurrences'] = sorted([o for s in source_rows for o in s['occurrences']], key=lambda o:(o['file'],o['line']))
        row['latest18'] = any(o['file'].startswith('18_') for o in row['occurrences'])
        row['cvpr_tier'] = next((tier for tier, ids in PROXIMITY['tiers'].items() if row['id'] in ids), '')
        row['cvpr_relation'] = PROXIMITY['labels'].get(row['cvpr_tier'], '')
        row['group_label'] = CURATION['groups'][row['group']]
        assert row['title'] and 'Verifying your browser' not in row['title']
        assert row['pdf_url'], row
        rows.append(row)
    assert len(seen) == len(set(seen)), 'A source is assigned twice'
    assert set(seen) | set(CURATION['nonpaper_source_ids']) == set(SOURCES), 'Missing source'
    assert not set(seen) & set(CURATION['nonpaper_source_ids'])
    assert len(rows) == 71
    return rows

def check_pdf(row):
    result = {'id': row['id'], 'url': row['pdf_url'], 'checked_at_local': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        with requests.get(row['pdf_url'], stream=True, timeout=(10,20), headers={'User-Agent':'Mozilla/5.0 (research bibliography link check)'}) as response:
            prefix = next(response.iter_content(chunk_size=1024), b'')
            result.update(http_status=response.status_code, resolved_url=response.url,
                          content_type=response.headers.get('Content-Type',''),
                          pdf_magic=prefix.lstrip().startswith(b'%PDF'))
            result['status'] = 'PDF_HEADER_OK' if response.status_code == 200 and result['pdf_magic'] else 'BROWSER_CHECK_REQUIRED'
    except Exception as exc:
        result.update(status='BROWSER_CHECK_REQUIRED', error=str(exc)[:350])
    print(row['id'], result['status'], flush=True)
    return result

def md_link(label, path):
    return '[' + label + '](<' + path + '>)'

def refs(row, prefix=''):
    byfile = {}
    for occurrence in row['occurrences']:
        byfile.setdefault(occurrence['file'], []).append(occurrence['line'])
    return ' · '.join(md_link(name[:2] + ':' + ','.join(map(str,sorted(set(lines)))), prefix + name)
                      for name,lines in byfile.items())

def render(rows, checks):
    checks = {c['id']:c for c in checks}
    for row in rows:
        row['pdf_check'] = checks[row['id']]
        row['link_status'] = 'PDF 응답 확인' if checks[row['id']]['status'] == 'PDF_HEADER_OK' else '브라우저 확인 필요'
    stats = Counter(r['group'] for r in rows)
    ok = sum(r['pdf_check']['status'] == 'PDF_HEADER_OK' for r in rows)
    latest = sum(r['latest18'] for r in rows)
    first = sorted([r for r in rows if r.get('priority')], key=lambda r:r['priority'])
    (OUT / 'reading_list.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['id','priority','latest18','cvpr_relation','group_label','alias','title','venue','role','source_url','pdf_url','link_status','historical_locations','metadata_note']
    with (OUT / 'reading_list.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {f:row.get(f,'') for f in fields}
            record['historical_locations'] = '; '.join(o['file'] + ':' + str(o['line']) for o in row['occurrences'])
            writer.writerow(record)

    old3 = next(ROOT.glob('03_*.md')).name
    old4 = next(ROOT.glob('04_*.md')).name
    old16 = next(ROOT.glob('16_*.md')).name
    old18 = next(ROOT.glob('18_*.md')).name
    header = f'''# 근접 선행연구 전체 목록·PDF와 기존 검토 기록

- 정리·링크 확인일: 2026-09-09, Asia/Seoul.
- 사용자 범위 정정: 지금 읽을 대상은 **현재 CVPR의 환자 DP 의료영상 diffusion·환자별 노출 감사 후보**다. [해당 후보 35편만 PDF 열기](reading_list_2026-09-09/cvpr_patient_dp.html) · [35편 목록과 기존 판정](reading_list_2026-09-09/cvpr_patient_dp.md) · [35편 CSV](reading_list_2026-09-09/cvpr_patient_dp.csv). 아래 71편은 다른 방향까지 포함한 탐색 이력의 보존 목록이다.
- 근접성 표현 정정: **35편은 관련 문헌 수이며 모두 직접 경쟁 연구는 아니다.** [핵심 대조 6편과 나머지의 역할](reading_list_2026-09-09/cvpr_proximity_review.md)을 별도로 표시했다. 6은 현재 주장·방법에 따른 우선 대조 선별 수이며 동일 주제의 선행을 모두 확정한 수가 아니다.
- 본학회 검토 정정: 기존 6편은 기여 중복 대조용 선별이며 SOTA 비교군 전체가 아니다. CDI(CVPR 2025) 등 직접 대조 누락과 발표 형식은 {md_link('22번 정정', next(ROOT.glob('22_*.md')).name)}을 우선한다. 아래는 기존 인용의 역사 목록이다.
- 범위: 이 폴더의 **01~18번 기존 문서에서 URL로 식별되는 문헌 전체**. 91개 고유 URL에서 논문 외 자료 8개를 분리하고 같은 논문의 proceedings/arXiv/PDF 링크를 합쳤다.
- 결과: **근접·관련 연구 67편 + 17번 탈옥 트렌드 참고 4편 = 71편**. 그중 **{latest}편은 최신 18번 판정에 직접 인용**되어 있다. 모든 항목의 근접성이나 검토 깊이가 같다는 뜻은 아니다.
- 열기: [검색·필터 가능한 PDF 목록](reading_list_2026-09-09/index.html) · [Excel용 CSV](reading_list_2026-09-09/reading_list.csv) · [전체 출처·행 번호 JSON](reading_list_2026-09-09/reading_list.json).
- PDF 링크 {len(rows)}개 중 **{ok}개는 HTTP 200과 PDF 시작 바이트를 확인**했다. 나머지 {len(rows)-ok}개는 아래에서 브라우저 확인 필요로 표시한다. PDF 전체 파일 무결성 검사나 전문 정독을 뜻하지 않는다. PDF 원문 일괄 저장은 수행하지 않았으며 링크를 눌러 열고 저장할 수 있다.

## 1. 이전에 가능성이 있다고 판단한 기록이 있는가

있다. 다만 **선행을 검토한 뒤에도 남은 조건부 연구 후보**라는 기록이며, CVPR 경쟁력이 이미 검증됐다는 결론은 아니다.

| 기존 기록 | 위치 | 판단의 정확한 범위 |
|---|---|---|
| 2026-09-02 초기 재검증 | {md_link('03번 §§2~4, 행 31~85',old3)} | 기존 기법의 점유 범위를 표로 대조하고, 당시 검색에서 찾지 못한 교집합을 신규성 후보로 제시. 모든 선행의 부재를 증명한 것은 아니라고 명시. |
| 2026-09-02~03 claim matrix | {md_link('04번',old4)} | patient aggregation·DP diffusion·multiplicity·다중영상 등 이미 존재하는 구성요소를 구분. 이후 후보 반증 결과도 반영. |
| 2026-09-03 방향 축소 | {md_link('16번 §§2~4, 행 30~99',old16)} | Kaiser·의료 ReID·unlearning 등의 충돌을 확인하고 환자별 생성 노출 위험으로 질문을 좁힘. |
| 2026-09-08 최신 판정, 09-09 상태 정정 | {md_link('18번 §0 및 §6, 행 12~30·285~304',old18)} | G8/P8 비교와 tail plot만으로는 CVPR main 기여가 약함. 새 환자 집합 감사법과 강한 DP 기준선 비교를 입증해야 제출 후보가 된다는 조건부 판정. |

따라서 이전 설명을 “다 읽고 봤으니 지금 형태 그대로 CVPR 가능성이 확인됐다”로 받아들이면 과하다. 정확한 의미는 **기록된 선행을 대조한 뒤, 단순 비교안은 강등하고 새 감사방법을 검증할 여지는 남겨두었다**는 것이다. 이번 목록 정리는 기존 판정을 긍정으로 상향하지 않는다.

## 2. 무엇을 어디까지 검토했다고 볼 수 있는가

- **기록으로 확인:** 03·04·10·16·18번의 연구군별 claim matrix, 선행 충돌, 후보 기각 이유, baseline·대조실험 요구사항.
- **기록으로 확인되지 않음:** 71편 각각의 전문·부록 정독 완료 장부, 증명과 구현의 독립 검증, 모든 선행의 재현 실험 완료. 전문을 보지 않았다는 단정도 아니며, 전부 완료했다는 증거가 없다는 뜻이다.
- **이번 작업:** 기존 기록 읽기, URL 전수 추출·중복 통합, 공식 proceedings·저자 공개본의 제목/링크 확인, 일부 PDF 첫 페이지·원문 관련 부분 확인, PDF 접속 상태 검사. 71편 전부의 새 심층 독해나 신규성 최종 확정은 수행하지 않았다.
- **검토할 때 남길 항목:** privacy unit/adjacency, 위협모델과 접근권, 기록 수·상관 보정, 기존 공격의 집계법, split-bias 통제, low-FPR 성능, 임상 효용, public-data/compute/epsilon 조건, 우리 방법이 추가하는 정확한 한 요소.

## 3. 먼저 읽을 15편

전체 71편 중 현재 기여 판단에 먼저 필요한 순서다. 제목 전체와 과거 기록 위치는 아래 전체 목록에서 확인할 수 있다.

| 순서 | 문헌 | 먼저 확인할 점 | PDF |
|---|---|---|---|
'''
    lines = [header]
    for row in first:
        lines.append(f"| {row['priority']} | {row['alias']} ({row['venue']}) | {row['role']} | [PDF]({row['pdf_url']}) |\n")
    lines.append('\n## 4. 전체 목록\n\n표의 **18번** 표시는 최신 기여도 판정 문서에 직접 URL이 있는 문헌이다. 기록 열은 문서 번호와 원래 행 번호다. 학회 최종본과 arXiv 공개본의 제목·버전·연도는 다를 수 있다.\n')
    for group, label in CURATION['groups'].items():
        lines.append(f'\n### {group}. {label} — {stats[group]}편\n\n| ID | 논문·발표 기록 | 우리 연구와의 관계 | 다운로드·접속 확인 | 기존 기록 |\n|---|---|---|---|---|\n')
        for row in rows:
            if row['group'] != group:
                continue
            label18 = ' **18번**' if row['latest18'] else ''
            note = '<br>' + row['metadata_note'] if row.get('metadata_note') else ''
            source_links = f"[원문 페이지]({row['source_url']})"
            if row.get('extra_source_url'):
                source_links += f" · [대체 출처]({row['extra_source_url']})"
            lines.append(f"| {row['id']}{label18} | **{row['alias']}**<br>{row['title']}<br>{row['venue']}{note} | {row['role']} | [PDF]({row['pdf_url']}) · {source_links}<br>{row['link_status']} | {refs(row)} |\n")
    lines.append('''
## 5. 누락과 검토 과장을 막기 위한 경계

- 이 목록은 기존 01~18번의 **명시적 URL 인용 전수 목록**이다. 전 세계 근접 논문을 모두 찾았다는 목록이 아니다.
- 기존 문서에 이름 또는 연구군만 나온 ViewDiff·MultiDiff·CDI, 일반 user-level DP theory (NeurIPS 2021/2023)는 특정 원문까지 대조했다는 인용 증거가 부족하다. 특정 논문을 추정해 “과거 검토 완료” 항목으로 보태지 않았다. 후속 독해에서 서지를 확정할 대상이다.
- DINOv2·RAD-DINO·BioViL-T 같은 사용 encoder와 FID/KID/PRDC 등 평가 지표의 원 논문은 이 71편 인용 집합과 별도로 방법론 참고문헌을 정비해야 한다.
- PFDM과 PF-LDM은 서로 다른 연구다. DPDM과 DP-LDM도 서로 다르다. 같은 논문의 arXiv/proceedings 링크만 통합했다. PIA/PIAN은 한 논문의 방법·변형으로 한 편으로 센다.
- Charles fixed-compute 연구는 기존 기록의 제목과 연결된 arXiv 공개본 제목이 다르다. B04에 둘의 연결을 설명하는 저자 출처를 붙였다. 원문 버전을 맞추기 전 학회본과 공개본을 같은 버전으로 인용하지 않는다.
- OpenReview의 브라우저 확인 또는 HTTP 403은 논문 부재의 증거가 아니다. PDF가 열려도 그것은 정독 완료의 증거가 아니다.

## 6. 논문 개수에서 분리한 기존 자료 8개

''')
    for sid in CURATION['nonpaper_source_ids']:
        source = SOURCES[sid]
        lines.append(f"- [{source.get('title') or source['url']}]({source['url']}) — 기존 URL #{sid}.\n")
    lines.append('''
## 7. 재현 가능한 정리 자료

- [원래 URL 91개와 인용 문맥](reading_list_2026-09-09/source_inventory.json)
- [공식 페이지·PDF 메타데이터 접속 기록](reading_list_2026-09-09/source_checks.json)
- [같은 논문 통합·분류·읽기 우선순위](reading_list_2026-09-09/curation.json)
- [최종 PDF 접속 기록](reading_list_2026-09-09/pdf_checks.json)
- [목록 생성 스크립트](reading_list_2026-09-09/build_reading_list.py)

이번 산출물은 문헌과 검토 범위의 기록이다. M0/K5/K10 실험, 공격 실행, 동결 프로토콜, 학위논문 본문은 변경하지 않았다. 최신 과학적 판정은 계속 18번이고, 이 20번은 읽기 목록과 검토 증거의 경계를 제공한다.
''')
    REPORT.write_text(''.join(lines),encoding='utf-8')

    esc = html.escape
    cards = []
    for row in sorted(rows,key=lambda r:(r.get('priority',999),r['id'])):
        search = esc(' '.join([row['title'],row['alias'],row['role'],row['venue'],row['group_label']]).lower(),quote=True)
        linkrefs = []
        used = set()
        for occurrence in row['occurrences']:
            name = occurrence['file']
            if name in used:
                continue
            used.add(name)
            linkrefs.append(f'<a href="../{quote(name)}">{esc(name[:2])}번</a>')
        priority = '<span class="tag priority">우선 ' + str(row['priority']) + '</span>' if row.get('priority') else ''
        latesttag = '<span class="tag">최신 18번 인용</span>' if row['latest18'] else ''
        statusclass = 'ok' if row['pdf_check']['status'] == 'PDF_HEADER_OK' else 'pending'
        cards.append(f'''<article data-search="{search}" data-group="{row['group']}" data-latest="{int(row['latest18'])}" data-priority="{int(bool(row.get('priority')))}" data-tier="{row['cvpr_tier']}">
<div class="tags"><span>{row['id']}</span>{priority}{latesttag}{'<span class="tag">'+esc(row['cvpr_relation'])+'</span>' if row['cvpr_relation'] else ''}</div>
<h2>{esc(row['alias'])}</h2><p class="title">{esc(row['title'])}</p><p class="venue">{esc(row['venue'])}</p>
<p>{esc(row['role'])}</p><div class="actions"><a class="pdf" href="{esc(row['pdf_url'],quote=True)}" target="_blank" rel="noopener">PDF 열기</a><a href="{esc(row['source_url'],quote=True)}" target="_blank" rel="noopener">원문 페이지</a><span class="{statusclass}">{esc(row['link_status'])}</span></div>
<p class="refs">기존 기록: {' · '.join(linkrefs)}</p>{'<p class="note">'+esc(row['metadata_note'])+'</p>' if row.get('metadata_note') else ''}</article>''')
    options = ''.join(f'<option value="{key}">{key}. {esc(label)} ({stats[key]})</option>' for key,label in CURATION['groups'].items())
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CVPR 근접 선행연구 PDF 목록</title>
<style>*{box-sizing:border-box}body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:0;background:#f5f6f8;color:#152232;line-height:1.65}main{max-width:1140px;margin:auto;padding:30px 22px}h1{font-size:28px;line-height:1.35}header p{max-width:950px}a{color:#145d99}.tools{position:sticky;top:0;background:#f5f6f8f5;padding:16px 0;display:flex;gap:10px;flex-wrap:wrap;z-index:1}input,select{font:inherit;padding:9px;border:1px solid #bcc6cf;border-radius:7px}input{flex:1;min-width:220px}select{max-width:100%}#count{font-weight:650}#list{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:15px}article{background:white;border:1px solid #dce1e5;border-radius:10px;padding:20px}article[hidden]{display:none}article h2{font-size:19px;line-height:1.35;margin:10px 0}.title{font-size:15px}.venue,.refs,.note{font-size:13px;color:#526477}.tags{display:flex;align-items:center;flex-wrap:wrap;gap:8px;font-size:12px;color:#526477}.tag{background:#edf1f4;padding:2px 7px;border-radius:4px}.priority{background:#e1f2f0;color:#126961}.actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px}.pdf{display:inline-block;background:#185f99;color:white;text-decoration:none;padding:7px 12px;border-radius:6px}.ok{color:#25754d}.pending{color:#9b5a00}.intro{border-left:4px solid #185f99;padding-left:14px}footer{margin-top:24px;font-size:13px;color:#526477}@media print{.tools{position:static}#list{display:block}article{break-inside:avoid;margin-bottom:15px}}</style></head><body><main>
'''
    page += f'''<header><h1>CVPR 근접 선행연구 · PDF 읽기 목록</h1><p><strong>근접·관련 67편 + 탈옥 트렌드 참고 4편</strong> · 최신 판정에 직접 인용 {latest}편 · 2026-09-09 정리</p><p class="intro">기존 기록의 결론은 조건부다. 단순 DP 방식 비교만으로는 기여가 약하며, 새 환자 집합 감사법의 차이를 입증해야 한다. 이 목록은 전부 정독·재현했다는 장부가 아니다.</p><p><a href="../{quote(REPORT.name)}">판단 근거·전체 목록 Markdown</a> · <a href="reading_list.csv" download>Excel용 CSV 저장</a></p><p>PDF {ok}/{len(rows)}개 응답 확인. PDF를 열고 브라우저의 저장 기능으로 다운로드할 수 있다. 나머지는 브라우저 확인 필요로 표시했다.</p></header>
<div class="tools"><input id="search" type="search" aria-label="논문 검색" placeholder="제목, 약칭, 비교할 점 검색"><select id="group" aria-label="연구 분야"><option value="">모든 분야</option>{options}</select><select id="scope" aria-label="읽기 범위"><option value="">전체 71편</option><option value="priority">먼저 읽을 15편</option><option value="latest">최신 18번 인용 35편</option></select></div><p id="count" aria-live="polite"></p><div id="list">{''.join(cards)}</div><footer>출처 범위: 01~18번의 URL 인용 전체. 기존 기록 링크는 로컬 Markdown 파일을 연다. HTTP/PDF 시작 바이트 확인은 전문 검토와 다르다.</footer>
'''
    page += '''</main><script>
const search = document.getElementById('search');
const group = document.getElementById('group');
const scope = document.getElementById('scope');
const cards = [...document.querySelectorAll('article')];
function filter() {
    const query = search.value.toLowerCase().trim();
    let visible = 0;
    for (const card of cards) {
        const matchesScope = !scope.value
            || (scope.value === 'priority' && card.dataset.priority === '1')
            || (scope.value === 'latest' && card.dataset.latest === '1')
            || (scope.value.startsWith('tier:') && card.dataset.tier === scope.value.slice(5));
        const show = matchesScope
            && (!query || card.dataset.search.includes(query))
            && (!group.value || card.dataset.group === group.value);
        card.hidden = !show;
        if (show) visible++;
    }
    document.getElementById('count').textContent = visible + '편 표시 / 전체 ' + cards.length + '편';
}
search.addEventListener('input', filter);
group.addEventListener('change', filter);
scope.addEventListener('change', filter);
filter();
</script></body></html>'''
    update_banner = '<p style="padding:14px;border:2px solid #2765a8;border-radius:8px"><strong>2026-09-10 직접 검토 업데이트:</strong> 기존 관련 67편 + 추가 12편의 개별 기록, 최신 비교군과 실제 공개 점수 재계산을 <a href="../research_2026-09-10/index.html">79편 검토 장부·드롭다운</a> 및 <a href="../research_2026-09-10/report.html">기여도 재판정</a>에 반영했다. 아래는 2026-09-09 인용 이력이다.</p>'
    page = page.replace('<header>', '<header>'+update_banner, 1)
    (OUT / 'index.html').write_text(page,encoding='utf-8')
    render_current_cvpr(rows, page, fields, update_banner)
    print(json.dumps({'papers':len(rows),'latest18':latest,'pdf_header_ok':ok,'groups':stats,'report':str(REPORT)},ensure_ascii=False),flush=True)

def render_current_cvpr(rows, page, fields, update_banner):
    current = [row for row in rows if row['latest18']]
    assert len(current) == 35
    assert not any(row['group'] in ('D', 'E', 'F') for row in current)
    counts = Counter(row['group'] for row in current)
    ok = sum(row['pdf_check']['status'] == 'PDF_HEADER_OK' for row in current)
    old18 = next(ROOT.glob('18_*.md')).name
    md = [f'''# 현재 CVPR 후보만: 환자 DP 의료영상 diffusion·환자별 노출 감사

- 범위 확인: 2026-09-09. 사용자가 지금 논의하는 대상은 이 CVPR 연구 하나라고 명확히 했다.
- 연구 질문: 한 환자의 여러 흉부 X-ray를 하나의 보호 단위로 보았을 때, 생성모델의 환자별 노출 위험을 어떻게 정확히 감사하고 동일 patient privacy budget에서 방어 방식의 차이를 검증할 것인가.
- 아래 **35편은 최신 18번 기여도 판정에서 직접 URL로 인용한 문헌 전체**다. 이 주제와 관련된 전 세계 논문 전부라는 뜻은 아니다.
- **35편 모두를 근접 연구로 부른 이전 답변은 정정한다.** 현재 핵심 주장·방법 대조에 우선 둘 6편, 직접 방법·평가 비교군 13편, 기반·확장 관련 문헌 16편으로 역할을 구분했다. [근접성 판단과 핵심 6편의 공통점·차이](cvpr_proximity_review.md)를 먼저 본다. 이는 현 후보에 따른 검토 우선순위이지 동일한 연구를 수행한 논문의 확정 개수가 아니다.
- **본학회 검토 정정:** 6편은 기여 중복 대조용 기존 선별이다. CDI(CVPR 2025) 등 직접 대조 누락이 확인됐으며 최신 본학회 SOTA 목록을 망라한 것이 아니다. {md_link('발표 형식·누락과 본학회 비교 우선 축', '../' + next(ROOT.glob('22_*.md')).name)}을 함께 본다.
- [PDF 검색 페이지](cvpr_patient_dp.html) · [35편 Excel CSV](cvpr_patient_dp.csv) · {md_link('기존 18번: 최신 기여도 판정', '../' + old18)}.
- PDF {ok}/35개 응답을 2026-09-09에 확인했다. 나머지는 브라우저 확인 필요로 표시한다.
- 기존 판단은 **단순 DP 방식 비교만으로는 기여가 약하며, 새 환자 집합 감사법의 유효성과 강한 기준선 대비 차이를 입증할 때 CVPR 후보를 유지**한다는 조건부 결론이다. 35편 모두의 전문·부록 정독 또는 재현 완료를 증명하는 기록은 확인되지 않는다.

졸업논문의 receipt·PP-Mark, 이전에 접은 생성방법 후보, 탈옥 트렌드 문헌은 이 목록에 포함하지 않았다. PFDM은 현재 연구의 출발점이자 18번의 직접 비교 문헌이므로 포함한다. 환자 위험·감사·공격 {counts['A']}편, DP 학습·보장 {counts['B']}편, PFDM {counts['C']}편이다.

''']
    for group, label in [('A','환자 위험·감사·공격·재식별'),('B','환자·사용자 DP와 DP diffusion'),('C','출발점 PFDM')]:
        md.append(f'## {label} — {counts[group]}편\n\n| ID | 문헌 | 발표 기록 | 우리 연구와 겹치는 점 | PDF | 기존 기록 |\n|---|---|---|---|---|---|\n')
        for row in current:
            if row['group'] != group:
                continue
            md.append(f"| {row['id']} | **{row['alias']}**<br>{row['title']} | {row['venue']} | {row['role']} | [PDF]({row['pdf_url']})<br>{row['link_status']} | {refs(row, '../')} |\n")
        md.append('\n')
    (OUT / 'cvpr_patient_dp.md').write_text(''.join(md), encoding='utf-8')
    with (OUT / 'cvpr_patient_dp.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in current:
            record = {field:row.get(field,'') for field in fields}
            record['historical_locations'] = '; '.join(o['file'] + ':' + str(o['line']) for o in row['occurrences'])
            writer.writerow(record)
    soup = BeautifulSoup(page, 'html.parser')
    soup.title.string = '현재 CVPR 환자 DP 의료영상 연구 — 선행 35편'
    for card in soup.select('article[data-latest="0"]'):
        card.decompose()
    replacement = BeautifulSoup(f'''<header><h1>현재 CVPR 연구 · 환자 DP 의료영상</h1><p><strong>현재 기여도 판정에 직접 인용한 35편</strong> · 2026-09-09 범위 정리</p><p class="intro">한 환자의 여러 흉부 X-ray를 하나의 보호 단위로 보고, 생성모델의 환자별 노출 위험을 감사하는 연구다. 새 환자 집합 감사법의 차이를 입증해야 한다는 기존의 조건부 판단을 유지한다.</p><p><a href="cvpr_patient_dp.md">35편 목록·판단 기록</a> · <a href="cvpr_patient_dp.csv" download>35편 CSV 저장</a></p><p>PDF {ok}/35개 응답 확인. 각 PDF를 열고 저장할 수 있다. PDF 접속 확인은 전문 정독·재현 완료와 다르다.</p></header>''', 'html.parser')
    replacement.header.append(BeautifulSoup(update_banner, 'html.parser'))
    soup.header.replace_with(replacement.header)
    soup.title.string = 'CVPR 기존 인용 35편 · 기여 중복 대조와 본학회 비교 정정'
    soup.header.h1.string = '현재 CVPR 연구 · 관련 문헌과 핵심 대조'
    clarification = BeautifulSoup('<p><strong>35편은 기존 18번의 URL 인용 목록이다.</strong> 그 안에서 기여 중복 대조 6편, 방법·평가 비교군 13편, 기반·확장 문헌 16편을 선별했다. 6편은 입문 목록이나 최신 본학회 SOTA 비교군 전체가 아니다. <a href="cvpr_proximity_review.md">기존 6편이 겹치는 부분과 차이 보기</a></p>', 'html.parser')
    soup.header.h1.insert_after(clarification.p)
    rationale = next(ROOT.glob('21_*.md'))
    rationale_paragraph = soup.new_tag('p')
    rationale_link = soup.new_tag('a', href='../' + quote(rationale.name))
    rationale_link.string = '선행을 검토하고도 방향을 유지한 이유·검증 조건'
    rationale_paragraph.append(rationale_link)
    soup.header.append(rationale_paragraph)
    maintrack = next(ROOT.glob('22_*.md'))
    maintrack_paragraph = soup.new_tag('p')
    maintrack_link = soup.new_tag('a', href='../' + quote(maintrack.name))
    maintrack_link.string = '본학회 비교 정정: CDI 등 누락·6편의 발표 형식·비교 우선 축'
    maintrack_paragraph.append(maintrack_link)
    maintrack_paragraph.append(' — 최신 본학회까지 충분히 검토했다는 해석은 철회한다. 새 확인 문헌은 연결 문서에 별도 기록했다.')
    soup.header.append(maintrack_paragraph)
    for option in soup.select('#group option'):
        value = option.get('value')
        if value and value not in counts:
            option.decompose()
        elif value:
            option.string = f'{value}. {CURATION["groups"][value]} ({counts[value]})'
    scope_select = soup.select_one('#scope')
    scope_select.clear()
    tier_counts = Counter(row['cvpr_tier'] for row in current)
    priority_count = sum(bool(row.get('priority')) for row in current)
    scope_options = [
        ('', f'전체 관련 문헌 {len(current)}편'),
        ('tier:1', f'기여 중복 대조 {tier_counts["1"]}편 (기존 선별)'),
        ('tier:2', f'방법·평가 비교군 {tier_counts["2"]}편'),
        ('tier:3', f'기반·확장 문헌 {tier_counts["3"]}편'),
        ('priority', f'먼저 읽을 {priority_count}편'),
    ]
    for value, label in scope_options:
        option = soup.new_tag('option', value=value)
        option.string = label
        scope_select.append(option)
    soup.footer.string = '현재 CVPR 후보의 기존 인용 35편. 출처·PDF 점검은 기존 71편 장부 기준이며, 후속 본학회 누락과 추가 대조는 22번 문서에 기록했다.'
    grid = soup.select_one('#list')
    for card in sorted(list(grid.select('article')), key=lambda card:int(card['data-tier'])):
        grid.append(card.extract())
    (OUT / 'cvpr_patient_dp.html').write_text(str(soup), encoding='utf-8')
    render_proximity_review(current)


def render_proximity_review(rows):
    by_id = {row['id']:row for row in rows}
    assigned = [pid for ids in PROXIMITY['tiers'].values() for pid in ids]
    assert len(assigned) == len(set(assigned)) == 35 and set(assigned) == set(by_id)
    text = ['# 현재 CVPR 문헌의 근접성 정정\n\n2026-09-09. **35편은 관련 문헌 전체이며, 모두 직접 경쟁하는 근접 연구라는 설명은 부정확했다.**\n\n' + PROXIMITY['qualification'] + '\n\n']
    text.append('현재 기여의 핵심은 환자 단위 위험, 환자 집합 membership 감사, diffusion 점수 보정과 의료 patient-DP 비교다. 이를 이미 점유한 부분과 대조하기 위한 우선 후보를 아래와 같이 골랐다. 논문 수를 줄였다고 신규성이 확인되거나 성공 가능성이 높아진 것은 아니다.\n\n')
    text.append(md_link('본학회 비교 정정: 발표 형식, CDI 등 누락과 비교 우선 축', '../' + next(ROOT.glob('22_*.md')).name) + '\n\n')
    for tier, ids in PROXIMITY['tiers'].items():
        text.append(f'## {PROXIMITY["labels"][tier]} — {len(ids)}편\n\n')
        if tier == '1':
            text.append('| 문헌 | 겹치는 핵심 | 여전히 다른 범위 | PDF |\n|---|---|---|---|\n')
            for pid in ids:
                row = by_id[pid]
                comparison = PROXIMITY['core_comparisons'][pid]
                text.append(f"| [{row['alias']}]({row.get('extra_source_url') or row['source_url']}) | {comparison['overlap']} | {comparison['difference']} | [PDF]({row['pdf_url']}) |\n")
        else:
            text.append('| 문헌 | 현재 후보에서의 역할 | PDF |\n|---|---|---|\n')
            for pid in ids:
                row = by_id[pid]
                text.append(f"| {row['alias']} | {row['role']} | [PDF]({row['pdf_url']}) |\n")
        text.append('\n')
    text.append('기반 문헌도 인용과 비교가 필요할 수 있다. 직접 경쟁성, 실험 baseline 필요성, 먼저 읽을 순서는 서로 다른 기준이다. 이번 핵심 6편의 공식 원문·초록과 기존 18번 대조표를 다시 확인했으며, 35편 전부의 새 전문 정독·재현을 수행한 것은 아니다.\n\n[현재 CVPR 관련 문헌 35편](cvpr_patient_dp.md) · [PDF 목록](cvpr_patient_dp.html) · [분류 명세](cvpr_proximity.json)\n')
    (OUT / 'cvpr_proximity_review.md').write_text(''.join(text), encoding='utf-8')


def main():
    rows = make_rows()
    if '--render-only' in sys.argv:
        checks = json.loads((OUT / 'pdf_checks.json').read_text(encoding='utf-8'))
    else:
        with cf.ThreadPoolExecutor(max_workers=6) as pool:
            checks = list(pool.map(check_pdf,rows))
        (OUT / 'pdf_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    render(rows,checks)

if __name__ == '__main__':
    main()
