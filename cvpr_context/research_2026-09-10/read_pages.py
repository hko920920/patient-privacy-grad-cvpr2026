"""Print explicit PDF pages for manual evidence review; does not assign review status."""
from pathlib import Path
import argparse, re
import fitz

p = argparse.ArgumentParser()
p.add_argument('selections', nargs='+', help='ID:1,3-5; ID alone prints a heading index')
p.add_argument('--limit', type=int, default=0, help='Optional displayed characters per page; visibly marked')
a = p.parse_args()
r = Path(__file__).resolve().parent
for selection in a.selections:
    ident, _, spec = selection.partition(':')
    d = fitz.open(r / 'pdfs' / (ident + '.pdf'))
    if not spec:
        print('\n', ident, 'pages', len(d))
        for i, page in enumerate(d):
            lines = page.get_text().splitlines()
            heads = [x for x in lines if re.match(r'^(?:[1-9][. ]|[A-Z]\. |Table [0-9]|Algorithm |Appendix|[A-Z][A-Z ]{7,})',x)]
            print(i+1, ' | '.join(heads[:8])[:500])
        continue
    pages=[]
    for v in spec.split(','):
        x, _, y = v.partition('-')
        pages.extend(range(int(x), int(y or x)+1))
    for n in pages:
        s = re.sub(r'[ \t]{2,}', ' ', d[n-1].get_text())
        if a.limit and len(s) > a.limit:
            s = s[:a.limit] + '\n[DISPLAY CUT: remainder of page not shown]'
        print(f'\n{ident} PDF p{n}\n{s}')
