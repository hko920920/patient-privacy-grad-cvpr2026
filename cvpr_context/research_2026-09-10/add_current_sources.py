from pathlib import Path
import json, concurrent.futures as cf
from acquire import fetch_pdf, save_json
r=Path(__file__).resolve().parent
extra=[
 ('X08','DP-FETA','IEEE S&P 2025','From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis','https://arxiv.org/pdf/2504.01395'),
 ('X09','FETA-Pro','USENIX Security 2026','From Easy to Hard++: Promoting Differentially Private Image Synthesis Through Spatial-Frequency Curriculum','https://www.usenix.org/sites/default/files/conference-files/sec26cycle1-final514.pdf'),
 ('X10','DP-SAD','ECCV 2024','Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01066.pdf'),
 ('X11','PFAMI','publication venue to verify from PDF','A Probabilistic Fluctuation based Membership Inference Attack for Diffusion Models','https://arxiv.org/pdf/2308.12143'),
 ('X12','MoFit','ICLR 2026','No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings','https://arxiv.org/pdf/2602.22689')]
rows=json.loads((r/'registry.json').read_text(encoding='utf-8'))
old={x['id'] for x in rows}
for ident,alias,venue,title,url in extra:
 if ident not in old: rows.append(dict(id=ident,alias=alias,venue=venue,title=title,source_url=url,pdf_url=url,latest18=False,scope='current_or_related_cvpr_research'))
save_json(r/'registry.json',rows)
acq={x['id']:x for x in json.loads((r/'acquisition.json').read_text(encoding='utf-8'))}
with cf.ThreadPoolExecutor(max_workers=3) as pool:
 for x in pool.map(fetch_pdf,[x for x in rows if x['id'] in {e[0] for e in extra}]):
  acq[x['id']]=x
  print(x['id'],x['download'],x.get('pages'),flush=True)
save_json(r/'acquisition.json',list(acq.values()))
