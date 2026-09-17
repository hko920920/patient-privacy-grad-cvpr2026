"""Acquire the single officially released ten-checkpoint ensemble, no inference."""
import re,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import gdown
R=Path(__file__).resolve().parents[1];P=R/'spec_sources/chexzero_review_20260917_v1';C=R.parent.parent/'code_working';DEST=C/'_models/chexzero_official_20220916'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 start=time.perf_counter();s=BeautifulSoup((P/'official_drive.html').read_text(encoding='utf-8'),'html.parser');items={}
 for e in s.select('[data-id]'):
  m=re.search(r'best_[A-Za-z0-9_.-]+\.pt',e.get_text(' ',strip=True))
  if m:items[e['data-id']]=m.group(0)
 if len(items)!=10:raise RuntimeError('Official release membership changed')
 DEST.mkdir(parents=True,exist_ok=True)
 release=dict(created_utc=datetime.now(timezone.utc).isoformat(),official_folder='https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno',
  source_html_sha256=sha(P/'official_drive.html'),selection='ALL ten official release files; no local performance-based checkpoint selection',
  files=[{'id':i,'filename':n,'url':'https://drive.google.com/file/d/'+i+'/view'} for i,n in sorted(items.items(),key=lambda t:t[1])])
 with (P/'official_release.json').open('x',encoding='utf-8') as f:json.dump(release,f,indent=2);f.write('\n')
 def get(row):
  t=time.perf_counter();p=DEST/row['filename'];part=p.with_suffix('.pt.part')
  try:
   if p.exists():raise RuntimeError('Pre-existing unbound checkpoint')
   result=gdown.download(id=row['id'],output=str(part),quiet=True,use_cookies=False,resume=False)
   if result is None or not part.exists() or part.stat().st_size<100_000_000:raise RuntimeError('Download not complete')
   with part.open('rb') as f:
    if f.read(2)!=b'PK':raise RuntimeError('Expected torch zip archive, not HTML')
   part.rename(p)
   rr=dict(**row,path=str(p),bytes=p.stat().st_size,sha256=sha(p),seconds=time.perf_counter()-t,status='DOWNLOADED')
  except Exception as e:rr=dict(**row,status='DOWNLOAD_FAILED',error=str(e),seconds=time.perf_counter()-t)
  print(json.dumps({'file':row['filename'],'status':rr['status'],'seconds':rr['seconds'],'error':rr.get('error','')}),flush=True)
  return rr
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(get,release['files']))
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,files=results,completed=sum(x['status']=='DOWNLOADED' for x in results),new_model_inference=0)
 with (P/'checkpoint_acquisition.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'completed':result['completed'],'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
