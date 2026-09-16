"""Fetch inspectable public source snapshots. Does not install or execute them."""
from pathlib import Path, PurePosixPath
import concurrent.futures as cf
import hashlib,json,zipfile
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'code_sources'
REPOS={'X01':'sprintml/copyrighted_data_identification','B08':'EzzzLi/DP-LORA',
       'B13':'TanqiuJiang/RAPID','X03':'py85252876/GSA','X06':'2019ChenGong/DPImageBench'}
SUFFIXES={'.py','.md','.yaml','.yml','.txt','.toml','.json','.sh'}

def fetch(item):
    pid,repo=item
    record={'id':pid,'repository':'https://github.com/'+repo,'status':'not_started','executed':False}
    try:
        response=requests.get('https://api.github.com/repos/'+repo,timeout=25)
        response.raise_for_status()
        meta=response.json()
        branch=meta['default_branch']
        response=requests.get('https://api.github.com/repos/'+repo+'/commits/'+branch,timeout=25)
        response.raise_for_status()
        sha=response.json()['sha']
        record.update(commit=sha,default_branch=branch)
        response=requests.get('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1',timeout=25)
        response.raise_for_status()
        tree=response.json()
        base=OUT/pid
        base.mkdir(exist_ok=True)
        (base/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        entries=[x for x in tree['tree'] if x['type']=='blob' and x.get('size',0)<500000 and (PurePosixPath(x['path']).suffix.lower() in SUFFIXES or PurePosixPath(x['path']).name.upper().startswith('LICENSE'))]
        # Download complete small text source; no datasets, checkpoints, binary objects, or installs.
        def one(entry):
            parts=PurePosixPath(entry['path'])
            if parts.is_absolute() or '..' in parts.parts: raise ValueError('Unsafe source path')
            url='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+entry['path']
            path=base.joinpath(*parts.parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            content=path.read_bytes() if path.exists() else b''
            blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if blob != entry['sha']:
                resp=requests.get(url,timeout=30)
                resp.raise_for_status()
                content=resp.content
                actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                if actual != entry['sha']:raise ValueError('Git blob mismatch: '+entry['path'])
                path.write_bytes(content)
            return {'path':entry['path'],'sha256':hashlib.sha256(content).hexdigest(),'url':url}
        with cf.ThreadPoolExecutor(max_workers=4) as pool: files=list(pool.map(one,entries))
        record.update(status='source_downloaded_not_executed',commit=sha,default_branch=branch,
                      license=(meta.get('license') or {}).get('spdx_id'),files=files,tree_truncated=tree.get('truncated',False))
    except Exception as exc:record.update(status='fetch_failed',error=str(exc))
    return record

def extract_supplements():
    results=[]
    for path in OUT.glob('*_supplement.zip'):
        pid=path.name.split('_')[0]
        base=OUT/(pid+'_supp')
        base.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            inventory=[{'path':i.filename,'bytes':i.file_size} for i in archive.infolist()]
            (base/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
            for info in archive.infolist():
                rel=PurePosixPath(info.filename)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
                if rel.suffix.lower() not in SUFFIXES or info.file_size>500000:continue
                target=base.joinpath(*rel.parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(info.filename))
        results.append({'id':pid,'status':'official_supplement_text_extracted_not_executed','zip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entries':inventory})
    return results

if __name__=='__main__':
    records=extract_supplements()
    (ROOT/'code_acquisition.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    for item in REPOS.items():
        record=fetch(item)
        records.append(record)
        (ROOT/'code_acquisition.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps({k:record[k] for k in ['id','status','commit','error'] if k in record}),flush=True)
