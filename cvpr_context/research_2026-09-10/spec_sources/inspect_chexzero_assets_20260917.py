"""Tensor-only checkpoint and architecture inspection; never call model forward."""
import sys,json,hashlib,time,gc,importlib.util
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import torch
R=Path(__file__).resolve().parents[1];P=R/'spec_sources/chexzero_review_20260917_v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def main():
 t=time.perf_counter();torch.set_num_threads(4);acq=json.loads((P/'checkpoint_acquisition.json').read_text())
 need(acq['completed']==10,'All ten official checkpoints required')
 spec=importlib.util.spec_from_file_location('chexzero_pinned_model',P/'source/model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 records=[];template=None
 for row in acq['files']:
  p=Path(row['path']);need(sha(p)==row['sha256'],'Asset SHA')
  state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  need(isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values()),'Tensor-only state')
  shape={k:list(v.shape) for k,v in state.items()}
  if template is None:template=shape
  need(shape==template,'Consistent architecture')
  need(all(torch.isfinite(v).all().item() for v in state.values()),'Finite weights')
  model=module.build_model(dict(state)).float().eval()
  loaded=model.state_dict();need(set(loaded)==set(state),'Full state key set')
  need(all(torch.equal(loaded[k],state[k].float()) for k in state),'All parameters loaded without additional rounding')
  records.append(dict(filename=p.name,sha256=row['sha256'],bytes=p.stat().st_size,state_keys=len(state),parameters=sum(v.numel() for v in state.values()),dtype_counts=dict(Counter(str(v.dtype) for v in state.values())),
   image_resolution=model.visual.input_resolution,patch_size=list(state['visual.conv1.weight'].shape[-2:]),context_length=model.context_length,embedding_dimension=state['text_projection'].shape[1],
   strict_load=True,all_tensors_finite=True,all_parameter_values_preserved=True,model_forward_calls=0))
  print(json.dumps({'filename':p.name,'keys':len(state),'strict_load':True}),flush=True)
  del state,loaded,model;gc.collect()
 # Tokenizer only: no image or text encoder evaluation.
 sys.path.insert(0,str(P/'source'))
 import clip
 prompts=['Emphysema','no Emphysema','Pneumothorax','no Pneumothorax']
 tok=clip.tokenize(prompts,context_length=77).tolist()
 need(len({tuple(x) for x in tok})==4,'Distinct fixed prompt tokens')
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PASS_ASSET_LOAD_ONLY_NOT_EFFICACY',seconds=time.perf_counter()-t,torch_version=torch.__version__,
  source_model_sha256=sha(P/'source/model.py'),source_tokenizer_sha256=sha(P/'source/simple_tokenizer.py'),acquisition_sha256=sha(P/'checkpoint_acquisition.json'),
  files=records,total_bytes=sum(x['bytes'] for x in records),prompts=prompts,token_ids=tok,model_forward_calls=0,GPU_calls=0,NIH_pixels_read=0,
  inspector_sha256=sha(__file__))
 with (P/'asset_inspection.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['status','seconds','total_bytes','model_forward_calls','GPU_calls']},indent=2))
if __name__=='__main__':main()
