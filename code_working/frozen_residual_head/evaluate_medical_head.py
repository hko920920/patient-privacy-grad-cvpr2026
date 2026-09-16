"""Pinned medical encoders and descriptive paired generation metrics."""
import argparse,gc,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .run_medical_head import OUT,check,npz
from public_medical_backbone.run_pilot import CODE,RESEARCH,require,sha,read,save,now
from dp_training import k5_evaluator as metrics
from dp_training import run_k5_evaluator_preflight as prior

HUB=Path.home()/'.cache/huggingface/hub'
RAD=HUB/'models--microsoft--rad-dino/snapshots'/prior.RAD_REVISION
BIO=HUB/'models--microsoft--BiomedVLP-BioViL-T/snapshots'/prior.BIO_REVISION

def prepare(out):
    check(out)
    fixed={'rad_weights':RAD/'model.safetensors','rad_config':RAD/'config.json','rad_processor':RAD/'preprocessor_config.json',
        'biovil_image_weights':BIO/'biovil_t_image_model_proj_size_128.pt','biovil_text_weights':BIO/'model.safetensors',
        'biovil_config':BIO/'config.json','biovil_configuration_code':BIO/'configuration_cxrbert.py','biovil_modeling_code':BIO/'modeling_cxrbert.py'}
    expected=read(CODE/'_reports/nih_cxr14_k5_evaluator_preflight_v1_001/public_report.json')['source_hashes']
    for name,path in fixed.items():require(sha(path).upper()==expected[name],'Pinned evaluator '+name)
    files=list(fixed.values())+[Path(__file__),Path(__file__).with_name('verify_medical_evaluation.py'),Path(metrics.__file__),Path(prior.__file__),
        RESEARCH/'TRACK1_MEDICAL_HEAD_EVALUATION_NOTE_20260916.md',out/'contract.json',out/'reference.json']
    save(out/'evaluation_contract.json',dict(schema='medical-head-evaluation/v1',created_utc=now(),source_sha256={str(p):sha(p) for p in files},
        rad_snapshot=str(RAD),bio_snapshot=str(BIO),reference_patients=40,generated_images=192,methods=['backbone','public','pooled'],
        reference_conditions=['no_finding','pleural_effusion'],matched_prompts=['normal','effusion'],specific_prompts=['normal','effusion','cardiomegaly'],
        KID='full-sample cubic formula; mixed32 is descriptive under shared latents; conditional16/20 and mean also reported',
        PRDC_nearest_k=5,bootstrap_seed=26091631,bootstrap_repeats=4000,bootstrap_unit='16 common latent blocks, three specific prompts averaged',
        encoder_replay_indices=list(range(8)),encoder_replay_atol=1e-5,clinical_claim=False))
    print('EVALUATION_CONTRACT_FIXED',flush=True)

def load_image(row):
    if row['kind']=='real':return metrics.load_real_p256_grayscale(Path(row['path']))
    with Image.open(row['path']) as im:
        require(im.size==(256,256) and im.mode=='RGB','Generated image boundary')
        return im.convert('L')

def encode(out):
    c=check(out);ec=read(out/'evaluation_contract.json');v=read(out/'generation_verification.json')
    require(v['complete'] and v['manifest_sha256']==sha(out/'generation_manifest.json'),'Verified generated images')
    for p,h in ec['source_sha256'].items():require(sha(p)==h,'Evaluator source changed '+p)
    prior.configure_torch();start=time.perf_counter()
    refs=[dict(kind='real',row_id='real|'+x['image_id'],**x) for x in read(out/'reference.json')]
    gen=[dict(kind='generated',row_id=x['image_id'],path=str(out/x['image_path']),sha256=x['image_sha256'],method=x['method'],prompt_id=x['prompt_id'],seed_id=x['seed_id']) for x in read(out/'generation_manifest.json')]
    rows=[v for i in range(4) for v in (refs[i],gen[i])]+refs[4:]+gen[4:]
    require(len(rows)==232 and len({x['row_id'] for x in rows})==232,'All232 evaluator inputs')
    for row in rows:require(sha(row['path'])==row['sha256'],'Image before encoding')
    save(out/'evaluation_rows.json',rows);images=[load_image(row) for row in rows]
    from transformers import AutoImageProcessor,AutoModel,AutoTokenizer
    proc=AutoImageProcessor.from_pretrained(RAD,local_files_only=True,use_fast=False)
    model=AutoModel.from_pretrained(RAD,local_files_only=True,use_safetensors=True).eval().cuda()
    def rad_batch(group):
        inp=proc(images=[im.convert('RGB') for im in group],return_tensors='pt')
        return model(**{k:v.cuda() for k,v in inp.items()}).pooler_output.float().cpu().numpy()
    tick=time.perf_counter()
    with torch.inference_mode():
        rad=np.concatenate([rad_batch(images[i:i+2]) for i in range(0,len(images),2)])
        rad_replay=np.concatenate([rad_batch(images[i:i+2]) for i in range(0,8,2)])
    rad_seconds=time.perf_counter()-tick;del model,proc;gc.collect();torch.cuda.empty_cache()
    require(rad.shape==(232,768) and np.isfinite(rad).all(),'RAD features')
    require(float(np.max(np.abs(rad_replay-rad[:8])))<=1e-5,'RAD replay drift')
    from health_multimodal.image import ImageEncoderType,ImageModel
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
    transform=create_chest_xray_transform_for_inference(resize=512,center_crop_size=448)
    model=ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,joint_feature_size=128,pretrained_model_path=BIO/'biovil_t_image_model_proj_size_128.pt').eval().cuda()
    def bio_batch(group):
        inp=torch.stack([transform(im) for im in group]).cuda();v=model(inp).projected_global_embedding
        return torch.nn.functional.normalize(v,dim=-1).float().cpu().numpy()
    tick=time.perf_counter()
    with torch.inference_mode():
        bio=np.concatenate([bio_batch(images[i:i+8]) for i in range(0,len(images),8)])
        bio_replay=bio_batch(images[:8])
    bio_seconds=time.perf_counter()-tick;del model,transform;gc.collect();torch.cuda.empty_cache()
    require(bio.shape==(232,128) and np.isfinite(bio).all(),'BioViL image features')
    require(float(np.max(np.abs(bio_replay-bio[:8])))<=1e-5,'BioViL replay drift')
    tokenizer=AutoTokenizer.from_pretrained(BIO,local_files_only=True,use_fast=False)
    model,loading=AutoModel.from_pretrained(BIO,trust_remote_code=True,local_files_only=True,use_safetensors=True,output_loading_info=True)
    require(sorted(loading.get('unexpected_keys',[]))==['bert.pooler.dense.bias','bert.pooler.dense.weight'] and not loading.get('missing_keys'),'Known BioViL text load boundary')
    model=model.eval().cuda();prompt_ids=list(c['prompts']);tokens=tokenizer([c['prompts'][k] for k in prompt_ids],padding=True,return_tensors='pt')
    with torch.inference_mode():text=model.get_projected_text_embeddings(tokens.input_ids.cuda(),tokens.attention_mask.cuda()).float().cpu().numpy()
    require(text.shape==(4,128) and np.max(np.abs(np.linalg.norm(text.astype(np.float64),axis=1)-1))<=1e-5,'Text vectors')
    npz(out/'evaluation_features.npz',rad=rad,bio=bio,text=text,rad_replay=rad_replay,bio_replay=bio_replay)
    for row in rows:require(sha(row['path'])==row['sha256'],'Image unchanged after encoding')
    save(out/'encoding.json',dict(complete=True,seconds=time.perf_counter()-start,images=232,generated=192,references=40,prompt_ids=prompt_ids,
        rad_seconds=rad_seconds,bio_image_seconds=bio_seconds,rad_replay_max_abs=float(np.max(np.abs(rad_replay-rad[:8]))),
        bio_replay_max_abs=float(np.max(np.abs(bio_replay-bio[:8]))),
        features_sha256=sha(out/'evaluation_features.npz'),rows_sha256=sha(out/'evaluation_rows.json'),evaluation_contract_sha256=sha(out/'evaluation_contract.json'),
        generation_manifest_sha256=sha(out/'generation_manifest.json'),clinical_validity=False))
    print('MEDICAL_ENCODERS_COMPLETE',flush=True)

def score(out):
    start=time.perf_counter();c=read(out/'contract.json');ec=read(out/'evaluation_contract.json');enc=read(out/'encoding.json')
    require(enc['features_sha256']==sha(out/'evaluation_features.npz') and enc['rows_sha256']==sha(out/'evaluation_rows.json'),'Fixed encoded inputs')
    rows=read(out/'evaluation_rows.json')
    with np.load(out/'evaluation_features.npz') as f:rad=f['rad'].astype(np.float64);bio=f['bio'].astype(np.float64);text=f['text'].astype(np.float64)
    real=[i for i,r in enumerate(rows) if r['kind']=='real'];specific=ec['specific_prompts'];pids=enc['prompt_ids'];scores=bio@text.T
    per_image=[];by_method={};maps={}
    for method in ec['methods']:
        allidx=[i for i,r in enumerate(rows) if r.get('method')==method]
        ii=[i for i in allidx if rows[i]['prompt_id'] in ec['matched_prompts']]
        conditional={}
        for prompt,condition in zip(ec['matched_prompts'],ec['reference_conditions']):
            ri=[i for i in real if rows[i]['matched_condition']==condition];gi=[i for i in allidx if rows[i]['prompt_id']==prompt]
            require(len(ri)==20 and len(gi)==16,'Fixed conditional metric counts')
            conditional[prompt]=metrics.kid_unbiased(rad[ri],rad[gi])
        by_method[method]=dict(KID_mixed_descriptive=metrics.kid_unbiased(rad[real],rad[ii]),KID_by_condition=conditional,
            KID_condition_mean=float(np.mean(list(conditional.values()))),PRDC=metrics.prdc(rad[real],rad[ii],nearest_k=5),
            effective_rank64=metrics.effective_rank(rad[allidx]),feature_variance64=float(np.var(rad[allidx],axis=0,ddof=1).sum()))
        maps[method]={}
        for i in allidx:
            pid=rows[i]['prompt_id'];matched=float(scores[i,pids.index(pid)])
            others=[scores[i,pids.index(p)] for p in specific if p!=pid]
            margin=matched-float(np.mean(others)) if pid in specific else None
            row=dict(image_id=rows[i]['row_id'],method=method,prompt_id=pid,seed_id=rows[i]['seed_id'],matched_cosine=matched,specific_margin=margin)
            per_image.append(row);maps[method][(row['seed_id'],pid)]=row
        relevant=[x for x in per_image if x['method']==method and x['prompt_id'] in specific]
        by_method[method]['specific_matched_cosine']=float(np.mean([x['matched_cosine'] for x in relevant]))
        by_method[method]['specific_margin']=float(np.mean([x['specific_margin'] for x in relevant]))
    rng=np.random.default_rng(ec['bootstrap_seed']);boot=rng.integers(0,16,size=(ec['bootstrap_repeats'],16));paired={}
    for a,b in [('public','backbone'),('pooled','public'),('pooled','backbone')]:
        item={}
        for metric in ['matched_cosine','specific_margin']:
            blocks=np.array([np.mean([maps[a][(sid,p)][metric]-maps[b][(sid,p)][metric] for p in specific]) for sid in c['generation_seeds']])
            means=blocks[boot].mean(1);item[metric]=dict(mean=float(blocks.mean()),positive_blocks=int(np.sum(blocks>0)),negative_blocks=int(np.sum(blocks<0)),blocks=blocks.tolist(),descriptive_95_interval=np.quantile(means,[.025,.975]).tolist())
        item['KID_condition_mean_difference']=by_method[a]['KID_condition_mean']-by_method[b]['KID_condition_mean']
        item['KID_mixed_difference']=by_method[a]['KID_mixed_descriptive']-by_method[b]['KID_mixed_descriptive'];paired[a+'_minus_'+b]=item
    save(out/'per_image_scores.json',per_image)
    result=dict(complete=True,seconds=time.perf_counter()-start,methods=by_method,paired=paired,
        features_sha256=sha(out/'evaluation_features.npz'),encoding_sha256=sha(out/'encoding.json'),per_image_sha256=sha(out/'per_image_scores.json'),
        evaluation_contract_sha256=sha(out/'evaluation_contract.json'),mixed_KID_unbiasedness_not_claimed=True,
        shared_latent_blocks=16,clinical_utility_proven=False,DP=False,code_sha256=sha(__file__))
    save(out/'evaluation.json',result);print({'methods':by_method,'pooled_minus_public':paired['pooled_minus_public']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','encode','score'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a.out)
