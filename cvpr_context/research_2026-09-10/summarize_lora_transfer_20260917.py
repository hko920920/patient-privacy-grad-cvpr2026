"""Render descriptive development figures and export aggregate public record."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parent
CODE=ROOT.parents[1]/'code_working'
OUT=CODE/'_reports/lora_transfer_20260917_v1'
OLD=CODE/'_reports/downstream_development_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(OUT/'result.json');verify=read(OUT/'verification.json');additional=read(OUT/'additional_input_verification.json')
    spec=ROOT/'spec_sources';arms=['L_public','L_pooled'];colors=['#2278a5','#c35d33']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    train={}
    for a,col in zip(arms,colors):
        train[a]=read(OUT/'training'/a/'report.json');rr=[json.loads(x) for x in (OUT/'training'/a/'trace.jsonl').read_text().splitlines()];loss=[r['loss'] for r in rr if r['committed']]
        axes[0].plot(np.arange(20,len(loss)+1),np.convolve(loss,np.ones(20)/20,mode='valid'),label=a,color=col)
        first=read(OUT/'runs'/f'{a}_11'/'trace.json')['trace'];axes[1].plot(np.arange(20,len(first)+1),np.convolve([r['loss'] for r in first],np.ones(20)/20,mode='valid'),label=a,color=col)
    axes[0].set(xlabel='Successful LoRA update',ylabel='Conditional epsilon MSE, moving mean20',title='Training loss: not downstream utility')
    axes[1].set(xlabel='Classifier update (seed11)',ylabel='BCE, moving mean20',yscale='log',title='Fixed classifier recipe')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(spec/'lora_transfer_loss_20260917.png',dpi=150);plt.close(fig)
    names=['R0','R1','S0','S1','S3','L_public','L_pooled','Dreal'];fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for mi,metric in enumerate(['AUROC','AP']):
        ax=axes[mi]
        for si,s in enumerate([11,23,37]):ax.scatter(np.arange(len(names))+(si-1)*.15,[result['scores'][a][si][metric] for a in names],s=23,label=f'seed{s}',alpha=.8)
        ax.scatter(np.arange(len(names)),[result['means'][a][metric] for a in names],marker='_',s=240,color='black',label='mean')
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel=metric,title='Weak-label method development');ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(spec/'lora_transfer_development_metrics_20260917.png',dpi=150);plt.close(fig)
    # Fixed first four latent blocks, both requested labels, four methods. No visual selection.
    cells=read(OUT/'contract.json')['plan']['generation']['cells'][:8]
    gm={(r['cell'],r['method']):(OUT,r) for r in read(OUT/'generation_manifest.json')}
    gm.update({(r['cell'],r['method']):(OLD,r) for r in read(OLD/'generation_manifest.json')})
    methods=[('public','Old public head'),('pooled','Old pooled head'),('lora_public','LoRA public'),('lora_pooled','LoRA pooled')]
    fig,axes=plt.subplots(4,8,figsize=(16,8.5))
    for i,(method,label) in enumerate(methods):
        for j,cell in enumerate(cells):
            root,r=gm[cell['cell'],method]
            with Image.open(root/r['image_path']) as im:axes[i,j].imshow(np.asarray(im),cmap='gray',vmin=0,vmax=255)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(cell['cell']+'\n'+cell['prompt'],fontsize=9)
            if j==0:axes[i,j].set_ylabel(label,fontsize=10)
    fig.suptitle('Fixed first4 shared latent blocks; requested labels are not expert diagnoses',fontsize=13)
    fig.tight_layout();fig.savefig(spec/'lora_transfer_first4blocks_20260917.png',dpi=130);plt.close(fig)
    gen=read(OUT/'generation_manifest.json');classifiers=[read(OUT/'runs'/f'{a}_{s}'/'trace.json') for a in arms for s in [11,23,37]]
    manifest_target=spec/'lora_transfer_generation_manifest_20260917.json'
    with manifest_target.open('x',encoding='utf-8') as f:json.dump(gen,f,ensure_ascii=False,indent=2);f.write('\n')
    summary=dict(schema='lora-transfer-public-record/v1',status='completed_verified',result=result,verification=verify,additional_input_verification={k:v for k,v in additional.items() if k not in ['source_file_sha256','access']},
        training={a:{k:v for k,v in t.items() if k not in ['exposure','access','changed_tensors']} for a,t in train.items()},
        training_changed_tensors={a:len(t['changed_tensors']) for a,t in train.items()},
        timing=dict(cache_seconds=sum(read(OUT/('cache_'+r)/'report.json')['seconds'] for r in ['public','private']),lora_training_seconds=sum(t['seconds'] for t in train.values()),
            generation_compute_seconds=sum(r['seconds'] for r in gen),generation_IO_seconds=sum(r['output_IO_seconds'] for r in gen),classifier_training_seconds=sum(r['seconds'] for r in classifiers),
            classifier_eval_seconds=sum(r['inference_seconds'] or 0 for a in arms for r in result['scores'][a])),
        peak_allocated=dict(lora_training=max(t['peak_allocated'] for t in train.values()),generation=max(r['peak_allocated'] for r in gen),classifier=max(r['peak_allocated'] for r in classifiers)),
        source_sha256={str(p.relative_to(CODE.parent)):sha(p) for p in [OUT/'contract.json',OUT/'result.json',OUT/'verification.json',OUT/'generation_manifest.json',manifest_target,OUT/'additional_input_verification.json',ROOT/'summarize_lora_transfer_20260917.py']},
        unchanged_old_failure=True,expert_final_opened=False,reserved_opened=False,new_DP=0,final_ready=False)
    target=spec/'lora_transfer_execution_record_20260917.json'
    with target.open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(timing=summary['timing'],peak=summary['peak_allocated'],means=result['means'],gate=result['engineering_gate_passed']),ensure_ascii=True,indent=2))
if __name__=='__main__':main()
