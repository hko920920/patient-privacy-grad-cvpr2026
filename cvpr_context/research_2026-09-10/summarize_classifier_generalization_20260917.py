"""Export aggregate diagnostic evidence and scientific plots. No model calls."""
from pathlib import Path
import json, hashlib, csv, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'code_working/_reports/classifier_generalization_20260917_v1'
ARMS=['R1','Dreal','L_public','L_pooled']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    if p.exists():
        assert read(p)==x, 'Do not overwrite changed aggregate evidence'
        return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    result=read(OUT/'result.json');verification=read(OUT/'verification.json');contract=read(OUT/'contract.json')
    assert verification['status']=='PASS_FIXED_INFERENCE_DIAGNOSTIC_INTEGRITY'
    means={};seed_rows=[]
    for arm in ARMS:
        rr=[r for r in result['model_results'] if r['arm']==arm]
        means[arm]={}
        for source in rr[0]['components']:
            parts=[r['components'][source] for r in rr]
            means[arm][source]={k:float(np.mean([p[k] for p in parts])) for k in
              ['AUROC','AP','BCE','balanced_BCE','exposure_weighted_BCE']}
            means[arm][source].update(images_per_seed=[p['images'] for p in parts],
                                     positive_images_per_seed=[p['positive_images'] for p in parts])
        means[arm]['development']={k:float(np.mean([r['development'][k] for r in rr])) for k in
                                   ['AUROC','AP','BCE','balanced_BCE']}
        means[arm]['development']['by_label']={str(y):{k:float(np.mean([r['development']['by_label'][str(y)][k] for r in rr]))
                   for k in ['BCE','logit_mean']}for y in (0,1)}
        means[arm]['logged_last50_BCE']=float(np.mean([r['training_log']['50']['mean_logged_BCE'] for r in rr]))
        for r in rr:
            for source,p in list(r['components'].items())+[('method_development',r['development'])]:
                seed_rows.append(dict(arm=arm,seed=r['seed'],source=source,images=p['images'],positives=p['positive_images'],
                                      AUROC=p['AUROC'],AP=p['AP'],BCE=p['BCE'],balanced_BCE=p['balanced_BCE']))
    write(ROOT/'spec_sources/classifier_generalization_record_20260917.json',
          dict(scope='Frozen-checkpoint diagnostic, not a new method efficacy test',
               contract_created_utc=contract['created_utc'],contract_sha256=sha(OUT/'contract.json'),
               result_sha256=sha(OUT/'result.json'),verification_sha256=sha(OUT/'verification.json'),
               source_sha256=contract['source_sha256'],budget=contract['budget'],means=means,
               model_results=result['model_results'],verification=verification,
               actual_inference_seconds=result['seconds'],all_model_states_unchanged=result['all_model_states_unchanged'],
               maximum_witness_logit_difference=result['maximum_witness_logit_difference'],
               method_efficacy_retested=False,expert_opened=False,reserved_opened=False,DP=False,final_ready=False))
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(seed_rows[0]));w.writeheader();w.writerows(seed_rows)
    table=ROOT/'spec_sources/classifier_generalization_seed_metrics_20260917.csv'
    if table.exists():
        assert table.read_text(encoding='utf-8').splitlines()==f.getvalue().splitlines()
    else:
        with table.open('x',encoding='utf-8',newline='') as target:target.write(f.getvalue())
    fig,ax=plt.subplots(figsize=(9,4.4))
    labels={'real/public':'Seen public real','real/private':'Seen private real',
            'synthetic/lora_public':'Seen requested-label synthetic','synthetic/lora_pooled':'Seen requested-label synthetic',
            'method_development':'Stored method development'}
    colors={'Seen public real':'#176977','Seen private real':'#628e45','Seen requested-label synthetic':'#946bb3','Stored method development':'#b04939'}
    for i,arm in enumerate(ARMS):
        sources=list(means[arm].keys());sources=[s for s in sources if s.startswith(('real/','synthetic/'))]+['method_development']
        offsets=np.linspace(-.22,.22,len(sources))
        for off,source in zip(offsets,sources):
            vals=[r['AUROC'] for r in seed_rows if r['arm']==arm and r['source']==source]
            name=labels[source];color=colors[name]
            ax.scatter(i+off+np.array([-.025,0,.025]),vals,c=color,s=32,alpha=.85)
            ax.plot([i+off-.04,i+off+.04],[np.mean(vals)]*2,color=color,lw=3,label=name if (i==0 or (i==1 and source=='real/private') or (i==2 and source.startswith('synthetic'))) else None)
    ax.axhline(.5,ls='--',c='#aab0b7',lw=1)
    ax.set(xticks=range(4),xticklabels=ARMS,ylabel='Image AUROC (raw logits)',ylim=(.46,1.035),
           title='Fixed eval mode: seen training images vs stored development')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    ax.legend(loc='center right',fontsize=8)
    fig.text(.5,.015,'Dots are classifier seeds; no independent new patient test. Real labels are weak; synthetic labels are requested.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(ROOT/'figures/classifier_generalization_auroc_20260917.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for y,color in [(0,'#347891'),(1,'#b04939')]:
        vals=[means[a]['development']['by_label'][str(y)]['logit_mean'] for a in ARMS]
        axes[0].plot(ARMS,vals,'o-',label=f'Weak label {y}',color=color)
        vals=[means[a]['development']['by_label'][str(y)]['BCE'] for a in ARMS]
        axes[1].plot(ARMS,vals,'o-',label=f'Weak label {y}',color=color)
    axes[0].set(ylabel='Mean raw logit',title='Development scores for both classes are low')
    axes[1].set(ylabel='Mean BCE',title='Class-wise loss exposes rare-positive errors')
    for ax in axes:ax.legend();ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.02,'Stored development: 223 positive / 5,047 images. Mean over 3 classifier seeds; no threshold or calibration change.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(ROOT/'figures/classifier_generalization_dev_logits_20260917.png',dpi=170);plt.close(fig)
    print(json.dumps(means,indent=2))
if __name__=='__main__':main()
