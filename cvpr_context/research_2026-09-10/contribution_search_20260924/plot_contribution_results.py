"""Standalone plots for the exploratory contribution search."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
data=json.loads((root/'joint_compiler_results.json').read_text())
methods=['ORIGINAL','DIRECT_SOURCE_PSEUDOLABEL','SOURCE_LABEL_SOLVE','JOINT']
labels=['Original\nhard labels','DINO\nrelabel','DINO\nlabel solve','DINO + public ResNet\nlabel solve']
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
colors=['#2363ad','#d36c28']
for ax,rec in zip(axes,['DenseNet','ResNet18']):
    for color,draw in zip(colors,['DP1','DP2']):
        vals=[data['metrics'][draw+'_'+m+'_eval_'+rec]['AUROC'] for m in methods]
        ax.plot(np.arange(4),vals,'o-',color=color,label=draw,linewidth=1.5)
        for x,y in enumerate(vals):
            other='DP2' if draw=='DP1' else 'DP1'
            other_y=data['metrics'][other+'_'+methods[x]+'_eval_'+rec]['AUROC']
            ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7 if y>=other_y else -14),textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.set_xticks(np.arange(4),labels,fontsize=9)
    ax.set_title(rec+('\nExcluded from label objective; development receiver' if rec=='DenseNet' else '\nUsed in the joint label objective'),fontsize=10)
    ax.grid(axis='y',alpha=.25);ax.set_ylim(.565,.74);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('V image-level AUROC')
axes[1].legend(frameon=False)
fig.suptitle('Same DP summaries and same 128 PNGs; only labels change',fontsize=12)
fig.tight_layout()
fig.savefig(root/'label_compiler_auroc.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'label_compiler_auroc.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(8,3.3))
names=[];points=[];lows=[];highs=[]
for ref,text in [('ORIGINAL','original hard labels'),('SOURCE_LABEL_SOLVE','DINO label solve')]:
    for draw in ('DP1','DP2'):
        c=data['comparisons'][draw+'_JOINT_eval_DenseNet_minus_'+ref]['AUROC']
        names.append(draw+' vs '+text);points.append(c['delta']);lows.append(c['patient_cluster_95'][0]);highs.append(c['patient_cluster_95'][1])
y=np.arange(4)
ax.errorbar(points,y,xerr=np.stack([np.array(points)-lows,np.array(highs)-points]),fmt='o',color='#2363ad',capsize=4)
ax.axvline(0,color='#777',linestyle='--',linewidth=1)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('AUROC difference: joint labels minus comparator')
ax.set_title('DenseNet: exploratory intervals conditional on the two fixed DP releases',fontsize=10)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
fig.text(.5,-.01,'2,000 paired patient-cluster bootstrap draws; intervals do not cover development selection or DP-noise variability.',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(root/'label_compiler_contrasts.png',dpi=180,bbox_inches='tight');fig.savefig(root/'label_compiler_contrasts.pdf',bbox_inches='tight')
print('Saved four plot artifacts.')
