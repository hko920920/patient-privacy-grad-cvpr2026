"""Bind supplied v3 as a separate amendment; never mutate prior experiments."""
from __future__ import annotations
import importlib.metadata, json, platform, subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from .contracts import *
from .datasets import manifests, EXPECTED, TRAIN, DEV, public_templates, derangement
from .encoders import BIO_PATH,BIO_REV,BIO_SHA,DENSE_PATH

def bind():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'w0_bindings.json').exists(),'W0 already bound; no overwrite')
    supplied_hashes={}
    for line in (SUPPLIED/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split(None,1); path=SUPPLIED/name.strip()
        require(sha(path)==expected,'Supplied bundle checksum mismatch: '+name)
        supplied_hashes[str(path)]=expected
    require(read(OUT/'reference_checks_rerun.json')['status']=='PASS_TOY_ALGEBRA_ONLY','Run supplied toy reference first')
    groups=manifests()
    prior=read(CODE/'_reports/real_support_plan_20260917_v1/plan.json')
    for path in (TRAIN,DEV):
        expected=prior['source_sha256'].get(str(path))
        require(expected==sha(path),'Prior manifest binding changed')
    require(sha(BIO_PATH)==BIO_SHA,'Wrong BioViL image checkpoint')
    from torchvision.models import DenseNet121_Weights
    DenseNet121_Weights.IMAGENET1K_V1.get_state_dict(progress=False,check_hash=True)
    require(DENSE_PATH.exists(),'DenseNet checkpoint not cached')
    import health_multimodal, torchvision
    health_root=Path(health_multimodal.__file__).parent
    health_files={str(p):sha(p) for p in sorted((health_root/'image').rglob('*.py'))}
    distribution=importlib.metadata.distribution('hi-ml-multimodal')
    direct_url=distribution.read_text('direct_url.json')
    commit=(json.loads(direct_url).get('vcs_info',{}).get('commit_id') if direct_url else None)
    upstream=subprocess.run(['git','ls-remote','https://github.com/hko920920/patient-privacy-grad-cvpr2026.git','refs/heads/main'],capture_output=True,text=True,check=True).stdout.split()[0]
    local=subprocess.run(['git','rev-parse','HEAD'],cwd=CODE,capture_output=True,text=True)
    with (SUPPLIED/'experiment_contract_template.yaml').open(encoding='utf-8') as f: cfg=yaml.safe_load(f)
    cfg['schema']='prrd-study-contract/v3-local-1'
    cfg['status']='W0_BOUND_PUBLIC_RUNTIME_PENDING'
    cfg['repo'].update(supplied_reference_commit=cfg['repo']['commit'],commit=upstream,
        remote_observed_at=now(),local_git_commit=local.stdout.strip() if local.returncode==0 else 'NOT_A_GIT_CHECKOUT',
        local_source_identity='per-file SHA256; remote commit is provenance, not identity of new uncommitted runtime')
    cfg['execution'].update(gpu_authorized=True,runtime_implemented=False,authorization_scope='W0/W1 public inputs; W2 requires measured cost and numeric approved time bound')
    bindings=dict(supplied_hashes)
    for path in (TRAIN,DEV,RESEARCH/'TRACK1_PATIENT_RELATION_DISTILLATION_COMPARISON_20260918.md',
                 RESEARCH/'TRACK1_PATIENT_RELATION_DISTILLATION_PROTOCOL_20260918.md',
                 CODE/'relation_distillation/settings_v1.json',CODE/'relation_distillation/moments.py',
                 CODE/'downstream_utility/data_v2.py',CODE/'downstream_utility/train_v2.py',
                 BIO_PATH,DENSE_PATH,CODE/'_reports/downstream_development_20260917_v1/cluster_bootstrap.npz'):
        bindings[str(path)]=sha(path)
    bindings.update(health_files)
    initialization={}
    for bank_seed in SEEDS:
        initialization[str(bank_seed)]=[{'image_id':r['image_id'],'path':r['path'],'sha256':r['sha256']} for r in public_templates(groups['P'],bank_seed)]
    save(OUT/'templates_P_private.json',initialization)
    save(OUT/'permutation_private.json',{str(s):derangement(102,s).tolist() for s in SEEDS})
    for path in (OUT/'templates_P_private.json',OUT/'permutation_private.json'): bindings[str(path)]=sha(path)
    for role in ('P','Q','V'):
        cfg['data'][role].update(manifest=str(DEV if role=='V' else TRAIN),
            manifest_sha256=sha(DEV if role=='V' else TRAIN),selector={'P':'real/public','Q':'diagnostic_real/former_classifier_selection','V':'method_development'}[role])
    cfg['source_encoder'].update(model_checkpoint=str(BIO_PATH),model_checkpoint_sha256=BIO_SHA,
        model_revision=BIO_REV,source_code_commit=commit or 'installed release '+distribution.version+'; tree SHA bound',
        source_package_version=distribution.version,embedding_layer='projected_global_embedding, L2-normalized as official inference engine',
        preprocessing_sha256=sha(health_root/'image/data/transforms.py'),
        differentiable_preprocess='torchvision tensor bilinear antialias resize short512 / center448 / repeat grayscale; no ImageNet normalize',
        PCA_weighting='equal P patient; equal images within patient; weighted q95')
    cfg['recipient_primary'].update(checkpoint=str(DENSE_PATH),checkpoint_sha256=sha(DENSE_PATH),
        preprocess=str(DenseNet121_Weights.IMAGENET1K_V1.transforms()),
        PCA_weighting='equal P patient; equal images within patient; weighted q95')
    cfg['recipient_final_reserved']['runtime_status']='NOT_LOADED_OR_EVALUATED; deferred phase, not a W0/W1 null'
    cfg['synthesis'].update(layout='negative0:64/positive64:128; point subsets0:32,64:96; pairs32:64 with96:128',
        joint_pair_loss='0.5*(squared mean-u error + Frobenius second-U error); same gamma1 as relation',
        template_policy='64 distinct public patients, same template cell j/j+64, common across five arms',
        augmentation_padding='zeros; align_corners=False; same continuous affine and brightness within pair')
    cfg['runtime']={'hardware':{'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
        'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__},
        'profile_warmup':20,'profile_timed':30,'profile_scope':'full128 clean+augmented two-pass update, final resolution active',
        'main_time_cap_seconds':None,'main_time_cap_status':'NOT_NUMERICALLY_SPECIFIED_IN_CURRENT_AUTHORIZATION',
        'source_recipient_eval_after_all_banks_sealed':True,'overwrite':False}
    cfg['bindings']=bindings
    amendments={'status':'PROSPECTIVE_V3_AMENDMENT_BEFORE_NEW_EFFICACY', 'old_settings_sha256':sha(CODE/'relation_distillation/settings_v1.json'),
        'source':'user supplied v3 master + final W0/W1 instructions',
        'changes':['DINOv2 -> BioViL-T image source; RN18 receiver -> DenseNet121',
            'PCA16 norm-normalization -> P-only q95 bounded scale; recipient PCA128',
            '12 -> 15 banks with genuine endpoint joint-moment comparator',
            'bank seeds101/211/307 ->101/202/303; private derangement per bank seed',
            'ridge.01 ->.1; renderer/AdamW/progressive levels/augmentation per master',
            'C-D engineering threshold .01 -> strictly positive; AP nondecrease vs B',
            'raw noisy synthesis targets; PSD repair only direct noisy readout, DP disabled',
            '4 update prior profile -> supplied20warmup+30timed whole-bank profile'],
        'runtime_clarifications':['Official PIL uint8 vs tensor resize roundoff is measured; consistent tensor geometry defines all new source inputs',
            'Patient-uniform P PCA/q95 retained explicitly; no P labels or Q/V features select the projection',
            'All resolution levels active for timed profile to avoid underestimating late-step cost',
            'R_joint uses raw endpoint mean/second moment loss with coefficient1/2; no extra dimension averaging',
            'No local git checkout; new source is bound by hashes, not falsely assigned to remote HEAD'],
        'prior_results_or_settings_changed':False,'new_Q_or_V_pixels':0,'DP_expert_reserved_authorized':False}
    save(OUT/'contract_amendment.json',amendments)
    save(OUT/'w0_bindings.json',cfg)
    save(OUT/'run_matrix_non_dp.json',[{'arm':a,'seed':s,'images':128,'steps':500,'status':'NOT_STARTED'} for s in SEEDS for a in ARMS])
    print(json.dumps({'status':cfg['status'],'remote_main':upstream,'local_git':cfg['repo']['local_git_commit'],
                      'counts':EXPECTED,'source_code_commit':cfg['source_encoder']['source_code_commit'],'output':str(OUT)},ensure_ascii=True),flush=True)

if __name__=='__main__': bind()
