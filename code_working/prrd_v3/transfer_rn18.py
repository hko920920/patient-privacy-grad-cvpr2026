"""Inject only a provider into the frozen corrected training function."""
import contextlib,copy,json,time,types
import numpy as np
from .contracts import OUT,save,require,install_legacy_guard

def provider(source,arm,run_seed,step):
    from downstream_utility import data_v2
    from downstream_utility.common import seed
    if arm=='R1': return data_v2.batch_records(source,arm,run_seed,step)
    require(arm.startswith('PRRD/'),'Unknown PRRD classifier arm')
    shared=data_v2.draw(source['real/public'],16,seed(run_seed,step,'public-shared'))
    other=data_v2.draw(source['synthetic/prrd'],16,seed(run_seed,step,'synthetic-extra'))
    return shared+other

def training_kernel():
    """Private globals copy keeps original module and frozen code object intact."""
    install_legacy_guard()
    from downstream_utility import train_v2
    g=dict(train_v2.train_steps.__globals__);g['batch_records']=provider
    return types.FunctionType(train_v2.train_steps.__code__,g,'prrd_train_steps',train_v2.train_steps.__defaults__)

def public_replay(group,access):
    from downstream_utility import data_v2,train_v2
    from downstream_utility.common import tensor_sha
    artifact=OUT/'classifier_public_replay.json'
    if artifact.exists(): return
    # All P metadata enters the original sampler. Decode only selected public pixels.
    records=[dict(r,index=i,array_key='P',namespace='real/public',role='public',
                  source_file_sha256=r['sha256']) for i,r in enumerate(group)]
    pool=data_v2.pools(records);source={'real/public':pool}
    selected=data_v2.batch_records(source,'R1',11,0)
    arrays={'P':{}}
    for r in selected:
        i=int(r['index'])
        if i not in arrays['P']: arrays['P'][i]=data_v2.letterbox(access.image(group[i]))
        r['array_sha256']=tensor_sha(arrays['P'][i])
    # data_v2.pools stores the same row dicts; only actually drawn records need pixels.
    init=train_v2.initial_state(11)
    a,ta=train_v2.train_steps(init,'R1',source,arrays,1,run_seed=11)
    b,tb=training_kernel()(init,'R1',source,arrays,1,run_seed=11)
    import torch
    require(all(torch.equal(a[k],b[k]) for k in a),'Corrected classifier update changed')
    for key in ('input_sha256','image_ids','labels','logits','loss','array_keys','array_indices'):
        require(ta['trace'][0][key]==tb['trace'][0][key],'Classifier provider mismatch: '+key)
    save(artifact,{'status':'PASS_CORRECTED_PUBLIC_PROVIDER_EXACT_ONE_STEP','updates':2,
                  'input_sha256':ta['trace'][0]['input_sha256'],
                  'initial_state_sha256':ta['initial_state_sha256'],'final_state_sha256':ta['final_state_sha256'],
                  'original_seconds':ta['seconds'],'provider_seconds':tb['seconds'],
                  'changed_trainable_tensors':len(ta['changed_trainable_parameters']),
                  'legacy_modules_loaded':False,'new_synthetic_or_private_inputs':0})

def disabled_dp():
    raise RuntimeError('DP is a separate authorized contract; no production sampler in W0/W1/W2')
