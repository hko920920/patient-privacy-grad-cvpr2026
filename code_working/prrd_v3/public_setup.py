"""Actual P-only feature extraction and immutable public projection artifacts."""
import json,time
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from .contracts import *
from .datasets import manifests,PixelAccess
from .encoders import Encoder,extract_public,fit_projection,state_hash,preprocess,pil_tensor

def run():
    setup(); cfg=read(OUT/'w0_bindings.json'); check_bindings(cfg)
    groups=manifests(); access=PixelAccess(groups,('P',)).install()
    results={}; started=time.perf_counter()
    for kind,dim in (('source',16),('recipient',128)):
        artifact=OUT/(kind+'_P_projection.npz'); receipt=OUT/(kind+'_P_projection.json')
        if receipt.exists():
            record=read(receipt); require(sha(artifact)==record['projection']['sha256'],'Resume projection SHA')
            require(sha(OUT/(kind+'_P_features.npz'))==record['feature_sha256'],'Resume feature SHA')
            results[kind]=record; continue
        require(not artifact.exists(),'Partial public artifact requires audited recovery')
        torch.manual_seed(918); encoder=Encoder(kind)
        before=state_hash(encoder.model)
        # A real PUBLIC input, tested before extracting the public PCA.
        image=access.image(groups['P'][0]); tensor=pil_tensor(image)
        if kind=='source':
            from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
            official=create_chest_xray_transform_for_inference(512,448)(image)
        else:
            from torchvision.models import DenseNet121_Weights
            official=DenseNet121_Weights.IMAGENET1K_V1.transforms()(image.convert('RGB'))
        actual=preprocess(tensor,kind)
        pixel_error=float((actual-official).abs().max())
        # Quantization to uint8 after PIL resize differs by <= one gray level;
        # receiver normalization scales that difference by at most 1/.224.
        tolerance=1/255*(1 if kind=='source' else 1/.224)+1e-5
        require(pixel_error<=tolerance,'Tensor/PIL resize difference beyond rounding tolerance')
        with torch.no_grad():
            a=encoder.raw(actual[None].cuda()); b=encoder.raw(official[None].cuda())
        feature_error=float((a-b).abs().max())
        # This is reported, not used to choose an encoder/recipe.
        input_gradient=None
        if kind=='source':
            x=TF.resize(tensor,[224,224],antialias=True)[None].cuda().requires_grad_(True)
            value=encoder.raw(preprocess(x,'source'))[0,0]
            gradient,=torch.autograd.grad(value,x)
            require(torch.isfinite(gradient).all() and gradient.abs().max()>0,'No finite synthetic-input gradient')
            input_gradient={'max_abs':float(gradient.abs().max()),'norm':float(gradient.norm())}
            del x,gradient,value
        features,seconds=extract_public(encoder,groups['P'],access,microbatch=4)
        feature_path=OUT/(kind+'_P_features.npz')
        require(not feature_path.exists(),'Feature cache overwrite prohibited')
        np.savez(feature_path,features=features,image_ids=np.array([r['image_id'] for r in groups['P']]))
        projection=fit_projection(features,groups['P'],dim,kind,artifact)
        require(state_hash(encoder.model)==before,'Encoder parameters/buffers mutated')
        require(not any(m.training for m in encoder.model.modules()),'A module left eval mode')
        record={'projection':projection,'feature_sha256':sha(feature_path),'seconds':seconds,
                'public_images':len(features),'encoder_state_sha256':before,'model_unchanged':True,
                'PIL_vs_tensor':{'maximum_preprocess_difference':pixel_error,'tolerance':tolerance,
                                 'maximum_raw_feature_difference':feature_error,'exact_equality_claimed':False},
                'input_gradient':input_gradient,'private_or_development_pixels':0,
                'recipient_not_used_for_synthesis_or_selection':True}
        save(receipt,record); results[kind]=record
        del encoder; torch.cuda.empty_cache()
    save(OUT/'public_setup.json',{'status':'PASS_P_ONLY_PROJECTIONS_AND_ENCODER_BINDING',
         'results':results,'access':access.report(),'seconds':time.perf_counter()-started})
    print(json.dumps({'status':'PASS_P_ONLY_PROJECTIONS_AND_ENCODER_BINDING','seconds':time.perf_counter()-started}),flush=True)

if __name__=='__main__': run()
