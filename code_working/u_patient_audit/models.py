import gc, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from .common import *
from dp_protocol.calibrate_xray_public_clip_norms import disable_broken_optional_onnx
disable_broken_optional_onnx()
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from huggingface_hub import snapshot_download
from peft import LoraConfig
from peft.utils.other import cast_mixed_precision_params

def setup():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def snapshot():
    return Path(snapshot_download(repo_id=MODEL_ID,revision=MODEL_REVISION,local_files_only=True))

def load_unet(adapter=None,training=False):
    torch.manual_seed(260914);torch.cuda.manual_seed_all(260914)
    unet=UNet2DConditionModel.from_pretrained(snapshot()/"unet",torch_dtype=torch.float16,
        variant="fp16",use_safetensors=True,local_files_only=True)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=8,lora_alpha=8,init_lora_weights="gaussian",
        target_modules=["to_q","to_k","to_v","to_out.0"]))
    cast_mixed_precision_params(unet,dtype=torch.float16)
    assert sum(p.numel() for p in unet.parameters() if p.requires_grad)==1659904
    if adapter is not None:
        state=torch.load(adapter,map_location="cpu",weights_only=True)
        state=state.get("adapter",state)
        expected={n for n,p in unet.named_parameters() if p.requires_grad}
        assert set(state)==expected
        result=unet.load_state_dict(state,strict=False)
        assert not result.unexpected_keys
    unet.enable_gradient_checkpointing()
    unet.train(training).to("cuda")
    if not training:unet.requires_grad_(False)
    return unet

def scheduler():
    s=DDPMScheduler.from_pretrained(snapshot()/"scheduler",local_files_only=True)
    assert s.config.num_train_timesteps==1000
    assert s.config.prediction_type in ("epsilon","v_prediction")
    return s

def diffusion_input(z,t,noise,sched):
    noisy=sched.add_noise(z,noise,t)
    target=noise if sched.config.prediction_type=="epsilon" else sched.get_velocity(z,noise,t)
    return noisy,target

def adapter_state(unet):
    return {n:p.detach().cpu().clone() for n,p in unet.named_parameters() if ".lora_" in n}

def release(*objects):
    gc.collect();torch.cuda.empty_cache()

def load_cache():
    metadata=json.loads((RUN/"cache"/"summary.json").read_text())
    path=RUN/"cache"/"cache.pt"
    assert digest(path)==metadata["cache_sha256"]
    assert digest(RUN/"cohort"/"lock.json")==metadata["cohort_lock_sha256"]
    return torch.load(path,map_location="cpu",weights_only=True)

