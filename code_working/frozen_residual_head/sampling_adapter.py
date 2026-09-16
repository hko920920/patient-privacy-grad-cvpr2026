"""Inference-only full64 output correction, matching the frozen training bank.

Captures the input to conv_out; does not mutate UNet weights or its output API.
The caller adds this wrapper's epsilon prediction before scheduler.step.
Only FP32, one conditional example and the declared epsilon schedule are supported.
"""
import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


class FrozenResidualAdapter:
    def __init__(self, unet, projection, weights, alphas, logsnr_min, logsnr_max, prediction_type):
        require(prediction_type == "epsilon", "Head was trained on epsilon residuals")
        require(tuple(projection.shape) == (320, 15), "Expected saved320x15 projection")
        require(tuple(weights.shape) == (64, 4), "Expected full64x4 weights")
        require(np.isfinite(projection).all() and np.isfinite(weights).all(), "Nonfinite weights")
        require(unet.conv_out.in_channels == 320 and unet.conv_out.out_channels == 4, "Wrong tap")
        require(next(unet.parameters()).dtype == torch.float32, "UNet must be FP32")
        self.unet = unet
        self.p = torch.from_numpy(np.array(projection, dtype=np.float32)).to(next(unet.parameters()).device)
        self.w = np.array(weights, dtype=np.float64, copy=True)
        self.alphas = np.array(alphas, dtype=np.float64, copy=True)
        require(self.alphas.shape == (1000,) and np.all((self.alphas>0)&(self.alphas<1)), "Wrong alpha schedule")
        self.lo, self.hi = float(logsnr_min), float(logsnr_max)
        self.handle = None
        self.captured = None
        self.capture_count = 0
        self.last = None

    def __enter__(self):
        require(self.handle is None, "Adapter already attached")
        def capture(module, inputs):
            require(len(inputs) == 1 and self.captured is None, "Unexpected or stale capture")
            self.captured = inputs[0].detach()
            self.capture_count += 1
        self.handle = self.unet.conv_out.register_forward_pre_hook(capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle, self.captured, self.last = None, None, None

    def predict(self, model_input, timestep, conditioning):
        require(self.handle is not None, "Use adapter as a context manager")
        require(not torch.is_grad_enabled(), "Inference mode required")
        require(tuple(model_input.shape) == (1,4,32,32), "Only fixed256px/batch1 supported")
        require(model_input.dtype == conditioning.dtype == torch.float32, "Inputs must be FP32")
        require(tuple(conditioning.shape) == (1,77,1024), "One conditional embedding required")
        require(self.captured is None, "Stale feature state")
        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        require(0 <= t < len(self.alphas), "Invalid timestep")
        count = self.capture_count
        try:
            base = self.unet(model_input, timestep, conditioning).sample
            require(self.capture_count == count+1 and self.captured is not None, "Tap not called exactly once")
            h = self.captured[0].permute(1,2,0).reshape(-1,320)
            small = torch.cat((torch.ones((len(h),1), device=h.device, dtype=torch.float32), h@self.p), dim=1)
            small = small.cpu().numpy().copy()
            alpha = self.alphas[t]
            u = 2*((np.log(alpha)-np.log1p(-alpha))-self.lo)/(self.hi-self.lo)-1
            require(-1-1e-12 <= u <= 1+1e-12, "Time outside training basis bounds")
            u = float(np.clip(u,-1,1))
            basis = np.array([1,u,(3*u*u-1)/2,(5*u*u*u-3*u)/2], dtype=np.float64)
            phi = (basis[None,:,None]*small.astype(np.float64)[:,None,:]).reshape(len(small),64)
            residual = phi@self.w
            require(np.isfinite(residual).all(), "Nonfinite correction")
            correction = torch.from_numpy(residual.astype(np.float32).reshape(32,32,4).transpose(2,0,1).copy())[None].to(base.device)
            corrected = base + correction
            require(bool(torch.isfinite(corrected).all()), "Nonfinite corrected prediction")
            self.last = {"features":small,"basis":basis,"residual":residual,
                         "base":base.detach().cpu().numpy().copy(),
                         "corrected":corrected.detach().cpu().numpy().copy()}
            return corrected
        finally:
            self.captured = None
