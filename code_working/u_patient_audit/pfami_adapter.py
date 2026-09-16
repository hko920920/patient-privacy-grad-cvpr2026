"""Source-bound PFAMI-stat measurement; no optimization or target training.

The authors' DDPM loss and feature preparation ASTs remain unchanged. The
wrapper supplies conditional epsilon predictions and explicitly seeded noise.
"""
import ast
import hashlib
import math
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT, digest

UPSTREAM = ROOT.parent / "CVPR 주제 탐색/research_2026-09-10/code_sources/X11"
SOURCE = UPSTREAM / "attack/attack_model_PFAMI.py"
SOURCE_SHA256 = "3bf2f4f28fbcfd3c4b4291a932e58a68332da3920c8fbc49fcde6e5414d83117"
COMMIT = "f80df339e11ff49db211a455c0a455d630cf34a7"


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def source_bindings():
    assert digest(SOURCE) == SOURCE_SHA256, "acquired PFAMI source changed"
    return dict(repository="https://github.com/wjfu99/MIA-Gen", commit=COMMIT,
        source_path=str(SOURCE), source_sha256=SOURCE_SHA256,
        methods_reused=["AttackModel.ddpm_loss", "AttackModel.feat_prepare"],
        source_member_label=0, source_eval_score="negative mean relative fluctuation",
        local_member_label=1, local_score="positive mean relative fluctuation",
        adaptation="conditional latent diffusion, fixed crop and explicit CRN policy; no NN/calibration")


class _TorchProxy:
    """Replace only the source's single noise draw with the recorded draw."""
    def __init__(self):
        self.noise = None
        self.draws = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn(self, shape, device=None):
        assert self.noise is not None and tuple(shape) == tuple(self.noise.shape)
        assert torch.device(device) == self.noise.device
        self.draws += 1
        return self.noise.clone()


def _source_methods():
    source_bindings()
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AttackModel")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"ddpm_loss", "feat_prepare"}]
    assert len(methods) == 2
    module = ast.Module(body=[ast.ClassDef(name="SourceMethods", bases=[], keywords=[],
                            body=methods, decorator_list=[])], type_ignores=[])
    proxy = _TorchProxy()
    # The released code uses np.int; supply its historical meaning without
    # modifying NumPy or the original method's source.
    np_proxy = SimpleNamespace(expand_dims=np.expand_dims, concatenate=np.concatenate,
                              zeros=np.zeros, ones=np.ones, isnan=np.isnan, int=int)
    namespace = dict(torch=proxy, np=np_proxy, F=F,
        utils=SimpleNamespace(tensor_to_ndarray=lambda *xs: tuple(x.detach().cpu().numpy() for x in xs)))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["SourceMethods"](), proxy


def source_statistic(original, cropped):
    """Evaluate the original stat branch using its selected perturbation slot 5."""
    original, cropped = np.asarray(original, dtype=np.float64), np.asarray(cropped, dtype=np.float64)
    assert original.ndim == 1 and original.shape == cropped.shape
    assert np.isfinite(original).all() and np.isfinite(cropped).all() and (original > 0).all()
    changes = np.zeros((1, len(original), 6), dtype=np.float64)
    changes[0, :, 5] = cropped - original
    data = SimpleNamespace(ori_losses=original[None, :], var_losses=changes)
    info = SimpleNamespace(mem_feat=data, nonmem_feat=data)
    methods, _ = _source_methods()
    values, labels = methods.feat_prepare(info, dict(calibration=False, attack_kind="stat"))
    assert labels.tolist() == [0, 1] and values[0] == values[1]
    return float(values[0])


def noise_seed(master_seed, image_id, view, timestep, stream):
    payload = f"pfami-v1|{int(master_seed)}|{image_id}|{view}|{int(timestep)}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class PFAMIAdapter:
    def __init__(self, unet, scheduler, hidden, timesteps, master_seed):
        self.unet, self.scheduler = unet, scheduler
        self.device = next(unet.parameters()).device
        self.unet.to(dtype=torch.float32).eval().requires_grad_(False)
        if hasattr(unet, "enable_adapters"):
            unet.enable_adapters()
        self.unet.requires_grad_(False)
        self.hidden = hidden.detach().clone().to(self.device, torch.float32)
        self.timesteps = tuple(map(int, timesteps))
        assert self.timesteps == tuple(range(0, 500, 50))
        self.master_seed = int(master_seed)
        self.prediction_type = str(scheduler.config.prediction_type)
        assert self.prediction_type == "epsilon", "the frozen current contract is epsilon"
        self.forward = self.backward = 0
        self.cuda_ms = 0.0
        self.versions = {n: p._version for n, p in unet.named_parameters()}
        self.methods, self.noise_proxy = _source_methods()
        self.pipeline = SimpleNamespace(unet=self, scheduler=scheduler)

    def eval(self):
        self.unet.eval()
        return self

    def __call__(self, noisy, timesteps):
        assert not torch.is_grad_enabled() and noisy.dtype == torch.float32 and noisy.shape[0] == 1
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        began = time.perf_counter()
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.unet(noisy, timesteps.long(), encoder_hidden_states=self.hidden).sample
        if self.device.type == "cuda":
            end.record(); end.synchronize()
            self.last_cuda_ms = float(start.elapsed_time(end))
        else:
            self.last_cuda_ms = None
        self.last_wall_seconds = time.perf_counter() - began
        assert output.dtype == torch.float32 and output.shape == noisy.shape
        assert bool(torch.isfinite(output).all())
        self.last_prediction_norm = float(output.norm())
        self.last_residual = (output - self.noise_proxy.noise).detach().cpu().clone()
        self.forward += 1
        self.cuda_ms += self.last_cuda_ms or 0.0
        return SimpleNamespace(sample=output)

    def assert_weights_unchanged(self):
        params = dict(self.unet.named_parameters())
        assert set(params) == set(self.versions)
        assert all(p._version == self.versions[n] and p.grad is None and not p.requires_grad
                   for n, p in params.items())
        return True

    def score(self, original_latent, crop_latent, image_id, stream="target"):
        assert stream in {"target", "base"}
        began, before = time.perf_counter(), self.forward
        latents = {"original": original_latent, "crop": crop_latent}
        cells, residuals = [], []
        with torch.no_grad():
            for timestep in self.timesteps:
                cell = dict(timestep=timestep)
                for view, source in latents.items():
                    latent = source.detach().clone().to(self.device, torch.float32)
                    assert latent.shape == (1, 4, 32, 32) and bool(torch.isfinite(latent).all())
                    seed = noise_seed(self.master_seed, image_id, view, timestep, stream)
                    noise = torch.randn(latent.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                                        dtype=torch.float32).to(self.device)
                    self.noise_proxy.noise, self.noise_proxy.draws = noise, 0
                    loss = self.methods.ddpm_loss(self.pipeline, latent, timestep)
                    assert self.noise_proxy.draws == 1 and np.asarray(loss).shape == (1,)
                    value = float(loss[0])
                    assert math.isfinite(value) and value > 0, "zero/nonfinite DDPM loss; no silent denominator repair"
                    cell[view + "_loss"] = value
                    cell[view + "_noise_seed"] = seed
                    cell[view + "_noise_sha256"] = tensor_digest(noise)
                    cell[view + "_noise_l2"] = float(noise.norm())
                    cell[view + "_prediction_l2"] = self.last_prediction_norm
                    cell[view + "_cuda_ms"] = self.last_cuda_ms
                    cell[view + "_wall_seconds"] = self.last_wall_seconds
                    residuals.append(self.last_residual[0])
                cell["relative"] = (cell["crop_loss"] - cell["original_loss"]) / cell["original_loss"]
                cell["finite"] = math.isfinite(cell["relative"])
                assert cell["finite"]
                cells.append(cell)
        score = source_statistic([r["original_loss"] for r in cells], [r["crop_loss"] for r in cells])
        independent = math.fsum(r["relative"] for r in cells) / len(cells)
        assert math.isclose(score, independent, rel_tol=1e-12, abs_tol=1e-12)
        self.assert_weights_unchanged()
        assert self.forward - before == 20
        return dict(image_id=image_id, stream=stream, cells=cells, score=score,
                    residuals=torch.stack(residuals).reshape(10, 2, 4, 32, 32),
                    forward=20, backward=0, finite=True, seconds=time.perf_counter()-began,
                    cuda_ms=math.fsum((r[v+"_cuda_ms"] or 0) for r in cells for v in ("original", "crop")))


def cpu_conformance():
    original = [0.2 + i / 50 for i in range(10)]
    crop = [v * (1.0 + (i-3) / 10) for i, v in enumerate(original)]
    observed = source_statistic(original, crop)
    expected = math.fsum((b-a)/a for a, b in zip(original, crop))/10
    assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
    seeds = [noise_seed(260914, "synthetic", view, t, stream)
             for view in ("original", "crop") for t in range(0,500,50) for stream in ("target","base")]
    assert len(set(seeds)) == len(seeds)
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, noisy, t, encoder_hidden_states=None):
            return SimpleNamespace(sample=noisy * .25 + self.anchor)
    sched = SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon"),
        add_noise=lambda z,n,t: .9*z+.1*n)
    probe = PFAMIAdapter(Oracle(),sched,torch.zeros(1,77,1024),range(0,500,50),260914)
    z = torch.linspace(-1,1,4096).reshape(1,4,32,32)
    result = probe.score(z,z*.8,"oracle")
    residual = result["residuals"]
    assert residual.shape == (10,2,4,32,32) and residual.dtype == torch.float32
    mse = residual.double().square().mean(dim=(2,3,4))
    errors = [abs(float(mse[i,j])-cell[v+"_loss"])
              for i,cell in enumerate(result["cells"]) for j,v in enumerate(("original","crop"))]
    assert max(errors) <= 16*torch.finfo(torch.float32).eps*float(mse.max())
    assert result["forward"] == 20 and probe.assert_weights_unchanged()
    return dict(status="PASS_CPU_SOURCE_STATISTIC_AND_SEED_CONFORMANCE", source=source_bindings(),
                source_score=observed, independent_score=expected, unique_seed_cases=len(seeds),
                ddpm_oracle_forward=20, residual_mse_max_abs=max(errors),
                GPU_executions=0, scope="source arithmetic and seed contract; not attack efficacy")
