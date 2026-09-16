"""Pinned CDI 26-feature extraction through unchanged released class ASTs.

This is a conditional cached-latent backend adaptation, not the source
CompVis checkpoint or the CDI set-level statistical pipeline. Literal GM
clean-latent masking and NO final second noising are intentionally retained.
No entry point, training, or GPU work runs on import.
"""
import ast
import hashlib
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize as scipy_minimize

from .common import ROOT, digest

UPSTREAM = ROOT.parent / "CVPR 주제 탐색/research_2026-09-10/code_sources/X01"
COMMIT = "dcd62258b0b3fde05d52aaecfade3b5f4c09507a"
FILES = {
    "src/models/GeneralLatentDiffusionWrapper.py": "4ad16b23f1d59cdca2c19fb0f1aca790c731877acddb26ca1060f02cabe2b064",
    "src/models/CompVisLatentDiffusionWrapper.py": "7d61e59bc2325f5b1bd3a4661d5c1f8fab3f50d5c2060406b824dee1a87ed5ab",
    "src/attacks/features_extraction/cdi.py": "f6a1c303d24384158a41bf29c2305a43586bc9a2c45d4b32546b0f6f5901c62d",
    "src/attacks/scores_computation/denoising_loss.py": "1764d0bdf6905cc7c3c9cdd0b6d10584767377996452ca6883978380c8c43571",
    "src/attacks/scores_computation/pia.py": "b4d5fad785db83e2b4bc4a70de30fa4275ebae362dcbfddb1438897f718855a0",
    "src/attacks/scores_computation/secmi.py": "10f1d642639d30f9f14f15a9d7657d819413a9ce077b68ff7aa7bcc4aee74dc2",
    "src/attacks/features_extraction/denoising_loss.py": "a6cfe0bb5f6eb92e2a70aaebdc7931828cf3e2a2ecdad61d124b9acd09f44aef",
    "src/attacks/features_extraction/secmi.py": "4e4f36d04f0cfe7f6dcab4812d66be4ed65680fbbb92910927d6b5bce85de9c3",
    "src/attacks/features_extraction/pia.py": "75c029c75473996e9543fcdfbe679b65ea05dc610354860202dfe09eda77232f",
    "src/attacks/features_extraction/gradient_masking.py": "b03cb131e58df86a89f7affd2e2addf25abdbc2d679e539687b7e77483073ccf",
    "src/attacks/features_extraction/multiple_loss.py": "024e40d2dba69b00c3d97330bd3ff8f44c3011f64ee0cf3ad19f42ed8bce5a6d",
    "src/attacks/features_extraction/noise_optim.py": "c5f934854e41a47c7c0810b9419a497b2bf0305878198105089fd52ef41f3024",
    "conf/attack/cdi.yaml": "e9b15e7a6297dc3dd8124d8df9fdc27d529d4773f415d0bf7b7a75b81bad8e95",
    "conf/attack/denoising_loss.yaml": "692668e8bb1903aa4e1e9ffa78e8f787806bb529f29d59e1f450f2f7946bdb09",
    "conf/attack/secmi_stat.yaml": "aa188200f526bc4f2c5c4fac3e92f3541636d8af1f809de9eb882e95d25fb6ad",
    "conf/attack/pia.yaml": "23bfa53601a4ca90e707945d132175c7bd619aa73350b4ed6ab63adbbff719ff",
    "conf/attack/pian.yaml": "efa48cbbd8c7987a5432ddf2b22dfe313bf0836b7eedffdadf17df3b135c55c0",
    "conf/attack/gradient_masking.yaml": "9c94e2674b346d7b3329d083af0301e12c1cde7b3f98d46a6d142a8c2e511688",
    "conf/attack/multiple_loss.yaml": "50ad33315408d604c65877eb41599bf207fe8541bca36df5a628855a46274556",
    "conf/attack/noise_optim.yaml": "16f59c521e545551b28ae5fbf86fb2d02d9835a46e55cd31cd7ced6d4bd74466",
}
FEATURE_NAMES = (
    ["Denoising Loss", "SecMI$_{stat}$", "PIA", "PIAN"]
    + [f"Gradient Masking_{i}" for i in range(10)]
    + [f"Multiple Loss_{i}" for i in range(10)]
    + ["Noise Optimization_0", "Noise Optimization_1"]
)
MODULES = ("denoising_loss", "secmi", "pia", "gradient_masking", "multiple_loss", "noise_optim")
TIMESTEPS = tuple(range(0, 1000, 100))
CLASS_NAMES = {
    "denoising_loss": "DenoisingLossExtractor",
    "secmi": "SecMIExtractor",
    "pia": "PIAExtractor",
    "gradient_masking": "GradientMaskingExtractor",
    "multiple_loss": "MultipleLossExtractor",
    "noise_optim": "NoiseOptimExtractor",
}


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _cpu(value):
    return value.detach().cpu().clone().contiguous()


def noise_seed(master_seed, image_id, stream, module, draw):
    payload = f"cdi-v1|{int(master_seed)}|{image_id}|{stream}|{module}|{int(draw)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def source_bindings():
    for relative, expected in FILES.items():
        assert digest(UPSTREAM / relative) == expected, "CDI acquired source changed: " + relative
    return {
        "repository": "https://github.com/sprintml/copyrighted_data_identification",
        "commit": COMMIT, "source_root": str(UPSTREAM), "files_sha256": dict(FILES),
        "classes_reused": list(CLASS_NAMES.values()) + [
            "GeneralLatentDiffusionWrapper", "SecMIStat", "DenoisingLossComputer",
            "PIAComputer", "PIANComputer"],
        "execution": "unchanged complete class ASTs; no upstream main/import graph executed",
        "feature_names": list(FEATURE_NAMES),
        "source_fidelity": {
            "GM": "gradient at z_t; mask replacement uses clean z0; masked target epsilon-z0",
            "ML": "one noise draw shared across ten timesteps; configured repetitions unused",
            "NO": "L-BFGS-B maxiter5; objective z_t+delta; final wrapper applies noise again",
            "PIA_PIAN": "shared initial epsilon prediction; p5; source normalization unchanged",
            "scores": "positive source norms in 26D order; no membership-polarity reversal",
        },
        "backend_adaptation": "identity encode on cached scaled latent; generic conditioning; current scheduler epsilon UNet",
        "statistical_scope": "26D extraction only; no fitted CDI scorer/reference test/efficacy evaluation",
    }


class _TorchProxy:
    """Only randn_like is replaced; every original class AST remains unchanged."""
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn_like(self, other, **kwargs):
        assert not kwargs, "unexpected upstream randn_like signature"
        owner = self.owner
        module = owner.active_module
        draws = owner.raw["noise_draws"][module]
        seed = noise_seed(owner.master_seed, owner.image_id, owner.stream, module, len(draws))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(tuple(other.shape), generator=generator, device="cpu", dtype=other.dtype)
        draws.append({"seed": seed, "sha256": tensor_digest(value), "noise": value.clone(),
                      "draw": len(draws)})
        return value.to(other.device)


def _source_class(relative, name, proxy, minimize_bridge):
    path = UPSTREAM / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(nodes) == 1
    namespace = dict(torch=proxy, T=torch.Tensor, Tuple=Tuple, List=List, Optional=Optional,
                     sqrt=math.sqrt, pi=math.pi, os=os, np=np, random=random,
                     FeatureExtractor=object, ScoreComputer=object, DiffusionModel=object,
                     minimize=minimize_bridge)
    # Class bodies and method bodies are the acquired AST, without substitutions.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class CDIAdapter:
    def __init__(self, unet, scheduler, hidden, master_seed=260914):
        source_bindings()
        self.unet, self.scheduler = unet, scheduler
        self.device = next(unet.parameters()).device
        unet.to(dtype=torch.float32)
        if getattr(unet, "peft_config", None) and hasattr(unet, "enable_adapters"):
            unet.enable_adapters()
        unet.eval().requires_grad_(False)
        self.hidden = hidden.detach().clone().to(self.device, torch.float32)
        assert self.hidden.ndim == 3 and self.hidden.shape[0] == 1
        assert bool(torch.isfinite(self.hidden).all())
        self.master_seed = int(master_seed)
        self.prediction_type = str(scheduler.config.prediction_type)
        assert self.prediction_type == "epsilon", "current CDI kernel contract requires epsilon prediction"
        self.alphas = _cpu(scheduler.alphas_cumprod).to(torch.float32)
        assert self.alphas.ndim == 1 and self.alphas.numel() == 1000
        assert bool(torch.isfinite(self.alphas).all())
        assert bool(((self.alphas > 0) & (self.alphas <= 1)).all())
        assert all(p.dtype == torch.float32 for p in unet.parameters() if p.is_floating_point())
        assert all(p.grad is None and not p.requires_grad for p in unet.parameters())
        self.initial_versions = {n: p._version for n, p in unet.named_parameters()}
        self.forward = self.backward = self.prediction_calls = 0
        self.active_module = None
        self.raw = {}
        self._forward_handle = unet.register_forward_hook(self._count_forward)
        self.proxy = _TorchProxy(self)
        general = _source_class("src/models/GeneralLatentDiffusionWrapper.py",
                                "GeneralLatentDiffusionWrapper", self.proxy, self._minimize)
        self.wrapper = general(SimpleNamespace(device=self.device))
        # Keep General.encode and General.predict_noise_from_latent dispatch intact.
        self.wrapper._encode = self._encode
        self.wrapper.noise_latents = self._noise_latents
        self.wrapper._predict_noise_from_latent = self._predict
        self.wrapper.get_alpha_cumprod = self.get_alpha_cumprod
        policies = {
            "denoising_loss": dict(timestep=100, n_repetitions=5),
            "secmi": dict(t0=0, t=100, step=10),
            "pia": dict(name="pia", t=200, p=5),
            "gradient_masking": dict(timesteps=",".join(map(str, TIMESTEPS)), masking_ratio=.2),
            "multiple_loss": dict(timesteps=",".join(map(str, TIMESTEPS)), n_repetitions=5),
            "noise_optim": dict(timestep=100, max_iter=5),
        }
        self.extractors = {}
        for module, clsname in CLASS_NAMES.items():
            cls = _source_class(f"src/attacks/features_extraction/{module}.py",
                                clsname, self.proxy, self._minimize)
            extractor = cls()
            extractor.model, extractor.device = self.wrapper, self.device
            extractor.attack_cfg = SimpleNamespace(**policies[module])
            self.extractors[module] = extractor
        self.stat = _source_class("src/attacks/scores_computation/secmi.py",
                                 "SecMIStat", self.proxy, self._minimize)()
        self.dl_stat = _source_class("src/attacks/scores_computation/denoising_loss.py",
                                    "DenoisingLossComputer", self.proxy, self._minimize)()
        self.pia_stat = _source_class("src/attacks/scores_computation/pia.py",
                                     "PIAComputer", self.proxy, self._minimize)()
        self.pian_stat = _source_class("src/attacks/scores_computation/pia.py",
                                      "PIANComputer", self.proxy, self._minimize)()
        gm = self.extractors["gradient_masking"]
        for method, key in (("obtain_gradients", "gradients"), ("get_masks", "masks"),
                            ("noise_top_values", "mixed_latents")):
            original = getattr(gm, method)
            def capture(*args, _original=original, _key=key, **kwargs):
                value = _original(*args, **kwargs)
                self.raw["gm"][_key] = _cpu(value)
                return value
            setattr(gm, method, capture)

    def _count_forward(self, module, args, output):
        self.forward += 1

    def _encode(self, latent):
        assert tuple(latent.shape) == (1, 4, 32, 32)
        assert latent.dtype == torch.float32 and latent.device == self.device
        assert not latent.requires_grad and bool(torch.isfinite(latent).all())
        return latent

    def get_alpha_cumprod(self, timestep):
        return float(self.alphas[int(timestep)])

    def _noise_latents(self, latent, timestep, noise):
        alpha = self.get_alpha_cumprod(timestep)
        # Current epsilon q_sample convention; scalar alpha matches the source bridge.
        return math.sqrt(alpha) * latent + math.sqrt(1 - alpha) * noise

    def _predict(self, latent, classes, timestep):
        assert tuple(latent.shape) == (1, 4, 32, 32) and latent.dtype == torch.float32
        assert latent.device == self.device and classes.dtype == torch.float32
        assert classes.shape == self.hidden.shape
        assert self.active_module in MODULES
        t = torch.tensor([int(timestep)], device=self.device, dtype=torch.long)
        before = self.forward
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.unet(latent, t, encoder_hidden_states=classes).sample
        assert self.forward == before + 1, "unexpected top-level UNet call count"
        assert output.shape == latent.shape and output.dtype == torch.float32
        assert bool(torch.isfinite(output).all())
        self.prediction_calls += 1
        entry = {
            "timestep": int(timestep), "alpha": self.get_alpha_cumprod(timestep),
            "input": _cpu(latent), "raw_prediction": _cpu(output), "epsilon": _cpu(output),
            "grad_enabled": bool(torch.is_grad_enabled()),
            "input_requires_grad": bool(latent.requires_grad),
        }
        self.raw["predictions"][self.active_module].append(entry)
        if output.requires_grad:
            def prediction_backward(gradient):
                self.backward += 1
                entry["epsilon_gradient"] = _cpu(gradient)
            output.register_hook(prediction_backward)
        if latent.requires_grad:
            def input_backward(gradient):
                entry["input_gradient"] = _cpu(gradient)
            latent.register_hook(input_backward)
        return output

    def _minimize(self, fun, x0, args=(), **kwargs):
        assert self.active_module == "noise_optim"
        assert kwargs["method"] == "L-BFGS-B" and kwargs["jac"] is True
        assert kwargs["options"] == {"maxiter": 5}
        self.raw["no"]["initial_noised_latent"] = _cpu(args[1])
        trace = self.raw["no"]["objective_trace"]
        def recorded_objective(x, *inner_args):
            before_f, before_b = self.forward, self.backward
            loss, gradient = fun(x, *inner_args)
            assert self.forward - before_f == 1 and self.backward - before_b == 1
            assert math.isfinite(loss) and bool(torch.isfinite(gradient).all())
            trace.append({"x": torch.from_numpy(np.array(x, copy=True)),
                          "gradient": _cpu(gradient), "loss": float(loss)})
            # Preserve the source's CPU torch gradient return to SciPy.
            return loss, gradient
        result = scipy_minimize(recorded_objective, x0, args=args, **kwargs)
        self.raw["no"]["delta"] = torch.from_numpy(np.array(result.x, copy=True)).reshape(1, 4, 32, 32)
        summary = {
            "nfev": int(result.nfev), "njev": int(result.njev), "nit": int(result.nit),
            "status": int(result.status), "success": bool(result.success),
            "message": str(result.message), "fun": float(result.fun),
        }
        assert summary["nfev"] == summary["njev"] and summary["nfev"] >= len(trace) > 0
        summary["objective_evaluations"] = len(trace)
        summary["memoized_evaluation_requests"] = summary["nfev"] - len(trace)
        assert math.isfinite(summary["fun"]) and bool(torch.isfinite(self.raw["no"]["delta"]).all())
        self.raw["no"]["optimizer"] = {**summary, "jac": torch.from_numpy(np.array(result.jac, copy=True))}
        return result

    def assert_weights_unchanged(self):
        named = dict(self.unet.named_parameters())
        assert set(named) == set(self.initial_versions)
        assert all(p._version == self.initial_versions[n] and p.grad is None and not p.requires_grad
                   for n, p in named.items()), "target parameter or gradient state changed"
        assert not self.unet.training
        return True

    def score(self, latent, image_id, stream="primary"):
        assert not torch.is_inference_mode_enabled(), "GM/NO require input autograd"
        started = time.perf_counter()
        self.assert_weights_unchanged()
        before_f, before_b, before_pred = self.forward, self.backward, self.prediction_calls
        self.image_id, self.stream = str(image_id), str(stream)
        latent = latent.detach().clone().to(self.device, torch.float32)
        self._encode(latent)
        self.raw = {
            "schema": "cdi-literal-26-raw/v1", "image_id": self.image_id, "stream": self.stream,
            "latent": _cpu(latent), "alphas": self.alphas.clone(),
            "hidden_sha256": tensor_digest(self.hidden),
            "noise_draws": {m: [] for m in MODULES},
            "predictions": {m: [] for m in MODULES},
            "module_outputs": {}, "gm": {}, "secmi": {}, "no": {"objective_trace": []},
        }
        modules = {}
        for module in MODULES:
            self.active_module = module
            block_started = time.perf_counter()
            f, b = self.forward, self.backward
            with torch.enable_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                output = self.extractors[module].process_batch((latent, self.hidden))
            saved = _cpu(output)
            assert bool(torch.isfinite(saved).all()), "nonfinite source feature: " + module
            self.raw["module_outputs"][module] = saved
            modules[module] = {
                "forward": self.forward - f, "backward": self.backward - b,
                "seconds": time.perf_counter() - block_started,
                "output_shape": list(saved.shape), "output_dtype": str(saved.dtype),
                "noise_draw_count": len(self.raw["noise_draws"][module]),
                "noise_seeds": [r["seed"] for r in self.raw["noise_draws"][module]],
                "noise_sha256": [r["sha256"] for r in self.raw["noise_draws"][module]],
            }
            assert len(self.raw["predictions"][module]) == modules[module]["forward"]
        self.active_module = None
        out = self.raw["module_outputs"]
        self.raw["secmi"] = {"z_det": out["secmi"][:, 0].clone(), "z_recon": out["secmi"][:, 1].clone()}
        first = [
            float(self.dl_stat.process_data(out["denoising_loss"], out["denoising_loss"])[0][0]),
            float(self.stat.compute_score(out["secmi"])[0]),
            float(self.pia_stat.process_data(out["pia"], out["pia"])[0][0]),
            float(self.pian_stat.process_data(out["pia"], out["pia"])[0][0]),
        ]
        features = first + out["gradient_masking"].reshape(-1).tolist() + out["multiple_loss"].reshape(-1).tolist() + out["noise_optim"].reshape(-1).tolist()
        assert len(features) == 26 and all(math.isfinite(v) and v >= 0 for v in features)
        opt = self.raw["no"]["optimizer"]
        q = opt["objective_evaluations"]
        expected = {"denoising_loss": (5, 0, 5), "secmi": (12, 0, 0), "pia": (3, 0, 0),
                    "gradient_masking": (20, 10, 1), "multiple_loss": (10, 0, 1),
                    "noise_optim": (q + 1, q, 1)}
        for module, (f, b, n) in expected.items():
            assert (modules[module]["forward"], modules[module]["backward"], modules[module]["noise_draw_count"]) == (f, b, n)
        gm_grad = self.raw["gm"]["gradients"]
        assert bool(torch.isfinite(gm_grad).all())
        assert bool((gm_grad.reshape(1, 10, -1).norm(dim=-1) > 0).all()), "zero GM input-gradient block"
        modules["gradient_masking"]["nonzero_input_gradient_timesteps"] = 10
        modules["noise_optim"]["optimizer"] = {k: v for k, v in opt.items() if k != "jac"}
        modules["noise_optim"]["objective_trace"] = [
            {"evaluation": i, "loss": r["loss"], "gradient_l2": float(r["gradient"].norm()),
             "gradient_max_abs": float(r["gradient"].abs().max())}
            for i, r in enumerate(self.raw["no"]["objective_trace"])
        ]
        forward, backward = self.forward - before_f, self.backward - before_b
        assert forward == self.prediction_calls - before_pred == 51 + q
        assert backward == 10 + q
        self.assert_weights_unchanged()
        return {
            "image_id": self.image_id, "stream": self.stream,
            "features": features, "feature_names": list(FEATURE_NAMES), "modules": modules,
            "forward": forward, "backward": backward, "seconds": time.perf_counter() - started,
            "prediction_type": self.prediction_type, "precision": "float32 UNet; literal scipy float64 perturbation",
            "batch_size": 1, "weights_unchanged": True, "raw": self.raw,
            "counter_method": "top-level UNet forward hook; predicted-epsilon autograd backward hook",
            "unet_training": bool(self.unet.training),
            "gradient_checkpointing_enabled": bool(getattr(self.unet, "is_gradient_checkpointing", False)),
        }


def cpu_conformance():
    """Synthetic CPU arithmetic/gradient bridge check; no actual model claim."""
    class AffineUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, x, t, encoder_hidden_states=None):
            return SimpleNamespace(sample=.13 * x + .017 * (int(t[0]) + 1) / 100 + 0 * self.anchor)

    alphas = torch.cumprod(1 - torch.linspace(.0001, .02, 1000, dtype=torch.float32), dim=0)
    scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon"), alphas_cumprod=alphas)
    latent = torch.linspace(-.5, .5, 4096, dtype=torch.float32).reshape(1, 4, 32, 32)
    hidden = torch.zeros(1, 77, 8, dtype=torch.float32)
    adapter = CDIAdapter(AffineUNet(), scheduler, hidden)
    started = time.perf_counter()
    result = adapter.score(latent, "synthetic-affine", stream="conformance")
    raw = result["raw"]
    eps_error = 0.0
    grad_error = 0.0
    for module, entries in raw["predictions"].items():
        for entry in entries:
            independent_epsilon = .13 * entry["input"] + .017 * (entry["timestep"] + 1) / 100
            eps_error = max(eps_error, float((entry["epsilon"] - independent_epsilon).abs().max()))
            if "input_gradient" in entry:
                grad_error = max(grad_error, float((entry["input_gradient"] - .13 * entry["epsilon_gradient"]).abs().max()))
    assert eps_error == 0.0 and grad_error <= 1e-7
    # No extra target call: independently rebuild source scalars from saved arrays.
    def norm(x, p=2):
        return float(torch.linalg.vector_norm(x.double().reshape(-1), ord=p))
    expected = []
    dl = raw["predictions"]["denoising_loss"]
    dl_noise = raw["noise_draws"]["denoising_loss"]
    expected.append(math.fsum(norm(e["epsilon"] - n["noise"]) for e, n in zip(dl, dl_noise)) / 5)
    expected.append(norm(raw["secmi"]["z_det"] - raw["secmi"]["z_recon"]))
    pia = raw["predictions"]["pia"]
    e0 = pia[0]["epsilon"]
    normalized = e0.numel() * math.sqrt(math.pi / 2) * e0 / e0.norm(p=1, dim=(1, 2, 3), keepdim=True)
    expected += [norm(e0 - pia[1]["epsilon"], 5), norm(normalized - pia[2]["epsilon"], 5)]
    gm_noise = raw["noise_draws"]["gradient_masking"][0]["noise"]
    for i, entry in enumerate(raw["predictions"]["gradient_masking"][10:]):
        mask = raw["gm"]["masks"][:, i]
        expected.append(norm(((gm_noise - latent) - entry["epsilon"])[mask]))
        mixed = latent.clone()
        mixed[mask] = gm_noise[mask]
        assert torch.equal(mixed, entry["input"])
        actual_mask = torch.zeros_like(latent, dtype=torch.bool).reshape(1, -1)
        indices = raw["gm"]["gradients"][:, i].abs().reshape(1, -1).topk(int(4096 * .2), dim=1).indices
        actual_mask.scatter_(1, indices, True)
        assert torch.equal(actual_mask.reshape_as(mask), mask)
    ml_noise = raw["noise_draws"]["multiple_loss"][0]["noise"]
    expected += [norm(entry["epsilon"] - ml_noise) for entry in raw["predictions"]["multiple_loss"]]
    no_noise = raw["noise_draws"]["noise_optim"][0]["noise"]
    no = raw["no"]
    final = raw["predictions"]["noise_optim"][-1]
    alpha = float(alphas[100])
    final_before_noising = (no["initial_noised_latent"] + no["delta"]).float()
    second_noised = math.sqrt(alpha) * final_before_noising + math.sqrt(1 - alpha) * no_noise
    assert torch.equal(final["input"], second_noised)
    expected += [norm(final["epsilon"] - no_noise), norm(no["delta"])]
    feature_error = max(abs(a - b) for a, b in zip(result["features"], expected))
    feature_relative_error = max(abs(a - b) / max(abs(b), 1e-10) for a, b in zip(result["features"], expected))
    assert feature_relative_error < 3e-6
    for entry in no["objective_trace"]:
        assert torch.isfinite(entry["gradient"]).all()
    assert adapter.assert_weights_unchanged()
    return {
        "status": "PASS_SYNTHETIC_CPU_CONFORMANCE", "GPU_executions": 0,
        "feature_count": 26, "forward": result["forward"], "backward": result["backward"],
        "epsilon_max_abs_error": eps_error, "input_gradient_max_abs_error": grad_error,
        "feature_float64_max_abs_error": feature_error,
        "feature_float64_max_relative_error": feature_relative_error,
        "GM_literal_clean_mask": True, "NO_literal_final_double_noise": True,
        "optimizer": {k: v for k, v in no["optimizer"].items() if k != "jac"},
        "seconds": time.perf_counter() - started, "source_bindings": source_bindings(),
        "scope": "unchanged AST execution with synthetic affine epsilon, CPU raw arithmetic and gradients; not actual-target or efficacy validation",
    }
