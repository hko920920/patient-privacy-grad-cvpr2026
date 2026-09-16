import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("poseidon_py.poseidon_hash")
pytest.importorskip("reedsolo")

from ppmark_v03.sync import DDIMSchedule, ddim_encode, make_ddim_schedule


def test_ddim_encode_identity_path():
    x = np.ones((8, 8), dtype=np.float32)
    sched = make_ddim_schedule(num_steps=3, beta_start=1e-5, beta_end=1e-5)
    out = ddim_encode(x, schedule=sched, noise_scale=0.0, normalize=False)
    # With zero noise and tiny betas, output should stay close to input.
    assert np.allclose(out, x, atol=1e-3)


def test_ddim_encode_custom_model_fn():
    x = np.zeros((4, 4), dtype=np.float32)
    sched = DDIMSchedule(timesteps=np.array([0, 1], dtype=np.int32), alpha_cumprod=np.array([0.9, 0.8], dtype=np.float32))

    def model_fn(x_t, t_idx):
        return np.ones_like(x_t) * (t_idx + 1)

    out = ddim_encode(x, model_fn=model_fn, schedule=sched, normalize=False)
    # Deterministic, shape-preserving path.
    assert out.shape == x.shape
    assert np.isfinite(out).all()
