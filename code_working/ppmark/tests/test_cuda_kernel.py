import numpy as np
import pytest
from pathlib import Path

cp = pytest.importorskip("cupy")
pytest.importorskip("poseidon_py.poseidon_hash")

from ppmark_v03.config import GlobalConfig
from ppmark_v03.cuda import CUDAWatermarkKernel, DeviceConfig, get_watermark_kernel
from ppmark_v03.embedding import inject_watermark
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.tables import InverseCDFTable


def test_cuda_matches_cpu_embedding():
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:  # type: ignore[attr-defined]
            pytest.skip("No CUDA devices detected for CuPy")
    except Exception:
        pytest.skip("CuPy CUDA runtime unavailable")

    cfg = GlobalConfig.load(Path("config_small.json"))
    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=b"test-key-material",
        count=cfg.image.sample_count(),
    )
    bit_sequence = [0, 1] * 16
    binding = 123456789

    cpu_result = inject_watermark(
        binding=binding,
        bit_sequence=bit_sequence,
        config=cfg,
        inverse_cdf=inverse_cdf,
        sample_set=sample_set,
    )

    try:
        gpu_kernel = get_watermark_kernel(DeviceConfig(backend="cuda", uniform_backend="cpu"), cfg, inverse_cdf)
    except RuntimeError as exc:
        pytest.skip(f"CUDA backend unavailable: {exc}")
    assert isinstance(gpu_kernel, CUDAWatermarkKernel)
    try:
        gpu_result = gpu_kernel.embed(binding=binding, bit_sequence=bit_sequence, sample_set=sample_set)
    except RuntimeError as exc:
        pytest.skip(f"CUDA embed unavailable: {exc}")

    assert np.allclose(cpu_result.latent_noise, gpu_result.latent_noise, atol=1e-6)
    assert len(gpu_result.sample_trace.entries) == len(sample_set.indices)

    # Spot-check first few sample observations for parity.
    for cpu_obs, gpu_obs in zip(cpu_result.sample_trace.entries[:5], gpu_result.sample_trace.entries[:5]):
        assert cpu_obs.index == gpu_obs.index
        assert np.isclose(cpu_obs.uniform, gpu_obs.uniform, atol=1e-6)
        assert np.isclose(cpu_obs.z_expected, gpu_obs.z_expected, atol=1e-6)
        assert np.isclose(cpu_obs.z_observed, gpu_obs.z_observed, atol=1e-6)


def test_cuda_xorshift_path_runs():
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:  # type: ignore[attr-defined]
            pytest.skip("No CUDA devices detected for CuPy")
    except Exception:
        pytest.skip("CuPy CUDA runtime unavailable")

    cfg = GlobalConfig.load(Path("config_small.json"))
    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=b"test-key-material",
        count=cfg.image.sample_count(),
    )
    bit_sequence = [0, 1] * 16
    binding = 987654321

    try:
        gpu_kernel = get_watermark_kernel(DeviceConfig(backend="cuda", uniform_backend="gpu_xorshift"), cfg, inverse_cdf)
    except RuntimeError as exc:
        pytest.skip(f"CUDA backend unavailable: {exc}")
    try:
        gpu_result = gpu_kernel.embed(binding=binding, bit_sequence=bit_sequence, sample_set=sample_set)
    except RuntimeError as exc:
        pytest.skip(f"CUDA embed unavailable: {exc}")

    assert gpu_result.latent_noise.shape == (cfg.image.height, cfg.image.width)
    assert len(gpu_result.sample_trace.entries) == len(sample_set.indices)
