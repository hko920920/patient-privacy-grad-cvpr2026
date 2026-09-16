import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("poseidon_py.poseidon_hash")
pytest.importorskip("reedsolo")

from ppmark_v03.config import GlobalConfig
from ppmark_v03.embedding import inject_watermark
from ppmark_v03.payload import build_payload_pallas
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.sync import extract_watermark, recover_bits
from ppmark_v03.tables import InverseCDFTable


def _prepare_embedding():
    cfg = GlobalConfig.load(Path("config_small.json"))
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    payload = build_payload_pallas(
        prompt="sync-test",
        seed=b"\x11" * 32,
        secret=b"\x22" * 32,
        codec=codec,
    )
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=payload.codeword,
        count=cfg.image.sample_count(),
    )
    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    embedding = inject_watermark(
        binding=payload.binding,
        bit_sequence=payload.bits,
        config=cfg,
        inverse_cdf=inverse_cdf,
        sample_set=sample_set,
    )
    return cfg, codec, payload, inverse_cdf, embedding


def test_recover_bits_round_trip():
    cfg, codec, payload, inverse_cdf, embedding = _prepare_embedding()
    result = recover_bits(
        aligned_image=embedding.latent_noise,
        binding=payload.binding,
        config=cfg,
        inverse_cdf=inverse_cdf,
        codec=codec,
    )

    assert result.message == payload.message_bytes
    assert result.codeword == payload.codeword
    assert len(result.bits) == len(payload.bits)


def test_extract_watermark_with_noise_and_resize():
    rng = np.random.default_rng(0)
    cfg, codec, payload, inverse_cdf, embedding = _prepare_embedding()
    noisy = embedding.latent_noise + 0.05 * rng.standard_normal(embedding.latent_noise.shape, dtype=np.float32)
    # Upscale to force coarse_align to run.
    enlarged = np.kron(noisy, np.ones((2, 2), dtype=np.float32))

    result = extract_watermark(
        image=enlarged,
        binding=payload.binding,
        config=cfg,
        inverse_cdf=inverse_cdf,
        codec=codec,
    )

    assert result.message == payload.message_bytes
    assert result.codeword == payload.codeword
