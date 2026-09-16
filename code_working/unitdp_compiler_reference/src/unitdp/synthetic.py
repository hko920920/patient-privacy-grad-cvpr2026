"""Synthetic windowed sequential benchmark for UnitDP smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unitdp.mapping import WindowMapping, WindowRecord


@dataclass(frozen=True)
class SyntheticWindowData:
    train_x: np.ndarray
    train_y: np.ndarray
    train_mapping: WindowMapping
    test_x: np.ndarray
    test_y: np.ndarray
    test_mapping: WindowMapping


def _build_owner_windows(
    owner_ids: list[str],
    window_length: int,
    stride: int,
    sequence_length: int,
    channels: int,
    rng: np.random.Generator,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, WindowMapping]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    records: list[WindowRecord] = []
    row_index = 0

    for owner_pos, owner in enumerate(owner_ids):
        owner_bias = rng.normal(loc=0.0, scale=0.8)
        trend = rng.normal(loc=0.0, scale=0.02)
        time = np.arange(sequence_length, dtype=np.float32)
        signal = rng.normal(size=(sequence_length, channels)).astype(np.float32)
        signal[:, 0] += owner_bias + trend * time
        signal[:, 1] += np.sin(time / 7.0 + owner_pos)
        if channels > 2:
            signal[:, 2:] += rng.normal(scale=0.2, size=(sequence_length, channels - 2))

        for start in range(0, sequence_length - window_length + 1, stride):
            end = start + window_length
            window = signal[start:end]
            score = float(window[:, 0].mean() + 0.35 * window[:, 1].mean())
            label = int(score > 0.0)
            features = np.concatenate(
                [
                    window.mean(axis=0),
                    window.std(axis=0),
                    window[-1] - window[0],
                ]
            ).astype(np.float32)
            x_rows.append(features)
            y_rows.append(label)
            records.append(
                WindowRecord(
                    window_id=f"{scenario}_{owner}_{start}",
                    owner_ids=(owner,),
                    start=start,
                    end=end,
                    row_index=row_index,
                    label=str(label),
                    scenario=scenario,
                )
            )
            row_index += 1

    return (
        np.stack(x_rows).astype(np.float32),
        np.array(y_rows, dtype=np.int64),
        WindowMapping(records),
    )


def make_synthetic_window_data(
    train_owners: int = 80,
    test_owners: int = 40,
    sequence_length: int = 96,
    window_length: int = 16,
    stride: int = 4,
    channels: int = 4,
    seed: int = 0,
) -> SyntheticWindowData:
    """Create a small owner/window classification benchmark."""

    rng = np.random.default_rng(seed)
    train_ids = [f"train_owner_{idx:04d}" for idx in range(train_owners)]
    test_ids = [f"test_owner_{idx:04d}" for idx in range(test_owners)]
    train_x, train_y, train_mapping = _build_owner_windows(
        train_ids,
        window_length,
        stride,
        sequence_length,
        channels,
        rng,
        "synthetic_train",
    )
    test_x, test_y, test_mapping = _build_owner_windows(
        test_ids,
        window_length,
        stride,
        sequence_length,
        channels,
        rng,
        "synthetic_test",
    )
    return SyntheticWindowData(
        train_x=train_x,
        train_y=train_y,
        train_mapping=train_mapping,
        test_x=test_x,
        test_y=test_y,
        test_mapping=test_mapping,
    )

