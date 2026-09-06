from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from train_torch import _evaluation_batches, _weighted_mean_losses


def test_evaluation_batches_contains_requested_number_of_full_batches() -> None:
    tokens = np.arange(1000, dtype=np.int64)
    result = _evaluation_batches(
        tokens,
        batch_size=4,
        context_length=8,
        batches=3,
        device=torch.device("cpu"),
        torch=torch,
    )

    assert len(result) == 3
    assert [int(inputs.shape[0]) for inputs, _ in result] == [4, 4, 4]
    assert all(inputs.shape == targets.shape == (4, 8) for inputs, targets in result)


def test_evaluation_batches_allows_short_final_batch_only_when_data_is_small() -> None:
    tokens = np.arange(20, dtype=np.int64)
    result = _evaluation_batches(
        tokens,
        batch_size=4,
        context_length=8,
        batches=5,
        device=torch.device("cpu"),
        torch=torch,
    )

    assert [int(inputs.shape[0]) for inputs, _ in result] == [4, 4, 4]


def test_weighted_mean_losses_does_not_overweight_short_final_batch() -> None:
    assert _weighted_mean_losses([1.0, 3.0], [8, 2]) == pytest.approx(1.4)
