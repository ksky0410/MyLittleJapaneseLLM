from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_torch import _advance_batch_rng, _schedule_steps


class TrainTorchScheduleTests(unittest.TestCase):
    def test_resume_rng_matches_skipping_the_same_number_of_batches(self) -> None:
        import numpy as np

        direct = np.random.default_rng(42)
        resumed = np.random.default_rng(42)
        for _ in range(5):
            direct.integers(0, 997, size=8)
        _advance_batch_rng(resumed, 5, 1024, 256, 8)

        np.testing.assert_array_equal(
            direct.integers(0, 997, size=8),
            resumed.integers(0, 997, size=8),
        )

    def test_legacy_schedule_keeps_checkpoint_at_evaluation_steps(self) -> None:
        evaluated = []
        checkpointed = []
        sampled = [0]
        for step in range(1, 8):
            should_evaluate, should_checkpoint, should_sample = _schedule_steps(
                step, 7, 3, 4, None
            )
            if should_evaluate:
                evaluated.append(step)
            if should_checkpoint:
                checkpointed.append(step)
            if should_sample:
                sampled.append(step)

        self.assertEqual(evaluated, [1, 3, 6, 7])
        self.assertEqual(checkpointed, [1, 3, 6, 7])
        self.assertEqual(sampled, [0, 4, 7])

    def test_explicit_checkpoint_interval_is_independent(self) -> None:
        evaluated = []
        checkpointed = []
        sampled = [0]
        for step in range(1, 8):
            should_evaluate, should_checkpoint, should_sample = _schedule_steps(
                step, 7, 3, 4, 5
            )
            if should_evaluate:
                evaluated.append(step)
            if should_checkpoint:
                checkpointed.append(step)
            if should_sample:
                sampled.append(step)

        self.assertEqual(evaluated, [1, 3, 6, 7])
        self.assertEqual(checkpointed, [5, 7])
        self.assertEqual(sampled, [0, 4, 7])

    def test_final_checkpoint_is_kept_when_interval_is_longer_than_run(self) -> None:
        self.assertEqual(
            [
                step
                for step in range(1, 8)
                if _schedule_steps(step, 7, 3, 4, 100)[1]
            ],
            [7],
        )
