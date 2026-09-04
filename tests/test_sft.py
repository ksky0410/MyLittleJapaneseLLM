from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from my_little_japanese_llm.sft import (
    load_rehearsal_tokens,
    load_sft_arrays,
    make_sft_batch,
    split_sft_rehearsal_batch_size,
    validate_rehearsal_ratio,
    validate_short_response_options,
)


class _FakeMX:
    @staticmethod
    def array(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)


class SFTDataTests(unittest.TestCase):
    def test_loads_and_batches_masked_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            np.savez(
                path,
                input_ids=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32),
                target_ids=np.array([[2, 3, 0], [5, 6, 0]], dtype=np.uint32),
                loss_mask=np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32),
            )
            arrays = load_sft_arrays(path, context_length=3)
            self.assertEqual(arrays["input_ids"].dtype, np.int32)
            self.assertEqual(arrays["loss_mask"].dtype, np.float32)
            batch = make_sft_batch(arrays, 4, np.random.default_rng(7), _FakeMX())
            self.assertEqual(batch[0].shape, (4, 3))
            self.assertEqual(batch[2].shape, (4, 3))

    def test_rejects_shape_and_mask_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            np.savez(
                path,
                input_ids=np.zeros((1, 3), dtype=np.int32),
                target_ids=np.zeros((1, 2), dtype=np.int32),
                loss_mask=np.zeros((1, 3), dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                load_sft_arrays(path, context_length=3)

            np.savez(
                path,
                input_ids=np.zeros((1, 3), dtype=np.int32),
                target_ids=np.zeros((1, 3), dtype=np.int32),
                loss_mask=np.array([[0, 2, 0]], dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                load_sft_arrays(path, context_length=3)

    def test_can_sample_short_responses_as_a_stratum(self) -> None:
        arrays = {
            "input_ids": np.arange(20, dtype=np.int32).reshape(4, 5),
            "target_ids": np.arange(20, dtype=np.int32).reshape(4, 5),
            "loss_mask": np.array(
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                ],
                dtype=np.float32,
            ),
        }
        batch = make_sft_batch(
            arrays,
            4,
            np.random.default_rng(7),
            _FakeMX(),
            short_response_ratio=0.5,
            short_response_max_tokens=2,
        )
        lengths = batch[2].sum(axis=1)
        self.assertEqual(int((lengths <= 2).sum()), 2)
        self.assertEqual(int((lengths > 2).sum()), 2)

    def test_validates_short_response_options(self) -> None:
        self.assertEqual(validate_short_response_options(None, None), (0.0, 8))
        self.assertEqual(validate_short_response_options(0.5, 8), (0.5, 8))
        with self.assertRaises(ValueError):
            validate_short_response_options(0.5, None)
        with self.assertRaises(ValueError):
            validate_short_response_options(1.0, 8)
        with self.assertRaises(ValueError):
            validate_short_response_options(0.5, 0)

    def test_splits_rehearsal_batch_and_validates_ratio(self) -> None:
        self.assertEqual(split_sft_rehearsal_batch_size(8, 0.25), (6, 2))
        self.assertEqual(split_sft_rehearsal_batch_size(8, 0.0), (8, 0))
        self.assertEqual(validate_rehearsal_ratio(0.25), 0.25)
        with self.assertRaises(ValueError):
            validate_rehearsal_ratio(-0.1)
        with self.assertRaises(ValueError):
            validate_rehearsal_ratio(1.0)

    def test_loads_uint32_rehearsal_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rehearsal.bin"
            np.array([1, 2, 3, 4], dtype=np.uint32).tofile(path)
            tokens = load_rehearsal_tokens(path)
            np.testing.assert_array_equal(
                tokens, np.array([1, 2, 3, 4], dtype=np.int32)
            )


if __name__ == "__main__":
    unittest.main()
