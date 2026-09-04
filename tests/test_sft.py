from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from my_little_japanese_llm.sft import load_sft_arrays, make_sft_batch


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


if __name__ == "__main__":
    unittest.main()
