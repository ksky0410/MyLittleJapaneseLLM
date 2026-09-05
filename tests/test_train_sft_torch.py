from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_sft_torch import (
    _load_base_checkpoint,
    _masked_cross_entropy,
    build_parser,
    validate_rehearsal_options,
)

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.torch_model import TorchJapaneseGPT
from my_little_japanese_llm.training import signature_from_config


class TrainSFTTorchOptionTests(unittest.TestCase):
    def test_rehearsal_options_are_paired_and_validated(self) -> None:
        self.assertEqual(validate_rehearsal_options(None, None), (None, 0.0))
        self.assertEqual(
            validate_rehearsal_options("tokens.bin", 0.25), ("tokens.bin", 0.25)
        )
        with self.assertRaises(ValueError):
            validate_rehearsal_options("tokens.bin", None)
        with self.assertRaises(ValueError):
            validate_rehearsal_options(None, 0.25)
        with self.assertRaises(ValueError):
            validate_rehearsal_options("tokens.bin", 1.0)
        with self.assertRaises(ValueError):
            validate_rehearsal_options("tokens.bin", float("nan"))

    def test_parser_accepts_cpu_and_rehearsal_arguments(self) -> None:
        args = build_parser().parse_args(
            [
                "--base-checkpoint",
                "base.pt",
                "--train-data",
                "train.npz",
                "--validation-data",
                "validation.npz",
                "--output-dir",
                "checkpoints",
                "--samples-dir",
                "samples",
                "--device",
                "cpu",
                "--no-amp",
                "--rehearsal-tokens",
                "tokens.bin",
                "--rehearsal-ratio",
                "0.25",
            ]
        )
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.no_amp)
        self.assertEqual(args.rehearsal_ratio, 0.25)


@unittest.skipUnless(torch is not None, "PyTorch未導入")
class TrainSFTTorchLossTests(unittest.TestCase):
    def test_masked_cross_entropy_ignores_unmasked_positions(self) -> None:
        functional = torch.nn.functional
        logits = torch.tensor(
            [
                [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, 1]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])
        actual = _masked_cross_entropy(logits, targets, loss_mask, functional)
        expected = functional.cross_entropy(
            logits.reshape(-1, 2)[[0, 2]],
            targets.reshape(-1)[[0, 2]],
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_all_zero_mask_returns_zero_without_nan(self) -> None:
        functional = torch.nn.functional
        logits = torch.zeros((1, 2, 3), requires_grad=True)
        targets = torch.tensor([[0, 1]])
        loss_mask = torch.zeros((1, 2))
        loss = _masked_cross_entropy(logits, targets, loss_mask, functional)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(loss))


@unittest.skipUnless(torch is not None, "PyTorch未導入")
class TrainSFTTorchCheckpointTests(unittest.TestCase):
    def test_train_torch_checkpoint_metadata_is_reloaded(self) -> None:
        config = load_config(ROOT / "configs/debug.toml")
        signature = signature_from_config(config, 32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pt"
            metadata_path = checkpoint.with_suffix(".json")
            model = TorchJapaneseGPT(32, 64, 2, 4, 64, 4)
            torch.save(model.state_dict(), checkpoint)
            metadata_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "weights_file": checkpoint.name,
                        "model": signature,
                        "weights_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            restored = TorchJapaneseGPT(32, 64, 2, 4, 64, 4)
            metadata = _load_base_checkpoint(restored, checkpoint, signature, torch)
            self.assertEqual(metadata["format_version"], 1)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
