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
    encode_generation_prompt,
    exclude_eos_from_loss,
    validate_eos_loss_weight,
    validate_rehearsal_options,
    weight_eos_in_loss,
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

    def test_parser_accepts_eos_exclusion(self) -> None:
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
                "--exclude-eos-from-sft-loss",
            ]
        )
        self.assertTrue(args.exclude_eos_from_sft_loss)

    def test_parser_accepts_eos_loss_weight(self) -> None:
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
                "--eos-loss-weight",
                "0.5",
            ]
        )
        self.assertEqual(args.eos_loss_weight, 0.5)

    def test_generation_prompt_supports_raw_and_conversation_templates(self) -> None:
        class Processor:
            def encode(self, text: str, out_type: type[int] = int) -> list[int]:
                del out_type
                return [len(text)]

            def eos_id(self) -> int:
                return 3

        processor = Processor()
        raw_ids, raw_rendered = encode_generation_prompt(
            processor, "こんにちは", "raw", "A", "B"
        )
        conversation_ids, conversation_rendered = encode_generation_prompt(
            processor, "こんにちは", "conversation", "DA", "DC"
        )
        self.assertEqual(raw_ids, [5])
        self.assertEqual(raw_rendered, "こんにちは")
        self.assertEqual(conversation_ids, [23, 14, 5, 3, 14])
        self.assertEqual(
            conversation_rendered,
            "<|startofconversation|><|speaker:DA|>こんにちは<eos:3><|speaker:DC|>",
        )

    def test_generation_prompt_rejects_unknown_template(self) -> None:
        class Processor:
            def encode(self, text: str, out_type: type[int] = int) -> list[int]:
                del text, out_type
                return [1]

            def eos_id(self) -> int:
                return 3

        with self.assertRaises(ValueError):
            encode_generation_prompt(Processor(), "こんにちは", "unknown", "A", "B")

    def test_exclude_eos_from_loss_only_removes_masked_eos(self) -> None:
        targets = torch.tensor([[1, 3, 2, 3]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        actual = exclude_eos_from_loss(targets, loss_mask, 3, torch)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(actual, expected))

    def test_eos_loss_weight_only_changes_masked_eos(self) -> None:
        targets = torch.tensor([[1, 3, 2, 3]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        actual = weight_eos_in_loss(targets, loss_mask, 3, 0.5, torch)
        expected = torch.tensor([[1.0, 0.5, 0.0, 0.5]])
        self.assertTrue(torch.equal(actual, expected))

    def test_eos_loss_weight_is_nonnegative_and_finite(self) -> None:
        self.assertEqual(validate_eos_loss_weight(0.0), 0.0)
        self.assertEqual(validate_eos_loss_weight(1.0), 1.0)
        with self.assertRaises(ValueError):
            validate_eos_loss_weight(-0.1)
        with self.assertRaises(ValueError):
            validate_eos_loss_weight(float("nan"))


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
