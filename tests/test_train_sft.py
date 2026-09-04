from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_sft import build_parser, validate_rehearsal_options


class TrainSFTOptionTests(unittest.TestCase):
    def test_rehearsal_options_are_paired(self) -> None:
        self.assertEqual(validate_rehearsal_options(None, None), (None, 0.0))
        self.assertEqual(
            validate_rehearsal_options("tokens.bin", 0.25), ("tokens.bin", 0.25)
        )
        with self.assertRaises(ValueError):
            validate_rehearsal_options("tokens.bin", None)
        with self.assertRaises(ValueError):
            validate_rehearsal_options(None, 0.25)

    def test_parser_accepts_short_response_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--base-checkpoint",
                "base.npz",
                "--train-data",
                "train.npz",
                "--validation-data",
                "val.npz",
                "--output-dir",
                "checkpoints",
                "--samples-dir",
                "samples",
                "--short-response-ratio",
                "0.5",
                "--short-response-max-tokens",
                "8",
            ]
        )
        self.assertEqual(args.short_response_ratio, 0.5)
        self.assertEqual(args.short_response_max_tokens, 8)


if __name__ == "__main__":
    unittest.main()
