from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_sft import validate_rehearsal_options


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


if __name__ == "__main__":
    unittest.main()
