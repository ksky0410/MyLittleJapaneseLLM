from __future__ import annotations

import sys
import unittest
from argparse import ArgumentTypeError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_domains import _parse_domain


class EvaluateDomainsTests(unittest.TestCase):
    def test_parse_domain(self) -> None:
        self.assertEqual(_parse_domain("medical=data.bin"), ("medical", "data.bin"))

    def test_parse_domain_rejects_invalid_value(self) -> None:
        with self.assertRaises(ArgumentTypeError):
            _parse_domain("missing-separator")
        with self.assertRaises(ArgumentTypeError):
            _parse_domain("=data.bin")
        with self.assertRaises(ArgumentTypeError):
            _parse_domain("medical=")


if __name__ == "__main__":
    unittest.main()
