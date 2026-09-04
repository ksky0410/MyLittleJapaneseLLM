from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tokenizer_report import FIXED_SAMPLES, build_report

from my_little_japanese_llm.tokenizer import train_sentencepiece


@unittest.skipUnless(importlib.util.find_spec("sentencepiece"), "SentencePiece未導入")
class TokenizerReportTests(unittest.TestCase):
    def test_report_contains_comparable_statistics_for_two_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text(
                "むかしむかし、小さな村がありました。\n"
                "人工知能とは、文章を学ぶ仕組みです。\n"
                "今日は良い天気です。\n",
                encoding="utf-8",
            )
            unigram, _, _ = train_sentencepiece(source, root / "unigram", 32, "unigram")
            bpe, _, _ = train_sentencepiece(source, root / "bpe", 32, "bpe")

            report = build_report([unigram, bpe], source)

            self.assertEqual(len(report["tokenizers"]), 2)
            self.assertEqual(report["input_path"], str(source.resolve()))
            for tokenizer in report["tokenizers"]:
                self.assertIn("model_path", tokenizer)
                self.assertGreater(tokenizer["vocab_size"], 0)
                self.assertGreater(tokenizer["total_token_count"], 0)
                self.assertGreater(tokenizer["average_characters_per_token"], 0)
                self.assertEqual(set(tokenizer["fixed_samples"]), set(FIXED_SAMPLES))
                for pieces in tokenizer["fixed_samples"].values():
                    self.assertGreater(len(pieces), 0)
