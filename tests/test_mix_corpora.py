from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_little_japanese_llm.mixing import mix_corpora


class _ConstantTokenProcessor:
    """単位境界とEOS加算を検証するためのSentencePiece代替。"""

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        del text, out_type
        return [1]


class MixCorporaTests(unittest.TestCase):
    def test_keeps_conversation_as_one_unit_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            general = root / "general.txt"
            medical = root / "medical.txt"
            conversation = root / "conversation.txt"
            general.write_text("一般文\n共通本文\n", encoding="utf-8")
            medical.write_text("共通本文\n医療本文\n", encoding="utf-8")
            conversation.write_text(
                "<|startofconversation|>\n"
                "<|speaker:A|>こんにちは。\n"
                "<|speaker:B|>元気です。\n"
                "<|endofconversation|>\n",
                encoding="utf-8",
            )
            output = root / "mixed.txt"
            manifest_path = root / "manifest.json"
            manifest = mix_corpora(
                [
                    ("general", general),
                    ("medical", medical),
                    ("conversation", conversation),
                ],
                output,
                manifest_path,
                seed=17,
            )
            text = output.read_text(encoding="utf-8")
            self.assertEqual(
                manifest, json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(manifest["input_unit_count"], 5)
            self.assertEqual(manifest["unique_unit_count"], 4)
            self.assertEqual(manifest["output_unit_count"], 4)
            self.assertEqual(manifest["sources"][1]["duplicate_units_removed"], 1)
            self.assertEqual(text.count("<|startofconversation|>"), 1)
            self.assertEqual(text.count("<|endofconversation|>"), 1)
            self.assertIn("<|speaker:A|>こんにちは。\n<|speaker:B|>元気です。", text)
            self.assertEqual(text.count("共通本文"), 1)
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )

    def test_seed_and_weighted_target_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources: list[tuple[str, Path, float]] = []
            for name, count, weight in (
                ("general", 8, 8.0),
                ("conversation", 4, 1.0),
                ("medical", 4, 1.0),
            ):
                path = root / f"{name}.txt"
                path.write_text(
                    "\n".join(f"{name}-{i}" for i in range(count)) + "\n",
                    encoding="utf-8",
                )
                sources.append((name, path, weight))
            first = mix_corpora(
                sources,
                root / "first.txt",
                root / "first.json",
                seed=123,
                target_units=10,
            )
            second = mix_corpora(
                sources,
                root / "second.txt",
                root / "second.json",
                seed=123,
                target_units=10,
            )
            self.assertEqual(
                (root / "first.txt").read_bytes(), (root / "second.txt").read_bytes()
            )
            self.assertEqual(first["target_units"], 10)
            self.assertEqual(first["output_unit_count"], 10)
            self.assertEqual(
                first["actual_adoption_share"],
                {"general": 0.8, "conversation": 0.1, "medical": 0.1},
            )
            self.assertEqual(
                first["actual_adoption_share"], second["actual_adoption_share"]
            )

    def test_reallocates_quota_when_source_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            large = root / "large.txt"
            small = root / "small.txt"
            large.write_text(
                "\n".join(f"大-{i}" for i in range(6)) + "\n", encoding="utf-8"
            )
            small.write_text("小-1\n", encoding="utf-8")
            manifest = mix_corpora(
                [("large", large, 1.0), ("small", small, 1.0)],
                root / "mixed.txt",
                root / "mixed.json",
                target_units=4,
            )
            self.assertEqual(manifest["output_unit_count"], 4)
            self.assertEqual(manifest["sources"][0]["adopted_unit_count"], 3)
            self.assertEqual(manifest["sources"][1]["adopted_unit_count"], 1)

    def test_selects_deterministically_by_token_budget_and_records_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            general = root / "general.txt"
            medical = root / "medical.txt"
            conversation = root / "conversation.txt"
            tokenizer = root / "tokenizer.model"
            general.write_text("一般一\n一般二\n一般三\n", encoding="utf-8")
            medical.write_text("医療一\n医療二\n医療三\n", encoding="utf-8")
            conversation.write_text(
                "<|startofconversation|>\n"
                "<|speaker:A|>こんにちは。\n"
                "<|endofconversation|>\n",
                encoding="utf-8",
            )
            tokenizer.write_bytes(b"test-tokenizer")
            sources = [
                ("general", general, 1.0),
                ("medical", medical, 1.0),
                ("conversation", conversation, 1.0),
            ]
            with patch(
                "my_little_japanese_llm.tokenizer.load_processor",
                return_value=_ConstantTokenProcessor(),
            ):
                first = mix_corpora(
                    sources,
                    root / "first.txt",
                    root / "first.json",
                    seed=17,
                    tokenizer_path=tokenizer,
                    target_tokens=18,
                )
                second = mix_corpora(
                    sources,
                    root / "second.txt",
                    root / "second.json",
                    seed=17,
                    tokenizer_path=tokenizer,
                    target_tokens=18,
                )

            self.assertEqual(
                (root / "first.txt").read_bytes(), (root / "second.txt").read_bytes()
            )
            self.assertEqual(first["target_units"], None)
            self.assertEqual(first["target_tokens"], 18)
            self.assertEqual(first["selected_token_count"], 18)
            self.assertEqual(first["tokenizer_path"], str(tokenizer.resolve()))
            self.assertEqual(
                first["tokenizer_sha256"],
                hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                sum(source["adopted_token_count"] for source in first["sources"]),
                first["selected_token_count"],
            )
            self.assertAlmostEqual(sum(first["actual_token_share"].values()), 1.0)
            self.assertEqual(first["actual_token_share"], second["actual_token_share"])
            self.assertEqual(first["output_unit_count"], 7)
            self.assertEqual(
                (root / "first.txt")
                .read_text(encoding="utf-8")
                .count("<|startofconversation|>"),
                1,
            )

    def test_rejects_conflicting_token_budget_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            tokenizer = root / "tokenizer.model"
            source.write_text("本文\n", encoding="utf-8")
            tokenizer.write_bytes(b"tokenizer")
            with self.assertRaises(ValueError):
                mix_corpora(
                    [("x", source)],
                    root / "out",
                    root / "manifest",
                    target_units=1,
                    target_tokens=1,
                    tokenizer_path=tokenizer,
                )
            with self.assertRaises(ValueError):
                mix_corpora(
                    [("x", source)],
                    root / "out",
                    root / "manifest",
                    target_tokens=1,
                )
            with self.assertRaises(ValueError):
                mix_corpora(
                    [("x", source)],
                    root / "out",
                    root / "manifest",
                    tokenizer_path=tokenizer,
                )

    def test_rejects_invalid_inputs_and_target_too_large(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("本文\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                mix_corpora(
                    [("x", source), ("x", source)], root / "out", root / "manifest"
                )
            with self.assertRaises(ValueError):
                mix_corpora([("x", source, 0.0)], root / "out", root / "manifest")
            with self.assertRaises(ValueError):
                mix_corpora([("x", source)], source, root / "manifest")
            with self.assertRaises(ValueError):
                mix_corpora(
                    [("x", source)], root / "out", root / "manifest", target_units=2
                )
            for name, content in (
                ("unclosed", "<|startofconversation|>\n発話\n"),
                ("orphan", "発話\n<|endofconversation|>\n"),
                ("nested", "<|startofconversation|>\n<|startofconversation|>\n"),
            ):
                bad = root / f"{name}.txt"
                bad.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    mix_corpora([("bad", bad)], root / "out", root / "manifest")


if __name__ == "__main__":
    unittest.main()
