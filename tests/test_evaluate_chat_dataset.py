from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chat_dataset import encode_history, select_examples


class _FakeProcessor:
    def encode(self, text: str, out_type: type[int] = int) -> list[int]:
        del out_type
        return [ord(text[0]) if text else 0]

    def eos_id(self) -> int:
        return 3


class EvaluateChatDatasetTests(unittest.TestCase):
    def test_selects_later_turns_deterministically(self) -> None:
        records = [
            {
                "conversation_id": "a",
                "turns": [
                    {"speaker_id": "A", "text": "一"},
                    {"speaker_id": "B", "text": "二"},
                    {"speaker_id": "A", "text": "三"},
                ],
            },
            {
                "conversation_id": "b",
                "turns": [
                    {"speaker_id": "A", "text": "四"},
                    {"speaker_id": "B", "text": "五"},
                ],
            },
        ]
        first = select_examples(records, 3, 42)
        second = select_examples(records, 3, 42)
        self.assertEqual(
            [(item["conversation_id"], item["target_index"]) for item in first],
            [(item["conversation_id"], item["target_index"]) for item in second],
        )
        self.assertTrue(all(item["target_index"] >= 1 for item in first))

    def test_history_excludes_target_body_and_includes_target_speaker(self) -> None:
        turns = [
            {"speaker_id": "A", "text": "最初"},
            {"speaker_id": "B", "text": "応答"},
        ]
        ids, rendered = encode_history(turns, 1, _FakeProcessor())
        self.assertEqual(ids, [ord("<"), ord("<"), ord("最"), 3, ord("<")])
        self.assertIn("<|speaker:B|>", rendered)
        self.assertNotIn("応答<eos", rendered)


if __name__ == "__main__":
    unittest.main()
