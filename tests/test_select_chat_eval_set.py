from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_chat_eval_set import classify_response_length, select_stratified_examples


class _LengthProcessor:
    def encode(self, text: str, out_type: type[int] = int) -> list[int]:
        del out_type
        return list(range(len(text)))

    def eos_id(self) -> int:
        return 3


class SelectChatEvalSetTests(unittest.TestCase):
    def test_classifies_response_lengths(self) -> None:
        self.assertEqual(classify_response_length(8), "short")
        self.assertEqual(classify_response_length(9), "medium")
        self.assertEqual(classify_response_length(24), "medium")
        self.assertEqual(classify_response_length(25), "long")

    def test_selects_one_unique_conversation_per_stratum(self) -> None:
        records = [
            {
                "conversation_id": "source:short",
                "turns": [
                    {"speaker_id": "A", "text": "a"},
                    {"speaker_id": "B", "text": "short"},
                ],
            },
            {
                "conversation_id": "source:medium",
                "turns": [
                    {"speaker_id": "A", "text": "a"},
                    {"speaker_id": "B", "text": "m" * 12},
                ],
            },
            {
                "conversation_id": "source:long",
                "turns": [
                    {"speaker_id": "A", "text": "a"},
                    {"speaker_id": "B", "text": "l" * 30},
                ],
            },
        ]
        examples = select_stratified_examples(
            records,
            set(),
            _LengthProcessor(),
            context_length=32,
            per_stratum=1,
            seed=42,
        )
        self.assertEqual(
            [item["stratum"] for item in examples], ["short", "medium", "long"]
        )
        self.assertEqual(
            len({item["conversation_id"] for item in examples}),
            3,
        )


if __name__ == "__main__":
    unittest.main()
