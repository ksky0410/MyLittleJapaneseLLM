from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_chat_sft import (
    build_conversation_example,
    make_training_example,
    prepare_chat_sft,
    truncate_and_pad,
)


class FakeProcessor:
    """markerと本文を目視しやすい整数列へ写像する小さなprocessor。"""

    def __init__(self) -> None:
        self._next_id = 10
        self._ids: dict[str, list[int]] = {}

    def encode(self, text: str, out_type: type[int] = int) -> list[int]:
        del out_type
        if text not in self._ids:
            self._ids[text] = [self._next_id]
            self._next_id += 1
        return self._ids[text]

    def eos_id(self) -> int:
        return 3

    def pad_id(self) -> int:
        return 0


class PrepareChatSFTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = FakeProcessor()
        self.turns = [
            {"speaker_id": "A", "text": "最初"},
            {"speaker_id": "B", "text": "応答"},
            {"speaker_id": "A", "text": "次の応答"},
        ]

    def test_masks_only_current_body_and_eos(self) -> None:
        ids, mask, body_count = build_conversation_example(
            self.turns, 1, self.processor
        )
        start = self.processor.encode("<|startofconversation|>")[0]
        speaker_a = self.processor.encode("<|speaker:A|>")[0]
        body_a = self.processor.encode("最初")[0]
        speaker_b = self.processor.encode("<|speaker:B|>")[0]
        body_b = self.processor.encode("応答")[0]
        end = self.processor.encode("<|endofconversation|>")[0]
        self.assertEqual(ids, [start, speaker_a, body_a, 3, speaker_b, body_b, 3, end])
        self.assertEqual(mask, [0, 0, 0, 0, 0, 1, 1, 0])
        self.assertEqual(body_count, 1)

    def test_creates_each_later_turn_as_a_target(self) -> None:
        first = build_conversation_example(self.turns, 1, self.processor)
        second = build_conversation_example(self.turns, 2, self.processor)
        self.assertEqual(sum(first[1]), 2)
        self.assertEqual(sum(second[1]), 2)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(second[0][-2], self.processor.eos_id())

    def test_left_truncates_and_right_pads_to_context_plus_one(self) -> None:
        ids, mask, truncated = truncate_and_pad(
            [1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 1, 0], 3, 0
        )
        self.assertTrue(truncated)
        self.assertEqual(ids, [3, 4, 5, 6])
        self.assertEqual(mask, [1, 1, 1, 0])

        ids, mask, truncated = truncate_and_pad([1, 2], [0, 1], 3, 0)
        self.assertFalse(truncated)
        self.assertEqual(ids, [1, 2, 0, 0])
        self.assertEqual(mask, [0, 1, 0, 0])

    def test_training_arrays_are_shifted_and_mask_targets_response(self) -> None:
        input_ids, target_ids, loss_mask, stats = make_training_example(
            self.turns, 1, self.processor, 8
        )
        self.assertEqual(input_ids.shape, (8,))
        self.assertEqual(target_ids.shape, (8,))
        self.assertEqual(loss_mask.shape, (8,))
        np.testing.assert_array_equal(target_ids[:-1], input_ids[1:])
        self.assertEqual(loss_mask.tolist(), [0, 0, 0, 0, 1, 1, 0, 0])
        self.assertEqual(stats["response_token_count"], 2)

    def test_prepare_writes_npz_and_manifest_for_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "conversation-v1"
            input_dir.mkdir()
            records = [
                {
                    "conversation_id": "one",
                    "turns": self.turns,
                },
                {
                    "conversation_id": "short",
                    "turns": [{"speaker_id": "A", "text": "一発話だけ"}],
                },
            ]
            content = (
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
                + "\n"
            )
            for split in ("train", "validation"):
                (input_dir / f"{split}.jsonl").write_text(content, encoding="utf-8")
            tokenizer = root / "tokenizer.model"
            tokenizer.write_bytes(b"fake-tokenizer")

            with patch("prepare_chat_sft.load_processor", return_value=self.processor):
                manifest = prepare_chat_sft(
                    tokenizer,
                    input_dir,
                    root / "prepared",
                    root / "prepared/manifest.json",
                    8,
                    seed=7,
                )
            self.assertEqual(manifest["format"], "chat-sft-preparation-v1")
            self.assertEqual(manifest["context_length"], 8)
            self.assertEqual(manifest["sequence_length"], 9)
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(set(manifest["splits"]), {"train", "validation"})
            for split in ("train", "validation"):
                stats = manifest["splits"][split]
                self.assertEqual(stats["conversation_count"], 2)
                self.assertEqual(stats["short_conversation_count"], 1)
                self.assertEqual(stats["example_count"], 2)
                self.assertEqual(stats["response_token_count"], 4)
                self.assertEqual(stats["array_shape"], [2, 8])
                with np.load(root / "prepared" / f"{split}.npz") as arrays:
                    self.assertEqual(arrays["input_ids"].shape, (2, 8))
                    self.assertEqual(arrays["target_ids"].shape, (2, 8))
                    self.assertEqual(arrays["loss_mask"].shape, (2, 8))
                    self.assertEqual(arrays["loss_mask"].dtype, np.uint8)
            self.assertEqual(
                json.loads((root / "prepared/manifest.json").read_text())[
                    "input_sha256"
                ],
                manifest["input_sha256"],
            )

    def test_rejects_malformed_turns(self) -> None:
        with self.assertRaises(ValueError):
            build_conversation_example(
                [{"speaker_id": "A", "text": "一"}], 1, self.processor
            )
        with self.assertRaises(ValueError):
            build_conversation_example(
                [{"speaker_id": "A", "text": "一"}, {"text": "二"}],
                1,
                self.processor,
            )


if __name__ == "__main__":
    unittest.main()
