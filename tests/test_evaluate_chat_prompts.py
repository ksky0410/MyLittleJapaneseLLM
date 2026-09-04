from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chat_prompts import load_prompts


class EvaluateChatPromptsTests(unittest.TestCase):
    def test_loads_issue_prompt_set(self) -> None:
        prompts = load_prompts(ROOT / "experiments/prompts/issue-1-chat-v1.json")
        self.assertEqual(len(prompts), 8)
        self.assertEqual(
            prompts[0],
            {
                "id": "casual-01",
                "category": "short-reply",
                "prompt": "まじで",
            },
        )
        self.assertEqual(len({item["id"] for item in prompts}), 8)

    def test_rejects_duplicate_ids_and_empty_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": "same", "category": "a", "prompt": "一"},
                        {"id": "same", "category": "b", "prompt": "二"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_prompts(path)
            path.write_text(
                json.dumps(
                    [{"id": "empty", "category": "a", "prompt": " "}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_prompts(path)


if __name__ == "__main__":
    unittest.main()
