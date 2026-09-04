from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chat_prompts import load_prompts, summarize_prompt_results


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

    def test_accepts_conversation_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "conversation",
                            "template": "conversation",
                            "prompt": "こんにちは",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prompts = load_prompts(path)
            self.assertEqual(prompts[0]["template"], "conversation")

            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "unknown",
                            "template": "unknown",
                            "prompt": "こんにちは",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_prompts(path)

    def test_summarizes_results_by_category(self) -> None:
        summary = summarize_prompt_results(
            [
                {
                    "category": "short",
                    "completion": "返答",
                    "completion_token_count": 4,
                    "eos_reached": True,
                },
                {
                    "category": "short",
                    "completion": "",
                    "completion_token_count": 0,
                    "eos_reached": False,
                },
            ]
        )
        self.assertEqual(summary["short"]["count"], 2)
        self.assertEqual(summary["short"]["empty_count"], 1)
        self.assertEqual(summary["short"]["eos_count"], 1)
        self.assertEqual(summary["short"]["mean_completion_tokens"], 2.0)


if __name__ == "__main__":
    unittest.main()
