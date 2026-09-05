from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chat_prompts import load_prompts
from evaluate_torch_prompt_set import apply_template


class EvaluateTorchPromptSetTests(unittest.TestCase):
    def test_applies_conversation_template_without_mutating_input(self) -> None:
        prompts = load_prompts(ROOT / "experiments/prompts/issue-1-chat-v1.json")
        converted = apply_template(prompts, "conversation")
        self.assertNotIn("template", prompts[0])
        self.assertEqual(converted[0]["template"], "conversation")
        self.assertEqual(converted[0]["prompt"], prompts[0]["prompt"])

    def test_rejects_unknown_template(self) -> None:
        with self.assertRaises(ValueError):
            apply_template([], "unknown")


if __name__ == "__main__":
    unittest.main()
