from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_chat_review_template import create_review_template


class CreateChatReviewTemplateTests(unittest.TestCase):
    def test_preserves_generation_and_leaves_human_labels_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.json"
            output = root / "review.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "conversation_id": "c1",
                                "record_index": 2,
                                "target_index": 3,
                                "target_speaker": "B",
                                "source": "mrmp",
                                "stratum": "short",
                                "history_truncated": False,
                                "train_text_overlap": False,
                                "rendered_prompt": "prompt",
                                "reference": "参照",
                                "completion": "生成",
                                "generated_token_count": 2,
                                "eos_reached": True,
                                "token_overlap_f1": 0.5,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            template = create_review_template(evaluation, output)

            self.assertEqual(template["review_status"], "pending_human_review")
            self.assertEqual(template["reviewed_count"], 0)
            review = template["reviews"][0]
            self.assertEqual(review["completion"], "生成")
            self.assertIsNone(review["context_fit"])
            self.assertIsNone(review["role_fit"])
            self.assertIsNone(review["not_collapsed"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
