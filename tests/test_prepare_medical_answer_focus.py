import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_medical_answer_focus import answer_focus_text, convert_record, prepare_answer_focus


class PrepareMedicalAnswerFocusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "number": "101A1",
            "exam_version": 101,
            "question": "問題です。",
            "options": {"a": "第一", "b": "第二"},
            "correct": ["b"],
        }

    def test_answer_focus_contains_only_the_answer_label(self) -> None:
        self.assertEqual(answer_focus_text(self.record), "正解はbです。")
        converted = convert_record(self.record, "train")
        self.assertEqual(converted["dataset"], "medical-qb-v2-answer-focus")
        self.assertIn("正解と理由を簡潔に答えてください", converted["turns"][0]["text"])
        self.assertEqual(converted["turns"][1]["text"], "正解はbです。")

    def test_prepare_writes_both_splits_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            content = json.dumps(self.record, ensure_ascii=False) + "\n"
            for split in ("train", "validation"):
                (source / f"{split}.jsonl").write_text(content, encoding="utf-8")
            manifest = prepare_answer_focus(
                source, root / "output", root / "manifest.json"
            )
            self.assertEqual(manifest["format"], "medical-qb-answer-focus-conversation-v1")
            self.assertEqual(manifest["splits"]["train"]["output_record_count"], 1)
            self.assertTrue((root / "output" / "validation.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
