import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_medical_sft import convert_record, prepare_medical_sft


class PrepareMedicalSFTTests(unittest.TestCase):
    def test_convert_record_uses_question_and_correct_explanation(self) -> None:
        record = {
            "number": "101A1",
            "exam_version": 101,
            "question": "本文です。",
            "options": {"a": "ａ　第一", "b": "ｂ　第二"},
            "correct": ["b"],
            "point": "重要です。",
            "explanations": {"a": "誤り。", "b": "正しい理由。"},
        }
        converted = convert_record(record, "train")
        self.assertEqual(converted["conversation_id"], "medical-qb-v2:101A1")
        self.assertIn("選択肢：a:ａ 第一", converted["turns"][0]["text"])
        self.assertIn("正解はbです", converted["turns"][1]["text"])
        self.assertIn("正しい理由", converted["turns"][1]["text"])
        self.assertIn("ポイント：重要です", converted["turns"][1]["text"])

    def test_prepare_preserves_source_and_writes_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            record = {
                "number": "101A1",
                "exam_version": 101,
                "question": "問題です。",
                "options": {"a": "ａ　正しい"},
                "correct": ["a"],
                "point": "",
                "explanations": {"a": "正しい。"},
            }
            for split in ("train", "validation"):
                (source / f"{split}.jsonl").write_text(
                    json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            output = root / "output"
            manifest_path = root / "manifest.json"
            manifest = prepare_medical_sft(source, output, manifest_path)
            self.assertEqual(manifest["splits"]["train"]["output_record_count"], 1)
            self.assertTrue((output / "validation.jsonl").is_file())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["format"],
                "medical-qb-sft-conversation-v1",
            )

    def test_drop_truncated_requires_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for split in ("train", "validation"):
                (source / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_medical_sft(
                    source,
                    root / "output",
                    root / "manifest.json",
                    drop_truncated=True,
                )


if __name__ == "__main__":
    unittest.main()
