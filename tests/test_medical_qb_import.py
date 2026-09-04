from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_medical_qb import import_medical_qb, read_only_connection


class MedicalQBImportTests(unittest.TestCase):
    def _make_database(self, path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE questions (
                    number TEXT PRIMARY KEY,
                    exam_version INTEGER,
                    pre_body TEXT DEFAULT '',
                    body TEXT DEFAULT '',
                    options_json TEXT DEFAULT '{}',
                    correct_answers_json TEXT DEFAULT '[]',
                    has_images INTEGER DEFAULT 0
                );
                CREATE TABLE descriptions (
                    number TEXT PRIMARY KEY,
                    json TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                """
                INSERT INTO questions
                (number, exam_version, pre_body, body, options_json,
                 correct_answers_json, has_images)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "118A1",
                        118,
                        "",
                        "通常問題<b>本文</b><br>画像を示す<img src='https://example.test/a.png'>",
                        json.dumps({"a": "ａ　第一選択", "b": "ｂ　第二選択"}),
                        '["a"]',
                        1,
                    ),
                    (
                        "119A1",
                        119,
                        "前置き<i>部分</i>",
                        "検証問題です。",
                        json.dumps({"a": "ａ　はい", "b": "ｂ　いいえ"}),
                        '["b"]',
                        0,
                    ),
                    (
                        "120A1",
                        120,
                        "",
                        "説明が欠けても採用する問題です。",
                        json.dumps({"a": "ａ　正しい"}),
                        '["a"]',
                        0,
                    ),
                    (
                        "700A1",
                        700,
                        "",
                        "チャレンジ問題です。",
                        json.dumps({"a": "ａ　正しい"}),
                        '["a"]',
                        0,
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO descriptions (number, json) VALUES (?, ?)",
                [
                    (
                        "118A1",
                        json.dumps(
                            {
                                "point": "<p>重要なポイント</p>",
                                "explanations": {
                                    "a": "<b>○ａ</b><br>正しい説明。\r\n続き。",
                                    "b": "<b>×ｂ</b><br>誤り。",
                                },
                            }
                        ),
                    ),
                    (
                        "119A1",
                        json.dumps(
                            {
                                "point": "",
                                "explanations": {"a": "<b>説明</b>"},
                            }
                        ),
                    ),
                    (
                        "700A1",
                        json.dumps(
                            {
                                "point": "チャレンジポイント",
                                "explanations": {"a": "正しい説明"},
                            }
                        ),
                    ),
                ],
            )

    def test_imports_read_only_sqlite_and_writes_clean_split_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "qb.sqlite"
            output_dir = root / "medical-qb-v1"
            self._make_database(database)
            input_hash = hashlib.sha256(database.read_bytes()).hexdigest()

            with read_only_connection(database) as connection:
                self.assertEqual(
                    connection.execute("PRAGMA query_only").fetchone()[0], 1
                )
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("CREATE TABLE should_not_exist (id INTEGER)")

            manifest = import_medical_qb(database, output_dir)
            loaded = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded, manifest)
            self.assertEqual(manifest["input_sha256"], input_hash)
            self.assertTrue(manifest["read_only"])
            self.assertEqual(manifest["questions_count"], 4)
            self.assertEqual(manifest["adopted_count"], 4)
            self.assertEqual(manifest["missing_count"], 1)
            self.assertEqual(manifest["image_count"], 1)
            self.assertEqual(
                manifest["exam_version_counts"],
                {"118": 1, "119": 1, "120": 1, "700": 1},
            )
            self.assertEqual(
                {
                    split: values["count"]
                    for split, values in manifest["splits"].items()
                },
                {"train": 1, "validation": 1, "test": 1, "challenge": 1},
            )
            self.assertEqual(manifest["challenge_versions"], [700])
            challenge_text = (output_dir / "challenge.txt").read_text(encoding="utf-8")
            self.assertIn("チャレンジ問題です。", challenge_text)

            jsonl = (output_dir / "train.jsonl").read_text(encoding="utf-8")
            text = (output_dir / "train.txt").read_text(encoding="utf-8")
            self.assertNotIn("https://example.test/a.png", jsonl + text)
            self.assertIn("[図表あり]", jsonl + text)
            self.assertIn("問題：通常問題本文 画像を示す[図表あり]", text)
            self.assertIn("選択肢：a：ａ　第一選択", text)
            self.assertIn("正解：a", text)
            self.assertIn("ポイント：重要なポイント", text)
            self.assertIn("選択肢解説：a：○ａ 正しい説明。 続き。", text)
            self.assertEqual(len(text.splitlines()), 1)
            self.assertEqual(
                manifest["splits"]["train"]["txt_sha256"],
                hashlib.sha256((output_dir / "train.txt").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                hashlib.sha256(database.read_bytes()).hexdigest(), input_hash
            )

    def test_split_versions_can_be_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "qb.sqlite"
            self._make_database(database)
            manifest = import_medical_qb(
                database,
                root / "custom",
                validation_versions=(118,),
                test_versions=(119,),
                challenge_versions=(700,),
            )
            self.assertEqual(manifest["validation_versions"], [118])
            self.assertEqual(manifest["test_versions"], [119])
            self.assertEqual(manifest["challenge_versions"], [700])
            self.assertEqual(manifest["splits"]["validation"]["count"], 1)
            self.assertEqual(manifest["splits"]["test"]["count"], 1)
            self.assertEqual(manifest["splits"]["challenge"]["count"], 1)
            self.assertEqual(manifest["splits"]["train"]["count"], 1)

    def test_split_version_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "qb.sqlite"
            self._make_database(database)
            with self.assertRaises(ValueError):
                import_medical_qb(
                    database,
                    root / "validation-overlap",
                    validation_versions=(700,),
                )
            with self.assertRaises(ValueError):
                import_medical_qb(
                    database,
                    root / "challenge-overlap",
                    test_versions=(700,),
                )


if __name__ == "__main__":
    unittest.main()
