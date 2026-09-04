from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_aozora import import_aozora


class AozoraImportTests(unittest.TestCase):
    def test_shift_jis_zip_is_cleaned_and_manifest_is_reproducible(self) -> None:
        source_text = (
            "小さな作品\n"
            "著者名\n"
            "\n"
            "-------------------------------------------------------\n"
            "【テキスト中に現れる記号について】\n"
            "［＃注記の説明］\n"
            "-------------------------------------------------------\n"
            "｜猫《ねこ》が歩く。［＃ここから改行］\n"
            "長い文章です。\n"
            "底本の情報ではありません。\n"
            "-------------------------------------------------------\n"
            "底本：テスト用の底本\n"
            "入力：テスト入力者\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sample.zip"
            output_path = root / "corpus.txt"
            manifest_path = root / "manifest.json"
            with zipfile.ZipFile(input_path, "w") as archive:
                archive.writestr("sample.txt", source_text.encode("shift_jis"))

            manifest = import_aozora(
                input_path,
                output_path,
                manifest_path,
                source="https://example.test/aozora/sample.zip",
                max_chars=5,
            )
            output = output_path.read_text(encoding="utf-8")
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                output,
                "猫が歩く。\n長い文章で\nす。\n底本の情報\nではありま\nせん。\n",
            )
            self.assertNotIn("作品", output)
            self.assertNotIn("注記", output)
            self.assertNotIn("［＃", output)
            self.assertNotIn("《", output)
            self.assertEqual(loaded, manifest)
            self.assertEqual(manifest["encoding"], "shift_jis")
            self.assertEqual(
                manifest["source"], "https://example.test/aozora/sample.zip"
            )
            self.assertEqual(manifest["input_member"], "sample.txt")
            self.assertEqual(manifest["annotation_count"], 2)
            self.assertEqual(manifest["ruby_count"], 1)
            self.assertEqual(manifest["split_paragraphs"], 2)
            self.assertEqual(manifest["split_segments"], 3)
            self.assertEqual(manifest["removed_lines"], 10)
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(input_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["input_lines"], len(source_text.splitlines()))
            self.assertEqual(manifest["output_lines"], 6)
            self.assertTrue(all(len(line) <= 5 for line in output.splitlines()))

    def test_shift_jis_txt_uses_default_manifest_path(self) -> None:
        source_text = (
            "作品名\n"
            "著者名\n"
            "-------------------------------------------------------\n"
            "【テキスト中に現れる記号について】\n"
            "-------------------------------------------------------\n"
            "本文中で「底本：昔の情報」と述べます。\n"
            "底本：テスト\n"
            "入力：テスト入力者\n"
            "校正：テスト校正者\n"
            "青空文庫作成ファイル：テスト\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sample.txt"
            output_path = root / "corpus.txt"
            input_path.write_bytes(source_text.encode("shift_jis"))

            manifest = import_aozora(input_path, output_path)

            output = output_path.read_text(encoding="utf-8")
            self.assertEqual(output, "本文中で「底本：昔の情報」と述べます。\n")
            self.assertIsNone(manifest["input_member"])
            self.assertEqual(manifest["source"], str(input_path.resolve()))
            self.assertTrue((root / "corpus.manifest.json").is_file())
            self.assertEqual(manifest["removed_lines"], 9)
            self.assertEqual(manifest["output_lines"], 1)
            self.assertIn("底本：昔の情報", output)
            self.assertNotIn("底本：テスト", output)
            self.assertNotIn("入力：", output)
            self.assertNotIn("校正：", output)
            self.assertNotIn(
                "青空文庫作成ファイル：",
                output,
            )


if __name__ == "__main__":
    unittest.main()
