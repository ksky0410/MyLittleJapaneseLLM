from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_wikimedia_wikipedia_ja import import_parquet


class WikimediaWikipediaImportTests(unittest.TestCase):
    def test_extracts_unique_articles_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.parquet"
            output_path = root / "corpus.txt"
            manifest_path = root / "manifest.json"
            parquet.write_table(
                pa.table(
                    {
                        "title": ["猫", "犬", "空記事", "猫"],
                        "text": [
                            "猫は哺乳類である。",
                            "犬が走る。",
                            None,
                            "猫は哺乳類である。",
                        ],
                        "url": ["a", "b", "c", "d"],
                    }
                ),
                input_path,
            )

            manifest = import_parquet(input_path, output_path, manifest_path)
            output = output_path.read_text(encoding="utf-8")
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(output, "猫は哺乳類である。\n犬が走る。\n")
            self.assertEqual(loaded, manifest)
            self.assertEqual(manifest["selected_documents"], 2)
            self.assertEqual(manifest["empty_rows"], 1)
            self.assertEqual(manifest["duplicate_rows_removed"], 1)
            self.assertEqual(manifest["dataset_subset"], "20231101.ja")
            self.assertEqual(manifest["license"], "CC BY-SA 3.0 and GFDL")
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(input_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
