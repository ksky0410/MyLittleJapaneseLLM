"""会話JSONLから指定sourceだけを抽出した派生JSONLを作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _common import repo_path

SPLITS = ("train", "validation", "test")
SOURCES = ("real-persona-chat", "mrmp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_source(
    input_dir: str | Path,
    output_dir: str | Path,
    source: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """元JSONLを変更せず、datasetフィールド一致の行だけを書き出す。"""

    if source not in SOURCES:
        raise ValueError(f"sourceは{SOURCES}のいずれかで指定してください: {source}")
    input_root = repo_path(input_dir).resolve()
    output_root = repo_path(output_dir).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"入力ディレクトリが見つかりません: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, Any] = {}
    for split in SPLITS:
        input_file = input_root / f"{split}.jsonl"
        if not input_file.is_file():
            raise FileNotFoundError(f"入力JSONLが見つかりません: {input_file}")
        selected: list[str] = []
        input_count = 0
        for line_number, line in enumerate(
            input_file.read_text(encoding="utf-8").splitlines(keepends=True), start=1
        ):
            if not line.strip():
                continue
            input_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONLを読めません: {input_file}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"JSONLのrecordがobjectではありません: {input_file}:{line_number}")
            if record.get("dataset") == source:
                selected.append(line)
        output_file = output_root / f"{split}.jsonl"
        output_file.write_text("".join(selected), encoding="utf-8")
        split_stats[split] = {
            "input_path": str(input_file),
            "input_sha256": sha256_file(input_file),
            "input_record_count": input_count,
            "output_path": str(output_file),
            "output_sha256": sha256_file(output_file),
            "output_record_count": len(selected),
        }

    manifest = {
        "format": "conversation-source-filter-v1",
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "source": source,
        "splits": split_stats,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="元conversation-v1 JSONLディレクトリ")
    parser.add_argument("--output", required=True, help="source別JSONLの出力ディレクトリ")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            filter_source(args.input, args.output, args.source, args.manifest),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
