"""Wikimedia Wikipedia日本語parquetから再現可能な学習本文を抽出する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from _common import repo_path

from my_little_japanese_llm.corpus import normalize_line
from my_little_japanese_llm.tokenizer import load_processor

DATASET_ID = "wikimedia/wikipedia"
DATASET_REVISION = "b8e579a0c09383e0e254c9980d56833d16048707"
DATASET_SUBSET = "20231101.ja"
DATASET_LICENSE = "CC BY-SA 3.0 and GFDL"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
WIKIMEDIA_DUMPS_URL = "https://dumps.wikimedia.org/"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parquet_batches(path: Path) -> Iterable[Any]:
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError(
            "parquet抽出にはpyarrowが必要です。.venv/bin/python -m pip install -e '.[data]' を実行してください"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    return parquet_file.iter_batches(
        columns=["title", "text", "url"], batch_size=1024
    )


def _article_text(title: object, text: object) -> str:
    normalized_title = normalize_line(str(title)) if title is not None else ""
    normalized_text = normalize_line(str(text)) if text is not None else ""
    if not normalized_text:
        return ""
    # 一部の記事本文はタイトルを省略して始まるため補う。すでに本文先頭に
    # タイトルがある場合は重複を避け、Wikipedia特有の二重化を防ぐ。
    if normalized_title and not normalized_text.startswith(normalized_title):
        return f"{normalized_title} {normalized_text}"
    return normalized_text


def import_parquet(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    skip_rows: int = 0,
    max_tokens: int | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    tokenizer_file = (
        Path(tokenizer_path).expanduser().resolve()
        if tokenizer_path is not None
        else None
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"入力parquetが見つかりません: {input_file}")
    if skip_rows < 0:
        raise ValueError("skip_rowsは0以上で指定してください")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokensは正の整数で指定してください")
    if max_tokens is not None and tokenizer_file is None:
        raise ValueError("max_tokensを指定した場合はtokenizer_pathが必要です")

    processor = load_processor(tokenizer_file) if tokenizer_file is not None else None
    selected: list[str] = []
    seen: set[str] = set()
    scanned_rows = 0
    skipped_rows = 0
    empty_rows = 0
    duplicate_rows = 0
    selected_tokens = 0
    stopped_by_budget = False

    for batch in _load_parquet_batches(input_file):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        for title, text in zip(titles, texts, strict=True):
            row_index = scanned_rows
            scanned_rows += 1
            if row_index < skip_rows:
                skipped_rows += 1
                continue
            document = _article_text(title, text)
            if not document:
                empty_rows += 1
                continue
            if document in seen:
                duplicate_rows += 1
                continue
            token_cost = (
                len(processor.encode(document, out_type=int)) + 1
                if processor is not None
                else 0
            )
            if max_tokens is not None and selected_tokens + token_cost > max_tokens:
                stopped_by_budget = True
                break
            seen.add(document)
            selected.append(document)
            selected_tokens += token_cost
        if stopped_by_budget:
            break

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    manifest = {
        "format": "wikimedia-wikipedia-ja-extract-v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_subset": DATASET_SUBSET,
        "dataset_url": DATASET_URL,
        "source_dumps_url": WIKIMEDIA_DUMPS_URL,
        "license": DATASET_LICENSE,
        "input_path": str(input_file),
        "input_size_bytes": input_file.stat().st_size,
        "input_sha256": sha256_file(input_file),
        "skip_rows": skip_rows,
        "max_tokens": max_tokens,
        "tokenizer_path": str(tokenizer_file) if tokenizer_file else None,
        "tokenizer_sha256": sha256_file(tokenizer_file)
        if tokenizer_file is not None
        else None,
        "title_prefix_policy": "prefix title when article text does not start with it",
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_characters": sum(map(len, selected)),
        "output_lines": len(selected),
        "scanned_rows": scanned_rows,
        "skipped_rows": skipped_rows,
        "empty_rows": empty_rows,
        "duplicate_rows_removed": duplicate_rows,
        "selected_documents": len(selected),
        "selected_tokens": selected_tokens if processor is not None else None,
        "stopped_by_token_budget": stopped_by_budget,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="入力parquet")
    parser.add_argument("--output", required=True, help="抽出したUTF-8本文")
    parser.add_argument("--manifest", required=True, help="抽出条件のJSON")
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    manifest = import_parquet(
        repo_path(args.input),
        repo_path(args.output),
        repo_path(args.manifest),
        skip_rows=args.skip_rows,
        max_tokens=args.max_tokens,
        tokenizer_path=repo_path(args.tokenizer) if args.tokenizer else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
