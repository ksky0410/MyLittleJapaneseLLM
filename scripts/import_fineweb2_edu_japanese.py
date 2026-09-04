"""FineWeb2 Edu Japaneseのparquetから再現可能なUTF-8本文を抽出する。"""

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

DATASET_ID = "hotchpotch/fineweb-2-edu-japanese"
DATASET_COMMIT = "180ca004c6a89b590daaad86cb062a07a5353c69"
DATASET_LICENSE = "ODC-By 1.0"
DATASET_TERMS = "Common Crawl Terms of Use"
DATASET_SUBSET = "small_tokens_cleaned"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    return parquet_file.iter_batches(columns=["text"], batch_size=1024)


def _select_documents(
    batches: Iterable[Any],
    *,
    skip_rows: int,
    max_tokens: int | None,
    processor: Any | None,
) -> tuple[list[str], dict[str, int]]:
    if skip_rows < 0:
        raise ValueError("skip_rowsは0以上で指定してください")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokensは正の整数で指定してください")
    if max_tokens is not None and processor is None:
        raise ValueError("max_tokensを指定した場合はTokenizerが必要です")

    selected: list[str] = []
    seen: set[str] = set()
    scanned_rows = 0
    skipped_rows = 0
    empty_rows = 0
    duplicate_rows = 0
    selected_tokens = 0
    for batch in batches:
        for raw_text in batch.column("text").to_pylist():
            row_index = scanned_rows
            scanned_rows += 1
            if row_index < skip_rows:
                skipped_rows += 1
                continue
            text = normalize_line(str(raw_text)) if raw_text is not None else ""
            if not text:
                empty_rows += 1
                continue
            if text in seen:
                duplicate_rows += 1
                continue
            token_cost = (
                len(processor.encode(text, out_type=int)) + 1
                if processor is not None
                else 0
            )
            if max_tokens is not None and selected_tokens + token_cost > max_tokens:
                break
            seen.add(text)
            selected.append(text)
            selected_tokens += token_cost
        else:
            continue
        break

    return selected, {
        "scanned_rows": scanned_rows,
        "skipped_rows": skipped_rows,
        "empty_rows": empty_rows,
        "duplicate_rows_removed": duplicate_rows,
        "selected_documents": len(selected),
        "selected_tokens": selected_tokens if processor is not None else 0,
    }


def import_parquet(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    skip_rows: int = 20_000,
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
    if max_tokens is not None and tokenizer_file is None:
        raise ValueError("max_tokensを指定した場合はtokenizer_pathが必要です")
    processor = load_processor(tokenizer_file) if tokenizer_file is not None else None
    selected, counts = _select_documents(
        _load_parquet_batches(input_file),
        skip_rows=skip_rows,
        max_tokens=max_tokens,
        processor=processor,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    output_sha256 = sha256_file(output_file)
    manifest = {
        "format": "fineweb2-edu-japanese-extract-v1",
        "dataset_id": DATASET_ID,
        "dataset_commit": DATASET_COMMIT,
        "dataset_subset": DATASET_SUBSET,
        "license": DATASET_LICENSE,
        "additional_terms": DATASET_TERMS,
        "input_path": str(input_file),
        "input_size_bytes": input_file.stat().st_size,
        "input_sha256": sha256_file(input_file),
        "skip_rows": skip_rows,
        "max_tokens": max_tokens,
        "tokenizer_path": str(tokenizer_file) if tokenizer_file else None,
        "tokenizer_sha256": sha256_file(tokenizer_file)
        if tokenizer_file is not None
        else None,
        "output_path": str(output_file),
        "output_sha256": output_sha256,
        "output_characters": sum(map(len, selected)),
        "output_lines": len(selected),
        **counts,
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
    parser.add_argument("--skip-rows", type=int, default=20_000)
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
