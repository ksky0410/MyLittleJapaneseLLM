"""複数のuint32 Token列を決定的に交互連結し、manifestへ記録する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import repo_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concat_token_bins(
    inputs: list[tuple[str, str | Path]],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    chunk_tokens: int,
) -> dict[str, Any]:
    """入力Token列をchunk単位でラウンドロビンに並べて連結する。"""

    if len(inputs) < 2:
        raise ValueError("入力Token列を少なくとも2つ指定してください")
    if chunk_tokens < 1:
        raise ValueError("chunk_tokensは1以上の整数で指定してください")
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    arrays: list[tuple[str, Path, np.ndarray]] = []
    input_stats: dict[str, Any] = {}
    for name, raw_path in inputs:
        input_file = repo_path(raw_path).resolve()
        if not input_file.is_file():
            raise FileNotFoundError(f"入力Token列が見つかりません: {input_file}")
        values = np.fromfile(input_file, dtype=np.uint32)
        if values.size == 0:
            raise ValueError(f"入力Token列が空です: {input_file}")
        arrays.append((name, input_file, values))
        input_stats[name] = {
            "path": str(input_file),
            "sha256": sha256_file(input_file),
            "token_count": int(values.size),
        }

    chunks: list[np.ndarray] = []
    positions = [0] * len(arrays)
    remaining = True
    while remaining:
        remaining = False
        for index, (_, _, values) in enumerate(arrays):
            start = positions[index]
            if start >= values.size:
                continue
            end = min(start + chunk_tokens, values.size)
            chunks.append(values[start:end])
            positions[index] = end
            remaining = True
    combined = np.concatenate(chunks).astype(np.uint32, copy=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.tofile(output_file)
    manifest = {
        "format": "uint32-token-concat-round-robin-v1",
        "chunk_tokens": chunk_tokens,
        "inputs": input_stats,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_token_count": int(combined.size),
        "dtype": "uint32",
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_input(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"入力はNAME=PATH形式で指定してください: {value}")
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunk-tokens", type=int, default=262144)
    args = parser.parse_args()
    inputs = [parse_input(value) for value in args.input]
    print(
        json.dumps(
            concat_token_bins(
                inputs,
                args.output,
                args.manifest,
                chunk_tokens=args.chunk_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
