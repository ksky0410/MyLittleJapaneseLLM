"""Tokenバイナリから固定長のnext-token batchを作る。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_tokens(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Tokenファイルが見つかりません: {source}")
    tokens = np.fromfile(source, dtype=np.uint32)
    if tokens.size < 3:
        raise ValueError(f"Token列が短すぎます（3 token未満）: {source}")
    return tokens.astype(np.int32, copy=False)


def _validate_length(tokens: np.ndarray, context_length: int) -> None:
    if context_length < 2:
        raise ValueError("context_length は2以上で指定してください")
    if tokens.size <= context_length:
        raise ValueError(
            f"Token数{tokens.size}ではcontext_length={context_length}のnext-token batchを作れません。"
            "データを増やすかcontext_lengthを短くしてください。"
        )


def make_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    mx: Any,
) -> tuple[Any, Any]:
    """ランダムな固定長batchをMLX arrayで返す。"""

    _validate_length(tokens, context_length)
    if batch_size <= 0:
        raise ValueError("batch_size は正の整数で指定してください")
    starts = rng.integers(0, tokens.size - context_length, size=batch_size)
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )
    return mx.array(inputs), mx.array(targets)


def evaluation_batches(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
    mx: Any,
) -> list[tuple[Any, Any]]:
    """検証用にデータ全体から決定的な固定長batchを作る。"""

    _validate_length(tokens, context_length)
    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_sizeとbatchesは正の整数で指定してください")
    available = tokens.size - context_length
    starts = np.linspace(0, available - 1, num=max(1, batches), dtype=np.int64)
    result = []
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset : offset + batch_size]
        inputs = np.stack(
            [tokens[start : start + context_length] for start in selected]
        )
        targets = np.stack(
            [tokens[start + 1 : start + context_length + 1] for start in selected]
        )
        result.append((mx.array(inputs), mx.array(targets)))
    return result
