"""応答tokenだけへlossをかける会話SFTのbatch処理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .model import require_mlx


def load_sft_arrays(path: str | Path, context_length: int) -> dict[str, np.ndarray]:
    """整形済みSFT npzを検証して読み込む。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SFTデータが見つかりません: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {"input_ids", "target_ids", "loss_mask"}
        if set(data.files) != required:
            raise ValueError(
                f"SFT npzの配列は{sorted(required)}で指定してください: {data.files}"
            )
        arrays = {name: np.asarray(data[name]) for name in required}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"SFT配列のshapeが一致しません: {shapes}")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[1] != context_length:
        raise ValueError(
            f"SFT配列は[N, {context_length}]の2次元で指定してください: {shape}"
        )
    if shape[0] == 0:
        raise ValueError("SFTデータが空です")
    if not np.isfinite(arrays["loss_mask"]).all():
        raise ValueError("loss_maskに有限でない値があります")
    if np.any(arrays["loss_mask"] < 0) or np.any(arrays["loss_mask"] > 1):
        raise ValueError("loss_maskは0から1の範囲で指定してください")
    return {
        "input_ids": arrays["input_ids"].astype(np.int32, copy=False),
        "target_ids": arrays["target_ids"].astype(np.int32, copy=False),
        "loss_mask": arrays["loss_mask"].astype(np.float32, copy=False),
    }


def make_sft_batch(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    mx: Any,
) -> tuple[Any, Any, Any]:
    """整形済みSFT配列から決定的なランダムbatchを作る。"""

    if batch_size <= 0:
        raise ValueError("batch_sizeは正の整数で指定してください")
    count = arrays["input_ids"].shape[0]
    if count == 0:
        raise ValueError("SFTデータが空です")
    indices = rng.integers(0, count, size=batch_size)
    return tuple(
        mx.array(arrays[name][indices])
        for name in ("input_ids", "target_ids", "loss_mask")
    )


def evaluation_sft_batches(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    batches: int,
    mx: Any,
) -> list[tuple[Any, Any, Any]]:
    """SFT validation配列から全体を等間隔に見る固定batchを作る。"""

    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_sizeとbatchesは正の整数で指定してください")
    count = arrays["input_ids"].shape[0]
    if count == 0:
        raise ValueError("SFTデータが空です")
    indices = np.linspace(
        0, count - 1, num=min(count, batches * batch_size), dtype=np.int64
    )
    result = []
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset : offset + batch_size]
        result.append(
            tuple(
                mx.array(arrays[name][selected])
                for name in ("input_ids", "target_ids", "loss_mask")
            )
        )
    return result


def masked_causal_lm_loss(model: Any, inputs: Any, targets: Any, loss_mask: Any) -> Any:
    """loss_maskが1のtarget位置だけでcausal LM lossを平均する。"""

    mx = require_mlx()
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).astype(logits.dtype)
    log_normalizer = mx.logsumexp(flat_logits, axis=-1)
    correct = mx.take_along_axis(flat_logits, flat_targets[:, None], axis=1).squeeze(-1)
    token_losses = log_normalizer - correct
    denominator = mx.maximum(mx.sum(flat_mask), mx.array(1.0, dtype=logits.dtype))
    return mx.sum(token_losses * flat_mask) / denominator


def evaluate_sft_loss(
    model: Any,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    batches: int,
    mx: Any,
) -> float:
    """SFT validationのmask付き平均lossを返す。"""

    losses = []
    for inputs, targets, loss_mask in evaluation_sft_batches(
        arrays, batch_size, batches, mx
    ):
        loss = masked_causal_lm_loss(model, inputs, targets, loss_mask)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses))
