"""MLX学習で共有するloss、評価、生成、checkpoint処理。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .data import evaluation_batches
from .model import model_signature, require_mlx


def causal_lm_loss(model: Any, inputs: Any, targets: Any) -> Any:
    """logitsと正解Tokenから平均causal cross entropyを計算する。"""

    mx = require_mlx()
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    log_normalizer = mx.logsumexp(flat_logits, axis=-1)
    correct = mx.take_along_axis(flat_logits, flat_targets[:, None], axis=1).squeeze(-1)
    return mx.mean(log_normalizer - correct)


def learning_rate(
    step: int, max_steps: int, base: float, minimum: float, warmup_steps: int
) -> float:
    if step < warmup_steps:
        return base * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def evaluate_loss(
    model: Any,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
) -> float:
    mx = require_mlx()
    losses = []
    for inputs, targets in evaluation_batches(
        tokens, batch_size, context_length, batches, mx
    ):
        loss = causal_lm_loss(model, inputs, targets)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


def generate_ids(
    model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_k: int,
    seed: int,
    eos_id: int | None = None,
) -> list[int]:
    """MLX forwardとnumpy samplingだけで再現可能な生成を行う。"""

    mx = require_mlx()
    if not prompt_ids:
        raise ValueError("promptを少なくとも1 token指定してください")
    if max_new_tokens <= 0 or temperature <= 0 or top_k < 0:
        raise ValueError("生成設定が不正です")
    rng = np.random.default_rng(seed)
    output = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = output[-context_length:]
        logits = model(mx.array([context]))[:, -1, :]
        mx.eval(logits)
        values = np.asarray(logits[0], dtype=np.float64) / temperature
        if top_k > 0 and top_k < values.size:
            candidate_ids = np.argpartition(values, -top_k)[-top_k:]
            candidate_values = values[candidate_ids]
        else:
            candidate_ids = np.arange(values.size)
            candidate_values = values
        candidate_values = candidate_values - candidate_values.max()
        probabilities = np.exp(candidate_values)
        probabilities /= probabilities.sum()
        next_id = int(rng.choice(candidate_ids, p=probabilities))
        output.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return output


def save_checkpoint(
    model: Any, path: str | Path, signature: dict[str, int], metrics: dict[str, Any]
) -> Path:
    """重みとJSON metadataを別々に保存する。pickleを使わず、ロード前に形状を検証する。"""

    weights_path = Path(path)
    if weights_path.suffix != ".npz":
        raise ValueError("checkpointの重みファイルは.npzを指定してください")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    metadata = {
        "format_version": 1,
        "weights_file": weights_path.name,
        "model": signature,
        "metrics": metrics,
    }
    metadata_path = weights_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return weights_path


def load_checkpoint(
    model: Any, path: str | Path, expected_signature: dict[str, int]
) -> dict[str, Any]:
    """metadataを先に検証してからMLX weightをロードする。"""

    weights_path = Path(path)
    metadata_path = weights_path.with_suffix(".json")
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpointが見つかりません: {weights_path}")
    if not metadata_path.is_file():
        raise ValueError(f"checkpoint metadataがありません: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != 1
        or metadata.get("weights_file") != weights_path.name
    ):
        raise ValueError(f"checkpoint metadataの形式が不正です: {metadata_path}")
    actual_signature = metadata.get("model")
    if actual_signature != expected_signature:
        raise ValueError(
            "checkpointと現在の設定が一致しません。"
            f" expected={expected_signature}, actual={actual_signature}"
        )
    require_mlx()
    model.load_weights(str(weights_path))
    return metadata


def signature_from_config(config: Any, vocab_size: int) -> dict[str, int]:
    return model_signature(
        vocab_size,
        config.model.dim,
        config.model.layers,
        config.model.heads,
        config.model.context_length,
        config.model.mlp_ratio,
    )
