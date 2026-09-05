"""PyTorchで応答部分だけを学習する会話SFTを実行する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from _common import repo_path
from train_torch import _generate, _schedule_steps

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.sft import (
    load_rehearsal_tokens,
    load_sft_arrays,
    split_sft_rehearsal_batch_size,
    validate_rehearsal_ratio,
)
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.torch_model import (
    TorchJapaneseGPT,
    parameter_count,
    require_torch,
)
from my_little_japanese_llm.training import (
    learning_rate,
    perplexity,
    signature_from_config,
)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rehearsal_options(
    rehearsal_tokens: str | None, rehearsal_ratio: float | None
) -> tuple[str | None, float]:
    """rehearsal用CLI引数の組み合わせを検証する。"""

    if (rehearsal_tokens is None) != (rehearsal_ratio is None):
        raise ValueError(
            "--rehearsal-tokensと--rehearsal-ratioは必ず同時に指定してください"
        )
    if rehearsal_ratio is None:
        return None, 0.0
    return rehearsal_tokens, validate_rehearsal_ratio(rehearsal_ratio)


def build_parser() -> argparse.ArgumentParser:
    """train_sft_torchのCLI parserを作る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/token-budget-chat-sft-5m-smoke.toml"
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto、cuda、cpuのいずれか")
    parser.add_argument("--no-amp", action="store_true", help="CUDAでもfloat32で実行")
    parser.add_argument(
        "--rehearsal-tokens",
        default=None,
        help="通常のraw uint32 Token列。rehearsal-ratioと同時に指定する",
    )
    parser.add_argument(
        "--rehearsal-ratio",
        type=float,
        default=None,
        help="rehearsal lossの重み（0以上1未満）",
    )
    return parser


def _masked_cross_entropy(logits: Any, targets: Any, loss_mask: Any, functional: Any) -> Any:
    """loss_maskが1の位置だけで平均したcross entropyを返す。"""

    if logits.ndim != 3:
        raise ValueError("logitsは[batch, sequence, vocab]の3次元で指定してください")
    if targets.shape != logits.shape[:2] or loss_mask.shape != logits.shape[:2]:
        raise ValueError("targetsとloss_maskはlogitsの先頭2次元と同じshapeが必要です")
    token_losses = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1).long(),
        reduction="none",
    )
    flat_mask = loss_mask.reshape(-1).to(device=logits.device, dtype=token_losses.dtype)
    denominator = flat_mask.sum().clamp_min(1.0)
    return (token_losses * flat_mask).sum() / denominator


def masked_causal_lm_loss(
    model: Any, inputs: Any, targets: Any, loss_mask: Any, functional: Any
) -> Any:
    """SFTのresponse-only lossを計算する。"""

    return _masked_cross_entropy(model(inputs), targets, loss_mask, functional)


def full_causal_lm_loss(model: Any, inputs: Any, targets: Any, functional: Any) -> Any:
    """rehearsal用の通常next-token lossを計算する。"""

    logits = model(inputs)
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1).long()
    )


def _sft_batch(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    device: Any,
    torch: Any,
) -> tuple[Any, Any, Any]:
    count = arrays["input_ids"].shape[0]
    if count == 0:
        raise ValueError("SFTデータが空です")
    indices = rng.integers(0, count, size=batch_size)
    return (
        torch.as_tensor(arrays["input_ids"][indices], dtype=torch.long, device=device),
        torch.as_tensor(arrays["target_ids"][indices], dtype=torch.long, device=device),
        torch.as_tensor(
            arrays["loss_mask"][indices], dtype=torch.float32, device=device
        ),
    )


def _evaluation_sft_batches(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    batches: int,
    device: Any,
    torch: Any,
) -> list[tuple[Any, Any, Any]]:
    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_sizeとbatchesは正の整数で指定してください")
    count = arrays["input_ids"].shape[0]
    if count == 0:
        raise ValueError("SFTデータが空です")
    indices = np.linspace(0, count - 1, num=min(count, batches * batch_size), dtype=np.int64)
    result = []
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset : offset + batch_size]
        result.append(
            (
                torch.as_tensor(
                    arrays["input_ids"][selected], dtype=torch.long, device=device
                ),
                torch.as_tensor(
                    arrays["target_ids"][selected], dtype=torch.long, device=device
                ),
                torch.as_tensor(
                    arrays["loss_mask"][selected],
                    dtype=torch.float32,
                    device=device,
                ),
            )
        )
    return result


def _evaluate_sft(
    model: Any,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    batches: int,
    device: Any,
    torch: Any,
    functional: Any,
    autocast_context: Any,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets, loss_mask in _evaluation_sft_batches(
            arrays, batch_size, batches, device, torch
        ):
            with autocast_context():
                loss = masked_causal_lm_loss(
                    model, inputs, targets, loss_mask, functional
                )
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def _load_base_checkpoint(
    model: Any,
    checkpoint_path: Path,
    expected_signature: dict[str, int | str],
    torch: Any,
) -> dict[str, Any]:
    """train_torch形式のmetadataを検証してPyTorch checkpointをreloadする。"""

    metadata_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"base checkpointが見つかりません: {checkpoint_path}")
    if not metadata_path.is_file():
        raise ValueError(f"base checkpoint metadataがありません: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != 1
        or metadata.get("weights_file") != checkpoint_path.name
    ):
        raise ValueError(f"base checkpoint metadataの形式が不正です: {metadata_path}")
    actual_signature = metadata.get("model")
    if isinstance(actual_signature, dict):
        actual_signature = {
            "position_embedding": "absolute",
            "norm_type": "layernorm",
            "ffn_type": "gelu",
            **actual_signature,
        }
    if actual_signature != expected_signature:
        raise ValueError(
            "base checkpointと現在の設定が一致しません。"
            f" expected={expected_signature}, actual={actual_signature}"
        )
    recorded_hash = metadata.get("weights_sha256")
    if recorded_hash is not None and recorded_hash != _sha256_file(checkpoint_path):
        raise ValueError(f"base checkpointのSHA-256がmetadataと一致しません: {checkpoint_path}")
    model_device = next(model.parameters()).device
    state_dict = torch.load(
        checkpoint_path, map_location=model_device, weights_only=True
    )
    if not isinstance(state_dict, dict):
        raise TypeError(f"base checkpointがstate_dictではありません: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return metadata


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("--deviceはauto、cuda、cpuのいずれかで指定してください")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cudaが指定されましたがCUDAが利用できません")
    return device


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rehearsal_tokens_arg, rehearsal_ratio = validate_rehearsal_options(
            args.rehearsal_tokens, args.rehearsal_ratio
        )
    except ValueError as error:
        parser.error(str(error))

    torch = require_torch()
    from torch.nn import functional

    config = load_config(repo_path(args.config))
    device = _device(torch, args.device)
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    train_arrays = load_sft_arrays(args.train_data, config.model.context_length)
    validation_arrays = load_sft_arrays(
        args.validation_data, config.model.context_length
    )
    rehearsal_path = (
        repo_path(rehearsal_tokens_arg).resolve()
        if rehearsal_tokens_arg is not None
        else None
    )
    rehearsal_tokens = (
        load_rehearsal_tokens(rehearsal_path) if rehearsal_path is not None else None
    )
    max_steps = args.max_steps if args.max_steps is not None else config.training.max_steps
    if max_steps <= 0 or max_steps > 1_000_000:
        raise ValueError("max_stepsは1以上1,000,000以下で指定してください")

    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
        torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(config.training.seed)
    model = TorchJapaneseGPT(
        vocab_size=vocab_size,
        dim=config.model.dim,
        layers=config.model.layers,
        heads=config.model.heads,
        context_length=config.model.context_length,
        mlp_ratio=config.model.mlp_ratio,
        position_embedding=config.model.position_embedding,
        norm_type=config.model.norm_type,
        ffn_type=config.model.ffn_type,
    ).to(device)
    signature = signature_from_config(config, vocab_size)
    base_checkpoint = repo_path(args.base_checkpoint).resolve()
    base_metadata = _load_base_checkpoint(model, base_checkpoint, signature, torch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def autocast_context():
        return torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        )

    output_dir = repo_path(args.output_dir).resolve()
    samples_dir = repo_path(args.samples_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    started = time.monotonic()
    checkpoint_interval = (
        config.training.checkpoint_interval or config.training.eval_interval
    )
    input_hashes = {
        "config_sha256": _sha256_file(config.source_path),
        "tokenizer_sha256": _sha256_file(config.paths.tokenizer_model),
        "train_data_sha256": _sha256_file(repo_path(args.train_data).resolve()),
        "validation_data_sha256": _sha256_file(
            repo_path(args.validation_data).resolve()
        ),
        "base_checkpoint_sha256": _sha256_file(base_checkpoint),
    }
    if rehearsal_path is not None:
        input_hashes["rehearsal_tokens_sha256"] = _sha256_file(rehearsal_path)

    def write_sample(step: int) -> str:
        prompt_ids = processor.encode(config.generation.prompt, out_type=int)
        ids = _generate(
            model,
            prompt_ids,
            config.generation.max_new_tokens,
            config.model.context_length,
            config.generation.temperature,
            config.generation.top_k,
            config.training.seed + step,
            int(processor.eos_id()),
            device,
            torch,
        )
        generated = processor.decode(ids)
        (samples_dir / f"step_{step:06d}.txt").write_text(
            f"prompt: {config.generation.prompt}\n\n{generated}\n", encoding="utf-8"
        )
        return generated

    write_sample(0)
    rehearsal_active = rehearsal_tokens is not None and rehearsal_ratio > 0
    if rehearsal_active:
        sft_batch_size, rehearsal_batch_size = split_sft_rehearsal_batch_size(
            config.training.batch_size, rehearsal_ratio
        )
    else:
        sft_batch_size = config.training.batch_size
        rehearsal_batch_size = 0
    best_validation_loss = float("inf")
    best_checkpoint_step: int | None = None
    best_checkpoint: Path | None = None
    latest_metrics: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        model.train()
        sft_inputs, sft_targets, sft_loss_mask = _sft_batch(
            train_arrays, sft_batch_size, rng, device, torch
        )
        if rehearsal_active:
            rehearsal_inputs, rehearsal_targets = _batch_for_rehearsal(
                rehearsal_tokens,
                rehearsal_batch_size,
                config.model.context_length,
                rng,
                device,
                torch,
            )
        else:
            rehearsal_inputs = rehearsal_targets = None
        lr = learning_rate(
            step - 1,
            max_steps,
            config.training.learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            sft_loss = masked_causal_lm_loss(
                model, sft_inputs, sft_targets, sft_loss_mask, functional
            )
            rehearsal_loss = None
            if rehearsal_active:
                rehearsal_loss = full_causal_lm_loss(
                    model, rehearsal_inputs, rehearsal_targets, functional
                )
                loss = (1.0 - rehearsal_ratio) * sft_loss + rehearsal_ratio * rehearsal_loss
            else:
                loss = sft_loss
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        should_evaluate, should_checkpoint, should_sample = _schedule_steps(
            step,
            max_steps,
            config.training.eval_interval,
            config.training.sample_interval,
            config.training.checkpoint_interval,
        )
        validation_loss: float | None = None
        if should_evaluate:
            validation_loss = _evaluate_sft(
                model,
                validation_arrays,
                config.training.batch_size,
                config.training.eval_batches,
                device,
                torch,
                functional,
                autocast_context,
            )
            latest_metrics = {
                "step": step,
                "train_loss": float(loss.detach().float().item()),
                "sft_train_loss": float(sft_loss.detach().float().item()),
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity(validation_loss),
                "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - started,
                "rehearsal_ratio": rehearsal_ratio,
            }
            if rehearsal_loss is not None:
                latest_metrics["rehearsal_train_loss"] = float(
                    rehearsal_loss.detach().float().item()
                )
            print(json.dumps(latest_metrics, ensure_ascii=False))
            _append_jsonl(metrics_path, latest_metrics)
        is_best = (
            validation_loss is not None and validation_loss < best_validation_loss
        )

        def save_checkpoint(
            checkpoint: Path,
            role: str,
            checkpoint_step: int = step,
            checkpoint_loss: Any = loss,
            checkpoint_metrics: dict[str, Any] = latest_metrics,
        ) -> None:
            torch.save(model.state_dict(), checkpoint)
            metadata = {
                "format_version": 1,
                "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch",
                "checkpoint_role": role,
                "checkpoint_step": checkpoint_step,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_train_loss": float(
                    checkpoint_loss.detach().float().item()
                ),
                "weights_file": checkpoint.name,
                "weights_bytes": checkpoint.stat().st_size,
                "weights_sha256": _sha256_file(checkpoint),
                "model": signature,
                "metrics": checkpoint_metrics,
                "inputs": input_hashes,
                "runtime": _runtime_info(torch, device, amp_enabled),
                "training_intervals": {
                    "eval_interval": config.training.eval_interval,
                    "sample_interval": config.training.sample_interval,
                    "checkpoint_interval": checkpoint_interval,
                },
                "base_checkpoint": {
                    "path": str(base_checkpoint),
                    "weights_sha256": base_metadata.get("weights_sha256"),
                },
                "sft": {
                    "train_examples": int(train_arrays["input_ids"].shape[0]),
                    "validation_examples": int(
                        validation_arrays["input_ids"].shape[0]
                    ),
                    "rehearsal_ratio": rehearsal_ratio,
                    "rehearsal_tokens": str(rehearsal_path)
                    if rehearsal_path is not None
                    else None,
                },
            }
            checkpoint.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if should_checkpoint:
            checkpoint = output_dir / f"step_{step:06d}.pt"
            save_checkpoint(checkpoint, "periodic")
            if is_best:
                best_checkpoint_step = step
                best_validation_loss = float(validation_loss)
                best_checkpoint = checkpoint
        elif is_best:
            best_checkpoint = output_dir / "best.pt"
            save_checkpoint(best_checkpoint, "best")
            best_checkpoint_step = step
            best_validation_loss = float(validation_loss)
        if (
            is_best
            and should_checkpoint
            and config.training.checkpoint_interval is not None
        ):
            best_snapshot = output_dir / "best.pt"
            save_checkpoint(best_snapshot, "best")
            best_checkpoint = best_snapshot
        if should_sample:
            write_sample(step)

    summary = {
        "format": "chat-sft-torch-training-v1",
        "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch",
        "runtime": _runtime_info(torch, device, amp_enabled),
        "inputs": input_hashes,
        "parameter_count": parameter_count(model),
        "training_intervals": {
            "eval_interval": config.training.eval_interval,
            "sample_interval": config.training.sample_interval,
            "checkpoint_interval": checkpoint_interval,
        },
        "final_step": max_steps,
        "best_checkpoint": best_checkpoint.name if best_checkpoint is not None else None,
        "best_checkpoint_step": best_checkpoint_step,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(base_checkpoint),
        "train_examples": int(train_arrays["input_ids"].shape[0]),
        "validation_examples": int(validation_arrays["input_ids"].shape[0]),
        "rehearsal_tokens": str(rehearsal_path) if rehearsal_path is not None else None,
        "rehearsal_ratio": rehearsal_ratio,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _batch_for_rehearsal(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    device: Any,
    torch: Any,
) -> tuple[Any, Any]:
    if tokens.size <= context_length:
        raise ValueError("rehearsal Token列がcontext_length以下です")
    starts = rng.integers(0, tokens.size - context_length, size=batch_size)
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def _runtime_info(torch: Any, device: Any, amp_enabled: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp_enabled": amp_enabled,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": properties.total_memory,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return info


if __name__ == "__main__":
    main()
