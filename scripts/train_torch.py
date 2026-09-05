"""PyTorch/CUDAでTinyJapaneseGPTを学習し、MLX版と同じ軽量成果物を保存する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.data import load_tokens
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.torch_model import TorchJapaneseGPT, parameter_count, require_torch
from my_little_japanese_llm.training import learning_rate, perplexity, signature_from_config


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_info(torch: object, device: object, amp_enabled: bool) -> dict[str, object]:
    info: dict[str, object] = {
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


def _batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    device: object,
    torch: object,
) -> tuple[object, object]:
    if tokens.size <= context_length:
        raise ValueError("学習Token列がcontext_length以下です")
    starts = rng.integers(0, tokens.size - context_length, size=batch_size)
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def _evaluation_batches(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
    device: object,
    torch: object,
) -> list[tuple[object, object]]:
    if tokens.size <= context_length:
        raise ValueError("検証Token列がcontext_length以下です")
    available = tokens.size - context_length
    starts = np.linspace(0, available - 1, num=max(1, batches), dtype=np.int64)
    result = []
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset : offset + batch_size]
        inputs = np.stack([tokens[start : start + context_length] for start in selected])
        targets = np.stack(
            [tokens[start + 1 : start + context_length + 1] for start in selected]
        )
        result.append(
            (
                torch.as_tensor(inputs, dtype=torch.long, device=device),
                torch.as_tensor(targets, dtype=torch.long, device=device),
            )
        )
    return result


def _loss(model: object, inputs: object, targets: object, F: object) -> object:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def _schedule_steps(
    step: int,
    max_steps: int,
    eval_interval: int,
    sample_interval: int,
    checkpoint_interval: int | None,
) -> tuple[bool, bool, bool]:
    """評価・生成・checkpointの節目を独立に決める。"""

    should_evaluate = (
        step == 1
        or step % min(eval_interval, 1000) == 0
        or step == max_steps
    )
    if checkpoint_interval is None:
        should_checkpoint = should_evaluate
    else:
        should_checkpoint = step % checkpoint_interval == 0 or step == max_steps
    should_sample = step % sample_interval == 0 or step == max_steps
    return should_evaluate, should_checkpoint, should_sample


def _evaluate(
    model: object,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
    device: object,
    torch: object,
    F: object,
    autocast_context: object,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in _evaluation_batches(
            tokens, batch_size, context_length, batches, device, torch
        ):
            with autocast_context():
                loss = _loss(model, inputs, targets, F)
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def _generate(
    model: object,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_k: int,
    seed: int,
    eos_id: int | None,
    device: object,
    torch: object,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("promptを少なくとも1 token指定してください")
    rng = np.random.default_rng(seed)
    output = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = output[-context_length:]
            tokens = torch.as_tensor([context], dtype=torch.long, device=device)
            logits = model(tokens)[0, -1].float().cpu().numpy().astype(np.float64)
            values = logits / temperature
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto、cuda、cpuのいずれか")
    parser.add_argument("--no-amp", action="store_true", help="CUDAでもfloat32で実行")
    args = parser.parse_args()

    torch = require_torch()
    import torch.nn.functional as F

    config = load_config(repo_path(args.config))
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    train_tokens = load_tokens(config.paths.train_tokens)
    val_tokens = load_tokens(config.paths.val_tokens)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cudaが指定されましたがCUDAが利用できません")
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
    ).to(device)
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

    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = (
        config.training.checkpoint_interval or config.training.eval_interval
    )
    metrics_path = config.paths.checkpoint_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    started = time.monotonic()
    signature = signature_from_config(config, vocab_size)
    input_hashes = {
        "config_sha256": _sha256_file(config.source_path),
        "tokenizer_sha256": _sha256_file(config.paths.tokenizer_model),
        "train_tokens_sha256": _sha256_file(config.paths.train_tokens),
        "val_tokens_sha256": _sha256_file(config.paths.val_tokens),
    }

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
        sample_path = config.paths.samples_dir / f"step_{step:06d}.txt"
        sample_path.write_text(
            f"prompt: {config.generation.prompt}\n\n{generated}\n", encoding="utf-8"
        )
        return generated

    write_sample(0)
    best_validation_loss = float("inf")
    best_checkpoint_step: int | None = None
    best_checkpoint: Path | None = None
    latest_metrics: dict = {}
    print(
        json.dumps(
            {
                "backend": "pytorch",
                "device": str(device),
                "amp_enabled": amp_enabled,
                "parameter_count": parameter_count(model),
            },
            ensure_ascii=False,
        )
    )

    for step in range(1, max_steps + 1):
        model.train()
        inputs, targets = _batch(
            train_tokens,
            config.training.batch_size,
            config.model.context_length,
            rng,
            device,
            torch,
        )
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
            loss = _loss(model, inputs, targets, F)
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
            validation_loss = _evaluate(
                model,
                val_tokens,
                config.training.batch_size,
                config.model.context_length,
                config.training.eval_batches,
                device,
                torch,
                F,
                autocast_context,
            )
            latest_metrics = {
                "step": step,
                "train_loss": float(loss.detach().float().item()),
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity(validation_loss),
                "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - started,
            }
            print(json.dumps(latest_metrics, ensure_ascii=False))
            _append_jsonl(metrics_path, latest_metrics)
        is_best = (
            validation_loss is not None and validation_loss < best_validation_loss
        )

        def save_checkpoint(checkpoint: Path, role: str) -> None:
            torch.save(model.state_dict(), checkpoint)
            runtime = _runtime_info(torch, device, amp_enabled)
            metadata = {
                "format_version": 1,
                "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch",
                "checkpoint_role": role,
                "checkpoint_step": step,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_train_loss": float(loss.detach().float().item()),
                "weights_file": checkpoint.name,
                "weights_bytes": checkpoint.stat().st_size,
                "weights_sha256": _sha256_file(checkpoint),
                "model": signature,
                "metrics": latest_metrics,
                "inputs": input_hashes,
                "runtime": runtime,
                "training_intervals": {
                    "eval_interval": config.training.eval_interval,
                    "sample_interval": config.training.sample_interval,
                    "checkpoint_interval": checkpoint_interval,
                },
            }
            checkpoint.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if should_checkpoint:
            checkpoint = config.paths.checkpoint_dir / f"step_{step:06d}.pt"
            save_checkpoint(checkpoint, "periodic")
            if is_best:
                best_checkpoint_step = step
                best_validation_loss = float(validation_loss)
                best_checkpoint = checkpoint
        elif is_best:
            best_checkpoint = config.paths.checkpoint_dir / "best.pt"
            save_checkpoint(best_checkpoint, "best")
            best_checkpoint_step = step
            best_validation_loss = float(validation_loss)

        if (
            is_best
            and should_checkpoint
            and config.training.checkpoint_interval is not None
        ):
            best_snapshot = config.paths.checkpoint_dir / "best.pt"
            save_checkpoint(best_snapshot, "best")
            best_checkpoint = best_snapshot
        if should_sample:
            write_sample(step)

    summary = {
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
        "best_checkpoint": (
            best_checkpoint.name if best_checkpoint is not None else None
        ),
        "best_checkpoint_step": best_checkpoint_step,
        "best_validation_loss": best_validation_loss,
        "checkpoint_interval": checkpoint_interval,
        "elapsed_seconds": time.monotonic() - started,
    }
    (config.paths.checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
