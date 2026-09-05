"""MLXでTinyJapaneseGPTを学習し、metrics/checkpoint/sampleを保存する。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.data import load_tokens, make_batch
from my_little_japanese_llm.model import TinyJapaneseGPT, require_mlx
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.training import (
    causal_lm_loss,
    evaluate_loss,
    generate_ids,
    learning_rate,
    perplexity,
    save_checkpoint,
    signature_from_config,
)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="短いsmoke用の一時的な上書き"
    )
    args = parser.parse_args()

    # --help時にはここまで到達しないため、MLXは学習実行時だけ要求する。
    mx = require_mlx()
    from mlx import nn, optimizers

    config = load_config(repo_path(args.config))
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    train_tokens = load_tokens(config.paths.train_tokens)
    val_tokens = load_tokens(config.paths.val_tokens)
    max_steps = (
        args.max_steps if args.max_steps is not None else config.training.max_steps
    )
    if max_steps <= 0 or max_steps > 1_000_000:
        raise ValueError("max_stepsは1以上1,000,000以下で指定してください")

    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    mx.random.seed(config.training.seed)
    rng = np.random.default_rng(config.training.seed)
    model = TinyJapaneseGPT(
        vocab_size=vocab_size,
        dim=config.model.dim,
        layers=config.model.layers,
        heads=config.model.heads,
        context_length=config.model.context_length,
        mlp_ratio=config.model.mlp_ratio,
        position_embedding=config.model.position_embedding,
        norm_type=config.model.norm_type,
    )
    optimizer = optimizers.AdamW(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
    signature = signature_from_config(config, vocab_size)
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.paths.checkpoint_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    started = time.monotonic()

    def write_sample(step: int) -> str:
        prompt_ids = processor.encode(config.generation.prompt, out_type=int)
        ids = generate_ids(
            model,
            prompt_ids,
            config.generation.max_new_tokens,
            config.model.context_length,
            config.generation.temperature,
            config.generation.top_k,
            config.training.seed + step,
            int(processor.eos_id()),
        )
        text = processor.decode(ids)
        sample_path = config.paths.samples_dir / f"step_{step:06d}.txt"
        sample_path.write_text(
            f"prompt: {config.generation.prompt}\n\n{text}\n", encoding="utf-8"
        )
        return text

    write_sample(0)
    best_validation_loss = float("inf")
    best_checkpoint: Path | None = None
    latest_metrics: dict = {}

    for step in range(1, max_steps + 1):
        inputs, targets = make_batch(
            train_tokens,
            config.training.batch_size,
            config.model.context_length,
            rng,
            mx,
        )
        lr = learning_rate(
            step - 1,
            max_steps,
            config.training.learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
        )
        optimizer.learning_rate = lr
        loss, gradients = loss_and_grad(model, inputs, targets)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

        should_log = (
            step == 1
            or step % min(config.training.eval_interval, 1000) == 0
            or step == max_steps
        )
        should_sample = step % config.training.sample_interval == 0 or step == max_steps
        if should_log:
            train_loss = float(loss.item())
            validation_loss = evaluate_loss(
                model,
                val_tokens,
                config.training.batch_size,
                config.model.context_length,
                config.training.eval_batches,
            )
            latest_metrics = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity(validation_loss),
                "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - started,
            }
            print(json.dumps(latest_metrics, ensure_ascii=False))
            _append_jsonl(metrics_path, latest_metrics)
            checkpoint = save_checkpoint(
                model,
                config.paths.checkpoint_dir / f"step_{step:06d}.npz",
                signature,
                latest_metrics,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_checkpoint = checkpoint
        if should_sample:
            write_sample(step)

    summary = {
        "final_step": max_steps,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": time.monotonic() - started,
    }
    (config.paths.checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
