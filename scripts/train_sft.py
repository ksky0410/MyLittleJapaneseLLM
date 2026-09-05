"""応答側loss maskingを使って既存checkpointを会話SFTする。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.model import TinyJapaneseGPT, require_mlx
from my_little_japanese_llm.sft import (
    combined_sft_rehearsal_loss,
    evaluate_sft_loss,
    full_causal_lm_loss,
    load_rehearsal_tokens,
    load_sft_arrays,
    make_sft_batch,
    make_sft_rehearsal_batch,
    masked_causal_lm_loss,
    split_sft_rehearsal_batch_size,
    validate_rehearsal_ratio,
    validate_short_response_options,
)
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.training import (
    generate_ids,
    learning_rate,
    load_checkpoint,
    perplexity,
    save_checkpoint,
    signature_from_config,
)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """train_sftのCLI parserを作る。MLXなしの引数検証にも使える。"""

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
    parser.add_argument(
        "--short-response-ratio",
        type=float,
        default=None,
        help="SFT batchに占める短い応答例の割合（0以上1未満）",
    )
    parser.add_argument(
        "--short-response-max-tokens",
        type=int,
        default=None,
        help="短い応答と判定するloss対象Token数の上限",
    )
    return parser


def validate_rehearsal_options(
    rehearsal_tokens: str | None, rehearsal_ratio: float | None
) -> tuple[str | None, float]:
    """rehearsal CLI引数の組み合わせを検証する。"""

    if (rehearsal_tokens is None) != (rehearsal_ratio is None):
        raise ValueError(
            "--rehearsal-tokensと--rehearsal-ratioは必ず同時に指定してください"
        )
    if rehearsal_ratio is None:
        return None, 0.0
    return rehearsal_tokens, validate_rehearsal_ratio(rehearsal_ratio)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rehearsal_tokens_arg, rehearsal_ratio = validate_rehearsal_options(
            args.rehearsal_tokens, args.rehearsal_ratio
        )
        short_response_ratio, short_response_max_tokens = (
            validate_short_response_options(
                args.short_response_ratio, args.short_response_max_tokens
            )
        )
    except ValueError as error:
        parser.error(str(error))

    mx = require_mlx()
    from mlx import nn, optimizers

    config = load_config(repo_path(args.config))
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
    base_checkpoint = repo_path(args.base_checkpoint).resolve()
    load_checkpoint(model, base_checkpoint, signature_from_config(config, vocab_size))
    optimizer = optimizers.AdamW(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    rehearsal_active = rehearsal_tokens is not None and rehearsal_ratio > 0
    if rehearsal_active:
        sft_batch_size, rehearsal_batch_size = split_sft_rehearsal_batch_size(
            config.training.batch_size, rehearsal_ratio
        )
        loss_and_grad = nn.value_and_grad(model, combined_sft_rehearsal_loss)
    else:
        sft_batch_size = config.training.batch_size
        rehearsal_batch_size = 0
        loss_and_grad = nn.value_and_grad(model, masked_causal_lm_loss)
    signature = signature_from_config(config, vocab_size)
    checkpoint_dir = repo_path(args.output_dir).resolve()
    samples_dir = repo_path(args.samples_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.jsonl"
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
        (samples_dir / f"step_{step:06d}.txt").write_text(
            f"prompt: {config.generation.prompt}\n\n{text}\n", encoding="utf-8"
        )
        return text

    write_sample(0)
    best_validation_loss = float("inf")
    best_checkpoint: Path | None = None
    for step in range(1, max_steps + 1):
        if rehearsal_active:
            (
                sft_inputs,
                sft_targets,
                sft_loss_mask,
                rehearsal_inputs,
                rehearsal_targets,
            ) = make_sft_rehearsal_batch(
                train_arrays,
                rehearsal_tokens,
                config.training.batch_size,
                config.model.context_length,
                rehearsal_ratio,
                rng,
                mx,
                short_response_ratio=short_response_ratio,
                short_response_max_tokens=short_response_max_tokens,
            )
        else:
            inputs, targets, loss_mask = make_sft_batch(
                train_arrays,
                config.training.batch_size,
                rng,
                mx,
                short_response_ratio=short_response_ratio,
                short_response_max_tokens=short_response_max_tokens,
            )
        lr = learning_rate(
            step - 1,
            max_steps,
            config.training.learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
        )
        optimizer.learning_rate = lr
        if rehearsal_active:
            loss, gradients = loss_and_grad(
                model,
                sft_inputs,
                sft_targets,
                sft_loss_mask,
                rehearsal_inputs,
                rehearsal_targets,
                rehearsal_ratio,
            )
        else:
            loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        should_log = (
            step == 1
            or step % min(config.training.eval_interval, 1000) == 0
            or step == max_steps
        )
        should_sample = step % config.training.sample_interval == 0 or step == max_steps
        if should_log:
            validation_loss = evaluate_sft_loss(
                model,
                validation_arrays,
                config.training.batch_size,
                config.training.eval_batches,
                mx,
            )
            metrics = {
                "step": step,
                "train_loss": float(loss.item()),
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity(validation_loss),
                "learning_rate": lr,
                "elapsed_seconds": time.monotonic() - started,
                "rehearsal_tokens": str(rehearsal_path)
                if rehearsal_path is not None
                else None,
                "rehearsal_ratio": rehearsal_ratio,
                "short_response_ratio": short_response_ratio,
                "short_response_max_tokens": short_response_max_tokens,
            }
            if rehearsal_active:
                sft_train_loss = masked_causal_lm_loss(
                    model, sft_inputs, sft_targets, sft_loss_mask
                )
                rehearsal_train_loss = full_causal_lm_loss(
                    model, rehearsal_inputs, rehearsal_targets
                )
                mx.eval(sft_train_loss, rehearsal_train_loss)
                metrics.update(
                    {
                        "sft_train_loss": float(sft_train_loss.item()),
                        "rehearsal_train_loss": float(rehearsal_train_loss.item()),
                        "sft_batch_size": sft_batch_size,
                        "rehearsal_batch_size": rehearsal_batch_size,
                    }
                )
            print(json.dumps(metrics, ensure_ascii=False))
            _append_jsonl(metrics_path, metrics)
            checkpoint = save_checkpoint(
                model,
                checkpoint_dir / f"step_{step:06d}.npz",
                signature,
                {**metrics, "base_checkpoint": str(base_checkpoint)},
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_checkpoint = checkpoint
        if should_sample:
            write_sample(step)

    summary = {
        "format": "chat-sft-training-v1",
        "final_step": max_steps,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(base_checkpoint),
        "train_examples": int(train_arrays["input_ids"].shape[0]),
        "validation_examples": int(validation_arrays["input_ids"].shape[0]),
        "rehearsal_tokens": str(rehearsal_path) if rehearsal_path is not None else None,
        "rehearsal_ratio": rehearsal_ratio,
        "short_response_ratio": short_response_ratio,
        "short_response_max_tokens": short_response_max_tokens,
        "elapsed_seconds": time.monotonic() - started,
    }
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
