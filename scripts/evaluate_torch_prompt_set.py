"""PyTorch checkpointを固定prompt群で評価し、raw/会話形式の差を記録する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import repo_path
from evaluate_chat_prompts import (
    _encode_prompt,
    _format_text,
    _sha256_file,
    load_prompts,
    summarize_prompt_results,
)
from evaluate_torch import _device, _generate, _load_model, _runtime
from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.torch_model import require_torch


def apply_template(
    prompts: list[dict[str, str]], template: str
) -> list[dict[str, str]]:
    """prompt群へ指定templateを適用する。"""

    if template not in {"raw", "conversation"}:
        raise ValueError("templateはrawまたはconversationで指定してください")
    return [{**item, "template": template} for item in prompts]


def evaluate_prompt_set(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    config = load_config(repo_path(args.config))
    device = _device(torch, args.device)
    checkpoint = repo_path(args.checkpoint).resolve()
    model, processor, metadata = _load_model(config, checkpoint, device, torch)
    prompts = apply_template(
        load_prompts(repo_path(args.prompt_file)), args.template
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config.generation.max_new_tokens
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else config.generation.temperature
    )
    top_k = args.top_k if args.top_k is not None else config.generation.top_k
    if max_new_tokens <= 0 or temperature <= 0 or top_k < 0:
        raise ValueError("生成設定が不正です")

    results: list[dict[str, Any]] = []
    for index, item in enumerate(prompts):
        prompt_ids, rendered_prompt = _encode_prompt(item, processor)
        output_ids = _generate(
            model,
            prompt_ids,
            max_new_tokens,
            config.model.context_length,
            temperature,
            top_k,
            args.seed + index,
            int(processor.eos_id()),
            device,
            torch,
        )
        completion_ids = output_ids[len(prompt_ids) :]
        results.append(
            {
                **item,
                "seed": args.seed + index,
                "prompt_token_count": len(prompt_ids),
                "rendered_prompt": rendered_prompt,
                "completion_token_count": len(completion_ids),
                "eos_reached": int(processor.eos_id()) in completion_ids,
                "completion": processor.decode(completion_ids),
            }
        )

    result: dict[str, Any] = {
        "format": "torch-chat-prompt-evaluation-v1",
        "prompt_file": str(repo_path(args.prompt_file).resolve()),
        "prompt_sha256": _sha256_file(repo_path(args.prompt_file).resolve()),
        "prompt_template": args.template,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": metadata.get(
            "checkpoint_step", metadata.get("metrics", {}).get("step")
        ),
        "config": str(config.source_path),
        "seed": args.seed,
        "runtime": _runtime(torch, device, device.type == "cuda" and not args.no_amp),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "category_summary": summarize_prompt_results(results),
        "results": results,
    }
    output = repo_path(args.output).resolve()
    text_output = repo_path(args.text_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_output.write_text(_format_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", default="experiments/prompts/issue-1-chat-v1.json")
    parser.add_argument("--template", choices=("raw", "conversation"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-output", required=True)
    evaluate_prompt_set(parser.parse_args())


if __name__ == "__main__":
    main()
