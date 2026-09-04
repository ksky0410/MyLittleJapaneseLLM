"""固定した短い会話promptでcheckpointの会話らしさを比較する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.model import TinyJapaneseGPT, require_mlx
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.training import (
    generate_ids,
    load_checkpoint,
    signature_from_config,
)

DEFAULT_PROMPT_FILE = "experiments/prompts/issue-1-chat-v1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    """JSONの固定promptを検証して読み込む。"""

    prompt_file = repo_path(path).resolve()
    if not prompt_file.is_file():
        raise FileNotFoundError(f"promptファイルが見つかりません: {prompt_file}")
    try:
        raw: Any = json.loads(prompt_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"prompt JSONを読めません: {prompt_file}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompt JSONは空でないobjectの配列にしてください")

    prompts: list[dict[str, str]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"prompt #{index + 1}がobjectではありません")
        prompt_id = item.get("id")
        category = item.get("category", "uncategorized")
        prompt = item.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt #{index + 1}のidが空です")
        if prompt_id in ids:
            raise ValueError(f"prompt idが重複しています: {prompt_id}")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"prompt {prompt_id}のcategoryが空です")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt {prompt_id}の本文が空です")
        ids.add(prompt_id)
        prompts.append({"id": prompt_id, "category": category, "prompt": prompt})
    return prompts


def _format_text(result: dict[str, Any]) -> str:
    lines = [
        "# Chat prompt evaluation",
        f"format: {result['format']}",
        f"prompt_file: {result['prompt_file']}",
        f"checkpoint: {result['checkpoint']}",
        f"checkpoint_step: {result['checkpoint_step']}",
        f"config: {result['config']}",
        f"seed: {result['seed']}",
        f"max_new_tokens: {result['generation']['max_new_tokens']}",
        f"temperature: {result['generation']['temperature']}",
        f"top_k: {result['generation']['top_k']}",
        "",
    ]
    for item in result["results"]:
        lines.extend(
            [
                f"## {item['id']} [{item['category']}]",
                f"prompt: {item['prompt']}",
                "completion:",
                item["completion"],
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_chat_prompts(
    config_path: str | Path,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    output_path: str | Path,
    text_output_path: str | Path,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """一つのcheckpointから固定prompt群を生成し、JSONと可読テキストを保存する。"""

    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokensは正の整数で指定してください")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperatureは正数で指定してください")
    if top_k is not None and top_k < 0:
        raise ValueError("top_kは0以上で指定してください")

    config_file = repo_path(config_path).resolve()
    checkpoint_file = repo_path(checkpoint_path).resolve()
    prompt_file = repo_path(prompt_path).resolve()
    output_file = repo_path(output_path).resolve()
    text_file = repo_path(text_output_path).resolve()
    if (
        output_file == text_file
        or output_file == prompt_file
        or text_file == prompt_file
    ):
        raise ValueError("入力promptと出力ファイルは別のパスにしてください")
    prompts = load_prompts(prompt_file)

    require_mlx()
    config = load_config(config_file)
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    model = TinyJapaneseGPT(
        vocab_size,
        config.model.dim,
        config.model.layers,
        config.model.heads,
        config.model.context_length,
        config.model.mlp_ratio,
        config.model.position_embedding,
    )
    metadata = load_checkpoint(
        model, checkpoint_file, signature_from_config(config, vocab_size)
    )
    generation = {
        "max_new_tokens": (
            max_new_tokens
            if max_new_tokens is not None
            else config.generation.max_new_tokens
        ),
        "temperature": (
            temperature if temperature is not None else config.generation.temperature
        ),
        "top_k": top_k if top_k is not None else config.generation.top_k,
    }
    base_seed = seed if seed is not None else config.training.seed
    results: list[dict[str, Any]] = []
    for index, item in enumerate(prompts):
        prompt_ids = processor.encode(item["prompt"], out_type=int)
        if not prompt_ids:
            raise ValueError(f"promptをToken化できません: {item['id']}")
        output_ids = generate_ids(
            model,
            prompt_ids,
            generation["max_new_tokens"],
            config.model.context_length,
            generation["temperature"],
            generation["top_k"],
            base_seed + index,
            int(processor.eos_id()),
        )
        completion = processor.decode(output_ids[len(prompt_ids) :])
        results.append(
            {
                **item,
                "seed": base_seed + index,
                "prompt_token_count": len(prompt_ids),
                "completion": completion,
            }
        )

    result: dict[str, Any] = {
        "format": "chat-prompt-evaluation-v1",
        "prompt_file": str(prompt_file),
        "prompt_sha256": _sha256_file(prompt_file),
        "checkpoint": str(checkpoint_file),
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "config": str(config_file),
        "seed": base_seed,
        "generation": generation,
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text_file.write_text(_format_text(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True, help="保存するJSON")
    parser.add_argument("--text-output", required=True, help="保存する可読テキスト")
    args = parser.parse_args()
    result = evaluate_chat_prompts(
        args.config,
        args.checkpoint,
        args.prompt_file,
        args.output,
        args.text_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
