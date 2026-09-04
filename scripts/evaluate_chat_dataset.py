"""held-out会話の履歴から次の発話を生成してcheckpointを比較する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
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

CONVERSATION_START = "<|startofconversation|>"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    """会話JSONLを読み、object recordだけを返す。"""

    source = repo_path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"会話JSONLが見つかりません: {source}")
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONLを読めません: {source}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"recordがobjectではありません: {source}:{line_number}")
            records.append(record)
    return records


def _turns(record: dict[str, Any]) -> list[dict[str, str]]:
    turns = record.get("turns")
    if not isinstance(turns, list):
        raise TypeError("会話recordのturnsは配列で指定してください")
    result: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise TypeError(f"turn #{index}がobjectではありません")
        speaker = turn.get("speaker_id")
        text = turn.get("text")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f"turn #{index}のspeaker_idが空です")
        if not isinstance(text, str) or not text:
            raise ValueError(f"turn #{index}のtextが空です")
        result.append({"speaker_id": speaker, "text": text})
    return result


def select_examples(
    records: list[dict[str, Any]], max_examples: int, seed: int
) -> list[dict[str, Any]]:
    """全会話の後続turnから再現可能に評価対象を選ぶ。"""

    if max_examples <= 0:
        raise ValueError("max_examplesは正の整数で指定してください")
    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        turns = _turns(record)
        if len(turns) < 2:
            continue
        conversation_id = record.get("conversation_id", f"record-{record_index}")
        if not isinstance(conversation_id, str) or not conversation_id:
            conversation_id = f"record-{record_index}"
        for target_index in range(1, len(turns)):
            candidates.append(
                {
                    "record_index": record_index,
                    "conversation_id": conversation_id,
                    "target_index": target_index,
                    "turns": turns,
                }
            )
    random.Random(seed).shuffle(candidates)
    return candidates[: min(max_examples, len(candidates))]


def encode_history(
    turns: list[dict[str, str]], target_index: int, processor: Any
) -> tuple[list[int], str]:
    """target本文を含めず、次話者markerまでをToken化する。"""

    if target_index < 1 or target_index >= len(turns):
        raise ValueError("target_indexは1以上かつturn数未満で指定してください")
    parts = [CONVERSATION_START]
    ids = [int(token) for token in processor.encode(CONVERSATION_START, out_type=int)]
    for turn in turns[:target_index]:
        speaker_marker = f"<|speaker:{turn['speaker_id']}|>"
        ids.extend(
            int(token) for token in processor.encode(speaker_marker, out_type=int)
        )
        ids.extend(int(token) for token in processor.encode(turn["text"], out_type=int))
        ids.append(int(processor.eos_id()))
        parts.append(f"{speaker_marker}{turn['text']}<eos:{int(processor.eos_id())}>")
    target_marker = f"<|speaker:{turns[target_index]['speaker_id']}|>"
    ids.extend(int(token) for token in processor.encode(target_marker, out_type=int))
    parts.append(target_marker)
    return ids, "".join(parts)


def token_overlap_scores(
    reference_ids: list[int], completion_ids: list[int], eos_id: int
) -> dict[str, float]:
    """EOSを除いた参照と生成のmultiset token overlapを計算する。"""

    reference = Counter(token for token in reference_ids if token != eos_id)
    completion = Counter(token for token in completion_ids if token != eos_id)
    overlap = sum((reference & completion).values())
    precision = overlap / sum(completion.values()) if completion else 0.0
    recall = overlap / sum(reference.values()) if reference else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_overlap_precision": precision,
        "token_overlap_recall": recall,
        "token_overlap_f1": f1,
    }


def _format_text(result: dict[str, Any]) -> str:
    lines = [
        "# Held-out chat dataset evaluation",
        f"format: {result['format']}",
        f"input: {result['input']}",
        f"checkpoint: {result['checkpoint']}",
        f"checkpoint_step: {result['checkpoint_step']}",
        f"seed: {result['seed']}",
        f"max_examples: {result['max_examples']}",
        f"max_new_tokens: {result['generation']['max_new_tokens']}",
        f"temperature: {result['generation']['temperature']}",
        f"top_k: {result['generation']['top_k']}",
        "",
    ]
    for index, item in enumerate(result["results"], start=1):
        lines.extend(
            [
                f"## example {index:03d}: {item['conversation_id']} turn {item['target_index']}",
                f"speaker: {item['target_speaker']}",
                f"prompt: {item['rendered_prompt']}",
                f"reference: {item['reference']}",
                "completion:",
                item["completion"],
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_chat_dataset(
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    text_output_path: str | Path,
    *,
    max_examples: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """held-out会話を評価し、生成内容をJSONとTXTへ保存する。"""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokensは正の整数で指定してください")
    config_file = repo_path(config_path).resolve()
    checkpoint_file = repo_path(checkpoint_path).resolve()
    input_file = repo_path(input_path).resolve()
    output_file = repo_path(output_path).resolve()
    text_file = repo_path(text_output_path).resolve()
    if output_file == text_file or output_file == input_file or text_file == input_file:
        raise ValueError("入力会話JSONLと出力ファイルは別のパスにしてください")
    records = _read_records(input_file)
    examples = select_examples(records, max_examples, seed)
    if not examples:
        raise ValueError("評価可能な2発話以上の会話がありません")

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
    results: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        turns = example["turns"]
        target_index = int(example["target_index"])
        prompt_ids, rendered_prompt = encode_history(turns, target_index, processor)
        output_ids = generate_ids(
            model,
            prompt_ids,
            max_new_tokens,
            config.model.context_length,
            config.generation.temperature,
            config.generation.top_k,
            seed + index,
            int(processor.eos_id()),
        )
        completion_ids = output_ids[len(prompt_ids) :]
        reference_ids = [
            int(token)
            for token in processor.encode(turns[target_index]["text"], out_type=int)
        ]
        overlap = token_overlap_scores(
            reference_ids, completion_ids, int(processor.eos_id())
        )
        results.append(
            {
                "conversation_id": example["conversation_id"],
                "record_index": example["record_index"],
                "target_index": target_index,
                "target_speaker": turns[target_index]["speaker_id"],
                "rendered_prompt": rendered_prompt,
                "prompt_token_count": len(prompt_ids),
                "reference": turns[target_index]["text"],
                "reference_token_count": len(
                    processor.encode(turns[target_index]["text"], out_type=int)
                ),
                "completion": processor.decode(completion_ids),
                "generated_token_count": len(completion_ids),
                "eos_reached": int(processor.eos_id()) in completion_ids,
                **overlap,
                "seed": seed + index,
            }
        )

    result: dict[str, Any] = {
        "format": "heldout-chat-dataset-evaluation-v1",
        "input": str(input_file),
        "input_sha256": _sha256_file(input_file),
        "checkpoint": str(checkpoint_file),
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "config": str(config_file),
        "seed": seed,
        "max_examples": max_examples,
        "selected_example_count": len(results),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": config.generation.temperature,
            "top_k": config.generation.top_k,
        },
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
    parser.add_argument("--input", required=True, help="validation会話JSONL")
    parser.add_argument("--output", required=True, help="保存するJSON")
    parser.add_argument("--text-output", required=True, help="保存する可読テキスト")
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_chat_dataset(
        args.config,
        args.checkpoint,
        args.input,
        args.output,
        args.text_output,
        max_examples=args.examples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
