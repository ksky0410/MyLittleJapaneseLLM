"""Issue #1会話SFTを応答機能の比率を考慮して決定的に選別する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from _common import repo_path
from my_little_japanese_llm.tokenizer import load_processor
from prepare_chat_sft import (
    _read_jsonl,
    _validate_turns,
    make_training_example,
)

GREETING_RE = re.compile(
    r"^(?:こんにちは|こんばんは|おはよう(?:ございます)?|"
    r"よろしく(?:お願いします)?|はじめまして|おつかれ(?:さま)?|"
    r"ありがとう(?:ございます)?)[!！。\sー〜～]*$"
)
QUESTION_END_RE = re.compile(
    r"(?:ですか|ますか|でしょうか|なのか|かな|かね|の|どう|何|なに|"
    r"いつ|どこ|誰|だれ|なぜ|どうして)[。！!？?\sー〜～…]*$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_input(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError(f"入力はNAME=PATH形式で指定してください: {value}")
    path = repo_path(raw_path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"入力ディレクトリが見つかりません: {path}")
    train_path = path / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train.jsonlが見つかりません: {train_path}")
    return name.strip(), path


def is_question_context(text: str) -> bool:
    """直前発話が質問として読めるかを保守的な規則で判定する。"""

    stripped = text.strip()
    return "?" in stripped or "？" in stripped or bool(QUESTION_END_RE.search(stripped))


def is_greeting_only(text: str) -> bool:
    return bool(GREETING_RE.fullmatch(text.strip()))


def build_candidates(
    source: str,
    records: list[dict[str, Any]],
    processor: Any,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        turns = _validate_turns(record)
        conversation_id = str(
            record.get("conversation_id", f"{source}:record-{record_index}")
        )
        for target_index in range(1, len(turns)):
            previous_text = turns[target_index - 1]["text"]
            response_text = turns[target_index]["text"]
            body_token_count = len(processor.encode(response_text, out_type=int))
            candidates.append(
                {
                    "source": source,
                    "record_index": record_index,
                    "target_index": target_index,
                    "conversation_id": conversation_id,
                    "source_file": record.get("source_file"),
                    "body_token_count": body_token_count,
                    "response_token_count": body_token_count + 1,
                    "question_context": is_question_context(previous_text),
                    "greeting_only": is_greeting_only(response_text),
                    "first_turn": target_index == 1,
                }
            )
    return candidates


def _category_token_count(candidates: list[dict[str, Any]], key: str) -> int:
    return sum(int(item["response_token_count"]) for item in candidates if item[key])


def select_candidates(
    candidates: list[dict[str, Any]],
    target_response_tokens: int,
    seed: int,
    *,
    question_token_fraction: float,
    max_greeting_token_fraction: float,
    max_first_turn_token_fraction: float,
) -> list[dict[str, Any]]:
    """カテゴリのtoken予算を優先し、足りない分を残りから埋める。"""

    if target_response_tokens <= 0:
        raise ValueError("target_response_tokensは正の整数で指定してください")
    for name, value in (
        ("question_token_fraction", question_token_fraction),
        ("max_greeting_token_fraction", max_greeting_token_fraction),
        ("max_first_turn_token_fraction", max_first_turn_token_fraction),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name}は0以上1以下で指定してください")

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int]] = set()
    selected_tokens = 0
    selected_greeting_tokens = 0
    selected_first_turn_tokens = 0
    greeting_limit = int(target_response_tokens * max_greeting_token_fraction)
    first_turn_limit = int(target_response_tokens * max_first_turn_token_fraction)

    def try_pick(item: dict[str, Any]) -> bool:
        nonlocal selected_tokens, selected_greeting_tokens, selected_first_turn_tokens
        item_id = (int(item["record_index"]), int(item["target_index"]))
        if item_id in selected_ids:
            return False
        response_tokens = int(item["response_token_count"])
        if item["greeting_only"] and selected_greeting_tokens + response_tokens > greeting_limit:
            return False
        if item["first_turn"] and selected_first_turn_tokens + response_tokens > first_turn_limit:
            return False
        selected.append(item)
        selected_ids.add(item_id)
        selected_tokens += response_tokens
        if item["greeting_only"]:
            selected_greeting_tokens += response_tokens
        if item["first_turn"]:
            selected_first_turn_tokens += response_tokens
        return True

    def pick_group(group: list[dict[str, Any]], budget: int) -> None:
        group_tokens = 0
        for item in group:
            if group_tokens >= budget:
                break
            if try_pick(item):
                group_tokens += int(item["response_token_count"])

    question_budget = int(target_response_tokens * question_token_fraction)
    pick_group([item for item in shuffled if item["question_context"]], question_budget)
    pick_group([item for item in shuffled if not item["question_context"]], target_response_tokens - question_budget)
    if selected_tokens < target_response_tokens:
        pick_group(shuffled, target_response_tokens - selected_tokens)
    if selected_tokens < target_response_tokens:
        raise ValueError(
            "カテゴリ制限後の候補がresponse token予算に不足しています: "
            f"{selected_tokens} < {target_response_tokens}"
        )
    random.Random(seed + 1).shuffle(selected)
    return selected


def _summary(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "example_count": len(candidates),
        "response_token_count": sum(int(item["response_token_count"]) for item in candidates),
        "body_token_count": sum(int(item["body_token_count"]) for item in candidates),
        "question_context_count": sum(bool(item["question_context"]) for item in candidates),
        "question_context_response_token_count": _category_token_count(candidates, "question_context"),
        "greeting_only_count": sum(bool(item["greeting_only"]) for item in candidates),
        "greeting_only_response_token_count": _category_token_count(candidates, "greeting_only"),
        "first_turn_count": sum(bool(item["first_turn"]) for item in candidates),
        "first_turn_response_token_count": _category_token_count(candidates, "first_turn"),
    }


def prepare_quality_sft(
    tokenizer_path: str | Path,
    inputs: list[tuple[str, Path]],
    output_path: str | Path,
    manifest_path: str | Path,
    context_length: int,
    target_response_tokens: int,
    seed: int,
    question_token_fraction: float,
    max_greeting_token_fraction: float,
    max_first_turn_token_fraction: float,
) -> dict[str, Any]:
    tokenizer_file = repo_path(tokenizer_path).resolve()
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizerが見つかりません: {tokenizer_file}")
    processor = load_processor(tokenizer_file)
    all_input_stats: dict[str, Any] = {}
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    selected_by_source: dict[str, list[dict[str, Any]]] = {}
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for source_index, (source, input_dir) in enumerate(inputs):
        input_file = input_dir / "train.jsonl"
        records = _read_jsonl(input_file)
        candidates = build_candidates(source, records, processor)
        selected = select_candidates(
            candidates,
            target_response_tokens,
            seed + source_index,
            question_token_fraction=question_token_fraction,
            max_greeting_token_fraction=max_greeting_token_fraction,
            max_first_turn_token_fraction=max_first_turn_token_fraction,
        )
        records_by_source[source] = records
        selected_by_source[source] = selected
        made = [
            make_training_example(
                records[int(item["record_index"])]["turns"],
                int(item["target_index"]),
                processor,
                context_length,
            )
            for item in selected
        ]
        arrays.append(
            (
                np.stack([item[0] for item in made]),
                np.stack([item[1] for item in made]),
                np.stack([item[2] for item in made]),
            )
        )
        all_input_stats[source] = {
            "input_path": str(input_file.resolve()),
            "input_sha256": sha256_file(input_file),
            "full": _summary(candidates),
            "selected": _summary(selected),
            "candidate_count": len(candidates),
        }

    input_ids = np.concatenate([item[0] for item in arrays], axis=0)
    target_ids = np.concatenate([item[1] for item in arrays], axis=0)
    loss_mask = np.concatenate([item[2] for item in arrays], axis=0)
    global_order = np.random.default_rng(seed + 2).permutation(input_ids.shape[0])
    input_ids = input_ids[global_order]
    target_ids = target_ids[global_order]
    loss_mask = loss_mask[global_order]
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )
    selected_provenance = [
        item
        for source in sorted(selected_by_source)
        for item in selected_by_source[source]
    ]
    manifest = {
        "format": "quality-aware-chat-sft-v1",
        "tokenizer_path": str(tokenizer_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "context_length": context_length,
        "target_response_tokens_per_source": target_response_tokens,
        "seed": seed,
        "question_token_fraction": question_token_fraction,
        "max_greeting_token_fraction": max_greeting_token_fraction,
        "max_first_turn_token_fraction": max_first_turn_token_fraction,
        "inputs": all_input_stats,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_example_count": int(input_ids.shape[0]),
        "output_response_token_count": int(loss_mask.astype(np.int64).sum()),
        "array_shape": list(input_ids.shape),
        "selected_provenance": selected_provenance,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--target-response-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--question-token-fraction", type=float, default=0.5)
    parser.add_argument("--max-greeting-token-fraction", type=float, default=0.02)
    parser.add_argument("--max-first-turn-token-fraction", type=float, default=0.05)
    args = parser.parse_args()
    inputs = [parse_input(value) for value in args.input]
    result = prepare_quality_sft(
        args.tokenizer,
        inputs,
        args.output,
        args.manifest,
        args.context_length,
        args.target_response_tokens,
        args.seed,
        args.question_token_fraction,
        args.max_greeting_token_fraction,
        args.max_first_turn_token_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
