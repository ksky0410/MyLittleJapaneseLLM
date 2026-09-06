"""応答機能のsource別比率を考慮してIssue #1会話SFTを作成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from _common import repo_path
from analyze_response_functions import classify_response_function
from my_little_japanese_llm.tokenizer import load_processor
from prepare_chat_sft import _read_jsonl, _validate_turns, make_training_example

CATEGORY_ORDER = (
    "question_answer",
    "topic_continuation",
    "other",
    "backchannel",
    "agreement_disagreement",
    "greeting",
    "closing",
)
SOURCE_CATEGORY_FRACTIONS = {
    "rpc": {
        "question_answer": 0.30,
        "topic_continuation": 0.45,
        "other": 0.20,
        "backchannel": 0.03,
        "agreement_disagreement": 0.015,
        "greeting": 0.004,
        "closing": 0.001,
    },
    "mrmp": {
        "question_answer": 0.13,
        "topic_continuation": 0.15,
        "other": 0.60,
        "backchannel": 0.09,
        "agreement_disagreement": 0.02,
        "greeting": 0.009,
        "closing": 0.001,
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_fractions(source: str) -> dict[str, float]:
    if source not in SOURCE_CATEGORY_FRACTIONS:
        raise ValueError(f"未知のsourceです: {source}")
    fractions = SOURCE_CATEGORY_FRACTIONS[source]
    if set(fractions) != set(CATEGORY_ORDER):
        raise ValueError(f"カテゴリ定義が不完全です: {source}")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"カテゴリ比率の合計が1ではありません: {source}")
    return dict(fractions)


def build_candidates(
    source: str, records: list[dict[str, Any]], processor: Any
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
            response_token_count = len(processor.encode(response_text, out_type=int)) + 1
            candidates.append(
                {
                    "source": source,
                    "record_index": record_index,
                    "target_index": target_index,
                    "conversation_id": conversation_id,
                    "source_file": record.get("source_file"),
                    "response_token_count": response_token_count,
                    "body_token_count": response_token_count - 1,
                    "category": classify_response_function(
                        previous_text, response_text, response_token_count
                    ),
                    "first_turn": target_index == 1,
                }
            )
    return candidates


def _summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for item in candidates:
        category = str(item["category"])
        counts[category] += 1
        tokens[category] += int(item["response_token_count"])
    total_tokens = sum(tokens.values())
    return {
        "example_count": len(candidates),
        "response_token_count": total_tokens,
        "categories": {
            category: {
                "example_count": counts[category],
                "response_token_count": tokens[category],
                "response_token_fraction": tokens[category] / total_tokens
                if total_tokens
                else 0.0,
            }
            for category in CATEGORY_ORDER
        },
    }


def select_candidates(
    candidates: list[dict[str, Any]],
    target_response_tokens: int,
    seed: int,
    source: str,
    *,
    max_greeting_token_fraction: float = 0.02,
    max_first_turn_token_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """カテゴリ予算を優先し、不足時は候補のあるカテゴリから補充する。"""

    if target_response_tokens <= 0:
        raise ValueError("target_response_tokensは正の整数で指定してください")
    fractions = validate_source_fractions(source)
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in shuffled:
        by_category[str(item["category"])].append(item)

    budgets = {
        category: int(target_response_tokens * fraction)
        for category, fraction in fractions.items()
    }
    greeting_limit = int(target_response_tokens * max_greeting_token_fraction)
    first_turn_limit = int(target_response_tokens * max_first_turn_token_fraction)
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int]] = set()
    selected_tokens = 0
    selected_greeting_tokens = 0
    selected_first_turn_tokens = 0
    selected_category_tokens: Counter[str] = Counter()

    def try_pick(item: dict[str, Any], *, enforce_category_budget: bool) -> bool:
        nonlocal selected_tokens, selected_greeting_tokens, selected_first_turn_tokens
        item_id = (int(item["record_index"]), int(item["target_index"]))
        if item_id in selected_ids:
            return False
        category = str(item["category"])
        response_tokens = int(item["response_token_count"])
        if enforce_category_budget and selected_category_tokens[category] >= budgets[category]:
            return False
        if (
            category == "greeting"
            and selected_greeting_tokens + response_tokens > greeting_limit
        ):
            return False
        if (
            item["first_turn"]
            and selected_first_turn_tokens + response_tokens > first_turn_limit
        ):
            return False
        selected.append(item)
        selected_ids.add(item_id)
        selected_tokens += response_tokens
        selected_category_tokens[category] += response_tokens
        if category == "greeting":
            selected_greeting_tokens += response_tokens
        if item["first_turn"]:
            selected_first_turn_tokens += response_tokens
        return True

    for category in CATEGORY_ORDER:
        for item in by_category[category]:
            if selected_tokens >= target_response_tokens:
                break
            if selected_category_tokens[category] >= budgets[category]:
                break
            try_pick(item, enforce_category_budget=True)

    while selected_tokens < target_response_tokens:
        candidates_with_budget = [
            item
            for item in shuffled
            if item["category"] in budgets
            and selected_category_tokens[str(item["category"])] < budgets[str(item["category"])]
        ]
        pool = candidates_with_budget or [
            item for item in shuffled if (int(item["record_index"]), int(item["target_index"])) not in selected_ids
        ]
        if not pool:
            raise ValueError(
                "カテゴリ制限後の候補がresponse token予算に不足しています: "
                f"{selected_tokens} < {target_response_tokens}"
            )
        picked = False
        for item in pool:
            if try_pick(item, enforce_category_budget=bool(candidates_with_budget)):
                picked = True
                break
        if not picked and candidates_with_budget:
            for item in shuffled:
                if try_pick(item, enforce_category_budget=False):
                    picked = True
                    break
        if not picked:
            raise ValueError("first_turnまたはgreetingの上限により候補を選べません")

    random.Random(seed + 1).shuffle(selected)
    return selected


def prepare_functional_sft(
    tokenizer_path: str | Path,
    inputs: list[tuple[str, Path]],
    output_path: str | Path,
    manifest_path: str | Path,
    context_length: int,
    target_response_tokens: int,
    seed: int,
) -> dict[str, Any]:
    tokenizer_file = repo_path(tokenizer_path).resolve()
    processor = load_processor(tokenizer_file)
    all_stats: dict[str, Any] = {}
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    selected_provenance: list[dict[str, Any]] = []
    for source_index, (source, input_dir) in enumerate(inputs):
        input_file = input_dir / "train.jsonl"
        records = _read_jsonl(input_file)
        candidates = build_candidates(source, records, processor)
        selected = select_candidates(
            candidates,
            target_response_tokens,
            seed + source_index,
            source,
        )
        made = [
            make_training_example(
                records[int(item["record_index"])] ["turns"],
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
        selected_provenance.extend(selected)
        all_stats[source] = {
            "input_path": str(input_file.resolve()),
            "input_sha256": sha256_file(input_file),
            "full": _summary(candidates),
            "selected": _summary(selected),
            "category_fractions": validate_source_fractions(source),
        }

    input_ids = np.concatenate([item[0] for item in arrays], axis=0)
    target_ids = np.concatenate([item[1] for item in arrays], axis=0)
    loss_mask = np.concatenate([item[2] for item in arrays], axis=0)
    global_order = np.random.default_rng(seed + len(inputs)).permutation(input_ids.shape[0])
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
    manifest = {
        "format": "functional-chat-sft-v1",
        "issue_url": "https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1",
        "classifier": "scripts/analyze_response_functions.py",
        "classifier_version": "2026-09-06-v3",
        "tokenizer_path": str(tokenizer_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "context_length": context_length,
        "target_response_tokens_per_source": target_response_tokens,
        "seed": seed,
        "inputs": all_stats,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_example_count": int(input_ids.shape[0]),
        "output_response_token_count": int(loss_mask.astype(np.int64).sum()),
        "array_shape": list(input_ids.shape),
        "selected_provenance": selected_provenance,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--target-response-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=9301)
    args = parser.parse_args()
    inputs = []
    for raw_input in args.input:
        name, separator, raw_path = raw_input.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"入力はNAME=PATH形式で指定してください: {raw_input}")
        input_dir = repo_path(raw_path).resolve()
        if not (input_dir / "train.jsonl").is_file():
            raise FileNotFoundError(f"train.jsonlが見つかりません: {input_dir}")
        inputs.append((name.strip(), input_dir))
    result = prepare_functional_sft(
        args.tokenizer,
        inputs,
        args.output,
        args.manifest,
        args.context_length,
        args.target_response_tokens,
        args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
