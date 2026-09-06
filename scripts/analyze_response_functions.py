"""Issue #1会話データの応答機能を再現可能な規則で集計する。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import repo_path
from my_little_japanese_llm.tokenizer import load_processor
from prepare_chat_sft import _read_jsonl, _validate_turns
from prepare_quality_chat_sft import (
    is_greeting_only,
    is_question_context,
    sha256_file,
)

BACKCHANNEL_RE = re.compile(
    r"^(?:うん|うーん|そう|そっか|なるほど|たしかに|ほんと|本当|"
    r"わかる|わかりました|了解|はい|いいえ|ええ|そうだね|そうですね|"
    r"なるほどね|笑|ｗ+|へえ|へー|まじ|マジ)[!！。…\sー〜～]*$"
)
BACKCHANNEL_WORD_RE = re.compile(
    r"(?:うん|そう|そっか|なるほど|たしかに|ほんと|本当|"
    r"わかる|了解|はい|いいえ|ええ|笑|ｗ+|へえ|へー|まじ|マジ)"
)
DISAGREEMENT_RE = re.compile(
    r"^(?:いや|でも|ただ|違う|ちがう|そうじゃ|それは違|"
    r"いやいや|いや、それ)[、,。！!？?\s]*"
)
CLOSING_RE = re.compile(
    r"(?:またね|おやすみ|お疲れさま|おつかれさま|ではまた|"
    r"失礼します|また明日|バイバイ|さようなら|ありがとう)[!！。\sー〜～]*$"
)


def classify_response_function(
    previous_text: str, response_text: str, response_token_count: int
) -> str:
    """直前発話と応答から、重複しない仮カテゴリを決める。"""

    response = response_text.strip()
    if is_greeting_only(response):
        return "greeting"
    if response_token_count <= 12 and CLOSING_RE.search(response):
        return "closing"
    if is_question_context(previous_text):
        return "question_answer"
    if response_token_count <= 12 and (
        BACKCHANNEL_RE.fullmatch(response)
        or BACKCHANNEL_WORD_RE.search(response)
    ):
        return "backchannel"
    if DISAGREEMENT_RE.search(response):
        return "agreement_disagreement"
    if response_token_count >= 16:
        return "topic_continuation"
    return "other"


def analyze_source(
    source: str, input_path: Path, processor: Any, sample_limit: int
) -> dict[str, Any]:
    records = _read_jsonl(input_path)
    counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_examples = 0
    total_tokens = 0
    for record_index, record in enumerate(records):
        turns = _validate_turns(record)
        conversation_id = str(
            record.get("conversation_id", f"{source}:record-{record_index}")
        )
        for target_index in range(1, len(turns)):
            previous_text = turns[target_index - 1]["text"]
            response_text = turns[target_index]["text"]
            response_token_count = len(processor.encode(response_text, out_type=int)) + 1
            category = classify_response_function(
                previous_text, response_text, response_token_count
            )
            counts[category] += 1
            token_counts[category] += response_token_count
            total_examples += 1
            total_tokens += response_token_count
            if len(samples[category]) < sample_limit:
                samples[category].append(
                    {
                        "conversation_id": conversation_id,
                        "record_index": record_index,
                        "target_index": target_index,
                        "previous": previous_text,
                        "response": response_text,
                        "response_token_count": response_token_count,
                    }
                )
    categories = sorted(counts)
    return {
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "record_count": len(records),
        "example_count": total_examples,
        "response_token_count": total_tokens,
        "categories": {
            category: {
                "example_count": counts[category],
                "response_token_count": token_counts[category],
                "example_fraction": counts[category] / total_examples
                if total_examples
                else 0.0,
                "response_token_fraction": token_counts[category] / total_tokens
                if total_tokens
                else 0.0,
                "samples": samples[category],
            }
            for category in categories
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()
    if args.sample_limit < 0:
        raise ValueError("--sample-limitは0以上で指定してください")
    processor = load_processor(repo_path(args.tokenizer).resolve())
    sources: dict[str, Any] = {}
    for raw_input in args.input:
        name, separator, raw_path = raw_input.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"入力はNAME=PATH形式で指定してください: {raw_input}")
        input_path = repo_path(raw_path).resolve() / "train.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"train.jsonlが見つかりません: {input_path}")
        sources[name.strip()] = analyze_source(
            name.strip(), input_path, processor, args.sample_limit
        )
    output_path = repo_path(args.output).resolve()
    report = {
        "format": "response-function-analysis-v1",
        "classifier": {
            "version": "2026-09-06-v3",
            "priority": [
                "greeting",
                "closing",
                "question_answer",
                "backchannel",
                "agreement_disagreement",
                "topic_continuation",
                "other",
            ],
            "note": "規則ベースの仮分類であり、人手ラベルではない",
        },
        "tokenizer": str(repo_path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256_file(repo_path(args.tokenizer).resolve()),
        "sources": sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
