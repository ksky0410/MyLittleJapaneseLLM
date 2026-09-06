"""Issue #1に対応する短い口語発話を実会話履歴付き評価manifestへ選ぶ。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import repo_path
from evaluate_chat_dataset import _read_records, _turns, encode_history
from my_little_japanese_llm.tokenizer import load_processor


PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("casual-agreement", "それな", r"^それな[ー〜～！？!?。、…笑ｗw\s]*$", "surface"),
    ("casual-reaction", "やば", r"^やば(?:い|すぎ|っ|す|いね|いです)?[ー〜～！？!?。、…笑ｗw\s]*$", "surface"),
    ("discourse-marker", "なんかさ", r"^なんかさ(?:[、,。！？!?…〜～\s].*)?$", "surface"),
    ("disagreement", "いやそれは", r"^いやそれは(?:[、,。！？!?…〜～\s].*)?$", "surface"),
    ("parting", "おつかれ", r"^おつかれ(?:さま)?(?:です|でした)?[ー〜～！？!?。、…笑ｗw\s]*$", "surface"),
    ("today-activity", "今日なにしてた？", r"^今日(?:は|、)?(?:なに|何)(?:を)?して(?:た|ました|いた)(?:か)?[ー〜～！？!?。、…笑ｗw\s]*$", "surface"),
    ("tomorrow-free", "明日ひま？", r"^明日(?:は)?[、,\s]*(?:ひま|暇)(?:ですか)?[ー〜～！？!?。、…笑ｗw\s]*$", "surface"),
    ("today-activity-intent", "今日の活動質問", r"^今日.{0,12}(?:なに|何).{0,12}(?:して|した|予定).{0,8}$", "semantic"),
    ("tomorrow-plan-intent", "明日の予定質問", r"^明日.{0,12}(?:予定|何する|何をする|暇|ひま).{0,8}$", "semantic"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_name(record: dict[str, Any]) -> str:
    source = record.get("source") or record.get("dataset")
    if source == "real-persona-chat":
        return "real-persona-chat"
    if source == "mrmp":
        return "mrmp"
    conversation_id = str(record.get("conversation_id", ""))
    return "mrmp" if conversation_id.startswith("mrmp:") else "real-persona-chat"


def select_candidates(
    records: list[dict[str, Any]], processor: Any, *, seed: int, per_source: int,
    context_length: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        turns = _turns(record)
        source = _source_name(record)
        conversation_id = str(record.get("conversation_id", f"record-{record_index}"))
        for input_index, turn in enumerate(turns[:-1]):
            text = turn["text"].strip()
            if not text:
                continue
            for category, reference_prompt, pattern, match_type in PATTERNS:
                if len(text) > 42:
                    continue
                if not re.search(pattern, text):
                    continue
                target_index = input_index + 1
                history_ids, _ = encode_history(turns, target_index, processor)
                candidates.append(
                    {
                        "record_index": record_index,
                        "conversation_id": conversation_id,
                        "target_index": target_index,
                        "target_speaker": turns[target_index]["speaker_id"],
                        "source": source,
                        "category": category,
                        "reference_prompt": reference_prompt,
                        "match_type": match_type,
                        "input": text,
                        "reference": turns[target_index]["text"],
                        "history_token_count": len(history_ids),
                        "history_truncated": len(history_ids) > context_length,
                    }
                )

    randomiser = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[(item["source"], item["category"])].append(item)
    selected: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for key in sorted(grouped):
        pool = list(grouped[key])
        randomiser.shuffle(pool)
        chosen = pool[:per_source]
        selected.extend(chosen)
        counts[f"{key[0]}:{key[1]}"] = {
            "available": len(pool),
            "selected": len(chosen),
            "selected_indices": [
                {"record_index": item["record_index"], "target_index": item["target_index"]}
                for item in chosen
            ],
        }
    selected.sort(key=lambda item: (item["source"], item["category"], item["record_index"], item["target_index"]))
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-sourceは正の整数で指定してください")

    input_path = repo_path(args.input).resolve()
    tokenizer_path = repo_path(args.tokenizer).resolve()
    output_path = repo_path(args.output).resolve()
    records = _read_records(input_path)
    processor = load_processor(tokenizer_path)
    selected, counts = select_candidates(
        records, processor, seed=args.seed, per_source=args.per_source,
        context_length=args.context_length,
    )
    examples = []
    for item in selected:
        example = {
            key: item[key]
            for key in (
                "record_index", "conversation_id", "target_index", "target_speaker",
                "source", "category", "match_type", "reference_prompt", "reference",
                "input", "history_token_count", "history_truncated",
            )
        }
        examples.append(example)
    result = {
        "format": "chat-eval-selection-v2-issue1-context",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "seed": args.seed,
        "per_source": args.per_source,
        "candidate_rules": [
            {"category": category, "reference_prompt": prompt, "match_type": match_type, "pattern": pattern}
            for category, prompt, pattern, match_type in PATTERNS
        ],
        "candidate_counts": counts,
        "selected_example_count": len(examples),
        "unique_conversation_count": len({item["conversation_id"] for item in examples}),
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
