"""未使用conversation test splitから層別評価例を固定する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from _common import repo_path
from evaluate_chat_dataset import _read_records, _turns, encode_history

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.tokenizer import load_processor

STRATA = ("short", "medium", "long")


def sha256_file(path: str | Path) -> str:
    """ファイルのSHA-256を返す。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_response_length(token_count: int) -> str:
    """応答本文Token数から評価層を決める。"""

    if token_count <= 8:
        return "short"
    if token_count <= 24:
        return "medium"
    return "long"


def _conversation_id(record: dict[str, Any], record_index: int) -> str:
    value = record.get("conversation_id", f"record-{record_index}")
    return value if isinstance(value, str) and value else f"record-{record_index}"


def _source_name(conversation_id: str) -> str:
    return conversation_id.split(":", 1)[0]


def _train_texts(records: list[dict[str, Any]]) -> set[str]:
    """train側の全発話本文を完全一致照合用setにする。"""

    texts: set[str] = set()
    for record in records:
        texts.update(turn["text"] for turn in _turns(record))
    return texts


def select_stratified_examples(
    records: list[dict[str, Any]],
    train_texts: set[str],
    processor: Any,
    context_length: int,
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    """一会話一例を守りながらshort/medium/longを選ぶ。"""

    if per_stratum <= 0:
        raise ValueError("per_stratumは正の整数で指定してください")
    if context_length <= 0:
        raise ValueError("context_lengthは正の整数で指定してください")
    candidates: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for record_index, record in enumerate(records):
        turns = _turns(record)
        conversation_id = _conversation_id(record, record_index)
        for target_index in range(1, len(turns)):
            reference = turns[target_index]["text"]
            reference_token_count = len(processor.encode(reference, out_type=int))
            stratum = classify_response_length(reference_token_count)
            prompt_ids, _ = encode_history(turns, target_index, processor)
            candidates[stratum].append(
                {
                    "record_index": record_index,
                    "conversation_id": conversation_id,
                    "target_index": target_index,
                    "target_speaker": turns[target_index]["speaker_id"],
                    "source": _source_name(conversation_id),
                    "stratum": stratum,
                    "reference": reference,
                    "reference_token_count": reference_token_count,
                    "history_token_count": len(prompt_ids),
                    "history_truncated": len(prompt_ids) > context_length,
                    "train_text_overlap": reference in train_texts,
                }
            )

    selected: list[dict[str, Any]] = []
    used_conversations: set[str] = set()
    for offset, stratum in enumerate(STRATA):
        pool = list(candidates[stratum])
        random.Random(seed + offset).shuffle(pool)
        picked = 0
        for candidate in pool:
            if candidate["conversation_id"] in used_conversations:
                continue
            selected.append(candidate)
            used_conversations.add(candidate["conversation_id"])
            picked += 1
            if picked == per_stratum:
                break
        if picked != per_stratum:
            raise ValueError(
                f"{stratum}層の選択数が不足しています: {picked}/{per_stratum}"
            )

    return selected


def build_selection_manifest(
    config_path: str | Path,
    input_path: str | Path,
    train_path: str | Path,
    output_path: str | Path,
    *,
    per_stratum: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """test JSONLから評価選択manifestを作成して保存する。"""

    config_file = repo_path(config_path).resolve()
    input_file = repo_path(input_path).resolve()
    train_file = repo_path(train_path).resolve()
    output_file = repo_path(output_path).resolve()
    if (
        input_file == train_file
        or input_file == output_file
        or train_file == output_file
    ):
        raise ValueError("入力と出力のパスを重複させないでください")
    if not input_file.is_file() or not train_file.is_file():
        raise FileNotFoundError("test/trainの会話JSONLが見つかりません")

    config = load_config(config_file)
    processor = load_processor(config.paths.tokenizer_model)
    records = _read_records(input_file)
    train_records = _read_records(train_file)
    examples = select_stratified_examples(
        records,
        _train_texts(train_records),
        processor,
        config.model.context_length,
        per_stratum=per_stratum,
        seed=seed,
    )
    counts = {
        stratum: sum(item["stratum"] == stratum for item in examples)
        for stratum in STRATA
    }
    manifest: dict[str, Any] = {
        "format": "chat-eval-selection-v1",
        "config": str(config_file),
        "input": str(input_file),
        "input_sha256": sha256_file(input_file),
        "train_input": str(train_file),
        "train_input_sha256": sha256_file(train_file),
        "tokenizer": str(repo_path(config.paths.tokenizer_model).resolve()),
        "tokenizer_sha256": sha256_file(
            repo_path(config.paths.tokenizer_model).resolve()
        ),
        "seed": seed,
        "context_length": config.model.context_length,
        "per_stratum": per_stratum,
        "selected_example_count": len(examples),
        "stratum_counts": counts,
        "unique_conversation_count": len(
            {item["conversation_id"] for item in examples}
        ),
        "examples": examples,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/token-budget-chat-sft-5m-smoke.toml"
    )
    parser.add_argument("--input", required=True, help="評価に使うtest会話JSONL")
    parser.add_argument(
        "--train-input", required=True, help="重複照合に使うtrain会話JSONL"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-stratum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_selection_manifest(
        args.config,
        args.input,
        args.train_input,
        args.output,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
