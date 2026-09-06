"""医師国家試験の問題を、質問回答形式のSFT用会話JSONLへ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import repo_path
from my_little_japanese_llm.tokenizer import load_processor
from prepare_chat_sft import make_training_example


SPLITS = ("train", "validation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL recordがobjectではありません: {path}:{line_number}")
            records.append(value)
    return records


def _compact(value: object) -> str:
    return " ".join(str(value or "").split())


def _question_text(record: dict[str, Any]) -> str:
    question = _compact(record.get("question"))
    options = record.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"optionsがobjectではありません: {record.get('number')}")
    option_text = "　".join(
        f"{key}:{_compact(value)}" for key, value in options.items()
    )
    return f"問題：{question}\n選択肢：{option_text}\n正解と理由を簡潔に答えてください。"


def _answer_text(record: dict[str, Any]) -> str:
    correct = record.get("correct")
    if not isinstance(correct, list) or not correct:
        raise ValueError(f"correctが空です: {record.get('number')}")
    explanations = record.get("explanations")
    if not isinstance(explanations, dict):
        explanations = {}
    reasons = [
        _compact(explanations.get(str(key)) or explanations.get(key))
        for key in correct
    ]
    reasons = [reason for reason in reasons if reason]
    answer = f"正解は{'、'.join(str(key) for key in correct)}です。"
    if reasons:
        answer += "理由は" + " ".join(reasons)
    point = _compact(record.get("point"))
    if point:
        answer += f" ポイント：{point}"
    return answer


def convert_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    number = _compact(record.get("number"))
    if not number:
        raise ValueError("問題番号が空です")
    return {
        "conversation_id": f"medical-qb-v2:{number}",
        "dataset": "medical-qb-v2",
        "source": source,
        "question_number": number,
        "exam_version": record.get("exam_version"),
        "turns": [
            {"turn_index": 0, "speaker_id": "DA", "text": _question_text(record)},
            {"turn_index": 1, "speaker_id": "DC", "text": _answer_text(record)},
        ],
    }


def prepare_medical_sft(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    context_length: int = 256,
    drop_truncated: bool = False,
) -> dict[str, Any]:
    source_root = repo_path(input_dir).resolve()
    destination = repo_path(output_dir).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"入力ディレクトリがありません: {source_root}")
    input_paths = {split: source_root / f"{split}.jsonl" for split in SPLITS}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("入力JSONLがありません: " + ", ".join(missing))
    processor = None
    if drop_truncated:
        if tokenizer_path is None:
            raise ValueError("drop_truncatedにはtokenizer_pathが必要です")
        processor = load_processor(repo_path(tokenizer_path).resolve())

    destination.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, Any] = {}
    for split, input_path in input_paths.items():
        records = _read_jsonl(input_path)
        converted: list[dict[str, Any]] = []
        skipped_records: list[dict[str, str]] = []
        for record in records:
            try:
                candidate = convert_record(record, split)
                if processor is not None:
                    _, _, _, stats = make_training_example(
                        candidate["turns"], 1, processor, context_length
                    )
                    if bool(stats["truncated"]):
                        raise ValueError(
                            f"context_length={context_length}を超えるため除外: {record.get('number')}"
                        )
                converted.append(candidate)
            except ValueError as error:
                skipped_records.append(
                    {"number": _compact(record.get("number")), "reason": str(error)}
                )
        output_path = destination / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in converted:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_stats[split] = {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "input_record_count": len(records),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "output_record_count": len(converted),
            "skipped_record_count": len(skipped_records),
            "skipped_records": skipped_records,
        }

    manifest = {
        "format": "medical-qb-sft-conversation-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(source_root),
        "output_dir": str(destination),
        "splits": split_stats,
        "speaker_mapping": {"question": "DA", "answer": "DC"},
        "answer_policy": "正解と正解選択肢の解説、pointだけをresponseへ含める",
        "context_filter": {
            "enabled": processor is not None,
            "tokenizer_path": str(repo_path(tokenizer_path).resolve())
            if tokenizer_path is not None
            else None,
            "context_length": context_length if processor is not None else None,
            "drop_truncated": drop_truncated,
        },
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="medical-qb-v2のtrain/validation JSONLディレクトリ")
    parser.add_argument("--output", required=True, help="変換後JSONLの出力ディレクトリ")
    parser.add_argument("--manifest", required=True, help="変換条件manifestの出力先")
    parser.add_argument("--tokenizer", default=None, help="長さフィルタ用SentencePiece .model")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--drop-truncated", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_medical_sft(
                args.input,
                args.output,
                args.manifest,
                tokenizer_path=args.tokenizer,
                context_length=args.context_length,
                drop_truncated=args.drop_truncated,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
