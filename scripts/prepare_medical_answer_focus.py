"""医師国家試験の正解ラベルだけを短く返す追加SFT会話を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import repo_path
from prepare_medical_sft import _compact, _question_text, _read_jsonl


SPLITS = ("train", "validation")


def sha256_file(path: Path) -> str:
    """ファイルのSHA-256を返す。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_focus_text(record: dict[str, Any]) -> str:
    """正解選択肢だけを、既存の評価器が読める形式で返す。"""

    correct = record.get("correct")
    if not isinstance(correct, list) or not correct:
        raise ValueError(f"correctが空です: {record.get('number')}")
    return f"正解は{'、'.join(str(key) for key in correct)}です。"


def convert_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    """元問題から、正解だけを返す一往復の会話を作る。"""

    number = _compact(record.get("number"))
    if not number:
        raise ValueError("問題番号が空です")
    return {
        "conversation_id": f"medical-qb-v2-answer-focus:{number}",
        "dataset": "medical-qb-v2-answer-focus",
        "source": source,
        "question_number": number,
        "exam_version": record.get("exam_version"),
        "turns": [
            {"turn_index": 0, "speaker_id": "DA", "text": _question_text(record)},
            {
                "turn_index": 1,
                "speaker_id": "DC",
                "text": answer_focus_text(record),
            },
        ],
    }


def prepare_answer_focus(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    filter_conversation_dir: str | Path | None = None,
) -> dict[str, Any]:
    """train/validationの問題JSONLをanswer-first会話JSONLへ変換する。

    ``filter_conversation_dir``を指定した場合は、既存の医療SFTに採用済みの
    問題番号だけを使う。これにより、元の医療SFTと同じ母集団で比較できる。
    """

    source_root = repo_path(input_dir).resolve()
    destination = repo_path(output_dir).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"入力ディレクトリがありません: {source_root}")
    filter_root = None
    if filter_conversation_dir is not None:
        filter_root = repo_path(filter_conversation_dir).resolve()
        if not filter_root.is_dir():
            raise NotADirectoryError(f"絞り込み会話ディレクトリがありません: {filter_root}")

    destination.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, Any] = {}
    for split in SPLITS:
        input_path = source_root / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"入力JSONLがありません: {input_path}")
        records = _read_jsonl(input_path)
        selected_numbers: set[str] | None = None
        filter_path = None
        if filter_root is not None:
            filter_path = filter_root / f"{split}.jsonl"
            if not filter_path.is_file():
                raise FileNotFoundError(f"絞り込み会話JSONLがありません: {filter_path}")
            selected_numbers = {
                _compact(record.get("question_number"))
                for record in _read_jsonl(filter_path)
            }
            records = [
                record
                for record in records
                if _compact(record.get("number")) in selected_numbers
            ]
        converted: list[dict[str, Any]] = []
        skipped_records: list[dict[str, str]] = []
        for record in records:
            try:
                converted.append(convert_record(record, split))
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
            "filter_path": str(filter_path) if filter_path is not None else None,
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "output_record_count": len(converted),
            "skipped_record_count": len(skipped_records),
            "skipped_records": skipped_records,
        }

    manifest = {
        "format": "medical-qb-answer-focus-conversation-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dir": str(source_root),
        "output_dir": str(destination),
        "filter_conversation_dir": str(filter_root) if filter_root is not None else None,
        "answer_policy": "正解選択肢だけを『正解は○です。』の形式で返す",
        "splits": split_stats,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--filter-conversation",
        default=None,
        help="既存SFTのtrain/validation.jsonlにある問題番号だけを採用する",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_answer_focus(
                args.input,
                args.output,
                args.manifest,
                filter_conversation_dir=args.filter_conversation,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
