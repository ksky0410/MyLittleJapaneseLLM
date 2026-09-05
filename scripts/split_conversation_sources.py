"""取り込み済み会話JSONLをsource別の会話ブロックへ分ける。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import repo_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"recordがobjectではありません: {path}:{line_number}")
            if not isinstance(value.get("dataset"), str) or not value["dataset"]:
                raise ValueError(f"datasetがありません: {path}:{line_number}")
            turns = value.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"turnsが空です: {path}:{line_number}")
            records.append(value)
    return records


def _format_record(record: dict[str, Any]) -> str:
    lines = ["<|startofconversation|>"]
    for index, turn in enumerate(record["turns"]):
        if not isinstance(turn, dict):
            raise TypeError(f"turn #{index}がobjectではありません")
        speaker = turn.get("speaker_id")
        text = turn.get("text")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f"turn #{index}のspeaker_idが空です")
        if not isinstance(text, str) or not text:
            raise ValueError(f"turn #{index}のtextが空です")
        lines.append(f"<|speaker:{speaker}|>{text}")
    lines.append("<|endofconversation|>")
    return "\n".join(lines)


def export_sources(
    input_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """splitごとにdataset別の会話ブロックとmanifestを作る。"""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    input_info: dict[str, dict[str, Any]] = {}
    for split in splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"会話JSONLが見つかりません: {input_path}")
        records = _read_records(input_path)
        input_info[split] = {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "record_count": len(records),
        }
        for record in records:
            source_records[split][str(record["dataset"])].append(record)

    outputs: dict[str, dict[str, Any]] = {}
    for split in splits:
        for source in sorted(source_records[split]):
            records = source_records[split][source]
            output_path = output_dir / f"{source}-{split}.txt"
            text = "\n\n".join(_format_record(record) for record in records) + "\n"
            output_path.write_text(text, encoding="utf-8")
            key = f"{source}:{split}"
            outputs[key] = {
                "source": source,
                "split": split,
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "conversation_count": len(records),
                "turn_count": sum(len(record["turns"]) for record in records),
                "character_count": sum(
                    len(turn["text"])
                    for record in records
                    for turn in record["turns"]
                ),
            }

    manifest = {
        "format": "conversation-source-split-v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_names": sorted({key.split(":", 1)[0] for key in outputs}),
        "splits": list(splits),
        "inputs": input_info,
        "outputs": outputs,
        "unit_rule": "一会話をstart/end markerを含む一つの論理単位として保存する。",
        "source_rule": "取り込み済みrecordのdatasetフィールドで分け、本文と話者markerを保持する。",
        "raw_data_rule": "入力元の公開リポジトリやmedilink_analysisの原本は変更しない。",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=("train", "validation", "test"),
        help="出力するsplit。省略時はtrain/validation/test全て",
    )
    args = parser.parse_args()
    splits = tuple(args.splits or ("train", "validation", "test"))
    manifest = export_sources(repo_path(args.input_dir), repo_path(args.output_dir), splits)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
