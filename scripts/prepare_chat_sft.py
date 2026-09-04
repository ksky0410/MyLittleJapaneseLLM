"""会話JSONLを、応答部分だけを学習するSFT用NPZへ整形する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from _common import repo_path

from my_little_japanese_llm.tokenizer import load_processor

CONVERSATION_START = "<|startofconversation|>"
CONVERSATION_END = "<|endofconversation|>"
DEFAULT_SEED = 42
SPLITS = ("train", "validation")


def sha256_file(path: str | Path) -> str:
    """ファイルのSHA-256を返す。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(processor: Any, text: str) -> list[int]:
    """SentencePiece processorまたはテスト用processorで本文をToken化する。"""

    return [int(token) for token in processor.encode(text, out_type=int)]


def _validate_turns(record: dict[str, Any]) -> list[dict[str, str]]:
    """既存conversation importerのturn schemaを検証して取り出す。"""

    turns = record.get("turns")
    if not isinstance(turns, list):
        raise TypeError("会話recordのturnsは配列で指定してください")
    validated: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise TypeError(f"turn #{index}がobjectではありません")
        speaker_id = turn.get("speaker_id")
        text = turn.get("text")
        if not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError(f"turn #{index}のspeaker_idが空です")
        if not isinstance(text, str) or not text:
            raise ValueError(f"turn #{index}のtextが空です")
        validated.append({"speaker_id": speaker_id, "text": text})
    return validated


def build_conversation_example(
    turns: Sequence[dict[str, str]],
    target_index: int,
    processor: Any,
) -> tuple[list[int], list[int], int]:
    """一つの応答例を、padding前のids・maskとして作る。

    ``target_index``より前のturnを文脈にし、target turnの本文とその直後の
    EOSだけを1にする。end markerはEOSの後ろに置き、教師対象にはしない。
    返り値は ``(ids, loss_mask, response_body_token_count)`` である。
    """

    if len(turns) < 2:
        raise ValueError("SFT例には少なくとも2発話が必要です")
    if target_index < 1 or target_index >= len(turns):
        raise ValueError("target_indexは1以上かつturn数未満で指定してください")
    validated_turns = _validate_turns({"turns": list(turns)})

    eos_id = int(processor.eos_id())
    ids = _encode(processor, CONVERSATION_START)
    mask = [0] * len(ids)
    response_body_token_count = 0
    for index, turn in enumerate(validated_turns[: target_index + 1]):
        speaker_marker = f"<|speaker:{turn['speaker_id']}|>"
        speaker_ids = _encode(processor, speaker_marker)
        ids.extend(speaker_ids)
        mask.extend([0] * len(speaker_ids))

        body_ids = _encode(processor, turn["text"])
        ids.extend(body_ids)
        is_target = index == target_index
        mask.extend([1 if is_target else 0] * len(body_ids))
        if is_target:
            response_body_token_count += len(body_ids)

        ids.append(eos_id)
        mask.append(1 if is_target else 0)

    end_ids = _encode(processor, CONVERSATION_END)
    ids.extend(end_ids)
    mask.extend([0] * len(end_ids))
    return ids, mask, response_body_token_count


def truncate_and_pad(
    ids: Sequence[int],
    loss_mask: Sequence[int],
    context_length: int,
    pad_id: int,
) -> tuple[list[int], list[int], bool]:
    """idsとmaskをcontext_length+1へ左切り・右paddingする。"""

    if context_length < 2:
        raise ValueError("context_lengthは2以上で指定してください")
    if len(ids) != len(loss_mask):
        raise ValueError("idsとloss_maskの長さが一致していません")
    sequence_length = context_length + 1
    truncated = len(ids) > sequence_length
    if truncated:
        ids = ids[-sequence_length:]
        loss_mask = loss_mask[-sequence_length:]
    padded_ids = list(ids) + [int(pad_id)] * (sequence_length - len(ids))
    padded_mask = list(loss_mask) + [0] * (sequence_length - len(loss_mask))
    return padded_ids, padded_mask, truncated


def make_training_example(
    turns: Sequence[dict[str, str]],
    target_index: int,
    processor: Any,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | bool]]:
    """一つの会話turnから、学習配列一行を作る。"""

    ids, mask, response_body_token_count = build_conversation_example(
        turns, target_index, processor
    )
    padded_ids, padded_mask, truncated = truncate_and_pad(
        ids, mask, context_length, int(processor.pad_id())
    )
    return (
        np.asarray(padded_ids[:-1], dtype=np.int32),
        np.asarray(padded_ids[1:], dtype=np.int32),
        np.asarray(padded_mask[1:], dtype=np.uint8),
        {
            "response_body_token_count": response_body_token_count,
            "response_token_count": int(sum(padded_mask)),
            "truncated": truncated,
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを読み込み、空行を除いてobjectだけを返す。"""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONLを読めません: {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONLのrecordがobjectではありません: {path}:{line_number}"
                )
            records.append(value)
    return records


def _resolve_input_splits(input_path: str | Path) -> dict[str, Path]:
    """入力ディレクトリからtrain/validation JSONLを解決する。"""

    root = repo_path(input_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"入力ディレクトリが見つかりません: {root}")
    paths = {split: root / f"{split}.jsonl" for split in SPLITS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"必要な入力JSONLが見つかりません: {', '.join(missing)}"
        )
    return paths


def _input_digest(paths: dict[str, Path], hashes: dict[str, str]) -> str:
    """split名と各JSONLのhashから入力全体の決定的なhashを作る。"""

    value = "".join(f"{split}\t{hashes[split]}\n" for split in SPLITS)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prepare_split(
    records: Iterable[dict[str, Any]],
    processor: Any,
    context_length: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """一つのsplitを整形し、配列と集計値を返す。"""

    examples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | bool]]
    ] = []
    conversation_count = 0
    short_conversation_count = 0
    for record in records:
        turns = _validate_turns(record)
        conversation_count += 1
        if len(turns) < 2:
            short_conversation_count += 1
            continue
        for target_index in range(1, len(turns)):
            examples.append(
                make_training_example(turns, target_index, processor, context_length)
            )

    random.Random(seed).shuffle(examples)
    if examples:
        input_ids = np.stack([example[0] for example in examples])
        target_ids = np.stack([example[1] for example in examples])
        loss_mask = np.stack([example[2] for example in examples])
    else:
        shape = (0, context_length)
        input_ids = np.empty(shape, dtype=np.int32)
        target_ids = np.empty(shape, dtype=np.int32)
        loss_mask = np.empty(shape, dtype=np.uint8)

    stats = {
        "conversation_count": conversation_count,
        "short_conversation_count": short_conversation_count,
        "example_count": len(examples),
        "response_token_count": int(
            sum(int(example[3]["response_token_count"]) for example in examples)
        ),
        "response_body_token_count": int(
            sum(int(example[3]["response_body_token_count"]) for example in examples)
        ),
        "truncated_example_count": sum(
            bool(example[3]["truncated"]) for example in examples
        ),
        "array_shape": list(input_ids.shape),
    }
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
    }, stats


def prepare_chat_sft(
    tokenizer_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    context_length: int,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """train/validation会話をSFT NPZへ変換し、manifestを保存する。"""

    if context_length < 2:
        raise ValueError("context_lengthは2以上で指定してください")
    tokenizer_file = repo_path(tokenizer_path).resolve()
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizerが見つかりません: {tokenizer_file}")
    split_paths = _resolve_input_splits(input_path)
    input_hashes = {split: sha256_file(path) for split, path in split_paths.items()}
    processor = load_processor(tokenizer_file)
    output_dir = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest: dict[str, Any] = {}
    for split in SPLITS:
        arrays, stats = _prepare_split(
            _read_jsonl(split_paths[split]), processor, context_length, seed
        )
        output_file = output_dir / f"{split}.npz"
        np.savez_compressed(output_file, **arrays)
        split_manifest[split] = {
            "input_path": str(split_paths[split]),
            "input_sha256": input_hashes[split],
            "npz_path": str(output_file),
            "npz_sha256": sha256_file(output_file),
            **stats,
        }

    manifest: dict[str, Any] = {
        "format": "chat-sft-preparation-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(repo_path(input_path).resolve()),
        "input_sha256": _input_digest(split_paths, input_hashes),
        "tokenizer_path": str(tokenizer_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "context_length": context_length,
        "sequence_length": context_length + 1,
        "seed": seed,
        "special_tokens": {
            "conversation_start": CONVERSATION_START,
            "conversation_end": CONVERSATION_END,
            "eos_id": int(processor.eos_id()),
            "pad_id": int(processor.pad_id()),
        },
        "splits": split_manifest,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model")
    parser.add_argument(
        "--input", required=True, help="train/validation JSONLのディレクトリ"
    )
    parser.add_argument(
        "--output", required=True, help="train.npz/validation.npzの出力ディレクトリ"
    )
    parser.add_argument("--manifest", required=True, help="出力manifest JSON")
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = prepare_chat_sft(
        args.tokenizer,
        args.input,
        args.output,
        args.manifest,
        args.context_length,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
