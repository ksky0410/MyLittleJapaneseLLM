"""複数sourceからresponse Token予算を均等配分したSFT NPZを作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import repo_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"入力はNAME=PATH形式で指定してください: {value}")
    return name.strip(), repo_path(path).resolve()


def select_until_budget(
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    loss_mask: np.ndarray,
    target_response_tokens: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """行を決定的な無作為順に選び、response Token予算を満たす。"""

    response_tokens = loss_mask.astype(np.int64).sum(axis=1)
    if int(response_tokens.sum()) < target_response_tokens:
        raise ValueError(
            f"sourceのresponse Token数が不足しています: {int(response_tokens.sum())} < {target_response_tokens}"
        )
    selected: list[int] = []
    total = 0
    for raw_index in rng.permutation(input_ids.shape[0]):
        index = int(raw_index)
        selected.append(index)
        total += int(response_tokens[index])
        if total >= target_response_tokens:
            break
    indices = np.asarray(selected, dtype=np.int64)
    return input_ids[indices], target_ids[indices], loss_mask[indices], total


def mix_sft_npz(
    inputs: list[tuple[str, Path]],
    output_path: str | Path,
    manifest_path: str | Path,
    target_response_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """各sourceへ均等に予算を割り当て、連結後に全体をshuffleする。"""

    if not inputs:
        raise ValueError("入力NPZを少なくとも一つ指定してください")
    if target_response_tokens <= 0:
        raise ValueError("target_response_tokensは正の整数で指定してください")
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    source_budget = target_response_tokens // len(inputs)
    budgets = [source_budget] * len(inputs)
    budgets[-1] += target_response_tokens - sum(budgets)
    rng = np.random.default_rng(seed)
    selected_by_source: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    source_stats: dict[str, Any] = {}
    context_length: int | None = None
    for (name, input_file), budget in zip(inputs, budgets, strict=True):
        if not input_file.is_file():
            raise FileNotFoundError(f"入力NPZが見つかりません: {input_file}")
        with np.load(input_file) as arrays:
            required = {"input_ids", "target_ids", "loss_mask"}
            if set(arrays.files) != required:
                raise ValueError(f"NPZの配列は{required}だけにしてください: {arrays.files}")
            input_ids = arrays["input_ids"]
            target_ids = arrays["target_ids"]
            loss_mask = arrays["loss_mask"]
            if input_ids.shape != target_ids.shape or input_ids.shape != loss_mask.shape:
                raise ValueError(f"配列のshapeが一致していません: {name}")
            if context_length is None:
                context_length = input_ids.shape[1]
            if input_ids.shape[1] != context_length:
                raise ValueError(f"context lengthが一致していません: {name}")
            selected = select_until_budget(
                input_ids, target_ids, loss_mask, budget, rng
            )
            selected_by_source.append(selected[:3])
            source_stats[name] = {
                "input_path": str(input_file),
                "input_sha256": sha256_file(input_file),
                "input_example_count": int(input_ids.shape[0]),
                "input_response_token_count": int(loss_mask.astype(np.int64).sum()),
                "target_response_tokens": budget,
                "selected_example_count": int(selected[0].shape[0]),
                "selected_response_token_count": selected[3],
            }

    input_ids = np.concatenate([item[0] for item in selected_by_source], axis=0)
    target_ids = np.concatenate([item[1] for item in selected_by_source], axis=0)
    loss_mask = np.concatenate([item[2] for item in selected_by_source], axis=0)
    order = rng.permutation(input_ids.shape[0])
    input_ids, target_ids, loss_mask = (
        input_ids[order],
        target_ids[order],
        loss_mask[order],
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )
    manifest = {
        "format": "sft-npz-equal-source-budget-v1",
        "seed": seed,
        "target_response_tokens": target_response_tokens,
        "sources": source_stats,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_example_count": int(input_ids.shape[0]),
        "output_response_token_count": int(loss_mask.astype(np.int64).sum()),
        "array_shape": list(input_ids.shape),
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-response-tokens", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inputs = [parse_input(value) for value in args.input]
    print(
        json.dumps(
            mix_sft_npz(
                inputs,
                args.output,
                args.manifest,
                args.target_response_tokens,
                args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
