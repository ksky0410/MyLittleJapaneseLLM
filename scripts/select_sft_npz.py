"""SFT NPZから応答Token予算を固定した決定的なsubsetを作る。"""

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


def select_examples(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    target_response_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """loss_maskの合計が指定予算以上になるまで行を無作為順に選ぶ。"""

    if target_response_tokens <= 0:
        raise ValueError("target_response_tokensは正の整数で指定してください")
    input_file = repo_path(input_path).resolve()
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
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
            raise ValueError("input_ids、target_ids、loss_maskのshapeが一致していません")
        response_tokens = loss_mask.astype(np.int64).sum(axis=1)

        total_response_tokens = int(response_tokens.sum())
        if total_response_tokens < target_response_tokens:
            raise ValueError(
                f"入力のresponse Token数が不足しています: {total_response_tokens} < {target_response_tokens}"
            )
        permutation = np.random.default_rng(seed).permutation(input_ids.shape[0])
        selected_indices: list[int] = []
        selected_response_tokens = 0
        for raw_index in permutation:
            index = int(raw_index)
            selected_indices.append(index)
            selected_response_tokens += int(response_tokens[index])
            if selected_response_tokens >= target_response_tokens:
                break
        indices = np.asarray(selected_indices, dtype=np.int64)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_file,
            input_ids=input_ids[indices],
            target_ids=target_ids[indices],
            loss_mask=loss_mask[indices],
        )

    manifest = {
        "format": "sft-npz-response-budget-selection-v1",
        "input_path": str(input_file),
        "input_sha256": sha256_file(input_file),
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "seed": seed,
        "target_response_tokens": target_response_tokens,
        "input_example_count": int(input_ids.shape[0]),
        "input_response_token_count": total_response_tokens,
        "selected_example_count": int(len(selected_indices)),
        "selected_response_token_count": selected_response_tokens,
        "array_shape": list(input_ids[indices].shape),
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
    parser.add_argument("--target-response-tokens", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            select_examples(
                args.input,
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
