"""SFT NPZの例を決定的に複製し、特定sourceの学習比率を上げる。"""

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


def repeat_sft_npz(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    """input_ids、target_ids、loss_maskをrepeat回分作り、全体をshuffleする。"""

    if repeat < 1:
        raise ValueError("repeatは1以上の整数で指定してください")
    input_file = repo_path(input_path).resolve()
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"入力NPZが見つかりません: {input_file}")
    with np.load(input_file) as arrays:
        required = {"input_ids", "target_ids", "loss_mask"}
        if set(arrays.files) != required:
            raise ValueError(f"NPZの配列は{required}だけにしてください: {arrays.files}")
        values = {key: arrays[key] for key in required}
    if values["input_ids"].shape != values["target_ids"].shape:
        raise ValueError("input_idsとtarget_idsのshapeが一致していません")
    if values["input_ids"].shape != values["loss_mask"].shape:
        raise ValueError("input_idsとloss_maskのshapeが一致していません")
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(values["input_ids"].shape[0]) for _ in range(repeat)]
    repeated: dict[str, np.ndarray] = {}
    for key, array in values.items():
        repeated[key] = np.concatenate(
            [array[order] for order in orders], axis=0
        )
    order = rng.permutation(repeated["input_ids"].shape[0])
    for key in repeated:
        repeated[key] = repeated[key][order]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **repeated)
    response_tokens = int(repeated["loss_mask"].astype(np.int64).sum())
    manifest = {
        "format": "sft-npz-repeat-v1",
        "input_path": str(input_file),
        "input_sha256": sha256_file(input_file),
        "input_example_count": int(values["input_ids"].shape[0]),
        "input_response_token_count": int(
            values["loss_mask"].astype(np.int64).sum()
        ),
        "repeat": repeat,
        "seed": seed,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_example_count": int(repeated["input_ids"].shape[0]),
        "output_response_token_count": response_tokens,
        "array_shape": list(repeated["input_ids"].shape),
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
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            repeat_sft_npz(
                args.input,
                args.output,
                args.manifest,
                repeat=args.repeat,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
