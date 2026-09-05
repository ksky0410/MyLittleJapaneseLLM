"""複数のSFT NPZを連結し、入力と配列のhashをmanifestへ保存する。"""

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


def concat_sft_npz(
    inputs: list[tuple[str, Path]], output_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """input_ids、target_ids、loss_maskを順番どおりに連結する。"""

    if not inputs:
        raise ValueError("入力NPZを少なくとも一つ指定してください")
    output_file = repo_path(output_path).resolve()
    manifest_file = repo_path(manifest_path).resolve()
    arrays_by_name: dict[str, dict[str, np.ndarray]] = {}
    input_stats: dict[str, Any] = {}
    expected_shape: tuple[int, int] | None = None
    for name, input_file in inputs:
        if not input_file.is_file():
            raise FileNotFoundError(f"入力NPZが見つかりません: {input_file}")
        with np.load(input_file) as arrays:
            required = {"input_ids", "target_ids", "loss_mask"}
            if set(arrays.files) != required:
                raise ValueError(f"NPZの配列は{required}だけにしてください: {arrays.files}")
            values = {key: arrays[key] for key in required}
        if values["input_ids"].shape != values["target_ids"].shape:
            raise ValueError(f"input_idsとtarget_idsのshapeが違います: {name}")
        if values["input_ids"].shape != values["loss_mask"].shape:
            raise ValueError(f"input_idsとloss_maskのshapeが違います: {name}")
        if expected_shape is None:
            expected_shape = (0, values["input_ids"].shape[1])
        if values["input_ids"].shape[1] != expected_shape[1]:
            raise ValueError(f"context lengthが一致していません: {name}")
        arrays_by_name[name] = values
        input_stats[name] = {
            "path": str(input_file),
            "sha256": sha256_file(input_file),
            "example_count": int(values["input_ids"].shape[0]),
            "response_token_count": int(values["loss_mask"].astype(np.int64).sum()),
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        input_ids=np.concatenate(
            [arrays_by_name[name]["input_ids"] for name, _ in inputs], axis=0
        ),
        target_ids=np.concatenate(
            [arrays_by_name[name]["target_ids"] for name, _ in inputs], axis=0
        ),
        loss_mask=np.concatenate(
            [arrays_by_name[name]["loss_mask"] for name, _ in inputs], axis=0
        ),
    )
    with np.load(output_file) as output_arrays:
        output_shape = list(output_arrays["input_ids"].shape)
        output_response_tokens = int(
            output_arrays["loss_mask"].astype(np.int64).sum()
        )
    manifest = {
        "format": "sft-npz-concat-v1",
        "inputs": input_stats,
        "output_path": str(output_file),
        "output_sha256": sha256_file(output_file),
        "output_example_count": output_shape[0],
        "output_response_token_count": output_response_tokens,
        "array_shape": output_shape,
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
    args = parser.parse_args()
    inputs = [parse_input(value) for value in args.input]
    print(
        json.dumps(
            concat_sft_npz(inputs, args.output, args.manifest),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
