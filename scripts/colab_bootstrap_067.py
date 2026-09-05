"""実験067のboth-SFT rehearsal ratio 0.20をColab GPUで実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp067_bundle.tar.gz")
PROJECT = Path("/content/small_llm_067")
CONFIG = "configs/issue1-both-20m-sft-source-rehearsal020-colab-3k.toml"
BASE = "artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt"
TRAIN_DATA = "artifacts/sft/issue1-both-balanced-v1/train.npz"
VALIDATION_DATA = "artifacts/sft/issue1-both-full-v1/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
OUTPUT = "issue1-both-20m-sft-source-rehearsal020-colab-3k"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not BUNDLE.is_file():
        raise FileNotFoundError(f"bundleが見つかりません: {BUNDLE}")
    if PROJECT.exists() and any(PROJECT.iterdir()):
        raise RuntimeError(f"既存の実験ディレクトリを上書きしません: {PROJECT}")
    PROJECT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as archive:
        archive.extractall(PROJECT, filter="data")

    required = {
        CONFIG: None,
        BASE: "326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751",
        TRAIN_DATA: "645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44",
        VALIDATION_DATA: "fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1",
        REHEARSAL: "d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090",
        TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
    }
    missing = []
    mismatches = {}
    for relative, expected in required.items():
        path = PROJECT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if expected is not None and sha256(path) != expected:
            mismatches[relative] = {"expected": expected, "actual": sha256(path)}
    if missing or mismatches:
        raise RuntimeError(
            "067 bundleの入力検証に失敗しました: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "067",
                "project_dir": str(PROJECT),
                "condition": "both-sft-rehearsal-ratio020",
                "verified_inputs": len(required),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "train_sft_torch.py"),
        "--config",
        CONFIG,
        "--base-checkpoint",
        BASE,
        "--train-data",
        TRAIN_DATA,
        "--validation-data",
        VALIDATION_DATA,
        "--output-dir",
        f"artifacts/checkpoints/{OUTPUT}",
        "--samples-dir",
        f"artifacts/samples/{OUTPUT}",
        "--lr-schedule-steps",
        "3000",
        "--eos-loss-weight",
        "0.5",
        "--rehearsal-tokens",
        REHEARSAL,
        "--rehearsal-ratio",
        "0.20",
        "--sample-template",
        "conversation",
        "--sample-speaker-a",
        "DA",
        "--sample-speaker-b",
        "DC",
        "--device",
        "auto",
    ]
    print(json.dumps({"experiment": "067", "steps": 3000}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
