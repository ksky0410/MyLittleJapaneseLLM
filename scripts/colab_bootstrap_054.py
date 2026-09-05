"""実験054の20M rehearsal ratio 0.50長時間SFTをColab T4で実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp054_bundle.tar.gz")
PROJECT = Path("/content/small_llm_054")
CONFIG = "configs/issue1-both-20m-rehearsal-ratio050-colab-3k.toml"
BASE = "artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt"
TRAIN_DATA = "artifacts/sft/chat-v1-context256/train.npz"
VALIDATION_DATA = "artifacts/sft/chat-v1-context256/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
OUTPUT = "artifacts/checkpoints/issue1-both-20m-rehearsal-ratio050-colab-3k"
SAMPLES = "artifacts/samples/issue1-both-20m-rehearsal-ratio050-colab-3k"


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
    required = [
        CONFIG,
        "scripts/train_sft_torch.py",
        "scripts/train_torch.py",
        "scripts/_common.py",
        BASE,
        f"{BASE[:-3]}.json",
        TRAIN_DATA,
        VALIDATION_DATA,
        REHEARSAL,
        "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model",
    ]
    hashes = {}
    for relative in required:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(f"bundle内に必要なファイルがありません: {relative}")
        hashes[relative] = sha256(path)
    print(
        json.dumps(
            {"experiment": "054", "project": str(PROJECT), "input_hashes": hashes},
            ensure_ascii=False,
            indent=2,
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
        OUTPUT,
        "--samples-dir",
        SAMPLES,
        "--rehearsal-tokens",
        REHEARSAL,
        "--rehearsal-ratio",
        "0.50",
        "--sample-template",
        "conversation",
        "--sample-speaker-a",
        "DA",
        "--sample-speaker-b",
        "DC",
        "--device",
        "auto",
    ]
    print(json.dumps({"experiment": "054", "condition": "ratio050", "steps": 3000}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
