"""実験065のboth-SFT+rehearsal 0.25をColab T4で実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp065_bundle.tar.gz")
PROJECT = Path("/content/small_llm_065")
BASE = "artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt"
CONFIG = "configs/issue1-both-20m-sft-source-rehearsal025-colab-3k.toml"
TRAIN = "artifacts/sft/issue1-both-balanced-v1/train.npz"
VALIDATION = "artifacts/sft/issue1-both-full-v1/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
OUTPUT = "artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal025-colab-3k"
SAMPLES = "artifacts/samples/issue1-both-20m-sft-source-rehearsal025-colab-3k"


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
        BASE,
        CONFIG,
        TRAIN,
        VALIDATION,
        REHEARSAL,
        "scripts/train_sft_torch.py",
        "scripts/train_torch.py",
        "scripts/_common.py",
        "src/my_little_japanese_llm/config.py",
        "src/my_little_japanese_llm/data.py",
        "src/my_little_japanese_llm/sft.py",
        "src/my_little_japanese_llm/tokenizer.py",
        "src/my_little_japanese_llm/torch_model.py",
        "src/my_little_japanese_llm/training.py",
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
            {"experiment": "065", "project": str(PROJECT), "input_hashes": hashes},
            ensure_ascii=False,
            indent=2,
        )
    )

    command = [
        sys.executable,
        str(PROJECT / "scripts" / "train_sft_torch.py"),
        "--config",
        CONFIG,
        "--base-checkpoint",
        BASE,
        "--train-data",
        TRAIN,
        "--validation-data",
        VALIDATION,
        "--output-dir",
        OUTPUT,
        "--samples-dir",
        SAMPLES,
        "--lr-schedule-steps",
        "3000",
        "--eos-loss-weight",
        "0.5",
        "--rehearsal-tokens",
        REHEARSAL,
        "--rehearsal-ratio",
        "0.25",
        "--sample-template",
        "conversation",
        "--sample-speaker-a",
        "DA",
        "--sample-speaker-b",
        "DC",
        "--device",
        "auto",
    ]
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
