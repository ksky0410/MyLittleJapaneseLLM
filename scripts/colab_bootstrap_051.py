"""実験051の20M SFT-only/rehearsal比較をColab T4で順に実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp051_bundle.tar.gz")
PROJECT = Path("/content/small_llm_051")
CONFIG = "configs/issue1-both-20m-sft-torch-colab-1k.toml"
BASE = "artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt"
TRAIN_DATA = "artifacts/sft/chat-v1-context256/train.npz"
VALIDATION_DATA = "artifacts/sft/chat-v1-context256/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_condition(output_name: str, samples_name: str, rehearsal: bool) -> None:
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
        f"artifacts/checkpoints/{output_name}",
        "--samples-dir",
        f"artifacts/samples/{samples_name}",
        "--device",
        "auto",
    ]
    if rehearsal:
        command.extend(
            [
                "--rehearsal-tokens",
                REHEARSAL,
                "--rehearsal-ratio",
                "0.25",
            ]
        )
    print(
        json.dumps(
            {
                "experiment": "051",
                "condition": "rehearsal" if rehearsal else "sft-only",
                "output_dir": output_name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT, check=True)


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
        "src/my_little_japanese_llm/torch_model.py",
        "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model",
        BASE,
        f"{BASE[:-3]}.json",
        TRAIN_DATA,
        VALIDATION_DATA,
        REHEARSAL,
    ]
    hashes = {}
    for relative in required:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(f"bundle内に必要なファイルがありません: {relative}")
        hashes[relative] = sha256(path)
    print(
        json.dumps(
            {"experiment": "051", "project": str(PROJECT), "input_hashes": hashes},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    run_condition(
        "issue1-both-20m-sft-torch-colab-1k",
        "issue1-both-20m-sft-torch-colab-1k",
        rehearsal=False,
    )
    run_condition(
        "issue1-both-20m-rehearsal-torch-colab-1k",
        "issue1-both-20m-rehearsal-torch-colab-1k",
        rehearsal=True,
    )


if __name__ == "__main__":
    main()
