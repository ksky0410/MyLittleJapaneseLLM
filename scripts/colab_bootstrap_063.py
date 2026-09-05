"""実験063のRPC/MRMP source-specific SFTをColab T4で順に実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp063_bundle.tar.gz")
PROJECT = Path("/content/small_llm_063")
BASE = "artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/best.pt"
JOBS = (
    {
        "name": "rpc",
        "config": "configs/issue1-rpc-20m-sft-source-colab-3k.toml",
        "train": "artifacts/sft/issue1-rpc-balanced-v1/train.npz",
        "validation": "artifacts/sft/issue1-rpc-full-v1/validation.npz",
        "output": "artifacts/checkpoints/issue1-rpc-20m-sft-source-colab-3k",
        "samples": "artifacts/samples/issue1-rpc-20m-sft-source-colab-3k",
    },
    {
        "name": "mrmp",
        "config": "configs/issue1-mrmp-20m-sft-source-colab-3k.toml",
        "train": "artifacts/sft/issue1-mrmp-balanced-v1/train.npz",
        "validation": "artifacts/sft/issue1-mrmp-full-v1/validation.npz",
        "output": "artifacts/checkpoints/issue1-mrmp-20m-sft-source-colab-3k",
        "samples": "artifacts/samples/issue1-mrmp-20m-sft-source-colab-3k",
    },
)


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
        "scripts/train_sft_torch.py",
        "scripts/_common.py",
        "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model",
    ]
    for job in JOBS:
        required.extend([job["config"], job["train"], job["validation"]])
    hashes = {}
    for relative in required:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(f"bundle内に必要なファイルがありません: {relative}")
        hashes[relative] = sha256(path)
    print(
        json.dumps(
            {"experiment": "063", "project": str(PROJECT), "input_hashes": hashes},
            ensure_ascii=False,
            indent=2,
        )
    )

    for job in JOBS:
        command = [
            sys.executable,
            str(PROJECT / "scripts" / "train_sft_torch.py"),
            "--config",
            job["config"],
            "--base-checkpoint",
            BASE,
            "--train-data",
            job["train"],
            "--validation-data",
            job["validation"],
            "--output-dir",
            job["output"],
            "--samples-dir",
            job["samples"],
            "--lr-schedule-steps",
            "3000",
            "--eos-loss-weight",
            "0.5",
            "--device",
            "auto",
        ]
        print(
            json.dumps(
                {"experiment": "063", "starting_source": job["name"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
