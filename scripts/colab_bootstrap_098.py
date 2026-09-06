"""実験098の20M Token・50Mモデル事前学習をColab GPUで実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp098_bundle.tar.gz")
PROJECT = Path("/content/small_llm_098")
CONFIG = "configs/issue1-both-50m-pretrain-20m-40k.toml"
TRAIN_TOKENS = "artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-20m-v1-train.bin"
VAL_TOKENS = "artifacts/tokens/fineweb2-edu-japanese-v1-test.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
TRAIN_SCRIPT = "scripts/train_torch.py"
OUTPUT = "issue1-both-50m-pretrain-20m-40k"


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
        CONFIG: "",
        TRAIN_TOKENS: "",
        VAL_TOKENS: "",
        TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
        TRAIN_SCRIPT: "",
    }
    missing: list[str] = []
    for relative in required:
        if not (PROJECT / relative).is_file():
            missing.append(relative)
    if missing:
        raise RuntimeError(json.dumps({"missing": missing}, ensure_ascii=False))
    print(
        json.dumps(
            {
                "experiment": "098",
                "project_dir": str(PROJECT),
                "condition": OUTPUT,
                "verified_inputs": len(required),
                "train_tokens_sha256": sha256(PROJECT / TRAIN_TOKENS),
                "val_tokens_sha256": sha256(PROJECT / VAL_TOKENS),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    command = [sys.executable, str(PROJECT / TRAIN_SCRIPT), "--config", CONFIG, "--device", "auto"]
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
