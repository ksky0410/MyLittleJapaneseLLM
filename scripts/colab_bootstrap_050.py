"""実験050の20M core/both比較をColab T4で順に実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp050_bundle.tar.gz")
PROJECT = Path("/content/small_llm_050")
CONFIGS = (
    "configs/issue1-core-20m-colab-2p5k.toml",
    "configs/issue1-both-20m-colab-2p5k.toml",
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
        *CONFIGS,
        "scripts/train_torch.py",
        "scripts/colab_bootstrap_train.py",
        "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model",
        "artifacts/tokens/issue1-core-1m-fineweb-train.bin",
        "artifacts/tokens/issue1-both-1m-fineweb-train.bin",
        "artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin",
    ]
    hashes = {}
    for relative in required:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(f"bundle内に必要なファイルがありません: {relative}")
        hashes[relative] = sha256(path)
    print(json.dumps({"experiment": "050", "project": str(PROJECT), "input_hashes": hashes}, ensure_ascii=False, indent=2))

    for config in CONFIGS:
        command = [
            sys.executable,
            str(PROJECT / "scripts" / "train_torch.py"),
            "--config",
            config,
            "--device",
            "auto",
        ]
        print(json.dumps({"experiment": "050", "starting_config": config}, ensure_ascii=False), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
