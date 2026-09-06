"""実験089の継続日本語事前学習をColab GPUで実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp089_bundle_reassembled.tar.gz")
PROJECT = Path("/content/small_llm_089")
CONFIG = "configs/issue1-both-50m-pretrain-5m-continuation-5k.toml"
INITIAL = "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt"
TRAIN_TOKENS = "artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin"
VAL_TOKENS = "artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
TRAIN_SCRIPT = "scripts/train_torch.py"
OUTPUT = "issue1-both-50m-pretrain-5m-continuation-5k"


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
        print(f"既存の展開済みprojectを検証して再利用します: {PROJECT}", flush=True)
    else:
        PROJECT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BUNDLE, "r:gz") as archive:
            archive.extractall(PROJECT, filter="data")
    required = {
        CONFIG: "669fddbfb2fbba31cdbc79f90d1afae5cb0185c830cd293c9d38afc5043dfa07",
        INITIAL: "1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb",
        "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.json": "4b6b56ad60730cc75a938dd8ef99aba6e713e03852043e8fe9175ef5d5c2813b",
        TRAIN_TOKENS: "54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c",
        VAL_TOKENS: "c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4",
        TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
        TRAIN_SCRIPT: "a4e926bbd6e93a544b3cceabf17ae0b5836d92396b95c3e4eb37a3992b3510df",
    }
    missing: list[str] = []
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in required.items():
        path = PROJECT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if expected and sha256(path) != expected:
            mismatches[relative] = {"expected": expected, "actual": sha256(path)}
    if missing or mismatches:
        raise RuntimeError(
            "089 bundleの入力検証に失敗しました: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, ensure_ascii=False)
        )
    print(json.dumps({
        "experiment": "089",
        "project_dir": str(PROJECT),
        "condition": "50m-pretrain-5m-continuation-5k",
        "verified_inputs": len(required),
        "max_steps": 5000,
        "initial_checkpoint": INITIAL,
    }, ensure_ascii=False), flush=True)
    command = [
        sys.executable, str(PROJECT / TRAIN_SCRIPT),
        "--config", CONFIG,
        "--initial-checkpoint", INITIAL,
        "--device", "auto",
    ]
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
