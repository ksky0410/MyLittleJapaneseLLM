"""実験082の081基盤へのseed123・標準SFTをColab GPUで実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp082_bundle_reassembled.tar.gz")
PROJECT = Path("/content/small_llm_082")
CONFIG = "configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml"
BASE = "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt"
TRAIN_DATA = "artifacts/sft/issue1-both-balanced-v1/train.npz"
VALIDATION_DATA = "artifacts/sft/issue1-both-full-v1/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
TRAIN_SCRIPT = "scripts/train_sft_torch.py"
OUTPUT = "issue1-both-50m-sft-from-5m-two-pass-seed123-3k"


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
        CONFIG: "",
        BASE: "1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb",
        TRAIN_DATA: "645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44",
        VALIDATION_DATA: "fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1",
        REHEARSAL: "d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090",
        TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
        TRAIN_SCRIPT: "",
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
            "082 bundleの入力検証に失敗しました: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "082",
                "project_dir": str(PROJECT),
                "condition": "both-50m-sft-from-5m-two-pass-rehearsal020-seed123",
                "verified_inputs": len(required),
                "long_response_ratio": 0.0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    command = [
        sys.executable,
        str(PROJECT / TRAIN_SCRIPT),
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
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
