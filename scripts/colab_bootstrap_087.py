"""実験087の長文応答層化SFTをColab GPUで実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp087_bundle_reassembled.tar.gz")
PROJECT = Path("/content/small_llm_087")
CONFIG = "configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025.toml"
BASE = "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt"
BASE_METADATA = "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.json"
TRAIN_DATA = "artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz"
VALIDATION_DATA = "artifacts/sft/issue1-both-full-v1/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
TRAIN_SCRIPT = "scripts/train_sft_torch.py"
COMMON_TRAIN_SCRIPT = "scripts/train_torch.py"
OUTPUT = "issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025"


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
        CONFIG: "ef795c1d17de3e985e778bbdd77d2705b208682d48f6dca9b6efd65c6866af5c",
        BASE: "1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb",
        BASE_METADATA: "4b6b56ad60730cc75a938dd8ef99aba6e713e03852043e8fe9175ef5d5c2813b",
        TRAIN_DATA: "001dc022a998abc5756f641b199988112db77ff42903485ff7a6fd6bd0e028a3",
        VALIDATION_DATA: "fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1",
        REHEARSAL: "d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090",
        TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
        TRAIN_SCRIPT: "",
        COMMON_TRAIN_SCRIPT: "c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e",
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
            "087 bundleの入力検証に失敗しました: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "087",
                "project_dir": str(PROJECT),
                "condition": "both-50m-sft-770k-each-rehearsal020-long025-seed123-10k",
                "verified_inputs": len(required),
                "max_steps": 10000,
                "long_response_ratio": 0.25,
                "long_response_min_tokens": 24,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    command = [
        sys.executable,
        str(PROJECT / TRAIN_SCRIPT),
        "--config", CONFIG,
        "--base-checkpoint", BASE,
        "--train-data", TRAIN_DATA,
        "--validation-data", VALIDATION_DATA,
        "--output-dir", f"artifacts/checkpoints/{OUTPUT}",
        "--samples-dir", f"artifacts/samples/{OUTPUT}",
        "--lr-schedule-steps", "10000",
        "--eos-loss-weight", "0.5",
        "--rehearsal-tokens", REHEARSAL,
        "--rehearsal-ratio", "0.20",
        "--long-response-ratio", "0.25",
        "--long-response-min-tokens", "24",
        "--sample-template", "conversation",
        "--sample-speaker-a", "DA",
        "--sample-speaker-b", "DC",
        "--device", "auto",
    ]
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
