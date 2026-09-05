"""実験065のboth-SFT rehearsal ratio 0.25をColab T4で実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/exp065_bundle.tar.gz")
PROJECT = Path("/content/small_llm_065")
CONFIG = "configs/issue1-both-20m-sft-rehearsal-ratio025-mps-3k.toml"
BASE = "artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt"
TRAIN_DATA = "artifacts/sft/issue1-both-balanced-v1/train.npz"
VALIDATION_DATA = "artifacts/sft/issue1-both-full-v1/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
OUTPUT = "issue1-both-20m-sft-rehearsal-ratio025-colab-3k"

EXPECTED_FILES = {
    CONFIG: "aee4d87dbf26da98f9febde94aeb8daabaad3b80e589d31ab98a13b439d7ec83",
    "scripts/train_sft_torch.py": "9df48784c54229c3590bda341087d5802ffcb52f2afa869cb25df63562b9f237",
    "scripts/train_torch.py": "c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e",
    "scripts/_common.py": "f6ccaee55a17750713d31bf0adcb93d0a08d4c9c899879804068391309a65d9e",
    "src/my_little_japanese_llm/config.py": "3ae8717563dd596a4af0f6d1321c207bbc94fd46e95e42265dde07d6903aec2c",
    "src/my_little_japanese_llm/torch_model.py": "adf294d9de0c794586a85ca1256b432569c60b989cf6eaf72a055ad795b7bd55",
    BASE: "326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751",
    f"{BASE[:-3]}.json": "a585fa721566432800d06bd8d6702ecc04b22468453c729dac8b4cbfd77ccafb",
    TRAIN_DATA: "645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44",
    VALIDATION_DATA: "fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1",
    REHEARSAL: "d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090",
    TOKENIZER: "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
}


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
    mismatches = {}
    for relative, expected in EXPECTED_FILES.items():
        path = PROJECT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "065 bundleの入力または実行コードhashが一致しません: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "065",
                "project_dir": str(PROJECT),
                "verified_files": len(EXPECTED_FILES),
                "condition": "both-sft-rehearsal-ratio025",
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
        "--rehearsal-tokens",
        REHEARSAL,
        "--rehearsal-ratio",
        "0.25",
        "--lr-schedule-steps",
        "3000",
        "--eos-loss-weight",
        "0.5",
        "--sample-template",
        "conversation",
        "--sample-speaker-a",
        "DA",
        "--sample-speaker-b",
        "DC",
        "--device",
        "auto",
    ]
    print(
        json.dumps({"experiment": "065", "steps": 3000}, ensure_ascii=False), flush=True
    )
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
