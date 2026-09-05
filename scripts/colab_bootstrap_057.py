"""実験057のEOS loss weight ablation SFTをColab T4で実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp057_bundle.tar.gz")
PROJECT = Path("/content/small_llm_057")
CONFIG = "configs/issue1-056base-rehearsal-ratio050-eos-ablation-colab-3k.toml"
BASE = "artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k/best.pt"
TRAIN_DATA = "artifacts/sft/chat-v1-context256/train.npz"
VALIDATION_DATA = "artifacts/sft/chat-v1-context256/validation.npz"
REHEARSAL = "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin"
TOKENIZER = "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model"
CONDITIONS = {
    "eos100": ("1.00", "issue1-056base-rehearsal-ratio050-eos100-colab-3k"),
    "eos050": ("0.50", "issue1-056base-rehearsal-ratio050-eos050-colab-3k"),
}

EXPECTED_FILES = {
    CONFIG: "73043053375a67210663684a999bc67340017936fd8486807374a933be7e4c2f",
    "scripts/train_sft_torch.py": "100c654d28fdd2817d8ea377588333802817f05c28a4a049aaf03942723fbbfc",
    "scripts/train_torch.py": "c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e",
    "scripts/_common.py": "f6ccaee55a17750713d31bf0adcb93d0a08d4c9c899879804068391309a65d9e",
    "src/my_little_japanese_llm/config.py": "3ae8717563dd596a4af0f6d1321c207bbc94fd46e95e42265dde07d6903aec2c",
    "src/my_little_japanese_llm/torch_model.py": "adf294d9de0c794586a85ca1256b432569c60b989cf6eaf72a055ad795b7bd55",
    BASE: "476d848edd7566ff259ee74469912c5ad828a471a44bca1e53b20cd8bc571b21",
    f"{BASE[:-3]}.json": "501f24e62dbe876cbd3538753f42125fb384b00307a22c2d2fec765998c38d5a",
    TRAIN_DATA: "400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609",
    VALIDATION_DATA: "5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4",
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
            "057 bundleの入力または実行コードhashが一致しません: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    print(json.dumps({"experiment": "057", "verified_files": len(EXPECTED_FILES)}, ensure_ascii=False), flush=True)
    for condition, (weight, output_name) in CONDITIONS.items():
        command = [
            sys.executable,
            str(PROJECT / "scripts" / "train_sft_torch.py"),
            "--config", CONFIG,
            "--base-checkpoint", BASE,
            "--train-data", TRAIN_DATA,
            "--validation-data", VALIDATION_DATA,
            "--output-dir", f"artifacts/checkpoints/{output_name}",
            "--samples-dir", f"artifacts/samples/{output_name}",
            "--rehearsal-tokens", REHEARSAL,
            "--rehearsal-ratio", "0.50",
            "--eos-loss-weight", weight,
            "--sample-template", "conversation",
            "--sample-speaker-a", "DA",
            "--sample-speaker-b", "DC",
            "--device", "auto",
        ]
        print(json.dumps({"experiment": "057", "condition": condition, "eos_loss_weight": float(weight), "steps": 3000}, ensure_ascii=False), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
