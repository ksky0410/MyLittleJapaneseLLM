"""実験056の20M modern-architecture pretrainingをColab T4で実行する。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path("/content/exp056_bundle.tar.gz")
PROJECT = Path("/content/small_llm_056")
CONFIG = "configs/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k.toml"

EXPECTED_FILES = {
    CONFIG: "b382e890e0cda18db24754662d6a30b8e4fb802092e58b20cc3c3654dd65007d",
    "scripts/train_torch.py": "c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e",
    "scripts/_common.py": "f6ccaee55a17750713d31bf0adcb93d0a08d4c9c899879804068391309a65d9e",
    "src/my_little_japanese_llm/config.py": "3ae8717563dd596a4af0f6d1321c207bbc94fd46e95e42265dde07d6903aec2c",
    "src/my_little_japanese_llm/torch_model.py": "adf294d9de0c794586a85ca1256b432569c60b989cf6eaf72a055ad795b7bd55",
    "src/my_little_japanese_llm/data.py": "c62cf74f017fa25e65a41ef833427b1b68f7d978af1006c4623644c5a549a166",
    "src/my_little_japanese_llm/tokenizer.py": "38f95bac7e340efeeb36262a7671bccaafd435434ab96d3544b442f17b4784f4",
    "src/my_little_japanese_llm/training.py": "0b70a4c2dfeec988ca1a11ee40cd6dfb39ab68318fb81d3250f5adb8ecdb5b02",
    "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model": "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
    "artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin": "3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a",
    "artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin": "c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4",
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
            "056 bundleの入力または実行コードhashが一致しません: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "056",
                "project_dir": str(PROJECT),
                "verified_files": len(EXPECTED_FILES),
                "config": CONFIG,
                "device_request": "auto",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "train_torch.py"),
            "--config",
            CONFIG,
            "--device",
            "auto",
        ],
        cwd=PROJECT,
        check=True,
    )


if __name__ == "__main__":
    main()
