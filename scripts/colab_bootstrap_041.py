"""実験041のColab bootstrapを固定configで起動する。"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path


EXPECTED_FILES = {
    "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml": "8b79fac4389c8bc781dafe1076e4083d19ca9c81543f15bfddf414b33ec01d95",
    "scripts/train_torch.py": "27d2fd7b93c456a598f16f7f83560af491c5ffb2243bcf2518b0a19a462b53d2",
    "scripts/colab_bootstrap_train.py": "bfac278f2e32f65d22379145d28380f6a687282c23267e51e65ec74916cab0c1",
    "artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model": "5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4",
    "artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin": "3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a",
    "artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin": "c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_bundle() -> Path:
    bundle = Path("/content/small_llm_bundle.tar.gz")
    project_dir = Path("/content/small_llm_041")
    if project_dir.exists() and any(project_dir.iterdir()):
        raise RuntimeError(
            f"{project_dir}が空ではありません。古いsessionを再利用せず、新規kernelを使ってください"
        )
    project_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(project_dir, filter="data")
    mismatches = {}
    for relative, expected in EXPECTED_FILES.items():
        actual_path = project_dir / relative
        actual = _sha256(actual_path) if actual_path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "041 bundleの入力または実行コードhashが一致しません: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    print(
        json.dumps(
            {
                "experiment": "041",
                "project_dir": str(project_dir),
                "verified_files": len(EXPECTED_FILES),
                "train_script_sha256": EXPECTED_FILES["scripts/train_torch.py"],
            },
            ensure_ascii=False,
        )
    )
    sys.path.insert(0, str(project_dir / "scripts"))
    return project_dir


if __name__ == "__main__":
    _extract_bundle()
    from colab_bootstrap_train import main

    sys.argv = [
        sys.argv[0],
        "--config",
        "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml",
        "--device",
        "auto",
    ]
    main()
