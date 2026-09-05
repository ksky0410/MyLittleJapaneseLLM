"""実験041の軽量成果物をColab側でarchive化して回収しやすくする。"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


PROJECT = Path("/content/small_llm")
CHECKPOINT_DIR = PROJECT / "artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k"
SAMPLES_DIR = PROJECT / "artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k"
LIGHT_ARCHIVE = Path("/content/exp041-lightweight.tar.gz")
MANIFEST = Path("/content/exp041-manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    roots = (CHECKPOINT_DIR, SAMPLES_DIR)
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    if not files:
        raise FileNotFoundError("実験041の軽量成果物が見つかりません")

    with tarfile.open(LIGHT_ARCHIVE, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT))

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    checkpoint_info = [
        {
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in checkpoints
    ]
    manifest = {
        "experiment": "041",
        "lightweight_archive": str(LIGHT_ARCHIVE),
        "lightweight_files": len(files),
        "checkpoints": checkpoint_info,
        "archive_bytes": LIGHT_ARCHIVE.stat().st_size,
        "archive_sha256": sha256(LIGHT_ARCHIVE),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
