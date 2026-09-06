"""実験080の軽量成果物をColab側でarchive化する。"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

PROJECT = Path("/content/small_llm_080")
CONDITION = "issue1-both-50m-pretrain-10m-5k"
CHECKPOINT_ROOT = PROJECT / "artifacts/checkpoints" / CONDITION
SAMPLES_ROOT = PROJECT / "artifacts/samples" / CONDITION
ARCHIVE = Path("/content/exp080-lightweight.tar.gz")
MANIFEST = Path("/content/exp080-manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    roots = (CHECKPOINT_ROOT, SAMPLES_ROOT)
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    if not files:
        raise FileNotFoundError("実験080の軽量成果物が見つかりません")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT))
    checkpoints = [
        {
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(CHECKPOINT_ROOT.glob("*.pt"))
    ]
    manifest = {
        "experiment": "080",
        "condition": CONDITION,
        "lightweight_files": len(files),
        "lightweight_archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "checkpoints": checkpoints,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
