"""実験086の軽量成果物とbest checkpointをColab側でarchive化する。"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

PROJECT = Path("/content/small_llm_086")
CONDITION = "issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each"
CHECKPOINT_ROOT = PROJECT / "artifacts/checkpoints" / CONDITION
SAMPLES_ROOT = PROJECT / "artifacts/samples" / CONDITION
ARCHIVE = Path("/content/exp086-lightweight.tar.gz")
BEST_ARCHIVE = Path("/content/exp086-best-checkpoint.tar")
MANIFEST = Path("/content/exp086-manifest.json")


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
        raise FileNotFoundError("実験086の軽量成果物が見つかりません")
    best_checkpoint = CHECKPOINT_ROOT / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"best checkpointが見つかりません: {best_checkpoint}")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT))
    with tarfile.open(BEST_ARCHIVE, "w") as archive:
        archive.add(best_checkpoint, arcname="best.pt")
    checkpoints = [
        {
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(CHECKPOINT_ROOT.glob("*.pt"))
    ]
    manifest = {
        "experiment": "086",
        "condition": CONDITION,
        "lightweight_files": len(files),
        "lightweight_archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "best_checkpoint_archive": str(BEST_ARCHIVE),
        "best_checkpoint_archive_bytes": BEST_ARCHIVE.stat().st_size,
        "best_checkpoint_archive_sha256": sha256(BEST_ARCHIVE),
        "checkpoints": checkpoints,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
