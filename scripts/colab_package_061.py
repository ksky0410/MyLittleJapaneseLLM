"""実験061の軽量成果物をColab側でarchive化する。"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


PROJECT = Path("/content/small_llm_061")
CONDITIONS = (
    "issue1-056base-rehearsal-ratio025-eos050-colab-6k-fixedlr3k",
    "issue1-056base-rehearsal-ratio075-eos050-colab-6k-fixedlr3k",
)
ARCHIVE = Path("/content/exp061-lightweight.tar.gz")
MANIFEST = Path("/content/exp061-manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    roots = tuple(
        root
        for condition in CONDITIONS
        for root in (
            PROJECT / "artifacts/checkpoints" / condition,
            PROJECT / "artifacts/samples" / condition,
        )
    )
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    if not files:
        raise FileNotFoundError("実験061の軽量成果物が見つかりません")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT))
    checkpoints = [
        {
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for condition in CONDITIONS
        for path in sorted((PROJECT / "artifacts/checkpoints" / condition).glob("*.pt"))
    ]
    manifest = {
        "experiment": "061",
        "conditions": CONDITIONS,
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
