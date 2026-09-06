"""HTTP upload制限を避けるため、実験089のbundle分割片をColab側で結合する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

PARTS = sorted(Path("/content").glob("exp089-part-*"))
TARGET = Path("/content/exp089_bundle_reassembled.tar.gz")
EXPECTED_BYTES = 0
EXPECTED_SHA256 = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not PARTS:
        raise FileNotFoundError("bundle分割片が見つかりません")
    if TARGET.exists():
        raise FileExistsError(f"既存bundleを上書きしません: {TARGET}")
    with TARGET.open("wb") as output:
        for part in PARTS:
            print(f"結合中: {part} ({part.stat().st_size} bytes)", flush=True)
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    actual_bytes = TARGET.stat().st_size
    actual_sha256 = sha256(TARGET)
    print({"bytes": actual_bytes, "sha256": actual_sha256}, flush=True)
    if EXPECTED_BYTES and (actual_bytes != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256):
        raise RuntimeError(f"bundle検証失敗: bytes={actual_bytes}, sha256={actual_sha256}")


if __name__ == "__main__":
    main()
