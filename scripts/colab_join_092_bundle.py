"""HTTP upload制限を避けるため、実験092のbundle分割片をColab側で結合する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

PARTS = sorted(Path("/content").glob("exp092-part-*"))
TARGET = Path("/content/exp092_bundle_reassembled.tar.gz")
EXPECTED_BYTES = 265098998
EXPECTED_SHA256 = "d3091b93d3b9c3318a4a6d35f97a3b73ab01bd5a3a19ac416dda54b33cdb215f"


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
    if EXPECTED_BYTES and actual_bytes != EXPECTED_BYTES:
        raise RuntimeError(f"bundle byte数検証失敗: {actual_bytes} != {EXPECTED_BYTES}")
    if EXPECTED_SHA256 and actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"bundle SHA-256検証失敗: {actual_sha256} != {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
