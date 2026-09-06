"""HTTP upload制限を避けるため、実験087のbundle分割片をColab側で結合する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

PARTS = sorted(Path("/content").glob("exp087-part-*"))
TARGET = Path("/content/exp087_bundle_reassembled.tar.gz")
EXPECTED_BYTES = 263_504_635
EXPECTED_SHA256 = "b67e0b307cb94926225ca0b868dc0b1b47283e54237e3e03e0aa711139e34d73"


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
        actual_bytes = TARGET.stat().st_size
        actual_sha256 = sha256(TARGET)
        if actual_bytes == EXPECTED_BYTES and actual_sha256 == EXPECTED_SHA256:
            print({"status": "already_verified", "bytes": actual_bytes, "sha256": actual_sha256}, flush=True)
            return
        raise FileExistsError(f"検証不能な既存bundleを上書きしません: {TARGET}")
    with TARGET.open("wb") as output:
        for part in PARTS:
            print(f"結合中: {part} ({part.stat().st_size} bytes)", flush=True)
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    actual_bytes = TARGET.stat().st_size
    actual_sha256 = sha256(TARGET)
    print({"bytes": actual_bytes, "sha256": actual_sha256}, flush=True)
    if actual_bytes != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"bundle検証失敗: bytes={actual_bytes}, sha256={actual_sha256}")


if __name__ == "__main__":
    main()
