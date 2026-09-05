"""実験058の分割bundleをColab上で連結し、SHA-256を検証する。"""

from __future__ import annotations

import hashlib
from pathlib import Path


PARTS = tuple(Path(f"/content/exp058_bundle_part_{index:02d}") for index in range(6))
BUNDLE = Path("/content/exp058_bundle.tar.gz")
EXPECTED_BYTES = 271784268
EXPECTED_SHA256 = "66b9105cdce3a8169bb586ef176971c188c146b9595131b6d73b0bfa2d38e59a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in PARTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle partが不足しています: {missing}")
    with BUNDLE.open("wb") as output:
        for part in PARTS:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    actual_bytes = BUNDLE.stat().st_size
    actual_sha256 = sha256(BUNDLE)
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(f"bundle bytes不一致: {actual_bytes} != {EXPECTED_BYTES}")
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(f"bundle SHA-256不一致: {actual_sha256} != {EXPECTED_SHA256}")
    print({"bytes": actual_bytes, "sha256": actual_sha256})


if __name__ == "__main__":
    main()
