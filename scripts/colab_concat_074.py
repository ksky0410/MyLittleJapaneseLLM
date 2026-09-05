"""実験074の分割bundleをColab上で連結し、bytesとSHA-256を検証する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

PARTS = tuple(Path(f"/content/exp074_bundle_part_{index:02d}") for index in range(6))
BUNDLE = Path("/content/exp074_bundle.tar.gz")
EXPECTED_BYTES = 236462382
EXPECTED_SHA256 = "4afb025504ca43fab266d5e44c7b64a9bf1cd8c8cfe396901aa7e2aca3b49bc6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in PARTS if not path.is_file()]
    if missing:
        raise FileNotFoundError("bundle partが不足しています: " + ", ".join(missing))
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
        raise ValueError(
            f"bundle SHA-256不一致: {actual_sha256} != {EXPECTED_SHA256}"
        )
    print(
        {
            "bundle": str(BUNDLE),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "parts": len(PARTS),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
