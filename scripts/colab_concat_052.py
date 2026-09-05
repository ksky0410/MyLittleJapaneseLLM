"""実験052の分割bundleをColab側で連結し、元hashを検証する。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_SHA256 = "94f28a741d6e3bebf922031ed8feafa1ecf2eaaacba54da8c38ab1e2950cbd35"
PARTS = [Path(f"/content/exp052_bundle_part_{index:02d}") for index in range(6)]
BUNDLE = Path("/content/exp052_bundle.tar.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in PARTS if not path.is_file()]
    if missing:
        raise FileNotFoundError("分割bundleが不足しています: " + ", ".join(missing))
    with BUNDLE.open("wb") as output:
        for part in PARTS:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    actual = sha256(BUNDLE)
    result = {
        "experiment": "052",
        "parts": [str(path) for path in PARTS],
        "bundle": str(BUNDLE),
        "bytes": BUNDLE.stat().st_size,
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": actual,
        "verified": actual == EXPECTED_SHA256,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if actual != EXPECTED_SHA256:
        raise RuntimeError("連結したbundleのSHA-256が一致しません")


if __name__ == "__main__":
    main()
