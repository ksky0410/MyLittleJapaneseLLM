"""日本語コーパスの軽い正規化、分割、記録。"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path


def normalize_line(line: str) -> str:
    """日本語を壊さない範囲でUnicodeと空白を正規化する。"""

    line = unicodedata.normalize("NFKC", line)
    line = line.replace("\t", " ").strip()
    return " ".join(line.split())


def read_documents(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"入力テキストが見つかりません: {source}")
    text = source.read_text(encoding="utf-8")
    documents = [normalize_line(line) for line in text.splitlines()]
    documents = [line for line in documents if line]
    if len(documents) < 2:
        raise ValueError("学習・検証に分けるため、2行以上のテキストを用意してください")
    return documents


def split_documents(
    documents: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio は0より大きく1未満で指定してください")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    output_dir: str | Path,
    train: list[str],
    val: list[str],
    source: Path,
    val_ratio: float,
    seed: int,
) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.txt"
    val_path = destination / "val.txt"
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val) + "\n", encoding="utf-8")
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "seed": seed,
        "val_ratio": val_ratio,
        "train_documents": len(train),
        "validation_documents": len(val),
        "train_characters": sum(map(len, train)),
        "validation_characters": sum(map(len, val)),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
