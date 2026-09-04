"""複数のUTF-8テキストコーパスを、重複なく決定的に混合する。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONVERSATION_START = "<|startofconversation|>"
CONVERSATION_END = "<|endofconversation|>"
SourceSpec = tuple[str, str | Path] | tuple[str, str | Path, float]


@dataclass
class _Source:
    name: str
    path: Path
    weight: float
    text: str
    input_bytes: bytes
    units: list[str]
    unique_units: list[str]
    duplicate_units_removed: int = 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_units(text: str, path: Path) -> list[str]:
    """通常の非空行、または会話ブロックを論理単位へ変換する。"""

    units: list[str] = []
    conversation: list[str] | None = None
    for line in text.splitlines():
        marker = line.strip()
        if marker == CONVERSATION_START:
            if conversation is not None:
                raise ValueError(f"会話区切りが入れ子になっています: {path}")
            conversation = [line]
        elif marker == CONVERSATION_END:
            if conversation is None:
                raise ValueError(f"会話終了区切りだけが現れました: {path}")
            conversation.append(line)
            units.append("\n".join(conversation))
            conversation = None
        elif conversation is not None:
            conversation.append(line)
        elif line.strip():
            units.append(line)
    if conversation is not None:
        raise ValueError(f"会話開始区切りに対応する終了区切りがありません: {path}")
    return units


def _source_spec(spec: SourceSpec) -> tuple[str, str | Path, float]:
    if len(spec) == 2:
        name, path = spec
        return name, path, 1.0
    name, path, weight = spec
    return name, path, float(weight)


def _read_source(spec: SourceSpec) -> _Source:
    name, raw_path, weight = _source_spec(spec)
    if not name:
        raise ValueError("source名は空にできません")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"weightは正の有限値で指定してください: {name}={weight}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"入力テキストが見つかりません: {path}")
    input_bytes = path.read_bytes()
    try:
        text = input_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"UTF-8として入力テキストを読めません: {path}") from error
    units = _read_units(text, path)
    return _Source(
        name=name,
        path=path,
        weight=weight,
        text=text,
        input_bytes=input_bytes,
        units=units,
        unique_units=[],
    )


def _deduplicate(sources: list[_Source]) -> None:
    seen: set[str] = set()
    for source in sources:
        for unit in source.units:
            if unit in seen:
                source.duplicate_units_removed += 1
            else:
                seen.add(unit)
                source.unique_units.append(unit)


def _choose_units(
    sources: list[_Source], *, target_units: int | None, seed: int
) -> list[tuple[str, str]]:
    available = sum(len(source.unique_units) for source in sources)
    if target_units is None:
        target = available
    else:
        if isinstance(target_units, bool) or target_units < 0:
            raise ValueError("target_unitsは0以上の整数で指定してください")
        target = int(target_units)
        if target > available:
            raise ValueError(
                f"target_unitsがunique単位数を超えています: {target} > {available}"
            )

    shuffled: dict[str, list[str]] = {}
    randomiser = random.Random(seed)
    for source in sources:
        shuffled[source.name] = list(source.unique_units)
        randomiser.shuffle(shuffled[source.name])

    positions = {source.name: 0 for source in sources}
    current = {source.name: 0.0 for source in sources}
    selected: list[tuple[str, str]] = []
    while len(selected) < target:
        active = [
            source
            for source in sources
            if positions[source.name] < len(shuffled[source.name])
        ]
        if not active:
            break
        total_weight = sum(source.weight for source in active)
        for source in active:
            current[source.name] += source.weight
        index, source = max(
            enumerate(active),
            key=lambda pair: (current[pair[1].name], -pair[0]),
        )
        del index
        current[source.name] -= total_weight
        unit = shuffled[source.name][positions[source.name]]
        positions[source.name] += 1
        selected.append((source.name, unit))
    return selected


def mix_corpora(
    sources: Iterable[SourceSpec],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 42,
    target_units: int | None = None,
) -> dict[str, Any]:
    """sourceを重複なく混ぜ、本文と再現用manifestを書き出す。"""

    specs = list(sources)
    if not specs:
        raise ValueError("sourceを一つ以上指定してください")
    source_data = [_read_source(spec) for spec in specs]
    names = [source.name for source in source_data]
    if len(names) != len(set(names)):
        raise ValueError("source名が重複しています")
    _deduplicate(source_data)

    destination = Path(output_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    source_paths = {source.path for source in source_data}
    if destination in source_paths or manifest_file in source_paths:
        raise ValueError("入力sourceをoutputまたはmanifestで上書きできません")
    if destination == manifest_file:
        raise ValueError("outputとmanifestには別のパスを指定してください")

    selected = _choose_units(source_data, target_units=target_units, seed=seed)
    output_text = "\n".join(unit for _, unit in selected)
    if output_text:
        output_text += "\n"
    output_bytes = output_text.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output_bytes)

    selected_counts = {source.name: 0 for source in source_data}
    selected_characters = {source.name: 0 for source in source_data}
    for name, unit in selected:
        selected_counts[name] += 1
        selected_characters[name] += len(unit)
    total_weight = sum(source.weight for source in source_data)
    total_selected_chars = sum(selected_characters.values())

    def share(value: float, total: float) -> float:
        return value / total if total else 0.0

    manifest: dict[str, Any] = {
        "format": "corpus-mix-v1",
        "seed": seed,
        "target_units": target_units,
        "algorithm": "sourceごとにshuffle後、available source間のsmooth weighted round-robinでquotaまで選択。枯渇sourceはactiveから外して残余を再配分する。",
        "weight_semantics": "weightはtarget_unitsに対する希望比率であり、単位の複製には使わない。",
        "unit_rule": "通常の非空行を一単位とし、会話startからendまでを改行込みの一単位とする。",
        "duplicate_rule": "本文完全一致をsource指定順で一度だけ採用する。",
        "requested_weight_share": {
            source.name: share(source.weight, total_weight) for source in source_data
        },
        "input_unit_count": sum(len(source.units) for source in source_data),
        "unique_unit_count": sum(len(source.unique_units) for source in source_data),
        "output_unit_count": len(selected),
        "output_lines": len(output_text.splitlines()),
        "output_character_count": len(output_text),
        "output_sha256": _sha256(output_bytes),
        "output_path": str(destination),
        "actual_adoption_share": {
            source.name: share(selected_counts[source.name], len(selected))
            for source in source_data
        },
        "actual_adoption_character_share": {
            source.name: share(selected_characters[source.name], total_selected_chars)
            for source in source_data
        },
        "sources": [],
    }
    for source in source_data:
        manifest["sources"].append(
            {
                "name": source.name,
                "input_path": str(source.path),
                "input_sha256": _sha256(source.input_bytes),
                "input_lines": len(source.text.splitlines()),
                "input_characters": len(source.text),
                "input_unit_count": len(source.units),
                "unique_unit_count": len(source.unique_units),
                "duplicate_units_removed": source.duplicate_units_removed,
                "weight": source.weight,
                "requested_weight_share": share(source.weight, total_weight),
                "adopted_unit_count": selected_counts[source.name],
                "adopted_characters": selected_characters[source.name],
                "actual_adoption_share": share(
                    selected_counts[source.name], len(selected)
                ),
            }
        )
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
