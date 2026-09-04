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


def _unit_token_cost(processor: Any, unit: str) -> int:
    """現在のテキストToken化規則で、論理単位のtoken数を数える。"""

    token_count = 0
    for line in unit.splitlines():
        line = line.strip()
        if not line:
            continue
        token_count += len(processor.encode(line, out_type=int)) + 1
    return token_count


def _choose_units_by_token_budget(
    sources: list[_Source],
    *,
    target_tokens: int,
    seed: int,
    token_costs: dict[str, list[int]],
) -> list[tuple[str, str, int]]:
    """重み付き公平キューで、予算を超えない論理単位を選ぶ。

    各sourceの単位を先に同じ乱数系列でshuffleし、単位のtoken costを含む
    weighted fair queueで次のsourceを決める。単位は分割・複製せず、残り予算に
    収まる候補だけを選ぶため、出力token数はtarget_tokens以下になる。
    """

    if target_tokens == 0:
        return []

    shuffled: dict[str, list[tuple[str, int]]] = {}
    randomiser = random.Random(seed)
    for source in sources:
        costs = token_costs[source.name]
        if len(costs) != len(source.unique_units):
            raise ValueError(f"token costの単位数が一致しません: {source.name}")
        pairs = list(zip(source.unique_units, costs, strict=True))
        randomiser.shuffle(pairs)
        shuffled[source.name] = pairs

    # next_finishはsourceごとの仮想的な送信完了時刻。cost/weightで進めることで、
    # 単位の長さが異なってもweightをtoken比率として扱える。
    next_finish = {source.name: 0.0 for source in sources}
    selected: list[tuple[str, str, int]] = []
    remaining = target_tokens
    source_order = {source.name: index for index, source in enumerate(sources)}

    while remaining > 0:
        candidates: list[tuple[float, int, _Source, int, int]] = []
        for source in sources:
            fitting_index = next(
                (
                    index
                    for index, (_, cost) in enumerate(shuffled[source.name])
                    if cost <= remaining
                ),
                None,
            )
            if fitting_index is None:
                continue
            _, cost = shuffled[source.name][fitting_index]
            finish = next_finish[source.name] + (cost / source.weight)
            candidates.append(
                (finish, source_order[source.name], source, fitting_index, cost)
            )
        if not candidates:
            break

        _, _, source, fitting_index, cost = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        unit, _ = shuffled[source.name].pop(fitting_index)
        finish = next_finish[source.name] + (cost / source.weight)
        next_finish[source.name] = finish
        remaining -= cost
        selected.append((source.name, unit, cost))
    return selected


def mix_corpora(
    sources: Iterable[SourceSpec],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 42,
    target_units: int | None = None,
    tokenizer_path: str | Path | None = None,
    target_tokens: int | None = None,
) -> dict[str, Any]:
    """sourceを重複なく混ぜ、本文と再現用manifestを書き出す。

    ``target_units``を指定した場合は従来どおり単位数で選ぶ。``target_tokens``を
    指定した場合は``tokenizer_path``のSentencePieceで各単位を測り、指定予算を
    超えない範囲でtoken数が近くなるように選ぶ。両方の指定はできない。
    """

    if target_units is not None and target_tokens is not None:
        raise ValueError("target_unitsとtarget_tokensは同時に指定できません")
    if target_tokens is not None:
        if isinstance(target_tokens, bool) or not isinstance(target_tokens, int):
            raise ValueError("target_tokensは0以上の整数で指定してください")
        if target_tokens < 0:
            raise ValueError("target_tokensは0以上の整数で指定してください")
        if tokenizer_path is None:
            raise ValueError("target_tokensにはtokenizer_pathが必要です")
    elif tokenizer_path is not None:
        raise ValueError("tokenizer_pathはtarget_tokensと一緒に指定してください")

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

    tokenizer_file: Path | None = None
    tokenizer_sha256: str | None = None
    token_costs: dict[str, list[int]] | None = None
    if target_tokens is not None:
        from .tokenizer import load_processor

        tokenizer_file = Path(tokenizer_path).expanduser().resolve()
        if not tokenizer_file.is_file():
            raise FileNotFoundError(
                f"Tokenizerモデルが見つかりません: {tokenizer_file}"
            )
        tokenizer_bytes = tokenizer_file.read_bytes()
        tokenizer_sha256 = _sha256(tokenizer_bytes)
        processor = load_processor(tokenizer_file)
        token_costs = {
            source.name: [
                _unit_token_cost(processor, unit) for unit in source.unique_units
            ]
            for source in source_data
        }

    if target_tokens is None:
        selected_by_source = _choose_units(
            source_data, target_units=target_units, seed=seed
        )
        selected = [(name, unit, None) for name, unit in selected_by_source]
    else:
        assert token_costs is not None
        selected = _choose_units_by_token_budget(
            source_data,
            target_tokens=target_tokens,
            seed=seed,
            token_costs=token_costs,
        )
    output_text = "\n".join(unit for _, unit, _ in selected)
    if output_text:
        output_text += "\n"
    output_bytes = output_text.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output_bytes)

    selected_counts = {source.name: 0 for source in source_data}
    selected_characters = {source.name: 0 for source in source_data}
    selected_tokens = {source.name: 0 for source in source_data}
    for name, unit, token_count in selected:
        selected_counts[name] += 1
        selected_characters[name] += len(unit)
        if token_count is not None:
            selected_tokens[name] += token_count
    total_weight = sum(source.weight for source in source_data)
    total_selected_chars = sum(selected_characters.values())
    total_selected_tokens = (
        sum(selected_tokens.values()) if target_tokens is not None else None
    )

    def share(value: float, total: float) -> float:
        return value / total if total else 0.0

    manifest: dict[str, Any] = {
        "format": "corpus-mix-v1",
        "seed": seed,
        "target_units": target_units,
        "target_tokens": target_tokens,
        "tokenizer_path": str(tokenizer_file) if tokenizer_file is not None else None,
        "tokenizer_sha256": tokenizer_sha256,
        "algorithm": (
            "sourceごとにshuffle後、available source間のsmooth weighted round-robinで"
            "quotaまで選択。枯渇sourceはactiveから外して残余を再配分する。"
            if target_tokens is None
            else "sourceごとにshuffle後、unit token cost / weightを仮想完了時刻へ加算する"
            "weighted fair queueで、target_tokensを超えない候補を決定的に選択。"
        ),
        "weight_semantics": (
            "weightは採用単位数またはtoken数に対する希望比率であり、"
            "単位の複製には使わない。"
        ),
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
        "selected_token_count": total_selected_tokens,
        "actual_adoption_share": {
            source.name: share(selected_counts[source.name], len(selected))
            for source in source_data
        },
        "actual_adoption_character_share": {
            source.name: share(selected_characters[source.name], total_selected_chars)
            for source in source_data
        },
        "actual_adoption_token_share": (
            {
                source.name: share(selected_tokens[source.name], total_selected_tokens)
                for source in source_data
            }
            if total_selected_tokens is not None
            else None
        ),
        "actual_token_share": (
            {
                source.name: share(selected_tokens[source.name], total_selected_tokens)
                for source in source_data
            }
            if total_selected_tokens is not None
            else None
        ),
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
                "adopted_token_count": (
                    selected_tokens[source.name] if target_tokens is not None else None
                ),
                "actual_adoption_share": share(
                    selected_counts[source.name], len(selected)
                ),
                "actual_token_share": (
                    share(selected_tokens[source.name], total_selected_tokens)
                    if total_selected_tokens is not None
                    else None
                ),
            }
        )
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
