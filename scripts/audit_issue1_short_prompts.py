"""Issue #1の短い口語promptとheld-out会話の分布を監査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from _common import repo_path
from analyze_response_functions import classify_response_function
from evaluate_chat_dataset import _read_records, _turns
from my_little_japanese_llm.tokenizer import load_processor


PROMPT_NAMES = (
    "まじで",
    "それな",
    "今日なにしてた？",
    "やば",
    "なんかさ",
    "いやそれは",
    "おつかれ",
    "明日ひま？",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _increment(counter: Counter[str], key: str) -> None:
    counter[key] += 1


def _nested_counts(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for item in items:
        cursor = root
        for key in keys[:-1]:
            value = str(item.get(key, "unknown"))
            cursor = cursor.setdefault(value, {})
        leaf = str(item.get(keys[-1], "unknown"))
        cursor[leaf] = int(cursor.get(leaf, 0)) + 1
    return root


def audit_evaluation(
    evaluation: dict[str, Any], selection: dict[str, Any] | None,
) -> dict[str, Any]:
    """最終held-out評価をsource・長さ・重複・履歴切り詰めで集計する。"""

    selection_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    if selection is not None:
        for item in selection.get("examples", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("conversation_id")),
                int(item.get("record_index", -1)),
                int(item.get("target_index", -1)),
            )
            selection_by_key[key] = item

    rows: list[dict[str, Any]] = []
    for item in evaluation.get("results", []):
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("conversation_id")),
            int(item.get("record_index", -1)),
            int(item.get("target_index", -1)),
        )
        selection_item = selection_by_key.get(key, {})
        row = {
            "conversation_id": item.get("conversation_id"),
            "record_index": item.get("record_index"),
            "target_index": item.get("target_index"),
            "source": item.get("source") or "unknown",
            "stratum": item.get("stratum") or "unstratified",
            "history_truncated": bool(item.get("history_truncated", False)),
            "train_text_overlap": bool(item.get("train_text_overlap", False)),
            "history_token_count": selection_item.get("history_token_count"),
            "prompt_token_count": item.get("prompt_token_count"),
            "reference_token_count": item.get("reference_token_count"),
            "generated_token_count": item.get("generated_token_count"),
            "token_overlap_f1": item.get("token_overlap_f1"),
            "eos_reached": item.get("eos_reached"),
            "reference": item.get("reference"),
        }
        rows.append(row)

    grouped: dict[str, Any] = {}
    for row in rows:
        group = grouped.setdefault(
            str(row["source"]),
            {"count": 0, "mean_f1": 0.0, "mean_reference_tokens": 0.0,
             "mean_generated_tokens": 0.0, "rows": []},
        )
        group["count"] += 1
        group["mean_f1"] += float(row["token_overlap_f1"] or 0.0)
        group["mean_reference_tokens"] += float(row["reference_token_count"] or 0.0)
        group["mean_generated_tokens"] += float(row["generated_token_count"] or 0.0)
        group["rows"].append(row)
    for group in grouped.values():
        count = max(1, int(group["count"]))
        for key in ("mean_f1", "mean_reference_tokens", "mean_generated_tokens"):
            group[key] = float(group[key]) / count

    distributions = {
        "source": _nested_counts(rows, ("source",)),
        "stratum": _nested_counts(rows, ("stratum",)),
        "source_stratum": _nested_counts(rows, ("source", "stratum")),
        "source_history_truncated": _nested_counts(
            rows, ("source", "history_truncated")
        ),
        "source_train_text_overlap": _nested_counts(
            rows, ("source", "train_text_overlap")
        ),
        "stratum_history_truncated": _nested_counts(
            rows, ("stratum", "history_truncated")
        ),
    }
    return {
        "count": len(rows),
        "distributions": distributions,
        "groups": grouped,
        "rows": rows,
    }


def scan_prompt_matches(
    records: list[dict[str, Any]],
    source: str,
    split: str,
    prompts: tuple[str, ...],
    token_count: Callable[[str], int],
    selected_keys: set[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    """会話turnの完全一致・部分一致と、直後応答を収集する。"""

    matches: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        turns = _turns(record)
        conversation_id = str(record.get("conversation_id", f"record-{record_index}"))
        for turn_index, turn in enumerate(turns):
            text = turn["text"]
            for prompt in prompts:
                if text == prompt:
                    match_type = "exact"
                elif prompt in text:
                    match_type = "substring"
                else:
                    continue
                has_response = turn_index + 1 < len(turns)
                response = turns[turn_index + 1] if has_response else None
                response_text = response["text"] if response else None
                response_tokens = token_count(response_text) if response_text else None
                category = (
                    classify_response_function(text, response_text or "", response_tokens or 0)
                    if response_text
                    else None
                )
                matches.append(
                    {
                        "prompt": prompt,
                        "match_type": match_type,
                        "source": source,
                        "split": split,
                        "record_index": record_index,
                        "conversation_id": conversation_id,
                        "source_dialogue_id": record.get("source_dialogue_id"),
                        "source_file": record.get("source_file"),
                        "turn_index": turn_index,
                        "input": text,
                        "input_token_count": token_count(text),
                        "response": response_text,
                        "response_token_count": response_tokens,
                        "response_function": category,
                        "response_speaker": response.get("speaker_id") if response else None,
                        "sft_selected": (source, record_index, turn_index + 1)
                        in selected_keys,
                    }
                )
    return matches


def summarize_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for prompt in PROMPT_NAMES:
        prompt_matches = [item for item in matches if item["prompt"] == prompt]
        summary[prompt] = {
            "total": len(prompt_matches),
            "exact": sum(item["match_type"] == "exact" for item in prompt_matches),
            "substring": sum(item["match_type"] == "substring" for item in prompt_matches),
            "with_response": sum(item["response"] is not None for item in prompt_matches),
            "sft_selected": sum(item["sft_selected"] for item in prompt_matches),
            "splits": dict(Counter(str(item["split"]) for item in prompt_matches)),
            "sources": dict(Counter(str(item["source"]) for item in prompt_matches)),
            "response_functions": dict(
                Counter(
                    str(item["response_function"])
                    for item in prompt_matches
                    if item["response_function"] is not None
                )
            ),
        }
    return summary


def _load_selected_keys(path: Path) -> tuple[set[tuple[str, int, int]], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    keys: set[tuple[str, int, int]] = set()
    for item in manifest.get("selected_provenance", []):
        if isinstance(item, dict):
            keys.add((str(item["source"]), int(item["record_index"]), int(item["target_index"])))
    return keys, sha256_file(path)


def _format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Issue #1短い口語prompt監査",
        "",
        f"- 評価例数：{result['evaluation']['count']}",
        f"- checkpoint：{result['checkpoint']}",
        f"- 実験コミット：{result['git_commit']}",
        "",
        "## held-out分布",
        "",
    ]
    for source, group in result["evaluation"]["groups"].items():
        lines.append(
            f"- {source}：{group['count']}例、平均F1 {group['mean_f1']:.6f}、"
            f"平均参照長 {group['mean_reference_tokens']:.2f} token、"
            f"平均生成長 {group['mean_generated_tokens']:.2f} token"
        )
    lines.extend(["", "## Issue #1 promptの出現状況", ""])
    for prompt, item in result["prompt_summary"].items():
        functions = ", ".join(
            f"{key}={value}" for key, value in item["response_functions"].items()
        ) or "なし"
        lines.append(
            f"- `{prompt}`：合計{item['total']}件（完全一致{item['exact']}、"
            f"部分一致{item['substring']}、応答あり{item['with_response']}、"
            f"SFT採用{item['sft_selected']}）。応答機能：{functions}"
        )
    lines.extend(["", "## 解釈", "", result["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selected-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-file", action="append", required=True, metavar="SOURCE=SPLIT=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()

    evaluation_path = repo_path(args.evaluation).resolve()
    selection_path = repo_path(args.selection).resolve()
    selected_manifest_path = repo_path(args.selected_manifest).resolve()
    checkpoint_path = repo_path(args.checkpoint).resolve()
    tokenizer_path = repo_path(args.tokenizer).resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_keys, selected_manifest_sha256 = _load_selected_keys(selected_manifest_path)
    processor = load_processor(tokenizer_path)

    def token_count(text: str) -> int:
        return len(processor.encode(text, out_type=int))

    all_matches: list[dict[str, Any]] = []
    source_inputs: list[dict[str, Any]] = []
    for spec in args.source_file:
        try:
            source, split, raw_path = spec.split("=", 2)
        except ValueError as error:
            raise ValueError(f"--source-fileはSOURCE=SPLIT=PATHで指定してください: {spec}") from error
        path = repo_path(raw_path).resolve()
        records = _read_records(path)
        source_inputs.append(
            {
                "source": source,
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": len(records),
            }
        )
        all_matches.extend(
            scan_prompt_matches(
                records,
                source,
                split,
                PROMPT_NAMES,
                token_count,
                selected_keys,
            )
        )

    import subprocess

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"

    result: dict[str, Any] = {
        "format": "issue1-short-prompt-audit-v1",
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_manifest_path": str(selected_manifest_path),
        "selected_manifest_sha256": selected_manifest_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "git_commit": git_commit,
        "evaluation": audit_evaluation(evaluation, selection),
        "source_inputs": source_inputs,
        "prompt_summary": summarize_matches(all_matches),
        "matches": all_matches,
        "interpretation": (
            "この監査はモデル性能を再評価するものではない。固定promptの出現数、"
            "実際の直後応答、SFT選択への採用状況、held-outの分布を分離して確認し、"
            "次の学習変更を一つに絞るために使う。"
        ),
    }
    output_path = repo_path(args.output).resolve()
    markdown_path = repo_path(args.markdown_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_format_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
