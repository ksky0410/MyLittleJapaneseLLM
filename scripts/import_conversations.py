"""日本語会話コーパスを会話単位で分割し、学習用JSONL/TXTへ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import repo_path

DEFAULT_LICENSE = "CC BY-SA 4.0"
DEFAULT_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.ja"
DATASET_FORMAT = "one JSON file per conversation with an utterances array"


@dataclass(frozen=True)
class SourceSpec:
    """一つの入力リポジトリと、その利用条件を表す。"""

    path: Path
    name: str
    url: str | None
    commit_sha: str | None
    license: str
    license_url: str


def sha256_bytes(value: bytes) -> str:
    """bytesのSHA-256を返す。"""

    return hashlib.sha256(value).hexdigest()


def _normalise_text(value: Any) -> str:
    """発話本文だけを学習用の一行へ正規化する。"""

    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(value.split())


def _git_value(path: Path, *arguments: str) -> str | None:
    """Gitの情報を読み取り専用で取得し、取得できなければNoneを返す。"""

    try:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_root(path: Path) -> Path | None:
    """入力ディレクトリ自身または親にあるGit作業木を探す。"""

    current = path.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _find_dialogue_files(root: Path) -> list[Path]:
    """リポジトリ内のdialogues/*.jsonを決定的な順序で探す。"""

    if not root.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {root}")

    dialogue_dirs: list[Path]
    if root.name == "dialogues":
        dialogue_dirs = [root]
    else:
        dialogue_dirs = sorted(
            path
            for path in root.rglob("dialogues")
            if path.is_dir() and ".git" not in path.parts
        )
    files = sorted(
        file
        for directory in dialogue_dirs
        for file in directory.rglob("*.json")
        if file.is_file() and ".git" not in file.parts
    )
    if not files:
        raise ValueError(f"dialogues/*.jsonが見つかりません: {root}")
    return files


def _source_name(path: Path) -> str:
    """入力パスから安定した表示名を作る。"""

    return path.name.replace("_", "-") or "conversation-source"


def _source_file_hashes(
    root: Path, files: Iterable[Path]
) -> tuple[list[dict[str, Any]], str]:
    """入力JSONそれぞれのSHA-256と、全ファイルを束ねたSHA-256を返す。"""

    records: list[dict[str, Any]] = []
    digest_lines: list[str] = []
    for file in files:
        relative = file.relative_to(root).as_posix()
        data = file.read_bytes()
        digest = sha256_bytes(data)
        records.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": digest,
            }
        )
        digest_lines.append(f"{relative}\t{digest}\n")
    return records, sha256_bytes("".join(digest_lines).encode("utf-8"))


def _ordered_utterances(raw: Any) -> list[dict[str, Any]]:
    """発話IDがある場合はそれを使い、発話の順番を固定する。"""

    if not isinstance(raw, list):
        return []
    indexed = list(enumerate(raw))
    if all(
        isinstance(item, dict) and isinstance(item.get("utterance_id"), (int, float))
        for _, item in indexed
    ):
        indexed.sort(key=lambda pair: (pair[1]["utterance_id"], pair[0]))
    return [item for _, item in indexed if isinstance(item, dict)]


def _read_conversations(
    spec: SourceSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一つのリポジトリから本文と入力ハッシュ情報を抽出する。"""

    root = spec.path.resolve()
    files = _find_dialogue_files(root)
    file_hashes, tree_sha256 = _source_file_hashes(root, files)
    conversations: list[dict[str, Any]] = []
    utterance_count = 0
    character_count = 0

    for file in files:
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"会話JSONを読めません: {file}") from error
        if not isinstance(raw, dict):
            raise TypeError(f"会話JSONのトップレベルがobjectではありません: {file}")

        dialogue_id = raw.get("dialogue_id")
        if dialogue_id is None:
            raise ValueError(f"dialogue_idがありません: {file}")
        turns: list[dict[str, Any]] = []
        for turn_index, utterance in enumerate(
            _ordered_utterances(raw.get("utterances"))
        ):
            speaker = utterance.get("interlocutor_id", f"speaker-{turn_index}")
            speaker_id = _normalise_text(speaker)
            text = _normalise_text(utterance.get("text"))
            if not text:
                continue
            turns.append(
                {
                    "turn_index": len(turns),
                    "speaker_id": speaker_id or f"speaker-{turn_index}",
                    "text": text,
                }
            )
        if not turns:
            continue

        source_dialogue_id = str(dialogue_id)
        source_file = file.relative_to(root).as_posix()
        # 実データには異なるJSONファイルで同じdialogue_idを持つ例がある。
        # 元IDは保持しつつ、会話識別子はファイルパスで一意にする。
        conversation_id = f"{spec.name}:{source_file}"
        conversations.append(
            {
                "conversation_id": conversation_id,
                "dataset": spec.name,
                "source_dialogue_id": source_dialogue_id,
                "source_file": source_file,
                "turns": turns,
            }
        )
        utterance_count += len(turns)
        character_count += sum(len(turn["text"]) for turn in turns)

    documentation: dict[str, str] = {}
    if spec.url:
        repository_url = spec.url.removesuffix("/").removesuffix(".git")
        revision = spec.commit_sha or "main"
        documentation = {
            "readme": f"{repository_url}/blob/{revision}/README.md",
            "license": f"{repository_url}/blob/{revision}/LICENSE",
        }
    provenance_files = {
        filename: {
            "bytes": len((root / filename).read_bytes()),
            "sha256": sha256_bytes((root / filename).read_bytes()),
        }
        for filename in ("README.md", "LICENSE")
        if (root / filename).is_file()
    }
    source_manifest = {
        "name": spec.name,
        "repository_url": spec.url,
        "commit_sha": spec.commit_sha,
        "license": spec.license,
        "license_url": spec.license_url,
        "provenance_documents": documentation,
        "provenance_files": provenance_files,
        "input_path": str(root),
        "format": DATASET_FORMAT,
        "source_tree_sha256": tree_sha256,
        "input_files": file_hashes,
        "input_file_count": len(file_hashes),
        "conversation_count": len(conversations),
        "utterance_count": utterance_count,
        "character_count": character_count,
    }
    return conversations, source_manifest


def _split_conversations(
    conversations: list[dict[str, Any]],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """会話を一度だけシャッフルし、会話単位で分割する。"""

    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratioとtest_ratioの合計は0以上1未満にしてください")
    shuffled = list(conversations)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    validation_count = int(total * validation_ratio)
    test_count = int(total * test_ratio)
    if validation_ratio > 0 and validation_count == 0 and total >= 3:
        validation_count = 1
    if test_ratio > 0 and test_count == 0 and total >= 3:
        test_count = 1

    while validation_count + test_count >= total and total:
        if test_count >= validation_count and test_count:
            test_count -= 1
        elif validation_count:
            validation_count -= 1
        else:
            break
    test_start = total - test_count
    validation_start = test_start - validation_count
    return {
        "train": shuffled[:validation_start],
        "validation": shuffled[validation_start:test_start],
        "test": shuffled[test_start:],
    }


def _text_record(conversation: dict[str, Any]) -> str:
    """会話JSONの本文部分を、話者境界を保った構造化TXTへ変換する。"""

    lines = ["<|startofconversation|>"]
    lines.extend(
        f"<|speaker:{turn['speaker_id']}|>{turn['text']}"
        for turn in conversation["turns"]
    )
    lines.append("<|endofconversation|>")
    return "\n".join(lines)


def _write_split(
    output_dir: Path, split: str, conversations: list[dict[str, Any]]
) -> dict[str, Any]:
    """一つのsplitをJSONL/TXTへ保存し、集計値を返す。"""

    jsonl_path = output_dir / f"{split}.jsonl"
    text_path = output_dir / f"{split}.txt"
    jsonl_lines: list[str] = []
    text_blocks: list[str] = []
    turn_count = 0
    character_count = 0
    for conversation in conversations:
        record = dict(conversation)
        record["split"] = split
        jsonl_lines.append(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )
        text_blocks.append(_text_record(conversation))
        turn_count += len(conversation["turns"])
        character_count += sum(len(turn["text"]) for turn in conversation["turns"])

    jsonl_bytes = ("\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")).encode(
        "utf-8"
    )
    text_bytes = ("\n\n".join(text_blocks) + ("\n" if text_blocks else "")).encode(
        "utf-8"
    )
    jsonl_path.write_bytes(jsonl_bytes)
    text_path.write_bytes(text_bytes)
    return {
        "conversation_count": len(conversations),
        "turn_count": turn_count,
        "character_count": character_count,
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256_bytes(jsonl_bytes),
        "txt": str(text_path),
        "txt_sha256": sha256_bytes(text_bytes),
    }


def _metadata_for_source(
    path: Path,
    name: str | None,
    url: str | None,
    commit: str | None,
    license_name: str,
    license_url: str,
) -> SourceSpec:
    """CLI引数とGitの情報からSourceSpecを作る。"""

    root = path.resolve()
    git_root = _git_root(root)
    git_path = git_root or root
    inferred_url = _git_value(git_path, "config", "--get", "remote.origin.url")
    inferred_commit = _git_value(git_path, "rev-parse", "HEAD")
    return SourceSpec(
        root,
        name or _source_name(root),
        url or inferred_url,
        commit or inferred_commit,
        license_name,
        license_url,
    )


def import_conversations(
    input_paths: Iterable[str | Path],
    output_dir: str | Path = "artifacts/corpus/conversation-v1",
    *,
    source_names: Iterable[str] | None = None,
    source_urls: Iterable[str] | None = None,
    source_commits: Iterable[str] | None = None,
    license_name: str = DEFAULT_LICENSE,
    license_url: str = DEFAULT_LICENSE_URL,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """会話リポジトリを混ぜ、会話単位のsplitとmanifestを書き出す。"""

    paths = [Path(path).expanduser() for path in input_paths]
    if not paths:
        raise ValueError("input_pathsが空です")
    names = list(source_names or [])
    urls = list(source_urls or [])
    commits = list(source_commits or [])
    for values, label in (
        (names, "source_names"),
        (urls, "source_urls"),
        (commits, "source_commits"),
    ):
        if values and len(values) != len(paths):
            raise ValueError(f"{label}はinput_pathsと同じ件数で指定してください")

    specs = [
        _metadata_for_source(
            path,
            names[index] if names else None,
            urls[index] if urls else None,
            commits[index] if commits else None,
            license_name,
            license_url,
        )
        for index, path in enumerate(paths)
    ]
    conversations: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for spec in specs:
        source_conversations, source_manifest = _read_conversations(spec)
        conversations.extend(source_conversations)
        source_manifests.append(source_manifest)

    if not conversations:
        raise ValueError("本文のある会話が見つかりません")
    conversation_ids = [
        conversation["conversation_id"] for conversation in conversations
    ]
    if len(set(conversation_ids)) != len(conversation_ids):
        raise ValueError(
            "conversation_idが重複しています。source_namesを分けてください"
        )

    splits = _split_conversations(conversations, validation_ratio, test_ratio, seed)
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    split_manifests = {
        split: _write_split(output_root, split, split_conversations)
        for split, split_conversations in splits.items()
    }
    source_split_counts: dict[str, dict[str, dict[str, int]]] = {
        spec.name: {
            split: {"conversation_count": 0, "turn_count": 0, "character_count": 0}
            for split in splits
        }
        for spec in specs
    }
    for split, split_conversations in splits.items():
        for conversation in split_conversations:
            counts = source_split_counts[conversation["dataset"]][split]
            counts["conversation_count"] += 1
            counts["turn_count"] += len(conversation["turns"])
            counts["character_count"] += sum(
                len(turn["text"]) for turn in conversation["turns"]
            )
    for source_manifest in source_manifests:
        source_manifest["splits"] = source_split_counts[source_manifest["name"]]
    assigned_ids = [
        conversation["conversation_id"]
        for split_conversations in splits.values()
        for conversation in split_conversations
    ]
    if set(assigned_ids) != set(conversation_ids) or len(assigned_ids) != len(
        set(assigned_ids)
    ):
        raise AssertionError("会話単位のsplitに失敗しました")

    manifest: dict[str, Any] = {
        "format": "conversation-import-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "license": license_name,
        "license_url": license_url,
        "usage_notes": [
            "公開元のREADMEとLICENSEに従って利用すること",
            "会話データから個人を特定しようとしないこと",
            "特定の話者へのなりすましに用いないこと",
            "話者の権利と、公開元が示す倫理上の注意を尊重すること",
        ],
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "input_count": len(specs),
        "conversation_count": len(conversations),
        "turn_count": sum(len(conversation["turns"]) for conversation in conversations),
        "character_count": sum(
            len(turn["text"])
            for conversation in conversations
            for turn in conversation["turns"]
        ),
        "sources": source_manifests,
        "splits": split_manifests,
        "output_dir": str(output_root.resolve()),
        "text_structure": "<|startofconversation|> and <|endofconversation|> delimit conversations; <|speaker:ID|> preserves turn speakers",
        "metadata_policy": "Only dialogue_id, source_file, speaker_id, turn order, and utterance text are exported; evaluations, timestamps, persona fields, mention metadata, and URL/image fields from source metadata are not copied. URLs appearing literally inside utterance text remain part of that text.",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="/tmpなどに取得した会話リポジトリのルート。複数指定可",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/corpus/conversation-v1",
        help="train/validation/test JSONL/TXTとmanifestの出力先",
    )
    parser.add_argument(
        "--source-name", action="append", help="inputごとのデータセット名"
    )
    parser.add_argument(
        "--source-url", action="append", help="inputごとの取得元リポジトリURL"
    )
    parser.add_argument(
        "--source-commit", action="append", help="inputごとのコミットSHA"
    )
    parser.add_argument(
        "--license", default=DEFAULT_LICENSE, help="データセットのライセンス名"
    )
    parser.add_argument("--license-url", default=DEFAULT_LICENSE_URL)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = import_conversations(
        args.input,
        repo_path(args.output_dir),
        source_names=args.source_name,
        source_urls=args.source_url,
        source_commits=args.source_commit,
        license_name=args.license,
        license_url=args.license_url,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"会話数: {manifest['conversation_count']}")
    print(f"発話数: {manifest['turn_count']}")
    for split, values in manifest["splits"].items():
        print(
            f"{split}: {values['conversation_count']}会話, {values['turn_count']}発話"
        )
    print(f"manifest: {repo_path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
