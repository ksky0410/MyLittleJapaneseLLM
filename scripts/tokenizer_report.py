"""複数のSentencePiece Tokenizerを同じ日本語テキストで比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import repo_path

from my_little_japanese_llm.tokenizer import load_processor

FIXED_SAMPLES = ("むかしむかし", "人工知能とは", "今日は良い天気です。")


def report_for_tokenizer(model_path: str | Path, input_text: str) -> dict:
    """1つのTokenizerについて、入力統計と固定サンプルを返す。"""

    processor = load_processor(model_path)
    token_ids = processor.encode(input_text, out_type=int)
    if not token_ids:
        raise ValueError("入力テキストを1 token以上用意してください")
    character_count = len(input_text.replace("\n", ""))
    return {
        "model_path": str(Path(model_path).resolve()),
        "vocab_size": int(processor.vocab_size()),
        "total_token_count": len(token_ids),
        "input_character_count": character_count,
        "average_characters_per_token": character_count / len(token_ids),
        "fixed_samples": {
            sample: processor.encode(sample, out_type=str) for sample in FIXED_SAMPLES
        },
    }


def build_report(tokenizer_paths: list[str | Path], input_path: str | Path) -> dict:
    """同一入力に対するTokenizer比較レポートを作る。"""

    if not tokenizer_paths:
        raise ValueError("Tokenizerを少なくとも1つ指定してください")
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"入力テキストが見つかりません: {source}")
    input_text = source.read_text(encoding="utf-8")
    if not input_text.strip():
        raise ValueError(f"入力テキストが空です: {source}")
    return {
        "input_path": str(source.resolve()),
        "input_character_count": len(input_text.replace("\n", "")),
        "tokenizers": [
            report_for_tokenizer(path, input_text) for path in tokenizer_paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        metavar="MODEL",
        help="比較するSentencePiece .model。複数回指定できます",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="TEXT",
        help="比較に使うUTF-8テキスト。相対パスはリポジトリルート基準",
    )
    args = parser.parse_args()
    report = build_report(
        [repo_path(path) for path in args.tokenizer],
        repo_path(args.input),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
