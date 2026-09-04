"""日本語テキストからSentencePiece UnigramまたはBPEを学習する。"""

from __future__ import annotations

import argparse

from _common import repo_path

from my_little_japanese_llm.tokenizer import (
    DEFAULT_MAX_SENTENCE_LENGTH,
    load_processor,
    train_sentencepiece,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="学習テキスト。相対パスはリポジトリルート基準"
    )
    parser.add_argument(
        "--model-prefix", required=True, help=".model/.vocabを作る接頭辞"
    )
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=DEFAULT_MAX_SENTENCE_LENGTH,
        help="SentencePieceへ渡す1文の最大長（UTF-8バイト数）",
    )
    args = parser.parse_args()

    model_path, vocab_path, effective_size = train_sentencepiece(
        repo_path(args.input),
        repo_path(args.model_prefix),
        args.vocab_size,
        args.model_type,
        args.max_sentence_length,
    )
    actual_size = int(load_processor(model_path).vocab_size())
    print(f"要求語彙数: {args.vocab_size}")
    print(f"SentencePieceに渡した語彙数: {effective_size}")
    print(f"実際の語彙数: {actual_size}")
    print(f"model: {model_path}")
    print(f"vocab: {vocab_path}")


if __name__ == "__main__":
    main()
