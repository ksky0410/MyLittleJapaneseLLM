"""SentencePieceの学習とテキストのToken化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_sentencepiece() -> Any:
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError(
            "SentencePieceが見つかりません。.venv/bin/python -m pip install -e '.[dev]' を実行してください。"
        ) from error
    return spm


def _required_vocab_size(input_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
    unique_characters = len(set(text.replace("\n", "")))
    # SentencePieceの特殊tokenと、未知文字用の余裕を含める。
    return unique_characters + 8


def train_sentencepiece(
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int,
    model_type: str = "unigram",
) -> tuple[Path, Path, int]:
    """SentencePieceを学習し、model/vocabのパスと実効語彙目標を返す。

    小さなコーパスでは、要求語彙数が文字種類数より小さいことがある。
    その場合はSentencePieceが失敗しないよう、必要な下限まで自動調整する。
    ``hard_vocab_limit=False`` により、要求値より実際の語彙が少ない場合も許容する。
    """

    if vocab_size < 16:
        raise ValueError("vocab_size は16以上で指定してください")
    if model_type not in {"unigram", "bpe"}:
        raise ValueError("model_type は unigram または bpe で指定してください")
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Tokenizer入力が見つかりません: {source}")
    spm = require_sentencepiece()
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    effective_vocab_size = max(vocab_size, _required_vocab_size(source))
    if effective_vocab_size != vocab_size:
        print(
            f"警告: 小コーパスの文字種類が多いため vocab_size を "
            f"{vocab_size} から {effective_vocab_size} に調整します。"
        )
    spm.SentencePieceTrainer.train(
        input=str(source),
        model_prefix=str(prefix),
        vocab_size=effective_vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        hard_vocab_limit=False,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_id=0,
    )
    return (
        prefix.with_suffix(".model"),
        prefix.with_suffix(".vocab"),
        effective_vocab_size,
    )


def load_processor(model_path: str | Path) -> Any:
    spm = require_sentencepiece()
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Tokenizerモデルが見つかりません: {model}")
    return spm.SentencePieceProcessor(model_file=str(model))


def encode_text_file(model_path: str | Path, input_path: str | Path) -> list[int]:
    """1ファイルを文ごとにencodeし、文境界ごとにEOSを挿入する。"""

    processor = load_processor(model_path)
    text = Path(input_path).read_text(encoding="utf-8")
    token_ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        token_ids.extend(processor.encode(line, out_type=int))
        token_ids.append(int(processor.eos_id()))
    if len(token_ids) < 3:
        raise ValueError(f"Tokenが少なすぎます: {input_path}")
    return token_ids
