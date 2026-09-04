"""複数のUTF-8コーパスを重複なく決定的に混合する。"""

from __future__ import annotations

import argparse
import json
import math

from _common import repo_path

from my_little_japanese_llm.mixing import mix_corpora


def _parse_source(value: str) -> tuple[str, str]:
    """NAME=PATH形式のsource指定を解釈する。"""

    name, separator, path = value.partition("=")
    name = name.strip()
    path = path.strip()
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("sourceはNAME=PATH形式で指定してください")
    return name, path


def _parse_weight(value: str) -> tuple[str, float]:
    """NAME=FLOAT形式のweight指定を解釈する。"""

    name, separator, raw_weight = value.partition("=")
    name = name.strip()
    raw_weight = raw_weight.strip()
    if not separator or not name or not raw_weight:
        raise argparse.ArgumentTypeError("weightはNAME=FLOAT形式で指定してください")
    try:
        weight = float(raw_weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"weightは数値で指定してください: {value}"
        ) from error
    if not math.isfinite(weight) or weight <= 0:
        raise argparse.ArgumentTypeError("weightは正の有限値で指定してください")
    return name, weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_source,
        metavar="NAME=PATH",
        help="UTF-8入力テキスト。複数指定可",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        type=_parse_weight,
        metavar="NAME=FLOAT",
        help="source間の選択優先度。省略時は全sourceが1.0",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-units",
        type=int,
        help="混合後に採用する論理単位数。省略時は全sourceのunique単位",
    )
    target_group.add_argument(
        "--target-tokens",
        type=int,
        help="SentencePieceで測った混合後token数の上限。単位を分割せず予算へ近づける",
    )
    parser.add_argument(
        "--tokenizer",
        help="--target-tokensで単位のtoken数を測るSentencePiece .model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="混合後UTF-8テキスト")
    parser.add_argument("--manifest", required=True, help="混合条件と集計値のJSON")
    args = parser.parse_args()

    source_names = [name for name, _ in args.source]
    if len(source_names) != len(set(source_names)):
        parser.error("source名は重複できません")
    weights: dict[str, float] = {}
    for name, weight in args.weight:
        if name in weights:
            parser.error(f"weight名が重複しています: {name}")
        weights[name] = weight
    unknown = sorted(set(weights) - set(source_names))
    if unknown:
        parser.error(f"sourceに存在しないweight名です: {', '.join(unknown)}")
    if args.target_tokens is not None and args.tokenizer is None:
        parser.error("--target-tokensには--tokenizerが必要です")
    if args.target_tokens is None and args.tokenizer is not None:
        parser.error("--tokenizerは--target-tokensと一緒に指定してください")
    sources = [
        (name, repo_path(path), weights.get(name, 1.0)) for name, path in args.source
    ]
    manifest = mix_corpora(
        sources,
        repo_path(args.output),
        repo_path(args.manifest),
        seed=args.seed,
        target_units=args.target_units,
        tokenizer_path=repo_path(args.tokenizer) if args.tokenizer else None,
        target_tokens=args.target_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
