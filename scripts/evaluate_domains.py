"""一つのcheckpointを複数domainのToken列で個別評価する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.data import load_tokens
from my_little_japanese_llm.model import TinyJapaneseGPT, require_mlx
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.training import (
    evaluate_loss,
    load_checkpoint,
    perplexity,
    signature_from_config,
)


def _parse_domain(value: str) -> tuple[str, str]:
    """NAME=PATH形式のdomain指定を解釈する。"""

    name, separator, path = value.partition("=")
    name = name.strip()
    path = path.strip()
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("domainはNAME=PATH形式で指定してください")
    return name, path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_domains(
    config_path: str | Path,
    checkpoint_path: str | Path,
    domains: list[tuple[str, str | Path]],
    output_path: str | Path,
    *,
    eval_batches: int | None = None,
) -> dict[str, object]:
    """checkpointを一度だけ読み、domainごとのlossをJSONへ保存する。"""

    if not domains:
        raise ValueError("domainを一つ以上指定してください")
    if eval_batches is not None and eval_batches <= 0:
        raise ValueError("eval_batchesは正の整数で指定してください")
    domain_names = [name for name, _ in domains]
    if any(not name for name in domain_names):
        raise ValueError("domain名は空にできません")
    if len(domain_names) != len(set(domain_names)):
        raise ValueError("domain名が重複しています")

    config_file = repo_path(config_path)
    checkpoint_file = repo_path(checkpoint_path).resolve()
    output_file = repo_path(output_path).resolve()
    token_files = [(name, repo_path(path).resolve()) for name, path in domains]
    if output_file in {path for _, path in token_files}:
        raise ValueError("入力Tokenをoutputで上書きできません")
    for _, token_file in token_files:
        if not token_file.is_file():
            raise FileNotFoundError(f"Tokenファイルが見つかりません: {token_file}")

    require_mlx()
    config = load_config(config_file)
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    model = TinyJapaneseGPT(
        vocab_size,
        config.model.dim,
        config.model.layers,
        config.model.heads,
        config.model.context_length,
        config.model.mlp_ratio,
        config.model.position_embedding,
        config.model.norm_type,
    )
    metadata = load_checkpoint(
        model,
        checkpoint_file,
        signature_from_config(config, vocab_size),
    )
    batches = eval_batches if eval_batches is not None else config.training.eval_batches
    domain_results: list[dict[str, object]] = []
    for name, token_file in token_files:
        tokens = load_tokens(token_file)
        loss = evaluate_loss(
            model,
            tokens,
            config.training.batch_size,
            config.model.context_length,
            batches,
        )
        domain_results.append(
            {
                "name": name,
                "token_path": str(token_file),
                "token_sha256": _sha256_file(token_file),
                "token_count": int(tokens.size),
                "validation_loss": loss,
                "perplexity": perplexity(loss),
            }
        )

    result: dict[str, object] = {
        "format": "domain-evaluation-v1",
        "config": str(config_file.resolve()),
        "checkpoint": str(checkpoint_file),
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "seed": config.training.seed,
        "eval_batches": batches,
        "domains": domain_results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        type=_parse_domain,
        metavar="NAME=PATH",
        help="評価するuint32 Token列。複数指定可",
    )
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.domain]
    if len(names) != len(set(names)):
        parser.error("domain名は重複できません")
    result = evaluate_domains(
        args.config,
        args.checkpoint,
        args.domain,
        args.output,
        eval_batches=args.eval_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
