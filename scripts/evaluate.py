"""metadataを検証したcheckpointのvalidation lossを再計算する。"""

from __future__ import annotations

import argparse
import json

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    require_mlx()
    config = load_config(repo_path(args.config))
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
        config.model.ffn_type,
    )
    metadata = load_checkpoint(
        model, repo_path(args.checkpoint), signature_from_config(config, vocab_size)
    )
    loss = evaluate_loss(
        model,
        load_tokens(config.paths.val_tokens),
        config.training.batch_size,
        config.model.context_length,
        config.training.eval_batches,
    )
    result = {
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "validation_loss": loss,
        "perplexity": perplexity(loss),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
