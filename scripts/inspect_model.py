"""Tokenizerとdebug設定からモデル形状・概算パラメータ数を確認する。"""

from __future__ import annotations

import argparse

from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.model import estimate_parameter_count
from my_little_japanese_llm.tokenizer import load_processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    args = parser.parse_args()
    config = load_config(repo_path(args.config))
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    parameters = estimate_parameter_count(
        vocab_size,
        config.model.dim,
        config.model.layers,
        config.model.heads,
        config.model.context_length,
        config.model.mlp_ratio,
    )
    print(f"vocab_size={vocab_size}")
    print(f"token_embedding_shape=({vocab_size}, {config.model.dim})")
    print(f"layers={config.model.layers}")
    print(f"dim={config.model.dim}")
    print(f"heads={config.model.heads}")
    print(f"context_length={config.model.context_length}")
    print(f"estimated_parameters={parameters:,}")


if __name__ == "__main__":
    main()
