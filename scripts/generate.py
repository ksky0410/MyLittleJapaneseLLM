"""metadataを検証したcheckpointから日本語を生成する。"""

from __future__ import annotations

import argparse

from _common import repo_path

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.model import TinyJapaneseGPT, require_mlx
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.training import (
    generate_ids,
    load_checkpoint,
    signature_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None, help="省略時は標準出力のみ")
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
    )
    load_checkpoint(
        model, repo_path(args.checkpoint), signature_from_config(config, vocab_size)
    )
    prompt = args.prompt if args.prompt is not None else config.generation.prompt
    max_new_tokens = args.max_new_tokens or config.generation.max_new_tokens
    ids = processor.encode(prompt, out_type=int)
    output = processor.decode(
        generate_ids(
            model,
            ids,
            max_new_tokens,
            config.model.context_length,
            config.generation.temperature,
            config.generation.top_k,
            config.training.seed,
            int(processor.eos_id()),
        )
    )
    print(output)
    if args.output is not None:
        output_path = repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"prompt: {prompt}\n\n{output}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
