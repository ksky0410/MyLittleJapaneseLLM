"""SentencePieceで1つのsplitをuint32のToken列へ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from _common import repo_path

from my_little_japanese_llm.tokenizer import encode_text_file, load_processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="train.txtまたはval.txt。境界を保つためsplitごとに実行",
    )
    parser.add_argument("--output", required=True, help="uint32 raw binaryの出力先")
    args = parser.parse_args()

    tokenizer_path = repo_path(args.tokenizer)
    input_path = repo_path(args.input)
    output_path = repo_path(args.output)
    ids = encode_text_file(tokenizer_path, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.uint32).tofile(output_path)
    metadata = {
        "format": "uint32_sentencepiece_ids_v1",
        "tokenizer": str(tokenizer_path.resolve()),
        "input": str(input_path.resolve()),
        "tokens": len(ids),
        "eos_id": int(load_processor(tokenizer_path).eos_id()),
        "vocab_size": int(load_processor(tokenizer_path).vocab_size()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"tokens: {len(ids)}")
    print(f"binary: {output_path}")
    print(f"metadata: {output_path}.json")


if __name__ == "__main__":
    main()
