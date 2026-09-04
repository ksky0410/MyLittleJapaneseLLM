"""日本語コーパスを正規化し、決定的にtrain/valへ分割する。"""

from __future__ import annotations

import argparse

from _common import repo_path

from my_little_japanese_llm.corpus import read_documents, split_documents, write_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/sample_ja.txt",
        help="UTF-8テキスト。相対パスはリポジトリルート基準",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/corpus", help="分割結果の出力先"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = repo_path(args.input)
    documents = read_documents(source)
    train, val = split_documents(documents, args.val_ratio, args.seed)
    manifest = write_split(
        repo_path(args.output_dir), train, val, source, args.val_ratio, args.seed
    )
    print(f"学習文書: {manifest['train_documents']}件")
    print(f"検証文書: {manifest['validation_documents']}件")
    print(f"学習文字数: {manifest['train_characters']}")
    print(f"検証文字数: {manifest['validation_characters']}")
    print(f"manifest: {repo_path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
