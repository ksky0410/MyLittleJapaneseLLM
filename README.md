# My Little Japanese LLM

Apple SiliconのMacBook上で、日本語Tokenizerと小さなdecoder-only Transformerをゼロから動かして学ぶための最小プロジェクトです。外部から巨大なデータセットを自動取得せず、内蔵サンプルか、ユーザーが指定したUTF-8テキストだけで一連の処理を実行できます。

このリポジトリには、コーパスの準備、SentencePiece Unigram/BPEの学習、Token化、MLXモデルの学習、validation lossの評価、checkpointの保存、固定promptの生成、モデル形状の確認が実装されています。実験の計画と結果は [AGENTS.md](AGENTS.md) の規則に従い、`experiments/notes/` に記録してください。

## 動作環境とインストール

本学習はApple Silicon MacとPython 3.11以上を想定し、MLXを使用します。プロジェクトのコマンドは、現在のシェルのPATHに依存しないよう `.venv/bin/python` を直接指定します。

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[apple,dev]"
```

MLXを使わずにコーパス処理、Tokenizer、設定、モデル形状、テストだけを確認する場合は、次のようにインストールできます。

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

SentencePieceやMLXがない場合、Tokenizerまたは学習系の実行時に、必要なインストールコマンドを含むエラーを表示します。`--help`、設定読み込み、corpus処理、モデル規模の概算はMLXを遅延importするため、MLXなしでも実行できます。

## 内蔵サンプルでdebugする

以下はリポジトリのルート以外から実行しても、相対パスをリポジトリルート基準で解決します。まず、内蔵サンプルを正規化し、seed 42で文書単位に決定的分割します。

```bash
.venv/bin/python scripts/prepare_data.py \
  --input data/sample_ja.txt \
  --output-dir artifacts/corpus \
  --val-ratio 0.2 \
  --seed 42
```

次に、学習splitからTokenizerを作ります。小さなコーパスで要求語彙数が必要な文字種類数より小さい場合は、スクリプトがSentencePieceに渡す語彙数を自動的に下限まで引き上げます。反対に、利用可能なpieceが少ない場合は `hard_vocab_limit=False` により実際の語彙数が要求値を下回ることがあります。実際の語彙数はコマンドの出力で確認してください。

```bash
.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/train.txt \
  --model-prefix artifacts/tokenizer/ja \
  --vocab-size 128 \
  --model-type unigram
```

学習splitと検証splitは別々にToken化します。splitを一つのファイルへ連結しないため、train/validationの境界が混ざりません。各binaryには、Token数、語彙数、EOS ID、SHA-256を記録したJSONも保存されます。

```bash
.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/ja.model \
  --input artifacts/corpus/train.txt \
  --output artifacts/tokens/train.bin

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/ja.model \
  --input artifacts/corpus/val.txt \
  --output artifacts/tokens/val.bin
```

UnigramとBPEを同じ入力で比較するときは、Tokenizerを複数回指定してJSONレポートを出力します。レポートには実語彙数、総token数、平均文字/token、固定した3つの日本語サンプルのpiece分割が含まれます。

```bash
.venv/bin/python scripts/tokenizer_report.py \
  --tokenizer artifacts/tokenizer/unigram.model \
  --tokenizer artifacts/tokenizer/bpe.model \
  --input artifacts/corpus/train.txt
```

学習前に、Tokenizerの実際の語彙数とモデルの入力・出力形状、概算パラメータ数を確認します。

```bash
.venv/bin/python scripts/inspect_model.py --config configs/debug.toml
```

実験ノートに予定を記録した後、debug学習を実行します。`configs/debug.toml` はdim 64、2層、context length 64、100 stepの小さな設定です。学習中はstep 1と20 stepごと、または最終stepにtrain loss、validation loss、perplexity、学習率をJSONLと標準出力へ記録します。ログ間隔は1000 stepを超えないようにしています。

```bash
.venv/bin/python scripts/train.py --config configs/debug.toml
```

学習によって次の成果物が作られます。

- `artifacts/checkpoints/step_000100.npz` と対応するmetadata JSON
- `artifacts/checkpoints/metrics.jsonl`
- `artifacts/checkpoints/summary.json`
- `artifacts/samples/step_000000.txt` と学習中の固定prompt生成結果

checkpointは重みファイルとJSON metadataに分けて保存します。生成と評価では、metadataのformat、モデルの語彙数・dim・層数・head数・context length・MLP倍率を現在の設定およびTokenizerと比較し、不一致ならロード前に拒否します。

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/debug.toml \
  --checkpoint artifacts/checkpoints/step_000100.npz

.venv/bin/python scripts/generate.py \
  --config configs/debug.toml \
  --checkpoint artifacts/checkpoints/step_000100.npz \
  --output artifacts/samples/manual.txt
```

## 手元のコーパスを使う

巨大なデータセットのダウンロードは行いません。手元のUTF-8テキストを指定して、同じコマンドを実行してください。入力は1行を1文または1段落として扱います。

```bash
.venv/bin/python scripts/prepare_data.py \
  --input /path/to/your/japanese.txt \
  --output-dir artifacts/my-corpus \
  --val-ratio 0.05 \
  --seed 42
```

その後、Tokenizerとencodeの `--input` を `artifacts/my-corpus/train.txt` と `artifacts/my-corpus/val.txt` に変更します。データの出所、ライセンス、前処理、ハッシュ、Token数は実験ノートへ記録してください。

## テストと入口の確認

プロジェクトの仮想環境を使い、軽量なテストを実行できます。MLXまたはSentencePieceがない環境では、それらを必要とするテストだけがskipされます。

```bash
.venv/bin/python -m pytest -q

for script in scripts/prepare_data.py scripts/train_tokenizer.py scripts/encode_data.py scripts/tokenizer_report.py scripts/inspect_model.py scripts/train.py scripts/generate.py scripts/evaluate.py; do
  .venv/bin/python "$script" --help >/dev/null
done
```

生成結果、checkpoint、ローカルデータ、キャッシュは `.gitignore` でGit管理対象外です。内蔵サンプル自体と、再現に必要なコード・設定・実験ノートは管理対象として残します。
