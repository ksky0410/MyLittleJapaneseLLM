# 実験119：未使用FineWeb2日本語shard 5・6による追加事前学習

## 実施前の計画

### 目的

Exp114ではFineWeb2のshard 3・4から約20M tokensを50Mモデルへ追加事前学習し、FineWeb validation lossは2.796272から2.777802へ改善した。しかし、FineWeb文書だけを続けて学習すると会話・医療の形式を忘れ、raw checkpointは自然な会話モデルとして使えなかった。Exp117・118のSFT比較でも、会話データ比率や長文samplingだけでは一般会話の自然さが安定して改善しなかった。

今回はExp114 raw bestを初期値に戻し、同じFineWeb2 Edu Japaneseの未使用shard 5・6から新たに約20M tokensを追加する。まず一般日本語の語彙・文体・知識側の基盤を増やし、その後にExp117で検証した会話SFTを再適用する二段階構成を用意する。raw checkpoint単体を会話モデルとして採用せず、SFT後の自然さを最終判定とする。

### 仮説

新しいFineWeb2文書を約20M tokens追加すれば、未使用FineWeb validation lossがExp114 rawの2.777802を下回り、自然な日本語の文脈・語彙・文章の続き方が改善する可能性がある。ただし、FineWebだけの追加事前学習は会話marker・EOS・医療QA形式を壊す可能性が高い。したがってraw評価では会話の改善を期待せず、後続SFTで一般会話と医療回答を戻せるかを確認する。

Exp114から同じ20M規模をもう一度追加することで、50Mモデルに対して累積FineWeb追加量を約40M tokensまで増やす。改善が飽和する場合は、今後はFineWebだけを増やすのではなく、一般文書・会話・医療を混ぜた継続事前学習へ切り替える。

### データ取得と前処理

Hugging Faceの`hotchpotch/fineweb-2-edu-japanese`、dataset commit `180ca004c6a89b590daaad86cb062a07a5353c69`、subset `small_tokens_cleaned`を固定する。次のURLからshard 5・6をRunpodへ取得する。

```text
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00005-of-00283.parquet?download=true
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00006-of-00283.parquet?download=true
```

各shardの先頭20,000行を除外し、現在のTokenizerで各10M tokensを上限として本文を抽出する。本文完全一致を除外してから、seed 11901でshard 5・6を同じ重みで混ぜ、合計約20M tokensを作る。既存のshard 1〜4、Wikipedia、会話、医療コーパスは新しいtrain列へ混ぜない。元parquetと`medilink_analysis`の原本は変更せず、URL、サイズ、SHA-256、抽出条件、混合条件、Token列のSHA-256を記録する。

### モデルと学習条件

- 初期checkpoint：Exp114 raw best、step 10,000、重みSHA-256 `4f6f9f4ddad1f717dbf170de8b1d5e704e1ef3975c1b4ab4e5e905abc5c3eca6`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習：batch size 8、10,000 step、約20.48M提示tokens、AdamW、learning rate 5e-6から5e-7、warmup 500、weight decay 0.1、seed 119
- validation：`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`、20 evaluation batches
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 設定：`configs/issue1-50m-pretrain-fineweb-new-shards56-runpod-10k.toml`

設定ファイルのSHA-256は`22b9d5d47cfdb3c1f847ff7e2f84550ebf17c4a234215ed6bf5e92bb6d5f0cbd`である。

### 成功・失敗の判定

新しい2 shardから約20M tokensを再現可能に抽出し、入力・本文・混合・Token列のmanifestを保存する。学習はNaN、OOM、shape errorなく10,000 stepを完走し、500 stepごとのvalidation lossと固定prompt生成を残す。FineWeb validation lossがExp114 rawの2.777802を下回ることを知識側の第一目標とする。ただし、raw lossの改善だけで自然な日本語の達成とは判定しない。後続SFTの初期値として有用かどうかを、会話・医療評価と生成本文で確認する。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-50m-pretrain-fineweb-new-shards56-runpod-10k.toml \
  --initial-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards34-runpod-10k/best.pt \
  --device cuda
```

## データ準備中の記録

データ取得、抽出、混合、Token化の各段階で、失敗した試行も含めて入力サイズ、hash、コマンド、生成物を追記する。完了後にRunpod上で入力hashを再確認してから学習を開始する。

## データ準備の結果

2026-09-07、shard 5・6のparquetをRunpodへ取得した。shard 5は269,589,405 bytes、入力SHA-256 `239a4cee8b8e87644d7e8376e3dd616281810bffc97047fae307bb7cda4b6411`、shard 6は268,992,270 bytes、入力SHA-256 `5b1f7206117b5c9e4d50498e3788b0943991c6468cc64560728d6c43c9f034f1`だった。HTTP 200で取得でき、途中停止や再試行による破損は確認されなかった。

先頭20,000行を除外して現在のTokenizerで抽出した結果、shard 5は19,585文書、9,999,758 tokens、本文SHA-256 `03808507303850d4626a6ac3a4e0d9db6a8587989f2fb001d3398032ab5bb444`、shard 6は19,552文書、9,999,909 tokens、本文SHA-256 `3ec0be4c354c3ef9a8823f888a5729e3421cbbca8759767e9e7334ee4870f916`となった。空行はなく、各shard内の本文完全一致重複は0件だった。

seed 11901、shard 5・6のweight 1:1、target 20,000,000 tokensで決定的に混合した。混合本文は39,137単位、19,999,667 tokens、本文SHA-256 `817deeccf8119ad999f900afac907a23039bd21ed35a81e0d83e94e7a8f96915`、mix manifest SHA-256 `6d9ff50f07e7c174a02c6794458503fde2a07d36224cb56e26bfe5884c57891d`となった。実測token比率はshard 5が49.9996%、shard 6が50.0004%である。Token化後のbinaryは19,999,667 tokens、SHA-256 `fd0917a05519c64e9f24b2c86d924ab4260da0342b2c2e558783357812eee741`、metadata SHA-256 `88673d9e0eb33171a846fb96c2919875b8f3a82abaf182422c41c457a929e99b`となった。

学習開始前のRunpod照合では、設定SHA-256 `22b9d5d47cfdb3c1f847ff7e2f84550ebf17c4a234215ed6bf5e92bb6d5f0cbd`、初期checkpoint SHA-256 `4f6f9f4ddad1f717dbf170de8b1d5e704e1ef3975c1b4ab4e5e905abc5c3eca6`、Tokenizer SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`が計画値と一致した。抽出・混合・Token化のログは`experiments/results/exp119/`へ保存した。

## 学習中の記録

学習開始後は少なくとも1,000 step以内ごとにvalidation loss、perplexity、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。崩れた生成も含め、すべてのsample TXTをGitHubへ保存する。
