# 実験114：未使用FineWeb2日本語shard 3・4による追加事前学習

## 実施前の計画

### 目的

実験111でFineWeb2 Edu Japaneseの未使用shard 1・2から約20M tokensを追加したところ、FineWeb validation lossは2.941318から2.796272へ大きく改善した。一方、raw事前学習だけでは会話形式と医療QAの回答形式が崩れた。実験114では、同じデータセットのさらに未使用なshard 3・4から約20M tokensを追加し、蒸留や教師モデルを使わずに、日本語の語彙・文体・一般知識をどこまで伸ばせるかを調べる。

今回のraw checkpointは会話モデルとして直接採用せず、FineWeb validationと固定生成を知識側の測定として保存する。その後、必要であれば実験113のanswer-focusを含むSFTへ再適用し、追加pretrainingが会話と医療QAへ残る効果を別実験で測定する。

### 仮説

新しいFineWeb2文書をさらに20M tokens学習すれば、実験111 raw bestのFineWeb validation loss 2.796272を下回り、未知文書への語彙適合と文体の安定性が改善する可能性がある。ただし、FineWebだけを続けると会話・医療の特殊形式を忘れるため、general、conversation、medicalのlossは実験111 rawより悪化するか、少なくとも改善しないと予想する。固定promptの文章が自然になるかは不確実であり、validation lossの改善だけでは自然な会話の改善とは判定しない。

### データ取得と前処理

Hugging Faceの`hotchpotch/fineweb-2-edu-japanese`、dataset commit `180ca004c6a89b590daaad86cb062a07a5353c69`、subset `small_tokens_cleaned`を固定する。次の未使用parquetをRunpodへ取得する。

```text
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00003-of-00283.parquet?download=true
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00004-of-00283.parquet?download=true
```

各shardの先頭20,000行を除外し、現在のTokenizerで各10M tokensを上限として本文を抽出する。本文完全一致を除外してから2つのsourceを同じ比率で混ぜ、合計約20M tokensを作る。既存のshard 1・2、Wikipedia、会話、医療コーパスは今回のtrain列へ混ぜない。元parquetと`medilink_analysis`の原本は変更せず、入力URL、サイズ、SHA-256、抽出条件、混合条件、Token列のSHA-256だけを記録する。

### モデルと学習条件

- 実験番号：114
- 実施日：2026-09-07
- 担当：Codex
- 実行環境：Runpod Pod `j9c46julmtbcb4`、NVIDIA A40、PyTorch CUDA
- 初期checkpoint：実験111 raw best、step 10,000
- 初期checkpoint SHA-256：`6957aaab539af1d6924d5c43a0c44a057a356c35dbac79c49fbe2279962468b9`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習：batch size 8、10,000 step、約20.48M提示tokens、AdamW、learning rate 5e-6から5e-7、warmup 500、weight decay 0.1、seed 114
- validation：FineWeb2 Edu Japanese test、20 evaluation batches
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 設定ファイル：`configs/issue1-50m-pretrain-fineweb-new-shards34-runpod-10k.toml`、SHA-256 `bfe833a8b8371c48af1bfb1203bd0338b1c72171cd963b368aff531c0873fb95`
- 学習コード：`scripts/train_torch.py`、SHA-256 `4695dfea5487fb7d912db762c0825a524aa921247dbfb670d74b5002cc4fe001`

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-50m-pretrain-fineweb-new-shards34-runpod-10k.toml \
  --initial-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards-runpod-10k/best.pt \
  --device cuda
```

### 成功・失敗の判定

2つの新規shardから約20M tokensを再現可能に抽出し、入力・本文・混合・Token列のmanifestを保存する。学習はNaN、OOM、shape errorなく10,000 stepを完走し、500 stepごとのvalidation lossと固定prompt生成を残す。FineWeb validation lossが実験111 rawの2.796272を下回ることを第一の性能目標とする。下回らない場合も、新しいデータが効かなかった反証として保存する。

## データ準備中の記録

ここに取得失敗、再試行、入力サイズとSHA-256、抽出結果、混合結果、Token化結果、使用したコマンドを発生順に追記する。失敗したコマンドも削除しない。

## 学習中の記録

ここに500 stepごとのvalidation loss、perplexity、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。崩れた生成も含め、すべてのsample TXTをGitHubへ保存する。

## 実験終了後の記録

ここに最良checkpoint、FineWeb loss、学習時間、raw生成評価、実験111との比較、仮説との一致・不一致、SFT再適用を行うかどうか、次に試す変更を追記する。checkpoint本体はGitへ追加せず、metadataとSHA-256だけを記録する。
