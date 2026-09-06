# 実験111：未使用FineWeb2日本語データ20M tokensの追加事前学習

## 実施前の計画

実験110では、実験109で追加した会話・医療データを含む20M tokensのraw事前学習checkpointへ、一般・医療SFTを8,000 step戻した。その結果、general・conversation・medicalのvalidation lossと医療F1は実験105・108より改善したが、FineWeb validation lossは少し悪化し、一般会話は短い相づちへ寄った。今回は、同じデータをさらに反復するのではなく、FineWeb2 Edu Japaneseのこれまで未使用のtrain shardから新しい日本語Web文書を約20M tokens抽出し、実験110のbest checkpointへ追加事前学習する。

今回の目的は、蒸留や教師モデルを使わずに、追加の高品質日本語文書と学習stepだけで、日本語の語彙・文体・一般知識を改善できるかを調べることである。raw事前学習で会話形式が崩れる可能性は実験109で確認済みなので、raw checkpointは知識側の候補として保存し、その後に実験110と同じSFTを再適用して総合評価する。SFTまで完了したcheckpointを実験111の最終候補とし、raw checkpoint単体を会話モデルとして採用しない。

### 仮説

実験109の追加コーパスは会話・医療validationを改善したが、FineWeb validationへの直接的な新規情報は限定的だった。未使用FineWeb2の新しいshardを20M tokens追加すれば、FineWeb lossと一般文書の語彙適合が改善し、SFT後にもgeneral validationの改善が一部残る可能性がある。learning rateは実験109の1e-5より低い5e-6へ下げ、実験110で回復した表現を一度に壊しすぎないようにする。ただし、raw事前学習だけでは会話EOSや話者形式を忘れると予想する。

### データ取得と固定条件

FineWeb2 Edu Japaneseのdataset commit `180ca004c6a89b590daaad86cb062a07a5353c69`、subset `small_tokens_cleaned`を使用する。これまで使った`train-00000-of-00283.parquet`とは別に、`train-00001-of-00283.parquet`と`train-00002-of-00283.parquet`を取得する。各shardの先頭20,000行は、配布元が案内する重複範囲を避けるため除外する。各shardからTokenizer上限10M tokensで本文を抽出し、重複を除いて2つのsourceを決定的に混ぜ、合計約20M tokensを作る。元のparquetはGitへ追加せず、URL、入力hash、抽出manifest、混合manifest、Token列hashを記録する。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`を固定する。validationは学習に使わない`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`を使う。既存のFineWeb2 shard 0、Wikipedia、会話、医療データは今回の新規train sourceへ混ぜず、情報源を分離して効果を判定する。

### モデルと学習条件

- 実験番号：111
- 実施日：2026-09-07
- 担当：Codex
- 実行環境：Runpod Pod `j9c46julmtbcb4`、NVIDIA A40、PyTorch CUDA
- 初期checkpoint：実験110 best、step 7,500
- 初期checkpoint SHA-256：`cbf18b9cbd39ec9256811e945ecc3e7fc99fe39d12d6987df7198027fd8ae492`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習：batch size 8、10,000 step、約20.48M提示tokens、AdamW、learning rate 5e-6から5e-7、warmup 500、weight decay 0.1、seed 111
- validation：FineWeb2 Edu Japanese test、20 evaluation batches
- 設定ファイル：`configs/issue1-50m-pretrain-fineweb-new-shards-runpod-10k.toml`
- 設定ファイルSHA-256：`8e7b9e005f9857f424899bb5d763589b0b58e8d4c65f3c24ee7fe31a4e386823`
- 学習コード：`scripts/train_torch.py`
- 学習コードSHA-256：`4695dfea5487fb7d912db762c0825a524aa921247dbfb670d74b5002cc4fe001`

### 実行コマンド

データ取得・抽出・混合・Token化を完了し、各ファイルのSHA-256を照合してから、次のコマンドを実行する。

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_torch.py \\
  --config configs/issue1-50m-pretrain-fineweb-new-shards-runpod-10k.toml \\
  --initial-checkpoint artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k/best.pt \\
  --device cuda
```

### 成功判定

新しい2 shardから約20M tokensを再現可能に抽出し、入力・出力hashをmanifestへ保存することをデータ準備の成功条件とする。学習はNaN、OOM、shape errorなく10,000 stepを完走し、500 stepごとのvalidation lossと生成文を保存する。FineWeb validation lossが実験110の2.941318から下がることを第一の性能目標とする。ただしraw checkpointの会話形式が崩れることは想定内であり、SFT再適用後のgeneral・conversation・medical loss、一般会話48例、医療162例で最終的な採用可否を判断する。

## データ準備中の記録

ここに取得URL、ファイルサイズ、入力hash、抽出結果、重複除去、Token数、manifest hashを追記する。取得や抽出が失敗した場合も、そのコマンドとエラーを残す。

## 学習中の記録

ここに500 stepごとのvalidation loss、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。生成文は崩れたものも含め、GitHubへ保存する。

## 実験終了後の記録

ここに最良checkpoint、FineWeb loss、学習時間、raw生成評価、SFT再適用の結果、実験110との比較、仮説との一致・不一致、次に試す変更を追記する。
