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

学習開始前に、Runpod上で設定SHA-256 `8e7b9e005f9857f424899bb5d763589b0b58e8d4c65f3c24ee7fe31a4e386823`、学習コードSHA-256 `4695dfea5487fb7d912db762c0825a524aa921247dbfb670d74b5002cc4fe001`、初期checkpoint SHA-256 `cbf18b9cbd39ec9256811e945ecc3e7fc99fe39d12d6987df7198027fd8ae492`、新規Token列SHA-256 `5aea382f8754df2c9d594003cb607489afbadbe6ac7c757f4c00069d1f33bf8e`が一致することを確認する。

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

2026-09-07に、次の2つのURLからFineWeb2 Edu JapaneseのparquetをRunpodへ取得した。shard 1は269,326,085 bytes、SHA-256は`91ca7d654aaae70f660fe933c6e101ee2b7e78438da9a267976ee3c9dc062be8`、shard 2は268,827,173 bytes、SHA-256は`3d899cd422633e0ef45e3602e952601aa53cd27f57c13d033d39d74c17a2c7c5`である。

```text
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00001-of-00283.parquet?download=true
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00002-of-00283.parquet?download=true
```

抽出の最初の試行では、Runpod側に`scripts/import_fineweb2_edu_japanese.py`がまだ同期されておらず、`can't open file '/workspace/exp100/scripts/import_fineweb2_edu_japanese.py'`で終了した。データ、checkpoint、既存Token列は変更されていない。`uv run --with pyarrow`によってpyarrowを一時環境へ導入したが、プロジェクトの依存ファイルは変更していない。この失敗を記録した後、同スクリプトをRunpodへ同期して抽出を再実行する。

スクリプトを同期した後の抽出は成功した。shard 1は19,687 documents、9,999,311 tokens、本文SHA-256 `6345d0d6fcba4d6308e44c8c77938f574382b61018a6877c26268a64ba2ee4df`、shard 2は19,583 documents、9,999,812 tokens、本文SHA-256 `68cb523c5dd56e5e246d0003f0679ed14ff03ea7c5716c543cb9539841b82ab3`となった。いずれも先頭20,000行を除外し、Tokenizer SHA-256は固定値と一致した。

その後、`mix_corpora.py`と`encode_data.py`がRunpod側に同期されていない状態で混合・Token化を開始し、`can't open file '/workspace/exp100/scripts/mix_corpora.py'`で終了した。この試行も学習や既存データを変更していない。過去の実験で必要だったスクリプト同期が不足していたため、`mix_corpora.py`、`encode_data.py`と、それらが使う既存のsource moduleを同期してから再実行する。

スクリプトを同期して混合・Token化を再実行した結果、重複除去後の39,270単位から19,999,123 tokensを採用した。shard 1のtoken shareは49.9987%、shard 2は50.0013%で、source間の重複は0件だった。混合本文のSHA-256は`cb665a35d2a384bb60343a2665c94970a4d46d3cddb92eb0624a040f889e58c8`、mix manifestのSHA-256は`2649109f659e7dc11ade8fb19523213f6e44659e0540f262d11e997c8a6f3633`である。Token化後のToken数は19,999,123、binaryのSHA-256は`5aea382f8754df2c9d594003cb607489afbadbe6ac7c757f4c00069d1f33bf8e`、metadataのSHA-256は`d061c406ebe9131a8bea1666a53240e3717623933819f327e46fd144e0ba1fcf`となった。shard manifestのSHA-256は、shard 1が`043493efa7cb4c6b7cb91022d495e9d3212065dc3f5ee0e78c88dde190caca9c`、shard 2が`b836874cc0380bb3e62e6de3e3b2c5fdf52372d42752ee73dae1967cfc1ca6ee`である。

混合・Token化に使ったスクリプトは、`mix_corpora.py` SHA-256 `35cef7df5c91adaac8d77933f845de8acd056d2d46cd7677ac3103ae0ab19255`、`encode_data.py` SHA-256 `fbc83816b93c2f92a6cd9681e494b0a0a1f7ba45b82640becdf6ee83d99c9e4e`である。Tokenizerとmixing・data moduleはローカルとRunpodでSHA-256が一致していることを確認した。

## 学習中の記録

ここに500 stepごとのvalidation loss、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。生成文は崩れたものも含め、GitHubへ保存する。

### 2026-09-07：step 1

実験110のbest checkpointから、Runpod A40上で追加事前学習を開始した。step 1のFineWeb validation lossは2.941314、perplexityは18.9407、learning rateは1.0e-8、経過時間は0.91秒だった。実験110の同じFineWeb評価値2.941318とほぼ一致しており、初期checkpointのreloadと評価経路は正常である。NaN、OOM、shape errorは発生していない。

## 実験終了後の記録

ここに最良checkpoint、FineWeb loss、学習時間、raw生成評価、SFT再適用の結果、実験110との比較、仮説との一致・不一致、次に試す変更を追記する。
