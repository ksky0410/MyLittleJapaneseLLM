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

2026-09-07に、予定したURLからshard 3・4のparquetをRunpodへ取得した。shard 3は269,300,696 bytes、SHA-256は`c2e40f85c363aebfbf72fb4f1486d6142f9ee0d4e3cdc2cfecb03fa9a6a5a2b8`、shard 4は268,420,983 bytes、SHA-256は`6a68a893804f2a9fadf26de01d2ede5f8ae8f3de12df1c246b8538c9e71d12c0`だった。`wget -c`に再試行を設定したが、確認時点で両方とも完全取得され、取得処理の失敗は発生していない。

ここに取得失敗、再試行、入力サイズとSHA-256、抽出結果、混合結果、Token化結果、使用したコマンドを発生順に追記する。失敗したコマンドも削除しない。

shard 3・4の抽出は並列に実行し、どちらも正常終了した。shard 3は先頭20,000行を除外した後19,654文書、9,999,914 tokensを選択し、本文SHA-256は`5e1d80c71ef4241143ecb5cd28224044f3e132ac64dbd2c845e3c69f858d9011`、manifest SHA-256は`437bd29dec3457dd9d7786b32951d9625eb9d8f41f2aeae8ca54e0decbb41149`だった。shard 4は19,583文書、9,999,582 tokensを選択し、本文SHA-256は`8c6b97bad4f08b8b08cd104e87d906ed0a146290b957f02089dd7b4fb504e215`、manifest SHA-256は`5ff03ac86f5ece928f1f13364750455f99250d2d129454d7ac37b5d6dbfad2e7`だった。両方とも空行と本文完全一致の重複は0件で、Tokenizer SHA-256は計画値と一致した。

抽出本文をseed 11401、shard 3:4 = 1:1で混合した。出力は39,237単位、19,999,496 tokens、本文SHA-256 `ff2afe12a7c00696e59772a243ee2e6e3d97514ba94a42cc1c065fa0f14409bf`、mix manifest SHA-256 `9a0ea3f1d4644d12477d5a90ce6eeb6dd29c3c8b4472f6d413a45bfb011f94eb`となった。実測Token比率はshard 3が50.0008%、shard 4が49.9992%で、重複除去は0件だった。SentencePieceでToken化し、Token列は19,999,496、binary SHA-256 `8d387f27ac75ce2c214a0a60d47b0f03d7d1f168fc17a85095264976fd6f91d6`、metadata SHA-256 `8f812e3e6c1117b4a6705324d595234bd0c79b66ab602cb7f48de680d9ccf91b`となった。

学習開始前のSHA照合では、新しい設定ファイルがRunpod側へまだ同期されておらず、`sha256sum`が設定ファイルだけを`No such file or directory`として終了した。Token列、validation、初期checkpoint、学習コードの照合は成功しており、学習や既存成果物には影響しない。設定ファイルを同期してから照合をやり直す。

## 学習中の記録

ここに500 stepごとのvalidation loss、perplexity、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。崩れた生成も含め、すべてのsample TXTをGitHubへ保存する。

### 2026-09-07：step 1

実験111 raw bestからRunpod A40上で追加事前学習を開始した。step 1のFineWeb validation lossは2.796276、perplexityは16.3835、learning rateは1.0e-8、経過時間は2.44秒だった。実験111 raw bestの評価値2.796272とほぼ一致しており、checkpointのreload、Token列、validation経路は正常である。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 500〜1,000

step 500のFineWeb validation lossは2.795288、perplexityは16.3673、learning rateは5.0e-6、経過時間は33.21秒だった。step 1,000ではloss 2.793580、perplexity 16.3394、learning rate 4.9694e-6、経過時間64.41秒となった。開始時から0.002696改善し、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,500〜2,000

step 1,500のFineWeb validation lossは2.792016、step 2,000は2.790129だった。step 2,000時点で開始時から0.006147改善し、learning rateは4.7292e-6、経過時間は128.18秒となった。validation lossは小さいながら一貫して低下しており、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 2,500〜3,000

step 2,500のFineWeb validation lossは2.788536、step 3,000は2.786680だった。step 3,000時点で開始時から0.009596改善し、learning rateは4.2744e-6、経過時間は192.42秒となった。validation lossは安定して低下している。

### 2026-09-07：step 3,500〜4,000

step 3,500のFineWeb validation lossは2.785577、step 4,000は2.784478だった。step 4,000時点で開始時から0.011798改善し、learning rateは3.6545e-6、経過時間は256.19秒となった。学習は安定しており、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,500〜5,500

step 4,500のFineWeb validation lossは2.782920、step 5,000は2.781984、step 5,500は2.781597だった。step 5,500時点で開始時から0.014678改善し、learning rateは2.5649e-6、経過時間は353.04秒となった。lossの低下は緩やかになったが、学習は安定している。

### 2026-09-07：step 6,000

step 6,000のFineWeb validation lossは2.780659、perplexityは16.1297、learning rateは2.1984e-6、経過時間は385.86秒だった。開始時から0.015616改善し、学習率が下がった後もvalidation lossは小幅に低下している。NaN、OOM、shape errorは発生していない。

## 実験終了後の記録

ここに最良checkpoint、FineWeb loss、学習時間、raw生成評価、実験111との比較、仮説との一致・不一致、SFT再適用を行うかどうか、次に試す変更を追記する。checkpoint本体はGitへ追加せず、metadataとSHA-256だけを記録する。
