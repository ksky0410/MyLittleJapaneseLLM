# 実験115：shard 3・4追加事前学習後のanswer-focus SFT

## 実施前の計画

### 目的

実験114でFineWeb2 Edu Japaneseの未使用shard 3・4を約20M tokens追加したraw checkpointへ、実験113と同じ一般会話・通常医療・answer-focus医療のSFTを適用する。実験113との比較によって、追加20M tokensの日本語事前学習が、SFT後の一般会話と医療QAの正答率・説明品質へ残るかを測定する。

### 仮説

実験114 rawはFineWeb validation lossを2.777802まで下げた一方、会話・医療の形式を忘れていた。実験113と同じSFTを再適用すれば、EOSと回答形式は回復すると予想する。さらに、追加FineWeb文書で増えた語彙・文体が残るなら、実験113よりgeneral・conversationのlossまたは一般会話F1が改善し、医療の完全一致33/162も維持または上回る可能性がある。ただし、raw pretrainingで知識の配置が変わったため、answer-focus SFTのvalidationが不安定になったり、医療理由の誤生成が残ったりする可能性もある。

### 比較条件

実験113から初期checkpointだけを変更し、その他を揃える。

- 初期checkpoint：実験114 raw best、step 10,000、SHA-256 `4f6f9f4ddad1f717dbf170de8b1d5e704e1ef3975c1b4ab4e5e905abc5c3eca6`
- 学習データ：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、一般会話約12.8万例、通常医療2,945例、answer-focus医療2,945例
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、実験113と同一
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 115
- 評価：FineWeb2、general、conversation、medicalのdomain loss、一般会話48例、医療162例

### 入力のSHA-256

学習開始前に次のハッシュをRunpod上で照合する。

- 設定：`configs/issue1-fineweb-shards34-answer-focus-sft-runpod-8k.toml`、SHA-256 `9c43cedeb66436bed8b807fbd9186263a1fa55ae26d2d0e70b977affb169c508`
- 学習コード：`scripts/train_sft_torch.py`、SHA-256 `bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`
- answer-focus train：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal Token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-fineweb-shards34-answer-focus-sft-runpod-8k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards34-runpod-10k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-fineweb-shards34-answer-focus-sft-runpod-8k \
  --samples-dir artifacts/samples/issue1-fineweb-shards34-answer-focus-sft-runpod-8k \
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功・失敗の判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと固定prompt生成を保存する。実験113と同じ評価器で比較し、一般会話F1、EOS、平均生成長、医療の回答抽出・完全一致・F1を確認する。追加pretraining後のSFTが実験113より全般に悪化しても、データ量を増やすだけではSFT性能が単調に伸びない反証として記録する。

## 学習中の記録

ここに1,000 stepを超えない間隔でvalidation loss、perplexity、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。悪い生成も削除しない。

### 2026-09-07：step 1〜250

Runpod A40上で実験114 raw bestからanswer-focus SFTを開始した。step 1のvalidation lossは2.937145、perplexityは18.8619、learning rateは1.0e-7、経過時間は0.89秒だった。step 250ではvalidation loss 2.829083、perplexity 16.9299、learning rate 1.9998e-5、経過時間21.47秒となった。実験113のstep 250よりvalidation lossは高いが、学習は安定しており、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 500〜1,000

step 500のvalidation lossは2.811753、step 750は2.823874、step 1,000は2.813150だった。最良値はstep 500の2.811753で、step 1,000時点のlearning rateは1.9538e-5、経過時間は84.18秒だった。実験113の同時点より高いものの、学習は安定している。

### 2026-09-07：step 1,250〜1,500

step 1,250のvalidation lossは2.828422、step 1,500は2.827187だった。step 500の2.811753を更新しておらず、step 1,500時点のlearning rateは1.8796e-5、経過時間は125.16秒だった。validation lossは実験113より高い状態で推移しているが、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,750〜2,250

step 1,750のvalidation lossは2.818044、step 2,000は2.808153、step 2,250は2.807672だった。step 2,250で最良値を更新したが、実験113の同時期よりまだ高い。step 2,250時点のlearning rateは1.7105e-5、経過時間は186.84秒で、学習は安定している。

### 2026-09-07：step 2,500〜3,000

step 2,500のvalidation lossは2.809942、step 2,750は2.810437、step 3,000は2.802531だった。step 3,000で最良値を更新し、step 3,000時点のlearning rateは1.4862e-5、経過時間は247.61秒となった。実験113との差は縮まっているが、まだ追加pretraining後の条件が高い。

## 実験終了後の記録

ここに最良checkpoint、学習時間、domain loss、一般会話48例、医療162例、実験113・114との比較、仮説の判定、次の一手を追記する。checkpoint本体はGitHubへ追加せず、metadata・SHA-256・生成文・評価全文を保存する。
