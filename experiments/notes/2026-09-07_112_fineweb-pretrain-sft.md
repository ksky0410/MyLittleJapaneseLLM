# 実験112：新規FineWeb追加事前学習checkpointへの一般・医療SFT

## 実施前の計画

実験111では、未使用FineWeb2日本語shard 1・2から19,999,123 tokensを抽出し、実験110のSFT済みbest checkpointへ10,000 stepの追加事前学習を行った。FineWeb validation lossは2.941318から2.796272へ大きく改善したが、raw checkpointの固定promptには「お客様の声」の反復が現れ、medical回答形式も崩れた。これはraw事前学習の知識側効果と会話形式の維持が別問題であることを示している。

今回は実験111 raw bestへ、実験110と同じ一般・医療SFTを8,000 step再適用する。目的は、新しいFineWeb文書で得た一般日本語の改善を残しながら、一般会話のEOS・話者形式と医療問題の回答形式を回復できるかを確認することである。教師LLMによる蒸留、reasoningデータ、医療データの追加倍率は使わず、pretraining後に元のSFTを戻す順序だけを検証する。

### 仮説

実験111 raw checkpointはFineWeb lossが大きく改善しているため、SFT後にもgeneral validation lossと一般会話の語彙適合へ一部の効果が残る可能性がある。実験110と同じSFTなら、一般会話のEOS 48/48と医療回答の抽出162/162を回復し、医療F1もrawの0.2467から0.36以上へ戻ると予想する。一方、SFTがFineWeb lossを押し戻すため、実験111 rawの2.796272をそのまま維持することは期待しない。

### 開始前の条件

- 実験番号：112
- 実施日：2026-09-07
- 担当：Codex
- 実行環境：Runpod Pod `j9c46julmtbcb4`、NVIDIA A40、PyTorch CUDA
- 初期checkpoint：実験111 raw best、step 10,000
- 初期checkpoint SHA-256：`6957aaab539af1d6924d5c43a0c44a057a356c35dbac79c49fbe2279962468b9`
- SFT train：`artifacts/sft/issue1-general-medical-concat-v1/train.npz`、SHA-256 `598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- SFT validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 112、rehearsal ratio 0.2
- 設定ファイル：`configs/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k.toml`
- 設定ファイルSHA-256：`2f0e0f0e5d24eaebda8340f4da34487eb6ab6420573f587713ec240447a3a1d0`
- 学習コード：`scripts/train_sft_torch.py`
- 学習コードSHA-256：`bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \\
  --config configs/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k.toml \\
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards-runpod-10k/best.pt \\
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \\
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \\
  --output-dir artifacts/checkpoints/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k \\
  --samples-dir artifacts/samples/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k \\
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \\
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと生成サンプルを保存する。実験111 rawで崩れた会話形式が回復し、一般会話48例のEOS 48/48、医療162例の回答形式抽出162/162を目標とする。実験110と比較してFineWeb、general、conversation、medicalのloss、一般会話F1、医療F1、医療正解率をすべて記録し、どの能力が追加FineWebから残ったかを切り分ける。

## 学習中の記録

ここに250 stepごとのvalidation loss、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。悪い生成や短すぎる応答も削除せず保存する。

### 2026-09-07：step 1〜250

Runpod A40上で実験111 raw bestからSFTを開始した。step 1のvalidation lossは2.913144、step 250は2.804113、step 250のlearning rateは1.9998e-5、経過時間は21.39秒だった。step 250の固定会話生成は「こんにちは!」に対して「こんにちは!」となり、raw checkpointで見られた長い反復はこの時点では現れていない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,000〜1,250

step 1,000のvalidation lossは2.811358、perplexityは16.6325、learning rateは1.9538e-5、経過時間は82.00秒だった。step 1,250ではloss 2.808899、perplexity 16.5916、learning rate 1.9209e-5、経過時間102.14秒となった。現時点の最良はstep 250の2.804113で、学習は安定している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 2,000〜2,250

step 2,000のvalidation lossは2.803952、perplexityは16.5098、learning rateは1.7739e-5、経過時間は161.93秒だった。step 2,000でvalidation lossの最良値をわずかに更新したが、step 2,250では2.811817へ戻った。学習率はまだ減衰前半であり、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,000

step 3,000のvalidation lossは2.789558、perplexityは16.2738、learning rateは1.4862e-5、経過時間は242.28秒だった。step 2,750の2.799209からさらに改善し、ここまでの最良値を更新した。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,000

step 4,000のvalidation lossは2.787631、perplexityは16.2425、learning rateは1.1366e-5、経過時間は322.61秒だった。最良値はstep 3,250の2.786897で、step 3,500以降はこの近辺で推移している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 5,000〜5,250

step 5,000のvalidation lossは2.769219、perplexityは15.9462、learning rateは7.8119e-6、経過時間は402.41秒だった。step 5,250ではloss 2.763094、perplexity 15.8488、learning rate 6.9821e-6、経過時間422.81秒となり、実験110のbest validation loss 2.773049を更新した。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,000

step 6,000のvalidation lossは2.761517、perplexityは15.8238、learning rateは4.7681e-6、経過時間は482.98秒だった。最良値はstep 5,500の2.758811で、実験110のbest 2.773049より0.014238低い。NaN、OOM、shape errorは発生していない。

## 実験終了後の記録

ここに最良checkpoint、学習時間、領域別loss、一般会話48例、医療162例、実験110・111との比較、仮説との一致・不一致、次に試す変更を追記する。
