# 実験110：追加事前学習checkpointへの一般・医療SFT再適用

## 実施前の計画

実験109では、SFT済みの50Mモデルへ、会話と医師国家試験を含む約20M tokensの追加事前学習を行った。FineWeb、会話、医療のvalidation lossは明確に改善した一方、一般会話と医療問題の固定評価ではEOS形式、返答の長さ、回答形式を大きく忘れた。これは追加データが無効だったという意味ではなく、raw next-token学習だけではSFTで獲得した応答形式を維持できないことを示している。

そこで今回は、実験109の最良checkpointへ、実験105〜108で使用した一般・医療の応答SFTを改めて8,000 step適用する。目的は、追加事前学習で改善した会話・医療の内部表現を残したまま、自然な返答形式と質問応答能力を回復できるかを確認することである。追加事前学習の後に十分なSFTを置く構成を、今後の標準候補にできるかを判断する。

### 仮説

実験109のraw validation loss改善は、会話と医療の語彙や文脈を追加データから学習できたことを示している。そのcheckpointに元の一般・医療SFTを重ねれば、実験105のSFT後checkpointよりも会話・医療の応答品質が高くなり、少なくともFineWeb lossの改善を一部維持できると予想する。ただし、追加事前学習で一般Web領域のlossが悪化しているため、一般会話の自然さが完全には戻らない可能性もある。

### 開始前の条件

- 実験番号：110
- 実施日：2026-09-07
- 担当：Codex
- 使用ブランチ：`main`
- 実行環境：Runpod Pod `j9c46julmtbcb4`、A40、PyTorch CUDA
- 初期checkpoint：実験109の最良checkpoint、step 9,500
- 初期checkpoint SHA-256：`0ceaeedef9d8ab8039861078bce673977a78a8bf7d329ff57096184b9f531cea`
- SFT train：`artifacts/sft/issue1-general-medical-concat-v1/train.npz`
- SFT train SHA-256：`598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- SFT validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`
- SFT validation SHA-256：`95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT設定：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 110、rehearsal ratio 0.2
- 設定ファイル：`configs/issue1-new-pretrain-general-medical-sft-runpod-8k.toml`
- 設定ファイルSHA-256：`dd75a98850863699d2befd162b279da874d59d46d3d55df601d27a3664f84c92`
- 学習コード：`scripts/train_sft_torch.py`
- 学習コードSHA-256：`bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`

周期checkpointは最新の1個だけ残すが、stepごとのmetrics、生成サンプル、評価結果、失敗ログは削除しない。大きな`.pt`本体はGitHubへ追加せず、最良checkpointのSHA-256と保存場所を記録する。生成された日本語は、良い結果だけでなく崩れた結果もすべてGitHubへ公開する。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \\
  --config configs/issue1-new-pretrain-general-medical-sft-runpod-8k.toml \\
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-new-japanese-20m-runpod-10k/best.pt \\
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \\
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \\
  --output-dir artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k \\
  --samples-dir artifacts/samples/issue1-new-pretrain-general-medical-sft-runpod-8k \\
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \\
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと生成サンプルを保存することを最低条件とする。性能面では、実験109で崩れた一般会話のEOSと返答長が回復し、医療評価で「正解は○です。」を抽出できる状態へ戻ることを重視する。実験105〜108の最良値と比べて、一般会話F1、医療正解率、領域別validation lossを比較する。すべての指標が同時に改善しなくても、raw追加事前学習の効果をSFT後まで残せたかを明確に切り分ける。

## 学習中の記録

ここに250 stepごとのvalidation loss、learning rate、elapsed time、GPUメモリ、固定prompt生成、警告、設定変更を追記する。実験が失敗した場合も、失敗した時点と原因を削除せずに残す。

### 2026-09-07：step 250

RunpodのA40上で、実験109のbest checkpointを初期値として学習を開始した。step 1のvalidation lossは3.183849、step 250は2.876897、step 250のlearning rateは1.9998e-5、経過時間は21.14秒だった。rehearsal ratioは0.2で、NaN、OOM、shape errorは発生していない。学習は継続中であり、step 250時点では生成品質の判定を保留する。

### 2026-09-07：step 500

step 500のvalidation lossは2.870976、perplexityは17.6542、learning rateは1.9935e-5、経過時間は41.47秒だった。学習中サンプルでは、`こんにちは！`に対して「こんにちは。今日はどこか行かれました?」と返しており、実験109のraw事前学習直後に失われた会話形式が早い段階で回復している。これは固定prompt一例に基づく暫定的な観察であり、一般48例と医療162例の最終評価で確認する。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,000

step 1,000のvalidation lossは2.875087、perplexityは17.7270、learning rateは1.9538e-5、経過時間は81.89秒だった。固定promptへの生成は「こんにちは!お願いします!」となり、会話の開始形式は維持されているが、step 500のサンプルより短い。学習中生成はtemperatureを含むため、この差だけで品質の悪化とは判断しない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 2,000

step 2,000のvalidation lossは2.846258、perplexityは17.2232、learning rateは1.7739e-5、経過時間は162.38秒だった。この時点でvalidation lossの最良値を更新した。固定promptへの生成は「こんにちは!宜しくお願いいたします。」となり、短いながらも話者markerの後に自然な挨拶を返している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,000

step 3,000のvalidation lossは2.832890、perplexityは16.9945、learning rateは1.4862e-5、経過時間は243.16秒だった。step 2,750の2.835384を経て最良値を更新しており、step 2,000以降もvalidation lossは緩やかに改善している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,000

step 4,000のvalidation lossは2.808299、perplexityは16.5817、learning rateは1.1366e-5、経過時間は324.20秒だった。ここまでの最良値を更新している。固定promptへの生成は「こんにちは!」に対して「こんにちは!」と短く返した。形式は崩れていないが、返答の長さはサンプリングにより揺れているため、最終の48例評価で平均長とEOS率を確認する。NaN、OOM、shape errorは発生していない。

## 実験終了後の記録

ここに実際の実行条件、最終loss、最良checkpointとstep、学習時間、生成評価、実験109および実験105との比較、予想との一致・不一致、次に試す変更を追記する。
