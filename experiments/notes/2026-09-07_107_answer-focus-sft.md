# 実験107：balanced SFT後の医療answer-focus追加SFT

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：準備中
- 使用ブランチ：`main`

実験105は一般会話の生成と各領域のvalidation lossを改善したが、医師国家試験162問の正解率は17.90%だった。実験106では医療データを4倍にして1,000 stepだけ追加学習し、正解率は19.14%まで上がったものの、一般会話の語句一致は悪化し、実験102の22.98%にも届かなかった。

今回は医療問題のデータ量をさらに大きく増やすのではなく、元の医療SFT例に加えて、同じ問題に対する短い回答例「正解は○です。」を一度だけ追加する。元の回答は理由を含むため、正解ラベル以外の説明トークンが多い。短い回答例で正解ラベルと終了位置への学習信号を強めれば、回答の先頭で正しい選択肢を出す能力が改善する可能性がある。

### 仮説

実験105の最良checkpointから、一般127,731例、元の医療2,945例、answer-focus医療2,945例を使って1,000 step学習する。医療の正解率は実験105の17.90%を上回り、実験106の19.14%に近いか、それ以上になると予想する。一般会話は実験105のEOS 48/48を維持し、平均生成長とF1の悪化を小さく抑えられると予想する。短い答えを追加しても医療の正答率が変わらない場合は、問題形式の暗記信号だけでは不十分で、選択肢を比較する学習データや評価方法を見直す必要がある。

### 開始前の条件

- 初期checkpoint：実験105の最良 `artifacts/checkpoints/issue1-balanced-pretrain-general-medical-sft-runpod-8k/best.pt`
- 初期checkpoint SHA-256：`1652603515b24e0538abeba01a63c53da1af4de87b51738b90acebe7326b9149`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 一般データ：実験102・105と同じ `artifacts/sft/issue1-general-medical-concat-v1/train.npz` 内の一般127,731例
- 元の医療データ：2,945例、元の理由付き回答を含む
- 追加医療データ：`artifacts/sft/issue1-medical-answer-focus-v1/train.npz`、実験102・105で採用した元の各問題に対して1例、回答本文は正解選択肢だけ
- 学習データ：一般、元医療、answer-focus医療を連結した `artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`
- validation：実験102・105・106と同じ `artifacts/sft/issue1-general-medical-concat-v1/validation.npz`
- rehearsal：`artifacts/mixed-ja-80-10-10-v2-train.bin`、比率20%
- 学習設定：`configs/issue1-balanced-pretrain-answer-focus-sft-runpod-1k.toml`
- 学習設定SHA-256：`409cd853ffa50ce3359eb6a048c9bab12c3cee5a0dfb84cb7e6efd5a12b30f74`
- 学習コード：`scripts/train_sft.py`、SHA-256 `1fb3c3c269fde247f25cf75e162f33e7e1dc3259b184770d86953f99016d2c22`
- 乱数seed：107
- 学習率：3e-6から3e-7までのcosine decay、warmup 100 steps
- 予定step：1,000 steps

実行前に、生成JSONL、NPZ、manifest、設定、checkpointの保存場所とSHA-256を記録する。検証用データにはanswer-focus例を追加せず、過去実験と同じ162問を使う。

最初のデータ生成では、医療問題全体をそのまま使うと、既存SFTが採用していない問題まで追加されることが分かった。そこで既存の `artifacts/corpus/medical-qb-sft-v1/{train,validation}.jsonl` にある問題番号を母集団として再生成する。元データ側の正解欄が空の問題は、既存処理と同様に除外し、manifestへ記録する。

再生成したanswer-focus会話はtrain 2,945例、validation 162例で、train JSONLのSHA-256は `8c702d252f78ce850da3d41f57bf174b0d943a30964f8f962956150eb50d7ea1` である。Tokenizerは実験102以降と同じ `mixed-ja-80-10-10-v2-unigram.model`（SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`）を使い、answer-focus train NPZは2,945例、応答本文18,577トークン、EOSを含む損失対象21,522トークン、SHA-256 `d15b59f3bc0a91b48145694b405b95fc954a76c60abc0c38ba6e49fbf4227f81`となった。

一般127,731例、元の医療2,945例、answer-focus医療2,945例を連結したtrain NPZは133,621例、損失対象1,736,042トークン、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262` である。初回に医療問題全体を使った生成では6,098例まで増えてしまったため採用せず、既存SFTと同じ問題番号に絞った生成へ修正した。この経緯は失敗として残す。

### 実行コマンド

Runpod Pod `j9c46julmtbcb4` のA40上で、次のコマンドを実行する。初期checkpoint、学習NPZ、validation NPZ、rehearsal token列はすべてRunpod上でSHA-256を照合してから開始する。

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft.py \
  --config configs/issue1-balanced-pretrain-answer-focus-sft-runpod-1k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-balanced-pretrain-general-medical-sft-runpod-8k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-balanced-pretrain-answer-focus-sft-runpod-1k \
  --samples-dir artifacts/samples/issue1-balanced-pretrain-answer-focus-sft-runpod-1k \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.2
```

### 成功判定

医療162問の正解率が実験105の17.90%を上回り、一般会話48例のEOS 48/48を維持することを第一条件とする。医療F1だけが上がって正解率が変わらない場合は、医療正答能力の改善とはみなさない。一般会話F1が大きく落ちる場合も総合モデルの採用候補にはしない。

## 実験中の記録

実験開始後、250 stepごとにvalidation loss、学習率、生成例、警告・停止の有無を追記する。

### 起動試行1：2026-09-07

Runpod上で予定コマンドを起動したが、`/workspace/exp100/scripts/train_sft.py` が存在せず、Pythonが開始直後に終了した。学習stepは0で、checkpointやデータへの変更はない。Runpodの作業ディレクトリには今回のデータだけを転送しており、実行コード本体の転送が不足していた。`scripts/train_sft.py` を追加転送し、import確認後に同じコマンドを再実行する。

## 実験終了後の記録

学習終了直後に、実際の条件、最終validation loss、最良checkpoint SHA-256、学習時間、ピークGPUメモリ、4領域loss、一般会話と医療会話の生成評価、162問の正解数を追記する。実験105・106と比較し、次に変える条件は一つか二つに絞る。
