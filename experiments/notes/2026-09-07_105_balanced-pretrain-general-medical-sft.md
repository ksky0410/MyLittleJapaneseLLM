# 実験105：balanced継続事前学習後の一般・医療SFT

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：実行前
- 使用ブランチ：`main`
- 目的：実験104で得たFineWeb改善を保ちながら、一般会話の自然な終了と医療QAの回答形式をSFTで回復する。

実験104では、初回20Mと追加20Mを交互に混ぜた40M tokenで継続事前学習することで、FineWeb lossを2.828304まで下げ、追加20Mだけの場合の会話・医療忘却を抑えた。ただし、会話loss 2.203025、医療loss 2.054636はSFT後の実験102より悪い。実験102で効果が確認できた一般・医療混合SFTを同じ条件で戻し、pretrainingとSFTの役割を分ける。

### 仮説

balanced継続事前学習checkpointに一般会話127,731例と医療QA2,945例のSFTを行えば、実験104より会話・医療のloss、EOS、回答形式が改善する。FineWeb lossは少し悪化する可能性があるが、実験102の19.18程度に留まり、知識側の改善を一部維持すると予想する。医療QAの4倍反復は今回は使わない。実験103で継続stepにより正解率が悪化したため、まず実験102の比率を基準に戻す。

### 開始前の条件

- 初期checkpoint：実験104最良 `artifacts/checkpoints/issue1-both-50m-pretrain-balanced-40m-runpod-5k/best.pt`
- 初期checkpoint SHA-256：`02ed0b3a01935aa4c1170dfa089e0282e38d4cb60d9d84dfc29526516f89c81e`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習データ：`artifacts/sft/issue1-general-medical-concat-v1/train.npz`、一般127,731例・医療2,945例、応答token1,714,520、SHA-256 `598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、比率20%
- tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 学習設定：`configs/issue1-balanced-pretrain-general-medical-sft-runpod-8k.toml`
- 乱数seed：105
- バッチサイズ：8
- 学習率：2e-5から2e-6までのcosine decay、warmup 200 steps
- 予定step：8,000 steps

### 成功判定

実験104より会話・医療lossが改善し、一般会話48例のEOS 48/48と医療回答形式を維持することを第一条件とする。医療162例の正解率は実験102の22.98%を上回ることを目標とする。FineWeb lossは実験104の2.828304から悪化しても、SFT前の一般会話・医療能力と引き換えに許容できる範囲かを比較する。checkpointは混合validationだけでなく、最終的に固定会話・医療評価で選ぶ。

## 実験中の記録

未実施。250 stepごとのmetricsと生成文を保存し、少なくとも500 stepごとに追記する。

## 実験終了後の記録

未実施。学習時間、最良checkpoint、4領域loss、一般会話・医療のEOSと正解率、採用判断を追記する。
