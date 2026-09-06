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

### 2026-09-07：step 500

実験104の最良checkpointから一般・医療混合SFTを開始した。step 1のvalidation lossは3.702144、step 250は3.109951、step 500は3.100838まで下がった。step 500の経過時間は40.90秒、学習率は1.993e-5だった。NaNやOOMは発生していない。

step 750のvalidation lossは3.066517、step 1,000は3.063408だった。step 1,000の経過時間は81.46秒、学習率は1.954e-5だった。実験102のSFTより初期lossは高いが、学習に伴って安定して下がっている。

step 1,250のvalidation lossは3.051065、step 1,500は3.034272まで下がった。step 1,500の経過時間は122.08秒、学習率は1.880e-5だった。現時点ではmixed validationが継続して改善している。

step 1,750のvalidation lossは3.029610、step 2,000は3.016137まで下がった。step 2,000の経過時間は162.66秒、学習率は1.774e-5だった。初期値からの適応は順調に進んでいる。

step 2,250でvalidation lossは3.005957まで下がったが、step 2,500では3.018204へ一時的に悪化し、step 3,000で3.008171となった。step 3,000の経過時間は242.96秒、学習率は1.486e-5だった。SFT validationは揺れながら改善しているため、医療・会話の固定評価を途中checkpointでも比較する必要がある。

step 3,250でvalidation lossは2.985787、step 3,500では2.979102まで下がった。step 3,500の経過時間は283.11秒、学習率は1.316e-5だった。step 2,500付近の一時的な悪化から再び改善している。

step 3,750では2.984342へ小さく悪化したが、step 4,000で2.968162まで下がった。step 4,000の経過時間は323.22秒、学習率は1.137e-5だった。最良validation lossを更新した。

step 4,250のvalidation lossは2.969603、step 4,500は2.972311へ一時的に悪化した。その後step 4,750で2.961130、step 5,000で2.955211まで改善した。step 5,000の経過時間は403.53秒、学習率は7.812e-6だった。

## 実験終了後の記録

### 2026-09-07：累積8,000 stepの本走と最終評価

8,000 stepまで学習し、最良checkpointは最終step 8,000だった。混合SFT validation lossは2.919135、perplexityは18.5253、学習時間は646.58秒、ピークGPUメモリは約1.49GBだった。最良checkpointの重みSHA-256は `1652603515b24e0538abeba01a63c53da1af4de87b51738b90acebe7326b9149` である。NaN、OOM、shape errorは発生しなかった。

領域別lossはFineWeb 2.921111、一般4.084512、会話2.070219、医療1.977347だった。実験102と比較すると、FineWebは2.953670から改善、一般は4.089575から改善、会話は2.100814から改善、医療は1.987503から改善し、4領域のvalidation lossはすべて改善した。balanced継続事前学習の知識側の改善を、一般・医療SFT後にも一部維持できた。

一般会話48例ではEOS 48/48、平均生成11.33 tokens、token overlap F1 0.2086だった。実験102のEOS 48/48、平均7.90 tokens、F1 0.1892より改善し、返答が一語だけで終わる比率が下がった。医療162例ではEOS 153/162、平均生成56.10 tokens、F1 0.3745だった。回答形式は162/162例で抽出できたが、正解は29例、正解率17.90%であり、実験102の37/161、22.98%を下回った。

以上から、実験105は自然な一般会話と日本語コーパス由来のvalidation lossを改善する候補として採用する。ただし、医療QAの正答モデルとしては採用しない。医療の正解率はlossや語句一致だけでは改善しなかったため、次は医療QAだけをさらに重くするのではなく、選択肢の識別を直接測れる小さな評価を学習中に行い、短い追加SFTのcheckpointを比較する。教師LLMによる蒸留はまだ使わない。
