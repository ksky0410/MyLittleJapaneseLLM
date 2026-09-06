# 実験104：初回20Mと追加20Mを混ぜた継続事前学習

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：準備中、実行前
- 使用ブランチ：`main`
- 目的：教師LLMによる蒸留を使わず、日本語の知識と文章生成能力を増やす。前回の追加20M tokenだけの継続事前学習で会話・医療形式を忘れた原因が、データの偏りにあるかを確認する。

実験101では追加FineWeb2・Wikipediaの約20M tokenだけを40,000 step学習し、FineWeb validationは改善したが、会話lossが2.18から3.86へ、医療lossが2.02から2.27へ悪化した。今回は、実験098で使った初回約20M tokenと追加約20M tokenを約262k tokenずつ交互に並べ、片方のコーパスへ連続して偏らない40M token列を作る。

### 仮説

初回コーパスと追加コーパスを交互に混ぜると、追加コーパスだけを使うよりFineWeb validationの改善は小さくなるかもしれない。しかし、データ分布の急変を抑え、初回学習で得た日本語の分布を保ちやすくなると予想する。継続事前学習後にFineWeb、一般、会話、医療のlossを測り、会話形式が壊れていないかを確認する。

### 開始前の条件

- 初期checkpoint：実験101の継続事前学習最良checkpoint `artifacts/checkpoints/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5/best.pt`
- 初期checkpoint SHA-256：`6057172a5a2b3b420c5c751388eead0b17e0dfaa41585259265859ab9bf016b4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 初回token列：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-20m-v1-train.bin`、19,987,750 tokens、SHA-256 `707561109835bed5bccd8126527debeb86e86f0e6daaa9abea4ea78ba63e54e6`
- 追加token列：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin`、19,993,334 tokens、SHA-256 `f19878618870a487ce5b0aab6970d6d72b2ef71ab76ee79520e7c3fe3341dec1`
- 連結後token列：約39,981,084 tokens。生成後のmanifestへ正確なtoken数とSHA-256を記録する。
- validation：`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`、2,061,459 tokens、SHA-256 `36d8d5c8bc92de1e168b8c3de9dd4ee975dec66f6b644b83bfbf9b239877161c`
- tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- token列の交互chunk長：262,144 tokens
- 学習設定：`configs/issue1-both-50m-pretrain-balanced-40m-runpod-5k.toml`
- 乱数seed：104
- バッチサイズ：8、context 256
- 学習率：1e-5から1e-6までのcosine decay、warmup 250 steps
- 予定step：5,000 steps

### 成功判定

FineWeb validation lossが実験101の2.835023を維持または改善し、会話・医療lossが実験101の3.860727・2.266281より明確に低くなることを第一の成功条件とする。一般会話生成のEOS率と固定prompt生成も保存する。raw継続事前学習だけで会話能力が壊れる場合は失敗として記録し、SFT前提の二段階手順へ進む。

## 実験中の記録

データ生成、学習開始、500 step以上の節目ごとに、このノートへ実測値を追記する。

### 2026-09-07：データ生成とstep 500

交互連結データを生成し、出力は39,981,084 tokens、SHA-256は `fa8442332ab19f2134cd4ba1888ae1bc3cfbec02fd8aefba9ec73fbb63c855ff` となった。ローカルとRunpod上のSHA-256が一致することを確認した。

Runpod A40で実験101の最良checkpointから学習を開始した。step 1のvalidation lossは2.835003、step 500では2.835177だった。step 500時点では初期値からほぼ変化しておらず、追加データの優位性はまだ確認できない。step 500の学習時間は33.72秒、学習率は9.939e-6だった。NaNやOOMは発生していない。

step 1,000ではvalidation lossが2.834757まで下がり、初期値をわずかに下回った。経過時間は65.77秒、学習率は9.459e-6だった。現時点では、追加コーパスだけを使った実験101で観察された会話能力の急激な悪化はまだ確認していない。

step 1,500ではvalidation lossが2.833571まで下がった。経過時間は97.77秒、学習率は8.550e-6だった。固定promptの生成には未知語記号や反復が残っているが、これはraw継続事前学習の途中経過として保存し、見た目だけで採用判断しない。

step 2,000ではvalidation lossが2.832611まで下がった。経過時間は131.00秒、学習率は7.310e-6だった。初期値からの改善は約0.00239で、まだ小さいが一方向に改善している。

step 2,500ではvalidation lossが2.831113まで下がった。経過時間は164.33秒、学習率は5.875e-6だった。追加20M tokenだけで学習した実験101の最良FineWeb loss 2.835023を下回ったが、会話・医療領域の評価は本走終了後に行う。

step 3,000ではvalidation lossが2.830287まで下がった。経過時間は197.32秒、学習率は4.398e-6だった。現時点でもNaNやOOMは発生していない。

step 3,500ではvalidation lossが2.828992まで下がった。経過時間は230.65秒、学習率は3.041e-6だった。実験101の最良FineWeb loss 2.835023より約0.006低く、混合データによる改善が継続している。

step 4,000ではvalidation lossが2.829277となり、step 3,500の最良値からわずかに悪化した。経過時間は263.69秒、学習率は1.951e-6だった。過学習または評価の揺れを区別するため、最終stepまで続けつつ最良checkpointはstep単位で保持する。

step 4,500ではvalidation lossが2.828304まで下がった。経過時間は296.89秒、学習率は1.245e-6だった。step 3,500の値も更新し、実験101の2.835023を約0.00672下回った。

## 実験終了後の記録

### 2026-09-07：累積5,000 stepの本走と領域評価

累積5,000 stepまで学習し、最良checkpointはFineWeb validation lossが最も低かったstep 4,500となった。最良checkpoint SHA-256は `02ed0b3a01935aa4c1170dfa089e0282e38d4cb60d9d84dfc29526516f89c81e`、学習時間は331.79秒、ピークGPUメモリは約1.53GBだった。NaN、OOM、shape errorは発生しなかった。

同じ20 eval batchesで領域評価した結果、FineWeb lossは2.828304、一般lossは4.218692、会話lossは2.203025、医療lossは2.054636だった。FineWebは実験101の2.835023を下回り、追加データだけを使った実験101の会話3.860727、医療2.266281より大幅に良い。初回データと追加データを交互に混ぜることで、追加データによる知識側の改善と、初回データ分布の保持を同時に実現できる可能性が示された。一方、一般・会話・医療lossは実験102のSFT後よりまだ悪いため、この段階を会話モデルとして採用しない。

固定promptのraw生成は、初期値の反復から途中の未知語記号、短いEOS、問い合わせ文らしい出力へ変化した。崩れた途中生成も `artifacts/samples/issue1-both-50m-pretrain-balanced-40m-runpod-5k/` に残した。

このcheckpointを初期値に、実験102と同じ一般127,731例・医療2,945例の混合SFTを再実施する。これにより、balanced continued pretrainingが持つFineWeb改善を保ったまま、会話の終了と医療回答形式を回復できるかを検証する。医療QAを4倍にした実験103は正解率が継続stepで悪化したため、次のSFTではまず実験102の比率へ戻し、医療正解率・一般会話EOS・FineWeb lossを複合的に評価する。
