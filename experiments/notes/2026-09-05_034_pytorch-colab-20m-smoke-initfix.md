# 実験034：PyTorch/CUDA版20M smokeの初期化修正

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験033ではColab T4上のPyTorch学習が実行自体は完了したものの、step 1のlossが243.464452となり失敗しました。調査の結果、PyTorchのEmbedding既定初期化がMLX側の初期スケールと一致していないことが原因候補になりました。

今回の仮説は、Token embeddingとabsolute position embeddingを標準偏差`1/sqrt(dim)`の正規分布で初期化すれば、初期logitsとlossが通常の語彙サイズ4,096に対応する水準へ戻り、100 stepの学習と生成が安定するというものです。モデル構造、Token列、Tokenizer、学習率、batch size、seed、Colab GPUは実験033と固定します。出力先だけを分け、033の失敗成果物を保全します。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`で、Token数4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算19,382,016 parametersです。batch size 8、最大100 step、evaluation/sample interval 100、evaluation batches 20、learning rate 3e-4、minimum learning rate 3e-5、warmup 50、weight decay 0.1、seed 42です。設定は`configs/fineweb2-mixed-ja-20m-torch-smoke-v2.toml`です。PyTorch標準AdamW、CUDA float16 autocast、GradScalerを使います。

## 実行前の再現情報

初期化修正とこのノートをcommit・pushした後、実験033と同じbundle構成で、v2設定を含むbundleを作成します。Colabセッション`torch20m-smoke`は再利用し、古い出力先には書き込みません。bootstrapへv2設定を渡して100 stepを実行し、終了後にv2の軽量成果物を回収します。

成功基準は、step 1のlossが異常値でなく、100 stepがNaN、shape error、Token列不足、メモリ不足なしに完走し、metrics、metadata、生成TXT、summaryが保存されることです。step 1のlossの厳密なMLX一致は、初期化とoptimizerの完全な移植が済んでいないため要求しません。

## 実験中の記録

未実施です。

## 結果と解釈

未実施です。

## 次に試すこと

smokeが成功すれば、PyTorch標準AdamW版の2,500 step実行へ進む前に、MLXとのoptimizer差分を確認します。その後、Colab T4で実験030相当の長時間学習を実行し、MacBook MLXとの差を記録します。
