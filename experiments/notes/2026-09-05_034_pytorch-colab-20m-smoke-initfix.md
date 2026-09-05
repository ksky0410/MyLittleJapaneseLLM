# 実験034：PyTorch/CUDA版20M smokeの初期化修正

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験033ではColab T4上のPyTorch学習が実行自体は完了したものの、step 1のlossが243.464452となり失敗しました。調査の結果、PyTorchのEmbedding既定初期化がMLX側の初期スケールと一致していないことが原因候補になりました。

今回の仮説は、Token embeddingとabsolute position embeddingを標準偏差`1/sqrt(dim)`の正規分布で初期化すれば、初期logitsとlossが通常の語彙サイズ4,096に対応する水準へ戻り、100 stepの学習と生成が安定するというものです。モデル構造、Token列、Tokenizer、学習率、batch size、seed、Colab GPUは実験033と固定します。出力先だけを分け、033の失敗成果物を保全します。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`で、Token数4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算19,382,016 parametersです。batch size 8、最大100 step、evaluation/sample interval 100、evaluation batches 20、learning rate 3e-4、minimum learning rate 3e-5、warmup 50、weight decay 0.1、seed 42です。設定は`configs/fineweb2-mixed-ja-20m-torch-smoke-v2.toml`です。PyTorch標準AdamW、CUDA float16 autocast、GradScalerを使います。

## 実行前の再現情報

初期化修正とこのノートをcommit・pushした後、実験033と同じbundle構成で、v2設定を含むbundleを作成します。Colabセッション`torch20m-smoke`は再利用し、古い出力先には書き込みません。bootstrapへv2設定を渡して100 stepを実行し、終了後にv2の軽量成果物を回収します。

実行コードcommitは`f0713c6`です。v2 bundleは`/tmp/small_llm-colab-034-1788568607.tar.gz`として作成し、SHA-256は`ee9629b55f7fb7869ca8ddb481c91ee701c59599d5150d24dca36373b34b42ad`です。bundleには実験033と同じコード・入力hashのファイルに加え、`configs/fineweb2-mixed-ja-20m-torch-smoke-v2.toml`を含めます。

成功基準は、step 1のlossが異常値でなく、100 stepがNaN、shape error、Token列不足、メモリ不足なしに完走し、metrics、metadata、生成TXT、summaryが保存されることです。step 1のlossの厳密なMLX一致は、初期化とoptimizerの完全な移植が済んでいないため要求しません。

## 実験中の記録

未実施です。

## 結果と解釈

初期化修正後のsmokeは成功しました。Colab T4上でPyTorch 2.11.0+cu128、CUDA 12.8、Python 3.13.15を使い、float16 autocastとGradScalerを有効にして100 stepを完走しました。GPUはTesla T4、総メモリは15,360MiBです。step 1のtrain lossは8.684756、general validation lossは8.782655、perplexityは6,520.166、step 100のtrain lossは6.537301、validation lossは7.003381、perplexityは1,100.347でした。step 100までNaN、shape error、Token列不足、メモリ不足は発生していません。

学習時間はstep 100時点で4.269秒、summary上では5.289秒でした。実験033の同条件に近いPyTorch smokeはstep 100時点で5.225秒でしたが、初期化不良でlossが崩れていたため、速度だけを厳密比較しません。MLX版20M smokeのstep 100は28.479秒であり、今回のT4が大幅に高速な参考値は得られました。GPUの最大メモリ割当はこの学習runではmetadataへ記録していないため未計測です。実験032の行列積probeでの最大割当は75,628,544 bytesでしたが、学習時の値とは区別します。

生成はstep 0では未知の漢字列が続きましたが、step 100では「今日はかするがの、、:ススはにのりにのンのきいにの、...」のように日本語の文字・助詞・句読点を含む列へ変化しました。まだ文法と意味は崩れておりますが、実験033の「今日は」の完全反復は解消しました。悪い生成を含む全文は`artifacts/samples/fineweb2-mixed-ja-20m-torch-smoke-v2/`に保存します。

最良checkpointは`artifacts/checkpoints/fineweb2-mixed-ja-20m-torch-smoke-v2/step_000100.pt`で、ローカル保存サイズは約74MiB、SHA-256は`e69bcb1795287617331bc8e6851abe1d880ca12a949628ef11883cd4ac4d2902`です。軽量成果物のSHA-256は、metricsが`34a1545051f92cb23082b98eab3c911da230e6f8ad04dbde82f68a95478c85f4`、step 1 metadataが`80191cb9bd0c7cea61815611068c5387258b1f421a995ce966fdd918d3280cc1`、step 100 metadataが`b7795e60314a711ecdb69671fe6caf03804be1584310e2b9a3bfafb61c2f13cf`、summaryが`d7844d0782faa7440bfe0b8e4c3a7487c8d70a4263355a93b49e2fc34f082b93`、step 0生成が`97973fcf09682820190c97797d9ec96b81f70b301937fc9945f8aaf84f0fdabc`、step 100生成が`680e3f1ad4734406aba25e7bb1ae1417cd100089824aaa18f25bfa41fb08f9c7`です。Colabセッション`torch20m-smoke`は成果物回収後に停止し、確認時点でアクティブセッションはありません。

初期化修正の仮説は支持されました。ただし、PyTorch標準AdamWのbias correction、Embedding以外の初期化、CUDA float16計算がMLXと異なるため、MLXとの数値parityはまだ未確認です。次はfloat32・同一初期重み・AdamW更新式を揃えた小型parity testを行い、差分の原因を切り分けます。

## 次に試すこと

smokeが成功すれば、PyTorch標準AdamW版の2,500 step実行へ進む前に、MLXとのoptimizer差分を確認します。その後、Colab T4で実験030相当の長時間学習を実行し、MacBook MLXとの差を記録します。
