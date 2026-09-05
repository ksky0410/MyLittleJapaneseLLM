# 実験033：PyTorch/CUDA版20MモデルのColab smoke

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験030では、Apple Silicon上のMLXで約19.4M parameterの日本語GPTを2,500 step学習し、general validation loss 5.158014まで改善しました。実験032ではGoogle Colab CLI 0.6.0からTesla T4、PyTorch 2.11.0+cu128、CUDA 12.8を利用できることを確認しました。

今回の目的は、現在のMLX専用学習をPyTorch/CUDAへ移植し、Colabで同じ20M級モデルを実行できる最小経路を確立することです。今回の100 stepは移植のsmokeであり、MLXの本学習結果との性能比較や、ColabのGPU速度の結論には使いません。仮説は、同じdecoder-only構造、Tokenizer、Token列、batch size、learning rate、seedをPyTorchへ移しても、NaNやshape errorなしにCUDA学習・checkpoint・生成を実行できるというものです。

## 実装上の差分

`src/my_little_japanese_llm/torch_model.py`にPyTorch版を追加しました。MLX版と同じvocab embedding、pre-LayerNorm、qkv attention、GELU MLP、residual connection、final LayerNorm、入力Embeddingとのweight tyingを使います。causal attentionはPyTorchの`scaled_dot_product_attention`を使い、absoluteとRoPEの両方を実装しました。`load_mlx_weights`も用意し、将来はMLXの`.npz`をPyTorchへ読み込んでframework差を検証できます。

`scripts/train_torch.py`は既存TOML設定を読み、Token window samplingとlearning-rate scheduleを既存実装から共有します。CUDAではfloat16 autocastとGradScalerを使い、metrics JSONL、step metadata、生成TXT、summaryをMLX版と同じディレクトリ構成で保存します。PyTorch checkpoint本体は`.pt`としてGit対象外にし、metadataと生成TXTは追跡します。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`で、Token数4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、概算19,382,016 parametersです。batch size 8、最大100 step、evaluation/sample interval 100、evaluation batches 20、learning rate 3e-4、minimum learning rate 3e-5、warmup 50、weight decay 0.1、seed 42を使います。設定は`configs/fineweb2-mixed-ja-20m-torch-smoke.toml`です。

## 実行前の再現情報

実行前のコードcommitは、PyTorch版とこのノートをcommit・pushした時点のcommitとして記録します。Colabには次のローカルファイルをbundleに含めます。

```text
src/my_little_japanese_llm/
scripts/_common.py
scripts/train_torch.py
scripts/colab_bootstrap_train.py
configs/fineweb2-mixed-ja-20m-torch-smoke.toml
artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model
artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin
artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin
```

T4 sessionを`colab new --session torch20m-smoke --gpu T4`で作成し、bundleを`/content/small_llm_bundle.tar.gz`へuploadします。必要ならColab VMへ`numpy`と`sentencepiece`をinstallし、`colab exec -s torch20m-smoke -f scripts/colab_bootstrap_train.py`で学習を起動します。結果取得後は、軽量成果物と生成TXTをローカルへdownloadし、hashを確認してからcommit・pushし、必ず`colab stop`でセッションを解放します。

成功基準は、T4上でCUDAが認識され、100 stepがNaN、shape error、Token列不足、メモリ不足なしに完走し、metrics、checkpoint metadata、生成TXT、summaryが保存されることです。step 100のvalidation lossは移植後の参考値として記録しますが、初期化とfloat16計算がMLXと異なるため、MLX結果との良し悪し比較には使いません。

## 実験中の記録

未実施です。

## 結果と解釈

未実施です。

## 次に試すこと

smokeが成功した場合は、同じPyTorch版で実験030相当の2,500 stepをColab T4で実行し、MacBook MLXの学習時間と比較します。その後、L4が利用できる場合の速度、50Mモデル、Wikipedia追加コーパス、context length 512の順に、同時に一つの要因だけを変えて調べます。smokeが失敗した場合はエラーとリソース状況を記録し、PyTorch実装を修正してから再実験します。
