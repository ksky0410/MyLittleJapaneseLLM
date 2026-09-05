# 実験040：中間Wikipedia比率7.5M Token列・約20MモデルのColab学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験039ではWikipediaを約26.6%含む7.5M Token列を5Mモデルへ5,000 step学習し、general validation loss 5.100070、Wikipedia test loss 4.362479、fixed chat F1 0.089495となりました。実験038のWikipediaなし5Mモデルと比べると、general lossはやや悪化しましたが、Wikipedia testとfixed chat F1を含むバランスは039が有望でした。

今回の目的は、039で選んだsource比率を約20M parameterモデルへ拡張し、モデル容量を増やすと日本語のlanguage modeling loss、Wikipedia適応、固定chatがどの程度改善するかを確認することです。モデルはdim 384、10層、6 heads、概算19,382,016 parametersとし、039と同じ7.5M Token列を5,000 step学習します。5Mモデルとの差では、Tokenizer、data、seed、batch size、schedule、optimizer backendをそろえ、変更をモデル容量に限定します。

学習はColab T4で実行します。T4はこのモデルを十分に載せられ、MacBook MLXで同規模を長時間学習するより短時間で実験できます。ただしColab側はPyTorch CUDA、float16 autocast、標準AdamWであり、MacBook MLXとはoptimizer実装と数値backendが異なります。そのため、039との厳密な因果比較ではなく、Colab PyTorch内の既存20M実験035との比較と、今後の20M級モデルの基準として扱います。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`、7,499,997 Token、SHA-256 `3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json`、実測Token比率は青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%です。元の医師国家試験データは変更せず、既存の加工済み本文とToken列を読み取り専用で使用します。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。検証Token列は039と同じgeneral validationを使用し、完了後にconversation、medical、FineWeb、Wikipediaのdomain評価とfixed chat-test-v1を実行します。

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算19,382,016 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、PyTorch標準AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42、CUDA float16 autocastとGradScalerを使用します。設定は`configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k.toml`です。

## 実行前の再現情報

実験039を完了したcommitは`0a32af2`です。実験040のconfig SHA-256は`93e2e1a970fb8c520aee259f19b72050cfc73f34fbb63f9eb35008bf65d9509a`です。実行コードは既存のPyTorch backend、`scripts/train_torch.py`、`scripts/colab_bootstrap_train.py`を使用し、コード変更は行いません。Colabへはコード、config、Tokenizer、Token列、general validationをbundleとして送り、学習後に軽量成果物とcheckpoint本体を回収します。

予定コマンドは次のとおりです。

```bash
colab new --session torch20m-wikipedia-mid-colab-5k --gpu T4
colab upload --session torch20m-wikipedia-mid-colab-5k <bundle> /content/small_llm_bundle.tar.gz
colab exec --session torch20m-wikipedia-mid-colab-5k --timeout 1800 --file scripts/colab_bootstrap_train.py
colab download --session torch20m-wikipedia-mid-colab-5k <remote-artifact> <local-artifact>
colab stop --session torch20m-wikipedia-mid-colab-5k
```

bootstrapは`/content/small_llm`へbundleを展開してから、`scripts/train_torch.py`を実行します。bundle展開失敗やColab CLIの引数問題が起きた場合も失敗として記録し、成果物を成功実験と混ぜません。成功基準は、5,000 stepがNaN、OOM、shape error、Token列不足なく完走し、100 step間隔のmetrics、checkpoint metadata、生成TXTを回収できることです。

## 実験中の記録

未実施です。Colab session作成、bundle hash、GPU情報、学習ログ、回収状況、停止状況を節目ごとに追記します。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。完了後は039の5M MLX結果と、同じPyTorch Colab backendの035を比較します。20Mモデルでsource比率の利点が確認できた場合は、Colab上で学習総Token数を増やし、RoPE、RMSNorm、SwiGLU、GQAを一つずつ導入します。結果が伸びない場合は、モデル容量よりdata量・Tokenizer・学習率scheduleを優先して調べます。
