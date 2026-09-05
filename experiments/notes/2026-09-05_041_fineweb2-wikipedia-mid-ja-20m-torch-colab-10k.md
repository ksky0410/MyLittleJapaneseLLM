# 実験041：中間Wikipedia比率7.5M Token列・約20Mモデルの10,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験040では、Wikipediaを約26.6%含む7.5M Token列を実測19,401,216 parameterのモデルへ5,000 step学習し、step 4,900でgeneral validation loss 4.924788、PPL 137.660を記録しました。一方、固定chat-test-v1のToken overlap F1は0.064172で、短い応答は生成できても長い履歴に沿った応答は安定しませんでした。

今回の仮説は、040と同じモデル・データ・Tokenizer・optimizer・seedを保ったまま学習stepだけを10,000へ延長すると、generalおよび各domainのlossがさらに下がり、生成文の一貫性も改善する可能性があるというものです。反対に、5,000 step以降でvalidation lossが横ばいまたは悪化し、chat F1も伸びない場合は、現状の制約は単純な学習不足ではなく、データ量、学習目標、会話形式、またはモデル構造にあると判断します。

## 使用するデータ、Tokenizer、モデル

学習Token列は040と同じ`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`です。Token数は7,499,997、SHA-256は`3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。実測source比率は青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%です。医師国家試験由来の元データは変更せず、既存の加工済みToken列を読み取り専用で使います。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、実測19,401,216 parametersです。

## 学習条件と再現情報

設定は`configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml`です。040から変えるのは最大step数と出力先だけで、batch size 8、evaluation/sample interval 100、evaluation batches 20、PyTorch標準AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42、CUDA float16 autocastとGradScalerを維持します。学習中の総Token exposureは、batch size 8とcontext length 256を掛けて、10,000 stepで約20.48M Tokenです。

開始前のGit commitは`473b816`です。設定ファイルのSHA-256は`3c3bd9a07a303d50f6ca17a9d29a02dd36f265671d9501f16596dc3532b90bab`です。実際に使用したbundleのSHA-256、ColabのPyTorch/CUDA/T4情報は、学習開始後に追記します。予定コマンドは次のとおりです。

```bash
colab new --session torch20m-wikipedia-mid-colab-10k --gpu T4
colab upload --session torch20m-wikipedia-mid-colab-10k <bundle> /content/small_llm_bundle.tar.gz
colab exec --session torch20m-wikipedia-mid-colab-10k --timeout 2400 --file scripts/colab_bootstrap_041.py
colab stop --session torch20m-wikipedia-mid-colab-10k
```

成功基準は、10,000 stepをNaN、OOM、shape error、Token列不足なく完走し、step 100間隔のmetrics・checkpoint metadata・生成文を回収できることです。失敗した場合も、発生したstep、エラー、回収できた成果物、次に確認することをこのノートへ残します。生成文はstep 0から最後のstepまで削除せずGitHubへpushします。

## 実験中の記録

未開始です。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。完了後は040との同一条件比較を行い、10,000 stepの延長効果が確認できなければ、データ量を増やす実験またはRoPE・RMSNorm・SwiGLU・GQAを一つずつ比較する実験へ進みます。
