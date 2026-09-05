# 実験041：中間Wikipedia比率7.5M Token列・約20Mモデルの10,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験040では、Wikipediaを約26.6%含む7.5M Token列を実測19,401,216 parameterのモデルへ5,000 step学習し、step 4,900でgeneral validation loss 4.924788、PPL 137.660を記録しました。一方、固定chat-test-v1のToken overlap F1は0.064172で、短い応答は生成できても長い履歴に沿った応答は安定しませんでした。

今回の仮説は、040と同じモデル・データ・Tokenizer・optimizer・seedを保ったまま学習stepだけを10,000へ延長すると、generalおよび各domainのlossがさらに下がり、生成文の一貫性も改善する可能性があるというものです。反対に、5,000 step以降でvalidation lossが横ばいまたは悪化し、chat F1も伸びない場合は、現状の制約は単純な学習不足ではなく、データ量、学習目標、会話形式、またはモデル構造にあると判断します。

## 使用するデータ、Tokenizer、モデル

学習Token列は040と同じ`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`です。Token数は7,499,997、SHA-256は`3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。実測source比率は青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%です。医師国家試験由来の元データは変更せず、既存の加工済みToken列を読み取り専用で使います。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、実測19,401,216 parametersです。

## 学習条件と再現情報

設定は`configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml`です。040から変えるのは最大step数と出力先だけで、batch size 8、evaluation/sample interval 100、evaluation batches 20、PyTorch標準AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42、CUDA float16 autocastとGradScalerを維持します。学習中の総Token exposureは、batch size 8とcontext length 256を掛けて、10,000 stepで約20.48M Tokenです。

開始前のGit commitは`473b816`です。設定ファイルのSHA-256は`3c3bd9a07a303d50f6ca17a9d29a02dd36f265671d9501f16596dc3532b90bab`です。学習bundleは`/tmp/small_llm-colab-041.tar.gz`、サイズ9.6MB、SHA-256は`872b9678c1e200d0feacf0483dade913980eb65d6654ff7f95c07fc4775c2a1f`です。ColabのPyTorch/CUDA/T4情報は、学習開始後に追記します。予定コマンドは次のとおりです。

```bash
colab new --session torch20m-wikipedia-mid-colab-10k --gpu T4
colab upload --session torch20m-wikipedia-mid-colab-10k <bundle> /content/small_llm_bundle.tar.gz
colab exec --session torch20m-wikipedia-mid-colab-10k --timeout 2400 --file scripts/colab_bootstrap_041.py
colab stop --session torch20m-wikipedia-mid-colab-10k
```

成功基準は、10,000 stepをNaN、OOM、shape error、Token列不足なく完走し、step 100間隔のmetrics・checkpoint metadata・生成文を回収できることです。失敗した場合も、発生したstep、エラー、回収できた成果物、次に確認することをこのノートへ残します。生成文はstep 0から最後のstepまで削除せずGitHubへpushします。

## 実験中の記録

2026-09-05、041専用の新規T4 session `torch20m-wikipedia-mid-colab-10k`の作成を試みましたが、Colab CLIがHTTP 412 `Precondition Failed`を返し、`TooManyAssignmentsError`で割当できませんでした。入力bundleやリポジトリは変更されていません。この失敗は、040の評価用session作成失敗と同様に削除せず記録します。

新規割当の代わりに`colab sessions`を再確認したところ、040で使用した既存session `torch20m-wikipedia-mid-colab-5k`がサーバー上に残っていることが分かりました。040の学習出力は別ディレクトリに保存済みで、041の出力先は`fineweb2-wikipedia-mid-ja-20m-torch-colab-10k`と分離されているため、同じT4 sessionへ041 bundleを上書きuploadして再利用します。再利用後は041の成果物回収を確認してから、明示的にsessionを停止し、もう一度`colab sessions`を確認します。

041 bundleのuploadは成功しましたが、実行要求から約20分経過しても、Colab側の041出力ディレクトリ、`metrics.jsonl`、step 1 checkpointはいずれも生成されませんでした。Colab CLIのログ取得もtimeoutし、既存kernelが実行要求を受け付けない状態と判断してローカルの実行待ちを中断しました。`colab download`で確認した際も`metrics.jsonl`と`summary.json`は存在せず、学習stepには到達していません。040の成果物を含む別ディレクトリは変更していません。sessionを停止し、停止後の`colab sessions`が空であることを確認しました。

その後の確認で、既存sessionのkernel再利用時に041ではなく040の学習処理が再実行され、040のmetrics、checkpoint metadata、生成文、評価結果が一時的に上書きされていたことが判明しました。これは041の成果物ではなく、040と同じ条件を再度走らせた想定外の再実行結果です。生成文と評価結果を含む全108ファイルを削除せず、`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-unexpected-rerun-before-041-stall/`、`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-unexpected-rerun-before-041-stall/`、および同名の`artifacts/evaluations/*unexpected-rerun-before-041-stall*`へ退避しました。040の正規記録はHEADの内容へ復元し、正規checkpointのSHA-256 `7f375a18adfbd55026711ad452320589296b0c3c399dd1887e354868c86e9667`と、元の最終validation loss 4.9294484456380205を再確認しました。041の結果としてこれらを扱わず、別の失敗・副産物として保存します。

## 結果と解釈

041の学習結果は未実施です。新規T4割当の失敗と、既存sessionを再利用した際のkernel応答停止という二つの失敗を削除せず記録しました。回収できた041の学習成果物はなく、metrics、checkpoint metadata、生成文は生成されていません。したがって、040とのlossや生成品質の比較もまだ行えません。

今回の失敗から、長時間実験を始める前に、sessionが新しいkernelとして実行可能かを短いprobeで確認し、probeの出力と最初のmetrics生成を検証する手順が必要だと分かりました。また、10,000 stepを100 step間隔で重み保存すると約100個、合計約7.7GBの`.pt`を書き込むため、次回の再実行ではmetricsと生成文の記録間隔を維持しつつ、重みcheckpointの保存間隔を別に制御できるようにして、Colab一時環境の保存負荷も抑えます。

## 次に試すこと

Colabの新規kernelが確保できたら、まずPython・PyTorch・T4のprobeを実行し、その後に041を再試行します。再試行前にcheckpoint保存間隔を設定から分離し、生成文とmetricsは100 step間隔、重み本体は必要な節目と最良checkpointだけを保存できるよう改善します。完走後は040との同一条件比較を行い、10,000 stepの延長効果が確認できなければ、データ量を増やす実験またはRoPE・RMSNorm・SwiGLU・GQAを一つずつ比較する実験へ進みます。
