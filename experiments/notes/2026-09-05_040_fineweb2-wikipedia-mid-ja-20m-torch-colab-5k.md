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

実験039を完了したcommitは`0a32af2`です。実験040の設定とノートを準備したcommitは`4a01fb1`で、040専用configを固定するwrapperを追加したcommitは`3ce40a2`、Colabの一時実行ディレクトリ問題を修正した実行コードcommitは`6515bac`です。config SHA-256は`93e2e1a970fb8c520aee259f19b72050cfc73f34fbb63f9eb35008bf65d9509a`です。実行コードは既存のPyTorch backend、`scripts/train_torch.py`、`scripts/colab_bootstrap_train.py`、`scripts/colab_bootstrap_040.py`を使用します。Colabへはコード、config、Tokenizer、Token列、general validationをbundleとして送り、学習後に軽量成果物とcheckpoint本体を回収します。

修正版送信用bundleは`/tmp/small_llm-colab-040.tar.gz`、SHA-256は`6b45fc705d46f7da88c210fc4d3c80cf7eb32effc51f5cac432f58589ae0584b`、サイズは9.6MBです。bundleには`src/my_little_japanese_llm`、`scripts/_common.py`、`scripts/train_torch.py`、`scripts/colab_bootstrap_train.py`、`scripts/colab_bootstrap_040.py`、設定、Tokenizer、7.5M Token列、general validation Token列を含めています。Colabにはこの修正版bundleをuploadします。

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

2026-09-05、名前付きT4 session `torch20m-wikipedia-mid-colab-5k`を作成し、bundleを`/content/small_llm_bundle.tar.gz`へuploadしました。最初のbundle hashは`5316852dee7950a94cfda5d494e9bae4b56b1609a3899f0bdccf49286d6ad103`でした。`colab exec --session torch20m-wikipedia-mid-colab-5k --timeout 1800 --file scripts/colab_bootstrap_040.py`を実行しましたが、Colabがwrapperを一時ディレクトリから評価するため、`from colab_bootstrap_train import main`で`ModuleNotFoundError`が発生しました。bundle展開、PyTorch初期化、Token列読み込み、学習stepには到達していません。この失敗は削除せず記録し、wrapperがbundleを先に展開してからimportする修正版へ更新します。

その後、wrapperをbundle展開後にimportする形へ修正し、修正版bundle hash `6b45fc705d46f7da88c210fc4d3c80cf7eb32effc51f5cac432f58589ae0584b`を使用して同じT4 sessionを再実行しました。修正版の学習では、初回失敗と混ざらないよう同じ実験名の出力ディレクトリを新規作成し、step 1から5,000までを完走させました。学習中はstep 100間隔でmetrics・checkpoint metadataを保存し、step 100間隔の生成文も省略せず出力しました。全checkpointのSHA-256一覧と軽量成果物archiveをColab側で生成し、最良checkpoint、metrics、summary、生成文を回収します。学習完了後、Colab sessionは停止し、`colab sessions`で稼働中セッションがないことを確認する予定です。

学習成果物を回収した後にPyTorch用の`evaluate_torch.py`を使い、同じcheckpointをgeneral、conversation、medical、FineWeb、Wikipediaの5 domainで評価し、既存の固定manifest `experiments/evaluation/chat-test-v1.json`にあるshort・medium・long各16例、合計48例のheld-out会話を生成評価します。MacBook側にはPyTorchがないため、評価もColab T4で行い、生成JSON/TXTと実行環境情報を回収します。評価用コードの追加とテストを完了した時点のGit commitは`a2ad887`、Colab評価wrapperを固定したcommitは`9aa4f7f`です。

2026-09-05、学習用T4 session停止後に評価専用T4 session `torch20m-wikipedia-mid-colab-eval`の作成を試みましたが、Colab CLIが`TooManyAssignmentsError`（HTTP 412 `Precondition Failed`）を返し、GPU割当前に停止しました。評価セッションは作成されておらず、入力データやcheckpointは変更していません。この失敗は削除せず、まずローカル仮想環境へのPyTorch追加が可能か確認し、可能ならCPU評価へ切り替えます。追加できない場合はColab割当上限が解消してから同じwrapperを再実行します。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。完了後は039の5M MLX結果と、同じPyTorch Colab backendの035を比較します。20Mモデルでsource比率の利点が確認できた場合は、Colab上で学習総Token数を増やし、RoPE、RMSNorm、SwiGLU、GQAを一つずつ導入します。結果が伸びない場合は、モデル容量よりdata量・Tokenizer・学習率scheduleを優先して調べます。
