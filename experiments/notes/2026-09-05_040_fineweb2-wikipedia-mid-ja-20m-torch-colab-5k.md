# 実験040：中間Wikipedia比率7.5M Token列・約20MモデルのColab学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験039ではWikipediaを約26.6%含む7.5M Token列を5Mモデルへ5,000 step学習し、general validation loss 5.100070、Wikipedia test loss 4.362479、fixed chat F1 0.089495となりました。実験038のWikipediaなし5Mモデルと比べると、general lossはやや悪化しましたが、Wikipedia testとfixed chat F1を含むバランスは039が有望でした。

今回の目的は、039で選んだsource比率を約20M parameterモデルへ拡張し、モデル容量を増やすと日本語のlanguage modeling loss、Wikipedia適応、固定chatがどの程度改善するかを確認することです。モデルはdim 384、10層、6 heads、実測19,401,216 parametersとし、039と同じ7.5M Token列を5,000 step学習します。5Mモデルとの差では、Tokenizer、data、seed、batch size、schedule、optimizer backendをそろえ、変更をモデル容量に限定します。

学習はColab T4で実行します。T4はこのモデルを十分に載せられ、MacBook MLXで同規模を長時間学習するより短時間で実験できます。ただしColab側はPyTorch CUDA、float16 autocast、標準AdamWであり、MacBook MLXとはoptimizer実装と数値backendが異なります。そのため、039との厳密な因果比較ではなく、Colab PyTorch内の既存20M実験035との比較と、今後の20M級モデルの基準として扱います。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`、7,499,997 Token、SHA-256 `3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json`、実測Token比率は青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%です。元の医師国家試験データは変更せず、既存の加工済み本文とToken列を読み取り専用で使用します。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。検証Token列は039と同じgeneral validationを使用し、完了後にconversation、medical、FineWeb、Wikipediaのdomain評価とfixed chat-test-v1を実行します。

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embedding、実測19,401,216 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、PyTorch標準AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42、CUDA float16 autocastとGradScalerを使用します。設定は`configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k.toml`です。

## 実行前の再現情報

実験039を完了したcommitは`0a32af2`です。実験040の設定とノートを準備したcommitは`4a01fb1`で、040専用configを固定するwrapperを追加したcommitは`3ce40a2`、Colabの一時実行ディレクトリ問題を修正した実行コードcommitは`6515bac`です。config SHA-256は`93e2e1a970fb8c520aee259f19b72050cfc73f34fbb63f9eb35008bf65d9509a`です。実行コードは既存のPyTorch backend、`scripts/train_torch.py`、`scripts/colab_bootstrap_train.py`、`scripts/colab_bootstrap_040.py`を使用します。Colabへはコード、config、Tokenizer、Token列、general validationをbundleとして送り、学習後に軽量成果物とcheckpoint本体を回収します。

修正版送信用bundleは`/tmp/small_llm-colab-040.tar.gz`、SHA-256は`6b45fc705d46f7da88c210fc4d3c80cf7eb32effc51f5cac432f58589ae0584b`、サイズは9.6MBです。bundleには`src/my_little_japanese_llm`、`scripts/_common.py`、`scripts/train_torch.py`、`scripts/colab_bootstrap_train.py`、`scripts/colab_bootstrap_040.py`、設定、Tokenizer、7.5M Token列、general validation Token列を含めています。Colabにはこの修正版bundleをuploadします。

予定コマンドは次のとおりです。

```bash
colab new --session torch20m-wikipedia-mid-colab-5k --gpu T4
colab upload --session torch20m-wikipedia-mid-colab-5k <bundle> /content/small_llm_bundle.tar.gz
colab exec --session torch20m-wikipedia-mid-colab-5k --timeout 1800 --file scripts/colab_bootstrap_040.py
colab download --session torch20m-wikipedia-mid-colab-5k <remote-artifact> <local-artifact>
colab stop --session torch20m-wikipedia-mid-colab-5k
```

bootstrapは`/content/small_llm`へbundleを展開してから、`scripts/train_torch.py`を実行します。bundle展開失敗やColab CLIの引数問題が起きた場合も失敗として記録し、成果物を成功実験と混ぜません。成功基準は、5,000 stepがNaN、OOM、shape error、Token列不足なく完走し、100 step間隔のmetrics、checkpoint metadata、生成TXTを回収できることです。

## 実験中の記録

2026-09-05、名前付きT4 session `torch20m-wikipedia-mid-colab-5k`を作成し、bundleを`/content/small_llm_bundle.tar.gz`へuploadしました。最初のbundle hashは`5316852dee7950a94cfda5d494e9bae4b56b1609a3899f0bdccf49286d6ad103`でした。`colab exec --session torch20m-wikipedia-mid-colab-5k --timeout 1800 --file scripts/colab_bootstrap_040.py`を実行しましたが、Colabがwrapperを一時ディレクトリから評価するため、`from colab_bootstrap_train import main`で`ModuleNotFoundError`が発生しました。bundle展開、PyTorch初期化、Token列読み込み、学習stepには到達していません。この失敗は削除せず記録し、wrapperがbundleを先に展開してからimportする修正版へ更新します。

その後、wrapperをbundle展開後にimportする形へ修正し、修正版bundle hash `6b45fc705d46f7da88c210fc4d3c80cf7eb32effc51f5cac432f58589ae0584b`を使用して同じT4 sessionを再実行しました。修正版の学習では、初回失敗と混ざらないよう同じ実験名の出力ディレクトリへstep 1から5,000までを完走させました。学習中はstep 100間隔でmetrics・checkpoint metadataを保存し、step 100間隔の生成文も省略せず出力しました。Colab側では全checkpointのSHA-256一覧と軽量成果物archiveを生成し、最良checkpoint、metrics、summary、生成文を回収しました。学習後にsessionを停止し、`colab sessions`で稼働中セッションがないことを確認しました。

学習成果物を回収した後、PyTorch用の`evaluate_torch.py`を`a2ad887`で追加し、全テストを通過させました。Colab評価wrapperは`9aa4f7f`で固定しました。評価用T4の新規割当が上限に達したため、評価は後述のとおりMacBook CPUへ切り替えました。評価出力形式に全48例の集計を追加する小修正を`8e91593`で行い、同じcheckpoint・入力・seed・生成条件でchat評価を再実行しました。

2026-09-05、学習用T4 session停止後に評価専用T4 session `torch20m-wikipedia-mid-colab-eval`の作成を試みましたが、Colab CLIが`TooManyAssignmentsError`（HTTP 412 `Precondition Failed`）を返し、GPU割当前に停止しました。評価セッションは作成されておらず、入力データやcheckpointは変更していません。この失敗は削除せず、まずローカル仮想環境へのPyTorch追加が可能か確認し、可能ならCPU評価へ切り替えます。追加できない場合はColab割当上限が解消してから同じwrapperを再実行します。

その後、PyTorch 2.14.0 macOS arm64をこのリポジトリの`.venv`へ追加し、CPUでdomain評価とchat評価を実行しました。general domainの再計算値はColab学習中に記録されたvalidation lossと小数点以下5桁程度まで一致しました。評価時は`--device cpu --no-amp`を指定し、学習時のCUDA AMPとは分けて記録しました。

## 結果と解釈

修正版の学習はNaN、OOM、shape error、Token列不足を起こさず、step 5,000まで完走しました。summaryでは最良checkpointがstep 4,900、最良general validation lossが4.9247881571、PPLが137.6601762、最終stepのgeneral validation lossが4.9294484456、PPLが138.3032096でした。最終train lossは3.5336384773で、学習時間は214.05秒でした。step 4,900時点のmetricsに記録されたtrain lossは4.1143980026、学習経過時間は208.64秒です。最良値は最終stepで更新されなかったため、評価にはstep 4,900を使いました。

実行環境はTesla T4、PyTorch 2.11.0+cu128、CUDA 12.8、float16 autocast有効でした。GPUメモリの最大確保量は689,949,184 bytes、予約量は723,517,440 bytesでした。モデルの実測parameter数は19,401,216です。Colab上で保存されたstep 100間隔のmetadata、metrics、summary、生成文はすべて回収し、`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k/`と`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k/`へ保存しました。生成文はstep 0とstep 100からstep 5,000までの各ファイルを残しています。

Colab側の軽量archiveは104ファイルを含み、SHA-256は`15a74d93e480215632875cfb31ed37b7a81fa9a2513a70a805ab9d876c38aaab`でした。最良checkpointは77,642,279 bytesで、SHA-256は`7f375a18adfbd55026711ad452320589296b0c3c399dd1887e354868c86e9667`です。全51 checkpointのSHA-256は`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k/colab_checkpoint_manifest.json`に保存し、重み本体は最良stepだけをローカルの`step_004900.pt`へ保管しました。大きな`.pt`本体はGitHubへpushせず、metadataとhash一覧を追跡対象にしています。

step 4,900のdomain評価は、general loss 4.924783、PPL 137.659、conversation loss 3.028522、PPL 20.667、medical loss 3.235062、PPL 25.408、FineWeb loss 4.044750、PPL 57.097、Wikipedia loss 4.151952、PPL 63.558でした。generalの差はColab学習中の4.924788から約5e-6であり、CPU再評価が同じ重みを読めていることを確認しました。評価結果は`artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-domains.json`に保存しました。

固定chat-test-v1の48例では、short・medium・longを各16例評価しました。全体のToken overlap F1は0.064172、precisionは0.147898、recallは0.051309、平均生成長は6.31 Token、EOS停止は48例中48例でした。層別F1はshortが0.101389、mediumが0.036385、longが0.054743でした。評価例のうち33例はcontext length 256を超えており、7例はtrain本文との重複フラグがありました。この条件は意図した固定評価セットの性質としてJSONに残しています。生成内容には「こんにちは!」「いいですね!」「わかります。」のような短い応答が含まれる一方、「お気にとう」「じゃが!」「私はこの日も良いですが」のように文脈から外れる出力もあり、自然な会話能力が得られたとは判定しません。全48例の本文は`artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-chat-test-v1.txt`、詳細JSONは同名`.json`へ保存しました。

039の5M MLXモデルと比べると、general lossは5.100070から4.924783、conversationは3.147053から3.028522、medicalは3.497018から3.235062、FineWebは4.227962から4.044750、Wikipediaは4.362479から4.151952へ改善しました。ただしbackendがMLXからPyTorch CUDAへ変わっているため、これはモデル容量だけの因果比較ではありません。chat F1は039の0.089495から0.064172へ下がり、domain lossの改善がそのままheld-out会話の応答品質へつながらないことが分かりました。さらに、035の20M PyTorch結果は7.5M列ではなく5M列を2,500 step学習したため、040との差には学習Token数とstep数も含まれます。この比較からモデルサイズ単独の効果を断定しない方針です。

今回の仮説のうち、「20Mへ拡張するとlanguage modeling lossとWikipedia適応が改善する」は探索的には支持されましたが、「固定chatも改善する」は支持されませんでした。現在のモデルはpretrainingだけで会話の語彙・短い相づち・EOS停止をある程度出せるものの、長い履歴では文脈に沿う応答を維持できません。次の比較では、モデル構造をすぐ変える前に、同じ20M構造で学習Token数を増やし、データ量の効果を分離します。

## 次に試すこと

次は実験041として、同じ20M・同じ7.5M Token列・同じTokenizer・同じoptimizer設定を保ったまま、学習stepを10,000へ増やします。これにより040の5,000 step時点がまだ学習途中だったのか、単にデータが不足しているのかを確認します。Colab割当上限が戻らない場合は、今回追加したPyTorchを使ってMacBook CPUで小さい検証を行い、GPUが確保でき次第、本学習へ切り替えます。

その後、同じ学習Token予算とseedを固定して、RoPE、RMSNorm、SwiGLU、GQAを一つずつ導入します。各構造変更ではdomain lossだけでなく、EOS停止率、生成長、固定chatの層別F1、崩壊した生成文を比較し、複数の現代的手法を一度に混ぜないようにします。
