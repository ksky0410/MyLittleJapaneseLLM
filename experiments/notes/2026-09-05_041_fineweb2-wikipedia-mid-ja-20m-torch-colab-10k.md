# 実験041：中間Wikipedia比率7.5M Token列・約20Mモデルの10,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験040では、Wikipediaを約26.6%含む7.5M Token列を実測19,401,216 parameterのモデルへ5,000 step学習し、step 4,900でgeneral validation loss 4.924788、PPL 137.660を記録しました。一方、固定chat-test-v1のToken overlap F1は0.064172で、短い応答は生成できても長い履歴に沿った応答は安定しませんでした。

今回の仮説は、040と同じモデル・データ・Tokenizer・optimizer・seedを保ったまま学習stepだけを10,000へ延長すると、generalおよび各domainのlossがさらに下がり、生成文の一貫性も改善する可能性があるというものです。反対に、5,000 step以降でvalidation lossが横ばいまたは悪化し、chat F1も伸びない場合は、現状の制約は単純な学習不足ではなく、データ量、学習目標、会話形式、またはモデル構造にあると判断します。

## 使用するデータ、Tokenizer、モデル

学習Token列は040と同じ`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`です。Token数は7,499,997、SHA-256は`3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。実測source比率は青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%です。医師国家試験由来の元データは変更せず、既存の加工済みToken列を読み取り専用で使います。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、実測19,401,216 parametersです。

## 学習条件と再現情報

設定は`configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml`です。当初の計画では040から最大step数と出力先だけを変え、それ以外のbatch size 8、evaluation/sample interval 100、evaluation batches 20、PyTorch標準AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42、CUDA float16 autocastとGradScalerを維持する予定でした。再試行用設定では、これらに加えてcheckpoint保存間隔だけを1,000 stepへ分離します。学習中の総Token exposureは、batch size 8とcontext length 256を掛けて、10,000 stepで約20.48M Tokenです。

計画を更新したGit commitは`7e4cc13`です。最初の失敗試行で用いた設定ファイルのSHA-256は`3c3bd9a07a303d50f6ca17a9d29a02dd36f265671d9501f16596dc3532b90bab`です。checkpoint保存間隔を分離したコードcommitは`22cafd6`、評価stepと実際のcheckpoint stepをmetadataで分けたcommitは`666d1f3`、新規kernelとbundleのhashを検証する再試行用commitは`5ff0463`です。再試行用設定ファイルのSHA-256は`8b79fac4389c8bc781dafe1076e4083d19ca9c81543f15bfddf414b33ec01d95`、bundleは`/tmp/small_llm-colab-041-5ff0463.tar.gz`、サイズ9.5MB、SHA-256は`27d776f394b73140e3aa901f7095add90eb0bf7ad7ebd6de10bddaa226e25b93`です。bundleからPythonのcacheは除外しています。ColabのPyTorch/CUDA/T4情報は、学習開始後に追記します。予定コマンドは次のとおりです。

```bash
colab new --session torch20m-wikipedia-mid-colab-10k --gpu T4
colab upload --session torch20m-wikipedia-mid-colab-10k <bundle> /content/small_llm_bundle.tar.gz
colab exec --session torch20m-wikipedia-mid-colab-10k --timeout 2400 --file scripts/colab_bootstrap_041.py
colab stop --session torch20m-wikipedia-mid-colab-10k
```

成功基準は、10,000 stepをNaN、OOM、shape error、Token列不足なく完走し、step 100間隔のmetrics・生成文と、step 1,000間隔のperiodic checkpoint metadata、およびvalidation loss最良時点のbest checkpointを回収できることです。20Mモデルの重みは約77MBあるため、学習中のcheckpoint保存を100 step間隔から1,000 step間隔へ分離し、生成文とmetricsの細かな記録は維持します。失敗した場合も、発生したstep、エラー、回収できた成果物、次に確認することをこのノートへ残します。生成文はstep 0から最後のstepまで削除せずGitHubへpushします。

## 実験中の記録

2026-09-05 12:15 JST、前回のsession再利用事故を避けるため、`torch20m-wikipedia-mid-colab-10k-r2`という新規sessionで`colab run --gpu T4 --keep scripts/colab_probe.py`を実行し、probe成功後に同じfresh kernelへbundleをuploadする計画を開始します。probeまたは新規GPU割当に失敗した場合は、学習を開始せず、CLI出力とsession状態をこのノートへ追記します。

12:15 JSTのfresh probeは、`colab run --session torch20m-wikipedia-mid-colab-10k-r2 --gpu T4 --keep --timeout 180 scripts/colab_probe.py`で実行しましたが、HTTP 412 `Precondition Failed`、`TooManyAssignmentsError`により割当できませんでした。コマンド終了後の`colab sessions`には、想定していなかった既存名`torch20m-wikipedia-mid-colab-10k`のT4 sessionが表示されたため、古いkernelを再利用しないよう直ちに`colab stop --session torch20m-wikipedia-mid-colab-10k`で停止し、その後`colab sessions`が空であることを確認しました。probeは実行されず、041の学習も開始しておりません。

12:16 JST、T4の割当制限がGPU種別固有かを切り分けるため、L4で同じprobeを一度だけ試す計画を追加しました。L4でも割当できない場合は、ColabのGPU割当待ちとして041を開始せず、MacBook側の次の実験準備へ戻ります。

12:17 JST、`colab run --session torch20m-wikipedia-mid-colab-10k-l4 --gpu L4 --keep --timeout 180 scripts/colab_probe.py`を実行しました。Colab backendは「L4 acceleratorはこのアカウントのquotaまたはentitlementでは利用できない」と拒否し、probeは実行されませんでした。直後の`colab sessions`は空でした。T4は割当制限、L4は利用権限不足と判断し、古いsessionの再利用やCPU上での041実行は行いません。

2026-09-05、041専用の新規T4 session `torch20m-wikipedia-mid-colab-10k`の作成を試みましたが、Colab CLIがHTTP 412 `Precondition Failed`を返し、`TooManyAssignmentsError`で割当できませんでした。入力bundleやリポジトリは変更されていません。この失敗は、040の評価用session作成失敗と同様に削除せず記録します。

新規割当の代わりに`colab sessions`を再確認したところ、040で使用した既存session `torch20m-wikipedia-mid-colab-5k`がサーバー上に残っていることが分かりました。040の学習出力は別ディレクトリに保存済みで、041の出力先は`fineweb2-wikipedia-mid-ja-20m-torch-colab-10k`と分離されているため、同じT4 sessionへ041 bundleを上書きuploadして再利用します。再利用後は041の成果物回収を確認してから、明示的にsessionを停止し、もう一度`colab sessions`を確認します。

041の長時間実験に向け、`training.checkpoint_interval`を追加しました。`eval_interval`と`sample_interval`は100のまま維持し、checkpointだけを1,000 step間隔で保存します。validation lossが更新された場合は`best.pt`を保存し、periodic checkpointとbest checkpointの役割をmetadataへ記録します。既存configでこの項目を省略した場合は従来どおりevaluation intervalをcheckpoint間隔として扱う後方互換実装です。変更は実験開始前にテストし、commitへ固定します。

2026-09-05 13:12 JST、新規T4 session `torch20m-wikipedia-mid-colab-10k-r3`のfresh probeに成功しました。Colab側はPython 3.13.15、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、15,360 MiB、compute capability 7.5でした。probeのmatmul 10回は0.185408626秒で、最大GPUメモリ使用量は75,628,544 bytesでした。sessionには安全bundle `/content/small_llm_bundle.tar.gz`をアップロードし、bootstrapは6項目のSHA-256を検証してから学習を起動しました。bundleのSHA-256は`27d776f394b73140e3aa901f7095add90eb0bf7ad7ebd6de10bddaa226e25b93`です。

bootstrapは検証用展開先`/content/small_llm_041`を使った後、`colab_bootstrap_train.py`の既定実行先である`/content/small_llm`へ学習用bundleを展開しました。そのため、学習成果物は`/content/small_llm/artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/`と`/content/small_llm/artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/`に生成されています。これは041の想定出力先と一致し、古いsessionやローカル出力先との競合はありません。

13:17 JST時点で、fresh T4学習はstep 5,500まで到達し、metrics 56件、step 0から5,500までの生成文、periodic checkpointはstep 1,000、2,000、3,000、4,000に生成されています。途中最新のstep 5,500はtrain loss 3.9399766922、general validation loss 4.7700667381、perplexity 117.9271119328、学習経過232.32秒です。この途中時点での最良validation lossもstep 5,500の4.7700667381です。回収した途中metricsは`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/metrics-colab-partial.jsonl`、step 4,700の生成文は`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/step_004700-colab-partial.txt`へ保存しました。学習はまだ継続中であり、これらは最終結果ではありません。

13:18 JSTごろ、追加回収を行ったところ学習はstep 8,500まで進んでいました。最新のstep 8,500はtrain loss 3.2670338154、general validation loss 4.5667727788、perplexity 96.2330427888、経過363.49秒でした。途中の最良validation lossもstep 8,500の4.5667727788です。回収時点のmetrics全体は`metrics-colab-partial-8000.jsonl`および同内容をstep 8,500時点の記録として保存した`metrics-colab-partial-8500.jsonl`に残し、SHA-256はどちらも`edd3f402fa93d909e77a88972bf17d918ab4a5c90f8dc60a074e9c1ad8f2de86`です。step 8,000の生成文は`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/step_008000-colab-partial.txt`、SHA-256は`79bb39cf1cc5a31dd84c84d173113eb07f6c410bc5c41d7098fd75c8c8e2aef0`です。periodic checkpointはstep 8,000まで存在し、学習はまだ継続中です。

再試行用コードでは、評価stepとcheckpoint保存stepが一致しない場合に備え、metadataの`checkpoint_step`を独立して保存し、評価スクリプトもこの値を優先して読むようにしました。ローカルの小型CPU統合確認では、evaluationがstep 1・3・6・7、periodic checkpointがstep 5・7、生成文がstep 0・4・7となり、step 5 metadataの保存step 5と直近validation step 3も区別できました。全テストは63件成功しました。

041 bundleのuploadは成功しましたが、実行要求から約20分経過しても、Colab側の041出力ディレクトリ、`metrics.jsonl`、step 1 checkpointはいずれも生成されませんでした。Colab CLIのログ取得もtimeoutし、既存kernelが実行要求を受け付けない状態と判断してローカルの実行待ちを中断しました。`colab download`で確認した際も`metrics.jsonl`と`summary.json`は存在せず、学習stepには到達していません。040の成果物を含む別ディレクトリは変更していません。sessionを停止し、停止後の`colab sessions`が空であることを確認しました。

その後の確認で、既存sessionのkernel再利用時に041ではなく040の学習処理が再実行され、040のmetrics、checkpoint metadata、生成文、評価結果が一時的に上書きされていたことが判明しました。これは041の成果物ではなく、040と同じ条件を再度走らせた想定外の再実行結果です。生成文と評価結果を含む全108ファイルを削除せず、`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-unexpected-rerun-before-041-stall/`、`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-unexpected-rerun-before-041-stall/`、および同名の`artifacts/evaluations/*unexpected-rerun-before-041-stall*`へ退避しました。040の正規記録はHEADの内容へ復元し、正規checkpointのSHA-256 `7f375a18adfbd55026711ad452320589296b0c3c399dd1887e354868c86e9667`と、元の最終validation loss 4.9294484456380205を再確認しました。041の結果としてこれらを扱わず、別の失敗・副産物として保存します。

## 結果と解釈

前半に記録した新規T4割当の失敗、既存sessionを再利用した際のkernel応答停止、L4 acceleratorの権限不足は、041の失敗履歴としてそのまま保持します。その後、2026-09-05 13:12 JSTに新規T4 session `torch20m-wikipedia-mid-colab-10k-r3`のfresh probeが成功し、041の再試行は10,000 stepまで完走しました。失敗履歴を成功結果で上書きせず、今回の完走結果を以下に記録します。

学習はstep 10,000で終了し、実測パラメータ数は19,401,216、経過時間は428.80秒でした。評価は100 stepごとに101件、生成文はstep 0からstep 10,000まで100 stepごとに保存され、periodic checkpoint metadataは1,000 stepごとに保存されました。validation lossは学習中に単調に改善し、最良checkpointは最終stepの`best.pt`、最良validation lossは4.5294143359、perplexityは92.7042516268でした。最終stepのtrain lossは3.5883781900、学習率は3.0000007080e-05でした。Colab側の実行環境はPython 3.13.15、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、15,360 MiBで、AMPを有効にしたCUDA学習です。重み本体はローカルの`best.pt`へ回収し、SHA-256 `f554bb5b6b2b1fe20b9318d1558465c0bb4407ddfa971593b853dc7f2aab868e`がColab上のマニフェストおよび`best.json`と一致しました。periodic weights全11個のサイズとSHA-256は、`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/colab_checkpoint_manifest.json`に記録しています。軽量成果物114ファイルのColabアーカイブSHA-256は`a6d41fc25a8cf3f1936d1852a62bf538f358d4b84626729628c3d4116253a291`です。

固定prompt `今日は`の生成は、step 0では「唆呵晴蚊…」という文字列の崩れが目立ちました。step 5,000では「今日は日本の漫画家としてもらう漫画家…」のように日本語らしい断片とWikipedia風の固有表現が現れ、step 10,000では「今日は Inc Internet Dentebe…」のように英数字を含む技術文書風の出力へ変化しました。日本語の文字列や頻出表現は学習されましたが、短い自然な物語を安定して生成できたとは言えません。生成結果は品質にかかわらず、`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/`へ全件保存しています。

回収した最良重みをローカルCPUで再評価したところ、general validation lossは4.5293409030（PPL 92.6974）、conversationは2.7032114665（PPL 14.9276）、medicalは2.6418840090（PPL 14.0396）でした。Colabの評価値とローカルCPUのgeneral値には評価バッチの乱数位置による小さな差があります。held-out chat-test-v1の48例では、EOS到達率48/48、平均生成Token数8.54、Token overlap F1は0.086434でした。040の5,000 step時点のgeneral loss 4.924788からは改善しましたが、chat F1の改善だけで会話能力が伸びたとは判断しません。今回のデータは一般Web 53.282%、Wikipedia 26.632%、会話6.651%、医療6.657%、青空文庫6.777%を混ぜた事前学習列であり、medical専用モデルではありません。なお、ローカル評価開始時にはシステムPythonに`sentencepiece`が見つからず一度停止しましたが、リポジトリの`.venv`には依存が導入済みだったため、環境を変更せず`.venv/bin/python`で再実行しました。

事前の予想どおり、学習stepを5,000から10,000へ延長するとgeneral lossは改善しました。一方、生成文は日本語らしい断片を含むものの、Wikipedia由来の文体や英数字の混入があり、固定chat評価も短いEOS出力に偏っています。したがって、今回の結果は「学習不足の一部は解消したが、会話能力や自然な生成には、より多いToken予算だけでなくデータ形式・モデル構造・instruction tuningの検討が必要」と解釈します。

今回の失敗から、長時間実験を始める前に、sessionが新しいkernelとして実行可能かを短いprobeで確認し、probeの出力と最初のmetrics生成を検証する手順が必要だと分かりました。また、10,000 stepを100 step間隔で重み保存すると約100個、合計約7.7GBの`.pt`を書き込むため、再試行用コードではmetricsと生成文の記録間隔を維持しつつ、重みcheckpointを1,000 step間隔と最良重み`best.pt`へ分離しました。保存stepと直近のvalidation stepもmetadataで別々に記録します。

## 次に試すこと

041の完走と成果物回収は完了しました。次は、今回のLayerNorm + GELU + absolute position構成を比較対象として固定し、同じ20M規模でLayerNorm + SwiGLU + RoPEを導入した構成を、まず短いsmoke実験で検証します。その後、ColabのT4が利用可能な場合は、今回と同じ7.5M Token列を使って10,000 stepへ延長し、構造変更と学習stepの効果を分離して比較します。結果が安定すれば、Wikipedia比率を保ったままToken予算を増やす実験、50M級への拡大、会話・医療データを混ぜたinstruction tuningへ進みます。今回と同じく、生成文は良否を問わず全ステップ保存し、開始前・途中・終了直後のノート更新とGitHub pushを継続します。
