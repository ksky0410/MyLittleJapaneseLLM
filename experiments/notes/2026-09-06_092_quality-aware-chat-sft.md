# 実験092：品質を考慮して選別したIssue #1会話データのSFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の会話コーパスを、単純な無作為subsetではなく、会話の機能を考慮して選別したデータへ置き換えます。今回の目的は、強いLLMから蒸留せず、同じ50Mモデルと同じ学習量のまま、自然な日本語会話を改善できるかを確かめることです。

086ではRPC・MRMPを多様化して同量ずつ学習しましたが、held-out chat F1は0.203292で、固定promptは挨拶へ縮退しました。087では24 token以上の長い応答を25%へ増やし、held-out chat F1は0.216545へ改善した一方、validation lossは086より悪化しました。092では、長い応答を一律に増やすのではなく、質問履歴を増やし、定型挨拶と初回発話を抑えることで、話題への応答と文脈継続を改善できるかを検証します。

学習条件は086・087と揃えます。50M baseは同じcheckpoint、SFTは10,000 step、batch sizeは8、学習率は5e-5から5e-6のcosine decay、seedは123、rehearsal ratioは0.20、EOS loss weightは0.50です。変更するのはSFT train dataだけです。validation、held-out chat評価、固定prompt、生成条件も揃えます。

成功条件は、086・087よりもshort・medium・longの複数層、または全体F1を改善し、一般日本語のvalidation lossを大きく悪化させないことです。固定promptで挨拶以外の応答や、質問に対応した内容が増えることも確認します。生成が長くなっただけ、lossだけが改善した場合は成功としません。

## 再現条件

092用の設定は`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware.toml`、Colab実行は`scripts/colab_bootstrap_092.py`、bundle結合は`scripts/colab_join_092_bundle.py`、成果物回収は`scripts/colab_package_092.py`です。実験開始前に、設定・コード・base checkpoint・validation・rehearsal token列・Tokenizer・品質選別NPZ・manifestのSHA-256を記録します。

準備コミットは`5a85b42`、設定のSHA-256は`91d18a72717c863fc32a9bd0ee2cc6a66038a8c8751999af466a3b6f34f3cd8e`、bootstrapは`b60b19a608b60655d87014c727f5304d57dc30b1f8af694d52084db18f67891f`、bundle結合スクリプトは`92f23549d27bb1dab94171a086c3a07e5a407d6fcad917d7b8b8b65545259b64`、成果物回収スクリプトは`f03b4a18a7e22f88bf7fbdce24ab13a6a868b1abc9084de20f0e0709849f1272`です。bundleは265,098,998 bytes、SHA-256は`d3091b93d3b9c3318a4a6d35f97a3b73ab01bd5a3a19ac416dda54b33cdb215f`で、ColabのHTTP upload制限を避けるため60 MiB以下の分割片として送ります。

品質選別データは`artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz`、manifestは同じディレクトリの`manifest.json`です。response token数はRPC 771,000、MRMP 770,975で、全体は1,541,975 tokenです。元の会話JSONLと`medilink_analysis`内の原本は変更しません。

## 学習前の記録

092の学習はまだ開始していません。設定と実行スクリプトを追加し、入力検証を通過するbundleを作成しました。Colab APIがHTTP 503を返した場合は、学習未実施のままエラーと時刻を追記し、同じ条件で再試行できる状態を保ちます。

## 実験中の記録

学習を開始できた場合は、100 stepごとのmetricsと固定生成、少なくとも1,000 stepごとの解釈を追記します。OOM、NaN、shape error、途中停止、生成の挨拶偏重も削除せず記録します。

2026-09-06 14:03 JST、`colab sessions`で既存セッションがないことを確認した後、`colab new --session exp092-quality-aware-sft --gpu T4`を実行しました。Colab APIのassignmentがHTTP 503 `Service Unavailable`で失敗し、GPUセッションは作成されませんでした。したがって、092の学習step、loss、生成結果はまだありません。bundleは作成済みで、再試行時には同じ分割片と同じSHA-256を使います。

前景MPSセッションではstep 1から学習が始まり、step 100のvalidation lossは3.511052、step 200は3.542971、step 300は3.541620、step 400は3.514441、step 500は3.519792でした。step 500のperplexityは33.777411、経過時間は416.18秒、learning rateは4.98199e-5でした。step 0から500までのmetrics、100 stepごとの生成、step 500のcheckpoint metadataを保存しています。固定promptのstep 500生成は「こんにちはー!」で、短い挨拶への縮退が続いています。一般validation lossは初期に大きく下がった後、step 100以降は横ばいであり、会話自然さの改善はまだ判断できません。

step 600はvalidation loss 3.487917、step 700は3.479720、step 800は3.472389、step 900は3.464336、step 1,000は3.464401でした。最良checkpointはstep 900、validation perplexity 31.955221、経過時間851.21秒です。step 1,000の固定prompt生成は「こんばんは!」となり、step 500の「こんにちはー!」から表面上の変化はありますが、まだ一往復の定型挨拶だけで、話題への応答や長い自然文は確認できません。生成本文はstep 0から1,000まで保存しています。現時点ではvalidation lossの改善を自然な会話性能の改善とは解釈せず、学習を継続します。

step 1,100ではvalidation loss 3.469782、step 1,200では3.488342へ一時的に悪化しましたが、step 1,300で3.433376、step 1,400で3.441852、step 1,500で3.427367まで改善しました。step 1,500のperplexityは30.795437、経過時間は1,546.04秒です。最良checkpointはstep 1,500へ更新されました。固定promptのstep 1,500生成は「こんにちは」で終了し、validation改善に対して会話生成の多様性はまだ増えていません。validation lossと固定promptの自然さが一致しない例として記録し、held-out chat評価まで結論を保留します。

step 1,600はvalidation loss 3.417355、step 1,700は3.420810、step 1,800は3.421289、step 1,900は3.378712、step 2,000は3.377987でした。step 2,000のperplexityは29.311707、経過時間は2,174.01秒で、最良checkpointはstep 2,000へ更新されました。固定promptのstep 2,000生成は「こんにちは!」で、依然として一往復の挨拶から広がっていません。一般validation lossは改善しているため学習を止めませんが、会話性能の成功判定には固定Issue #1 prompt群とheld-out chat F1を使います。

step 2,100はvalidation loss 3.358708、step 2,200は3.351035、step 2,300は3.344531、step 2,400は3.328543、step 2,500は3.311811でした。step 2,500のperplexityは27.434761、経過時間は2,701.42秒です。step 2,500で最良checkpointが更新され、固定promptは「おはようございます!」となりました。挨拶の種類は変化していますが、質問への応答や話題継続はまだ確認できません。validation改善と会話自然さを分けて評価する方針は維持します。

step 2,600はvalidation loss 3.312458、step 2,700は3.298765、step 2,800は3.295309、step 2,900は3.292916、step 3,000は3.266383でした。step 3,000のperplexityは26.216333、経過時間は3,209.34秒です。step 3,000で最良checkpointが更新されました。固定promptの出力は「こんばんは!」で、依然として短い挨拶に限られています。一般validationは改善していますが、Issue #1の狙いである自然な会話性能はまだ未判定です。

step 3,100はvalidation loss 3.281472、step 3,200は3.270768、step 3,300は3.274873、step 3,400は3.275212、step 3,500は3.251657でした。step 3,500のperplexityは25.833104、経過時間は3,706.69秒で、最良checkpointが更新されました。固定promptのstep 3,500生成は「こんにちは!」です。validationは改善を続けていますが、挨拶以外の応答が出ていないため、自然な会話能力の改善はまだ確認できません。

step 3,600はvalidation loss 3.249674、step 3,700は3.247966、step 3,800は3.230070、step 3,900は3.237198、step 4,000は3.223094でした。step 4,000のperplexityは25.105681、経過時間は4,214.68秒です。step 4,000で最良checkpointが更新されました。固定promptは「こんにちは!」のままで、一般validationの改善と会話生成の広がりはまだ一致していません。

step 4,100はvalidation loss 3.231673、step 4,200は3.219601、step 4,300は3.208805、step 4,400は3.196438、step 4,500は3.213047でした。step 4,400のperplexityは24.445310、経過時間は4,619.82秒で、最良checkpointが更新されました。step 4,500の固定生成は「よろしくお願いします!」となり、挨拶以外の短い定型応答が現れました。ただし、自然な話題応答を示すものではないため、まだ成功判定にはしません。

step 4,600はvalidation loss 3.193851、step 4,700は3.191092、step 4,800は3.177835、step 4,900は3.167158、step 5,000は3.168787でした。step 4,900のperplexityは23.739915、経過時間は5,124.07秒です。step 4,900で最良checkpointが更新されました。step 5,000の固定生成は「こんにちは!ありがとうございます!よろしくお願いします!」となり、短い定型応答が連結される変化が見えました。自然な話題応答か、単なる挨拶列の暗記かは、最終評価で切り分けます。

step 5,100はvalidation loss 3.170002、step 5,200は3.162290、step 5,300は3.167446、step 5,400は3.162243、step 5,500は3.153414でした。step 5,500のperplexityは23.415872、経過時間は6,124.46秒です。step 5,500で最良checkpointが更新されました。固定promptのstep 5,500生成は「こんにちは!」へ戻っており、step 5,000の定型応答連結は安定した能力とはまだ見なせません。生成の揺れも含めて保存し、後半の評価で比較します。

step 5,600はvalidation loss 3.142920、step 5,700は3.145978、step 5,800は3.137923、step 5,900は3.130502、step 6,000は3.134078でした。step 5,900のperplexityは22.885472、経過時間は6,839.18秒で、最良checkpointが更新されました。step 6,000の固定生成は「こんにちは!最近何か詳しくお願いいたしますー!」となり、挨拶から続く語句が少し長くなりました。ただし文法的には不自然で、自然な話題応答と評価できる段階ではありません。良化と崩れの両方をそのまま保存します。

step 6,100はvalidation loss 3.123706、step 6,200は3.130502、step 6,300は3.136202、step 6,400は3.136349、step 6,500は3.126902、step 6,600は3.121405、step 6,700は3.115601、step 6,800は3.116168、step 6,900は3.120221、step 7,000は3.114227でした。step 7,000のperplexityは22.516017、経過時間は8,027.75秒で、最良checkpointがstep 7,000へ更新されました。step 7,000の固定生成は「こんにちはー!」で、自然な話題応答にはまだ到達していません。validationの改善と生成品質の差を保ったまま、残り3,000 stepを実行します。

step 7,100はvalidation loss 3.114232、step 7,200は3.109955、step 7,300は3.102167、step 7,400は3.095141、step 7,500は3.090104まで改善しました。step 7,500のmetricsと生成文は標準出力・`metrics.jsonl`・samplesへ書き込まれましたが、同時に周期checkpointを書き込む際、ディスク空き容量不足によるPyTorch I/Oエラーが発生しました。`best.pt`はstep 7,400の完全なcheckpointとして残り、`step_007500.pt`は約88MiBの不完全なファイルになりました。学習プロセスはstep 7,500で終了し、step 7,500の不完全checkpointは再利用不能と確認したうえで除去します。これは学習失敗ではなく、保存容量の管理失敗です。

再開前に、生成文・metrics・checkpoint metadataを残し、周期checkpointの重い`.pt`本体は`best.pt`以外を整理します。`.pt`本体はGitHubへ追加せず、metadataに保存済みのSHA-256とstepを根拠として残します。これにより、元の会話データ、Tokenizer、既存のbest checkpoint、全生成文を保持したまま、再開に必要な空き容量を確保します。再開は同じ092の続きとして扱わず、step 7,400のbest weightsからoptimizerを初期化する別の継続試行として記録します。

Colab停止中の代替可否を確認するため、Apple Silicon実機のMPSが利用可能かを`torch 2.14.0`で確認しました。`torch.backends.mps.is_built()`と`is_available()`はいずれも`True`でした。まずは本番条件を変更しない2 stepのMPS smokeを、専用出力先`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-smoke`と`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-smoke`へ保存します。これは本番092の性能結果には混ぜず、速度・ロード・loss計算・生成経路だけを確認する補助実験です。

smokeは2026-09-06に完了しました。MPS、PyTorch 2.14.0、AMPなしで、step 1のvalidation lossは4.042009、step 2は4.035694でした。50,207,616 parameters、train 127,731例、validation 49,045例を読み込み、NaN、OOM、shape errorはありませんでした。2 stepの経過時間は19.40秒で、step 0・1・2の生成サンプルとmetrics、checkpoint metadata、summaryを専用出力先へ保存しました。この結果は本番の性能比較には使わず、実行経路が正常であることだけを示します。

ColabのHTTP 503が継続しているため、同じ092条件をMPSで本学習します。出力先は`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-10k`と`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-10k`です。MPSはCUDA AMPを使わず、seed、batch、学習率、rehearsal、EOS loss weight、評価・生成間隔は設定どおりにします。Colab版とはbackendだけが違うため、092の主結果としてbackendを明記して比較します。

最初のバックグラウンド起動は、step 0の出力を残さず終了しました。原因はログから特定できなかったため、学習結果には含めません。その後、前景セッションで同じコマンドを再実行し、学習を開始できました。

## step 7,400 checkpointの評価

step 7,500で周期checkpointの保存に失敗したため、完全な重みとして残ったstep 7,400の`best.pt`を評価しました。評価に使ったcheckpointのSHA-256は`a92cbc8125982a0e4fe779a1ae22725fb68ae0be6ca41ff59fa5dde2b6a6dd3a`です。092は10,000 stepを完走した実験ではなく、以下はstep 7,400時点の暫定評価として扱います。

まず、Issue #1で定めた8個の固定promptをconversation形式で生成しました。8例すべてがEOSへ到達し、平均生成長は短いものの、入力に応じた会話としては不十分でした。`まじで`には「こんばんは!」、`それな`には「どうもー!」、`今日なにしてた？`には「今日は夕ご飯でした。今日は暑かったです。」、`やば`には「こんばんは」、`なんかさ`には「はじめまして、最近お料理ありますか?」、`いやそれは`には「こんばんはー」、`おつかれ`には「こんばんはー」、`明日ひま？`には「こんばんはー!」を生成しました。`今日なにしてた？`と`なんかさ`に少し長い出力は出ましたが、自然な応答というより学習済み定型句の組み合わせに見えます。生成結果は`artifacts/evaluations/issue1-both-50m-quality-aware-mps-best-step7400-issue1-prompts-conversation.json`と同名の`.txt`に保存しています。JSONのSHA-256は`fcd84cc6cc7cfe571ea51c34ebab07519275306853c99ddcc68e54b3d66f015f`、テキストのSHA-256は`95dfe517aed0e0ffb84f7e5d255fac252a3e069c38fce8793715a53fdf15e6ba`です。

次に、48例のheld-out chat-testを同じTorch評価スクリプトで実行しました。先に誤ってMLX用の`evaluate_chat_dataset.py`へTorch checkpointを渡す試行があり、`Unknown file format pt`で終了しました。この試行は評価結果に含めず、生成物も作成されていません。正しいTorch評価では48例すべてがEOSへ到達し、平均生成長は7.8125 token、token overlap F1は0.212413でした。層別のF1はshort 0.348001、medium 0.171042、long 0.118196でした。出力は`artifacts/evaluations/issue1-both-50m-quality-aware-mps-best-step7400-chat-test-v1.json`と同名の`.txt`に保存しています。JSONのSHA-256は`5cd58b0ebbef09d2d9061f4eea51ba7a1009efec2031329a2f40d50aef41d9c1`、テキストのSHA-256は`e51cde5b8990f6b07dffd654c4a76713127fa6c7858fe7c858906b0141568856`です。入力JSONLのSHA-256は`65f534a8e63acf056bcbcbc7c827d62ff7dedfd383be382cc299c056dec90ce5`、selectionのSHA-256は`ab2f372d4c6d5000ab0a8ec91c8d8c22837b6ffa2005e79db3f63fdc7a8ab530`です。

086の全体F1 0.203292に対しては0.009121改善しましたが、087の0.216545には0.004132届きませんでした。087との比較では、092のmedium F1 0.171042は0.034154高い一方、shortは0.348001で0.033519低く、longは0.118196で0.013030低くなりました。ただし086・087は10,000 step付近のbest checkpoint、092はstep 7,400のcheckpointであり、学習stepとbackendも完全には揃っていません。そのため、これを厳密な勝敗とは扱わず、品質選別だけで明確な改善が出たとは結論しません。

同じcheckpointで5領域のvalidationも測定しました。generalはloss 4.352421、perplexity 77.6663、conversationはloss 2.437993、perplexity 11.4500、medicalはloss 2.530518、perplexity 12.5600、RPCはloss 2.394490、perplexity 10.9626、MRMPはloss 2.028177、perplexity 7.6002でした。出力は`artifacts/evaluations/issue1-both-50m-quality-aware-mps-best-step7400-domains.json`に保存し、SHA-256は`449f4038f3bc5bf93abffeef1fb3c13254d937becfd868993262ec3bb538f0f2`です。会話系データのlossは低いものの、固定promptとheld-out F1が伸びていないため、学習データへの適合だけで会話能力を判断してはいけないことが改めて確認できました。

## 実験終了後の結果と解釈

092は、ColabのHTTP 503を避けるためMPSで実行しました。step 7,500まで学習計算は進み、validation lossはstep 7,500で3.090104まで低下しましたが、周期checkpointの保存時にディスク容量不足によるI/Oエラーが起きました。完全な最良checkpointはstep 7,400で、そこでのvalidation lossは3.095141、perplexityは22.090362です。従って、学習が発散した実験ではなく、保存方式とディスク管理に失敗して途中終了した実験です。

品質を考慮した選別により、086よりheld-out全体F1は上がり、medium層の語彙的重なりも改善しました。しかし、087の長文応答サンプリングほどshort・long層を改善せず、固定promptでは8例中ほぼすべてが挨拶へ戻りました。今回の仮説である「質問履歴を増やし、定型挨拶を抑えれば自然な会話応答が増える」は、少なくともこのcheckpointでは支持されません。質問の割合だけでなく、応答機能、話題継続、否定、相づち、誘い、終了などを明示的に分ける必要があります。また、会話データをSFTへ追加したことで、一般validation lossとheld-out会話性能が一致しない問題も続いています。

step 7,400の重み、metrics、metadata、step 7,500までの生成サンプル、3種類の評価結果は保持します。容量回復のため周期checkpointの古い`.pt`本体を整理しましたが、生成文と評価結果を削除してはいません。092は「10,000 step完走・採用モデル」ではなく、保存失敗を含む失敗実験として記録します。

## 次に試すこと

次の093では、質問数の単純な増加ではなく、応答機能ごとの配分を変える実験を優先します。具体的には、短い挨拶だけでなく、質問への回答、相づち、同意・否定、話題継続、終了を別カテゴリとして数え、validationの分布から大きく外れない範囲でbatch内の割合を管理します。強いLLMからの蒸留はまだ使わず、Issue #1のRPC・MRMP原データから作れる範囲で改善を試みます。

その前に、学習スクリプトのcheckpoint保存を見直します。周期ごとに大きな`.pt`を積み上げず、`best.pt`と直近1個だけを保持し、保存前後に空き容量を確認する方式へ変更します。これを別の小さな検証実験で確認してから、093を開始します。step 7,400からの継続はoptimizer状態を保存していないため、092の厳密な再開とはせず、実施する場合は新しい実験番号を発行します。
