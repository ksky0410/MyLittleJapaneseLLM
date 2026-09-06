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

Colab停止中の代替可否を確認するため、Apple Silicon実機のMPSが利用可能かを`torch 2.14.0`で確認しました。`torch.backends.mps.is_built()`と`is_available()`はいずれも`True`でした。まずは本番条件を変更しない2 stepのMPS smokeを、専用出力先`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-smoke`と`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-smoke`へ保存します。これは本番092の性能結果には混ぜず、速度・ロード・loss計算・生成経路だけを確認する補助実験です。

smokeは2026-09-06に完了しました。MPS、PyTorch 2.14.0、AMPなしで、step 1のvalidation lossは4.042009、step 2は4.035694でした。50,207,616 parameters、train 127,731例、validation 49,045例を読み込み、NaN、OOM、shape errorはありませんでした。2 stepの経過時間は19.40秒で、step 0・1・2の生成サンプルとmetrics、checkpoint metadata、summaryを専用出力先へ保存しました。この結果は本番の性能比較には使わず、実行経路が正常であることだけを示します。

ColabのHTTP 503が継続しているため、同じ092条件をMPSで本学習します。出力先は`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-10k`と`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-quality-aware-mps-10k`です。MPSはCUDA AMPを使わず、seed、batch、学習率、rehearsal、EOS loss weight、評価・生成間隔は設定どおりにします。Colab版とはbackendだけが違うため、092の主結果としてbackendを明記して比較します。

最初のバックグラウンド起動は、step 0の出力を残さず終了しました。原因はログから特定できなかったため、学習結果には含めません。その後、前景セッションで同じコマンドを再実行し、学習を開始できました。

## 実験終了後の結果と解釈

学習未実施の場合は、ColabのHTTP status、session状態、bundle hashを記録します。学習できた場合は、最終・最良validation loss、perplexity、chat F1、最良checkpoint、学習時間、最大メモリ、生成サンプルの保存場所を追記し、086・087と比較します。

## 次に試すこと

092の結果が改善した場合は、質問履歴比率またはsource別の配分を一つだけ変え、改善要因を切り分けます。改善しない場合は、カテゴリ分類の誤り、質問文の過剰選別、MRMPの候補不足を監査します。一般日本語の性能が低い場合は、SFT前の継続事前学習089の結果と組み合わせて、基礎能力と会話適応を分離して評価します。
