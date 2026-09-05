# 実験037：Wikipedia追加10M Token列の5Mモデル・5,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験036では、Wikipediaを加えた約10M Token列を5Mモデルへ2,500 step学習した結果、general validation loss 5.522978となり、Wikipediaなし5Mモデルの実験029の5.290503より悪化しました。会話・医療・FineWeb validationと固定chat F1も悪化しました。

今回の仮説は、Wikipedia追加で学習Token列を約10Mへ増やしたにもかかわらず2,500 stepへ固定したため、データへの反復が不足していたというものです。同じモデル、Tokenizer、10M Token列、batch size、seedを使って5,000 stepへ延長し、validation lossと生成が改善するか確認します。反復不足が主因なら、step 2,500以降もlossが下がり、少なくともgeneral・Wikipedia validationの改善傾向が続くはずです。

この実験では最大stepを変えるため、cosine learning-rate scheduleの終点も5,000 stepへ変わります。したがって、これは「同じ総学習Token数でデータ源だけを比較する実験」ではなく、「10M Token列をより長いscheduleで学習する追試」です。Wikipediaなし5M Token列の5,000 step対照条件は別実験として後で行います。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`、9,999,973 Token、SHA-256 `d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。Wikipedia専用validationは`artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin`、998,845 Token、SHA-256 `2898e8ab7385dc7beb26e4ba956639eaa791b059a1a7e763ae9d4b958e09d269`です。Tokenizerはvocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

モデルはdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算5,197,920 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-wikipedia-augmented-ja-5m-5k.toml`です。

## 実行前の再現情報

実験036の結果commitは`575c02d`です。実験037のconfig SHA-256は`a0efd68cb9b1f4d895129359697c382fe3975038209974f3df5a99d7e6753a1f`です。学習はMacBook上の既存MLX環境で行い、smokeは省略します。実験036で同じモデル・Token列の100 step smokeと2,500 step本学習が成功済みであるため、今回は5,000 step本学習のみを実行します。

本学習は同日開始し、実行中は100 stepごとにmetrics、checkpoint metadata、生成TXTを保存しました。最終的に5,000 stepまで完走し、途中でNaN、shape error、データ長エラー、メモリ不足は発生しませんでした。実行中の中間値は「実験中の記録」に時系列で追記しています。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-5k.toml
```

成功基準は、5,000 stepがNaN、shape error、データ長エラー、メモリ不足なく完走し、1,000 stepを超えない間隔でmetrics、checkpoint metadata、生成TXTが保存されることです。実験036のstep2,500値と比較し、step5,000でvalidation lossがさらに下がるか、固定chat評価が回復するかを確認します。意味的な会話品質は自動overlap指標だけで断定しません。

## 実験中の記録

step 2,000ではvalidation lossが5.512431、step 2,500では5.436808まで下がりました。実験036の同じstep 2,500のvalidation loss 5.525606より0.088798低く、現時点では学習期間を延長する仮説と一致する方向です。ただし、実験036とはlearning-rate scheduleの終点も異なるため、Wikipedia追加の効果だけとは解釈しません。step 2,900ではvalidation loss 5.395232、step 3,100では5.348437、step 3,500では5.325272、step 3,800では5.294717、step 4,000では5.274800、step 4,200では5.265985、step 4,400では5.243676、step 4,700では5.233565まで下がり、延長後の改善が続いています。step 4,500の5.245535のような小さな反発はありましたが、全体としては下降傾向です。step 4,700時点では学習時間979.11秒で、学習は継続中です。step 4,700までの生成結果も`artifacts/samples/fineweb2-wikipedia-augmented-ja-5m-5k/step_004700.txt`に保存しています。
step 5,000で学習が完了しました。最終validation lossは5.216578、perplexityは184.302、metrics上の経過時間は1,042.91秒でした。最終生成結果は`artifacts/samples/fineweb2-wikipedia-augmented-ja-5m-5k/step_005000.txt`に保存し、完走後にdomain評価とfixed chat評価を実行しました。

## 結果と解釈

2026-09-05、5,000 stepを正常に完走しました。最終stepがそのまま最良checkpointで、train lossは4.305170、general validation lossは5.216578、perplexityは184.302でした。学習時間はmetrics上で1,042.91秒、summary上で1,043.57秒です。学習率は最終的に`3.000003e-5`となりました。成功基準としていた異常なしの完走、1,000 step以下の記録間隔、checkpoint metadataと生成TXTの保存を満たしています。最良checkpointは`artifacts/checkpoints/fineweb2-wikipedia-augmented-ja-5m-5k/step_005000.npz`です。

実験036の同じWikipedia追加Token列による2,500 step条件と比べると、general validation lossは5.525606から5.216578へ0.309028改善し、perplexityは251.038から184.302へ下がりました。実験029のWikipediaなし約5M Token列・2,500 step条件のgeneral loss 5.290503も0.073925下回っています。ただし037は学習stepを2倍にし、cosine scheduleの終点も5,000 stepへ変更しています。したがって、この差はWikipedia追加単独の効果ではなく、「10M Token列を長いscheduleで学習した条件」の結果です。今回の主仮説である、データ量を増やした場合に学習期間を延長すれば反復不足による悪化を回復できるという見立ては、general lossについて支持されました。

最良checkpointのドメイン別評価では、general loss 5.216578（PPL 184.302）、conversation loss 3.187662（PPL 24.232）、medical loss 3.626501（PPL 37.581）、FineWeb test loss 4.250461（PPL 70.138）、Wikipedia test loss 4.275204（PPL 71.895）でした。Wikipedia専用validationも学習中の混合比率に対応して低下し、Wikipedia文体への適応が確認できます。ただし、Wikipedia testは今回初めて追加した評価なので、追加前との差分を因果的な改善とは扱いません。

固定chat-test-v1の48例では、EOS到達が48/48、平均生成Token数は4.875、overlap precisionは0.129099、recallは0.052964、F1は0.063502でした。実験036のF1 0.056060からは改善しましたが、Wikipediaなしの実験029のF1 0.072297には届いていません。層別F1はshort 0.099901、medium 0.076449、long 0.014157で、longの意味対応は特に弱いままです。生成は「こんにちは!」「ですよね。そう」のような短い自然な返答が一部で見られる一方、話題と無関係な「最近!」「お酒はいい!」や空文字もあり、自然な日本語会話モデルとしては未達です。固定chat評価は自動Token overlapであり、意味的な正しさを完全には測れないため、結果JSONと全文TXTをそのまま保存しました。

固定prompt `今日は`への最終生成は、冒頭こそ日本語断片を含むものの、後半で句読点や中黒が連続し、文法と意味の一貫性が崩れています。この出力は品質が悪いものとして削除せず、`artifacts/samples/fineweb2-wikipedia-augmented-ja-5m-5k/step_005000.txt`に保存しています。学習中のstep 0から5,000までの生成結果も100 step間隔で同じディレクトリへ保存しており、改善と崩れの両方を追跡できます。

以上から、今回の延長学習はlanguage modeling lossと短い会話の自動指標を回復させましたが、会話の内容理解や長い履歴への応答品質までは回復させませんでした。次の対照では、Wikipediaなしの約5M Token列を同じ5,000 step・同じscheduleで学習し、037との差がデータ源によるものか、単に学習期間によるものかを分離します。

成果物のSHA-256は、metricsが`8f88a02129c2bd383318f7e63ce3598e551a6abe3832162344ac9ba42451f90e`、summaryが`77800dabb414ccc0dc9d4e377eae135bf44b69deed1687b7f215f4a5754b2710`、step 5,000 metadataが`c2185ef80a27c6231139bbb2a3abe49e154515a6cf524583d4badef83cc43109`、step 5,000生成TXTが`c5b1620e1c4258f2812755e868ee7e9396c5dd3bbbba378cabf6696841f445f9`です。domain評価JSONは`4b56b2a7b9a354775f5670d42642fa85c8de08c484a526eafbdf80ef1385a69a`、fixed chat評価JSONは`3885c27ecbc0899f346d9f7ed5d580f53d8078b78f34decde0481bd55dc38f8a`、fixed chat生成TXTは`0e134b9cfc8b64b541e7ccc640351bd027d2f151c8b202a574e92c26466fe622`です。

## 次に試すこと

Wikipedia追加による2,500 step時の悪化は、5,000 stepの延長でgeneral lossと短いchat指標がかなり回復しました。次はWikipediaなし約5M Token列を同じ5,000 step・同じscheduleで学習し、データ量を増やしたことと学習時間を延長したことを分離します。その対照後、会話・医療・Wikipediaのsource比率を変えるablationへ進みます。データ条件の比較を終えてから、RoPE、RMSNorm、SwiGLU、GQAなど現代的な構造要素を一つずつ導入し、各変更を独立した実験として記録します。Colabは、20M以上のモデルや大量Tokenの学習でMacBookより有利ですが、まずはデータ対照をMLXでそろえ、backend差を混ぜない方針です。
