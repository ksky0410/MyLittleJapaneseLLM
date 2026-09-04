# 実験029：FineWeb混合コーパスでの学習step延長

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験028ではFineWeb2 Edu Japaneseの`small_tokens_cleaned`を追加した約5M Tokenの混合コーパスを作成しましたが、500 stepでは既存general validation、会話validation、医療validation、固定会話testのいずれも、約1M Tokenの対照条件を上回りませんでした。実験028の学習ではbatch size 8、context 256、500 stepのため、1回の学習で処理したwindow Token数は約1,024,000です。5M Tokenの候補プールからランダムにwindowを取る現在の実装では、500 stepのままでは追加データを十分に見ていない可能性があります。

今回の仮説は、同じFineWeb混合Token列を2,500 stepまで学習すれば、500 step条件よりgeneral validation lossとFineWeb validation lossが下がり、出力が極端に崩れる頻度も減るというものです。反対に、2,500 stepでvalidation lossが頭打ちまたは悪化し、生成文も改善しない場合は、学習step不足だけでなく、現在のTokenizer・source混合・モデル容量のいずれかが主なボトルネックだと判断します。

この実験ではデータ、Tokenizer、モデル構造、optimizerの種類、batch size、seedを変更しません。ただし現在の実装ではcosine learning rate scheduleの終点が`max_steps`に依存するため、最大step数を500から2,500へ変えると、同じstepでの学習率も変わります。したがって今回は「長い学習予算に合わせてschedule horizonも延長した条件」として扱い、step 500までのmetricsとcheckpointが実験028と一致することは期待しません。この実装上の制約を含めて記録し、必要なら後続実験でschedule horizonを独立指定できるようにします。

## 使用するデータとTokenizer

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`です。実際のToken数は4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。混合条件、source別Token比率、入力本文のhashは`artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json`へ記録済みです。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`で、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、実語彙数は4,096です。既存general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`を使います。追加source側のvalidationとして、FineWeb test由来の`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`（2,061,459 Token、SHA-256 `36d8d5c8bc92de1e168b8c3de9dd4ee975dec66f6b644b83bfbf9b239877161c`）もdomain評価へ使います。会話・医療のvalidationは実験028と同じToken列を使います。

## モデルと学習条件

モデルはvocab size 4,096、dim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、概算5,197,920 parametersです。batch size 8、最大2,500 step、evaluation interval 100、sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-mixed-ja-5m-2p5k.toml`へ固定します。

対照は実験028の500 step条件、補助的な比較は実験017の約1M Token・500 step条件です。主比較では同じFineWeb混合Token列の500 stepと2,500 stepを比べますが、学習率schedule horizonも異なるため、結果は「長時間学習条件」の効果として解釈します。100 stepごとのvalidation loss、perplexity、生成文を追跡し、学習途中の出力は悪いものも削除せずGitHubへ保存します。

## 実行前の再現情報

実験028を記録したcommitは`d3b148c`です。実験029の専用configとこのノートをcommit・pushしてから、以下のコマンドを実行します。

```bash
.venv/bin/python scripts/train.py --config configs/fineweb2-mixed-ja-5m-2p5k.toml
```

成功基準は、2,500 stepまでNaN、shape error、データ長エラー、途中停止なしに完走し、100 step以下の間隔でmetrics、checkpoint metadata、固定prompt生成文を保存することです。性能面では、general validation lossまたはFineWeb validation lossが500 stepから0.10以上下がることを有望な変化の目安とします。ただしloss低下だけでは会話能力の改善とは扱わず、実験027の固定chat-test-v1を最後に同じ条件で評価します。

## 実験中の記録

実験開始前は未実施と記録していました。

2026-09-05、step 500まで到達しました。train lossは5.293273、general validation lossは5.938881、perplexityは379.510、学習率は`0.0002945857`でした。実験028の500 step条件は学習率が`0.0000300167`までcosine decayしていたため、step 500のlossが一致しないのは異常ではなく、`max_steps`をschedule horizonへ使う実装による予定された差です。2,500 stepまで継続し、最終的なlossと生成の変化を確認します。

step 1,000まで到達しました。train lossは4.797299、general validation lossは5.606614、perplexityは272.221、学習率は`0.0002381486`です。general validation lossは実験028の500 step条件6.004154から0.397541下がり、実験017の約1M Token・500 step条件5.606362にも近づきました。ただし、step 1,000時点でも学習率scheduleが異なり、実験017とは学習step数も異なるため、これは「FineWeb追加だけの効果」ではなく、長い学習予算とschedule延長を含む中間結果です。学習は2,500 stepまで継続します。

## 結果と解釈

2026-09-05に2,500 stepを完走しました。最終train lossは4.005231、general validation lossは5.290503、perplexityは198.443、最終学習率は`0.0000300001`でした。summary上の学習時間は551.22秒で、最良checkpointは`artifacts/checkpoints/fineweb2-mixed-ja-5m-2p5k/step_002500.npz`です。NaN、shape error、データ長エラー、途中停止は発生しませんでした。最大メモリと温度は専用計測を実施していないため、未計測です。

実験028の同じFineWeb混合データ・5Mモデル・500 step条件と比べると、general lossは6.004154から5.290503へ0.713651低下しました。FineWeb test lossは5.281133から4.560703へ0.720430低下し、会話validation lossは3.966163から3.341280へ0.624882低下、医療validation lossは5.217864から3.883457へ1.334407低下しました。実験017の約1M Token・500 step条件と比べてもgeneralは5.606362から0.315859低く、会話は3.852320から0.511040低く、医療は4.909000から1.025543低くなっています。ただし、029は学習Token列だけでなく、`max_steps`に応じてcosine learning rate scheduleの終点も延長しているため、これらをFineWeb追加単独の効果とは解釈しません。今回の結果は「広い5M候補プールを長いscheduleで十分に学習すると、500 step時点の短期悪化が解消される」ことを示す探索結果です。

固定chat-test-v1の48例では、EOS到達が48/48、平均生成長が6.60 Token、precision・recall・F1が0.1425、0.0653、0.0723でした。実験028の500 step条件はEOS 47/48、平均8.79 Token、F1 0.0421でしたので、029は全体F1が0.0302上がり、生成長は2.19 Token短くなりました。層別F1もshort 0.0835、medium 0.0755、long 0.0579となり、実験028の0.0493、0.0424、0.0345から全層で上がりました。実験017のstep 500ベース条件F1 0.0505と比べても上がっています。

一方、生成文の意味的な品質はまだ不十分です。固定testでは、挨拶に対する「そうなんですね!」、話題に対する空欄や「私も、いか?」、長い履歴に対する「はい。」のような定型・短文が目立ちました。実験028の崩れた長文より停止と日本語らしさは改善していますが、参照応答の内容や話者役割を安定して捉えたとは言えません。Token overlap F1の改善には、正しい内容の学習だけでなく、無関係な長文を出さず短く終了する変化も含まれています。

今回の仮説は、一般validationとFineWeb validationのloss改善について支持されました。500 stepでは約1M Tokenしか処理しないため、5M候補プールの追加が短期では不利に見えましたが、2,500 stepへ延長するとlossは一貫して改善し、FineWeb testを含む複数domainで効果が確認されました。ただし、学習率schedule horizonの変更が含まれるため、純粋なstep数の因果効果はまだ分離できていません。また、会話testの意味評価は未実施で、train本文完全一致候補7例と履歴切り詰め33例という実験027の注意点も引き継いでいます。

学習・評価パイプライン、checkpoint保存、100 stepごとの生成記録、domain評価、固定chat testはすべて成功しました。性能面ではlanguage modeling lossとoverlap指標は改善しましたが、自然な日本語会話という最終目標は未達です。失敗した短い出力や崩れた出力を含む全sampleをGitHubへ保存し、次のモデル容量実験の基準として使用します。

## 次に試すこと

次は、今回のFineWeb混合Token列、Tokenizer、学習step 2,500を固定し、`dim=384・layers=10・heads=6・context=256`の約19.4M parameterモデルへ拡張します。まず50〜100 stepのsmokeで速度とメモリを確認し、その後に同じ2,500 stepを実行します。モデル容量実験ではTokenizer、データ、seed、schedule horizonを変えず、5Mモデルとの比較を可能にします。会話評価の意味面については、別途48例の人手レビュー用テンプレートを作成し、自動overlap指標と混同しない形で追加します。

## 成果物のハッシュ

metricsのSHA-256は`f9da848dd2085cd7cc2f6704999f9f6f0144e987c6a9325de8fe138e61e2d28b`、summaryは`12fc16d2fd9bb81770a82b2adf98c925c253b73a750947c2f8818cd849e506e3`、step 2,500 metadataは`cf7084d4ea233dafeab606de73f2dd858915bc2f1d826168c69b0eb43fa86165`です。step 2,500生成文は`3950580065707e36d5d2cb312aef2bc890abd6669e16210e82d753bb9a5ba1fb`、固定chat-test TXTは`794d5e777a83c867fa3c2673dd1fb06ff5652de6857eb8ce988bb1bad6168916`、domain評価JSONは`f71c99f8adc422354eb0fbf84ddda39a5611e68b35c8dc883e4bce6c49fb03e4`、固定chat評価JSONは`63dc7a56f40d1c9dc1244b87f009cbbc425a2373ce94032566e898fb1a50a6ca`です。metricsにはstep 1と100〜2,500の100 step間隔、合計26行を保存しています。
