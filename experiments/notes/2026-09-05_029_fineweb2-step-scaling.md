# 実験029：FineWeb混合コーパスでの学習step延長

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験028ではFineWeb2 Edu Japaneseの`small_tokens_cleaned`を追加した約5M Tokenの混合コーパスを作成しましたが、500 stepでは既存general validation、会話validation、医療validation、固定会話testのいずれも、約1M Tokenの対照条件を上回りませんでした。実験028の学習ではbatch size 8、context 256、500 stepのため、1回の学習で処理したwindow Token数は約1,024,000です。5M Tokenの候補プールからランダムにwindowを取る現在の実装では、500 stepのままでは追加データを十分に見ていない可能性があります。

今回の仮説は、同じFineWeb混合Token列を2,500 stepまで学習すれば、500 step条件よりgeneral validation lossとFineWeb validation lossが下がり、出力が極端に崩れる頻度も減るというものです。反対に、2,500 stepでvalidation lossが頭打ちまたは悪化し、生成文も改善しない場合は、学習step不足だけでなく、現在のTokenizer・source混合・モデル容量のいずれかが主なボトルネックだと判断します。

この実験ではデータ、Tokenizer、モデル構造、optimizer、learning rate schedule、batch size、seedを変更しません。変更するのは最大step数と、それに伴う専用checkpoint・sample保存先だけです。実験028の500 step結果と同じseedで最初から学習するため、step 500までのmetricsとcheckpointが一致することも再現性の確認に使います。

## 使用するデータとTokenizer

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`です。実際のToken数は4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。混合条件、source別Token比率、入力本文のhashは`artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json`へ記録済みです。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`で、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、実語彙数は4,096です。既存general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`を使います。追加source側のvalidationとして、FineWeb test由来の`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`（2,061,459 Token、SHA-256 `36d8d5c8bc92de1e168b8c3de9dd4ee975dec66f6b644b83bfbf9b239877161c`）もdomain評価へ使います。会話・医療のvalidationは実験028と同じToken列を使います。

## モデルと学習条件

モデルはvocab size 4,096、dim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embeddingで、概算5,197,920 parametersです。batch size 8、最大2,500 step、evaluation interval 100、sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-mixed-ja-5m-2p5k.toml`へ固定します。

対照は実験028の500 step条件、補助的な比較は実験017の約1M Token・500 step条件です。主比較では同じFineWeb混合Token列の500 stepと2,500 stepを比べ、500 stepまでの同一性を確認した後、100 stepごとのvalidation loss、perplexity、生成文を追跡します。学習途中の出力は悪いものも削除せずGitHubへ保存します。

## 実行前の再現情報

実験028を記録したcommitは`d3b148c`です。実験029の専用configとこのノートをcommit・pushしてから、以下のコマンドを実行します。

```bash
.venv/bin/python scripts/train.py --config configs/fineweb2-mixed-ja-5m-2p5k.toml
```

成功基準は、2,500 stepまでNaN、shape error、データ長エラー、途中停止なしに完走し、100 step以下の間隔でmetrics、checkpoint metadata、固定prompt生成文を保存することです。性能面では、general validation lossまたはFineWeb validation lossが500 stepから0.10以上下がることを有望な変化の目安とします。ただしloss低下だけでは会話能力の改善とは扱わず、実験027の固定chat-test-v1を最後に同じ条件で評価します。

## 実験中の記録

未実施です。

## 結果と解釈

未実施です。

## 次に試すこと

step延長で改善が確認できれば、同じFineWeb混合Token列・Tokenizer・学習stepを固定し、`dim=384・layers=10・heads=6・context=256`の約19.4M parameterモデルへ拡張します。step延長で改善しなければ、モデル容量を増やす前にTokenizerの再学習またはFineWeb source比率の再設計を別実験として検討します。
