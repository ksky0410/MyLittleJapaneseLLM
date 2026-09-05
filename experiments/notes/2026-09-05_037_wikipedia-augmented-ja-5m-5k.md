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

本学習は同日開始し、step 2,500まで正常に進行しています。step 500ではtrain loss 5.263340、validation loss 6.219744、perplexity 502.574、step 1,000ではtrain loss 4.764680、validation loss 5.774941、perplexity 322.125、step 1,200ではtrain loss 4.705955、validation loss 5.739346、perplexity 310.861、step 2,000ではtrain loss 4.340868、validation loss 5.512431、perplexity 247.753、step 2,500ではtrain loss 4.080081、validation loss 5.436808、perplexity 229.708でした。step 2,500時点の学習率は`1.786027e-4`、経過時間は470.15秒です。step 2,500までNaN、shape error、データ長エラー、メモリ不足は発生しておらず、学習は継続中です。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-5k.toml
```

成功基準は、5,000 stepがNaN、shape error、データ長エラー、メモリ不足なく完走し、1,000 stepを超えない間隔でmetrics、checkpoint metadata、生成TXTが保存されることです。実験036のstep2,500値と比較し、step5,000でvalidation lossがさらに下がるか、固定chat評価が回復するかを確認します。意味的な会話品質は自動overlap指標だけで断定しません。

## 実験中の記録

step 2,000ではvalidation lossが5.512431、step 2,500では5.436808まで下がりました。実験036の同じstep 2,500のvalidation loss 5.525606より0.088798低く、現時点では学習期間を延長する仮説と一致する方向です。ただし、実験036とはlearning-rate scheduleの終点も異なるため、Wikipedia追加の効果だけとは解釈しません。step 2,500時点では学習時間470.15秒で、学習は継続中です。step 2,500までの生成結果も`artifacts/samples/fineweb2-wikipedia-augmented-ja-5m-5k/step_002500.txt`に保存しています。

## 結果と解釈

未実施です。

## 次に試すこと

lossとchatが回復した場合は、Wikipediaなし5M Token列を5,000 step学習する対照を追加します。回復しない場合は、Wikipediaのsource比率を下げたablationへ進みます。データの反復条件を確認した後、Colab PyTorch側でも10M Token・20Mまたは50Mモデルを試し、backendとデータ量を混同しないようにします。
