# 実験038：Wikipediaなし5M Token列の5,000 step対照

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験037では、青空文庫・FineWeb・Wikipedia・会話・医療を混ぜた約10M Token列を5Mモデルへ5,000 step学習し、general validation loss 5.216578、固定chat-test F1 0.063502となりました。実験029のWikipediaなし約5M Token列・2,500 step条件のgeneral loss 5.290503には届きましたが、実験037では学習stepとschedule horizonを同時に変更しています。

今回の目的は、実験037の改善を学習期間の延長とWikipedia追加に分けることです。Wikipediaなしの約5M Token列を、モデル、Tokenizer、batch size、seed、optimizer、learning-rate schedule、総学習stepを実験037と完全にそろえて学習します。Wikipedia追加が有効なら037が038よりgeneral loss、Wikipedia testを除く既存domain、固定chatのいずれかで優位になると予想します。単に学習期間の延長が主因なら、038も037に近いgeneral lossまで下がり、会話ではWikipediaを含まない038が同等以上になる可能性があります。

この比較は、同じ計算量で「約5M Token候補を二周近く見る条件」と「約10M Token候補を一周近く見る条件」を比べるものです。したがって、データの重複回数と多様性も含めた実用的な対照であり、各Tokenを一度だけ見せる厳密なデータ量比較ではありません。実験037の改善をWikipedia単独の因果効果と断定するための実験ではなく、次のsource ablationへ進むための探索的な対照です。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`です。実際のToken数は4,999,958、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。混合元のsource構成、前処理、Token比率は`artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json`に保存済みです。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。検証Token列は037と同じ`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`を使用します。会話、医療、FineWeb、Wikipediaのdomain評価とfixed chat評価も037と同じ評価Token列・選択ファイルを使います。

モデルはdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算5,197,920 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-mixed-ja-5m-5k.toml`に保存します。

## 実行前の再現情報

比較対象の実験037を完了したcommitは`bee443e`です。実験038のconfig SHA-256は`3f2430adf5eff81c78cad5a892e35297e0d97cc047252759f7ab201f9c7457f3`です。学習はMacBook上の既存MLX環境で行います。037と同じ実装・backendを使うことで、Colab PyTorch実験035とのbackend差をこの比較へ持ち込みません。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py --config configs/fineweb2-mixed-ja-5m-5k.toml
```

成功基準は、5,000 stepをNaN、shape error、データ長エラー、メモリ不足なく完走し、1,000 step以下の間隔でmetrics、checkpoint metadata、固定promptの生成TXTを保存することです。完走後はgeneral、conversation、medical、FineWeb、Wikipediaのdomain評価とfixed chat-test-v1を037と同じ条件で実行します。自動overlap F1だけで自然な会話品質を断定せず、生成TXTを目視できる状態で残します。

## 実験中の記録

2026-09-05、学習を開始しました。step 1ではtrain loss 8.721458、general validation loss 8.801644、perplexity 6645.158でした。warmup中のstep 300ではtrain loss 5.494156、validation loss 6.306765、perplexity 548.269、learning rate `3.000000e-4`、経過時間32.85秒でした。step 1,000ではtrain loss 4.797681、validation loss 5.606802、perplexity 272.272、learning rate `2.855307e-4`、経過時間173.07秒、step 1,100ではtrain loss 4.449255、validation loss 5.548042、perplexity 256.734、step 1,400ではtrain loss 4.247522、validation loss 5.433506、perplexity 228.951、step 1,600ではtrain loss 4.552619、validation loss 5.383194、perplexity 217.717、step 1,900ではtrain loss 4.249821、validation loss 5.301405、perplexity 200.618、step 2,100ではtrain loss 4.242826、validation loss 5.260210、perplexity 192.522でした。step 2,000では5.303236へ一時的に反発し、step 2,200では5.273854、step 2,300では5.275253となった後、step 2,400では5.218160、step 2,500では5.212595、step 2,600では5.196833、step 2,700では5.180368、step 2,800では5.159169、step 2,900では5.149507まで下がりました。step 2,900時点のperplexityは172.346、learning rateは`1.426345e-4`、経過時間は591.61秒です。ここまでNaN、shape error、データ長エラー、メモリ不足は発生していません。step 300から2,900までの生成結果は`artifacts/samples/fineweb2-mixed-ja-5m-5k/`へ保存されています。学習は継続中です。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。037と038の差を確認した後、Wikipediaのsource比率を下げたablation、または会話・医療の比率を保ったまま高品質一般文書を増やす実験へ進みます。データ条件の比較が終わってから、RoPE、RMSNorm、SwiGLU、GQAなどの構造変更を一つずつ導入します。
