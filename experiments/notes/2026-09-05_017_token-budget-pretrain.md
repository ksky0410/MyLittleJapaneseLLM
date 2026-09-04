# 実験017：Token予算混合コーパスの5M級事前学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験016で作ったToken予算混合コーパスが、単位数ベース混合コーパスと比べて小型モデルの学習結果を変えるかを確認します。今回の仮説は、学習Tokenのsource比率をmanifestで把握しやすくしたことで、長い会話・医療問題が意図せず過大になる条件を明確に比較できるというものです。ただし実験016では一般sourceが枯渇し、実測比率はgeneral 50.83%、conversation 24.57%、medical 24.60%でした。そのため「一般80%のモデル」とは呼ばず、Token予算で制御した現状の混合モデルとして扱います。

比較の基準は、同じTokenizer・モデル構造・seed・学習stepの既存expanded mixed v2 absoluteモデルです。新しい学習はabsolute position embeddingだけを使い、RoPEやSFTなど他の変更は入れません。モデルはdim 240、6層、6 heads、context 256、batch size 8、学習率3e-4、warmup 300、weight decay 0.1、seed 42、最大500 stepです。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`を再利用し、学習Token列だけを実016の混合出力から作ります。検証は既存のgeneral validation Token列で行い、終了後にconversation・medicalのdomain評価とIssue #1固定会話prompt評価も行います。

実験前のGitコミットは`709810b`（`exp: plan token budget pretraining`）です。入力混合出力のSHA-256は`d0eb6691a25107fb2ab94b91c2a366e2a80a5fd720797fec27434a29d1cea000`、TokenizerのSHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。Token化コマンドは次のとおりです。

```bash
.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-1m.txt \
  --output artifacts/tokens/mixed-ja-token-budget-1m-train.bin
```

学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py --config configs/token-budget-mixed-ja-5m-smoke.toml
```

成功判定は、Token化がmanifestの測定規則と矛盾せず、500 stepまで完了することです。validation loss、domain別loss、固定会話prompt生成、学習時間、エラーやメモリ問題を既存のGit追跡対象へ保存します。単位数ベース混合モデルよりlossが低いことは成功条件ではなく、違いを解釈できる記録を残すことを優先します。

## 実験中の記録

2026-09-05にToken化を実行し、エラーなく999,997 tokenを得ました。Token列のbinary SHA-256は`91423864b083009c6ed10cad7645263b5b8c60f7ea8e9d1c3af1f25474cad471`で、混合manifestの`selected_token_count`と一致しました。実効語彙数は4,096、EOS IDは3です。binary本体は大きいためGitへ追加せず、同名JSON metadataとこのノートへ保存条件を残します。

Token化記録をpushした後、同日中に学習を開始しました。100 stepごとにmetricsと固定prompt生成を保存し、途中停止はありませんでした。validation lossの推移は、step 1が8.799506、step 100が6.567255、step 200が6.266910、step 300が5.950939、step 400が5.710444、step 500が5.606362でした。validation perplexityはstep 500で272.1523です。train lossはstep 500で5.179236でした。

stepごとのmetrics、checkpoint metadata、固定prompt生成は次のディレクトリへ保存しました。重み本体の`.npz`はGitへ追加せず、metadataと実験ノートから保存先を追跡できるようにします。

- `artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/metrics.jsonl`
- `artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/summary.json`
- `artifacts/samples/token-budget-mixed-ja-5m-smoke/`

## 結果と解釈

500 stepまで完了し、最良checkpointはstep 500でした。学習時間は84.55秒です。既存expanded mixed v2 absoluteモデルのgeneral validation loss 5.735948に対して、今回のToken予算混合モデルは5.606362となり、general validationだけでは約0.129586低くなりました。ただし、データの並びとsource配分を変更した比較であり、1 seed・500 stepの探索結果です。これだけでToken予算方式の優位性や汎用性を結論づけません。

学習中の固定promptはstep 0から500まで残しています。step 0の出力はほぼ崩れたToken列でしたが、step 500では日本語らしい断片が増えました。一方、会話らしさや医療内容の正確さをこの生成だけから主張できないため、同じcheckpointのdomain別lossとIssue #1固定promptを別途評価します。

## 次に試すこと

次に、既存expanded absoluteモデルと同じgeneral・conversation・medical評価、およびIssue #1固定promptを実行します。Token予算混合モデルで会話文体が改善しない場合は、次に会話SFTの応答側loss maskingへ進みます。
