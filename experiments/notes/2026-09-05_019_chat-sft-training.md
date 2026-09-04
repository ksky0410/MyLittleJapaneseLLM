# 実験019：Token予算pretraining checkpointへの会話SFT

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験017で学習したToken予算混合モデルを初期値として、実験018の話者境界付きSFTデータを追加学習します。今回の仮説は、通常のpretrainingで会話文体が生じなかったモデルでも、履歴を文脈として与え、現在の返答本文とEOSだけへlossをかけることで、Issue #1の短い会話promptに対するcompletionが改善する可能性がある、というものです。

base checkpointは`artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz`です。SFTデータは`artifacts/sft/chat-v1-context256/{train,validation}.npz`で、整形条件とSHA-256は実験018のmanifestに記録しています。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、モデルはdim 240・6層・6 heads・context 256・absolute position embeddingです。学習条件はbatch size 8、最大500 step、seed 42、学習率5e-5、minimum learning rate 5e-6、warmup 50、weight decay 0.01です。SFT validation lossは応答maskが1のTokenだけで計算します。

SFT前後で、SFT validation lossだけでなく、通常のgeneral・conversation・medical validation lossも評価します。Issue #1の8固定promptも同じtemperature 0.8、top-k 40、各promptのseed 42〜49で再生成します。SFTで会話以外のlossが悪化する可能性も結果として記録します。自然な返答が一つでも出れば成功と決めつけず、空出力・文語・医療形式への漏れも含めて保存します。

実験前のGitコミットは`1d24563`（`exp: plan masked chat sft training`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --output-dir artifacts/checkpoints/token-budget-chat-sft-5m-smoke \
  --samples-dir artifacts/samples/token-budget-chat-sft-5m-smoke
```

成功判定は、500 stepまで完了し、mask付きtrain/validation loss、checkpoint metadata、stepごとの生成サンプルが保存されることです。会話能力の向上は別途固定promptとdomain評価で判定します。

## 実験中の記録

2026-09-05に500 stepまで完了しました。エラーやメモリ不足はありませんでした。mask付きvalidation lossの推移は、step 1が5.400093、step 100が4.804685、step 200が4.717910、step 300が4.670404、step 400が4.633231、step 500が4.623544でした。validation perplexityはstep 500で101.8543、train lossは4.570463、学習時間は141.35秒です。最良checkpointはstep 500です。

通常の生成サンプルは、step 0では「今日は左生の後と云を開い」といった文語的な長文でしたが、step 300では「今日は今日か、いい!私は、と聞いた。」、step 500では「今日はおかけてあってます。」となりました。短く止まる傾向は現れましたが、自然な応答とはまだ言えません。良化だけでなく、文法崩れも含めてstep 0〜500の全サンプルを保存しています。

- `artifacts/checkpoints/token-budget-chat-sft-5m-smoke/metrics.jsonl`
- `artifacts/checkpoints/token-budget-chat-sft-5m-smoke/summary.json`
- `artifacts/samples/token-budget-chat-sft-5m-smoke/`

## 結果と解釈

会話SFT validation lossは学習中一貫して下がり、応答本文とEOSへ限定した目的関数は正常に最適化されました。これは「会話SFTデータの応答tokenを予測する」学習が進んだことを示しますが、自然な会話能力の獲得を意味しません。特に通常の生成promptはSFTデータの入力形式と異なるため、次の固定会話prompt評価で本来の仮説を確認します。

## 次に試すこと

SFT後に同じcheckpointを固定promptとdomain評価へ通します。改善しない場合は、履歴の切り詰め方、学習率、会話のturn選択、短い応答を重視するサンプリングのいずれか一つを変更します。
