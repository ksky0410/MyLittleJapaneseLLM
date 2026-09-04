# 実験021：pretraining replayを混ぜた会話SFT

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験019の会話SFT-onlyでは、応答mask付きvalidation lossは下がったものの、通常のgeneral・conversation・medical lossがすべて悪化しました。実験020の構造化promptでは「こんにちは」への偏りも確認しました。そこで今回は、SFT batchの一部へ実験017で使ったpretraining Token列をrehearsalとして再提示し、SFT-onlyとの違いを調べます。

仮説は、SFT masked lossを75%、pretraining full lossを25%の目的関数として別々に計算して結合すれば、会話応答の学習を続けながら、一般・医療・会話の全Token分布の忘却を抑えられる可能性があるというものです。SFTとrehearsalを同じToken数で平均せず、各lossを独立に平均してから重み付けします。これにより、256 tokenのpretraining batchが、短い応答maskを持つSFT batchを単純なToken数で圧倒しないようにします。

base checkpointは実験017の`artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz`、SFTデータは実験018の`artifacts/sft/chat-v1-context256/{train,validation}.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-token-budget-1m-train.bin`です。モデル、Tokenizer、seed、SFT学習率5e-5、weight decay 0.01、500 stepは実験019と同一です。差分はrehearsal Token列の追加と、SFT 0.75・rehearsal 0.25のloss結合だけです。

評価では、SFT mask付きvalidation loss、通常のgeneral・conversation・medical validation loss、rawおよび構造化Issue #1固定promptをSFT-onlyと比較します。SFT側の出力が少し自然になっても、固定8件だけでは一般化と断定しません。忘却が減ったか、会話入力形式に対する条件付き応答が改善したかを別々に見ます。

実験前の計画コミットは`5639aea`（`exp: plan chat rehearsal sft`）、rehearsal実装コミットは`6c06049`（`feat: add pretraining rehearsal to chat sft`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-rehearsal-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --rehearsal-tokens artifacts/tokens/mixed-ja-token-budget-1m-train.bin \
  --rehearsal-ratio 0.25 \
  --output-dir artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-smoke \
  --samples-dir artifacts/samples/token-budget-chat-rehearsal-sft-5m-smoke
```

成功判定は、rehearsalなしの既存SFTを壊さず、500 stepまでmetrics・checkpoint・生成サンプルが保存されることです。通常domain lossの悪化幅と、構造化promptの偏りがSFT-onlyより小さくなることを改善の目安としますが、事前の予想が外れた場合もそのまま残します。

## 実験中の記録

2026-09-05に500 stepまで完了しました。エラーやメモリ不足はありませんでした。mask付きvalidation lossはstep 1が5.400670、step 100が4.812702、step 200が4.739223、step 300が4.687804、step 400が4.644852、step 500が4.635125でした。step 500のperplexityは103.0408、結合train lossは4.667452、学習時間は90.98秒です。最良checkpointはstep 500でした。

step 500の内訳はSFT train loss 4.402709、rehearsal full train loss 5.446867、batch sizeはSFT 6・rehearsal 2でした。結合lossは、両方をToken数で直接平均せず、SFT 75%・rehearsal 25%で計算しています。通常の「今日は」生成はstep 0の文語的長文からstep 300の崩れた長文、step 500の「今日は今日から覚らない。」へ変化しました。SFT-onlyと同様に短くなる傾向はありますが、自然な文章とは言えません。全stepの生成とmetricsを保存しています。

- `artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-smoke/metrics.jsonl`
- `artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-smoke/summary.json`
- `artifacts/samples/token-budget-chat-rehearsal-sft-5m-smoke/`

## 結果と解釈

rehearsalありのmask付きvalidation loss 4.635125は、SFT-onlyの4.623544より0.011581高く、会話SFT目的の最適化はわずかに弱くなりました。これはSFT側の学習を完全に犠牲にせず、別目的を同時に最適化した結果として予想どおりです。ただし、忘却抑制と会話応答の改善は通常domain評価と構造化promptで確認する必要があります。

## 次に試すこと

rehearsalで忘却が抑えられたら、ratio 0.1または0.5を一つずつ比較します。抑えられなければ、次は会話データの短い応答を重視するサンプリングか、履歴切り詰めの別条件を試します。
