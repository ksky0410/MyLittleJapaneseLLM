# 実験021：pretraining replayを混ぜた会話SFT

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験019の会話SFT-onlyでは、応答mask付きvalidation lossは下がったものの、通常のgeneral・conversation・medical lossがすべて悪化しました。実験020の構造化promptでは「こんにちは」への偏りも確認しました。そこで今回は、SFT batchの一部へ実験017で使ったpretraining Token列をrehearsalとして再提示し、SFT-onlyとの違いを調べます。

仮説は、SFT masked lossを75%、pretraining full lossを25%の目的関数として別々に計算して結合すれば、会話応答の学習を続けながら、一般・医療・会話の全Token分布の忘却を抑えられる可能性があるというものです。SFTとrehearsalを同じToken数で平均せず、各lossを独立に平均してから重み付けします。これにより、256 tokenのpretraining batchが、短い応答maskを持つSFT batchを単純なToken数で圧倒しないようにします。

base checkpointは実験017の`artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz`、SFTデータは実験018の`artifacts/sft/chat-v1-context256/{train,validation}.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-token-budget-1m-train.bin`です。モデル、Tokenizer、seed、SFT学習率5e-5、weight decay 0.01、500 stepは実験019と同一です。差分はrehearsal Token列の追加と、SFT 0.75・rehearsal 0.25のloss結合だけです。

評価では、SFT mask付きvalidation loss、通常のgeneral・conversation・medical validation loss、rawおよび構造化Issue #1固定promptをSFT-onlyと比較します。SFT側の出力が少し自然になっても、固定8件だけでは一般化と断定しません。忘却が減ったか、会話入力形式に対する条件付き応答が改善したかを別々に見ます。

実験前のGitコミットは`5639aea`（`exp: plan chat rehearsal sft`）です。rehearsal実装が入った後にこの記録へ実装コミットを追記します。使用コマンドは次のとおりです。

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

未実施です。100 step以内の間隔でSFT loss、rehearsal loss、結合loss、validation loss、生成を確認します。

## 結果と解釈

未実施です。

## 次に試すこと

rehearsalで忘却が抑えられたら、ratio 0.1または0.5を一つずつ比較します。抑えられなければ、次は会話データの短い応答を重視するサンプリングか、履歴切り詰めの別条件を試します。
