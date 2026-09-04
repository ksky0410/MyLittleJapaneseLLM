# 実験026：短い応答を層化した会話SFT

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験025では2,000 stepまでSFTを延長しましたが、Issue #1の固定promptは「こんにちは」「こんばんは」などの定型挨拶へ偏り、短い入力へ内容に応じて返す能力は確認できませんでした。一方、rehearsal ratio 0.25は通常domainの忘却を抑え、held-out会話のEOS停止とToken overlap F1もSFT-onlyより良好でした。

今回はモデル構造、Tokenizer、学習率、学習step、rehearsal条件を変えず、SFT batch内の例の選び方だけを変更します。会話SFT trainのloss maskを調べると、EOSを含む応答Token数が8以下の例は82,904/396,966例（20.9%）でした。短い返答が少数派であるため、各batchのSFT例8行のうち4行を応答Token数8以下から、残り4行をそれ以外の例から選ぶ層化samplingを導入します。短い例の選択は既存のランダム選択と同じく復元抽出とし、validation配列とrehearsal Token列は変更しません。

仮説は、短い応答を過剰代表するとIssue #1の「まじで」「それな」「やば」「おつかれ」のような短文promptで、過度に長いcompletionを出す割合が下がり、カテゴリ別の短文応答評価が改善するというものです。rehearsal ratio 0.25を併用するため、general・conversation・medicalの通常domain lossは実験025のrehearsal 0.25に近く保たれると予想します。反対に、短文へ寄せすぎるとheld-outの長い発話でrecallが低下する可能性があります。

比較対象は、実験025で得た通常rehearsal 0.25のstep 2,000 checkpointです。新条件は同じbase checkpointから、`short_response_ratio=0.5`、`short_response_max_tokens=8`、rehearsal ratio 0.25、最大2,000 stepで学習します。model dim 240、6層、6 heads、context 256、absolute position embedding、batch size 8、learning rate 5e-5、minimum learning rate 5e-6、warmup 50、weight decay 0.01、seed 42は維持します。

開始前のGitコミットは、短文sampling機能を実装する直前の`992c16c`（`exp: record 2k SFT scaling results`）です。実装後、CLIとbatch samplingのテスト、全テスト、ruffを通過しました。固定prompt評価にはカテゴリ別の空出力・EOS停止・平均生成Token数の集計も追加し、今回の評価JSONとTXTへ保存します。

実験条件を確定する実行コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-rehearsal-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --rehearsal-tokens artifacts/tokens/mixed-ja-token-budget-1m-train.bin \
  --rehearsal-ratio 0.25 \
  --short-response-ratio 0.5 \
  --short-response-max-tokens 8 \
  --output-dir artifacts/checkpoints/token-budget-chat-rehearsal-short-sft-5m-2k \
  --samples-dir artifacts/samples/token-budget-chat-rehearsal-short-sft-5m-2k \
  --max-steps 2000
```

この実装状態のテストは49 passed、ruff checkは成功、ruff format --checkも成功でした。実験開始前に作業treeをcleanにし、実装commitをこのノートへ追記します。

成功判定は、新条件が2,000 stepまで完走し、100 step間隔のmetrics、checkpoint metadata、全stepの生成TXT、summary、domain評価、構造化固定prompt評価、held-out 24例評価を保存できることです。短文層化の効果は、固定promptのカテゴリ別出力、生成長、EOS停止、held-out overlap F1、通常domain lossを合わせて判断し、改善しない場合もそのまま記録します。

## 実験中の記録

未実施です。コード実装とテストの結果、学習中のstepごとのlossおよび生成をここへ追記します。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。短文samplingの効果が確認できた場合は、短文以外のカテゴリへsamplingを広げず、まずrehearsal比率との組み合わせを検討します。効果がなければ、同じ条件でsamplingを長く回す前に、応答内容の重複・履歴長・話者markerの影響を調べます。
