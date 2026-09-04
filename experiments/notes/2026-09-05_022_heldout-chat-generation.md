# 実験022：held-out会話履歴からの次発話生成

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。Issue #1の固定短文promptだけでは、学習時の入力分布と実際の会話履歴の差を評価できませんでした。そこで、会話validation splitの各会話から過去turnと次話者markerだけをpromptにし、含めていない次turnを生成する評価を追加します。Issue #1が求める「話者境界を保った会話形式」「固定promptだけでなく生成例の保存」に対応する評価です。[Issue #1](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)

仮説は、構造化promptで「こんにちは」へ偏ったSFT-onlyモデルでも、実際の履歴が長く入れば参照発話の条件に応じたcompletionが少し現れる可能性があるというものです。一方で、SFTデータの約69.6%がcontext 256へ収める際に左切り詰めになっているため、validation生成でも履歴が失われる影響が残ると予想します。

比較対象は同じTokenizer・モデル構造・seedで学習した次の3 checkpointです。

- pretraining：`token-budget-mixed-ja-5m-smoke/step_000500.npz`
- SFT-only：`token-budget-chat-sft-5m-smoke/step_000500.npz`
- rehearsal 0.25：`token-budget-chat-rehearsal-sft-5m-smoke/step_000500.npz`

validation JSONLからseed 42で最大24例を選び、各会話の2発話目以降を候補とします。target本文はpromptへ含めず、開始marker、履歴turn、履歴EOS、target speaker markerまでをToken化します。生成はtemperature 0.8、top-k 40、最大64 token、例ごとにseed 42から順に加算します。completionが空か、EOSで停止したか、生成Token数、参照本文をJSONとTXTへ残します。

実験前のGitコミットは、評価器実装後に記録します。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/evaluate_chat_dataset.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --input artifacts/corpus/conversation-v1/validation.jsonl \
  --output artifacts/evaluations/token-budget-mixed-ja-5m-smoke-heldout-chat.json \
  --text-output artifacts/samples/token-budget-mixed-ja-5m-smoke/heldout-chat.txt \
  --examples 24 --max-new-tokens 64 --seed 42

.venv/bin/python scripts/evaluate_chat_dataset.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-chat-sft-5m-smoke/step_000500.npz \
  --input artifacts/corpus/conversation-v1/validation.jsonl \
  --output artifacts/evaluations/token-budget-chat-sft-5m-smoke-heldout-chat.json \
  --text-output artifacts/samples/token-budget-chat-sft-5m-smoke/heldout-chat.txt \
  --examples 24 --max-new-tokens 64 --seed 42

.venv/bin/python scripts/evaluate_chat_dataset.py \
  --config configs/token-budget-chat-rehearsal-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-smoke/step_000500.npz \
  --input artifacts/corpus/conversation-v1/validation.jsonl \
  --output artifacts/evaluations/token-budget-chat-rehearsal-sft-5m-smoke-heldout-chat.json \
  --text-output artifacts/samples/token-budget-chat-rehearsal-sft-5m-smoke/heldout-chat.txt \
  --examples 24 --max-new-tokens 64 --seed 42
```

成功判定は、3 checkpointで同じ24例が選ばれ、target本文をpromptへ混入させず、JSON・TXTが保存されることです。自然さの優劣はcompletionの空率・停止率・参照との重なりを補助指標として使いますが、24例だけで会話能力の一般化を断定しません。

## 実験中の記録

未実施です。各checkpointの生成結果、空completion数、EOS停止数、平均生成長、代表例を追記します。

## 結果と解釈

未実施です。

## 次に試すこと

実データ履歴でもSFTの条件付き応答が弱ければ、短い応答を過剰代表しない層化sampling、あるいは教師モデルからの会話応答蒸留を検討します。実データ評価が改善する場合は、例数を増やして複数seedで再確認します。
