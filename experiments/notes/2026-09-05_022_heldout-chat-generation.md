# 実験022：held-out会話履歴からの次発話生成

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。Issue #1の固定短文promptだけでは、学習時の入力分布と実際の会話履歴の差を評価できませんでした。そこで、会話validation splitの各会話から過去turnと次話者markerだけをpromptにし、含めていない次turnを生成する評価を追加します。Issue #1が求める「話者境界を保った会話形式」「固定promptだけでなく生成例の保存」に対応する評価です。[Issue #1](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)

仮説は、構造化promptで「こんにちは」へ偏ったSFT-onlyモデルでも、実際の履歴が長く入れば参照発話の条件に応じたcompletionが少し現れる可能性があるというものです。一方で、SFTデータの約69.6%がcontext 256へ収める際に左切り詰めになっているため、validation生成でも履歴が失われる影響が残ると予想します。

比較対象は同じTokenizer・モデル構造・seedで学習した次の3 checkpointです。

- pretraining：`token-budget-mixed-ja-5m-smoke/step_000500.npz`
- SFT-only：`token-budget-chat-sft-5m-smoke/step_000500.npz`
- rehearsal 0.25：`token-budget-chat-rehearsal-sft-5m-smoke/step_000500.npz`

validation JSONLからseed 42で最大24例を選び、各会話の2発話目以降を候補とします。target本文はpromptへ含めず、開始marker、履歴turn、履歴EOS、target speaker markerまでをToken化します。生成はtemperature 0.8、top-k 40、最大64 token、例ごとにseed 42から順に加算します。completionが空か、EOSで停止したか、生成Token数、参照本文をJSONとTXTへ残します。

実験前のGitコミットは`3e9ea2a`（`eval: add heldout chat generation`）です。使用コマンドは次のとおりです。

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

2026-09-05に3 checkpointの評価を完了しました。3モデルともseed 42で同じ24例が選ばれ、選択例の識別子とtarget indexのSHA-256は`eff39e1a3ef62a87a0c3b353ca7a3e954149a17da755e254b08358f7d7f71389`で一致しました。target本文をpromptへ含めず、各結果には履歴prompt、参照発話、completionを保存しています。エラーや空completionはありませんでした。

生成の集計は次のとおりです。

- pretraining：EOS停止24/24、平均生成4.92 token
- SFT-only：EOS停止20/24、平均生成27.29 token
- rehearsal 0.25：EOS停止23/24、平均生成14.50 token

pretrainingは「そうなんですね。」「なるほど!」など短い汎用相づちが中心でした。SFT-onlyは「確かに」など会話らしい接続を出す例が増えましたが、文法が崩れたまま長く続く例が多く、参照内容に十分対応していません。rehearsalはSFT-onlyより短く停止しやすくなりましたが、「そうですね。」などの汎用返答へ戻る例もあり、内容理解の改善とは断定できません。

代表例として、参照「ああ、朝は時間が進むのが速く感じますよね。」に対し、pretrainingは「それのか?」、SFT-onlyは「そうですね。私はちょっと海道の方で…」、rehearsalは「確かに。どんな味ですか?」でした。rehearsalは会話形式を保ちながらも、参照内容とはずれています。これは、SFTでcompletionを長くする能力は学習できても、5M級・500 step・現在のデータ形式では条件付きの意味対応が弱いことを示す探索的な結果です。

成果物は次のとおりです。

- [pretraining held-out JSON](../../artifacts/evaluations/token-budget-mixed-ja-5m-smoke-heldout-chat.json)
- [pretraining held-out TXT](../../artifacts/samples/token-budget-mixed-ja-5m-smoke/heldout-chat.txt)
- [SFT-only held-out JSON](../../artifacts/evaluations/token-budget-chat-sft-5m-smoke-heldout-chat.json)
- [SFT-only held-out TXT](../../artifacts/samples/token-budget-chat-sft-5m-smoke/heldout-chat.txt)
- [rehearsal held-out JSON](../../artifacts/evaluations/token-budget-chat-rehearsal-sft-5m-smoke-heldout-chat.json)
- [rehearsal held-out TXT](../../artifacts/samples/token-budget-chat-rehearsal-sft-5m-smoke/heldout-chat.txt)

## 結果と解釈

「学習時と同じ会話形式ならSFTの条件付き応答が改善する」という仮説は、一部だけ確認されました。SFT-onlyはpretrainingよりcompletionが長くなり、自然な接続語を含む例も現れましたが、生成長の増加と意味的な正答は一致しませんでした。rehearsalはEOS停止率と生成長をpretraining寄りへ戻し、domain lossで確認した忘却抑制とも整合しますが、参照発話への対応は十分ではありません。

したがって現時点のボトルネックは、単なる会話形式やloss maskingではなく、限られた学習stepで「どの発話へ何を返すか」を学ぶデータ量・モデル容量・学習例の構成です。24例評価は定性的な探索であり、一般化の証明ではありませんが、次の実験では短い応答を単に増やすだけでなく、実際の履歴とtargetの意味対応を検証できる評価指標を追加します。

## 次に試すこと

次は、validation会話のtarget発話と生成completionのToken overlapやROUGE-Lのような簡易指標を追加し、目視だけでなく内容対応も比較します。その後、短い応答だけを過剰代表しない層化sampling、または教師モデルからの会話応答蒸留を一つずつ検討します。実データ評価の例数を増やす前に、まず指標の再現性を固めます。
