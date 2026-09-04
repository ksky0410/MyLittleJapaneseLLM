# 実験020：学習時と同じ会話テンプレートでの固定prompt評価

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験019ではrawな短文だけをモデルへ渡して会話SFT後の生成を評価しましたが、SFT学習時には開始marker、話者marker、EOSを含む構造化入力を使っていました。そこで今回は、Issue #1の8固定promptを学習時と同じ形式へレンダリングし、raw評価との差を確認します。

仮説は、SFTモデルがraw入力ではなく、学習時と同じ`<|startofconversation|><|speaker:A|>発話<EOS><|speaker:B|>`の形式でのみ改善を示す可能性があるというものです。比較対象は、同じToken予算pretrainingを行ったSFT前のabsolute checkpointと、実験019の500 step SFT checkpointです。両方へ同じTokenizer、同じ8 prompt、temperature 0.8、top-k 40、各prompt seed 42〜49を適用します。

テンプレート版では、文字列をそのままencodeするのではなく、学習データ整形と同じ単位で開始marker、話者A marker、prompt本文、EOS、話者B markerをToken列へ連結します。EOSはTokenizerのID 3を使用します。JSONには元のpromptとレンダリング表現、completion、Token数を保存し、TXTには全出力を保存します。

実験前のGitコミットは`b16790e`（`eval: support structured chat prompts`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/evaluate_chat_prompts.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --prompt-file experiments/prompts/issue-1-chat-sft-v1.json \
  --output artifacts/evaluations/token-budget-mixed-ja-5m-smoke-chat-sft-format.json \
  --text-output artifacts/samples/token-budget-mixed-ja-5m-smoke/chat-issue-1-sft-format.txt

.venv/bin/python scripts/evaluate_chat_prompts.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-chat-sft-5m-smoke/step_000500.npz \
  --prompt-file experiments/prompts/issue-1-chat-sft-v1.json \
  --output artifacts/evaluations/token-budget-chat-sft-5m-smoke-chat-sft-format.json \
  --text-output artifacts/samples/token-budget-chat-sft-5m-smoke/chat-issue-1-sft-format.txt
```

成功判定は両checkpointで8件のJSON・TXTが保存され、同一条件の再実行で結果が一致することです。テンプレート版でSFTモデルの出力が改善しても、8件だけの定性的な観察であり、会話能力の一般化とは断定しません。空出力や崩れた出力もそのまま残します。

## 実験中の記録

2026-09-05に両checkpointの評価を完了しました。エラーはなく、両方とも8 promptすべてでcompletionが生成され、空completionは0件でした。pretraining側JSONのSHA-256は`0cd7d2505f60b927e4c5006523da7bf87a0f13bff074e7f67f556f764ed9f378`、SFT側JSONのSHA-256は`53325b3a4f0623e04ad0cf62f66ccfeea9c204041fdd48c45906e39d5d24f4e5`です。

pretraining側は「まじで」への「おー」、「今日なにしてた？」への「そうですね。」、「いやそれは」への「そうですね」など、短いcompletionと長いcompletionが混在しました。SFT側は8件中7件が「こんにちは」を含むcompletionになり、入力に応じた使い分けよりも、会話応答の頻出パターンを強く出しました。「明日ひま？」だけは「@っていました」となりました。SFT側の出力は形式に適合したものの、内容の自然さと条件付き応答は不十分です。

保存先は次のとおりです。

- [pretraining構造化prompt JSON](../../artifacts/evaluations/token-budget-mixed-ja-5m-smoke-chat-sft-format.json)
- [pretraining構造化promptテキスト](../../artifacts/samples/token-budget-mixed-ja-5m-smoke/chat-issue-1-sft-format.txt)
- [SFT構造化prompt JSON](../../artifacts/evaluations/token-budget-chat-sft-5m-smoke-chat-sft-format.json)
- [SFT構造化promptテキスト](../../artifacts/samples/token-budget-chat-sft-5m-smoke/chat-issue-1-sft-format.txt)

## 結果と解釈

仮説の前半、すなわちSFTモデルが学習時と同じ構造化形式でのみraw形式とは異なる挙動を示す可能性は確認できました。しかし、SFT側のcompletionはpromptに応じて内容を変えるより「こんにちは」へ崩れており、Issue #1が求める短い会話の自然さが改善したとは言えません。SFT前モデルも8件すべてでcompletionを出したため、空出力の解消だけを成果とみなすこともできません。

また、SFT後の通常domain loss悪化と合わせると、会話データだけを500 step追加する設定は、一般日本語・医療・会話の保持と条件付き応答の両方で不十分です。次は会話SFTの各batchへ一般pretrainingデータを少量混ぜ、忘却を抑えながら応答maskを維持できるかを検証します。比較対象と差分を一つに絞ります。

## 次に試すこと

まず会話SFTの各batchへ一般pretraining batchを一定割合で混ぜる「rehearsal」方式を実装し、SFT-onlyとの違いを比較します。改善した場合も、複数seedと実際のvalidation会話で再確認します。
