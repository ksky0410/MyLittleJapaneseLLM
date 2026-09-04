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

未実施です。各checkpointの完了、出力SHA-256、空completion数、内容の傾向を追記します。

## 結果と解釈

未実施です。

## 次に試すこと

テンプレート版でも改善しない場合は、SFTデータの長い履歴切り詰めと、会話データに一般日本語を混ぜる忘却対策を別実験にします。改善した場合も、複数seedと実際のvalidation会話で再確認します。
