# 実験015：Issue #1 固定会話プロンプト比較

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。この実験では、GitHub Issue #1で提案されている会話らしさの確認方法を、再現可能な固定プロンプト評価として最初に実装します。Issue #1の主眼は、知識量だけではなく、短い発話、相づち、くだけた表現、助詞や文末表現を比較することです。[Issue #1](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)

今回の仮説は、拡張した一般日本語・会話・医療の混合コーパスで同じ500 step学習した場合、RoPEモデルはabsolute position embeddingモデルより、短い入力からの局所的な続きを少し安定して生成する可能性がある、というものです。ただし、これまでの比較は位置表現の変更に伴う初期化順序やMLXのコンパイル順序まで完全には揃っていないため、生成結果だけで因果的な優劣を断定しません。まずは探索的な観察として扱います。

比較対象は次の2つです。

- absolute：`configs/expanded-mixed-ja-5m-smoke.toml`、step 500
- RoPE：`configs/expanded-rope-mixed-ja-5m-smoke.toml`、step 500

どちらも、同じ`mixed-ja-80-10-10-v2`の学習Token列、同じTokenizer、dim 240・6層・6 heads・context 256、batch size 8、学習率3e-4、seed 42、500 stepです。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`で、モデルの生成条件はtemperature 0.8、top-k 40、最大160 tokenです。各プロンプトではseedを42から順に加算し、モデル間で同じプロンプトに同じseedを割り当てます。

Issue #1の候補を参考に、`experiments/prompts/issue-1-chat-v1.json`へ次の8入力を固定しました。「まじで」「それな」「今日なにしてた？」「やば」「なんかさ」「いやそれは」「おつかれ」「明日ひま？」です。出力は短い返答、相づち、短い会話ターン、くだけた表現、文脈接続、不同意、別れ際、誘いへの反応を観察します。

実行前のGitコミットは`c6a0d0c`（`eval: add fixed chat prompt comparison`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/evaluate_chat_prompts.py \
  --config configs/expanded-mixed-ja-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/expanded-mixed-ja-5m-smoke/step_000500.npz \
  --prompt-file experiments/prompts/issue-1-chat-v1.json \
  --output artifacts/evaluations/expanded-mixed-ja-5m-smoke-chat.json \
  --text-output artifacts/samples/expanded-mixed-ja-5m-smoke/chat-issue-1.txt

.venv/bin/python scripts/evaluate_chat_prompts.py \
  --config configs/expanded-rope-mixed-ja-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/expanded-rope-mixed-ja-5m-smoke/step_000500.npz \
  --prompt-file experiments/prompts/issue-1-chat-v1.json \
  --output artifacts/evaluations/expanded-rope-mixed-ja-5m-smoke-chat.json \
  --text-output artifacts/samples/expanded-rope-mixed-ja-5m-smoke/chat-issue-1.txt
```

成功判定は、両checkpointについて8プロンプトの出力とJSONが保存され、同じ条件で再実行したときにJSONの生成結果が一致することです。自然さの優劣は、この実験だけでは確定せず、後続の会話専用SFTや複数seed比較の候補を決める材料とします。

## 実験中の記録

2026-09-05に予定どおりabsoluteとRoPEのcheckpointを順に評価しました。MLXの実行でエラーやメモリ不足は発生しませんでした。absolute側のJSON SHA-256は`eeb914373a64e738daf548950bad0d9930154b54990c550bdea19a663edf34c3`で、同じコマンドを一時ディレクトリへ再実行したJSONと完全一致しました。したがって、少なくともこの評価スクリプトとseedの範囲では生成結果を再現できます。

RoPE側の生成も正常に完了しました。8プロンプトすべてについて結果が保存されましたが、`今日なにしてた？`と`明日ひま？`は両モデルとも空のcompletionになりました。出力は途中で省略せず、次のファイルへ保存しています。

- [absolute JSON](../../artifacts/evaluations/expanded-mixed-ja-5m-smoke-chat.json)
- [absolute生成テキスト](../../artifacts/samples/expanded-mixed-ja-5m-smoke/chat-issue-1.txt)
- [RoPE JSON](../../artifacts/evaluations/expanded-rope-mixed-ja-5m-smoke-chat.json)
- [RoPE生成テキスト](../../artifacts/samples/expanded-rope-mixed-ja-5m-smoke/chat-issue-1.txt)

## 結果と解釈

validation lossだけを見ると、absoluteのgeneral validation lossは5.7359479268、RoPEは5.5338133176でした。固定会話プロンプトでは、absoluteは「いやそれは」に対して「それは今はそうでない。」を生成し、RoPEは「それな」に対して「こきいです。」、「おつかれ」に対して「します。」を生成しました。一方で、absoluteは「まじで」「それな」「なんかさ」で医療問題の断片を長く続け、RoPEも「まじで」「いやそれは」などで医療定型文や崩れた文を生成しました。両方に医療形式の強い漏れがあり、会話自然性の改善とは解釈できません。

今回の結果は、混合コーパスの文字比率が一般34.17%、会話45.18%、医療20.65%であっても、500 step・約5M級の短い学習では会話の短い応答能力が生じないことを示す探索的な失敗例です。特に会話データを通常のnext-token pretrainingとして扱い、話者境界や応答側lossの制御をしていないことが、次の改善候補です。また、RoPEのvalidation lossが低いことと、会話promptへの応答が自然であることは一致しませんでした。

## 次に試すこと

まずIssue #1の会話データをpretraining混合から切り分け、話者境界を保った会話SFT用形式と、応答側だけへlossをかけるmaskingを導入します。そのうえで同じ8プロンプトを再評価し、通常の混合pretrainingとの差を確認します。並行して、source比率を文字数ではなくtoken予算で制御し、医療データが短い単位数で過剰に効く問題を減らします。
