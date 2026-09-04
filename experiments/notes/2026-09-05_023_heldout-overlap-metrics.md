# 実験023：held-out会話生成へのToken overlap指標追加

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験022では、held-out会話24例に対する生成を保存し、SFT-onlyがcompletionを長くする一方、rehearsalが生成長とEOS停止をpretraining側へ戻すことを確認しました。ただし、生成が長くなったことと参照発話へ対応したことを区別できていませんでした。

今回は、EOSを除いた参照発話と生成completionのmultiset Token overlap precision・recall・F1を追加します。仮説は、SFT-onlyの長文化が内容対応を伴っていなければ、生成長は増えてもoverlap F1は改善しないというものです。rehearsalは短くなった分、recallは低くなる可能性がありますが、genericな長文化を抑えたかを併せて判断します。

比較対象は実験022と同じpretraining、SFT-only、rehearsal 0.25の3 checkpointです。validation JSONLの同じ24例、seed 42、temperature 0.8、top-k 40、max 64 tokenを使います。指標は表面的なToken一致に過ぎず、意味的な正しさや自然さの証明ではありません。数値は目視確認と併用します。

実験前のGitコミットは`29d5641`（`eval: add heldout overlap metrics`）です。各評価は次の既存コマンドを再実行し、出力JSONとTXTを更新します。

```bash
.venv/bin/python scripts/evaluate_chat_dataset.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --input artifacts/corpus/conversation-v1/validation.jsonl \
  --output artifacts/evaluations/token-budget-mixed-ja-5m-smoke-heldout-chat.json \
  --text-output artifacts/samples/token-budget-mixed-ja-5m-smoke/heldout-chat.txt \
  --examples 24 --max-new-tokens 64 --seed 42
```

SFT-onlyとrehearsalも同じcheckpoint差し替えで実行します。成功判定は3モデルの選択例が一致し、各結果に3つのoverlap指標が有限値で保存されることです。評価JSONの更新自体も、既存の生成結果を上書きした変更として記録します。

## 実験中の記録

未実施です。モデルごとの平均precision・recall・F1、EOS停止数、平均生成長、代表例を追記します。

## 結果と解釈

未実施です。

## 次に試すこと

overlap F1が生成長と連動しない場合は、次に文字列編集距離やROUGE-Lを追加する前に、簡易指標が有効な差分を示すかを確認します。SFT-onlyがoverlapでも改善しない場合は、学習データのtargetと履歴の条件付けを見直します。
