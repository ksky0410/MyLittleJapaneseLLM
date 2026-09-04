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

2026-09-05に3 checkpointを同じ24例で再評価しました。指標追加後のJSON SHA-256は、pretrainingが`62cf553535701ea6de1c85720b8c625759459617a595ff0b001d418856bc6fed`、SFT-onlyが`34f333925416debc56cd874e3fec1fe06ece4272b23bf4518966205d5ae6cd20`、rehearsalが`e8c10a1f05ecc6b3e284c5c581a615259711b1f75bec0aee1caa3a713fb304a5`です。選択例の識別子とtarget indexは前実験と同じSHA-256`eff39e1a3ef62a87a0c3b353ca7a3e954149a17da755e254b08358f7d7f71389`でした。

平均値は次のとおりです。

- pretraining：平均生成4.92 token、EOS停止24/24、precision 0.0858、recall 0.0234、F1 0.0355
- SFT-only：平均生成27.29 token、EOS停止20/24、precision 0.1858、recall 0.2506、F1 0.1557
- rehearsal 0.25：平均生成14.50 token、EOS停止23/24、precision 0.2419、recall 0.2124、F1 0.1844

SFT-onlyはcompletionを長くしてrecallを上げましたが、rehearsalはそれより短く、precisionとF1が高くなりました。これは24例・1 seedの探索結果にすぎず、Token overlapは意味理解の証明ではありませんが、rehearsalが単なる生成短縮ではなく、参照発話と共有するTokenを相対的に増やしたことを示します。

## 結果と解釈

「SFT-onlyの長文化が内容対応を伴わなければ、生成長は増えてもoverlap F1は改善しない」という仮説は、今回の値では支持されませんでした。SFT-onlyのF1はpretrainingより改善し、held-out会話の内容へ部分的に近づいた可能性があります。ただしrehearsalのF1がさらに高く、EOS停止率も高かったため、rehearsalは内容対応と暴走抑制のバランスを改善した候補です。

一方、Token overlapは「参照をそのまま繰り返した」場合にも上がります。したがって、これを自然さや意味的正しさとして扱わず、既存の生成TXT・domain loss・会話形式と合わせて解釈します。次の検証では例数とseedを増やし、長さ別にF1を分解し、単純な相づちが高得点を得ていないかを確認します。

## 次に試すこと

まず24例から例数を増やし、少なくとも複数seedでoverlap F1・生成長・EOS停止を再計算します。その後、相づちや定型挨拶を分離した層別評価を追加し、学習データのtargetと履歴の条件付けが本当に効いているかを確認します。
