# 実験024：held-out会話生成の複数seed評価

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験023ではvalidation会話24例・seed 42でToken overlapを比較し、rehearsal 0.25が平均F1 0.1844でSFT-onlyの0.1557とpretrainingの0.0355を上回りました。ただし、24例と1 seedでは評価対象とsamplingの偶然を除けません。今回は評価対象の選択seedと生成seedを42・43・44へ増やし、同じ3 checkpointを各seedで比較します。

仮説は、rehearsalのF1優位とEOS停止率の高さがseedを変えても大きく崩れないというものです。反対に差がseed依存なら、実験023の結論は探索的な候補に留めます。各seedではvalidation JSONLから24例を決定的に選び、3モデルへ同じ選択例を与えます。生成はtemperature 0.8、top-k 40、最大64 tokenです。

比較対象は、Token予算pretraining、会話SFT-only、rehearsal ratio 0.25の各step 500 checkpointです。target本文をpromptへ含めず、EOSを除くToken overlap precision・recall・F1、生成Token数、EOS停止を保存します。Token overlapは意味的評価ではないため、seed間の傾向確認に限定します。

実験前のGitコミットは`511e748`（`exp: record heldout overlap metrics`）です。次のコマンドを、seedごとに3 checkpointへ実行します。

```bash
for seed in 42 43 44; do
  .venv/bin/python scripts/evaluate_chat_dataset.py \
    --config configs/token-budget-chat-sft-5m-smoke.toml \
    --checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
    --input artifacts/corpus/conversation-v1/validation.jsonl \
    --output artifacts/evaluations/token-budget-mixed-ja-5m-smoke-heldout-chat-seed-${seed}.json \
    --text-output artifacts/samples/token-budget-mixed-ja-5m-smoke/heldout-chat-seed-${seed}.txt \
    --examples 24 --max-new-tokens 64 --seed ${seed}
done
```

SFT-onlyとrehearsalも同じ形式で実行します。成功判定は各seedで3モデルの選択例ハッシュが一致し、全9 JSON・TXTが保存されることです。平均値とseedごとの差を記録し、rehearsalの優位が再現するかを判断します。

## 実験中の記録

未実施です。seedごとの例数、EOS停止数、平均生成長、overlap precision・recall・F1、選択例ハッシュを追記します。

## 結果と解釈

未実施です。

## 次に試すこと

rehearsalの傾向が再現すれば、次は相づち・定型挨拶を分離した層別評価へ進みます。再現しなければ、評価例数を増やす前にデータ分割と生成温度の影響を切り分けます。
