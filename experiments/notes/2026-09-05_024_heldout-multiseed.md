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

2026-09-05に3 seed × 3 checkpointの評価を完了しました。各seedの選択例ハッシュは、seed 42が`eff39e1a3ef62a87a0c3b353ca7a3e954149a17da755e254b08358f7d7f71389`、seed 43が`d7d2a00263455463d9f2a60812aecc608bf09fca35b3cea779d76eb2b3364378`、seed 44が`c658b99b14f254a42531eb8f7a7eed8c003acf01130a8e21a385317a17f7f111`でした。各seedで3モデルのハッシュが一致し、モデル間の比較対象は揃っています。

3 seed・72例を合算した集計は次のとおりです。

- pretraining：EOS停止72/72、平均生成4.60 token、precision 0.1423、recall 0.0388、F1 0.0543
- SFT-only：EOS停止62/72、平均生成25.17 token、precision 0.2178、recall 0.2372、F1 0.1659
- rehearsal 0.25：EOS停止71/72、平均生成14.28 token、precision 0.2463、recall 0.1951、F1 0.1757

rehearsalはSFT-onlyより平均生成長を10.89 token短くし、EOS停止を高く保ちながらprecisionを0.0285、F1を0.0098上げました。一方、recallは0.0422下がっています。seedを増やしても、rehearsalのF1がSFT-onlyとpretrainingの間で最も高い傾向は維持されましたが、差は大きくなく、意味的な正しさの証明ではありません。

各seedの全生成結果は、悪いcompletionやEOS未到達を含めて次のディレクトリへ保存しています。

- `artifacts/evaluations/*heldout-chat-seed-*.json`
- `artifacts/samples/*/heldout-chat-seed-*.txt`

## 結果と解釈

仮説の「rehearsalのF1優位とEOS停止率の高さがseedを変えても大きく崩れない」は、今回の3 seed・72例では支持されました。ただし、rehearsalはrecallを犠牲にしてprecisionと停止性を上げているため、すべての面でSFT-onlyを上回ったわけではありません。SFT-onlyは参照Tokenを広く含む長いcompletionを生成し、rehearsalはより短い返答へ制約されました。

この結果から、rehearsal ratio 0.25は次の実験の基準条件として採用します。しかし、overlap F1は同じTokenを出したかしか測らないため、自然な相づち、意味的な応答、話者の役割を評価できません。Issue #1の主目的に近づくには、層別promptと人手確認可能な小規模評価を追加する必要があります。

## 次に試すこと

rehearsalの傾向は再現しましたので、次は相づち・定型挨拶・質問・同意/不同意・話題転換へ分けた層別評価を追加します。その後、短い応答を過剰代表するsamplingを一つだけ導入し、F1と実際の内容対応が同時に改善するかを確認します。
