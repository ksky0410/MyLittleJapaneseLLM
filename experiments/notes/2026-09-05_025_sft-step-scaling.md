# 実験025：会話SFTの学習step数スケーリング

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験019〜024では、5M級モデルを500 stepだけ会話SFTし、rehearsalによる忘却抑制を確認しました。しかしtrain batchで実際に見るSFT例は約4,000件に過ぎず、396,966例ある会話trainの一部しか見ていません。今回は、SFT-onlyとrehearsal 0.25を同じ2,000 stepまで延長し、500 stepで見えた差が学習不足によるものか、方法の差として残るかを調べます。

仮説は、学習stepを増やすとSFT-onlyの条件付き応答とheld-out overlap F1が改善する一方、通常domain lossの忘却は強くなる可能性があり、rehearsalはその悪化を抑えるというものです。SFT-onlyとrehearsalを同じbase checkpointから独立に再開し、差分はrehearsal objectiveの有無だけにします。

両方ともモデルはdim 240・6層・6 heads・context 256・absolute position embedding、Tokenizer・SFT data・seed 42・batch size 8・学習率5e-5・minimum learning rate 5e-6・warmup 50・weight decay 0.01です。SFT validation lossは応答maskだけで計算します。rehearsal側はpretraining Token列を25%の独立full lossとして結合します。`--max-steps 2000`で学習stepだけを設定ファイルの500から変更します。

学習後は、SFT mask validation、通常のgeneral・conversation・medical validation、Issue #1構造化prompt、held-out validation会話24例のoverlap指標を両モデルで比較します。長時間実験のため、1000 stepを超えてmetrics記録を空けないよう、既存の100 step間隔を保ちます。SFT-onlyが自然になったように見えても、生成TXT・domain loss・overlap F1を分けて評価します。

実験前のGitコミットは`10b5fbd`（`exp: record multiseed heldout chat eval`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --output-dir artifacts/checkpoints/token-budget-chat-sft-5m-2k \
  --samples-dir artifacts/samples/token-budget-chat-sft-5m-2k \
  --max-steps 2000

.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-rehearsal-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --rehearsal-tokens artifacts/tokens/mixed-ja-token-budget-1m-train.bin \
  --rehearsal-ratio 0.25 \
  --output-dir artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-2k \
  --samples-dir artifacts/samples/token-budget-chat-rehearsal-sft-5m-2k \
  --max-steps 2000
```

成功判定は両モデルが2,000 stepまで完了し、100 step間隔のmetrics、全stepの生成サンプル、summary、checkpoint metadataが保存されることです。SFT-onlyとrehearsalの優劣は、事前の仮説に合わない結果も含めて評価します。

## 実験中の記録

学習は2026-09-05に開始し、SFT-onlyとrehearsal 0.25を並列に実行しています。100 step間隔でmetricsを保存し、1,000 step時点ではSFT-onlyのvalidation lossが4.4665（perplexity 87.05）でした。rehearsal側は900 step時点でvalidation loss 4.5039（perplexity 90.37）であり、途中経過ではSFT-onlyがやや先行しています。これは学習率が減衰中の途中値であり、忘却とheld-out応答の結果は学習完了後に評価します。

1,000 stepまでのログは各モデルの`metrics.jsonl`に保存されています。学習中の固定prompt生成も各stepのTXTとして保存される設定です。

## 結果と解釈

未実施です。

## 次に試すこと

2,000 stepで改善が見えた場合は、学習量を増やす前に短い会話・質問・相づちの層別評価を追加します。改善しなければ、次は学習データのtarget turn選択または教師モデル蒸留へ進みます。
