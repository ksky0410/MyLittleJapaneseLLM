# 実験101：Runpod 50M継続事前学習の低学習率比較

## 目的

実験100では、Runpod 40,000 step事前学習モデルへ未使用20M-tokenコーパスを追加する継続事前学習を開始したが、最大学習率`1e-4`へ到達した後に既存FineWeb validation lossが悪化した。実験101では初期checkpointと新規token列を固定し、最大学習率だけを`3e-5`へ下げる。これにより、既存の日本語能力を壊さずに新しい文書を取り込めるかを確認する。

## 事前仮説

継続事前学習ではランダム初期値からのpretrainingより小さい学習率が適切であり、`3e-5`ならstep 500以降のvalidation悪化が弱まる、またはvalidation lossが2.9未満へ下がると予想する。もし低学習率でも改善せず、新規データの分布とvalidationのずれが主因なら、学習率を下げてもlossの改善は限定的になる。

この実験も会話SFTではないため、質問への直接回答や自然な雑談の改善は別途評価する。事前学習終了後、良いcheckpointを会話SFTの基盤へ渡す。

## 条件

- 実施日：2026年9月7日
- 担当：Codex
- 初期checkpoint：実験098の`artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt`
- 初期checkpoint SHA-256：`83e8be941b645823efd1ae0a358d2c4521faa49b58de7696229298973bd25ac7`
- 追加train binary：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin`
- 追加train binary SHA-256：`f19878618870a487ce5b0aab6970d6d72b2ef71ab76ee79520e7c3fe3341dec1`
- 追加tokens：19,993,334
- モデル：dim 576、12層、9 heads、RoPE、LayerNorm、SwiGLU、context 256、50,207,616 parameters
- 学習：batch size 8、40,000 step、約81.92M提示tokens、seed 101
- optimizer：AdamW。重みだけを初期checkpointから読み込み、optimizer stateは初期化
- 学習率：`3e-5`から`3e-6`までcosine decay、warmup 1,000 step、weight decay 0.1
- 評価：FineWeb2 Japanese testを500 stepごとに20 evaluation batchesで測定し、同じpromptの生成文を保存
- GPU：Runpod A40 Secure。前回実測速度は約14.5 step/秒、約$0.49/時

## 実行前の成功基準

step 500以降のFineWeb validation lossが実験100のstep 500 `2.927500`から悪化せず、40,000 stepまでに実験098のbest `2.973267`を安定して下回ることを有望な結果とする。validationが改善しても、固定chat-testの生成全文と新しい会話testで自然さを確認する。NaN、OOM、shape errorなく完走し、500 stepごとのmetrics・生成文・checkpoint metadataを保存する。

## 開始前の実行記録

このノートは実験100の高学習率条件を停止した直後に作成する。実験100の失敗結果、データ準備、入力hashは上書きせず、実験101の出力先を分離する。開始前に設定のSHA-256、Runpod Pod ID、bundle SHA-256、GPUを追記し、学習開始後は500 stepを超えて記録を空けない。

## 実行コマンド

```bash
PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5.toml \
  --initial-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt \
  --device cuda
```

## 結果

学習中にmetrics、生成、checkpoint、GPU速度、料金、停止理由を追記する。終了後にbest step、FineWeb・各domain validation、固定chat-test、新規会話test、人手レビュー用サンプルを記録し、会話SFTへ渡すcheckpointを明記する。
