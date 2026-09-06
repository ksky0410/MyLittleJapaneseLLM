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
- 設定SHA-256：`a719b97d30c602076ace58c238d647b19b3160dd81f2a0f2ce9fc3455d62463b`
- Runpod Pod：`j9c46julmtbcb4`（CA-MTL-1）
- 転送bundle SHA-256：`beb3beb7339d2e81946d1fca61f1234ef3cb3bb1e97db5a2a750c8a22c3efac4`

## 実行前の成功基準

step 500以降のFineWeb validation lossが実験100のstep 500 `2.927500`から悪化せず、40,000 stepまでに実験098のbest `2.973267`を安定して下回ることを有望な結果とする。validationが改善しても、固定chat-testの生成全文と新しい会話testで自然さを確認する。NaN、OOM、shape errorなく完走し、500 stepごとのmetrics・生成文・checkpoint metadataを保存する。

## 開始前の実行記録

このノートは実験100の高学習率条件を停止した直後に作成する。実験100の失敗結果、データ準備、入力hashは上書きせず、実験101の出力先を分離する。開始前に設定のSHA-256、Runpod Pod ID、bundle SHA-256、GPUを追記し、学習開始後は500 stepを超えて記録を空けない。

### 開始前の追加準備

医師国家試験データを後段SFTで使える質問回答形式へ変換する`prepare_medical_sft.py`とテストを追加した。元の`artifacts/corpus/medical-qb-v2`は読み取り専用で扱い、正解欄が空の問題と、context 256で質問または回答が切れる問題を学習用から除外する。変換後のSFT候補はtrain 2,945例、response 172,545 tokens、validation 162例、response 9,277 tokensとなり、SFT配列の`truncated_example_count`はtrain・validationとも0である。除外件数と問題番号は`artifacts/corpus/medical-qb-sft-v1/manifest.json`へ記録した。

Runpodの同じA40 Pod `j9c46julmtbcb4`上で、実験101の2,000 step pilotを本番と別の`exp101-pilot`へ実行した。step 500、1,000、1,500、2,000のFineWeb validation lossはそれぞれ2.931313、2.914403、2.902358、2.896122となり、step 1の2.973276から一貫して改善した。step 2,000のbest checkpoint SHA-256は`e2e23d652fd365716c5f97b68f8da8144332aa1ddb3d9eba465b3f4fe229759f`である。高学習率の実験100とは異なり、step 500以降にvalidation lossが悪化しなかったため、このpilotを有望と判断する。metrics、best metadata、生成サンプルは`artifacts/checkpoints/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5-pilot/`と対応するsamplesディレクトリへ回収した。

本番はpilotのstep 2,000 best重みから`--start-step 2000`で継続し、累積40,000 stepまで学習する。これによりpilotの2,000 stepを捨てず、学習率scheduleと乱数の進行も連続させる。

### 本番開始の追記

pilotの入力と結果を確認後、同じRunpod Pod `j9c46julmtbcb4`で本番プロセスをPID 808として起動した。学習ログはPod上の`/workspace/exp100/exp101.log`、軽量metricsと生成文は設定どおりの`issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5`ディレクトリへ保存する。pilotのstep 2,000 best重みを初期値に使い、`--start-step 2000`で累積stepを継続している。終了までPodを保持し、途中で500 step以上記録を空けない。

## 実行コマンド

```bash
PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5.toml \
  --initial-checkpoint artifacts/checkpoints/exp101-pilot/best.pt \
  --start-step 2000 \
  --device cuda
```

## 結果

学習中にmetrics、生成、checkpoint、GPU速度、料金、停止理由を追記する。終了後にbest step、FineWeb・各domain validation、固定chat-test、新規会話test、人手レビュー用サンプルを記録し、会話SFTへ渡すcheckpointを明記する。

### 途中経過（2026年9月7日、累積step 5,500）

Runpod上の本番プロセスは継続中で、NaN、OOM、shape errorは発生していない。pilotのstep 2,000 best重みから再開した後、FineWeb validation lossはstep 2,500の2.894975からstep 5,500の2.876570まで緩やかに改善した。step 5,000では2.878091、step 5,500ではPerplexity 17.753284である。実験098のbest validation loss 2.973267と比べ、現時点でlossは約3.3%低く、Perplexityは約9%低い。ただし、これは事前学習評価の改善であり、自然な会話応答の改善を意味しないため、後段の会話SFTと生成比較を省略しない。

step 5,500時点のtrain lossは2.395376、学習率は`2.9123e-5`である。step 5,000から5,500までの経過時間は約33秒で、A40は約1.9GBのGPUメモリを使用し、異常なメモリ増加は見られない。metricsと生成サンプルはPod上の設定出力先に保存されており、学習完了後に回収する。

step 6,000、6,500、7,000のvalidation lossはそれぞれ2.874430、2.874084、2.870859となった。step 7,000のPerplexityは17.652169、train lossは2.843273、学習率は`2.8454e-5`である。step 5,500から7,000までの間も学習プロセスは正常に動作し、validation lossは悪化していない。step 7,000時点では、実験098のbest loss 2.973267に対して約3.4%低い。

その後もstep 7,500、8,000、8,500、9,000、9,500のvalidation lossはそれぞれ2.868662、2.867681、2.864839、2.863642、2.859826となった。step 9,500のPerplexityは17.458489、train lossは2.910493、学習率は`2.6958e-5`である。step 7,000以降もlossは一度も悪化せず、step 9,500時点では実験098のbest lossより約3.8%低い。Runpod A40上のプロセスは正常に継続している。

step 10,000、10,500、11,000、11,500、12,000のvalidation lossはそれぞれ2.863249、2.861999、2.860042、2.856804、2.855655となった。step 12,000のPerplexityは17.385823、train lossは2.655377、学習率は`2.5039e-5`である。validation lossは引き続き改善し、step 12,000時点では実験098のbest lossより約4.0%低い。SFT用データのRunpod転送も完了し、事前学習終了後に同じPodで続けられる状態にした。
