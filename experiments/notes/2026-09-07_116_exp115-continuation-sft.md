# 実験116：exp115 checkpointの低学習率SFT継続

## 実施前の計画

### 目的

実験115の最良checkpointを初期値として、同じ一般会話・通常医療・answer-focus医療のSFTを低い学習率で追加8,000 step行う。実験115では8,000 stepの最後までvalidation lossが改善し続け、一般会話の固定評価も過去条件より良かったため、学習stepを増やすことが自然な日本語と質問応答能力の改善につながるかを検証する。

### 仮説

exp115のbest checkpointから急に大きな学習率へ戻さず、2e-6から2e-7へゆっくり減衰させれば、既に得た会話形式を壊さず、語彙選択や文脈への適合が少し改善すると予想する。反対に、同じデータをさらに8,000 step見せることで訓練例の定型句を過剰に繰り返し、一般会話F1や医療正答率が下がる可能性もある。validation lossだけでなく、固定一般会話48例、医療162例、生成全文を用いて判断する。

### 比較条件

exp115とデータ・モデル・評価器を揃え、初期checkpointと継続学習の学習率・seed・sample promptだけを変更する。

- 初期checkpoint：exp115 best、step 8,000、重みSHA-256 `e0c317ff57d8199a04c05f4751742367183701c1472b2c297a156f40e19beb5a`
- 学習データ：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 継続後の累積step：16,000。追加stepは8,000
- batch size：8。学習率：2e-6から2e-7。warmup：0。weight decay：0.01。seed：116
- 設定：`configs/issue1-exp115-continuation-sft-runpod-16k.toml`

### 成功・失敗の判定

NaN、OOM、shape errorなく16,000 stepまで完走し、250 stepごとのvalidation lossと生成文を保存する。exp115に対して一般会話F1が維持または改善し、EOS 48/48を維持できれば、継続学習を自然な会話改善の候補とする。医療の完全一致が改善すれば追加の副次的成果とする。validation lossだけが改善して生成文が定型句や無関係な付け足しへ崩れた場合は、stepを増やすだけでは不十分と判断する。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-exp115-continuation-sft-runpod-16k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-fineweb-shards34-answer-focus-sft-runpod-8k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-exp115-continuation-sft-runpod-16k \
  --samples-dir artifacts/samples/issue1-exp115-continuation-sft-runpod-16k \
  --device cuda --start-step 8000 --max-steps 16000 --lr-schedule-steps 16000 \
  --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

学習開始前に、設定・学習コード・入力データ・exp115 checkpointのハッシュをRunpod上で照合する。学習中は少なくとも1,000 step以内ごとにこのノートへvalidation loss、学習率、経過時間、生成文、警告を追記する。学習終了後はexp115と同じ評価を行い、checkpoint本体をGitHubへ追加せず、メタデータ・ハッシュ・全生成文・評価全文・ログを保存する。

## 学習中の記録

学習開始前。ここへstep 8,250以降のvalidation lossと生成結果を追記する。
