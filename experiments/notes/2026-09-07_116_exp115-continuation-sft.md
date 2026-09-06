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

### 2026-09-07：初回起動の設定検証失敗

最初の起動は、設定の`training.warmup_steps`へ0を指定したため、設定検証で停止した。`warmup_steps`は正の整数だけを受け付ける実装であり、モデルのforward、checkpoint読み込み、学習データの読み込みより前に終了している。Runpod上の出力ディレクトリにはcheckpointや生成文は作成されず、exp115のcheckpointと入力データは変更されていない。設定を1へ修正し、継続学習開始時点ではwarmupの影響がほぼない条件として再実行する。

### 2026-09-07：step 8,250〜8,750

設定修正後の再実行は正常に開始した。step 8,250のvalidation lossは2.754303、step 8,500は2.754326、step 8,750は2.753209だった。step 8,750でexp115のbest 2.754026を更新し、learning rateは9.682e-7、経過時間は62.64秒だった。学習率を抑えた継続でも初期には改善が見られている。NaN、OOM、shape errorは発生していない。各stepの生成文は `artifacts/samples/issue1-exp115-continuation-sft-runpod-16k/` に保存されている。

### 2026-09-07：step 9,000〜9,250

step 9,000のvalidation lossは2.752965、step 9,250は2.751872だった。exp115のbestから合計0.002154改善しており、learning rateはstep 9,250で8.816e-7、経過時間は104.43秒だった。低学習率でvalidation lossは単調ではないものの、現時点では改善傾向を保っている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 9,500〜10,000

step 9,500のvalidation lossは2.751341、step 9,750は2.750261、step 10,000は2.750426だった。step 9,750で現時点の最良値を更新し、exp115のbestから0.003765改善した。step 10,000のlearning rateは7.558e-7、経過時間は166.36秒だった。step 10,000でわずかな反発はあるが、継続学習によるvalidation lossの改善は続いている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 10,250〜10,750

step 10,250のvalidation lossは2.750998、step 10,500は2.750023、step 10,750は2.748146だった。step 10,750で現時点の最良値を更新し、exp115のbestから0.005880改善した。step 10,750のlearning rateは6.375e-7、経過時間は227.13秒だった。学習率が下がる中でもvalidation lossは改善しており、同じSFTデータを追加で見せる効果がまだ残っている可能性がある。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 11,000〜11,500

step 11,000のvalidation lossは2.748025、step 11,250は2.747518、step 11,500は2.748593だった。step 11,250で現時点の最良値を更新し、exp115のbestから0.006507改善した。step 11,500では小さな反発が見られたが、直前までの改善幅は維持されている。step 11,500のlearning rateは5.292e-7、経過時間は288.44秒だった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 11,750〜12,250

step 11,750のvalidation lossは2.747498、step 12,000は2.748418、step 12,250は2.748064だった。step 11,750で現時点の最良値をわずかに更新し、exp115のbestから0.006528改善した。step 12,250のlearning rateは4.333e-7、経過時間は349.26秒だった。step 11,250以降は改善がほぼ横ばいであり、継続学習の効果が飽和し始めた可能性がある。学習は安定している。

### 2026-09-07：step 12,500〜13,000

step 12,500のvalidation lossは2.748631、step 12,750は2.749006、step 13,000は2.748297だった。step 11,750のbest 2.747498は更新できず、step 13,000のlearning rateは3.518e-7、経過時間は409.08秒だった。低学習率の継続による改善はstep 11,750付近で頭打ちとなり、その後は小さな揺らぎの範囲に入った可能性が高い。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 13,250〜13,750

step 13,250のvalidation lossは2.748865、step 13,500は2.748030、step 13,750は2.748123だった。最良値はstep 11,750の2.747498のままで、step 13,750のlearning rateは2.865e-7、経過時間は469.24秒だった。step 13,000以降も大きな悪化はないが、追加学習によるvalidation改善は確認できない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 14,000〜14,500

step 14,000のvalidation lossは2.748182、step 14,250は2.748196、step 14,500は2.748038だった。最良値はstep 11,750の2.747498から更新されず、step 14,500のlearning rateは2.388e-7、経過時間は529.04秒だった。後半はvalidation lossが2.748前後で安定しているが、exp115をさらに下回る新しい改善は確認できていない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 14,750〜15,000

step 14,750のvalidation lossは2.747659、step 15,000は2.747452だった。step 15,000で最良値をわずかに更新し、exp115のbestから0.006573改善した。step 15,000のlearning rateは2.173e-7、経過時間は569.18秒だった。終盤に小さいながら改善が戻っているため、step 16,000まで記録を継続する。
